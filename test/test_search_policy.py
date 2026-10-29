"""Optimization policy identity and active knobs remain separate from model facts."""
from copy import deepcopy
from dataclasses import dataclass
from types import SimpleNamespace
import argparse
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import atlas_model_selection as selection
import atlas_search_policy as policy
from test_model_selection import SelectionFixture


class SearchPolicyTest(SelectionFixture, unittest.TestCase):
    def load(self, definition):
        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary) / "policy.json"
            source.write_text(json.dumps(definition))
            with patch.object(policy, "POLICY_PATH", source):
                return policy.load_policy()

    def test_fixed_execution_contract_preserves_effective_configuration(self):
        from atlas_model_timing import configuration
        @dataclass
        class Architecture:
            numerics: str = "default"
            randomize_init: bool = True
            example_capacity: int = 32
        cfg = SimpleNamespace(arch_state_config=Architecture(), example_latency=7)
        model_module = SimpleNamespace(DefaultHardwareConfig=lambda: cfg)
        with patch.dict(sys.modules, {"npu_model.configs.hardware.default": model_module}):
            actual, values = configuration()
        self.assertIs(actual, cfg)
        self.assertEqual(values, {"arch_state_config": {"numerics": "rtl", "randomize_init": False,
                                                       "example_capacity": 32},
                                  "example_latency": 7, "ignore_runtime_errors": False})

    def test_execution_safety_contract_mutations_refuse(self):
        for name, value in (("numerics", "float"), ("randomize_init", True),
                            ("ignore_runtime_errors", True), ("ignore_runtime_errors", 0),
                            ("schema", "unimplemented-v2")):
            changed = deepcopy(policy.POLICY)
            changed["execution_contract"][name] = value
            with self.subTest(name=name, value=value), self.assertRaisesRegex(ValueError, "execution contract"):
                self.load(changed)

    def test_canonical_identity_and_effective_overrides(self):
        effective = {"random_seeds": [9], "workers": 2, "search_group_evaluations": 3}
        receipt = policy.policy_receipt(effective)
        self.assertEqual(receipt["definition_sha256"], policy.canonical_hash(policy.POLICY))
        digest = receipt.pop("sha256")
        self.assertEqual(digest, policy.canonical_hash(receipt))
        effective["workers"] = 1
        self.assertNotEqual(digest, policy.policy_receipt(effective)["sha256"])
        self.assertEqual(receipt["compiler_policy_sha256"], policy.COMPILER_POLICY_SHA256)
        self.assertEqual(receipt["effective"]["workers"], 2)


    def test_generated_cpp_bounds_and_projection_are_from_the_definition(self):
        header = policy.generated_header()
        for name, value in (("kTimingSearchCycles", "timing_search_cycles"),
                            ("kGeneratedWordLimit", "generated_word_limit"),
                            ("kSpillRewriteLimit", "spill_rewrite_limit")):
            self.assertIn(f"{name} = {policy.POLICY['engineering']['compiler'][value]};", header)
        self.assertIn(policy.COMPILER_POLICY_SHA256, header)
        self.assertIn(f"kDMAProposalSerializationCycles = {policy.POLICY['dma_proposal']['intercommand_serialization_cycles']};", header)
        self.assertIn(policy.canonical_hash(policy.POLICY["dma_proposal"]), header)

    def test_dma_serialization_has_one_active_definition_and_identity(self):
        changed = deepcopy(policy.POLICY)
        changed["dma_proposal"]["intercommand_serialization_cycles"] = 2
        loaded = self.load(changed)
        self.assertNotEqual(policy.compiler_policy_hash(loaded), policy.COMPILER_POLICY_SHA256)
        with patch.object(policy, "POLICY", loaded), patch.object(policy, "COMPILER_POLICY_SHA256", policy.compiler_policy_hash(loaded)):
            receipt = policy.policy_receipt({})
        self.assertEqual(receipt["definition"]["dma_proposal"], loaded["dma_proposal"])
        self.assertEqual(receipt["compiler_policy_sha256"], policy.compiler_policy_hash(loaded))
        loaded["presets"]["full"]["max_candidates"] += 1
        loaded["pools"]["random_seeds"] = [93]
        self.assertEqual(policy.compiler_policy_hash(loaded), policy.compiler_policy_hash(changed))

    def test_algorithm_changes_refuse_instead_of_becoming_unused_labels(self):
        changed = deepcopy(policy.POLICY)
        changed["algorithms"]["group_neighbors"] = "unimplemented-v2"
        with self.assertRaisesRegex(ValueError, "algorithm revision"):
            self.load(changed)

    def test_unknown_fields_and_invalid_numeric_bounds_refuse(self):
        for mutate in (lambda value: value.update(unknown_variant={}),
                       lambda value: value["engineering"]["compiler"].update(unused_limit=17),
                       lambda value: value["engineering"]["scorer"].update(max_cycles=0),
                       lambda value: value["dma_proposal"].update(unused_gap=1),
                       lambda value: value["dma_proposal"].update(classification="model_fact"),
                       lambda value: value["dma_proposal"].update(intercommand_serialization_cycles=0),
                       lambda value: value["dma_proposal"].update(intercommand_serialization_cycles=True),
                       lambda value: value["dma_proposal"].update(intercommand_serialization_cycles=2**31),
                       lambda value: value["candidates"].update(batch_width=0),
                       lambda value: value["candidates"].update(curve_group_wait_placements=["source", "default"]),
                       lambda value: value["presets"]["full"].update(max_candidates=False)):
            changed = deepcopy(policy.POLICY)
            mutate(changed)
            with self.subTest(changed=changed), self.assertRaises(ValueError):
                self.load(changed)

    def test_shuffle_seed_and_wait_pool_drive_proposals(self):
        graph = {"preds": [[], [0], [], [], [], []]}
        order = ((0, 1, 2, 3, 4, 5),)
        baseline = list(selection.group_order_neighbors([graph], order))
        changed = dict(selection.POOLS, neighbor_shuffle_seed=17)
        with patch.object(selection, "POOLS", changed):
            alternatives = list(selection.group_order_neighbors([graph], order))
        self.assertEqual(set(baseline), set(alternatives))
        self.assertNotEqual(baseline, alternatives)

    def test_cli_defaults_and_effective_receipt_match_actual_attempts(self):
        parser = argparse.ArgumentParser()
        selection.add_model_arguments(parser)
        args = parser.parse_args(["--provider", "model", "--workers", "2", "--max-candidates", "5"])
        options = selection.model_arguments(args)
        options.update(random_seeds=(19,), search_group_evaluations=0,
                       mreg_candidates=((7,) * 64,), store_window_candidates=((3,),),
                       pack_reload_candidates=False, pack_preload_candidates=False)
        receipt = self.select(scorer=lambda *args, **kwargs: 11, **options)
        effective = receipt["search_policy"]["effective"]
        self.assertEqual(effective["random_seeds"], [19])
        self.assertEqual(effective["mreg_candidates"], [[7] * 64])
        self.assertEqual(effective["store_window_candidates"], [[3]])
        self.assertEqual(effective["objective"], ["cycles", "index"])
        self.assertEqual(effective["max_candidates"], 5)
        self.assertEqual(len(receipt["candidates"]), 5)
        self.assertEqual(receipt["search_policy"], json.loads((self.directory / "compiled/selection.json").read_text())["search_policy"])
        self.assertNotIn("op_timing_sha256", receipt["search_policy"])

    def test_legacy_synthetic_secondary_objective_is_refused(self):
        changed = deepcopy(policy.POLICY)
        changed["objectives"]["curve"] = ["cycles", "fifo43_cycles", "index"]
        with self.assertRaisesRegex(ValueError, "selection objective"):
            self.load(changed)

    def test_compiler_policy_mismatch_is_fatal_even_with_a_custom_scorer(self):
        with self.assertRaisesRegex(ValueError, "rebuild"):
            self.select(runner=self.override_run("", 0, 0, policy="0" * 64), workers=2)
        receipt = json.loads((self.directory / "compiled/selection.json").read_text())
        self.assertEqual(receipt["candidates"][0]["error_stage"], "policy")
        self.assertIn("search_policy", receipt)


if __name__ == "__main__":
    unittest.main()
