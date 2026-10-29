"""Search settings drive compiler bounds and candidate attempts."""
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
                policy.fixed_execution_contract(changed["execution_contract"])

    def test_generated_cpp_bounds_follow_the_definition(self):
        header = policy.generated_header()
        for name, value in (("kTimingSearchCycles", "timing_search_cycles"),
                            ("kGeneratedWordLimit", "generated_word_limit"),
                            ("kSpillRewriteLimit", "spill_rewrite_limit")):
            self.assertIn(f"{name} = {policy.POLICY['engineering']['compiler'][value]};", header)
        self.assertIn(f"kDMAProposalSerializationCycles = {policy.POLICY['dma_proposal']['intercommand_serialization_cycles']};", header)
        self.assertIn(policy.canonical_hash(policy.POLICY["dma_proposal"]), header)

    def test_dma_serialization_reaches_the_compiler(self):
        changed = deepcopy(policy.POLICY)
        changed["dma_proposal"]["intercommand_serialization_cycles"] = 2
        with patch.object(policy, "POLICY", changed):
            header = policy.generated_header()
        self.assertIn("kDMAProposalSerializationCycles = 2;", header)

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
                       mreg_candidates=((7,) * 64,))
        receipt = self.select(scorer=lambda *args, **kwargs: 11, **options)
        effective = receipt["search_policy"]["effective"]
        self.assertEqual(effective["random_seeds"], [19])
        self.assertEqual(effective["mreg_candidates"], [[7] * 64])
        self.assertEqual(effective["max_candidates"], 5)
        self.assertEqual(len(receipt["candidates"]), 4)
        self.assertEqual(receipt["search_policy"], json.loads((self.directory / "compiled/selection.json").read_text())["search_policy"])
        self.assertNotIn("op_timing_sha256", receipt["search_policy"])

if __name__ == "__main__":
    unittest.main()
