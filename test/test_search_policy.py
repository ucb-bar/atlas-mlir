"""Optimization policy identity and active knobs remain separate from model facts."""
from copy import deepcopy
from dataclasses import dataclass
from types import SimpleNamespace
import argparse
import json
from pathlib import Path
import random
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import atlas_model_selection as selection
import atlas_search_policy as policy


class SearchPolicyTest(unittest.TestCase):
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
            with self.subTest(name=name, value=value), tempfile.TemporaryDirectory() as temporary:
                source = Path(temporary) / "policy.json"
                source.write_text(json.dumps(changed))
                with patch.object(policy, "POLICY_PATH", source), self.assertRaisesRegex(ValueError, "execution contract"):
                    policy.load_policy()

    def test_canonical_identity_and_effective_overrides(self):
        effective = {"random_seeds": [9], "workers": 2, "search_evaluations": 3}
        receipt = policy.policy_receipt(effective)
        self.assertEqual(receipt["definition_sha256"], policy.canonical_hash(policy.POLICY))
        digest = receipt.pop("sha256")
        self.assertEqual(digest, policy.canonical_hash(receipt))
        self.assertEqual(digest, policy.canonical_hash(dict(reversed(list(receipt.items())))))
        effective["workers"] = 1
        self.assertNotEqual(digest, policy.policy_receipt(effective)["sha256"])
        # Changing a run budget does not invent a new model-fact identity.
        self.assertEqual(receipt["compiler_policy_sha256"], policy.COMPILER_POLICY_SHA256)
        self.assertEqual(receipt["effective"]["workers"], 2)

    def test_budget_and_seed_changes_preserve_derived_projection_semantics(self):
        from atlas_derive_replay import projection_program
        from atlas_model_replay import semantic_digest
        source = dict(schema="atlas.core_edges.v1", clock_step=1,
                      phases=["s1_previous_state", "execution_units", "fetch"],
                      dma_wait_observes="previous_edge_flags", implicit_exu_stalls=False,
                      checked_methods=["Core.tick", "Core._tick_s1"], source_contract="atlas.model_frontend.v1")
        baseline = projection_program(source)
        changed = deepcopy(policy.POLICY)
        changed["presets"]["full"]["max_candidates"] += 1
        changed["pools"]["random_seeds"] = [103, 107]
        changed["pools"]["neighbor_shuffle_seed"] += 1
        with patch.object(policy, "POLICY", changed):
            regenerated = projection_program(source)
            changed_receipt = policy.policy_receipt({})
        self.assertEqual(semantic_digest(baseline), semantic_digest(regenerated))
        self.assertEqual(baseline, regenerated)
        self.assertNotEqual(changed_receipt["definition_sha256"], policy.policy_receipt({})["definition_sha256"])

    def test_generated_cpp_bounds_and_projection_are_from_the_definition(self):
        header = policy.generated_header()
        for name, value in (("kTimingSearchCycles", "timing_search_cycles"),
                            ("kGeneratedWordLimit", "generated_word_limit"),
                            ("kSpillRewriteLimit", "spill_rewrite_limit")):
            self.assertIn(f"{name} = {policy.POLICY['engineering']['compiler'][value]};", header)
        self.assertIn(policy.COMPILER_POLICY_SHA256, header)
        self.assertIn(policy.canonical_hash(policy.POLICY["replay_projection"]), header)
        self.assertIn(f"kDMAProposalSerializationCycles = {policy.POLICY['dma_proposal']['intercommand_serialization_cycles']};", header)
        self.assertIn(policy.canonical_hash(policy.POLICY["dma_proposal"]), header)
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "AtlasSearchPolicy.h"
            subprocess.run([sys.executable, "-B", str(ROOT / "tools/atlas_search_policy.py"),
                            "--header", str(output)], check=True)
            self.assertEqual(header, output.read_text())

    def test_dma_serialization_has_one_active_definition_and_identity(self):
        changed = deepcopy(policy.POLICY)
        changed["dma_proposal"]["intercommand_serialization_cycles"] = 2
        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary) / "policy.json"
            source.write_text(json.dumps(changed))
            with patch.object(policy, "POLICY_PATH", source):
                loaded = policy.load_policy()
        self.assertNotEqual(policy.compiler_policy_hash(loaded), policy.COMPILER_POLICY_SHA256)
        with patch.object(policy, "POLICY", loaded), patch.object(policy, "COMPILER_POLICY_SHA256", policy.compiler_policy_hash(loaded)):
            header = policy.generated_header()
            receipt = policy.policy_receipt({})
        self.assertIn("kDMAProposalSerializationCycles = 2;", header)
        self.assertIn(json.dumps(loaded["dma_proposal"], sort_keys=True, separators=(",", ":")), header)
        self.assertEqual(receipt["definition"]["dma_proposal"], loaded["dma_proposal"])
        self.assertEqual(receipt["compiler_policy_sha256"], policy.compiler_policy_hash(loaded))
        # Candidate budgets and seeds cannot change the compiler adapter hash.
        loaded["presets"]["full"]["max_candidates"] += 1
        loaded["pools"]["random_seeds"] = [93]
        self.assertEqual(policy.compiler_policy_hash(loaded), policy.compiler_policy_hash(changed))

    def test_algorithm_changes_refuse_instead_of_becoming_unused_labels(self):
        changed = deepcopy(policy.POLICY)
        changed["algorithms"]["order_neighbors"] = "unimplemented-v2"
        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary) / "policy.json"
            source.write_text(json.dumps(changed))
            with patch.object(policy, "POLICY_PATH", source), self.assertRaisesRegex(ValueError, "algorithm revision"):
                policy.load_policy()

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
                       lambda value: value["presets"]["full"].update(max_candidates=False)):
            changed = deepcopy(policy.POLICY)
            mutate(changed)
            with self.subTest(changed=changed), tempfile.TemporaryDirectory() as temporary:
                source = Path(temporary) / "policy.json"
                source.write_text(json.dumps(changed))
                with patch.object(policy, "POLICY_PATH", source), self.assertRaises(ValueError):
                    policy.load_policy()

    def test_shuffle_seed_and_wait_pool_drive_proposals(self):
        graph = {"preds": [[], [], [], []]}
        order = ((0, 1, 2, 3),)
        baseline = list(selection.order_neighbors([graph], order))
        changed = dict(selection.POOLS, neighbor_shuffle_seed=17)
        with patch.object(selection, "POOLS", changed):
            alternatives = list(selection.order_neighbors([graph], order))
        self.assertEqual(set(baseline), set(alternatives))
        self.assertNotEqual(baseline, alternatives)
        changed = dict(selection.CANDIDATES, curve_wait_placements=["source", "default"])
        with patch.object(selection, "CANDIDATES", changed):
            candidates = list(selection.order_candidates(selection.Candidate(), [order], True))
        self.assertEqual([item.wait_placement for item in candidates], ["source", "default"])

    def test_cli_defaults_and_effective_receipt_match_actual_attempts(self):
        parser = argparse.ArgumentParser()
        selection.add_model_arguments(parser)
        args = parser.parse_args(["--provider", "model", "--workers", "2", "--max-candidates", "5"])
        options = selection.model_arguments(args)
        options.update(random_seeds=(19,), search_evaluations=0, search_group_evaluations=0,
                       mreg_candidates=((7,) * 64,), store_window_candidates=((3,),),
                       pack_reload_candidates=False, pack_preload_candidates=False)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source.mlir"
            source.write_text("module {}")
            programs = {}
            def command(argv, *, cwd, **kwargs):
                if Path(argv[0]).name == "atlas-emit":
                    return subprocess.CompletedProcess(argv, 0, json.dumps(programs[cwd]), "")
                (cwd / argv[argv.index("-o") + 1]).write_text("module {}")
                if any(arg.startswith("--atlas-compile-candidate=") for arg in argv):
                    (cwd / "allocation.json").write_text('{"fits": true}')
                    programs[cwd] = {"compiler_search_policy_sha256": policy.COMPILER_POLICY_SHA256,
                                     "instructions": [{"word_u32": 0}]}
                return subprocess.CompletedProcess(argv, 0, "", "")
            with patch.object(selection.subprocess, "run", side_effect=command):
                receipt = selection.select_program(source, root / "output", root,
                                                   scorer=lambda *args, **kwargs: 11, **options)
            effective = receipt["search_policy"]["effective"]
            self.assertEqual(effective["random_seeds"], [19])
            self.assertEqual(effective["mreg_candidates"], [[7] * 64])
            self.assertEqual(effective["store_window_candidates"], [[3]])
            self.assertEqual(effective["objective"], ["cycles", "index"])
            self.assertEqual(effective["max_candidates"], 5)
            self.assertEqual(len(receipt["candidates"]), 5)
            self.assertEqual(receipt["search_policy"], json.loads((root / "output/selection.json").read_text())["search_policy"])
            self.assertNotIn("op_timing_sha256", receipt["search_policy"])

    def test_objective_policy_is_native_only_and_fifo43_is_explicit(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source.mlir"
            source.write_text("module {}")
            words = {}
            def command(argv, *, cwd, **kwargs):
                if Path(argv[0]).name == "atlas-emit":
                    return subprocess.CompletedProcess(argv, 0, json.dumps({
                        "compiler_search_policy_sha256": policy.COMPILER_POLICY_SHA256,
                        "instructions": [{"word_u32": words[cwd]}]}), "")
                (cwd / argv[argv.index("-o") + 1]).write_text("module {}")
                if any(arg.startswith("--atlas-compile-candidate=") for arg in argv):
                    (cwd / "allocation.json").write_text('{"fits": true}')
                    words[cwd] = int(any("lookahead=1" in arg for arg in argv))
                return subprocess.CompletedProcess(argv, 0, "", "")
            profiles = []
            def score(program, *, profile):
                profiles.append(profile)
                return 10 if profile == "curve" else 100 - program["instructions"][0]["word_u32"]
            with patch.object(selection.subprocess, "run", side_effect=command):
                for profile in ("curve", "fifo43"):
                    profiles.clear()
                    receipt = selection.select_program(source, root / profile, root, scorer=score,
                                                       profile=profile, max_candidates=2, workers=2,
                                                       curve_candidates=False)
                    self.assertEqual(receipt["selected_index"], 0 if profile == "curve" else 1)
                    self.assertEqual(set(profiles), {profile})
                    self.assertEqual(receipt["search_policy"]["effective"]["objective"], ["cycles", "index"])
                    self.assertEqual(receipt["objective"]["minimize"], policy.POLICY["objectives"][profile])
                    self.assertEqual(receipt["objective"]["policy"], policy.POLICY["algorithms"]["selection_objective"])
                    self.assertEqual(receipt["objective"]["tie_rule"], policy.POLICY["algorithms"]["selection_tie"])

    def test_legacy_synthetic_secondary_objective_is_refused(self):
        changed = deepcopy(policy.POLICY)
        changed["objectives"]["curve"] = ["cycles", "fifo43_cycles", "index"]
        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary) / "policy.json"
            source.write_text(json.dumps(changed))
            with patch.object(policy, "POLICY_PATH", source), self.assertRaisesRegex(ValueError, "selection objective"):
                policy.load_policy()

    def test_compiler_policy_mismatch_is_fatal_even_with_a_custom_scorer(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source.mlir"
            source.write_text("module {}")
            def command(argv, *, cwd, **kwargs):
                if Path(argv[0]).name == "atlas-emit":
                    return subprocess.CompletedProcess(argv, 0, json.dumps({
                        "compiler_search_policy_sha256": "0" * 64, "instructions": []}), "")
                (cwd / argv[argv.index("-o") + 1]).write_text("module {}")
                return subprocess.CompletedProcess(argv, 0, "", "")
            with patch.object(selection.subprocess, "run", side_effect=command), self.assertRaisesRegex(ValueError, "rebuild"):
                selection.select_program(source, root / "output", root, workers=2,
                                         scorer=lambda *args, **kwargs: 0)
            receipt = json.loads((root / "output/selection.json").read_text())
            self.assertEqual(receipt["candidates"][0]["error_stage"], "policy")
            self.assertIn("search_policy", receipt)


if __name__ == "__main__":
    unittest.main()
