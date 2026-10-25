"""Structural source parameters and their bounded admission contracts."""
import ast
from copy import deepcopy
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))

import atlas_derive_memory as memory
import atlas_derive_overlap as overlap
from atlas_derive_mxu import _plan, UnsupportedMXU
from test_derive_overlap import fixtures


class CapacityRulesTests(unittest.TestCase):
    def test_one_slot_or_tracker_binds_pair_minimum(self):
        facts, contract = fixtures()
        baseline = overlap.SourceRules(facts, contract)
        changed = deepcopy(contract)
        changed["capacity"].update(vpu_slots=1, mxu0_compute=1)
        strict = overlap.SourceRules(facts, changed)
        independent = dict(mreg_banks="disjoint")
        self.assertEqual(baseline.minimum_gap("vpu.relu", "vpu.exp2", independent), 1)
        self.assertEqual(strict.minimum_gap("vpu.relu", "vpu.exp2", independent), 65)
        independent.update(accumulator="other", weight_slot="same")
        self.assertEqual(baseline.minimum_gap("mxu0.matmul", "mxu0.matmul", independent), 32)
        self.assertEqual(strict.minimum_gap("mxu0.matmul", "mxu0.matmul", independent), 95)

    def test_chain_slots_and_tracker_final_edge_follow_selected_capacity(self):
        facts, contract = fixtures()
        # The three-slot saturation probe uses one additional independent lane box.
        sin = deepcopy(next(b for b in facts["op_timing"] if b["name"] == "vpu.relu"))
        sin["name"] = "vpu.sin"
        facts["op_timing"].append(sin)
        facts["reservation_releases"]["vpu.sin"] = deepcopy(facts["reservation_releases"]["vpu.relu"])
        for slots in (1, 2, 3):
            changed = deepcopy(contract)
            changed["capacity"].update(vpu_slots=slots, mxu0_compute=2, mxu1_compute=1)
            rules = overlap.SourceRules(facts, changed)
            samples = [s for s in overlap.chain_samples(rules) if s["category"] == "capacity"]
            self.assertEqual(len(samples), 6)
            for sample in samples:
                issues = [0]
                for gap in sample["gaps"]:
                    issues.append(issues[-1] + gap)
                conflict = rules.capacity_conflict(sample["operations"], issues)
                with self.subTest(slots=slots, sample=sample["id"]):
                    if sample["expectation"] == "positive":
                        self.assertIsNone(conflict)
                    else:
                        self.assertIsNotNone(conflict)


@unittest.skipUnless(os.environ.get("ATLAS_DERIVE_MODEL_ROOT"), "set ATLAS_DERIVE_MODEL_ROOT for reviewed source guards")
class CapacitySourceTests(unittest.TestCase):
    def source(self, name):
        return overlap.safe_path(Path(os.environ["ATLAS_DERIVE_MODEL_ROOT"]) / "npu_model" / "hardware" / name).read_text()

    def test_depth_changes_preserve_isolated_timing_but_not_control_flow(self):
        source = self.source("mxu.py")
        expected = _plan(source)
        for first, second in ((1, 1), (2, 4), (64, 64)):
            changed = source.replace("inflight_depth = 3", f"inflight_depth = {first}").replace("inflight_depth = 2", f"inflight_depth = {second}")
            self.assertEqual(_plan(changed), expected)
        for value in (0, 65, "True", "self.config.depth"):
            with self.subTest(value=value), self.assertRaises(UnsupportedMXU):
                _plan(source.replace("inflight_depth = 3", f"inflight_depth = {value}"))
        for old, new in (("len(compute) >= self.inflight_depth", "len(compute) > self.inflight_depth"),
                         ("PORT_CYCLES = 32", "PORT_CYCLES = 16"),
                         ('"weight": ((1,), (0,))', '"weight": ((0,), (1,))'),
                         ('"push_bf16": ((0, 1),)', '"push_bf16": ((0,),)'),
                         ("all(free(port) for port in choice)", "any(free(port) for port in choice)")):
            with self.subTest(mutation=new), self.assertRaises(UnsupportedMXU):
                _plan(source.replace(old, new))

    def test_only_exact_slot_parameter_positions_bypass_shape_pin(self):
        source = self.source("vpu.py")
        for slots in (1, 2, 3):
            changed = source.replace("ISSUE_SLOTS = 2", f"ISSUE_SLOTS = {slots}")
            overlap._guard_ast(ast.parse(changed), "vpu.py")
        for slots in (0, 4, True):
            with self.subTest(slots=slots), self.assertRaises(ValueError):
                overlap._guard_ast(ast.parse(source.replace("ISSUE_SLOTS = 2", f"ISSUE_SLOTS = {slots}")), "vpu.py")
        for old, new in (("cycle - op.issued < op.write_last", "cycle - op.issued <= op.write_last"),
                         ("return VPU_OP_LATENCIES[mnemonic] - 1", "return VPU_OP_LATENCIES[mnemonic]")):
            with self.subTest(mutation=new), self.assertRaises(ValueError):
                overlap._guard_ast(ast.parse(source.replace(old, new)), "vpu.py")

    def test_xlu_duration_is_resolved_after_complete_source_review(self):
        source = self.source("xlu.py")
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "xlu.py"
            with patch.object(memory.inspect, "getsourcefile", return_value=str(path)):
                for old, new in (("TRANSPOSE_CYCLES = 66", "TRANSPOSE_CYCLES = 65"),
                                 ("TRANSPOSE_CYCLES - 1 - age", "TRANSPOSE_CYCLES - age")):
                    path.write_text(source.replace(old, new))
                    with self.subTest(mutation=new), self.assertRaises(memory.UnsupportedMemory):
                        memory._source(SimpleNamespace())


if __name__ == "__main__":
    unittest.main()
