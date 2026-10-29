"""Source contracts, complete ISA coverage, and fatal derivation validation."""
from contextlib import redirect_stderr
from dataclasses import replace
from io import StringIO
import json
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import atlas_model_timing as timing
import atlas_derive_vpu as derive
import atlas_derive_mxu as mxu
import atlas_derive_memory as memory
from atlas_model_schema import LOCK
from atlas_model_diagnostics import ModelTimingValidationError
from model_evidence import isolate_model_imports


class VPUShapeTest(unittest.TestCase):
    def test_both_source_shape_pins_fail_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            for name in ("vpu.py", "bank_conflict.py"):
                source = Path(directory) / name
                source.write_text("def changed_timing():\n    return 0\n")
                with self.subTest(name=name), patch.object(derive.inspect, "getsourcefile", return_value=str(source)):
                    with self.assertRaisesRegex(derive.UnsupportedVPU, "unsupported VPU timing semantics in " + name):
                        derive._check_source(SimpleNamespace(), latency_table=name == "vpu.py")


class DerivationGateTest(unittest.TestCase):
    def test_wrong_native_timing_fact_is_rejected(self):
        observed = dict(events={"write": dict(count=32, first_age=35, last_age=66, step=1)},
                        first_free_age=67)
        with tempfile.TemporaryDirectory() as directory, \
                patch.object(timing, "observe_vpu", return_value=observed):
            good = {"vpu.add": observed}
            report, _ = timing.validate_derived(object(), good, {}, Path(directory))
            self.assertTrue(report["passed"])
            wrong = {"vpu.add": dict(observed, first_free_age=68)}
            with self.assertRaises(ModelTimingValidationError):
                timing.validate_derived(object(), wrong, {}, Path(directory))


@unittest.skipUnless(os.environ.get("ATLAS_DERIVE_MODEL_ROOT"), "set ATLAS_DERIVE_MODEL_ROOT for source contracts")
class NativeModelTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        isolate_model_imports(cls, os.environ["ATLAS_DERIVE_MODEL_ROOT"])
        cls.cfg, _ = timing.configuration()


class ModelDerivationTest(NativeModelTest):
    def test_derivation_needs_no_numerical_state_or_unit_ticks(self):
        from npu_model.hardware.arch_state import ArchState
        from npu_model.hardware.exu import StagedExecution
        from npu_model.hardware.mxu import _MatrixExecutionUnit
        from npu_model.hardware.lsu import LoadStoreUnit
        from npu_model.hardware.xlu import CrossLaneExecutionUnit
        from npu_model.hardware.vpu import VectorExecutionUnit
        with patch.object(ArchState, "__init__", side_effect=AssertionError("allocated state")), \
                patch.object(StagedExecution, "__init__", side_effect=AssertionError("staged arithmetic")), \
                patch.object(_MatrixExecutionUnit, "tick", side_effect=AssertionError("MXU tick")), \
                patch.object(VectorExecutionUnit, "tick", side_effect=AssertionError("VPU tick")), \
                patch.object(LoadStoreUnit, "tick", side_effect=AssertionError("LSU tick")), \
                patch.object(CrossLaneExecutionUnit, "tick", side_effect=AssertionError("XLU tick")):
            self.assertEqual(len(mxu.derive_mxu(self.cfg)[0]), 14)
            self.assertEqual(len(derive.derive_vpu(self.cfg)[0]), 29)
            blocks, releases = memory.derive_memory(self.cfg)
        self.assertEqual(len(blocks), 6)
        self.assertEqual(releases["vlsu.vstore"]["mreg_read"]["release_age"], 34)
        self.assertEqual(releases["xlu.transpose"]["mreg_read"]["release_age"], 33)
        self.assertEqual(releases["scalar_lsu.load"]["vmem_read"]["available_age"], 2)

    def test_mxu_source_latency_moves_compute_window_and_drain(self):
        source, _ = mxu._sources(self.cfg)
        baseline = mxu._blocks(mxu._plan(source))
        changed = mxu._blocks(mxu._plan(source.replace("compute_first = 3\n", "compute_first = 7\n")))
        self.assertEqual(changed["mxu1.matmul"]["first_free_age"], baseline["mxu1.matmul"]["first_free_age"] + 4)
        self.assertEqual(changed["mxu1.matmul"]["events"]["acc_write"]["first_age"], 7)
        self.assertEqual(changed["mxu0.matmul"], baseline["mxu0.matmul"])
        self.assertEqual(changed["mxu1.push_weight"], baseline["mxu1.push_weight"])

    def test_mxu_incomplete_rows_or_early_releases_fail_closed(self):
        source, _ = mxu._sources(self.cfg)
        for changed in (source.replace("0 <= age < 32", "0 <= age < 31"),
                        source.replace("age == 32 and", "age == 30 and")):
            self.assertTrue(source != changed, "negative mutation must change the source")
            with self.assertRaises(mxu.UnsupportedMXU):
                mxu._plan(changed)

class DerivedVPUTest(NativeModelTest):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        from npu_model.hardware import vpu
        cls.vpu, cls.timing = vpu, timing

    def test_latency_edits_change_drain_and_reservations_from_source_table(self):
        baseline = derive.derive_vpu(self.cfg)[0]
        with patch.dict(self.vpu.VPU_OP_LATENCIES, {"vadd.bf16": 80}):
            changed, releases = derive.derive_vpu(self.cfg)
        self.assertEqual(changed["vpu.add"]["first_free_age"], 80)
        self.assertEqual(changed["vpu.add"]["events"], baseline["vpu.add"]["events"])
        self.assertEqual(releases["vpu.add"]["mreg_write"]["available_age"], 80)
        self.assertEqual(releases["vpu.add"]["mreg_write"]["release_age"], 79)
        self.assertEqual({name: block for name, block in changed.items() if name != "vpu.add"},
                         {name: block for name, block in baseline.items() if name != "vpu.add"})

    def test_bad_latencies_fail_closed(self):
        for latency in (0, -1, True, 1.5, 4096, 32, 64, 65):
            with self.subTest(latency=latency), patch.dict(self.vpu.VPU_OP_LATENCIES, {"vadd.bf16": latency}):
                with self.assertRaises(derive.UnsupportedVPU):
                    derive.derive_vpu(self.cfg)[0]

    def test_unsupported_geometry_and_isa_fail_closed(self):
        cfg = type(self.cfg)()
        cfg.arch_state_config = replace(self.cfg.arch_state_config, mrf_depth=16)
        with self.assertRaisesRegex(derive.UnsupportedVPU, "requires 64 MREGs"):
            derive.derive_vpu(cfg)
        cfg.arch_state_config = self.cfg.arch_state_config
        cfg.isa = SimpleNamespace(operations={name: cls for name, cls in self.cfg.isa.operations.items()
                                              if name != "vadd.bf16"})
        with self.assertRaisesRegex(derive.UnsupportedVPU, "operation coverage"):
            derive.derive_vpu(cfg)

    @unittest.skipUnless(os.environ.get("ATLAS_DERIVE_OBSERVE"), "set ATLAS_DERIVE_OBSERVE for native validation")
    def test_vpu_fact_matches_native_execution(self):
        blocks, releases = derive.derive_vpu(self.cfg)
        with tempfile.TemporaryDirectory() as directory:
            report, _ = timing.validate_derived(self.cfg, blocks, releases, Path(directory))
        self.assertTrue(report["passed"])
        blocks["vpu.add"]["first_free_age"] += 1
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(ModelTimingValidationError):
                timing.validate_derived(self.cfg, blocks, releases, Path(directory))


if __name__ == "__main__":
    unittest.main()
