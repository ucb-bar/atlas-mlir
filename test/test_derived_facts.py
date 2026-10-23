"""Source contracts, complete ISA coverage, and fatal derivation validation."""
from dataclasses import replace
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
from atlas_model_facts import safe_path, identity


def retained_probe_cases():
    """Fixed disjoint operand representatives; no boundary/alias variants."""
    for mnemonic, suffix in timing.VPU_BLOCKS.items():
        if mnemonic.startswith("vli."):
            assembly = f"{mnemonic} m4, 0"
        elif mnemonic in {"vpack.bf16.fp8", "vunpack.fp8.bf16"}:
            assembly = f"{mnemonic} m4, m0, e0"
        elif mnemonic in timing.BINARY:
            assembly = f"{mnemonic} m4, m0, m2"
        else:
            assembly = f"{mnemonic} m4, m0"
        yield "vpu." + suffix, "vpu", assembly
    for mxu in ("mxu0", "mxu1"):
        for mnemonic, suffix in timing.MXU_BLOCKS.items():
            if mnemonic == "vmatpush.weight":
                operands = "w0, m0"
            elif mnemonic.startswith("vmatpush"):
                operands = "acc0, m0"
            elif mnemonic == "vmatpop.fp8.acc":
                operands = "m4, acc0, e0"
            elif mnemonic == "vmatpop.bf16.acc":
                operands = "m4, acc0"
            else:
                operands = "acc0, m0, w0"
            yield f"{mxu}.{suffix}", mxu, f"{mnemonic}.{mxu} {operands}"
    yield "vlsu.vload", "vlsu", "vload m4, 0(x0)"
    yield "vlsu.vstore", "vlsu", "vstore m0, 0(x0)"
    yield "xlu.transpose", "xlu", "vtrpose.xlu m4, m0"
    yield "scalar_lsu.load", "scalar_lsu", "lw x4, 0(x0)"
    yield "scalar_lsu.scale_load", "scalar_lsu", "seld e4, 0(x0)"
    yield "scalar_lsu.store", "scalar_lsu", "sw x1, 0(x0)"


class DerivationGateTest(unittest.TestCase):
    def test_manifest_rejects_symlink_root_before_traversal(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "npu_model").symlink_to(root / "unvisited")
            with patch.object(timing.os, "scandir", side_effect=AssertionError("traversed symlink")):
                with self.assertRaisesRegex(RuntimeError, "symlink"):
                    timing.source_digest(root)

    def test_safe_path_rejects_symlink_ancestors(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "link").symlink_to(root / "unvisited")
            with self.assertRaisesRegex(ValueError, "symlink"):
                safe_path(root / "link" / "child")

    def test_validator_forwards_configuration_and_fails_noninterference(self):
        cfg = object()
        block = dict(name="xlu.transpose", first_free_age=2, events={})
        raw = {key: [] for key in ("architectural_states", "control_states", "pre_edges")}
        raw.update(initial_state_sha256="a", final_state_sha256="b", first_free_age=2, complete_count=1,
                   total_instructions=1, busy_cycles=2, instruction_word=0, configuration={}, reservations=[])
        plain = dict(raw, final_state_sha256="changed")
        with tempfile.TemporaryDirectory() as directory, \
                patch.object(timing, "isa_cases", return_value=[("xlu.transpose", "xlu", "probe")]), \
                patch.object(timing, "observe_case", side_effect=[(block, raw), (block, plain)] * 2) as observe:
            with self.assertRaisesRegex(RuntimeError, "validation failed"):
                timing.validate_derived(cfg, {"xlu.transpose": block}, {}, Path(directory))
            self.assertTrue(all(call.kwargs["cfg"] is cfg for call in observe.call_args_list))
            report = json.loads((Path(directory) / "validation-report.json").read_text())
            self.assertFalse(report["passed"])
            self.assertTrue(all("observer changed" in check["error"] for check in report["checks"]))

    def test_identity_tracks_source_and_numerical_data_independently(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data = root / "npu_model/configs/data"
            data.mkdir(parents=True)
            source = root / "npu_model/model.py"
            source.write_text("value = 1\n")
            table = data / "table.bin"
            table.write_bytes(b"one")
            original, _, _ = identity(root)
            source.write_text("value = 2\n")
            changed_source, _, _ = identity(root)
            self.assertNotEqual(original["source_sha256"], changed_source["source_sha256"])
            self.assertEqual(original["numerical_data_sha256"], changed_source["numerical_data_sha256"])
            self.assertEqual(changed_source["revision"], changed_source["source_sha256"][:40])
            table.write_bytes(b"two")
            changed_data, _, _ = identity(root)
            self.assertEqual(changed_source["source_sha256"], changed_data["source_sha256"])
            self.assertNotEqual(changed_source["numerical_data_sha256"], changed_data["numerical_data_sha256"])

    def test_memory_derivation_reports_shared_source_failures_as_memory_errors(self):
        for error_type in (mxu.UnsupportedMXU, memory.UnsupportedVPU):
            error = error_type("unsupported source helper semantics")
            with self.subTest(error_type=error_type), patch.object(memory, "_derive_memory", side_effect=error):
                with self.assertRaisesRegex(memory.UnsupportedMemory, "unsupported source helper semantics") as raised:
                    memory.derive_memory(object())
                self.assertIs(raised.exception.__cause__, error)

    def test_unknown_mxu_semantics_fail_closed(self):
        with self.assertRaises(mxu.UnsupportedMXU):
            mxu._plan("class DifferentEngine: pass")


@unittest.skipUnless(os.environ.get("ATLAS_DERIVE_MODEL_ROOT"), "set ATLAS_DERIVE_MODEL_ROOT for source contracts")
class NativeModelTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        sys.path.insert(0, os.environ["ATLAS_DERIVE_MODEL_ROOT"])
        cls.cfg, _ = timing.configuration()


class ModelDerivationTest(NativeModelTest):
    def test_isa_metadata_reproduces_all_retained_probe_operands(self):
        self.assertEqual(timing.isa_cases(self.cfg), list(retained_probe_cases()))

    def test_mxu_and_memory_need_no_numerical_state_or_unit_ticks(self):
        from npu_model.hardware.arch_state import ArchState
        from npu_model.hardware.exu import StagedExecution
        from npu_model.hardware.mxu import _MatrixExecutionUnit
        from npu_model.hardware.lsu import LoadStoreUnit
        from npu_model.hardware.xlu import CrossLaneExecutionUnit
        with patch.object(ArchState, "__init__", side_effect=AssertionError("allocated state")), \
                patch.object(StagedExecution, "__init__", side_effect=AssertionError("staged arithmetic")), \
                patch.object(_MatrixExecutionUnit, "tick", side_effect=AssertionError("MXU tick")), \
                patch.object(LoadStoreUnit, "tick", side_effect=AssertionError("LSU tick")), \
                patch.object(CrossLaneExecutionUnit, "tick", side_effect=AssertionError("XLU tick")):
            self.assertEqual(len(mxu.derive_mxu(self.cfg)[0]), 14)
            blocks, releases = memory.derive_memory(self.cfg)
        self.assertEqual(len(blocks), 6)
        self.assertEqual(releases["vlsu.vstore"]["mreg_read"]["release_age"], 34)
        self.assertEqual(releases["xlu.transpose"]["mreg_read"]["release_age"], 33)
        self.assertEqual(releases["scalar_lsu.load"]["vmem_read"]["available_age"], 2)

    @unittest.skipUnless(os.environ.get("ATLAS_DERIVE_REFERENCE"), "set ATLAS_DERIVE_REFERENCE for measured parity")
    def test_all_blocks_are_canonical_byte_equal_to_retained_measured_export(self):
        from atlas_derive_vpu import derive_vpu
        expected = json.loads(Path(os.environ["ATLAS_DERIVE_REFERENCE"]).read_text())
        derived = derive_vpu(self.cfg)[0] | mxu.derive_mxu(self.cfg)[0] | memory.derive_memory(self.cfg)[0]
        self.assertEqual(len(derived), 49)
        self.assertEqual(timing.encoded(derived), timing.encoded({b["name"]: b for b in expected["op_timing"]}))

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

    def test_real_vpu_to_mxu_overwrite_rejects_release_edge_accepts_next_age(self):
        import torch
        from io import StringIO
        from unittest.mock import Mock
        from npu_model.hardware.arch_state import ArchState
        from npu_model.hardware.bank_conflict import BankConflictError
        from npu_model.hardware.vpu import VectorExecutionUnit
        from npu_model.hardware.mxu import MatrixExecutionUnitInner
        from npu_model.logging.logger import Logger
        from npu_model.software.instruction import Uop
        from npu_model.util.converter import stream_to_instrs
        for gap in (63, 64):
            with self.subTest(gap=gap):
                state = ArchState(self.cfg.arch_state_config)
                try:
                    for bank in (0, 1):
                        state.mrf[bank].view(torch.bfloat16).fill_(0.5 + bank * 0.25)
                    state.acc["mxu1"][0].fill_(2)
                    expected = [state.mrf[bank].clone() for bank in (0, 1)]
                    vector = VectorExecutionUnit("vpu", Mock(spec=Logger), state, config=self.cfg)
                    matrix = MatrixExecutionUnitInner("mxu1", Mock(spec=Logger), state, config=self.cfg)
                    instructions = stream_to_instrs(StringIO("vmov m4, m0\nvmatpop.bf16.acc.mxu1 m0, acc0\n"))
                    first, second = [Uop(instruction, pc=i * 4) for i, instruction in enumerate(instructions)]
                    def execute():
                        for age in range(gap + 35):
                            vector.tick(first if age == 0 else None)
                            matrix.tick(second if age == gap else None)
                    if gap == 63:
                        with self.assertRaises(BankConflictError):
                            execute()
                    else:
                        execute()
                        self.assertFalse(vector.has_in_flight or matrix.has_in_flight)
                        self.assertTrue(all(torch.equal(state.mrf[bank + 4], expected[bank]) for bank in (0, 1)))
                finally:
                    state.close()


class VPUShapeTest(unittest.TestCase):
    def test_unsupported_source_shape_fails_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "vpu.py"
            source.write_text("def changed_timing():\n    return 0\n")
            with patch.object(derive.inspect, "getsourcefile", return_value=str(source)):
                with self.assertRaisesRegex(derive.UnsupportedVPU, "unsupported VPU timing semantics"):
                    derive._check_source(SimpleNamespace(), latency_table=True)


class DerivedVPUTest(NativeModelTest):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        from npu_model.hardware import vpu
        cls.vpu, cls.timing = vpu, timing

    def test_derivation_never_executes_or_ticks_and_preserves_model_methods(self):
        from npu_model.hardware.arch_state import ArchState
        from npu_model.hardware.exu import StagedExecution
        methods = (self.vpu.StagedExecution, self.vpu.VectorExecutionUnit._accept, StagedExecution._execute)
        with patch.object(ArchState, "__init__", side_effect=AssertionError("architectural state allocated")), \
                patch.object(StagedExecution, "__init__", side_effect=AssertionError("numerical staging executed")), \
                patch.object(self.vpu.VectorExecutionUnit, "tick", side_effect=AssertionError("unit tick executed")), \
                patch.object(self.vpu.VectorExecutionUnit, "_advance", side_effect=AssertionError("unit advanced")):
            self.assertEqual(len(derive.derive_vpu(self.cfg)[0]), 29)
        self.assertEqual(methods, (self.vpu.StagedExecution, self.vpu.VectorExecutionUnit._accept, StagedExecution._execute))


    def test_latency_edits_change_drain_and_reservations_from_source_table(self):
        baseline = derive.derive_vpu(self.cfg)[0]
        with patch.dict(self.vpu.VPU_OP_LATENCIES, {"vadd.bf16": 80}):
            changed, releases = derive.derive_vpu(self.cfg)
        self.assertEqual(changed["vpu.add"]["first_free_age"], 80)
        self.assertEqual(changed["vpu.add"]["next_issue_age"], 80)
        self.assertEqual(changed["vpu.add"]["events"], baseline["vpu.add"]["events"])
        self.assertEqual(releases["vpu.add"]["mreg_write"]["available_age"], 80)
        self.assertEqual(releases["vpu.add"]["mreg_write"]["release_age"], 79)
        self.assertEqual({name: block for name, block in changed.items() if name != "vpu.add"},
                         {name: block for name, block in baseline.items() if name != "vpu.add"})

    def test_reservations_include_banks_before_first_write_and_defer_releases(self):
        releases = derive.derive_vpu(self.cfg)[1]
        self.assertEqual(releases["vpu.add"]["mreg_read"],
                         dict(banks=[0, 1, 2, 3], first_age=0, release_age=63, available_age=64))
        self.assertEqual(releases["vpu.add"]["mreg_write"],
                         dict(banks=[4, 5], first_age=0, release_age=65, available_age=66))
        self.assertEqual(releases["vpu.rsum"]["mreg_read"]["available_age"], 32)
        self.assertEqual(releases["vpu.csum"]["mreg_read"]["available_age"], 128)
        self.assertEqual(releases["vpu.fp8unpack"]["mreg_read"]["banks"], [0])
        self.assertEqual(releases["vpu.fp8pack"]["mreg_write"]["banks"], [4])
        self.assertNotIn("mreg_read", releases["vpu.vliAll"])

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

    @unittest.skipUnless(os.environ.get("ATLAS_DERIVE_OBSERVE"), "set ATLAS_DERIVE_OBSERVE for numerical validation")
    def test_all_blocks_observed_with_noninterference(self):
        blocks, releases = derive.derive_vpu(self.cfg)
        cases = [case for case in self.timing.isa_cases(self.cfg) if case[1] == "vpu"]
        with tempfile.TemporaryDirectory() as directory, patch.object(self.timing, "isa_cases", return_value=cases):
            report, _ = self.timing.validate_derived(self.cfg, blocks, releases, Path(directory))
        self.assertTrue(report["passed"])
        self.assertEqual(report["observed_runs"], 58)

    @unittest.skipUnless(os.environ.get("ATLAS_DERIVE_OBSERVE"), "set ATLAS_DERIVE_OBSERVE for numerical validation")
    def test_changed_latency_matches_observed_drain(self):
        with patch.dict(self.vpu.VPU_OP_LATENCIES, {"vadd.bf16": 80}):
            expected = derive.derive_vpu(self.cfg)[0]["vpu.add"]
            block, _ = self.timing.observe_case("vpu.add", "vpu", "vadd.bf16 m4, m0, m2", nonzero=True)
        self.assertEqual(block, expected)


if __name__ == "__main__":
    unittest.main()
