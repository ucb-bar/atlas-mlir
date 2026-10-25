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
from atlas_model_facts import identity, note_unqualified_model, overlap_bundle
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
    def test_qualification_note_is_quiet_for_pins_and_nonfatal_for_changed_content(self):
        source, configuration = LOCK["source_sha256"], LOCK["configuration_sha256"]
        cases = [(source, configuration, False), ("0" * 64, configuration, True),
                 (source, "0" * 64, True), ("0" * 64, "0" * 64, True)]
        for actual_source, actual_configuration, changed in cases:
            with self.subTest(source=actual_source, configuration=actual_configuration):
                output = StringIO()
                with redirect_stderr(output):
                    self.assertIsNone(note_unqualified_model(actual_source, actual_configuration))
                lines = output.getvalue().splitlines()
                self.assertEqual(len(lines), int(changed))
                if changed:
                    self.assertIn("results are not from the qualified revision", lines[0])

    def test_overlap_gate_failure_never_publishes_bundle_lock(self):
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory)
            (out / "model-timing.json").write_text("{}")
            with patch("atlas_derive_overlap.generate", return_value={"pairs": [object()]}), \
                 patch("atlas_validate_overlap.validate", side_effect=RuntimeError("negative boundary failed")):
                with self.assertRaisesRegex(ModelTimingValidationError, "negative boundary failed"):
                    overlap_bundle(out, out)
            self.assertFalse((out / "facts.lock.json").exists())
            self.assertFalse((out / "model-overlap.json").exists())

    def test_overlap_pipeline_validates_frozen_bound_proposal_before_returning_digests(self):
        from test_derived_contract import overlap_fixture
        from atlas_derived_contract import overlap_semantics
        rules = overlap_fixture()
        digest = timing.sha256(timing.encoded(overlap_semantics(rules)))
        for bound in (digest, "0" * 64):
            with self.subTest(bound=bound), tempfile.TemporaryDirectory() as directory:
                out, calls = Path(directory), []
                (out / "model-timing.json").write_text(json.dumps(dict(overlap_rules_sha256=bound)))
                def generate(model, facts, candidate, samples):
                    calls.append("derive")
                    candidate.write_text("frozen source rules")
                    return rules
                def validate(model, facts, candidate, samples, reports, final, **options):
                    self.assertEqual((candidate.read_text(), calls, options),
                                     ("frozen source rules", ["derive"], {"workers": 1}))
                    reports.mkdir()
                    (reports / "validation-report.json").write_text("native gate passed")
                    final.write_text("qualified source rules")
                    calls.append("validate")
                    return {"seconds": 1}
                with patch("atlas_derive_overlap.generate", side_effect=generate), \
                        patch("atlas_validate_overlap.validate", side_effect=validate) as gate:
                    if bound != digest:
                        with self.assertRaisesRegex(ValueError, "overlap proposal differs from the sealed derived semantics"):
                            overlap_bundle(out, out)
                        gate.assert_not_called()
                        self.assertFalse((out / "model-overlap.json").exists())
                    else:
                        result = overlap_bundle(out, out)
                        self.assertEqual(calls, ["derive", "validate"])
                        self.assertEqual(result["overlap_sha256"], timing.sha256(b"qualified source rules"))
                        self.assertEqual(result["overlap_validation_sha256"], timing.sha256(b"native gate passed"))

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
        self.assertEqual(changed["vpu.add"]["next_issue_age"], 80)
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

    @unittest.skipUnless(os.environ.get("ATLAS_DERIVE_OBSERVE"), "set ATLAS_DERIVE_OBSERVE for numerical validation")
    def test_all_blocks_observed_with_noninterference(self):
        blocks, releases = {}, {}
        for derive_blocks in (derive.derive_vpu, mxu.derive_mxu, memory.derive_memory):
            values, reservations = derive_blocks(self.cfg)
            blocks.update(values)
            releases.update(reservations)
        with tempfile.TemporaryDirectory() as directory:
            report, _ = self.timing.validate_derived(self.cfg, blocks, releases, Path(directory))
        self.assertTrue(report["passed"])
        self.assertEqual(report["observed_runs"], 98)

    @unittest.skipUnless(os.environ.get("ATLAS_DERIVE_OBSERVE"), "set ATLAS_DERIVE_OBSERVE for numerical validation")
    def test_changed_latency_matches_observed_drain(self):
        with patch.dict(self.vpu.VPU_OP_LATENCIES, {"vadd.bf16": 80}):
            expected = derive.derive_vpu(self.cfg)[0]["vpu.add"]
            block, _ = self.timing.observe_case("vpu.add", "vpu", "vadd.bf16 m4, m0, m2", nonzero=True)
        self.assertEqual(block, expected)


if __name__ == "__main__":
    unittest.main()
