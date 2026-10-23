"""DMA issue qualification refuses changed predicates and failed native gates."""
from copy import deepcopy
import os
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from atlas_derive_dma_admission import (SOURCE_NAMES, UnsupportedDMAAdmission,
                                        derive_dma_admission, review_sources)
from atlas_validate_dma_admission import validate_dma_admission, _semantics, sample_plan
from atlas_model_diagnostics import ModelTimingValidationError
from model_evidence import isolate_model_imports


@unittest.skipUnless(os.environ.get("ATLAS_DERIVE_MODEL_ROOT"), "set ATLAS_DERIVE_MODEL_ROOT for DMA source/native qualification")
class DMAAdmissionTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = Path(os.environ["ATLAS_DERIVE_MODEL_ROOT"])
        isolate_model_imports(cls, str(cls.model))
        from atlas_model_timing import configuration
        from atlas_derive_curve import derive_curve
        cls.cfg, config = configuration()
        cls.facts = dict(model=dict(configuration=config), dma_timing=derive_curve(cls.cfg))

    def test_derivation_allocates_no_core_and_executes_no_dma(self):
        from npu_model.hardware.core import Core
        from npu_model.hardware.dma import DmaExecutionUnit
        with patch.object(Core, "__init__", side_effect=AssertionError("allocated Core")), \
                patch.object(DmaExecutionUnit, "tick", side_effect=AssertionError("executed DMA")):
            candidate = derive_dma_admission(self.model, self.facts)
        self.assertFalse(candidate["validation"]["passed"])
        self.assertEqual(candidate["issue"], dict(ordinary_gap=1, first_engine_age=1, store_read_to_request_cycles=2))
        self.assertEqual(candidate["semantics"]["fixed_dma_windows"], "none")

    def test_changed_channel_wait_grant_and_completion_predicates_refuse(self):
        sources = {name: (self.model / "npu_model" / "hardware" / name).read_text() for name in SOURCE_NAMES}
        mutations = [("core.py", "assert not state.check_flag(insn.funct3)", "assert state.check_flag(insn.funct3)"),
                     ("core.py", 'self.stall_reason = "dma.wait"', 'self.stall_reason = "delay"'),
                     ("dma.py", "bank != write_bank", "bank == write_bank"),
                     ("dma.py", "if dispatched and will_be_zero:", "if dispatched or will_be_zero:")]
        for filename, original, replacement in mutations:
            self.assertIn(original, sources[filename])
            changed = dict(sources, **{filename: sources[filename].replace(original, replacement, 1)})
            with self.subTest(filename=filename, predicate=original), self.assertRaises(UnsupportedDMAAdmission):
                review_sources(changed)

    def test_semantic_tampering_rejected_before_native_execution(self):
        candidate = derive_dma_admission(self.model, self.facts)
        candidate["semantics"]["channel_reuse"] = "no_wait_needed"
        with patch("atlas_validate_dma_admission._run", side_effect=AssertionError("ran changed candidate")):
            with self.assertRaisesRegex(ModelTimingValidationError, "differs from selected source"):
                validate_dma_admission(self.model, self.facts, candidate)

    def test_changed_capacity_is_bound_in_admission(self):
        candidate = derive_dma_admission(self.model, self.facts)
        changed = deepcopy(self.facts)
        changed["model"]["configuration"]["dma_max_in_flight"] += 1
        other = derive_dma_admission(self.model, changed)
        self.assertNotEqual(_semantics(candidate), _semantics(other))

    def test_serial_state_failure_cannot_publish_validation(self):
        candidate = derive_dma_admission(self.model, self.facts)
        sample = sample_plan(self.facts)[0]
        with patch("atlas_validate_dma_admission.sample_plan", return_value=[sample]), \
                patch("atlas_validate_dma_admission._run", side_effect=[dict(state_sha256="a"), dict(state_sha256="b")]):
            with self.assertRaisesRegex(ModelTimingValidationError, "serial numerical mismatch"):
                validate_dma_admission(self.model, self.facts, candidate)
        self.assertFalse(candidate["validation"]["passed"])

    def test_full_native_issue_numeric_and_rejection_gate(self):
        candidate = derive_dma_admission(self.model, self.facts)
        qualified, report = validate_dma_admission(self.model, self.facts, candidate)
        self.assertEqual(_semantics(qualified), _semantics(candidate))
        self.assertTrue(report["inputs_unchanged"])
        self.assertTrue(qualified["validation"]["passed"])
        self.assertGreaterEqual(report["negative_runs"], 6)
        self.assertEqual(report["positive_runs"], report["numerical_checks"])
        self.assertTrue(any(run.get("denied_grants", 0) for run in report["runs"]))
        self.assertFalse(candidate["validation"]["passed"])


if __name__ == "__main__":
    unittest.main()
