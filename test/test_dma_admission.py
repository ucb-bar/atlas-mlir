"""DMA source extraction refuses unsupported predicates and configuration."""
from copy import deepcopy
import os
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from atlas_derive_dma_admission import (SOURCE_NAMES, UnsupportedDMAAdmission,
                                        derive_dma_admission, review_sources)
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

    def test_changed_capacity_is_bound_in_admission(self):
        candidate = derive_dma_admission(self.model, self.facts)
        changed = deepcopy(self.facts)
        changed["model"]["configuration"]["dma_max_in_flight"] += 1
        with self.assertRaisesRegex(UnsupportedDMAAdmission, "authoritative geometry"):
            derive_dma_admission(self.model, changed)
        changed["dma_timing"]["max_in_flight"] += 1
        other = derive_dma_admission(self.model, changed)
        self.assertNotEqual(candidate["configuration"], other["configuration"])

    def test_shared_source_phase_and_startup_disagreement_refuse(self):
        for field, value in (("source_contract", {}), ("startup_cycles", dict(load=2, store=3)),
                             ("issue_interval_cycles", 2)):
            changed = deepcopy(self.facts)
            changed["dma_timing"][field] = value
            with self.subTest(field=field), self.assertRaisesRegex(UnsupportedDMAAdmission, "authoritative"):
                derive_dma_admission(self.model, changed)
        changed = deepcopy(self.facts)
        changed["dma_timing"]["proposal_contract"]["source_evidence"]["edge_order"].reverse()
        with self.assertRaisesRegex(UnsupportedDMAAdmission, "phase fields"):
            derive_dma_admission(self.model, changed)

if __name__ == "__main__":
    unittest.main()
