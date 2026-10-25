"""Canonical proposal operands and source-checked timing-class invariance."""
import ast
import os
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from atlas_derive_costs import UnsupportedCosts, check_timing_conditions, derive_costs
from model_evidence import isolate_model_imports


class PredicateProofTest(unittest.TestCase):
    def test_register_selectors_may_address_rows_but_not_choose_timing(self):
        valid = ast.parse("def rows(insn, age):\n bank = insn.vd\n return [(bank, age)] if age < 32 else []").body[0]
        check_timing_conditions(valid, {"vd"})
        invalid = ast.parse("def rows(insn, age):\n bank = insn.vd\n last = bank\n return [(bank, age)] if age < last else []").body[0]
        with self.assertRaisesRegex(UnsupportedCosts, "register-dependent timing condition"):
            check_timing_conditions(invalid, {"vd"})
        unpacked = ast.parse("def rows(insn, age):\n name, bank = ('op', insn.vd)\n return [(bank, age)] if age < bank else []").body[0]
        with self.assertRaisesRegex(UnsupportedCosts, "register-dependent timing condition"):
            check_timing_conditions(unpacked, {"vd"})


@unittest.skipUnless(os.environ.get("ATLAS_DERIVE_MODEL_ROOT"), "set ATLAS_DERIVE_MODEL_ROOT for native source contracts")
class SourceCostTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        isolate_model_imports(cls, os.environ["ATLAS_DERIVE_MODEL_ROOT"])
        from atlas_model_timing import configuration
        from atlas_derive_arch import derive_arch
        from atlas_derive_vocabulary import derive_isa_metadata
        cls.cfg, _ = configuration()
        cls.architecture = derive_arch(cls.cfg)
        cls.metadata = derive_isa_metadata(cls.cfg)

    def derive(self, **kwargs):
        return derive_costs(self.cfg, kwargs.get("architecture", self.architecture),
                            kwargs.get("metadata", self.metadata))

    def test_planner_source_mutation_refuses_without_pin_refresh(self):
        import atlas_derive_vpu as vpu
        with patch.dict(vpu.SOURCE_SHAPES, {"vpu.py": "0" * 64}):
            with self.assertRaisesRegex(vpu.UnsupportedVPU, "unsupported VPU timing semantics"):
                self.derive()

    def test_geometry_span_drives_placements_and_capacity_fails_closed(self):
        wider = dict(self.architecture, bf16_registers_per_tile=4)
        self.assertEqual(self.derive(architecture=wider)["probes"]["vpack.bf16.fp8"][1]["rs2"], 20)
        small = dict(self.architecture, mreg_count=4)
        with self.assertRaisesRegex(UnsupportedCosts, "canonical disjoint cost probes exceed"):
            self.derive(architecture=small)

if __name__ == "__main__":
    unittest.main()
