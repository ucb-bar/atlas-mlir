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

    def test_default_probe_signature_matches_previous_adapter(self):
        from npu_model.isa import EXU
        value = self.derive()
        self.assertEqual(value["timing_class"]["interval_scope"], "selected_representative_pair_only")
        for name, pair in value["probes"].items():
            metadata = self.metadata.get(name)
            first, second = dict(rd=0, rs1=2, rs2=4, imm=0), dict(rd=6, rs1=8, rs2=10, imm=0)
            if self.cfg.isa.operations[name].exu == EXU.SCALAR:
                fields = {row.name for row in __import__("dataclasses").fields(self.cfg.isa.operations[name])}
                for field in ("rs1", "rs2"):
                    if field not in fields:
                        first[field] = second[field] = 0
            op_class = metadata.get("op_class") if metadata else None
            if op_class in {"WeightPush", "AccPushFp8", "AccPushBf16"}:
                first, second = dict(rd=0, rs1=0, rs2=0, imm=0), dict(rd=1, rs1=2, rs2=0, imm=0)
            elif op_class in {"PopFp8", "PopBf16"}:
                first, second = dict(rd=0, rs1=0, rs2=0, imm=0), dict(rd=2, rs1=0, rs2=1, imm=0)
            elif op_class in {"MatMul", "MatMulAcc"}:
                first, second = dict(rd=0, rs1=0, rs2=0, imm=0), dict(rd=1, rs1=2, rs2=1, imm=0)
            elif name == "sw":
                first["rd"] = second["rd"] = 0
            elif name in {"vload", "vstore", "lw", "seld", "vtrpose.xlu"} or op_class in {"VpuRowReduce", "VpuColReduce"}:
                first["rs2"] = second["rs2"] = 0
            elif op_class == "VpuElementwise" and "vs2" not in {row["name"] for row in metadata["operands"]}:
                first["rs2"] = second["rs2"] = 0
            elif op_class in {"VpuLoadImmPair", "VpuLoadImmSingle"}:
                for field in ("rs1", "rs2"):
                    first[field] = second[field] = 0
            with self.subTest(name=name):
                self.assertEqual(pair, [first, second])

    def test_geometry_span_drives_placements_and_capacity_fails_closed(self):
        wider = dict(self.architecture, bf16_registers_per_tile=4)
        self.assertEqual(self.derive(architecture=wider)["probes"]["vpack.bf16.fp8"][1]["rs2"], 20)
        small = dict(self.architecture, mreg_count=4)
        with self.assertRaisesRegex(UnsupportedCosts, "canonical disjoint cost probes exceed"):
            self.derive(architecture=small)

    def test_planner_source_mutation_refuses_without_pin_refresh(self):
        from npu_model.hardware import vpu
        import atlas_derive_vpu
        original = atlas_derive_vpu.SOURCE_SHAPES["vpu.py"]
        try:
            atlas_derive_vpu.SOURCE_SHAPES["vpu.py"] = "0" * 64
            with self.assertRaisesRegex(Exception, "unsupported VPU timing semantics"):
                self.derive()
        finally:
            atlas_derive_vpu.SOURCE_SHAPES["vpu.py"] = original

    def test_derivation_does_not_run_core_or_numerics(self):
        from npu_model.hardware.core import Core
        from npu_model.hardware.arch_state import ArchState
        with patch.object(Core, "tick", side_effect=AssertionError("model execution")), \
                patch.object(ArchState, "__init__", side_effect=AssertionError("numerical state")):
            self.assertTrue(self.derive()["probes"])


if __name__ == "__main__":
    unittest.main()
