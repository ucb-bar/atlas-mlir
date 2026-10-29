"""Derived cost probes and their operand-invariance refusals."""
import ast
from dataclasses import replace
import os
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from atlas_derive_costs import check_timing_conditions, derive_costs
from atlas_derive_arch import derive_arch
from atlas_model_diagnostics import UnsupportedModelTiming
from model_evidence import NativeModelFixture


class InvarianceTest(unittest.TestCase):
    def test_register_predicates_refuse(self):
        for condition in ("insn.vd == 0", "alias == 0"):
            tree = ast.parse("def planner(insn):\n    alias = insn.vd\n    if " + condition + ":\n        return 1\n")
            with self.subTest(condition=condition), self.assertRaises(UnsupportedModelTiming):
                check_timing_conditions(tree.body[0], {"vd"})


@unittest.skipUnless(os.environ.get("ATLAS_DERIVE_MODEL_ROOT"), "set ATLAS_DERIVE_MODEL_ROOT")
class CostProbeTest(NativeModelFixture, unittest.TestCase):
    def test_geometry_and_isa_operands(self):
        architecture = derive_arch(self.cfg)
        pairs = derive_costs(self.cfg, architecture)
        self.assertEqual(len(pairs), 89)
        self.assertEqual(pairs["vadd.bf16"][1], dict(rd=6, rs1=8, rs2=10, imm=0))
        self.assertEqual(pairs["vmatpop.bf16.acc.mxu0"][1], dict(rd=2, rs1=0, rs2=1, imm=0))
        wider = derive_costs(self.cfg, dict(architecture, bf16_registers_per_tile=4))
        self.assertEqual(wider["vpack.bf16.fp8"][1]["rs2"], 20)
        for values in (dict(architecture, mreg_count=4), dict(architecture, bf16_registers_per_tile=0)):
            with self.subTest(values=values), self.assertRaises(UnsupportedModelTiming):
                derive_costs(self.cfg, values)

    def test_imported_model_path_refuses(self):
        import npu_model
        import tempfile
        from unittest.mock import patch
        from atlas_model_facts import regenerate
        with tempfile.TemporaryDirectory() as directory, \
                patch.object(npu_model, "__file__", "/wrong/npu_model/__init__.py"):
            with self.assertRaisesRegex(UnsupportedModelTiming, "imported npu_model differs"):
                regenerate(os.environ["ATLAS_DERIVE_MODEL_ROOT"], Path(directory) / "facts")
