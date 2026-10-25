"""ISA/resource vocabulary extraction and refusal of stale compiler snapshots."""
from copy import deepcopy
from dataclasses import dataclass
import os
from pathlib import Path
import sys
import unittest
from model_evidence import NativeModelFixture
from unittest.mock import patch

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import atlas_derive_vocabulary as vocabulary
from atlas_model_schema import VOCABULARY, canonical


class VocabularyContractTest(unittest.TestCase):
    def test_unhandled_resource_syntax_fails_closed_even_when_original_read_remains(self):
        class ExponentReg: pass
        @dataclass
        class Instruction:
            es1: ExponentReg
            mnemonic = "test.resource"
            def exec(self, state): pass
        forms = {
            "state aliases": "alias = state\n    alias.write_erf(self.es1, 0)",
            "keyword state accessor": "state.write_erf(reg=self.es1, value=0)",
            "direct state attribute": "state.erf[self.es1] = 0",
            "unresolved helper dispatch": "module.helper(state, self.es1)",
            "keyword helper": "helper(state=state, reg=self.es1)",
            "unresolved register selector": "state.write_erf(int(self.es1), 0)",
            "unresolved register selector for read_erf": "state.read_erf(self.es1 + 1)",
            "unresolved register selector for write_erf": "shifted = self.es1 + 1\n    state.write_erf(shifted, 0)",
            "unresolved helper dispatch receiving a register": "module.helper(self.es1)",
            "state accessor aliases": "writer = state.write_erf\n    writer(self.es1, 0)",
            "AnnAssign": "reg: int = self.es1\n    state.write_erf(reg, 0)",
        }
        for diagnostic, addition in forms.items():
            with self.subTest(syntax=diagnostic):
                source = "def exec(self, state):\n    state.read_erf(self.es1)\n    " + addition + "\n"
                with patch("inspect.getsource", return_value=source), \
                        patch("inspect.getsourcefile", return_value=__file__):
                    with self.assertRaisesRegex(vocabulary.UnsupportedVocabulary, diagnostic):
                        vocabulary._isa_directions(Instruction)

    def test_prohibited_path_refused_before_filesystem_access(self):
        with patch.object(Path, "is_symlink", side_effect=AssertionError("filesystem access")):
            for path in ("/tmp/HAMMER/source.py", "/tmp/vLsI/../source.py"):
                with self.assertRaisesRegex(vocabulary.UnsupportedVocabulary, "prohibited"):
                    vocabulary._safe(path)

    def test_dimensions_come_from_row_requests(self):
        requests = [(age, age // 2, age % 2) for age in range(4)]
        self.assertEqual(vocabulary._dimensions(requests, 0, 2),
                         [dict(offset=0, registers=2, passes=1)])
        self.assertEqual(vocabulary._dimensions(requests + [(age+4, bank, row) for age, bank, row in requests], 0, 2),
                         [dict(offset=0, registers=2, passes=2)])
        simultaneous = [(row, bank, row) for row in range(2) for bank in range(2)]
        self.assertEqual(vocabulary._dimensions(simultaneous, 0, 2),
                         [dict(offset=0, registers=1, passes=1), dict(offset=1, registers=1, passes=1)])

    def test_nonuniform_and_shifted_row_plans_have_precise_refusals(self):
        with self.assertRaisesRegex(vocabulary.UnsupportedVocabulary, "noncontiguous or shifted"):
            vocabulary._dimensions([(0, 1, 0)], 0, 1)
        with self.assertRaisesRegex(vocabulary.UnsupportedVocabulary, "incomplete or nonuniform"):
            vocabulary._dimensions([(0, 0, 0)], 0, 2)

@unittest.skipUnless(os.environ.get("ATLAS_DERIVE_MODEL_ROOT"), "set ATLAS_DERIVE_MODEL_ROOT for ISA source contracts")
class NativeVocabularyTest(NativeModelFixture, unittest.TestCase):
    def test_pinned_model_reproduces_vocabulary_without_numerical_execution(self):
        from npu_model.hardware.arch_state import ArchState
        from npu_model.hardware.exu import StagedExecution
        from npu_model.hardware.mxu import _MatrixExecutionUnit
        from npu_model.hardware.vpu import VectorExecutionUnit
        from npu_model.configs.isa_definition import VADD_BF16
        with patch.object(ArchState, "__init__", side_effect=AssertionError("state allocation")), \
                patch.object(StagedExecution, "__init__", side_effect=AssertionError("numerical staging")), \
                patch.object(_MatrixExecutionUnit, "tick", side_effect=AssertionError("unit tick")), \
                patch.object(VectorExecutionUnit, "tick", side_effect=AssertionError("unit tick")), \
                patch.object(VADD_BF16, "exec", side_effect=AssertionError("instruction execution")):
            self.assertEqual(canonical(vocabulary.validate_vocabulary(self.cfg)), canonical(VOCABULARY))

    def test_isa_types_assembly_order_and_all_resource_directions_are_extracted(self):
        metadata = vocabulary.derive_isa_metadata(self.cfg)
        self.assertEqual(len(metadata), 49)
        self.assertEqual(metadata["vpack.bf16.fp8"]["compiler_operands"], "md m2 e1")
        self.assertEqual(metadata["vmatpop.fp8.acc.mxu0"]["compiler_operands"], "md a2 e1")
        self.assertEqual(metadata["vstore"]["compiler_operands"], "md @")
        memory = {row["name"]: row for row in metadata["vstore"]["operands"]}
        self.assertTrue(memory["vd"]["read"])
        self.assertFalse(memory["vd"]["write"])
        self.assertTrue(memory["rs1"]["read"])
        matmul = {row["name"]: row for row in metadata["vmatmul.acc.mxu0"]["operands"]}
        self.assertEqual((matmul["vs2"]["type"], matmul["vs2"]["resource"], matmul["vs2"]["read"]),
                         ("WeightBuffer", "Weight", True))
        self.assertTrue(matmul["vd"]["read"] and matmul["vd"]["write"])
        self.assertEqual(metadata["vmatpush.weight.mxu0"]["mreg_ports"], [1])

    def test_model_read_pass_change_propagates_then_refuses_stale_snapshot(self):
        from npu_model.hardware import vpu
        original = vpu.VectorExecutionUnit._reads
        def extra_pass(operation, age):
            requests = original(operation, age)
            if operation.uop.insn.mnemonic in vpu._COL_REDUCE and age < 64:
                requests += original(operation, age)
            return requests
        with patch.object(vpu.VectorExecutionUnit, "_reads", staticmethod(extra_pass)):
            derived = vocabulary.derive_vocabulary(self.cfg)
            self.assertEqual(derived["vpu_operand_roles"]["VpuColReduce"][0]["passes"], 3)
            with self.assertRaisesRegex(vocabulary.UnsupportedVocabulary, "stale: vpu_operand_roles"):
                vocabulary.validate_vocabulary(self.cfg)

    def test_isa_resource_type_change_gets_precise_unsupported_error(self):
        from npu_model.isa_types import ExponentReg
        cls = self.cfg.isa.operations["vadd.bf16"]
        with patch.object(cls.__dataclass_fields__["vs1"], "type", ExponentReg):
            with self.assertRaisesRegex(vocabulary.UnsupportedVocabulary, "unmapped reads resource"):
                vocabulary.derive_vocabulary(self.cfg)

    def test_model_exponent_write_requires_a_supported_compiler_adapter(self):
        cls = self.cfg.isa.operations["vpack.bf16.fp8"]
        def write_scale(self, state):
            state.write_erf(self.es1, 0)
        with patch.object(cls, "exec", write_scale):
            with self.assertRaisesRegex(vocabulary.UnsupportedVocabulary, "unsupported exponent resource directions.*write"):
                vocabulary.derive_vocabulary(self.cfg)

    def test_model_dispatch_change_is_refused(self):
        from npu_model.isa import EXU
        cls = self.cfg.isa.operations["vadd.bf16"]
        with patch.object(cls, "exu", EXU.MATRIX_INNER):
            with self.assertRaisesRegex(vocabulary.UnsupportedVocabulary, "unsupported ISA dispatch for vadd.bf16"):
                vocabulary.derive_vocabulary(self.cfg)

    def test_memory_shifted_register_selector_is_refused(self):
        cls = self.cfg.isa.operations["vload"]
        def shifted_destination(self, state):
            state.read_xrf(self.rs1)
            state.write_mrf_u8(self.vd + 1, None)
        with patch.object(cls, "exec", shifted_destination):
            with self.assertRaisesRegex(vocabulary.UnsupportedVocabulary, "unresolved register selector for write_mrf_u8"):
                vocabulary.derive_vocabulary(self.cfg)

    def test_new_model_instruction_requires_explicit_compiler_adapter(self):
        from types import SimpleNamespace
        cfg = deepcopy(self.cfg)
        cfg.isa = SimpleNamespace(operations=dict(self.cfg.isa.operations))
        cfg.isa.operations["vnew.bf16"] = cfg.isa.operations["vadd.bf16"]
        with self.assertRaisesRegex(vocabulary.UnsupportedVocabulary, "added=.*vnew.bf16"):
            vocabulary.derive_vocabulary(cfg)

    def test_memory_dispatch_change_is_refused(self):
        from npu_model.isa import EXU
        cls = self.cfg.isa.operations["vload"]
        with patch.object(cls, "exu", EXU.SCALAR):
            with self.assertRaisesRegex(vocabulary.UnsupportedVocabulary, "unsupported ISA dispatch for vload"):
                vocabulary.derive_vocabulary(self.cfg)

    def test_unknown_model_resource_accessor_is_refused(self):
        cls = self.cfg.isa.operations["vpack.bf16.fp8"]
        def extra_resource(self, state):
            scale = self.es1
            state.read_erf(scale)
            state.read_newfile(scale)
        with patch.object(cls, "exec", extra_resource):
            with self.assertRaisesRegex(vocabulary.UnsupportedVocabulary, "unsupported ISA resource accessor read_newfile"):
                vocabulary.derive_vocabulary(self.cfg)

    def test_extra_unmapped_model_resource_is_refused(self):
        original = vocabulary._issue
        def issue(*args):
            unit, plan, staged = original(*args)
            staged["reads"] = list(staged["reads"]) + [("wb", "mxu0", 0)]
            return unit, plan, staged
        with patch.object(vocabulary, "_issue", issue):
            with self.assertRaisesRegex(vocabulary.UnsupportedVocabulary, "unmapped reads resource"):
                vocabulary.derive_vocabulary(self.cfg)

    def test_handwritten_operand_count_cannot_override_planner(self):
        changed = deepcopy(VOCABULARY)
        changed["vpu_operand_roles"]["VpuPack"][0]["registers"] = 1
        with self.assertRaisesRegex(vocabulary.UnsupportedVocabulary, "stale: vpu_operand_roles"):
            vocabulary.validate_vocabulary(self.cfg, changed)

    def test_bundle_adapter_refuses_unrepresentable_base_or_immediate_fields(self):
        from types import SimpleNamespace
        from npu_model.isa_types import Bundled, ExponentReg, Imm12, Imm16, Named, ScalarReg
        scalar, imm12 = Named(ScalarReg, "rs1"), Named(Imm12, "imm")
        for bundle in (Bundled(Named(ExponentReg, "rs1"), imm12), Bundled(Named(ScalarReg, "rs2"), imm12),
                       Bundled(scalar, Named(Imm16, "imm")), Bundled(scalar, Named(Imm12, "other"))):
            operation = SimpleNamespace(mnemonic="test.bundle", params=[bundle])
            with self.subTest(bundle=bundle.format_arg()), self.assertRaisesRegex(
                    vocabulary.UnsupportedVocabulary, "unsupported bundled assembly operand"):
                vocabulary._compiler_operands(operation)


if __name__ == "__main__":
    unittest.main()
