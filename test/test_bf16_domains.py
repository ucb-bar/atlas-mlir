"""Separate masked-score SUB, nonpositive EXP, and finite gradual MUL domains."""
import gzip
import hashlib
import json
import os
from pathlib import Path
import struct
import sys
import tempfile
import unittest
from dataclasses import replace

ROOT = Path(__file__).resolve().parents[1]
MERLIN = os.environ.get("MERLIN_NATIVE_ROOT")
BIN = os.environ.get("ATLAS_OOT_BIN_DIR")


def request_for(op):
    from atlas_mlir_native.bf16_domains import build_bf16_primitive_request
    return build_bf16_primitive_request(op)


class BF16DomainsReferenceTest(unittest.TestCase):
    def test_exp_gradual_underflow_boundaries_and_legacy_refusal(self):
        from atlas_mlir_native.bf16_domains import vexp_bf16_nonpositive_bits as exp
        from atlas_mlir_native.exp_lane import vexp_bf16_bits as legacy
        for code, expected in ((0xc2ae,0x00b4),(0xc2af,0x006d),(0xc2b0,0x0042),(0xc2b8,1),
                               (0xc2c0,0),(0xc300,0),(0xff80,0),(0,0x3f80),(0x8000,0x3f80),(0x8001,0x3f80)):
            self.assertEqual(exp(code), expected)
        for code in (0x3f80,0x0001,0xc301,0x7f80,0x7fc0,0xffc0,True):
            with self.assertRaises(ValueError):
                exp(code)
        for code in (0xc2af,0xc300,0xff80):
            with self.assertRaises(ValueError):
                legacy(code)

    def test_exp_entire_admitted_domain_matches_pinned_rom_if_available(self):
        from atlas_mlir_native.bf16_domains import vexp_bf16_nonpositive_bits as exp
        rom = os.environ.get("ATLAS_BF16_EXP_ROM")
        if not rom:
            self.skipTest("set ATLAS_BF16_EXP_ROM for pinned original47 RTL table comparison")
        data = Path(rom).read_bytes()
        self.assertEqual(hashlib.sha256(data).hexdigest(), "c868283c0118ac478b2acd44af4e10d5c02bcd236d73e188e1c7ce33f112e566")
        expected = struct.unpack("<65536H", gzip.decompress(data))
        codes = (0,0xff80,*range(0x8000,0xc301))
        for code in codes:
            self.assertEqual(exp(code), expected[code], hex(code))

    def test_finite_mul_signed_zero_ties_subnormals_overflow(self):
        from atlas_mlir_native.bf16_domains import mul_bf16_finite_bits as mul
        self.assertEqual(mul(0x0042,0x4380),0x0404)  # EXP(-88)*256 is normal.
        for a,b,expected in ((1,0x3f00,0),(3,0x3f00,2),(0x8001,0x3f00,0x8000),
                             (0x8000,0x3f80,0x8000),(0x8000,0xbf80,0),
                             (0x0080,0x3f00,0x0040),(0x7f7f,0x4000,0x7f80)):
            self.assertEqual(mul(a,b),expected)
        for code in (0x7f80,0xff80,0x7fc0,0xffc0,True,-1,65536):
            with self.assertRaises(ValueError):
                mul(code,0x3f80)

    def test_sub_asymmetric_infinity_and_callback_contracts(self):
        from atlas_mlir_native.reference import evaluate_bf16_sub_nonnan, evaluate_bf16_mul_finite, evaluate_bf16_exp_nonpositive
        from merlin.semantic_compiler.reference import TensorValue
        request = request_for("sub_bf16_nonnan")
        node = request.nodes[-1]
        pairs = ((0x3f80,0x7f80,0xff80),(0xff80,0x3f80,0xff80),(0xff80,0x7f80,0xff80),
                 (0x8000,0,0x8000),(0x3f80,0x3f80,0),(0x7f7f,0xff7f,0x7f80),(1,0,1),(0,0x8001,1))
        for lhs,rhs,expected in pairs:
            args = tuple(TensorValue(node.type,(code,)*1024) for code in (lhs,rhs))
            self.assertEqual(set(evaluate_bf16_sub_nonnan(node,args).elements),{expected})
        for lhs,rhs in ((0x7f80,0),(0,0xff80),(0x7fc0,0),(0,0xffc0)):
            with self.assertRaises(ValueError):
                evaluate_bf16_sub_nonnan(node,tuple(TensorValue(node.type,(code,)*1024) for code in (lhs,rhs)))
        for op, callback, args in (("exp_bf16_nonpositive",evaluate_bf16_exp_nonpositive,(0xc2b0,)),
                                   ("mul_bf16_finite",evaluate_bf16_mul_finite,(0x0042,0x4380))):
            node = request_for(op).nodes[-1]
            values = tuple(TensorValue(node.type,(code,)*1024) for code in args)
            self.assertEqual(len(callback(node,values).elements),1024)
            with self.assertRaises(ValueError):
                callback(replace(node,attrs=()),values)


@unittest.skipUnless(MERLIN and BIN, "set MERLIN_NATIVE_ROOT and ATLAS_OOT_BIN_DIR")
class BF16DomainsNativeTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        sys.path[:0] = [str(ROOT / "python"), str(Path(MERLIN) / "src")]
        from atlas_mlir_native.binding import AtlasNativeBinding
        from merlin.semantic_compiler.snapshot import build_native_snapshot,packaged_egg_bridge_path
        cls.tmp = tempfile.TemporaryDirectory()
        cls.work = Path(os.environ.get("ATLAS_BF16_DOMAIN_QUALIFICATION_DIR",cls.tmp.name))
        cls.work.mkdir(parents=True,exist_ok=True)
        cls.binding = AtlasNativeBinding
        cls.snapshot = build_native_snapshot(cls.binding.profile(),destination=cls.work/"snapshot",
            crate=packaged_egg_bridge_path(),cargo_target_dir=Path(os.environ.get("CARGO_TARGET_DIR",cls.tmp.name))/"mxu1-cargo",
            source_revision="separate-masked-sub-nonpositive-exp-finite-mul-domains")

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()
        sys.path.remove(str(ROOT/"python"))
        sys.path.remove(str(Path(MERLIN)/"src"))

    def compile(self, request, name):
        from merlin.semantic_compiler.search import SearchLimits
        from merlin.semantic_compiler.target_binding import verify_native_publication
        fixed = {node.id:i+3 for i,node in enumerate(request.nodes[:-1])}
        manifest = self.binding.compile(self.snapshot,request,fixed_inputs=fixed,fixed_outputs=(35,),
            target_source=ROOT,destination=self.work/name,limits=SearchLimits())
        verify_native_publication(self.work/name,manifest,engine="merlin_native",request_digest=request.digest(),target_identity=request.target_identity)
        return json.loads((self.work/name/"physical-program.json").read_bytes())

    def test_selected_native_domains_and_constant_mask_roles(self):
        from merlin.semantic_compiler.model import ConstantBinding
        from merlin.semantic_compiler.target_binding import NativeCompilationError
        for op,mode in (("sub_bf16_nonnan","sub"),("exp_bf16_nonpositive","exp"),("mul_bf16_finite","mul")):
            program = self.compile(request_for(op),op)
            rows = [row for row in program["instructions"] if row["operation"] in ("atlas.vpu_unary","atlas.vpu_binary")]
            self.assertEqual([row["fields"]["kind"] for row in rows],[mode])
        base = request_for("sub_bf16_nonnan")
        for port in (0,1):
            nodes = list(base.nodes)
            nodes[port] = replace(nodes[port],op="constant",effect="constant")
            request = replace(base,nodes=tuple(nodes),input_storages=tuple(item for item in base.input_storages if item[0] != nodes[port].id),
                              constants=(ConstantBinding(nodes[port].id,"bf16-le-raw",(b"\x80\x7f"*1024).hex()),))
            if port == 0:
                with self.assertRaises(NativeCompilationError):
                    self.compile(request,"invalid-lhs-posinf")
            else:
                self.compile(request,"rhs-posinf-mask")


if __name__ == "__main__":
    unittest.main()
