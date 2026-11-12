"""Hierarchical norm source binding, byte transports, and native templates."""

from dataclasses import replace
import json
import os
from pathlib import Path
import struct
import tempfile
import unittest

from atlas_mlir_native.binding import TARGET_IDENTITY
from atlas_mlir_native.hierarchical_rmsnorm import POLICY, prepare_source, compile_source, pack_partial_columns, runtime_admission
from merlin.tests.ir.test_rmsnorm_source import source_graph

ROOT = Path(__file__).resolve().parents[1]


def source_request(width=2048, rows=1):
    return replace(source_graph((rows,width)),target_identity=TARGET_IDENTITY)


def selected(source, exponent=-7):
    return dict(policy=POLICY,source_request_digest=source.digest(),provenance_sha256="12"*32,
                output_scale_exponent=exponent)


def evaluate_stage(request, inputs):
    """Independent primitive semantic check, never a compiler input."""
    from atlas_mlir_native.reference import (evaluate_bf16_binary_zero, evaluate_bf16_row_sum,
        evaluate_bf16_sqrt, evaluate_bf16_recip, evaluate_rounded_scaled_pack)
    from merlin.semantic_compiler.reference import TensorValue
    values={key:TensorValue(request.node(key).type,tuple(words))for key,words in inputs.items()}
    for binding in request.constants:
        words=tuple(x for(x,)in struct.iter_unpack("<H",bytes.fromhex(binding.data_hex)))
        values[binding.node_id]=TensorValue(request.node(binding.node_id).type,words)
    methods={"row_sum_bf16":evaluate_bf16_row_sum,"sqrt_bf16_bounded":evaluate_bf16_sqrt,
             "recip_bf16_bounded":evaluate_bf16_recip,"pack_fp8":evaluate_rounded_scaled_pack}
    for node in request.nodes:
        if node.effect=="pure":
            operands=tuple(values[key]for key in node.inputs)
            values[node.id]=methods.get(node.op,evaluate_bf16_binary_zero)(node,operands)
    return values[request.outputs[0]].elements


class HierarchicalRMSNormTest(unittest.TestCase):
    def test_policy_shape_source_and_publication(self):
        source=source_request()
        plan=prepare_source(source,selected(source))
        self.assertEqual(plan.launches_per_row_group,132)
        self.assertEqual(plan.region.hidden_consumers,("residual",))
        self.assertEqual([n.op for n in plan.stages["inverse"].nodes if n.effect=="pure"],
            ["mul_bf16_zero","add_bf16_zero","sqrt_bf16_bounded","recip_bf16_bounded"])
        self.assertEqual(plan.stages["publish"].nodes[-1].op,"pack_fp8")
        self.assertEqual(plan.record()["transport"][0]["destination_columns"],list(range(32)))
        self.assertTrue(all(not t["arithmetic"]for t in plan.record()["transport"]))
        rebound=prepare_source(source,selected(source)|{"provenance_sha256":"34"*32})
        self.assertTrue(all(plan.stages[k].digest()!=rebound.stages[k].digest() for k in plan.stages))
        for width in (1024,32768):
            current=source_request(width)
            self.assertEqual(prepare_source(current,selected(current)).width,width)
        for change in ({"source_request_digest":"00"*32},{"provenance_sha256":"bad"},
                       {"output_scale_exponent":-33},{"unexpected":1}):
            with self.assertRaises(ValueError):
                prepare_source(source,selected(source)|change)
        current=source_request(3072)
        with self.assertRaises(ValueError):
            prepare_source(current,selected(current))

    def test_column_transport_copies_only_completed_lane_bytes(self):
        tiles=[b"".join(struct.pack("<H",r*100+c+k*4096)for r in range(32)for c in range(32))for k in range(2)]
        result=struct.unpack("<1024H",pack_partial_columns(tiles))
        for r in range(32):
            self.assertEqual(result[r*32:r*32+32],(r*100,r*100+4096)+(0,)*30)
        with self.assertRaises(ValueError):
            pack_partial_columns([bytes(2047)])

    def test_comparison_only_admission_proves_exact_epsilon_sum(self):
        from atlas_mlir_native.reference import _normal_bf16
        from fractions import Fraction
        epsilon=Fraction(21,2**21)
        for width in (1024,2048,4096,8192,16384,32768):
            admission=runtime_admission(width)
            threshold=_normal_bf16(admission["nonzero_row_abs_max_min_bf16"])
            self.assertGreaterEqual(threshold**2/width,Fraction(1,2**33))
            self.assertEqual(admission["hidden_abs_max_bf16"],0x4000)
        for word in range((127-33)<<7,((127+2)<<7)+1):
            exact=_normal_bf16(word)+epsilon
            self.assertLessEqual(exact.numerator.bit_length(),24)

    def test_width2048_native_semantics_match_original47_arithmetic(self):
        # Direct original47 arithmetic is a separate oracle, not an instruction
        # execution claim. The production compiler never imports npu_model.
        try:
            import torch
            from npu_model.configs.numerics import RtlNumerics as rtl
        except ImportError:
            self.skipTest("original47 arithmetic is an optional external oracle")
        torch.set_num_threads(1)
        source=source_request()
        plan=prepare_source(source,selected(source))
        hidden=[(0x3a80,0x3b00,0x3b80,0x3c00,0x3c80,0x3d00,0xba80,0xbc80)[i%8]for i in range(2048)]
        gamma=[(0x3f80,0x3f00,0x4000,0xbf80)[i%4]for i in range(2048)]
        tile=lambda words:struct.pack("<1024H",*words)
        partials=[]
        for k in range(64):
            words=tuple(hidden[k*32:k*32+32])+ (0,)*992
            partials.append(tile(evaluate_stage(plan.stages["square_reduce"],{"hidden":words})))
        group=[]
        for k in range(2):
            words=struct.unpack("<1024H",pack_partial_columns(partials[k*32:k*32+32]))
            group.append(tile(evaluate_stage(plan.stages["reduce"],{"partials":words})))
        total=evaluate_stage(plan.stages["reduce"],{"partials":struct.unpack("<1024H",pack_partial_columns(group))})
        inverse=evaluate_stage(plan.stages["inverse"],{"total":total})
        output=[]
        for k in range(64):
            words=tuple(hidden[k*32:k*32+32])+(0,)*992
            weights=tuple(gamma[k*32:k*32+32])*32
            result=evaluate_stage(plan.stages["publish"],{"hidden":words,"inverse":inverse,"gamma":weights})
            self.assertTrue(all(x==0 for x in result[32:]))
            output.extend(result[:32])
        def tensor(words):
            return torch.tensor(words,dtype=torch.uint16).view(torch.bfloat16)
        h,g=tensor(hidden),tensor(gamma)
        square=rtl.mul(h,h).reshape(2,32,32)
        partial=rtl.row_sum(square).reshape(2,32)
        groups=rtl.row_sum(partial).reshape(2)
        padded=torch.zeros(32,dtype=torch.bfloat16);padded[:2]=groups
        summed=rtl.row_sum(padded)
        mean=rtl.mul(summed,tensor([0x3a00]))
        shifted=rtl.add(mean,tensor([0x3728]))
        inv=rtl.unary("rcp",rtl.unary("sqrt",shifted))
        normalized=rtl.mul(rtl.mul(h,inv),g)
        reference=rtl.to_fp8(rtl.mul(normalized,tensor([0x4300])),127).view(torch.uint8).tolist()
        self.assertEqual(list(output),reference)
        self.assertEqual(inverse[0],inv.view(torch.uint16).item())
        # The operation-domain check refuses a rounded FP32 epsilon addition.
        with self.assertRaises(ValueError):
            evaluate_stage(plan.stages["inverse"],{"total":(0x5f80,)*1024})


@unittest.skipUnless(os.environ.get("MERLIN_NATIVE_ROOT") and os.environ.get("ATLAS_OOT_BIN_DIR"),"select native tools")
class HierarchicalRMSNormNativeTest(unittest.TestCase):
    def test_four_templates_compile_through_native_pipeline(self):
        from atlas_mlir_native.binding import AtlasNativeBinding
        from merlin.semantic_compiler.snapshot import build_native_snapshot,packaged_egg_bridge_path
        from merlin.semantic_compiler.search import SearchLimits
        from merlin.semantic_compiler.target_binding import verify_native_publication
        retained=os.environ.get("ATLAS_RMSNORM_TEST_OUT")
        temporary=None if retained else tempfile.TemporaryDirectory(dir=os.environ.get("TMPDIR"))
        work=Path(retained or temporary.name);work.mkdir(parents=True,exist_ok=True)
        try:
            snapshot=build_native_snapshot(AtlasNativeBinding.profile(),destination=work/"snapshot",
                crate=packaged_egg_bridge_path(),cargo_target_dir=Path(os.environ.get("ATLAS_MXU1_CARGO_TARGET",work/"cargo")),
                source_revision="hierarchical-rmsnorm-templates")
            source=source_request()
            policy=selected(source)
            plan=prepare_source(source,policy)
            record=compile_source(snapshot,source,policy,target_source=ROOT,destination=work/"norm",
                limits=SearchLimits(candidate_nodes=256,wall_timeout_s=120))
            for kind,request in plan.stages.items():
                directory=work/"norm"/kind;manifest=json.loads((directory/"manifest.json").read_text())
                verify_native_publication(directory,manifest,engine="merlin_native",request_digest=request.digest(),target_identity=TARGET_IDENTITY)
                self.assertEqual(record["compiled_stages"][kind]["request_digest"],request.digest())
                self.assertTrue((directory/"program.bin").stat().st_size>0)
        finally:
            if temporary:
                temporary.cleanup()


if __name__=="__main__":
    unittest.main()
