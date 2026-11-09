"""Standalone source exponential keeps its explicit precision and operation."""
from dataclasses import replace
import unittest
import numpy as np
from atlas_mlir_native.binding import TARGET_IDENTITY, BF16_EXP_ATTRS
from atlas_mlir_native.source_exp import POLICY, input_spec, prepare_source
from merlin.semantic_compiler.model import ConstantBinding, KernelRequest, SemanticNode, TensorType, pointwise_identity_maps


def source_request(shape=(3,7)):
    type_=TensorType(shape,'f32','raw-bit-pattern')
    x=SemanticNode('x','input',(),type_,effect='input')
    y=SemanticNode('y','source_f32_exp_unqualified',('x',),type_,
        (('numeric_semantics','source-f32-exp-unqualified-v1'),),index_maps=pointwise_identity_maps(len(shape),1))
    return KernelRequest((x,y),('y',),('external',),(('x','external'),),TARGET_IDENTITY,'diagnostic')


class SourceExpTest(unittest.TestCase):
    def test_source_remains_one_exp_with_bounded_conversion(self):
        from atlas_mlir_native.pointwise import prepare_plan, virtual_program
        source=source_request()
        precision=prepare_source(source,POLICY)
        self.assertEqual(precision.source,source)
        self.assertEqual([node.op for node in precision.request.nodes],['input','exp_bf16'])
        self.assertEqual(precision.request.nodes[-1].attrs,BF16_EXP_ATTRS)
        self.assertEqual(precision.record()['output_conversion'],'bf16-to-f32-exact')
        plan=prepare_plan(precision.request)
        self.assertEqual(plan.body_request().nodes[-1].op,'exp_bf16')
        text=virtual_program(plan,(0x90000000,),0x90001000,'atlas.vpu_exp_bf16_tile')
        self.assertIn('kind = "exp"',text)
        self.assertNotIn('sigmoid',text)
        np.testing.assert_array_equal(input_spec().encode(np.array([-32.,-0.,0.,32.],np.float32)),
                                      np.array([0xc200,0x8000,0,0x4200],np.uint16))
        for value in (33.,-33.,np.inf,np.nan,np.float32(1e-40)):
            with self.assertRaises(ValueError): input_spec().encode(np.array([value],np.float32))

    def test_exp_needs_exact_source_and_separate_selection(self):
        from atlas_mlir_native.pointwise import prepare_source as prepare_pointwise
        source=source_request()
        for policy in (None,'use-host-exp'):
            with self.assertRaisesRegex(ValueError,'explicitly selected'): prepare_source(source,policy)
        with self.assertRaises(ValueError): prepare_pointwise(source,allow_math=True)
        self.assertEqual(prepare_pointwise(source,exp_policy=POLICY).request.nodes[-1].op,'exp_bf16')
        output=source.nodes[-1]
        for node in (replace(output,attrs=(('numeric_semantics','source-f32-exp-other'),)),
                     replace(output,index_maps=()),replace(output,op='source_f32_composite_unqualified')):
            with self.assertRaises(ValueError): prepare_source(replace(source,nodes=(source.nodes[0],node)),POLICY)

    def test_constants_are_preflighted_before_native_selection(self):
        source=source_request(shape=(1,1))
        x=replace(source.nodes[0],op='constant',effect='constant')
        changed=replace(source,nodes=(x,source.nodes[-1]),input_storages=(),
            constants=(ConstantBinding('x','f32-le-raw',np.array([64.],np.float32).tobytes().hex()),))
        with self.assertRaises(ValueError): prepare_source(changed,POLICY)

    def test_application_policy_admits_exp_only_with_already_quantized(self):
        from atlas_mlir_native.application import AtlasApplicationAdapter
        from atlas_mlir_native.already_quantized import POLICY as INTAKE_POLICY
        from atlas_mlir_native.application_precision import CONTRACTION_POLICY
        from atlas_mlir_native.row_softmax import SOFTMAX_POLICY
        base=dict(schema='atlas.diagnostic_application_precision.v1',source_sha256='0'*64,target_identity=TARGET_IDENTITY,
            contraction=CONTRACTION_POLICY,row_softmax=SOFTMAX_POLICY,host_support=['transpose','reshape','copy'])
        with self.assertRaises(Exception): AtlasApplicationAdapter({**base,'source_exp':POLICY},original_source_sha256='0'*64)
        selected=AtlasApplicationAdapter({**base,'already_quantized':INTAKE_POLICY,'source_exp':POLICY},original_source_sha256='0'*64)
        region=selected.plan(source_request()).selected[0]
        self.assertEqual(region.placement,'accelerator')
        self.assertEqual(dict(region.metadata)['kind'],'pointwise_math')
