"""Authored FP8 byte transport, signed scales, static packing, and C ABI."""
import ctypes
from dataclasses import replace
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

import numpy as np
from atlas_mlir_native.already_quantized import POLICY, pack_operand, prepare_source, restore_bf16, _restore_bf16_c
from atlas_mlir_native.binding import TARGET_IDENTITY
from atlas_mlir_native.static_quantize import pack_bf16, emit_c
from merlin.semantic_compiler.model import KernelRequest, SemanticNode, TensorType


def source_request(m=3,n=35,k=65,batch=(2,)):
    def inp(key,shape,dtype):
        return SemanticNode(key,'input',(),TensorType(shape,dtype,'raw-bit-pattern'),effect='input')
    nodes=(inp('a',(*batch,m,k),'i8'),inp('sx',(),'i32'),inp('w',(n,k),'i8'),inp('sw',(n,1),'i32'))
    attrs=tuple(sorted(dict(semantic_kind='prequantized_contraction',code_dtype='uint8',exponent_dtype='int32',
        weight_format='atlas_e4m3',dequantization='e4m3(code)*2^exponent',rhs_transposed='true',
        output_rescale='2^(lhs_exp+rhs_exp)',bias_order='after_output_cast',output_dtype='bfloat16').items()))
    y=SemanticNode('y','atlas.qlinear',tuple(node.id for node in nodes),TensorType((*batch,m,n),'bf16','raw-bit-pattern'),attrs)
    return KernelRequest((*nodes,y),('y',),('external',),tuple((node.id,'external') for node in nodes),TARGET_IDENTITY,'diagnostic')


def quantizer_source(dtype='bf16',fixed=-7,count=65536):
    x=SemanticNode('x','input',(),TensorType((count,),dtype,'raw-bit-pattern'),effect='input')
    attrs=tuple(sorted(dict(semantic_kind='static_activation_quantizer',mode='static',code_dtype='uint8',
        weight_format='atlas_e4m3',exponent_dtype='int32',dequantization='e4m3(code)*2^exponent',
        reduce='tensor',rule='ref',static_exp=fixed,tag='',rows=32).items()))
    codes=SemanticNode('codes','atlas.quantize_act.codes',('x',),TensorType((count,),'i8','raw-bit-pattern'),attrs)
    exponent=SemanticNode('exponent','atlas.quantize_act.exponent',('x',),TensorType((),'i32','raw-bit-pattern'),attrs)
    return KernelRequest((x,codes,exponent),('codes','exponent'),('external','external'),(('x','external'),),TARGET_IDENTITY,'diagnostic')


class AlreadyQuantizedTest(unittest.TestCase):
    def test_bf16_contraction_requires_explicit_unchanged_weight_host_policy(self):
        from atlas_mlir_native.application import AtlasApplicationAdapter
        from atlas_mlir_native.application_precision import CONTRACTION_POLICY
        from atlas_mlir_native.row_softmax import SOFTMAX_POLICY
        from merlin.compile.c_bf16_linear import SOURCE_POLICY as HOST_POLICY
        from merlin.semantic_compiler.model import IndexMap
        policy = dict(schema='atlas.diagnostic_application_precision.v1', source_sha256='0'*64,
            target_identity=TARGET_IDENTITY, contraction=CONTRACTION_POLICY,
            row_softmax=SOFTMAX_POLICY, host_support=['transpose','reshape','copy'], already_quantized=POLICY)
        nodes = tuple(SemanticNode(key, 'input', (), TensorType(shape, 'bf16', 'raw-bit-pattern'), effect='input')
                      for key, shape in (('a',(2,3)),('b',(3,5)),('seed',(2,5))))
        maps = tuple(IndexMap(3, rows, (0,0)) for rows in
                     (((1,0,0),(0,0,1)),((0,0,1),(0,1,0)),((1,0,0),(0,1,0)),((1,0,0),(0,1,0))))
        result = SemanticNode('y','source_bf16_matmul_unqualified',('a','b','seed'),nodes[-1].type,
            attrs=(('reduction_order','unresolved'),('numeric_semantics','source-bf16-mul-add-unqualified-v1')), index_maps=maps)
        source = KernelRequest((*nodes,result),('y',),('external',),
            tuple((node.id,'external') for node in nodes),TARGET_IDENTITY,'diagnostic')
        with self.assertRaises(Exception):
            AtlasApplicationAdapter(policy, original_source_sha256='0'*64).plan(source)
        selected = AtlasApplicationAdapter({**policy,'host_bf16_linear':HOST_POLICY}, original_source_sha256='0'*64)
        plan = selected.plan(source)
        self.assertEqual(plan.selected[0].placement, 'host')
        self.assertEqual(dict(plan.selected[0].metadata)['kind'], 'host_bf16_linear')
        for refused in ('unknown', 'bf16-product-rne-tree32-f32-rne-bf16-rne-tiles-ltr-chop-v1'):
            with self.subTest(policy=refused), self.assertRaisesRegex(Exception, 'host BF16 linear policy differs'):
                AtlasApplicationAdapter({**policy,'host_bf16_linear':refused},original_source_sha256='0'*64)

    def test_immutable_intake_refuses_unquantized_contraction_before_selection(self):
        from atlas_mlir_native.application import AtlasApplicationAdapter
        from atlas_mlir_native.application_precision import CONTRACTION_POLICY
        from atlas_mlir_native.row_softmax import SOFTMAX_POLICY
        policy = dict(schema='atlas.diagnostic_application_precision.v1', source_sha256='0'*64,
            target_identity=TARGET_IDENTITY, contraction=CONTRACTION_POLICY,
            row_softmax=SOFTMAX_POLICY, host_support=['transpose','reshape','copy'], already_quantized=POLICY)
        adapter = AtlasApplicationAdapter(policy, original_source_sha256='0'*64)
        source = source_request()
        floating = replace(source.node('y'), id='unquantized', op='source_f32_matmul_unqualified')
        for nodes in ((*source.nodes[:-1], floating), (*source.nodes, floating)):
            with self.assertRaisesRegex(Exception, 'refuses floating contractions requiring operand recoding'):
                adapter.plan(replace(source, nodes=nodes, outputs=('unquantized',)))

    def test_logical_codes_survive_tail_padding_and_rhs_transport(self):
        values=np.resize(np.array([0,8,0x38,0x77,0x78,0x7e,0x88,0xfe],np.uint8),(35,65))
        frozen=values.tobytes()
        for row in (0,32):
            for k in (0,32,64):
                packed=np.frombuffer(pack_operand(values,row=row,k=k),np.uint8).reshape(32,32)
                window=values[row:row+32,k:k+32]
                np.testing.assert_array_equal(packed[:window.shape[0],:window.shape[1]],window)
                self.assertTrue(np.all(packed[window.shape[0]:]==0))
                self.assertTrue(np.all(packed[:,window.shape[1]:]==0))
        self.assertEqual(values.tobytes(),frozen)
        for reserved in (1,7,0x7f,0x80,0x81,0xff):
            with self.assertRaises(ValueError): pack_operand(np.array([[reserved]],np.uint8))

    def test_signed_row_channel_restore_and_domain_refusal(self):
        words=np.array([[0x3f80,0x4000],[0xbf80,0]],np.uint16)
        got=restore_bf16(words,np.array([[-7],[3]]),np.array([[4,-2]]))
        np.testing.assert_array_equal(got,np.array([[0x3e00,0x3b80],[0xc300,0]],np.uint16))
        np.testing.assert_array_equal(restore_bf16(words,0,0,output_dtype='f32'),np.array([[1,2],[-1,0]],np.float32))
        with self.assertRaisesRegex(ValueError,'domain'): restore_bf16(words,1000,0)
        with self.assertRaisesRegex(ValueError,'signed'): restore_bf16(words,0.5,0)
        with self.assertRaisesRegex(ValueError,'signed'): restore_bf16(words,np.uint8(127),0)

    def test_arbitrary_k_batched_plan_uses_raw_cap448_bodies(self):
        source=source_request(k=5633)
        plan=prepare_source(source)
        self.assertEqual((plan.m,plan.n,plan.k,plan.batch_shape),(3,35,5633,(2,)))
        self.assertEqual(plan.operand_ids,('a','w'))
        self.assertEqual(plan.exponent_ids,('sx','sw'))
        self.assertEqual(plan.reset.node('result').op,'matmul')
        self.assertEqual(plan.continuation.node('result').op,'matmul_accumulate')
        self.assertEqual(plan.record()['weight_conversion'],'none; unchanged logical FP8 codes')
        changed=replace(source.node('y'),attrs=tuple((key,'wrong' if key=='weight_format' else value) for key,value in source.node('y').attrs))
        with self.assertRaisesRegex(Exception,'weight format'):
            prepare_source(replace(source,nodes=(*source.nodes[:-1],changed)))

    def test_readout_requires_representable_nonzero_bf16_factor_even_for_zero(self):
        # The reference materializes BF16(2**(sx+sw)) before VMUL.
        # Exact nonzero subnormal factors preserve normal results and zeros.
        for word,sx,sw in ((0x3980,64,64),(0,64,64),(0x4980,-70,-70),(0,-70,-70),(0x4980,-67,-67)):
            with self.subTest(word=word,sx=sx,sw=sw), self.assertRaisesRegex(ValueError,'factor.*domain'):
                restore_bf16(np.array([word],np.uint16),sx,sw)
        for word,sx,sw in ((0x3f80,-64,-63),(0x7f00,1,0),(0x007f,0,0),(0x7f80,0,0)):
            with self.subTest(word=word,sx=sx,sw=sw), self.assertRaisesRegex(ValueError,'scaled readout'):
                restore_bf16(np.array([word],np.uint16),sx,sw)
        for word,sx,sw,wanted in ((0x3f80,-63,-63,0x0080),(0x3f80,64,63,0x7f00),
                                  (0,-63,-63,0),(0,64,63,0),(0xbf80,0,0,0xbf80),
                                      (0x4980,-66,-67,0x0700),(0x4000,-64,-63,0x0080),
                                      (0,-66,-67,0),(0x8000,-66,-67,0x8000)):
            self.assertEqual(int(restore_bf16(np.array([word],np.uint16),sx,sw)[0]),wanted)

    def test_generated_readout_refuses_scale_overflow_and_underflow_atomically(self):
        if not shutil.which('cc'): self.skipTest('C compiler unavailable')
        code='#include <stdint.h>\n'+_restore_bf16_c('readout')+'''
int restore(uint32_t word,int32_t lhs,int32_t rhs,uint32_t *out) {
  return readout_restore(word,lhs,rhs,out);
}
'''
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);(root/'restore.c').write_text(code)
            subprocess.run(['cc','-std=c11','-Wall','-Wextra','-Werror','-O2','-shared','-fPIC',
                            'restore.c','-o','restore.so'],cwd=root,check=True,capture_output=True)
            library=ctypes.CDLL(str(root/'restore.so'))
            library.restore.argtypes=[ctypes.c_uint32,ctypes.c_int32,ctypes.c_int32,ctypes.POINTER(ctypes.c_uint32)]
            for word,sx,sw in ((0x3980,64,64),(0,64,64),(0x4980,-70,-70),(0,-70,-70),(0x4980,-67,-67),
                               (0x3f80,-64,-63),(0x7f00,1,0),(0x007f,0,0),(0x7f80,0,0)):
                out=ctypes.c_uint32(0xa5a5)
                self.assertEqual(library.restore(word,sx,sw,ctypes.byref(out)),2)
                self.assertEqual(out.value,0xa5a5)
            for word,sx,sw,wanted in ((0x3f80,-63,-63,0x0080),(0x3f80,64,63,0x7f00),
                                      (0,-63,-63,0),(0,64,63,0),(0xbf80,0,0,0xbf80),
                                      (0x4980,-66,-67,0x0700),(0x4000,-64,-63,0x0080),
                                      (0,-66,-67,0),(0x8000,-66,-67,0x8000)):
                out=ctypes.c_uint32(0xa5a5)
                self.assertEqual(library.restore(word,sx,sw,ctypes.byref(out)),0)
                self.assertEqual(out.value,wanted)

    def test_application_selects_explicit_byte_intake_and_device_packer(self):
        from atlas_mlir_native.application import AtlasApplicationAdapter
        from atlas_mlir_native.application_precision import CONTRACTION_POLICY
        from atlas_mlir_native.row_softmax import SOFTMAX_POLICY
        policy=dict(schema='atlas.diagnostic_application_precision.v1',source_sha256='0'*64,target_identity=TARGET_IDENTITY,
            contraction=CONTRACTION_POLICY,row_softmax=SOFTMAX_POLICY,host_support=['transpose','reshape','copy'],already_quantized=POLICY)
        adapter=AtlasApplicationAdapter(policy,original_source_sha256='0'*64)
        self.assertEqual(dict(adapter.plan(source_request()).selected[0].metadata)['kind'],'already_quantized')
        self.assertEqual(adapter.plan(quantizer_source()).selected[0].placement,'accelerator')
        host = AtlasApplicationAdapter({**policy, 'activation_quantizer': 'host'}, original_source_sha256='0'*64)
        self.assertEqual(dict(host.plan(quantizer_source()).selected[0].metadata)['kind'], 'static_quantize')
        self.assertEqual(host.plan(quantizer_source()).selected[0].placement, 'host')
        from test_activation_quantize_device import dynamic_source
        wide_exponents = dynamic_source(shape=(4096, 1), reduce='block', rows=4096)
        self.assertEqual(dict(adapter.plan(wide_exponents).selected[0].metadata)['kind'], 'dynamic_quantize')
        from merlin.semantic_compiler.codegen import CodegenOptions, codegen_options
        with codegen_options(CodegenOptions(io_memory_bytes=8192)):
            chunked = adapter.plan(quantizer_source(count=65536)).selected[0]
        self.assertEqual(chunked.placement, 'accelerator')
        self.assertEqual(dict(chunked.metadata)['kind'], 'activation_quantize_device')
        self.assertIn('atlas.qlinear',adapter.source_operation_translators)
        from atlas_mlir_native.pointwise import BF16_POINTWISE_POLICY
        selected=AtlasApplicationAdapter({**policy,'bf16_pointwise':BF16_POINTWISE_POLICY,
            'already_quantized_mask':'bf16-minfinite-zero-bounded32-nonempty-v1'},original_source_sha256='0'*64)
        self.assertEqual(selected.bf16_pointwise,BF16_POINTWISE_POLICY)
        self.assertEqual(selected.already_quantized_mask_sentinel,0xff7f0000)
        self.assertIsNone(adapter.already_quantized_mask_sentinel)
        with self.assertRaises(Exception):
            AtlasApplicationAdapter({**policy,'already_quantized_mask':'accept-any-finite'},original_source_sha256='0'*64)
        with self.assertRaises(Exception):
            AtlasApplicationAdapter({k:v for k,v in policy.items() if k!='already_quantized'},original_source_sha256='0'*64).plan(source_request())

    def test_static_packer_edges_and_generated_c_all_bf16_patterns(self):
        # Known boundaries include rounding into the minimum normal, finite
        # exponent15, saturation, signed zeros, NaNs, and infinities.
        bits=np.array([0,0x8000,0x3c78,0x3c77,0x43e0,0xc3e0,0x43f0,0x7f80,0xff80,0x7fc1],np.uint16)
        np.testing.assert_array_equal(pack_bf16(bits,0),np.array([0,0,8,0,126,254,126,126,254,0],np.uint8))
        if not shutil.which('cc'): self.skipTest('C compiler unavailable')
        values=np.arange(65536,dtype=np.uint16)
        for fixed in (-127,-7,0,9,127):
            source=quantizer_source(fixed=fixed)
            record,files=emit_c(source,'pack')
            with tempfile.TemporaryDirectory() as temp:
                root=Path(temp)
                for key,data in files.items():
                    path=root/key;path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(data)
                subprocess.run(['cc','-std=c11','-Wall','-Wextra','-Werror','-O2','-shared','-fPIC','pack.c','-o','pack.so'],cwd=root,check=True,capture_output=True)
                lib=ctypes.CDLL(str(root/'pack.so'));lib.pack.argtypes=[ctypes.c_void_p,ctypes.POINTER(ctypes.c_void_p),ctypes.POINTER(ctypes.c_void_p)]
                out=np.full(65536,0xa5,np.uint8); exp=ctypes.c_int32()
                args=(ctypes.c_void_p*1)(values.ctypes.data); results=(ctypes.c_void_p*2)(out.ctypes.data,ctypes.addressof(exp))
                self.assertEqual(lib.pack(None,args,results),0)
                self.assertEqual(exp.value,fixed)
                np.testing.assert_array_equal(out,pack_bf16(values,fixed))

    def test_dynamic_generated_c_produces_runtime_exponents_for_each_reduction(self):
        from atlas_mlir_native.dynamic_quantize import emit_c as emit_dynamic_c
        if not shutil.which('cc'): self.skipTest('C compiler unavailable')
        shape=(2,3,5,7); count=np.prod(shape); row_count=np.prod(shape[:-1]); k=shape[-1]; m=shape[-2]
        values=((np.arange(count,dtype=np.uint32)*311+7)&65535).astype(np.uint16)
        # Every special class appears, and a maximum above448 forces nc to
        # differ from ref in the first head while the final group stays short.
        values[:7]=[0x43f0,0xc3f0,0x7f80,0xff80,0x7fc1,0,0x8000]
        for reduction in ('tensor','head','token','block','head_block'):
            for rule in ('ref','nc'):
                exp_shape=() if reduction=='tensor' else (*shape[:-2],1,1) if reduction=='head' else (*shape[:-1],1)
                original=quantizer_source(count=int(count)); attrs=dict(original.node('codes').attrs)
                attrs.pop('static_exp');attrs.update(mode='dynamic',semantic_kind='dynamic_activation_quantizer',reduce=reduction,rule=rule,rows=3)
                attrs=tuple(sorted(attrs.items()))
                source=replace(original,nodes=(replace(original.node('x'),type=TensorType(shape,'bf16','raw-bit-pattern')),
                    replace(original.node('codes'),type=TensorType(shape,'i8','raw-bit-pattern'),attrs=attrs),
                    replace(original.node('exponent'),type=TensorType(exp_shape,'i32','raw-bit-pattern'),attrs=attrs)))
                if reduction=='tensor': groups=[(0,int(row_count))]
                elif reduction=='head': groups=[(first,first+m) for first in range(0,int(row_count),m)]
                elif reduction=='token': groups=[(first,first+1) for first in range(int(row_count))]
                elif reduction=='block': groups=[(first,min(first+3,int(row_count))) for first in range(0,int(row_count),3)]
                else: groups=[(head+first,head+min(first+3,m)) for head in range(0,int(row_count),m) for first in range(0,m,3)]
                wanted_codes=np.empty(values.size,np.uint8); wanted_exponents=np.empty(int(np.prod(exp_shape)),np.int32)
                for group,(first,end) in enumerate(groups):
                    block=values[first*k:end*k]
                    decoded=(block.astype(np.uint32)<<16).view(np.float32)
                    finite=np.where(np.isfinite(decoded),np.abs(decoded),0)
                    maximum=float(finite.max()); exponent=int(np.frexp(maximum)[1])-9 if maximum else 0
                    if rule=='nc' and maximum>np.ldexp(448.,exponent): exponent+=1
                    wanted_codes[first*k:end*k]=pack_bf16(block,exponent)
                    if reduction in ('tensor','head'): wanted_exponents[group]=exponent
                    else: wanted_exponents[first:end]=exponent
                record,files=emit_dynamic_c(source,'dynamic_pack')
                self.assertEqual(record['exponent_source'],'runtime BF16 amax; signed i32 data')
                with tempfile.TemporaryDirectory() as temp:
                    root=Path(temp)
                    for key,data in files.items():
                        path=root/key;path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(data)
                    subprocess.run(['cc','-std=c11','-Wall','-Wextra','-Werror','-O2','-shared','-fPIC','dynamic_pack.c','-o','pack.so'],cwd=root,check=True,capture_output=True)
                    lib=ctypes.CDLL(str(root/'pack.so'));lib.dynamic_pack.argtypes=[ctypes.c_void_p,ctypes.POINTER(ctypes.c_void_p),ctypes.POINTER(ctypes.c_void_p)]
                    actual=np.zeros(values.size,np.uint8); exponents=np.zeros(wanted_exponents.size,np.int32)
                    args=(ctypes.c_void_p*1)(values.ctypes.data); result=(ctypes.c_void_p*2)(actual.ctypes.data,exponents.ctypes.data)
                    self.assertEqual(lib.dynamic_pack(None,args,result),0)
                    np.testing.assert_array_equal(exponents,wanted_exponents)
                    np.testing.assert_array_equal(actual,wanted_codes)
                    # Scaling live input changes the signed exponent producer.
                    finite_values=np.full(values.size,0x3f80,np.uint16)
                    args[0]=finite_values.ctypes.data
                    self.assertEqual(lib.dynamic_pack(None,args,result),0)
                    self.assertTrue(np.all(exponents==-8))
                    finite_values[:]=0x4400
                    self.assertEqual(lib.dynamic_pack(None,args,result),0)
                    self.assertTrue(np.all(exponents==1))

    def test_f32_packer_rounds_to_bf16_and_preserves_nan_class(self):
        if not shutil.which('cc'): self.skipTest('C compiler unavailable')
        bits=np.array([0x7f800001,0xff800001,0x7f800000,0xff800000,0,0x80000000,
                       0x3f808000,0x3f880000,0x3f980000],np.uint32)
        expected=np.array([0,0,126,254,0,0,0x38,0x38,0x3a],np.uint8)
        _,files=emit_c(quantizer_source(dtype='f32',fixed=0,count=bits.size),'pack')
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)
            for key,data in files.items():
                path=root/key;path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(data)
            subprocess.run(['cc','-std=c11','-Wall','-Wextra','-Werror','-O2','-shared','-fPIC','pack.c','-o','pack.so'],cwd=root,check=True,capture_output=True)
            lib=ctypes.CDLL(str(root/'pack.so'));lib.pack.argtypes=[ctypes.c_void_p,ctypes.POINTER(ctypes.c_void_p),ctypes.POINTER(ctypes.c_void_p)]
            out=np.zeros(bits.size,np.uint8);exp=ctypes.c_int32()
            args=(ctypes.c_void_p*1)(bits.ctypes.data);result=(ctypes.c_void_p*2)(out.ctypes.data,ctypes.addressof(exp))
            self.assertEqual(lib.pack(None,args,result),0)
            np.testing.assert_array_equal(out,expected)


    def test_parsed_quantizers_preserve_integer_storage_policies_through_generated_c(self):
        from atlas_mlir_native.source_translation import translate_activation_quantizer
        from merlin.semantic_compiler.linalg_bridge import translate_linalg_text
        from atlas_mlir_native.dynamic_quantize import emit_c as emit_dynamic_c
        if not shutil.which('cc'): self.skipTest('C compiler unavailable')
        words=np.array([[0x3f80,0x4000,0],[0xbf80,0x3f00,0x4080]],np.uint16)
        frozen=words.tobytes()
        for mode,reduce,exp_type,extra,emitter,wanted_codes,wanted_exponents in (
            ('static','tensor','tensor<i32>','static_exp=0:i32,',emit_c,
             [0x38,0x40,0,0xb8,0x30,0x48],[0]),
            ('dynamic','token','tensor<2x1xi32>','',emit_dynamic_c,
             [0x70,0x78,0,0xe8,0x60,0x78],[-7,-6]),
        ):
            source=f'''module {{ func.func @work(%x:tensor<2x3xbf16>) -> (tensor<2x3xi8>,{exp_type}) {{
              %q:2 = "atlas.quantize_act"(%x) {{mode="{mode}",reduce="{reduce}",rule="ref",
                code_dtype="uint8",exponent_dtype="int32",weight_format="atlas_e4m3",
                dequantization="e4m3(code)*2^exponent",rows=32:i32,{extra}tag=""}}
                : (tensor<2x3xbf16>) -> (tensor<2x3xi8>,{exp_type})
              func.return %q#0,%q#1 : tensor<2x3xi8>,{exp_type}
            }} }}'''
            request=translate_linalg_text(source,entry='work',target_identity=TARGET_IDENTITY,
                lowering_policy='diagnostic',source_operation_translators={
                    'atlas.quantize_act':translate_activation_quantizer}).request
            self.assertEqual(request.node(request.outputs[0]).type.numerical_policy,'signed-i8')
            self.assertEqual(request.node(request.outputs[1]).type.numerical_policy,'i32-wrap-k-ascending')
            record,files=emitter(request,'pack')
            self.assertEqual(record['source_request_digest'],request.digest())
            with tempfile.TemporaryDirectory() as temp:
                root=Path(temp)
                for key,data in files.items():
                    path=root/key;path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(data)
                subprocess.run(['cc','-std=c11','-Wall','-Wextra','-Werror','-O2','-shared','-fPIC',
                                'pack.c','-o','pack.so'],cwd=root,check=True,capture_output=True)
                library=ctypes.CDLL(str(root/'pack.so'))
                library.pack.argtypes=[ctypes.c_void_p,ctypes.POINTER(ctypes.c_void_p),ctypes.POINTER(ctypes.c_void_p)]
                codes=np.full(words.size,0xa5,np.uint8)
                exponents=np.full(len(wanted_exponents),12345,np.int32)
                inputs=(ctypes.c_void_p*1)(words.ctypes.data)
                outputs=(ctypes.c_void_p*2)(codes.ctypes.data,exponents.ctypes.data)
                self.assertEqual(library.pack(None,inputs,outputs),0)
                np.testing.assert_array_equal(codes,np.array(wanted_codes,np.uint8))
                np.testing.assert_array_equal(exponents,np.array(wanted_exponents,np.int32))
                self.assertEqual(words.tobytes(),frozen)

    def test_quantizer_rejects_duplicate_outputs_and_allocated_layouts(self):
        from atlas_mlir_native.static_quantize import prepare_source as prepare_static
        from atlas_mlir_native.dynamic_quantize import prepare_source as prepare_dynamic
        source=quantizer_source(count=8)
        attrs=dict(source.node('codes').attrs);attrs.pop('static_exp')
        attrs.update(mode='dynamic',semantic_kind='dynamic_activation_quantizer')
        dynamic=replace(source,nodes=(source.nodes[0],*(replace(node,attrs=tuple(sorted(attrs.items()))) for node in source.nodes[1:])))
        for request,prepare in ((source,prepare_static),(dynamic,prepare_dynamic)):
            for changed in (replace(request,outputs=('codes','exponent','codes'),output_storages=('external',)*3),
                            replace(request,outputs=('codes','codes')),
                            replace(request,physical_extents=(('x',8),)),
                            replace(request,physical_extents=(('x',8),),boundary_layouts=(('x','non-dense'),),
                                    output_layouts=('non-dense','non-dense'))):
                with self.assertRaisesRegex(ValueError,'distinct dense'):
                    prepare(changed)
            wrong=replace(request.node('codes'),type=replace(request.node('codes').type,numerical_policy='unknown-encoding'))
            with self.assertRaisesRegex(ValueError,'raw-bit-pattern'):
                prepare(replace(request,nodes=(request.nodes[0],wrong,request.nodes[2])))

    def test_parsed_integer_storage_carriers_keep_code_and_exponent_data(self):
        import json
        from atlas_mlir_native.application import AtlasApplicationAdapter
        from atlas_mlir_native.application_precision import CONTRACTION_POLICY
        from atlas_mlir_native.row_softmax import SOFTMAX_POLICY
        from merlin.semantic_compiler.linalg_bridge import translate_linalg_text
        def adapter(policy=None):
            policy=policy or dict(schema='atlas.diagnostic_application_precision.v1',source_sha256='0'*64,
                target_identity=TARGET_IDENTITY,contraction=CONTRACTION_POLICY,row_softmax=SOFTMAX_POLICY,
                host_support=['transpose','reshape','copy'],already_quantized=POLICY)
            return AtlasApplicationAdapter(policy,original_source_sha256=policy['source_sha256'])
        for mode in ('static','dynamic'):
            static=', static_exp = -7 : i32' if mode=='static' else ''
            text=f'''module {{
              func.func @forward(%w: tensor<4x3xi8>, %sw: tensor<4x1xi32>, %x: tensor<1x2x3xbf16>) -> tensor<1x2x4xbf16> {{
                %codes, %sx = "atlas.quantize_act"(%x) {{code_dtype = "uint8", exponent_dtype = "int32", weight_format = "atlas_e4m3", dequantization = "e4m3(code)*2^exponent", mode = "{mode}", reduce = "tensor", rule = "ref", rows = 32 : i32, tag = ""{static}}} : (tensor<1x2x3xbf16>) -> (tensor<1x2x3xi8>, tensor<i32>)
                %y = "atlas.qlinear"(%codes, %sx, %w, %sw) {{code_dtype = "uint8", exponent_dtype = "int32", weight_format = "atlas_e4m3", dequantization = "e4m3(code)*2^exponent", rhs_transposed = "true", output_rescale = "2^(lhs_exp+rhs_exp)", output_dtype = "bfloat16", bias_order = "after_output_cast"}} : (tensor<1x2x3xi8>, tensor<i32>, tensor<4x3xi8>, tensor<4x1xi32>) -> tensor<1x2x4xbf16>
                return %y : tensor<1x2x4xbf16>
              }}
            }}'''
            selected=adapter()
            source=translate_linalg_text(text,entry='forward',target_identity=TARGET_IDENTITY,lowering_policy='diagnostic',
                                        source_operation_translators=selected.source_operation_translators).request
            codes=next(node for node in source.nodes if node.op=='atlas.quantize_act.codes')
            exponent=next(node for node in source.nodes if node.op=='atlas.quantize_act.exponent')
            self.assertEqual(codes.type.numerical_policy,'signed-i8')
            self.assertEqual(exponent.type.numerical_policy,'i32-wrap-k-ascending')
            self.assertEqual([dict(region.metadata)['kind'] for region in selected.plan(source).selected],
                             [('static_quantize' if mode == 'static' else 'dynamic_quantize'),'already_quantized'])
            self.assertEqual(selected.plan(source).selected[0].placement, 'host')
        # When the integration receipts are present, replay the exact staged
        # source that exposed this mismatch, including its saved carrier ABI.
        workspace=Path(__file__).resolve().parents[2]
        for name in ('qlinear-external-r1','attention-e2e-r2'):
            root=workspace/'out'/name
            if not (root/'source-request.json').is_file() or not (root/'staged/program.mlir').is_file():
                continue
            saved=KernelRequest.from_record(json.loads((root/'source-request.json').read_text()))
            selected=adapter(json.loads((root/'precision.json').read_text()))
            translated=translate_linalg_text((root/'staged/program.mlir').read_text(),entry='forward',
                target_identity=TARGET_IDENTITY,lowering_policy='diagnostic',
                source_operation_translators=selected.source_operation_translators).request
            self.assertTrue(selected.plan(translated).selected)
            self.assertTrue(selected.plan(saved).selected)
            for request in (saved,translated):
                carriers={node.type.numerical_policy for node in request.nodes if node.op.startswith('atlas.quantize_act.')}
                self.assertEqual(carriers,{'signed-i8','i32-wrap-k-ascending'})


    def test_readout_factor_must_be_a_finite_nonzero_bf16_power_of_two(self):
        for word,sx,sw in ((0,127,127),(0x4980,-70,-70),(0x3680,64,64),(0x4980,-67,-67)):
            with self.assertRaisesRegex(ValueError,'readout factor'):
                restore_bf16(np.array([[word]],np.uint16),np.int32(sx),np.int32(sw))
        self.assertEqual(int(restore_bf16(np.array([[0x4980]],np.uint16),np.int32(-66),np.int32(-67))[0,0]),0x0700)
        self.assertEqual(int(restore_bf16(np.array([[0x3680]],np.uint16),np.int32(63),np.int32(64))[0,0]),0x7600)
        self.assertEqual(int(restore_bf16(np.array([[0]],np.uint16),np.int32(63),np.int32(64))[0,0]),0)

    def test_generated_c_factor_preflight_before_callbacks_and_batched_association(self):
        """Test the full host dispatcher with a mock device, without a native build.

        Only publication authentication is mocked for the one-word test image.
        Real generated K32 images have a separate model-backed directed audit.
        """
        import json
        from hashlib import sha256
        from unittest.mock import patch
        from atlas_mlir_native import already_quantized as native
        from merlin.semantic_compiler.model import pointwise_identity_maps
        if not shutil.which('cc'):self.skipTest('C compiler unavailable')
        atlas=Path(__file__).resolve().parents[1]
        shared=source_request(m=2,n=3,k=32,batch=(2,))
        shared=replace(shared,nodes=tuple(replace(node,type=TensorType((2,2,1),'i32','raw-bit-pattern'))
                       if node.id=='sx' else node for node in shared.nodes))
        head=source_request(m=2,n=3,k=32,batch=(2,3))
        head_nodes=[]
        for node in head.nodes:
            if node.id=='w':node=replace(node,type=TensorType((2,3,3,32),'i8','raw-bit-pattern'))
            if node.id in ('sx','sw'):node=replace(node,type=TensorType((2,3,1,1),'i32','raw-bit-pattern'))
            if node.id=='y':node=replace(node,op='atlas.qk_matmul',type=replace(node.type,dtype='f32'),
                attrs=tuple((key,'float32' if key=='output_dtype' else value) for key,value in node.attrs))
            head_nodes.append(node)
        head=replace(head,nodes=tuple(head_nodes))
        for source in (shared,head):
            with tempfile.TemporaryDirectory() as temp:
                root=Path(temp);compiled=root/'compiled';compiled.mkdir()
                plan=native.prepare_source(source)
                (compiled/'already-quantized-plan.json').write_text(json.dumps(plan.record()))
                for role,body in (('reset',plan.reset),('continue',plan.continuation)):
                    directory=compiled/role;directory.mkdir()
                    abi=native.AtlasNativeBinding.plan_boundary_abi(body)
                    addresses=[native.DRAM_BASE+abi['fixed_inputs'][node.id]*native.TILE_BYTES for node in body.nodes if node.effect=='input']
                    (directory/'manifest.json').write_text(json.dumps({'target_inventory_sha256':sha256((atlas/'docs/selected-variant-inventory.json').read_bytes()).hexdigest()}))
                    (directory/'source-request.json').write_text(json.dumps(body.record()))
                    (directory/'execution_plan.json').write_text(json.dumps(dict(input_addresses=addresses,
                        output_addresses=[native.DRAM_BASE+abi['fixed_outputs'][0]*native.TILE_BYTES],operation='mxu1_cap448_seeded_ordered')))
                    (directory/'program.bin').write_bytes((0x00100073).to_bytes(4,'little'))
                    (directory/'physical-program.json').write_text(json.dumps({'instructions':[{'word_u32':0x00100073}]}))
                with patch.object(native,'verify_native_publication'):
                    _,files=native.prepare_c_package(source,compiled,'probe',target_source=atlas)
                package=root/'c';package.mkdir()
                for key,data in files.items():
                    path=package/key;path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(data)
                (package/'shim.c').write_text('''#include "probe.h"
static atlas_host host; static probe_context region; static probe_source_context source;
static uint8_t raw[PROBE_WORKSPACE_BYTES],converted[PROBE_SOURCE_WORKSPACE_BYTES];
static unsigned calls,starts; static uint16_t result_word;
static uint32_t read32(void *c,uintptr_t a) {(void)c;++calls;return a==0x40004u?starts:5u;}
static void write32(void *c,uintptr_t a,uint32_t v) {(void)c;++calls;if(a==0x40018u&&v==1u)++starts;}
static int put(void *c,uintptr_t a,const uint8_t *v,size_t n) {(void)c;(void)a;(void)v;(void)n;++calls;return 0;}
static int get(void *c,uintptr_t a,uint8_t *v,size_t n) {(void)c;(void)a;++calls;for(size_t i=0;i<n;++i)v[i]=(uint8_t)(result_word>>((i%2u)*8u));return 0;}
unsigned callback_count(void) {return calls;}
int invoke(const void *a,const void *sx,const void *w,const void *sw,void *out,uint16_t word) {
 calls=starts=0;result_word=word;
 if(atlas_host_init(&host,(atlas_host_map){0x20000u,0x40000u,131072u},(atlas_host_mmio){read32,write32,0},(atlas_host_memory){1,0,0}))return -1;
 if(probe_bind(&region,&host,(atlas_tile_memory_io){put,get,0},raw,sizeof(raw),20u))return -2;
 if(probe_source_bind(&source,&region,converted,sizeof(converted)))return -3;
 const void *inputs[]={a,sx,w,sw};void *outputs[]={out};return probe_source_dispatch(&source,inputs,outputs);
}
''')
                subprocess.run(['cc','-std=c11','-O2','-Wall','-Wextra','-Werror','-shared','-fPIC','probe.c','runtime/atlas_host.c','shim.c','-o','probe.so'],cwd=package,check=True,capture_output=True)
                lib=ctypes.CDLL(str(package/'probe.so'));lib.invoke.argtypes=[ctypes.c_void_p]*5+[ctypes.c_uint16]
                a=np.zeros(source.node('a').type.shape,np.uint8);w=np.zeros(source.node('w').type.shape,np.uint8)
                output_bytes=np.prod(source.node('y').type.shape)*(2 if source.node('y').type.dtype=='bf16' else 4)
                def run(sx,sw,word,invalid=False,expected_word=0):
                    sx=np.ascontiguousarray(sx,dtype=np.int32);sw=np.ascontiguousarray(sw,dtype=np.int32)
                    output=np.full(int(output_bytes),0xa5,np.uint8)
                    status=lib.invoke(a.ctypes.data,sx.ctypes.data,w.ctypes.data,sw.ctypes.data,output.ctypes.data,word)
                    if invalid:
                        self.assertEqual(status,2);self.assertEqual(lib.callback_count(),0)
                        self.assertTrue(np.all(output==0xa5))
                    else:
                        self.assertEqual(status,0);self.assertGreater(lib.callback_count(),0)
                        actual=np.frombuffer(output,dtype='<u2' if source.node('y').type.dtype=='bf16' else '<u4')
                        self.assertTrue(np.all(actual==expected_word))
                sx_shape=source.node('sx').type.shape;sw_shape=source.node('sw').type.shape
                for word,sx,sw in ((0,127,127),(0x4980,-70,-70),(0,-70,-70),(0x3680,64,64),(0x4980,-67,-67)):
                    run(np.full(sx_shape,sx,np.int32),np.full(sw_shape,sw,np.int32),word,invalid=True)
                for word,sx,sw,expected in ((0x4980,-66,-67,0x0700),(0x3680,63,64,0x7600)):
                    run(np.full(sx_shape,sx,np.int32),np.full(sw_shape,sw,np.int32),word,
                        expected_word=expected if source.node('y').type.dtype=='bf16' else expected<<16)
                if source is shared:
                    # The first batch is legal; the later one must fail before
                    # any first-batch I/O. Padding catches an erroneous RHS batch stride.
                    sx=np.array([[[-7],[-7]],[[64],[64]]],np.int32)
                    sw=np.array([64,64,64,-127,-127,-127],np.int32)
                    run(sx,sw,0,invalid=True)
                else:
                    # Opposite head scales are legal in each head; a global
                    # cross-head extrema check would incorrectly reject them.
                    sx=np.array([-127,127,-127,127,-127,127],np.int32).reshape(sx_shape)
                    run(sx,-sx,0x3f80,expected_word=0x3f800000)
                    sx=np.zeros(sx_shape,np.int32);sw=np.zeros(sw_shape,np.int32)
                    sx.flat[4]=sw.flat[4]=64
                    run(sx,sw,0,invalid=True)
