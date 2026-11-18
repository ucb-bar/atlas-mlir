"""Execute emitted FP8 guards over every byte, including finite exponent 15."""
import ctypes
from pathlib import Path
import subprocess
import tempfile
import unittest
import copy

from atlas_mlir_native.contraction_source_c import fp8_invalid_c


class SourceDomainTests(unittest.TestCase):
    def test_source_bound_schedule_mutations_refuse(self):
        from atlas_mlir_native.application import AtlasApplicationAdapter
        from atlas_mlir_native.application_precision import CONTRACTION_POLICY, COLUMN_SCALED_CONTRACTION_POLICY, prepare_contraction
        from atlas_mlir_native.binding import TARGET_IDENTITY
        from atlas_mlir_native.row_softmax import SOFTMAX_POLICY
        from merlin.compile.semantic_regions import extract_semantic_region
        from merlin.semantic_compiler.model import ConstantBinding, IndexMap, KernelRequest, SemanticNode, TensorType
        from merlin.semantic_compiler.target_binding import NativeCompilationError
        f32=lambda shape: TensorType(shape,'f32','raw-bit-pattern')
        maps=tuple(IndexMap(3,rows,(0,0)) for rows in (((1,0,0),(0,0,1)),((0,0,1),(0,1,0)),((1,0,0),(0,1,0))))
        source=KernelRequest((SemanticNode('a','input',(),f32((1,32)),effect='input'),
            SemanticNode('w','input',(),f32((32,64)),effect='input'),
            SemanticNode('zero','constant',(),f32((1,64)),effect='constant'),
            SemanticNode('dot','source_f32_matmul_unqualified',('a','w','zero'),f32((1,64)),
                (('numeric_semantics','source-f32-mul-add-unqualified-v1'),('reduction_order','unresolved')),
                index_maps=(*maps,maps[-1]))),('dot',),('external',),(('a','external'),('w','external')),
            TARGET_IDENTITY,'diagnostic','source-control',(ConstantBinding('zero','f32-le-raw','00'*(64*4)),))
        region=extract_semantic_region(source,('dot',))
        base=dict(schema='atlas.diagnostic_application_precision.v1',source_sha256='0'*64,target_identity=TARGET_IDENTITY,
            contraction=CONTRACTION_POLICY,row_softmax=SOFTMAX_POLICY,host_support=['transpose','reshape','copy'])
        schedule=dict(source_output='dot',policy=COLUMN_SCALED_CONTRACTION_POLICY,source_request_digest=region.digest(),
            provenance_sha256='1'*64,lhs_encode_exponent=9,rhs_column_encode_exponents=[[0,32,9],[32,64,10]])
        policy=dict(schema='atlas.diagnostic_application_precision.v22',source_sha256='0'*64,base_policy=base,contraction_schedules=[schedule])
        def plan(value):
            return AtlasApplicationAdapter(value,original_source_sha256='0'*64).plan(source)
        plan(policy)
        bad=copy.deepcopy(policy);bad['contraction_schedules'][0]['source_request_digest']='f'*64
        with self.assertRaisesRegex(ValueError,'source or policy'):
            plan(bad)
        bad=copy.deepcopy(policy);extra=copy.deepcopy(schedule);extra['source_output']='unused';bad['contraction_schedules'].append(extra)
        with self.assertRaisesRegex(NativeCompilationError,'exact source contractions'):
            plan(bad)
        bad=copy.deepcopy(policy);bad['contraction_schedules'][0]['rhs_column_encode_exponents'][1][0]=33
        with self.assertRaisesRegex(ValueError,'partition'):
            plan(bad)
        # Square operands conceal the geometry error if scale-axis meaning is
        # not checked: a K-dependent RHS scale cannot be restored by N column.
        from dataclasses import replace
        from merlin.semantic_compiler.contraction_precision import lower_semantic_contraction
        square=replace(source,nodes=tuple(replace(node,type=f32((32,32)) if node.id=='w' else f32((1,32)))
                       if node.id in {'w','zero','dot'} else node for node in source.nodes),
                       constants=(ConstantBinding('zero','f32-le-raw','00'*(32*4)),))
        chosen={key:value for key,value in schedule.items() if key!='source_output'}
        chosen.update(source_request_digest=square.digest(),rhs_column_encode_exponents=[[0,16,9],[16,32,10]])
        precision=prepare_contraction(square,operand_scale_exponents=chosen)
        lhs,rhs,out=precision.profile.conversion_specs
        for specs in ((lhs,replace(rhs,scale_axis=1),out),(lhs,rhs,replace(out,scale_axis=0))):
            with self.assertRaisesRegex(ValueError,'logical N'):
                lower_semantic_contraction(square,profile=replace(precision.profile,conversion_specs=specs))

    def test_selected_finite_top_and_legacy_refusal(self):
        source = '#include <stdint.h>\n'
        for name, finite_top in (("legacy",False),("selected",True)):
            source += f'int {name}(unsigned code) {{ return !{fp8_invalid_c("code",finite_top=finite_top)}; }}\n'
        with tempfile.TemporaryDirectory() as root:
            root=Path(root);(root/'guard.c').write_text(source)
            subprocess.run(['gcc','-std=c11','-Wall','-Wextra','-Werror','-shared','-fPIC',str(root/'guard.c'),'-o',str(root/'guard.so')],check=True)
            lib=ctypes.CDLL(str(root/'guard.so'))
            for code in range(256):
                magnitude=code&127
                self.assertEqual(bool(lib.selected(code)),code==0 or 8<=magnitude<=126)
                self.assertEqual(bool(lib.legacy(code)),code==0 or 8<=magnitude<=119)
            for code in (0x7e,0xfe):
                self.assertEqual(lib.selected(code),1)
                self.assertEqual(lib.legacy(code),0)
            for code in (0x7f,0xff,0x80,0x01,0x81):
                self.assertEqual(lib.selected(code),0)


if __name__=='__main__':
    unittest.main()
