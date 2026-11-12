"""Exact in-tile products after ordered quantized contraction restoration."""
from pathlib import Path
import json
import os
import subprocess
import tempfile
import unittest

import numpy as np

from atlas_mlir_native.already_quantized import prepare_source
from atlas_mlir_native.binding import DRAM_BASE
from atlas_mlir_native.quantized_loop import execution_layout,virtual_source
from test_exponent_restoration import BIN,ROOT,compile_fixture,raw_fixture
from test_quantized_epilogue import case,packed_extra,packed_exponents
from test_quantized_loop import packed


class QuantizedEpilogueLoweringTest(unittest.TestCase):
    def test_vector_factor_bounds_preserve_selected_product_domain(self):
        from atlas_mlir_native.exponent_restoration import fast_factor_range
        for dtype,operand in (('bf16',0),('f32',1)):
            plan=prepare_source(case(dtype,operand)[0])
            lower,upper=fast_factor_range(plan)
            self.assertEqual((lower,upper),(-18,15))
            spec=plan.epilogue_precision.operand_specs[plan.epilogue_contract_operand]
            for exponent in range(-26,33):
                for sx in range(lower,upper+1):
                    self.assertTrue(-126 <= exponent+sx <= 127)
                    for sw in range(lower,upper+1):
                        target=exponent+sx+sw
                        for sign in (0,0x8000):
                            for mantissa in (0,127):
                                code=sign|((target+127)<<7)|mantissa
                                self.assertTrue(any(first<=code<=last for first,last in spec.admitted_code_ranges))

    @unittest.skipUnless((BIN/'atlas-opt').is_file(),'Atlas compiler binaries unavailable')
    def test_tail_batch_dtypes_and_operand_order_pass_default_checks(self):
        # Four-tile images exercise the explicit zero fallback budget. The
        # two-tile counterparts retain successful full scalar admission.
        for dtype,operand,n in (('bf16',0,35),('f32',1,35),('bf16',0,32),('f32',1,32)):
            source,_,_=case(dtype,operand,n=n);plan=prepare_source(source);layout=execution_layout(plan)
            self.assertEqual(layout['input_addresses'][2],layout['status_address']+layout['status_bytes'])
            self.assertEqual(layout['input_buffer_bytes'][2],layout['output_bytes'])
            virtual=virtual_source(plan,layout);_,program=compile_fixture(virtual)
            self.assertTrue(any(row['operation']=='atlas.vpu_binary' and row['fields']['kind']=='mul' for row in program['instructions']))
            self.assertEqual(sum(row['operation']=='atlas.trap' for row in program['instructions']),1)

    @unittest.skipUnless(os.environ.get('ATLAS_RUN_NPU_MODEL')=='1','select pinned npu_model explicitly')
    def test_original_model_k65_tails_admission_clamping_and_status_carry(self):
        groups=[]
        for dtype,operand,n in (('bf16',0,35),('f32',1,35),('bf16',0,32),('f32',1,32)):
            source,_,_=case(dtype,operand,n=n);plan=prepare_source(source);layout=execution_layout(plan)
            scalar_fallback=bool(layout['restoration_admission']['scalar_fallback_tile_budget'])
            _,program=compile_fixture(virtual_source(plan,layout))
            binary=b''.join(row['word_u32'].to_bytes(4,'little') for row in program['instructions'])
            cases=[]
            for label in ('valid','valid_negative','vector_lower','vector_upper','signed_zero_extra','following_operand_domain','bad_exponents_cancel','zero_scale_min'):
                a=np.full(source.node('a').type.shape,0 if label=='zero_scale_min' else 0xb8 if label=='valid_negative' else 0x38,np.uint8)
                b=np.full(source.node('w').type.shape,0 if label=='zero_scale_min' else 0x38,np.uint8)
                sx=np.array(0,dtype='<i4');sw=np.zeros(source.node('sw').type.shape,dtype='<i4')
                expected=np.full((2,plan.m,plan.n),0x4302,dtype='<u2')
                if label in ('valid','valid_negative'):
                    sw[:,0]=np.arange(plan.n)%3-1
                    expected[:]=(0xc282 if label=='valid_negative' else 0x4282)+((sw[:,0]+1).reshape(1,1,plan.n)<<7)
                elif label in ('vector_lower','vector_upper'):
                    scale=-18 if label=='vector_lower' else 15
                    sx.fill(scale);sw.fill(scale);expected.fill(0x4302+2*scale*128)
                elif label=='signed_zero_extra':
                    expected[:,:,0]=0x8000;expected[:,:,1]=0
                elif label=='following_operand_domain':
                    # This restored result is finite/normal but exceeds the
                    # following pointwise encoder's bounded input domain.
                    sw[0,0]=64
                    # A zero-budget image refuses the entire affected tile;
                    # host publication is atomic on its sticky status. Later
                    # fast tiles still retain their exact device results.
                    expected[:,:,:min(32,plan.n) if not scalar_fallback else 1]=0
                elif label=='bad_exponents_cancel':
                    # Wrapped/summed restoration could look completely valid;
                    # operand admission must still clamp the VPU input to +0.
                    sx.fill(128);sw.fill(-128);expected.fill(0)
                else:
                    sx.fill(-127);sw.fill(-6);expected.fill(0)
                extras=np.full((2,plan.m,plan.n),0x4000,dtype='<u2')
                if label=='signed_zero_extra':
                    extras[:,:,0]=0x8000;extras[:,:,1]=0
                payloads=[packed(a).tobytes(),packed(b).tobytes(),packed_extra(extras,plan).tobytes()]
                size=layout['input_addresses'][2]-DRAM_BASE+layout['input_buffer_bytes'][2]+2048
                memory=bytearray([0xa5])*size
                for address,payload in zip(layout['input_addresses'],payloads,strict=True):
                    at=address-DRAM_BASE;memory[at:at+len(payload)]=payload
                for values,geometry,address in zip((sx,sw),layout['exponent_layouts'],layout['exponent_addresses'],strict=True):
                    payload=packed_exponents(values,geometry).tobytes();at=address-DRAM_BASE;memory[at:at+len(payload)]=payload
                wanted=bytearray(memory);payload=packed_extra(expected,plan).tobytes();at=layout['output_addresses'][0]-DRAM_BASE
                wanted[at:at+len(payload)]=payload
                rejected=(label in ('following_operand_domain','bad_exponents_cancel') or
                          (label=='zero_scale_min' and not scalar_fallback));at=layout['status_address']-DRAM_BASE
                wanted[at:at+2048]=int(rejected).to_bytes(4,'little')+bytes(2044)
                cases.append(dict(label=label,memory=memory.hex(),expected_memory=wanted.hex()))
            groups.append(dict(kind=f'quantized_epilogue_{dtype}_operand{operand}_n{n}',binary=binary.hex(),cases=cases))
        # Direct completed-readout injection isolates the raw-admission guard.
        source,_,_=case('f32',1,m=32,n=32,k=32,batch=(2,));plan=prepare_source(source);layout=execution_layout(plan)
        _,program=compile_fixture(raw_fixture(plan,layout));binary=b''.join(row['word_u32'].to_bytes(4,'little') for row in program['instructions'])
        cases=[]
        for label,word in (('negative_zero',0x8000),('nonfinite',0x7f80)):
            size=layout['input_addresses'][2]-DRAM_BASE+layout['output_bytes']+2048;memory=bytearray(size)
            raw=np.full((2,1024),0x3f80,dtype='<u2');raw[0].fill(word);memory[:raw.nbytes]=raw.tobytes()
            extra=np.full((2,32,32),0x4000,dtype='<u2');payload=packed_extra(extra,plan).tobytes();at=layout['input_addresses'][2]-DRAM_BASE
            memory[at:at+len(payload)]=payload
            expected=np.full((2,32,32),0x4000,dtype='<u2');expected[0].fill(0);wanted=bytearray(memory)
            payload=packed_extra(expected,plan).tobytes();at=layout['output_addresses'][0]-DRAM_BASE;wanted[at:at+len(payload)]=payload
            at=layout['status_address']-DRAM_BASE;wanted[at:at+2048]=b'\1\0\0\0'+bytes(2044)
            cases.append(dict(label=label,memory=memory.hex(),expected_memory=wanted.hex()))
        groups.append(dict(kind='quantized_epilogue_raw_clamping',binary=binary.hex(),cases=cases))
        with tempfile.TemporaryDirectory(prefix='atlas-quantized-epilogue-model-') as directory:
            directory=Path(directory);spec=directory/'spec.json';receipt=directory/'receipt.json';spec.write_text(json.dumps(groups))
            result=subprocess.run([os.environ['IMPORT_MODEL_PYTHON'],str(Path(__file__).with_name('device_boundary_model.py')),
                str(spec),str(receipt),str(ROOT)],capture_output=True,text=True,timeout=180)
            self.assertEqual(result.returncode,0,result.stdout+result.stderr)
            output=os.environ.get('ATLAS_QUANTIZED_EPILOGUE_RECEIPT')
            if output:Path(output).write_bytes(receipt.read_bytes())


if __name__=='__main__':unittest.main()
