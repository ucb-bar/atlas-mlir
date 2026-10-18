"""Source admission, complete-K preservation and independent target oracle."""
from pathlib import Path
import hashlib,json,sys,tempfile,unittest
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
from atlas_sota_virtual import (ORACLE_SHA256,encode_inputs,expected_output,load_source,
                                oracle,sha,validate_arrays,virtual_source)
from atlas_egglog_ir import parse_program


class SotaVirtualTests(unittest.TestCase):
    def arrays(self,k=64):
        return {'A':np.ones((32,k),dtype=np.float32),'B':np.ones((k,32),dtype=np.float32),
                'native_C':np.zeros((32,32),dtype=np.float32)}

    def test_exact_oracle_pin_and_native_C_independence(self):
        self.assertEqual(sha(oracle.__file__),ORACLE_SHA256)
        arrays=self.arrays();encoded,scales=encode_inputs(arrays)
        exponent=sum(x['scale_exponent'] for x in scales.values())
        expected,_=expected_output(encoded,exponent)
        self.assertEqual(expected,b'\x80\x42'*1024) # 64.0 in BF16, physical halves.
        arrays['native_C'].fill(123456)
        changed,changed_scales=encode_inputs(arrays)
        self.assertEqual(scales,changed_scales)
        self.assertEqual(expected_output(changed,exponent)[0],expected)

    def test_complete_K_and_original_order(self):
        for k in (64,960,2048):
            program=parse_program(virtual_source(k,-10,0x90040000))
            operations=program.operations
            self.assertEqual(sum(x.op=='virtual_mxu_reset' for x in operations),1)
            accumulates=[x for x in operations if x.op=='virtual_mxu_accumulate']
            self.assertEqual(len(accumulates),k//32-1)
            for tile,op in enumerate(accumulates,1):
                self.assertEqual(op.operands[1:],(f'%a{tile}',f'%w{tile}',f'%acc{tile-1}'))
            self.assertEqual(sum(x.op=='virtual_xlu_transpose' for x in operations),k//32)
            self.assertEqual(sum(x.op=='virtual_dma_load_fp8' for x in operations),2*k//32)
            self.assertEqual(sum(x.op=='virtual_vpu_binary' for x in operations),1)
            self.assertTrue(all(dict(x.attrs).get('unit')=='0 : i32' for x in operations if x.op=='virtual_mxu_load_weight'))
            self.assertFalse(any(x.op=='virtual_pack_fp8' for x in operations))

    def test_virtual_memory_ABI_rejection(self):
        with self.assertRaises(ValueError):virtual_source(64,-10,0x90001020)
        with self.assertRaises(ValueError):virtual_source(64,-10,0x90000000)

    def test_shape_dtype_and_raw_authentication_rejections(self):
        arrays=self.arrays()
        self.assertEqual(validate_arrays(arrays,'float32'),64)
        with self.assertRaises(ValueError):validate_arrays(arrays,'bfloat16')
        for name in ('A','B'):
            arrays[name+'_bf16']=np.full(arrays[name].shape,0x3F80,dtype=np.uint16)
        self.assertEqual(validate_arrays(arrays,'bfloat16'),64)
        arrays['A_bf16'][0,0]=0
        with self.assertRaises(ValueError):validate_arrays(arrays,'bfloat16')
        arrays=self.arrays(33)
        with self.assertRaises(ValueError):validate_arrays(arrays,'float32')
        arrays=self.arrays();arrays['A'][0,0]=np.inf
        with self.assertRaises(ValueError):validate_arrays(arrays,'float32')

    def test_pinned_archive_and_complete_K_admission(self):
        with tempfile.TemporaryDirectory() as temp:
            base=Path(temp);input=base/'operands.npz';manifest=base/'manifest.json'
            np.savez(input,**self.arrays())
            doc={'schema_version':1,'source':{'kind':'synthetic_admission_test_only'},'source_dtype':'float32',
                 'npz_sha256':sha(input),'site':{'k_range':[0,64],'complete_k':True}}
            manifest.write_text(json.dumps(doc))
            args={'npz_sha256':sha(input),'manifest_sha256':sha(manifest)}
            arrays,_=load_source(input,manifest,**args)
            self.assertEqual(arrays['A'].shape,(32,64))
            with self.assertRaises(ValueError):load_source(input,manifest,npz_sha256='0'*64,manifest_sha256=sha(manifest))
            doc['site']['k_range']=[0,32];manifest.write_text(json.dumps(doc))
            with self.assertRaises(ValueError):load_source(input,manifest,npz_sha256=sha(input),manifest_sha256=sha(manifest))
            doc['site']={'k_range':[0,64]};manifest.write_text(json.dumps(doc))
            with self.assertRaises(ValueError):load_source(input,manifest,npz_sha256=sha(input),manifest_sha256=sha(manifest))


if __name__=='__main__':unittest.main()
