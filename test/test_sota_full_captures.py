"""Portable full-source/tile admission and whole-tensor quantization tests."""
from pathlib import Path
import json,sys,tempfile,unittest
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
from atlas_sota_virtual import (encode_inputs,encode_full_inputs,expected_output,load_source,
 materialize,select_tile,sha,validate_arrays,validate_full_site)


class FullCaptureTests(unittest.TestCase):
    def arrays(self,m=64,k=64,n=64):
        return {'A':np.ones((m,k),np.float32),'B':np.ones((k,n),np.float32),'native_C':np.zeros((m,n),np.float32)}

    def manifest(self,arrays):
        return dict(schema_version=1,source={'kind':'portable synthetic admission only'},source_dtype='float32',
                    site=dict(k_range=[0,arrays['A'].shape[1]],complete_k=True,
                              matrix_shapes={n:list(arrays[n].shape) for n in ('A','B','native_C')}))

    def write(self,folder,arrays,doc):
        path=folder/'operands.npz';manifest=folder/'manifest.json';np.savez(path,**arrays);doc['npz_sha256']=sha(path);manifest.write_text(json.dumps(doc))
        return path,manifest,dict(npz_sha256=sha(path),manifest_sha256=sha(manifest))

    def test_optimized_full_recipe_matches_retained_scalar_recipe(self):
        rng=np.random.default_rng(194);arrays=self.arrays()
        for name in ('A','B'):arrays[name][:]=rng.uniform(-10,10,size=arrays[name].shape).astype(np.float32)
        # Include float32 tie-rounding and signedzero, independently rounded by old scalar oracle.
        bits=np.array([0,0x80000000,0x3f807fff,0x3f808000,0x3f808001,0x3f818000,0xbf808000,1,0x80000001],np.uint32)
        arrays['A'].flat[:len(bits)]=bits.view(np.float32)
        expected,scales=encode_inputs(arrays);actual,full_scales=encode_full_inputs(arrays)
        self.assertEqual(scales,full_scales)
        for n in expected:np.testing.assert_array_equal(actual[n],expected[n])

    def test_scales_before_slice_and_complete_K_coordinates(self):
        arrays=self.arrays();arrays['A'][-1,-1]=256;arrays['B'][-1,-1]=128
        full,scales=encode_full_inputs(arrays);selected,coords=select_tile(full,0,0)
        local,local_scales=encode_inputs({'A':arrays['A'][:32],'B':arrays['B'][:,:32]})
        self.assertEqual([scales[n]['scale_exponent'] for n in ('A','B')],[0,-1])
        self.assertEqual([local_scales[n]['scale_exponent'] for n in ('A','B')],[-8,-8])
        self.assertFalse(np.array_equal(selected['A'],local['A']))
        np.testing.assert_array_equal(selected['A'],full['A'][:32,:]);np.testing.assert_array_equal(selected['B'],full['B'][:,:32])
        self.assertEqual(coords,dict(rows=[0,32],output_columns=[0,32],k_range=[0,64],native_rows=32,padded_rows=0))
        for r,c in ((-32,0),(1,0),(64,0),(0,-32),(0,1),(0,64),(True,0)):
            with self.assertRaises(ValueError):select_tile(full,r,c)
        self.assertEqual(select_tile(full,32,32)[1]['rows'],[32,64])

    def test_padding_authentication_and_explicit_tail_labels(self):
        arrays=self.arrays();arrays['A'][49:]=0;doc=self.manifest(arrays)
        doc['site']|=dict(native_rows=49,valid_rows=49,padding=dict(kind='exported_matrix_zero_rows_only',rows=[49,64],native_masks_or_positions_extended=False))
        self.assertEqual(validate_full_site(arrays,doc),49)
        encoded,_=encode_full_inputs(arrays);tile,selection=select_tile(encoded,32,32,native_rows=49)
        self.assertEqual((selection['native_rows'],selection['padded_rows']),(17,15));self.assertTrue(np.all(tile['A'][17:]==0))
        arrays['A'][49,0]=1
        with self.assertRaises(ValueError):validate_full_site(arrays,doc)
        arrays['A'][49,0]=0;doc['site']['padding']['native_masks_or_positions_extended']=True
        with self.assertRaises(ValueError):validate_full_site(arrays,doc)
        doc['site']['padding']=False
        with self.assertRaises(ValueError):validate_full_site(arrays,doc)

    def test_full_raw_authentication_shape_extent_and_hash(self):
        arrays=self.arrays();doc=self.manifest(arrays)
        self.assertEqual(validate_arrays(arrays,'float32',allow_full=True),64)
        with self.assertRaises(ValueError):validate_arrays(arrays,'float32')
        for name in ('A','B','native_C'):arrays[name+'_bf16']=(arrays[name].view(np.uint32)>>16).astype(np.uint16)
        self.assertEqual(validate_arrays(arrays,'bfloat16',allow_full=True),64)
        del arrays['native_C_bf16']
        with self.assertRaises(ValueError):validate_arrays(arrays,'bfloat16',allow_full=True)
        arrays['native_C_bf16']=np.zeros((64,64),np.uint16);arrays['B_bf16'][0,0]=0
        with self.assertRaises(ValueError):validate_arrays(arrays,'bfloat16',allow_full=True)
        arrays=self.arrays();doc=self.manifest(arrays)
        with tempfile.TemporaryDirectory() as temp:
            path,manifest,pins=self.write(Path(temp),arrays,doc);loaded,_=load_source(path,manifest,allow_full=True,**pins);self.assertEqual(loaded['A'].shape,(64,64))
            with self.assertRaises(ValueError):load_source(path,manifest,allow_full=True,npz_sha256='0'*64,manifest_sha256=sha(manifest))
            doc['site']['matrix_shapes']['A']=[32,64];manifest.write_text(json.dumps(doc))
            with self.assertRaises(ValueError):load_source(path,manifest,allow_full=True,npz_sha256=sha(path),manifest_sha256=sha(manifest))

    def test_selected_materialization_global_scale_and_diagnostic_C(self):
        arrays=self.arrays();arrays['A'][-1,-1]=256;arrays['B'][-1,-1]=128;arrays['native_C'].fill(999)
        with tempfile.TemporaryDirectory() as temp:
            folder=Path(temp);path,manifest,pins=self.write(folder,arrays,self.manifest(arrays));r=materialize(path,manifest,folder/'out',name='full',row_start=0,column_start=0,**pins)
            self.assertEqual(r['source_shape'],[64,64,64]);self.assertEqual(r['output_rescale_exponent'],-1);self.assertFalse(r['native_C_used_as_oracle'])
            self.assertEqual((folder/'out/expected-output.bin').read_bytes(),b'\x80\x42'*1024)
            with self.assertRaises(ValueError):materialize(path,manifest,folder/'bad',name='full',row_start=0,**pins)

    def test_default_payload_byte_identity_against_preextension_snapshot(self):
        arrays=self.arrays(32,64,32);doc=self.manifest(arrays)
        with tempfile.TemporaryDirectory() as temp:
            folder=Path(temp);path,manifest,pins=self.write(folder,arrays,doc);materialize(path,manifest,folder/'out',name='legacy',**pins)
            expected={'source.mlir':'bfccdaa636242aea87c6ffd2cecea251e16d8097d35338a7e5db0e6f4bf6c5dc','initial.bin':'9f219d7fafe4de1f264fe50dc0385c81051f8bbb17e823f5db957b402be40a3c','expected.bin':'5da3d88da8ce95c11765aa09a823f2c4d7291edf219590378ac2e14a8ca4752f','expected-output.bin':'80217e7ba007fb93c85ed9d86d65bd1da0a9d0b4488d4183ea446b8f8109e1b4'}
            self.assertEqual({n:sha(folder/'out'/n) for n in expected},expected)

if __name__=='__main__':unittest.main()
