"""Narrow semantic gates: addresses, ordered K chains, slots and patch ABI."""
import io
import unittest
from atlas_program_reuse_resident import admission, assembly, dma_load_bytes

from reuse_proof import prove


class ResidentReuseTests(unittest.TestCase):
    def test_target_capacity_and_compulsory_dma(self):
        r = admission(128,768,768,688384,-19)
        self.assertEqual(r['a_byte_range'], [262144,360448])
        self.assertEqual(r['b_column_byte_range'], [360448,385024])
        self.assertEqual(r['resident_bytes'], 122880)
        self.assertEqual(dma_load_bytes(128,768,768), 688128)
        self.assertEqual(r['dma_store_bytes'], 196608)
    def test_complete_address_and_k_semantics(self):
        for shape in ((64,32,32),(64,64,64),(128,768,768)):
            m, k, n = shape
            with self.subTest(shape=shape):
                source = assembly(m, k, n, m*k+k*n+256, -19)
                counts = prove(source, m, k, n, -19, m*k+k*n)['counts']
                self.assertEqual(counts['vli.all'], 1)
                self.assertEqual(counts['vtrpose.xlu'], k*n//1024)
                self.assertEqual(counts['vmatpush.weight.mxu0'], (m//64)*(k//32)*(n//32))
    def test_reject_unsupported_layouts(self):
        for args in ((32,768,768,0,-19),(128,2048,768,0,-19),
                     (128,768,768,688383,-19),(128,768,768,688384,None),
                     (128,768,768,688384,-127)):
            with self.subTest(args=args):
                with self.assertRaises(ValueError): admission(*args)
    def test_original_rescale_patch_protocol(self):
        # Optional dependency supplied by the existing model environment.
        from npu_model.util.converter import input_to_program
        words = []
        for exponent in (-126,-19,0,127):
            p = input_to_program(io.StringIO(assembly(128,768,768,688384,exponent)))
            raw = p.assemble()
            self.assertLessEqual(len(raw), 8192)
            indices = [i for i, w in enumerate(raw) if (w & ((1<<23)-1)) == 0x25f]
            self.assertEqual(len(indices), 1)
            index = indices[0]
            self.assertEqual(raw[index], ((exponent+127)<<23)|0x25f)
            words.append((index,raw))
        for index, raw in words:
            self.assertEqual(index, words[0][0])
            changed = [i for i,(a,b) in enumerate(zip(raw,words[0][1])) if a!=b]
            self.assertTrue(changed in ([],[index]))

if __name__ == '__main__': unittest.main()
