"""Independent scalar-flow/operand provenance proof, without numerical timing."""

from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).absolute().parents[1] / 'tools'))
import atlas_program_reuse_stream as producer

from reuse_proof import prove


class StreamReuseTests(unittest.TestCase):
    def test_order_addresses_capacity_and_compulsory_traffic(self):
        for shape in ((128,3072,768), (32,32,32), (64,64,96), (96,3072,64), (128,2048,32)):
            for exponent in (None, -15):
                m,k,n = shape
                with self.subTest(shape=shape, exponent=exponent):
                    result = prove(producer.assembly(m,k,n,m*k+k*n+256,exponent),m,k,n,exponent,m*k+k*n)
                    self.assertEqual((result['load_bytes'],result['store_bytes']), (m*k+k*n,2*m*n))

    def test_row2_bank_transition_and_disjoint_regions(self):
        plan = producer.layout(128,3072,768)
        self.assertEqual(plan, dict(a_bank1_bytes=262144,a_bank0_bytes=131072,
                                   b_word=32768,b_bytes=98304,output_word=57344))
        addresses = [producer.a_word(i)*4 for i in range(0,393216,1024)]
        self.assertEqual(len(set(addresses)),384)
        self.assertEqual(addresses[255:257],[523264,0])
        self.assertTrue(all(a >= 262144 or a + 1024 <= 131072 for a in addresses))

    def test_refuses_overflow_and_bad_abi(self):
        for shape in ((160,3072,768),(128,4096,768),(0,32,32),(32,31,32)):
            with self.assertRaises(ValueError):
                producer.assembly(*shape,10000000)
        for offset in (0,8193,2**32-32):
            with self.assertRaises(ValueError):
                producer.assembly(64,64,64,offset)

    def test_negative_controls(self):
        m=k=n=64
        source = producer.assembly(m,k,n,m*k+k*n+256,-15)
        for old,new in [('vtrpose.xlu m1, m1','nop'),('vmatmul.acc.mxu0','vmatmul.mxu0'),
                        ('acc1','acc0'),('vstore m3, 0(x9)','vstore m2, 0(x9)'),
                        ('add x22, x22, x16','add x22, x22, x8'),
                        ('addi x19, x19, 256','addi x19, x19, 512')]:
            with self.subTest(mutation=old), self.assertRaises((AssertionError,KeyError)):
                prove(source.replace(old,new),m,k,n,-15,m*k+k*n)


if __name__ == '__main__':
    unittest.main()
