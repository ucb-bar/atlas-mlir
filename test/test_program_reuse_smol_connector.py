"""Prove exact operand order and DMA ownership before experimental execution."""
from pathlib import Path
import sys
import unittest
sys.path.insert(0, str(Path(__file__).absolute().parents[1] / 'tools'))
import atlas_program_reuse_smol_connector as producer
from reuse_proof import prove


class ConnectorTest(unittest.TestCase):
    def test_complete_symbolic_flow(self):
        m, k, n = producer.SHAPE
        source = producer.assembly(m, k, n, m*k+k*n+256, -15)
        result = prove(source, m, k, n, -15, producer.dma_load_bytes(m, k, n))
        self.assertEqual(result['load_bytes'], 24379392)
        self.assertEqual(result['store_bytes'], 122880)
        self.assertEqual(result['counts']['vmatpop.bf16.acc.mxu0'], 60)
        for changed in (source.replace('dma.wait ch1', 'nop'),
                        source.replace('vtrpose.xlu m1, m1', 'nop'),
                        source.replace('vmatmul.acc.mxu0', 'vmatmul.mxu0'),
                        source.replace('vmul.bf16 m2, m2, m4', 'nop')):
            with self.assertRaises(AssertionError):
                prove(changed, m, k, n, -15, producer.dma_load_bytes(m, k, n))

    def test_rejected_inputs_and_one_rescale_patch(self):
        for shape in ((64., 12288, 960), (32, 12288, 960), (64, 12256, 960)):
            with self.assertRaises(ValueError):
                producer.layout(*shape)
        args = (64, 12288, 960, 12583168)
        for exponent in (None, True, -127, 128):
            with self.assertRaises(ValueError):
                producer.assembly(*args, exponent)
        first = producer.assembly(*args, -15).splitlines()
        second = producer.assembly(*args, -14).splitlines()
        self.assertEqual(sum(a != b for a, b in zip(first, second)), 1)


if __name__ == '__main__':
    unittest.main()
