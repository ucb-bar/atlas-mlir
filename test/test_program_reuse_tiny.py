"""Symbolic origin/permutation/order proofs; no numerical model execution."""
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).absolute().parents[1] / 'tools'))
import atlas_program_reuse_tiny as producer

SHAPE = (128, 2048, 2048)
OUTPUT = 4456704


from reuse_proof import prove as prove_program


def prove(source, exponent=-14):
    return prove_program(source, *SHAPE, exponent)


class TinyReuseTests(unittest.TestCase):


    def test_negative_controls_detect_semantic_breakage(self):
        source = producer.assembly(*SHAPE, OUTPUT, -14)
        changes = {
            'A bank alias': ('li x1, 0x10000', 'li x1, 0'),
            'incomplete preload': ('li x21, 256', 'li x21, 255'),
            'B DRAM stride': ('addi x20, x20, 1024', 'addi x20, x20, 2048'),
            'missing transpose': ('vtrpose.xlu m1, m1', 'nop'),
            'missing cached transpose': ('vstore m1, 0(x19)', 'nop'),
            'K order': ('addi x7, x7, 256', 'addi x7, x7, 512'),
            'accumulator alias': ('vmatmul.acc.mxu0 acc1', 'vmatmul.acc.mxu0 acc0'),
            'reset continuation': ('vmatmul.acc.mxu0', 'vmatmul.mxu0'),
            'wrong pop': ('vmatpop.bf16.acc.mxu0 m2, acc1', 'vmatpop.bf16.acc.mxu0 m2, acc0'),
            'missing scale': ('vmul.bf16 m2, m2, m4', 'nop'),
            'wrong scale': ('vli.all m4, 14464', 'vli.all m4, 14592'),
            'output scratch alias': ('li x3, 0x4000', 'li x3, 0x3f00'),
            'swapped BF16 halves': ('vstore m3, 0(x9)', 'vstore m2, 0(x9)'),
            'output row stride': ('li x22, 131072', 'li x22, 2048'),
            'output column alias': ('add x12, x12, x8', 'add x12, x12, x0'),
            'undrained DMA': ('dma.wait ch0', 'nop'),
            'unwaited prefetch': ('dma.wait ch1', 'nop'),
            'wrong DRAM page': ('li x30, 1', 'li x30, 0'),
        }
        for name, (old, new) in changes.items():
            self.assertIn(old, source)
            with self.subTest(control=name), self.assertRaises((AssertionError, KeyError)):
                prove(source.replace(old, new))

    def test_admission(self):
        for shape in ((64, 2048, 2048), (128, 1024, 2048), (128, 2048, 512), (160, 2048, 2048)):
            with self.subTest(shape=shape), self.assertRaises(ValueError):
                producer.assembly(*shape, OUTPUT, -14)
            with self.assertRaises(ValueError):
                producer.dma_load_bytes(*shape)
        for exponent in (True, False, 0.0, float("inf"), float("nan"), -127, 128):
            with self.assertRaises(ValueError):
                producer.assembly(*SHAPE, OUTPUT, exponent)

        for offset in (0, OUTPUT - 1, OUTPUT + 1, float(OUTPUT), True):
            with self.subTest(offset=offset), self.assertRaises(ValueError):
                producer.assembly(*SHAPE, offset, -14)


if __name__ == '__main__':
    unittest.main()
