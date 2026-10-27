"""Shared symbolic coverage and mutations for all public reuse recipes."""
import importlib
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).absolute().parents[1] / 'tools'))
import atlas_program as program
from reuse_proof import prove


def assembly(shape, exponent=-13):
    m, k, n = shape
    return program.assembly(m, k, n, m * k + k * n + 256, exponent, traffic_reuse=True)


class ReuseProofTests(unittest.TestCase):
    def test_all_dispatch_shapes(self):
        for m, k, n in program._TRAFFIC_REUSE_MODULES:
            with self.subTest(shape=(m, k, n)):
                result = prove(assembly((m, k, n)), m, k, n, -13)
                self.assertEqual(result['store_bytes'], 2 * m * n)

    def test_optional_direct_rescale(self):
        for shape, name in program._TRAFFIC_REUSE_MODULES.items():
            if name not in ('atlas_program_reuse_tiny', 'atlas_program_reuse_tiny_breadth',
                            'atlas_program_reuse_stream'):
                continue
            m, k, n = shape
            source = importlib.import_module(name).assembly(m, k, n, m * k + k * n + 256, None)
            with self.subTest(shape=shape):
                prove(source, m, k, n, None)

    def test_one_static_rescale_patch_for_every_shape(self):
        for shape in program._TRAFFIC_REUSE_MODULES:
            original = assembly(shape, 0).splitlines()
            sites = [i for i, line in enumerate(original) if line.startswith('vli.all ')]
            with self.subTest(shape=shape):
                self.assertEqual(len(sites), 1)
                for exponent in (-126, -13, 127):
                    expected = original.copy()
                    expected[sites[0]] = f'vli.all m4, {(exponent + 127) << 7}'
                    self.assertEqual(expected, assembly(shape, exponent).splitlines())

    def reject_mutations(self, shape, changes):
        source = assembly(shape)
        for old, new in changes:
            self.assertIn(old, source)
            with self.subTest(shape=shape, mutation=old), self.assertRaises((AssertionError, KeyError)):
                prove(source.replace(old, new), *shape, -13)

    def test_blocked_negative_controls(self):
        for shape in ((224, 2304, 256), (128, 2560, 960)):
            self.reject_mutations(shape, [
                ('vtrpose.xlu m1, m1', 'nop'),
                ('vmatmul.acc.mxu0', 'vmatmul.mxu0'),
                ('acc1', 'acc0'),
                ('vstore m3, 0(x9)', 'vstore m2, 0(x9)'),
                ('add x22, x22, x16', 'add x22, x22, x8'),
                ('addi x19, x19, 256', 'addi x19, x19, 512'),
                ('dma.wait ch0', 'nop'),
                ('dma.wait ch1', 'nop'),
                ('vmul.bf16 m2, m2, m4', 'nop')])

    def test_compact_resnet_pair_loop_negative_controls(self):
        self.reject_mutations((12544, 160, 64), [
            ('add x23, x23, x25', 'add x23, x23, x8'),
            ('addi x24, x24, -1', 'addi x24, x24, -2'),
            ('add x7, x7, x23', 'add x7, x7, x0')])

    def test_tiny_and_prefetch_negative_controls(self):
        for shape in ((128, 2048, 256), (32, 5632, 1024), (128, 5632, 1024)):
            changes = [
                ('vtrpose.xlu m1, m1', 'nop'),
                ('vmatmul.acc.mxu0', 'vmatmul.mxu0'),
                ('vstore m3, 0(x9)', 'vstore m2, 0(x9)'),
                ('dma.wait ch0', 'nop'),
                ('dma.wait ch1', 'nop'),
                ('vmul.bf16 m2, m2, m4', 'nop'),
                ('li x30, 1', 'li x30, 0')]
            if shape[0] != 32:
                changes += [('add x23, x23, x22', 'add x23, x23, x8')]
            if shape[1] == 5632:
                changes += [('dma.config x30, ch1', 'nop')]
            if shape == (128, 5632, 1024):
                changes += [('addi x7, x7, 256', 'addi x7, x7, 512'), ('add x12, x12, x22', 'add x12, x12, x0')]
            self.reject_mutations(shape, changes)

    def test_column_prefetch_ownership(self):
        for shape in ((64, 736, 2048), (128, 768, 768), (128, 2048, 2048),
                      (128, 3072, 768), (12544, 160, 64)):
            self.reject_mutations(shape, [
                ('dma.load x24, x20, x5, ch1', 'dma.load x19, x20, x5, ch1'),
                ('addi x24, x19, 256', 'addi x24, x19, 512'),
                ('dma.wait ch1', 'nop')])

    def test_scale_and_memory_boundary_negative_controls(self):
        self.reject_mutations((128, 2048, 256), [
            ('vli.all m4, 14592', 'vli.all m4, 14720'),
            ('vli.all m4, 14592', 'vli.all m4, 14592\nvli.all m4, 14592'),
            # Correct values scaled too late must not validate earlier stores.
            ('vmul.bf16 m2, m2, m4\ndelay 66\nvstore m2, 0(x3)\ndelay 34\nvstore m3, 0(x9)',
             'vstore m2, 0(x3)\ndelay 34\nvstore m3, 0(x9)\ndelay 34\nvmul.bf16 m2, m2, m4'),
            ('li x1, 0x10000', 'li x1, 0x20000'),
            ('li x3, 0x4000', 'li x3, 0xff00'),
            ('li x9, 0x4100', 'li x9, 0x10000')])
        self.reject_mutations((32, 5632, 1024), [
            ('li x19, 65536', 'li x19, 65280')])  # 4-KiB DMA crosses banks.


if __name__ == '__main__':
    unittest.main()
