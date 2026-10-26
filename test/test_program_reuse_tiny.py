"""Symbolic origin/permutation/order proofs; no numerical model execution."""
from collections import Counter
from pathlib import Path
import re
import sys
import unittest

sys.path.insert(0, str(Path(__file__).absolute().parents[1] / 'tools'))
import atlas_program_reuse_tiny as producer

SHAPE = (128, 2048, 2048)
OUTPUT = 4456704


def transpose(values):
    return tuple(values[r * 32 + c] for c in range(32) for r in range(32))


def prove(source, exponent=-14):
    """Follow all original byte identities and ordered contractions to DRAM.

    Symbolic output bytes represent the unchanged ordered BF16 MXU0 chain,
    BF16 pop halves and optional scale. This is not numerical qualification.
    """
    lines = source.splitlines()
    labels = {line[:-1]: i for i, line in enumerate(lines) if line.endswith(':')}
    regs, memory, mregs, scales, weights, accs = {'x0': 0}, {}, {}, {}, {}, {}
    loads, stores, popped, products = [], [], set(), Counter()
    pending, configured, pc = False, False, 0

    def address(arg):
        return regs[re.fullmatch(r'0\((x\d+)\)', arg).group(1)] * 4

    def put(base, values, scale=None):
        assert base % 1024 == 0 and 0 <= base <= 524288 - 1024
        assert len(values) == 1024
        memory[base] = (values, scale)

    for _ in range(500000):
        line = lines[pc]
        op, *args = line.replace(',', '').split()
        pc += 1
        if op.endswith(':') or op in ('nop', 'delay'):
            continue
        if op == 'li':
            regs[args[0]] = int(args[1], 0)
        elif op in ('add', 'addi'):
            regs[args[0]] = regs[args[1]] + (regs[args[2]] if op == 'add' else int(args[2]))
        elif op in ('beq', 'jal'):
            assert lines[pc] == 'nop', 'inert branch delay slot'
            if op == 'jal' or regs[args[0]] == regs[args[1]]:
                pc = labels[args[-1]]
        elif op == 'dma.config':
            assert regs[args[0]] == 1 and args[1] == 'ch0', '4-GiB DRAM page'
            configured = True
        elif op == 'dma.wait':
            assert args == ['ch0']
            pending = False
        elif op == 'dma.load':
            assert configured and not pending and args[3] == 'ch0'
            dest, origin, size = (regs[x] for x in args[:3])
            dest *= 4
            assert size == 1024 and origin % 1024 == 0 and 0 <= origin < 4456448
            if origin < 262144:
                assert dest == 262144 + origin, 'A bank capacity/address'
            else:
                assert dest == (origin - 262144) % 65536, 'B column panel address'
            put(dest, tuple(range(origin, origin + size)))
            loads.append((dest, origin, size))
            pending = True
        elif op == 'vload':
            assert not pending
            mregs[args[0]], scales[args[0]] = memory[address(args[1])]
        elif op == 'vtrpose.xlu':
            mregs[args[0]] = transpose(mregs[args[1]])
        elif op == 'vmatpush.weight.mxu0':
            assert args[0] in ('w0', 'w1')
            weights[args[0]] = mregs[args[1]]
        elif op in ('vmatmul.mxu0', 'vmatmul.acc.mxu0'):
            slot, activation, weight = args
            assert slot in ('acc0', 'acc1')
            a, b = mregs[activation], weights[weight]
            origin_a, origin_b = a[0], b[0]
            row, inner = divmod(origin_a // 1024, 64)
            column, b_inner = divmod((origin_b - 262144) // 1024, 64)
            assert 0 <= row < 4 and 0 <= column < 64 and inner == b_inner
            assert a == tuple(range(origin_a, origin_a + 1024)), 'original A byte order'
            assert b == transpose(tuple(range(origin_b, origin_b + 1024))), 'original B transpose'
            if op == 'vmatmul.mxu0':
                assert inner == 0, 'reset only first K tile'
                accs[slot] = []
            chain = accs[slot]
            assert inner == len(chain), 'strict per-output K order'
            assert not chain or chain[0][:2] == (row, column), 'accumulator output identity'
            chain.append((row, column, inner))
            products[row, column] += 1
        elif op == 'vmatpop.bf16.acc.mxu0':
            assert args[0] == 'm2'
            chain = accs[args[1]]
            assert len(chain) == 64
            row, column, _ = chain[0]
            assert (row, column) not in popped
            popped.add((row, column))
            tokens = tuple(-((row * 64 + column) * 2048 + i + 1) for i in range(2048))
            mregs['m2'], mregs['m3'] = tokens[:1024], tokens[1024:]
            scales['m2'] = scales['m3'] = None
        elif op == 'vli.all':
            assert args[0] == 'm4'
            mregs['m4'] = int(args[1])
        elif op == 'vmul.bf16':
            assert args == ['m2', 'm2', 'm4'] and exponent is not None
            assert mregs['m4'] == (exponent + 127) << 7
            assert scales['m2'] is None and scales['m3'] is None, 'scale exactly once'
            scales['m2'] = scales['m3'] = exponent
        elif op == 'vstore':
            dest = address(args[1])
            assert not pending
            if args[0] == 'm1':
                assert 0 <= dest < 65536, 'B cache stays below output scratch'
            else:
                assert args[0] in ('m2', 'm3') and dest in (65536, 66560), 'output scratch bound'
            put(dest, mregs[args[0]], scales.get(args[0]))
        elif op == 'dma.store':
            assert configured and not pending and args[3] == 'ch0'
            dest, src, size = (regs[x] for x in args[:3])
            assert size == 2048 and src * 4 == 65536
            assert OUTPUT <= dest < OUTPUT + 524288 and (dest - OUTPUT) % 2048 == 0
            low, high = memory[src * 4], memory[src * 4 + 1024]
            output = (dest - OUTPUT) // 2048
            expected = tuple(-(output * 2048 + i + 1) for i in range(2048))
            assert low[0] + high[0] == expected, 'output tile and BF16 half ordering'
            assert low[1] == high[1] == exponent, 'exact rescale before store'
            stores.append((dest, src, size))
            pending = True
        elif op == 'ecall':
            assert not pending and configured
            assert len(loads) == 4352 and [x[1] for x in loads] == list(range(0, 4456448, 1024))
            assert len(stores) == 256 and sorted(x[0] for x in stores) == list(range(OUTPUT, OUTPUT + 524288, 2048))
            assert products == Counter({(r, c): 64 for r in range(4) for c in range(64)})
            assert len(popped) == 256
            return {'load_bytes': sum(x[2] for x in loads), 'store_bytes': sum(x[2] for x in stores),
                    'products': sum(products.values()), 'outputs': len(stores)}
        else:
            raise AssertionError('unsupported instruction: ' + op)
    raise AssertionError('program did not terminate')


class TinyReuseTests(unittest.TestCase):
    def test_complete_origin_order_capacity_and_output_proof(self):
        self.assertEqual(prove(producer.assembly(*SHAPE, OUTPUT, -14)),
                         {'load_bytes': 4456448, 'store_bytes': 524288, 'products': 16384, 'outputs': 256})
        self.assertEqual(producer.dma_load_bytes(*SHAPE), 4456448)

    def test_unit_scale_and_extreme_rescale(self):
        for exponent in (None, -126, 127):
            with self.subTest(exponent=exponent):
                prove(producer.assembly(*SHAPE, OUTPUT, exponent), exponent)

    def test_one_static_rescale_patch_site(self):
        base = producer.assembly(*SHAPE, OUTPUT, 0).splitlines()
        positions = [i for i, line in enumerate(base) if line.startswith('vli.all ')]
        self.assertEqual(len(positions), 1)
        for exponent in (-126, -14, -1, 1, 127):
            alternate = producer.assembly(*SHAPE, OUTPUT, exponent).splitlines()
            patched = base.copy()
            patched[positions[0]] = f'vli.all m4, {(exponent + 127) << 7}'
            self.assertEqual(patched, alternate)

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
            'wrong DRAM page': ('li x30, 1', 'li x30, 0'),
        }
        for name, (old, new) in changes.items():
            self.assertIn(old, source)
            with self.subTest(control=name), self.assertRaises((AssertionError, KeyError)):
                prove(source.replace(old, new))

    def test_admission(self):
        for shape in ((64, 2048, 2048), (128, 1024, 2048), (128, 2048, 256), (160, 2048, 2048)):
            with self.subTest(shape=shape), self.assertRaises(ValueError):
                producer.assembly(*shape, OUTPUT, -14)
            with self.assertRaises(ValueError):
                producer.dma_load_bytes(*shape)
        for exponent in (-127, 128):
            with self.assertRaises(ValueError):
                producer.assembly(*SHAPE, OUTPUT, exponent)


if __name__ == '__main__':
    unittest.main()
