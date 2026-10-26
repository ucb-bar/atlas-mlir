"""Full symbolic address/order proof including asynchronous prefetch ownership."""
from collections import Counter
import hashlib
from pathlib import Path
import re
import sys
import unittest
sys.path.insert(0, str(Path(__file__).absolute().parents[1] / 'tools'))
import atlas_program_reuse_tiny_breadth as producer


def prove(source, shape, exponent):
    m, k, n = shape
    kt, nt = k // 32, n // 32
    output = m * k + k * n + 256
    lines = source.splitlines()
    labels = {line[:-1]: i for i, line in enumerate(lines) if line.endswith(':')}
    regs, memory, mregs, accs, pending, configured = {'x0': 0}, {}, {}, {}, {}, set()
    counts, stores, popped = Counter(), set(), set()
    pc = 0
    def addr(arg):
        return regs[re.fullmatch(r'0\((x\d+)\)', arg).group(1)] * 4
    def accessible(base, size):
        assert base % 1024 == 0 and 0 <= base < base + size <= 524288
        assert base // 262144 == (base + size - 1) // 262144
        for _, start, length in pending.values():
            assert base + size <= start or start + length <= base, 'pending DMA hazard'
    for _ in range(1500000):
        op, *args = lines[pc].replace(',', '').split(); pc += 1
        if op.endswith(':') or op in ('nop', 'delay'):continue
        if op == 'li':regs[args[0]] = int(args[1], 0)
        elif op in ('add', 'addi'):
            regs[args[0]] = regs[args[1]] + (regs[args[2]] if op == 'add' else int(args[2]))
        elif op in ('beq', 'jal'):
            assert lines[pc] == 'nop'
            if op == 'jal' or regs[args[0]] == regs[args[1]]:pc = labels[args[-1]]
        elif op == 'dma.config':
            assert regs[args[0]] == 1
            configured.add(args[1])
        elif op == 'dma.wait':
            assert args[0] in pending, 'wait matches command'
            writes, _, _ = pending.pop(args[0]);memory.update(writes)
        elif op in ('dma.load', 'dma.store'):
            first, second, size = (regs[x] for x in args[:3]);ch = args[3]
            assert ch in configured and ch not in pending
            base = (first if op == 'dma.load' else second) * 4
            accessible(base, size)
            assert 0 < size <= 4096 and size % 1024 == 0
            writes = {}
            if op == 'dma.load':
                assert 0 <= second < second + size <= m * k + k * n and second % 1024 == 0
                for offset in range(0, size, 1024):
                    writes[base + offset] = ('input', second + offset, False)
                    counts[second + offset] += 1
            else:
                assert size == 2048 and output <= first < output + m * n * 2
                assert (first - output) % 2048 == 0 and first not in stores
                r, c = divmod((first - output) // 2048, nt)
                assert memory[base] == ('out', r, c, 0, exponent)
                assert memory[base + 1024] == ('out', r, c, 1, exponent)
                stores.add(first)
            pending[ch] = (writes, base, size)
        elif op == 'vload':
            base = addr(args[1]);accessible(base, 1024);mregs[args[0]] = memory[base]
        elif op == 'vtrpose.xlu':
            kind, origin, transposed = mregs[args[1]];assert kind == 'input' and not transposed
            mregs[args[0]] = kind, origin, True
        elif op == 'vmatpush.weight.mxu0':
            assert args[0] == 'w0';weight = mregs[args[1]]
        elif op in ('vmatmul.mxu0', 'vmatmul.acc.mxu0'):
            slot, activation, wb = args;assert slot in ('acc0', 'acc1') and wb == 'w0'
            a, b = mregs[activation], weight
            assert a[0] == b[0] == 'input' and not a[2] and b[2]
            assert 0 <= a[1] < m * k <= b[1] < m * k + k * n
            r, inner = divmod(a[1] // 1024, kt)
            c, binner = divmod((b[1] - m * k) // 1024, kt)
            assert inner == binner
            if op == 'vmatmul.mxu0':assert inner == 0;accs[slot] = (r, c, 0)
            assert accs[slot] == (r, c, inner), 'exact increasing K, output ownership'
            accs[slot] = r, c, inner + 1
        elif op == 'vmatpop.bf16.acc.mxu0':
            assert args[0] == 'm2';r, c, inner = accs[args[1]]
            assert inner == kt and (r, c) not in popped;popped.add((r, c))
            for half in range(2):mregs[f'm{2 + half}'] = ('out', r, c, half, None)
        elif op == 'vli.all':
            assert args[0] == 'm4';scale = int(args[1]);assert scale == (exponent + 127) << 7
        elif op == 'vmul.bf16':
            assert args == ['m2', 'm2', 'm4'] and scale == (exponent + 127) << 7
            for half in range(2):
                old = mregs[f'm{2 + half}'];assert old[-1] is None
                mregs[f'm{2 + half}'] = old[:-1] + (exponent,)
        elif op == 'vstore':
            base = addr(args[1]);accessible(base, 1024);memory[base] = mregs[args[0]]
        elif op == 'ecall':
            assert not pending
            expected = Counter({i: 1 if i < m * k else max(1, m // 64)
                                for i in range(0, m * k + k * n, 1024)})
            assert counts == expected
            assert stores == set(range(output, output + m * n * 2, 2048))
            assert popped == {(r, c) for r in range(m // 32) for c in range(nt)}
            return sum(counts.values()) * 1024
        else:raise AssertionError(op)
    raise AssertionError('nontermination')


class BreadthTests(unittest.TestCase):
    def test_original_two_programs_are_unchanged(self):
        originals = {
            (128,5632,1024): 'dc942a73b9e8e61b770f5b44fd1411326ad84647bbbc1cbb0d1b70ef8dc5fbd2',
            (32,2048,2048): '6bbe476863b12a923462f8bff7b4285765c6bfadc36cc27a6969433b151a66e5',
        }
        for (m,k,n), expected in originals.items():
            with self.subTest(shape=(m,k,n)):
                source = producer.assembly(m,k,n,m*k+k*n+256,0)
                self.assertEqual(hashlib.sha256(source.encode()).hexdigest(), expected)

    def test_all_addresses_order_hazards_and_extreme_scales(self):
        for shape in producer.SHAPES:
            for exponent in (None, -126, -18, 127):
                with self.subTest(shape=shape, exponent=exponent):
                    m, k, n = shape
                    self.assertEqual(prove(producer.assembly(*shape, m*k+k*n+256, exponent), shape, exponent),
                                     producer.dma_load_bytes(*shape))
    def test_unique_host_patch(self):
        for m, k, n in producer.SHAPES:
            a = producer.assembly(m, k, n, m*k+k*n+256, -18).splitlines()
            b = producer.assembly(m, k, n, m*k+k*n+256, -19).splitlines()
            diffs = [(x, y) for x, y in zip(a,b) if x != y]
            self.assertEqual(len(diffs), 1)
            self.assertTrue(diffs[0][0].startswith('vli.all m4,'))
    def test_negative_controls(self):
        shape = (128,5632,1024);source = producer.assembly(*shape,6488320,-18)
        changes = [('dma.wait ch1','nop'), ('vtrpose.xlu m1, m1','nop'),
                   ('addi x7, x7, 256','addi x7, x7, 512'),
                   ('vmatmul.acc.mxu0','vmatmul.mxu0'), ('vstore m3, 0(x9)','vstore m2, 0(x9)'),
                   ('vmul.bf16 m2, m2, m4','nop'), ('li x30, 1','li x30, 0'),
                   ('add x12, x12, x22','add x12, x12, x0')]
        for old,new in changes:
            with self.subTest(control=old), self.assertRaises((AssertionError, KeyError)):
                prove(source.replace(old,new),shape,-18)
    def test_admission(self):
        for shape in [(32.,2048,1024), (32,2048,1024.), (32,2048,1056)]:
            with self.subTest(shape=shape), self.assertRaises(ValueError):
                producer.allocation(*shape)
        for args in [(128,2048,2048,4456704,-14),(128,5632,1024,0,-18),
                     (32,2048,2048,4260096,False),(32,2048,2048,4260096,-127)]:
            with self.assertRaises(ValueError):producer.assembly(*args)

if __name__ == '__main__':unittest.main()
