"""Independent scalar-flow/operand provenance proof, without numerical timing."""

from collections import Counter
from pathlib import Path
import re
import sys
import unittest

sys.path.insert(0, str(Path(__file__).absolute().parents[1] / 'tools'))
import atlas_program_reuse_stream as producer


def prove(source, m, k, n, exponent):
    lines = source.splitlines()
    labels = {s[:-1]: i for i, s in enumerate(lines) if s.endswith(':')}
    regs, memory, matrix, accum = {'x0': 0}, {}, {}, {}
    loads, stores, counts = [], {}, Counter()
    weight, pc = None, 0

    def tile(address):
        assert address % 1024 == 0 and 0 <= address <= 524288 - 1024
        assert address // 262144 == (address + 1023) // 262144
        return address

    def address(arg):
        return tile(4 * regs[re.fullmatch(r'0\((x\d+)\)', arg).group(1)])

    for _ in range(1000000):
        op, *args = lines[pc].replace(',', '').split()
        pc += 1
        if op.endswith(':'):
            continue
        counts[op] += 1
        if op in ('nop', 'delay', 'dma.wait'):
            pass
        elif op == 'dma.config':
            assert regs[args[0]] == 1 and args[1] in ('ch0', 'ch1')
        elif op == 'li':
            regs[args[0]] = int(args[1], 0)
        elif op in ('add', 'addi'):
            regs[args[0]] = regs[args[1]] + (regs[args[2]] if op == 'add' else int(args[2]))
            assert 0 <= regs[args[0]] < 2**32
        elif op in ('beq', 'jal'):
            assert lines[pc] == 'nop'
            if op == 'jal' or regs[args[0]] == regs[args[1]]:
                pc = labels[args[-1]]
        elif op == 'dma.load':
            dst, src, size = (regs[a] for a in args[:3])
            assert size == 1024 and src % 1024 == 0 and 0 <= src < m*k+k*n
            memory[tile(dst * 4)] = ('input', src, False)
            loads.append((dst * 4, src, size))
        elif op == 'vload':
            matrix[args[0]] = memory[address(args[1])]
        elif op == 'vstore':
            memory[address(args[1])] = matrix[args[0]]
        elif op == 'vtrpose.xlu':
            kind, origin, transposed = matrix[args[1]]
            assert kind == 'input'
            matrix[args[0]] = (kind, origin, not transposed)
        elif op == 'vmatpush.weight.mxu0':
            assert args[0] == 'w0'
            weight = matrix[args[1]]
        elif op in ('vmatmul.mxu0', 'vmatmul.acc.mxu0'):
            acc = args[0]
            assert acc in ('acc0', 'acc1') and args[2] == 'w0'
            kind, a, at = matrix[args[1]]
            wkind, b, bt = weight
            assert kind == wkind == 'input' and not at and bt
            row, inner = divmod(a // 1024, k // 32)
            col, binner = divmod((b - m*k) // 1024, k // 32)
            assert 0 <= a < m*k <= b < m*k+k*n and inner == binner
            if op == 'vmatmul.mxu0':
                assert inner == 0
                accum[acc] = (row, col, 1)
            else:
                assert accum[acc] == (row, col, inner)
                accum[acc] = (row, col, inner + 1)
        elif op == 'vmatpop.bf16.acc.mxu0':
            row, col, inner = accum.pop(args[1])
            assert inner == k // 32 and args[0] == 'm2'
            matrix['m2'], matrix['m3'] = ('output', row, col, 0), ('output', row, col, 1)
        elif op == 'vli.all':
            assert args == ['m4', str((exponent + 127) << 7)]
            matrix['m4'] = exponent
        elif op == 'vmul.bf16':
            assert args == ['m2', 'm2', 'm4'] and matrix['m4'] == exponent
        elif op == 'dma.store':
            dst, src, size = (regs[a] for a in args[:3])
            first, second = memory[tile(src*4)], memory[tile(src*4+1024)]
            assert first[0] == 'output' and first[3] == 0 and second == (*first[:3], 1)
            _, row, col, _ = first
            assert dst == m*k+k*n+256 + (row*(n//32)+col)*2048 and size == 2048
            assert (row, col) not in stores
            stores[row, col] = dst
        elif op == 'ecall':
            assert not accum and len(stores) == m*n//1024
            assert sorted(src for _, src, _ in loads) == list(range(0, m*k+k*n, 1024))
            assert counts['vmatmul.mxu0'] == m*n//1024
            assert counts['vmatmul.acc.mxu0'] == m*n//1024*(k//32-1)
            assert counts['vmatpop.bf16.acc.mxu0'] == m*n//1024
            assert counts['vli.all'] == (1 if exponent is not None else 0)
            assert counts['vmul.bf16'] == (m*n//1024 if exponent is not None else 0)
            return dict(load_bytes=sum(x[2] for x in loads), store_bytes=len(stores)*2048,
                        counts=dict(counts), loads=loads)
        else:
            raise AssertionError(op)
    raise AssertionError('nontermination')


class StreamReuseTests(unittest.TestCase):
    def test_order_addresses_capacity_and_compulsory_traffic(self):
        for shape in ((128,3072,768), (32,32,32), (64,64,96), (96,3072,64), (128,2048,32)):
            for exponent in (None, -15):
                m,k,n = shape
                with self.subTest(shape=shape, exponent=exponent):
                    result = prove(producer.assembly(m,k,n,m*k+k*n+256,exponent),m,k,n,exponent)
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
                prove(source.replace(old,new),m,k,n,-15)


if __name__ == '__main__':
    unittest.main()
