"""Independent scalar-flow/operand provenance proof, without numerical timing."""

from collections import Counter
import hashlib
from pathlib import Path
import re
import sys
import unittest

sys.path.insert(0, str(Path(__file__).absolute().parents[1] / 'tools'))
import atlas_program_reuse_resnet as producer


def prove(source, m, k, n, exponent):
    lines = source.splitlines()
    labels = {s[:-1]: i for i, s in enumerate(lines) if s.endswith(':')}
    regs, memory, matrix, accum = {'x0': 0}, {}, {}, {}
    loads, stores, counts = [], {}, Counter()
    weight, pc = None, 0
    dma_pending = False
    scaled = set()

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
        if op == 'dma.wait':
            assert args == ['ch0']
            dma_pending = False
        elif op in ('nop', 'delay'):
            pass
        elif op == 'dma.config':
            assert regs[args[0]] == 1 and args[1] == 'ch0'
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
            assert not dma_pending
            dma_pending = True
            dst, src, size = (regs[a] for a in args[:3])
            assert size == 1024 and src % 1024 == 0 and 0 <= src < m*k+k*n
            memory[tile(dst * 4)] = ('input', src, False)
            loads.append((dst * 4, src, size))
        elif op == 'vload':
            assert not dma_pending
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
            kind,row,col,half=matrix['m2']
            assert kind == 'output' and (row,col) not in scaled
            scaled.add((row,col))
        elif op == 'dma.store':
            assert not dma_pending
            dma_pending = True
            dst, src, size = (regs[a] for a in args[:3])
            first, second = memory[tile(src*4)], memory[tile(src*4+1024)]
            assert first[0] == 'output' and first[3] == 0 and second == (*first[:3], 1)
            _, row, col, _ = first
            assert dst == m*k+k*n+256 + (row*(n//32)+col)*2048 and size == 2048
            assert (row,col) in scaled
            scaled.remove((row,col))
            assert (row, col) not in stores
            stores[row, col] = dst
        elif op == 'ecall':
            assert not dma_pending and not scaled
            assert not accum and len(stores) == m*n//1024
            expected_loads = list(range(0, m*k, 1024)) + list(range(m*k, m*k+k*n, 1024)) * len(producer.layout(m,k,n)['blocks'])
            assert sorted(src for _, src, _ in loads) == sorted(expected_loads)
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


class ResNetReuseTests(unittest.TestCase):
    def test_original_three_programs_are_unchanged(self):
        # Zero-rescale assembly identities captured before broadening admission.
        originals = {
            (64,4608,512): '3e0febeda4234345f8892786f4ce15cde2ac82ed48f1aacfce27fddb361bf808',
            (224,2304,256): '0274a95acb093e8dba22109b6b6cc1d66cdf30215dec708a4184f2348c8a5a9d',
            (800,1152,128): '1eb4e08d6504bd3c43671e4e9424fd215420cfb30a85af4605f491cf298dafa5',
        }
        for (m,k,n), expected in originals.items():
            with self.subTest(shape=(m,k,n)):
                source = producer.assembly(m,k,n,m*k+k*n+256,0)
                self.assertEqual(hashlib.sha256(source.encode()).hexdigest(), expected)

    def test_complete_operand_origin_order_and_outputs(self):
        for m,k,n in sorted(producer.SHAPES):
            with self.subTest(shape=(m,k,n)):
                result=prove(producer.assembly(m,k,n,m*k+k*n+256,-13),m,k,n,-13)
                self.assertEqual(result['load_bytes'],producer.dma_load_bytes(m,k,n))
                self.assertEqual(result['store_bytes'],2*m*n)

    def test_banks_and_live_regions(self):
        for shape in producer.SHAPES:
            plan=producer.layout(*shape)
            for block in plan['blocks']:
                regions=[plan['b_column_range'],plan['output_scratch_range'],block['a_bank0_range'],block['a_bank1_range']]
                occupied=set()
                for start,end in regions:
                    self.assertLessEqual(end,524288)
                    if end>start:
                        self.assertEqual(start//262144,(end-1)//262144)
                    tiles=set(range(start,end,1024))
                    self.assertFalse(occupied & tiles)
                    occupied |= tiles

    def test_negative_controls(self):
        m,k,n=224,2304,256
        s=producer.assembly(m,k,n,m*k+k*n+256,-13)
        for old,new in [('vtrpose.xlu m1, m1','nop'),('vmatmul.acc.mxu0','vmatmul.mxu0'),('acc1','acc0'),('vstore m3, 0(x9)','vstore m2, 0(x9)'),('add x22, x22, x16','add x22, x22, x8'),('addi x19, x19, 256','addi x19, x19, 512'),('dma.wait ch0','nop'),('vmul.bf16 m2, m2, m4','nop')]:
            with self.subTest(mutation=old), self.assertRaises((AssertionError,KeyError)):
                prove(s.replace(old,new),m,k,n,-13)

    def test_one_static_patch_word_source(self):
        for m,k,n in producer.SHAPES:
            source=producer.assembly(m,k,n,m*k+k*n+256,0).splitlines()
            sites=[i for i,line in enumerate(source) if line.startswith('vli.all')]
            self.assertEqual(len(sites),1)
            for exponent in (-126,-13,127):
                expected=source.copy()
                expected[sites[0]]=f'vli.all m4, {(exponent+127)<<7}'
                self.assertEqual(expected,producer.assembly(m,k,n,m*k+k*n+256,exponent).splitlines())

    def test_strict_admission(self):
        for shape in [(224.,2304,256),(225,2304,256),(32,32,32),(32,2048,1024)]:
            with self.assertRaises(ValueError): producer.layout(*shape)
        for offset,exp in [(0,-13),(1106177,-13),(1106176,None),(1106176,-127),(1106176,True)]:
            with self.assertRaises(ValueError): producer.assembly(224,2304,256,offset,exp)

if __name__=='__main__': unittest.main()
