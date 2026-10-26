"""Narrow semantic gates: addresses, ordered K chains, slots and patch ABI."""
import io
import re
import unittest
from collections import Counter
from atlas_program_reuse_resident import admission, assembly, dma_load_bytes


def trace_contract(m, k, n):
    output = m*k + k*n + 256
    source = assembly(m, k, n, output, -19)
    instructions, labels = [], {}
    for line in source.splitlines():
        if line.endswith(':'):
            labels[line[:-1]] = len(instructions)
        else:
            instructions.append(re.split(r'[ ,]+', line))
    x = [0]*32
    vm, mrf, acc = {}, {}, {}
    weight = None
    loads, stores, commands = [], [], Counter()
    pc, pending, steps = 0, None, 0
    def reg(s): return int(s[1:])
    while True:
        words = instructions[pc]
        op, args = words[0], words[1:]
        if op == 'ecall': break
        steps += 1
        if steps > 1000000: raise AssertionError('nonterminating scalar loop')
        commands[op] += 1
        redirect = None
        if op == 'li': x[reg(args[0])] = int(args[1], 0)
        elif op == 'add': x[reg(args[0])] = x[reg(args[1])] + x[reg(args[2])]
        elif op == 'addi': x[reg(args[0])] = x[reg(args[1])] + int(args[2], 0)
        elif op == 'beq':
            if x[reg(args[0])] == x[reg(args[1])]: redirect = labels[args[2]]
        elif op == 'jal': redirect = labels[args[1]]
        elif op == 'dma.load':
            destination, address, size = (x[reg(a)] for a in args[:3])
            assert size == 1024
            loads.append((address, size))
            if address < m*k:
                tile = address // 1024
                value = ('A', tile // (k//32), tile % (k//32))
            else:
                tile = (address-m*k) // 1024
                value = ('B-raw', tile // (k//32), tile % (k//32))
            vm[4*destination] = value
        elif op == 'vload':
            address = 4*x[reg(args[1][2:-1])]
            mrf[args[0]] = vm[address]
        elif op == 'vtrpose.xlu':
            value = mrf[args[1]]
            assert value[0] == 'B-raw'
            mrf[args[0]] = ('B-transposed', *value[1:])
        elif op == 'vmatpush.weight.mxu0':
            assert args[0] == 'w0'
            weight = mrf[args[1]]
            assert weight[0] == 'B-transposed'
        elif op in ('vmatmul.mxu0', 'vmatmul.acc.mxu0'):
            a = mrf[args[1]]
            assert a[0] == 'A' and a[2] == weight[2]
            slot = int(args[0][3:])
            assert slot in (0, 1)
            if op == 'vmatmul.mxu0':
                assert a[2] == 0
                acc[slot] = [a[1], weight[1], [0]]
            else:
                state = acc[slot]
                assert (state[0], state[1], state[2][-1]+1) == (a[1], weight[1], a[2])
                state[2].append(a[2])
        elif op == 'vmatpop.bf16.acc.mxu0':
            state = acc[int(args[1][3:])]
            assert state[2] == list(range(k//32))
            dest = int(args[0][1:])
            assert dest == 2
            mrf[f'm{dest}'] = ('C', state[0], state[1], 0)
            mrf[f'm{dest+1}'] = ('C', state[0], state[1], 1)
        elif op == 'vli.all':
            assert args == ['m4', str((-19+127)<<7)]
        elif op == 'vmul.bf16': assert args == ['m2', 'm2', 'm4']
        elif op == 'vstore':
            address = 4*x[reg(args[1][2:-1])]
            vm[address] = mrf[args[0]]
        elif op == 'dma.store':
            address, source_word, size = (x[reg(a)] for a in args[:3])
            low, high = vm[source_word*4], vm[source_word*4+1024]
            assert low[0] == high[0] == 'C' and low[1:3] == high[1:3]
            assert low[3] == 0 and high[3] == 1 and size == 2048
            assert address == output + (low[1]*(n//32)+low[2])*2048
            stores.append((address, size))
        elif op not in ('delay', 'dma.wait', 'dma.config', 'nop'):
            raise AssertionError(op)
        x[0] = 0
        pc = pending if pending is not None else pc+1
        pending = redirect
    assert sorted(loads) == [(i, 1024) for i in range(0, m*k+k*n, 1024)]
    assert sorted(stores) == [(i, 2048) for i in range(output, output+2*m*n, 2048)]
    assert commands['vli.all'] == 1
    assert commands['vtrpose.xlu'] == k*n//1024
    assert commands['vmatpush.weight.mxu0'] == (m//64)*(k//32)*(n//32)
    assert commands['vmatmul.mxu0']+commands['vmatmul.acc.mxu0'] == (m//32)*(k//32)*(n//32)
    return commands


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
            with self.subTest(shape=shape): trace_contract(*shape)
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
