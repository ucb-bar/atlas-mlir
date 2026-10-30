"""Shared symbolic operand, DMA ownership and output proof; no timing claim."""
from collections import Counter
import re

import atlas_program as program


def prove(source, m, k, n, exponent, load_budget=None, *, legacy=None):
    """Execute scalar flow and track complete 1-KiB input/output tile identities.

    B traffic comes from the public accounting API, independently anchored by
    the receipt-derived table in test_program_traffic_reuse. Direct helper tests
    supply their compulsory-traffic budget explicitly. Reuse loads A once;
    legacy modes check their resident or streamed input multiplicities.
    DMA regions remain inaccessible until their own channel is waited. Output
    tokens retain both BF16 halves and whether rescaling preceded each store.
    """
    lines = source.splitlines()
    labels = {line[:-1]: i for i, line in enumerate(lines) if line.endswith(':')}
    assert len(labels) == sum(line.endswith(':') for line in lines)
    assert sum(line.startswith('vli.all ') for line in lines) == (exponent is not None)
    if legacy is not None:
        assert legacy in ('none', 'A', 'B', 'BT')
        a_repeats = 1 if legacy == 'A' else n // 32
        b_repeats = 1 if legacy in ('B', 'BT') else m // 32
        load_budget = m * k * a_repeats + k * n * b_repeats
    elif load_budget is None:
        load_budget = program.dma_load_bytes(m, k, n, traffic_reuse=True)
    if legacy is None:
        a_repeats = 1
        b_repeats, remainder = divmod(load_budget - m * k, k * n)
        assert b_repeats >= 1 and remainder == 0
    regs, memory, matrix, accum = {'x0': 0}, {}, {}, {}
    pending, configured = {}, set()
    loads, stores, popped, counts = [], {}, set(), Counter()
    pc, weight = 0, None
    core, arithmetic, products, b_stores, output_stores = [], [], [], [], []
    instructions, delays, in_compute = 0, 0, False

    def accessible(base, size=1024):
        assert base % 1024 == 0 and 0 <= base < base + size <= 524288
        assert base // 262144 == (base + size - 1) // 262144
        assert all(base + size <= start or end <= base for start, end, _ in pending.values()), 'pending DMA hazard'
        return base

    def address(arg):
        return accessible(4 * regs[re.fullmatch(r'0\((x\d+)\)', arg).group(1)])

    for _ in range(3000000):
        line = lines[pc]
        op, *args = line.replace(',', '').split()
        pc += 1
        if op.endswith(':'):
            in_compute |= op == 'rows_loop:'
            continue
        counts[op] += 1
        instructions += 1
        if legacy is not None and op.startswith('v'):
            normalized = re.sub(r'vload m([01]), 0\(x\d+\)', r'vload m\1, input', line)
            arithmetic.append(normalized)
            if in_compute and op != 'vtrpose.xlu':
                core.append(line)
        if op == 'dma.wait':
            assert args[0] in configured and args[0] in pending
            _, _, writes = pending.pop(args[0])
            memory.update(writes)
        elif op == 'delay':
            delays += int(args[0])
        elif op == 'nop':
            pass
        elif op == 'dma.config':
            assert regs[args[0]] == 1 and args[1] in ('ch0', 'ch1')
            assert args[1] not in pending
            configured.add(args[1])
        elif op == 'li':
            regs[args[0]] = int(args[1], 0)
            assert args[0] != 'x0' and 0 <= regs[args[0]] < 2**32
        elif op in ('add', 'addi'):
            regs[args[0]] = regs[args[1]] + (regs[args[2]] if op == 'add' else int(args[2]))
            assert args[0] != 'x0' and 0 <= regs[args[0]] < 2**32
        elif op in ('beq', 'jal'):
            assert lines[pc] == 'nop', 'inert branch delay slot'
            if op == 'jal':
                assert args[0] == 'x0'
            if op == 'jal' or regs[args[0]] == regs[args[1]]:
                pc = labels[args[-1]]
        elif op == 'dma.load':
            dst, src, size = (regs[a] for a in args[:3])
            dst *= 4
            ch = args[3]
            assert ch in configured and ch not in pending and size in (1024, 4096)
            assert src % 1024 == 0 and 0 <= src < src + size <= m * k + k * n
            accessible(dst, size)
            writes = {dst + delta: ('input', src + delta, False) for delta in range(0, size, 1024)}
            pending[ch] = (dst, dst + size, writes)
            loads.append((dst, src, size))
            if legacy is not None:
                assert size == 1024
                resident = legacy == 'A' if src < m * k else legacy in ('B', 'BT')
                expected = 262144 + (src if src < m * k else src - m * k)
                assert dst == (expected if resident else (8192 if src < m * k else 9216))
                if resident:
                    assert not in_compute
        elif op == 'vload':
            matrix[args[0]] = memory[address(args[1])]
        elif op == 'vstore':
            dst = address(args[1])
            memory[dst] = matrix[args[0]]
            if legacy is not None and args[0] == 'm1':
                assert legacy == 'BT' and not in_compute
                kind, origin, transposed = matrix['m1']
                assert kind == 'input' and transposed and dst == 262144 + origin - m * k
                b_stores.append((dst, origin))
        elif op == 'vtrpose.xlu':
            kind, origin, transposed = matrix[args[1]]
            assert kind == 'input' and not transposed
            matrix[args[0]] = (kind, origin, True)
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
            col, binner = divmod((b - m * k) // 1024, k // 32)
            assert 0 <= a < m * k <= b < m * k + k * n and inner == binner
            assert 0 <= row < m // 32 and 0 <= col < n // 32
            if legacy is not None:
                output, expected_inner = divmod(len(products), k // 32)
                assert (row, col, inner) == (*divmod(output, n // 32), expected_inner)
                products.append((a, b, op == 'vmatmul.acc.mxu0'))
            if op == 'vmatmul.mxu0':
                assert inner == 0 and acc not in accum, 'first K overwrites a free slot'
                accum[acc] = (row, col, 1)
            else:
                assert accum[acc] == (row, col, inner), 'exact increasing K and output ownership'
                accum[acc] = (row, col, inner + 1)
        elif op == 'vmatpop.bf16.acc.mxu0':
            row, col, inner = accum.pop(args[1])
            assert inner == k // 32 and args[0] == 'm2' and (row, col) not in popped
            popped.add((row, col))
            for half in range(2):
                matrix[f'm{2 + half}'] = ('output', row, col, half, None)
        elif op == 'vli.all':
            assert exponent is not None and args == ['m4', str((exponent + 127) << 7)]
            matrix['m4'] = exponent
        elif op == 'vmul.bf16':
            assert exponent is not None and args == ['m2', 'm2', 'm4'] and matrix['m4'] == exponent
            kind, row, col, half, scale = matrix['m2']
            assert kind == 'output' and half == 0 and scale is None
            assert matrix['m3'] == ('output', row, col, 1, None)
            for half in range(2):
                matrix[f'm{2 + half}'] = ('output', row, col, half, exponent)
        elif op == 'dma.store':
            dst, src, size = (regs[a] for a in args[:3])
            src *= 4
            ch = args[3]
            assert ch in configured and ch not in pending and size == 2048
            accessible(src, size)
            first, second = memory[src], memory[src + 1024]
            kind, row, col, half, scale = first
            assert kind == 'output' and half == 0 and scale == exponent
            assert second == ('output', row, col, 1, exponent)
            assert dst == m * k + k * n + 256 + (row * (n // 32) + col) * 2048
            assert (row, col) not in stores
            if legacy is not None:
                assert src == 10240
            output_stores.append((dst, src // 4, size))
            stores[row, col] = dst
            pending[ch] = (src, src + size, {})
        elif op == 'ecall':
            outputs = {(r, c) for r in range(m // 32) for c in range(n // 32)}
            assert not pending and not accum and set(stores) == popped == outputs
            origins = [src + d for _, src, size in loads for d in range(0, size, 1024)]
            expected = (list(range(0, m * k, 1024)) * a_repeats
                        + list(range(m * k, m * k + k * n, 1024)) * b_repeats)
            assert Counter(origins) == Counter(expected)
            assert sum(size for _, _, size in loads) == load_budget
            assert counts['vmatmul.mxu0'] == len(outputs)
            assert counts['vmatmul.acc.mxu0'] == len(outputs) * (k // 32 - 1)
            assert counts['vmatpop.bf16.acc.mxu0'] == len(outputs)
            assert counts['vmul.bf16'] == (len(outputs) if exponent is not None else 0)
            if legacy in ('A', 'B', 'BT'):
                start, end = (0, m * k) if legacy == 'A' else (m * k, m * k + k * n)
                resident_loads = [item for item in loads if start <= item[1] < end]
                assert resident_loads == [(262144 + i - start, i, 1024) for i in range(start, end, 1024)]
                expected_stores = [(262144 + i, m * k + i) for i in range(0, k * n, 1024)]
                assert b_stores == (expected_stores if legacy == 'BT' else [])
            return dict(load_bytes=load_budget, store_bytes=len(stores) * 2048,
                        counts=dict(counts), loads=loads, stores=output_stores,
                        products=products, core=core, arithmetic=arithmetic,
                        instructions=instructions, delays=delays, b_stores=b_stores)
        else:
            raise AssertionError(op)
    raise AssertionError('nontermination')
