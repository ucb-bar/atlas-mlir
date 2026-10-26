"""Compact, bank-bounded A-block/B-column reuse for captured Smol shapes.

Packed DRAM inputs and guarded output offsets are unchanged. Each row block
loads A once into bank 1 followed by bank 0; each B column is transposed once
per block. Two MXU0 slots share weight pushes and retain increasing-K order.
The first K operation overwrites, all later operations accumulate, and each
output has one BF16 pop/rescale. Delays match the established generator.
"""

BANK_BYTES = 262144
SHAPES = {
    (32, 32, 960),
    (64, 32, 736),
    (64, 320, 320),
    (64, 736, 32),
    (64, 736, 320),
    (64, 736, 736),
    (64, 736, 960),
    (64, 736, 2048),
    (64, 960, 320),
    (64, 960, 736),
    (64, 960, 960),
    (64, 960, 2560),
    (64, 1440, 736),
    (64, 2048, 736),
    (64, 2560, 960),
    (128, 320, 320),
    (128, 768, 3072),
    (128, 960, 320),
    (128, 960, 960),
    (128, 960, 2560),
    (128, 2560, 960),
}


def layout(m, k, n):
    if any(type(v) is not int for v in (m, k, n)) or (m, k, n) not in SHAPES:
        raise ValueError("Smol reuse requires one of the captured physical shapes")
    b_bytes = k * 32
    a_bank0 = b_bytes + 2048
    if max(0, m * k - BANK_BYTES) + a_bank0 > BANK_BYTES:
        raise ValueError("resident A, B column and output must fit two banks")
    rows = m // 32
    blocks = []
    for first in range(0, m // 32, rows):
        count = min(rows, m // 32 - first)
        size = count * k * 32
        blocks.append(dict(first_row_tile=first, row_tiles=count,
                           a_dram_range=[first * k * 32, (first + count) * k * 32],
                           a_bank1_range=[BANK_BYTES, BANK_BYTES + min(size, BANK_BYTES)],
                           a_bank0_range=[a_bank0, a_bank0 + max(0, size - BANK_BYTES)]))
    return dict(b_column_range=[0, b_bytes], output_scratch_range=[b_bytes, a_bank0],
                a_bank0_base=a_bank0, blocks=blocks, bank_bytes=BANK_BYTES,
                dma_load_bytes=m * k + len(blocks) * k * n,
                dma_store_bytes=2 * m * n)


def dma_load_bytes(m, k, n):
    return layout(m, k, n)['dma_load_bytes']


def a_word(offset, bank0_base):
    return (BANK_BYTES + offset if offset < BANK_BYTES
            else bank0_base + offset - BANK_BYTES) // 4


def assembly(m, k, n, output_offset, rescale_exponent=0):
    plan = layout(m, k, n)
    if type(output_offset) is not int or output_offset != m * k + k * n + 256:
        raise ValueError("output must preserve the captured input/guard layout")
    if type(rescale_exponent) is not int or not -126 <= rescale_exponent <= 127:
        raise ValueError("one normal finite BF16 rescale instruction is required")
    if output_offset + 2 * m * n + 256 >= 2**32:
        raise ValueError("guarded image must fit the original DRAM low word")
    out = plan['output_scratch_range'][0] // 4
    base0 = plan['a_bank0_base']
    lines = ['li x5, 1024', 'li x8, 2048', f'li x3, {out}', f'li x9, {out + 256}',
             f'li x16, {n * 64}', 'li x30, 1', 'dma.config x30, ch0',
             f'vli.all m4, {(rescale_exponent + 127) << 7}', 'delay 66']
    for bi, block in enumerate(plan['blocks']):
        tag = f'block_{bi}'
        size = block['row_tiles'] * k * 32
        for pi, (offset, count) in enumerate(((0, min(size, BANK_BYTES)),
                                            (BANK_BYTES, max(0, size - BANK_BYTES)))):
            if not count:
                continue
            label = f'{tag}_a_{pi}'
            lines += [f'li x19, {a_word(offset, base0)}',
                      f"li x20, {block['a_dram_range'][0] + offset}",
                      f'li x21, {count // 1024}', label + ':',
                      'dma.load x19, x20, x5, ch0', 'dma.wait ch0',
                      'addi x19, x19, 256', 'addi x20, x20, 1024',
                      'addi x21, x21, -1', f'beq x21, x0, {label}_done', 'nop',
                      f'jal x0, {label}', 'nop', label + '_done:']
        lines += [f'li x15, {m * k}', f'li x17, {n // 32}',
                  f"li x12, {output_offset + block['first_row_tile'] * n * 64}",
                  tag + '_columns:', 'li x19, 0', 'add x20, x15, x0',
                  f'li x21, {k // 32}', tag + '_b:',
                  'dma.load x19, x20, x5, ch0', 'dma.wait ch0',
                  'vload m1, 0(x19)', 'delay 34', 'vtrpose.xlu m1, m1', 'delay 66',
                  'vstore m1, 0(x19)', 'delay 34', 'addi x19, x19, 256',
                  'addi x20, x20, 1024', 'addi x21, x21, -1',
                  f'beq x21, x0, {tag}_b_done', 'nop', f'jal x0, {tag}_b', 'nop',
                  tag + '_b_done:', 'add x15, x20, x0', 'add x22, x12, x0']
        for row in range(0, block['row_tiles'], 2):
            rows = min(2, block['row_tiles'] - row)
            # Split K loops exactly where either A pointer crosses banks. No
            # vector/DMA tile crosses a bank, and no pointer-wrap speculation.
            cuts = {0, 1, k // 32}
            for acc in range(rows):
                crossing = (BANK_BYTES - (row + acc) * k * 32) // 1024
                if 0 < crossing < k // 32:
                    cuts.add(crossing)
            cuts = sorted(cuts)
            for si, (first, end) in enumerate(zip(cuts, cuts[1:])):
                label = f'{tag}_row_{row}_k_{si}'
                lines += [f'li x6, {first * 256}', f'li x21, {end - first}']
                for acc in range(rows):
                    addr = a_word((row + acc) * k * 32 + first * 1024, base0)
                    lines += [f'li x{4 if acc == 0 else 7}, {addr}']
                lines += [label + ':', 'vload m1, 0(x6)', 'delay 34',
                          'vmatpush.weight.mxu0 w0, m1', 'delay 32']
                op = 'vmatmul.mxu0' if first == 0 else 'vmatmul.acc.mxu0'
                for acc in range(rows):
                    reg = 4 if acc == 0 else 7
                    lines += [f'vload m0, 0(x{reg})', 'delay 34',
                              f'{op} acc{acc}, m0, w0', 'delay 96', f'addi x{reg}, x{reg}, 256']
                lines += ['addi x6, x6, 256']
                if end - first > 1:
                    lines += ['addi x21, x21, -1', f'beq x21, x0, {label}_done', 'nop',
                              f'jal x0, {label}', 'nop', label + '_done:']
            for acc in range(rows):
                lines += [f'vmatpop.bf16.acc.mxu0 m2, acc{acc}', 'delay 34',
                          'vmul.bf16 m2, m2, m4', 'delay 66', 'vstore m2, 0(x3)', 'delay 34',
                          'vstore m3, 0(x9)', 'delay 34', 'dma.store x22, x3, x8, ch0',
                          'dma.wait ch0', 'add x22, x22, x16']
        lines += ['add x12, x12, x8', 'addi x17, x17, -1',
                  f'beq x17, x0, {tag}_done', 'nop', f'jal x0, {tag}_columns', 'nop', tag + '_done:']
    return '\n'.join(lines + ['nop', 'ecall']) + '\n'
