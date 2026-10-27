"""Experimental captured connector: one A row panel resident, B tiles streamed.

This helper is not public dispatch admission. Qualification requires both
captured launches on the pinned model before changing the public allowlist.
"""
from atlas_program_reuse_common import store_output

BANK_BYTES = 262144
SHAPE = (64, 12288, 960)


def layout(m, k, n):
    if any(type(value) is not int for value in (m, k, n)) or (m, k, n) != SHAPE:
        raise ValueError("connector recipe requires captured physical shape 64/12288/960")
    panel = 32 * k
    spill = panel - BANK_BYTES
    return dict(a_bank1_range=[BANK_BYTES, 2 * BANK_BYTES], a_bank0_range=[0, spill],
                b_buffers=[[spill, spill + 1024], [spill + 1024, spill + 2048]],
                output_range=[spill + 2048, spill + 4096],
                dma_load_bytes=m * k + (m // 32) * k * n, dma_store_bytes=2 * m * n)


def dma_load_bytes(m, k, n):
    return layout(m, k, n)['dma_load_bytes']


def a_word(offset):
    return (BANK_BYTES + offset if offset < BANK_BYTES else offset - BANK_BYTES) // 4


def assembly(m, k, n, output_offset, rescale_exponent):
    plan = layout(m, k, n)
    if type(output_offset) is not int or output_offset != m * k + k * n + 256:
        raise ValueError("connector output must retain captured guarded placement")
    if type(rescale_exponent) is not int or not -126 <= rescale_exponent <= 127:
        raise ValueError("connector rescale must be one finite normal BF16 factor")
    b0, b1 = [span[0] // 4 for span in plan['b_buffers']]
    output = plan['output_range'][0] // 4
    lines = ['li x5, 1024', 'li x8, 2048', f'li x3, {output}', f'li x9, {output + 256}',
             'li x30, 1', 'dma.config x30, ch0', 'dma.config x30, ch1',
             f'vli.all m4, {(rescale_exponent + 127) << 7}', 'delay 66']
    for row in range(m // 32):
        tag = f'row_{row}'
        for part, (start, size) in enumerate(((0, BANK_BYTES), (BANK_BYTES, 32*k-BANK_BYTES))):
            label = f'{tag}_preload_{part}'
            lines += [f'li x19, {a_word(start)}', f'li x20, {row * 32 * k + start}',
                      f'li x21, {size // 1024}', label + ':',
                      'dma.load x19, x20, x5, ch0', 'dma.wait ch0',
                      'addi x19, x19, 256', 'addi x20, x20, 1024', 'addi x21, x21, -1',
                      f'beq x21, x0, {label}_done', 'nop', f'jal x0, {label}', 'nop', label + '_done:']
        lines += [f'li x15, {m * k}', f'li x17, {n // 32}',
                  f'li x22, {output_offset + row * n * 64}', tag + '_columns:',
                  f'li x19, {b0}', f'li x24, {b1}', 'add x20, x15, x0',
                  'dma.load x19, x20, x5, ch0', 'dma.wait ch0', 'addi x20, x20, 1024']
        # The cuts isolate overwrite, bank crossing and final non-prefetch K tile.
        cuts = (0, 1, BANK_BYTES // 1024, k // 32 - 1, k // 32)
        for segment, (first, end) in enumerate(zip(cuts, cuts[1:])):
            label = f'{tag}_k_{segment}'
            final = first == k // 32 - 1
            lines += [f'li x4, {a_word(first * 1024)}', f'li x21, {end - first}', label + ':']
            if not final:
                lines += ['dma.load x24, x20, x5, ch1', 'addi x20, x20, 1024']
            lines += ['vload m1, 0(x19)', 'delay 34', 'vtrpose.xlu m1, m1', 'delay 66',
                      'vmatpush.weight.mxu0 w0, m1', 'delay 32', 'vload m0, 0(x4)', 'delay 34',
                      ('vmatmul.mxu0' if first == 0 else 'vmatmul.acc.mxu0') + ' acc0, m0, w0',
                      'delay 96', 'addi x4, x4, 256']
            if not final:
                lines += ['dma.wait ch1', 'add x25, x19, x0', 'add x19, x24, x0', 'add x24, x25, x0']
            if end - first > 1:
                lines += ['addi x21, x21, -1', f'beq x21, x0, {label}_done', 'nop',
                          f'jal x0, {label}', 'nop', label + '_done:']
        lines += store_output(0, 22, rescale_exponent)
        lines += ['add x15, x20, x0', 'add x22, x22, x8', 'addi x17, x17, -1',
                  f'beq x17, x0, {tag}_done', 'nop', f'jal x0, {tag}_columns', 'nop', tag + '_done:']
    return '\n'.join(lines + ['nop', 'ecall']) + '\n'
