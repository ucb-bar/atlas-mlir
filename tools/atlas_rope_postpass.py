"""Serialized rotate-half RoPE over BF16 q/k retained by a producer launch.

One BF16 32x32 cosine, sine, and negated-sine tile per physical row tile is
shared by all 64-channel heads. Each output is RNE(RNE(x*cos) + RNE(r*sin)),
the host rotary formula: the low half uses r = -x[d+32] as x[d+32] * -sin,
and the add is the exact RNE sequence. The producer stores no unrotated DRAM
output; this pass writes rotated BF16 tiles to its original output extent.
"""
from dataclasses import dataclass

from atlas_epilogue import Epilogue
from atlas_program_checks import check_matrix_registers
from atlas_resident_output import retain_output, RESIDENT_H_WORD
from atlas_rmsnorm_postpass import _load_pair
from atlas_rne_add import CONSTANTS, rne_add

EXTRA_WORD = 1048576 // 4
STAGE_WORD = 1310720 // 4


@dataclass(frozen=True)
class ResidentRopeEpilogue(Epilogue):
    scratch_registers: tuple[int, int] = (10, 11)
    cos_offset: int = 0
    sin_offset: int = 0
    neg_sin_offset: int = 0
    head_dimension: int = 64

    def __post_init__(self):
        super().__post_init__()
        if self.head_dimension != 64 or self.n % 64 or 2 * self.m * self.n > 524288:
            raise ValueError('resident RoPE requires head64 and h fitting banks 2 and 3')
        if self.output_dtype != 'bf16' or self.output_layout != 'tiles':
            raise ValueError('resident RoPE output must be BF16 tiles')
        if set(self.scratch_registers) & {13, 16, 17, 18, 20, 21, 30, 31}:
            raise ValueError('RoPE requires an explicit recipe-free post-pass scratch pair')
        if 192 * self.m > 262144:
            raise ValueError('RoPE coefficients must fit bank 4')
        regions = self.rope_regions
        for i, region in enumerate(regions):
            offset, size = region['offset'], region['size_bytes']
            if type(offset) is not int or offset < 0 or offset % 32 or offset + size >= 2**32:
                raise ValueError('RoPE tables require aligned low32 input regions')
            for other in (*regions[:i], dict(offset=self.output_offset, size_bytes=self.output_bytes),
                          *Epilogue.input_regions.fget(self)):
                if offset < other['offset'] + other['size_bytes'] and other['offset'] < offset + size:
                    raise ValueError('RoPE table overlaps another input or output')

    @property
    def rope_regions(self):
        return tuple(dict(offset=offset, size_bytes=64 * self.m, layout='row_tiles_head64_half', dtype='bf16')
                     for offset in (self.cos_offset, self.sin_offset, self.neg_sin_offset))

    @property
    def dma_load_bytes(self):
        return super().dma_load_bytes + 192 * self.m

    @property
    def input_regions(self):
        return super().input_regions + self.rope_regions

    def store(self, output_reg):
        return retain_output(self, output_reg)


def postpass(e):
    if not isinstance(e, ResidentRopeEpilogue):
        raise TypeError('postpass requires ResidentRopeEpilogue')
    lines = []
    for index, region in enumerate(e.rope_regions):
        for offset in range(0, region['size_bytes'], 4096):
            size = min(4096, region['size_bytes'] - offset)
            lines += [f"li x15, {EXTRA_WORD + (index * 64 * e.m + offset) // 4}",
                      f"li x19, {region['offset'] + offset}", f'li x22, {size}',
                      'dma.load x15, x19, x22, ch0', 'dma.wait ch0']
    # lo/hi halves in m54/m56; cos, sin, and -sin in m58/m60/m62; products in m0/m2; result m52.
    def half(stage):
        return [*rne_add(0, 2, 52, constants=False), f'li x3, {stage}', 'vstore m52, 0(x3)', 'delay 34',
                'addi x3, x3, 256', 'vstore m53, 0(x3)', 'delay 34']
    lines += [line for register, bits in CONSTANTS for line in (f'vli.all m{register}, {bits}', 'delay 66')]
    lines += [f'li x16, {RESIDENT_H_WORD}', f'li x20, {e.output_offset}', f'li x17, {EXTRA_WORD}',
              f'li x21, {EXTRA_WORD + 16 * e.m}', f'li x15, {EXTRA_WORD + 32 * e.m}', f'li x13, {e.m // 32}',
              'rope_row:', *_load_pair(58, 17), *_load_pair(60, 21), *_load_pair(62, 15),
              f'li x18, {e.n // 64}', 'rope_head:', *_load_pair(54, 16),
              'addi x19, x16, 512', *_load_pair(56, 19, scratch=22),
              'vmul.bf16 m0, m54, m58', 'delay 66', 'vmul.bf16 m2, m56, m62', 'delay 66', *half(STAGE_WORD),
              'vmul.bf16 m0, m56, m58', 'delay 66', 'vmul.bf16 m2, m54, m60', 'delay 66', *half(STAGE_WORD + 512),
              f'li x3, {STAGE_WORD}', 'li x22, 4096', 'dma.store x20, x3, x22, ch0', 'dma.wait ch0',
              'addi x16, x16, 1024', 'add x20, x20, x22', 'addi x18, x18, -1',
              'bne x18, x0, rope_head', 'nop', 'addi x17, x17, 512', 'addi x21, x21, 512', 'addi x15, x15, 512',
              'addi x13, x13, -1', 'bne x13, x0, rope_row', 'nop']
    check_matrix_registers('\n'.join(lines))
    return lines


def append_postpass(source, epilogue):
    check_matrix_registers(source)
    lines = source.rstrip().splitlines()
    if not lines or lines[-1] != 'ecall' or lines.count('ecall') != 1:
        raise ValueError('expected one final ECALL')
    result = '\n'.join(lines[:-1] + postpass(epilogue) + ['ecall']) + '\n'
    check_matrix_registers(result)
    return result
