"""Resident LayerNorm/RMSNorm extensions with the original 2048 path intact.

Inputs with real_n < n must have zero h and gamma in padded columns. The
qualification helper validates this contract before producing any fixture.
Only m10/m11 and m12/m13 hold statistics; constants and operands use low pairs.
"""
from dataclasses import dataclass
import struct

from atlas_epilogue import Epilogue
from atlas_program_checks import check_matrix_registers
from atlas_rmsnorm_postpass import (ResidentRMSNormEpilogue, H_WORD, EXTRA_WORD, STAGE_WORD,
                                   MASK_BYTES, _load_pair, postpass as original_postpass)


@dataclass(frozen=True)
class ResidentNormEpilogue(ResidentRMSNormEpilogue):
    norm_kind: str = "rms"
    real_n: int | None = None
    norm_bias_offset: int | None = None

    def __post_init__(self):
        Epilogue.__post_init__(self)
        if self.m not in (32, 64, 128, 192) or 2 * self.m * self.n > 524288:
            raise ValueError("resident norm rows must fit banks 2 and 3")
        if (self.norm_kind, self.n, self.real_n) not in (("rms", 2048, 2048), ("rms", 960, 960),
                                                       ("rms", 736, 720), ("layer", 768, 768)):
            raise ValueError("unsupported norm kind, physical width, or real width")
        if self.output_dtype != "bf16" or self.output_layout != "tiles":
            raise ValueError("resident norm requires original BF16 tile output")
        if set(self.scratch_registers) & {30, 31}:
            raise ValueError("resident norm requires an explicit recipe-selected scratch pair")
        if set(self.scratch_registers) & {12, 13, 14, 16, 17, 18, 20, 21}:
            raise ValueError("norm scratch overlaps post-pass live scalar state")
        if type(self.norm_scale_exponent) is not int or not -126 <= self.norm_scale_exponent <= 119:
            raise ValueError("norm FP8 scale must have finite normal BF16 clipping limits")
        if (self.norm_kind == "layer") != (self.norm_bias_offset is not None):
            raise ValueError("LayerNorm requires beta; RMSNorm has no beta")
        regions = self.norm_regions
        for i, region in enumerate(regions):
            offset, size = region['offset'], region['size_bytes']
            if type(offset) is not int or offset < 0 or offset % 32 or offset + size >= 2**32:
                raise ValueError("norm extras require aligned low32 input regions")
            for other in (*regions[:i], dict(offset=self.output_offset, size_bytes=self.output_bytes),
                          *Epilogue.input_regions.fget(self)):
                if offset < other['offset'] + other['size_bytes'] and other['offset'] < offset + size:
                    raise ValueError("norm extras overlap another input or output")

    @property
    def norm_regions(self):
        regions = [dict(offset=self.norm_mask_offset, size_bytes=MASK_BYTES, layout="onehot_column_tiles", dtype="bf16"),
                   dict(offset=self.norm_weight_offset, size_bytes=64 * self.n, layout="channel", dtype="bf16")]
        if self.norm_bias_offset is not None:
            regions.append(dict(offset=self.norm_bias_offset, size_bytes=64 * self.n, layout="channel", dtype="bf16"))
        return tuple(regions)

    @property
    def dma_load_bytes(self):
        return Epilogue.dma_load_bytes.fget(self) + sum(region['size_bytes'] for region in self.norm_regions)

    @property
    def input_regions(self):
        return Epilogue.input_regions.fget(self) + self.norm_regions


def _bits(value):
    raw = struct.unpack('<I', struct.pack('<f', value))[0]
    return (raw + 0x7fff + ((raw >> 16) & 1)) >> 16


def _load_extras(e):
    lines = []
    target = EXTRA_WORD
    for index, region in enumerate(e.norm_regions):
        offset, size = region['offset'], region['size_bytes']
        count, tail = divmod(size, 4096)
        lines += [f"li x15, {target}", f"li x16, {offset}", "li x17, 4096", f"li x18, {count}",
                  f"norm_extra_{index}:", "dma.load x15, x16, x17, ch0", "dma.wait ch0",
                  "addi x15, x15, 1024", "li x19, 4096", "add x16, x16, x19", "addi x18, x18, -1",
                  f"bne x18, x0, norm_extra_{index}", "nop"]
        if tail:
            lines += [f"li x17, {tail}", "dma.load x15, x16, x17, ch0", "dma.wait ch0"]
        target += size // 4
    return lines


def _stats(e, accumulator, label, *, centered=False, square=True, count=None):
    count = e.n // 32 if count is None else count
    lines = [f"li x17, {EXTRA_WORD}", f"li x18, {count}", label + ':', *_load_pair(2, 16)]
    if centered:
        lines += ["vsub.bf16 m2, m2, m12", "delay 66"]
    if square:
        lines += ["vmul.bf16 m6, m2, m2", "delay 66"]
    lines += [f"vredsum.row.bf16 m6, m{6 if square else 2}", "delay 39", *_load_pair(8, 17),
              "vmul.bf16 m6, m6, m8", "delay 66", f"vadd.bf16 m{accumulator}, m{accumulator}, m6", "delay 66",
              "addi x16, x16, 512", "addi x17, x17, 512", "addi x18, x18, -1",
              f"bne x18, x0, {label}", "nop", f"vredsum.row.bf16 m{accumulator}, m{accumulator}", "delay 39"]
    return lines


def postpass(e):
    if not isinstance(e, ResidentNormEpilogue):
        raise TypeError("postpass requires ResidentNormEpilogue")
    # This path is deliberately byte-identical to the previously gated source.
    if (e.m, e.n, e.norm_kind, e.scratch_registers) == (128, 2048, 'rms', (10, 11)):
        return original_postpass(e)
    lines = _load_extras(e)
    norm_base = e.output_offset + 2 * e.m * e.n
    gamma_word = EXTRA_WORD + MASK_BYTES // 4
    lines += [f"li x14, {H_WORD}", f"li x20, {norm_base}", f"li x13, {e.m // 32}", "norm_row:",
              "vli.all m10, 0", "delay 66", "vli.all m12, 0", "delay 66", "add x16, x14, x0"]
    if e.norm_kind == 'layer':
        lines += _stats(e, 12, 'norm_mean', square=False)
        lines += [f"vli.all m6, {_bits(1 / e.real_n)}", "delay 66", "vmul.bf16 m12, m12, m6", "delay 66",
                  "add x16, x14, x0"]
        lines += _stats(e, 10, 'norm_variance', centered=True)
    elif e.n <= 1024:
        lines += _stats(e, 10, 'norm_sumsq')
    else:
        lines += _stats(e, 10, 'norm_sumsq0', count=32)
        lines += _stats(e, 12, 'norm_sumsq1', count=32)
        lines += [f"li x17, {EXTRA_WORD}", *_load_pair(8, 17), "vmul.bf16 m10, m10, m8", "delay 66",
                  "addi x17, x17, 512", *_load_pair(8, 17), "vmul.bf16 m12, m12, m8", "delay 66",
                  "vadd.bf16 m10, m10, m12", "delay 66", "vredsum.row.bf16 m10, m10", "delay 39"]
    lines += [f"vli.all m6, {_bits(1 / e.real_n)}", "delay 66", "vmul.bf16 m10, m10, m6", "delay 66",
              f"vli.all m6, {_bits(1e-6 if e.norm_kind == 'layer' else 1e-5)}", "delay 66",
              "vadd.bf16 m10, m10, m6", "delay 66", "vsqrt.bf16 m10, m10", "delay 66",
              "vrecip.bf16 m10, m10", "delay 66", "add x16, x14, x0", f"li x17, {gamma_word}",
              f"li x18, {e.n // 32}", f"li x3, {STAGE_WORD}", f"li x9, {STAGE_WORD + 256}"]
    if e.norm_kind == 'layer':
        lines += [f"li x21, {gamma_word + 16 * e.n}"]
    lines += ["norm_output:", *_load_pair(2, 16)]
    if e.norm_kind == 'layer':
        lines += ["vsub.bf16 m2, m2, m12", "delay 66"]
    lines += ["vmul.bf16 m2, m2, m10", "delay 66", *_load_pair(6, 17), "vmul.bf16 m2, m2, m6", "delay 66"]
    if e.norm_kind == 'layer':
        lines += [*_load_pair(6, 21), "vadd.bf16 m2, m2, m6", "delay 66", "addi x21, x21, 512"]
    bridge = Epilogue(e.m, e.n, norm_base, output_dtype='fp8', output_scale_exponent=e.norm_scale_exponent,
                      scratch_registers=e.scratch_registers)
    lines += bridge.store(20)
    lines += ["addi x16, x16, 512", "addi x17, x17, 512", "li x19, 2048", "add x20, x20, x19",
              "addi x18, x18, -1", "bne x18, x0, norm_output", "nop", f"li x19, {16 * e.n}",
              "add x14, x14, x19", "addi x13, x13, -1", "bne x13, x0, norm_row", "nop"]
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
