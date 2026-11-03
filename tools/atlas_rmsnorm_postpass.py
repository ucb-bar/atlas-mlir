"""Opt-in 128x2048 resident RMSNorm prototype, independent of core recipes.

The core retains post-residual BF16 h in banks 2/3 and stores its original
BF16 output. The serialized post-pass computes row statistics with two pairs,
then emits natural FP8 tiles immediately after h in the same output extent.
"""
from dataclasses import dataclass

from atlas_epilogue import Epilogue
from atlas_program_checks import check_matrix_registers
from atlas_resident_output import retain_output

H_WORD = 524288 // 4
EXTRA_WORD = 1048576 // 4
STAGE_WORD = 1310720 // 4
MASK_BYTES = 65536
GAMMA_BYTES = 131072


@dataclass(frozen=True)
class ResidentRMSNormEpilogue(Epilogue):
    norm_weight_offset: int = 0
    norm_mask_offset: int = 0
    norm_scale_exponent: int = -8

    def __post_init__(self):
        super().__post_init__()
        if (self.m, self.n) != (128, 2048) or self.output_dtype != "bf16" or self.output_layout != "tiles":
            raise ValueError("resident RMSNorm requires 128x2048 BF16 tiles")
        if self.scratch_registers != (10, 11):
            raise ValueError("resident RMSNorm requires Tiny scratch x10/x11")
        if type(self.norm_scale_exponent) is not int or not -126 <= self.norm_scale_exponent <= 119:
            raise ValueError("norm FP8 scale must have finite normal BF16 clipping limits")
        regions = [(self.norm_mask_offset, MASK_BYTES), (self.norm_weight_offset, GAMMA_BYTES)]
        for offset, size in regions:
            if type(offset) is not int or offset < 0 or offset % 32 or offset + size >= 2**32:
                raise ValueError("norm extras require aligned low32 input regions")
            if offset < self.output_offset + self.output_bytes and self.output_offset < offset + size:
                raise ValueError("norm extras overlap output")
        a, b = regions
        if a[0] < b[0] + b[1] and b[0] < a[0] + a[1]:
            raise ValueError("norm extras overlap each other")

    @property
    def output_bytes(self):
        return 3 * self.m * self.n

    @property
    def dma_store_bytes(self):
        return self.output_bytes

    @property
    def dma_load_bytes(self):
        return super().dma_load_bytes + MASK_BYTES + GAMMA_BYTES

    @property
    def input_regions(self):
        return super().input_regions + (
            dict(offset=self.norm_mask_offset, size_bytes=MASK_BYTES, layout="onehot_column_tiles", dtype="bf16"),
            dict(offset=self.norm_weight_offset, size_bytes=GAMMA_BYTES, layout="channel", dtype="bf16"))

    def store(self, output_reg):
        return retain_output(self, output_reg) + super().store(output_reg)


def _load_pair(register, pointer, scratch=19):
    return [f"vload m{register}, 0(x{pointer})", "delay 34", f"addi x{scratch}, x{pointer}, 256",
            f"vload m{register + 1}, 0(x{scratch})", "delay 34"]


def postpass(epilogue):
    """Assembly fragment; DMA ch0 already uses the launch arena high half."""
    e = epilogue
    if not isinstance(e, ResidentRMSNormEpilogue):
        raise TypeError("postpass requires ResidentRMSNormEpilogue")
    lines = []
    for name, offset, size, target in (("masks", e.norm_mask_offset, MASK_BYTES, EXTRA_WORD),
                                        ("gamma", e.norm_weight_offset, GAMMA_BYTES, EXTRA_WORD + MASK_BYTES // 4)):
        lines += [f"li x15, {target}", f"li x16, {offset}", "li x17, 4096", f"li x18, {size // 4096}",
                  f"norm_load_{name}:", "dma.load x15, x16, x17, ch0", "dma.wait ch0",
                  "addi x15, x15, 1024", "li x19, 4096", "add x16, x16, x19", "addi x18, x18, -1",
                  f"bne x18, x0, norm_load_{name}", "nop"]
    norm_base = e.output_offset + 2 * e.m * e.n
    lines += [f"li x14, {H_WORD}", f"li x20, {norm_base}", "li x13, 4", "norm_row:",
              "vli.all m10, 0", "delay 66", "vli.all m12, 0", "delay 66", "add x16, x14, x0"]
    for group, accumulator in enumerate((10, 12)):
        lines += [f"li x17, {EXTRA_WORD}", "li x18, 32", f"norm_stats_{group}:",
                  *_load_pair(2, 16), "vmul.bf16 m6, m2, m2", "delay 66",
                  "vredsum.row.bf16 m6, m6", "delay 39", *_load_pair(8, 17),
                  "vmul.bf16 m6, m6, m8", "delay 66", f"vadd.bf16 m{accumulator}, m{accumulator}, m6", "delay 66",
                  "addi x16, x16, 512", "addi x17, x17, 512", "addi x18, x18, -1",
                  f"bne x18, x0, norm_stats_{group}", "nop"]
    lines += ["vredsum.row.bf16 m10, m10", "delay 39", "vredsum.row.bf16 m12, m12", "delay 39",
              f"li x17, {EXTRA_WORD}", *_load_pair(8, 17), "vmul.bf16 m10, m10, m8", "delay 66",
              "addi x17, x17, 512", *_load_pair(8, 17), "vmul.bf16 m12, m12, m8", "delay 66",
              "vadd.bf16 m10, m10, m12", "delay 66", "vredsum.row.bf16 m10, m10", "delay 39",
              "vli.all m6, 14848", "delay 66", "vmul.bf16 m10, m10, m6", "delay 66",
              "vli.all m6, 14120", "delay 66", "vadd.bf16 m10, m10, m6", "delay 66",
              "vsqrt.bf16 m10, m10", "delay 66", "vrecip.bf16 m10, m10", "delay 66",
              "add x16, x14, x0", f"li x17, {EXTRA_WORD + MASK_BYTES // 4}", "li x18, 64",
              f"li x3, {STAGE_WORD}", f"li x9, {STAGE_WORD + 256}", "norm_output:", *_load_pair(2, 16),
              "vmul.bf16 m2, m2, m10", "delay 66", *_load_pair(6, 17), "vmul.bf16 m2, m2, m6", "delay 66"]
    bridge = Epilogue(e.m, e.n, norm_base, output_dtype="fp8", output_scale_exponent=e.norm_scale_exponent,
                      scratch_registers=(10, 11))
    lines += bridge.store(20)
    lines += ["addi x16, x16, 512", "addi x17, x17, 512", "li x19, 2048", "add x20, x20, x19",
              "addi x18, x18, -1", "bne x18, x0, norm_output", "nop", "li x19, 32768", "add x14, x14, x19",
              "addi x13, x13, -1", "bne x13, x0, norm_row", "nop"]
    check_matrix_registers("\n".join(lines))
    return lines


def append_postpass(source, epilogue):
    """Replace only the final ECALL, retaining the original recipe verbatim."""
    check_matrix_registers(source)
    lines = source.rstrip().splitlines()
    if not lines or lines[-1] != "ecall" or lines.count("ecall") != 1:
        raise ValueError("expected one final ECALL")
    result = "\n".join(lines[:-1] + postpass(epilogue) + ["ecall"]) + "\n"
    check_matrix_registers(result)
    return result
