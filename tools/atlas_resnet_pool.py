"""Device 3x3 stride-two max pooling of FP8 channels-last tensors.

The pool is exact for nonnegative normal FP8 inputs. VUNPACK and VPACK at
unit scale round-trip every such byte, VMAXIMUM orders them as the bytes
order, and an MXU0 product with a one-hot FP8 row (1.0 or zero) moves a byte
into BF16 without rounding. Padded taps contribute zero, which never exceeds
a nonnegative maximum.

Each input row is staged once in its own 8 KiB VMEM slot, so every 1 KiB
register of a row is aligned. A register holds the position pairs (2x, 2x+1)
of one output group. The vertical maximum of the three input rows is
elementwise. The horizontal taps 2x - 1 and 2x + 1 sit one position away in
the same register, so constant one-hot shift matrices on MXU0 bring them into
the lanes of position 2x (the left neighbor of a group's first pair comes
from the previous group's register). The first half of every pair then holds
the output, which MXU1 returns to FP8 and one DMA per output position stores.
"""

from dataclasses import dataclass

REGISTER_BYTES = 1024
BANK_BYTES = 256 * 1024
VMEM_BYTES = 6 * BANK_BYTES
MAX_DMA_BYTES = 4096
SLOT_BYTES = 8192
ONE = 0x38  # FP8 E4M3 1.0


@dataclass(frozen=True)
class MaxPool:
    source_offset: int
    constants_offset: int
    output_offset: int
    channels: int
    height: int
    width: int

    @property
    def output_height(self):
        return self.height // 2

    @property
    def output_width(self):
        return self.width // 2

    @property
    def row_bytes(self):
        return self.width * self.channels

    @property
    def position_lines(self):
        return self.channels // 32

    @property
    def pairs_per_group(self):
        return REGISTER_BYTES // (2 * self.channels)

    @property
    def groups(self):
        return self.output_width // self.pairs_per_group

    @property
    def source_bytes(self):
        return self.height * self.row_bytes

    @property
    def output_bytes(self):
        return self.output_height * self.output_width * self.channels

    @property
    def dma_load_bytes(self):
        return self.source_bytes + len(self.constants())

    @property
    def dma_store_bytes(self):
        return self.output_bytes

    @property
    def input_regions(self):
        return (dict(offset=self.source_offset, size_bytes=self.source_bytes, dtype="fp8", layout="channels_last"),
                dict(offset=self.constants_offset, size_bytes=len(self.constants()), dtype="fp8", layout="shift_matrices"))

    def constants(self):
        """Identity, shift-up, shift-down, and previous-group wrap selectors, one 32x32 FP8 tile each."""
        s = self.position_lines
        tiles = [bytearray(REGISTER_BYTES) for _ in range(4)]
        for row in range(32):
            tiles[0][33 * row] = ONE
            if row + s < 32:
                tiles[1][32 * row + row + s] = ONE
            if row >= s:
                tiles[2][32 * row + row - s] = ONE
            else:
                tiles[3][32 * row + row + 32 - s] = ONE
        return b"".join(tiles)

    def validate(self, arena_bytes=None):
        if any(type(v) is not int or v <= 0 for v in (self.channels, self.height, self.width)):
            raise ValueError("max pool requires positive integer geometry")
        if self.channels % 32 or REGISTER_BYTES % (2 * self.channels):
            raise ValueError("max pool channels must be 32 to 512 and divide a register pair group")
        if self.height % 2 or self.width % 2 or self.output_width % self.pairs_per_group:
            raise ValueError("max pool requires even spatial dimensions that fill whole register groups")
        if (self.output_height * self.output_width) % 32:
            raise ValueError("max pool output rows must fill whole 32-row tiles")
        if self.row_bytes > SLOT_BYTES or self.height * SLOT_BYTES > 4 * BANK_BYTES:
            raise ValueError("max pool rows must fit their VMEM slots below the constant bank")
        regions = ((self.source_offset, self.source_bytes), (self.constants_offset, len(self.constants())),
                   (self.output_offset, self.output_bytes))
        for offset, size in regions:
            if type(offset) is not int or offset < 0 or offset % 32 or offset + size > (arena_bytes or 2**32):
                raise ValueError("max pool region must be aligned and inside the arena")
        for i, (a, x) in enumerate(regions):
            for b, y in regions[i + 1:]:
                if a < b + y and b < a + x:
                    raise ValueError("max pool regions overlap")

    def assembly(self, arena_base):
        self.validate()
        if type(arena_base) is not int or arena_base < 0 or arena_base % 64 or arena_base >= 2**64:
            raise ValueError("arena base must be a physical 64-byte-aligned address")
        low = arena_base & 0xffffffff
        constants = 4 * BANK_BYTES // 4
        lines = [f"li x30, {arena_base >> 32}", "dma.config x30, ch0", "seli e0, 127", "seli e1, 127",
                 f"li x5, {SLOT_BYTES // 4}", f"li x6, {self.channels}", f"li x7, {MAX_DMA_BYTES}",
                 f"li x1, {constants}", f"li x2, {low + self.constants_offset}", "dma.load x1, x2, x7, ch0", "dma.wait ch0"]
        lines += [item for i in range(4) for item in (f"vload m{16 + i}, {8 * i}(x1)", "delay 34")]
        lines += ["li x1, 0", f"li x2, {low + self.source_offset}", f"li x3, {self.height}", "stage_rows:",
                  "add x9, x1, x0"]
        for start in range(0, self.row_bytes, MAX_DMA_BYTES):
            piece = min(MAX_DMA_BYTES, self.row_bytes - start)
            lines += [f"li x4, {piece}", "dma.load x9, x2, x4, ch0", "dma.wait ch0", "add x2, x2, x4"]
            if start + piece < self.row_bytes:
                lines += [f"addi x9, x9, {piece // 4}"]
        lines += ["add x1, x1, x5", "addi x3, x3, -1", "bne x3, x0, stage_rows", "nop",
                  f"li x13, {5 * BANK_BYTES // 4}", "li x11, 0", f"li x12, {low + self.output_offset}",
                  f"li x20, {self.output_width * self.channels}"]
        lines += self._output_row(2) + ["add x11, x11, x5"]
        lines += [f"li x14, {self.output_height - 1}", "rows_loop:"] + self._output_row(3)
        lines += ["add x11, x11, x5", "add x11, x11, x5", "addi x14, x14, -1", "bne x14, x0, rows_loop", "nop", "ecall"]
        return "\n".join(lines) + "\n"

    def _output_row(self, rows):
        """One output row from the input rows starting at VMEM word x11; x12 is its DRAM output."""
        lines = []
        for group in range(self.groups):
            lines += [f"addi x17, x11, {group * REGISTER_BYTES // 4}"]
            for row in range(rows):
                if row:
                    lines += ["add x17, x17, x5"]
                lines += ["vload m10, 0(x17)", "delay 34", f"vunpack.fp8.bf16 m{14 if row else 12}, m10, e0", "delay 67"]
                if row:
                    lines += ["vmaximum.bf16 m12, m12, m14", "delay 66"]
            lines += ["vpack.bf16.fp8 m10, m12, e0", "delay 66", "vtrpose.xlu m11, m10", "delay 66"]
            if group:
                # w0 still holds the previous group's transpose: its last position is this group's left tap.
                lines += ["vmatmul.mxu0 acc0, m19, w0", "delay 96"]
            lines += ["vmatpush.weight.mxu0 w0, m11", "delay 32",
                      "vmatmul.mxu0 acc1, m16, w0", "delay 96", "vmatpop.bf16.acc.mxu0 m2, acc1", "delay 34",
                      "vmatmul.mxu0 acc1, m17, w0", "delay 96", "vmatpop.bf16.acc.mxu0 m4, acc1", "delay 34",
                      "vmaximum.bf16 m2, m2, m4", "delay 66",
                      f"vmatmul{'.acc' if group else ''}.mxu0 acc0, m18, w0", "delay 96",
                      "vmatpop.bf16.acc.mxu0 m4, acc0", "delay 34", "vmaximum.bf16 m2, m2, m4", "delay 66",
                      "vmatpush.acc.bf16.mxu1 acc0, m2", "delay 34", "vmatpop.fp8.acc.mxu1 m6, acc0, e1", "delay 34",
                      "vstore m6, 0(x13)", "delay 34", "add x18, x13, x0",
                      f"li x19, {group * self.pairs_per_group * self.channels}", "add x19, x19, x12"]
            for pair in range(self.pairs_per_group):
                lines += ["dma.store x19, x18, x6, ch0", "dma.wait ch0"]
                if pair + 1 < self.pairs_per_group:
                    lines += [f"addi x18, x18, {self.channels // 2}", f"addi x19, x19, {self.channels}"]
        return lines + ["add x12, x12, x20"]
