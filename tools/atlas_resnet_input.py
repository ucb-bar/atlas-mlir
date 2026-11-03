"""Device row gathers from FP8 channels-last tensors into tiled MXU A operands.

A 1x1 input gathers one 32-byte channel line per output row. A kxk input
(odd kernel, same padding) orders K as (ky, kx, channel) with the weights
permuted to match, so every A tile row is still one 32-byte channel line of
one tap. Padded taps and padded output rows read the declared zero line.
"""

from dataclasses import dataclass
import math


@dataclass(frozen=True)
class DirectInput:
    offset: int
    channels: int
    height: int
    width: int
    stride: int = 1
    zero_offset: int = 0
    kernel: int = 1
    padding: int = 0

    @property
    def output_height(self):
        return (self.height + 2 * self.padding - self.kernel) // self.stride + 1

    @property
    def output_width(self):
        return (self.width + 2 * self.padding - self.kernel) // self.stride + 1

    @property
    def output_rows(self):
        return self.output_height * self.output_width

    @property
    def source_bytes(self):
        return ((self.height * self.width + 31) // 32) * 32 * self.channels

    @property
    def input_regions(self):
        return (dict(offset=self.offset, size_bytes=self.source_bytes, dtype="fp8", layout="channels_last"),
                dict(offset=self.zero_offset, size_bytes=32, dtype="uint8", layout="zero"))

    def validate(self, m, k, *, output_offset=None, output_bytes=None):
        if any(type(value) is not int or value <= 0 for value in (self.channels, self.height, self.width)):
            raise ValueError("direct input requires positive integer channels and spatial dimensions")
        if self.channels % 32 or self.channels & (self.channels - 1):
            raise ValueError("direct input channels must be a power of two divisible by 32")
        if type(self.stride) is not int or self.stride not in (1, 2):
            raise ValueError("direct input supports stride one or two")
        if type(self.kernel) is not int or self.kernel < 1 or not self.kernel % 2 or self.padding != self.kernel // 2:
            raise ValueError("direct input supports odd kernels with same padding")
        if (m, k) != (((self.output_rows + 31) // 32) * 32, self.kernel * self.kernel * self.channels):
            raise ValueError("direct input geometry differs from the physical contraction")
        if type(self.offset) is int and self.offset + self.height * self.width * self.channels >= 2**32:
            raise ValueError("direct input source exclusive end must fit an unsigned 32-bit scalar register")
        for offset, size in ((self.offset, self.source_bytes), (self.zero_offset, 32)):
            if type(offset) is not int or offset < 0 or offset % 32 or offset + size > 2**32:
                raise ValueError("direct input region must be aligned and inside the DMA low32 window")
        if self.offset < self.zero_offset + 32 and self.zero_offset < self.offset + self.source_bytes:
            raise ValueError("direct input source overlaps its zero line")
        output_width = self.output_width
        last = ((m - 1) // output_width) * self.stride * self.width + ((m - 1) % output_width) * self.stride
        if self.offset + (last + 1) * self.channels > 2**32:
            raise ValueError("padded spatial-row address arithmetic exceeds the DMA low32 window")
        if output_offset is not None:
            if type(output_bytes) is not int or output_bytes <= 0:
                raise ValueError("output extent is required for direct-input preservation checks")
            for region in self.input_regions:
                if region['offset'] < output_offset + output_bytes and output_offset < region['offset'] + region['size_bytes']:
                    raise ValueError("direct input or zero line overlaps output")

    def gather(self, destination_reg, virtual_source_reg, label):
        """Load one virtual tiled A tile, preserving caller registers and using only x26..x31.

        The source register is the byte offset in the legacy A matrix, not a
        pointer into the NHWC source. Every output row requests one 32-byte
        line, including padded spatial rows, which read the declared zero line.
        """
        if self.kernel != 1:
            raise ValueError("kxk direct inputs fill whole A blocks with gather_block")
        if destination_reg in range(26, 32) or virtual_source_reg in range(26, 32):
            raise ValueError("gather caller operands overlap its scratch registers")
        channels_shift = int(math.log2(self.channels))
        lines = [f"add x26, x{destination_reg}, x0", f"li x31, {self.channels * 32 - 1}",
                 f"and x30, x{virtual_source_reg}, x31", "srli x30, x30, 5",
                 f"srli x27, x{virtual_source_reg}, {channels_shift + 5}", "slli x27, x27, 5"]
        if self.stride == 2:
            output_width = (self.width + 1) // 2
            lines += ["li x29, 0", f"li x28, {output_width}", f"li x31, {2 * self.width * self.channels}",
                      label + "_map:", f"bltu x27, x28, {label}_mapped", "nop", "sub x27, x27, x28",
                      "add x29, x29, x31", f"jal x0, {label}_map", "nop", label + "_mapped:",
                      "sub x28, x28, x27", f"slli x27, x27, {channels_shift + 1}", "add x27, x27, x29"]
        else:
            lines += [f"slli x27, x27, {channels_shift}"]
        lines += ["add x27, x27, x30", f"li x31, {self.offset}", "add x27, x27, x31",
                  f"li x29, {self.offset + self.height * self.width * self.channels}", "li x31, 32",
                  label + "_rows:", "add x30, x27, x0", f"bltu x27, x29, {label}_valid", "nop",
                  f"li x30, {self.zero_offset}", label + "_valid:", "dma.load x26, x30, x31, ch0", "dma.wait ch0",
                  "addi x26, x26, 8"]
        increment = self.stride * self.channels
        lines += ([f"addi x27, x27, {increment}"] if increment < 2048 else
                  [f"li x30, {increment}", "add x27, x27, x30"])
        if self.stride == 2:
            extra = 2 * (self.width - (self.width + 1) // 2) * self.channels
            lines += ["addi x28, x28, -1", f"bne x28, x0, {label}_no_wrap", "nop",
                      f"li x28, {(self.width + 1) // 2}", f"li x30, {extra}", "add x27, x27, x30",
                      label + "_no_wrap:"]
        return lines + [f"addi x30, x{destination_reg}, 256", f"bne x26, x30, {label}_rows", "nop"]


    def gather_block(self, first_row_tile, row_tiles, bank_words, bank0_word, label):
        """Fill the blocked recipe's A block for a kxk input, tile by tile in (row tile, K tile) order.

        A tiles start at VMEM word bank_words; the tile that would reach word
        2 * bank_words continues at bank0_word instead. Taps are unrolled per K
        tile, so each tap's offset and bounds are immediates. Registers x3, x5,
        x8, x9, x10, x11, x16, and x30 are untouched; every other one is scratch
        that the recipe sets again after the A load.
        """
        s, c, taps = self.stride, self.channels, self.kernel * self.kernel
        start = first_row_tile * 32
        oy, ox = divmod(start, self.output_width)
        lines = [f"li x1, {self.height}", f"li x2, {self.width}", f"li x4, {self.output_height * s}",
                 f"li x6, {self.output_width * s}", "li x7, 32", f"li x12, {self.zero_offset}", f"li x13, {s * c}",
                 f"li x14, {(s * self.width - s * self.output_width) * c}", f"li x19, {bank_words}", f"li x20, {row_tiles}",
                 f"li x21, {oy * s}", f"li x22, {ox * s}", f"li x23, {self.offset + (oy * s * self.width + ox * s) * c}",
                 label + "_tiles:"]
        for tap in range(taps):
            ky, kx = divmod(tap, self.kernel)
            for block in range(c // 32):
                tile = f"{label}_{tap}_{block}"
                offset = ((ky - self.padding) * self.width + kx - self.padding) * c + 32 * block
                lines += [f"li x15, {offset & 0xffffffff}", f"li x17, {(ky - self.padding) & 0xffffffff}",
                          f"li x18, {(kx - self.padding) & 0xffffffff}", "add x24, x21, x0", "add x25, x22, x0",
                          "add x26, x23, x0", "li x27, 32", tile + "_row:", "add x29, x12, x0",
                          f"bgeu x24, x4, {tile}_load", "nop", "add x28, x24, x17", f"bgeu x28, x1, {tile}_load", "nop",
                          "add x28, x25, x18", f"bgeu x28, x2, {tile}_load", "nop", "add x29, x26, x15", tile + "_load:",
                          "dma.load x19, x29, x7, ch0", "dma.wait ch0", "addi x19, x19, 8", *self._advance(tile),
                          "addi x27, x27, -1", f"bne x27, x0, {tile}_row", "nop", f"li x28, {2 * bank_words}",
                          f"bne x19, x28, {tile}_same_bank", "nop", f"li x19, {bank0_word}", tile + "_same_bank:"]
        lines += ["li x27, 32", label + "_next:", *self._advance(label + "_next", 21, 22, 23),
                  "addi x27, x27, -1", f"bne x27, x0, {label}_next", "nop",
                  "addi x20, x20, -1", f"beq x20, x0, {label}_done", "nop", f"jal x0, {label}_tiles", "nop", label + "_done:"]
        return lines

    def _advance(self, label, row=24, column=25, address=26):
        """Step one output position in row-major order: column, then row with the stride-two skip."""
        return [f"add x{address}, x{address}, x13", f"addi x{column}, x{column}, {self.stride}",
                f"bne x{column}, x6, {label}_column", "nop", f"li x{column}, 0", f"addi x{row}, x{row}, {self.stride}",
                f"add x{address}, x{address}, x14", label + "_column:"]


def kernel_major_weights(weights, channels, kernel, n):
    """Permute N-tile-major tiled FP8 weights from K order (c, ky, kx) to (ky, kx, c)."""
    import numpy as np
    k = channels * kernel * kernel
    tiles = np.frombuffer(weights, np.uint8).reshape(n // 32, k // 32, 32, 32)
    matrix = tiles.transpose(1, 2, 0, 3).reshape(k, n)
    taps = matrix.reshape(channels, kernel * kernel, n).transpose(1, 0, 2).reshape(k, n)
    return taps.reshape(k // 32, 32, n // 32, 32).transpose(2, 0, 1, 3).tobytes()


def relocate_assembly(source, arena_base, *, arena_bytes=None, weight_source=None):
    """Relocate base/blocked source while preserving all six gather scratch registers.

    x8 becomes the fixed low32 displacement. Its original constant 2048 use
    is materialized only where needed for a logical BF16 output increment or
    BF16 DMA store. DMA address registers are restored immediately afterwards.
    """
    import re
    if type(arena_base) is not int or arena_base < 0 or arena_base % 64 or arena_base >= 2**64:
        raise ValueError("arena base must be a physical 64-byte-aligned address")
    if weight_source is not None and weight_source.arena_base != arena_base:
        raise ValueError("weight source arena binding differs from relocation")
    displacement = arena_base & 0xffffffff
    if arena_bytes is not None and (type(arena_bytes) is not int or arena_bytes <= 0 or displacement + arena_bytes > 2**32):
        raise ValueError("relocated arena must fit one DMA low32 window")
    if source.splitlines().count("li x30, 1") != 1 or source.splitlines().count("li x8, 2048") != 1:
        raise ValueError("relocation requires the base/blocked retained DMA setup and output increment")
    lines = []
    for line in source.splitlines():
        if line == "li x30, 1":
            lines.append(f"li x30, {arena_base >> 32}")
            continue
        if line == "li x8, 2048":
            lines.append(f"li x8, {displacement}")
            continue
        advance = re.fullmatch(r"add x(\d+), x\1, x8", line)
        if advance:
            lines += ["li x30, 2048", f"add x{advance[1]}, x{advance[1]}, x30"]
            continue
        transfer = re.fullmatch(r"dma\.(load|store) x(\d+), x(\d+), x(\d+), ch([0-7])", line)
        if transfer:
            op, rd, rs1, rs2, channel = transfer.groups()
            if channel in ("2", "3"):
                if weight_source is None:
                    raise ValueError("immutable weight DMA requires explicit source metadata")
                if op != "load":
                    raise ValueError("immutable weight channels only support loads")
                lines.append(line)
                continue
            if rs2 == "8":
                if op != "store" or rd in ("8", "31") or rs1 in ("8", "31"):
                    raise ValueError("unsupported DMA use of the output increment register")
                lines.append("li x31, 2048")
                rs2 = "31"
                line = f"dma.store x{rd}, x{rs1}, x{rs2}, ch{channel}"
            dram = rs1 if op == "load" else rd
            others = (rd, rs2) if op == "load" else (rs1, rs2)
            if dram in ("0", "8") or dram in others or "8" in others:
                raise ValueError("DMA relocation requires distinct writable address and length operands")
            lines += [f"add x{dram}, x{dram}, x8", line, f"sub x{dram}, x{dram}, x8"]
            continue
        if re.search(r"\bx8\b", line) or re.search(r"\bdma\.(load|store)\b", line):
            raise ValueError("unsupported x8 use or DMA transfer syntax")
        if "dma.config" in line and not (re.fullmatch(r"dma.config x30, ch[01]", line) or
                                                   re.fullmatch(r"dma.config x(?:6|15|20), ch[23]", line)):
            raise ValueError("unexpected DMA configuration source")
        lines.append(line)
    return "\n".join(lines) + "\n"
