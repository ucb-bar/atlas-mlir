"""Ordered BF16 tile epilogues and explicit staged-input accounting.

The contraction recipes retain their BF16 logical output pointers. Conversion
maps those pointers to compact FP8 tiles or channels-last rows at the store.
Scalar scratch defaults to x30/x31; recipes may select another free pair.
m6..m9 are scratch; MXU0 accumulators and m4/m5 stay live. The exact add_rne
step also writes m10..m51 and m54..m57, so checked_assembly admits it only
when the recipe itself names none of the registers the hook writes.
"""

from dataclasses import dataclass
import math

from atlas_rne_add import rne_add


@dataclass(frozen=True)
class TileInput:
    """A staged tile tensor, or one BF16 broadcast tile per column for channel.

    Step("add", TileInput(offset, layout="channel")) is a per-column bias add.
    Channel tiles are ordered by column and reused across all output tile rows.
    """

    offset: int
    layout: str = "tiles"
    dtype: str = "bf16"
    scale_exponent: int = 0

    def bytes(self, m, n):
        return m * n if self.dtype == "fp8" else 2 * (m * n if self.layout == "tiles" else 32 * n)


@dataclass(frozen=True)
class Step:
    op: str
    operand: TileInput | None = None


@dataclass(frozen=True)
class Epilogue:
    m: int
    n: int
    output_offset: int
    steps: tuple[Step, ...] = ()
    output_dtype: str = "bf16"
    output_scale_exponent: int = 0
    output_layout: str = "tiles"
    scratch_registers: tuple[int, int] = (30, 31)
    output_row_stride: int | None = None
    output_column_tile: int = 0

    def __post_init__(self):
        if (not isinstance(self.scratch_registers, tuple) or len(self.scratch_registers) != 2
            or any(type(r) is not int or not 1 <= r <= 31 or r in (3, 8, 9) for r in self.scratch_registers)
            or self.scratch_registers[0] == self.scratch_registers[1]):
            raise ValueError("hook scratch requires two distinct nonzero scalar registers")
        if any(type(v) is not int or v <= 0 or v % 32 for v in (self.m, self.n)):
            raise ValueError("epilogue physical dimensions must be positive multiples of 32")
        if type(self.output_offset) is not int or self.output_offset < 0 or self.output_offset % 32:
            raise ValueError("epilogue output offset must be DMA aligned")
        if type(self.output_column_tile) is not int or self.output_column_tile < 0:
            raise ValueError("column-tile base must be a nonnegative integer")
        if self.output_row_stride is not None:
            if (type(self.output_row_stride) is not int or self.output_row_stride % 1024
                or self.output_row_stride < (self.output_column_tile * 32 + self.n) * 32):
                raise ValueError("consumer tile-row stride must contain the producer shard")
        if (self.output_row_stride is not None or self.output_column_tile) and (
            self.output_dtype != "fp8" or self.output_layout != "tiles"):
            raise ValueError("consumer stride and column base require FP8 tiles")
        if self.output_column_tile and self.output_row_stride is None:
            raise ValueError("column-tile base requires an explicit consumer stride")
        if self.output_offset + max(2 * self.m * self.n, self.output_bytes) >= 2**32:
            raise ValueError("epilogue output exceeds DMA low32 window")
        if self.output_dtype not in ("bf16", "fp8") or self.output_layout not in ("tiles", "channels_last"):
            raise ValueError("unsupported epilogue output format")
        if self.output_layout == "channels_last" and (self.output_dtype != "fp8" or self.n & (self.n - 1)):
            raise ValueError("channels-last output requires FP8 and power-of-two padded channels")
        if type(self.output_scale_exponent) is not int or not -126 <= self.output_scale_exponent <= 119:
            raise ValueError("FP8 scale and clipping bounds must be normal finite BF16")
        if self.output_dtype == "bf16" and self.output_scale_exponent != 0:
            raise ValueError("BF16 output has no FP8 scale")
        for step in self.steps:
            if step.op not in ("add", "add_rne", "mul", "relu", "exp", "recip", "silu", "gelu_tanh", "connector_scale"):
                raise ValueError("unsupported ordered epilogue step")
            if step.op == "add_rne" and (step.operand is None or step.operand.layout != "tiles"
                                         or step.operand.dtype != "bf16" or self.output_dtype != "bf16"):
                raise ValueError("exact residual add requires BF16 tile operand and output")
            if (step.op in ("add", "add_rne", "mul")) != isinstance(step.operand, TileInput):
                raise ValueError("binary steps require a staged tile, unary steps have no operand")
            if step.operand:
                operand = step.operand
                if operand.dtype not in ("bf16", "fp8"):
                    raise ValueError("operand dtype must be BF16 or FP8")
                if operand.layout not in (("channels_last",) if operand.dtype == "fp8" else ("tiles", "channel")):
                    raise ValueError("BF16 operands require tiles/channel, FP8 operands require channels_last")
                if type(operand.scale_exponent) is not int or not -126 <= operand.scale_exponent <= 127:
                    raise ValueError("operand scale must be normal finite BF16")
                if operand.dtype == "bf16" and operand.scale_exponent:
                    raise ValueError("BF16 operands have no FP8 scale")
                if operand.dtype == "fp8" and self.n & (self.n - 1):
                    raise ValueError("channels-last FP8 operands require power-of-two padded channels")
                size = operand.bytes(self.m, self.n)
                if type(operand.offset) is not int or operand.offset < 0 or operand.offset % 32:
                    raise ValueError("operand offset must be DMA aligned")
                if operand.offset < self.output_offset + self.output_bytes and self.output_offset < operand.offset + size:
                    raise ValueError("epilogue operand overlaps output")
                if operand.offset + size >= 2**32:
                    raise ValueError("operand exceeds DMA low32 window")

    @property
    def output_bytes(self):
        if self.output_row_stride is not None:
            return (self.m // 32 - 1) * self.output_row_stride + self.output_column_tile * 1024 + self.n * 32
        return self.m * self.n * (2 if self.output_dtype == "bf16" else 1)

    @property
    def dma_store_bytes(self):
        """Actual transferred bytes; output_bytes includes untouched stride holes."""
        return self.m * self.n * (2 if self.output_dtype == "bf16" else 1)

    @property
    def dma_load_bytes(self):
        return sum(1024 if step.operand.dtype == "fp8" else 2048 for step in self.steps if step.operand) * (self.m // 32) * (self.n // 32)

    @property
    def input_regions(self):
        return tuple(dict(offset=step.operand.offset, size_bytes=step.operand.bytes(self.m, self.n),
                          layout=step.operand.layout, dtype=step.operand.dtype, scale_exponent=step.operand.scale_exponent)
                     for step in self.steps if step.operand)

    def _registers(self, lines):
        return [line.replace("@s0", f"x{self.scratch_registers[0]}")
                    .replace("@s1", f"x{self.scratch_registers[1]}") for line in lines]

    def _check_output_register(self, output_reg):
        if (type(output_reg) is not int or not 1 <= output_reg <= 31
            or output_reg in (*self.scratch_registers, 3, 8, 9)):
            raise ValueError("output pointer must be distinct from hook and store scratch")

    def _delta(self, output_reg):
        return [f"li @s0, {self.output_offset}", f"sub @s0, x{output_reg}, @s0"]

    def apply(self, output_reg):
        self._check_output_register(output_reg)
        lines = []
        for step in self.steps:
            if step.operand:
                if step.operand.dtype == "fp8":
                    lines += self._fp8_operand(output_reg, step.operand)
                    lines += [f"v{step.op}.bf16 m2, m2, m6", "delay 66"]
                    continue
                lines += self._delta(output_reg)
                if step.operand.layout == "channel":
                    if not self.n & (self.n - 1):
                        lines += [f"li @s1, {self.n * 64 - 1}", "and @s0, @s0, @s1"]
                    else:
                        # Remove one BF16 tile-row stride per boundary crossed.
                        # Compare the original pointer so both scratch registers
                        # suffice, without labels or clobbering live pointers.
                        stride = self.n * 64
                        for row in range(1, self.m // 32):
                            lines += [f"li @s1, {self.output_offset + row * stride}",
                                      f"sltu @s1, x{output_reg}, @s1", "xori @s1, @s1, 1"]
                            previous = 0
                            for bit in range(stride.bit_length()):
                                if stride & (1 << bit):
                                    if bit != previous:
                                        lines += [f"slli @s1, @s1, {bit - previous}"]
                                    lines += ["sub @s0, @s0, @s1"]
                                    previous = bit
                # The exact add keeps m4/m5 and m0/m1, reading the residual from m56.
                residual = 56 if step.op == "add_rne" else 6
                lines += [f"li @s1, {step.operand.offset}", "add @s0, @s0, @s1", "li @s1, 2048",
                          "dma.load x3, @s0, @s1, ch0", "dma.wait ch0", f"vload m{residual}, 0(x3)", "delay 34",
                          f"vload m{residual + 1}, 0(x9)", "delay 34"]
                lines += (rne_add(56, 2, 2, sign=54) if step.op == "add_rne" else
                          [f"v{step.op}.bf16 m2, m2, m6", "delay 66"])
            elif step.op == "connector_scale":
                # SmolVLA materializes sqrt(960) as BF16 31 before multiplying.
                lines += ["vli.all m8, 16888", "delay 66", "vmul.bf16 m2, m2, m8", "delay 66"]
            elif step.op == "gelu_tanh":
                # Native tanh GELU sequence, retaining x until the last multiply.
                # Constants are BF16 RNE encodings of .044715, sqrt(2/pi), 1, .5.
                lines += ["vsquare.bf16 m6, m2", "delay 66", "vmul.bf16 m6, m6, m2", "delay 66",
                          "vli.all m8, 15671", "delay 66", "vmul.bf16 m6, m6, m8", "delay 66",
                          "vadd.bf16 m6, m2, m6", "delay 66", "vli.all m8, 16204", "delay 66",
                          "vmul.bf16 m6, m6, m8", "delay 66", "vtanh.bf16 m6, m6", "delay 66",
                          "vli.all m8, 16256", "delay 66", "vadd.bf16 m6, m6, m8", "delay 66",
                          "vli.all m8, 16128", "delay 66", "vmul.bf16 m6, m6, m8", "delay 66",
                          "vmul.bf16 m2, m2, m6", "delay 66"]
            elif step.op == "silu":
                # Preserve v in m2 until the final multiply. Constants are
                # short lived and never share the high MRF banks with live data.
                lines += ["vli.all m8, 49024", "delay 66", "vmul.bf16 m6, m2, m8", "delay 66",
                          "vli.all m8, 49840", "delay 66", "vmaximum.bf16 m6, m6, m8", "delay 66",
                          "vexp.bf16 m6, m6", "delay 66", "vli.all m8, 16256", "delay 66",
                          "vadd.bf16 m6, m6, m8", "delay 66", "vrecip.bf16 m6, m6", "delay 66",
                          "vmul.bf16 m2, m2, m6", "delay 66"]
            else:
                lines += [f"v{step.op}.bf16 m2, m2", "delay 66"]
        return self._registers(lines)

    def _fp8_operand(self, output_reg, operand):
        # The logical BF16 output pointer determines the NHWC row/channel tile.
        lines = self._delta(output_reg)
        lines += ["srli @s0, @s0, 1", f"li @s1, {-(self.n * 32) & 0xffffffff}", "and @s0, @s0, @s1",
                  f"li @s1, {self.output_offset}", f"sub @s1, x{output_reg}, @s1", "srli @s1, @s1, 6"]
        shift = 32 - int(math.log2(self.n))
        lines += ([f"andi @s1, @s1, {self.n - 1}"] if self.n <= 2048 else
                  [f"slli @s1, @s1, {shift}", f"srli @s1, @s1, {shift}"])
        lines += ["add @s0, @s0, @s1", f"li @s1, {operand.offset}", "add @s0, @s0, @s1", "li @s1, 32"]
        for row in range(32):
            lines += ["dma.load x3, @s0, @s1, ch0", "dma.wait ch0"]
            if row != 31:
                lines += ["addi x3, x3, 8"]
                lines += ([f"addi @s0, @s0, {self.n}"] if self.n < 2048 else
                          [f"li @s1, {self.n}", "add @s0, @s0, @s1", "li @s1, 32"])
        lines += ["addi x3, x3, -248", "vload m6, 0(x3)", "delay 34",
                  "vmatpush.acc.fp8.mxu1 acc0, m6", "delay 34", "vmatpop.bf16.acc.mxu1 m6, acc0", "delay 34"]
        if operand.scale_exponent:
            lines += [f"vli.all m8, {(127 + operand.scale_exponent) << 7}", "delay 66",
                      "vmul.bf16 m6, m6, m8", "delay 66"]
        return lines

    def store(self, output_reg):
        self._check_output_register(output_reg)
        if self.output_dtype == "bf16":
            return ["vstore m2, 0(x3)", "delay 34", "vstore m3, 0(x9)", "delay 34",
                    f"dma.store x{output_reg}, x3, x8, ch0", "dma.wait ch0"]
        # Clip before MXU pop: its RTL rounding can otherwise encode 480 as NaN.
        # MXU1 is scratch so pending MXU0 accumulator tiles remain untouched.
        import struct
        limit = 448.0 * 2.0**self.output_scale_exponent
        bits = struct.unpack("<I", struct.pack("<f", limit))[0] >> 16
        lines = [f"vli.all m8, {bits | 0x8000}", "delay 66", "vmaximum.bf16 m2, m2, m8", "delay 66",
                 f"vli.all m8, {bits}", "delay 66", "vminimum.bf16 m2, m2, m8", "delay 66",
                 "vmatpush.acc.bf16.mxu1 acc0, m2", "delay 34",
                 f"seli e0, {127 - self.output_scale_exponent}", "vmatpop.fp8.acc.mxu1 m6, acc0, e0", "delay 34",
                 "vstore m6, 0(x3)", "delay 34"]
        lines += self._delta(output_reg)
        if self.output_layout == "tiles":
            lines += ["srli @s0, @s0, 1"]
            correction = 0 if self.output_row_stride is None else self.output_row_stride - self.n * 32
            if correction:
                # Add one stride correction for each row boundary crossed. Only two
                # scalar scratch registers are needed, including non-power-of-two N.
                for row in range(1, self.m // 32):
                    lines += [f"li @s1, {self.output_offset + row * self.n * 64}",
                              f"sltu @s1, x{output_reg}, @s1", "xori @s1, @s1, 1"]
                    previous = 0
                    for bit in range(correction.bit_length()):
                        if correction & (1 << bit):
                            if bit != previous:
                                lines += [f"slli @s1, @s1, {bit - previous}"]
                            lines += ["add @s0, @s0, @s1"]
                            previous = bit
            lines += [f"li @s1, {self.output_offset + self.output_column_tile * 1024}", "add @s0, @s0, @s1",
                      "li @s1, 1024", "dma.store @s0, x3, @s1, ch0", "dma.wait ch0"]
        else:
            # delta = row_tile*n*64 + column_tile*2048. Convert to NHWC.
            lines += ["srli @s0, @s0, 1", f"li @s1, {-(self.n * 32) & 0xffffffff}", "and @s0, @s0, @s1",
                      f"li @s1, {self.output_offset}", f"sub @s1, x{output_reg}, @s1", "srli @s1, @s1, 6",
                      f"andi @s1, @s1, {self.n - 1}" if self.n <= 2048 else f"slli @s1, @s1, {32 - int(math.log2(self.n))}"]
            if self.n > 2048:
                lines += [f"srli @s1, @s1, {32 - int(math.log2(self.n))}"]
            lines += ["add @s0, @s0, @s1", f"li @s1, {self.output_offset}", "add @s0, @s0, @s1", "li @s1, 32"]
            for row in range(32):
                lines += [f"dma.store @s0, x3, @s1, ch0", "dma.wait ch0"]
                if row != 31:
                    lines += ["addi x3, x3, 8", f"addi @s0, @s0, {self.n}"] if self.n < 2048 else [
                        "addi x3, x3, 8", f"li @s1, {self.n}", "add @s0, @s0, @s1", "li @s1, 32"]
            lines += ["addi x3, x3, -248"]
        return self._registers(lines)
