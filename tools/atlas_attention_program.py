"""Parameterized head-dimension-64 FP8 attention in Atlas text assembly.

BF16 operands and output use the existing matmul layout: row block, head,
32-column tile, left/right 16-column halves. Scales denote original / encoded.
The mask table is shared across heads and is read by the device, not unrolled.
"""

from dataclasses import dataclass
import math
import struct


@dataclass(frozen=True)
class AttentionConfig:
    query_rows: int
    keys: int
    query_heads: int
    kv_heads: int
    head_dim: int = 64
    mask: str = "none"
    query_start: int | None = None
    scale: float = 0.125
    q_exponent: int = 0
    k_exponent: int = 0
    v_exponent: int = 0
    p_exponent: int = -8

    def __post_init__(self):
        if any(type(x) is not int or x <= 0 for x in (self.query_rows, self.keys, self.query_heads, self.kv_heads)):
            raise ValueError("attention dimensions must be positive integers")
        if self.head_dim != 64 or self.query_heads % self.kv_heads:
            raise ValueError("require head dimension 64 and integral GQA group")
        if self.mask not in ("none", "causal", "padded", "arbitrary"):
            raise ValueError("unknown mask kind")
        if not math.isfinite(self.scale) or self.scale <= 0:
            raise ValueError("attention scale must be positive and finite")
        for value in (self.q_exponent, self.k_exponent, self.v_exponent, self.p_exponent):
            if type(value) is not int or not -60 <= value <= 60:
                raise ValueError("FP8 scale exponent must be an integer in -60..60")
        _bf16(self.scale * 2.0 ** (self.q_exponent + self.k_exponent))

    @property
    def query_tiles(self):
        return (self.query_rows + 31) // 32

    @property
    def key_tiles(self):
        return (self.keys + 31) // 32


def _bf16(value):
    bits = struct.unpack("<I", struct.pack("<f", value))[0]
    if bits & 65535:
        raise ValueError("assembly constants must be exactly representable BF16")
    return bits >> 16


def tile_masks(config, mask_tiles=None):
    """Return allowed Boolean [query_tiles,key_tiles,32,32] tiles.

    An arbitrary table accepts None for a full tile, False for a skipped tile,
    or a Boolean [32,32] tile. Padding always masks rows and keys outside shape.
    """
    import numpy as np

    c = config
    result = np.ones((c.query_tiles, c.key_tiles, 32, 32), dtype=bool)
    if c.mask == "arbitrary" and mask_tiles is None:
        raise ValueError("arbitrary attention requires a mask-tile table")
    if mask_tiles is not None:
        if len(mask_tiles) != c.query_tiles or any(len(row) != c.key_tiles for row in mask_tiles):
            raise ValueError("mask table shape differs from query/key tile counts")
        for qi, row in enumerate(mask_tiles):
            for ki, tile in enumerate(row):
                if tile is not None:
                    result[qi, ki] = np.asarray(tile, dtype=bool)
    start = c.keys - c.query_rows if c.query_start is None else c.query_start
    for qi in range(c.query_tiles):
        qr = np.arange(qi * 32, (qi + 1) * 32)
        for ki in range(c.key_tiles):
            kr = np.arange(ki * 32, (ki + 1) * 32)
            result[qi, ki] &= (qr[:, None] < c.query_rows) & (kr[None, :] < c.keys)
            if c.mask == "causal":
                result[qi, ki] &= kr[None, :] <= start + qr[:, None]
    return result


def pack_tiles(value):
    """Encode BF16 [heads,rows,64] in the projection's native tiled layout."""
    import numpy as np
    import torch

    if isinstance(value, torch.Tensor):
        value = value.to(torch.bfloat16).contiguous().view(torch.uint16).numpy()
    value = np.asarray(value)
    if value.dtype != np.uint16 or value.ndim != 3 or value.shape[2] != 64:
        raise ValueError("expected BF16 words [heads,rows,64]")
    heads, rows, _ = value.shape
    padded = np.zeros((heads, (rows + 31) // 32 * 32, 64), dtype="<u2")
    padded[:, :rows] = value
    return b"".join(padded[h, r:r + 32, d:d + 16].tobytes()
        for r in range(0, padded.shape[1], 32) for h in range(heads) for d in range(0, 64, 16))


def unpack_tiles(data, heads, rows):
    import numpy as np

    padded_rows = (rows + 31) // 32 * 32
    words = np.frombuffer(data, dtype="<u2").reshape(padded_rows // 32, heads, 4, 32, 16)
    return words.transpose(1, 0, 3, 2, 4).reshape(heads, padded_rows, 64)[:, :rows].copy()


def stage(config, q, k, v, mask_tiles=None):
    """Build one exact arena and its offsets, including immutable mask descriptors."""
    import numpy as np

    c = config
    allowed = tile_masks(c, mask_tiles)
    for value, shape in zip((q, k, v), ((c.query_heads, c.query_rows, 64),
        (c.kv_heads, c.keys, 64), (c.kv_heads, c.keys, 64))):
        if tuple(value.shape) != shape:
            raise ValueError("input tensor shapes differ from attention configuration")
    parts = [pack_tiles(x) for x in (q, k, v)]
    expected_sizes = [c.query_tiles * c.query_heads * 4096, c.key_tiles * c.kv_heads * 4096,
        c.key_tiles * c.kv_heads * 4096]
    if [len(x) for x in parts] != expected_sizes:
        raise ValueError("input tensor shapes differ from attention configuration")
    offsets = {"q": 0, "k": len(parts[0]), "v": len(parts[0]) + len(parts[1])}
    arena = bytearray(b"".join(parts))
    offsets["descriptors"] = len(arena)
    table_size = (c.query_tiles * c.key_tiles * 8 + 31) // 32 * 32
    arena.extend(bytes(table_size))
    descriptors = []
    for qi in range(c.query_tiles):
        for ki in range(c.key_tiles):
            tile = allowed[qi, ki]
            if not tile.any():
                descriptors.append((0, 1))
            elif tile.all():
                descriptors.append((0, 0))
            else:
                descriptors.append((len(arena), 0))
                mask = np.where(tile, 0, 0x7f80).astype("<u2")
                arena.extend(mask[:, :16].tobytes() + mask[:, 16:].tobytes())
    table = b"".join(struct.pack("<II", *entry) for entry in descriptors)
    arena[offsets["descriptors"]:offsets["descriptors"] + len(table)] = table
    offsets["descriptor_bytes"] = table_size
    offsets["output"] = len(arena) + 256
    offsets["output_bytes"] = c.query_tiles * c.query_heads * 4096
    arena.extend(b"\xa5" * 256 + b"\xcd" * offsets["output_bytes"] + b"\x5a" * 256)
    return bytes(arena), offsets


def assembly(config, offsets, *, dram_base=0x90000000):
    """Generate a bounded physical-register program, with loops over heads and tiles."""
    c = config
    if c.key_tiles * 2048 > 256 * 1024 or offsets["descriptor_bytes"] > 128 * 1024:
        raise ValueError("score tiles or descriptors exceed reserved VMEM banks")
    if dram_base % 32 or dram_base < 0 or dram_base + offsets["output"] + offsets["output_bytes"] >= 2**32:
        raise ValueError("arena must fit one 32-bit aligned DMA window")
    lines = []

    def emit(*items):
        lines.extend(items)

    def op(instruction, delay=66):
        emit(instruction, f"delay {delay}")

    def pair_load(reg, base="x2", offset=0):
        op(f"vload m{reg}, {offset}({base})", 34)
        op(f"vload m{reg + 1}, {offset + 8}({base})", 34)

    def pair_store(reg, base, offset=0):
        op(f"vstore m{reg}, {offset}({base})", 34)
        op(f"vstore m{reg + 1}, {offset + 8}({base})", 34)

    def quantize(dest, exponent_reg, offset=0):
        pair_load(12, offset=offset)
        bound = (40, 56, 48)[exponent_reg]
        op(f"vminimum.bf16 m12, m12, m{bound}")
        op(f"vmaximum.bf16 m12, m12, m{bound + 2}")
        op("vmatpush.acc.bf16.mxu0 acc0, m12", 32)
        op(f"vmatpop.fp8.acc.mxu0 m{dest}, acc0, e{exponent_reg}", 34)

    def key_loop_start(label):
        emit(f"li x11, {c.key_tiles}", "add x12, x13, x0", "li x10, 0x10000", f"{label}:")

    def key_loop_end(label, extra=()):
        emit(*extra, "addi x10, x10, 512", "addi x12, x12, 8", "addi x11, x11, -1",
            f"beq x11, x0, {label}_done", "nop", f"jal x0, {label}", "nop", f"{label}_done:")

    emit("li x30, 0", "dma.config x30, ch0", "dma.wait ch0", "li x2, 0x800", "li x3, 0xc00",
        "li x5, 4096", "li x9, 2048", f"li x23, {c.kv_heads * 4096}",
        f"li x19, {dram_base + offsets['q']}", f"li x20, {dram_base + offsets['output']}",
        f"li x26, {dram_base + offsets['k']}", f"li x27, {dram_base + offsets['v']}",
        f"li x15, {c.query_heads}", f"li x16, {c.query_heads // c.kv_heads}",
        f"seli e0, {127 - c.q_exponent}", f"seli e1, {127 - c.k_exponent}",
        f"seli e2, {127 - c.v_exponent}", "seli e3, 127", "li x21, 0x4000",
        f"li x22, {dram_base + offsets['descriptors']}")
    for offset in range(0, offsets["descriptor_bytes"], 4096):
        emit(f"li x31, {min(4096, offsets['descriptor_bytes'] - offset)}",
            "dma.load x21, x22, x31, ch0", "dma.wait ch0")
        if offset + 4096 < offsets["descriptor_bytes"]:
            emit("li x31, 4096", "add x22, x22, x31", "addi x21, x21, 1024")
    op("vli.all m24, 0")
    op(f"vli.all m8, {_bf16(c.scale * 2.0 ** (c.q_exponent + c.k_exponent))}")
    op("vli.all m26, 65407")  # Most negative finite BF16 for empty rows.
    op(f"vli.all m28, {_bf16(2.0 ** -c.p_exponent)}")
    op(f"vli.all m30, {_bf16(2.0 ** (c.p_exponent + c.v_exponent))}")
    op("vli.all m62, 16256")  # One for empty-row denominator clamp.
    for index, exponent in enumerate((c.q_exponent, c.k_exponent, c.v_exponent)):
        bound = (40, 56, 48)[index]
        op(f"vli.all m{bound}, {_bf16(448.0 * 2.0 ** exponent)}")
        op(f"vli.all m{bound + 2}, {_bf16(-448.0 * 2.0 ** exponent)}")
    op(f"vli.all m60, {_bf16(448.0)}")
    op(f"vli.all m54, {_bf16(-128.0)}")
    emit("head_loop:", "add x4, x19, x0", "add x8, x20, x0", f"li x14, {c.query_tiles}",
        "li x13, 0x10000", "query_loop:", "dma.load x2, x4, x5, ch0", "dma.wait ch0")
    quantize(0, 0)
    quantize(2, 0, 16)
    op("vli.all m4, 65408")  # Negative infinity.
    emit("add x6, x26, x0")
    key_loop_start("scores_loop")
    emit("lw x22, 4(x12)", "delay 4", "bne x22, x0, scores_masked", "nop",
        "dma.load x2, x6, x5, ch0", "dma.wait ch0")
    op("vmatpush.acc.bf16.mxu1 acc0, m24", 32)
    for dim in range(2):
        quantize(14, 1, dim * 16)
        op("vmatpush.weight.mxu1 w0, m14", 32)
        op(f"vmatmul.acc.mxu1 acc0, m{dim * 2}, w0", 96)
    op("vmatpop.bf16.acc.mxu1 m16, acc0", 34)
    op("vmul.bf16 m16, m16, m8")
    emit("lw x22, 0(x12)", "delay 4", "beq x22, x0, scores_ready", "nop",
        f"li x31, {dram_base}", "add x22, x22, x31", "dma.load x2, x22, x9, ch0", "dma.wait ch0")
    pair_load(18)
    op("vsub.bf16 m16, m16, m18")
    emit("jal x0, scores_ready", "nop", "scores_masked:")
    op("vli.all m16, 65408")
    emit("scores_ready:")
    pair_store(16, "x10")
    op("vredmax.row.bf16 m22, m16", 39)
    op("vmaximum.bf16 m4, m4, m22")
    key_loop_end("scores_loop", ("add x6, x6, x23",))
    op("vmaximum.bf16 m4, m4, m26")
    op("vli.all m6, 0")
    key_loop_start("exp_loop")
    emit("lw x22, 4(x12)", "delay 4", "bne x22, x0, exp_masked", "nop")
    pair_load(16, "x10")
    op("vsub.bf16 m20, m16, m4")
    op("vmaximum.bf16 m20, m20, m54")
    emit("lw x22, 0(x12)", "delay 4", "beq x22, x0, exp_unmasked", "nop",
        f"li x31, {dram_base}", "add x22, x22, x31", "dma.load x2, x22, x9, ch0", "dma.wait ch0")
    pair_load(18)
    op("vsub.bf16 m20, m20, m18")
    emit("exp_unmasked:")
    op("vexp.bf16 m20, m20")
    emit("jal x0, exp_ready", "nop", "exp_masked:")
    op("vli.all m20, 0")
    emit("exp_ready:")
    pair_store(20, "x10")
    op("vredsum.row.bf16 m22, m20", 39)
    op("vadd.bf16 m6, m6, m22")
    key_loop_end("exp_loop")
    op("vmaximum.bf16 m6, m6, m62")
    op("vrecip.bf16 m10, m6")
    op("vmul.bf16 m10, m10, m28")
    op("vmatpush.acc.bf16.mxu1 acc0, m24", 32)
    op("vmatpush.acc.bf16.mxu1 acc1, m24", 32)
    emit("add x7, x27, x0")
    key_loop_start("pv_loop")
    emit("lw x22, 4(x12)", "delay 4", "bne x22, x0, pv_skip", "nop")
    pair_load(20, "x10")
    op("vmul.bf16 m20, m20, m10")
    op("vminimum.bf16 m20, m20, m60")
    op("vmatpush.acc.bf16.mxu0 acc0, m20", 32)
    op("vmatpop.fp8.acc.mxu0 m36, acc0, e3", 34)
    emit("dma.load x2, x7, x5, ch0", "dma.wait ch0")
    for dim in range(2):
        quantize(14, 2, dim * 16)
        op("vtrpose.xlu m14, m14")
        op("vmatpush.weight.mxu1 w0, m14", 32)
        op(f"vmatmul.acc.mxu1 acc{dim}, m36, w0", 96)
    emit("pv_skip:")
    key_loop_end("pv_loop", ("add x7, x7, x23",))
    for dim in range(2):
        op(f"vmatpop.bf16.acc.mxu1 m{32 + dim * 2}, acc{dim}", 34)
        op(f"vmul.bf16 m{32 + dim * 2}, m{32 + dim * 2}, m30")
        pair_store(32 + dim * 2, "x3", dim * 16)
    emit("dma.store x8, x3, x5, ch0", "dma.wait ch0", f"li x31, {c.query_heads * 4096}",
        "add x4, x4, x31", "add x8, x8, x31", f"addi x13, x13, {c.key_tiles * 8}",
        "addi x14, x14, -1", "beq x14, x0, query_done", "nop", "jal x0, query_loop", "nop", "query_done:",
        "li x31, 4096", "add x19, x19, x31", "add x20, x20, x31", "addi x16, x16, -1",
        "bne x16, x0, kv_same", "nop", "add x26, x26, x31", "add x27, x27, x31",
        f"li x16, {c.query_heads // c.kv_heads}", "kv_same:", "addi x15, x15, -1",
        "beq x15, x0, finished", "nop", "jal x0, head_loop", "nop", "finished:", "dma.wait ch0", "delay 128", "ecall")
    return "\n".join(lines) + "\n"
