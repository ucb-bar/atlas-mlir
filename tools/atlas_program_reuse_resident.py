"""FP8 contraction with resident A and one reusable transposed B column.

This opt-in assembly helper preserves the original 4-GiB DMA base and packed
row-major output-tile layout. A and a K-by-32 B column share VMEM bank 1.
Two physical MXU0 accumulators cover a pair of 32-row tiles; every accumulator
visits K tiles in increasing order. One B load and transpose serves all rows,
and one weight push serves both rows of a pair. Only acc0/acc1 and w0 are used.

DELAY values below are the conservative spacings already emitted by
atlas_program.py: VLOAD/VSTORE/POP 34, transpose/VLI/MUL 66, weight push 32,
and MXU0 matmul 96. They are not fitted to a benchmark or memory profile.
"""

from atlas_program_reuse_blocked import load_b_column

BANK_BYTES = 256 * 1024
RESIDENT_BYTE_BASE = BANK_BYTES
TILE_BYTES = 32 * 32
OUTPUT_TILE_BYTES = 2 * TILE_BYTES
GUARD_BYTES = 256


def admission(m, k, n, output_offset, rescale_exponent):
    """Fail closed unless two-row groups and both resident regions fit."""
    if any(type(x) is not int or x <= 0 or x % 32 for x in (m, k, n)):
        raise ValueError("physical dimensions must be positive multiples of 32")
    if m % 64:
        raise ValueError("resident reuse requires complete pairs of 32-row tiles")
    a_bytes, b_column_bytes = m * k, k * 32
    if a_bytes + b_column_bytes > BANK_BYTES:
        raise ValueError("A plus one B column must fit one 256-KiB VMEM bank")
    if type(output_offset) is not int or output_offset != m * k + k * n + GUARD_BYTES:
        raise ValueError("output offset must preserve the original input and guard layout")
    if type(rescale_exponent) is not int or not -126 <= rescale_exponent <= 127:
        raise ValueError("one normal finite BF16 rescale instruction is required")
    if output_offset + 2 * m * n + GUARD_BYTES >= 2**32:
        raise ValueError("DRAM image must fit the original DMA low-32 window")
    return {
        "a_byte_range": [RESIDENT_BYTE_BASE, RESIDENT_BYTE_BASE + a_bytes],
        "b_column_byte_range": [RESIDENT_BYTE_BASE + a_bytes,
                                RESIDENT_BYTE_BASE + a_bytes + b_column_bytes],
        "resident_bytes": a_bytes + b_column_bytes,
        "bank_bytes": BANK_BYTES,
        "accumulator_slots": [0, 1],
        "weight_slots": [0],
        "dma_load_bytes": a_bytes + k * n,
        "dma_store_bytes": 2 * m * n,
    }


def dma_load_bytes(m, k, n):
    admission(m, k, n, m * k + k * n + GUARD_BYTES, 0)
    return m * k + k * n


def assembly(m, k, n, output_offset, rescale_exponent=0):
    regions = admission(m, k, n, output_offset, rescale_exponent)
    a_word = regions["a_byte_range"][0] // 4
    b_word = regions["b_column_byte_range"][0] // 4
    a_row_words = k * 32 // 4
    out_row_bytes = (n // 32) * OUTPUT_TILE_BYTES
    lines = [
        f"li x1, {a_word}", f"li x2, {b_word}", "li x3, 0xa00",
        "li x5, 1024", "li x8, 2048", "li x9, 0xb00",
        f"li x14, {n // 32}", f"li x15, {m * k}",
        f"li x16, {a_row_words}", f"li x17, {out_row_bytes}",
        f"li x18, {2 * a_row_words}", f"li x23, {output_offset}",
        f"li x26, {2 * out_row_bytes}", "li x30, 1", "dma.config x30, ch0", *(["dma.config x30, ch1"] if k > 32 else []),
        # Exactly one VLI m4 preserves the existing 0x25f rescale patch protocol.
        # m4/m5 remain dedicated scale registers throughout the invocation.
        f"vli.all m4, {(rescale_exponent + 127) << 7}", "delay 66",
        "add x19, x1, x0", "li x20, 0", f"li x21, {m * k // TILE_BYTES}",
        "reuse_a_load:", "dma.load x19, x20, x5, ch0", "dma.wait ch0",
        "addi x19, x19, 256", "addi x20, x20, 1024", "addi x21, x21, -1",
        "beq x21, x0, reuse_a_loaded", "nop", "jal x0, reuse_a_load", "nop",
        "reuse_a_loaded:", "reuse_column:",
        "add x19, x2, x0", "add x20, x15, x0", f"li x21, {k // 32}",
        *load_b_column("reuse_b_load"), "add x15, x20, x0",
        "add x10, x1, x0", "add x11, x10, x16", "add x24, x23, x0",
        f"li x13, {m // 64}", "reuse_row_pair:",
        "add x4, x10, x0", "add x7, x11, x0", "add x6, x2, x0",
    ]
    for tile in range(k // 32):
        op = "vmatmul.mxu0" if tile == 0 else "vmatmul.acc.mxu0"
        lines += [
            "vload m1, 0(x6)", "delay 34",
            "vmatpush.weight.mxu0 w0, m1", "delay 32",
            "vload m0, 0(x4)", "delay 34", f"{op} acc0, m0, w0", "delay 96",
            "vload m0, 0(x7)", "delay 34", f"{op} acc1, m0, w0", "delay 96",
            "addi x4, x4, 256", "addi x7, x7, 256", "addi x6, x6, 256",
        ]
    lines += ["add x12, x24, x0"]
    for acc in (0, 1):
        # Pops are deliberately sequential: m2/m3 are drained to DRAM before
        # reusing them for the other accumulator. m4/m5 never overlap a pop.
        lines += [
            f"vmatpop.bf16.acc.mxu0 m2, acc{acc}", "delay 34",
            "vmul.bf16 m2, m2, m4", "delay 66",
            "vstore m2, 0(x3)", "delay 34", "vstore m3, 0(x9)", "delay 34",
            "dma.store x12, x3, x8, ch0", "dma.wait ch0",
        ]
        if acc == 0:
            lines += ["add x12, x12, x17"]
    lines += [
        "add x10, x10, x18", "add x11, x11, x18", "add x24, x24, x26",
        "addi x13, x13, -1", "beq x13, x0, reuse_rows_done", "nop",
        "jal x0, reuse_row_pair", "nop", "reuse_rows_done:",
        "add x23, x23, x8", "addi x14, x14, -1",
        "beq x14, x0, reuse_finished", "nop", "jal x0, reuse_column", "nop",
        "reuse_finished:", "nop", "ecall",
    ]
    return "\n".join(lines) + "\n"
