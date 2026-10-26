"""Two-bank A residency, streamed resident B columns, and paired MXU0 rows.

All addresses below are VMEM words, while DRAM offsets/counts are bytes.
Each accumulator finishes its ordered full K reduction before its sole pop.
The delays are the existing conservative atlas_program.py instruction recipe.
"""

BANK_BYTES = 256 * 1024
TILE_BYTES = 1024


def layout(m, k, n):
    if any(type(d) is not int or d <= 0 or d % 32 for d in (m, k, n)):
        raise ValueError("physical dimensions must be positive multiples of 32")
    a_bytes, b_panel = m * k, k * 32
    a_spill = max(0, a_bytes - BANK_BYTES)
    if a_spill + b_panel + 2048 > BANK_BYTES:
        raise ValueError("resident A, B column and output scratch must fit two 256-KiB banks")
    return dict(a_bank1_bytes=min(a_bytes, BANK_BYTES), a_bank0_bytes=a_spill,
                b_word=a_spill // 4, b_bytes=b_panel,
                output_word=(a_spill + b_panel) // 4)


def dma_load_bytes(m, k, n):
    layout(m, k, n)
    return m * k + k * n


def a_word(byte_offset):
    """Injectively map packed A bytes: first bank 1, then bank 0."""
    return ((BANK_BYTES + byte_offset) if byte_offset < BANK_BYTES
            else (byte_offset - BANK_BYTES)) // 4


def assembly(m, k, n, output_offset, rescale_exponent=None):
    plan = layout(m, k, n)
    if (type(output_offset) is not int or output_offset < m * k + k * n
            or output_offset % 32 or output_offset + 2 * m * n > 2**32):
        raise ValueError("output must be aligned, after inputs, and within the DRAM low word")
    if rescale_exponent is not None and (type(rescale_exponent) is not int
            or not -126 <= rescale_exponent <= 127):
        raise ValueError("VPU rescale factor must be normal finite BF16")
    b, out = plan['b_word'], plan['output_word']
    lines = ["li x5, 1024", "li x8, 2048", f"li x3, {out}", f"li x9, {out + 256}",
             "li x30, 1", "dma.config x30, ch0", f"li x15, {m * k}",
             f"li x12, {output_offset}", f"li x17, {n // 32}", f"li x16, {n * 64}"]
    # One patchable scale immediate preserves the captured host template protocol.
    if rescale_exponent is not None:
        lines += [f"vli.all m4, {(rescale_exponent + 127) << 7}", "delay 66"]
    # Each preload loop remains inside a single physical bank, even at its end.
    for part, (origin, count) in enumerate(((0, plan['a_bank1_bytes']),
                                         (BANK_BYTES, plan['a_bank0_bytes']))):
        if not count:
            continue
        lines += [f"li x19, {a_word(origin)}", f"li x20, {origin}", f"li x21, {count // 1024}",
                  f"preload_a_{part}:", "dma.load x19, x20, x5, ch0", "dma.wait ch0",
                  "addi x19, x19, 256", "addi x20, x20, 1024", "addi x21, x21, -1",
                  f"beq x21, x0, preload_a_{part}_done", "nop", f"jal x0, preload_a_{part}", "nop",
                  f"preload_a_{part}_done:"]
    lines += ["columns_loop:", f"li x19, {b}", "add x20, x15, x0", f"li x21, {k // 32}",
              "preload_b:", "dma.load x19, x20, x5, ch0", "dma.wait ch0",
              "vload m1, 0(x19)", "delay 34", "vtrpose.xlu m1, m1", "delay 66",
              "vstore m1, 0(x19)", "delay 34", "addi x19, x19, 256", "addi x20, x20, 1024",
              "addi x21, x21, -1", "beq x21, x0, preload_b_done", "nop", "jal x0, preload_b", "nop",
              "preload_b_done:", "add x15, x20, x0", "add x22, x12, x0"]
    for row in range(0, m // 32, 2):
        rows = min(2, m // 32 - row)
        lines += [f"li x6, {b}"]
        for inner in range(k // 32):
            lines += ["vload m1, 0(x6)", "delay 34", "vmatpush.weight.mxu0 w0, m1", "delay 32"]
            for acc in range(rows):
                origin = (row + acc) * k * 32 + inner * 1024
                lines += [f"li x1, {a_word(origin)}", "vload m0, 0(x1)", "delay 34",
                          ("vmatmul.mxu0" if inner == 0 else "vmatmul.acc.mxu0") + f" acc{acc}, m0, w0",
                          "delay 96"]
            lines += ["addi x6, x6, 256"]
        for acc in range(rows):
            lines += [f"vmatpop.bf16.acc.mxu0 m2, acc{acc}", "delay 34"]
            if rescale_exponent is not None:
                lines += ["vmul.bf16 m2, m2, m4", "delay 66"]
            lines += ["vstore m2, 0(x3)", "delay 34", "vstore m3, 0(x9)", "delay 34",
                      "dma.store x22, x3, x8, ch0", "dma.wait ch0", "add x22, x22, x16"]
    lines += ["add x12, x12, x8", "addi x17, x17, -1", "beq x17, x0, finished", "nop",
              "jal x0, columns_loop", "nop", "finished:", "nop", "ecall"]
    return "\n".join(lines) + "\n"
