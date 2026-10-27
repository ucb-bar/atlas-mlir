"""Opt-in Tiny 128x2048xN compulsory-DMA reuse recipe.

VMEM operands are four-byte words, while DRAM operands are bytes. A occupies
bank 1 exactly; bank 0 holds a 64-KiB transposed B column panel and 2-KiB
output scratch. MXU0 has two accumulators, so four rows run as two row pairs.
All spacing is the conservative established atlas_program schedule: load /
store / pop 34, transpose / BF16 arithmetic 66, weight push 32, compute 96.
"""

from atlas_program_reuse_common import load_b_column, store_output


SHAPES = ((128, 2048, 2048), (128, 2048, 256), (128, 2048, 1280), (128, 2048, 1536))

def assembly(m, k, n, output_offset, rescale_exponent=None):
    if any(type(v) is not int for v in (m, k, n)) or (m, k, n) not in SHAPES:
        raise ValueError("Tiny reuse requires a captured physical shape")
    if type(output_offset) is not int or output_offset != m * k + k * n + 256:
        raise ValueError("output must preserve the original input and guard layout")
    if rescale_exponent is not None and (type(rescale_exponent) is not int or not -126 <= rescale_exponent <= 127):
        raise ValueError("VPU rescale factor must be normal finite integer BF16")
    lines = [
        "li x1, 0x10000", "li x3, 0x4000", "li x9, 0x4100",
        "li x5, 1024", "li x8, 2048", f"li x12, {output_offset}",
        f"li x14, {n // 32}", f"li x22, {64 * n}", "li x30, 1", "dma.config x30, ch0", "dma.config x30, ch1",
        "add x19, x1, x0", "li x20, 0", "li x21, 256", "resident_a_loop:",
        "dma.load x19, x20, x5, ch0", "dma.wait ch0",
        "addi x19, x19, 256", "addi x20, x20, 1024", "addi x21, x21, -1",
        "beq x21, x0, resident_a_done", "nop", "jal x0, resident_a_loop", "nop",
        "resident_a_done:", "columns_loop:", "li x19, 0", "li x21, 64",
        *load_b_column("column_load_loop"), "add x23, x12, x0",
    ]
    # One static VLI preserves the existing single-word host rescale patch.
    # m4 remains live until all four outputs in the column are scaled.
    if rescale_exponent is not None:
        lines += [f"vli.all m4, {(rescale_exponent + 127) << 7}", "delay 66"]
    for pair in range(2):
        lines += [f"li x4, {0x10000 + pair * 32768}", f"li x7, {0x10000 + pair * 32768 + 16384}", "li x6, 0"]
        for tile in range(k // 32):
            compute = "vmatmul.mxu0" if tile == 0 else "vmatmul.acc.mxu0"
            lines += [
                "vload m1, 0(x6)", "delay 34", "vmatpush.weight.mxu0 w0, m1", "delay 32",
                "vload m0, 0(x4)", "delay 34", f"{compute} acc0, m0, w0", "delay 96",
                "vload m0, 0(x7)", "delay 34", f"{compute} acc1, m0, w0", "delay 96",
                "addi x4, x4, 256", "addi x7, x7, 256", "addi x6, x6, 256",
            ]
        for acc in range(2):
            lines += store_output(acc, 23, rescale_exponent) + ["add x23, x23, x22"]
    lines += ["add x12, x12, x8", "addi x14, x14, -1", "beq x14, x0, finished", "nop",
              "jal x0, columns_loop", "nop", "finished:", "nop", "ecall"]
    return "\n".join(lines) + "\n"


def dma_load_bytes(m, k, n):
    if any(type(v) is not int for v in (m, k, n)) or (m, k, n) not in SHAPES:
        raise ValueError("Tiny reuse requires a captured physical shape")
    return m * k + k * n
