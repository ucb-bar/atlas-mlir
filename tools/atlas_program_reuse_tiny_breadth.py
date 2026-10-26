"""Opt-in ordered MXU0 row residency with four-tile B prefetch.

Two rows share each B weight for the admitted M128 shape; each admitted M32
shape uses one row. SHAPES below defines the exact coverage. The 4096-byte B
buffers alternate, and each incoming DMA is
waited before its first use. All matrix/LSU spacing is the established
atlas_program spacing; no numerical reassociation or partial pop occurs.
VMEM addresses in the assembly are four-byte words, DRAM offsets bytes.
"""
SHAPES = ((128, 5632, 1024), (32, 2048, 2048), (32, 2048, 1024),
          (32, 5632, 1024), (32, 2048, 1536), (32, 2048, 256),
          (32, 2048, 1280))


def allocation(m, k, n):
    if any(type(v) is not int for v in (m, k, n)) or (m, k, n) not in SHAPES:
        raise ValueError("Prefetch reuse requires a captured physical shape")
    rows = min(m // 32, 2)
    a = [(0 if rows == 2 else 262144, 32 * k)]
    if rows == 2:
        a.append((262144, 32 * k))
    b = 32 * k if rows == 2 else 0
    return dict(a=a, b_buffers=[(b, 4096), (b + 4096, 4096)],
                output=(b + 8192, 2048), rows=rows,
                dma_load_bytes=m * k + (m // (32 * rows)) * k * n)


def dma_load_bytes(m, k, n):
    return allocation(m, k, n)['dma_load_bytes']


def assembly(m, k, n, output_offset, rescale_exponent=None):
    spec = allocation(m, k, n)
    if type(output_offset) is not int or output_offset != m * k + k * n + 256:
        raise ValueError("Tiny breadth requires the original guarded output offset")
    if rescale_exponent is not None and (type(rescale_exponent) is not int or
                                         not -126 <= rescale_exponent <= 127):
        raise ValueError("VPU rescale factor must be normal finite integer BF16")
    rows = spec['rows']
    b0, b1 = (p[0] // 4 for p in spec['b_buffers'])
    out = spec['output'][0] // 4
    lines = ["li x30, 1", "dma.config x30, ch0", "dma.config x30, ch1",
             "li x5, 4096", "li x8, 2048", f"li x3, {out}",
             f"li x9, {out + 256}", f"li x12, {output_offset}",
             f"li x22, {64 * n}", "li x29, 1", "li x20, 0",
             f"li x28, {m // (32 * rows)}"]
    if rescale_exponent is not None:
        lines += [f"vli.all m4, {(rescale_exponent + 127) << 7}", "delay 66"]
    lines += ["pairs:"]
    for row, (base, size) in enumerate(spec['a']):
        label = f'a_load_{row}'
        lines += [f"li x19, {base // 4}", f"li x21, {size // 4096}", label + ':',
                  "dma.load x19, x20, x5, ch0", "dma.wait ch0",
                  "addi x19, x19, 1024", "add x20, x20, x5", "addi x21, x21, -1",
                  f"beq x21, x0, {label}_done", "nop", f"jal x0, {label}", "nop", label + '_done:']
    lines += ["add x27, x20, x0", f"li x20, {m * k}", f"li x14, {n // 32}", "columns:",
              f"li x4, {spec['a'][0][0] // 4}"]
    if rows == 2:
        lines += [f"li x7, {spec['a'][1][0] // 4}"]
    lines += [f"li x16, {b0}", f"li x17, {b1}", f"li x24, {k // 128}", "li x25, 0",
              "dma.load x16, x20, x5, ch0", "dma.wait ch0", "add x20, x20, x5",
              "blocks:", "beq x24, x29, no_prefetch", "nop",
              "dma.load x17, x20, x5, ch1", "add x20, x20, x5", "no_prefetch:",
              "add x6, x16, x0", "li x21, 4", "tiles:",
              "vload m1, 0(x6)", "delay 34", "vtrpose.xlu m1, m1", "delay 66",
              "vmatpush.weight.mxu0 w0, m1", "delay 32",
              "beq x25, x0, first_tile", "nop"]
    for opcode, label in [('vmatmul.acc.mxu0', 'accumulate'), ('vmatmul.mxu0', 'first_tile')]:
        if label == 'first_tile':
            lines += [label + ':']
        for row in range(rows):
            lines += [f"vload m0, 0(x{4 if row == 0 else 7})", "delay 34",
                      f"{opcode} acc{row}, m0, w0", "delay 96"]
        if label == 'accumulate':
            lines += ["jal x0, tile_done", "nop"]
    lines += ["tile_done:", "addi x25, x25, 1", "addi x4, x4, 256"]
    if rows == 2:
        lines += ["addi x7, x7, 256"]
    lines += ["addi x6, x6, 256", "addi x21, x21, -1", "beq x21, x0, block_done", "nop",
              "jal x0, tiles", "nop", "block_done:", "beq x24, x29, column_done", "nop",
              "dma.wait ch1", "add x18, x16, x0", "add x16, x17, x0", "add x17, x18, x0",
              "addi x24, x24, -1", "jal x0, blocks", "nop", "column_done:", "add x23, x12, x0"]
    for row in range(rows):
        lines += [f"vmatpop.bf16.acc.mxu0 m2, acc{row}", "delay 34"]
        if rescale_exponent is not None:
            lines += ["vmul.bf16 m2, m2, m4", "delay 66"]
        lines += ["vstore m2, 0(x3)", "delay 34", "vstore m3, 0(x9)", "delay 34",
                  "dma.store x23, x3, x8, ch0", "dma.wait ch0", "add x23, x23, x22"]
    lines += ["add x12, x12, x8", "addi x14, x14, -1", "beq x14, x0, pair_done", "nop",
              "jal x0, columns", "nop", "pair_done:"]
    if rows == 2:
        lines += ["add x12, x12, x22"]
    lines += ["add x20, x27, x0", "addi x28, x28, -1", "beq x28, x0, finished", "nop",
              "jal x0, pairs", "nop", "finished:", "nop", "ecall"]
    return '\n'.join(lines) + '\n'
