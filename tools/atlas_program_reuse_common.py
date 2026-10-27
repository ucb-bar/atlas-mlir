"""Shared resident-column loading and ordered BF16 output emission."""

def load_b_column(label):
    """Load/transpose a resident column, prefetching only its next disjoint tile.

    x19/x20/x21 hold VMEM words, DRAM bytes and remaining tiles; x24 is scratch.
    ch0 supplies the first tile, ch1 the following tiles. Each DMA is waited
    before use, and the final tile issues no speculative request.
    """
    transpose = ['vload m1, 0(x19)', 'delay 34', 'vtrpose.xlu m1, m1', 'delay 66',
                 'vstore m1, 0(x19)', 'delay 34']
    return ['dma.load x19, x20, x5, ch0', 'dma.wait ch0', label + ':',
            'addi x24, x19, 256', 'addi x20, x20, 1024', 'addi x21, x21, -1',
            f'beq x21, x0, {label}_last', 'nop', 'dma.load x24, x20, x5, ch1',
            *transpose, 'dma.wait ch1', 'add x19, x24, x0', f'jal x0, {label}', 'nop',
            label + '_last:', *transpose, label + '_done:']



def store_output(acc, output_reg, rescale_exponent):
    """Pop once, optionally rescale, then drain both output halves to DRAM."""
    lines = [f"vmatpop.bf16.acc.mxu0 m2, acc{acc}", "delay 34"]
    if rescale_exponent is not None:
        lines += ["vmul.bf16 m2, m2, m4", "delay 66"]
    return lines + ["vstore m2, 0(x3)", "delay 34", "vstore m3, 0(x9)", "delay 34",
                    f"dma.store x{output_reg}, x3, x8, ch0", "dma.wait ch0"]
