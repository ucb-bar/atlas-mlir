"""Shared resident-column loading and ordered BF16 output emission."""

def load_b_column(label, channels=(0, 1)):
    """Load/transpose a resident column, prefetching only its next disjoint tile.

    x19/x20/x21 hold VMEM words, DRAM bytes and remaining tiles; x24 is scratch.
    ch0 supplies the first tile, ch1 the following tiles. Each DMA is waited
    before use, and the final tile issues no speculative request.
    """
    transpose = ['vload m1, 0(x19)', 'delay 34', 'vtrpose.xlu m1, m1', 'delay 66',
                 'vstore m1, 0(x19)', 'delay 34']
    first, prefetch = channels
    return [f'dma.load x19, x20, x5, ch{first}', f'dma.wait ch{first}', label + ':',
            'addi x24, x19, 256', 'addi x20, x20, 1024', 'addi x21, x21, -1',
            f'beq x21, x0, {label}_last', 'nop', f'dma.load x24, x20, x5, ch{prefetch}',
            *transpose, f'dma.wait ch{prefetch}', 'add x19, x24, x0', f'jal x0, {label}', 'nop',
            label + '_last:', *transpose, label + '_done:']



def store_output(acc, output_reg, rescale_exponent, epilogue=None, *, initialize_rescale=False):
    """Pop once, optionally rescale, then drain both output halves to DRAM."""
    lines = [f"vmatpop.bf16.acc.mxu0 m2, acc{acc}", "delay 34"]
    if rescale_exponent is not None:
        if initialize_rescale:
            lines += [f"vli.all m4, {(rescale_exponent + 127) << 7}", "delay 66"]
        lines += ["vmul.bf16 m2, m2, m4", "delay 66"]
    if epilogue is not None:
        return lines + epilogue.apply(output_reg) + epilogue.store(output_reg)
    return lines + ["vstore m2, 0(x3)", "delay 34", "vstore m3, 0(x9)", "delay 34",
                    f"dma.store x{output_reg}, x3, x8, ch0", "dma.wait ch0"]
