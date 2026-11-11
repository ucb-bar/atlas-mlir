"""Complete byte spans admitted by the Atlas DMA target contract."""

# DRAM, the EE290 scratchpad, and Atlas VMEM banks 3-5 on the system bus.
# A program using the VMEM window keeps its own VMEM use below PROGRAM_VMEM_BYTES.
VMEM_BUS_BASE = 0x20000000
VMEM_BYTES = 1536 * 1024
PROGRAM_VMEM_BYTES = 768 * 1024
VMEM_BUS_WINDOW = (VMEM_BUS_BASE + PROGRAM_VMEM_BYTES, VMEM_BUS_BASE + VMEM_BYTES)
DMA_MEMORY_WINDOWS = ((0x80000000, 1 << 32), (0x08000000, 0x08010000), VMEM_BUS_WINDOW)


def dma_span_in_window(address, size):
    """Require a positive span wholly contained in one DMA window."""
    return (type(address) is int and type(size) is int and size > 0
            and any(start <= address < address + size <= end
                    for start, end in DMA_MEMORY_WINDOWS))
