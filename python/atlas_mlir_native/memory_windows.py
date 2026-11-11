"""Complete byte spans admitted by the Atlas DMA target contract."""

DMA_MEMORY_WINDOWS = ((0x80000000, 1 << 32), (0x08000000, 0x08010000))


def dma_span_in_window(address, size):
    """Require a positive span wholly contained in one DMA window."""
    return (type(address) is int and type(size) is int and size > 0
            and any(start <= address < address + size <= end
                    for start, end in DMA_MEMORY_WINDOWS))
