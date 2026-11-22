"""Complete byte spans admitted by the Atlas DMA target contract."""

from .soc import SOC


def dma_memory_windows(soc=None):
    soc = SOC if soc is None else soc
    return tuple((region.base, region.end)
                 for region in (soc.memory_region("dma"), soc.memory_region("host_scratch"),
                                soc.memory_region("atlas_data")))


def dma_span_in_window(address, size, *, soc=None):
    """Require a positive span wholly contained in one DMA window."""
    return (type(address) is int and type(size) is int and size > 0
            and any(start <= address < address + size <= end
                    for start, end in dma_memory_windows(soc)))
