"""Static physical DRAM admission for bare ELF packages."""
import struct
from elf_image import Image


def audit(out, dram_base, dram_bytes):
    """Check p_memsz and every SHF_ALLOC section, including NOLOAD/stack/BSS."""
    raw = (out / "host.elf").read_bytes(); image = Image(raw)
    limit = dram_base + dram_bytes
    if dram_base < 0 or dram_bytes <= 0 or limit > 1 << 64:
        raise ValueError("invalid physical DRAM range")
    spans = []
    for s in image.segments:
        start, size = s["physical"], s["memory_bytes"]
        if (s["address"] != start or not dram_base <= start <= limit - size
                or s["file_bytes"] > size or s["offset"] + s["file_bytes"] > len(raw)):
            raise ValueError("PT_LOAD outside physical DRAM range")
        if size and any(start < end and begin < start + size for begin, end in spans):
            raise ValueError("overlapping physical PT_LOAD ranges")
        if size: spans.append((start, start + size))
    shoff = struct.unpack_from("<Q", raw, 40)[0]
    shsize, count = struct.unpack_from("<HH", raw, 58)
    allocated = 0
    for i in range(count):
        s = struct.unpack_from("<IIQQQQIIQQ", raw, shoff + i * shsize)
        if s[2] & 2 and s[5]:
            start, size = s[3], s[5]; allocated += 1
            if (not dram_base <= start <= limit - size
                    or not any(begin <= start and start + size <= end for begin, end in spans)):
                raise ValueError("allocated section outside physical DRAM/PT_LOAD range")
    # HTIF CRT reserves heap and one boot-hart stack only through symbols,
    # outside PT_LOAD p_memsz. The freestanding layer/block use alloc sections.
    symbols = {name: row["address"] for name, row in image.symbols.items()}
    reserved = []
    if "__stack_start" in symbols:
        start = symbols["__stack_start"]
        reserved.append(("stack", start, start + symbols["__stack_size"]))
    if "__heap_end" in symbols:
        reserved.append(("heap", symbols["_end"], symbols["__heap_end"]))
    for name, start, end in reserved:
        if not dram_base <= start <= end <= limit:
            raise ValueError(name + " outside physical DRAM range")
        if any(start < hi and lo < end for lo, hi in spans):
            raise ValueError(name + " overlaps physical PT_LOAD")
    if not any(begin <= image.entry < end for begin, end in spans):
        raise ValueError("entry outside physical DRAM")
    return dict(dram_base=dram_base, dram_bytes=dram_bytes, segments=image.segments, allocated_sections=allocated,
        reserved=reserved, boot_harts=1)
