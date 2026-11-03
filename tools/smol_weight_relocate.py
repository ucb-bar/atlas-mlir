"""Build-time DMA relocation and physical DRAM bounds for bare ELF packages."""
import re

DEFAULT_ARENA = 0x100000000


def _dram_register(load, rd, rs1, rs2):
    reg, others = (rs1, (rd, rs2)) if load else (rd, (rs1, rs2))
    if reg == 0 or reg in others:
        raise ValueError("DMA relocation requires a writable, distinct DRAM register")
    return reg


def relocate_assembly(source, old_base, new_base, *, force=False, weight_source=None):
    """Keep local offsets/VMEM operands, change high32 and displace DMA bytes."""
    if old_base == new_base and not force:
        return source
    if old_base != DEFAULT_ARENA or re.search(r"\bx31\b", source):
        raise ValueError("relocation requires the retained base and unused x31")
    if weight_source is not None and weight_source.arena_base != new_base:
        raise ValueError("immutable weight arena binding mismatch")
    setup = re.compile(r"^li x30, 1$", re.MULTILINE)
    if len(setup.findall(source)) != 1:
        raise ValueError("expected one retained DMA high-half setup")
    source = setup.sub(f"li x30, {new_base >> 32}", source)
    lines = [f"li x31, {new_base & 0xffffffff}"]
    for line in source.splitlines():
        m = re.fullmatch(r"dma\.(load|store) x(\d+), x(\d+), x(\d+), ch[0-7]", line)
        if re.search(r"\bdma\.(load|store)\b", line) and m is None:
            raise ValueError("unsupported DMA transfer syntax")
        if weight_source is not None and re.fullmatch(r"dma\.(?:load|wait).*ch[23]", line):
            if line.startswith("dma.load") and m is None:
                raise ValueError("unsupported immutable DMA transfer")
            lines.append(line)
            continue
        if weight_source is not None and re.fullmatch(r"dma.config x(?:6|15|20), ch[23]", line):
            lines.append(line)
            continue
        if re.search(r"\bdma\.config\b", line) and not re.fullmatch(r"dma\.config x30, ch[0-7]", line):
            raise ValueError("unexpected DMA configuration source or syntax")
        reg = _dram_register(m[1] == "load", *map(int, m.group(2, 3, 4))) if m else None
        lines.extend([f"add x{reg}, x{reg}, x31", line, f"sub x{reg}, x{reg}, x31"] if m else [line])
    return "\n".join(lines) + "\n"
