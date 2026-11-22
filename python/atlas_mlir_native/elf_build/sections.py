"""Measure allocated ELF64 RISC-V sections without a linker estimate."""
import hashlib
from pathlib import Path
import struct


def allowed(path):
    path = Path(path)
    # Refuse links before following them, including links in parent components.
    for component in reversed((path.absolute(), *path.absolute().parents)):
        if component.is_symlink():
            raise ValueError("symlink paths are not accepted")
    return path


def measure(path):
    path = allowed(path)
    data = path.read_bytes()
    if len(data) < 64 or data[:7] != b"\x7fELF\x02\x01\x01":
        raise ValueError("expected ELF64 little-endian version 1")
    kind, machine, version = struct.unpack_from("<HHI", data, 16)
    if machine != 243 or version != 1 or kind not in (1, 2):
        raise ValueError("expected RISC-V relocatable or executable ELF")
    shoff = struct.unpack_from("<Q", data, 40)[0]
    ehsize, _, _, shsize, count, strings_index = struct.unpack_from("<6H", data, 52)
    if (ehsize != 64 or shsize != 64 or not count or strings_index >= count
            or shoff < 64 or shoff + count * shsize > len(data)):
        raise ValueError("invalid or unsupported section table")
    headers = [struct.unpack_from("<IIQQQQIIQQ", data, shoff + i * shsize)
               for i in range(count)]
    for _, typ, _, _, offset, size, *_ in headers:
        if typ != 8 and (offset > len(data) or size > len(data) - offset):
            raise ValueError("truncated section")
    strings = headers[strings_index]
    if strings[1] != 3:
        raise ValueError("invalid section name table")
    names = data[strings[4]:strings[4] + strings[5]]
    result = []
    for no, typ, flags, address, _, size, _, _, alignment, _ in headers:
        if no >= len(names) or names.find(b"\0", no) < 0:
            raise ValueError("invalid section name")
        if alignment and alignment & (alignment - 1):
            raise ValueError("invalid section alignment")
        if flags & 2:
            result.append(dict(name=names[no:names.index(b"\0", no)].decode(),
                               type=typ, flags=flags, size=size,
                               alignment=alignment, address=address))
    return dict(path=str(path), sha256=hashlib.sha256(data).hexdigest(),
                elf_type=kind, sections=result,
                allocated_bytes=sum(s["size"] for s in result))
