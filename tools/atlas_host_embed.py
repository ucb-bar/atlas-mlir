#!/usr/bin/env python3
"""Turn a checked reset-entry capsule into a C header for the host build."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import struct
import string
import sys


def _mapped_regions(memmap: dict) -> list[dict]:
    """Read Chipyard's generated host address map without inferring DMA reachability."""
    rows = memmap.get("mapping")
    if not isinstance(rows, list) or not rows:
        raise ValueError("SoC memmap has no mapping rows")
    result = []
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError("SoC memmap row is not an object")
        values = {}
        for key in ("base", "size", "r", "w", "x", "c"):
            field = row.get(key)
            if not isinstance(field, list) or len(field) != 1:
                raise ValueError(f"SoC memmap needs one {key} value per row")
            values[key] = field[0]
        if (any(type(values[key]) is not int or values[key] < 0
                for key in ("base", "size"))
                or not values["size"]
                or any(type(values[key]) is not bool for key in ("r", "w", "x", "c"))):
            raise ValueError("SoC memmap has an invalid range or permission")
        names = row.get("names", [])
        if not isinstance(names, list) or any(not isinstance(name, str) for name in names):
            raise ValueError("SoC memmap has invalid mapping names")
        values["names"] = names
        result.append(values)
    return result


def _cover_span(rows: list[dict], start: int, size: int, *, name: str,
                access: str = "rw", uncached: bool = False) -> dict:
    if type(start) is not int or type(size) is not int or start < 0 or size <= 0:
        raise ValueError(f"{name} has an invalid address range")
    position = start
    end = start + size
    used = []
    while position < end:
        choices = [row for row in rows
                   if row["base"] <= position < row["base"] + row["size"]
                   and all(row[permission] for permission in access)
                   and (not uncached or not row["c"])]
        if not choices:
            raise ValueError(f"SoC memmap does not cover {name} at 0x{position:x}"
                             " with required host permissions/cacheability")
        chosen = max(choices, key=lambda row: row["base"] + row["size"])
        used.append(chosen)
        position = min(end, chosen["base"] + chosen["size"])
    return {
        "address": start,
        "size_bytes": size,
        "cacheable": any(row["c"] for row in used),
        "mapping_names": [name for row in used for name in row["names"]],
    }


def check_soc_memmap(host_map: dict, manifest: dict, memmap: dict) -> dict:
    """Check host-visible IMEM, CSR, and declared DRAM spans before handoff."""
    rows = _mapped_regions(memmap)
    checked = {
        "imem": _cover_span(rows, host_map["imem_base"],
                            host_map["imem_capacity_bytes"], name="Atlas IMEM", uncached=True),
        "csr": _cover_span(rows, host_map["csr_base"], 0x1c,
                           name="Atlas status/start CSR", uncached=True),
        "dram": [],
    }
    for region in manifest["dram"]["regions"]:
        checked["dram"].append({
            "name": region["name"],
            **_cover_span(rows, region["address"], region["size_bytes"],
                          name=f"DRAM {region['name']}", uncached=False),
        })
    return checked


def check_host_elf(elf: bytes, memmap: dict) -> dict:
    """Check RV64 bare-metal load segments against the generated host map."""
    if (len(elf) < 64 or elf[:6] != b"\x7fELF\x02\x01" or
            elf[6] != 1):
        raise ValueError("host ELF must be ELF64 little-endian")
    e_type, machine = struct.unpack_from("<HH", elf, 16)
    if e_type != 2 or machine != 243:
        raise ValueError("host ELF must be a RISC-V executable")
    entry, phoff = struct.unpack_from("<QQ", elf, 24)
    phentsize, phnum = struct.unpack_from("<HH", elf, 54)
    if phentsize != 56 or not phnum or phoff + phentsize * phnum > len(elf):
        raise ValueError("host ELF has a missing or truncated program header table")
    rows = _mapped_regions(memmap)
    segments = []
    executable_entry = False
    for index in range(phnum):
        kind, flags, offset, virtual, physical, filesz, memsz, alignment = struct.unpack_from(
            "<IIQQQQQQ", elf, phoff + index * phentsize)
        if kind != 1:
            continue
        if (not memsz or filesz > memsz or offset + filesz > len(elf) or
                physical != virtual or physical + memsz > 1 << 64 or
                not flags & 4 or flags & ~7 or
                (alignment > 1 and
                 (alignment & (alignment - 1) or virtual % alignment != offset % alignment))):
            raise ValueError(f"host ELF LOAD segment {index} is invalid for bare-metal launch")
        access = "r" + ("w" if flags & 2 else "") + ("x" if flags & 1 else "")
        checked = _cover_span(rows, physical, memsz,
                              name=f"host ELF LOAD segment {index}", access=access)
        segments.append({"index": index, "flags": flags, **checked})
        if flags & 1 and physical <= entry < physical + memsz:
            executable_entry = True
    ordered = sorted(segments, key=lambda segment: segment["address"])
    if any(left["address"] + left["size_bytes"] > right["address"]
           for left, right in zip(ordered, ordered[1:])):
        raise ValueError("host ELF LOAD segments overlap in physical memory")
    if not segments or not executable_entry:
        raise ValueError("host ELF entry is not in a mapped executable LOAD segment")
    return {"entry": entry, "load_segments": segments}


def check_soc_memmap_hash(host_map: dict, memmap_bytes: bytes) -> None:
    if host_map.get("soc_memmap_sha256") != hashlib.sha256(memmap_bytes).hexdigest():
        raise ValueError("SoC memmap hash differs from bitstream-bound host map")


def render(manifest: dict, program: bytes, host_map: dict) -> str:
    if manifest.get("schema") != "atlas.boot-capsule.v1":
        raise ValueError("unsupported Atlas capsule")
    if host_map.get("schema") != "atlas.host-map.v1":
        raise ValueError("unsupported host map")
    soc_config = host_map.get("soc_config_source")
    soc_revision = host_map.get("soc_source_revision")
    if not isinstance(soc_config, str) or not soc_config:
        raise ValueError("host map requires a SoC configuration source")
    if not isinstance(soc_revision, str) or len(soc_revision) != 40 or any(
            digit not in string.hexdigits for digit in soc_revision):
        raise ValueError("host map requires a pinned SoC source revision")
    if manifest.get("dram", {}).get("selected_rtl_revision") != host_map.get("rtl_revision"):
        raise ValueError("capsule and host map select different RTL")
    if manifest.get("program_sha256") != hashlib.sha256(program).hexdigest():
        raise ValueError("program hash differs from capsule")
    if not program or len(program) % 4 or manifest.get("program_words") != len(program) // 4:
        raise ValueError("program length differs from capsule")
    if manifest.get("entry_pc_word") != 0 or manifest.get("completion", {}).get("kind") != "ecall_halt":
        raise ValueError("host loader requires a reset-entry ECALL program")
    if manifest.get("imem_tl_byte_base") != host_map.get("imem_base") or \
            manifest.get("imem_capacity_bytes") != host_map.get("imem_capacity_bytes") or \
            manifest.get("start_csr_tl_byte_address") != host_map.get("start_csr_offset"):
        raise ValueError("capsule and host map disagree on Atlas MMIO")
    if any(not isinstance(host_map.get(key), int) or host_map[key] < 0
           for key in ("imem_base", "imem_capacity_bytes", "csr_base", "start_csr_offset")):
        raise ValueError("host map addresses must be nonnegative integers")
    if len(program) > host_map["imem_capacity_bytes"]:
        raise ValueError("program exceeds IMEM")
    words = struct.unpack(f"<{len(program) // 4}I", program)
    lines = [
        "/* Generated from a checked Atlas capsule. Do not edit. */",
        "#ifndef ATLAS_PROGRAM_H",
        "#define ATLAS_PROGRAM_H",
        "#include <stdint.h>",
        '#include "atlas_host.h"',
        f'#define ATLAS_PROGRAM_SHA256 "{manifest["program_sha256"]}"',
        f'#define ATLAS_PROGRAM_RTL_REVISION "{host_map["rtl_revision"]}"',
        f'#define ATLAS_PROGRAM_SOC_CONFIG {json.dumps(soc_config)}',
        f'#define ATLAS_PROGRAM_SOC_REVISION "{soc_revision}"',
        f"#define ATLAS_PROGRAM_BITSTREAM_BOUND {int(bool(host_map.get('bitstream_identity')))}",
        f"#define ATLAS_PROGRAM_WORD_COUNT {len(words)}u",
        "static const uint32_t atlas_program_words[ATLAS_PROGRAM_WORD_COUNT] = {",
    ]
    lines.extend(f"    0x{word:08x}u," for word in words)
    lines.extend([
        "};",
        "static const atlas_host_map atlas_program_map = {",
        f"    0x{host_map['imem_base']:x}u, 0x{host_map['csr_base']:x}u,",
        f"    0x{host_map['imem_capacity_bytes']:x}u",
        "};",
        "#endif",
        "",
    ])
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capsule", type=Path, required=True)
    parser.add_argument("--host-map", type=Path, required=True)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--soc-memmap", type=Path,
                        help="generated Chipyard memmap.json from the selected SoC build")
    parser.add_argument("--host-elf", type=Path,
                        help="final RV64 bare-metal ELF to check against the generated map")
    parser.add_argument("--check-only", action="store_true",
                        help="validate the final handoff without writing another header")
    parser.add_argument("--require-bitstream", action="store_true",
                        help="refuse a source-only map without a bitstream identity")
    args = parser.parse_args()
    try:
        if ((args.check_only and args.out is not None) or
                (not args.check_only and args.out is None)):
            raise ValueError("supply --out for header generation or --check-only")
        if args.out is not None and args.out.exists():
            raise ValueError("output header already exists")
        if args.host_elf is not None and args.soc_memmap is None:
            raise ValueError("host ELF check requires the generated SoC memmap")
        manifest = json.loads((args.capsule / "manifest.json").read_text())
        program = (args.capsule / manifest["program_file"]).read_bytes()
        host_map = json.loads(args.host_map.read_text())
        if args.require_bitstream and not host_map.get("bitstream_identity"):
            raise ValueError("host map is not bound to a bitstream")
        if args.require_bitstream and args.soc_memmap is None:
            raise ValueError("bitstream handoff requires the generated SoC memmap")
        header = render(manifest, program, host_map)
        mapped = None
        host_elf = None
        memmap_hash = None
        if args.soc_memmap is not None:
            memmap_bytes = args.soc_memmap.read_bytes()
            memmap_hash = hashlib.sha256(memmap_bytes).hexdigest()
            if args.require_bitstream:
                check_soc_memmap_hash(host_map, memmap_bytes)
            memmap = json.loads(memmap_bytes)
            mapped = check_soc_memmap(host_map, manifest, memmap)
            if args.host_elf is not None:
                elf_bytes = args.host_elf.read_bytes()
                host_elf = check_host_elf(elf_bytes, memmap)
                host_elf["sha256"] = hashlib.sha256(elf_bytes).hexdigest()
        if args.out is not None:
            args.out.parent.mkdir(parents=True, exist_ok=True)
            args.out.write_text(header)
    except (OSError, ValueError, KeyError, TypeError) as error:
        print(json.dumps({"status": "FAIL", "error": str(error)}), file=sys.stderr)
        return 1
    print(json.dumps({"status": "PASS", "out": str(args.out) if args.out else None,
                      "program_words": manifest["program_words"],
                      "bitstream_bound": bool(host_map.get("bitstream_identity")),
                      "soc_memmap_checked": mapped is not None,
                      "soc_memmap_sha256": memmap_hash,
                      "mapped_regions": mapped,
                      "host_elf": host_elf}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
