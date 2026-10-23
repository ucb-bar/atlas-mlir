#!/usr/bin/env python3
"""Package an unchanged bare-metal model ELF for existing TSI loaders."""

import argparse
import hashlib
import json
from pathlib import Path
import re

from atlas_host_embed import _cover_span, _mapped_regions, check_host_elf
from atlas_path_guard import checked
from host_iss.elf_image import Image

IMEM_BASE = 0x20000
IMEM_BYTES = 0x20000
CSR_BASE = 0x40000


def physical(value):
    path = Path(value)
    if any(re.search(r"hammer|vlsi", part, re.I) for part in path.parts):
        raise ValueError("restricted path")
    return checked(path)


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def overlaps(left, left_size, right, right_size):
    return left < right + right_size and right < left + left_size


def describe(raw, dma_base, dma_bytes, *, memmap=None, observe=()):
    if (type(dma_base) is not int or type(dma_bytes) is not int
            or dma_base < 0 or dma_bytes <= 0 or dma_base + dma_bytes > 1 << 64
            or dma_base % 32 or dma_bytes % 32
            or (dma_base & 0xffffffff) + dma_bytes > 1 << 32):
        raise ValueError("DMA arena must be aligned and fit one 32-bit DMA window")
    # The broad row permits structural ELF inspection without inventing a target map.
    structural = {"mapping": [{"base": [0], "size": [1 << 64], "r": [True],
                               "w": [True], "x": [True], "c": [True]}]}
    footprint = check_host_elf(raw, structural if memmap is None else memmap)
    image = Image(raw)
    if any(segment["flags"] & 1
           and overlaps(segment["physical"], segment["memory_bytes"], dma_base, dma_bytes)
           for segment in image.segments):
        raise ValueError("DMA arena overlaps executable host memory")
    for base, size in ((IMEM_BASE, IMEM_BYTES), (CSR_BASE, 0x1c)):
        if overlaps(dma_base, dma_bytes, base, size) or any(
                overlaps(segment["physical"], segment["memory_bytes"], base, size)
                for segment in image.segments):
            raise ValueError("host or DMA memory overlaps Atlas MMIO")
    symbols = {}
    for name in dict.fromkeys(("tohost", "fromhost", *observe)):
        symbol = image.symbols.get(name)
        if symbol is None or not symbol["size"]:
            raise ValueError("missing sized completion/observation symbol: " + name)
        address, size = symbol["address"], symbol["size"]
        if name in ("tohost", "fromhost") and (size != 8 or address % 8):
            raise ValueError(name + " must be an aligned 64-bit HTIF word")
        if overlaps(address, size, dma_base, dma_bytes):
            raise ValueError("DMA arena overlaps completion/observation symbol: " + name)
        if not any(segment["flags"] & 2
                   and segment["physical"] <= address
                   and address + size <= segment["physical"] + segment["memory_bytes"]
                   for segment in image.segments):
            raise ValueError("completion/observation symbol is not in writable loaded RAM")
        symbols[name] = {"address": address, "size_bytes": size}
    if overlaps(symbols["tohost"]["address"], 8, symbols["fromhost"]["address"], 8):
        raise ValueError("HTIF words overlap")
    loads = []
    for segment in image.segments:
        loads.append({**segment, "zero_fill_address": segment["physical"] + segment["file_bytes"],
                      "zero_fill_bytes": segment["memory_bytes"] - segment["file_bytes"]})
    ranges = sorted((segment["physical"], segment["physical"] + segment["memory_bytes"])
                    for segment in image.segments)
    ranges.append((dma_base, dma_base + dma_bytes))
    merged = []
    for start, end in sorted(ranges):
        if merged and start <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    result = {"entry": footprint["entry"], "elf_file_bytes": len(raw),
              "load_segments": loads, "dma": {"address": dma_base, "size_bytes": dma_bytes},
              "resident_union_bytes": sum(end - start for start, end in merged),
              "resident_ranges": merged, "symbols": symbols,
              "memory_map_checked": memmap is not None}
    if memmap is not None:
        rows = _mapped_regions(memmap)
        result["target_mapping"] = {
            "imem": _cover_span(rows, IMEM_BASE, IMEM_BYTES, name="Atlas IMEM", uncached=True),
            "csr": _cover_span(rows, CSR_BASE, 0x1c, name="Atlas CSR", uncached=True),
            "dma": _cover_span(rows, dma_base, dma_bytes, name="Atlas DMA arena"),
        }
    return result


def package(*, elf, name, dma_base, dma_bytes, out, soc_memmap=None,
            target_identity=None, observe=()):
    if (not re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9_-]*", name)
            or re.search(r"hammer|vlsi", name, re.I)):
        raise ValueError("invalid workload name")
    if bool(soc_memmap) != bool(target_identity):
        raise ValueError("supply both the target memory map and target identity")
    source, destination = physical(elf), physical(out)
    map_path = physical(soc_memmap) if soc_memmap else None
    raw = source.read_bytes()
    map_raw = map_path.read_bytes() if map_path else None
    footprint = describe(raw, dma_base, dma_bytes,
                         memmap=json.loads(map_raw) if map_raw is not None else None,
                         observe=observe)
    manifest = {"schema": "atlas.model-handoff.v1", "name": name,
                "status": "PREPARATION_ONLY", "numerical_evidence": None,
                "elf_sha256": digest(raw), "target_identity": target_identity,
                "soc_memmap_sha256": digest(map_raw) if map_raw is not None else None,
                "hardware_execution": False, "cache_coherence_qualified": False,
                "completion": {"protocol": "HTIF", "success_tohost": 1,
                               "failure_tohost": "(exit_code << 1) | 1"},
                "footprint": footprint}
    workload = {"benchmark_name": name, "common_bootbinary": "host.elf",
                "common_rootfs": None, "common_outputs": [],
                "common_simulation_outputs": ["uartlog"],
                "workloads": [{"name": name}]}
    # Exclusive creation preserves any existing handoff and its evidence.
    destination.mkdir(parents=True, exist_ok=False)
    (destination / "host.elf").write_bytes(raw)
    (destination / "handoff.json").write_text(json.dumps(manifest, indent=2) + "\n")
    (destination / "workload.json").write_text(json.dumps(workload, indent=2) + "\n")
    if map_raw is not None:
        (destination / "soc-memmap.json").write_bytes(map_raw)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--elf", required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--dma-base", type=lambda value: int(value, 0), default=0x100000000)
    parser.add_argument("--dma-bytes", type=lambda value: int(value, 0), required=True)
    parser.add_argument("--observe-symbol", action="append", default=[])
    parser.add_argument("--soc-memmap")
    parser.add_argument("--target-identity")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    package(elf=args.elf, name=args.name, dma_base=args.dma_base, dma_bytes=args.dma_bytes,
            out=args.out, soc_memmap=args.soc_memmap, target_identity=args.target_identity,
            observe=args.observe_symbol)


if __name__ == "__main__":
    main()
