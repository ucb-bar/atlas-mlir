#!/usr/bin/env python3
"""Check the graph program against a selected SoC map before host handoff.

The generated map object binds host MMIO addresses. These checks establish
host address coverage; the platform still has to establish Atlas DMA access,
cache maintenance, and execution on its bitstream.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from itertools import pairwise
from pathlib import Path

from atlas_host_embed import check_host_elf, check_soc_memmap, check_soc_memmap_hash


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _read_object(path: Path) -> dict:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise TypeError(f"{path} must contain a JSON object")
    return value


def _regions(plan: dict) -> list[dict]:
    addresses = plan.get("input_addresses")
    sizes = plan.get("input_payload_bytes", plan.get("input_buffer_bytes"))
    outputs = plan.get("output_addresses")
    output_bytes = plan.get("output_bytes")
    if (not isinstance(addresses, list) or not addresses or
            not isinstance(sizes, list) or len(addresses) != len(sizes) or
            not isinstance(outputs, list) or not outputs or
            type(output_bytes) is not int or output_bytes <= 0):
        raise ValueError("native graph plan has invalid physical buffer lists")
    regions = []
    for kind, pairs in (("input", zip(addresses, sizes, strict=True)),
                        ("output", ((address, output_bytes) for address in outputs))):
        for index, (address, size) in enumerate(pairs):
            if (type(address) is not int or type(size) is not int or
                    address <= 0 or size <= 0 or address + size > 1 << 32):
                raise ValueError(f"{kind} {index} has an invalid DMA address or size")
            regions.append({"name": f"{kind}_{index}", "address": address,
                            "size_bytes": size})
    ordered = sorted(regions, key=lambda row: row["address"])
    if any(left["address"] + left["size_bytes"] > right["address"]
           for left, right in pairwise(ordered)):
        raise ValueError("graph physical input/output buffers overlap")
    return regions


def _compilation_images(compiled):
    if (compiled / "abi.json").is_file():
        abi = _read_object(compiled / "abi.json")
        if (abi.get("schema") != "merlin.c_handoff.v1" or not isinstance(abi.get("files"), dict)
                or not isinstance(abi.get("programs"), list) or not abi["programs"]):
            raise ValueError("unsupported C handoff ABI")
        for name, digest in abi["files"].items():
            path = Path(name)
            if path.is_absolute() or ".." in path.parts or (compiled/path).is_symlink() or _sha((compiled/path).read_bytes()) != digest:
                raise ValueError("C handoff member identity differs")
        images = []
        for image in abi["programs"]:
            path = image["path"]
            if abi["files"].get(path) != image["sha256"]:
                raise ValueError("C handoff image identity differs")
            program = (compiled/path).read_bytes()
            if not program or len(program) % 4 or len(program) != image["bytes"] or not isinstance(image["execution"], dict):
                raise ValueError("C handoff image has no complete execution ABI")
            images.append((program, image["execution"]))
        return abi, images
    manifest = _read_object(compiled / "manifest.json")
    plan_bytes = (compiled / "execution_plan.json").read_bytes()
    plan = json.loads(plan_bytes)
    program = (compiled / "program.bin").read_bytes()
    if (manifest.get("schema") != "atlas.native_tile_compilation.v6" or
            manifest.get("engine") != "merlin_native" or
            not isinstance(plan, dict) or
            plan.get("schema") != "atlas.native_tile_execution_plan.v8" or
            manifest.get("execution_plan_sha256") != _sha(plan_bytes) or
            manifest.get("binary_sha256") != _sha(program) or
            not program or len(program) % 4):
        raise ValueError("native graph manifest, plan, or program identity differs")
    return manifest, [(program, plan)]


def check(compiled: Path, host_map_path: Path, memmap_path: Path,
          host_elf: Path | None = None) -> tuple[dict, dict]:
    manifest, images = _compilation_images(compiled)
    program = max((binary for binary, _ in images), key=len)
    host_map = _read_object(host_map_path)
    memmap_bytes = memmap_path.read_bytes()
    if (host_map.get("schema") != "atlas.host-map.v1" or
            not isinstance(host_map.get("bitstream_identity"), str) or
            not host_map["bitstream_identity"] or
            not isinstance(host_map.get("soc_config_source"), str) or
            not host_map["soc_config_source"] or
            not isinstance(host_map.get("soc_source_revision"), str) or
            re.fullmatch(r"[0-9a-fA-F]{40}", host_map["soc_source_revision"]) is None or
            not isinstance(host_map.get("rtl_revision"), str) or
            not isinstance(manifest.get("target_identity"), str) or
            manifest["target_identity"].split(":", 1)[0] !=
            "atlas-rtl-" + host_map["rtl_revision"]):
        raise ValueError("selected host map is unbound or selects different Atlas sources")
    for key in ("imem_base", "imem_capacity_bytes", "csr_base"):
        value = host_map.get(key)
        if type(value) is not int or value < 0 or value > 0xffffffff or value & 3:
            raise ValueError(f"host map has an invalid {key}")
    if (host_map.get("start_csr_offset") != 0x18 or
            host_map["imem_capacity_bytes"] < len(program) or
            host_map["imem_base"] + host_map["imem_capacity_bytes"] > 1 << 32 or
            host_map["csr_base"] + 0x1c > 1 << 32):
        raise ValueError("host map disagrees with the compiled image or host CSR layout")
    check_soc_memmap_hash(host_map, memmap_bytes)
    memmap = json.loads(memmap_bytes)
    image_regions = [_regions(plan) for _, plan in images]
    regions = [row for group in image_regions for row in group]
    for constant in manifest.get("constant_inputs", []):
        path = constant.get("path")
        if (not isinstance(path, str) or Path(path).is_absolute() or
                ".." in Path(path).parts or
                not any(row["address"] == constant.get("address") and
                        row["size_bytes"] == constant.get("bytes")
                        for row in regions) or
                _sha((compiled / path).read_bytes()) != constant.get("sha256")):
            raise ValueError("native constant placement or identity differs")
    mappings = [check_soc_memmap(host_map, {"dram": {"regions": group}}, memmap) for group in image_regions]
    mapped = mappings[0] if len(mappings) == 1 else {"programs": mappings}
    checked_elf = None
    if host_elf is not None:
        elf = host_elf.read_bytes()
        checked_elf = check_host_elf(elf, memmap)
        checked_elf["sha256"] = _sha(elf)
        # Host code, data, and BSS must not share bytes with the Atlas program
        # window, its CSRs, or any DRAM buffer the device reads or writes.
        reserved = [(row["name"], row["address"], row["size_bytes"]) for row in regions]
        reserved += [("IMEM", host_map["imem_base"], host_map["imem_capacity_bytes"]),
                     ("CSR", host_map["csr_base"], 0x1c)]
        for segment in checked_elf["load_segments"]:
            start, end = segment["address"], segment["address"] + segment["size_bytes"]
            for name, address, size in reserved:
                if start < address + size and address < end:
                    raise ValueError(f"host ELF LOAD segment {segment['index']} "
                                     f"overlaps Atlas {name}")
    result = {
        "schema": "atlas.graph_platform_check.v1",
        "status": "platform_elf_checked" if checked_elf else "platform_inputs_checked",
        "scope": "host-visible map and ELF; Atlas DMA/cache/FPGA execution unqualified",
        "bitstream_identity": host_map["bitstream_identity"],
        "soc_config_source": host_map["soc_config_source"],
        "soc_source_revision": host_map["soc_source_revision"],
        "soc_memmap_sha256": _sha(memmap_bytes),
        "host_map_sha256": _sha(host_map_path.read_bytes()),
        "program_sha256": _sha(program) if len(images) == 1 else None,
        "program_sha256s": [_sha(binary) for binary, _ in images],
        "mapped_regions": mapped,
        "host_elf": checked_elf,
    }
    return result, host_map


def render_map_c(host_map: dict) -> str:
    return ("/* Generated from a checked SoC map. Do not edit. */\n"
            '#include "atlas_host.h"\n'
            "const atlas_host_map atlas_selected_host_map = {\n"
            f"  0x{host_map['imem_base']:x}u, 0x{host_map['csr_base']:x}u, "
            f"0x{host_map['imem_capacity_bytes']:x}u\n"
            "};\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--compiled", required=True, type=Path)
    parser.add_argument("--host-map", required=True, type=Path)
    parser.add_argument("--soc-memmap", required=True, type=Path)
    parser.add_argument("--host-elf", type=Path)
    parser.add_argument("--out-c", type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    if args.out_c is not None and args.host_elf is not None:
        parser.error("generate the map before linking the final host ELF")
    if args.out.exists() or args.out.is_symlink() or (args.out_c is not None and
            (args.out_c.exists() or args.out_c.is_symlink())):
        parser.error("fresh output paths required")
    try:
        result, host_map = check(args.compiled, args.host_map, args.soc_memmap,
                                 args.host_elf)
        if args.out_c is not None:
            args.out_c.parent.mkdir(parents=True, exist_ok=True)
            args.out_c.write_text(render_map_c(host_map))
    except (OSError, ValueError, TypeError, KeyError) as error:
        result = {"schema": "atlas.graph_platform_check.v1", "status": "FAIL",
                  "error": str(error)}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": result["status"], "out": str(args.out)}))
    return 0 if result["status"] != "FAIL" else 1


if __name__ == "__main__":
    raise SystemExit(main())
