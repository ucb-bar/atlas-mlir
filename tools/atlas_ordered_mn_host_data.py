#!/usr/bin/env python3
"""Export one verified reusable ordered-K program as an M/N host bundle."""

from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path

if __package__:
    from .atlas_ordered_k_host_data import _member, _sha
    from .atlas_ordered_k_host_data import prepare as prepare_k
else:
    from atlas_ordered_k_host_data import _member, _sha
    from atlas_ordered_k_host_data import prepare as prepare_k

ROOT = Path(__file__).resolve().parents[1]
RUNTIME_FILES = (
    "atlas_host.h", "atlas_host.c",
    "atlas_ordered_k.h", "atlas_ordered_k.c",
    "atlas_ordered_mn.h", "atlas_ordered_mn.c",
)


def prepare_mn(compiled: Path, verification: Path, host_map: Path,
               packer_header: Path, boundary_header: Path,
               f32_boundary: str | None = None) -> tuple[dict[str, bytes], dict]:
    k_header, k_record, packer_bytes, boundary_bytes = prepare_k(
        compiled, verification, host_map, packer_header, boundary_header,
        f32_boundary,
    )
    manifest = json.loads(_member(compiled, "manifest.json"))
    receipt = json.loads(verification.read_text())
    outer_bytes = _member(compiled, "ordered-mn-tiling.json")
    outer = json.loads(outer_bytes)
    inner_path = compiled / "ordered-k-tiling.json"
    inner = (json.loads(_member(compiled, "ordered-k-tiling.json"))
             if inner_path.exists() or inner_path.is_symlink() else None)
    tracked = manifest.get("stage_artifacts", {})
    has_tails = outer.get("schema") == "merlin.ordered_mn_tiling.v2"
    if (
        outer.get("schema") not in {"merlin.ordered_mn_tiling.v1", "merlin.ordered_mn_tiling.v2"}
        or (has_tails and outer.get("tail_pad_element_hex") != "00")
        or (not has_tails and "tail_pad_element_hex" in outer)
        or manifest.get("outer_source_request_digest") != outer.get("source_request_digest")
        or tracked.get("ordered-mn-tiling.json") != _sha(outer_bytes)
        or "ordered-mn-tiling.json" not in receipt.get("checked_stage_artifacts", [])
        or outer.get("selected_request_digest") != (
            inner.get("source_request_digest") if inner is not None
            else manifest.get("request_digest"))
        or outer.get("tile_m") != 32 or outer.get("tile_n") != 32
        or type(outer.get("selected_tile_index")) is not int
        or not isinstance(outer.get("tiles"), list)
    ):
        raise ValueError("outer matrix partition differs from verified native artifact")
    tiles = outer["tiles"]
    selected = outer["selected_tile_index"]
    if not 0 <= selected < len(tiles) or not isinstance(tiles[0], dict):
        raise ValueError("selected output tile is absent")
    slices = tiles[0].get("input_slices")
    if not isinstance(slices, list) or len(slices) != 2 or not all(isinstance(item, dict) for item in slices):
        raise ValueError("outer matrix inputs are absent")
    left, right = slices
    m, n = left.get("source_rows"), right.get("source_rows")
    k = left.get("source_columns")
    source_names = ([item["source_input"] for item in inner["source_slices"][:2]]
                    if inner is not None else k_record["source_inputs"])
    m_tiles = (m + 31) // 32 if type(m) is int and m > 0 else 0
    n_tiles = (n + 31) // 32 if type(n) is int and n > 0 else 0
    if (
        any(type(extent) is not int or extent <= 0 for extent in (m, n, k))
        or (has_tails != bool(m % 32 or n % 32))
        or k != (inner["source_slices"][0]["source_columns"]
                 if inner is not None else 32)
        or k_record["window_count"] != 2 * ((k + 31) // 32)
        or left.get("source_input") != source_names[0]
        or right.get("source_input") != source_names[1]
        or len(tiles) != m_tiles * n_tiles
        or any(tile.get("request_digest") != outer["selected_request_digest"]
               for tile in tiles)
    ):
        raise ValueError("outer matrix geometry cannot reuse the selected tile program")
    for index, tile in enumerate(tiles):
        expected_m = (index // n_tiles) * 32
        expected_n = (index % n_tiles) * 32
        expected_valid_m = min(32, m - expected_m)
        expected_valid_n = min(32, n - expected_n)
        expected_slices = [
            {
                "tile_input": left["tile_input"],
                "source_input": left["source_input"],
                "source_rows": m, "source_columns": k,
                "row_start": expected_m, "column_start": 0,
                "rows": expected_valid_m, "columns": k,
            },
            {
                "tile_input": right["tile_input"],
                "source_input": right["source_input"],
                "source_rows": n, "source_columns": k,
                "row_start": expected_n, "column_start": 0,
                "rows": expected_valid_n, "columns": k,
            },
        ]
        if (
            not isinstance(tile, dict)
            or tile.get("m_start") != expected_m
            or tile.get("n_start") != expected_n
            or tile.get("input_slices") != expected_slices
            or (has_tails and (tile.get("valid_m"), tile.get("valid_n"))
                != (expected_valid_m, expected_valid_n))
            or (not has_tails and ("valid_m" in tile or "valid_n" in tile))
        ):
            raise ValueError("outer matrix slices omit, overlap, or reorder a source region")

    header = "\n".join((
        ("/* One verified program reused over every padded 32x32 output tile. */"
         if has_tails else
         "/* One verified program reused over every exact 32x32 output tile. */"),
        "#ifndef ATLAS_ORDERED_MN_DATA_H",
        "#define ATLAS_ORDERED_MN_DATA_H",
        '#include "atlas_ordered_k_data.h"',
        '#include "atlas_ordered_mn.h"',
        f"#define ATLAS_ORDERED_MN_ROWS {m}u",
        f"#define ATLAS_ORDERED_MN_COLUMNS {n}u",
        f"#define ATLAS_ORDERED_MN_REDUCTION {k}u",
        f"#define ATLAS_ORDERED_MN_TILE_COUNT {len(tiles)}u",
        *(("#define ATLAS_ORDERED_MN_HAS_TAILS 1u",) if has_tails else ()),
        "#endif",
        "",
    ))
    files = {
        "atlas_ordered_k_data.h": k_header.encode(),
        "atlas_ordered_mn_data.h": header.encode(),
        "row_major_k_window.h": packer_bytes,
        "finite_float_boundary.h": boundary_bytes,
    }
    files.update({name: (ROOT / "runtime" / name).read_bytes() for name in RUNTIME_FILES})
    if any(not payload for payload in files.values()):
        raise ValueError("ordered M/N host source is empty")
    record = {
        "schema": "atlas.ordered_mn_host_data.v2" if has_tails else "atlas.ordered_mn_host_data.v1",
        "status": k_record["status"],
        "engine": "merlin_native",
        "source_request_digest": outer["source_request_digest"],
        "tile_request_digest": outer["selected_request_digest"],
        "program_sha256": k_record["program_sha256"],
        "native_manifest_sha256": k_record["native_manifest_sha256"],
        "native_verification_sha256": k_record["native_verification_sha256"],
        "outer_tiling_sha256": _sha(outer_bytes),
        "host_map_sha256": k_record["host_map_sha256"],
        "m_rows": m,
        "n_rows": n,
        "k_columns": k,
        "tile_count": len(tiles),
        "selected_tile_index": selected,
        "runtime_input_domain": k_record["runtime_input_domain"],
        "source_domain_coverage": k_record["source_domain_coverage"],
        "files": {name: _sha(payload) for name, payload in sorted(files.items())},
    }
    if "f32_boundary" in k_record:
        record["f32_boundary"] = k_record["f32_boundary"]
    if has_tails:
        record["has_tails"] = True
    return files, record


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--compiled", type=Path, required=True)
    parser.add_argument("--verification", type=Path, required=True)
    parser.add_argument("--host-map", type=Path, required=True)
    parser.add_argument("--packer-header", type=Path, required=True)
    parser.add_argument("--boundary-header", type=Path, required=True)
    parser.add_argument("--f32-boundary", choices=("diagnostic-mxu0-f32",))
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists() or args.out.is_symlink():
        parser.error("fresh output directory required")
    try:
        files, record = prepare_mn(
            args.compiled, args.verification, args.host_map,
            args.packer_header, args.boundary_header, args.f32_boundary,
        )
    except (OSError, ValueError, TypeError, KeyError, IndexError) as error:
        parser.error(str(error))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="ordered-mn-host-", dir=args.out.parent) as temporary:
        staged = Path(temporary) / "host-data"
        staged.mkdir()
        for name, payload in files.items():
            (staged / name).write_bytes(payload)
        (staged / "manifest.json").write_text(json.dumps(record, sort_keys=True, indent=2) + "\n")
        staged.rename(args.out)
    print(json.dumps({"status": record["status"], "out": str(args.out)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
