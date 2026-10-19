#!/usr/bin/env python3
"""Bind one verified source-matmul tile grid to its outlined MLIR call."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

if __package__:
    from .atlas_model_tile_bridge import _source_file, _symbol_name
    from .atlas_source_mn_host_data import RUNTIME_FILES
else:
    from atlas_model_tile_bridge import _source_file, _symbol_name
    from atlas_source_mn_host_data import RUNTIME_FILES


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def render_wrapper(symbol: str, *, frozen_rhs: bool, rank: int = 2) -> tuple[str, str, str]:
    if rank not in (2, 3):
        raise ValueError("source grid wrapper needs rank two or three")
    symbol = _symbol_name(symbol)
    descriptor = f"atlas_f32_memref{rank}"
    runner = f"atlas_source_mn_run_memref{rank}"
    suffix = _sha(symbol.encode())[:12]
    binder = f"atlas_bind_source_mn_{suffix}"
    checked = f"atlas_source_mn_checked_{suffix}"
    callback = f"atlas_source_mn_ciface_{suffix}"
    map_name = f"atlas_source_mn_map_{suffix}"
    guard = f"ATLAS_SOURCE_MN_MODEL_WRAPPER_{suffix.upper()}_H"
    frozen = (
        "  .frozen_rhs_bits = atlas_source_mn_frozen_rhs_bits,\n"
        "  .frozen_rhs_elements = ATLAS_SOURCE_MN_BATCHES * "
        "ATLAS_SOURCE_MN_REDUCTION * ATLAS_SOURCE_MN_COLUMNS,\n"
        if frozen_rhs else ""
    )
    header = f"""/* Generated from a checked Merlin-native source tile grid. */
#ifndef {guard}
#define {guard}
#include "atlas_source_mn_memref.h"
int {binder}(atlas_host *host, atlas_tile_memory_io io, size_t max_polls);
atlas_host_map {map_name}(void);
atlas_ordered_k_status {checked}({descriptor} *result,
    const {descriptor} *lhs, const {descriptor} *rhs,
    atlas_host_result *device_result);
#endif
"""
    source = f"""/* Generated from a checked Merlin-native source tile grid. */
#include "atlas_source_mn_model_wrapper.h"
#include "atlas_source_mn_data.h"

#if !ATLAS_SOURCE_MN_BITSTREAM_BOUND && !defined(ATLAS_DIAGNOSTIC_ALLOW_UNBOUND)
#error "Bind the selected Atlas program to a bitstream before deployment"
#endif
#if !defined(ATLAS_PLATFORM_CACHE_SYNC) && !defined(ATLAS_DIAGNOSTIC_ASSUME_COHERENT)
#error "Supply a qualified cache policy or select diagnostic coherence"
#endif
#ifndef ATLAS_DIAGNOSTIC_PRECISION_POLICY
#error "This source-bound precision policy is diagnostic, not model-qualified"
#endif

static atlas_ordered_k_tile tile;
static _Alignas(float) uint8_t scratch[2048u];
static atlas_host_buffer buffers[ATLAS_SOURCE_MN_WINDOW_COUNT + 1u];
static const atlas_source_mn_memref_binding binding = {{
  .tile = &tile,
  .encode_lhs = atlas_source_mn_encode_lhs,
  .encode_rhs = atlas_source_mn_encode_rhs,
  .batches = ATLAS_SOURCE_MN_BATCHES,
  .m = ATLAS_SOURCE_MN_ROWS,
  .n = ATLAS_SOURCE_MN_COLUMNS,
  .k = ATLAS_SOURCE_MN_REDUCTION,
{frozen}}};

int {binder}(atlas_host *host, atlas_tile_memory_io io, size_t max_polls) {{
  return atlas_source_mn_bind(&tile, host, io.context, io.write, io.read,
                              max_polls, scratch, sizeof scratch,
                              buffers, ATLAS_SOURCE_MN_WINDOW_COUNT + 1u);
}}

atlas_host_map {map_name}(void) {{ return atlas_source_mn_host_map; }}

atlas_ordered_k_status {checked}({descriptor} *result,
    const {descriptor} *lhs, const {descriptor} *rhs,
    atlas_host_result *device_result) {{
  return {runner}(&binding, lhs, rhs, result, device_result);
}}

void {callback}({descriptor} *result, const {descriptor} *lhs,
                const {descriptor} *rhs)
    __asm__("_mlir_ciface_{symbol}");
void {callback}({descriptor} *result, const {descriptor} *lhs,
                const {descriptor} *rhs) {{
  atlas_host_result device_result = ATLAS_HOST_INVALID;
  atlas_ordered_k_status status = {checked}(result, lhs, rhs, &device_result);
  if (status == ATLAS_ORDERED_K_OK) return;
  atlas_model_tile_status failure = ATLAS_MODEL_TILE_INVALID;
  if (status == ATLAS_ORDERED_K_INPUT_DOMAIN) failure = ATLAS_MODEL_TILE_INPUT_DOMAIN;
  if (status == ATLAS_ORDERED_K_CONSTANT_MISMATCH)
    failure = ATLAS_MODEL_TILE_CONSTANT_MISMATCH;
  if (status == ATLAS_ORDERED_K_MEMORY_ERROR) failure = ATLAS_MODEL_TILE_MEMORY_ERROR;
  if (status == ATLAS_ORDERED_K_DEVICE_ERROR) failure = ATLAS_MODEL_TILE_DEVICE_ERROR;
  atlas_model_tile_fail(failure, device_result);
}}
"""
    return header, source, binder


def prepare(probe: Path, region_id: str, host_bundle: Path) -> dict:
    from merlin.frontends.linalg_mlir import parse_mlir_file
    from xdsl.dialects import func

    probe_manifest = json.loads((probe / "manifest.json").read_text())
    bundle_path = host_bundle / "manifest.json"
    if host_bundle.is_symlink() or bundle_path.is_symlink():
        raise ValueError("source host bundle is linked")
    bundle_bytes = bundle_path.read_bytes()
    bundle = json.loads(bundle_bytes)
    expected_files = {*RUNTIME_FILES, "atlas_source_mn_data.h",
                      "row_major_k_window.h", "finite_float_boundary.h"}
    files = bundle.get("files")
    if (bundle.get("schema") != "atlas.source_mn_host_data.v1"
            or bundle.get("engine") != "merlin_native"
            or bundle.get("status") not in {"diagnostic_unbound", "diagnostic_unqualified"}
            or bundle.get("rank") not in (2, 3)
            or type(bundle.get("batch_count")) is not int
            or bundle["batch_count"] < 1
            or (bundle["rank"] == 2 and bundle["batch_count"] != 1)
            or not isinstance(files, dict) or set(files) != expected_files
            or {path.name for path in host_bundle.iterdir()} != expected_files | {"manifest.json"}):
        raise ValueError("source host bundle is not a checked rank-two or rank-three grid")
    for name, digest in files.items():
        if _sha(_source_file(host_bundle, name).read_bytes()) != digest:
            raise ValueError(f"source host member changed: {name}")
    identity = bundle.get("source_identity", {})
    bridge = probe_manifest.get("host_bridge", {})
    bridge_path = _source_file(probe, bridge.get("path"))
    if (probe_manifest.get("schema") != "merlin.native_model_region_probe.v1"
            or probe_manifest.get("engine") != "merlin_native"
            or probe_manifest.get("status") != "incomplete"
            or bundle.get("target_identity") != probe_manifest.get("target_identity")
            or identity != {"original_source_sha256": probe_manifest.get("original_source_sha256"),
                            "staged_program_sha256": probe_manifest.get("staged_program_sha256"),
                            "precision_policy_sha256": probe_manifest.get("precision_policy_sha256"),
                            "region_id": region_id}
            or bridge.get("schema") != "merlin.diagnostic_model_host_bridge.v1"
            or bridge.get("status") != "typed_declarations_only"
            or bridge.get("sha256") != _sha(bridge_path.read_bytes())):
        raise ValueError("source host bundle and model probe differ")
    rank = bundle["rank"]
    batches = bundle["batch_count"]
    region_rows = probe_manifest.get(
        "regions" if rank == 2 else "batched_contractions", [],
    )
    regions = [row for row in region_rows if row.get("region_id") == region_id]
    bindings = [row for row in bridge.get("bindings", []) if row.get("region_id") == region_id]
    if len(regions) != 1 or len(bindings) != 1:
        raise ValueError("source grid has no unique model region binding")
    region, selected = regions[0], bindings[0]
    symbol = _symbol_name(selected.get("symbol"))
    m, n, k = (bundle.get(key) for key in ("m_rows", "n_rows", "k_columns"))
    source_prefix = f"{batches}x" if rank == 3 else ""
    if (any(type(value) is not int or value <= 0 for value in (m, n, k))
            or (rank == 3 and region.get("batch_shape") != [batches])
            or (region.get("tiling_status") if rank == 2 else region.get("status"))
            != "isolated_tiles_emitted"
            or region.get("tile_count") != bundle.get("tile_count")
            or selected.get("selected_route") != "tiled_native"
            or selected.get("operation_id") != region.get("operation_id")
            or selected.get("input_types") != [f"tensor<{source_prefix}{m}x{k}xf32>",
                                               f"tensor<{source_prefix}{k}x{n}xf32>"]
            or selected.get("result_types") != [f"tensor<{source_prefix}{m}x{n}xf32>"]
            or symbol not in bridge.get("external_symbols", [])
            or symbol not in bridge.get("c_interface_symbols", [])):
        raise ValueError("source grid differs from the outlined two-input matmul")
    declarations = [op for op in parse_mlir_file(bridge_path).body.block.ops
                    if isinstance(op, func.FuncOp) and op.sym_name.data == symbol]
    if (len(declarations) != 1 or declarations[0].body.blocks
            or "llvm.emit_c_interface" not in declarations[0].attributes):
        raise ValueError("source grid has no external MLIR C-interface declaration")
    tiles = [row for row in probe_manifest.get(
        "named_tiles" if rank == 2 else "batched_tiles", [],
    )
             if row.get("region_id") == region_id]
    tile_records = bundle.get("tiles")
    if (not isinstance(tile_records, list) or len(tiles) != len(tile_records)
            or len(tiles) != bundle["tile_count"]
            or sorted(row.get("tile_index") for row in tiles) != list(range(len(tiles)))):
        raise ValueError("source grid is missing selected model tiles")
    for tile, record in zip(sorted(tiles, key=lambda row: row["tile_index"]),
                            tile_records, strict=True):
        artifact = tile.get("artifact")
        if (tile.get("status") != "emitted"
                or tile.get("static_inputs_status") != "packed_or_runtime"
                or tile.get("operation_id") != region["operation_id"]
                or tile.get("request_digest") != record.get("request_digest")
                or tile.get("binary_sha256") != bundle.get("program_sha256")
                or tile.get("tile_index") != record.get("tile")
                or not isinstance(artifact, str)
                or not re.fullmatch(
                    r"regions/" + ("named" if rank == 2 else "batched")
                    + r"-[0-9]{4}-tile-[0-9]{4}", artifact,
                )
                or _sha(_source_file(probe, artifact + "/manifest.json").read_bytes())
                != record.get("manifest_sha256")):
            raise ValueError("source grid tile differs from the model probe")
    if selected.get("tile_request_digests") != [
        record["request_digest"] for record in tile_records
    ]:
        raise ValueError("outlined call has a different native tile roster")
    frozen_rhs = bundle.get("frozen_rhs") is not None
    if frozen_rhs != ("#define ATLAS_SOURCE_MN_FROZEN_RHS 1" in
                      _source_file(host_bundle, "atlas_source_mn_data.h").read_text()):
        raise ValueError("source host bundle has inconsistent frozen RHS")
    header, source, binder = render_wrapper(symbol, frozen_rhs=frozen_rhs, rank=rank)
    generated = {
        "atlas_source_mn_model_wrapper.h": header,
        "atlas_source_mn_model_wrapper.c": source,
    }
    return {
        "files": generated,
        "record": {
            "schema": "atlas.diagnostic_source_mn_model_wrapper.v1",
            "status": bundle["status"], "engine": "merlin_native",
            "region_id": region_id, "source_operation_id": region["operation_id"],
            "outlined_symbol": symbol, "binding_function": binder,
            "rank": rank, "batch_count": batches,
            "native_request_digests": selected["tile_request_digests"],
            "program_sha256": bundle["program_sha256"],
            "host_bundle_manifest_sha256": _sha(bundle_bytes),
            "host_bridge_sha256": bridge["sha256"],
            "frozen_rhs": frozen_rhs,
            "linked": False, "executed": False,
        },
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--probe", required=True, type=Path)
    parser.add_argument("--region-id", required=True)
    parser.add_argument("--host-bundle", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args(argv)
    try:
        if args.out.exists() or args.out.is_symlink():
            raise ValueError("fresh wrapper output directory required")
        prepared = prepare(args.probe, args.region_id, args.host_bundle)
        args.out.mkdir(parents=True)
        for name, content in prepared["files"].items():
            (args.out / name).write_text(content)
        record = prepared["record"]
        record["files"] = {
            name: _sha(content.encode()) for name, content in prepared["files"].items()
        }
        (args.out / "manifest.json").write_text(json.dumps(record, sort_keys=True, indent=2) + "\n")
    except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError) as error:
        print(json.dumps({"status": "FAIL", "reason": str(error)}, sort_keys=True), file=sys.stderr)
        return 1
    print(json.dumps({"status": record["status"], "out": str(args.out)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
