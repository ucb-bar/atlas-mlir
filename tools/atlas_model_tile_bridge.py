#!/usr/bin/env python3
"""Bind one verified Merlin-native tile to an outlined MLIR C-interface symbol."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

if __package__:
    from .atlas_captured_host_data import _native_import, source_tile_window
    from .atlas_captured_host_data import render as render_host_data
    from .atlas_host_embed import render as render_program
else:
    from atlas_captured_host_data import _native_import, source_tile_window
    from atlas_captured_host_data import render as render_host_data
    from atlas_host_embed import render as render_program


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _source_file(root: Path, relative: str) -> Path:
    if not isinstance(relative, str) or not relative or Path(relative).is_absolute():
        raise ValueError("bridge artifact path must be relative")
    path = root / relative
    if path.is_symlink() or not path.resolve().is_relative_to(root.resolve()) or not path.is_file():
        raise ValueError("bridge artifact escapes or is absent from the probe")
    return path


def _symbol_name(symbol: str) -> str:
    if not isinstance(symbol, str) or not symbol or not symbol.isascii() or any(
        not (character.isalnum() or character in "_$") for character in symbol
    ):
        raise ValueError("outlined symbol cannot be represented as a C linker name")
    return symbol


def render_wrapper(symbol: str, constant_operand: int,
                   transpose_input: tuple[int, int],
                   source_shape: tuple[int, int, int] | None = None) -> tuple[str, str, str]:
    """Emit one typed C wrapper; all numerical and physical facts come from headers."""
    symbol = _symbol_name(symbol)
    if constant_operand not in (0, 1, 2):
        raise ValueError("selected constant operand must be lhs, rhs, or absent")
    if len(transpose_input) != 2 or any(value not in (0, 1) for value in transpose_input):
        raise ValueError("selected input layouts must be identity or transpose")
    if source_shape is not None and (
        len(source_shape) != 3 or any(type(value) is not int or not 1 <= value <= 32
                                      for value in source_shape)
    ):
        raise ValueError("selected source window must fit one physical tile")
    suffix = _sha(symbol.encode())[:12]
    binder = f"atlas_bind_tile_{suffix}"
    map_name = f"atlas_tile_map_{suffix}"
    checked = f"atlas_tile_run_checked_{suffix}"
    callback = f"atlas_tile_ciface_{suffix}"
    guard = f"ATLAS_MODEL_TILE_WRAPPER_{suffix.upper()}_H"
    call = ("atlas_model_tile_run(&tile, lhs, rhs, result, device_result)"
            if source_shape is None else
            "atlas_model_tile_run_window(&tile, lhs, rhs, result, "
            f"{source_shape[0]}u, {source_shape[1]}u, {source_shape[2]}u, device_result)")
    if constant_operand == 2:
        input_setup = """  tile.input_addresses[0] = ATLAS_CAPTURED_INPUT_ADDRESS_0;
  tile.input_addresses[1] = ATLAS_CAPTURED_INPUT_ADDRESS_1;
  tile.constant_operand = 2;"""
    else:
        input_setup = f"""  tile.input_addresses[{constant_operand}] = atlas_captured_constants[0].address;
  tile.input_addresses[{1 - constant_operand}] = ATLAS_CAPTURED_INPUT_ADDRESS;
  tile.constant_bytes = atlas_captured_constants[0].data;
  tile.constant_bytes_length = atlas_captured_constants[0].size;
  tile.constant_operand = {constant_operand};"""
    header = f"""/* Generated from a verified Merlin-native tile. */
#ifndef {guard}
#define {guard}
#include "atlas_model_tile.h"
/* Returns nonzero if an earlier launch is still active or inputs are invalid. */
int {binder}(atlas_host *host, atlas_tile_memory_io io, size_t max_polls);
atlas_host_map {map_name}(void);
/* Returns the tile failure status without entering the MLIR void callback. */
atlas_model_tile_status {checked}(atlas_f32_memref2 *result,
    const atlas_f32_memref2 *lhs, const atlas_f32_memref2 *rhs,
    atlas_host_result *device_result);
#endif
"""
    source = f"""/* Generated from a verified Merlin-native tile. */
#include "atlas_model_tile_wrapper.h"
#include <atlas_program.h>
#include <atlas_captured_data.h>

#if !ATLAS_PROGRAM_BITSTREAM_BOUND && !defined(ATLAS_DIAGNOSTIC_ALLOW_UNBOUND)
#error "Bind the selected Atlas program to a bitstream before deployment"
#endif
#if !defined(ATLAS_PLATFORM_CACHE_SYNC) && !defined(ATLAS_DIAGNOSTIC_ASSUME_COHERENT)
#error "Supply a qualified cache policy or select diagnostic coherence"
#endif
#ifndef ATLAS_DIAGNOSTIC_PRECISION_POLICY
#error "This source-bound precision policy is diagnostic, not model-qualified"
#endif

static atlas_model_tile tile;
static _Alignas(float) uint8_t scratch[ATLAS_MODEL_TILE_SCRATCH_BYTES];

int {binder}(atlas_host *host, atlas_tile_memory_io io, size_t max_polls) {{
  if (!host || host->active || (tile.host && tile.host->active) ||
      !io.write || !io.read || !max_polls)
    return 1;
  tile = (atlas_model_tile){{0}};
  tile.host = host;
  tile.io = io;
  tile.program_words = atlas_program_words;
  tile.program_word_count = ATLAS_PROGRAM_WORD_COUNT;
{input_setup}
  tile.output_address = ATLAS_CAPTURED_OUTPUT_ADDRESS;
  tile.transpose_input[0] = {transpose_input[0]};
  tile.transpose_input[1] = {transpose_input[1]};
  tile.pack[0] = atlas_pack_runtime_input_0_f32;
  tile.pack[1] = atlas_pack_runtime_input_1_f32;
  tile.unpack = atlas_unpack_output_bf16_f32;
  tile.max_polls = max_polls;
  tile.scratch = scratch;
  tile.scratch_bytes = sizeof(scratch);
  return 0;
}}

atlas_host_map {map_name}(void) {{ return atlas_program_map; }}

atlas_model_tile_status {checked}(atlas_f32_memref2 *result,
    const atlas_f32_memref2 *lhs, const atlas_f32_memref2 *rhs,
    atlas_host_result *device_result) {{
  return {call};
}}

void {callback}(atlas_f32_memref2 *result, const atlas_f32_memref2 *lhs,
                const atlas_f32_memref2 *rhs)
    __asm__("_mlir_ciface_{symbol}");
void {callback}(atlas_f32_memref2 *result, const atlas_f32_memref2 *lhs,
                const atlas_f32_memref2 *rhs) {{
  atlas_host_result device_result;
  atlas_model_tile_status status = {checked}(result, lhs, rhs, &device_result);
  if (status != ATLAS_MODEL_TILE_OK)
    atlas_model_tile_fail(status, device_result);
}}
"""
    return header, source, binder


def prepare(probe: Path, region_id: str, verification: Path, capsule: Path,
            host_map: Path) -> dict[str, str | dict]:
    probe_manifest = json.loads((probe / "manifest.json").read_text())
    bridge = probe_manifest.get("host_bridge", {})
    bridge_path = _source_file(probe, bridge.get("path"))
    if (probe_manifest.get("schema") != "merlin.native_model_region_probe.v1"
            or probe_manifest.get("engine") != "merlin_native"
            or bridge.get("schema") != "merlin.diagnostic_model_host_bridge.v1"
            or bridge.get("status") != "typed_declarations_only"
            or bridge.get("linked") is not False or bridge.get("executed") is not False
            or type(bridge.get("strict_native_eligible")) is not bool
            or bridge.get("sha256") != _sha(bridge_path.read_bytes())):
        raise ValueError("probe has no intact diagnostic native host bridge")
    regions = [row for row in probe_manifest.get("regions", []) if row.get("region_id") == region_id]
    bindings = [row for row in bridge.get("bindings", []) if row.get("region_id") == region_id]
    if len(regions) != 1 or len(bindings) != 1:
        raise ValueError("selected source region has no unique host binding")
    region, binding = regions[0], bindings[0]
    symbol = _symbol_name(binding.get("symbol"))
    if (region.get("status") != "emitted" or binding.get("native_status") != "emitted"
            or binding.get("operation_id") != region.get("operation_id")
            or binding.get("request_digest") != region.get("request_digest")
            or symbol not in bridge.get("external_symbols", [])
            or symbol not in bridge.get("c_interface_symbols", [])):
        raise ValueError("native region and outlined C-interface types differ")
    from merlin.frontends.linalg_mlir import parse_mlir_file
    from xdsl.dialects import func

    module = parse_mlir_file(bridge_path)
    declarations = [op for op in module.body.block.ops
                    if isinstance(op, func.FuncOp) and op.sym_name.data == symbol]
    if (len(declarations) != 1 or declarations[0].body.blocks
            or "llvm.emit_c_interface" not in declarations[0].attributes):
        raise ValueError("selected host symbol is not an external C-interface declaration")

    artifact = region.get("artifact")
    if not isinstance(artifact, str) or not artifact.startswith("regions/region-"):
        raise ValueError("selected native region has no bounded artifact path")
    compiled = probe / artifact
    if compiled.is_symlink() or not compiled.is_dir() or not compiled.resolve().is_relative_to(probe.resolve()):
        raise ValueError("native artifact escapes the probe")
    native_manifest = json.loads((compiled / "manifest.json").read_text())
    if (native_manifest.get("request_digest") != region["request_digest"]
            or native_manifest.get("binary_sha256") != region.get("binary_sha256")
            or native_manifest.get("target_identity") != probe_manifest.get("target_identity")):
        raise ValueError("native artifact differs from the model region record")
    imported = _native_import(compiled, verification)
    if len(imported["constants"]) > 1 or any(
        row["payload_bytes"] != 1024 for row in imported["constants"]
    ):
        raise ValueError("model tile wrapper supports at most one packed constant")
    constants = native_manifest["constant_inputs"]
    if len(constants) != len(imported["constants"]):
        raise ValueError("native constant and host payload count differ")
    if constants:
        if constants[0].get("role") not in ("lhs", "rhs"):
            raise ValueError("native constant has no operand role")
        constant_operand = 0 if constants[0]["role"] == "lhs" else 1
    else:
        constant_operand = 2
    precision = json.loads((compiled / "precision-plan.json").read_text())
    m, n, k = source_tile_window(precision)
    if (binding.get("input_types") != [f"tensor<{m}x{k}xf32>",
                                          f"tensor<{k}x{n}xf32>"]
            or binding.get("result_types") != [f"tensor<{m}x{n}xf32>"]):
        raise ValueError("native region and outlined C-interface types differ")
    inputs = precision.get("inputs", [])
    if (len(inputs) != 2 or any(
        not isinstance(item, dict) or item.get("source_type", {}).get("dtype") != "f32"
        for item in inputs
    )):
        raise ValueError("outlined f32 inputs need selected FP8 packers")
    axes = [item.get("target_axes_from_source") for item in inputs]
    if any(item not in ([0, 1], [1, 0]) for item in axes):
        raise ValueError("outlined input layout needs identity or transpose")
    transpose_input = tuple(int(item == [1, 0]) for item in axes)

    capsule_manifest = json.loads((capsule / "manifest.json").read_text())
    program = (capsule / capsule_manifest["program_file"]).read_bytes()
    selected_map = json.loads(host_map.read_text())
    program_header = render_program(capsule_manifest, program, selected_map)
    data_header = render_host_data(compiled, capsule, native_verification=verification)
    source_shape = None if (m, n, k) == (32, 32, 32) else (m, n, k)
    wrapper_header, wrapper_source, binder = render_wrapper(
        symbol, constant_operand, transpose_input, source_shape)
    return {
        "files": {
            "atlas_program.h": program_header,
            "atlas_captured_data.h": data_header,
            "atlas_model_tile_wrapper.h": wrapper_header,
            "atlas_model_tile_wrapper.c": wrapper_source,
        },
        "record": {
            "schema": "atlas.diagnostic_model_tile_wrapper.v1",
            "status": "diagnostic_unbound" if not selected_map.get("bitstream_identity") else "diagnostic_unqualified",
            "engine": "merlin_native",
            "region_id": region_id,
            "source_operation_id": region["operation_id"],
            "outlined_symbol": symbol,
            "binding_function": binder,
            "constant_operand": constant_operand,
            "transpose_input": list(transpose_input),
            **({"source_shape": list(source_shape)} if source_shape else {}),
            "native_request_digest": region["request_digest"],
            "native_binary_sha256": native_manifest["binary_sha256"],
            "native_manifest_sha256": _sha((compiled / "manifest.json").read_bytes()),
            "native_verification_sha256": _sha(verification.read_bytes()),
            "host_bridge_sha256": bridge["sha256"],
            "bitstream_bound": bool(selected_map.get("bitstream_identity")),
            "linked": False,
            "executed": False,
        },
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--probe", required=True, type=Path)
    parser.add_argument("--region-id", required=True)
    parser.add_argument("--native-verification", required=True, type=Path)
    parser.add_argument("--capsule", required=True, type=Path)
    parser.add_argument("--host-map", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args(argv)
    try:
        if args.out.exists() or args.out.is_symlink():
            raise ValueError("fresh wrapper output directory required")
        prepared = prepare(args.probe, args.region_id, args.native_verification,
                           args.capsule, args.host_map)
        args.out.mkdir(parents=True)
        files = prepared["files"]
        record = prepared["record"]
        for name, content in files.items():
            (args.out / name).write_text(content)
        record["files"] = {name: _sha(content.encode()) for name, content in files.items()}
        (args.out / "manifest.json").write_text(json.dumps(record, sort_keys=True, indent=2) + "\n")
    except (OSError, ValueError, KeyError, TypeError, ImportError, json.JSONDecodeError) as error:
        print(json.dumps({"status": "FAIL", "reason": str(error)}, sort_keys=True), file=sys.stderr)
        return 1
    print(json.dumps({"status": record["status"], "out": str(args.out)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
