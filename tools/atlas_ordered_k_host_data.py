#!/usr/bin/env python3
"""Export one verified MXU0 tile with one or more exact K windows."""

from __future__ import annotations

import argparse
import hashlib
import json
import struct
import tempfile
from pathlib import Path


def _sha(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _member(root: Path, name: str) -> bytes:
    path = root / name
    if path.is_symlink() or not path.is_file() or not path.resolve().is_relative_to(root.resolve()):
        raise ValueError(f"native artifact member is absent or escapes: {name}")
    return path.read_bytes()


def prepare(compiled: Path, verification: Path, host_map: Path,
            packer_header: Path, boundary_header: Path,
            f32_boundary: str | None = None) -> tuple[str, dict, bytes, bytes]:
    if packer_header.name != "row_major_k_window.h" or packer_header.is_symlink():
        raise ValueError("ordered-K host data requires the selected Merlin packer header")
    if boundary_header.name != "finite_float_boundary.h" or boundary_header.is_symlink():
        raise ValueError("ordered-K host data requires the selected Merlin BF16 boundary header")
    packer_bytes = packer_header.read_bytes()
    boundary_bytes = boundary_header.read_bytes()
    if not packer_bytes or not boundary_bytes:
        raise ValueError("selected Merlin runtime header is empty")
    manifest_bytes = _member(compiled, "manifest.json")
    manifest = json.loads(manifest_bytes)
    receipt = json.loads(verification.read_text())
    plan_bytes = _member(compiled, "execution_plan.json")
    plan = json.loads(plan_bytes)
    tiling_path = compiled / "ordered-k-tiling.json"
    has_k_partition = tiling_path.exists() or tiling_path.is_symlink()
    tiling_bytes = _member(compiled, "ordered-k-tiling.json") if has_k_partition else None
    tiling = json.loads(tiling_bytes) if tiling_bytes is not None else None
    binary = _member(compiled, "program.bin")
    selected_map = json.loads(host_map.read_text())
    target = manifest.get("target_identity", "")
    revision = target.removeprefix("atlas-rtl-").split(":", 1)[0]
    soc_revision = selected_map.get("soc_source_revision", "")
    tracked = manifest.get("stage_artifacts", {})
    if (
        manifest.get("schema") != "atlas.native_tile_compilation.v6"
        or manifest.get("engine") != "merlin_native"
        or manifest.get("artifact_kind") != "program"
        or manifest.get("compile_mode") != "diagnostic"
        or manifest.get("qualification_scope") != "diagnostic_only"
        or (has_k_partition and (
            manifest.get("source_request_digest") != tiling.get("source_request_digest")
            or manifest.get("request_digest") != tiling.get("tiled_request_digest")))
        or (not has_k_partition and "source_request_digest" in manifest)
        or manifest.get("execution_plan_sha256") != _sha(plan_bytes)
        or manifest.get("binary_sha256") != _sha(binary)
        or (has_k_partition and tracked.get("ordered-k-tiling.json") != _sha(tiling_bytes))
        or (not has_k_partition and "ordered-k-tiling.json" in tracked)
        or len(revision) != 40 or any(character not in "0123456789abcdef" for character in revision)
        or not target.startswith(f"atlas-rtl-{revision}:")
        or selected_map.get("schema") != "atlas.host-map.v1"
        or selected_map.get("rtl_revision") != revision
        or not isinstance(selected_map.get("soc_config_source"), str)
        or not selected_map["soc_config_source"]
        or type(soc_revision) is not str or len(soc_revision) != 40
        or any(character not in "0123456789abcdef" for character in soc_revision)
        or type(selected_map.get("imem_base")) is not int
        or type(selected_map.get("csr_base")) is not int
        or type(selected_map.get("imem_capacity_bytes")) is not int
        or selected_map["imem_base"] < 0 or selected_map["imem_base"] % 4
        or selected_map["csr_base"] < 0 or selected_map["csr_base"] % 4
        or selected_map["imem_capacity_bytes"] < 4
        or selected_map["imem_capacity_bytes"] % 4
        or selected_map.get("start_csr_offset") != 24
        or len(binary) % 4 or not binary
        or len(binary) > selected_map["imem_capacity_bytes"]
    ):
        raise ValueError("ordered-K artifact or selected host map differs")
    if (
        receipt.get("schema") != "merlin.native_artifact_verification.v1"
        or receipt.get("status") != "verified_artifacts"
        or receipt.get("engine") != "merlin_native"
        or receipt.get("source_kind") != "parsed_linalg"
        or receipt.get("selected_record_checked") is not True
        or Path(receipt.get("artifact", "")).resolve() != compiled.resolve()
        or receipt.get("manifest_sha256") != _sha(manifest_bytes)
        or receipt.get("request_digest") != manifest["request_digest"]
        or receipt.get("target_identity") != target
        or receipt.get("binary_sha256") != _sha(binary)
        or not isinstance(receipt.get("checked_stage_artifacts"), list)
        or (has_k_partition and "ordered-k-tiling.json"
            not in receipt.get("checked_stage_artifacts", []))
        or (not has_k_partition and "ordered-k-tiling.json"
            in receipt.get("checked_stage_artifacts", []))
    ):
        raise ValueError("ordered-K artifact has no matching native verification")

    if has_k_partition:
        slices = tiling.get("source_slices")
        tile_count = tiling.get("tile_count")
        source_request_digest = tiling.get("source_request_digest")
        has_head_pad = tiling.get("schema") == "merlin.ordered_k_tiling.v2"
    else:
        obligations = plan.get("boundary_domain_obligations")
        if (not isinstance(obligations, list) or len(obligations) != 2
                or any(not isinstance(item, dict)
                       or item.get("required_domain") != "finite-normal-e4m3-or-positive-zero"
                       or item.get("status") != "caller_premise_unchecked"
                       for item in obligations)):
            raise ValueError("single-K source inputs are absent")
        slices = [
            {"source_input": item.get("input"), "tile_input": item.get("input"),
             "rows": 32, "source_columns": 32, "k_start": 0, "tile_columns": 32}
            for item in obligations
        ]
        tile_count = 1
        source_request_digest = manifest["request_digest"]
        has_head_pad = False
    addresses = plan.get("input_addresses")
    source_columns = (slices[0].get("source_columns")
                      if isinstance(slices, list) and slices and isinstance(slices[0], dict)
                      else None)
    head_count = source_columns % 32 if type(source_columns) is int else 0
    if (
        (has_k_partition and (
            tiling.get("schema") not in {
                "merlin.ordered_k_tiling.v1", "merlin.ordered_k_tiling.v2"}
            or tiling.get("tile_k") != 32
            or (has_head_pad and tiling.get("head_pad_element_hex") != "00")
            or (not has_head_pad and "head_pad_element_hex" in tiling)))
        or type(tile_count) is not int or tile_count < 1
        or type(source_columns) is not int or source_columns <= 0
        or source_columns != (head_count + 32 * (tile_count - 1)
                              if has_head_pad else 32 * tile_count)
        or has_head_pad != bool(head_count)
        or not isinstance(slices, list) or len(slices) != 2 * tile_count
        or not isinstance(addresses, list) or len(addresses) != len(slices)
        or (not has_k_partition and (
            plan.get("input_address") != addresses[0]
            or plan.get("weight_address") != addresses[1]))
        or plan.get("operation") != (f"mxu0_k{tile_count}" if tile_count > 1
                                     else "mxu0_reset")
        or plan.get("schema") != "atlas.native_tile_execution_plan.v8"
        or plan.get("numerical_policy") != "mxu0-e4m3-ordered-bf16-finite-v1"
        or plan.get("input_domain") != "finite-normal-e4m3-or-positive-zero"
        or plan.get("runtime_domain_guard") is not False
        or plan.get("input_bytes") != 1024 or plan.get("output_bytes") != 2048
        or plan.get("output_physical_layouts") != ["bf16_pair_halves"]
        or plan.get("output_addresses") != [plan.get("output_address")]
    ):
        raise ValueError("ordered-K source windows or physical ABI differ")
    sources = [slices[0].get("source_input"), slices[1].get("source_input")]
    if (any(type(name) is not str or not name or not name.isascii()
            or any(not (character.isalnum() or character == "_") for character in name)
            for name in sources) or sources[0] == sources[1]):
        raise ValueError("ordered-K logical sources are malformed")
    windows = []
    for index, (item, address) in enumerate(zip(slices, addresses)):
        k_index = index // 2
        expected_count = head_count if has_head_pad and k_index == 0 else 32
        expected_start = (0 if k_index == 0 else
                          head_count + (k_index - 1) * 32 if has_head_pad
                          else k_index * 32)
        if (
            not isinstance(item, dict)
            or item.get("source_input") != sources[index % 2]
            or item.get("tile_input") != (f"{sources[index % 2]}__k{index // 2}"
                                          if has_k_partition else sources[index % 2])
            or item.get("rows") != 32
            or item.get("source_columns") != source_columns
            or item.get("k_start") != expected_start
            or item.get("tile_columns") != 32
            or (has_head_pad and (
                item.get("source_count") != expected_count
                or item.get("destination_column_start") != (
                    32 - head_count if k_index == 0 else 0)))
            or (not has_head_pad and (
                "source_count" in item or "destination_column_start" in item))
            or type(address) is not int or address <= 0 or address + 1024 > 1 << 32
        ):
            raise ValueError("ordered-K input placement does not match source windows")
        windows.append((index % 2, address, item, expected_count))
    output = plan["output_address"]
    spans = [(address, 1024) for _, address, _, _ in windows] + [(output, 2048)]
    if (
        type(output) is not int or output <= 0 or output + 2048 > 1 << 32
        or any(left < right + right_size and right < left + left_size
               for index, (left, left_size) in enumerate(spans)
               for right, right_size in spans[index + 1:])
    ):
        raise ValueError("ordered-K physical inputs and output overlap")
    words = struct.unpack(f"<{len(binary) // 4}I", binary)
    selected_boundary = None
    selected_codebook = None
    if f32_boundary is not None:
        if f32_boundary != "diagnostic-mxu0-f32":
            raise ValueError("unknown ordered-K f32 boundary policy")
        from atlas_mlir_native.binding import diagnostic_mxu0_f32_profile
        from merlin.semantic_compiler.precision_codegen import (
            finite_f32_codebook,
            render_finite_f32_codebook_c,
        )

        profile = diagnostic_mxu0_f32_profile()
        selected_boundary = profile.conversion_specs[0]
        if (
            profile.target_identity != target
            or profile.result_policy != plan["numerical_policy"]
            or profile.conversion_specs[1] != selected_boundary
            or [list(pair) for pair in selected_boundary.admitted_code_ranges]
            != [[0, 0], [8, 119], [136, 247]]
        ):
            raise ValueError("selected f32 boundary differs from the verified raw-FP8 program")
        selected_codebook = finite_f32_codebook(selected_boundary)
    f32_lines = []
    if selected_codebook is not None:
        f32_lines = [
            "#define ATLAS_ORDERED_K_SOURCE_F32_DIAGNOSTIC 1",
            f'#define ATLAS_ORDERED_K_F32_POLICY "{selected_boundary.policy}"',
            *render_finite_f32_codebook_c("atlas_ordered_k_f32_codebook", selected_codebook),
            "static inline int atlas_ordered_k_encode_source_f32(",
            "    const float *source, uint8_t *destination) {",
            "  return merlin_pack_f32_u8(source, destination,",
            f"                            32u * {source_columns}u,",
            "                            &atlas_ordered_k_f32_codebook);",
            "}",
        ]
    lines = [
        "/* Generated from a verified diagnostic Merlin-native Atlas program. */",
        "#ifndef ATLAS_ORDERED_K_DATA_H",
        "#define ATLAS_ORDERED_K_DATA_H",
        '#include "atlas_ordered_k.h"',
        '#include "row_major_k_window.h"',
        '#include "finite_float_boundary.h"',
        f'#define ATLAS_ORDERED_K_PROGRAM_SHA256 "{_sha(binary)}"',
        f'#define ATLAS_ORDERED_K_RTL_REVISION "{revision}"',
        '#define ATLAS_ORDERED_K_RUNTIME_DOMAIN "positive-zero-or-e4m3-exp1-to14"',
        "#define ATLAS_ORDERED_K_DIAGNOSTIC_ONLY 1",
        f"#define ATLAS_ORDERED_K_BITSTREAM_BOUND {int(bool(selected_map.get('bitstream_identity')))}",
        f"#define ATLAS_ORDERED_K_PROGRAM_WORD_COUNT {len(words)}u",
        f"#define ATLAS_ORDERED_K_WINDOW_COUNT {len(windows)}u",
        f"#define ATLAS_ORDERED_K_OUTPUT_ADDRESS 0x{output:x}u",
        "static const uint32_t atlas_ordered_k_program_words[ATLAS_ORDERED_K_PROGRAM_WORD_COUNT] = {",
        *(f"    0x{word:08x}u," for word in words),
        "};",
        "static const atlas_ordered_k_window atlas_ordered_k_windows[ATLAS_ORDERED_K_WINDOW_COUNT] = {",
        *(f"    {{{source}, 32u, {source_columns}u, {item['k_start']}u, "
          f"{source_count}u, 0x{address:x}u}},"
          for source, address, item, source_count in windows),
        "};",
        "static const atlas_host_map atlas_ordered_k_host_map = {",
        f"    0x{selected_map['imem_base']:x}u, 0x{selected_map['csr_base']:x}u,",
        f"    0x{selected_map['imem_capacity_bytes']:x}u",
        "};",
        "#ifndef ATLAS_DIAGNOSTIC_ORDERED_K",
        '#error "Ordered-K host data is diagnostic until selected-SoC qualification"',
        "#endif",
        "static inline int atlas_ordered_k_bind(",
        "    atlas_ordered_k_tile *tile, atlas_host *host, void *memory_context,",
        "    atlas_ordered_k_write write, atlas_ordered_k_read read,",
        "    size_t max_polls, uint8_t *scratch, size_t scratch_bytes,",
        "    atlas_host_buffer *buffers, size_t buffer_capacity) {",
        "  if (!tile || !host || host->active || !write || !read || !max_polls ||",
        "      !scratch || scratch_bytes < 2048u || !buffers ||",
        "      buffer_capacity < ATLAS_ORDERED_K_WINDOW_COUNT + 1u ||",
        "      !atlas_ordered_k_binding_valid(tile, host, scratch, scratch_bytes, buffers, buffer_capacity))",
        "    return 1;",
        "  *tile = (atlas_ordered_k_tile){",
        "      .host = host, .memory_context = memory_context,",
        "      .write = write, .read = read,",
        "      .pack = merlin_pack_row_major_k_window,",
        "      .program_words = atlas_ordered_k_program_words,",
        "      .program_word_count = ATLAS_ORDERED_K_PROGRAM_WORD_COUNT,",
        "      .windows = atlas_ordered_k_windows,",
        "      .window_count = ATLAS_ORDERED_K_WINDOW_COUNT,",
        "      .output_address = ATLAS_ORDERED_K_OUTPUT_ADDRESS,",
        "      .max_polls = max_polls, .scratch = scratch,",
        "      .scratch_bytes = scratch_bytes,",
        "      .active_buffers = buffers, .active_buffer_capacity = buffer_capacity,",
        "  };",
        "  return 0;",
        "}",
        "static inline int atlas_ordered_k_readout_f32(",
        "    const volatile uint8_t *physical, float *logical) {",
        "  return merlin_unpack_split_bf16_f32(physical, logical, 32u, 32u, 16u);",
        "}",
        *f32_lines,
        "#endif",
        "",
    ]
    record = {
        "schema": "atlas.ordered_k_host_data.v1",
        "status": "diagnostic_unbound" if not selected_map.get("bitstream_identity")
                  else "diagnostic_unqualified",
        "engine": "merlin_native",
        "source_request_digest": source_request_digest,
        "request_digest": manifest["request_digest"],
        "program_sha256": _sha(binary),
        "native_manifest_sha256": _sha(manifest_bytes),
        "native_verification_sha256": _sha(verification.read_bytes()),
        "host_map_sha256": _sha(host_map.read_bytes()),
        "packer_sha256": _sha(packer_bytes),
        "boundary_sha256": _sha(boundary_bytes),
        "window_count": len(windows),
        "source_inputs": sources,
        "output_address": output,
        "runtime_input_domain": "positive-zero-or-e4m3-exp1-to14",
        "runtime_admitted_code_ranges": [[0, 0], [8, 119], [136, 247]],
        "source_domain_coverage": "runtime_subset_unqualified",
    }
    if selected_boundary is not None and selected_codebook is not None:
        record["f32_boundary"] = {
            "status": "diagnostic_approximation_unqualified_model_quality",
            "source_dtype": "f32",
            "conversion": selected_boundary.record(),
            "codebook": selected_codebook.record(),
        }
    return "\n".join(lines), record, packer_bytes, boundary_bytes


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--compiled", type=Path, required=True)
    parser.add_argument("--verification", type=Path, required=True)
    parser.add_argument("--host-map", type=Path, required=True)
    parser.add_argument("--packer-header", type=Path, required=True,
                        help="selected Merlin row_major_k_window.h to bundle")
    parser.add_argument("--boundary-header", type=Path, required=True,
                        help="selected Merlin finite_float_boundary.h to bundle")
    parser.add_argument("--f32-boundary", choices=("diagnostic-mxu0-f32",),
                        help="explicit diagnostic f32-to-FP8 approximation for host inputs")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        parser.error("fresh output directory required")
    try:
        header, record, packer_bytes, boundary_bytes = prepare(
            args.compiled, args.verification, args.host_map,
            args.packer_header, args.boundary_header, args.f32_boundary)
    except (OSError, ValueError, TypeError, KeyError) as error:
        parser.error(str(error))
    record["header_sha256"] = _sha(header.encode())
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="ordered-k-host-", dir=args.out.parent) as temporary:
        staged = Path(temporary) / "host-data"
        staged.mkdir()
        (staged / "atlas_ordered_k_data.h").write_text(header)
        (staged / "row_major_k_window.h").write_bytes(packer_bytes)
        (staged / "finite_float_boundary.h").write_bytes(boundary_bytes)
        (staged / "manifest.json").write_text(json.dumps(record, sort_keys=True, indent=2) + "\n")
        staged.rename(args.out)
    print(json.dumps({"status": record["status"], "out": str(args.out)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
