#!/usr/bin/env python3
"""Bundle a verified source-bound f32 matmul grid for the Atlas host runtime."""

from __future__ import annotations

import argparse
import json
import re
import struct
import tempfile
from pathlib import Path

if __package__:
    from .atlas_ordered_k_host_data import _member, _sha
else:
    from atlas_ordered_k_host_data import _member, _sha

ROOT = Path(__file__).resolve().parents[1]
RUNTIME_FILES = (
    "atlas_host.h", "atlas_host.c", "atlas_ordered_k.h", "atlas_ordered_k.c",
    "atlas_ordered_mn.h", "atlas_ordered_mn.c", "atlas_source_mn.h",
    "atlas_source_mn.c", "atlas_model_tile.h", "atlas_source_mn_memref.h",
    "atlas_source_mn_memref.c",
)


def _tile_paths(root: Path) -> list[Path]:
    if root.is_symlink() or not root.is_dir():
        raise ValueError("source tile root is absent or linked")
    paths = [path for path in root.iterdir() if re.fullmatch(r"tile-[0-9]+", path.name)]
    paths.sort(key=lambda path: int(path.name[5:]))
    if (not paths or any(path.is_symlink() or not path.is_dir() or
                         path.name != f"tile-{index}" or
                         not (root / f"tile-{index}-verification.json").is_file()
                         for index, path in enumerate(paths))):
        raise ValueError("source output tiles or verification receipts are missing")
    return paths


def _checked_tile(path: Path, receipt_path: Path) -> tuple[dict, dict, dict, bytes, bytes]:
    manifest_bytes = _member(path, "manifest.json")
    plan_bytes = _member(path, "execution_plan.json")
    precision_bytes = _member(path, "precision-plan.json")
    binary = _member(path, "program.bin")
    manifest = json.loads(manifest_bytes)
    plan = json.loads(plan_bytes)
    precision = json.loads(precision_bytes)
    if receipt_path.is_symlink():
        raise ValueError("native verification receipt is linked")
    receipt_bytes = receipt_path.read_bytes()
    receipt = json.loads(receipt_bytes)
    if any(not isinstance(item, dict) for item in (manifest, plan, precision, receipt)):
        raise TypeError("native source tile records must be objects")
    target = manifest.get("target_identity")
    tracked = manifest.get("stage_artifacts", {})
    checked = receipt.get("checked_stage_artifacts", [])
    if (manifest.get("schema") != "atlas.native_tile_compilation.v6"
            or manifest.get("engine") != "merlin_native"
            or manifest.get("artifact_kind") != "program"
            or manifest.get("compile_mode") != "diagnostic"
            or manifest.get("qualification_scope") != "diagnostic_only"
            or not isinstance(manifest.get("constant_inputs"), list)
            or not re.fullmatch(r"atlas-rtl-[0-9a-f]{40}:[a-z0-9-]+", target or "")
            or manifest.get("execution_plan_sha256") != _sha(plan_bytes)
            or manifest.get("binary_sha256") != _sha(binary)
            or not isinstance(tracked, dict)
            or not isinstance(checked, list)
            or len(checked) != len(tracked)
            or set(checked) != set(tracked)
            or tracked.get("precision-plan.json") != _sha(precision_bytes)
            or not binary or len(binary) % 4
            or receipt.get("schema") != "merlin.native_artifact_verification.v1"
            or receipt.get("status") != "verified_artifacts"
            or receipt.get("engine") != "merlin_native"
            or receipt.get("source_kind") != "source_bound_diagnostic_tile"
            or receipt.get("selected_record_checked") is not True
            or Path(receipt.get("artifact", "")).resolve() != path.resolve()
            or receipt.get("manifest_sha256") != _sha(manifest_bytes)
            or receipt.get("binary_sha256") != _sha(binary)
            or receipt.get("request_digest") != manifest.get("request_digest")
            or receipt.get("target_identity") != target
            or "precision-plan.json" not in checked
            or precision.get("request_digest") != manifest.get("request_digest")
            or precision.get("quality_status") != "unqualified"):
        raise ValueError(f"native source tile is unverified or changed: {path.name}")
    if any(_sha(_member(path, name)) != digest for name, digest in tracked.items()):
        raise ValueError(f"native source tile stage changed: {path.name}")
    return manifest, plan, precision, binary, receipt_bytes


def _shape(shape: object, rank: int) -> bool:
    return (isinstance(shape, list) and len(shape) == rank
            and all(type(value) is int and value > 0 for value in shape))


def _source_geometry(precision: dict) -> tuple[int, int, int, int, int]:
    output = precision.get("output", {})
    inputs = precision.get("inputs", [])
    if (not isinstance(output, dict) or not isinstance(inputs, list) or len(inputs) < 2
            or any(not isinstance(item, dict) for item in inputs)):
        raise TypeError("source matmul inputs and output must be records")
    result_type = output.get("source_type", {})
    if (not isinstance(result_type, dict)
            or any(not isinstance(inputs[index].get("source_type"), dict)
                   for index in (0, 1))):
        raise TypeError("source matmul operand types must be records")
    shape = result_type.get("shape")
    rank = len(shape) if isinstance(shape, list) else 0
    if (rank not in (2, 3)
            or not _shape(shape, rank)
            or result_type.get("dtype") != "f32"
            or not _shape(inputs[0].get("source_type", {}).get("shape"), rank)
            or not _shape(inputs[1].get("source_type", {}).get("shape"), rank)):
        raise ValueError("source f32 matmul geometry is missing")
    batch = shape[0] if rank == 3 else 1
    m, n = shape[-2:]
    lhs_shape = inputs[0]["source_type"]["shape"]
    rhs_shape = inputs[1]["source_type"]["shape"]
    lhs_m, k = lhs_shape[-2:]
    rhs_k, rhs_n = rhs_shape[-2:]
    if ((lhs_m, rhs_k, rhs_n) != (m, k, n)
            or (rank == 3 and (lhs_shape[0] != batch or rhs_shape[0] != batch))
            or max(batch * m * k * 4, batch * n * k * 4,
                   batch * m * n * 4) > 0xffffffff or
            2 * ((k + 31) // 32) > 128):
        raise ValueError("source contraction exceeds the checked host ABI")
    return batch, m, n, k, rank


def _conversion_specs(precision: dict, batch: int, m: int, n: int, k: int,
                      rank: int, batch_index: int, m_start: int,
                      n_start: int) -> tuple[dict, dict]:
    from merlin.semantic_compiler.precision_values import FloatBoundarySpec

    count = (k + 31) // 32
    inputs = precision["inputs"]
    output = precision["output"]
    batch_prefix = [batch] if rank == 3 else []
    source_batch_index = [batch_index] if rank == 3 else []
    valid_m, valid_n = min(32, m - m_start), min(32, n - n_start)
    if len(inputs) != 2 * count:
        raise ValueError("source K windows are missing")
    role_specs = []
    for index, item in enumerate(inputs):
        operand, segment = index % 2, index // 2
        first_count = k % 32 or 32
        segment_start = 0 if segment == 0 else first_count + (segment - 1) * 32
        segment_count = first_count if segment == 0 else 32
        role = ("lhs", "rhs")[operand]
        expected_role = f"{role}__k{segment}" if k > 32 else role
        source_offsets = ([m_start, segment_start] if operand == 0
                          else [segment_start, n_start])
        window_shape = ([valid_m, segment_count] if operand == 0
                        else [segment_count, valid_n])
        target_axes = [0, 1] if operand == 0 else [1, 0]
        target_window_shape = [window_shape[axis] for axis in target_axes]
        expected_pad = ("positive_zero_f32" if target_window_shape != [32, 32]
                        else None)
        raw = item.get("conversion_spec")
        if not isinstance(raw, dict) or not isinstance(raw.get("admitted_code_ranges"), list):
            raise TypeError("source FP8 policy is absent or malformed")
        try:
            spec = FloatBoundarySpec(
                raw["policy"], raw["format"], raw["direction"],
                min_normal_exponent_code=raw["min_normal_exponent_code"],
                max_normal_exponent_code=raw["max_normal_exponent_code"],
                flush_below_normal=raw["flush_below_normal"],
                admitted_code_ranges=tuple(tuple(pair) for pair in raw["admitted_code_ranges"]),
                pre_encode_scale_exponent=raw.get("pre_encode_scale_exponent", 0),
                canonicalize_negative_zero=raw.get("canonicalize_negative_zero", False),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("source FP8 policy cannot be reconstructed") from exc
        if (item.get("source_role") != expected_role
                or item.get("status") != "unmaterialized_boundary_obligation"
                or item.get("source_type", {}).get("shape") !=
                (batch_prefix + ([m, k] if operand == 0 else [k, n]))
                or item.get("source_type", {}).get("dtype") != "f32"
                or item.get("source_batch_index", []) != source_batch_index
                or item.get("target_type", {}).get("shape") != [32, 32]
                or item.get("target_type", {}).get("dtype") != "fp8_e4m3"
                or item.get("source_offsets") != source_offsets
                or item.get("source_window_shape") != window_shape
                or item.get("target_offsets") != [0, 32 - segment_count]
                or item.get("target_axes_from_source") != target_axes
                or item.get("padding_value") != expected_pad
                or item.get("conversion_policy") != spec.policy
                or spec.record() != raw
                or spec.format != "fp8_e4m3" or spec.direction != "encode"
                or spec.pre_encode_scale_exponent != 0):
            raise ValueError("source FP8 window or conversion differs")
        role_specs.append(raw)
    if (any(role_specs[index] != role_specs[index % 2] for index in range(len(role_specs)))
            or output.get("status") != "unmaterialized_boundary_obligation"
            or output.get("source_role") != "result"
            or output.get("source_type", {}).get("shape") != batch_prefix + [m, n]
            or output.get("source_type", {}).get("dtype") != "f32"
            or output.get("source_batch_index", []) != source_batch_index
            or output.get("target_type", {}).get("shape") != [32, 32]
            or output.get("target_type", {}).get("dtype") != "bf16"
            or output.get("target_type", {}).get("numerical_policy") !=
            "mxu0-e4m3-ordered-bf16-finite-v1"
            or output.get("source_offsets") != [m_start, n_start]
            or output.get("source_window_shape") != [valid_m, valid_n]
            or output.get("target_offsets") != [0, 0]
            or output.get("target_axes_from_source") != [0, 1]
            or output.get("conversion_policy") != "bf16-to-f32-exact"
            or output.get("conversion_spec") != FloatBoundarySpec(
                "bf16-to-f32-exact", "bf16", "decode").record()):
        raise ValueError("source result window or precision differs")
    return role_specs[0], role_specs[1]


def prepare(tiles_root: Path, host_map: Path, packer_header: Path,
            boundary_header: Path, *, staged_root: Path | None = None) -> tuple[dict[str, bytes], dict]:
    from atlas_mlir_native.binding import AtlasNativeBinding
    from merlin.frontends.compile_inputs import load_compile_inputs
    from merlin.semantic_compiler.precision_codegen import (
        finite_f32_codebook,
        render_finite_f32_codebook_c,
    )
    from merlin.semantic_compiler.precision_values import FloatBoundarySpec

    if (packer_header.name != "row_major_k_window.h" or packer_header.is_symlink()
            or boundary_header.name != "finite_float_boundary.h" or boundary_header.is_symlink()):
        raise ValueError("source host bundle needs the selected Merlin C headers")
    packer_bytes, boundary_bytes = packer_header.read_bytes(), boundary_header.read_bytes()
    if not packer_bytes or not boundary_bytes:
        raise ValueError("source host boundary header is empty")
    selected_map = json.loads(host_map.read_text())
    if not isinstance(selected_map, dict):
        raise TypeError("selected host map must be a record")
    paths = _tile_paths(tiles_root)
    staged = load_compile_inputs(staged_root) if staged_root is not None else None
    first_manifest, first_plan, first_precision, binary, _ = _checked_tile(
        paths[0], tiles_root / "tile-0-verification.json")
    batch_count, m, n, k, rank = _source_geometry(first_precision)
    count = (k + 31) // 32
    matrix_tiles = ((m + 31) // 32) * ((n + 31) // 32)
    expected_tiles = batch_count * matrix_tiles
    target = first_manifest["target_identity"]
    revision = target.split(":", 1)[0].removeprefix("atlas-rtl-")
    if (len(paths) != expected_tiles or selected_map.get("schema") != "atlas.host-map.v1"
            or selected_map.get("rtl_revision") != revision
            or not isinstance(selected_map.get("soc_config_source"), str)
            or not selected_map["soc_config_source"]
            or not re.fullmatch(r"[0-9a-f]{40}", selected_map.get("soc_source_revision", ""))
            or any(type(selected_map.get(key)) is not int for key in (
                "imem_base", "csr_base", "imem_capacity_bytes", "start_csr_offset"))
            or selected_map["imem_base"] < 0 or selected_map["imem_base"] % 4
            or selected_map["csr_base"] < 0 or selected_map["csr_base"] % 4
            or selected_map["imem_capacity_bytes"] < len(binary)
            or selected_map["imem_capacity_bytes"] % 4
            or selected_map["start_csr_offset"] != 24):
        raise ValueError("source tiles or selected host map differ")
    addresses = first_plan.get("input_addresses")
    output_address = first_plan.get("output_address")
    if (first_plan.get("schema") != "atlas.native_tile_execution_plan.v8"
            or first_plan.get("operation") != (f"mxu0_k{count}" if count > 1 else "mxu0_reset")
            or first_plan.get("numerical_policy") != "mxu0-e4m3-ordered-bf16-finite-v1"
            or first_plan.get("input_domain") != "finite-normal-e4m3-or-positive-zero"
            or first_plan.get("input_bytes") != 1024
            or first_plan.get("output_bytes") != 2048
            or first_plan.get("output_physical_layouts") != ["bf16_pair_halves"]
            or first_plan.get("runtime_domain_guard") is not False
            or first_plan.get("producer_domain_obligations") != []
            or first_plan.get("output_addresses") != [output_address]
            or not isinstance(addresses, list) or len(addresses) != 2 * count
            or any(type(address) is not int or address <= 0 or address + 1024 > 1 << 32
                   for address in addresses)
            or type(output_address) is not int or output_address <= 0
            or output_address + 2048 > 1 << 32):
        raise ValueError("source Atlas input/output ABI is unsupported")
    expected_roles = [f"{role}__k{segment}" if k > 32 else role
                      for segment in range(count) for role in ("lhs", "rhs")]
    if first_plan.get("boundary_domain_obligations") != [
        {"input": role, "required_domain": "finite-normal-e4m3-or-positive-zero",
         "status": "caller_premise_unchecked"} for role in expected_roles
    ]:
        raise ValueError("source Atlas input-domain obligations differ")
    spans = [(address, 1024) for address in addresses] + [(output_address, 2048)]
    if any(left < right + right_size and right < left + left_size
           for index, (left, left_size) in enumerate(spans)
           for right, right_size in spans[index + 1:]):
        raise ValueError("source Atlas input/output placements overlap")
    identity = tuple(first_precision.get(key) for key in (
        "original_source_sha256", "staged_program_sha256", "precision_policy_sha256",
        "region_id"))
    if (any(not re.fullmatch(r"[0-9a-f]{64}", item or "") for item in identity[:3])
            or not isinstance(identity[3], str) or not identity[3]):
        raise ValueError("source identity is incomplete")
    conversion_specs = None
    source_values = None
    tile_records = []
    frozen_rhs = None
    frozen_source = None
    frozen_mode = None
    n_tiles = (n + 31) // 32
    for index, path in enumerate(paths):
        manifest, plan, precision, current_binary, receipt_bytes = _checked_tile(
            path, tiles_root / f"tile-{index}-verification.json")
        batch_index = index // matrix_tiles
        matrix_index = index % matrix_tiles
        m_start, n_start = (matrix_index // n_tiles) * 32, (matrix_index % n_tiles) * 32
        origin = [m_start, n_start]
        expected_origin = origin if matrix_tiles > 1 else None
        if (manifest.get("target_identity") != target or current_binary != binary
                or plan != first_plan
                or _source_geometry(precision) != (batch_count, m, n, k, rank)
                or tuple(precision.get(key) for key in (
                    "original_source_sha256", "staged_program_sha256",
                    "precision_policy_sha256", "region_id")) != identity
                or precision.get("batch_index", []) != ([batch_index] if rank == 3 else [])
                or precision.get("ordered_k_continuation") is not (k > 32)
                or precision.get("tile_origin") != expected_origin
                or precision.get("schema") != (
                    "merlin.diagnostic_precision_tile.v4" if rank == 3 else
                    "merlin.diagnostic_precision_tile.v3" if k > 32 else
                    "merlin.diagnostic_precision_tile.v2" if expected_tiles > 1 else
                    "merlin.diagnostic_precision_tile.v1")):
            raise ValueError("source tiles do not share one program and source policy")
        specs = _conversion_specs(
            precision, batch_count, m, n, k, rank, batch_index, m_start, n_start,
        )
        values = (
            precision["inputs"][0].get("source_value"),
            precision["inputs"][1].get("source_value"),
            precision["output"].get("source_value"),
        )
        if (any(value is None for value in values)
                or any(precision["inputs"][part].get("source_value") != values[part % 2]
                       for part in range(len(precision["inputs"])))):
            raise ValueError("source value identities are incomplete")
        if source_values is None:
            source_values = values
        elif values != source_values:
            raise ValueError("source output tiles use different graph values")
        if conversion_specs is None:
            conversion_specs = specs
        elif specs != conversion_specs:
            raise ValueError("source output tiles use different FP8 policies")
        constants = manifest["constant_inputs"]
        if frozen_mode is None:
            frozen_mode = bool(constants)
        elif frozen_mode != bool(constants):
            raise ValueError("source tiles mix frozen and runtime RHS")
        if constants and staged is None:
            raise ValueError("declared constants require --compile-inputs")
        if staged is not None:
            tile = AtlasNativeBinding.prepare_precision_tile(
                staged, identity[3], tile_index=index,
            )
            if tile.record() != precision:
                raise ValueError("staged source differs from a verified precision tile")
            prepared = AtlasNativeBinding.prepare_precision_constants(tile, staged_root, plan)
            if [item.record() for item in prepared] != constants:
                raise ValueError("declared constants differ from the verified tile")
            expected_rhs = {f"rhs__k{part}" if k > 32 else "rhs" for part in range(count)}
            if prepared and {item.role for item in prepared} != expected_rhs:
                raise ValueError("source host supports only a complete frozen RHS")
            for item in prepared:
                if item.payload is None or _member(path, item.record()["path"]) != item.payload:
                    raise ValueError("verified frozen RHS payload differs from staged weights")
                source = item.source
                source_record = {key: value for key, value in source.record().items() if key != "role"}
                if frozen_rhs is None:
                    frozen_rhs = source.value
                    frozen_source = source_record
                elif source_record != frozen_source or not (source.value == frozen_rhs).all():
                    raise ValueError("source tiles have different frozen RHS weights")
        tile_records.append({
            "tile": index, "request_digest": manifest["request_digest"],
            "batch_index": batch_index,
            "matrix_tile_index": matrix_index,
            "manifest_sha256": _sha(_member(path, "manifest.json")),
            "verification_sha256": _sha(receipt_bytes),
            "precision_plan_sha256": _sha(_member(path, "precision-plan.json")),
        })
    assert conversion_specs is not None
    codebooks = []
    for raw in conversion_specs:
        spec = FloatBoundarySpec(
            raw["policy"], raw["format"], raw["direction"],
            min_normal_exponent_code=raw["min_normal_exponent_code"],
            max_normal_exponent_code=raw["max_normal_exponent_code"],
            flush_below_normal=raw["flush_below_normal"],
            admitted_code_ranges=tuple(tuple(pair) for pair in raw["admitted_code_ranges"]),
            canonicalize_negative_zero=raw.get("canonicalize_negative_zero", False),
        )
        codebooks.append(finite_f32_codebook(spec))
    words = struct.unpack(f"<{len(binary) // 4}I", binary)
    windows = []
    head = k % 32 or 32
    for segment in range(count):
        start = 0 if segment == 0 else head + (segment - 1) * 32
        columns = head if segment == 0 else 32
        for operand in range(2):
            windows.append((operand, start, columns, addresses[2 * segment + operand]))
    lines = [
        "/* Generated from verified source-bound diagnostic Atlas tiles. */",
        "#ifndef ATLAS_SOURCE_MN_DATA_H",
        "#define ATLAS_SOURCE_MN_DATA_H",
        '#include "atlas_source_mn.h"',
        '#include "row_major_k_window.h"',
        '#include "finite_float_boundary.h"',
        f"#define ATLAS_SOURCE_MN_ROWS {m}u",
        f"#define ATLAS_SOURCE_MN_COLUMNS {n}u",
        f"#define ATLAS_SOURCE_MN_REDUCTION {k}u",
        f"#define ATLAS_SOURCE_MN_BATCHES {batch_count}u",
        f"#define ATLAS_SOURCE_MN_FROZEN_RHS {int(frozen_rhs is not None)}",
        f"#define ATLAS_SOURCE_MN_TILE_COUNT {matrix_tiles}u",
        f"#define ATLAS_SOURCE_MN_WINDOW_COUNT {len(windows)}u",
        f"#define ATLAS_SOURCE_MN_OUTPUT_ADDRESS 0x{output_address:x}u",
        f'#define ATLAS_SOURCE_MN_PROGRAM_SHA256 "{_sha(binary)}"',
        f'#define ATLAS_SOURCE_MN_RTL_REVISION "{revision}"',
        f"#define ATLAS_SOURCE_MN_BITSTREAM_BOUND {int(bool(selected_map.get('bitstream_identity')))}",
        "#define ATLAS_SOURCE_MN_DIAGNOSTIC_ONLY 1",
        "#ifndef ATLAS_DIAGNOSTIC_SOURCE_MN",
        '#error "Source-bound f32 host data is diagnostic until precision and SoC qualification"',
        "#endif",
        f"static const uint32_t atlas_source_mn_program_words[{len(words)}] = {{",
        *(f"    0x{word:08x}u," for word in words),
        "};",
        "static const atlas_ordered_k_window atlas_source_mn_windows[ATLAS_SOURCE_MN_WINDOW_COUNT] = {",
        *(f"    {{{operand}u, 32u, {k}u, {start}u, {columns}u, 0x{address:x}u}},"
          for operand, start, columns, address in windows),
        "};",
        "static const atlas_host_map atlas_source_mn_host_map = {",
        f"    0x{selected_map['imem_base']:x}u, 0x{selected_map['csr_base']:x}u,",
        f"    0x{selected_map['imem_capacity_bytes']:x}u",
        "};",
    ]
    if frozen_rhs is not None:
        if frozen_rhs.shape != ((batch_count, k, n) if rank == 3 else (k, n)):
            raise ValueError("frozen RHS has the wrong source shape")
        import numpy as np

        bits = np.ascontiguousarray(frozen_rhs, dtype="<f4").view("<u4").flat
        lines.extend((
            f"static const uint32_t atlas_source_mn_frozen_rhs_bits[{batch_count * k * n}] = {{",
            *(f"    0x{int(value):08x}u," for value in bits),
            "};",
        ))
    for role, codebook in zip(("lhs", "rhs"), codebooks, strict=True):
        lines.extend(render_finite_f32_codebook_c(f"atlas_source_mn_{role}_codebook", codebook))
        lines.extend((
            f"static inline int atlas_source_mn_encode_{role}(uint32_t bits,",
            "    const void *policy, uint8_t *code) {",
            "  (void)policy;",
            f"  return merlin_f32_u8_encode_bits(bits, &atlas_source_mn_{role}_codebook, code);",
            "}",
        ))
    lines.extend((
        "static inline int atlas_source_mn_bind(",
        "    atlas_ordered_k_tile *tile, atlas_host *host, void *context,",
        "    atlas_ordered_k_write write, atlas_ordered_k_read read,",
        "    size_t max_polls, uint8_t *scratch, size_t scratch_bytes,",
        "    atlas_host_buffer *buffers, size_t buffer_capacity) {",
        "  if (!tile || !host || host->active || !write || !read || !max_polls ||",
        "      !scratch || scratch_bytes < 2048u || !buffers ||",
        "      buffer_capacity < ATLAS_SOURCE_MN_WINDOW_COUNT + 1u ||",
        "      !atlas_ordered_k_binding_valid(tile, host, scratch, scratch_bytes, buffers, buffer_capacity))",
        "    return 1;",
        "  *tile = (atlas_ordered_k_tile){",
        "      .host = host, .memory_context = context, .write = write, .read = read,",
        "      .pack = merlin_pack_row_major_k_window,",
        "      .program_words = atlas_source_mn_program_words,",
        f"      .program_word_count = {len(words)}u,",
        "      .windows = atlas_source_mn_windows,",
        "      .window_count = ATLAS_SOURCE_MN_WINDOW_COUNT,",
        f"      .output_address = 0x{output_address:x}u,",
        "      .max_polls = max_polls, .scratch = scratch,",
        "      .scratch_bytes = scratch_bytes,",
        "      .active_buffers = buffers, .active_buffer_capacity = buffer_capacity,",
        "  };",
        "  return 0;",
        "}",
        "#endif",
        "",
    ))
    files = {
        "atlas_source_mn_data.h": "\n".join(lines).encode(),
        "row_major_k_window.h": packer_bytes,
        "finite_float_boundary.h": boundary_bytes,
    }
    files.update({name: (ROOT / "runtime" / name).read_bytes() for name in RUNTIME_FILES})
    record = {
        "schema": "atlas.source_mn_host_data.v1",
        "status": "diagnostic_unbound" if not selected_map.get("bitstream_identity")
                  else "diagnostic_unqualified",
        "engine": "merlin_native",
        "source_identity": {
            "original_source_sha256": identity[0], "staged_program_sha256": identity[1],
            "precision_policy_sha256": identity[2], "region_id": identity[3],
        },
        "target_identity": target,
        "program_sha256": _sha(binary),
        "host_map_sha256": _sha(host_map.read_bytes()),
        "rank": rank, "batch_count": batch_count, "m_rows": m, "n_rows": n, "k_columns": k,
        "tile_count": expected_tiles,
        "matrix_tile_count": matrix_tiles,
        "window_count": len(windows),
        "frozen_rhs": frozen_source,
        "f32_boundaries": {role: {"spec": raw, "codebook": codebook.record()}
                           for role, raw, codebook in zip(
                               ("lhs", "rhs"), conversion_specs, codebooks, strict=True)},
        "tiles": tile_records,
        "files": {name: _sha(payload) for name, payload in sorted(files.items())},
    }
    return files, record


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tiles-root", type=Path, required=True)
    parser.add_argument("--host-map", type=Path, required=True)
    parser.add_argument("--packer-header", type=Path, required=True)
    parser.add_argument("--boundary-header", type=Path, required=True)
    parser.add_argument("--compile-inputs", type=Path,
                        help="compiler-only staged source, required for frozen RHS weights")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists() or args.out.is_symlink():
        parser.error("fresh output directory required")
    try:
        files, record = prepare(args.tiles_root, args.host_map,
                                args.packer_header, args.boundary_header,
                                staged_root=args.compile_inputs)
    except (OSError, ValueError, TypeError, KeyError, IndexError, json.JSONDecodeError) as exc:
        parser.error(str(exc))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="source-mn-host-", dir=args.out.parent) as temporary:
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
