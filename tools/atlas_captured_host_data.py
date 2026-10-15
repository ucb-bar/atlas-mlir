#!/usr/bin/env python3
"""Emit fixed DRAM buffers from a checked captured or native Atlas tile."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path


def source_tile_window(precision: dict) -> tuple[int, int, int]:
    """Validate a complete, head-padded single-tile f32 matmul boundary."""
    inputs = precision.get("inputs")
    output = precision.get("output")
    if not isinstance(inputs, list) or len(inputs) != 2 or not all(
        isinstance(item, dict) for item in inputs
    ) or not isinstance(output, dict):
        raise ValueError("native tile has no two-input source window")
    shapes = [item.get("source_type", {}).get("shape") for item in inputs]
    shape = output.get("source_type", {}).get("shape")
    if not all(
        isinstance(item, list) and len(item) == 2 and
        all(type(dimension) is int and 1 <= dimension <= 32 for dimension in item)
        for item in [*shapes, shape]
    ):
        raise ValueError("native source window exceeds one physical tile")
    m, k = shapes[0]
    rhs_k, n = shapes[1]
    if rhs_k != k or shape != [m, n]:
        raise ValueError("native source matmul dimensions do not match")
    for index, item in enumerate(inputs):
        expected_pad = ("positive_zero_f32" if
                        (m if index == 0 else n, k) != (32, 32) else None)
        if (item.get("source_role") != ("lhs", "rhs")[index]
                or item.get("source_offsets") != [0, 0]
                or item.get("source_window_shape") != shapes[index]
                or item.get("target_offsets") != [0, 32 - k]
                or item.get("target_axes_from_source") != ([0, 1] if index == 0 else [1, 0])
                or item.get("padding_value") != expected_pad
                or item.get("target_type", {}).get("shape") != [32, 32]):
            raise ValueError("native source input window differs from the physical tile")
    if (output.get("source_role") != "result"
            or output.get("source_offsets") != [0, 0]
            or output.get("source_window_shape") != [m, n]
            or output.get("target_offsets") != [0, 0]
            or output.get("target_axes_from_source") != [0, 1]
            or output.get("target_type", {}).get("shape") != [32, 32]):
        raise ValueError("native source output window has no exact BF16-to-f32 readback")
    return m, n, k


def _native_import(compiled: Path, verification_path: Path) -> dict:
    """Adapt one verified native tile to the existing fixed-address host handoff."""
    manifest_bytes = (compiled / "manifest.json").read_bytes()
    manifest = json.loads(manifest_bytes)
    plan_bytes = (compiled / "execution_plan.json").read_bytes()
    plan = json.loads(plan_bytes)
    verified = json.loads(verification_path.read_text())
    target = manifest.get("target_identity")
    match = re.fullmatch(r"atlas-rtl-([0-9a-f]{40}):[a-z0-9-]+", target or "")
    if (manifest.get("schema") != "atlas.native_tile_compilation.v6"
            or manifest.get("engine") != "merlin_native"
            or manifest.get("artifact_kind") != "program"
            or manifest.get("compile_mode") != "diagnostic"
            or manifest.get("qualification_scope") != "diagnostic_only"
            or not match or plan.get("schema") != "atlas.native_tile_execution_plan.v8"
            or manifest.get("execution_plan_sha256") != hashlib.sha256(plan_bytes).hexdigest()):
        raise ValueError("native host handoff requires a checked diagnostic Atlas tile")
    program = compiled / "program.bin"
    if (program.is_symlink() or not program.is_file()
            or hashlib.sha256(program.read_bytes()).hexdigest() != manifest.get("binary_sha256")):
        raise ValueError("native program differs from compiler manifest")
    if (verified.get("schema") != "merlin.native_artifact_verification.v1"
            or verified.get("status") != "verified_artifacts"
            or verified.get("engine") != "merlin_native"
            or verified.get("source_kind") not in (
                "source_bound_diagnostic_tile", "typed_request")
            or verified.get("selected_record_checked") is not True
            or Path(verified.get("artifact", "")).resolve() != compiled.resolve()
            or verified.get("request_digest") != manifest.get("request_digest")
            or verified.get("target_identity") != target
            or verified.get("binary_sha256") != manifest.get("binary_sha256")
            or verified.get("manifest_sha256") != hashlib.sha256(manifest_bytes).hexdigest()):
        raise ValueError("native verification does not match the selected program")
    if verified["source_kind"] == "typed_request":
        inputs = plan.get("input_addresses")
        input_sizes = plan.get("input_payload_bytes")
        outputs = plan.get("output_addresses")
        layouts = plan.get("output_physical_layouts")
        constants = manifest.get("constant_inputs", [])
        checked = verified.get("checked_stage_artifacts", [])
        tracked = manifest.get("stage_artifacts", {})
        if (not isinstance(constants, list) or len(constants) > 16
                or not isinstance(checked, list) or not isinstance(tracked, dict)
                or plan.get("constant_inputs", []) != constants):
            raise ValueError("typed native constants differ from the checked execution plan")
        if (plan.get("shape") != [32, 32] or plan.get("dtype") != "bf16"
                or not isinstance(inputs, list) or not 1 <= len(inputs) <= 16
                or not isinstance(input_sizes, list) or len(input_sizes) != len(inputs)
                or any(size not in (1024, 2048) for size in input_sizes)
                or not isinstance(outputs, list) or not 1 <= len(outputs) <= 16
                or layouts != ["bf16_pair_halves"] * len(outputs)
                or plan.get("output_bytes") != 2048
                or plan.get("input_address") != inputs[0]
                or plan.get("output_address") != outputs[0]
                or plan.get("output_numerical_policies") is None
                or len(plan["output_numerical_policies"]) != len(outputs)
                or any(type(address) is not int or address < 0 or address % 32
                       for address in (*inputs, *outputs))
                or any(address != outputs[0] + index * 2048
                       for index, address in enumerate(outputs))):
            raise ValueError("typed native tile has an unsupported physical ABI")
        windows = [(address, address + size) for address, size in zip(inputs, input_sizes)]
        windows.extend((address, address + 2048) for address in outputs)
        if any(left[0] < right[1] and right[0] < left[1]
               for index, left in enumerate(windows)
               for right in windows[index + 1:]):
            raise ValueError("typed native tile has overlapping physical buffers")
        constant_by_address = {}
        rows = []
        for row in constants:
            if not isinstance(row, dict) or set(row) != {
                "node_id", "address", "bytes", "encoding", "physical_layout", "path", "sha256",
            }:
                raise ValueError("typed native constant has malformed placement")
            address, path = row["address"], row["path"]
            if (type(address) is not int or address not in inputs
                    or address in constant_by_address
                    or input_sizes[inputs.index(address)] != 2048
                    or row["bytes"] != 2048 or row["encoding"] != "bf16-le-raw"
                    or row["physical_layout"] != "bf16_pair_halves"
                    or not isinstance(row["node_id"], str) or not row["node_id"]
                    or not isinstance(path, str)
                    or path != f"constants/constant-{len(rows):03d}.bin"
                    or path not in checked or tracked.get(path) != row["sha256"]):
                raise ValueError("typed native constant is not checked at its selected address")
            constant_by_address[address] = row
            rows.append({
                "slot": inputs.index(address), "source_key": row["node_id"],
                "payload_file": path, "payload_bytes": 2048,
                "payload_sha256": row["sha256"], "dram_address": address,
            })
        if sum(size == 2048 for size in input_sizes) != len(rows):
            raise ValueError("typed native BF16 input lacks a declared constant payload")
        return {
            "schema": "atlas.oot.captured_linalg_import.v1",
            "selected_rtl_revision": match.group(1),
            "program": {"program_sha256": manifest["binary_sha256"]},
            "input_slots": [
                {"slot": index,
                 "source_kind": "generated" if address in constant_by_address else "input",
                 "source_key": constant_by_address[address]["node_id"]
                 if address in constant_by_address else None,
                 "dram_address": address, "payload_bytes": size}
                for index, (address, size) in enumerate(zip(inputs, input_sizes))
            ],
            "outputs": [
                {"index": index, "dram_address": address,
                 "payload_bytes": 2048, "physical_layout": "bf16_pair_halves"}
                for index, address in enumerate(outputs)
            ],
            "constants": rows,
        }
    constants = manifest.get("constant_inputs")
    if not isinstance(constants, list) or len(constants) > 1:
        raise ValueError("native host handoff supports zero or one declared constant")
    addresses = plan.get("input_addresses")
    size = plan.get("input_bytes")
    output = plan.get("output_address")
    output_size = plan.get("output_bytes")
    if (not isinstance(addresses, list) or len(addresses) != 2
            or any(type(address) is not int or address < 0 for address in addresses)
            or addresses[0] == addresses[1]
            or type(size) is not int or size <= 0
            or type(output) is not int or output < 0
            or type(output_size) is not int or output_size <= 0
            or plan.get("output_physical_layouts") != ["bf16_pair_halves"]
            or plan.get("output_addresses") != [output]):
        raise ValueError("native tile has an unsupported fixed-address input/output ABI")
    rows = []
    constant_addresses = set()
    checked = verified.get("checked_stage_artifacts", [])
    tracked = manifest.get("stage_artifacts", {})
    for row in constants:
        if not isinstance(row, dict) or row.get("status") != "packed":
            raise ValueError("native constant is blocked or malformed")
        path = row.get("path")
        address = row.get("address")
        role = row.get("role")
        expected_address = addresses[0] if role == "lhs" else addresses[1] if role == "rhs" else None
        if (path != f"constants/{role}.bin"
                or role not in ("lhs", "rhs")
                or path not in checked or path not in tracked
                or type(address) is not int or address != expected_address
                or address in constant_addresses
                or row.get("bytes") != size or row.get("sha256") != tracked[path]):
            raise ValueError("native constant does not match selected input placement")
        constant_addresses.add(address)
        rows.append({"payload_file": path, "payload_bytes": size,
                     "payload_sha256": row["sha256"], "dram_address": address})
    runtime = [address for address in addresses if address not in constant_addresses]
    if len(runtime) not in (1, 2) or size != 1024 or output_size != 2048:
        raise ValueError("native host handoff requires one or two runtime FP8 tiles and one BF16 output")
    precision_bytes = (compiled / "precision-plan.json").read_bytes()
    if (tracked.get("precision-plan.json") != hashlib.sha256(precision_bytes).hexdigest()
            or "precision-plan.json" not in checked):
        raise ValueError("native runtime conversion differs from the verified precision plan")
    precision = json.loads(precision_bytes)
    m, n, k = source_tile_window(precision)
    conversions = precision.get("inputs")
    if not isinstance(conversions, list) or len(conversions) != 2:
        raise ValueError("native precision plan has no two-input conversion")
    for index, conversion in enumerate(conversions):
        if (conversion.get("source_role") != ("lhs" if index == 0 else "rhs")
                or conversion.get("source_type", {}).get("dtype") != "f32"
                or conversion.get("source_type", {}).get("shape") !=
                ([m, k] if index == 0 else [k, n])
                or conversion.get("target_type", {}).get("dtype") != "fp8_e4m3"
                or conversion.get("target_type", {}).get("shape") != [32, 32]
                or conversion.get("target_axes_from_source") not in ([0, 1], [1, 0])):
            raise ValueError("native runtime input needs a supported f32 to FP8 boundary")
    from atlas_mlir_native.precision_abi import BF16_HALF_COLUMNS, TILE_EDGE
    from merlin.semantic_compiler.precision_codegen import finite_f32_codebook
    from merlin.semantic_compiler.precision_values import FloatBoundarySpec

    tables = []
    for conversion in conversions:
        raw_spec = conversion.get("conversion_spec")
        if not isinstance(raw_spec, dict) or conversion.get("conversion_policy") != raw_spec.get("policy"):
            raise ValueError("native runtime input has no selected conversion policy")
        ranges = raw_spec.get("admitted_code_ranges")
        if not isinstance(ranges, list):
            raise TypeError("native runtime conversion has malformed admitted codes")
        spec = FloatBoundarySpec(
            raw_spec["policy"], raw_spec["format"], raw_spec["direction"],
            min_normal_exponent_code=raw_spec["min_normal_exponent_code"],
            max_normal_exponent_code=raw_spec["max_normal_exponent_code"],
            flush_below_normal=raw_spec["flush_below_normal"],
            admitted_code_ranges=tuple(tuple(pair) for pair in ranges),
            pre_encode_scale_exponent=raw_spec.get("pre_encode_scale_exponent", 0),
            canonicalize_negative_zero=raw_spec.get("canonicalize_negative_zero", False),
        )
        if spec.record() != raw_spec:
            raise ValueError("native runtime conversion contains unknown or changed fields")
        tables.append(finite_f32_codebook(spec).record())
    readback = precision.get("output")
    if (not isinstance(readback, dict)
            or readback.get("source_role") != "result"
            or readback.get("source_type", {}).get("dtype") != "f32"
            or readback.get("source_type", {}).get("shape") != [m, n]
            or readback.get("target_type", {}).get("dtype") != "bf16"
            or readback.get("target_type", {}).get("shape") != [TILE_EDGE, TILE_EDGE]
            or readback.get("target_type", {}).get("numerical_policy") != plan.get("numerical_policy")
            or readback.get("target_axes_from_source") != [0, 1]
            or readback.get("conversion_policy") != "bf16-to-f32-exact"
            or readback.get("conversion_spec") != FloatBoundarySpec(
                "bf16-to-f32-exact", "bf16", "decode",
            ).record()):
        raise ValueError("native result needs the selected exact BF16-to-f32 readback")
    return {
        "schema": "atlas.oot.captured_linalg_import.v1",
        "selected_rtl_revision": match.group(1),
        "program": {"program_sha256": manifest["binary_sha256"]},
        "input_slots": [{"source_kind": "input", "dram_address": address,
                         "payload_bytes": size} for address in runtime],
        "output": {"dram_address": output, "payload_bytes": output_size},
        "constants": rows,
        "runtime_conversions": tables,
        "result_readback": {"rows": TILE_EDGE, "columns": TILE_EDGE,
                            "first_columns": BF16_HALF_COLUMNS},
    }


def render(compiled: Path, capsule: Path, *, native_verification: Path | None = None) -> str:
    imported_path = compiled / "import-manifest.json"
    if imported_path.exists() or imported_path.is_symlink():
        if native_verification is not None or (compiled / "manifest.json").exists():
            raise ValueError("host handoff must select one compiler artifact format")
        imported_bytes = imported_path.read_bytes()
        imported = json.loads(imported_bytes)
    else:
        if native_verification is None:
            raise ValueError("native host handoff requires a fresh native-verify receipt")
        imported_bytes = None
        imported = _native_import(compiled, native_verification)
    packed = json.loads((capsule / "manifest.json").read_text())
    if imported.get("schema") != "atlas.oot.captured_linalg_import.v1":
        raise ValueError("unsupported captured Linalg import")
    if packed.get("schema") != "atlas.boot-capsule.v1":
        raise ValueError("unsupported Atlas capsule")
    captured_hash = packed.get("captured_import_manifest_sha256")
    if (captured_hash is not None
            and (imported_bytes is None
                 or hashlib.sha256(imported_bytes).hexdigest() != captured_hash)):
        raise ValueError("captured import changed after capsule packaging")
    if packed.get("captured_output_count") is not None and captured_hash is None:
        raise ValueError("capsule omits captured import identity")
    program_path = capsule / "program.bin"
    if (packed.get("program_file") != "program.bin" or program_path.is_symlink()
            or not program_path.is_file()
            or hashlib.sha256(program_path.read_bytes()).hexdigest() != packed.get("program_sha256")):
        raise ValueError("boot capsule program differs from its manifest")
    if imported.get("selected_rtl_revision") != packed.get("dram", {}).get("selected_rtl_revision"):
        raise ValueError("compiler and capsule select different RTL")
    if imported.get("program", {}).get("program_sha256") != packed.get("program_sha256"):
        raise ValueError("compiler and capsule contain different instruction streams")
    outputs = imported.get("outputs")
    if outputs is None:
        outputs = [imported.get("output")]
    elif imported.get("output") is not None and outputs != [imported["output"]]:
        raise ValueError("captured output descriptions disagree")
    if (not isinstance(outputs, list) or not 1 <= len(outputs) <= 16
            or any(not isinstance(output, dict) for output in outputs)
            or packed.get("captured_output_count", len(outputs)) != len(outputs)
            or (len(outputs) > 1 and imported_bytes is not None
                and "captured_output_count" not in packed)):
        raise ValueError("captured output count differs from capsule")
    declared_slots = imported.get("input_slots")
    if not isinstance(declared_slots, list) or any(
        not isinstance(slot, dict) for slot in declared_slots
    ):
        raise ValueError("host handoff has malformed input slots")
    slots = [slot for slot in declared_slots if slot.get("source_kind") == "input"]
    constant_slots = [slot for slot in declared_slots
                      if slot.get("source_kind") in ("param", "generated")]
    if (not 1 <= len(slots) <= 16
            or len(slots) + len(constant_slots) != len(declared_slots)):
        raise ValueError("host handoff requires one to sixteen runtime input tiles")
    constant_rows = imported.get("constants", [])
    if not isinstance(constant_rows, list) or any(
        not isinstance(row, dict) for row in constant_rows
    ):
        raise ValueError("host handoff has malformed constant rows")
    if constant_slots or any("slot" in row for row in constant_rows):
        declared_by_id = {slot.get("slot"): slot for slot in constant_slots}
        staged_by_id = {row.get("slot"): row for row in constant_rows}
        if (len(declared_by_id) != len(constant_slots)
                or len(staged_by_id) != len(constant_rows)
                or set(declared_by_id) != set(staged_by_id)
                or any(type(identifier) is not int for identifier in declared_by_id)
                or any(slot.get("dram_address") != staged_by_id[identifier].get("dram_address")
                       or slot.get("payload_bytes") != staged_by_id[identifier].get("payload_bytes")
                       or slot.get("source_key") != staged_by_id[identifier].get("source_key")
                       for identifier, slot in declared_by_id.items())):
            raise ValueError("declared constant slot differs from staged payload")
    rows = packed["dram"]["regions"]
    input_ranges = [(r["address"], r["address"] + r["size_bytes"])
                    for r in rows if r["kind"] == "input"]
    output_ranges = [(r["address"], r["address"] + r["size_bytes"])
                     for r in rows if r["kind"] == "output"]
    if len(input_ranges) != 1 or len(output_ranges) != 1:
        raise ValueError("host handoff requires one input/constant region and one output region")
    output_base = outputs[0].get("dram_address")
    if (type(output_base) is not int
            or any(type(output.get("dram_address")) is not int
                   or output["dram_address"] != output_base + index * 2048
                   or output.get("payload_bytes") != 2048
                   or ("index" in output and output["index"] != index)
                   or ("physical_layout" in output
                       and output["physical_layout"] != "bf16_pair_halves")
                   for index, output in enumerate(outputs))
            or any(slot.get("payload_bytes") != 1024
                   or type(slot.get("dram_address")) is not int
                   or not input_ranges[0][0] <= slot["dram_address"]
                   or slot["dram_address"] + 1024 > input_ranges[0][1] for slot in slots)
            or len({slot["dram_address"] for slot in slots}) != len(slots)
            or any(left["dram_address"] < right["dram_address"] + 1024
                   and right["dram_address"] < left["dram_address"] + 1024
                   for index, left in enumerate(slots) for right in slots[index + 1:])
            or not output_ranges[0][0] <= output_base
            or output_base + len(outputs) * 2048 > output_ranges[0][1]):
        raise ValueError("compiler tile addresses escape the packaged layout")
    lines = [
        "/* Generated from a checked Atlas tile. Do not edit. */",
        "#ifndef ATLAS_CAPTURED_DATA_H",
        "#define ATLAS_CAPTURED_DATA_H",
        "#include <stddef.h>",
        "#include <stdint.h>",
        f"#define ATLAS_CAPTURED_INPUT_COUNT {len(slots)}u",
        "#define ATLAS_CAPTURED_INPUT_BYTES 1024u",
        f"#define ATLAS_CAPTURED_INPUT_REGION_ADDRESS 0x{input_ranges[0][0]:08x}u",
        f"#define ATLAS_CAPTURED_INPUT_REGION_BYTES {input_ranges[0][1] - input_ranges[0][0]}u",
        f"#define ATLAS_CAPTURED_OUTPUT_COUNT {len(outputs)}u",
        "#define ATLAS_CAPTURED_OUTPUT_BYTES 2048u",
        "typedef struct { uintptr_t address; size_t size; const uint8_t *data; } atlas_init_blob;",
        "typedef struct { uintptr_t address; size_t size; } atlas_output_slot;",
    ]
    for index, output in enumerate(outputs):
        lines.append(f"#define ATLAS_CAPTURED_OUTPUT_ADDRESS_{index} "
                     f"0x{output['dram_address']:08x}u")
    if len(outputs) == 1:
        lines.append("#define ATLAS_CAPTURED_OUTPUT_ADDRESS ATLAS_CAPTURED_OUTPUT_ADDRESS_0")
    lines.append("static const atlas_output_slot "
                 "atlas_captured_outputs[ATLAS_CAPTURED_OUTPUT_COUNT] = {")
    lines.extend(f"    {{ATLAS_CAPTURED_OUTPUT_ADDRESS_{index}, "
                 "ATLAS_CAPTURED_OUTPUT_BYTES},"
                 for index in range(len(outputs)))
    lines.append("};")
    if len(slots) == 1:
        lines.insert(5, f"#define ATLAS_CAPTURED_INPUT_ADDRESS 0x{slots[0]['dram_address']:08x}u")
    else:
        for index, slot in enumerate(slots):
            lines.insert(5 + index, f"#define ATLAS_CAPTURED_INPUT_ADDRESS_{index} "
                                    f"0x{slot['dram_address']:08x}u")
    runtime_conversions = imported.get("runtime_conversions")
    if runtime_conversions is None and imported.get("runtime_conversion") is not None:
        runtime_conversions = [imported["runtime_conversion"]] * 2
    if runtime_conversions is not None:
        if not isinstance(runtime_conversions, list) or len(runtime_conversions) != 2:
            raise ValueError("native host handoff needs two selected runtime conversions")
        from merlin.semantic_compiler.precision_codegen import (
            FiniteF32Codebook,
            render_finite_f32_codebook_c,
        )

        lines.extend(['#include "finite_float_boundary.h"',
                      "#define ATLAS_CAPTURED_RUNTIME_INPUT_F32 1u"])
        for index, runtime_conversion in enumerate(runtime_conversions):
            if not isinstance(runtime_conversion, dict):
                raise TypeError("native host handoff has a malformed runtime conversion")
            table = FiniteF32Codebook(
                tuple(runtime_conversion["positive_codes"]),
                tuple(runtime_conversion["midpoint_bits"]),
                runtime_conversion["zero_gate_bits"],
                tuple(runtime_conversion["admitted_mask"]),
                runtime_conversion.get("canonicalize_negative_zero", False),
            )
            lines.extend(render_finite_f32_codebook_c(f"atlas_runtime_codebook_{index}", table))
            lines.extend([
                f"static inline int atlas_pack_runtime_input_{index}_f32(const float *source, uint8_t *destination) {{",
                "    return merlin_pack_f32_u8(source, destination, ATLAS_CAPTURED_INPUT_BYTES,",
                f"                              &atlas_runtime_codebook_{index});",
                "}",
            ])
        if runtime_conversions[0] == runtime_conversions[1]:
            lines.extend([
                "static inline int atlas_pack_runtime_input_f32(const float *source, uint8_t *destination) {",
                "    return atlas_pack_runtime_input_0_f32(source, destination);",
                "}",
            ])
        readback = imported["result_readback"]
        lines.extend([
            "static inline int atlas_unpack_output_bf16_f32(const volatile uint8_t *source,",
            "                                               float *destination) {",
            "    return merlin_unpack_split_bf16_f32(source, destination,",
            f"                                        {readback['rows']}u, {readback['columns']}u,",
            f"                                        {readback['first_columns']}u);",
            "}",
        ])
    blobs = []
    occupied = [(slot["dram_address"], slot["dram_address"] + 1024) for slot in slots]
    for index, row in enumerate(constant_rows):
        path = compiled / row["payload_file"]
        if not path.resolve().is_relative_to(compiled.resolve()):
            raise ValueError("constant path escapes compiler output")
        data = path.read_bytes()
        if len(data) != row["payload_bytes"] or hashlib.sha256(data).hexdigest() != row["payload_sha256"]:
            raise ValueError("constant payload differs from compiler manifest")
        address = row["dram_address"]
        end = address + len(data)
        if not input_ranges[0][0] <= address or end > input_ranges[0][1] or any(
                address < hi and lo < end for lo, hi in occupied):
            raise ValueError("constant placement overlaps or escapes input region")
        occupied.append((address, end))
        name = f"atlas_constant_{index}"
        lines.append(f"static const uint8_t {name}[{len(data)}] = {{")
        lines.extend("    " + ", ".join(f"0x{byte:02x}" for byte in data[start:start + 16]) + ","
                     for start in range(0, len(data), 16))
        lines.append("};")
        blobs.append((name, address, len(data)))
    lines.append(f"#define ATLAS_CAPTURED_CONSTANT_COUNT {len(blobs)}u")
    if blobs:
        lines.append("static const atlas_init_blob atlas_captured_constants[ATLAS_CAPTURED_CONSTANT_COUNT] = {")
        lines.extend(f"    {{0x{address:08x}u, {size}u, {name}}},"
                     for name, address, size in blobs)
        lines.append("};")
    lines.extend(["#endif", ""])
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--compiled", type=Path, required=True)
    parser.add_argument("--capsule", type=Path, required=True)
    parser.add_argument("--native-verification", type=Path,
                        help="native-verify receipt for a source-bound native tile")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    try:
        if args.out.exists():
            raise ValueError("output header already exists")
        content = render(args.compiled, args.capsule,
                         native_verification=args.native_verification)
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(content)
    except (OSError, ValueError, KeyError, TypeError, ImportError, json.JSONDecodeError) as error:
        print(json.dumps({"status": "FAIL", "error": str(error)}), file=sys.stderr)
        return 1
    print(json.dumps({"status": "PASS", "out": str(args.out)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
