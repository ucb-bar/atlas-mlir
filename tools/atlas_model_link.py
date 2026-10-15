#!/usr/bin/env python3
"""Link a checked Merlin host object with selected Atlas tile or grid wrappers.

This diagnostic stage produces a relocatable RV64 object. It does not supply
the C runtime, a platform entry point, a bitstream binding, or execution.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import struct
import subprocess
import sys
from pathlib import Path

if __package__:
    from .atlas_model_tile_bridge import render_wrapper
    from .atlas_source_mn_host_data import RUNTIME_FILES as SOURCE_RUNTIME_FILES
    from .atlas_source_mn_model_bridge import prepare as prepare_source_grid_wrapper
    from .check_program_decode import audit_program, decode_binary
    from .check_variant_inventory import load_inventory
else:
    from atlas_model_tile_bridge import render_wrapper
    from atlas_source_mn_host_data import RUNTIME_FILES as SOURCE_RUNTIME_FILES
    from atlas_source_mn_model_bridge import prepare as prepare_source_grid_wrapper
    from check_program_decode import audit_program, decode_binary
    from check_variant_inventory import load_inventory

ROOT = Path(__file__).resolve().parents[1]
FLAGS = (
    "--target=riscv64-unknown-elf", "-march=rv64gcv", "-mabi=lp64d",
    "-mcmodel=medany", "-O2", "-ffreestanding", "-fno-builtin",
)
RUNTIME_UNDEFINED = {"calloc", "expf", "free", "malloc", "memcmp", "memcpy", "memset", "rsqrtf"}


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _symbols(output: str) -> dict[str, str]:
    symbols: dict[str, str] = {}
    for line in output.splitlines():
        fields = line.split()
        if len(fields) < 2 or len(fields[1]) != 1:
            raise ValueError("nm produced a malformed symbol row")
        name, kind = fields[:2]
        if kind.islower():
            # Static tables in separate wrapper translation units may share names.
            continue
        if name in symbols:
            raise ValueError(f"nm produced a duplicate symbol: {name}")
        symbols[name] = kind
    if not symbols:
        raise ValueError("nm produced no symbols")
    return symbols


def _program_header_digest(text: str, *, bitstream_bound: bool) -> str:
    """Read only the fixed generated word-array grammar, never execute C text."""
    sha = re.search(r'(?m)^#define ATLAS_PROGRAM_SHA256 "([0-9a-f]{64})"$', text)
    count = re.search(r"(?m)^#define ATLAS_PROGRAM_WORD_COUNT ([1-9][0-9]*)u$", text)
    bound = re.search(r"(?m)^#define ATLAS_PROGRAM_BITSTREAM_BOUND ([01])$", text)
    array = re.search(
        r"(?ms)^static const uint32_t atlas_program_words\[ATLAS_PROGRAM_WORD_COUNT\] = \{\n(.*?)^\};$",
        text,
    )
    if not sha or not count or not bound or not array or bool(int(bound[1])) != bitstream_bound:
        raise ValueError("wrapper program header has a missing or changed image identity")
    body = array[1]
    words = re.findall(r"0x([0-9a-fA-F]{8})u", body)
    residue = re.sub(r"0x[0-9a-fA-F]{8}u", "", body)
    if (len(words) != int(count[1]) or len(words) > 32768
            or residue.replace(",", "").strip()):
        raise ValueError("wrapper program header has a malformed instruction array")
    digest = hashlib.sha256(struct.pack(f"<{len(words)}I", *(int(word, 16) for word in words))).hexdigest()
    if digest != sha[1]:
        raise ValueError("wrapper program header words differ from its declared image")
    return digest


def _selection(probe: dict, host_check: dict, host_object: Path,
               wrappers: list[tuple[Path, dict]], *, probe_path: Path | None = None,
               source_bundles: list[Path] | None = None) -> list[dict]:
    if (probe.get("schema") != "merlin.native_model_region_probe.v1"
            or probe.get("engine") != "merlin_native"
            or probe.get("status") != "incomplete"):
        raise ValueError("model link needs an incomplete native model probe")
    bridge = probe.get("host_bridge", {})
    selected = bridge.get("external_symbols")
    if (bridge.get("schema") != "merlin.diagnostic_model_host_bridge.v1"
            or bridge.get("status") != "typed_declarations_only"
            or not isinstance(selected, list) or not selected
            or len(selected) != len(set(selected))):
        raise ValueError("model probe has no selected C-interface calls")
    if (host_check.get("schema") != "merlin.native_host_object_check.v1"
            or host_check.get("status") != "object_symbols_checked"
            or host_check.get("selected_device_symbols") != selected
            or host_check.get("required_native_unmet") != bridge.get("required_native_unmet")
            or host_check.get("host_object_sha256") != _sha(host_object)):
        raise ValueError("host object check differs from the selected model calls")
    wiring = host_check.get("entry_dataflow")
    if (not isinstance(wiring, dict)
            or wiring.get("schema") != "merlin.host_entry_dataflow_check.v2"
            or wiring.get("status") != "source_to_host_body_checked"
            or wiring.get("operation_count") != probe.get("source_operation_count")
            or wiring.get("native_external_call_count") != len(selected)
            or type(wiring.get("host_defined_call_count")) is not int
            or wiring["host_defined_call_count"] < 0
            or type(wiring.get("host_linalg_body_correspondence_checked_count")) is not int
            or wiring.get("host_linalg_body_correspondence_checked_count")
            != wiring.get("host_defined_call_count")
            or type(wiring.get("checked_operand_edges")) is not int
            or wiring["checked_operand_edges"] < 0):
        raise ValueError("host object has no matching source-to-entry wiring check")
    accounting = host_check.get("source_route_accounting")
    counts = accounting.get("state_counts") if isinstance(accounting, dict) else None
    allowed_states = {
        "native_external_unexecuted", "required_native_retained_host",
        "host_body_numerics_unqualified", "entry_support_numerics_unqualified",
        "model_output_unexecuted",
    }
    if (not isinstance(accounting, dict)
            or accounting.get("schema") != "merlin.model_source_route_accounting.v1"
            or accounting.get("operation_count") != probe.get("source_operation_count")
            or not isinstance(counts, dict) or set(counts) - allowed_states
            or any(type(value) is not int or value < 0 for value in counts.values())
            or sum(counts.values()) != probe.get("source_operation_count")
            or counts.get("native_external_unexecuted") != len(selected)
            or counts.get("required_native_retained_host", 0)
            != len(bridge["required_native_unmet"])
            or counts.get("model_output_unexecuted") != 1
            or counts.get("host_body_numerics_unqualified", 0)
            + counts.get("required_native_retained_host", 0)
            != wiring["host_defined_call_count"]
            or not isinstance(accounting.get("operations"), list)
            or len(accounting["operations"]) != probe.get("source_operation_count")):
        raise ValueError("host object has no complete source route accounting")
    abi = host_check.get("entry_abi")
    if (not isinstance(abi, dict)
            or type(abi.get("source_input_count")) is not int
            or type(abi.get("source_result_count")) is not int
            or abi["source_result_count"] < 1
            or abi.get("input_descriptor_positions") != list(range(abi["source_input_count"]))
            or abi.get("result_descriptor_positions") != list(
                range(abi["source_input_count"],
                      abi["source_input_count"] + abi["source_result_count"])
            )
            or len(abi.get("descriptor_types", [])) != (
                abi["source_input_count"] + abi["source_result_count"]
            )):
        raise ValueError("host object has no checked appended-result C-interface ABI")
    source_regions = [*probe["regions"], *probe.get("batched_contractions", [])]
    regions = {row["region_id"]: row for row in source_regions}
    if len(regions) != len(source_regions):
        raise ValueError("model probe has duplicate source region IDs")
    bindings = {row["symbol"]: row for row in bridge["bindings"]}
    rows = []
    seen = set()
    bundles = {}
    for directory in source_bundles or []:
        manifest = directory / "manifest.json"
        if directory.is_symlink() or manifest.is_symlink() or not manifest.is_file():
            raise ValueError("selected source host bundle is missing or linked")
        digest = _sha(manifest)
        if digest in bundles:
            raise ValueError("selected source host bundle is duplicated")
        bundles[digest] = directory
    for directory, wrapper in wrappers:
        symbol = wrapper.get("outlined_symbol")
        if (wrapper.get("schema") not in {
            "atlas.diagnostic_model_tile_wrapper.v1",
            "atlas.diagnostic_source_mn_model_wrapper.v1",
        } or wrapper.get("engine") != "merlin_native"
                or wrapper.get("linked") is not False or wrapper.get("executed") is not False
                or not isinstance(symbol, str) or symbol not in selected or symbol in seen):
            raise ValueError("wrapper does not identify one selected native call")
        seen.add(symbol)
        region = regions.get(wrapper.get("region_id"))
        binding = bindings.get(symbol)
        if (not region or not binding or binding["region_id"] != wrapper["region_id"]
                or binding["operation_id"] != wrapper.get("source_operation_id")
                or bridge["sha256"] != wrapper.get("host_bridge_sha256")):
            raise ValueError("wrapper differs from the selected native region")
        if wrapper["schema"] == "atlas.diagnostic_source_mn_model_wrapper.v1":
            bundle = bundles.get(wrapper.get("host_bundle_manifest_sha256"))
            if (probe_path is None or bundle is None
                    or binding.get("selected_route") != "tiled_native"):
                raise ValueError("source grid wrapper has no selected host bundle")
            if json.loads((probe_path / "manifest.json").read_text()) != probe:
                raise ValueError("source grid wrapper differs from the model probe")
            expected = prepare_source_grid_wrapper(probe_path, wrapper["region_id"], bundle)
            expected_record = expected["record"]
            if {key: value for key, value in wrapper.items() if key != "files"} != expected_record:
                raise ValueError("source grid wrapper record differs from the selected renderer")
            files = wrapper.get("files")
            if (not isinstance(files, dict) or set(files) != set(expected["files"])
                    or any((directory / name).is_symlink()
                           or not (directory / name).is_file()
                           or _sha(directory / name) != files[name]
                           or (directory / name).read_text() != rendered
                           for name, rendered in expected["files"].items())):
                raise ValueError("source grid wrapper code differs from the selected renderer")
            rows.append({
                "region_id": wrapper["region_id"], "symbol": symbol,
                "binder": wrapper["binding_function"],
                "map_name": wrapper["binding_function"].replace(
                    "atlas_bind_source_mn_", "atlas_source_mn_map_", 1,
                ),
                "kind": "source_grid", "bundle_dir": str(bundle.resolve()),
                "wrapper_manifest_sha256": _sha(directory / "manifest.json"),
                "wrapper_dir": str(directory.resolve()),
            })
            continue
        if (type(wrapper.get("bitstream_bound")) is not bool
                or wrapper.get("status") != (
                    "diagnostic_unqualified" if wrapper["bitstream_bound"] else "diagnostic_unbound"
                )
                or region["request_digest"] != wrapper.get("native_request_digest")
                or region["binary_sha256"] != wrapper.get("native_binary_sha256")
                or region["status"] != "emitted"
                or region["static_inputs_status"] != "packed_or_runtime"):
            raise ValueError("wrapper differs from the selected native region")
        binder = wrapper.get("binding_function")
        if not isinstance(binder, str) or not re.fullmatch(r"atlas_bind_tile_[0-9a-f]{12}", binder):
            raise ValueError("wrapper has an invalid binding function")
        files = wrapper.get("files")
        expected_files = {
            "atlas_program.h", "atlas_captured_data.h",
            "atlas_model_tile_wrapper.h", "atlas_model_tile_wrapper.c",
        }
        if not isinstance(files, dict) or set(files) != expected_files:
            raise ValueError("wrapper has an incomplete source roster")
        for name, digest in files.items():
            path = directory / name
            if path.is_symlink() or not path.is_file() or _sha(path) != digest:
                raise ValueError(f"wrapper source changed: {name}")
        if _program_header_digest(
            (directory / "atlas_program.h").read_text(),
            bitstream_bound=wrapper.get("bitstream_bound") is True,
        ) != wrapper["native_binary_sha256"]:
            raise ValueError("wrapper program words differ from the selected native binary")
        shape = wrapper.get("source_shape", [32, 32, 32])
        if (not isinstance(shape, list) or len(shape) != 3 or
                any(type(value) is not int or not 1 <= value <= 32 for value in shape)):
            raise ValueError("wrapper has an invalid source window")
        m, n, k = shape
        if (binding.get("input_types") != [f"tensor<{m}x{k}xf32>",
                                          f"tensor<{k}x{n}xf32>"] or
                binding.get("result_types") != [f"tensor<{m}x{n}xf32>"]):
            raise ValueError("wrapper source window differs from selected C-interface types")
        rendered_header, rendered_source, rendered_binder = render_wrapper(
            symbol, wrapper.get("constant_operand"), tuple(wrapper.get("transpose_input", [])),
            tuple(wrapper["source_shape"]) if "source_shape" in wrapper else None,
        )
        if (binder != rendered_binder
                or (directory / "atlas_model_tile_wrapper.h").read_text() != rendered_header
                or (directory / "atlas_model_tile_wrapper.c").read_text() != rendered_source):
            raise ValueError("wrapper code differs from the selected Atlas renderer")
        rows.append({
            "region_id": wrapper["region_id"], "symbol": symbol, "binder": binder,
            "map_name": binder.replace("atlas_bind_tile_", "atlas_tile_map_", 1),
            "kind": "single_tile",
            "wrapper_manifest_sha256": _sha(directory / "manifest.json"),
            "wrapper_dir": str(directory.resolve()),
        })
    if set(selected) != seen or len(rows) != len(selected):
        raise ValueError("wrapper roster does not cover exactly the selected native calls")
    return sorted(rows, key=lambda row: selected.index(row["symbol"]))


def _check_native_reset_words(probe_path: Path, probe: dict,
                              host_check: dict) -> list[dict]:
    """Tie each dropped zero initializer to a reset word in its selected image.

    This checks each selected output tile's encoded reset and accumulation
    sequence, not the MXU's numerical behavior or timing.
    """
    selected = probe["host_bridge"]["external_symbols"]
    wiring = host_check["entry_dataflow"]
    if (wiring.get("initialization_semantics") != "source_drop_and_host_initializer_checked"
            or wiring.get("native_reset_semantics") != "not_checked"
            or wiring.get("native_zero_initialization_obligation_count") != len(selected)):
        raise ValueError("selected native calls lack checked positive-zero source starts")
    inventory = load_inventory()
    source_regions = [*probe["regions"], *probe.get("batched_contractions", [])]
    regions = {row["region_id"]: row for row in source_regions}
    if len(regions) != len(source_regions):
        raise ValueError("selected native reset has duplicate source regions")
    checked = []
    expected_count = 0
    for binding in probe["host_bridge"]["bindings"]:
        if binding["symbol"] not in selected:
            continue
        region = regions[binding["region_id"]]
        batched = "batch_shape" in region
        if binding.get("selected_route") == "tiled_native":
            artifacts = sorted(
                (tile for tile in probe.get(
                    "batched_tiles" if batched else "named_tiles", [],
                )
                 if tile["region_id"] == region["region_id"]),
                key=lambda tile: tile["tile_index"],
            )
            if (len(artifacts) != region.get("tile_count")
                    or [tile["request_digest"] for tile in artifacts]
                    != binding.get("tile_request_digests")):
                raise ValueError("selected native reset has an incomplete tile grid")
            relative_pattern = (
                r"regions/batched-[0-9]{4}-tile-[0-9]{4}" if batched
                else r"regions/named-[0-9]{4}-tile-[0-9]{4}"
            )
        else:
            artifacts = [region]
            relative_pattern = r"regions/region-[0-9]{4}"
        expected_count += len(artifacts)
        for selected_artifact in artifacts:
            relative = selected_artifact.get("artifact")
            if (not isinstance(relative, str) or not re.fullmatch(relative_pattern, relative)):
                raise ValueError("selected native reset has no safe region artifact")
            artifact = probe_path / relative
            if (artifact.is_symlink() or not artifact.is_dir()
                    or not artifact.resolve().is_relative_to(probe_path.resolve())):
                raise ValueError("selected native reset artifact is missing or escapes the probe")
            manifest_path = artifact / "manifest.json"
            binary_path = artifact / "program.bin"
            program_path = artifact / "physical-program.json"
            plan_path = artifact / "execution_plan.json"
            if any(path.is_symlink() or not path.is_file() for path in
                   (manifest_path, binary_path, program_path, plan_path)):
                raise ValueError("selected native reset has a missing or linked stage file")
            manifest = json.loads(manifest_path.read_text())
            binary = binary_path.read_bytes()
            if (manifest.get("schema") != "atlas.native_tile_compilation.v6"
                    or manifest.get("artifact_kind") != "program"
                    or manifest.get("target_identity") != probe.get("target_identity")
                    or manifest.get("target_inventory_sha256") != _sha(
                        ROOT / "docs/selected-variant-inventory.json")
                    or manifest.get("engine") != "merlin_native"
                    or manifest.get("request_digest") != selected_artifact["request_digest"]
                    or manifest.get("binary_sha256") != selected_artifact["binary_sha256"]
                    or hashlib.sha256(binary).hexdigest() != selected_artifact["binary_sha256"]
                    or manifest.get("stage_artifacts", {}).get("physical-program.json") != _sha(program_path)
                    or manifest.get("execution_plan_sha256") != _sha(plan_path)):
                raise ValueError("selected native reset program differs from its checked artifact")
            program = json.loads(program_path.read_text())
            plan = json.loads(plan_path.read_text())
            if program.get("selected_rtl_revision") != inventory["selected_sources"]["rtl_revision"]:
                raise ValueError("selected reset program has a different RTL revision")
            audit_program(inventory, program, binary)
            decoded = decode_binary(inventory, binary)["instructions"]
            operation = plan.get("operation")
            if not isinstance(operation, str) or not re.fullmatch(
                    r"mxu[01]_(?:reset|k[2-9][0-9]*)", operation):
                raise ValueError("selected native tile does not declare reset execution")
            unit = int(operation[3])
            matmuls = [row for row in decoded if row["operation"] == "atlas.mxu_matmul"]
            expected_matmuls = 1 if operation.endswith("reset") else int(operation.split("k")[1])
            if len(matmuls) != expected_matmuls:
                raise ValueError("selected native tile has no declared reset/accumulate sequence")
            if binding.get("selected_route") != "tiled_native" and (
                    len(matmuls) != 1 or matmuls[0]["fields"]["unit"] != unit
                    or matmuls[0]["fields"]["accumulate"] is not False):
                raise ValueError("selected native tile has no unique encoded reset matmul")
            if (not matmuls or matmuls[0]["fields"]["unit"] != unit
                    or matmuls[0]["fields"]["accumulate"] is not False
                    or any(row["fields"]["unit"] != unit
                           or row["fields"]["acc_slot"] != matmuls[0]["fields"]["acc_slot"]
                           or row["fields"]["accumulate"] is not True
                           for row in matmuls[1:])):
                raise ValueError("selected native tile has no ordered encoded reset chain")
            reset = matmuls[0]
            for index, matmul in enumerate(matmuls):
                previous = matmuls[index - 1]["word_index"] + 1 if index else 0
                push = [row for row in decoded[previous:matmul["word_index"]]
                        if row["operation"] == "atlas.mxu_push"
                        and row["fields"]["unit"] == unit
                        and row["fields"]["slot"] == matmul["fields"]["weight_slot"]
                        and row["fields"]["kind"] == "weight_fp8"]
                if len(push) != 1:
                    raise ValueError("encoded matmul lacks its matching weight push")
            pop = [row for row in decoded
                   if row["operation"] == "atlas.mxu_pop"
                   and row["fields"]["unit"] == unit
                   and row["fields"]["slot"] == reset["fields"]["acc_slot"]
                   and row["fields"]["format"] == "bf16"]
            if len(pop) != 1 or pop[0]["word_index"] <= matmuls[-1]["word_index"]:
                raise ValueError("encoded reset lacks its final BF16 readout")
            checked.append({
                "region_id": region["region_id"],
                **({"tile_index": selected_artifact["tile_index"]}
                   if binding.get("selected_route") == "tiled_native" else {}),
                "binary_sha256": selected_artifact["binary_sha256"],
                "reset_word_index": reset["word_index"],
                "reset_mode": reset["mode_id"],
                "unit": unit,
            })
    if len(checked) != expected_count:
        raise ValueError("native reset check did not cover every selected model call")
    return checked


def _check_link_symbols(output: str, selected: list[str], entry: str) -> list[str]:
    symbols = _symbols(output)
    required = {"_mlir_ciface_" + entry, "atlas_invoke_checked_entry",
                *("_mlir_ciface_" + name for name in selected)}
    if any(symbols.get(name) not in {"T", "W"} for name in required):
        raise ValueError("linked object lacks a defined model entry or selected device call")
    if any(name.startswith("_mlir_ciface_") and kind == "U" for name, kind in symbols.items()):
        raise ValueError("linked object retains an unresolved MLIR device call")
    unresolved = sorted(name for name, kind in symbols.items() if kind == "U")
    if not set(unresolved).issubset(RUNTIME_UNDEFINED):
        raise ValueError("linked object has an unexpected unresolved runtime roster")
    return unresolved


def _render_entry_adapter(entry: str, abi: dict) -> tuple[str, str]:
    """Expose the checked appended-result ABI through one counted C call."""
    if not isinstance(entry, str) or not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", entry):
        raise ValueError("model entry has no supported C symbol")
    input_count = abi["source_input_count"]
    result_count = abi["source_result_count"]
    types = abi["descriptor_types"]
    if (type(input_count) is not int or type(result_count) is not int
            or input_count < 1 or result_count < 1
            or len(types) != input_count + result_count):
        raise ValueError("model entry has no supported descriptor roster")
    dimensions = []
    for typ in types:
        match = re.fullmatch(
            r"memref<([1-9][0-9]*)(?:x([1-9][0-9]*))?(?:x([1-9][0-9]*))?xf32>",
            typ,
        )
        if match is None:
            raise ValueError("model entry needs static rank-one to rank-three f32 memrefs")
        dimensions.append(tuple(int(value) for value in match.groups() if value is not None))
    if any(len(shape) != 2 for shape in dimensions):
        fields = [
            f"  const atlas_f32_memref{len(shape)} *input_{index};"
            for index, shape in enumerate(dimensions[:input_count])
        ]
        fields.extend(
            f"  atlas_f32_memref{len(shape)} *result_{index};"
            for index, shape in enumerate(dimensions[input_count:])
        )
        header = "\n".join([
            "#ifndef ATLAS_MODEL_ENTRY_H", "#define ATLAS_MODEL_ENTRY_H",
            '#include "atlas_model_tile.h"',
            "#ifdef __cplusplus", 'extern "C" {', "#endif",
            "/* Caller supplies storage for every selected model result. */",
            "typedef struct {", *fields, "} atlas_model_entry_call;",
            "int atlas_invoke_checked_entry(const atlas_model_entry_call *call);",
            "#ifdef __cplusplus", "}", "#endif",
            "#endif", "",
        ])
        params = [
            f"const atlas_f32_memref{len(shape)} *input_{index}"
            for index, shape in enumerate(dimensions[:input_count])
        ]
        params.extend(
            f"atlas_f32_memref{len(shape)} *result_{index}"
            for index, shape in enumerate(dimensions[input_count:])
        )
        checks = [
            f"  if (!call->{name}) return 1;"
            for name in [*(f"input_{i}" for i in range(input_count)),
                         *(f"result_{i}" for i in range(result_count))]
        ]
        for index, shape in enumerate(dimensions[:input_count]):
            name = f"call->input_{index}"
            checks.append(
                f"  if (!{name}->aligned || {name}->offset < 0 || "
                + " || ".join(f"{name}->sizes[{axis}] != {extent} || "
                             f"{name}->strides[{axis}] <= 0"
                             for axis, extent in enumerate(shape))
                + f" || !atlas_entry_span_safe({name}->offset, {name}->sizes, "
                  f"{name}->strides, {len(shape)})) return 2;"
            )
        result_checks = []
        for index, shape in enumerate(dimensions[input_count:]):
            name = f"call->result_{index}"
            result_checks.append(
                f"  if (!{name}->aligned || "
                + " || ".join(f"{name}->sizes[{axis}] != {extent} || "
                             f"{name}->strides[{axis}] <= 0"
                             for axis, extent in enumerate(shape))
                + f" || !atlas_entry_span_safe({name}->offset, {name}->sizes, "
                  f"{name}->strides, {len(shape)})) return 3;"
            )
        arguments = [f"call->input_{i}" for i in range(input_count)]
        arguments.extend(f"call->result_{i}" for i in range(result_count))
        source = "\n".join([
            '#include "atlas_model_entry.h"',
            "static int atlas_entry_span_safe(int64_t offset, const int64_t *sizes,",
            "                                 const int64_t *strides, unsigned rank) {",
            "  if (offset < 0) return 0;",
            "  int64_t remaining = INT64_MAX - offset;",
            "  for (unsigned axis = 0; axis < rank; ++axis) {",
            "    if (sizes[axis] <= 0 || strides[axis] <= 0) return 0;",
            "    if (sizes[axis] > 1) {",
            "      if (strides[axis] > remaining / (sizes[axis] - 1)) return 0;",
            "      remaining -= strides[axis] * (sizes[axis] - 1);",
            "    }",
            "  }",
            "  return 1;",
            "}",
            f"extern void _mlir_ciface_{entry}({', '.join(params)});",
            "int atlas_invoke_checked_entry(const atlas_model_entry_call *call) {",
            "  if (!call) return 1;", *checks, *result_checks,
            f"  _mlir_ciface_{entry}({', '.join(arguments)});",
            *result_checks,
            "  return 0;", "}", "",
        ])
        return header, source
    header = """#ifndef ATLAS_MODEL_ENTRY_H
#define ATLAS_MODEL_ENTRY_H
#include "atlas_model_tile.h"
#ifdef __cplusplus
extern "C" {
#endif
/* Call after atlas_bind_model_tiles. The caller supplies result storage. */
int atlas_invoke_checked_entry(const atlas_f32_memref2 *inputs, size_t input_count,
                               atlas_f32_memref2 *results, size_t result_count);
#ifdef __cplusplus
}
#endif
#endif
"""
    params = [f"const atlas_f32_memref2 *input_{index}" for index in range(input_count)]
    params.extend(f"atlas_f32_memref2 *result_{index}" for index in range(result_count))
    checks = [
        f"  if (!inputs[{index}].aligned || inputs[{index}].offset < 0 ||"
        f" inputs[{index}].sizes[0] != {rows} || inputs[{index}].sizes[1] != {cols} ||"
        f" inputs[{index}].strides[0] <= 0 || inputs[{index}].strides[1] <= 0) return 2;"
        for index, (rows, cols) in enumerate(dimensions[:input_count])
    ]
    result_checks = [
        f"  if (!results[{index}].aligned || results[{index}].offset < 0 ||"
        f" results[{index}].sizes[0] != {rows} || results[{index}].sizes[1] != {cols} ||"
        f" results[{index}].strides[0] <= 0 || results[{index}].strides[1] <= 0) return 3;"
        for index, (rows, cols) in enumerate(dimensions[input_count:])
    ]
    arguments = [f"&inputs[{index}]" for index in range(input_count)]
    arguments.extend(f"&results[{index}]" for index in range(result_count))
    source = "\n".join([
        '#include "atlas_model_entry.h"',
        f"extern void _mlir_ciface_{entry}({', '.join(params)});",
        "int atlas_invoke_checked_entry(const atlas_f32_memref2 *inputs, size_t input_count,",
        "                               atlas_f32_memref2 *results, size_t result_count) {",
        f"  if (!inputs || !results || input_count != {input_count}u ||"
        + f" result_count != {result_count}u) return 1;",
        *checks,
        *result_checks,
        f"  _mlir_ciface_{entry}({', '.join(arguments)});",
        *result_checks,
        "  return 0;",
        "}",
        "",
    ])
    return header, source


def _render_bindings(rows: list[dict]) -> tuple[str, str]:
    """Initialize one Atlas host and bind each selected C-interface call."""
    if not rows:
        raise ValueError("model binding needs at least one selected tile")
    header = """#ifndef ATLAS_MODEL_BINDINGS_H
#define ATLAS_MODEL_BINDINGS_H
#include "atlas_model_tile.h"
int atlas_bind_model_tiles(atlas_host *host, atlas_host_mmio mmio,
                           atlas_host_memory memory, atlas_tile_memory_io io,
                           size_t max_polls);
#endif
"""
    declarations = []
    for row in rows:
        binder = row["binder"]
        map_name = row.get("map_name", binder.replace("atlas_bind_tile_", "atlas_tile_map_", 1))
        declarations.extend([
            f"extern int {binder}(atlas_host *, atlas_tile_memory_io, size_t);",
            f"extern atlas_host_map {map_name}(void);",
        ])
    first_map = rows[0].get("map_name", rows[0]["binder"].replace(
        "atlas_bind_tile_", "atlas_tile_map_", 1,
    ))
    comparisons = []
    for row in rows[1:]:
        map_name = row.get("map_name", row["binder"].replace(
            "atlas_bind_tile_", "atlas_tile_map_", 1,
        ))
        comparisons.append(
            f"  atlas_host_map other_{map_name} = {map_name}();\n"
            f"  if (map.imem_base != other_{map_name}.imem_base ||\n"
            f"      map.csr_base != other_{map_name}.csr_base ||\n"
            f"      map.imem_capacity_bytes != other_{map_name}.imem_capacity_bytes) return 2;"
        )
    calls = [f"  if ({row['binder']}(host, io, max_polls)) return {index + 4};"
             for index, row in enumerate(rows)]
    source = "\n".join([
        '#include "atlas_model_bindings.h"',
        *declarations,
        "int atlas_bind_model_tiles(atlas_host *host, atlas_host_mmio mmio,",
        "                           atlas_host_memory memory, atlas_tile_memory_io io,",
        "                           size_t max_polls) {",
        "  if (!host || !io.write || !io.read || !max_polls) return 1;",
        f"  atlas_host_map map = {first_map}();",
        *comparisons,
        "  if (atlas_host_init(host, map, mmio, memory) != ATLAS_HOST_OK) return 3;",
        *calls,
        "  return 0;",
        "}",
        "",
    ])
    return header, source


def _run(command: list[str], log: Path) -> None:
    result = subprocess.run(command, capture_output=True, text=True, check=False, timeout=120)
    log.write_text(result.stdout + result.stderr)
    if result.returncode:
        raise ValueError(f"tool failed ({result.returncode}): {log}")


def build(args: argparse.Namespace) -> dict:
    out = Path(args.out)
    if out.exists() or out.is_symlink():
        raise ValueError("fresh model link output directory required")
    probe_path = Path(args.probe).resolve(strict=True)
    host_check_path = Path(args.host_check).resolve(strict=True)
    host_object = Path(args.host_object).resolve(strict=True)
    clang = Path(args.clang).resolve(strict=True)
    nm = Path(args.nm).resolve(strict=True)
    runtime_include = Path(args.merlin_runtime_include).resolve(strict=True)
    if not all(path.is_file() for path in (host_check_path, host_object, clang, nm)):
        raise ValueError("selected host object or tools are missing")
    probe_bytes = (probe_path / "manifest.json").read_bytes()
    probe = json.loads(probe_bytes)
    host_check = json.loads(host_check_path.read_text())
    if host_check.get("probe_manifest_sha256") != hashlib.sha256(probe_bytes).hexdigest():
        raise ValueError("host object check differs from model probe")
    wrapper_dirs = [Path(name).resolve(strict=True) for name in args.wrapper]
    wrappers = [(directory, json.loads((directory / "manifest.json").read_text()))
                for directory in wrapper_dirs]
    source_bundles = [Path(name).resolve(strict=True)
                      for name in getattr(args, "source_bundle", [])]
    rows = _selection(
        probe, host_check, host_object, wrappers,
        probe_path=probe_path, source_bundles=source_bundles,
    )
    reset_words = _check_native_reset_words(probe_path, probe, host_check)
    if not (runtime_include / "finite_float_boundary.h").is_file():
        raise ValueError("selected Merlin runtime include lacks the precision boundary")
    out.mkdir(parents=True)
    objects = []
    commands = []
    entry_header, entry_source = _render_entry_adapter(
        probe["host_bridge"]["entry_c_interface_symbol"], host_check["entry_abi"],
    )
    (out / "atlas_model_entry.h").write_text(entry_header)
    (out / "atlas_model_entry.c").write_text(entry_source)
    entry_object = out / "atlas_model_entry.o"
    entry_command = [
        str(clang), *FLAGS, "-I", str(ROOT / "runtime"), "-I", str(out),
        "-c", str(out / "atlas_model_entry.c"), "-o", str(entry_object),
    ]
    _run(entry_command, out / "atlas_model_entry.log")
    objects.append(entry_object)
    commands.append(entry_command)
    bindings_header, bindings_source = _render_bindings(rows)
    (out / "atlas_model_bindings.h").write_text(bindings_header)
    (out / "atlas_model_bindings.c").write_text(bindings_source)
    binder_object = out / "atlas_model_bindings.o"
    binder_command = [
        str(clang), *FLAGS, "-I", str(ROOT / "runtime"), "-I", str(out),
        "-c", str(out / "atlas_model_bindings.c"), "-o", str(binder_object),
    ]
    _run(binder_command, out / "atlas_model_bindings.log")
    objects.append(binder_object)
    commands.append(binder_command)
    for index, row in enumerate(rows):
        grid = row["kind"] == "source_grid"
        source = Path(row["wrapper_dir"]) / (
            "atlas_source_mn_model_wrapper.c" if grid else "atlas_model_tile_wrapper.c"
        )
        obj = out / f"wrapper-{index:04d}.o"
        command = [
            str(clang), *FLAGS, "-DATLAS_DIAGNOSTIC_ALLOW_UNBOUND=1",
            "-DATLAS_DIAGNOSTIC_ASSUME_COHERENT=1",
            "-DATLAS_DIAGNOSTIC_PRECISION_POLICY=1",
            *(["-DATLAS_DIAGNOSTIC_SOURCE_MN=1", "-I", row["bundle_dir"]]
              if grid else []),
            "-I", str(ROOT / "runtime"), "-I", str(runtime_include),
            "-I", row["wrapper_dir"], "-c", str(source), "-o", str(obj),
        ]
        _run(command, out / f"wrapper-{index:04d}.log")
        objects.append(obj)
        commands.append(command)
    if any(row["kind"] == "source_grid" for row in rows):
        for row in rows:
            if row["kind"] != "source_grid":
                continue
            bundle = Path(row["bundle_dir"])
            if any(_sha(bundle / name) != _sha(ROOT / "runtime" / name)
                   for name in SOURCE_RUNTIME_FILES):
                raise ValueError("source grid runtime differs from selected Atlas runtime")
    runtime_names = ["atlas_host", "atlas_model_tile"]
    if any(row["kind"] == "source_grid" for row in rows):
        runtime_names.extend(("atlas_ordered_k", "atlas_ordered_mn",
                              "atlas_source_mn", "atlas_source_mn_memref"))
    for name in runtime_names:
        source = ROOT / "runtime" / f"{name}.c"
        obj = out / f"{name}.o"
        command = [str(clang), *FLAGS, "-DATLAS_FREESTANDING=1",
                   "-I", str(ROOT / "runtime"), "-c", str(source), "-o", str(obj)]
        _run(command, out / f"{name}.log")
        objects.append(obj)
        commands.append(command)
    linked = out / "combined.o"
    command = [str(clang), *FLAGS[:4], "-nostdlib", "-fuse-ld=lld", "-Wl,-r",
               str(host_object), *(str(obj) for obj in objects), "-o", str(linked)]
    _run(command, out / "link.log")
    commands.append(command)
    nm_result = subprocess.run([str(nm), "-P", str(linked)],
                               capture_output=True, text=True, check=False, timeout=15)
    if nm_result.returncode:
        raise ValueError("nm could not inspect the combined host/device object")
    unresolved = _check_link_symbols(
        nm_result.stdout, [row["symbol"] for row in rows],
        probe["host_bridge"]["entry_c_interface_symbol"],
    )
    return {
        "schema": "atlas.diagnostic_model_link.v1",
        "status": "relocatable_link_checked",
        "scope": "selected native calls resolved in one RV64 relocatable object; no final ELF or execution",
        "probe_manifest_sha256": hashlib.sha256(probe_bytes).hexdigest(),
        "host_check_sha256": _sha(host_check_path),
        "host_object_sha256": _sha(host_object),
        "entry_abi": host_check["entry_abi"],
        "linker_source_sha256": _sha(Path(__file__)),
        "runtime_source_sha256": {
            name: _sha(ROOT / "runtime" / name)
            for base in runtime_names for name in (f"{base}.c", f"{base}.h")
        },
        "finite_float_boundary_sha256": _sha(runtime_include / "finite_float_boundary.h"),
        "clang_sha256": _sha(clang), "nm_sha256": _sha(nm),
        "selected_wrappers": rows,
        "checked_native_reset_words": reset_words,
        "entry_header_sha256": _sha(out / "atlas_model_entry.h"),
        "entry_source_sha256": _sha(out / "atlas_model_entry.c"),
        "bindings_header_sha256": _sha(out / "atlas_model_bindings.h"),
        "bindings_source_sha256": _sha(out / "atlas_model_bindings.c"),
        "required_native_unmet": probe["host_bridge"]["required_native_unmet"],
        "source_route_state_counts": host_check["source_route_accounting"]["state_counts"],
        "unresolved_runtime_symbols": unresolved,
        "commands": commands,
        "object_sha256": {obj.name: _sha(obj) for obj in objects},
        "combined_object": "combined.o",
        "combined_object_sha256": _sha(linked),
        "final_elf": False, "executed": False,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--probe", required=True)
    parser.add_argument("--host-check", required=True)
    parser.add_argument("--host-object", required=True)
    parser.add_argument("--wrapper", action="append", required=True)
    parser.add_argument("--source-bundle", action="append", default=[],
                        help="verified source-grid host bundle for a grid wrapper")
    parser.add_argument("--merlin-runtime-include", required=True)
    parser.add_argument("--clang", required=True)
    parser.add_argument("--nm", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)
    out = Path(args.out)
    preexisting_output = out.exists() or out.is_symlink()
    try:
        report = build(args)
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError,
            subprocess.TimeoutExpired) as error:
        report = {"schema": "atlas.diagnostic_model_link.v1",
                  "status": "compile_error", "reason": str(error)}
        if not preexisting_output:
            if not out.exists() and not out.is_symlink():
                out.mkdir(parents=True)
            if out.is_dir():
                (out / "manifest.json").write_text(json.dumps(report, sort_keys=True, indent=2) + "\n")
        print(json.dumps(report, sort_keys=True), file=sys.stderr)
        return 1
    (Path(args.out) / "manifest.json").write_text(json.dumps(report, sort_keys=True, indent=2) + "\n")
    print(json.dumps({"status": report["status"], "out": args.out}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
