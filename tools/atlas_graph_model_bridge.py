"""Bind a checked native Atlas graph to its typed MLIR host call."""

from __future__ import annotations

import argparse
import hashlib
import json
import struct
import sys
from pathlib import Path


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _member(root: Path, name: str) -> bytes:
    path = Path(name)
    if (not name or path.is_absolute() or path.as_posix() != name
            or ".." in path.parts or (root / path).is_symlink()
            or not (root / path).is_file()
            or not (root / path).resolve().is_relative_to(root.resolve())):
        raise ValueError(f"graph member is absent or unsafe: {name}")
    return (root / path).read_bytes()


def _float_spec(raw: dict):
    from merlin.semantic_compiler.precision_values import FloatBoundarySpec

    if not isinstance(raw, dict) or raw.get("direction") != "encode":
        raise ValueError("graph input has no selected encode boundary")
    return FloatBoundarySpec(
        raw["policy"], raw["format"], raw["direction"],
        min_normal_exponent_code=raw["min_normal_exponent_code"],
        max_normal_exponent_code=raw["max_normal_exponent_code"],
        flush_below_normal=raw["flush_below_normal"],
        admitted_code_ranges=tuple(tuple(pair) for pair in raw["admitted_code_ranges"]),
        pre_encode_scale_exponent=raw.get("pre_encode_scale_exponent", 0),
        canonicalize_negative_zero=raw.get("canonicalize_negative_zero", False),
    )


def _render_bytes(name: str, payload: bytes) -> list[str]:
    return [
        f"static const uint8_t {name}[{len(payload)}] = {{",
        *("  " + ", ".join(f"0x{value:02x}u" for value in payload[start:start + 16]) + ","
          for start in range(0, len(payload), 16)),
        "};",
    ]


def prepare(source_path: Path, bridge_root: Path, compiled: Path) -> tuple[dict, dict[str, str]]:
    from merlin.frontends.linalg_mlir import parse_mlir_text
    from merlin.semantic_compiler.precision_codegen import (
        finite_f32_codebook,
        render_finite_f32_codebook_c,
    )
    from merlin.semantic_compiler.target_binding import verify_native_publication
    from merlin.targetgen.contract.model_host_bridge import (
        diagnostic_whole_graph_host_bridge,
    )
    from xdsl.dialects import func

    if bridge_root.is_symlink() or compiled.is_symlink():
        raise ValueError("graph bridge and native compilation must be real directories")
    bridge_bytes = _member(bridge_root, "manifest.json")
    bridge = json.loads(bridge_bytes)
    compilation_bytes = _member(compiled, "manifest.json")
    compilation = json.loads(compilation_bytes)
    if not isinstance(bridge, dict) or not isinstance(compilation, dict):
        raise TypeError("graph bridge and native compilation need object manifests")
    verify_native_publication(
        compiled, compilation, engine="merlin_native",
        request_digest=compilation.get("request_digest"),
        target_identity=compilation.get("target_identity"),
    )
    selected = json.loads(_member(compiled, "native-selection.json"))
    precision = json.loads(_member(compiled, "precision-plan.json"))
    plan = json.loads(_member(compiled, "execution_plan.json"))
    if any(not isinstance(item, dict) for item in (selected, precision, plan)):
        raise ValueError("native graph stages must be object records")
    binary = _member(compiled, "program.bin")
    mlir = _member(bridge_root, "host-bridge.mlir")
    source_bytes = source_path.read_bytes()
    symbol = bridge.get("device_symbol")
    if (bridge.get("schema") != "merlin.diagnostic_whole_graph_host_bridge.v1"
            or bridge.get("status") != "typed_declarations_only"
            or bridge.get("engine") != "merlin_native"
            or bridge.get("linked") is not False
            or bridge.get("executed") is not False
            or bridge.get("strict_native_eligible") is not False
            or bridge.get("host_bridge_sha256") != _sha(mlir)
            or bridge.get("native_manifest_sha256") != _sha(compilation_bytes)
            or bridge.get("request_digest") != compilation.get("request_digest")
            or bridge.get("target_identity") != compilation.get("target_identity")
            or bridge.get("program_sha256") != _sha(binary)
            or precision.get("source_sha256") != bridge.get("source_sha256")
            or bridge.get("source_sha256") != _sha(source_bytes)
            or selected.get("source_outputs") != bridge.get("source_outputs")
            or plan.get("output_addresses") != bridge.get("output_addresses")
            or not isinstance(symbol, str) or not symbol.isascii()
            or not symbol.isidentifier()
            or bridge.get("c_interface_symbol") != "_mlir_ciface_" + symbol
            or not binary or len(binary) % 4):
        raise ValueError("graph host bridge differs from the checked native program")
    expected, expected_mlir = diagnostic_whole_graph_host_bridge(
        source_bytes.decode(), bridge["entry"], device_symbol=symbol,
        compilation=compilation, selection=selected,
        precision_plan=precision, execution_plan=plan,
    )
    if (mlir != expected_mlir.encode()
            or any(bridge.get(key) != value for key, value in expected.items())):
        raise ValueError("graph host bridge differs from replayed source lowering")

    module = parse_mlir_text(mlir.decode())
    module.verify()
    declarations = [op for op in module.body.block.ops
                    if isinstance(op, func.FuncOp) and op.sym_name.data == symbol]
    if (len(declarations) != 1 or declarations[0].body.blocks
            or "llvm.emit_c_interface" not in declarations[0].attributes):
        raise ValueError("selected graph callback lacks an external MLIR C-interface declaration")
    source_inputs = bridge.get("input_types")
    source_outputs = bridge.get("result_types")
    entry = bridge.get("entry")
    shape = plan.get("shape")
    if (not isinstance(shape, list) or shape != [32, 32]
            or plan.get("schema") != "atlas.native_tile_execution_plan.v8"
            or plan.get("dtype") != "bf16"
            or not isinstance(source_inputs, list) or not source_inputs
            or not isinstance(source_outputs, list) or not source_outputs
            or source_inputs != ["tensor<32x32xf32>"] * len(source_inputs)
            or source_outputs != ["tensor<32x32xf32>"] * len(source_outputs)
            or not isinstance(entry, str) or not entry.isascii() or not entry.isidentifier()
            or [str(item) for item in declarations[0].function_type.inputs.data] != source_inputs
            or [str(item) for item in declarations[0].function_type.outputs.data] != source_outputs):
        raise ValueError("selected graph has no supported static f32 host signature")
    input_count, output_count = len(source_inputs), len(source_outputs)
    addresses = plan.get("input_addresses")
    sizes = plan.get("input_payload_bytes")
    output_addresses = plan.get("output_addresses")
    layouts = plan.get("output_physical_layouts")
    constants = compilation.get("constant_inputs")
    conversions = precision.get("boundary_conversions")
    if (not isinstance(addresses, list) or not isinstance(sizes, list)
            or not isinstance(output_addresses, list) or not isinstance(layouts, list)
            or not isinstance(constants, list) or not isinstance(conversions, list)
            or any(not isinstance(item, dict) for item in constants + conversions)
            or len(addresses) != len(sizes)
            or len(addresses) != input_count + len(constants)
            or len(output_addresses) != output_count
            or layouts != ["bf16_pair_halves"] * output_count
            or plan.get("output_bytes") != 2048
            or sizes != [1024] * input_count + [item.get("bytes") for item in constants]
            or len(conversions) != input_count + output_count
            or [item.get("source") for item in conversions[input_count:]]
            != selected.get("source_outputs")):
        raise ValueError("graph physical slots or source conversions differ")
    # Keep the emitted runtime restricted to the selected 32x32 register-pair
    # layout. Any new shape/layout needs a separately checked target binding.
    if (any(type(value) is not int or not 0 <= value <= 0xffffffff
            for value in addresses + output_addresses)
            or len(set(addresses + output_addresses)) != len(addresses + output_addresses)):
        raise ValueError("graph physical addresses are invalid or repeated")
    codebooks: dict[str, str] = {}
    codebook_lines: list[str] = []
    input_slots = []
    for index, conversion in enumerate(conversions[:input_count]):
        if (not isinstance(conversion, dict)
                or conversion.get("source") != f"arg{index}"
                or conversion.get("source_dtype") != "f32"
                or conversion.get("target_dtype") != "fp8_e4m3"):
            raise ValueError("graph input conversion is missing or reordered")
        spec = _float_spec(conversion.get("conversion_spec"))
        key = json.dumps(spec.record(), sort_keys=True)
        if key not in codebooks:
            name = f"atlas_graph_codebook_{len(codebooks)}"
            codebook_lines.extend(render_finite_f32_codebook_c(name, finite_f32_codebook(spec)))
            codebooks[key] = name
        input_slots.append(f"  {{0x{addresses[index]:x}u, 32u, 32u, 1024u, &{codebooks[key]}}},")

    constant_lines: list[str] = []
    constant_slots = []
    for index, item in enumerate(constants):
        if not isinstance(item, dict):
            raise TypeError("graph constant slot is malformed")
        payload = _member(compiled, item.get("path"))
        address = addresses[input_count + index]
        if (item.get("address") != address or item.get("bytes") != len(payload)
                or item.get("sha256") != _sha(payload)):
            raise ValueError("graph constant bytes differ from their physical slot")
        name = f"atlas_graph_constant_{index}"
        constant_lines.extend(_render_bytes(name, payload))
        constant_slots.append(f"  {{0x{address:x}u, {name}, {len(payload)}u}},")
    words = struct.unpack(f"<{len(binary) // 4}I", binary)
    data = [
        "/* Generated from an exact diagnostic Merlin-native graph program. */",
        "#ifndef ATLAS_GRAPH_DATA_H", "#define ATLAS_GRAPH_DATA_H",
        '#include "atlas_graph_memref.h"',
        f"#define ATLAS_GRAPH_INPUT_COUNT {input_count}u",
        f"#define ATLAS_GRAPH_OUTPUT_COUNT {output_count}u",
        f"#define ATLAS_GRAPH_CONSTANT_COUNT {len(constants)}u",
        f"#define ATLAS_GRAPH_PROGRAM_WORD_COUNT {len(words)}u",
        f'#define ATLAS_GRAPH_PROGRAM_SHA256 "{_sha(binary)}"',
        "#define ATLAS_GRAPH_DIAGNOSTIC_ONLY 1",
        *codebook_lines,
        f"static const uint32_t atlas_graph_program_words[{len(words)}] = {{",
        *(f"  0x{word:08x}u," for word in words),
        "};",
        f"static const atlas_graph_input_slot atlas_graph_inputs[{input_count}] = {{",
        *input_slots, "};",
        *constant_lines,
        f"static const atlas_graph_constant_slot atlas_graph_constants[{max(1, len(constants))}] = {{",
        *(constant_slots or ["  {0u, 0, 0u},"]), "};",
        f"static const atlas_graph_output_slot atlas_graph_outputs[{output_count}] = {{",
        *(f"  {{0x{address:x}u, 32u, 32u, 16u, 2048u}},"
          for address in output_addresses),
        "};", "#endif", "",
    ]
    suffix = _sha(symbol.encode())[:12]
    binder = f"atlas_bind_graph_{suffix}"
    checked = f"atlas_run_graph_checked_{suffix}"
    result_type = f"atlas_graph_results_{suffix}"
    params = ", ".join(f"const atlas_f32_memref2 *arg{index}"
                       for index in range(input_count))
    arguments = ", ".join(f"arg{index}" for index in range(input_count))
    header = f"""/* Generated C interface for one checked native graph. */
#ifndef ATLAS_GRAPH_MODEL_WRAPPER_H
#define ATLAS_GRAPH_MODEL_WRAPPER_H
#include "atlas_graph_memref.h"
#define ATLAS_GRAPH_BINDING_FUNCTION {binder}
typedef struct {{ atlas_f32_memref2 values[{output_count}]; }} {result_type};
int {binder}(atlas_host *host, atlas_tile_memory_io io, size_t max_polls);
atlas_model_tile_status {checked}({result_type} *result, {params},
                                  atlas_host_result *device_result);
#endif
"""
    source = f"""/* Generated from a checked native graph and typed host bridge. */
#include "atlas_graph_model_wrapper.h"
#include "atlas_graph_data.h"
#ifndef ATLAS_DIAGNOSTIC_GRAPH_PRECISION
#error "Graph f32-to-FP8/BF16 precision is diagnostic until model qualification"
#endif
static atlas_host_buffer active[ATLAS_GRAPH_INPUT_COUNT +
                                ATLAS_GRAPH_CONSTANT_COUNT + ATLAS_GRAPH_OUTPUT_COUNT];
static atlas_graph_memref_binding binding = {{
  .program_words = atlas_graph_program_words,
  .program_word_count = ATLAS_GRAPH_PROGRAM_WORD_COUNT,
  .inputs = atlas_graph_inputs, .input_count = ATLAS_GRAPH_INPUT_COUNT,
  .constants = atlas_graph_constants, .constant_count = ATLAS_GRAPH_CONSTANT_COUNT,
  .outputs = atlas_graph_outputs, .output_count = ATLAS_GRAPH_OUTPUT_COUNT,
  .active_buffers = active,
  .active_buffer_capacity = ATLAS_GRAPH_INPUT_COUNT +
                             ATLAS_GRAPH_CONSTANT_COUNT + ATLAS_GRAPH_OUTPUT_COUNT,
}};
int {binder}(atlas_host *host, atlas_tile_memory_io io, size_t max_polls) {{
  if (!host || host->active || !io.write || !io.read || !max_polls) return 1;
  binding.host = host;
  binding.io = io;
  binding.max_polls = max_polls;
  return 0;
}}
atlas_model_tile_status {checked}({result_type} *result, {params},
                                  atlas_host_result *device_result) {{
  const atlas_f32_memref2 *arguments[] = {{{arguments}}};
  if (!result) return ATLAS_MODEL_TILE_INVALID;
  return atlas_graph_run_memrefs(&binding, arguments, ATLAS_GRAPH_INPUT_COUNT,
                                 result->values, ATLAS_GRAPH_OUTPUT_COUNT,
                                 device_result);
}}
void atlas_graph_ciface_{suffix}({result_type} *result, {params})
    __asm__("_mlir_ciface_{symbol}");
void atlas_graph_ciface_{suffix}({result_type} *result, {params}) {{
  atlas_host_result device_result = ATLAS_HOST_INVALID;
  atlas_model_tile_status status = {checked}(result, {arguments}, &device_result);
  if (status != ATLAS_MODEL_TILE_OK) atlas_model_tile_fail(status, device_result);
}}
"""
    host_entry = f"""/* Call the checked MLIR host entry with its declared output arity. */
#include "atlas_graph_data.h"
#include "atlas_graph_host_entry.h"
void _mlir_ciface_{entry}({", ".join(
    ["const atlas_f32_memref2 *" for _ in range(input_count)] +
    ["atlas_f32_memref2 *" for _ in range(output_count)]
)});
void atlas_graph_invoke_host(const atlas_f32_memref2 *inputs,
                             atlas_f32_memref2 *outputs) {{
  _mlir_ciface_{entry}({", ".join(
    [f"&inputs[{index}]" for index in range(input_count)] +
    [f"&outputs[{index}]" for index in range(output_count)]
  )});
}}
"""
    files = {
        "atlas_graph_data.h": "\n".join(data),
        "atlas_graph_model_wrapper.h": header,
        "atlas_graph_model_wrapper.c": source,
        "atlas_graph_host_entry.h": (
            "#ifndef ATLAS_GRAPH_HOST_ENTRY_H\n"
            "#define ATLAS_GRAPH_HOST_ENTRY_H\n"
            '#include "atlas_graph_memref.h"\n'
            "void atlas_graph_invoke_host(const atlas_f32_memref2 *inputs,\n"
            "                             atlas_f32_memref2 *outputs);\n"
            "#endif\n"
        ),
        "atlas_graph_host_entry.c": host_entry,
    }
    record = {
        "schema": "atlas.diagnostic_graph_model_wrapper.v1",
        "status": "generated_unlinked",
        "engine": "merlin_native",
        "device_symbol": symbol,
        "binding_function": binder,
        "checked_function": checked,
        "source_sha256": bridge["source_sha256"],
        "request_digest": bridge["request_digest"],
        "program_sha256": _sha(binary),
        "native_manifest_sha256": _sha(compilation_bytes),
        "host_bridge_manifest_sha256": _sha(bridge_bytes),
        "input_count": input_count,
        "output_count": output_count,
        "constant_count": len(constants),
        "linked": False,
        "executed": False,
    }
    return record, files


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--linalg", required=True, type=Path)
    parser.add_argument("--host-bridge", required=True, type=Path)
    parser.add_argument("--compiled", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args(argv)
    try:
        if args.out.exists() or args.out.is_symlink():
            raise ValueError("fresh graph wrapper output directory required")
        record, files = prepare(args.linalg, args.host_bridge, args.compiled)
        args.out.mkdir(parents=True)
        for name, content in files.items():
            (args.out / name).write_text(content)
        record["files"] = {name: _sha(content.encode()) for name, content in files.items()}
        (args.out / "manifest.json").write_text(json.dumps(record, sort_keys=True, indent=2) + "\n")
        print(json.dumps({"status": record["status"], "out": str(args.out)}, sort_keys=True))
        return 0
    except (OSError, ValueError, TypeError, KeyError, ImportError, UnicodeError) as error:
        if not args.out.exists() and not args.out.is_symlink():
            args.out.mkdir(parents=True, exist_ok=True)
            (args.out / "manifest.json").write_text(json.dumps({
                "schema": "atlas.diagnostic_graph_model_wrapper.v1",
                "status": "compile_error", "reason": str(error),
            }, sort_keys=True, indent=2) + "\n")
        print(json.dumps({"status": "compile_error", "reason": str(error)}, sort_keys=True),
              file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
