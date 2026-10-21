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
    precision_bytes = _member(compiled, "precision-plan.json")
    plan_bytes = _member(compiled, "execution_plan.json")
    precision = json.loads(precision_bytes)
    plan = json.loads(plan_bytes)
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
        "target_identity": compilation["target_identity"],
        "execution_plan_sha256": _sha(plan_bytes),
        "precision_plan_sha256": _sha(precision_bytes),
        "native_manifest_sha256": _sha(compilation_bytes),
        "host_bridge_manifest_sha256": _sha(bridge_bytes),
        "input_count": input_count,
        "output_count": output_count,
        "constant_count": len(constants),
        "source_inputs": [f"arg{index}" for index in range(input_count)],
        "source_outputs": bridge["source_outputs"],
        "linked": False,
        "executed": False,
    }
    return record, files


def prepare_c_package(record: dict, generated: dict[str, str], compiled: Path,
                      name: str, *, compile_inputs: Path | None = None) -> tuple[dict, dict[str, bytes]]:
    """Package the existing checked callback and runtime without a host main."""
    from merlin.common.paths import merlin_dir
    from merlin.compile.c_sources import c_source_pair

    runtime = Path(__file__).resolve().parents[1] / "runtime"
    public = {member: _member(runtime, member).decode()
              for member in ("atlas_host.h", "atlas_model_tile.h")}
    private = {
        "finite_float_boundary.h": (Path(merlin_dir()) / "runtime/c/finite_float_boundary.h").read_text(),
        "atlas_graph_memref.h": _member(runtime, "atlas_graph_memref.h").decode(),
        **{member: generated[member] for member in (
            "atlas_graph_data.h", "atlas_graph_model_wrapper.h")},
    }
    sources = {member: _member(runtime, member).decode()
               for member in ("atlas_host.c", "atlas_model_tile.c", "atlas_graph_memref.c")}
    sources["atlas_graph_model_wrapper.c"] = generated["atlas_graph_model_wrapper.c"]
    inputs, outputs = record["input_count"], record["output_count"]
    suffix = _sha(record["device_symbol"].encode())[:12]
    result_type = f"atlas_graph_results_{suffix}"
    arguments = ", ".join(f"&views[{index}]" for index in range(inputs))
    declarations = f"""
#define {name.upper()}_INPUT_COUNT {inputs}u
#define {name.upper()}_OUTPUT_COUNT {outputs}u
#define {name.upper()}_TENSOR_ELEMENTS 1024u
#define {name.upper()}_DIAGNOSTIC_PRECISION 1
int {name}_bind(atlas_host *host, atlas_tile_memory_io io, size_t max_polls);
/* Dense 32x32 binary32 host buffers; selected FP8/BF16 conversion is inside.
 * Output buffers must not overlap. Calls are synchronous and not reentrant.
 * Finish a timed-out device launch before calling again or reusing buffers. */
atlas_model_tile_status {name}_run(const float *const *inputs,
                                  float *const *outputs,
                                  atlas_host_result *device_result);
const uint32_t *{name}_program(size_t *word_count);
/* Merlin serial-dispatch ABI. Uses the same binding and precision as _run;
 * context is unused. Zero means the device call and readback have completed. */
int {name}_dispatch(void *context, const void *const *inputs, void *const *outputs);
"""
    implementation = f"""
int {name}_bind(atlas_host *host, atlas_tile_memory_io io, size_t max_polls) {{
  return {record['binding_function']}(host, io, max_polls);
}}
const uint32_t *{name}_program(size_t *word_count) {{
  if (word_count) *word_count = ATLAS_GRAPH_PROGRAM_WORD_COUNT;
  return atlas_graph_program_words;
}}
atlas_model_tile_status {name}_run(const float *const *inputs,
                                  float *const *outputs,
                                  atlas_host_result *device_result) {{
  if (device_result) *device_result = ATLAS_HOST_INVALID;
  if (!inputs || !outputs) return ATLAS_MODEL_TILE_INVALID;
  atlas_f32_memref2 views[{inputs}];
  for (size_t i = 0; i < {inputs}u; ++i) {{
    if (!inputs[i]) return ATLAS_MODEL_TILE_INVALID;
    views[i] = (atlas_f32_memref2){{(float *)inputs[i], (float *)inputs[i], 0,
                                 {{32, 32}}, {{32, 1}}}};
  }}
  for (size_t i = 0; i < {outputs}u; ++i) {{
    uintptr_t begin = (uintptr_t)outputs[i];
    if (!begin || begin > UINTPTR_MAX - 4096u) return ATLAS_MODEL_TILE_INVALID;
    for (size_t j = 0; j < i; ++j) {{
      uintptr_t other = (uintptr_t)outputs[j];
      if (begin < other + 4096u && other < begin + 4096u)
        return ATLAS_MODEL_TILE_INVALID;
    }}
  }}
  {result_type} result = {{0}};
  atlas_model_tile_status status = {record['checked_function']}(
      &result, {arguments}, device_result);
  if (status != ATLAS_MODEL_TILE_OK) return status;
  for (size_t i = 0; i < {outputs}u; ++i) {{
    __builtin_memcpy(outputs[i], result.values[i].aligned, 4096u);
    free(result.values[i].allocated);
  }}
  return ATLAS_MODEL_TILE_OK;
}}
int {name}_dispatch(void *context, const void *const *inputs, void *const *outputs) {{
  (void)context;
  if (!inputs || !outputs) return ATLAS_MODEL_TILE_INVALID;
  const float *typed_inputs[{inputs}];
  float *typed_outputs[{outputs}];
  for (size_t i = 0; i < {inputs}u; ++i) typed_inputs[i] = (const float *)inputs[i];
  for (size_t i = 0; i < {outputs}u; ++i) typed_outputs[i] = (float *)outputs[i];
  atlas_host_result device;
  return (int){name}_run(typed_inputs, typed_outputs, &device);
}}
"""
    frozen = None
    if compile_inputs is not None:
        import numpy as np
        from merlin.compile.c_parameters import freeze_c_arguments
        from merlin.frontends.compile_inputs import (
            load_compile_inputs,
            load_lifted_argument_bytes,
        )

        frozen = freeze_c_arguments(
            compile_inputs, name=name, source_sha256=record["source_sha256"])
        if len(frozen.arguments) != inputs:
            raise ValueError("staged argument count differs from compiled graph")
        staged = load_compile_inputs(compile_inputs)
        precision = json.loads(_member(compiled, "precision-plan.json"))
        for index, payload in load_lifted_argument_bytes(compile_inputs, staged).items():
            _float_spec(precision["boundary_conversions"][index]["conversion_spec"]).encode(
                np.frombuffer(payload, dtype="<f4"))
        count = len(frozen.runtime_inputs)
        declarations += f"""
#define {name.upper()}_MODEL_INPUT_COUNT {count}u
/* Declared parameters are embedded. Only runtime inputs are passed here,
 * in captured input order. The binding, outputs and status match _run. */
atlas_model_tile_status {name}_model_run(const float *const *inputs,
                                        float *const *outputs,
                                        atlas_host_result *device_result);
int {name}_model_dispatch(void *context, const void *const *inputs, void *const *outputs);
"""
        implementation += frozen.declarations + f"""
atlas_model_tile_status {name}_model_run(const float *const *inputs,
                                        float *const *outputs,
                                        atlas_host_result *device_result) {{
  if (device_result) *device_result = ATLAS_HOST_INVALID;
  {'if (!inputs) return ATLAS_MODEL_TILE_INVALID;' if count else '(void)inputs;'}
  const float *arguments[{inputs}] = {{{', '.join(frozen.arguments)}}};
  return {name}_run(arguments, outputs, device_result);
}}
int {name}_model_dispatch(void *context, const void *const *inputs, void *const *outputs) {{
  (void)context;
  if ({'!inputs || ' if count else ''}!outputs) return ATLAS_MODEL_TILE_INVALID;
  const float *typed_inputs[{max(1, count)}] = {{0}};
  float *typed_outputs[{outputs}];
  for (size_t i = 0; i < {count}u; ++i) typed_inputs[i] = (const float *)inputs[i];
  for (size_t i = 0; i < {outputs}u; ++i) typed_outputs[i] = (float *)outputs[i];
  atlas_host_result device;
  return (int){name}_model_run(typed_inputs, typed_outputs, &device);
}}
"""
    pair = c_source_pair(name, public_headers=public, private_headers=private,
                         sources=sources, declarations=declarations,
                         implementation=implementation)
    files = {member: text.encode() for member, text in pair.items()}
    compilation_bytes = _member(compiled, "manifest.json")
    binary = _member(compiled, "program.bin")
    execution_plan = _member(compiled, "execution_plan.json")
    precision_plan = _member(compiled, "precision-plan.json")
    if (_sha(compilation_bytes) != record["native_manifest_sha256"] or
            _sha(binary) != record["program_sha256"] or
            _sha(execution_plan) != record["execution_plan_sha256"] or
            _sha(precision_plan) != record["precision_plan_sha256"]):
        raise ValueError("native compilation changed during C export")
    files[name + ".bin"] = binary
    files[name + ".words.txt"] = "".join(
        f"{word:08x}\n" for (word,) in struct.iter_unpack("<I", binary)).encode()
    for item in json.loads(compilation_bytes)["constant_inputs"]:
        if Path(item["path"]).parts[0] != "constants":
            raise ValueError("C export constants must use the constants directory")
        data = _member(compiled, item["path"])
        if _sha(data) != item["sha256"]:
            raise ValueError("constant changed during C export")
        files[item["path"]] = data
    files["README.md"] = f"""# {name}

Compile `{name}.c` and include `{name}.h` in your application. The C file embeds
the device program, constants and selected conversion tables; no repository
include path or external binary loader is required. `{name}.bin` contains
little-endian 32-bit Atlas words; `{name}.words.txt` lists them in hex.
`constants/` contains the same constant payloads embedded in the C file.

Use C11 and `-DATLAS_DIAGNOSTIC_GRAPH_PRECISION=1`. This prototype uses a selected
FP8/BF16 precision policy; original-model quality and FPGA execution are not
qualified by this export. Link against the platform C allocator (`malloc` and
`free`) and ordinary C memory functions. Do not use fast-math options.
Intermediate value-domain assumptions remain in `device_layout` in the manifest;
the runtime does not check all of them.

The application supplies startup, linker script, IMEM/CSR map, MMIO callbacks,
DRAM read/write callbacks and cache maintenance. Reserve the device addresses
listed in `manifest.json`. No `main`, mailbox loop, linker script or host ELF is
included. The source includes the existing Atlas load/start/wait implementation.

1. Initialize an `atlas_host` with `atlas_host_init` and the platform services.
2. Call `{name}_bind(&host, io, max_polls)` and check that it returns zero.
3. Pass {inputs} input pointers and {outputs} output pointers to `{name}_run`.
   Each points to 1024 contiguous `float` elements. Check both returned statuses.

Outputs are caller-owned. On failure they remain unchanged. A timeout leaves
the device active: finish or reset that launch before reusing its buffers.
Calls use one static binding and must not overlap.

Input order is the source function's `arg0`, `arg1`, etc. Output order is
`source_outputs` in the manifest. This exports one whole-graph Atlas call;
it does not translate arbitrary CPU computation from LLVM into C.

For a Merlin serial dispatch program, bind `{name}_dispatch` as the compiled
pointer-ABI implementation of this graph node. It calls the same synchronous
`{name}_run`, ignores its context pointer and returns zero only after readback.
Host operations before or after it need their own compiled wrappers; there is
no CPU fallback when the device call fails.
""".encode()
    result = dict(record)
    result.update({
        "schema": "atlas.graph_c_package.v1", "package_name": name,
        "entry_function": name + "_run", "binding_function": name + "_bind",
        "dispatch_function": name + "_dispatch", "dispatch_completion": "synchronous",
        "lowering_pipeline": json.loads(compilation_bytes).get("lowering_pipeline", "unspecified"),
        "source_members": {member: _sha(text.encode()) for member, text in
                           {**public, **private, **sources}.items()},
        "device_layout": json.loads(execution_plan),
        "precision_plan": json.loads(precision_plan),
        "startup_included": False, "elf_included": False,
    })
    if frozen is not None:
        result["model_interface"] = {
            "entry_function": name + "_model_run",
            "dispatch_function": name + "_model_dispatch",
            "runtime_inputs": list(frozen.runtime_inputs),
            "parameters": list(frozen.parameters),
            "weights_sha256": frozen.weights_sha256,
            "state": "stateless",
        }
        files["README.md"] += f"""
## Captured model entry

`{name}_model_run` embeds the declared parameters and accepts only the
{len(frozen.runtime_inputs)} runtime input pointers listed in
`model_interface.runtime_inputs` in the manifest. Use
`{name.upper()}_MODEL_INPUT_COUNT` for this pointer array. Outputs and platform
binding are the same as `{name}_run`. `{name}_model_dispatch` provides the
corresponding Merlin dispatch entry. Neither entry reads a capture directory
at runtime. Mutable captured buffers require a separate state interface and
are rejected by this export.
""".encode()
    return result, files


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--linalg", type=Path)
    source.add_argument("--request", type=Path, help="typed contraction or row-softmax request for C/H export")
    parser.add_argument("--host-bridge", type=Path)
    parser.add_argument("--linalg-entry", help="source entry for an explicit row-softmax precision policy")
    parser.add_argument("--diagnostic-graph-policy", type=Path,
                        help="source row-softmax precision policy; no LLVM host bridge is required")
    parser.add_argument("--compiled", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--c-package-name", help="emit one self-contained C/H pair and device binaries")
    parser.add_argument("--runtime-linkage", choices=("embedded", "separate"), default="embedded",
                        help="link the Atlas host runtime once when combining region C files")
    parser.add_argument("--compile-inputs", type=Path,
                        help="embed declared parameters from a checked compiler-input stage")
    args = parser.parse_args(argv)
    if args.c_package_name and args.out.resolve().is_relative_to(args.compiled.resolve()):
        print(json.dumps({"status": "compile_error",
                          "reason": "C package output must be outside the native compilation"}),
              file=sys.stderr)
        return 2
    try:
        if args.runtime_linkage == "separate" and (not args.c_package_name or
                args.request is None and args.diagnostic_graph_policy is None):
            raise ValueError("separate runtime linkage requires a contraction or row-softmax C package")
        if args.compile_inputs is not None and not args.c_package_name:
            raise ValueError("--compile-inputs requires --c-package-name")
        if args.out.exists() or args.out.is_symlink():
            raise ValueError("fresh graph wrapper output directory required")
        if args.diagnostic_graph_policy is not None:
            if args.linalg is None or not args.c_package_name or args.host_bridge is not None:
                raise ValueError("source row-softmax export requires --linalg, --c-package-name and no host bridge")
            from atlas_mlir_native.softmax_c_source import prepare_source_c_package

            record, payloads = prepare_source_c_package(
                args.linalg,args.diagnostic_graph_policy,args.compiled,args.c_package_name,
                entry=args.linalg_entry,target_source=Path(__file__).resolve().parents[1],
                compile_inputs=args.compile_inputs, runtime_linkage=args.runtime_linkage)
        elif args.linalg_entry is not None:
            raise ValueError("--linalg-entry requires --diagnostic-graph-policy")
        elif args.request is not None:
            if not args.c_package_name or args.host_bridge is not None:
                raise ValueError("--request requires --c-package-name and no graph bridge")
            from merlin.semantic_compiler.model import KernelRequest

            request = KernelRequest.from_record(json.loads(args.request.read_text()))
            target_source = Path(__file__).resolve().parents[1]
            if request.nodes[-1].op == "softmax":
                if args.compile_inputs is not None:
                    raise ValueError("row-softmax export takes an explicit BF16 request, not capture parameters")
                from atlas_mlir_native.softmax_c import (
                    prepare_c_package as prepare_softmax_c,
                )

                record, payloads = prepare_softmax_c(args.request,args.compiled,args.c_package_name,
                                                    target_source=target_source, runtime_linkage=args.runtime_linkage)
            else:
                from atlas_mlir_native.looped_c import (
                    prepare_c_package as prepare_looped_c,
                )

                record, payloads = prepare_looped_c(args.request,args.compiled,args.c_package_name,
                                                   target_source=target_source,
                                                   compile_inputs=args.compile_inputs, runtime_linkage=args.runtime_linkage)
        else:
            if args.host_bridge is None:
                raise ValueError("--linalg requires its checked --host-bridge")
            record, files = prepare(args.linalg, args.host_bridge, args.compiled)
            if args.c_package_name:
                record, payloads = prepare_c_package(record, files, args.compiled, args.c_package_name,
                                                    compile_inputs=args.compile_inputs)
            else:
                payloads = {name: content.encode() for name, content in files.items()}
        args.out.mkdir(parents=True)
        for name, content in payloads.items():
            (args.out / name).parent.mkdir(parents=True, exist_ok=True)
            (args.out / name).write_bytes(content)
        record["files"] = {name: _sha(content) for name, content in payloads.items()}
        (args.out / "manifest.json").write_text(json.dumps(record, sort_keys=True, indent=2) + "\n")
        print(json.dumps({"status": record["status"], "out": str(args.out)}, sort_keys=True))
        return 0
    except (OSError, ValueError, TypeError, KeyError, ImportError, UnicodeError, RuntimeError) as error:
        status = getattr(error, "status", "compile_error")
        if not args.out.exists() and not args.out.is_symlink():
            args.out.mkdir(parents=True, exist_ok=True)
            (args.out / "manifest.json").write_text(json.dumps({
                "schema": ("atlas.row_softmax_c_package.v1" if args.diagnostic_graph_policy is not None else
                           "atlas.looped_c_package.v1" if args.request is not None else
                           "atlas.diagnostic_graph_model_wrapper.v1"),
                "status": status, "reason": str(error),
            }, sort_keys=True, indent=2) + "\n")
        print(json.dumps({"status": status, "reason": str(error)}, sort_keys=True),
              file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
