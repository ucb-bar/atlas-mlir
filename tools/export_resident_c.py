#!/usr/bin/env python3
"""Export a resident Atlas image, static data and buffer descriptions as C/H.

The platform supplies loading, cache maintenance and launch code. This exporter
does not generate main(), startup code, a linker script or an ELF.
"""

import argparse
import hashlib
import json
from pathlib import Path


def prepare(compiled, compile_inputs, policy_path, name, *, target_source):
    import numpy as np
    from atlas_mlir_native.binding import (
        DRAM_BASE,
        TILE_BYTES,
        check_io_memory_manifest,
    )
    from atlas_mlir_native.graph_precision import (
        diagnostic_graph_input_spec,
        lower_diagnostic_graph,
    )
    from atlas_mlir_native.resident_graph import memory_plan

    from merlin.compile.c_sources import c_source_pair
    from merlin.frontends.compile_inputs import (
        load_compile_inputs,
        load_lifted_argument_bytes,
    )
    from merlin.semantic_compiler.model import KernelRequest
    from merlin.semantic_compiler.target_binding import verify_native_publication

    manifest = json.loads((compiled / "manifest.json").read_text())
    check_io_memory_manifest(manifest)
    request = KernelRequest.from_record(
        json.loads((compiled / "source-request.json").read_text())
    )
    verify_native_publication(
        compiled,
        manifest,
        engine="merlin_native",
        request_digest=request.digest(),
        target_identity=request.target_identity,
    )
    execution = json.loads((compiled / "execution_plan.json").read_text())
    if (
        execution.get("schema") != "atlas.resident_graph_execution_plan.v1"
        or execution.get("launches_per_invocation") != 1
    ):
        raise ValueError("C data export requires one checked resident graph")
    _, slots, sizes = memory_plan(request)
    locations = {row["node"]: row for row in execution["buffers"]}
    if locations != {
        key: {
            "node": key,
            "address": DRAM_BASE + slot * TILE_BYTES,
            "bytes": sizes[key],
        }
        for key, slot in slots.items()
    }:
        raise ValueError("resident buffers differ from the replayed frame layout")
    staged = load_compile_inputs(compile_inputs)
    precision = json.loads((compiled / "precision-plan.json").read_text())
    policy = json.loads(policy_path.read_text())
    replayed = lower_diagnostic_graph(staged.program, entry=staged.entry, policy=policy)
    if replayed.request.digest() != request.digest() or replayed.record() != precision:
        raise ValueError(
            "C export cannot replay the published source precision transformation"
        )
    if (
        precision["source_sha256"]
        != hashlib.sha256(staged.program.encode()).hexdigest()
        or precision["policy_sha256"]
        != hashlib.sha256(
            json.dumps(policy, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        or manifest["target_inventory_sha256"]
        != hashlib.sha256(
            (target_source / "docs/selected-variant-inventory.json").read_bytes()
        ).hexdigest()
    ):
        raise ValueError(
            "C export source, precision or target differs from compilation"
        )
    weights = load_lifted_argument_bytes(compile_inputs, staged)
    inputs = [n for n in request.nodes if n.op == "input"]
    if len(inputs) != len(staged.argument_bindings):
        raise ValueError("C export argument order differs from capture")
    spec = diagnostic_graph_input_spec(policy)
    static, ports, data = [], [], {}
    for i, node in enumerate(inputs):
        row = {
            **locations[node.id],
            "shape": list(node.type.shape),
            "dtype": node.type.dtype,
            "layout": "mn_tiles_fp8_row_major",
            "source": dict(staged.argument_bindings[i]),
        }
        if i not in weights:
            ports.append(row)
            continue
        values = np.frombuffer(weights[i], dtype="<f4").reshape(node.type.shape)
        codes = spec.encode(values)
        payload = b"".join(
            codes[m : m + 32, n : n + 32].tobytes()
            for m in range(0, codes.shape[0], 32)
            for n in range(0, codes.shape[1], 32)
        )
        if len(payload) != row["bytes"]:
            raise ValueError("static operand differs from published payload size")
        relative = f"data/static-{len(static):03}.bin"
        data[relative] = payload
        static.append({**row, "path": relative})
    from atlas_mlir_native.layout import pack_bf16_matrix

    bindings = {c.node_id: c for c in request.constants}
    # Publication records follow placement order; canonical requests sort by ID.
    constant_ids = [c["node_id"] for c in manifest["constant_inputs"]]
    if len(constant_ids) != len(bindings) or set(constant_ids) != set(bindings):
        raise ValueError("C export constant inventory differs from the source")
    for constant in manifest["constant_inputs"]:
        binding = bindings[constant["node_id"]]
        path = Path(constant["path"])
        if (
            path.is_absolute()
            or ".." in path.parts
            or (compiled / path).is_symlink()
            or not (compiled / path).resolve().is_relative_to(compiled.resolve())
        ):
            raise ValueError("C constant path escapes its compilation")
        payload = (compiled / path).read_bytes()
        if payload != pack_bf16_matrix(
            bytes.fromhex(binding.data_hex), request.node(binding.node_id).type.shape
        ):
            raise ValueError("C constant bytes differ from source semantics")
        relative = f"data/static-{len(static):03}.bin"
        data[relative] = payload
        static.append({**locations[constant["node_id"]], "path": relative})
    output = [
        {
            **locations[key],
            "shape": list(request.node(key).type.shape),
            "dtype": "bf16",
            "layout": "mn_tiles_bf16_pair_halves",
        }
        for key in request.outputs
    ]
    immutable = {row["node"] for row in execution["inputs"]}
    written = [row for row in execution["buffers"] if row["node"] not in immutable]
    begin = min(row["address"] for row in written)
    end = max(
        row["address"] + ((row["bytes"] + TILE_BYTES - 1) // TILE_BYTES) * TILE_BYTES
        for row in written
    )
    binary = (compiled / "program.bin").read_bytes()
    upper = name.upper()
    declarations = f"""
#define {upper}_PROGRAM_WORDS {len(binary) // 4}u
#define {upper}_INPUT_COUNT {len(ports)}u
#define {upper}_OUTPUT_COUNT {len(output)}u
#define {upper}_STATIC_COUNT {len(static)}u
#define {upper}_DEVICE_BUFFER_COUNT {len(execution["inputs"]) + 1}u
#define {upper}_INPUT_SCALE_EXPONENT {spec.pre_encode_scale_exponent}
typedef struct {{ uintptr_t address; size_t bytes; const uint8_t *data; }} {name}_static_segment;
typedef struct {{ uintptr_t address; size_t bytes; uint32_t rows, columns; }} {name}_tensor;
extern const uint32_t {name}_program[{upper}_PROGRAM_WORDS];
extern const {name}_static_segment {name}_static_data[{max(1, len(static))}];
/* Inputs: packed E4M3 bytes. Outputs: BF16 register halves, in M/N tile order.
 * Shapes and the exact numerical policy are also recorded in manifest.json. */
extern const {name}_tensor {name}_inputs[{max(1, len(ports))}];
extern const {name}_tensor {name}_outputs[{upper}_OUTPUT_COUNT];
/* Read-only operands followed by one private writable DRAM arena. */
extern const atlas_host_buffer {name}_device_buffers[{upper}_DEVICE_BUFFER_COUNT];
"""
    words = ",".join(
        f"0x{int.from_bytes(binary[i : i + 4], 'little'):08x}u"
        for i in range(0, len(binary), 4)
    )
    definitions = [f"const uint32_t {name}_program[] = {{{words}}};"]
    for i, row in enumerate(static):
        values = ",".join(f"0x{byte:02x}u" for byte in data[row["path"]])
        definitions.append(f"static const uint8_t {name}_data_{i}[] = {{{values}}};")
    initializer = (
        ",".join(
            f"{{{r['address']}u,{r['bytes']}u,{name}_data_{i}}}"
            for i, r in enumerate(static)
        )
        or "{0,0,0}"
    )
    definitions.append(
        f"const {name}_static_segment {name}_static_data[] = {{{initializer}}};"
    )
    for kind, rows in (("inputs", ports), ("outputs", output)):
        initializer = (
            ",".join(
                f"{{{r['address']}u,{r['bytes']}u,{r['shape'][0]}u,{r['shape'][1]}u}}"
                for r in rows
            )
            or "{0,0,0,0}"
        )
        definitions.append(f"const {name}_tensor {name}_{kind}[] = {{{initializer}}};")
    buffers = ",".join(
        f"{{{r['address']}u,{r['bytes']}u,ATLAS_HOST_DEVICE_READ}}"
        for r in execution["inputs"]
    )
    buffers += f",{{{begin}u,{end - begin}u,ATLAS_HOST_DEVICE_WRITE}}"
    definitions.append(
        f"const atlas_host_buffer {name}_device_buffers[] = {{{buffers}}};"
    )
    rendered = c_source_pair(
        name,
        public_headers={
            "atlas_host.h": (target_source / "runtime/atlas_host.h").read_text()
        },
        private_headers={},
        sources={"image.c": "\n".join(definitions)},
        declarations=declarations,
    )
    files = {key: value.encode() for key, value in rendered.items()}
    files.update(data)
    files["program.bin"] = binary
    record = {
        "schema": "atlas.resident_c_package.v1",
        "engine": "merlin_native",
        "artifact_kind": "program",
        "program_sha256": hashlib.sha256(binary).hexdigest(),
        "compilation_manifest_sha256": hashlib.sha256(
            (compiled / "manifest.json").read_bytes()
        ).hexdigest(),
        "target_identity": request.target_identity,
        "input_conversion": spec.record(),
        "inputs": ports,
        "outputs": output,
        "static_data": static,
        "private_dram": {"address": begin, "bytes": end - begin},
        "qualification": "diagnostic; platform memory map and cache maintenance require matching",
        "files": {
            key: hashlib.sha256(value).hexdigest() for key, value in files.items()
        },
    }
    verify_native_publication(
        compiled,
        manifest,
        engine="merlin_native",
        request_digest=request.digest(),
        target_identity=request.target_identity,
    )
    if hashlib.sha256(binary).hexdigest() != manifest["binary_sha256"]:
        raise ValueError("resident image changed during C export")
    return record, files


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("compiled", "compile-inputs", "policy", "out"):
        parser.add_argument("--" + key, type=Path, required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument(
        "--io-memory-bytes",
        type=int,
        help="same external-buffer window used for compilation",
    )
    parser.add_argument(
        "--target-source", type=Path, default=Path(__file__).resolve().parents[1]
    )
    args = parser.parse_args()
    if args.out.exists():
        parser.error("fresh C package directory required")
    from merlin.semantic_compiler.codegen import CodegenOptions, codegen_options

    with codegen_options(CodegenOptions(io_memory_bytes=args.io_memory_bytes)):
        record, files = prepare(
            args.compiled,
            args.compile_inputs,
            args.policy,
            args.name,
            target_source=args.target_source,
        )
    args.out.mkdir(parents=True, exist_ok=False)
    for name, payload in files.items():
        path = args.out / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)
    (args.out / "manifest.json").write_text(
        json.dumps(record, indent=2, sort_keys=True) + "\n"
    )
    print(
        json.dumps(
            {
                "status": "exported",
                "out": str(args.out),
                "program_sha256": record["program_sha256"],
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
