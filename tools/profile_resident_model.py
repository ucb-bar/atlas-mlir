#!/usr/bin/env python3
"""Check and measure a connected resident graph on the pinned npu_model.

This evaluator consumes runtime samples only after compilation. Device cycles
and optional native C runtime timings are separate execution tiers.
"""

import argparse
import hashlib
import importlib.metadata
import json
import struct
import sys
from dataclasses import replace
from functools import lru_cache
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "test"))


def physical_bytes(value):
    """Independent layout indexing; do not call compiler packing helpers."""
    rows, columns = value.type.shape
    width = 2 if value.type.dtype == "bf16" else 1
    result = bytearray()
    for mr in range(0, rows, 32):
        for nc in range(0, columns, 32):
            tile = bytearray(1024 * width)
            for r in range(32):
                for c in range(32):
                    at = (
                        (c // 16) * 1024 + (r * 16 + c % 16) * 2
                        if width == 2
                        else r * 32 + c
                    )
                    code = value.elements[(mr + r) * columns + nc + c]
                    tile[at : at + width] = code.to_bytes(width, "little")
            result.extend(tile)
    return bytes(result)


def reference_values(request, arguments):
    from atlas_mlir_native import reference
    from atlas_mlir_native.binding import (
        MXU0_INPUT_DOMAIN,
        MXU0_POLICY,
        ROUNDED_PACK_POLICY,
    )
    from rtl.graph_semantic_checker import _raw_bf16
    from test_mxu_numerics import dot

    from merlin.semantic_compiler.reference import TensorValue, evaluate_graph

    @lru_cache(maxsize=65536)
    def checked_dot(weights, activations):
        for code in (*weights, *activations):
            if code and not (8 <= code <= 119 or 136 <= code <= 247):
                raise ValueError(
                    "packed producer exceeds the selected MXU domain: "
                    + MXU0_INPUT_DOMAIN
                )
        return dot(weights, activations)

    def contraction(node, operands):
        a, w = operands
        m, k = a.type.shape
        n, wk = w.type.shape
        if k != wk or node.type.shape != (m, n):
            raise ValueError("reference contraction shape mismatch")
        return TensorValue(
            node.type,
            tuple(
                checked_dot(
                    w.elements[j * k : (j + 1) * k], a.elements[i * k : (i + 1) * k]
                )
                for i in range(m)
                for j in range(n)
            ),
        )

    def pack(node, operands):
        source = operands[0]
        return reference.evaluate_rounded_scaled_pack(
            node,
            (
                TensorValue(
                    replace(source.type, numerical_policy="raw-bit-pattern"),
                    source.elements,
                ),
            ),
        )

    callbacks = {
        ("matmul", MXU0_POLICY): contraction,
        ("pack_fp8", ROUNDED_PACK_POLICY): pack,
    }

    def binary(node, operands):
        if any(
            v.type.dtype != "bf16"
            or v.type.numerical_policy not in {"raw-bit-pattern", MXU0_POLICY}
            for v in operands
        ):
            raise ValueError(
                "reference pointwise input has an unsupported BF16 representation"
            )
        normalized = tuple(
            TensorValue(replace(v.type, numerical_policy="raw-bit-pattern"), v.elements)
            for v in operands
        )
        return reference.evaluate_bf16_binary_zero(node, normalized)

    for op in ("add", "sub", "mul"):
        callbacks[(op + "_bf16_zero", "raw-bit-pattern")] = binary
    for op in ("exp", "recip", "row_max", "row_sum"):
        callbacks[(op + "_bf16", "raw-bit-pattern")] = _raw_bf16(
            getattr(reference, f"evaluate_bf16_{op}")
        )
    constants = {
        c.node_id: TensorValue(
            request.node(c.node_id).type,
            tuple(
                struct.unpack(f"<{len(c.data_hex) // 4}H", bytes.fromhex(c.data_hex))
            ),
        )
        for c in request.constants
    }
    roots = tuple(n.id for n in request.nodes if n.effect == "pure")
    observed = evaluate_graph(
        replace(request, outputs=roots, output_storages=("external",) * len(roots)),
        arguments,
        constants=constants,
        primitives=callbacks,
    )
    return {**arguments, **constants, **dict(zip(roots, observed, strict=True))}


def rtl_fixture(root, compiled, request, execution, values, variant, max_cycles):
    """Declare immutable inputs and compare every byte in the private arena."""
    folder = root / f"input{variant}"
    folder.mkdir(parents=True, exist_ok=False)
    immutable = {row["node"] for row in execution["inputs"]}
    written = [row for row in execution["buffers"] if row["node"] not in immutable]
    first = min(row["address"] for row in written)
    end = max(
        row["address"] + ((row["bytes"] + 2047) // 2048) * 2048 for row in written
    )
    expected = bytearray(b"\xa5" * (end - first))
    for row in written:
        payload = physical_bytes(values[row["node"]])
        offset = row["address"] - first
        expected[offset : offset + len(payload)] = payload
    binary = (compiled / "program.bin").read_bytes()
    (folder / "program.bin").write_bytes(binary)
    (folder / "expected.bin").write_bytes(expected)
    properties = [
        "schema=atlas.rtl_tile_fixture.v3",
        f"name=resident_wide_{variant}",
        f"program_sha256={hashlib.sha256(binary).hexdigest()}",
        f"output.address={first:#x}",
        f"output_count={len(expected) // 2048}",
        f"max_cycles={max_cycles}",
    ]
    count = 0
    for row in execution["inputs"]:
        payload = physical_bytes(values[row["node"]])
        width = 2048 if request.node(row["node"]).type.dtype == "bf16" else 1024
        for offset in range(0, len(payload), width):
            (folder / f"input-{count}.bin").write_bytes(
                payload[offset : offset + width]
            )
            properties.append(f"input.{count}.address={row['address'] + offset:#x}")
            count += 1
    properties.append(f"input_count={count}")
    (folder / "manifest.properties").write_text("\n".join(properties) + "\n")
    return {
        "path": str(folder),
        "program_sha256": hashlib.sha256(binary).hexdigest(),
        "input_count": count,
        "output_bytes": len(expected),
        "reference": "independent_ordered_arithmetic_and_byte_layout",
    }


def measure(args):
    import numpy as np
    import torch
    from atlas_mlir_native.binding import check_io_memory_manifest
    from atlas_mlir_native.graph_precision import diagnostic_graph_input_spec
    from npu_model.configs import isa_definition as isa
    from npu_model.configs.hardware.default import DefaultHardwareConfig
    from npu_model.logging import LoggerConfig
    from npu_model.simulation import Simulation
    from npu_model.software.program import InstantiableProgram
    from tools.check_program_decode import decode_binary
    from tools.model_instructions import model_instruction

    from merlin.frontends.compile_inputs import (
        load_compile_inputs,
        load_lifted_argument_bytes,
    )
    from merlin.semantic_compiler.model import KernelRequest
    from merlin.semantic_compiler.reference import TensorValue
    from merlin.semantic_compiler.target_binding import verify_native_publication

    revision = "337e9f249f938a1dff007ddf8a763e5cd4b3aa2b"
    origin = json.loads(
        importlib.metadata.distribution("npu-model").read_text("direct_url.json")
        or "{}"
    )
    if (
        revision not in origin.get("url", "")
        and origin.get("vcs_info", {}).get("commit_id") != revision
    ):
        raise ValueError("install the pinned model revision")
    torch.set_num_threads(1)
    root = args.compiled
    manifest = json.loads((root / "manifest.json").read_text())
    check_io_memory_manifest(manifest)
    request = KernelRequest.from_record(
        json.loads((root / "source-request.json").read_text())
    )
    verify_native_publication(
        root,
        manifest,
        engine="merlin_native",
        request_digest=request.digest(),
        target_identity=request.target_identity,
    )
    execution = json.loads((root / "execution_plan.json").read_text())
    if (
        execution.get("schema") != "atlas.resident_graph_execution_plan.v1"
        or execution.get("launches_per_invocation") != 1
    ):
        raise ValueError("resident profiler needs one complete connected image")
    staged = load_compile_inputs(args.compile_inputs)
    precision = json.loads((root / "precision-plan.json").read_text())
    policy = json.loads(args.policy.read_text())
    if (
        precision["source_sha256"]
        != hashlib.sha256(staged.program.encode()).hexdigest()
        or precision["policy_sha256"]
        != hashlib.sha256(
            json.dumps(policy, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
    ):
        raise ValueError("profile source or precision differs from compilation")
    spec = diagnostic_graph_input_spec(policy)
    weights = load_lifted_argument_bytes(args.compile_inputs, staged)
    inputs = [n for n in request.nodes if n.op == "input"]
    if len(inputs) != len(staged.argument_bindings):
        raise ValueError("capture argument order differs")
    panels = []
    for variant in range(2):
        arguments = {}
        for i, node in enumerate(inputs):
            # Generated panels are evaluation inputs, never compilation inputs.
            values = (
                np.frombuffer(weights[i], dtype="<f4").reshape(node.type.shape)
                if i in weights
                else (
                    (
                        (
                            np.arange(np.prod(node.type.shape)).reshape(node.type.shape)
                            % 23
                        )
                        - 11
                    )
                    * (0.03125 if variant else 0.0625)
                ).astype(np.float32)
            )
            arguments[node.id] = TensorValue(
                node.type, tuple(map(int, spec.encode(values).ravel()))
            )
        panels.append(reference_values(request, arguments))
    if args.rtl_fixtures is not None:
        args.rtl_fixtures.mkdir(parents=True, exist_ok=False)
        fixtures = [
            rtl_fixture(
                args.rtl_fixtures,
                root,
                request,
                execution,
                values,
                i,
                args.rtl_max_cycles,
            )
            for i, values in enumerate(panels)
        ]
        (args.rtl_fixtures / "fixtures.json").write_text(
            json.dumps(fixtures, indent=2) + "\n"
        )
    binary = (root / "program.bin").read_bytes()
    inventory = json.loads((ROOT / "docs/selected-variant-inventory.json").read_text())
    instructions = [
        model_instruction(row, inventory, isa)
        for row in decode_binary(inventory, binary)["instructions"]
    ]
    locations = {row["node"]: row for row in execution["buffers"]}
    base = min(row["address"] for row in locations.values()) - 64
    stop = max(row["address"] + row["bytes"] for row in locations.values()) + 64
    immutable = {row["node"] for row in execution["inputs"]}
    work = args.out.with_suffix(".work")
    work.mkdir(parents=True, exist_ok=False)
    host = None
    if getattr(args, "profile_host_runtime", False):
        from tools.export_resident_c import prepare
        from tools.profile_host_runtime import HostRuntimeProfile

        record, files = prepare(
            root, args.compile_inputs, args.policy, "resident_image", target_source=ROOT
        )
        host = HostRuntimeProfile(work / "host-runtime", record, files, binary, ROOT)
    samples = []
    for iteration in range(args.warmup + args.repetitions):
        expected_values = panels[iteration % 2]
        memory = bytearray((i * 17 + 31) % 256 for i in range(stop - base))
        expected = bytearray(memory)
        for key, location in locations.items():
            payload = physical_bytes(expected_values[key])
            if len(payload) != location["bytes"]:
                raise ValueError(
                    "independent layout differs from published buffer size"
                )
            offset = location["address"] - base
            expected[offset : offset + len(payload)] = payload
            if key in immutable:
                memory[offset : offset + len(payload)] = payload
        instance = InstantiableProgram(instructions)
        instance.dram_base = base
        instance.memory_regions = [(0, torch.tensor(list(memory), dtype=torch.uint8))]
        config = DefaultHardwareConfig()
        config.arch_state_config = replace(
            config.arch_state_config,
            numerics="rtl",
            randomize_init=True,
            init_seed=73 + iteration,
        )
        simulation = Simulation(
            config,
            LoggerConfig(filename=str(work / f"run-{iteration}.log")),
            instance,
            verbose=False,
        )
        state = simulation.core.arch_state
        try:
            if host is not None:
                host.begin(state)
            simulation.run(max_cycles=2000000)
            if not state.halted or simulation.get_stats().runtime_errors:
                raise ValueError("resident model execution failed or timed out")
            actual = bytes(state.read_dram(base, len(memory)).tolist())
            if actual != bytes(expected):
                first = next(
                    i
                    for i, (a, b) in enumerate(zip(actual, expected, strict=True))
                    if a != b
                )
                raise ValueError(
                    f"resident output/intermediate/input/guard mismatch at 0x{base + first:x}: "
                    f"{actual[first]:02x} != {expected[first]:02x}"
                )
            host_sample = host.end(state) if host is not None else None
            samples.append(
                {
                    "input_set": iteration % 2,
                    "warmup": iteration < args.warmup,
                    "device_cycles": simulation.get_stats().cycles,
                    **(
                        {"host_runtime": host_sample} if host_sample is not None else {}
                    ),
                }
            )
        finally:
            state.close()
    if host is not None:
        host_samples = [row["host_runtime"] for row in samples]
        if (
            sum(row["image_loads"] for row in host_samples) != 1
            or sum(row["image_reuses"] for row in host_samples) != len(samples) - 1
            or sum(row["words_written"] for row in host_samples) != len(binary) // 4
        ):
            raise ValueError("expected one cold image load followed by residency reuse")
    physical = json.loads((root / "physical-program.json").read_text())
    return {
        "correctness": "passed",
        "execution_tier": "npu_model_with_native_C_control"
        if host is not None
        else "npu_model_device_image",
        "model_revision": revision,
        "program_sha256": hashlib.sha256(binary).hexdigest(),
        "program_bytes": len(binary),
        "device_cycles": [r["device_cycles"] for r in samples if not r["warmup"]],
        "samples": samples,
        "spills": physical["spill_stores"],
        "reloads": physical["spill_reloads"],
        "checks": {
            "input_sets": 2,
            "outputs": len(request.outputs),
            "stored_buffers": len(locations),
            "launches_per_invocation": 1,
            "inputs_and_guards": "unchanged",
        },
        "host_timing": host.provenance
        if host is not None
        else "not_measured; select --profile-host-runtime or a platform runner",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("compiled", "compile-inputs", "policy", "out"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repetitions", type=int, default=5)
    parser.add_argument(
        "--io-memory-bytes",
        type=int,
        help="same external-buffer window used for compilation",
    )
    parser.add_argument(
        "--profile-host-runtime",
        action="store_true",
        help="also time the native C runtime through virtual MMIO; not FPGA latency",
    )
    parser.add_argument(
        "--rtl-fixtures",
        type=Path,
        help="also write fresh selected-AtlasCore evaluation fixtures",
    )
    parser.add_argument(
        "--rtl-max-cycles",
        type=int,
        help="explicit hardware watchdog, chosen from calibration",
    )
    args = parser.parse_args()
    if args.out.exists() or args.warmup < 0 or args.repetitions < 2:
        parser.error(
            "fresh output, nonnegative warmup and at least two repetitions required"
        )
    if args.rtl_fixtures and (
        args.rtl_max_cycles is None or not 1 <= args.rtl_max_cycles <= 10000000
    ):
        parser.error("RTL fixtures require --rtl-max-cycles in [1,10000000]")
    from merlin.semantic_compiler.codegen import CodegenOptions, codegen_options

    with codegen_options(CodegenOptions(io_memory_bytes=args.io_memory_bytes)):
        args.out.write_text(json.dumps(measure(args), indent=2, sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
