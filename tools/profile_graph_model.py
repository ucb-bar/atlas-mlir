#!/usr/bin/env python3
"""Execute a captured single-image graph through generated C and npu_model.

This is an evaluation tool. Reference callbacks and runtime samples never enter
ordinary compilation. Host nanoseconds describe this native callback harness;
they are not Shuttle cycles or FPGA launch latency.
"""

import argparse
import ctypes
import dataclasses
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import shutil
import struct
import subprocess
import sys
from time import perf_counter_ns

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "test"))


def reference_outputs(request, arguments):
    """Use the existing public exact arithmetic checker, independently of ISA execution."""
    from atlas_mlir_native.binding import (
        MXU0_POLICY,
        ROUNDED_PACK_POLICY,
        SCALED_PACK_POLICY,
    )
    from atlas_mlir_native import reference
    from merlin.semantic_compiler.reference import TensorValue, evaluate_graph
    from rtl.graph_semantic_checker import evaluate_mxu0_reset, _raw_bf16

    callbacks = {("matmul", MXU0_POLICY): evaluate_mxu0_reset}
    for op in ("add", "sub", "mul"):
        callbacks[(op + "_bf16", "raw-bit-pattern")] = _raw_bf16(
            getattr(reference, f"evaluate_bf16_{op}")
        )
        callbacks[(op + "_bf16_zero", "raw-bit-pattern")] = _raw_bf16(
            reference.evaluate_bf16_binary_zero
        )
    for op in ("exp", "recip", "row_max", "row_sum"):
        callbacks[(op + "_bf16", "raw-bit-pattern")] = _raw_bf16(
            getattr(reference, f"evaluate_bf16_{op}")
        )
    callbacks[("pack_fp8", ROUNDED_PACK_POLICY)] = _raw_bf16(
        reference.evaluate_rounded_scaled_pack
    )
    callbacks[("pack_fp8", SCALED_PACK_POLICY)] = _raw_bf16(
        reference.evaluate_scaled_pack
    )
    constants = {
        row.node_id: TensorValue(
            request.node(row.node_id).type,
            tuple(
                struct.unpack(
                    f"<{len(row.data_hex) // 4}H", bytes.fromhex(row.data_hex)
                )
            ),
        )
        for row in request.constants
    }
    return evaluate_graph(request, arguments, constants=constants, primitives=callbacks)


def measure(args):
    import numpy as np
    import torch
    from merlin.frontends.compile_inputs import (
        load_compile_inputs,
        load_lifted_argument_bytes,
    )
    from merlin.semantic_compiler.model import KernelRequest
    from merlin.semantic_compiler.precision_values import FloatBoundarySpec
    from merlin.semantic_compiler.reference import TensorValue
    from npu_model.configs import isa_definition as isa
    from npu_model.configs.hardware.default import DefaultHardwareConfig
    from npu_model.logging import LoggerConfig
    from npu_model.simulation import Simulation
    from npu_model.software.program import InstantiableProgram
    from tools.atlas_graph_model_bridge import prepare, prepare_c_package
    from tools.check_program_decode import decode_binary
    from tools.model_instructions import model_instruction

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
    binary = (root / "program.bin").read_bytes()
    manifest = json.loads((root / "manifest.json").read_text())
    if (
        manifest["artifact_kind"] != "program"
        or manifest["binary_sha256"] != hashlib.sha256(binary).hexdigest()
    ):
        raise ValueError("single-image profiling requires the exact final program")
    request = KernelRequest.from_record(
        json.loads((root / "source-request.json").read_text())
    )
    if request.digest() != manifest["request_digest"]:
        raise ValueError("request differs from compiled program")
    execution = json.loads((root / "execution_plan.json").read_text())
    if execution["operation"] != "mxu0_bf16_dag":
        raise ValueError(
            "graph profiler requires an admitted single-image MXU/VPU graph"
        )
    staged = load_compile_inputs(args.compile_inputs)
    if (
        hashlib.sha256(staged.program.encode()).hexdigest()
        != json.loads((root / "precision-plan.json").read_text())["source_sha256"]
    ):
        raise ValueError("capture differs from the compiled source")
    work = args.out.with_suffix(".work")
    work.mkdir(parents=True, exist_ok=False)
    bridge = work / "bridge"
    command = [
        sys.executable,
        "-m",
        "merlin.targetgen.cli",
        "native-bridge-graph",
        "--linalg",
        str(args.compile_inputs / "program.mlir"),
        "--entry",
        staged.entry,
        "--compiled",
        str(root),
        "--device-symbol",
        "profile_graph",
        "--out",
        str(bridge),
    ]
    result = subprocess.run(command, capture_output=True, text=True)
    (work / "bridge.log").write_text(result.stdout + result.stderr)
    if result.returncode:
        raise ValueError("cannot replay the source/host interface; see bridge.log")
    record, generated = prepare(args.compile_inputs / "program.mlir", bridge, root)
    package, payloads = prepare_c_package(
        record, generated, root, "profiled", compile_inputs=args.compile_inputs
    )
    for name, payload in payloads.items():
        path = work / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)
    binding = work / "binding.c"
    binding.write_text("""#define _POSIX_C_SOURCE 200809L
#include "profiled.h"
#include <time.h>
static atlas_host host;
static uint64_t ticks(void *p) { (void)p; struct timespec t;
  clock_gettime(CLOCK_MONOTONIC,&t); return (uint64_t)t.tv_sec*1000000000u+t.tv_nsec; }
int bind(atlas_host_read32 rd,atlas_host_write32 wr,atlas_tile_write put,atlas_tile_read get) {
  if (atlas_host_init(&host,(atlas_host_map){0x20000,0x40000,131072},
      (atlas_host_mmio){rd,wr,0},(atlas_host_memory){1,0,0})) return -1;
  if (atlas_host_set_clock(&host,ticks,0)) return -1;
  return profiled_bind(&host,(atlas_tile_memory_io){put,get,0},8);
}
int call(const float *input,float *first,float *second) {
  const float *inputs[]={input}; float *outputs[]={first,second}; atlas_host_result device;
  return profiled_model_run(inputs,outputs,&device);
}
uint64_t counter(unsigned n) {
  uint64_t values[]={host.timings.load_ticks,host.timings.start_ticks,host.timings.wait_ticks,
                    host.image_loads,host.image_reuses,host.imem_words_written,host.launches};
  return n<7?values[n]:0;
}
""")
    if (
        len(package["model_interface"]["runtime_inputs"]) != 1
        or package["output_count"] != 2
    ):
        raise ValueError(
            "this C profiling harness currently admits one input and two outputs"
        )
    cc = shutil.which(os.environ.get("CC", "cc"))
    if cc is None:
        raise ValueError("a native C compiler is required")
    result = subprocess.run(
        [
            cc,
            "-std=c11",
            "-O2",
            "-Wall",
            "-Wextra",
            "-Werror",
            "-shared",
            "-fPIC",
            "-DATLAS_DIAGNOSTIC_GRAPH_PRECISION=1",
            str(work / "profiled.c"),
            str(binding),
            "-o",
            str(work / "graph.so"),
        ],
        capture_output=True,
        text=True,
    )
    (work / "C-build.log").write_text(result.stdout + result.stderr)
    if result.returncode:
        raise ValueError("generated C/H build failed; see C-build.log")
    library = ctypes.CDLL(str(work / "graph.so"))
    RD = ctypes.CFUNCTYPE(ctypes.c_uint32, ctypes.c_void_p, ctypes.c_size_t)
    WR = ctypes.CFUNCTYPE(None, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_uint32)
    MEM = ctypes.CFUNCTYPE(
        ctypes.c_int, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_void_p, ctypes.c_size_t
    )
    library.bind.argtypes = [RD, WR, MEM, MEM]
    library.call.argtypes = [ctypes.c_void_p] * 3
    library.counter.argtypes = [ctypes.c_uint]
    library.counter.restype = ctypes.c_uint64
    inventory = json.loads((ROOT / "docs/selected-variant-inventory.json").read_text())
    instructions = [
        model_instruction(row, inventory, isa)
        for row in decode_binary(inventory, binary)["instructions"]
    ]
    base = min(execution["input_addresses"] + execution["output_addresses"]) - 64
    end = max(execution["input_addresses"] + execution["output_addresses"]) + 2112
    memory = bytearray((i * 17 + 31) % 256 for i in range(end - base))
    control = {
        "running": False,
        "polls": 0,
        "starts": 0,
        "simulation_ns": 0,
        "cycles": 0,
    }
    words, errors = {}, []

    def read32(_, address):
        if address == 0x40004:
            return control["starts"]
        if address != 0x40008:
            errors.append("unknown CSR read")
            return 0
        if not control["running"]:
            return 5
        control["polls"] += 1
        if control["polls"] == 1:
            return 0
        control["running"] = False
        return 5

    def write32(_, address, word):
        if 0x20000 <= address < 0x40000:
            words[(address - 0x20000) // 4] = word
            return
        started = perf_counter_ns()
        try:
            if address != 0x40018 or word != 1 or control["running"]:
                raise ValueError("invalid or overlapping launch")
            loaded = struct.pack(
                f"<{len(binary) // 4}I",
                *(words.get(i, -1) for i in range(len(binary) // 4)),
            )
            if loaded != binary:
                raise ValueError("runtime launched different IMEM bytes")
            control.update(running=True, polls=0, starts=control["starts"] + 1)
            program = InstantiableProgram(instructions)
            program.dram_base = base
            initial = bytes(memory)
            program.memory_regions = [
                (0, torch.tensor(list(memory), dtype=torch.uint8))
            ]
            config = DefaultHardwareConfig()
            config.arch_state_config = dataclasses.replace(
                config.arch_state_config,
                numerics="rtl",
                randomize_init=True,
                init_seed=317 + control["starts"],
            )
            simulation = Simulation(
                config,
                LoggerConfig(filename=str(work / f"model-{control['starts']}.log")),
                program,
                verbose=False,
            )
            state = simulation.core.arch_state
            try:
                state.vmem.fill_(0xCD)
                simulation.run(max_cycles=1000000)
                stats = simulation.get_stats()
                if not state.halted or stats.runtime_errors:
                    raise ValueError("device program did not halt correctly")
                observed = bytes(state.read_dram(base, len(memory)).tolist())
                untouched = bytearray(observed)
                for destination in execution["output_addresses"]:
                    start = destination - base
                    untouched[start : start + 2048] = initial[start : start + 2048]
                if untouched != initial:
                    raise ValueError("input or guard memory changed")
                memory[:] = observed
                control["cycles"] = stats.cycles
            finally:
                state.close()
        except Exception as error:
            errors.append(str(error))
        finally:
            control["simulation_ns"] += perf_counter_ns() - started

    def put(_, address, pointer, size):
        if control["running"] or not base <= address <= end - size:
            return 1
        memory[address - base : address - base + size] = ctypes.string_at(pointer, size)
        return 0

    def get(_, address, pointer, size):
        if control["running"] or not base <= address <= end - size:
            return 1
        ctypes.memmove(
            pointer, bytes(memory[address - base : address - base + size]), size
        )
        return 0

    callbacks = RD(read32), WR(write32), MEM(put), MEM(get)
    if library.bind(*callbacks):
        raise ValueError("generated binding failed")
    lifted = load_lifted_argument_bytes(args.compile_inputs, staged)
    precision = json.loads((root / "precision-plan.json").read_text())
    reference_cache = {}
    samples = []
    for iteration in range(args.warmup + args.repetitions):
        variant = iteration % 2
        tokens = (
            (np.arange(1024, dtype=np.float32).reshape(32, 32) * (variant + 1) % 29)
            - 14
        ) / 16
        if variant not in reference_cache:
            arguments = {}
            for index, (node, _) in enumerate(request.input_storages):
                raw = (
                    np.frombuffer(lifted[index], dtype="<f4").reshape(32, 32)
                    if index in lifted
                    else tokens
                )
                fields = precision["boundary_conversions"][index]["conversion_spec"]
                spec = FloatBoundarySpec(
                    **{
                        **fields,
                        "admitted_code_ranges": tuple(
                            tuple(p) for p in fields["admitted_code_ranges"]
                        ),
                    }
                )
                arguments[node] = TensorValue(
                    request.node(node).type,
                    tuple(int(v) for v in spec.encode(raw).flat),
                )
            reference_cache[variant] = reference_outputs(request, arguments)
        outputs = [np.full(1026, -991, dtype=np.float32) for _ in range(2)]
        before = [library.counter(i) for i in range(7)]
        input_before = tokens.tobytes()
        old_simulation = control["simulation_ns"]
        old_starts = control["starts"]
        begin = perf_counter_ns()
        status = library.call(
            tokens.ctypes.data, outputs[0].ctypes.data + 4, outputs[1].ctypes.data + 4
        )
        elapsed = perf_counter_ns() - begin
        simulation_ns = control["simulation_ns"] - old_simulation
        after = [library.counter(i) for i in range(7)]
        if status or errors or control["starts"] - old_starts != 1:
            raise ValueError(f"single-launch C execution failed: {status}, {errors}")
        if tokens.tobytes() != input_before:
            raise ValueError("host input was modified")
        for output, expected, address in zip(
            outputs,
            reference_cache[variant],
            execution["output_addresses"],
            strict=True,
        ):
            if output[0] != -991 or output[-1] != -991:
                raise ValueError("host output guard changed")
            expected_f32 = (np.array(expected.elements, dtype=np.uint32) << 16).view(
                np.float32
            )
            if not np.array_equal(
                output[1:-1].view(np.uint32), expected_f32.view(np.uint32)
            ):
                raise ValueError(
                    "generated C output differs from independent semantic reference"
                )
        row = {
            "device_cycles": control["cycles"],
            "host_ns": elapsed - simulation_ns,
            "load_ns": after[0] - before[0],
            "start_ns": after[1] - before[1] - simulation_ns,
            "wait_ns": after[2] - before[2],
            "image_loads": after[3] - before[3],
            "image_reuses": after[4] - before[4],
        }
        if any(row[k] < 0 for k in ("host_ns", "start_ns")):
            raise ValueError("host clock and simulator intervals disagree")
        row["warmup"] = iteration < args.warmup
        samples.append(row)
    if library.counter(3) != 1 or library.counter(6) != args.warmup + args.repetitions:
        raise ValueError(
            "expected one initial image load and one launch per invocation"
        )
    physical = json.loads((root / "physical-program.json").read_text())
    measured = [row for row in samples if not row["warmup"]]
    return {
        "program_sha256": hashlib.sha256(binary).hexdigest(),
        "correctness": "passed",
        "device_cycles": [r["device_cycles"] for r in measured],
        "warm_host_ticks": [r["host_ns"] for r in measured],
        "host_clock_unit": "nanoseconds",
        "host_platform": platform.machine(),
        "samples": samples,
        "execution_tier": "npu_model_with_native_C_dispatch",
        "model_revision": revision,
        "host_timing": "native callback harness; synchronous simulator interval excluded; not Shuttle or FPGA latency",
        "checks": {
            "input_sets": 2,
            "outputs": 2,
            "inputs_and_guards": "unchanged",
            "launches_per_invocation": 1,
        },
        "spills": physical.get("spill_stores"),
        "reloads": physical.get("spill_reloads"),
        "reduction_tiles": 1,
        "effective_K_unroll": 1,
        "K_unroll_applicability": "one hardware tile per contraction; no K loop",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--compiled", type=Path, required=True)
    parser.add_argument("--compile-inputs", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repetitions", type=int, default=5)
    args = parser.parse_args()
    if args.out.exists() or args.warmup < 0 or args.repetitions < 2:
        parser.error(
            "fresh output, nonnegative warmup and at least two repetitions required"
        )
    report = measure(args)
    args.out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
