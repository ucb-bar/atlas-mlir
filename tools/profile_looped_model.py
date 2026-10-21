#!/usr/bin/env python3
"""Check and measure a compact contraction on npu_model; not a compiler input."""

import argparse
import dataclasses
import hashlib
import importlib.metadata
import json
import struct
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def measure(args):
    import torch
    from atlas_mlir_native.binding import check_io_memory_manifest
    from atlas_mlir_native.looped_tiling import prepare_execution_plan
    from npu_model.configs import isa_definition as isa
    from npu_model.configs.hardware.default import DefaultHardwareConfig
    from npu_model.logging import LoggerConfig
    from npu_model.simulation import Simulation
    from npu_model.software.program import InstantiableProgram
    from tools.check_program_decode import decode_binary
    from tools.model_instructions import model_instruction

    from merlin.semantic_compiler.model import KernelRequest
    from merlin.semantic_compiler.ordered_mnk_tiling import materialize_looped_mnk_inputs

    root = args.compiled
    check_io_memory_manifest(json.loads((root / "manifest.json").read_text()))
    request = KernelRequest.from_record(json.loads((root / "source-request.json").read_text()))
    plan, _, streaming = prepare_execution_plan(request)
    if streaming or len(request.nodes) != 3 or request.nodes[-1].op != "matmul":
        raise ValueError("model profiler currently admits one non-streamed, unseeded contraction")
    m, n, k = plan.extents
    execution = json.loads((root / "execution_plan.json").read_text())
    binary = (root / "program.bin").read_bytes()
    inventory = json.loads((ROOT / "docs/selected-variant-inventory.json").read_text())
    decoded = decode_binary(inventory, binary)
    instructions = [model_instruction(row, inventory, isa) for row in decoded["instructions"]]
    origin = json.loads(importlib.metadata.distribution("npu-model").read_text("direct_url.json") or "{}")
    revision = "337e9f249f938a1dff007ddf8a763e5cd4b3aa2b"
    if revision not in origin.get("url", "") and origin.get("vcs_info", {}).get("commit_id") != revision:
        raise ValueError("install the documented model revision before profiling")
    torch.set_num_threads(1)
    samples = []
    for iteration in range(args.warmup + args.repetitions):
        variant = iteration % 2
        # Two sparse, signed witnesses. Nonzero terms in the first and final
        # K tiles exercise reset and continuation; their sums are exactly BF16.
        active = sorted({0, k - 1})
        a = bytes((0x38 if variant == 0 else 0x40) if i in active else 0 for _ in range(m) for i in range(k))
        w = bytes((0xB8 if c % 2 else 0x38) if i in active else 0 for c in range(n) for i in range(k))
        packed = materialize_looped_mnk_inputs(plan, {request.nodes[0].id: a, request.nodes[1].id: w})
        expected = bytearray(plan.buffer_bytes[2])
        for r in range(m):
            for c in range(n):
                value = len(active) * (1 + variant) * (-1 if c % 2 else 1)
                bits = struct.unpack("<I", struct.pack("<f", value))[0] >> 16
                tile = (r // 32) * plan.counts[1] + c // 32
                lane = ((c % 32) // 16) * 512 + (r % 32) * 16 + c % 16
                struct.pack_into("<H", expected, tile * 2048 + lane * 2, bits)
        output = execution["output_addresses"][0]
        base = min([*execution["input_addresses"], output]) - 64
        end = (
            max(
                [
                    output + len(expected),
                    *(addr + len(data) for addr, data in zip(execution["input_addresses"], packed, strict=True)),
                ]
            )
            + 64
        )
        initial = bytearray((i * 13 + 71) % 256 for i in range(end - base))
        for addr, data in zip(execution["input_addresses"], packed, strict=True):
            initial[addr - base : addr - base + len(data)] = data
        program = InstantiableProgram(instructions)
        program.dram_base = base
        program.memory_regions = [(0, torch.tensor(list(initial), dtype=torch.uint8))]
        config = DefaultHardwareConfig()
        config.arch_state_config = dataclasses.replace(
            config.arch_state_config, numerics="rtl", randomize_init=True, init_seed=317 + iteration
        )
        simulation = Simulation(
            config, LoggerConfig(filename=str(args.out.with_suffix(f".{iteration}.log"))), program, verbose=False
        )
        state = simulation.core.arch_state
        try:
            state.vmem.fill_(0xCD)
            simulation.run(max_cycles=500000)
            stats = simulation.get_stats()
            if not state.halted or stats.runtime_errors:
                raise ValueError("model failed to finish the selected program")
            initial[output - base : output - base + len(expected)] = expected
            if bytes(state.read_dram(base, len(initial)).tolist()) != initial:
                raise ValueError("model output, preserved input or surrounding memory differs")
            if iteration >= args.warmup:
                samples.append(stats.cycles)
        finally:
            state.close()
    return {
        "program_sha256": hashlib.sha256(binary).hexdigest(),
        "correctness": "passed",
        "device_cycles": samples,
        "warm_host_ticks": None,
        "launch_overhead_ticks": None,
        "execution_tier": "npu_model",
        "model_revision": revision,
        "checks": {"input_sets": 2, "output": "exact", "inputs_and_guards": "unchanged"},
        "spills": json.loads((root / "physical-program.json").read_text()).get("spill_stores"),
        "reloads": json.loads((root / "physical-program.json").read_text()).get("spill_reloads"),
        "host_timing": "unavailable: model instruction execution excludes CPU dispatch",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--compiled", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repetitions", type=int, default=5)
    parser.add_argument("--io-memory-bytes", type=int, help="same external-buffer window used for compilation")
    args = parser.parse_args()
    if args.warmup < 0 or args.repetitions < 2:
        parser.error("profiling needs nonnegative warmup and at least two checked repetitions")
    if args.out.exists():
        parser.error("fresh measurement output required")
    try:
        from merlin.semantic_compiler.codegen import CodegenOptions, codegen_options

        with codegen_options(CodegenOptions(io_memory_bytes=args.io_memory_bytes)):
            report = measure(args)
    except (ValueError, OSError) as error:
        print(str(error), file=sys.stderr)
        return 1
    args.out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
