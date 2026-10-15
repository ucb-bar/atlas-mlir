#!/usr/bin/env python3
"""Serialize fixed-image Atlas tiles for a bounded MxK by KxN contraction."""

import argparse
from datetime import datetime, timezone
import io
import json
import os
from pathlib import Path
import platform
import resource
import struct
import sys
import time

from atlas_tile_arithmetic import cache_statistics, mxu0_dot, output_halves, rescale_bf16
from run_model_tile import (RECIPES, assembly, check_bindings, configure_model,
                            encode_operands, execute_negative_controls, execute_tile,
                            input_sources, load_bundle, metrics, model_binding,
                            safe_path, sha256, validate_arrays, write_json)


def code_sources():
    return {name: sha256(Path(__file__).with_name(name)) for name in
            ("run_model_contraction.py", "run_model_tile.py", "atlas_tile_arithmetic.py")}


def run(args, out, receipt):
    import numpy as np
    model_root = safe_path(args.model_root)
    model, torch, assemble, pin = configure_model(model_root)
    model_before, input_before = model_binding(model_root), input_sources(args)
    receipt["model_binding_before"] = model_before
    arrays, manifest = load_bundle(args)
    if args.smoke:
        a, b = arrays["A"], arrays["B"]
        a = np.concatenate((a, a[::-1] * np.float32(-0.5)))
        b = np.concatenate((b, b[:, ::-1] * np.float32(0.5)), axis=1)
        arrays = {"A": a, "B": b, "native_C": (a.astype(np.float64) @ b).astype(np.float32)}
        manifest["site"] = {"M": 64, "K": 64, "N": 64}
    a, b, native = validate_arrays(arrays, manifest.get("source_dtype"), matrix=True)
    np.savez(out / "source.npz", **arrays)
    write_json(out / "source-manifest.json", manifest)
    encoded, converted, conversions = encode_operands(arrays, args.recipe)
    rescale_exponent = sum(item["scale_exponent"] for item in conversions.values())
    np.savez(out / "encoded.npz", **encoded)
    k_tiles = a.shape[1] // 32
    input_size, guard_size = 2 * k_tiles * 1024, 256
    output_offset = input_size + guard_size
    source = assembly(k_tiles, output_offset, rescale_exponent if args.recipe == "tensor-pow2" else None)
    program = assemble(io.StringIO(source))
    words = program.assemble()
    if len(words) > 32768 or words[-1] != 0x73:
        raise ValueError("fixed image does not fit IMEM or lacks ECALL")
    (out / "program.S").write_text(source)
    (out / "program.bin").write_bytes(b"".join(struct.pack("<I", word) for word in words))
    (out / "program.hex").write_text("".join(f"{word:08x}\n" for word in words))
    requested = {"load_bytes": input_size, "store_bytes": 2048}
    receipt.update({"recipe": RECIPES[args.recipe], "shape": [a.shape[0], a.shape[1], b.shape[1]],
                    "source_inputs": input_before, "conversion": conversions,
                    "output_rescale_exponent": rescale_exponent,
                    "output_rescale_execution": "VLI_ALL_and_VMUL_BF16" if args.recipe == "tensor-pow2" else "none",
                    "native_fidelity_claim": False, "raw_image_execution_claim": False,
                    "model_input": "new_assembler_decoded_instruction_objects",
                    "invocations": "serialized_fresh_core_instances_one_python_process",
                    "timing_boundary": "sum_of_cold_model_fetch_to_ECALL_invocations_excluding_host_staging",
                    "instruction_words": len(words), "fixed_image_sha256": sha256(out / "program.bin"),
                    "memory": {"dram_base": 1 << 32, "input_bytes_per_tile": input_size,
                               "output_offset": output_offset, "guard_bytes_each": guard_size,
                               "dram_working_set_bytes": input_size + 2048 + 2 * guard_size,
                               "transfer_address_alignment_bytes": 256, "vmem_staging_bytes": 4096},
                    "model": {"revision": pin, "module_file": str(safe_path(model.__file__)),
                              "torch_version": torch.__version__, "numerics": "rtl",
                              "backend": "curve", "curve": "ee290sim_vcs_probe.json",
                              "curve_sha256": model_before["curve_sha256"]}, "tiles": []})
    expected_global = np.zeros(native.shape, dtype=np.uint16)
    raw_accumulators = np.zeros(native.shape, dtype=np.uint16)
    actual_global = np.zeros(native.shape, dtype=np.uint16)
    for row in range(0, a.shape[0], 32):
        for column in range(0, b.shape[1], 32):
            tile_index = len(receipt["tiles"])
            destination = out / f"tile-{tile_index:04d}"
            destination.mkdir()
            tile_a = encoded["A"][row:row + 32]
            tile_b = encoded["B"][:, column:column + 32]
            raw_expected = [[mxu0_dot(tile_a[r], tile_b[:, c]) for c in range(32)] for r in range(32)]
            expected = ([[rescale_bf16(bits, rescale_exponent) for bits in values] for values in raw_expected]
                        if args.recipe == "tensor-pow2" else raw_expected)
            expected_bytes = output_halves(expected)
            (destination / "expected-bf16-halves.bin").write_bytes(expected_bytes)
            payload = b"".join(tile_a[:, k * 32:(k + 1) * 32].tobytes() for k in range(k_tiles))
            payload += b"".join(tile_b[k * 32:(k + 1) * 32, :].tobytes() for k in range(k_tiles))
            initial = payload + b"\xA5" * guard_size + b"\xCD" * 2048 + b"\x5A" * guard_size
            (destination / "initial-dram.bin").write_bytes(initial)
            result = execute_tile(program, initial, expected_bytes, output_offset, requested,
                                  destination, 0, args.max_cycles, 42 + tile_index)
            actual_bytes = (destination / "actual-bf16-halves-0.bin").read_bytes()
            halves = np.frombuffer(actual_bytes, dtype="<u2").reshape(2, 32, 16)
            actual_global[row:row + 32, column:column + 32] = np.concatenate((halves[0], halves[1]), axis=1)
            expected_global[row:row + 32, column:column + 32] = expected
            raw_accumulators[row:row + 32, column:column + 32] = raw_expected
            tile_receipt = {"row_range": [row, row + 32], "column_range": [column, column + 32],
                            "k_range": [0, a.shape[1]], "execution": result,
                            "initial_dram_sha256": sha256(destination / "initial-dram.bin"),
                            "expected_sha256": sha256(destination / "expected-bf16-halves.bin"),
                            "actual_sha256": sha256(destination / "actual-bf16-halves-0.bin"),
                            "cycles_trace_sha256": sha256(destination / "cycles-0.jsonl"),
                            "perfetto_sha256": sha256(destination / "perfetto-0.json")}
            write_json(destination / "receipt.json", tile_receipt)
            receipt["tiles"].append(tile_receipt)
            if tile_index == 0 and (args.negative_controls or args.smoke):
                receipt["executed_negative_controls"] = execute_negative_controls(
                    source, initial, expected_bytes, output_offset, requested,
                    out, args.max_cycles, assemble)
    if not np.array_equal(actual_global, expected_global):
        raise AssertionError("assembled global output disagrees with independent full contraction")
    target = (actual_global.astype(np.uint32) << 16).view(np.float32)
    np.savez(out / "result.npz", target_C=target, actual_bits=actual_global,
             expected_bits=expected_global, raw_accumulator_bits=raw_accumulators)
    receipt["diagnostic_errors"] = {"target_vs_native_C": metrics(target, native),
                                    "bf16_operands_float64_matmul_vs_native_C": metrics(
                                        converted["A"].astype(np.float64) @ converted["B"].astype(np.float64), native)}
    receipt["total_model_cycles"] = sum(tile["execution"]["cycles"] for tile in receipt["tiles"])
    receipt["observed_dma"] = {key: sum(tile["execution"]["observed"].get(key, 0) for tile in receipt["tiles"])
                               for key in ("load_bytes", "store_bytes", "response_beats")}
    receipt["oracle_cache"] = cache_statistics()
    check_bindings(args, model_root, receipt, model_before, input_before)
    receipt["status"] = "passed_model_only"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-root", required=True)
    parser.add_argument("--input")
    parser.add_argument("--manifest")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--negative-controls", action="store_true")
    parser.add_argument("--recipe", choices=tuple(RECIPES), default="unit")
    parser.add_argument("--out", required=True)
    parser.add_argument("--max-cycles", type=int, default=500000)
    args = parser.parse_args()
    out = safe_path(args.out)
    if out.exists():
        parser.error("output destination must be new")
    out.mkdir(parents=True)
    start = time.monotonic()
    receipt = {"status": "failed", "claim": "Atlas model only; no hardware or native-quality verdict",
               "command": sys.argv, "cwd": os.getcwd(), "python": sys.executable,
               "pid": os.getpid(), "started_utc": datetime.now(timezone.utc).isoformat(),
               "cpu_affinity": sorted(os.sched_getaffinity(0)),
               "python_version": platform.python_version(), "code_before": code_sources(),
               "environment": {key: os.environ.get(key) for key in
                               ("PYTHONPATH", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")}}
    try:
        run(args, out, receipt)
        receipt["code_after"] = code_sources()
        if receipt["code_before"] != receipt["code_after"]:
            raise AssertionError("runner/oracle changed during execution")
    except Exception as error:
        receipt["status"] = "failed"
        receipt["error"] = {"type": type(error).__name__, "message": str(error)}
        raise
    finally:
        receipt["wall_seconds"] = time.monotonic() - start
        receipt["max_rss_kib"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        receipt["artifacts"] = {path.name: sha256(path) for path in out.iterdir()
                                if not any(s in path.name.lower() for s in ("hammer", "vlsi"))
                                and path.is_file() and path.name != "receipt.json"}
        write_json(out / "receipt.json", receipt)
    print(json.dumps({"status": receipt["status"], "shape": receipt["shape"],
                      "tiles": len(receipt["tiles"]), "model_cycles": receipt["total_model_cycles"]}))


if __name__ == "__main__":
    main()
