#!/usr/bin/env python3
"""Execute a complete bounded contraction in one conservative Atlas program."""

import argparse
from datetime import datetime, timezone
import io
import json
import os
from pathlib import Path
import resource
import struct
import sys
import time
import zipfile

from atlas_tile_arithmetic import mxu0_dot, output_halves, rescale_bf16
from run_model_tile import (OutputMismatch, RECIPES, check_bindings, configure_model,
                            encode_operands, execute_tile, input_sources, load_bundle, metrics,
                            model_binding, safe_path, sha256, validate_archive_members,
                            validate_arrays, write_json)


def assembly(m, k, n, output_offset, rescale_exponent=None, *, resident_b=False,
             pretranspose_resident_b=False):
    if pretranspose_resident_b and not resident_b:
        raise ValueError("--pretranspose-resident-b requires --resident-b")
    # Keep B in bank 1; bank 0 retains the existing A/output staging windows.
    if resident_b and k * n > 256 * 1024:
        raise ValueError("resident B must fit one 256-KiB VMEM bank")
    stride = k * 32
    lines = ["li x1, 0x800", "li x2, " + ("0x10000" if resident_b else "0x900"), "li x3, 0xa00", "li x5, 1024",
             "li x8, 2048", "li x9, 0xb00", "li x10, 0", f"li x12, {output_offset}",
             f"li x13, {m // 32}", f"li x15, {m * k}", f"li x16, {stride}",
             f"li x17, {n // 32}", "li x30, 1", "dma.config x30, ch0"]
    if resident_b:
        # DMA/VLOAD VMEM operands count four-byte words; DRAM operands count bytes.
        lines += ["add x19, x2, x0", "add x20, x15, x0", f"li x21, {k * n // 1024}",
                  f"li x18, {stride // 4}", "resident_b_loop:",
                  "dma.load x19, x20, x5, ch0", "dma.wait ch0"]
        if pretranspose_resident_b:
            # Cache the raw byte transpose in the same bank-1 tile, in-program.
            lines += ["vload m1, 0(x19)", "delay 34", "vtrpose.xlu m1, m1", "delay 66",
                      "vstore m1, 0(x19)", "delay 34"]
        lines += ["addi x19, x19, 256", "addi x20, x20, 1024", "addi x21, x21, -1",
                  "beq x21, x0, resident_b_done", "nop", "jal x0, resident_b_loop", "nop",
                  "resident_b_done:"]
    lines += ["rows_loop:", "add x11, " + ("x2" if resident_b else "x15") + ", x0", "add x14, x17, x0",
             "columns_loop:", "add x4, x10, x0", "add x6, x11, x0"]
    for tile in range(k // 32):
        lines += ["dma.load x1, x4, x5, ch0", "dma.wait ch0"]
        if not resident_b:
            lines += ["dma.load x2, x6, x5, ch0", "dma.wait ch0"]
        lines += ["vload m0, 0(x1)", "delay 34", "vload m1, 0(" + ("x6" if resident_b else "x2") + ")", "delay 34"]
        if not pretranspose_resident_b:
            lines += ["vtrpose.xlu m1, m1", "delay 66"]
        lines += ["vmatpush.weight.mxu0 w0, m1", "delay 32",
                  ("vmatmul.mxu0" if tile == 0 else "vmatmul.acc.mxu0") + " acc0, m0, w0",
                  "delay 96", "addi x4, x4, 1024", "addi x6, x6, " + ("256" if resident_b else "1024")]
    lines += ["vmatpop.bf16.acc.mxu0 m2, acc0", "delay 34"]
    if rescale_exponent is not None:
        if not -126 <= rescale_exponent <= 127:
            raise ValueError("VPU rescale factor must be normal finite BF16")
        lines += [f"vli.all m4, {(rescale_exponent + 127) << 7}", "delay 66",
                  "vmul.bf16 m2, m2, m4", "delay 66"]
    lines += ["vstore m2, 0(x3)", "delay 34", "vstore m3, 0(x9)", "delay 34",
              "dma.store x12, x3, x8, ch0", "dma.wait ch0", "add x12, x12, x8",
              "add x11, x11, " + ("x18" if resident_b else "x16"), "addi x14, x14, -1", "beq x14, x0, columns_done", "nop",
              "jal x0, columns_loop", "nop", "columns_done:", "add x10, x10, x16",
              "addi x13, x13, -1", "beq x13, x0, finished", "nop", "jal x0, rows_loop", "nop",
              "finished:", "nop", "ecall"]
    return "\n".join(lines) + "\n"


def preflight_bundle(args):
    """Admit the NPZ central directory before any input digest or content read."""
    if args.smoke:
        if args.input or args.manifest:
            raise ValueError("--smoke excludes --input/--manifest")
        return
    if not args.input or not args.manifest:
        raise ValueError("require --input and --manifest, or --smoke")
    with zipfile.ZipFile(safe_path(args.input)) as archive:
        validate_archive_members(archive.infolist())


def code_sources():
    return {name: sha256(Path(__file__).with_name(name)) for name in
            ("run_model_program_contraction.py", "run_model_tile.py", "atlas_tile_arithmetic.py")}


def negative_controls(source, initial, expected, offset, requested, out, max_cycles, assemble, tile_count):
    mutations = {"omitted_B_transpose": source.replace("vtrpose.xlu m1, m1", "nop"), "wrong_oracle": source}
    if "vmatmul.acc.mxu0" in source:
        mutations["reset_continuations"] = source.replace("vmatmul.acc.mxu0", "vmatmul.mxu0")
    if tile_count > 1:
        mutations["aliased_output_tiles"] = source.replace("add x12, x12, x8", "add x12, x12, x0")
    controls = {}
    for name, text in mutations.items():
        destination = out / ("control-" + name)
        destination.mkdir()
        changed = bytes([expected[0] ^ 1]) + expected[1:] if name == "wrong_oracle" else expected
        (destination / "program.S").write_text(text)
        (destination / "expected.bin").write_bytes(changed)
        program = assemble(io.StringIO(text))
        (destination / "program.bin").write_bytes(b"".join(struct.pack("<I", word) for word in program.assemble()))
        try:
            execute_tile(program, initial, changed, offset, requested, destination, 0, max_cycles)
        except OutputMismatch as error:
            actual = (destination / "actual-bf16-halves-0.bin").read_bytes()
            if name == "wrong_oracle" and actual != expected:
                raise AssertionError("Wrong-oracle control changed actual execution") from error
            controls[name] = {"status": "expected_output_mismatch", "execution": error.context,
                              "message": str(error), "actual_sha256": sha256(destination / "actual-bf16-halves-0.bin"),
                              "program_sha256": sha256(destination / "program.bin"),
                              "expected_sha256": sha256(destination / "expected.bin")}
            write_json(destination / "receipt.json", controls[name])
        else:
            raise AssertionError("Negative control did not distinguish " + name)
    return controls


def run(args, out, receipt):
    pretranspose_resident_b = getattr(args, "pretranspose_resident_b", False)
    if pretranspose_resident_b and not args.resident_b:
        raise ValueError("--pretranspose-resident-b requires --resident-b")
    import numpy as np
    preflight_bundle(args)
    model_root = safe_path(args.model_root)
    model, torch, assemble, pin = configure_model(model_root)
    model_before, input_before = model_binding(model_root), input_sources(args)
    arrays, manifest = load_bundle(args)
    if args.smoke:
        a, b = arrays["A"], arrays["B"]
        a = np.concatenate((a, a[::-1] * np.float32(-0.5)))
        b = np.concatenate((b, b[:, ::-1] * np.float32(0.5)), axis=1)
        arrays = {"A": a, "B": b, "native_C": (a.astype(np.float64) @ b).astype(np.float32)}
        manifest["site"] = {"M": 64, "K": 64, "N": 64}
    a, b, native = validate_arrays(arrays, manifest.get("source_dtype"), matrix=True)
    m, k, n = a.shape[0], a.shape[1], b.shape[1]
    if args.resident_b and k * n > 256 * 1024:
        raise ValueError("resident B must fit one 256-KiB VMEM bank")
    encoded, converted, conversions = encode_operands(arrays, args.recipe)
    exponent = sum(item["scale_exponent"] for item in conversions.values())
    np.savez(out / "source.npz", **arrays)
    np.savez(out / "encoded.npz", **encoded)
    write_json(out / "source-manifest.json", manifest)
    inputs = b"".join(encoded["A"][row:row + 32, inner:inner + 32].tobytes()
                      for row in range(0, m, 32) for inner in range(0, k, 32))
    inputs += b"".join(encoded["B"][inner:inner + 32, column:column + 32].tobytes()
                       for column in range(0, n, 32) for inner in range(0, k, 32))
    expected, ranges, physical = np.zeros((m, n), dtype=np.uint16), [], []
    for row in range(0, m, 32):
        for column in range(0, n, 32):
            raw = [[mxu0_dot(encoded["A"][r], encoded["B"][:, c])
                    for c in range(column, column + 32)] for r in range(row, row + 32)]
            corrected = ([[rescale_bf16(bits, exponent) for bits in values] for values in raw]
                         if args.recipe == "tensor-pow2" else raw)
            expected[row:row + 32, column:column + 32] = corrected
            ranges.append({"row_range": [row, row + 32], "column_range": [column, column + 32], "k_range": [0, k]})
            physical.append(output_halves(corrected))
    expected_bytes = b"".join(physical)
    (out / "expected-bf16-tiles.bin").write_bytes(expected_bytes)
    offset = len(inputs) + 256
    initial = inputs + b"\xA5" * 256 + b"\xCD" * len(expected_bytes) + b"\x5A" * 256
    source = assembly(m, k, n, offset, exponent if args.recipe == "tensor-pow2" else None,
                      resident_b=args.resident_b, pretranspose_resident_b=pretranspose_resident_b)
    (out / "program.S").write_text(source)
    program = assemble(io.StringIO(source))
    words = program.assemble()
    if len(words) > 32768 or words[-1] != 0x73:
        raise ValueError("program does not fit IMEM or lacks ECALL")
    (out / "program.bin").write_bytes(b"".join(struct.pack("<I", word) for word in words))
    (out / "initial-dram.bin").write_bytes(initial)
    requested = {"load_bytes": len(ranges) * (2 * k // 32 * 1024), "store_bytes": len(expected_bytes)}
    if args.resident_b:
        requested["load_bytes"] = m * k * (n // 32) + k * n
    receipt.update(recipe=RECIPES[args.recipe], shape=[m, k, n], source_inputs=input_before,
                   model_binding_before=model_before, conversion=conversions, tiles=ranges,
                   output_rescale_exponent=exponent, instruction_words=len(words), requested_dma=requested,
                   model={"revision": pin, "module_file": str(safe_path(model.__file__)), "torch_version": torch.__version__,
                          "numerics": "rtl", "backend": "curve", "curve": "ee290sim_vcs_probe.json",
                          "curve_sha256": model_before["curve_sha256"]},
                   memory={"dram_base": 1 << 32, "input_bytes": len(inputs), "output_offset": offset,
                           "output_bytes": len(expected_bytes), "guard_bytes_each": 256, "total_dram_bytes": len(initial)},
                   timing_boundary="one_fetch_to_ECALL_instruction_program_excluding_host_conversion_packing_and_loading",
                   invocations="one_reset_core_per_repeat_all_tiles_in_one_instruction_program",
                   resident_operand_reuse_claim=args.resident_b,
                   operand_reuse={"mode": ("resident_b_pretransposed_v1" if pretranspose_resident_b else
                                           "resident_b_v1" if args.resident_b else "none"),
                                  "resident_b_bytes": k * n if args.resident_b else 0,
                                  "resident_b_vmem_byte_base": 0x40000 if args.resident_b else None},
                   raw_image_execution_claim=False, native_fidelity_claim=False,
                   model_input="new_assembler_decoded_instruction_objects", runs=[])
    max_cycles = args.max_cycles or len(ranges) * (k // 32 * 2500 + 5000)
    for repeat in range(args.repeats):
        receipt["runs"].append(execute_tile(program, initial, expected_bytes, offset, requested, out, repeat, max_cycles, 42 + repeat))
    if args.smoke or args.negative_controls:
        receipt["executed_negative_controls"] = negative_controls(
            source, initial, expected_bytes, offset, requested, out, max_cycles, assemble, len(ranges))
    first = (out / "actual-bf16-halves-0.bin").read_bytes()
    actual = np.zeros_like(expected)
    for index, tile in enumerate(ranges):
        halves = np.frombuffer(first[index * 2048:(index + 1) * 2048], dtype="<u2").reshape(2, 32, 16)
        r0, r1 = tile["row_range"]
        c0, c1 = tile["column_range"]
        actual[r0:r1, c0:c1] = np.concatenate((halves[0], halves[1]), axis=1)
    if not np.array_equal(actual, expected):
        raise AssertionError("Global actual words differ from the independent contraction")
    target = (actual.astype(np.uint32) << 16).view(np.float32)
    np.savez(out / "result.npz", target_C=target, actual_bits=actual, expected_bits=expected)
    receipt["diagnostic_errors"] = {"target_vs_native_C": metrics(target, native)}
    check_bindings(args, model_root, receipt, model_before, input_before)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-root", required=True)
    parser.add_argument("--input")
    parser.add_argument("--manifest")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--negative-controls", action="store_true")
    parser.add_argument("--resident-b", action="store_true",
                        help="load packed B once into one 256-KiB VMEM bank; refuse larger tensors")
    parser.add_argument("--pretranspose-resident-b", action="store_true",
                        help="transpose each resident B tile once in-program; requires --resident-b")
    parser.add_argument("--recipe", choices=tuple(RECIPES), default="unit")
    parser.add_argument("--out", required=True)
    parser.add_argument("--max-cycles", type=int)
    parser.add_argument("--repeats", type=int, choices=(1, 2), default=2)
    args = parser.parse_args()
    if args.pretranspose_resident_b and not args.resident_b:
        parser.error("--pretranspose-resident-b requires --resident-b")
    out = safe_path(args.out)
    out.mkdir(parents=True, exist_ok=False)
    start = time.monotonic()
    before = code_sources()
    source_dir = out / "producer-sources"
    source_dir.mkdir()
    for name, sha in before.items():
        (source_dir / name).write_bytes(Path(__file__).with_name(name).read_bytes())
        if sha256(source_dir / name) != sha:
            raise AssertionError("Producer source snapshot changed")
    receipt = {"status": "failed", "claim": "One complete instruction program in the model; no hardware/native-quality verdict",
               "command": sys.argv, "cwd": os.getcwd(), "python": sys.executable,
               "cpu_affinity": sorted(os.sched_getaffinity(0)), "pid": os.getpid(),
               "started_utc": datetime.now(timezone.utc).isoformat(), "code_before": before,
               "environment": {key: os.environ.get(key) for key in
                               ("PYTHONPATH", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")}}
    try:
        run(args, out, receipt)
        receipt["code_after"] = code_sources()
        if receipt["code_after"] != before:
            raise AssertionError("Producer source changed during execution")
        receipt["status"] = "passed_model_only"
    except Exception as error:
        receipt.update(status="failed", error={"type": type(error).__name__, "message": str(error)})
        raise
    finally:
        receipt.update(wall_seconds=time.monotonic() - start, max_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                       artifacts={path.name: sha256(path) for path in out.iterdir()
                                  if not any(bad in path.name.lower() for bad in ("hammer", "vlsi"))
                                  and path.is_file() and path.name != "receipt.json"})
        write_json(out / "receipt.json", receipt)
    print(json.dumps({"status": receipt["status"], "shape": receipt["shape"], "cycles": [r["cycles"] for r in receipt["runs"]]}))


if __name__ == "__main__":
    main()
