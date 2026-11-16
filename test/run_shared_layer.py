"""Test-only simulator transport for a compiled layer; never a compiler oracle.

Each final binary is independently decoded and every output tile is compared
with Torch FP8 conversion and an ordered BF16 recurrence on the actual runtime
operands. Host partitions are separately checked against Torch using recorded
device outputs at contraction boundaries. A free-running whole-graph reference
is reported separately because small host f32 reduction differences can cross
a quantization midpoint and amplify through subsequent contractions. No device contraction falls back to host math.
"""

import argparse
import ctypes
import json
import multiprocessing
import os
import subprocess
import traceback
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
from functools import lru_cache
from hashlib import sha256
from pathlib import Path
from types import SimpleNamespace

import npu_model
import numpy as np
import torch
from atlas_mlir_native.layout import unpack_bf16_pair_halves
from atlas_mlir_native.shared_dense_io import pack_inputs
from merlin.llvmlower.abi import HostModel, make_descriptor
from merlin.runtime.fp8_formats import float_to_codes
from npu_model.configs.hardware.default import DefaultHardwareConfig
from npu_model.logging import LoggerConfig
from npu_model.simulation import Simulation
from npu_model.software.program import InstantiableProgram
from torch.utils._python_dispatch import TorchDispatchMode

from test.models.tinyllama_layer import Layer
from test.run_shared_model_case import instructions
from tools.check_program_decode import audit_program
from tools.check_variant_inventory import load_inventory


def model_arrays(manifest, capture):
    """Validate the complete static ABI before exposing buffers to native code."""
    arrays = np.load(capture / "inputs.npz")
    inputs = [
        np.array(arrays[f"arg{i}"], copy=True, order="C")
        for i in range(len(manifest["signature"]["inputs"]))
    ]
    numpy_types = {
        "f32": np.float32,
        "f64": np.float64,
        "i1": np.bool_,
        "i8": np.int8,
        "i16": np.int16,
        "i32": np.int32,
        "i64": np.int64,
    }
    signature = manifest["signature"]
    input_dtypes = signature.get("inputs_dtypes", ["f32"] * len(inputs))
    for value, shape, dtype in zip(
        inputs, signature["inputs"], input_dtypes, strict=True
    ):
        if value.shape != tuple(shape) or value.dtype != np.dtype(numpy_types[dtype]):
            raise ValueError("runtime input shape/dtype differs from declared ABI")
    output_dtypes = signature.get("outputs_dtypes", ["f32"] * len(signature["outputs"]))
    outputs = [
        np.full(shape, np.nan if dtype in ("f32", "f64") else 0, numpy_types[dtype])
        for shape, dtype in zip(signature["outputs"], output_dtypes, strict=True)
    ]
    return inputs, outputs


def reference(a, b):
    """Independent conversion implementation and left-to-right scalar K order."""

    def quantize(v):
        v = v.clamp(-448, 448).to(torch.float8_e4m3fn).float()
        return torch.where(v.abs() < 2**-6, 0.0, v)

    a, b = quantize(a), quantize(b)
    result = torch.zeros((*a.shape[:-1], b.shape[-1]), dtype=torch.float32)
    for k in range(a.shape[-1]):
        result = (
            (
                result.double()
                + a[..., :, k, None].double() * b[..., None, k, :].double()
            )
            .float()
            .bfloat16()
            .float()
        )
    return result


class ReferenceContractions(TorchDispatchMode):
    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        if func in (torch.ops.aten.mm.default, torch.ops.aten.bmm.default):
            return reference(*args)
        return func(*args, **(kwargs or {}))


class ReplayContractions(TorchDispatchMode):
    """Check every host partition without quantization-discontinuity feedback.

    This never supplies simulator expected bytes: device tiles were already
    independently checked on the actual operands. Replay uses their recorded
    outputs to compare each surrounding host region against ordinary Torch.
    """

    def __init__(self, boundaries):
        super().__init__()
        self.boundaries = boundaries
        self.index = 0
        self.errors = []

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        if func in (torch.ops.aten.mm.default, torch.ops.aten.bmm.default):
            a, b, result = self.boundaries[self.index]
            errors = []
            for observed, actual in zip(args, (a, b), strict=True):
                observed = observed.numpy()
                np.testing.assert_allclose(observed, actual, rtol=1e-5, atol=1e-5)
                left = np.frombuffer(encoded(observed), np.uint8)
                right = np.frombuffer(encoded(actual), np.uint8)
                errors.append(
                    {
                        "max_abs": float(np.max(np.abs(observed - actual))),
                        "different_fp8_codes": int(np.count_nonzero(left != right)),
                    }
                )
            self.errors.append(errors)
            self.index += 1
            return torch.from_numpy(result.copy())
        return func(*args, **(kwargs or {}))


def view(pointer, shape):
    desc = ctypes.cast(pointer, ctypes.POINTER(make_descriptor(len(shape)))).contents
    if (
        tuple(desc.sizes) != tuple(shape)
        or desc.offset < 0
        or not desc.aligned
        or any(s <= 0 for s in desc.strides)
    ):
        raise ValueError("unexpected runtime descriptor")
    count = 1 + sum((size - 1) * stride for size, stride in zip(shape, desc.strides))
    backing = (ctypes.c_float * count).from_address(desc.aligned + 4 * desc.offset)
    return np.ndarray(
        shape,
        dtype=np.float32,
        buffer=backing,
        strides=tuple(4 * s for s in desc.strides),
    )


def encoded(value):
    if not np.isfinite(value).all():
        raise ValueError("contraction input must be finite")
    codes = float_to_codes(value, "f8E4M3FN").astype(np.uint8)
    codes[(codes & 0x78) == 0] = 0
    return codes.tobytes()


@lru_cache
def decoded(directory, key):
    return instructions((directory / key / "program.bin").read_bytes())


def execute_tile(task):
    directory, destination, key, spec, a, b = task
    torch.set_num_threads(1)
    selected = SimpleNamespace(**spec["selected"])
    offset = spec["output_base"] - 0x90000000
    initial = bytearray([0xCD]) * (offset + 2048 + 64)
    for address, data in pack_inputs(selected, encoded(a), encoded(b)):
        initial[address : address + len(data)] = data
    program = InstantiableProgram(decoded(directory, key))
    program.dram_base = 0x90000000
    program.memory_regions = [
        (0, torch.from_numpy(np.frombuffer(initial, dtype=np.uint8).copy()))
    ]
    config = DefaultHardwareConfig()
    config.arch_state_config = replace(config.arch_state_config, numerics="rtl")
    sim = Simulation(
        config,
        LoggerConfig(filename=str(destination / f"worker-{os.getpid()}.trace.json")),
        program,
        verbose=False,
    )
    try:
        sim.run(max_cycles=1000000)
        stats = sim.get_stats()
        assert sim.core.is_finished() and not stats.runtime_errors, (
            "device failed to finish correctly"
        )
        actual = bytes(
            sim.core.arch_state.read_dram(program.dram_base, len(initial)).tolist()
        )
        assert (
            actual[:offset] == initial[:offset]
            and actual[offset + 2048 :] == initial[offset + 2048 :]
        ), "input/guard clobber"
        logical = unpack_bf16_pair_halves(actual[offset : offset + 2048])
        values = (
            (np.frombuffer(logical, dtype="<u2").astype(np.uint32) << 16)
            .view(np.float32)
            .reshape(32, 32)
        )
        expected = reference(torch.from_numpy(a), torch.from_numpy(b)).numpy()
        np.testing.assert_array_equal(values.view(np.uint32), expected.view(np.uint32))
        return values, stats.cycles
    finally:
        sim.close()


def execute(directory, capture, destination, workers=1):
    if type(workers) is not int or not 1 <= workers <= 32:
        raise ValueError("workers must be between one and 32")
    destination.mkdir(parents=True, exist_ok=False)
    manifest = json.loads((directory / "manifest.json").read_text())
    model_root = Path(npu_model.__file__).resolve().parent.parent
    revision = subprocess.check_output(
        ["git", "-C", str(model_root), "rev-parse", "HEAD"], text=True
    ).strip()
    assert revision == manifest["model_revision"], "different selected model revision"
    assert not subprocess.check_output(
        ["git", "-C", str(model_root), "status", "--porcelain", "--untracked-files=no"],
        text=True,
    ), "dirty model source"
    for name, digest in manifest["artifacts"].items():
        assert sha256((directory / name).read_bytes()).hexdigest() == digest, name
    torch.set_num_threads(1)
    kernels = {}
    for key in manifest["kernels"]:
        for name, digest in manifest["kernels"][key]["artifacts"].items():
            assert (
                sha256((directory / key / name).read_bytes()).hexdigest() == digest
            ), (key, name)
        binary = (directory / key / "program.bin").read_bytes()
        assert (
            audit_program(
                load_inventory(),
                json.loads((directory / key / "program.json").read_text()),
                binary,
            )["status"]
            == "PASS"
        )
        kernels[key] = instructions(binary)
    records = []
    boundaries = []
    errors = []

    pool = (
        ProcessPoolExecutor(
            max_workers=workers, mp_context=multiprocessing.get_context("spawn")
        )
        if workers > 1
        else None
    )

    def dispatch(index, ap, bp, cp):
        try:
            call = manifest["calls"][index]
            a, b = [view(p, s) for p, s in zip((ap, bp), call["input_shapes"])]
            c = view(cp, call["output_shape"])
            batch = a.shape[0] if a.ndim == 3 else 1
            av, bv, cv = (v.reshape((batch, *v.shape[-2:])) for v in (a, b, c))
            key = call["kernel"]
            cycles, tiles = 0, 0
            tasks, locations = [], []
            for z in range(batch):
                for m in range(0, a.shape[-2], 32):
                    for n in range(0, b.shape[-1], 32):
                        aa = np.zeros((32, int(key)), np.float32)
                        bb = np.zeros((int(key), 32), np.float32)
                        rows, cols = min(32, a.shape[-2] - m), min(32, b.shape[-1] - n)
                        aa[:rows, : a.shape[-1]] = av[z, m : m + rows, :]
                        bb[: b.shape[-2], :cols] = bv[z, :, n : n + cols]
                        tasks.append(
                            (
                                directory,
                                destination,
                                key,
                                manifest["kernels"][key],
                                aa,
                                bb,
                            )
                        )
                        locations.append((z, m, n, rows, cols))
            results = (
                pool.map(execute_tile, tasks) if pool else map(execute_tile, tasks)
            )
            for (z, m, n, rows, cols), (result, elapsed) in zip(locations, results):
                cv[z, m : m + rows, n : n + cols] = result[:rows, :cols]
                cycles += elapsed
                tiles += 1
            record = {
                "call": index,
                "tiles": tiles,
                "cycles": cycles,
                "shape": call["output_shape"],
            }
            boundaries.append((a.copy(), b.copy(), c.copy()))
            np.savez(destination / f"call-{index}.npz", lhs=a, rhs=b, output=c)
            records.append(record)
            print(json.dumps(record), flush=True)
            return 0
        except BaseException:  # noqa: BLE001 - exceptions cannot cross a ctypes callback
            errors.append(traceback.format_exc())
            print(errors[-1], flush=True)
            return 1

    inputs, outputs = model_arrays(manifest, capture)
    host = HostModel.load(str(directory / "model.so"), name=manifest["entry"])
    callback_type = ctypes.CFUNCTYPE(
        ctypes.c_int, ctypes.c_int64, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p
    )
    callback = callback_type(dispatch)
    host.lib.merlin_set_dispatch.argtypes = [callback_type]
    host.lib.merlin_set_dispatch(callback)
    try:
        host([(v.ctypes.data, v.shape) for v in (*inputs, *outputs)])
    finally:
        if pool:
            pool.shutdown()
    assert not errors and [r["call"] for r in records] == list(
        range(len(manifest["calls"]))
    )
    np.savez(
        destination / "outputs.npz", **{f"output{i}": v for i, v in enumerate(outputs)}
    )
    evidence = {
        "device_tiles_bit_exact": True,
        "calls": records,
        "cycles": sum(r["cycles"] for r in records),
        "pipeline": manifest["pipeline"],
        "model_revision": revision,
        "torch_version": torch.__version__,
        "workers": workers,
        "scope": "model simulator device cycles only; host and transfer costs excluded",
    }
    (destination / "execution.json").write_text(json.dumps(evidence, indent=2) + "\n")
    return manifest, inputs, outputs, boundaries, records, revision


def run(directory, capture, destination, workers=1):
    manifest, inputs, outputs, boundaries, records, revision = execute(
        directory, capture, destination, workers
    )
    heads = manifest["calls"][3]["input_shapes"][0][0]
    dim = inputs[0].shape[-1] // heads
    kv_heads = inputs[4].shape[0] // dim
    np.savez(
        destination / "outputs.npz", **{f"output{i}": v for i, v in enumerate(outputs)}
    )
    replay = ReplayContractions(boundaries)
    with replay, torch.no_grad():
        expected = Layer(heads, kv_heads)(*[torch.from_numpy(v) for v in inputs])
    assert replay.index == len(boundaries)
    replay_diffs = []
    for got, want in zip(outputs, expected, strict=True):
        want = want.numpy()
        np.testing.assert_allclose(got, want, rtol=1e-5, atol=1e-5)
        replay_diffs.append(float(np.max(np.abs(got - want))))
    with ReferenceContractions(), torch.no_grad():
        independent = Layer(heads, kv_heads)(*[torch.from_numpy(v) for v in inputs])
    free_diffs = [
        float(np.max(np.abs(got - want.numpy())))
        for got, want in zip(outputs, independent, strict=True)
    ]
    report = {
        "partition_checks_passed": True,
        "device_tiles_bit_exact": True,
        "host_tolerance": {"rtol": 1e-5, "atol": 1e-5},
        "host_boundary_errors": replay.errors,
        "calls": records,
        "host_replay_output_max_abs_errors": replay_diffs,
        "free_reference_output_max_abs_errors": free_diffs,
        "cycles": sum(r["cycles"] for r in records),
        "pipeline": manifest["pipeline"],
        "model_revision": revision,
        "torch_version": torch.__version__,
        "workers": workers,
        "scope": "model simulator device cycles only; host execution and transfer costs excluded",
    }
    (destination / "result.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("directory", type=Path)
    parser.add_argument("capture", type=Path)
    parser.add_argument("destination", type=Path)
    args = parser.parse_args()
    print(
        json.dumps(
            run(
                args.directory.resolve(),
                args.capture.resolve(),
                args.destination.resolve(),
                args.workers,
            )
        )
    )
