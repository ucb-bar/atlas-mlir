"""Model-independent diagnostic validation of compiled host/device partitioning.

The supplied capture includes source-consistent synthetic inputs and independent
Torch f32 outputs. Check its original compiled host graph against those outputs,
then execute every outlined contraction on Atlas and compare the resulting graph
with independent ordered arithmetic in the same host image. The latter isolates
transport/tiling errors; it is not an independent whole-graph accuracy proof.
"""

import argparse
import ctypes
import json
import traceback
from pathlib import Path

import numpy as np
import torch
from merlin.llvmlower.abi import HostModel

from test.run_shared_layer import execute, model_arrays, reference, view


def run(directory, capture, destination, f32_host, workers=1):
    manifest = json.loads((directory / "manifest.json").read_text())
    inputs, baseline = model_arrays(manifest, capture)
    if any(v.dtype != np.float32 for v in baseline):
        raise ValueError("diagnostic output comparison requires f32 results")
    golden = np.load(capture / "torch-f32.npz")
    host = HostModel.load(str(f32_host / "model.so"), name=manifest["entry"])
    host([(v.ctypes.data, v.shape) for v in (*inputs, *baseline)])
    host_errors = []
    for i, got in enumerate(baseline):
        want = golden[f"output{i}"]
        np.testing.assert_allclose(got, want, rtol=2e-4, atol=2e-4)
        host_errors.append(float(np.max(np.abs(got - want))))
    manifest, inputs, outputs, _boundaries, _records, _revision = execute(
        directory, capture, destination, workers
    )
    software = [np.empty_like(v) for v in outputs]
    host = HostModel.load(str(directory / "model.so"), name=manifest["entry"])
    visited = []

    errors = []

    def dispatch(index, ap, bp, cp):
        try:
            call = manifest["calls"][index]
            a, b = [
                view(p, s) for p, s in zip((ap, bp), call["input_shapes"], strict=True)
            ]
            c = view(cp, call["output_shape"])
            c[:] = reference(
                torch.from_numpy(a.copy()), torch.from_numpy(b.copy())
            ).numpy()
            visited.append(index)
            return 0
        except BaseException:  # noqa: BLE001 - exceptions cannot cross a ctypes callback
            errors.append(traceback.format_exc())
            print(errors[-1], flush=True)
            return 1

    callback_type = ctypes.CFUNCTYPE(
        ctypes.c_int, ctypes.c_int64, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p
    )
    callback = callback_type(dispatch)
    host.lib.merlin_set_dispatch.argtypes = [callback_type]
    host.lib.merlin_set_dispatch(callback)
    host([(v.ctypes.data, v.shape) for v in (*inputs, *software)])
    assert not errors and visited == list(range(len(manifest["calls"])))
    for actual, expected in zip(outputs, software, strict=True):
        np.testing.assert_array_equal(actual.view(np.uint32), expected.view(np.uint32))
    report = json.loads((destination / "execution.json").read_text())
    report.update(
        original_f32_host_matches_torch=True,
        original_f32_host_max_abs_errors=host_errors,
        full_outputs_match_ordered_software_callbacks=True,
        reference_scope="original f32 graph checked against Torch; approximate device graph checked against same host with independent arithmetic",
        approximation_max_abs_errors=[
            float(np.max(np.abs(v - golden[f"output{i}"])))
            for i, v in enumerate(outputs)
        ],
    )
    (destination / "result.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("capture", type=Path)
    parser.add_argument("destination", type=Path)
    parser.add_argument("--f32-host", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()
    run(
        args.directory.resolve(),
        args.capture.resolve(),
        args.destination.resolve(),
        args.f32_host.resolve(),
        args.workers,
    )
