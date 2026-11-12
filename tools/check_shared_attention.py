#!/usr/bin/env python3
"""Independent numerical witnesses for shared attention/MLP lowering; no agents."""

import argparse
import json
import struct
import subprocess
from pathlib import Path

import numpy as np
from atlas_mlir_native.exp_lane import vexp_bf16_bits
from atlas_mlir_native.layout import pack_bf16_pair_halves
from atlas_mlir_native.pack_lane import pack_bf16_lane
from atlas_mlir_native.recip_lane import vrecip_bf16_bits
from atlas_mlir_native.shared_dense import MODEL_REVISION
from atlas_mlir_native.shared_tensor_graph import (
    PRECISION,
    compile_tensor_graph,
    pack_tensor_inputs,
)
from merlin.frontends.linalg_tensor_graph import read_tensor_graph

from tools.check_shared_dense import execute_bytes


def bf16(value):
    bits = struct.unpack("<I", struct.pack("<f", float(value)))[0]
    return ((bits + 0x7FFF + ((bits >> 16) & 1)) >> 16) & 65535


def decode(code):
    return struct.unpack("<f", struct.pack("<I", int(code) << 16))[0]


def matrix(a, b):
    # Independent scalar recurrence; no compiler or model arithmetic helpers.
    a = np.vectorize(lambda x: pack_bf16_lane(int(x), 127))(a)
    b = np.vectorize(lambda x: pack_bf16_lane(int(x), 127))(b)

    def fp8(x):
        return (
            0.0
            if x == 0
            else (-1 if x & 128 else 1) * (8 + (x & 7)) * 2.0 ** (((x >> 3) & 15) - 10)
        )

    av = np.vectorize(fp8)(a)
    bv = np.vectorize(fp8)(b)
    out = np.zeros((a.shape[0], b.shape[1]), dtype=np.uint16)
    for i in range(out.shape[0]):
        for j in range(out.shape[1]):
            acc = 0
            for k in range(a.shape[1]):
                acc = bf16(decode(acc) + av[i, k] * bv[k, j])
            out[i, j] = acc
    return out


def binary(a, b, kind):
    x = np.vectorize(decode)(a)
    y = np.vectorize(decode)(b)
    value = x * y if kind == "mul" else x - y if kind == "sub" else x + y
    if kind == "mul":
        return np.vectorize(bf16, otypes=[np.uint16])(value)
    return (np.asarray(value, dtype="<f4").view("<u4") >> 16).astype(np.uint16)


def reference(arguments):
    # The source fixture's high-level expression, independent of the shared
    # reader, its node IDs, emitted virtual IR and physical instruction stream.
    wq, wk, wv, w1, w2, x = [
        np.vectorize(bf16, otypes=[np.uint16])(arguments["input" + str(i)])
        for i in range(6)
    ]
    q, k, v = (matrix(x, w.T) for w in (wq, wk, wv))
    raw_scores = matrix(q, k.T)
    scores = binary(
        raw_scores, np.full(raw_scores.shape, bf16(0.176776692), np.uint16), "mul"
    )
    maximum = np.vectorize(bf16, otypes=[np.uint16])(
        np.max(np.vectorize(decode)(scores), axis=1, keepdims=True)
    )
    shifted = binary(scores, maximum, "sub")
    exponent = np.vectorize(vexp_bf16_bits, otypes=[np.uint16])(shifted.astype(object))
    vals = np.vectorize(decode)(exponent).astype(np.float32)
    while vals.shape[1] > 1:
        vals = (vals[:, ::2] + vals[:, 1::2]).astype(np.float32)
    total = np.vectorize(bf16, otypes=[np.uint16])(vals)
    inverse = np.vectorize(vrecip_bf16_bits, otypes=[np.uint16])(total.astype(object))
    probability = binary(exponent, inverse, "mul")
    attended = matrix(probability, v)
    hidden = matrix(attended, w1.T)
    relu = np.where(hidden & 32768, 0, hidden).astype(np.uint16)
    out = matrix(relu, w2.T)
    return [
        q,
        k,
        v,
        raw_scores,
        scores,
        shifted,
        exponent,
        probability,
        attended,
        hidden,
        relu,
        out,
    ]


def fixture(source, width, retained):
    # Test-only structural variation: longer projection reduction and optional
    # retained stage outputs. Production lowering never sees these selectors.
    from io import StringIO

    from merlin.frontends.linalg_text import parse_mlir_text
    from xdsl.dialects import builtin, func
    from xdsl.dialects.linalg.ops import GenericOp, MatmulOp, TransposeOp
    from xdsl.printer import Printer

    module = parse_mlir_text(source)
    fn = module.body.block.first_op
    for index in (0, 1, 2, 5):
        fn.body.block.args[index]._type = builtin.TensorType(builtin.f32, (32, width))
    for op in list(fn.body.block.ops):
        if isinstance(op, TransposeOp) and op.operands[0] in fn.body.block.args[:3]:
            ty = builtin.TensorType(builtin.f32, (width, 32))
            op.operands[1]._type = ty
            op.results[0]._type = ty
    if retained:
        values = [
            op.results[0]
            for op in fn.body.block.ops
            if isinstance(op, (MatmulOp, GenericOp))
        ]
        ret = fn.body.block.last_op
        fn.body.block.insert_op_before(func.ReturnOp(*values), ret)
        fn.body.block.erase_op(ret)
    fn.update_function_type()
    module.verify()
    buf = StringIO()
    Printer(stream=buf).print_op(module)
    return buf.getvalue()


def quality_metrics(arguments, codes):
    wq, wk, wv, w1, w2, x = [arguments["input" + str(i)] for i in range(6)]
    q, k, v = (x @ w.T for w in (wq, wk, wv))
    scores = (q @ k.T) * np.float32(0.176776692)
    p = np.exp(scores - np.max(scores, axis=1, keepdims=True))
    p = p / np.sum(p, axis=1, keepdims=True)
    expected = np.maximum((p @ v) @ w1.T, 0) @ w2.T
    actual = np.vectorize(decode)(codes)
    error = actual - expected
    return {
        "max_abs": float(np.max(np.abs(error))),
        "rmse": float(np.sqrt(np.mean(error**2))),
        "reference_max_abs": float(np.max(np.abs(expected))),
        "status": "diagnostic f32 comparison; no quality admission threshold",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--atlas-bin-dir", required=True, type=Path)
    parser.add_argument("--model-root", required=True, type=Path)
    parser.add_argument("--model-python", required=True, type=Path)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    assert (
        subprocess.check_output(
            ["git", "-C", str(args.model_root), "rev-parse", "HEAD"], text=True
        ).strip()
        == MODEL_REVISION
    )
    original = (root / "test/examples/captured_attention_mlp32_linalg.mlir").read_text()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    rows = []
    for width in (32, 64):
        for retained in (False, True):
            source = fixture(original, width, retained)
            request = read_tensor_graph(
                source, entry="forward", target_identity="diagnostic"
            )
            for mode in ("source-order", "optimized"):
                directory = args.output_dir / f"w{width}-retained{int(retained)}-{mode}"
                manifest = compile_tensor_graph(
                    source,
                    entry="forward",
                    precision=PRECISION,
                    destination=directory,
                    atlas_bin_dir=args.atlas_bin_dir,
                    pipeline=mode,
                )
                for seed in (7, 19):
                    rng = np.random.default_rng(seed)
                    arguments = {}
                    for i in range(5):
                        w = np.zeros((32, width if i < 3 else 32), np.float32)
                        for c in range(0, w.shape[1], 32):
                            w[np.arange(32), c + rng.permutation(32)] = rng.choice(
                                [0.5, 1.0, -0.5], size=32
                            )
                        arguments["input" + str(i)] = w
                    arguments["input5"] = rng.choice(
                        [-0.5, 0, 0.5, 1.0], size=(32, width)
                    ).astype(np.float32)
                    stages = reference(arguments)
                    expected = stages if retained else stages[-1:]
                    assert (
                        np.count_nonzero(stages[-1]) > 32
                        and len(np.unique(stages[-1])) > 4
                    ), "vacuous witness"
                    quality = quality_metrics(arguments, stages[-1])
                    payload = pack_tensor_inputs(request, manifest["abi"], arguments)
                    outputs = [
                        pack_bf16_pair_halves(x.astype("<u2").tobytes())
                        for x in expected
                    ]
                    result = execute_bytes(
                        root,
                        directory,
                        "seed" + str(seed),
                        payload,
                        outputs,
                        model_python=args.model_python,
                        model_root=args.model_root,
                    )
                    rows.append(
                        dict(
                            width=width,
                            retained=retained,
                            mode=mode,
                            seed=seed,
                            quality=quality,
                            **result,
                        )
                    )
                    print(rows[-1], flush=True)
                    (args.output_dir / "results.json").write_text(
                        json.dumps(rows, indent=2) + "\n"
                    )


if __name__ == "__main__":
    main()
