#!/usr/bin/env python3
"""Write an FP8 matrix product request, Y[M,N] = A[M,K] @ W[N,K].T."""

import argparse
import json
from pathlib import Path


def main():
    from atlas_mlir_native.binding import MXU0_INPUT_DOMAIN, MXU0_POLICY, TARGET_IDENTITY, _mxu0_maps

    from merlin.semantic_compiler.model import KernelRequest, SemanticNode, TensorType

    parser = argparse.ArgumentParser(description=__doc__)
    for dimension in ("m", "k", "n"):
        parser.add_argument("--" + dimension, type=int, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if min(args.m, args.k, args.n) < 1:
        parser.error("matrix dimensions must be positive")
    nodes = tuple(
        SemanticNode(
            key,
            "input",
            (),
            TensorType(shape, "fp8_e4m3", "raw-bit-pattern"),
            (("input_domain", MXU0_INPUT_DOMAIN),),
            "input",
        )
        for key, shape in (("a", (args.m, args.k)), ("w", (args.n, args.k)))
    )
    result = SemanticNode(
        "y", "matmul", ("a", "w"), TensorType((args.m, args.n), "bf16", MXU0_POLICY), index_maps=_mxu0_maps()
    )
    request = KernelRequest(
        (*nodes, result),
        ("y",),
        ("external",),
        (("a", "external"), ("w", "external")),
        TARGET_IDENTITY,
        lowering_policy="diagnostic",
    )
    with args.out.open("x") as stream:
        stream.write(json.dumps(request.record(), indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
