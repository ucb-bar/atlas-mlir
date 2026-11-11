#!/usr/bin/env python3
"""Execute shared source-to-graph lowering in both schedules without agents.

PACK witnesses explicitly stay in the declared seven-code domain. Passing them
does not qualify arbitrary BF16 truncation or turn source premises into proofs.
"""

import argparse
from itertools import product
import json
from pathlib import Path
import random
import subprocess

from atlas_mlir_native.layout import pack_bf16_pair_halves, unpack_bf16_pair_halves
from atlas_mlir_native.shared_dense import MODEL_REVISION
from atlas_mlir_native.shared_graph import compile_graph, pack_graph_inputs, plan_graph
from merlin.frontends.linalg_text import parse_mlir_text
from tools.check_shared_dense import execute_bytes, witness


def physical_tiles(logical, side):
    return [pack_bf16_pair_halves(b"".join(
        logical[((mr+r)*side+nc)*2:((mr+r)*side+nc+32)*2] for r in range(32)))
        for mr in range(0, side, 32) for nc in range(0, side, 32)]


def logical_tiles(tiles, side):
    logical = bytearray(side*side*2)
    for index, physical in enumerate(tiles):
        tile = unpack_bf16_pair_halves(physical)
        mr, nc = divmod(index, side//32)
        for r in range(32):
            start = ((mr*32+r)*side+nc*32)*2
            logical[start:start+64] = tile[r*64:(r+1)*64]
    return bytes(logical)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--atlas-bin-dir", required=True, type=Path)
    parser.add_argument("--model-root", required=True, type=Path)
    parser.add_argument("--model-python", required=True, type=Path)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    revision = subprocess.check_output(["git", "-C", str(args.model_root), "rev-parse", "HEAD"], text=True).strip()
    if revision != MODEL_REVISION:
        parser.error(f"expected model {MODEL_REVISION}, got {revision}")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    rows = []
    for name, side in product(("native_bf16_relu_linalg", "native_mxu0_relu_linalg",
                               "native_mxu0_relu_pack_mxu0_linalg"), (32, 64)):
        source = (root / "test/examples" / (name + ".mlir")).read_text()
        is_relu = name == "native_bf16_relu_linalg"
        chain = "pack" in name
        if is_relu:
            # Keep the original input live and observable across ReLU.
            source = source.replace(') -> tensor<32x32xbf16> {',
                                    ') -> (tensor<32x32xbf16>, tensor<32x32xbf16>) {', 1).replace(
                'func.return %result : tensor<32x32xbf16>',
                'func.return %input, %result : tensor<32x32xbf16>, tensor<32x32xbf16>')
        source = source.replace("32x32", f"{side}x{side}")
        entry = parse_mlir_text(source).body.block.first_op.sym_name.data
        request, _ = plan_graph(source, entry=entry)
        for pipeline in ("source-order", "optimized"):
            directory = args.output_dir / (name + f"-{side}-" + pipeline)
            manifest = compile_graph(source, entry=entry, destination=directory,
                                     atlas_bin_dir=args.atlas_bin_dir, pipeline=pipeline)
            for phase in (2, 3):
                if is_relu:
                    rng = random.Random(1000+phase)
                    codes = [0, 0x8000, 0x0080, 0x8080, 0x7f7f, 0xff7f]
                    while len(codes) < side * side:
                        code = rng.randrange(65536)
                        if 0 < (code >> 7) & 255 < 255:
                            codes.append(code)
                    logical = b"".join(code.to_bytes(2, "little") for code in codes)
                    relu = b"".join((0 if code & 0x8000 else code).to_bytes(2, "little") for code in codes)
                    arguments = {"input0": logical}
                    outputs = [*physical_tiles(logical, side), *physical_tiles(relu, side)]
                else:
                    lhs, rhs, tiles = witness(side, side, side, "permutation" if chain else "finite-range", phase)
                    arguments = {"input0": lhs, "input1": rhs}
                    logical = logical_tiles(tiles, side)
                    h = b"".join((0 if int.from_bytes(logical[i:i+2], "little") & 0x8000 else
                                  int.from_bytes(logical[i:i+2], "little")).to_bytes(2, "little")
                                 for i in range(0, len(logical), 2))
                    outputs = physical_tiles(h, side)
                    if chain:
                        # Independently establish the cast premise for every intermediate lane.
                        assert all(int.from_bytes(h[i:i+2], "little") in (0, 0x3f00, 0x3f80, 0x4000)
                                   for i in range(0, len(h), 2))
                        arguments["input2"] = bytes(0x38 if r == (5*c+3) % side else 0
                                                    for c in range(side) for r in range(side))
                        y = b"".join(h[(r*side+(5*c+3)%side)*2:(r*side+(5*c+3)%side)*2+2]
                                     for r in range(side) for c in range(side))
                        outputs.extend(physical_tiles(y, side))
                inputs = pack_graph_inputs(request, manifest["abi"], arguments)
                result = execute_bytes(root, directory, f"witness-{phase}", inputs, outputs,
                                       model_python=args.model_python, model_root=args.model_root)
                row = dict(source=name, side=side, pipeline=pipeline, phase=phase, **result,
                           timing_checked=manifest["shared_issue_timing"]["instructions"],
                           binary_sha256=manifest["binary_sha256"])
                rows.append(row)
                (args.output_dir / "results.json").write_text(json.dumps(rows, indent=2) + "\n")
                print(json.dumps(row), flush=True)


if __name__ == "__main__":
    main()
