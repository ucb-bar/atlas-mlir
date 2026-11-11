#!/usr/bin/env python3
"""Small shared-pass regression and Atlas reference comparison; no agent loop.

Use the Merlin Python environment for this driver and a separate --model-python
for the selected model. Each program is compiled once and reused with different
runtime bytes. Results describe delegated lowering, never generated-compiler
performance. Independent sparse and rounding witnesses avoid a model oracle.
"""

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import struct

from atlas_mlir_native.shared_dense import INPUT_BASE, OUTPUT_BASE, MODEL_REVISION, compile_dense, pack_inputs, plan


def source_for(root, m, k, n):
    from merlin.frontends.linalg_mlir import parse_mlir_text
    from merlin.xdsl_dialects._common import text
    from xdsl.dialects.builtin import TensorType

    module = parse_mlir_text((root / "test/examples/native_mxu0_ordered_linalg.mlir").read_text())
    function = next(iter(module.body.block.ops))
    shapes = ((m, k), (n, k))
    # Test-fixture construction only. Production lowering reads exact input IR.
    for argument, shape in zip(function.body.block.args, shapes):
        argument._type = TensorType(argument.type.element_type, shape)
    for op in function.body.block.ops:
        for result in op.results:
            if isinstance(result.type, TensorType):
                result._type = TensorType(result.type.element_type, (m, n))
    function.update_function_type()
    module.verify()
    return text(module)


def witness(m, k, n, kind, phase):
    codes = (0, 0x30, 0x38, 0x40, 0xb0, 0xb8, 0xc0)
    bits = {0: 0, 0x30: 0x3f00, 0x38: 0x3f80, 0x40: 0x4000,
            0xb0: 0xbf00, 0xb8: 0xbf80, 0xc0: 0xc000}
    if kind == "permutation":
        lhs = bytes(codes[(i*3+j*5+phase) % len(codes)] for i in range(m) for j in range(k))
        rhs = bytes(0x38 if r == (c*3+5+phase) % k else 0 for c in range(n) for r in range(k))
        expected = [bits[lhs[i*k+(j*3+5+phase) % k]] for i in range(m) for j in range(n)]
    elif kind == "dense":
        lhs, rhs = bytes([0x38])*(m*k), bytes([0x38])*(n*k)
        expected = [{32: 0x4200, 64: 0x4280}[k]]*(m*n)
    elif kind == "finite-range":
        # Independent integer oracle: every normal E4M3 value is an integer
        # multiple of 2^-9, so products and sums can be exact integers in
        # units of 2^-18. Round to eight significant bits after each FMA.
        # This never calls the compiler's or model's numerical helpers.
        legal = (0, *range(8, 127), *range(136, 255))
        lhs = bytes(legal[(i*19+r*13+phase) % len(legal)] for i in range(m) for r in range(k))
        rhs = bytes(legal[(j*23+r*17+7+phase*3) % len(legal)] for j in range(n) for r in range(k))
        def units(code):
            return 0 if code == 0 else (-1 if code & 128 else 1) * ((8+(code & 7)) << (((code >> 3) & 15)-1))
        expected = []
        for i in range(m):
            for j in range(n):
                acc = 0
                for r in range(k):
                    total = acc + units(lhs[i*k+r])*units(rhs[j*k+r])
                    magnitude, shift = abs(total), max(0, abs(total).bit_length()-8)
                    q, rem = divmod(magnitude, 1 << shift)
                    q += 2*rem > (1 << shift) or (2*rem == (1 << shift) and q % 2)
                    acc = (-1 if total < 0 else 1) * (q << shift)
                if not acc:
                    expected.append(0)
                else:
                    length = abs(acc).bit_length()
                    significand = abs(acc) >> (length-8) if length >= 8 else abs(acc) << (8-length)
                    expected.append((0x8000 if acc < 0 else 0) | ((length-1-18+127) << 7) | (significand-128))
    elif kind == "tile-boundary":
        # Round each 1/256 contribution independently after a tile ending in
        # 1.0; accumulating the next tile separately would instead yield 1+1/128.
        lhs = bytes(0x38 if r == 31 else 0x18 if r in (32, 33) else 0 for _ in range(m) for r in range(k))
        rhs = bytes(0x38 if r == 31 else 0x18 if r in (32, 33) else 0 for _ in range(n) for r in range(k))
        expected = [0x3f80]*(m*n)
    else:
        # 1 +/- 1/256 +/- 1/256, in order. Positive terms each tie to
        # even 1.0. Negative terms are representable below 1.0.
        lhs = bytes(0x38 if r == 0 else (0x18 if kind == "round-up" else 0x98) if r < 3 else 0
                    for _ in range(m) for r in range(k))
        rhs = bytes(0x38 if r == 0 else 0x18 if r < 3 else 0 for _ in range(n) for r in range(k))
        expected = [0x3f80 if kind == "round-up" else 0x3f7e]*(m*n)
    tiles = []
    for mr in range(0, m, 32):
        for nc in range(0, n, 32):
            tiles.append(b"".join(expected[(mr+r)*n+nc+c].to_bytes(2, "little")
                                  for half in range(2) for r in range(32)
                                  for c in range(half*16, half*16+16)))
    return lhs, rhs, tiles


def run_case(root, directory, selected, *, shape, kind, phase, model_python, model_root, relu=False, chain=False):
    lhs, rhs, tiles = witness(*shape, kind, phase)
    if relu:
        tiles = [b"".join((0 if int.from_bytes(tile[i:i+2], "little") & 0x8000 else
                           int.from_bytes(tile[i:i+2], "little")).to_bytes(2, "little")
                          for i in range(0, len(tile), 2)) for tile in tiles]
    inputs = pack_inputs(selected, lhs, rhs)
    if chain:
        assert relu and kind == "permutation" and shape == (32, 32, 32)
        w1 = bytes(0x38 if r == (5*c+3) % 32 else 0 for c in range(32) for r in range(32))
        inputs = (*inputs, (4096, w1))
        h = tiles[0]
        offsets = [(((5*c+3) % 32)//16)*1024+r*32+(((5*c+3) % 32) % 16)*2
                   for half in range(2) for r in range(32) for c in range(half*16, half*16+16)]
        tiles.append(b"".join(h[offset:offset+2] for offset in offsets))
    return execute_bytes(root, directory, f"{kind}-{phase}", inputs, tiles,
                         model_python=model_python, model_root=model_root)


def execute_bytes(root, directory, name, inputs, tiles, *, model_python, model_root):
    """Run test-only runtime bytes after compilation; compare exact outputs and guards."""
    outputs = [(OUTPUT_BASE-INPUT_BASE+i*2048, data) for i, data in enumerate(tiles)]
    guards = [(offset+len(data), b"G"*(2048-len(data))) for offset, data in inputs if len(data) < 2048]
    guards += [(OUTPUT_BASE-INPUT_BASE-64, b"L"*64), (OUTPUT_BASE-INPUT_BASE+len(tiles)*2048, b"R"*64)]
    initial = [*inputs, *guards, *((offset, b"?"*len(data)) for offset, data in outputs)]
    encode = lambda rows: [(offset, data.hex()) for offset, data in rows]
    case = directory / f"{name}.json"
    case.write_text(json.dumps(dict(dram_base=INPUT_BASE, initial=encode(initial), expected=encode(outputs),
                                    preserved=encode([*inputs, *guards]))) + "\n")
    env = {**os.environ, "PYTHONPATH": os.pathsep.join((str(model_root), str(root)))}
    run = subprocess.run([str(model_python), str(root / "test/run_shared_model_case.py"), str(directory), str(case)],
                         env=env, text=True, capture_output=True)
    if run.returncode:
        raise RuntimeError(f"{directory.name}/{case.name}:\n{run.stdout}\n{run.stderr}")
    return json.loads(case.with_suffix(".result.json").read_text())


def compile_reference(directory, bins, pipeline, *, relu=False, chain_source=None):
    """Independent, direct Atlas tile program used only as a test comparator."""
    directory.mkdir()
    source = f'''module {{
      func.func @reference() -> !atlas.virtual_state attributes {{
        atlas.input_dram_base = {INPUT_BASE} : i64, atlas.output_dram_base = {OUTPUT_BASE} : i64}} {{
        %s0 = "atlas.virtual_start"() : () -> !atlas.virtual_state
        %s1, %a = "atlas.virtual_input_fp8"(%s0) {{index = 0 : i32}} : (!atlas.virtual_state) -> (!atlas.virtual_state, !atlas.virtual_fp8)
        %s2, %w = "atlas.virtual_input_fp8"(%s1) {{index = 1 : i32}} : (!atlas.virtual_state) -> (!atlas.virtual_state, !atlas.virtual_fp8)
        %y = "atlas.virtual_mxu_matmul"(%a, %w) {{unit = 0 : i32}} : (!atlas.virtual_fp8, !atlas.virtual_fp8) -> !atlas.virtual_bf16
        %s3 = "atlas.virtual_output_bf16"(%s2, %y) {{index = 0 : i32}} : (!atlas.virtual_state, !atlas.virtual_bf16) -> !atlas.virtual_state
        return %s3 : !atlas.virtual_state
      }}
    }}'''
    if relu:
        source = source.replace('        %s3 =', '        %z = "atlas.virtual_vpu_unary"(%y) {kind = "relu"} : (!atlas.virtual_bf16) -> !atlas.virtual_bf16\n        %s3 =')
        source = source.replace('(%s2, %y)', '(%s2, %z)')
    if chain_source is not None:
        source = chain_source
    (directory / "virtual.mlir").write_text(source)
    subprocess.run([str(bins / "atlas-opt"), str(directory / "virtual.mlir"),
                    "--atlas-compile-unscheduled" if pipeline == "source-order" else "--atlas-compile",
                    "--verify-atlas-generated-schedule", "--verify-atlas-machine-stream",
                    "-o", str(directory / "machine.mlir")], check=True, capture_output=True)
    words = subprocess.run([str(bins / "atlas-emit"), str(directory / "machine.mlir")],
                           check=True, capture_output=True, text=True).stdout
    program = subprocess.run([str(bins / "atlas-emit"), "--program-json", str(directory / "machine.mlir")],
                             check=True, capture_output=True, text=True).stdout
    (directory / "program.bin").write_bytes(b"".join(struct.pack("<I", int(word, 16)) for word in words.split()))
    (directory / "program.json").write_text(program)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--atlas-bin-dir", type=Path, required=True)
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--model-python", type=Path, required=True)
    parser.add_argument("--reference-only", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    revision = subprocess.run(["git", "-C", str(args.model_root), "rev-parse", "HEAD"],
                              check=True, text=True, capture_output=True).stdout.strip()
    if revision != MODEL_REVISION:
        parser.error(f"this baseline requires npu_model {MODEL_REVISION}; got {revision}")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    (args.output_dir / "baseline.json").write_text(json.dumps({
        "model_revision": revision, "model_root": str(args.model_root.resolve()),
        "atlas_bin_dir": str(args.atlas_bin_dir.resolve()), "arguments": sys.argv[1:],
        "scope": "model execution of shared logical scheduling with delegated Atlas lowering",
    }, indent=2) + "\n")
    rows = []
    for shape in (() if args.reference_only else ((32, 32, 32), (32, 64, 32), (64, 32, 64), (64, 64, 32))):
        source = source_for(root, *shape)
        selected, _ = plan(source, entry="mxu0_tile")
        for pipeline in ("source-order", "optimized"):
            directory = args.output_dir / ("x".join(map(str, shape)) + "-" + pipeline)
            manifest = compile_dense(source, entry="mxu0_tile", destination=directory,
                                     atlas_bin_dir=args.atlas_bin_dir, pipeline=pipeline)
            for kind, phase in (("permutation", 0), ("permutation", 1), ("dense", 0),
                                ("round-up", 0), ("round-down", 0), ("finite-range", 0), ("tile-boundary", 0)):
                result = run_case(root, directory, selected, shape=shape, kind=kind, phase=phase,
                                  model_python=args.model_python, model_root=args.model_root)
                row = dict(shape=shape, pipeline=pipeline, witness=kind, phase=phase,
                           binary_sha256=manifest["binary_sha256"], **result)
                rows.append(row)
                print(json.dumps(row), flush=True)
                (args.output_dir / "results.json").write_text(json.dumps(rows, indent=2) + "\n")
    selected, _ = plan(source_for(root, 32, 32, 32), entry="mxu0_tile")
    for pipeline in ("source-order", "optimized"):
        directory = args.output_dir / ("reference-" + pipeline)
        compile_reference(directory, args.atlas_bin_dir, pipeline)
        for kind, phase in (("permutation", 0), ("permutation", 1), ("dense", 0),
                            ("round-up", 0), ("round-down", 0), ("finite-range", 0), ("tile-boundary", 0)):
            result = run_case(root, directory, selected, shape=(32, 32, 32), kind=kind, phase=phase,
                              model_python=args.model_python, model_root=args.model_root)
            rows.append(dict(shape=(32, 32, 32), backend="direct Atlas reference", pipeline=pipeline,
                             witness=kind, phase=phase, **result))
            print(json.dumps(rows[-1]), flush=True)
            (args.output_dir / "results.json").write_text(json.dumps(rows, indent=2) + "\n")
        directory = args.output_dir / ("reference-relu-" + pipeline)
        compile_reference(directory, args.atlas_bin_dir, pipeline, relu=True)
        for phase in (0, 1):
            result = run_case(root, directory, selected, shape=(32, 32, 32), kind="permutation", phase=phase,
                              model_python=args.model_python, model_root=args.model_root, relu=True)
            rows.append(dict(shape=(32, 32, 32), backend="direct Atlas matmul + ReLU reference", pipeline=pipeline,
                             witness="permutation", phase=phase, **result))
            print(json.dumps(rows[-1]), flush=True)
            (args.output_dir / "results.json").write_text(json.dumps(rows, indent=2) + "\n")
        # Reuse the README's existing virtual MLP fixture; select MXU0 at both
        # stages and retain the ReLU result as an additional observable output.
        source = (root / "test/examples/virtual_fp8_two_layer_mlp.mlir").read_text()
        source = source.replace("2415927296", str(OUTPUT_BASE)).replace("unit = 1 : i32", "unit = 0 : i32")
        source = source.replace('    %io4 = "atlas.virtual_output_bf16"(%io3, %y)',
                                '    %saved = "atlas.virtual_output_bf16"(%io3, %h1) {index = 0 : i32} : (!atlas.virtual_state, !atlas.virtual_bf16) -> !atlas.virtual_state\n'
                                '    %io4 = "atlas.virtual_output_bf16"(%saved, %y)')
        source = source.replace('(%saved, %y) {index = 0', '(%saved, %y) {index = 1')
        directory = args.output_dir / ("reference-pack-chain-" + pipeline)
        compile_reference(directory, args.atlas_bin_dir, pipeline, chain_source=source)
        for phase in (0, 1):
            result = run_case(root, directory, selected, shape=(32, 32, 32), kind="permutation", phase=phase,
                              model_python=args.model_python, model_root=args.model_root, relu=True, chain=True)
            rows.append(dict(shape=(32, 32, 32), backend="direct Atlas ReLU/PACK/second matmul reference",
                             pipeline=pipeline, witness="permutation", phase=phase, **result))
            print(json.dumps(rows[-1]), flush=True)
            (args.output_dir / "results.json").write_text(json.dumps(rows, indent=2) + "\n")


if __name__ == "__main__":
    main()
