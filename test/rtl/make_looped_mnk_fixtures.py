#!/usr/bin/env python3
"""Compile whole-contraction requests and prepare independent RTL fixtures."""

import argparse
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from test_looped_mnk import ROOT, SHAPES, build_snapshot, case_data, source_request


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        parser.error("--out must be fresh")
    args.out.mkdir(parents=True)
    started = time.perf_counter()
    snapshot = build_snapshot(args.out)
    generation_seconds = time.perf_counter() - started
    records = []
    for shape in SHAPES:
        name = "mnk_" + "_".join(map(str, shape))
        source = args.out / (name + ".request.json")
        source.write_text(json.dumps(source_request(*shape).record(), sort_keys=True, indent=2) + "\n")
        root = args.out / name
        command = [
            sys.executable,
            "-m",
            "merlin.targetgen.cli",
            "native-compile",
            "--engine",
            "merlin_native",
            "--support",
            "atlas",
            "--snapshot",
            str(snapshot.root),
            "--request",
            str(source),
            "--mode",
            "diagnostic",
            "--abi",
            "auto",
            "--target-source",
            str(ROOT),
            "--out",
            str(root),
        ]
        # Compile in a child before constructing runtime inputs or expected outputs.
        started = time.perf_counter()
        subprocess.run(command, check=True, capture_output=True, text=True)
        compile_seconds = time.perf_counter() - started
        _, plan, inputs, expected = case_data(shape)
        execution = json.loads((root / "execution_plan.json").read_text())
        binary = (root / "program.bin").read_bytes()
        (root / "expected.bin").write_bytes(expected)
        properties = [
            "schema=atlas.rtl_tile_fixture.v2",
            f"name={name}",
            f"input_count={sum(len(p) // 1024 for p in inputs)}",
            f"output_count={plan.counts[0] * plan.counts[1]}",
            f"program_sha256={hashlib.sha256(binary).hexdigest()}",
            f"output.address={execution['output_addresses'][0]:#x}",
        ]
        i = 0
        for operand, (base, payload) in enumerate(zip(execution["input_addresses"], inputs, strict=True)):
            for start in range(0, len(payload), 1024):
                (root / f"input-{i}.bin").write_bytes(payload[start : start + 1024])
                properties.append(f"input.{i}.address={base + start:#x}")
                properties.append(f"input.{i}.read_count={plan.counts[1 if operand == 0 else 0]}")
                i += 1
        (root / "manifest.properties").write_text("\n".join(properties) + "\n")
        records.append(
            {
                "name": name,
                "m": shape[0],
                "n": shape[1],
                "k": shape[2],
                "words": len(binary) // 4,
                "compile_seconds": round(compile_seconds, 3),
                "loop_counts": list(plan.counts),
            }
        )
    (args.out / "fixtures.json").write_text(json.dumps(records, indent=2) + "\n")
    (args.out / "generation.json").write_text(
        json.dumps(
            {
                "native_build_seconds": round(generation_seconds, 3),
                "compiler_process_inputs": "typed request, target and snapshot only",
            },
            indent=2,
        )
        + "\n"
    )
    print(json.dumps(records))


if __name__ == "__main__":
    main()
