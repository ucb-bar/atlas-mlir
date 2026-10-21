#!/usr/bin/env python3
"""Compile a staged graph or tensor request and export one device image as C/H."""

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("snapshot", "out"):
        parser.add_argument("--" + key, type=Path, required=True)
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--compile-inputs", type=Path, help="Merlin's staged capture directory")
    inputs.add_argument("--request", type=Path, help="typed native kernel request")
    parser.add_argument("--name", required=True, help="C identifier for the exported arrays")
    parser.add_argument("--target-source", type=Path, default=Path(__file__).resolve().parents[3])
    parser.add_argument("--policy", type=Path)
    parser.add_argument("--io-memory-bytes", type=int, default=1048576)
    parser.add_argument("--reduction-unroll", choices=("1", "2", "4", "8"), default="4")
    parser.add_argument("--pipeline", choices=("mandatory", "optimized"), default="mandatory")
    args = parser.parse_args()
    if args.out.exists() or args.out.is_symlink():
        parser.error("output directory must be fresh")
    source = args.target_source.resolve()
    inputs = args.compile_inputs.resolve() if args.compile_inputs else None
    snapshot = args.snapshot.resolve()
    policy = (args.policy or source / "test/examples/attention_mlp_graph_scaled_arithmetic.json").resolve()
    if (
        (inputs is not None and not (inputs / "program.mlir").is_file())
        or (args.request is not None and not args.request.is_file())
        or not (snapshot / "manifest.json").is_file()
    ):
        parser.error("select source inputs and a generated native snapshot")
    args.out.mkdir(parents=True)
    output = args.out.resolve()
    (output / "logs").mkdir()
    environment = {**os.environ, "ATLAS_NATIVE_PIPELINE": args.pipeline}
    commands = []

    def run(stage, arguments):
        command = [sys.executable, *map(str, arguments)]
        start = time.monotonic()
        with (
            (output / "logs" / (stage + ".stdout")).open("w") as stdout,
            (output / "logs" / (stage + ".stderr")).open("w") as stderr,
        ):
            result = subprocess.run(command, stdout=stdout, stderr=stderr, env=environment, check=False)
        commands.append(
            {"stage": stage, "argv": command, "returncode": result.returncode, "seconds": time.monotonic() - start}
        )
        (output / "commands.json").write_text(json.dumps(commands, indent=2) + "\n")
        if result.returncode:
            print(f"{stage} failed; see {output / 'logs' / (stage + '.stderr')}", file=sys.stderr)
            raise SystemExit(result.returncode)

    source_flags = (
        ["--linalg", inputs / "program.mlir", "--linalg-entry", "forward", "--diagnostic-graph-policy", policy]
        if inputs is not None
        else ["--request", args.request.resolve()]
    )
    common = [
        "--engine",
        "merlin_native",
        "--support",
        "atlas",
        "--snapshot",
        snapshot,
        *source_flags,
        "--mode",
        "diagnostic",
        "--io-memory-bytes",
        args.io_memory_bytes,
    ]
    run(
        "compile",
        [
            "-m",
            "merlin.targetgen.cli",
            "native-compile",
            *common,
            "--abi",
            "auto",
            "--target-source",
            source,
            "--search-limits",
            source / "test/examples/attention_mlp_graph_search_limits.json",
            "--reduction-unroll",
            args.reduction_unroll,
            "--out",
            output / "compiler",
            "--status-file",
            output / "compile-status.json",
        ],
    )
    status = json.loads((output / "compile-status.json").read_text())
    if status.get("status") != "emitted" or status.get("artifact_kind") != "program":
        raise SystemExit("this example requires one device image; no C package was exported")
    run(
        "verify",
        [
            "-m",
            "merlin.targetgen.cli",
            "native-verify",
            *common,
            "--artifact",
            output / "compiler",
            "--out",
            output / "verification.json",
        ],
    )
    # Use the exporter beside this example; --target-source can select a separately
    # built source snapshot while preserving its target inventory and Python ABI.
    if inputs is not None:
        exporter = Path(__file__).resolve().parents[3] / "tools/export_resident_c.py"
        run(
            "export",
            [
                exporter,
                "--compiled",
                output / "compiler",
                "--compile-inputs",
                inputs,
                "--policy",
                policy,
                "--name",
                args.name,
                "--target-source",
                source,
                "--io-memory-bytes",
                args.io_memory_bytes,
                "--out",
                output / "c",
            ],
        )
    else:
        from atlas_mlir_native.looped_c import prepare_c_package

        from merlin.semantic_compiler.codegen import CodegenOptions, codegen_options

        with codegen_options(CodegenOptions(io_memory_bytes=args.io_memory_bytes)):
            record, files = prepare_c_package(
                output / "compiler/source-request.json",
                output / "compiler",
                args.name,
                target_source=source,
                runtime_linkage="separate",
            )
        (output / "c").mkdir()
        for filename, payload in files.items():
            member = output / "c" / filename
            member.parent.mkdir(parents=True, exist_ok=True)
            member.write_bytes(payload)
        (output / "c/manifest.json").write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps({"status": "exported", "c_package": str(output / "c"), "execution": "not_run"}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
