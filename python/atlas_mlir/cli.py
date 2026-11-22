"""Compile, deploy, and run Atlas applications through model2MLIR and Merlin."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys

from . import deploy, toolchain
from .driver import build, capture
from .provenance import environment_record


def _capture_options(parser):
    parser.add_argument("--dtype", choices=("fp32", "bf16", "fp16", "fp8", "int8"), default="bf16")
    parser.add_argument("--capture-python", help="interpreter containing the pinned capture stack")
    parser.add_argument("--contract", type=Path, help="original atlas-torchao manifest.json")
    parser.add_argument("--quant-policy", type=Path, help="original atlas-torchao policy.json")
    parser.add_argument("--expected-contract-sha256")
    parser.add_argument("--expected-policy-sha256")


def _capture_kwargs(args):
    return {key: getattr(args, key) for key in ("dtype", "capture_python", "contract",
            "quant_policy", "expected_contract_sha256", "expected_policy_sha256", "timeout")}


def _build(args):
    result = build(args.source, out=args.output, precision_policy=args.precision_policy,
        captured=args.capture, snapshot=args.snapshot, target_source=args.target_source,
        already_quantized=args.already_quantized, application_parameters=args.application_parameters,
        application_name=args.application_name, application_memory_bytes=args.application_memory_bytes,
        application_state=args.application_state,
        io_memory_bytes=args.io_memory_bytes, recipe=args.recipe, board=args.board,
        application_residency=args.application_residency, application_region_loop=args.application_region_loop,
        application_deployment=args.application_deployment, host_clock_hz=args.host_clock_hz,
        measured_host_bytes=args.measured_host_bytes,
        force=args.force, **_capture_kwargs(args))
    print(f"Compiled and verified diagnostic application: {result.application}\nReport: {result.report}")
    return 0


def _capture(args):
    result = capture(args.source, out=args.output, **_capture_kwargs(args))
    print(f"Materialized capture: {result}")
    return 0


def _doctor(args):
    record = environment_record(capture=args.capture)
    for name in ("atlas-opt", "atlas-emit"):
        toolchain.run([str(toolchain.bin_dir() / name), "--help"], name)
    for file in ("runtime/atlas_host.c", "runtime/atlas_host.h", "runtime/atlas_host_layout.h",
                 "docs/selected-variant-inventory.json"):
        if not (toolchain.data_root() / file).is_file():
            raise toolchain.ToolchainError(f"missing target support file: {file}")
    print(json.dumps(record, indent=2, sort_keys=True))
    print("Atlas tools, support files, hashes and plugin identities checked.")
    return 0


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    # Historical direct importer is intentionally only an explicitly labeled demo.
    if argv and argv[0] in ("demo", "tile-demo"):
        from .tile_demo_cli import main as tile_main
        return tile_main(["demo" if argv[0] == "demo" else "build", *argv[1:]])
    parser = argparse.ArgumentParser(prog="atlasc", description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    deploy.add_commands(commands)
    cp = commands.add_parser("capture", help="materialize a live loader with Merlin/model2MLIR")
    cp.add_argument("source", type=Path)
    cp.add_argument("-o", "--output", type=Path, required=True)
    _capture_options(cp)
    cp.add_argument("--timeout", type=float, default=1800, help="hang guard: seconds a step may run without writing its log or outputs before it is aborted")
    cp.set_defaults(run=_capture)
    bp = commands.add_parser("build", help="capture, stage, compile and verify a Merlin application")
    source = bp.add_mutually_exclusive_group(required=True)
    source.add_argument("source", type=Path, nargs="?", help="live .py loader defining get_model_and_inputs()")
    source.add_argument("--capture", type=Path, help="materialized Merlin capture directory")
    bp.add_argument("-o", "--output", type=Path, default=Path("atlas-out/application"))
    bp.add_argument("--precision-policy", type=Path, required=True, help="source-bound JSON, or .py defining get_precision_policy(source_sha256)")
    bp.add_argument("--snapshot", type=Path, help="existing pinned native snapshot; otherwise build one (requires Cargo)")
    bp.add_argument("--target-source", type=Path, help="override installed target support root")
    _capture_options(bp)
    bp.add_argument("--already-quantized", action="store_true", help="preserve authored codes for an existing --capture")
    bp.add_argument("--application-parameters", choices=("external", "embedded"), default="external")
    bp.add_argument("--application-name", default="application")
    bp.add_argument("--application-state", type=Path, help="source-bound application state input/update interface")
    bp.add_argument("--application-memory-bytes", type=int, required=True, help="explicit host tensor arena budget")
    bp.add_argument("--io-memory-bytes", type=int)
    bp.add_argument("--recipe", help="recipe owned by the selected Atlas provider")
    bp.add_argument("--board", default="sophialake", help="board owned by the selected Atlas provider")
    bp.add_argument("--application-deployment", choices=("onchip", "board"), help=argparse.SUPPRESS)
    bp.add_argument("--host-clock-hz", type=int, help=argparse.SUPPRESS)
    bp.add_argument("--measured-host-bytes", type=int, help=argparse.SUPPRESS)
    bp.add_argument("--application-residency", action="store_true")
    bp.add_argument("--application-region-loop", action="store_true")
    bp.add_argument("--timeout", type=float, default=1800)
    bp.add_argument("--force", action="store_true", help="replace only a prior atlasc output directory")
    bp.set_defaults(run=_build)
    setup = commands.add_parser("setup-capture", help="fetch pinned atlas-torchao exporter and reference sources")
    def setup_capture(args):
        from .torchao import setup
        print(setup())
        return 0
    setup.set_defaults(run=setup_capture)
    dp = commands.add_parser("doctor", help="check tools, build record, dependency pins and unique plugins")
    dp.add_argument("--capture", action="store_true")
    dp.set_defaults(run=_doctor)
    commands.add_parser("demo", help="historical standalone captured MLP tile demo")
    commands.add_parser("tile-demo", help="historical direct importer; not the Merlin application path")
    args = parser.parse_args(argv)
    try:
        if getattr(args, "timeout", 1) <= 0:
            raise ValueError("--timeout must be positive")
        return args.run(args)
    except (OSError, ValueError, subprocess.SubprocessError, toolchain.ToolchainError, toolchain.ToolFailure) as error:
        print(f"atlasc: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
