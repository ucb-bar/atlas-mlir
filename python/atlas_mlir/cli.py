"""atlasc: compile a PyTorch model for Atlas, or check the installed toolchain."""

from __future__ import annotations

import argparse
import importlib.metadata
import subprocess
import sys
from pathlib import Path

from . import policy, toolchain
from .capture import Capture, CaptureError
from .driver import build


def _demo(args: argparse.Namespace) -> int:
    examples = toolchain.data_root() / "test/examples"
    args.source = None
    args.linalg = examples / "captured_mlp32_linalg.mlir"
    args.arguments = examples / "captured_mlp32_arguments.json"
    args.weights = examples / "captured_mlp32_weights.safetensors"
    args.inputs = None
    args.policy = "fp8"
    args.host_cc = None
    args.no_host_elf = True
    return _build(args)


def _build(args: argparse.Namespace) -> int:
    captured = None
    if args.linalg:
        missing = [flag for flag, value in (("--arguments", args.arguments),
                                            ("--weights", args.weights)) if value is None]
        if missing or args.source:
            sys.exit("atlasc: --linalg needs --arguments and --weights, and no model source")
        captured = Capture(args.linalg, args.arguments, args.weights, args.inputs)
    elif not args.source:
        sys.exit("atlasc: give a model (loader .py or .pt2) or --linalg")
    name = args.source.stem if args.source else args.linalg.stem
    out = args.output or Path("atlas-out") / name
    try:
        result = build(args.source, out=out, captured=captured, example_inputs=args.inputs,
                       policy=args.policy, force=args.force, host_cc=args.host_cc,
                       host_elf=not args.no_host_elf)
    except (CaptureError, OSError, ValueError,
            toolchain.ToolchainError, toolchain.ToolFailure) as error:
        print(f"atlasc: {error}", file=sys.stderr)
        return 1
    print(f"atlasc: compiled {args.source or args.linalg}")
    print(f"  program   {result.program} ({result.words} words)")
    if result.elf:
        print(f"  host ELF  {result.elf}")
    print(f"  report    {result.report}")
    for note in result.notes:
        print(f"  note      {note}")
    return 0


def _version(distribution: str) -> str | None:
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        return None


def _doctor(args: argparse.Namespace) -> int:
    """Report each piece atlasc uses; fail only on what compilation needs."""
    rows: list[tuple[str, str, bool]] = []
    try:
        bins = toolchain.bin_dir()
        for name in ("atlas-opt", "atlas-emit"):
            probe = subprocess.run([str(bins / name), "--help"],
                                   capture_output=True, text=True, check=False)
            rows.append((name, str(bins / name) if probe.returncode == 0 else
                         probe.stderr.strip(), probe.returncode == 0))
    except (OSError, toolchain.ToolchainError) as error:
        rows.append(("atlas-opt, atlas-emit", str(error), False))
    try:
        rows.append(("support files", str(toolchain.data_root()), True))
        rows.append(("RTL revision", toolchain.rtl_revision(), True))
    except toolchain.ToolchainError as error:
        rows.append(("support files", str(error), False))
    distributions = ["xdsl", "numpy", "safetensors"]
    if args.capture:
        distributions.extend(["torch", "torch-mlir"])
    for distribution in distributions:
        version = _version(distribution)
        rows.append((distribution, version or "missing", version is not None))
    if args.capture:
        try:
            from .capture import _torch
            _torch()
            rows.append(("PyTorch capture imports", "loaded", True))
        except (CaptureError, OSError) as error:
            rows.append(("PyTorch capture imports", str(error), False))
    cc = toolchain.host_cc()
    rows.append(("RISC-V host compiler", cc or f"missing ({toolchain.HOST_CC}, from "
                 "Chipyard's riscv-tools); program.bin still builds", True))
    width = max(len(label) for label, _, _ in rows)
    for label, detail, ok in rows:
        print(f"{'ok  ' if ok else 'FAIL'} {label:<{width}}  {detail}")
    return 0 if all(ok for _, _, ok in rows) else 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="atlasc", description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    compile_parser = commands.add_parser(
        "build", help="compile a model to an Atlas program and host ELF")
    compile_parser.add_argument("source", type=Path, nargs="?",
                                help="loader .py defining get_model_and_inputs(), or a .pt2")
    compile_parser.add_argument("-o", "--output", type=Path,
                                help="output directory (default atlas-out/<name>)")
    compile_parser.add_argument("--policy", default="fp8",
                                help=f"preset ({', '.join(policy.PRESETS)}) or policy JSON")
    compile_parser.add_argument("--force", action="store_true",
                                help="replace an earlier atlasc build in the output directory")
    compile_parser.add_argument("--host-cc", help=f"RISC-V C compiler (default {toolchain.HOST_CC})")
    compile_parser.add_argument("--no-host-elf", action="store_true",
                                help="stop after the program and host headers")
    captured = compile_parser.add_argument_group("already captured Linalg")
    captured.add_argument("--linalg", type=Path)
    captured.add_argument("--arguments", type=Path, help="argument manifest JSON")
    captured.add_argument("--weights", type=Path, help="safetensors parameters")
    captured.add_argument("--inputs", type=Path,
                          help="safetensors example inputs by graph input name")
    compile_parser.set_defaults(run=_build)

    doctor = commands.add_parser("doctor", help="check the tools atlasc uses")
    doctor.add_argument("--capture", action="store_true", help="also check PyTorch capture")
    doctor.set_defaults(run=_doctor)

    demo = commands.add_parser("demo", help="compile the bundled two-layer MLP")
    demo.add_argument("-o", "--output", type=Path, default=Path("atlas-out/demo"))
    demo.add_argument("--force", action="store_true")
    demo.set_defaults(run=_demo)

    args = parser.parse_args(argv)
    return args.run(args)


if __name__ == "__main__":
    raise SystemExit(main())
