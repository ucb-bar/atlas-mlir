#!/usr/bin/env python3
"""Lower one virtual Atlas program through every stage, keeping each artifact.

The output directory receives:
  1-virtual.mlir    the source, after --verify-atlas-virtual-stream
  2-scheduled.mlir  after --schedule-atlas-virtual
  3-allocated.mlir  untimed machine IR, after --lower-atlas-virtual-to-machine
  3-allocation.json the allocation plan: each value's registers, MXU slot or
                    DMA channel, peak demand against capacity per register
                    kind, the VMEM windows used, whether it fits, and every
                    error; written even when the program does not fit
  4-timed.mlir      after --schedule-atlas-stream (the default timing pass)
  5-words.hex       instruction words, from atlas-emit
  5-map.json        the word-to-operation map, from atlas-emit --map-json
  5-program.json    the decoded program, from atlas-emit --program-json
  commands.txt      every command, in order
  log.txt           each command's diagnostics and exit status

The driver stops at the first failing command; its diagnostic ends log.txt.
If allocation fails, 3-allocation.json still names each resource that does
not fit.
"""

from __future__ import annotations

import argparse
import os
import shlex
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def stages(source: Path, timing: str) -> list[tuple[str, str, Path | str, list[str]]]:
    """(artifact, tool, input, options) for each command, in order."""
    return [
        ("1-virtual.mlir", "atlas-opt", source, ["--verify-atlas-virtual-stream"]),
        ("2-scheduled.mlir", "atlas-opt", "1-virtual.mlir",
         ["--schedule-atlas-virtual", "--verify-atlas-virtual-stream"]),
        ("3-allocated.mlir", "atlas-opt", "2-scheduled.mlir",
         ["--lower-atlas-virtual-to-machine=allocation-report=3-allocation.json",
          "--verify-atlas-generated-schedule"]),
        ("4-timed.mlir", "atlas-opt", "3-allocated.mlir",
         [f"--{timing}", "--verify-atlas-timing", "--verify-atlas-generated-schedule"]),
        ("5-words.hex", "atlas-emit", "4-timed.mlir", []),
        ("5-map.json", "atlas-emit", "4-timed.mlir", ["--map-json"]),
        ("5-program.json", "atlas-emit", "4-timed.mlir", ["--program-json"]),
    ]


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("source", type=Path, help="virtual Atlas program")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--atlas-bin-dir", type=Path,
                        default=Path(os.environ.get("ATLAS_OOT_BIN_DIR", ROOT / "build/bin")))
    parser.add_argument("--timing", choices=("insert-atlas-delays", "schedule-atlas-stream"),
                        default="schedule-atlas-stream",
                        help="machine timing pass (default: schedule-atlas-stream)")
    args = parser.parse_args()
    try:
        source = args.source.resolve(strict=True)
        bin_dir = args.atlas_bin_dir.resolve(strict=True)
        output = args.output_dir.resolve()
        if output.exists() and (not output.is_dir() or any(output.iterdir())):
            parser.error("--output-dir must be an empty directory to prevent stale artifacts")
        output.mkdir(parents=True, exist_ok=True)
    except OSError as error:
        parser.error(str(error))

    commands, log = [], []
    for artifact, tool, given, options in stages(source, args.timing):
        command = [str(bin_dir / tool), *options, str(output / given)]
        if tool == "atlas-opt":
            # A diagnostic names the failing resource; the operation dump would bury it.
            command += ["--mlir-print-op-on-diagnostic=false", "-o", str(output / artifact)]
        # Run in the artifact directory so the pass option has a simple relative
        # filename. MLIR pass options have their own quoting grammar, distinct
        # from argv and shell quoting. Record the working directory for replay.
        commands.append(f"cd {shlex.quote(str(output))} && " + shlex.join(command) +
                        ("" if tool == "atlas-opt" else f" > {shlex.quote(str(output / artifact))}"))
        (output / "commands.txt").write_text("\n".join(commands) + "\n")
        try:
            result = subprocess.run(command, cwd=output, text=True, capture_output=True)
        except OSError as error:
            result = subprocess.CompletedProcess(command, 1, "", f"{error}\n")
        if tool == "atlas-emit" and result.returncode == 0:
            (output / artifact).write_text(result.stdout)
        log.append(f"$ {commands[-1]}\n{result.stderr}exit {result.returncode}\n")
        (output / "log.txt").write_text("\n".join(log))
        if result.returncode:
            print(f"{artifact} failed; see {output / 'log.txt'}\n{result.stderr}",
                  end="", file=sys.stderr)
            return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
