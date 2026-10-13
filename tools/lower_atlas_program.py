#!/usr/bin/env python3
"""Lower one virtual Atlas program through every stage, keeping each artifact.

The output directory receives:
  1-virtual.mlir    the source, after --verify-atlas-virtual-stream
  2-scheduled.mlir  after --schedule-atlas-virtual
  3-allocated.mlir  untimed machine IR, after --lower-atlas-virtual-to-machine
  4-timed.mlir      after the timing pass
  5-words.hex       instruction words, from atlas-emit
  5-map.json        the word-to-operation map, from atlas-emit --map-json
  5-program.json    the decoded program, from atlas-emit --program-json
  commands.txt      every command, in order
  log.txt           each command's diagnostics and exit status

The driver stops at the first failing command; its diagnostic ends log.txt.
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
         ["--lower-atlas-virtual-to-machine", "--verify-atlas-generated-schedule"]),
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
                        default="insert-atlas-delays")
    args = parser.parse_args()
    source = args.source.resolve(strict=True)
    bin_dir = args.atlas_bin_dir.resolve(strict=True)
    output = args.output_dir.resolve()
    if output.exists() and any(output.iterdir()):
        parser.error("--output-dir must be empty to prevent stale artifacts")
    output.mkdir(parents=True, exist_ok=True)

    commands, log = [], []
    for artifact, tool, given, options in stages(source, args.timing):
        command = [str(bin_dir / tool), *options, str(output / given)]
        if tool == "atlas-opt":
            # A diagnostic names the failing resource; the operation dump would bury it.
            command += ["--mlir-print-op-on-diagnostic=false", "-o", str(output / artifact)]
        commands.append(shlex.join(command) +
                        ("" if tool == "atlas-opt" else f" > {shlex.quote(str(output / artifact))}"))
        result = subprocess.run(command, text=True, capture_output=True)
        if tool == "atlas-emit" and result.returncode == 0:
            (output / artifact).write_text(result.stdout)
        log.append(f"$ {commands[-1]}\n{result.stderr}exit {result.returncode}\n")
        (output / "commands.txt").write_text("\n".join(commands) + "\n")
        (output / "log.txt").write_text("\n".join(log))
        if result.returncode:
            print(f"{artifact} failed; see {output / 'log.txt'}\n{result.stderr}",
                  end="", file=sys.stderr)
            return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
