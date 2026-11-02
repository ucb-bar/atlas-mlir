"""Locate atlas-opt, atlas-emit, the packaged support files and the RISC-V host compiler."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

PACKAGE = Path(__file__).resolve().parent
SOURCE_ROOT = PACKAGE.parents[1]
HOST_CC = "riscv64-unknown-elf-gcc"


class ToolchainError(RuntimeError):
    """A required Atlas tool or support file is missing."""


class ToolFailure(RuntimeError):
    """A compiler or packaging step refused its input or failed."""


def data_root() -> Path:
    """The wheel's copy of tools/, runtime/, docs/ and the startup files, or the source tree.

    Both have the same layout, so the tools/ scripts find their sibling files either way.
    """
    for root in (PACKAGE / "share", SOURCE_ROOT):
        if (root / "tools/compile_atlas_linalg_tile.py").is_file():
            return root
    raise ToolchainError("Atlas support files are missing; reinstall atlas-mlir")


def bin_dir() -> Path:
    """ATLAS_OOT_BIN_DIR if set, else the wheel's tools, else a source build in build/bin."""
    override = os.environ.get("ATLAS_OOT_BIN_DIR")
    candidates = [Path(override)] if override else [PACKAGE / "bin", SOURCE_ROOT / "build/bin"]
    for directory in candidates:
        if all((directory / name).is_file() for name in ("atlas-opt", "atlas-emit")):
            return directory
    searched = ", ".join(str(directory) for directory in candidates)
    raise ToolchainError(f"atlas-opt and atlas-emit not found in {searched}")


def tool(name: str) -> Path:
    return data_root() / "tools" / name


def run(command: list[str], step: str) -> str:
    """Run one step; on failure raise with the last lines it printed, which name the reason."""
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    if result.returncode:
        detail = (result.stderr.strip() or result.stdout.strip()).splitlines()[-5:]
        raise ToolFailure(f"{step} failed: " + "\n  ".join(detail or [f"exit {result.returncode}"]))
    return result.stdout


def run_tool(name: str, *arguments: str | Path, step: str) -> str:
    """Run one of the repository's tools/ scripts with this interpreter."""
    return run([sys.executable, str(tool(name)), *map(str, arguments)], step)


def rtl_revision() -> str:
    inventory = data_root() / "docs/selected-variant-inventory.json"
    return json.loads(inventory.read_text())["selected_sources"]["rtl_revision"]


def host_cc(requested: str | None = None) -> str | None:
    """The RISC-V C compiler for the host ELF: the one requested, else Chipyard's riscv-tools GCC."""
    if requested:
        found = shutil.which(requested)
        if not found:
            raise ToolchainError(f"host compiler not found: {requested}")
        return found
    found = shutil.which(HOST_CC)
    if found:
        return found
    riscv = os.environ.get("RISCV")  # Set by Chipyard's env.sh.
    if riscv and (Path(riscv) / "bin" / HOST_CC).is_file():
        return str(Path(riscv) / "bin" / HOST_CC)
    return None


def _exec(name: str) -> None:
    try:
        path = bin_dir() / name
    except ToolchainError as error:
        sys.exit(f"{name}: {error}")
    os.execv(path, [str(path), *sys.argv[1:]])


def atlas_opt() -> None:
    _exec("atlas-opt")


def atlas_emit() -> None:
    _exec("atlas-emit")
