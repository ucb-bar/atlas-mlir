"""Pinned source intake for atlas-torchao's not-yet-packaged exporter modules."""
from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys

from .provenance import PINS, _git
from .toolchain import ToolchainError

URL = "https://github.com/ucb-ee290c/atlas-torchao.git"


def source_root():
    return Path(os.environ.get("ATLAS_TORCHAO_ROOT", Path.home() / ".cache" / "atlas-mlir" /
                               "atlas-torchao" / PINS["atlas-torchao"])).expanduser().resolve()


def check(root):
    if _git(root, "rev-parse", "HEAD") != PINS["atlas-torchao"]:
        raise ToolchainError("atlas-torchao revision differs; run atlasc setup-capture in a fresh directory")
    if _git(root, "status", "--porcelain", "--untracked-files=no"):
        raise ToolchainError("atlas-torchao checkout or reference submodules have tracked changes")
    for line in _git(root, "submodule", "status", "third_party/pi0-quant", "third_party/smolVLA-quant").splitlines():
        if line.startswith(("-", "+", "U")):
            raise ToolchainError("atlas-torchao reference submodules are absent or at a different revision")


def setup():
    root = source_root()
    if not root.exists():
        root.mkdir(parents=True)
        for args in (("init",), ("remote", "add", "origin", URL),
                     ("fetch", "--depth", "1", "origin", PINS["atlas-torchao"]),
                     ("checkout", "--detach", "FETCH_HEAD")):
            subprocess.run(["git", "-C", str(root), *args], check=True)
        subprocess.run(["git", "-C", str(root), "submodule", "update", "--init", "--depth", "1",
                        "third_party/pi0-quant", "third_party/smolVLA-quant"], check=True)
    check(root)
    return root


def activate():
    root = source_root()
    if not root.exists():
        raise ToolchainError("quantized capture needs the pinned exporter sources: run atlasc setup-capture")
    check(root)
    # Upstream exporter modules use top-level imports and are missing from its
    # published package list. Keep this scoped to the capture subprocess; do not
    # install its conflicting xDSL pins or unrelated training dependencies.
    for directory in (root / "src", root / "src/exporter"):
        sys.path.insert(0, str(directory))
    return {"revision": PINS["atlas-torchao"], "reference_submodules": _git(root, "submodule", "status",
            "third_party/pi0-quant", "third_party/smolVLA-quant")}
