"""Pinned source intake for atlas-torchao's not-yet-packaged exporter modules."""
from __future__ import annotations

import ast
import importlib
import importlib.util
import os
from pathlib import Path
import subprocess
import sys

from .provenance import PINS, _git
from .toolchain import ToolchainError

URL = "https://github.com/ucb-ee290c/atlas-torchao.git"
# The pinned exporter declares its reference repositories, with their pinned commits, in this file.
REFERENCE_LOADER = "src/reference/ref_loader.py"


def source_root():
    return Path(os.environ.get("ATLAS_TORCHAO_ROOT", Path.home() / ".cache" / "atlas-mlir" /
                               "atlas-torchao" / PINS["atlas-torchao"])).expanduser().resolve()


def reference_submodules(root):
    """Return the reference submodule paths that the pinned exporter declares, checked against its Git metadata.

    The names and commits come from the PINS table of REFERENCE_LOADER at the pinned revision. Each one must be
    a submodule in that revision's .gitmodules whose recorded commit equals the declared one.
    """
    revision = PINS["atlas-torchao"]

    def read(*args):
        try:
            return _git(root, *args)
        except subprocess.CalledProcessError:
            return ""
    source = read("show", f"{revision}:{REFERENCE_LOADER}")
    if not source:
        raise ToolchainError(f"atlas-torchao {revision} lacks {REFERENCE_LOADER}")
    tree = ast.parse(source)
    tables = [node.value for node in tree.body if isinstance(node, ast.Assign)
              and [getattr(target, "id", None) for target in node.targets] == ["PINS"]]
    if len(tables) != 1:
        raise ToolchainError(f"atlas-torchao {REFERENCE_LOADER} must assign one PINS table")
    try:
        declared = ast.literal_eval(tables[0])
    except ValueError:
        declared = None
    if not isinstance(declared, dict) or not declared:
        raise ToolchainError(f"atlas-torchao {REFERENCE_LOADER} PINS must be a nonempty literal table")
    paths = []
    for name, commit in declared.items():
        path = f"third_party/{name}"
        if not isinstance(name, str) or not name or "/" in name or name.startswith("."):
            raise ToolchainError(f"atlas-torchao reference name {name!r} is not a plain directory name")
        if not isinstance(commit, str) or len(commit) != 40 or set(commit) - set("0123456789abcdef"):
            raise ToolchainError(f"atlas-torchao reference {name} has no full commit id")
        if read("config", "--blob", f"{revision}:.gitmodules", "--get", f"submodule.{path}.path") != path:
            raise ToolchainError(f"atlas-torchao reference {path} is not a declared submodule")
        if read("ls-tree", revision, "--", path).split() != ["160000", "commit", str(commit), path]:
            raise ToolchainError(f"atlas-torchao reference {path} is not pinned at its declared commit")
        paths.append(path)
    return tuple(paths)


def check(root):
    if _git(root, "rev-parse", "HEAD") != PINS["atlas-torchao"]:
        raise ToolchainError("atlas-torchao revision differs; run atlasc setup-capture in a fresh directory")
    if _git(root, "status", "--porcelain", "--untracked-files=no"):
        raise ToolchainError("atlas-torchao checkout or reference submodules have tracked changes")
    for line in _git(root, "submodule", "status", *reference_submodules(root)).splitlines():
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
                        *reference_submodules(root)], check=True)
    check(root)
    return root


class _OneModulePerFile:
    """Resolve a top-level import of an exporter module to its ``exporter.`` package module.

    With both src and src/exporter on sys.path, each exporter file has two module names. Executing
    a file twice re-registers its torch.library operators, and a program loaded before that keeps
    freed dispatcher handles, so capture fails intermittently.
    """

    def __init__(self, directory):
        self.directory = directory

    def find_spec(self, name, path=None, target=None):
        if path is None and (self.directory / f"{name}.py").is_file():
            return importlib.util.spec_from_loader(name, self)
        return None

    def create_module(self, spec):
        return importlib.import_module(f"exporter.{spec.name}")

    def exec_module(self, module):
        pass


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
    sys.meta_path.insert(0, _OneModulePerFile(root / "src/exporter"))
    return {"revision": PINS["atlas-torchao"], "reference_submodules": _git(root, "submodule", "status",
            *reference_submodules(root))}
