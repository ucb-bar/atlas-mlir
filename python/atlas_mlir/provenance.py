"""Identity checks for the wheel and its pinned compiler dependencies."""
from __future__ import annotations

import hashlib
import importlib.metadata as metadata
import importlib.util
import json
from pathlib import Path
import re
import subprocess
import tomllib
from urllib.parse import unquote, urlparse

from . import toolchain


def install_urls(requirements):
    """The pinned Merlin and model2MLIR URLs in atlas-mlir's install requirements."""
    urls = {}
    for requirement in requirements:
        name, _, url = requirement.partition("@")
        name = re.split(r"[\s\[]", name.strip(), maxsplit=1)[0]
        if name in ("merlin", "m2m"):
            if name in urls:
                raise toolchain.ToolchainError(f"atlas-mlir requires {name} more than once")
            urls[name] = url.split(";")[0].strip()
            if not re.search(r"[0-9a-f]{40}", urls[name]):
                raise toolchain.ToolchainError(f"atlas-mlir must pin {name} to a commit")
    if urls.keys() != {"merlin", "m2m"}:
        raise toolchain.ToolchainError("atlas-mlir must require pinned merlin and m2m URLs")
    return urls


def _requirements():
    """pyproject.toml is the one pin authority; an installed wheel carries it as Requires-Dist."""
    if (toolchain.PACKAGE / "build-record.json").is_file():
        return metadata.requires("atlas-mlir") or []
    project = tomllib.loads((toolchain.SOURCE_ROOT / "pyproject.toml").read_text())["project"]
    return [*project["dependencies"], *project["optional-dependencies"]["capture"]]


INSTALL_URLS = install_urls(_requirements())
PINS = {
    **{name: re.search(r"[0-9a-f]{40}", url).group() for name, url in INSTALL_URLS.items()},
    "atlas-torchao": "ce19b4ef057e72e24cec4744ecea7d6adb0920e8",
    "npu-model": "b97b2f5a97e432141c4ef8504bec396651d77a8f",
}


def _git(path, *args):
    return subprocess.check_output(["git", "-C", str(path), *args], text=True,
                                   stderr=subprocess.DEVNULL).strip()


def dependency_record(name, *, required=True):
    try:
        dist = metadata.distribution(name)
    except metadata.PackageNotFoundError:
        if required:
            raise toolchain.ToolchainError(f"{name} is missing; install the pinned compiler dependencies")
        return None
    result = {"version": dist.version}
    if name in PINS:
        direct = json.loads(dist.read_text("direct_url.json") or "{}")
        revision = direct.get("vcs_info", {}).get("commit_id")
        if name == "merlin" and direct.get("url") == INSTALL_URLS["merlin"]:
            archive_hash = direct.get("archive_info", {}).get("hashes", {}).get("sha256")
            if not archive_hash:
                raise toolchain.ToolchainError("Merlin archive installation lacks a content hash")
            revision = PINS[name]
            result["archive_sha256"] = archive_hash
        if direct.get("dir_info", {}).get("editable"):
            root = Path(unquote(urlparse(direct["url"]).path))
            revision = _git(root, "rev-parse", "HEAD")
            module = {"npu-model": "npu_model"}.get(name, name)
            spec = importlib.util.find_spec(module)
            if spec is None or not spec.origin or not Path(spec.origin).resolve().is_relative_to(root):
                raise toolchain.ToolchainError(f"imported {name} is outside its pinned editable checkout")
            if _git(root, "status", "--porcelain", "--untracked-files=no"):
                raise toolchain.ToolchainError(f"{name} checkout has tracked changes")
        if revision != PINS[name]:
            raise toolchain.ToolchainError(f"{name} must be installed from commit {PINS[name]}; found {revision}")
        result["revision"] = revision
    return result


def unique_plugins():
    for group, name in (("merlin.native_target_compilers", "atlas"),
                        ("m2m.quantization_adapters", "atlas_fp8")):
        entries = [e for e in metadata.entry_points(group=group) if e.name == name]
        if len(entries) != 1 or entries[0].dist.metadata["Name"] != "atlas-mlir":
            raise toolchain.ToolchainError(f"expected one {group}:{name} owned by atlas-mlir; "
                "remove the legacy atlas-mlir-native-support / atlas-m2m-adapter installations")


def build_record():
    record_path = toolchain.PACKAGE / "build-record.json"
    if not record_path.is_file():
        root = toolchain.SOURCE_ROOT
        try:
            return {"kind": "source-checkout", "revision": _git(root, "rev-parse", "HEAD"),
                    "dirty": bool(_git(root, "status", "--porcelain", "--untracked-files=no"))}
        except subprocess.CalledProcessError as error:
            raise toolchain.ToolchainError("wheel build record is missing") from error
    record = json.loads(record_path.read_text())
    for name, expected in record["files"].items():
        path = toolchain.PACKAGE / name
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise toolchain.ToolchainError(f"packaged file hash mismatch: {name}; reinstall atlas-mlir")
    return record


def environment_record(*, capture=False):
    unique_plugins()
    # Merlin pins z3-solver because buffer placement counts its work.
    names = ["merlin", "xdsl", "numpy", "safetensors", "z3-solver"]
    if capture:
        names += ["m2m", "torch"]
    dependencies = {name: dependency_record(name) for name in names}
    if dependencies["xdsl"]["version"] != "0.68.0":
        raise toolchain.ToolchainError("the compiler is qualified with xdsl==0.68.0")
    if capture and dependencies["torch"]["version"] != "2.11.0+cpu":
        raise toolchain.ToolchainError("capture requires torch==2.11.0+cpu from the PyTorch CPU index")
    return {"atlas": build_record(), "dependencies": dependencies}
