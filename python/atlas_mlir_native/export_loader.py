"""Authenticate saved exporter bundles without executing their graph.

Hashes establish local content integrity, not a signature or exporter revision
approval. Legacy policy facts are source contracts, never target qualification.
Framework imports and exporter registrations belong to the later graph consumer.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import struct
from typing import Any


class ExportLoadError(ValueError):
    """A saved export is incomplete, inconsistent, or unsafe to read."""


_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_DTYPES = {"BOOL": 1, "U8": 1, "I8": 1, "I16": 2, "U16": 2,
           "I32": 4, "U32": 4, "I64": 8, "U64": 8,
           "BF16": 2, "F16": 2, "F32": 4, "F64": 8,
           "F8_E4M3": 1, "F8_E5M2": 1}


def _fail(message: str):
    raise ExportLoadError(message)


def _safe_path(path: Path) -> Path:
    # Validate spelling before any filesystem access; reject symlinks instead of
    # resolving a link whose destination may leave the admitted tree.
    path = Path(path)
    if ".." in path.parts:
        _fail("parent traversal is not permitted")
    path = Path(os.path.abspath(path))
    for part in reversed((path, *path.parents)):
        if part.is_symlink():
            _fail(f"symlink is not permitted: {part}")
    return path


def _member(base: Path, name: str) -> Path:
    if not isinstance(name, str) or "\\" in name:
        _fail("inventory paths must be relative POSIX paths")
    relative = PurePosixPath(name)
    if relative.is_absolute() or not relative.parts or any(p in (".", "..") for p in name.split("/")):
        _fail(f"invalid inventory path: {name!r}")
    return _safe_path(base / relative)


def _json(path: Path) -> dict[str, Any]:
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                _fail(f"duplicate JSON key in {path.name}: {key}")
            result[key] = value
        return result
    try:
        value = json.loads(path.read_text(), object_pairs_hook=pairs,
                           parse_constant=lambda value: _fail(f"nonfinite JSON: {value}"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ExportLoadError(f"cannot read JSON {path}: {exc}") from exc
    if not isinstance(value, dict):
        _fail(f"expected JSON object: {path}")
    return value


def _hash(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as exc:
        raise ExportLoadError(f"cannot read inventory member {path}: {exc}") from exc
    return digest.hexdigest()


@dataclass(frozen=True)
class TensorRecord:
    name: str
    dtype: str
    shape: tuple[int, ...]
    path: Path
    data_offset: int
    nbytes: int

    def read_bytes(self) -> bytes:
        """Read original tensor storage; perform no numerical conversion."""
        _safe_path(self.path)
        with self.path.open("rb") as stream:
            stream.seek(self.data_offset)
            data = stream.read(self.nbytes)
        if len(data) != self.nbytes:
            _fail(f"truncated tensor {self.name}")
        return data

    def sha256(self) -> str:
        """Hash this tensor's original storage without materializing it."""
        _safe_path(self.path)
        digest = hashlib.sha256()
        with self.path.open("rb") as stream:
            stream.seek(self.data_offset)
            remaining = self.nbytes
            while remaining:
                block = stream.read(min(remaining, 1024 * 1024))
                if not block:
                    _fail(f"truncated tensor {self.name}")
                digest.update(block)
                remaining -= len(block)
        return digest.hexdigest()

    def to_numpy(self):
        """Expose storage, with BF16 and float8 represented by integer bits."""
        import numpy as np
        types = {"BOOL": "?", "U8": "u1", "I8": "i1", "I16": "<i2",
                 "U16": "<u2", "BF16": "<u2", "F16": "<f2", "F32": "<f4",
                 "F64": "<f8", "I32": "<i4", "I64": "<i8", "U32": "<u4",
                 "U64": "<u8", "F8_E4M3": "u1", "F8_E5M2": "u1"}
        return np.frombuffer(self.read_bytes(), dtype=types[self.dtype]).reshape(self.shape)


@dataclass(frozen=True)
class FixtureCase:
    index: int
    inputs: tuple[TensorRecord, ...]
    outputs: tuple[TensorRecord, ...]


@dataclass(frozen=True)
class ExportBundle:
    root: Path
    export_dir: Path
    fx_dir: Path
    graph_path: Path
    fixtures_path: Path
    fixtures: tuple[FixtureCase, ...]
    metadata: dict[str, Any]
    inventory: dict[str, dict[str, Any]]

    def verify_unchanged(self) -> None:
        """Reauthenticate before a later consumer deserializes source artifacts."""
        for name, entry in self.inventory.items():
            path = _member(self.root, name)
            if path.stat().st_size != entry["bytes"] or _hash(path) != entry["sha256"]:
                _fail(f"bundle changed after intake: {name}")


def _tensors(path: Path) -> dict[str, TensorRecord]:
    try:
        with path.open("rb") as stream:
            prefix = stream.read(8)
            if len(prefix) != 8:
                _fail(f"invalid safetensors header: {path}")
            length = struct.unpack("<Q", prefix)[0]
            if length > min(path.stat().st_size - 8, 100_000_000):
                _fail(f"invalid safetensors header length: {path}")
            header = json.loads(stream.read(length), object_pairs_hook=_unique_pairs)
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ExportLoadError(f"invalid safetensors {path}: {exc}") from exc
    if not isinstance(header, dict):
        _fail(f"invalid safetensors header object: {path}")
    result = {}
    intervals = []
    for name, entry in header.items():
        if name == "__metadata__":
            continue
        if not isinstance(entry, dict):
            _fail(f"invalid tensor descriptor: {name}")
        dtype, shape, offsets = (entry.get(k) for k in ("dtype", "shape", "data_offsets"))
        if (not isinstance(dtype, str) or dtype not in _DTYPES or not isinstance(shape, list)
                or any(type(n) is not int or n < 0 for n in shape)
                or not isinstance(offsets, list) or len(offsets) != 2
                or any(type(n) is not int or n < 0 for n in offsets)):
            _fail(f"invalid tensor storage: {name}")
        start, end = offsets
        size = math.prod(shape) * _DTYPES[dtype]
        if end - start != size:
            _fail(f"tensor shape/length mismatch: {name}")
        result[name] = TensorRecord(name, dtype, tuple(shape), path, 8 + length + start, size)
        intervals.append((start, end))
    cursor = 0
    for start, end in sorted(intervals):
        if start != cursor:
            _fail(f"overlapping or incomplete tensor storage: {path}")
        cursor = end
    if 8 + length + cursor != path.stat().st_size:
        _fail(f"tensor payload length mismatch: {path}")
    return result


def _unique_pairs(items):
    result = {}
    for key, value in items:
        if key in result:
            _fail(f"duplicate tensor header key: {key}")
        result[key] = value
    return result


def _fixtures(tensors: dict[str, TensorRecord]) -> tuple[FixtureCase, ...]:
    cases = {}
    for name, tensor in tensors.items():
        match = re.fullmatch(r"case_(\d+)\.(input|output)_(\d+)", name)
        if not match:
            _fail(f"unsupported fixture tensor name: {name}")
        case, kind, index = match.groups()
        group = cases.setdefault(int(case), {"input": {}, "output": {}})[kind]
        if int(index) in group:
            _fail(f"duplicate fixture index: {name}")
        group[int(index)] = tensor
    if not cases or sorted(cases) != list(range(len(cases))):
        _fail("fixture cases must be nonempty and contiguous")
    result = []
    for index, case in sorted(cases.items()):
        for kind, group in case.items():
            if not group or sorted(group) != list(range(len(group))):
                _fail(f"fixture {kind} indices must be nonempty and contiguous")
        result.append(FixtureCase(index, tuple(case["input"][i] for i in range(len(case["input"]))),
                                  tuple(case["output"][i] for i in range(len(case["output"])))))
    return tuple(result)


def load_export(path: str | os.PathLike[str]) -> ExportBundle:
    """Read an export_module out_dir or a saved export/fx bundle.

    Passing its export/ or fx/ directory is also accepted. The operation hashes
    every declared inventory member, checks cross-manifest bindings, and reads
    tensor headers. It never imports torch, runs a graph, or changes weights.
    """
    root = _safe_path(Path(path))
    if root.name in ("export", "fx"):
        root = root.parent
    export_dir, fx_dir = _member(root, "export"), _member(root, "fx")
    inventory: dict[str, dict[str, Any]] = {}

    def record(file: Path, expected=None):
        name = file.relative_to(root).as_posix()
        try:
            size = file.stat().st_size
        except OSError as exc:
            raise ExportLoadError(f"missing inventory member: {name}") from exc
        if expected is not None:
            if (not isinstance(expected, dict) or not isinstance(expected.get("sha256"), str)
                    or not _SHA256.fullmatch(expected["sha256"])
                    or type(expected.get("bytes")) is not int or expected["bytes"] < 0):
                _fail(f"invalid inventory digest/length: {name}")
            if size != expected["bytes"]:
                _fail(f"inventory length mismatch: {name}")
        actual = {"sha256": _hash(file), "bytes": size}
        if expected is not None and actual["sha256"] != expected["sha256"]:
            _fail(f"inventory hash mismatch: {name}")
        if name in inventory and inventory[name] != actual:
            _fail(f"conflicting inventory declarations: {name}")
        inventory[name] = actual
        return actual["sha256"]

    def manifest(directory: Path, name: str, schema: str):
        file = _member(directory, name)
        record(file)
        value = _json(file)
        if value.get("schema") != schema or not isinstance(value.get("files"), dict) or not value["files"]:
            _fail(f"unsupported or incomplete manifest: {name}")
        for filename, expected in value["files"].items():
            record(_member(directory, filename), expected)
        return value

    exported = manifest(export_dir, "manifest.json", "atlas_torchao.export_manifest.v1")
    fx = manifest(fx_dir, "fx_manifest.json", "atlas_torchao.fx_bundle.v1")
    for name in ("model.safetensors", "policy.json"):
        if name not in exported["files"]:
            _fail(f"export inventory omits {name}")
    for name in ("model.pt2", "fixtures.safetensors", "policy.json", "buffer_inventory.json", "graph_inspection.json"):
        if name not in fx["files"]:
            _fail(f"FX inventory omits {name}")
    policy_hash = inventory["export/policy.json"]["sha256"]
    if exported.get("policy_sha256") != policy_hash:
        _fail("export policy binding mismatch")
    provenance = fx.get("provenance", {})
    if not isinstance(provenance, dict) or provenance.get("export_manifest_sha256") != inventory["export/manifest.json"]["sha256"]:
        _fail("FX/export manifest binding mismatch")
    optional_manifest = _member(root, "bundle_manifest.json")
    if optional_manifest.exists():
        record(optional_manifest)
        bundle_manifest = _json(optional_manifest)
        if not isinstance(bundle_manifest.get("files"), dict):
            _fail("bundle manifest is missing its inventory")
        for name, expected in bundle_manifest["files"].items():
            record(_member(root, name), expected)
    verification_path = _member(root, "verify_bundle.json")
    record(verification_path)
    verification = _json(verification_path)
    checks = verification.get("checks")
    if (verification.get("ok") is not True or verification.get("failed") != []
            or not isinstance(checks, dict) or not checks
            or any(not isinstance(v, dict) or v.get("ok") is not True for v in checks.values())):
        _fail("supplied bundle verification did not pass")
    policy = _json(_member(export_dir, "policy.json"))
    fx_policy = _json(_member(fx_dir, "policy.json"))
    if policy.get("schema") != "atlas_torchao.execution_policy.v1" or fx_policy.get("schema") != policy["schema"]:
        _fail("unsupported execution policy schema")
    weights = _tensors(_member(export_dir, "model.safetensors"))
    sites = policy.get("sites")
    if not isinstance(sites, dict):
        _fail("policy is missing site contracts")
    quantized_sites = []
    for name, site in sites.items():
        if not isinstance(site, dict):
            _fail(f"invalid site contract: {name}")
        if site.get("status") != "quantized":
            continue
        weight = site.get("weight")
        if weight is not None:
            if not isinstance(weight, dict):
                _fail(f"invalid weight contract: {name}")
            for field, dtype in (("codes", "U8"), ("scale_exp", "I32")):
                tensor = weights.get(weight.get(field))
                if tensor is None or tensor.dtype != dtype:
                    _fail(f"quantized weight contract/storage mismatch: {name}.{field}")
                declared_shape = weight.get(field + "_shape")
                if declared_shape is not None and declared_shape != list(tensor.shape):
                    _fail(f"quantized weight shape mismatch: {name}.{field}")
        quantized_sites.append(name)
    buffer_inventory = _json(_member(fx_dir, "buffer_inventory.json"))
    if quantized_sites and not buffer_inventory:
        _fail("quantized graph buffer inventory is empty")
    tensor_digests = {}
    for name, item in buffer_inventory.items():
        if (not isinstance(item, dict) or not isinstance(item.get("sha256"), str)
                or not _SHA256.fullmatch(item["sha256"])
                or not isinstance(item.get("shape"), list)
                or any(type(n) is not int or n < 0 for n in item["shape"])
                or not isinstance(item.get("dtype"), str)):
            _fail(f"invalid graph buffer inventory: {name}")
        if name.rsplit(".", 1)[-1] in ("weight_codes", "weight_scale_exp"):
            # Capture wrappers may add a module prefix. Match source tensor
            # names structurally, and refuse an ambiguous suffix.
            matches = [tensor for key, tensor in weights.items()
                       if name == key or name.endswith("." + key)]
            if len(matches) != 1:
                _fail(f"graph weight buffer has no unique source tensor: {name}")
            tensor, = matches
            expected_dtype = {"U8": "torch.uint8", "I32": "torch.int32"}.get(tensor.dtype)
            if item["shape"] != list(tensor.shape) or item["dtype"] != expected_dtype:
                _fail(f"graph/source tensor storage mismatch: {name}")
            if tensor.name not in tensor_digests:
                tensor_digests[tensor.name] = tensor.sha256()
            if item["sha256"] != tensor_digests[tensor.name]:
                _fail(f"graph/source tensor hash mismatch: {name}")
    native = fx.get("native_fp8", {})
    already_quantized = bool(quantized_sites)
    if already_quantized and (not isinstance(native, dict) or native.get("storage") != "uint8" or native.get("weight_format") != "atlas_e4m3"):
        _fail("quantized policy is missing native FP8 graph storage evidence")
    source_contract = {key: policy.get(key) for key in ("operand", "scale", "accumulate", "sites", "runtime")}
    contract_hash = hashlib.sha256(json.dumps(source_contract, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    fixtures_path = _member(fx_dir, "fixtures.safetensors")
    fixture_cases = _fixtures(_tensors(fixtures_path))
    metadata = {"schema": "atlas.authenticated_export.v1", "already_quantized": already_quantized,
                "quantized_sites": quantized_sites, "operand_dtype": policy.get("operand", {}).get("dtype"),
                "source_contract": source_contract, "source_contract_sha256": contract_hash,
                "target_contract_sha256": exported.get("contract_sha256"),
                "policy_sha256": policy_hash, "fx_policy_sha256": inventory["fx/policy.json"]["sha256"],
                "export_manifest_sha256": inventory["export/manifest.json"]["sha256"],
                "fx_manifest_sha256": inventory["fx/fx_manifest.json"]["sha256"],
                "provenance": exported.get("provenance", {}), "verification": verification,
                "policy": policy, "fx_policy": fx_policy, "export_manifest": exported, "fx_manifest": fx,
                "buffer_inventory": buffer_inventory,
                "weight_tensors": {name: {"dtype": value.dtype, "shape": list(value.shape)} for name, value in weights.items()},
                "authentication": "local manifest content integrity; no signature or revision approval"}
    return ExportBundle(root, export_dir, fx_dir, _member(fx_dir, "model.pt2"),
                        fixtures_path, fixture_cases, metadata, inventory)
