"""Authenticated source-only capture intake; never reads emitted programs or goldens."""
from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path
import struct
import zipfile

import numpy as np


def checked_path(path):
    path = Path(path)
    if ".." in path.parts:
        raise ValueError("parent traversal is not permitted")
    path = path.absolute()
    # Reject links before following them, including directory components.
    for component in reversed((path, *path.parents)):
        if component.is_symlink():
            raise ValueError("symlink is not permitted")
    return path


def digest(path):
    result = sha256()
    with checked_path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            result.update(block)
    return result.hexdigest()


def capture_sources(path, *, receipt_sha256):
    """Authenticate only inputs, constants, and original source trace against receipt."""
    root = checked_path(path)
    receipt_path = checked_path(root / "capture_receipt.json")
    if digest(receipt_path) != receipt_sha256:
        raise ValueError("capture receipt differs from trusted digest")
    receipt = json.loads(receipt_path.read_text())
    inventory = receipt["artifacts"]
    authenticated = {}
    for name in ("frontend-trace.json", "weights.safetensors", "inputs.npz"):
        record = inventory[name]
        file = checked_path(root / name)
        if file.stat().st_size != record["bytes"] or digest(file) != record["sha256"]:
            raise ValueError("capture artifact differs: " + name)
        authenticated[name] = record["sha256"]
    trace = json.loads(checked_path(root / "frontend-trace.json").read_text())
    nodes = trace["graphs"]["quantized"]["nodes"]
    return root, nodes, {"receipt_sha256": digest(receipt_path), "artifacts": authenticated}


def safetensor_constants(path):
    """Read raw tensor storage without framework deserialization or dtype casts."""
    path = checked_path(path)
    with path.open("rb") as stream:
        size = struct.unpack("<Q", stream.read(8))[0]
        if size > path.stat().st_size - 8:
            raise ValueError("invalid safetensors header")
        header = json.loads(stream.read(size))
    from . import Tensor
    dtypes = {"BF16": ("<u2", "bf16"), "F32": ("<u4", "f32"),
              "I64": ("<i8", "i64"), "I32": ("<i4", "i32"),
              "U8": ("u1", "u8"), "BOOL": ("?", "bool")}
    result = {}
    end_of_file = path.stat().st_size
    intervals = []
    for name, record in header.items():
        if name == "__metadata__":
            continue
        storage, logical = dtypes[record["dtype"]]
        begin, end = record["data_offsets"]
        shape = tuple(record["shape"])
        if (any(type(d) is not int or d < 0 for d in shape)
                or type(begin) is not int or type(end) is not int):
            raise ValueError("invalid tensor dimensions or offsets")
        intervals.append((begin, end))
        expected = int(np.prod(shape, dtype=object)) * np.dtype(storage).itemsize
        if begin < 0 or end - begin != expected or 8 + size + end > end_of_file:
            raise ValueError("invalid tensor storage: " + name)
        raw = np.memmap(path, dtype=storage, mode="r", offset=8 + size + begin, shape=shape)
        result[name] = Tensor(raw, logical)
    cursor = 0
    for begin, end in sorted(intervals):
        if begin != cursor or end < begin:
            raise ValueError("overlapping or incomplete tensor storage")
        cursor = end
    if cursor != end_of_file - size - 8:
        raise ValueError("unclaimed tensor storage")
    return result


def capture_inputs(path, nodes, supplied=None):
    """The capture's user inputs, or `supplied` user inputs in their place (input_000, input_001, ... in source order)."""
    from . import Tensor
    users = [n for n in nodes if n["op"] == "placeholder" and n.get("input_kind") == "USER_INPUT"]
    result = {}
    name = (lambda index: f"input_{index:03d}") if supplied else (lambda index: f"in{index}")
    with np.load(checked_path(supplied or Path(path) / "inputs.npz"), allow_pickle=False) as arrays:
        if supplied and set(arrays.files) != {name(index) for index in range(len(users))}:
            raise ValueError("supplied inputs must be exactly the source user inputs input_000 to input_%03d" % (len(users) - 1))
        for index, node in enumerate(users):
            raw = np.ascontiguousarray(arrays[name(index)])
            meta = node["results"][0]
            dtype = meta["dtype"]
            if dtype == "bfloat16":
                if raw.dtype == np.float32 and not np.any(raw.view(np.uint32) & 0xffff):
                    raw = (raw.view(np.uint32) >> 16).astype(np.uint16)
                if raw.dtype != np.uint16:
                    raise ValueError("BF16 input does not contain exact words")
                logical = "bf16"
            else:
                storage, logical = {"int64": (np.dtype("int64"), "i64"), "float32": (np.dtype("float32"), "f32"),
                                    "bool": (np.dtype("bool"), "bool")}[dtype]
                if raw.dtype != storage:
                    raise ValueError("input dtype differs")
                if logical == "f32":
                    raw = raw.view(np.uint32)
            if list(raw.shape) != meta["shape"]:
                raise ValueError("input shape differs")
            result[node["id"]] = Tensor(raw, logical)
    return result


def export_nodes(path):
    """Inspect serialized source graph metadata, without loading a framework program."""
    with zipfile.ZipFile(checked_path(path)) as archive:
        from .export_io import _member
        candidates = [_member(n) for n in archive.namelist()
                      if n.endswith(("/models/model.json", "serialized_exported_program.json"))]
        if len(candidates) != 1:
            raise ValueError("expected one serialized export graph")
        document = json.loads(archive.read(candidates[0]))
    nodes = document["graph_module"]["graph"]["nodes"]
    return [{"id": n.get("name", f"node_{i}"), "op": "call_function",
             "target": n["target"].removeprefix("torch.ops."),
             "args": n.get("inputs", []), "kwargs": {}} for i, n in enumerate(nodes)]
