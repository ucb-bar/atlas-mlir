"""Read authenticated PT2 source graphs and raw storage, without torch or pickle.

This is an adapter for PyTorch's serialized export schema 8.17, not an export
executor. Returned nodes use the independent evaluator's source-trace schema.
Graph outputs retain the serialized export's flat output order.
"""
from __future__ import annotations

import json
import math
from pathlib import PurePosixPath
import struct
import zipfile

import numpy as np

from .evaluator import Tensor
from .source_io import checked_path


# torch/_export/serde/schema.py, SCHEMA_VERSION=(8, 17). These are serialization
# enum values, not torch runtime dtypes; no framework module is imported.
_DTYPES = {1: ("uint8", "u1", "u8"), 4: ("int32", "<i4", "i32"),
           5: ("int64", "<i8", "i64"), 7: ("float32", "<u4", "f32"),
           12: ("bool", "?", "bool"), 13: ("bfloat16", "<u2", "bf16")}
_FIXTURE_DTYPES = {"U8": "u8", "I32": "i32", "I64": "i64",
                   "F32": "f32", "BOOL": "bool", "BF16": "bf16"}


def _member(name):
    if (not isinstance(name, str) or "\\" in name
            or PurePosixPath(name).is_absolute()
            or any(p in ("", ".", "..") for p in name.split("/"))):
        raise ValueError("invalid serialized export member")
    return name


def _json(archive, name):
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("duplicate serialized export JSON key")
            result[key] = value
        return result
    return json.loads(archive.read(_member(name)), object_pairs_hook=unique,
                      parse_constant=lambda v: (_ for _ in ()).throw(
                          ValueError("nonfinite serialized export JSON")))


def _union(value):
    if not isinstance(value, dict) or len(value) != 1:
        raise ValueError("invalid serialized export union")
    return next(iter(value.items()))


def _dtype(value):
    if type(value) is not int or value not in _DTYPES:
        raise ValueError(f"unsupported serialized source dtype {value!r}")
    return _DTYPES[value]


def _integer(value):
    kind, number = _union(value)
    if kind != "as_int" or type(number) is not int:
        raise ValueError("symbolic tensor dimensions/strides require runtime shape support")
    return number


def _meta(meta):
    dtype, storage, logical = _dtype(meta["dtype"])
    shape = [_integer(x) for x in meta["sizes"]]
    if any(x < 0 for x in shape) or meta.get("layout") != 7:
        raise ValueError("only nonnegative strided source tensor shapes are admitted")
    device = meta["device"]
    if device != {"type": "cpu", "index": None}:
        raise ValueError("only CPU source tensor metadata is admitted")
    return dtype, storage, logical, shape


def _value_id(name):
    return "value:" + name


def _ref(name):
    if not isinstance(name, str) or not name:
        raise ValueError("invalid serialized source value name")
    return {"value_id": _value_id(name)}


def _argument(value):
    kind, value = _union(value)
    if kind == "as_tensor":
        return _ref(value["name"])
    if kind in {"as_tensors", "as_nested_tensors"}:
        def tensors(x):
            return [tensors(y) for y in x] if isinstance(x, list) else _ref(x["name"])
        return tensors(value)
    if kind == "as_none":
        return None
    if kind in {"as_int", "as_float", "as_bool", "as_string",
                "as_ints", "as_floats", "as_bools", "as_strings", "as_int_lists"}:
        return value
    if kind in {"as_sym_int", "as_sym_bool", "as_sym_float"}:
        subkind, value = _union(value)
        if subkind == "as_name":
            return _ref(value)
        if subkind in {"as_int", "as_bool", "as_float"}:
            return value
    if kind in {"as_sym_ints", "as_sym_bools", "as_sym_floats"}:
        singular = kind[:-1]
        return [_argument({singular: x}) for x in value]
    if kind == "as_scalar_type":
        return {"kind": "dtype", "value": _dtype(value)[0]}
    if kind == "as_device":
        if value != {"type": "cpu", "index": None}:
            raise ValueError("only CPU serialized device arguments are admitted")
        return {"kind": "device", "value": "cpu"}
    if kind == "as_layout" and value == 7:
        return {"kind": "layout", "value": "strided"}
    if kind == "as_memory_format" and value in {1, 4}:
        return {"kind": "memory_format", "value": {1: "contiguous_format", 4: "preserve_format"}[value]}
    raise ValueError(f"unsupported serialized source argument {kind}")


def _result(value, tensor_values):
    kind, item = _union(value)
    if kind == "as_tensor":
        name = item["name"]
        dtype, _, _, shape = _meta(tensor_values[name])
        return {"id": _value_id(name), "dtype": dtype, "shape": shape, "device": "cpu"}
    if kind in {"as_sym_int", "as_sym_bool", "as_sym_float"}:
        subkind, name = _union(item)
        if subkind == "as_name":
            return {"id": _value_id(name)}
    raise ValueError(f"unsupported serialized source result {kind}")


def _storage(archive, path, prefix, record):
    if record.get("use_pickle") is not False:
        raise ValueError("pickled source tensors are not admitted")
    dtype, storage, logical, shape = _meta(record["tensor_meta"])
    meta = record["tensor_meta"]
    strides = [_integer(x) for x in meta["strides"]]
    offset = _integer(meta["storage_offset"])
    if len(strides) != len(shape) or offset < 0 or any(x < 0 for x in strides):
        raise ValueError("invalid source tensor strides/storage offset")
    name = _member(prefix + "/" + _member(record["path_name"]))
    info = archive.getinfo(name)
    itemsize = np.dtype(storage).itemsize
    extent = offset + (1 + sum((d - 1) * s for d, s in zip(shape, strides))
                       if math.prod(shape) else 0)
    if info.file_size < extent * itemsize or info.file_size % itemsize:
        raise ValueError("source tensor storage is truncated or misaligned")
    if info.flag_bits & 1:
        raise ValueError("encrypted source storage is not admitted")
    if info.compress_type == zipfile.ZIP_STORED and info.file_size:
        if info.compress_size != info.file_size:
            raise ValueError("stored source member has inconsistent compressed length")
        # Keep ZipFile's filename and overlapping-member checks when mapping
        # storage directly. Opening performs these checks without reading data.
        with archive.open(info):
            pass
        # The enclosing PT2 is authenticated by ExportBundle. Map only its raw
        # storage payload; do not deserialize sample_inputs or any pickle.
        with path.open("rb") as stream:
            stream.seek(info.header_offset)
            header = stream.read(30)
        if len(header) != 30 or header[:4] != b"PK\x03\x04":
            raise ValueError("invalid local source storage header")
        filename_size, extra_size = struct.unpack_from("<HH", header, 26)
        begin = info.header_offset + 30 + filename_size + extra_size
        if begin + info.file_size > path.stat().st_size:
            raise ValueError("source storage exceeds authenticated archive")
        raw = np.memmap(path, dtype="u1", mode="r", offset=begin, shape=info.file_size)
    else:
        raw = archive.read(name)
    words = np.ndarray(tuple(shape), dtype=storage, buffer=raw, offset=offset * itemsize,
                       strides=tuple(x * itemsize for x in strides))
    return Tensor(words, logical)


def export_sources(bundle, case_index=0):
    """Return ``(nodes, constants, inputs)`` from an authenticated ExportBundle.

    Only fixture inputs are read. Supplied fixture outputs never participate in
    generating reference values. Unsupported schemas, mutations, tensor kinds,
    and unbacked symbolic shapes fail explicitly rather than invoking torch.
    """
    if type(case_index) is not int or not 0 <= case_index < len(bundle.fixtures):
        raise ValueError("fixture case index is outside the authenticated bundle")
    bundle.verify_unchanged()
    path = checked_path(bundle.graph_path)
    with zipfile.ZipFile(path) as archive:
        names = archive.namelist()
        if len(set(names)) != len(names):
            raise ValueError("duplicate serialized export archive member")
        candidates = [_member(n) for n in names if n.endswith("/models/model.json")]
        if len(candidates) != 1:
            raise ValueError("expected one raw-storage PT2 source graph")
        model_path, = candidates
        document = _json(archive, model_path)
        if document.get("schema_version") != {"major": 8, "minor": 17}:
            raise ValueError("unsupported serialized export schema; expected 8.17")
        prefix = model_path.removesuffix("/models/model.json")
        if archive.read(_member(prefix + "/byteorder")) != b"little":
            raise ValueError("only little-endian raw source storage is admitted")
        module = document["graph_module"]
        graph, signature = module["graph"], module["signature"]
        configs = {}
        for group in ("weights", "constants"):
            base = prefix + "/data/" + group
            config = _json(archive, base + "/model_" + group + "_config.json")["config"]
            for name, record in config.items():
                if name in configs:
                    raise ValueError("duplicate serialized source tensor binding")
                configs[name] = base, record
        nodes, constants, inputs, ids, values = [], {}, {}, set(), set()

        def append(node):
            if node["id"] in ids or node["id"] in values:
                raise ValueError("duplicate serialized source node/value id")
            ids.add(node["id"])
            for result in node.get("results", []):
                if result["id"] in values or result["id"] in ids:
                    raise ValueError("duplicate serialized source result id")
                values.add(result["id"])
            nodes.append(node)

        input_specs = signature["input_specs"]
        if len(graph["inputs"]) != len(input_specs):
            raise ValueError("source graph/signature input count differs")
        fixture_inputs = iter(bundle.fixtures[case_index].inputs)
        for source_input, spec in zip(graph["inputs"], input_specs):
            kind, binding = _union(spec)
            name = _union(source_input)[1]["name"]
            argument = binding["arg"]
            binding_name = argument.get("name", argument.get("as_tensor", {}).get("name"))
            if name != binding_name:
                raise ValueError("source graph/signature input binding differs")
            result = _result(source_input, graph["tensor_values"])
            node = {"id": name, "op": "placeholder", "target": name, "results": [result]}
            if kind == "user_input":
                node["input_kind"] = "USER_INPUT"
                fixture = next(fixture_inputs, None)
                if fixture is None or fixture.dtype not in _FIXTURE_DTYPES:
                    raise ValueError("missing or unsupported source fixture input")
                value = Tensor(fixture.to_numpy(), _FIXTURE_DTYPES[fixture.dtype])
                inputs[name] = value
            elif kind in {"parameter", "buffer", "tensor_constant"}:
                target = binding[kind + "_name"]
                node["input_kind"] = {"parameter": "PARAMETER", "buffer": "BUFFER",
                                      "tensor_constant": "CONSTANT_TENSOR"}[kind]
                node["input_target"] = target
                if target not in configs:
                    raise ValueError("source tensor has no raw serialized binding: " + target)
                base, record = configs[target]
                value = _storage(archive, path, base, record)
                constants[target] = value
            else:
                raise ValueError("unsupported source input signature kind: " + kind)
            if value.dtype != _dtype(graph["tensor_values"][name]["dtype"])[2] or list(value.raw.shape) != result["shape"]:
                raise ValueError("source input storage differs from graph metadata: " + name)
            append(node)
        if next(fixture_inputs, None) is not None:
            raise ValueError("fixture has more inputs than source signature")
        for index, source in enumerate(graph["nodes"]):
            name = source.get("name") or "source_node_" + str(index)
            try:
                args, kwargs = [], {}
                for item in source["inputs"]:
                    value = _argument(item["arg"])
                    if item.get("kind") == 1:
                        if kwargs:
                            raise ValueError("positional source argument follows a keyword")
                        args.append(value)
                    elif item.get("kind") == 2 and item["name"] not in kwargs:
                        kwargs[item["name"]] = value
                    else:
                        raise ValueError("invalid serialized source argument kind/name")
                append({"id": name, "op": "call_function",
                        "target": source["target"].removeprefix("torch.ops."),
                        "args": args, "kwargs": kwargs,
                        "results": [_result(x, graph["tensor_values"]) for x in source["outputs"]]})
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(f"serialized source node {name} ({source['target']}): {exc}") from exc
        outputs = signature["output_specs"]
        if len(outputs) != len(graph["outputs"]):
            raise ValueError("source graph/signature output count differs")
        for value, spec in zip(graph["outputs"], outputs):
            kind, record = _union(spec)
            if kind != "user_output" or record["arg"] != value:
                raise ValueError("source mutations or non-user outputs are not admitted")
        output = [_argument(x) for x in graph["outputs"]]
        if graph["is_single_tensor_return"]:
            if len(output) != 1:
                raise ValueError("single source output has inconsistent arity")
            output = output[0]
        append({"id": "__source_output__", "op": "output", "target": "output", "args": [output]})
    return nodes, constants, inputs
