"""A source-ordered, NumPy storage interpreter with an explicit stop boundary.

Source arithmetic dispatches to independent host and selected-device handlers.
Reductions require declared order; device arithmetic requires pinned target rules.
No framework kernels, compiler evaluator, emitted code, or supplied reference
values participate. ``inspect_ops`` is a static coverage audit, not execution
or a bit-exact qualification result.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np


_DTYPES = {
    "bf16": "bf16", "bfloat16": "bf16", "f32": "f32", "float32": "f32",
    "i64": "i64", "int64": "i64", "i32": "i32", "int32": "i32",
    "u8": "u8", "uint8": "u8", "bool": "bool",
}
_STORAGE = {"bf16": np.dtype("uint16"), "f32": np.dtype("uint32"),
            "i64": np.dtype("int64"), "i32": np.dtype("int32"),
            "u8": np.dtype("uint8"), "bool": np.dtype("bool")}
_HOST_OPS = frozenset({
    "aten.embedding.default", "aten.unsqueeze.default", "aten.expand.default", "aten.expand_as.default",
    "aten._assert_tensor_metadata.default", "aten.to.dtype", "aten.to.dtype_layout",
    "aten.transpose.int", "aten.permute.default", "aten.cat.default",
    "aten.reshape.default", "aten.view.default", "aten.flatten.using_ints", "aten.contiguous.default",
    "aten.clone.default", "aten.detach.default", "aten.alias.default", "aten.dropout.default",
    "aten.slice.Tensor", "aten.select.int", "<built-in function getitem>", "aten.im2col.default",
    "aten.pad.default", "aten.split.Tensor",
})



def _dtype(value: str) -> str:
    key = str(value).removeprefix("torch.")
    if key not in _DTYPES:
        raise ValueError(f"unsupported logical dtype {value!r}")
    return _DTYPES[key]


@dataclass(frozen=True)
class Tensor:
    """Logical tensor; BF16 and F32 are unsigned raw bit arrays.

    Float32 arrays are also accepted for F32 and reinterpreted without casting.
    Integer tensors require exactly the declared storage dtype, preventing
    accidental numeric conversion of source bytes.
    """

    raw: np.ndarray
    dtype: str

    def __post_init__(self):
        dtype = _dtype(self.dtype)
        raw = np.asarray(self.raw)
        if dtype == "f32" and raw.dtype == np.dtype("float32"):
            raw = raw.view(np.uint32)
        if raw.dtype != _STORAGE[dtype]:
            raise ValueError(f"{dtype} requires {_STORAGE[dtype]} raw storage; got {raw.dtype}")
        raw = raw.view()
        raw.flags.writeable = False
        object.__setattr__(self, "raw", raw)
        object.__setattr__(self, "dtype", dtype)


class EvaluationError(ValueError):
    """The first source node that could not be evaluated; no outputs certified."""

    def __init__(self, node, reason, completed_node_ids=()):
        self.node_id = str(node.get("id", "<missing-id>"))
        self.target = str(node.get("target", node.get("op", "<missing-op>")))
        self.reason = str(reason)
        self.completed_node_ids = tuple(completed_node_ids)
        super().__init__(f"source node {self.node_id} ({self.target}): {self.reason}")

    def record(self):
        return {"node_id": self.node_id, "target": self.target, "reason": self.reason,
                "completed_nodes": len(self.completed_node_ids)}


@dataclass(frozen=True)
class EvaluationResult:
    outputs: Any
    values: Mapping[str, Any]
    completed_node_ids: tuple[str, ...]
    ordinary_framework_kernels_called: bool = False


def _unsupported(node):
    op, target = node.get("op"), str(node.get("target", ""))
    if op == "placeholder":
        if node.get("input_kind") not in {"USER_INPUT", "PARAMETER", "BUFFER", "CONSTANT_TENSOR"}:
            return "unsupported or missing source input role"
        try:
            for result in node.get("results", []):
                if result.get("dtype") is not None:
                    _dtype(result["dtype"])
        except ValueError as exc:
            return str(exc)
        return None
    if op == "output":
        return None
    if op != "call_function":
        return f"unsupported source node kind {op!r}"
    from . import host, device
    if target not in _BUFFER_OPS:
        target = _out_of_place(target) or target
    if target not in _HOST_OPS | _BUFFER_OPS and not host.handles(target) and not device.handles(target):
        return "no independent source handler"
    return None


# A fresh buffer written in place: torch.empty_like, then copy_ into views of it. Views are numpy views, so a
# write through one reaches every other view of the same storage, as in torch.
_BUFFER_OPS = frozenset({"aten.empty_like.default", "aten.copy_.default"})
# Ops whose result aliases its first operand in torch and in this evaluator (numpy views), and which read no values.
_VIEW_OPS = frozenset({"aten.slice.Tensor", "aten.select.int", "aten.view.default", "aten.reshape.default",
                       "aten.unsqueeze.default", "aten.transpose.int", "aten.permute.default", "aten.expand.default",
                       "aten.alias.default", "aten.detach.default", "aten._assert_tensor_metadata.default"})


def _out_of_place(target):
    """aten.NAME_.OVERLOAD's out-of-place form aten.NAME.OVERLOAD, or None for an op that is not in-place."""
    parts = target.split(".")
    if len(parts) == 3 and parts[0] == "aten" and parts[1].endswith("_") and not parts[1].startswith("_"):
        return f"aten.{parts[1][:-1]}.{parts[2]}"
    return None


def _root(raw):
    while isinstance(raw.base, np.ndarray):
        raw = raw.base
    return raw


def _elements(view, root):
    """Flat element indices of root that a numpy view of it covers."""
    offset = (view.__array_interface__["data"][0] - root.__array_interface__["data"][0]) // root.itemsize
    index = np.arange(root.size).reshape(-1)[offset:]
    return np.lib.stride_tricks.as_strided(index, view.shape, tuple(s // root.itemsize * index.itemsize for s in view.strides))


class _Buffers:
    """Fresh empty_like storage, which elements copy_ has written, and whether any value has read it."""

    def __init__(self):
        self._fresh = {}

    def empty_like(self, args, kwargs):
        _arity(args, kwargs, 1, allowed=("dtype", "layout", "device", "pin_memory", "memory_format"))
        _cpu_options(kwargs)
        value = _tensor(args[0])
        dtype = _dtype(kwargs["dtype"]) if kwargs.get("dtype") is not None else value.dtype
        root = np.zeros(value.raw.shape, _STORAGE[dtype])
        self._fresh[id(root)] = (root, np.zeros(root.size, bool), [False])
        return Tensor(root, dtype)

    def copy_(self, args, kwargs):
        _arity(args, kwargs, 2, 3)
        destination, source = _tensor(args[0]), _tensor(args[1])
        if len(args) > 2 and args[2] is not False:
            raise ValueError("asynchronous copy_ is unsupported")
        entry = self._fresh.get(id(_root(destination.raw)))
        if entry is None:
            raise ValueError("copy_ writes only into a view of a fresh empty_like buffer")
        root, written, read = entry
        if read[0]:
            raise ValueError("copy_ after a value has read its empty_like buffer is not admitted")
        if source.dtype != destination.dtype:
            raise ValueError("copy_ with a dtype change is not admitted")
        self.read(source)
        elements = _elements(destination.raw, root).reshape(-1)
        if np.unique(elements).size != elements.size:
            raise ValueError("copy_ destination overlaps itself")
        root.reshape(-1)[elements] = np.broadcast_to(source.raw, destination.raw.shape).reshape(-1)
        written[elements] = True
        return destination

    def read(self, value):
        """Refuse a value that reads an unwritten element of a fresh buffer; mark the buffer read."""
        if isinstance(value, (tuple, list)):
            for item in value:
                self.read(item)
        elif isinstance(value, Mapping):
            for item in value.values():
                self.read(item)
        elif isinstance(value, Tensor) and self._fresh:
            entry = self._fresh.get(id(_root(value.raw)))
            if entry is not None:
                root, written, read = entry
                if not written[_elements(value.raw, root)].all():
                    raise ValueError("reads an element of an empty_like buffer that no copy_ has written")
                read[0] = True


def _single_reader(nodes):
    """Node ids whose value exactly one node reads."""
    readers = Counter()

    def visit(value, seen):
        if isinstance(value, Mapping):
            if "node_id" in value and value["node_id"] not in seen:
                seen.add(value["node_id"])
                readers[value["node_id"]] += 1
            for item in value.values():
                visit(item, seen)
        elif isinstance(value, (tuple, list)):
            for item in value:
                visit(item, seen)

    for node in nodes:
        visit([node.get("args", []), node.get("kwargs", {})], set())
    return {key for key, count in readers.items() if count == 1}


def inspect_ops(nodes: Sequence[Mapping], target_rules: Mapping | None = None) -> dict:
    """Report static coverage, retaining the first source occurrence of each op.

    ``supported`` means a handler exists; runtime shapes, argument domains and
    metadata are still checked by ``evaluate``. Target rules cannot turn an
    unqualified handler into a supported operation.
    """
    supported, unsupported = {}, {}
    counts = Counter()
    first = None
    for node in nodes:
        target = str(node.get("target", node.get("op", "")))
        # Placeholder names are bindings, not operation families.
        key = str(node.get("op")) if node.get("op") in {"placeholder", "output"} else target
        counts[key] += 1
        reason = _unsupported(node)
        record = {"target": key, "first_node_id": str(node.get("id", "<missing-id>"))}
        if reason:
            record["reason"] = reason
            unsupported.setdefault(key, record)
            if first is None:
                first = {"node_id": record["first_node_id"], "target": target, "reason": reason}
        else:
            supported.setdefault(key, record)
    for group in (supported, unsupported):
        for key, record in group.items():
            record["count"] = counts[key]
    return {"schema": "atlas.source_reference_coverage.v1", "status": "unsupported" if first else "covered",
            "audit_only": True, "executed": False, "node_count": sum(counts.values()),
            "supported": list(supported.values()), "unsupported": list(unsupported.values()),
            "first_unsupported": first, "ordinary_framework_kernels_called": False}


def _tensor(value):
    if not isinstance(value, Tensor):
        raise ValueError("expected a logical Tensor")
    return value


def _axis(value, rank, *, insert=False):
    if type(value) is not int:
        raise ValueError("axis must be an integer")
    extent = rank + int(insert)
    if not -extent <= value < extent:
        raise ValueError("axis is outside source tensor rank")
    return value % extent


def _arity(args, kwargs, lo, hi=None, allowed=()):
    if not lo <= len(args) <= (lo if hi is None else hi) or set(kwargs) - set(allowed):
        raise ValueError("unsupported source argument signature")


def _cpu_options(kwargs):
    if kwargs.get("device") not in {None, "cpu"} or kwargs.get("layout") not in {None, "torch.strided", "strided"}:
        raise ValueError("only CPU strided tensor semantics are admitted")
    if kwargs.get("memory_format") not in {None, "torch.contiguous_format", "contiguous_format", "torch.preserve_format", "preserve_format"}:
        raise ValueError("unsupported memory format")
    if kwargs.get("pin_memory") not in {None, False} or kwargs.get("non_blocking") not in {None, False}:
        raise ValueError("asynchronous or pinned transfer semantics are unsupported")


def _convert(value, dtype):
    value, dtype = _tensor(value), _dtype(dtype)
    if value.dtype == dtype:
        return Tensor(value.raw.copy(), dtype)
    if value.dtype == "bf16" and dtype == "f32":
        return Tensor(value.raw.astype(np.uint32) << np.uint32(16), "f32")
    if value.dtype in {"i64", "i32", "u8", "bool"} and dtype == "f32":
        return Tensor(value.raw.astype(np.float32), "f32")
    if value.dtype == "f32" and dtype == "bf16":
        raw = value.raw
        if np.any((raw & np.uint32(0x7f800000)) == np.uint32(0x7f800000)):
            raise ValueError("F32-to-BF16 conversion is qualified only for finite source values")
        rounded = (raw + np.uint32(0x7fff) + ((raw >> np.uint32(16)) & np.uint32(1))) >> np.uint32(16)
        return Tensor(rounded.astype(np.uint16), "bf16")
    raise ValueError(f"unsupported source dtype conversion {value.dtype} to {dtype}")


def _fill_raw(fill, dtype):
    """Source bits of a pad value: Python float to F32 to BF16, both round to nearest even, as torch fills."""
    if fill is None:
        fill = 0
    if type(fill) not in {int, float}:
        raise ValueError("pad value must be a number")
    if dtype in {"bf16", "f32"}:
        limit = (2 - 2**-7) * 2.0**127 if dtype == "bf16" else float(np.finfo(np.float32).max)
        if not abs(fill) <= limit:
            raise ValueError(f"pad value {fill!r} is not a finite {dtype} value")
        single = Tensor(np.array(fill, np.float32), "f32")
        return (single if dtype == "f32" else _convert(single, "bf16")).raw
    if dtype == "bool" or not float(fill).is_integer():
        raise ValueError(f"pad value {fill!r} is supported for {dtype} only as an integer")
    info = np.iinfo(_STORAGE[dtype])
    if not info.min <= int(fill) <= info.max:
        raise ValueError(f"pad value {fill!r} is outside {dtype}")
    return _STORAGE[dtype].type(int(fill))


def _pad(args, kwargs):
    """aten.pad.default in constant mode: crop for negative pads, then fill, as torch's constant_pad_nd."""
    _arity(args, kwargs, 2, 4, ("mode", "value"))
    options = dict(zip(("mode", "value"), args[2:]))
    if set(options) & set(kwargs):
        raise ValueError("duplicate source pad option")
    options.update(kwargs)
    value, pad, mode = _tensor(args[0]), args[1], options.get("mode", "constant")
    if mode != "constant":
        raise ValueError(f"pad mode {mode!r} is not implemented by the reference; only 'constant' is")
    rank = value.raw.ndim
    if (not isinstance(pad, (list, tuple)) or any(type(x) is not int for x in pad)
            or len(pad) % 2 or len(pad) > 2 * rank):
        raise ValueError("pad must be an even-length integer list covering at most every source axis")
    crop, widths = [slice(None)] * rank, [(0, 0)] * rank
    for j in range(len(pad) // 2):
        axis, (lo, hi) = rank - 1 - j, pad[2 * j:2 * j + 2]
        size = value.raw.shape[axis]
        if size + min(lo, 0) + min(hi, 0) < 0:
            raise ValueError("negative pads remove more than the source axis holds")
        crop[axis] = slice(-min(lo, 0), size + min(hi, 0))
        widths[axis] = (max(lo, 0), max(hi, 0))
    fill = _fill_raw(options.get("value"), value.dtype)
    return Tensor(np.pad(value.raw[tuple(crop)], widths, constant_values=fill), value.dtype)


def _host_quantize_act(args, kwargs, rules):
    """Evaluate the authored host quantizer with the shared raw encoding rules."""
    from . import device
    x, mode, reduce, rule, static_exp, code_dtype, tag, rows = device._arguments(
        args, kwargs, ("x", "mode", "reduce", "rule", "static_exp", "code_dtype", "tag", "rows"), ("", 32))
    if (not rules.get("authenticated_placement") or mode not in {"dynamic", "static"}
            or rules.get("selected_routes") != ["source_" + mode + "_quantize"]):
        raise ValueError("host activation quantizer requires its authenticated selected route")
    if not isinstance(x, Tensor) or x.dtype not in {"bf16", "f32"}:
        raise ValueError("activation quantizer requires BF16 or F32 source bytes")
    if code_dtype != "uint8" or not isinstance(tag, str):
        raise ValueError("activation quantizer requires uint8 codes and a string tag")
    if (type(rows) is not int or not 1 <= rows < 1 << 31
            or static_exp is not None and (type(static_exp) is not int or not -(1 << 31) <= static_exp < 1 << 31)):
        raise ValueError("quantizer rows/static_exp must be bounded source integers")
    if not x.raw.size or mode == "dynamic" and not x.raw.ndim:
        raise ValueError("host activation quantizer requires nonempty logical storage")
    values = x.raw if x.dtype == "bf16" else x.raw.view(np.float32)
    codes, exponent = device.quantize_act(values, mode, reduce, rule, static_exp, rows)
    return Tensor(codes, "u8"), Tensor(exponent, "i32")


def _call(target, args, kwargs, target_rules=None):
    from . import host, device, placed_device
    if (target_rules or {}).get("numerical_stages") is not None:
        from . import stages
        return stages.call(target, args, kwargs, target_rules)
    if (target_rules or {}).get("execution") == "device" and placed_device.handles(target):
        return placed_device.call(target, args, kwargs, target_rules)
    if device.handles(target):
        if ((target_rules or {}).get("execution") == "host" and
                ((target_rules or {}).get("authenticated_placement") or "selected_routes" in (target_rules or {}))):
            if target == "atlas.quantize_act.default":
                return _host_quantize_act(args, kwargs, target_rules)
            raise ValueError("no host source adapter for placed Atlas operation " + target)
        return device.call(target, args, kwargs, target_rules)
    if (target_rules or {}).get("execution") == "device" and target not in _HOST_OPS:
        raise ValueError("no placed-device source adapter for " + target)
    if host.handles(target):
        return host.call(target, args, kwargs, target_rules)
    if target == "aten.embedding.default":
        _arity(args, kwargs, 2, 5)
        weight, indices = map(_tensor, args[:2])
        if weight.raw.ndim != 2 or indices.dtype not in {"i32", "i64"}:
            raise ValueError("embedding requires a matrix and integer indices")
        if len(args) > 2 and (type(args[2]) is not int or not -weight.raw.shape[0] <= args[2] < weight.raw.shape[0]):
            raise ValueError("invalid source padding index")
        if any(type(x) is not bool for x in args[3:]):
            raise ValueError("invalid embedding flags")
        if np.any(indices.raw < 0) or np.any(indices.raw >= weight.raw.shape[0]):
            raise ValueError("embedding index is outside unchanged source weight")
        return Tensor(weight.raw[indices.raw], weight.dtype)
    if target == "aten._assert_tensor_metadata.default":
        _arity(args, kwargs, 1, allowed=("dtype", "device", "layout", "size", "stride"))
        value = _tensor(args[0])
        _cpu_options(kwargs)
        if kwargs.get("dtype") is not None and _dtype(kwargs["dtype"]) != value.dtype:
            raise ValueError("source dtype assertion failed")
        if kwargs.get("size") is not None and tuple(kwargs["size"]) != value.raw.shape:
            raise ValueError("source shape assertion failed")
        if kwargs.get("stride") is not None and tuple(kwargs["stride"]) != tuple(x // value.raw.itemsize for x in value.raw.strides):
            raise ValueError("source stride assertion failed")
        return None
    if target in {"aten.to.dtype", "aten.to.dtype_layout"}:
        if target == "aten.to.dtype":
            _arity(args, kwargs, 2, 5, ("non_blocking", "copy", "memory_format"))
            dtype = args[1]
            for key, value in zip(("non_blocking", "copy", "memory_format"), args[2:]):
                if key in kwargs:
                    raise ValueError("duplicate source conversion option")
                kwargs[key] = value
        else:
            _arity(args, kwargs, 1, allowed=("dtype", "device", "layout", "pin_memory", "non_blocking", "copy", "memory_format"))
            dtype = kwargs.get("dtype", _tensor(args[0]).dtype)
        _cpu_options(kwargs)
        if kwargs.get("copy") not in {None, False, True}:
            raise ValueError("invalid conversion copy flag")
        return _convert(args[0], dtype)
    if target == "<built-in function getitem>":
        _arity(args, kwargs, 2)
        if not isinstance(args[0], (tuple, list)) or type(args[1]) is not int:
            raise ValueError("getitem only supports source result containers")
        return args[0][args[1]]
    if target == "aten.pad.default":
        return _pad(args, kwargs)
    value = _tensor(args[0])
    if target in {"aten.unsqueeze.default", "aten.transpose.int", "aten.permute.default", "aten.reshape.default", "aten.view.default", "aten.expand.default"}:
        _arity(args, kwargs, 3 if target == "aten.transpose.int" else 2, allowed=("implicit",) if target == "aten.expand.default" else ())
        if target == "aten.unsqueeze.default":
            raw = np.expand_dims(value.raw, _axis(args[1], value.raw.ndim, insert=True))
        elif target == "aten.transpose.int":
            raw = np.swapaxes(value.raw, _axis(args[1], value.raw.ndim), _axis(args[2], value.raw.ndim))
        elif target == "aten.permute.default":
            dims = tuple(_axis(x, value.raw.ndim) for x in args[1])
            if len(dims) != value.raw.ndim or len(set(dims)) != len(dims):
                raise ValueError("permutation must use every source axis once")
            raw = value.raw.transpose(dims)
        elif target in {"aten.reshape.default", "aten.view.default"}:
            if target == "aten.view.default" and not value.raw.flags.c_contiguous:
                raise ValueError("noncontiguous source view has no admitted stride proof")
            raw = value.raw.reshape(tuple(args[1]))
        else:
            if kwargs.get("implicit") not in {None, False}:
                raise ValueError("implicit expand is unsupported")
            shape = list(args[1])
            extra = len(shape) - value.raw.ndim
            if extra < 0:
                raise ValueError("expanded rank is smaller than source rank")
            for i, size in enumerate(shape):
                if type(size) is not int or size < -1 or (size == -1 and i < extra):
                    raise ValueError("invalid source expansion dimension")
                if size == -1:
                    shape[i] = value.raw.shape[i - extra]
            raw = np.broadcast_to(value.raw, tuple(shape))
        return Tensor(raw, value.dtype)
    if target in {"aten.contiguous.default", "aten.clone.default", "aten.detach.default", "aten.alias.default"}:
        _arity(args, kwargs, 1, allowed=() if target in {"aten.detach.default", "aten.alias.default"} else ("memory_format",))
        _cpu_options(kwargs)
        if target == "aten.contiguous.default" and kwargs.get("memory_format") in {"torch.preserve_format", "preserve_format"}:
            raise ValueError("preserve_format is invalid for contiguous")
        if target in {"aten.detach.default", "aten.alias.default"}:
            return Tensor(value.raw, value.dtype)
        if target == "aten.clone.default" and not value.raw.flags.c_contiguous:
            raise ValueError("noncontiguous clone requires a qualified source stride implementation")
        return Tensor(np.array(value.raw, copy=True, order="C"), value.dtype)
    if target == "aten.slice.Tensor":
        _arity(args, kwargs, 1, 5)
        axis = _axis(args[1] if len(args) > 1 else 0, value.raw.ndim)
        start = args[2] if len(args) > 2 else None
        end = args[3] if len(args) > 3 else None
        step = args[4] if len(args) > 4 else 1
        if type(step) is not int or step <= 0:
            raise ValueError("source slice step must be positive")
        slices = [slice(None)] * value.raw.ndim
        slices[axis] = slice(start, end, step)
        return Tensor(value.raw[tuple(slices)], value.dtype)
    if target == "aten.select.int":
        _arity(args, kwargs, 3)
        axis = _axis(args[1], value.raw.ndim)
        slices = [slice(None)] * value.raw.ndim
        slices[axis] = args[2]
        return Tensor(value.raw[tuple(slices)], value.dtype)
    raise ValueError("source handler is not implemented")


def _resolve(value, env, owners):
    if isinstance(value, Mapping):
        if "value_id" in value or "node_id" in value:
            key = value.get("value_id", value.get("node_id"))
            if "value_id" in value and "node_id" in value and owners.get(value["value_id"]) != value["node_id"]:
                raise ValueError("source value/node reference ownership mismatch")
            if key not in env:
                raise ValueError(f"source dependency is unavailable: {key}")
            return env[key]
        if value.get("kind") in {"dtype", "device", "layout", "memory_format"}:
            return value["value"]
        return {k: _resolve(v, env, owners) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [_resolve(v, env, owners) for v in value]
    return value


def _bind(node, value, env, owners):
    node_id = node["id"]
    if node_id in env:
        raise ValueError("duplicate source node id")
    results = node.get("results", [])
    if len(results) > 1:
        if not isinstance(value, (tuple, list)) or len(value) != len(results):
            raise ValueError("multiple source results require a matching result container")
        for spec, item in zip(results, value):
            if not isinstance(item, Tensor) or item.dtype != _dtype(spec["dtype"]) or item.raw.shape != tuple(spec["shape"]):
                raise ValueError("evaluated result container differs from source dtype/shape")
    if isinstance(value, Tensor) and results and node.get("op") != "output":
        spec = results[0]
        if spec.get("dtype") is None or spec.get("shape") is None:
            raise ValueError("tensor result lacks declared source dtype or shape")
        if value.dtype != _dtype(spec["dtype"]) or value.raw.shape != tuple(spec["shape"]):
            raise ValueError("evaluated tensor differs from source result dtype/shape")
        if spec.get("device") not in {None, "cpu"}:
            raise ValueError("non-CPU source result is unsupported")
    env[node_id] = value
    for index, spec in enumerate(results):
        key = spec["id"]
        if key in env:
            raise ValueError("duplicate source value id")
        owners[key] = node_id
        env[key] = value[index] if len(results) > 1 else value


def _input(node, constants, inputs):
    source = inputs if node.get("input_kind") == "USER_INPUT" else constants
    candidates = [node.get("input_target"), node.get("target"), node.get("id")]
    key = next((key for key in candidates if key is not None and key in source), None)
    if key is None:
        raise ValueError("missing unchanged source input binding")
    incoming = source[key]
    if isinstance(incoming, Tensor):
        return incoming
    results = node.get("results", [])
    if len(results) != 1 or results[0].get("dtype") is None:
        raise ValueError("input requires one declared tensor result")
    return Tensor(incoming, results[0]["dtype"])


def evaluate(nodes: Sequence[Mapping], constants: Mapping, inputs: Mapping,
             target_rules: Mapping | None = None, *, observer=None, decisions=None) -> EvaluationResult:
    """Evaluate in source list order, or raise at the first unsupported node.

    Bindings may use ``input_target``, placeholder ``target``, or node id.
    The source metadata controls the logical dtype; unsigned BF16/F32 inputs
    remain raw bits. ``target_rules`` declares source arithmetic order, host-math
    supplier, and selected device unit independently of compiler outputs.
    """
    if not isinstance(constants, Mapping) or not isinstance(inputs, Mapping):
        raise TypeError("constants and inputs must be source binding mappings")
    env, owners, completed = {}, {}, []
    output = None
    seen_output = False
    buffers, single_reader = _Buffers(), _single_reader(nodes)
    producers = {node.get("id"): node for node in nodes}
    for node in nodes:
        try:
            if seen_output:
                raise ValueError("source node appears after output")
            reason = _unsupported(node)
            if reason:
                raise ValueError(reason)
            op = node["op"]
            if op == "placeholder":
                value = _input(node, constants, inputs)
            else:
                args = _resolve(node.get("args", []), env, owners)
                kwargs = _resolve(node.get("kwargs", {}), env, owners)
                if node.get("target") == "<built-in function getitem>" and node.get("args"):
                    source = node["args"][0]
                    if isinstance(source, Mapping) and "node_id" in source:
                        container = env[source["node_id"]]
                        if isinstance(container, (tuple, list)):
                            args[0] = container
                if node.get("target") not in _VIEW_OPS | _BUFFER_OPS:
                    buffers.read([args, kwargs])
                if op == "output":
                    _arity(args, kwargs, 1)
                    value = output = args[0]
                    seen_output = True
                elif node["target"] == "aten.empty_like.default":
                    value = buffers.empty_like(args, kwargs)
                elif node["target"] == "aten.copy_.default":
                    value = buffers.copy_(args, kwargs)
                elif _out_of_place(node["target"]):
                    # In place on a value nothing else reads equals out of place under this node's own placement.
                    self_ref = node.get("args", [None])[0]
                    producer = producers.get(self_ref.get("node_id")) if isinstance(self_ref, Mapping) else None
                    if (producer is None or producer.get("op") != "call_function" or self_ref["node_id"] not in single_reader
                            or producer.get("target") in _HOST_OPS | _VIEW_OPS | _BUFFER_OPS):
                        raise ValueError("in-place source op is admitted only on a computed value that no other node reads")
                    rules = decisions.node_rules(node, target_rules or {}) if decisions else target_rules
                    value = _call(_out_of_place(node["target"]), args, kwargs, rules)
                    if not isinstance(value, Tensor) or value.raw.shape != args[0].raw.shape:
                        raise ValueError("in-place result differs from its destination's shape")
                    if value.dtype != args[0].dtype:
                        # torch computes in the promoted dtype, then rounds once into the destination's dtype.
                        if {value.dtype, args[0].dtype} != {"f32", "bf16"}:
                            raise ValueError("in-place result dtype cannot be cast to its destination's")
                        value = _convert(value, args[0].dtype)
                elif node["target"] == "aten.cat.default":
                    _arity(args, kwargs, 1, 2)
                    tensors = [_tensor(x) for x in args[0]]
                    if not tensors or len({x.dtype for x in tensors}) != 1:
                        raise ValueError("cat requires tensors of one logical dtype")
                    axis = _axis(args[1] if len(args) > 1 else 0, tensors[0].raw.ndim)
                    value = Tensor(np.concatenate([x.raw for x in tensors], axis=axis), tensors[0].dtype)
                else:
                    rules = decisions.node_rules(node, target_rules or {}) if decisions else target_rules
                    value = _call(node["target"], args, kwargs, rules)
            _bind(node, value, env, owners)
            completed.append(node["id"])
            if observer is not None:
                observer(node, value)
        except (ValueError, KeyError, TypeError, IndexError, OverflowError) as exc:
            raise EvaluationError(node, str(exc), completed) from exc
    if not seen_output:
        raise EvaluationError({"id": "<end>", "target": "output"}, "source graph has no output node", completed)
    return EvaluationResult(output, env, tuple(completed))
