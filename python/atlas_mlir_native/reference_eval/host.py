"""Generic independent host operators over logical source tensors.

NumPy supplies storage and individual IEEE binary32 arithmetic instructions;
no framework kernel, BLAS, or NumPy transcendental participates. Reductions
and host matmul require an explicit lexicographic binary32 order. Transcendentals
use merlin.runtime.host_math's authenticated fixed ABI and shared policy.
"""
from __future__ import annotations

import importlib.util
import json
from collections.abc import Mapping
from pathlib import Path

import numpy as np

from .evaluator import Tensor, _axis, _convert, _cpu_options, _dtype, _tensor
from .source_io import checked_path
from ..export_loader import _safe_path

_CONSTRUCTORS = {"aten.arange.default", "aten.arange.start", "aten.arange.start_step",
                 "aten.ones.default", "aten.zeros.default", "aten.full.default",
                 "aten.ones_like.default", "aten.zeros_like.default", "aten.full_like.default"}
_BINARY = {f"aten.{op}.{overload}": op for op in
           ("add", "sub", "mul", "div", "eq", "ne", "lt", "le", "gt", "ge")
           for overload in ("Tensor", "Scalar")}
_BINARY.update({"aten.maximum.default": "maximum", "aten.minimum.default": "minimum",
                "aten.pow.Tensor_Scalar": "pow", "aten.pow.Tensor_Tensor": "pow"})
_UNARY = {"aten.neg.default": "neg", "aten.abs.default": "abs",
          "aten.relu.default": "relu", "aten.square.default": "square",
          "aten.logical_not.default": "logical_not"}
_MATH = {f"aten.{op}.default": name for op, name in
         (("sin", "sinf"), ("cos", "cosf"), ("exp", "expf"), ("log", "logf"),
          ("rsqrt", "rsqrtf"), ("sqrt", "sqrtf"), ("tanh", "tanhf"), ("erf", "erff"))}
_REDUCTIONS = {"aten.sum.default", "aten.sum.dim_IntList", "aten.mean.default",
               "aten.mean.dim", "aten.amax.default", "aten.amin.default"}
_COMPOSITES = {"aten.softmax.int", "aten._softmax.default", "aten.silu.default", "aten.sigmoid.default",
               "aten.gelu.default", "aten.rms_norm.default", "aten.layer_norm.default",
               "aten.native_layer_norm.default"}
_MATMUL = {"aten.matmul.default", "aten.mm.default", "aten.bmm.default", "aten.linear.default"}
_CONVOLUTIONS = {"aten.conv2d.default", "aten.conv2d.padding"}
_OTHER = {"aten.where.self", "aten.masked_fill.Scalar", "aten.clamp.default",
          "aten.triu.default", "aten.tril.default"}

_EXTRA = {"aten.rsub.Scalar", "aten.reciprocal.default", "aten.flatten.using_ints",
          "aten.bucketize.Tensor", "aten.new_ones.default", "aten.__and__.Tensor",
          "aten.lift_fresh_copy.default", "aten.detach_.default", "aten.where.ScalarOther",
          "aten.expand_as.default", "aten.split.Tensor", "aten.min.dim",
          "aten.broadcast_tensors.default", "aten.im2col.default", "aten.cumsum.default",
          "aten.to.device", "aten.max_pool2d.default", "aten.batch_norm.default", "aten.dropout.default",
          "aten.index.Tensor"}


def handles(target):
    return target in (_CONSTRUCTORS | _BINARY.keys() | _UNARY.keys() | _MATH.keys()
                      | _REDUCTIONS | _COMPOSITES | _MATMUL | _CONVOLUTIONS | _OTHER | _EXTRA)


def _signature(args, kwargs, lo, hi=None, allowed=()):
    if not lo <= len(args) <= (lo if hi is None else hi) or set(kwargs) - set(allowed):
        raise ValueError("unsupported host source argument signature")


def _numbers(value):
    value = _tensor(value)
    if value.dtype == "bf16":
        return (value.raw.astype(np.uint32) << np.uint32(16)).view(np.float32)
    return value.raw.view(np.float32) if value.dtype == "f32" else value.raw


def _pack(value, dtype):
    dtype = _dtype(dtype)
    if dtype in {"f32", "bf16"}:
        result = Tensor(np.asarray(value, dtype=np.float32), "f32")
        return _convert(result, "bf16") if dtype == "bf16" else result
    return Tensor(np.asarray(value, dtype={"i64": np.int64, "i32": np.int32,
                                         "u8": np.uint8, "bool": np.bool_}[dtype]), dtype)


def _result_dtype(*values):
    types = {x.dtype for x in values if isinstance(x, Tensor)}
    if not types:
        raise ValueError("host arithmetic requires a logical Tensor operand")
    if len(types) == 1:
        return next(iter(types))
    if types <= {"bf16", "f32"}:
        return "f32"
    raise ValueError("mixed host tensor dtype promotion is not qualified")


def _operand(value, dtype):
    if isinstance(value, Tensor):
        return _numbers(value)
    if not isinstance(value, (int, float, bool)):
        raise ValueError("host arithmetic operand must be a tensor or source scalar")
    if dtype in {"f32", "bf16"}:
        return np.float32(value)
    if isinstance(value, float):
        raise ValueError("floating scalar promotion of an integer tensor is not qualified")
    return value


def _bf16_add(a, b, subtract=False):
    """Exact integer BF16 sum with a single RNE boundary rounding."""
    a, b = np.broadcast_arrays(a, b)
    result = np.empty(a.shape, np.uint16)
    for index in np.ndindex(a.shape):
        left, right = int(a[index]), int(b[index]) ^ (0x8000 if subtract else 0)
        if (left & 0x7fff) > 0x7f80 or (right & 0x7fff) > 0x7f80:
            raise ValueError("source BF16 add/subtract NaN is unqualified")
        if (left & 0x7fff) == 0x7f80 or (right & 0x7fff) == 0x7f80:
            if (left & 0x7fff) == (right & 0x7fff) == 0x7f80 and (left ^ right) & 0x8000:
                raise ValueError("source BF16 infinity cancellation is unqualified")
            result[index] = left if (left & 0x7fff) == 0x7f80 else right
            continue
        def integer(code):
            exponent, mantissa = (code >> 7) & 255, code & 127
            number = mantissa if exponent == 0 else (128 + mantissa) << (exponent - 1)
            return -number if code & 0x8000 else number
        total = integer(left) + integer(right)
        if total == 0:
            result[index] = (left & right) & 0x8000 if not (left & 0x7fff or right & 0x7fff) else 0
            continue
        sign, magnitude = (0x8000 if total < 0 else 0), abs(total)
        shift = max(0, magnitude.bit_length() - 8)
        rounded, remainder = divmod(magnitude, 1 << shift)
        if shift and (remainder > 1 << (shift - 1) or remainder == 1 << (shift - 1) and rounded & 1):
            rounded += 1
        if rounded == 256:
            rounded, shift = 128, shift + 1
        exponent = shift + 1
        result[index] = sign | (rounded if rounded < 128 else 0x7f80 if exponent >= 255 else exponent << 7 | (rounded - 128))
    return Tensor(result, "bf16")


TORCH_F32_SUM = "torch-2.11-cpu-f32-sum-v1"


def _cascade(rows):
    """The wheel's scalar row sum: four interleaved partials, each a four-level cascade, then a left fold."""
    length = rows.shape[1]
    size = length // 4
    power = max(4, (0 if size <= 1 else int(size - 1).bit_length()) // 4)
    step, mask = 1 << power, (1 << power) - 1
    levels = np.zeros((4, rows.shape[0], 4), np.float32)
    index = 0
    while index + step <= size:
        for _ in range(step):
            levels[0] = np.add(levels[0], rows[:, index * 4:index * 4 + 4])
            index += 1
        for level in range(1, 4):
            levels[level] = np.add(levels[level], levels[level - 1])
            levels[level - 1] = 0
            if index & (mask << (level * power)):
                break
    for at in range(index, size):
        levels[0] = np.add(levels[0], rows[:, at * 4:at * 4 + 4])
    for level in range(1, 4):
        levels[level] = np.add(levels[level], levels[level - 1])
    partial = levels[3].copy()
    for column in range(size * 4, length):
        partial[:, 0] = np.add(partial[:, 0], rows[:, column])
    total = partial[:, 0]
    for lane in range(1, 4):
        total = np.add(total, partial[:, lane])
    return total


def _torch_f32_sum(numbers):
    """Sum the last axis in the order of the pinned wheel's kernel, for every leading shape."""
    outer, length = numbers.shape[:-1], numbers.shape[-1]
    rows = numbers.reshape(-1, length)
    count = length // 8
    if not count:
        return _cascade(rows).reshape(outer)
    lanes = [_cascade(rows[:, lane:count * 8:8]) for lane in range(8)]
    if length % 8:
        tail = rows[:, count * 8]
        for column in range(count * 8 + 1, length):
            tail = np.add(tail, rows[:, column])
        lanes[0] = np.add(lanes[0], tail)
    total = lanes[0]
    for lane in lanes[1:]:
        total = np.add(total, lane)
    return total.reshape(outer)


def _sleef_expf(x):
    """SLEEF expf at 1.0 ULP with fused steps, as the wheel's vector kernels call it."""
    x = np.asarray(x, np.float32)
    d = np.clip(x, np.float32(-104), np.float32(100))
    q = np.rint(np.multiply(d, np.float32(1.442695040888963407359924681001892137426645954152985934135449406931)))
    s = _fma32(q, np.float32(-0.693145751953125), d)
    s = _fma32(q, np.float32(-1.428606765330187045e-06), s)
    u = np.full(x.shape, np.float32(0.000198527617612853646278381), np.float32)
    for c in (0.00139304355252534151077271, 0.00833336077630519866943359, 0.0416664853692054748535156,
              0.166666671633720397949219, 0.5):
        u = _fma32(u, s, np.float32(c))
    u = np.add(np.float32(1), _fma32(np.multiply(s, s), u, s))
    n = q.astype(np.int32)
    half = n >> 1
    for part in (half, n - half):
        u = np.multiply(u, ((part + 127).astype(np.uint32) << 23).view(np.float32))
    u = np.where(x < -104, np.float32(0), u)
    return np.where(x > 100, np.float32(np.inf), u)


def _ordered(rules, target):
    order = (rules or {}).get("host_order")
    if isinstance(order, Mapping):
        order = order.get(target, order.get("default"))
    if order not in {"lexicographic-f32", TORCH_F32_SUM} or (order == TORCH_F32_SUM and target not in _REDUCTIONS):
        raise ValueError("host reduction/matmul requires declared host_order='lexicographic-f32'")
    return order


def _host_math(function, raw, rules):
    config = (rules or {}).get("host_math")
    if not isinstance(config, Mapping) or not {"binding_path", "build_dir"} <= config.keys():
        raise ValueError(f"{function} requires the merlin.runtime.host_math binding_path/build_dir and shared policy")
    def safe(path):
        return checked_path(_safe_path(Path(path)))
    binding = safe(config["binding_path"])
    build = safe(config["build_dir"])
    policy = safe(config.get("policy_path", binding.parent / "host-unary-math-policy.json"))
    # The supplier accesses these descendants itself. Admit every path first,
    # rejecting symlinks before loading any policy, sources or shared object.
    supplier = json.loads(policy.read_text())
    for path in (binding.parent / "host_math.h", binding.parent / "third_party/core-math/REVISION",
                 build / "build-receipt.json", build / "libmerlin-host-math.so"):
        safe(path)
    for entry in supplier.get("functions", {}).values():
        safe(binding.parent / "third_party/core-math" / entry["source"])
    spec = importlib.util.spec_from_file_location("atlas_reference_host_math", binding)
    if spec is None or spec.loader is None:
        raise ValueError("merlin.runtime.host_math Python binding is unavailable")
    module = importlib.util.module_from_spec(spec)
    # Loading a supplier must not write a __pycache__ there.
    exec(compile(binding.read_bytes(), str(binding), "exec"), module.__dict__)
    # Admission precedes loading/calling a shared library; missing host-math functions
    # therefore stop with their exact names rather than reaching any fallback.
    module.load_policy(policy, required=[function])
    engine = module.HostMath(build, policy_path=policy, required=[function])
    raw = np.asarray(raw, dtype=np.uint32)
    result = np.empty(raw.shape, np.uint32)
    for index in np.ndindex(raw.shape):
        result[index] = engine.call_bits(function, int(raw[index]))
    return result.view(np.float32)


def _unary_math(function, value, rules):
    value = _tensor(value)
    if value.dtype not in {"f32", "bf16"}:
        raise ValueError("host transcendental requires F32 or BF16 input")
    return _pack(_host_math(function, _numbers(value).view(np.uint32), rules), value.dtype)


def _constructor(target, args, kwargs):
    allowed = ("dtype", "device", "layout", "pin_memory", "memory_format")
    _cpu_options(kwargs)
    like = "_like." in target
    fill = 1 if "ones" in target else 0
    if "arange" in target:
        count = {"aten.arange.default": 1, "aten.arange.start": 2, "aten.arange.start_step": 3}[target]
        _signature(args, kwargs, count, allowed=allowed)
        start, stop, step = (0, args[0], 1) if count == 1 else (args[0], args[1], args[2] if count == 3 else 1)
        if any(type(x) is not int for x in (start, stop, step)):
            raise ValueError("floating arange scalar generation is not qualified")
        if step == 0 or (stop - start) * step < 0:
            raise ValueError("invalid source arange step/direction")
        dtype = _dtype(kwargs.get("dtype") or "i64")
        return _pack(list(range(start, stop, step)), dtype)
    _signature(args, kwargs, 2 if "full" in target else 1, allowed=allowed)
    if like:
        source = _tensor(args[0])
        shape, dtype = source.raw.shape, _dtype(kwargs.get("dtype") or source.dtype)
    else:
        shape, dtype = tuple(args[0]), _dtype(kwargs.get("dtype") or "f32")
    if any(type(n) is not int or n < 0 for n in shape):
        raise ValueError("constructor shape must contain nonnegative integers")
    if "full" in target:
        fill = args[1]
    if not isinstance(fill, (int, float, bool)):
        raise ValueError("constructor fill must be a source scalar")
    return _pack(np.full(shape, fill), dtype)


def _binary(target, args, kwargs):
    op = _BINARY[target]
    _signature(args, kwargs, 2, allowed=("alpha",) if op in {"add", "sub"} else ())
    dtype = _result_dtype(*args)
    if op in {"add", "sub"} and dtype == "bf16" and all(isinstance(x, Tensor) for x in args) and kwargs.get("alpha", 1) == 1:
        return _bf16_add(args[0].raw, args[1].raw, subtract=op == "sub")
    a, b = (_operand(x, dtype) for x in args)
    if op in {"add", "sub"}:
        alpha = kwargs.get("alpha", 1)
        if not isinstance(alpha, (int, float)):
            raise ValueError("alpha must be a source scalar")
        if alpha != 1:
            b = np.multiply(b, _operand(alpha, dtype), dtype=np.float32 if dtype in {"f32", "bf16"} else None)
    operations = {"add": np.add, "sub": np.subtract, "mul": np.multiply,
                  "div": np.divide, "maximum": np.maximum, "minimum": np.minimum,
                  "eq": np.equal, "ne": np.not_equal, "lt": np.less,
                  "le": np.less_equal, "gt": np.greater, "ge": np.greater_equal}
    if op == "pow":
        if not isinstance(args[1], (int, float)) or args[1] != 2:
            raise ValueError("host pow is qualified only for the source square exponent 2")
        result = np.multiply(a, a, dtype=np.float32 if dtype in {"f32", "bf16"} else None)
    else:
        if op == "div" and dtype not in {"f32", "bf16"}:
            raise ValueError("integer true-division promotion is not qualified")
        with np.errstate(all="ignore"):
            result = operations[op](a, b)
    return _pack(result, "bool" if op in {"eq", "ne", "lt", "le", "gt", "ge"} else dtype)


def _reduce(value, dims, keepdim, kind, rules, target, dtype=None):
    value = _tensor(value)
    if value.dtype not in {"bf16", "f32"}:
        raise ValueError("ordered host reduction requires BF16 or F32 input")
    if dtype is not None and dtype not in {"f32", "bf16"}:
        raise ValueError("floating host reduction to integer dtype is not qualified")
    order = _ordered(rules, target)
    dims = tuple(range(value.raw.ndim)) if dims is None or dims == [] else tuple(dims) if isinstance(dims, (list, tuple)) else (dims,)
    axes = tuple(_axis(x, value.raw.ndim) for x in dims)
    if len(set(axes)) != len(axes):
        raise ValueError("duplicate source reduction dimension")
    # Source dimensions are traversed in increasing logical-axis order.
    axes = tuple(sorted(axes))
    other = tuple(x for x in range(value.raw.ndim) if x not in axes)
    numbers = _numbers(value).transpose(other + axes)
    outer = tuple(value.raw.shape[x] for x in other)
    count = int(np.prod([value.raw.shape[x] for x in axes], dtype=np.int64))
    numbers = numbers.reshape(outer + (count,))
    if kind in {"min", "max"} and count == 0:
        raise ValueError("extrema reduction of empty source dimension")
    result = np.zeros(outer, np.float32)
    if order == TORCH_F32_SUM:
        if value.dtype != "f32" or kind not in {"sum", "mean"} or axes != (value.raw.ndim - 1,) or not outer or not count:
            raise ValueError("the torch f32 sum order is qualified for a nonempty f32 last axis")
        if int(np.prod(outer, dtype=np.int64)) == 1 and count >= 1 << 15:
            raise ValueError("the declared f32 sum order refuses one row of 32,768 or more elements")
        # The source tensor's own strides, not the compiler's declaration, decide whether eager uses this order.
        if count > 1 and value.raw.strides[-1] != value.raw.itemsize:
            raise ValueError("the torch f32 sum order matches eager only for an operand contiguous along the reduced axis")
        if not np.isfinite(numbers).all():
            raise ValueError("the torch f32 sum order is qualified for finite inputs")
        result = _torch_f32_sum(numbers)
    else:
        if kind in {"min", "max"}:
            result = numbers[..., 0].copy()
        for i in range(1 if kind in {"min", "max"} else 0, count):
            result = ({"min": np.minimum, "max": np.maximum}.get(kind, np.add))(result, numbers[..., i])
    if kind == "mean":
        with np.errstate(all="ignore"):
            result = np.divide(result, np.float32(count))
    if keepdim:
        result = result.reshape(tuple(1 if x in axes else value.raw.shape[x] for x in range(value.raw.ndim)))
    return _pack(result, dtype or value.dtype)


def _matmul(target, args, kwargs, rules):
    _signature(args, kwargs, 2, 3 if target == "aten.linear.default" else 2)
    a, b = map(_tensor, args[:2])
    if a.dtype != b.dtype or a.dtype not in {"f32", "bf16"}:
        raise ValueError("host matmul requires matching F32 or BF16 source operands")
    left, right = _numbers(a), _numbers(b)
    if target == "aten.linear.default":
        if right.ndim != 2:
            raise ValueError("linear weight must be a matrix")
        right = right.T
    if left.ndim == 0 or right.ndim == 0:
        raise ValueError("matmul operands must have rank at least one")
    if target == "aten.mm.default" and (left.ndim != 2 or right.ndim != 2):
        raise ValueError("mm requires matrices")
    if target == "aten.bmm.default" and (left.ndim != 3 or right.ndim != 3 or left.shape[0] != right.shape[0]):
        raise ValueError("bmm requires equal batch dimensions and rank three")
    av, bv = left.ndim == 1, right.ndim == 1
    if av: left = left[np.newaxis, :]
    if bv: right = right[:, np.newaxis]
    if left.shape[-1] != right.shape[-2]:
        raise ValueError("matmul contracting dimensions differ")
    k_size = left.shape[-1]
    if k_size != 1:
        if (rules or {}).get("execution") != "host":
            raise ValueError("matmul requires an explicitly declared host execution path; device matmul needs the qualified RTL adapter")
        _ordered(rules, target)
    batch = np.broadcast_shapes(left.shape[:-2], right.shape[:-2])
    left = np.broadcast_to(left, batch + left.shape[-2:])
    right = np.broadcast_to(right, batch + right.shape[-2:])
    result = np.zeros(batch + (left.shape[-2], right.shape[-1]), np.float32)
    for k in range(k_size):
        product = np.multiply(left[..., :, k, None], right[..., None, k, :], dtype=np.float32)
        result = product if k_size == 1 else np.add(result, product, dtype=np.float32)
    if av: result = np.squeeze(result, axis=-2)
    if bv: result = np.squeeze(result, axis=-1)
    output = _pack(result, a.dtype)
    if len(args) == 3 and args[2] is not None:
        bias = _tensor(args[2])
        if bias.dtype != a.dtype or bias.raw.shape != (right.shape[-1],):
            raise ValueError("linear bias must be a matching source-dtype vector")
        if a.dtype == "bf16":
            return _bf16_add(output.raw, bias.raw)
        output = _pack(np.add(result, _numbers(bias), dtype=np.float32), "f32")
    return output


def call(target, args, kwargs, target_rules=None):
    """Evaluate one source host op, refusing unavailable math or undeclared order."""
    rules = target_rules or {}
    if not handles(target):
        raise ValueError(f"no independent host handler for {target}")
    kwargs = dict(kwargs)
    if target in _EXTRA:
        return _extra(target, args, kwargs, rules)
    if target in _CONSTRUCTORS:
        return _constructor(target, args, kwargs)
    if target in _BINARY:
        return _binary(target, args, kwargs)
    if target in _MATMUL:
        return _matmul(target, args, kwargs, rules)
    if target in _CONVOLUTIONS:
        return _conv2d(target, args, kwargs, rules)
    if target in _MATH:
        _signature(args, kwargs, 1)
        if target == "aten.rsqrt.default" and _tensor(args[0]).dtype == "f32":
            # The wheel's kernel rounds twice: the IEEE square root, then its reciprocal.
            x = _numbers(_tensor(args[0]))
            if not np.isfinite(x).all() or (x < 0).any():
                raise ValueError("source f32 rsqrt is qualified for finite nonnegative inputs")
            with np.errstate(divide="ignore"):
                return _pack(np.divide(np.float32(1), np.sqrt(x)), "f32")
        return _unary_math(_MATH[target], args[0], rules)
    if target in _UNARY:
        _signature(args, kwargs, 1)
        value, op = _tensor(args[0]), _UNARY[target]
        if op in {"neg", "abs"} and value.dtype in {"f32", "bf16"}:
            sign = np.array(0x80000000 if value.dtype == "f32" else 0x8000, dtype=value.raw.dtype)
            return Tensor(value.raw ^ sign if op == "neg" else value.raw & ~sign, value.dtype)
        numbers = _numbers(value)
        if op == "relu":
            # A source ReLU retains -0 and NaN payloads; maximum(x, +0)
            # would select +0 on ties and silently change source bits.
            return Tensor(np.where(numbers < 0, np.zeros((), value.raw.dtype), value.raw), value.dtype)
        result = {"neg": np.negative, "abs": np.abs, "relu": lambda x: np.maximum(x, 0),
                  "square": lambda x: np.multiply(x, x), "logical_not": np.logical_not}[op](numbers)
        return _pack(result, "bool" if op == "logical_not" else value.dtype)
    if target in _REDUCTIONS:
        _signature(args, kwargs, 1, 3, ("dtype", "dim", "keepdim"))
        dims = args[1] if len(args) > 1 else kwargs.get("dim")
        keep = args[2] if len(args) > 2 else kwargs.get("keepdim", False)
        if type(keep) is not bool:
            raise ValueError("reduction keepdim must be Boolean")
        kind = "mean" if "mean" in target else "max" if "amax" in target else "min" if "amin" in target else "sum"
        return _reduce(args[0], dims, keep, kind, rules, target, _dtype(kwargs["dtype"]) if kwargs.get("dtype") else None)
    if target in _COMPOSITES:
        return _composite(target, args, kwargs, rules)
    if target == "aten.where.self":
        _signature(args, kwargs, 3)
        condition = _tensor(args[0])
        if condition.dtype != "bool": raise ValueError("where condition must be Boolean")
        dtype = _result_dtype(*args[1:])
        return _pack(np.where(condition.raw, *[_operand(x, dtype) for x in args[1:]]), dtype)
    if target == "aten.masked_fill.Scalar":
        _signature(args, kwargs, 3)
        value, mask = map(_tensor, args[:2])
        if mask.dtype != "bool": raise ValueError("masked_fill mask must be Boolean")
        return _pack(np.where(mask.raw, _operand(args[2], value.dtype), _numbers(value)), value.dtype)
    if target == "aten.clamp.default":
        _signature(args, kwargs, 1, 3, ("min", "max"))
        value = _tensor(args[0]); result = _numbers(value)
        lo = args[1] if len(args) > 1 else kwargs.get("min")
        hi = args[2] if len(args) > 2 else kwargs.get("max")
        if lo is None and hi is None: raise ValueError("clamp requires a bound")
        if lo is not None: result = np.maximum(result, _operand(lo, value.dtype))
        if hi is not None: result = np.minimum(result, _operand(hi, value.dtype))
        return _pack(result, value.dtype)
    _signature(args, kwargs, 1, 2, ("diagonal",))
    value = _tensor(args[0]); diagonal = args[1] if len(args) > 1 else kwargs.get("diagonal", 0)
    if value.raw.ndim < 2 or type(diagonal) is not int: raise ValueError("triangular mask requires rank at least two and integer diagonal")
    return Tensor((np.triu if "triu" in target else np.tril)(value.raw, diagonal), value.dtype)


def _index(args, kwargs):
    """torch advanced indexing with integer index tensors; None keeps a dimension whole. torch and numpy share the
    rules: indices broadcast together, adjacent indexed dimensions stay in place, separated ones move to the front."""
    _signature(args, kwargs, 2)
    value, indices = _tensor(args[0]), args[1]
    if not isinstance(indices, (list, tuple)) or not indices or len(indices) > value.raw.ndim:
        raise ValueError("index requires one entry per leading source dimension")
    keys = []
    for axis, entry in enumerate(indices):
        if entry is None:
            keys.append(slice(None))
            continue
        entry = _tensor(entry)
        if entry.dtype not in {"i64", "i32"}:
            raise ValueError("index is qualified only for integer index tensors")
        extent = value.raw.shape[axis]
        if np.any(entry.raw < -extent) or np.any(entry.raw >= extent):
            raise ValueError("index is outside the source dimension")
        keys.append(entry.raw)
    if all(isinstance(key, slice) for key in keys):
        raise ValueError("index requires at least one index tensor")
    # A copy, not ascontiguousarray, which would promote a 0-dimensional result to rank one.
    return Tensor(np.asarray(value.raw[tuple(keys)]).copy(order="C"), value.dtype)


def _extra(target, args, kwargs, rules):
    if target == "aten.index.Tensor":
        return _index(args, kwargs)
    if target == "aten.rsub.Scalar":
        _signature(args, kwargs, 2, allowed=("alpha",))
        value = _tensor(args[0])
        if value.dtype not in {"f32", "bf16"}:
            raise ValueError("rsub is qualified only for F32 or BF16 source input")
        alpha = kwargs.get("alpha", 1)
        if not isinstance(alpha, (int, float)):
            raise ValueError("alpha must be a source scalar")
        scaled = np.multiply(_numbers(value), _operand(alpha, value.dtype))
        return _pack(np.subtract(_operand(args[1], value.dtype), scaled), value.dtype)
    if target == "aten.reciprocal.default":
        _signature(args, kwargs, 1)
        value = _tensor(args[0])
        if value.dtype not in {"f32", "bf16"}:
            raise ValueError("integer reciprocal promotion is not qualified")
        return _binary("aten.div.Tensor", [1, value], {})
    if target == "aten.where.ScalarOther":
        _signature(args, kwargs, 3)
        return call("aten.where.self", args, kwargs, rules)
    if target == "aten.new_ones.default":
        _signature(args, kwargs, 2, allowed=("dtype", "device", "layout", "pin_memory"))
        value = _tensor(args[0])
        return _constructor("aten.ones.default", [args[1]], {**kwargs, "dtype": kwargs.get("dtype") or value.dtype})
    if target == "aten.__and__.Tensor":
        _signature(args, kwargs, 2)
        left, right = map(_tensor, args)
        if left.dtype != right.dtype or left.dtype not in {"bool", "i64", "i32", "u8"}:
            raise ValueError("bitwise and requires matching integer or Boolean tensors")
        return Tensor(np.bitwise_and(left.raw, right.raw), left.dtype)
    if target == "aten.broadcast_tensors.default":
        _signature(args, kwargs, 1)
        tensors = list(map(_tensor, args[0]))
        if not tensors:
            return []
        shape = np.broadcast_shapes(*(t.raw.shape for t in tensors))
        return [Tensor(np.broadcast_to(t.raw, shape), t.dtype) for t in tensors]
    if target == "aten.bucketize.Tensor":
        _signature(args, kwargs, 2, allowed=("out_int32", "right"))
        value, bounds = map(_tensor, args)
        if value.dtype != bounds.dtype or bounds.raw.ndim != 1:
            raise ValueError("bucketize requires matching dtypes and vector boundaries")
        out32, right = kwargs.get("out_int32", False), kwargs.get("right", False)
        if type(out32) is not bool or type(right) is not bool:
            raise ValueError("bucketize flags must be Boolean")
        b, x = _numbers(bounds), _numbers(value)
        if np.any(b[1:] <= b[:-1]) or np.any(np.isnan(b)) or np.any(np.isnan(x)):
            raise ValueError("bucketize requires strictly increasing, non-NaN boundaries and non-NaN values")
        return _pack(np.searchsorted(b, x, side="right" if right else "left"), "i32" if out32 else "i64")
    if target == "aten.to.device":
        _signature(args, kwargs, 3, 6, ("non_blocking", "copy", "memory_format"))
        options = dict(zip(("non_blocking", "copy", "memory_format"), args[3:]))
        if options.keys() & kwargs.keys():
            raise ValueError("duplicate source conversion option")
        options.update(kwargs)
        _cpu_options({"device": args[1], **options})
        if type(options.get("copy", False)) is not bool:
            raise ValueError("copy must be Boolean")
        value = _tensor(args[0])
        return _convert(value, args[2] or value.dtype)
    if target == "aten.dropout.default":
        # Inference dropout is the identity; training dropout draws random masks.
        _signature(args, kwargs, 3)
        if args[2] is not False:
            raise ValueError("training dropout is unqualified")
        value = _tensor(args[0])
        return Tensor(value.raw.copy(), value.dtype)
    if target == "aten.max_pool2d.default":
        return _max_pool2d(args, kwargs)
    if target == "aten.batch_norm.default":
        return _batch_norm(args, kwargs)
    if target == "aten.im2col.default":
        return _im2col(args, kwargs)
    _signature(args, kwargs, 1, {"aten.flatten.using_ints": 3, "aten.expand_as.default": 2,
                                "aten.split.Tensor": 3, "aten.min.dim": 3,
                                "aten.cumsum.default": 2}.get(target, 1),
               ("dtype",) if target == "aten.cumsum.default" else ())
    value = _tensor(args[0])
    if target in {"aten.lift_fresh_copy.default", "aten.detach_.default"}:
        return Tensor(value.raw.copy(), value.dtype)
    if target == "aten.flatten.using_ints":
        rank = value.raw.ndim
        start, end = args[1] if len(args) > 1 else 0, args[2] if len(args) > 2 else -1
        if rank == 0:
            if start not in {0, -1} or end not in {0, -1}:
                raise ValueError("flatten scalar axis is outside source rank")
            return Tensor(value.raw.reshape(1), value.dtype)
        start, end = _axis(start, rank), _axis(end, rank)
        if start > end:
            raise ValueError("flatten start dimension follows end dimension")
        shape = value.raw.shape
        return Tensor(value.raw.reshape(shape[:start] + (int(np.prod(shape[start:end+1])),) + shape[end+1:]), value.dtype)
    if target == "aten.expand_as.default":
        if len(args) != 2:
            raise ValueError("expand_as requires another tensor")
        return Tensor(np.broadcast_to(value.raw, _tensor(args[1]).raw.shape), value.dtype)
    if target == "aten.split.Tensor":
        if len(args) < 2 or type(args[1]) is not int or args[1] <= 0:
            raise ValueError("split requires a positive integer split size")
        dim = _axis(args[2] if len(args) > 2 else 0, value.raw.ndim)
        return tuple(Tensor(x, value.dtype) for x in np.split(value.raw, range(args[1], value.raw.shape[dim], args[1]), axis=dim))
    if target == "aten.min.dim":
        if len(args) < 2:
            raise ValueError("min.dim requires a dimension")
        dim = _axis(args[1], value.raw.ndim)
        keep = args[2] if len(args) > 2 else False
        if type(keep) is not bool or value.raw.shape[dim] == 0:
            raise ValueError("min.dim requires Boolean keepdim and nonempty dimension")
        numbers = _numbers(value)
        if np.any(np.isnan(numbers)):
            raise ValueError("min.dim NaN selection is not qualified")
        index = np.argmin(numbers, axis=dim, keepdims=True)
        output = np.take_along_axis(value.raw, index, axis=dim)
        if not keep:
            index, output = np.squeeze(index, axis=dim), np.squeeze(output, axis=dim)
        return Tensor(output, value.dtype), _pack(index, "i64")
    if target == "aten.cumsum.default":
        if len(args) != 2:
            raise ValueError("cumsum requires a dimension")
        dim = _axis(args[1], value.raw.ndim)
        dtype = _dtype(kwargs.get("dtype") or ("i64" if value.dtype in {"bool", "u8", "i32"} else value.dtype))
        if dtype not in {"f32", "bf16", "i64", "i32"}:
            raise ValueError("unsupported cumsum dtype")
        if value.dtype in {"f32", "bf16"}:
            if dtype not in {"f32", "bf16"}:
                raise ValueError("floating cumsum to integer is not qualified")
        if dtype in {"f32", "bf16"}:
            _ordered(rules, target)
        numbers = np.moveaxis(_numbers(value), dim, -1)
        storage = np.float32 if dtype in {"f32", "bf16"} else np.int64 if dtype == "i64" else np.int32
        result = np.empty(numbers.shape, storage)
        total = np.zeros(numbers.shape[:-1], storage)
        for i in range(numbers.shape[-1]):
            total = np.add(total, numbers[..., i], dtype=storage)
            result[..., i] = total
        return _pack(np.moveaxis(result, -1, dim), dtype)
    raise ValueError(f"unimplemented extra host handler {target}")


def _binary32(value, zero_negative=False):
    """Round an exact rational to binary32 with ties to even; return the encoding."""
    from fractions import Fraction
    if value == 0:
        return 0x80000000 if zero_negative else 0
    sign, value = (0x80000000, -value) if value < 0 else (0, value)
    exponent = value.numerator.bit_length() - value.denominator.bit_length()
    if Fraction(2) ** exponent > value:
        exponent -= 1
    quantum = max(exponent, -126) - 23
    scaled = value / Fraction(2) ** quantum
    significand, remainder = divmod(scaled.numerator, scaled.denominator)
    if 2 * remainder > scaled.denominator or 2 * remainder == scaled.denominator and significand & 1:
        significand += 1
    if significand == 1 << 24:
        significand, quantum = significand >> 1, quantum + 1
    if significand < 1 << 23:
        return sign | significand
    if quantum + 150 >= 255:
        return sign | 0x7f800000
    return sign | (quantum + 150) << 23 | (significand - (1 << 23))


def _rational(bits, width=32):
    from fractions import Fraction
    mantissa_bits = 23 if width == 32 else 7
    exponent, mantissa = (bits >> mantissa_bits) & 255, bits & ((1 << mantissa_bits) - 1)
    if exponent == 255:
        raise ValueError("batch_norm term is not finite")
    significand = mantissa | (1 << mantissa_bits if exponent else 0)
    value = Fraction(significand) * Fraction(2) ** ((exponent or 1) - 127 - mantissa_bits)
    return -value if bits >> (width - 1) else value


def _batch_norm(args, kwargs):
    """PyTorch's CPU inference kernel for BF16 input, each binary32 step rounded once to nearest even.

    alpha = (1 / sqrt(var + eps)) * weight, beta = bias - mean * alpha (fused), y = bf16(x * alpha + beta) (fused).
    Checked bit for bit against torch 2.11.0+cpu on random finite BF16 input with random BF16 or F32 statistics,
    with and without affine terms, at eps 1e-5; not every statistics value or epsilon. Non-finite input is refused.
    """
    from fractions import Fraction
    from math import isqrt
    _signature(args, kwargs, 9)
    value, weight, bias, mean, var, training, _momentum, eps = (_tensor(args[0]), *args[1:8])
    if training is not False or value.dtype != "bf16" or value.raw.ndim < 2 or not isinstance(eps, float):
        raise ValueError("batch_norm is qualified only for BF16 inference input")
    channels = value.raw.shape[1]
    terms = {}
    for name, term in (("weight", weight), ("bias", bias), ("mean", mean), ("var", var)):
        if term is None and name in {"weight", "bias"}:
            continue
        term = _tensor(term)
        if term.dtype not in {"bf16", "f32"} or term.raw.shape != (channels,):
            raise ValueError("batch_norm statistics must be per-channel BF16 or F32")
        width = 16 if term.dtype == "bf16" else 32
        terms[name] = [(_rational(int(bits), width), bool(int(bits) >> (width - 1))) for bits in term.raw]
    raw = value.raw.astype(np.uint32)
    if ((raw & 0x7f80) == 0x7f80).any():
        raise ValueError("batch_norm non-finite input is unqualified")
    epsilon = _rational(_binary32(Fraction(eps)))
    alphas, betas = [], []
    for c in range(channels):
        total = _rational(_binary32(terms["var"][c][0] + epsilon))
        if total <= 0:
            raise ValueError("batch_norm variance plus epsilon must be positive")
        scaled = total * Fraction(2) ** 200
        root = isqrt(scaled.numerator // scaled.denominator)
        exact = root * root * scaled.denominator == scaled.numerator
        inverse = _rational(_binary32(1 / _rational(_binary32(Fraction(4 * root + (0 if exact else 1), 1 << 102)))))
        alpha_bits = _binary32(inverse * terms["weight"][c][0], terms["weight"][c][1]) if "weight" in terms else _binary32(inverse)
        alpha = _rational(alpha_bits)
        product, (addend, addend_negative) = -terms["mean"][c][0] * alpha, terms["bias"][c] if "bias" in terms else (Fraction(0), False)
        product_negative = (not terms["mean"][c][1]) ^ bool(alpha_bits >> 31)
        beta_bits = _binary32(product + addend, product_negative and addend_negative and not product and not addend)
        _rational(beta_bits)
        alphas.append((alpha, bool(alpha_bits >> 31)))
        betas.append((_rational(beta_bits), bool(beta_bits >> 31)))
    result = np.empty(raw.shape, np.uint16)
    for index in np.ndindex(raw.shape):
        bits = int(raw[index])
        (alpha, alpha_negative), (beta, beta_negative) = alphas[index[1]], betas[index[1]]
        x = _rational(bits, 16)
        product = x * alpha
        wide = _binary32(product + beta, (bool(bits >> 15) ^ alpha_negative) and beta_negative and not product and not beta)
        upper, lower = wide >> 16, wide & 0xffff
        if (wide & 0x7f800000) != 0x7f800000 and (lower > 0x8000 or lower == 0x8000 and upper & 1):
            upper += 1
        result[index] = upper
    return Tensor(result, "bf16")


def _max_pool2d(args, kwargs):
    """PyTorch's CPU scan per window in row-major order: take a value only if it is greater.

    Equal values (including -0 and +0) keep the first. Padding reads are -inf and never win.
    NaN input is refused: PyTorch's BF16 kernel returns a canonical NaN, not an input payload.
    ceil_mode is refused.
    """
    _signature(args, kwargs, 2, 6)
    value = _tensor(args[0])
    if value.dtype not in {"bf16", "f32"} or value.raw.ndim != 4:
        raise ValueError("max_pool2d is qualified only for NCHW BF16 or F32 source input")
    def pair(index, default):
        item = args[index] if len(args) > index else default
        item = [item] * 2 if isinstance(item, int) else list(item) or list(default)
        if len(item) == 1:
            item *= 2
        if len(item) != 2 or any(type(x) is not int for x in item):
            raise ValueError("max_pool2d requires integer kernel/stride/padding/dilation pairs")
        return item
    (kh, kw) = pair(1, None)
    (sh, sw), (ph, pw), (dh, dw) = pair(2, (kh, kw)), pair(3, (0, 0)), pair(4, (1, 1))
    if (len(args) > 5 and args[5] is not False) or min(kh, kw, sh, sw, dh, dw) < 1 or min(ph, pw) < 0:
        raise ValueError("max_pool2d ceil_mode or nonpositive geometry is unqualified")
    batch, channels, height, width = value.raw.shape
    oh, ow = (height + 2*ph - dh*(kh-1) - 1)//sh + 1, (width + 2*pw - dw*(kw-1) - 1)//sw + 1
    if oh <= 0 or ow <= 0:
        raise ValueError("max_pool2d output shape must be positive")
    numbers = _numbers(value)
    if np.isnan(numbers).any():
        raise ValueError("max_pool2d NaN input is unqualified")
    result = np.empty((batch, channels, oh, ow), value.raw.dtype)
    for b, c, r, q in np.ndindex(batch, channels, oh, ow):
        best = best_number = None
        for i, j in np.ndindex(kh, kw):
            row, col = r*sh - ph + i*dh, q*sw - pw + j*dw
            if 0 <= row < height and 0 <= col < width:
                number = numbers[b, c, row, col]
                if best is None or number > best_number:
                    best, best_number = value.raw[b, c, row, col], number
        result[b, c, r, q] = (0xff80 if value.dtype == "bf16" else 0xff800000) if best is None else best
    return Tensor(result, value.dtype)


def _im2col(args, kwargs):
    _signature(args, kwargs, 5)
    value = _tensor(args[0])
    if value.raw.ndim not in {3, 4}:
        raise ValueError("im2col requires CHW or NCHW source input")
    pairs = []
    for index, pair in enumerate(args[1:]):
        if not isinstance(pair, (list, tuple)) or len(pair) != 2 or any(type(x) is not int or x < (0 if index == 2 else 1) for x in pair):
            raise ValueError("im2col requires integer kernel/dilation/padding/stride pairs")
        pairs.append(pair)
    (kh, kw), (dh, dw), (ph, pw), (sh, sw) = pairs
    source = value.raw if value.raw.ndim == 4 else value.raw[None]
    batch, channels, height, width = source.shape
    oh, ow = (height + 2*ph - dh*(kh-1) - 1)//sh + 1, (width + 2*pw - dw*(kw-1) - 1)//sw + 1
    if oh <= 0 or ow <= 0:
        raise ValueError("im2col calculated block shape must be positive")
    result = np.zeros((batch, channels, kh, kw, oh, ow), value.raw.dtype)
    for y, x, by, bx in np.ndindex(kh, kw, oh, ow):
        iy, ix = by*sh - ph + y*dh, bx*sw - pw + x*dw
        if 0 <= iy < height and 0 <= ix < width:
            result[:, :, y, x, by, bx] = source[:, :, iy, ix]
    result = result.reshape(batch, channels*kh*kw, oh*ow)
    return Tensor(result if value.raw.ndim == 4 else result[0], value.dtype)


def _pair(value, name, minimum):
    pair = [value, value] if type(value) is int else list(value) if isinstance(value, (list, tuple)) else None
    if pair is not None and len(pair) == 1:
        pair = pair * 2
    if pair is None or len(pair) != 2 or any(type(x) is not int or x < minimum for x in pair):
        raise ValueError(f"conv2d {name} must be one or two integers of at least {minimum}")
    return pair


def _conv2d(target, args, kwargs, rules):
    """BF16 conv2d in the reference's lexicographic-f32 convention, as its matmul: for each output, binary32 products
    of the BF16 operands, summed from +0 in ascending (input channel, kernel row, kernel column) order with one binary32
    rounding per addition, then the bias added in binary32, then one BF16 RNE rounding. torch leaves the order
    unspecified, so the reference fixes it here, and a compiler must match it (Merlin's declared host BF16 affine
    contraction does). The order must be declared for this target by name, not by a default that admits matmul.
    Only one group, NCHW input, and zero padding (integers for conv2d.default, "valid" for conv2d.padding) qualify."""
    _signature(args, kwargs, 2, 7, ("bias", "stride", "padding", "dilation", "groups"))
    names = ("bias", "stride", "padding", "dilation", "groups")
    options = dict(zip(names, args[2:]))
    if options.keys() & kwargs.keys():
        raise ValueError("duplicate conv2d argument")
    options.update(kwargs)
    value, weight = _tensor(args[0]), _tensor(args[1])
    bias = options.get("bias")
    if value.dtype != "bf16" or weight.dtype != "bf16" or (bias is not None and _tensor(bias).dtype != "bf16"):
        raise ValueError("host conv2d is qualified only for BF16 input, weight, and bias")
    if (rules or {}).get("execution") != "host":
        raise ValueError("conv2d requires an explicitly declared host execution path")
    order = (rules or {}).get("host_order")
    if not isinstance(order, Mapping) or order.get(target) != "lexicographic-f32":
        raise ValueError("conv2d requires host_order to declare 'lexicographic-f32' for " + target)
    groups = options.get("groups", 1)
    if type(groups) is not int or groups != 1:
        raise ValueError("grouped conv2d is not qualified")
    padding = options.get("padding", "valid" if target == "aten.conv2d.padding" else 0)
    if target == "aten.conv2d.padding":
        if padding != "valid":
            raise ValueError("conv2d.padding is qualified only for padding 'valid'")
        padding = 0
    elif isinstance(padding, str):
        raise ValueError("conv2d.default takes integer padding")
    (sh, sw), (ph, pw), (dh, dw) = (_pair(options.get("stride", 1), "stride", 1), _pair(padding, "padding", 0),
                                    _pair(options.get("dilation", 1), "dilation", 1))
    if value.raw.ndim != 4 or weight.raw.ndim != 4 or weight.raw.shape[1] != value.raw.shape[1]:
        raise ValueError("conv2d requires NCHW input and OIHW weight with matching channels")
    out_channels, channels, kh, kw = weight.raw.shape
    columns = _im2col([value, [kh, kw], [dh, dw], [ph, pw], [sh, sw]], {})
    batch, _, height, width = value.raw.shape
    oh, ow = (height + 2*ph - dh*(kh-1) - 1)//sh + 1, (width + 2*pw - dw*(kw-1) - 1)//sw + 1
    x, w = _numbers(columns), _numbers(weight).reshape(out_channels, channels*kh*kw)
    total = np.zeros((batch, out_channels, oh*ow), np.float32)
    for k in range(channels*kh*kw):
        product = np.multiply(w[None, :, k, None], x[:, None, k, :], dtype=np.float32)
        total = np.add(total, product, dtype=np.float32)
    if bias is not None:
        if _tensor(bias).raw.shape != (out_channels,):
            raise ValueError("conv2d bias must be one value per output channel")
        total = np.add(total, _numbers(bias)[None, :, None], dtype=np.float32)
    return _pack(total.reshape(batch, out_channels, oh, ow), "bf16")


_LANES, _CHUNK = 8, 16


def _fma32(a, b, c):
    """Correctly rounded binary32 RN(a*b+c), elementwise.

    The binary64 product of two binary32 values is exact; the binary64 sum can
    then round, and rounding that to binary32 differs from one rounding only
    when it lands exactly on a binary32 midpoint, which is redone in rationals.
    """
    from fractions import Fraction

    a, b, c = (np.asarray(v, np.float32) for v in np.broadcast_arrays(a, b, c))
    wide = a.astype(np.float64) * b.astype(np.float64) + c.astype(np.float64)
    result = np.array(wide.astype(np.float32))
    ties = (np.asarray(wide).view(np.uint64) & np.uint64((1 << 29) - 1)) == np.uint64(1 << 28)
    for i in np.flatnonzero(ties.reshape(-1)):
        exact = Fraction(float(a.flat[i])) * Fraction(float(b.flat[i])) + Fraction(float(c.flat[i]))
        low = np.float32(float(exact))
        options = (low, np.nextafter(low, np.float32(np.inf)), np.nextafter(low, np.float32(-np.inf)))
        result.flat[i] = min(options, key=lambda v: (abs(Fraction(float(v)) - exact), int(np.array(v).view(np.uint32)) & 1))
    return result if result.ndim else np.float32(result)


def _add_moments_vec(add, m1_add, m2_add, m0, m1, m2):
    """torch AddMomentsVec as the pinned wheel dispatches it: rounded m1 update, fused m2 update."""
    n = m0 + add
    c = np.float32(np.float32(add) / np.float32(n)) if n else np.float32(0)
    delta = np.subtract(m1_add, m1)
    m2_tmp = np.add(m2, m2_add)
    c_delta = np.multiply(c, delta)
    m0_delta = np.multiply(delta, np.float32(m0))
    return n, np.add(m1, c_delta), _fma32(m0_delta, c_delta, m2_tmp)


def _torch_bf16_row_moments(x):
    """torch RowwiseMoments for one widened BF16 row: chunked Welford, cascade, lane combination."""
    width = x.size
    vectors = width // (2 * _LANES)
    chunks = -(-vectors // _CHUNK)
    depth = (chunks - 1).bit_length() if chunks > 1 else 0
    zero = np.zeros(_LANES, np.float32)
    stack = [[0, zero.copy(), zero.copy()] for _ in range(max(depth, 1))]
    weights = [np.float32(np.float32(1) / np.float32(j + 1)) for j in range(_CHUNK)]
    for i in range(chunks):
        count = min(_CHUNK, vectors - i * _CHUNK)
        halves = []
        for h in range(2):
            m1, m2 = zero.copy(), zero.copy()
            for j in range(count):
                start = (i * _CHUNK + j) * 2 * _LANES + h * _LANES
                v = x[start:start + _LANES]
                delta = np.subtract(v, m1)
                m1 = _fma32(delta, weights[j], m1)
                m2 = _fma32(delta, np.subtract(v, m1), m2)
            halves.append((m1, m2))
        for m1, m2 in halves:
            stack[0] = list(_add_moments_vec(count, m1, m2, *stack[0]))
        mask, level = i + 1, 1
        while level < depth and not mask & 1:
            stack[level] = list(_add_moments_vec(*stack[level - 1], *stack[level]))
            stack[level - 1] = [0, zero.copy(), zero.copy()]
            mask >>= 1
            level += 1
    for level in range(1, depth):
        stack[0] = list(_add_moments_vec(*stack[level], *stack[0]))
    m0, m1, m2 = 0, np.float32(0), np.float32(0)
    for v in x[vectors * 2 * _LANES:]:
        delta = np.float32(v - m1)
        m0 += 1
        m1 = np.float32(m1 + np.float32(delta / np.float32(m0)))
        m2 = np.float32(m2 + np.float32(delta * np.float32(v - m1)))
    for lane in range(_LANES):  # scalar AddMoments
        add = 2 * vectors
        n = m0 + add
        c = np.float32(np.float32(add) / np.float32(n)) if n else np.float32(0)
        delta = np.float32(stack[0][1][lane] - m1)
        m1 = _fma32(c, delta, m1)
        term = np.float32(np.float32(delta * delta) * c)
        m2 = np.float32(m2 + _fma32(term, np.float32(m0), stack[0][2][lane]))
        m0 = n
    return m1, np.float32(m2 / np.float32(width))


def _torch_bf16_layer_norm(target, value, shape, options):
    """torch eager CPU BF16 layer_norm as the pinned 2.11.0+cpu wheel dispatches it on AVX2 and AVX512 hosts.

    Its DEFAULT capability does not fuse multiply-adds and is not matched.
    """
    eps = options.get("eps", 1e-5)
    rows = _numbers(value).reshape(-1, int(np.prod(shape)))
    affine = []
    for name, identity in (("weight", np.float32(1)), ("bias", np.float32(0))):
        term = options.get(name)
        if term is None:
            affine.append(np.full(rows.shape[1], identity, np.float32))
            continue
        term = _tensor(term)
        if term.raw.shape != shape or term.dtype != value.dtype:
            raise ValueError("normalization affine source dtype/shape differs")
        affine.append(_numbers(term).reshape(-1))
    result = np.empty(rows.shape, np.float32)
    means, inverses = np.empty(rows.shape[0], np.float32), np.empty(rows.shape[0], np.float32)
    for r, row in enumerate(rows):
        mean, variance = _torch_bf16_row_moments(row)
        rstd = np.float32(np.float32(1) / np.sqrt(np.float32(variance + np.float32(eps))))
        offset = np.float32(np.float32(-mean) * rstd)
        result[r] = _fma32(_fma32(row, rstd, offset), affine[0], affine[1])
        means[r], inverses[r] = mean, rstd
    output = _pack(result.reshape(value.raw.shape), "bf16")
    if target != "aten.native_layer_norm.default":
        return output
    keep = value.raw.shape[:value.raw.ndim - len(shape)] + (1,) * len(shape)
    return output, _pack(means.reshape(keep), "bf16"), _pack(inverses.reshape(keep), "bf16")


def _composite(target, args, kwargs, rules):
    if target == "aten.sigmoid.default":
        # PyTorch's CPU BF16 kernel: binary32 1/(1+exp(-x)), one BF16 rounding.
        _signature(args, kwargs, 1)
        value = _tensor(args[0]); x = _numbers(value)
        if value.dtype != "bf16": raise ValueError("source sigmoid is qualified for BF16 only")
        if np.isnan(x).any(): raise ValueError("source BF16 sigmoid NaN input is unqualified")
        exp = _host_math("expf", np.negative(x).view(np.uint32), rules)
        return _pack(np.divide(np.float32(1), np.add(np.float32(1), exp)), value.dtype)
    if target in {"aten.silu.default", "aten.gelu.default"}:
        _signature(args, kwargs, 1, 2 if "gelu" in target else 1, ("approximate",) if "gelu" in target else ())
        value = _tensor(args[0]); x = _numbers(value)
        if value.dtype not in {"f32", "bf16"}: raise ValueError("activation requires floating input")
        if "silu" in target:
            # The pinned wheel's kernel: SLEEF expf in binary32, x/(1+e), one rounding to a BF16 result.
            if not np.isfinite(x).all(): raise ValueError("source SiLU is qualified for finite inputs")
            return _pack(np.divide(x, np.add(np.float32(1), _sleef_expf(np.negative(x)))), value.dtype)
        approximate = args[1] if len(args) > 1 else kwargs.get("approximate", "none")
        if approximate == "none":
            erf = _host_math("erff", np.multiply(x, np.float32(0.7071067811865476)).view(np.uint32), rules)
            result = np.multiply(np.multiply(x, np.float32(0.5)), np.add(np.float32(1), erf))
        elif approximate == "tanh":
            cube = np.multiply(np.multiply(x, x), x)
            inner = np.multiply(np.float32(0.7978845608028654), np.add(x, np.multiply(np.float32(0.044715), cube)))
            tanh = _host_math("tanhf", inner.view(np.uint32), rules)
            result = np.multiply(np.multiply(x, np.float32(0.5)), np.add(np.float32(1), tanh))
        else: raise ValueError("unknown GELU approximation")
        return _pack(result, value.dtype)
    if target in {"aten.softmax.int", "aten._softmax.default"}:
        _signature(args, kwargs, 2, 3, ("dtype",) if target == "aten.softmax.int" else ())
        value = _tensor(args[0]); dtype = value.dtype
        if target == "aten._softmax.default" and len(args) > 2 and args[2] is not False:
            raise ValueError("half_to_float softmax is not qualified")
        if target == "aten.softmax.int":
            requested = args[2] if len(args) > 2 else kwargs.get("dtype")
            if requested is not None: dtype = _dtype(requested)
        value = _pack(_numbers(value), "f32")
        maximum = _reduce(value, [args[1]], True, "max", rules, target)
        shifted = np.subtract(_numbers(value), _numbers(maximum))
        exp = _host_math("expf", shifted.view(np.uint32), rules)
        total = _reduce(_pack(exp, "f32"), [args[1]], True, "sum", rules, target)
        return _pack(np.divide(exp, _numbers(total)), dtype)
    names = ("weight", "eps") if target == "aten.rms_norm.default" else ("weight", "bias", "eps", "cudnn_enable")
    _signature(args, kwargs, 2, 2 + len(names), names)
    options = dict(zip(names, args[2:]))
    if options.keys() & kwargs.keys(): raise ValueError("duplicate normalization source option")
    options.update(kwargs)
    value = _tensor(args[0]); shape = tuple(args[1]); rank = value.raw.ndim
    if not shape or tuple(value.raw.shape[-len(shape):]) != shape: raise ValueError("normalization shape must match trailing source dimensions")
    if value.dtype not in {"f32", "bf16"}: raise ValueError("normalization requires floating input")
    x = _pack(_numbers(value), "f32"); dims = list(range(rank-len(shape), rank))
    rms = target == "aten.rms_norm.default"
    if not rms and value.dtype == "bf16":
        return _torch_bf16_layer_norm(target, value, shape, options)
    weight = options.get("weight")
    bias = None if rms else options.get("bias")
    eps = options.get("eps", None if rms else 1e-5)
    eps = (0.0078125 if value.dtype == "bf16" else 1.1920928955078125e-7) if eps is None else eps
    mean = _pack(np.zeros(tuple(1 if i in dims else n for i,n in enumerate(value.raw.shape)), np.float32), "f32") if rms else _reduce(x, dims, True, "mean", rules, target)
    centered = np.subtract(_numbers(x), _numbers(mean))
    variance = _reduce(_pack(np.multiply(centered, centered), "f32"), dims, True, "mean", rules, target)
    inv = _host_math("rsqrtf", np.add(_numbers(variance), np.float32(eps)).view(np.uint32), rules)
    result = np.multiply(centered, inv)
    for affine, operation in ((weight, np.multiply), (bias, np.add)):
        if affine is not None:
            affine = _tensor(affine)
            if affine.raw.shape != shape or affine.dtype != value.dtype: raise ValueError("normalization affine source dtype/shape differs")
            result = operation(result, _numbers(affine))
    output = _pack(result, value.dtype)
    return (output, mean, _pack(inv, "f32")) if target == "aten.native_layer_norm.default" else output
