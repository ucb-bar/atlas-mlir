"""Aten arithmetic selected for the device by an authenticated placement plan.

These adapters implement selected pointwise, exponential, and reduction
instruction sequences by source operation kind. Plain VPU F32 operands narrow
with RNE and results carry widened BF16 words. Exact-float routes instead
preserve the source operation's single RNE rounding boundary. Reduction tiles use the model's adjacent FP32
tree, BF16 RNE readout and left-to-right, seeded RTL add continuation. Torch is
used here only for model storage conversion and authenticated model API calls.
"""
from __future__ import annotations

from functools import lru_cache
import operator
from pathlib import Path
import subprocess
import sys

import numpy as np

from . import device
from .evaluator import Tensor, _dtype

REDUCTION_POLICY = "bf16-adjacent-fp32-tree-rne-seeded-tile-sum-ltr-chop-v1"
NONNEGATIVE_REDUCTION_POLICY = "bf16-nonnegative-normal-adjacent-fp32-tree-rne-seeded-tile-sum-ltr-chop-v1"
_BINARY = {f"aten.{op}.{overload}": op for op in ("add", "sub", "mul", "div")
           for overload in ("Tensor", "Scalar")}
_UNARY = {f"aten.{op}.default": op for op in
          ("square", "rsqrt", "sqrt", "reciprocal", "exp", "neg", "abs", "relu", "silu")}
_SOFTMAX = {"aten.softmax.int", "aten._softmax.default"}
# Row-softmax policies, score domains, and the mask encoding implemented here, named as
# the compiler declares them per selected region (atlas-mlir row_softmax.py and softmax_c_source.py).
_SOFTMAX_POLICIES = {"source_softmax": "bf16-tiled-row-softmax-ltr-fp32-rne-chop-rcp-lut-v1",
                     "source_masked_softmax": "bf16-tiled-masked-row-softmax-zero-neginf-v1"}
ABSOLUTE32_DOMAIN = "bf16-signed-zero-or-normal-exp8-to132-abs-at-most32-v1"
ROWSPAN80_DOMAIN = "bf16-row-relative-span80-pad32-v1"
_SENTINEL = {"f32": 0xff7f0000, "bf16": 0xff7f}
_REDUCTIONS = {"aten.sum.dim_IntList", "aten.sum.default", "aten.mean.dim", "aten.mean.default"}


def handles(target):
    """Static adapter coverage; placement, rules and domains are checked later."""
    return target in _SOFTMAX or target in _BINARY or target in _UNARY or target in _REDUCTIONS or target == "aten.pow.Tensor_Scalar"


@lru_cache(maxsize=8)
def _pinned_table(root, name):
    return subprocess.check_output(["git", "-C", root, "show",
                                   f"{device.MODEL_REVISION}:npu_model/configs/data/{name}.bin.gz"])


def _rtl(rules, unaries=()):
    device._rules(rules, contraction=False)
    module = sys.modules["npu_model.configs.numerics"]
    root = str(Path(module.__file__).parents[2])
    for name in unaries:
        if name not in {"rcp", "sqrt", "exp", "relu"}:
            raise ValueError("unsupported placed-device unary table")
        path = device._safe_path(Path(root) / f"npu_model/configs/data/{name}.bin.gz")
        if path.read_bytes() != _pinned_table(root, name):
            raise ValueError("device unary table differs from pinned model: " + name)
    return module.NUMERICS["rtl"]


def _narrow(raw):
    raw = np.asarray(raw, dtype=np.uint32)
    if np.any((raw & 0x7fffffff) > 0x7f800000):
        raise ValueError("unqualified source NaN payload in placed-device f32 cast")
    return ((raw + np.uint32(0x7fff) + ((raw >> 16) & 1)) >> 16).astype(np.uint16)


def _operand(value):
    if isinstance(value, Tensor):
        if value.dtype not in {"bf16", "f32"}:
            raise ValueError("placed-device arithmetic requires BF16 or F32 source operands")
        return value.raw if value.dtype == "bf16" else _narrow(value.raw)
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise ValueError("placed-device operand must be a floating tensor or source scalar")
    with np.errstate(over="ignore", invalid="ignore"):
        return _narrow(np.asarray(value, np.float32).view(np.uint32))


def _dtype_of(values):
    types = {value.dtype for value in values if isinstance(value, Tensor)}
    if not types or not types <= {"bf16", "f32"}:
        raise ValueError("placed-device arithmetic requires BF16 or F32 source tensors")
    return "f32" if "f32" in types else "bf16"


def _bounded(bits, *, positive=False, exponential=False):
    magnitude = bits & 0x7fff
    valid = ((magnitude == 0) | ((magnitude >= (0x80 if exponential else 0x2000)) &
                               (magnitude <= (0x4200 if exponential else 0x5f00))))
    if positive:
        valid &= (bits < 0x8000) & (magnitude != 0)
    if not np.all(valid):
        raise ValueError("placed-device operand leaves the selected BF16 code domain")
    return bits


def _normal_result(bits):
    exponent = (bits >> 7) & 255
    if np.any(((bits & 0x7fff) != 0) & ((exponent == 0) | (exponent == 255))):
        raise ValueError("placed-device result leaves the finite normal/signed-zero domain")
    return bits


def _torch_bf16(bits):
    import torch
    return torch.from_numpy(np.array(bits, np.uint16, copy=True, order="C")).view(torch.bfloat16)


def _bits(value):
    import torch
    return value.contiguous().view(torch.uint16).numpy().copy()


def _pack(bits, dtype):
    bits = np.asarray(bits, np.uint16)
    return Tensor(bits if dtype == "bf16" else bits.astype(np.uint32) << np.uint32(16), dtype)


def _pointwise(rtl, operation, operands):
    operands = [_bounded(value, positive=(operation == "div" and index == 1))
                for index, value in enumerate(operands)]
    operands = np.broadcast_arrays(*operands)
    values = [_torch_bf16(value) for value in operands]
    if operation in {"add", "sub", "mul"}:
        result = getattr(rtl, operation)(*values)
    elif operation == "div":
        result = rtl.mul(values[0], rtl.unary("rcp", values[1]))
    elif operation == "rsqrt":
        _bounded(operands[0], positive=True)
        result = rtl.unary("rcp", rtl.unary("sqrt", values[0]))
    elif operation in {"sqrt", "reciprocal"}:
        _bounded(operands[0], positive=True)
        result = rtl.unary("sqrt" if operation == "sqrt" else "rcp", values[0])
    else:
        raise ValueError("unsupported placed-device pointwise sequence")
    return _normal_result(_bits(result))


def _reduction_operand(bits, policy):
    if policy == REDUCTION_POLICY:
        bits = _bounded(bits).copy()
        bits[bits == 0x8000] = 0
        return bits
    if policy != NONNEGATIVE_REDUCTION_POLICY:
        raise ValueError("unknown selected device reduction policy")
    if np.any((bits != 0) & ((bits < 0x0080) | (bits > 0x7f7f))):
        raise ValueError("selected nonnegative reduction requires positive normal BF16 or positive zero")
    return bits


def _sum(rtl, bits, policy):
    bits = _reduction_operand(bits, policy)
    # The incoming instruction accumulator is an explicit positive-zero seed.
    total = _torch_bf16(np.zeros((*bits.shape[:-1], 1), np.uint16))
    for start in range(0, bits.shape[-1], 32):
        width = min(32, bits.shape[-1] - start)
        tile = np.zeros((*bits.shape[:-1], 32), np.uint16)
        tile[..., :width] = bits[..., start:start + width]
        partial = rtl.row_sum(_torch_bf16(tile))
        if policy == NONNEGATIVE_REDUCTION_POLICY:
            _reduction_operand(_bits(partial), policy)
        total = rtl.add(total, partial)
        if policy == NONNEGATIVE_REDUCTION_POLICY:
            _reduction_operand(_bits(total), policy)
    return _bits(total)[..., 0]


def _signature(args, kwargs, minimum, maximum=None, allowed=()):
    if not minimum <= len(args) <= (minimum if maximum is None else maximum) or set(kwargs) - set(allowed):
        raise ValueError("unsupported placed-device source argument signature")


def _reduce(target, args, kwargs, rules):
    default = target.endswith(".default")
    _signature(args, kwargs, 1, 1 if default else 3, ("dtype",) if default else ("dim", "keepdim", "dtype"))
    value = args[0]
    dtype = _dtype_of([value])
    if kwargs.get("dtype") is not None:
        dtype = _dtype(kwargs["dtype"])
        if dtype not in {"bf16", "f32"}:
            raise ValueError("placed-device reduction dtype must be floating")
    rank = value.raw.ndim
    axes = None if default else (args[1] if len(args) > 1 else kwargs.get("dim"))
    keepdim = False if default else (args[2] if len(args) > 2 else kwargs.get("keepdim", False))
    if not isinstance(keepdim, bool):
        raise ValueError("placed-device reduction keepdim must be boolean")
    if axes is None or isinstance(axes, (tuple, list)) and not axes:
        axes = tuple(range(rank))
    elif isinstance(axes, int):
        axes = (axes,)
    try:
        axes = tuple(operator.index(axis) for axis in axes)
    except TypeError as exc:
        raise ValueError("placed-device reduction axes must be integral") from exc
    if any(axis < -rank or axis >= rank for axis in axes):
        raise ValueError("placed-device reduction axis is out of bounds")
    axes = tuple(axis % rank for axis in axes)
    if len(set(axes)) != len(axes):
        raise ValueError("placed-device reduction axes must be unique")
    if axes != tuple(range(rank - len(axes), rank)):
        raise ValueError("selected device reduction requires final contiguous axes")
    shape = value.raw.shape[:rank - len(axes)]
    count = 1
    for extent in value.raw.shape[rank - len(axes):]:
        count *= extent
    bits = _operand(value).reshape((*shape, count))
    rtl = _rtl(rules, ("rcp",) if "mean" in target else ())
    policy = rules.get("reduction_policy", rules.get("reduction", REDUCTION_POLICY))
    result = _sum(rtl, bits, policy)
    if "mean" in target:
        if count == 0:
            raise ValueError("empty selected device mean is unqualified")
        result = _pointwise(rtl, "div", [result, _operand(count)])
    if keepdim:
        result = result.reshape((*shape, *((1,) * len(axes))))
    return _pack(result, dtype)


def _exact_float(target, args, kwargs, rules):
    """Evaluate the route contract, independently of its checked instruction sequence.

    exact_float_device.OPS declares IEEE BF16 add/sub RNE, exact sign flips,
    and RNE casts. Provenance-shared F32 source additions retain F32 semantics.
    No emitted arithmetic is read. The integer BF16 primitive
    implements a single exact sum and boundary rounding, including cancellation.
    """
    from .host import _bf16_add
    _rtl(rules)  # retain pinned target admission even for an exact route
    operation = _BINARY.get(target)
    if operation in {"add", "sub"}:
        _signature(args, kwargs, 2, allowed=("alpha",))
        if kwargs.get("alpha", 1) != 1:
            raise ValueError("exact-float scalar modifier is unqualified")
        dtype = _dtype_of(args)
        if dtype == "bf16" and all(isinstance(value, Tensor) for value in args):
            return _bf16_add(args[0].raw, args[1].raw, subtract=operation == "sub")
        # Source F32 operations can share provenance with the route's exact
        # casts. Preserve the source dtype and its single RNE boundary.
        from .host import _operand as source_operand, _pack as source_pack
        operands = [source_operand(value, dtype) for value in args]
        if any(np.any(~np.isfinite(value)) for value in operands):
            raise ValueError("exact-float nonfinite F32 arithmetic is unqualified")
        with np.errstate(over="ignore", under="ignore", invalid="ignore"):
            result = (np.add if operation == "add" else np.subtract)(*operands, dtype=np.float32)
        return source_pack(result, dtype)
    if target == "aten.neg.default":
        _signature(args, kwargs, 1)
        value = args[0]
        dtype = _dtype_of([value])
        mask = 0x8000 if dtype == "bf16" else 0x80000000
        return Tensor(value.raw ^ np.array(mask, value.raw.dtype), dtype)
    raise ValueError("unsupported source_exact_float_device route operation: " + target)


def _declared_domain(rules, route):
    """The selected region's declared score domain, after checking its policy and mask encoding."""
    if "softmax_policy" in rules:
        raise ValueError("target rules may not declare softmax_policy; the selected application declares it")
    declared = rules.get("softmax") or {}
    if declared.get("policy") != _SOFTMAX_POLICIES[route]:
        raise ValueError(f"{route} requires its declared row-softmax policy, not {declared.get('policy')}")
    if declared.get("score_domain") not in (ABSOLUTE32_DOMAIN, ROWSPAN80_DOMAIN):
        raise ValueError(f"unsupported declared softmax score domain {declared.get('score_domain')}")
    if route == "source_masked_softmax" and declared.get("exclusion_bits") != _SENTINEL["f32"]:
        raise ValueError(f"masked softmax requires the declared finite exclusion bits 0xff7f0000, not {declared.get('exclusion_bits')}")
    return declared["score_domain"]


def _softmax(target, args, kwargs, rules, *, masked):
    """Selected tiled row softmax with explicit arithmetic boundaries.

    Both routes follow their policy's declared tile decomposition: left-to-right
    tile maxima and sums, one reciprocal, and a final multiply. The masked route
    multiplies every exponential tile by its keep tile; the plain route keeps
    every key and multiplies only a partial last tile by its column mask. A
    masked input already includes its additive mask. Only retained scores are
    observable here, so no claim about erased pre-mask scores is made.
    """
    _signature(args, kwargs, 2, 3, ("dtype",) if target == "aten.softmax.int" else ("half_to_float",))
    value, axis = args[:2]
    dtype = _dtype_of([value])
    if axis not in (-1, value.raw.ndim - 1) or value.raw.ndim < 2:
        raise ValueError("selected softmax requires the final axis and rank at least two")
    if kwargs.get("dtype") is not None or len(args) == 3 and args[2] not in (None, False) or kwargs.get("half_to_float", False):
        raise ValueError("selected softmax dtype modifier is unqualified")
    bits = _operand(value)
    count = bits.shape[-1]
    if not 0 < count <= 65536:
        raise ValueError("selected softmax requires 1..65536 keys")
    domain = _declared_domain(rules, "source_masked_softmax" if masked else "source_softmax")
    keep = value.raw != _SENTINEL[dtype] if masked else np.ones(bits.shape, bool)
    if not np.all(np.any(keep, axis=-1)):
        raise ValueError("selected softmax requires a retained key in every row")
    magnitude, exponent = bits & 0x7fff, (bits >> 7) & 255
    if domain == ABSOLUTE32_DOMAIN:
        invalid = keep & ((magnitude > 0x4200) | ((magnitude != 0) & ((exponent < 8) | (exponent > 132))))
        if np.any(invalid):
            index = tuple(map(int, np.argwhere(invalid)[0]))
            raise ValueError(f"selected softmax retained score leaves the declared {domain} domain: index={index}, bits=0x{int(bits[index]):04x}")
    else:
        if count > 1024 or np.any(keep & (((magnitude != 0) & (exponent < 8)) | (exponent == 255))):
            raise ValueError("rowspan80 softmax requires 1..1024 keys and finite normal/signed-zero scores")
        numbers = (bits.astype(np.uint32) << 16).view(np.float32)
        maximum = np.max(np.where(keep, numbers, -np.inf), axis=-1)
        high = np.take_along_axis(bits, np.argmax(np.where(keep, numbers, -np.inf), axis=-1)[..., None], axis=-1)
        low = np.take_along_axis(bits, np.argmin(np.where(keep, numbers, np.inf), axis=-1)[..., None], axis=-1)
        def units(code):
            code = int(code)
            exp, frac = (code >> 7) & 255, code & 127
            number = frac if exp == 0 else (128 + frac) << (exp - 1)
            return -number if code & 0x8000 else number
        if np.any((maximum < -32) | (maximum > 48)) or any(
                units(a) - units(b) > (80 << 133) for a, b in zip(high.flat, low.flat, strict=True)):
            raise ValueError("rowspan80 softmax row maximum/span leaves declared domain")
    rtl = _rtl(rules, ("exp", "rcp"))
    tiles, masks, maxima, widths = [], [], [], []
    for start in range(0, count, 32):
        width = min(32, count - start)
        tile = np.full((*bits.shape[:-1], 32), 0xc200, np.uint16)
        valid = np.zeros(tile.shape, bool)
        valid[..., :width] = keep[..., start:start + width]
        tile[..., :width] = np.where(valid[..., :width], bits[..., start:start + width], 0xc200)
        tensor = _torch_bf16(tile)
        tiles.append(tensor)
        widths.append(width)
        masks.append(_torch_bf16(np.where(valid, 0x3f80, 0).astype(np.uint16)))
        maxima.append(rtl.amax(tensor, -1))
    maximum = maxima[0]
    for partial in maxima[1:]:
        maximum = rtl.maximum(maximum, partial)
    exponentials, sums = [], []
    for tile, mask, width in zip(tiles, masks, widths, strict=True):
        shifted = rtl.sub(tile, maximum)
        if domain == ROWSPAN80_DOMAIN:
            offsets = (_bits(shifted).astype(np.uint32) << 16).view(np.float32)
            if np.any(~np.isfinite(offsets)) or np.any((offsets < -80) | (offsets > 0)):
                raise ValueError("rowspan80 softmax instruction subtraction leaves [-80,0]")
        exponential = rtl.unary("exp", shifted)
        if masked or width < 32:
            exponential = rtl.mul(exponential, mask)
        exponentials.append(exponential)
        sums.append(rtl.row_sum(exponential))
    denominator = sums[0]
    for partial in sums[1:]:
        denominator = rtl.add(denominator, partial)
    reciprocal = rtl.unary("rcp", denominator)
    result = np.concatenate([_bits(rtl.mul(value, reciprocal)) for value in exponentials], axis=-1)[..., :count]
    return _pack(result, dtype)


def _masked_scores(args, kwargs, rules):
    """Source score plus additive mask, computed inside a selected masked row softmax.

    Roles come from the source structure the route recognizes, never from
    operand values: the add's first operand is the scores, with the sum's
    shape, and its second is the mask, which may broadcast (Merlin's
    recognize_masked_row_softmax). The add keeps its source semantics. The
    mask's every code must be positive zero or the BF16 minimum, and the sum
    must equal that finite sentinel exactly where the mask excludes, so the
    softmax reads exclusion from the sum. Any other form refuses.
    """
    from .host import call as source_call
    _declared_domain(rules, "source_masked_softmax")
    _signature(args, kwargs, 2, allowed=("alpha",))
    if kwargs.get("alpha", 1) != 1 or not all(isinstance(value, Tensor) for value in args):
        raise ValueError("masked softmax add requires two source tensors without a scalar modifier")
    dtype = _dtype_of(args)
    scores, mask = args
    if mask.dtype not in _SENTINEL or not np.all((mask.raw == 0) | (mask.raw == _SENTINEL[mask.dtype])):
        raise ValueError("masked softmax mask operand codes must be zero or the BF16 minimum")
    result = source_call("aten.add.Tensor", args, kwargs)
    if scores.raw.shape != result.raw.shape:
        raise ValueError("masked softmax scores operand must have the sum's shape")
    masked = np.broadcast_to(mask.raw == _SENTINEL[mask.dtype], result.raw.shape)
    if np.any((result.raw == _SENTINEL[dtype]) != masked):
        raise ValueError("masked softmax sum differs from the finite sentinel exactly where the mask excludes")
    return result


def call(target, args, kwargs, rules):
    """Run a selected instruction sequence; never infer a node's placement."""
    if not handles(target):
        raise ValueError("no placed-device source adapter for " + str(target))
    routes = rules.get("selected_routes", [])
    from .stages import composite_silu_route
    composite_silu = composite_silu_route(target, routes)
    if "source_exact_float_device" in routes and not composite_silu:
        if routes != ["source_exact_float_device"]:
            raise ValueError("conflicting exact-float numerical routes")
        return _exact_float(target, args, kwargs, rules)
    softmax_routes = {"source_softmax": False, "source_masked_softmax": True}
    if target in _SOFTMAX or set(routes) & softmax_routes.keys():
        if len(routes) != 1 or routes[0] not in softmax_routes:
            raise ValueError("softmax requires one selected device row-softmax route")
        if target in _SOFTMAX:
            return _softmax(target, args, kwargs, rules, masked=softmax_routes[routes[0]])
        if softmax_routes[routes[0]] and target == "aten.add.Tensor":
            return _masked_scores(args, kwargs, rules)
        raise ValueError(f"{routes[0]} route declares no semantics for {target}")
    if target in _REDUCTIONS:
        return _reduce(target, args, kwargs, rules)
    if target in _BINARY:
        operation = _BINARY[target]
        _signature(args, kwargs, 2, allowed=("alpha",) if operation in {"add", "sub"} else ("rounding_mode",) if operation == "div" else ())
        if kwargs.get("alpha", 1) != 1 or kwargs.get("rounding_mode") is not None:
            raise ValueError("placed-device scalar modifier is unqualified")
        rtl = _rtl(rules, ("rcp",) if operation == "div" else ())
        return _pack(_pointwise(rtl, operation, [_operand(value) for value in args]), _dtype_of(args))
    operation = "square" if target == "aten.pow.Tensor_Scalar" else _UNARY[target]
    _signature(args, kwargs, 2 if target == "aten.pow.Tensor_Scalar" else 1)
    if target == "aten.pow.Tensor_Scalar" and (isinstance(args[1], bool) or args[1] != 2):
        raise ValueError("placed-device power requires source square exponent 2")
    value, dtype = args[0], _dtype_of([args[0]])
    tables = {"rsqrt": ("sqrt", "rcp"), "sqrt": ("sqrt",), "reciprocal": ("rcp",),
              "exp": ("exp",), "relu": ("relu",), "silu": ("exp", "rcp")}.get(operation, ())
    rtl = _rtl(rules, tables)
    if operation in {"neg", "abs"}:
        mask = 0x8000 if dtype == "bf16" else 0x80000000
        return Tensor(value.raw ^ np.array(mask, value.raw.dtype) if operation == "neg" else
                      value.raw & np.array(mask - 1, value.raw.dtype), dtype)
    bits = _operand(value)
    if operation == "square":
        result = _pointwise(rtl, "mul", [bits, bits])
    elif operation in {"exp", "relu"}:
        _bounded(bits, exponential=operation == "exp")
        result = _bits(rtl.unary(operation, _torch_bf16(bits)))
    elif operation == "silu":
        _bounded(bits)  # sigmoid route's normal input floor, 2**-63
        exponential = _bits(rtl.unary("exp", _torch_bf16(_bounded(bits ^ np.uint16(0x8000), exponential=True))))
        denominator = _pointwise(rtl, "add", [np.asarray(0x3f80, np.uint16), exponential])
        result = _pointwise(rtl, "div", [bits, denominator])
    else:
        result = _pointwise(rtl, operation, [bits])
    return _pack(result, dtype)
