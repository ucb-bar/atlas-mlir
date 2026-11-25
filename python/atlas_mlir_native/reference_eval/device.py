"""Generic Atlas device arithmetic, independently evaluated from source bytes.

Integer datapaths and activation scales use independently qualified
integer arithmetic helpers. No graph, model policy, emitted program, or framework
numerical kernel supplies reference values. The pinned model is imported only
by :func:`qualify` to cross-check the selected MXU and FP8 pack paths.

Contractions require explicit target rules: ``npu_model_revision``,
``numerics="rtl"``, optional ``selected_unit`` (mxu0/mxu1) for diagnostic overrides, and
``k_order="ascending-k32"``. Otherwise the shared target rule selects the unit from the declared compiler route.
``npu_model_root`` may be supplied in rules or ATLAS_NPU_MODEL_ROOT. Model
qualification checks the selected unit, including continuation and a witness
that distinguishes MXU0 from MXU1; it is not a silicon qualification claim.
"""
from __future__ import annotations

from collections.abc import Mapping
from functools import lru_cache
import hashlib
import importlib
import math
import operator
import os
from pathlib import Path
import subprocess
import sys

import numpy as np

from .evaluator import Tensor

MODEL_REVISION = "b97b2f5a97e432141c4ef8504bec396651d77a8f"
from .target_rules import K_ORDER
_TARGETS = frozenset({"atlas.quantize_act.default", "atlas.qlinear.default",
                      "atlas.qk_matmul.default", "atlas.pv_matmul.default",
                      "atlas.qconv2d.default"})
_MODEL_SOURCES = ("npu_model/__init__.py", "npu_model/configs/__init__.py",
                  "npu_model/configs/numerics.py", "npu_model/configs/rtl_math.py")

def _raw(value, width):
    array = np.asarray(value)
    if array.dtype.kind not in "ui" or np.any(array < 0) or np.any(array > (1 << width) - 1):
        raise ValueError(f"expected unsigned {width}-bit integer encodings")
    return array.astype(np.uint8 if width == 8 else np.uint16, copy=False)

def bf16_values(bits):
    """Decode raw BF16 bits to float32, preserving signed zero and payloads."""
    return (_raw(bits, 16).astype(np.uint32) << 16).view(np.float32)

def bf16_bits(values):
    """Float32 to BF16, round nearest ties even, canonical positive quiet NaN."""
    with np.errstate(over="ignore", invalid="ignore"):
        values = np.asarray(values, dtype=np.float32)
    raw = values.view(np.uint32)
    rounded = (raw + np.uint32(0x7fff) + ((raw >> 16) & 1)) >> 16
    return np.where(np.isnan(values), 0x7fc0, rounded).astype(np.uint16)

def _scale_shift(scale_code):
    code = np.asarray(scale_code)
    if code.dtype.kind not in "iu" or np.any(code < 0) or np.any(code > 255):
        raise ValueError("scale_code must contain raw ERF bytes in [0, 255]")
    return np.clip(code.astype(np.int32) - 127, -128, 127)

def pack_bf16(bits, scale_code, *, mxu=False):
    """FP8Pack (divide) or BF16ScaleToE4M3 accumulator pop (multiply).

    Both flush BF16 denormals and NaN to positive zero and saturate infinity.
    MXU pop alone preserves encoding 0x7f/0xff when finite input rounds to 480.
    """
    raw, shift = np.broadcast_arrays(_raw(bits, 16).astype(np.int32), _scale_shift(scale_code))
    sign, exp, mant = raw >> 15, (raw >> 7) & 255, raw & 127
    adjusted = exp - 127 + (shift if mxu else -shift)
    sig = 128 | mant
    rounded = (sig >> 4) + (((sig & 8) != 0) & (((sig & 7) != 0) | (((sig >> 4) & 1) != 0)))
    adjusted = adjusted + (rounded == 16)
    frac = (np.where(rounded == 16, 8, rounded) - 8) & 7
    saturated = (adjusted > 8) | ((not mxu) & (adjusted == 8) & (frac == 7))
    output = (sign << 7) | ((adjusted + 7) << 3) | frac
    output = np.where(saturated, (sign << 7) | 126, output)
    output = np.where(adjusted < -6, 0, output)
    output = np.where(exp == 255, np.where(mant != 0, 0, (sign << 7) | 126), output)
    return np.where(exp == 0, 0, output).astype(np.uint8)

def multiply_bf16(a_bits, b_bits):
    """VMUL_BF16: BF16 RNE multiplication with HardFloat's default NaN."""
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        return bf16_bits(bf16_values(a_bits) * bf16_values(b_bits))

def readout_bf16(acc_bits, exponent):
    """BF16 accumulator pop followed by VMUL with BF16(2**exponent).

    Exponent is the signed dequantization exponent, usually sx + sw, rather
    than the biased ERF encoding accepted by pack_bf16 and unpack_fp8.
    """
    exponent = np.asarray(exponent)
    if exponent.dtype.kind not in "iu":
        raise ValueError("readout exponent must be integral")
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        scale = bf16_bits(np.ldexp(np.ones(exponent.shape, dtype=np.float32),
                                  np.clip(exponent, -1000, 1000).astype(np.int32)))
    return multiply_bf16(acc_bits, scale)

def _product(a, b):
    a, b = np.asarray(a, dtype=np.int64), np.asarray(b, dtype=np.int64)
    ae, be = (a >> 3) & 15, (b >> 3) & 15
    sig = (8 | (a & 7)) * (8 | (b & 7))
    carry = sig >> 7
    valid = (ae != 0) & (be != 0)
    return (a ^ b) >> 7, np.where(valid, ae + be - 1 + carry, 0), np.where(valid, np.where(carry != 0, sig, sig << 1) & 127, 0)

_BIT_LENGTH = np.array([x.bit_length() for x in range(4096)], dtype=np.int64)

def _systolic_step(a, b, accumulator):
    sign, exponent, fraction = _product(a, b)
    exponent += 114
    acc = accumulator.astype(np.int64)
    ae, am, az = (acc >> 7) & 255, acc & 127, acc >> 15
    p, q = (128 | fraction) << 3, (128 | am) << 3
    product_larger = (exponent > ae) | ((exponent == ae) & (p >= q))
    e = np.where(product_larger, exponent, ae)
    diff = np.abs(exponent - ae)
    result_sign = np.where(product_larger, sign, az)
    big, small = np.where(product_larger, p, q), np.where(product_larger, q, p)
    shift = np.minimum(diff, 11)
    sticky = (small & ((1 << shift) - 1)) != 0
    small = (small >> shift) | sticky
    sig = np.where(sign == az, big + small, big - small)
    zero = sig == 0
    carry = (sig & 2048) != 0
    sticky = sticky | (carry & ((sig & 1) != 0))
    sig = np.where(carry, sig >> 1, sig)
    e = np.where(carry, (e + 1) & 255, e)
    normalize = (~carry) & ((sig & 1024) == 0) & (~zero)
    shift = np.where(normalize, np.minimum(11 - _BIT_LENGTH[sig], e), 0)
    sig, e = (sig << shift) & 2047, e - shift
    frac = ((sig >> 3) & 127) + (((sig & 4) != 0) & (((sig & 3) != 0) | sticky | ((sig & 8) != 0)))
    rounded_carry = frac == 128
    e = np.where(rounded_carry, (e + 1) & 255, e)
    frac = np.where(rounded_carry, 0, frac)
    result = (result_sign << 15) | (np.minimum(e, 254) << 7) | np.where(e > 254, 127, frac)
    result = np.where(zero, 0, result)
    result = np.where(ae == 0, (sign << 15) | (exponent << 7) | fraction, result)
    return np.where(exponent == 114, acc, result).astype(np.uint16)

def _inner_chunk(a, b, accumulator):
    """One 32-lane AnchorAccumulationTree, including incoming BF16 partial."""
    acc = accumulator.astype(np.int64)
    pe, pm = (acc >> 7) & 255, acc & 127
    anchor = np.where((acc & 32767) != 0, pe - 127, -1024)
    products = []
    for k in range(a.shape[-1]):
        product = _product(a[..., :, k, None], b[..., None, k, :])
        products.append(product)
        anchor = np.maximum(anchor, np.where(product[1] != 0, product[1] - 13, -1024))
    anchor += 7
    total = np.zeros(acc.shape, dtype=np.int64)
    for sign, exponent, fraction in products:
        # The 7-bit headroom bounds every valid shift below 64.
        shift = np.clip(anchor - (exponent - 13), 0, 63)
        magnitude = np.where(exponent != 0, ((128 | fraction) << 24) >> shift, 0)
        total += np.where(sign != 0, -magnitude, magnitude)
    shift = anchor - (pe - 127) - 24
    sig = np.where(pe != 0, 128, 0) | pm
    magnitude = np.where(shift >= 0, sig >> np.clip(shift, 0, 63),
                         (sig << np.clip(-shift, 0, 31)) & 0xffffffff)
    magnitude = np.where((shift >= 32) | (shift < -31), 0, magnitude)
    total += np.where((acc & 32768) != 0, -magnitude, magnitude)
    total = ((total + (1 << 31)) & 0xffffffff) - (1 << 31)
    mag = np.abs(total)
    # Every positive 32-bit magnitude has an exact double representation.
    lead = np.frexp(mag.astype(np.float64))[1].astype(np.int64) - 1
    shift = np.maximum(0, lead - 7)
    sig = (mag >> shift) << np.maximum(0, 7 - lead)  # the RTL normalizes sums shorter than eight bits
    remainder = mag & ((1 << shift) - 1)
    half = 1 << np.maximum(0, shift - 1)
    sig += (shift != 0) & ((remainder > half) | ((remainder == half) & ((sig & 1) != 0)))
    carry = sig >= 256
    sig, lead = np.where(carry, sig >> 1, sig), lead + carry
    exponent = lead + anchor - 31 + 127
    result = ((total < 0).astype(np.int64) << 15) | (exponent << 7) | (sig & 127)
    result = np.where(exponent >= 255, ((total < 0).astype(np.int64) << 15) | 0x7f7f, result)
    return np.where((total == 0) | (exponent <= 0), 0, result).astype(np.uint16)

def matmul_codes(a, b, unit="mxu0", *, initial=None, chunk_size=32):
    """A[..., M, K] @ B[..., K, N], yielding raw BF16 accumulators.

    K advances in ascending order. Each 32-lane instruction continues from
    the preceding BF16 partial. MXU0 rounds each custom FMA; MXU1 rounds each
    32-lane anchor tree. A short final instruction is implicitly zero padded.
    initial, if supplied, is an incoming raw BF16 accumulator tile.
    """
    if unit not in ("mxu0", "mxu1"):
        raise ValueError("unit must be mxu0 or mxu1")
    if chunk_size != 32:
        raise ValueError("only the pinned 32-lane instruction schedule is supported")
    a, b = _raw(a, 8), _raw(b, 8)
    if a.ndim < 2 or b.ndim < 2 or a.shape[-1] != b.shape[-2]:
        raise ValueError("matmul_codes requires [..., M, K] and [..., K, N]")
    batch = np.broadcast_shapes(a.shape[:-2], b.shape[:-2])
    a = np.broadcast_to(a, batch + a.shape[-2:])
    b = np.broadcast_to(b, batch + b.shape[-2:])
    shape = batch + (a.shape[-2], b.shape[-1])
    acc = np.zeros(shape, dtype=np.uint16) if initial is None else np.broadcast_to(_raw(initial, 16), shape).copy()
    for start in range(0, a.shape[-1], 32):
        stop = min(start + 32, a.shape[-1])
        if unit == "mxu1":
            acc = _inner_chunk(a[..., start:stop], b[..., start:stop, :], acc)
        else:
            for k in range(start, stop):
                acc = _systolic_step(a[..., :, k, None], b[..., None, k, :], acc)
    return acc

def _scale_exp(values: np.ndarray, axes, rule: str) -> np.ndarray:
    # Nonfinite values do not contribute to the graph's scale selection.
    magnitude = np.abs(values)
    finite = np.where(np.isfinite(magnitude), magnitude, np.float32(0))
    maximum = np.max(finite, axis=axes, keepdims=axes is not None)
    _, exponent = np.frexp(maximum)
    scale = np.where(maximum == 0, 0, exponent - 9).astype(np.int32)
    if rule == "nc":
        # Float64 represents every BF16 maximum and this power-of-two product
        # exactly, including tiny inputs whose requested scales we later refuse.
        limit = np.ldexp(np.float64(448), scale)
        scale = scale + (maximum > limit).astype(np.int32)
    return np.asarray(scale, dtype=np.int32)

def _row_group_exp(values: np.ndarray, rows: int, rule: str) -> np.ndarray:
    lead, m, k = values.shape[:-2], values.shape[-2], values.shape[-1]
    result = np.empty((*lead, m, 1), dtype=np.int32)
    for start in range(0, m, rows):
        stop = min(start + rows, m)
        result[..., start:stop, :] = _scale_exp(
            values[..., start:stop, :], (-2, -1), rule
        )
    return result

def quantize_act(
    x,
    mode: str,
    reduce: str,
    rule: str,
    static_exp: int | None,
    rows: int = 32,
) -> tuple[np.ndarray, np.ndarray]:
    """Return Atlas uint8 codes and the graph's int32 power-of-two exponent.

    Floating inputs are rounded to BF16 before selecting scales. A uint16 input
    explicitly supplies BF16 bit patterns. ``block`` groups flattened rows;
    ``head_block`` groups rows independently within each leading matrix.
    Scales outside -127..127 are refused, never clamped. Code 255 would
    encode exponent 128 but the pinned Atlas model clamps its shift to 127.
    """
    rows = operator.index(rows)
    if rows < 1:
        raise ValueError("rows must be >= 1")
    if mode not in ("dynamic", "static"):
        raise ValueError(f"unsupported activation mode: {mode!r}")
    if reduce not in ("tensor", "token", "head", "block", "head_block"):
        raise ValueError(f"unsupported activation reduction: {reduce!r}")
    if rule not in ("ref", "nc"):
        raise ValueError(f"unsupported activation rule: {rule!r}")
    if (mode == "static") != (static_exp is not None):
        raise ValueError("static_exp is required exactly when mode='static'")

    raw = np.asarray(x)
    bits = raw if raw.dtype == np.uint16 else bf16_bits(raw)
    values = bf16_values(bits)
    if mode == "static":
        requested = operator.index(static_exp)
        if not -127 <= requested <= 127:
            raise ValueError("activation exponent has no exact supported E8M0 scale")
        scale = np.asarray(requested, dtype=np.int32)
    else:
        minimum_rank = {"tensor": 0, "token": 1, "head": 2, "block": 1, "head_block": 2}[reduce]
        if values.ndim < minimum_rank:
            raise ValueError(f"{reduce!r} reduction requires rank >= {minimum_rank}")
        if values.size == 0:
            raise ValueError("dynamic activation quantization requires a nonempty tensor")
        if reduce == "block":
            scale = _row_group_exp(values.reshape(-1, values.shape[-1]), rows, rule)
            scale = scale.reshape(*values.shape[:-1], 1)
        elif reduce == "head_block":
            scale = _row_group_exp(values, rows, rule)
        else:
            axes = {"tensor": None, "token": (-1,), "head": (-2, -1)}[reduce]
            scale = _scale_exp(values, axes, rule)
        if np.any((scale < -127) | (scale > 127)):
            raise ValueError("activation exponent has no exact supported E8M0 scale")
    codes = pack_bf16(bits, scale_code=scale + np.int32(127))
    return np.asarray(codes, dtype=np.uint8), np.asarray(scale, dtype=np.int32)

def _codes(value):
    value = np.asarray(value)
    if value.dtype != np.uint8:
        raise ValueError("code operands must be raw uint8, with no dtype conversion")
    exp, mant = (value >> 3) & 15, value & 7
    if np.any(((exp == 0) & (mant != 0)) | ((exp == 15) & (mant == 7))):
        raise ValueError("graph contains forbidden E4M3 subnormal or reserved codes")
    return value

def _row_exponents(exp, shape):
    exp = np.asarray(exp)
    if exp.dtype.kind not in "iu":
        raise ValueError("exponents must be integers")
    if np.any(exp < -(2**31)) or np.any(exp > 2**31 - 1):
        raise ValueError("operand exponents must be representable in the source int32 format")
    try:
        return np.broadcast_to(exp, shape[:-1] + (1,)).astype(np.int64)
    except ValueError as e:
        raise ValueError("exponents must be scalar or constant over each contraction row") from e


def handles(target):
    """Whether an Atlas schema has an adapter; rules and domains still checked."""
    return target in _TARGETS


def _safe_path(value):
    from .source_io import checked_path
    return checked_path(value)


@lru_cache(maxsize=8)
def _pinned_sources(root):
    commit = subprocess.check_output(
        ["git", "-C", root, "rev-parse", "HEAD"], text=True).strip()
    if commit != MODEL_REVISION:
        raise ValueError("device model revision differs from pinned " + MODEL_REVISION)
    return tuple(subprocess.check_output(
        ["git", "-C", root, "show", f"{MODEL_REVISION}:{name}"])
        for name in _MODEL_SOURCES)


def _authenticate(root):
    root = _safe_path(root)
    # Check HEAD each time, including after the pinned byte cache is populated.
    commit = subprocess.check_output(
        ["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip()
    if commit != MODEL_REVISION:
        raise ValueError("device model revision differs from pinned " + MODEL_REVISION)
    hashes = []
    for name, expected in zip(_MODEL_SOURCES, _pinned_sources(str(root)), strict=True):
        actual = _safe_path(root / name).read_bytes()
        if actual != expected:
            raise ValueError("device model source modified: " + name)
        hashes.append(hashlib.sha256(actual).hexdigest())
    return str(root), tuple(hashes)


def _rules(target_rules, *, contraction, k_extent=None):
    if not isinstance(target_rules, Mapping):
        raise ValueError("device arithmetic requires declared target rules")
    if target_rules.get("npu_model_revision") not in {MODEL_REVISION, MODEL_REVISION[:7]}:
        raise ValueError("device arithmetic requires pinned npu_model " + MODEL_REVISION)
    if target_rules.get("numerics") != "rtl":
        raise ValueError("device arithmetic requires explicit numerics=rtl")
    unit = target_rules.get("selected_unit")
    if contraction and "selected_unit" not in target_rules and k_extent is not None:
        from .target_rules import select_mxu
        unit = select_mxu(k_extent, route=target_rules.get("contraction_route", "already-quantized"),
                          policy_unit=target_rules.get("policy_unit"))
    if contraction:
        if not isinstance(unit, str) or unit.lower() not in {"mxu0", "mxu1"}:
            raise ValueError("device contraction needs a declared selected_unit: mxu0 or mxu1")
        unit = unit.lower()
        if target_rules.get("k_order") != K_ORDER:
            raise ValueError("device contraction needs declared k_order=ascending-k32")
    else:
        unit = None
    root = target_rules.get("npu_model_root") or os.environ.get("ATLAS_NPU_MODEL_ROOT")
    if not root:
        raise ValueError("device arithmetic requires the pinned npu_model_root")
    root, hashes = _authenticate(root)
    _qualify(root, hashes, unit)
    return unit


@lru_cache(maxsize=16)
def _qualify(root, source_hashes, unit):
    """Cross-check independent integers with the chosen pinned RTL model path."""
    old_path, old_bytecode = list(sys.path), sys.dont_write_bytecode
    try:
        sys.dont_write_bytecode = True
        sys.path.insert(0, root)
        module = importlib.import_module("npu_model.configs.numerics")
        expected = Path(root) / "npu_model/configs/numerics.py"
        if _safe_path(module.__file__) != expected:
            raise ValueError("loaded npu_model differs from authenticated model root")
        for name in ("npu_model", "npu_model.configs", "npu_model.configs.rtl_math"):
            loaded = sys.modules[name]
            relative = name.replace(".", "/")
            relative += "/__init__.py" if hasattr(loaded, "__path__") else ".py"
            if _safe_path(loaded.__file__) != Path(root) / relative:
                raise ValueError("loaded model dependency differs from authenticated source: " + name)
        import torch
        torch.set_num_threads(1)
        rtl = module.NUMERICS["rtl"]
        if not isinstance(rtl, module.RtlNumerics):
            raise ValueError("selected model numerics is not RtlNumerics")
        # Every BF16 input encoding is checked at scale 1. Extra boundary scales
        # exercise exponent clamping, flush, saturation, and RNE at a new shift.
        all_bits = np.arange(65536, dtype=np.uint16)
        bf = torch.from_numpy(all_bits).view(torch.bfloat16)
        for scale in (0, 127, 254):
            expected_codes = rtl.to_fp8(bf, scale).view(torch.uint8).numpy()
            if not np.array_equal(pack_bf16(all_bits, scale), expected_codes):
                raise ValueError("independent FP8 pack differs from pinned RTL model")
        witnesses = 0
        if unit is not None:
            method = rtl.systolic_matmul if unit == "mxu0" else rtl.inner_product_matmul
            rng = np.random.default_rng(0xA71A)
            palette = np.array([0, 0x08, 0x18, 0x30, 0x38, 0x40, 0x77,
                                0x80, 0x88, 0x98, 0xB0, 0xB8, 0xC0, 0xF7], np.uint8)
            for k in (3, 32, 33, 65):
                a = rng.choice(palette, (2, k))
                b = rng.choice(palette, (k, 3))
                # Tie witness gives 0x3f80 on MXU0 and 0x3f81 on MXU1.
                a[0, :3], b[:3, 0] = [0x38, 0x18, 0x18], [0x38, 0x18, 0x18]
                actual = matmul_codes(a, b, unit=unit)
                partial = torch.zeros((2, 3), dtype=torch.bfloat16)
                for start in range(0, k, 32):
                    aa = np.zeros((2, 32), np.uint8)
                    bb = np.zeros((32, 3), np.uint8)
                    count = min(32, k - start)
                    aa[:, :count], bb[:count, :] = a[:, start:start + count], b[start:start + count]
                    partial = method(torch.from_numpy(aa).view(torch.float8_e4m3fn),
                                     torch.from_numpy(bb).view(torch.float8_e4m3fn), partial)
                expected_bits = partial.view(torch.uint16).numpy()
                if not np.array_equal(actual, expected_bits):
                    index = tuple(np.argwhere(actual != expected_bits)[0])
                    raise ValueError(f"independent {unit} differs from pinned RTL model at K={k}, {index}")
                witnesses += actual.size
            tie_a = np.array([[0x38, 0x18, 0x18]], np.uint8)
            tie_b = tie_a.T.copy()
            expected_tie = 0x3f80 if unit == "mxu0" else 0x3f81
            if matmul_codes(tie_a, tie_b, unit=unit).item() != expected_tie:
                raise ValueError("selected-unit rounding witness failed")
        return {"revision": MODEL_REVISION, "numerics": "rtl", "selected_unit": unit,
                "k_order": K_ORDER if unit else None, "model_mismatches": 0,
                "pack_encodings_checked": 65536 * 3, "contraction_cells_checked": witnesses,
                "source_sha256": dict(zip(_MODEL_SOURCES, source_hashes, strict=True)),
                "scope": "pinned-model-cross-check", "silicon_qualified": False}
    finally:
        sys.path[:] = old_path
        sys.dont_write_bytecode = old_bytecode


def qualify(target_rules):
    """Authenticate and cross-check one explicitly selected unit, returning evidence."""
    _rules(target_rules, contraction=True)
    root = target_rules.get("npu_model_root") or os.environ.get("ATLAS_NPU_MODEL_ROOT")
    root, hashes = _authenticate(root)
    return dict(_qualify(root, hashes, target_rules["selected_unit"].lower()))


def _arguments(args, kwargs, names, defaults=()):
    if len(args) > len(names) or set(kwargs) - set(names):
        raise ValueError("unsupported device source argument signature")
    values = dict(zip(names, args))
    if values.keys() & kwargs.keys():
        raise ValueError("duplicate device source argument")
    values.update(kwargs)
    optional = dict(zip(names[len(names) - len(defaults):], defaults)) if defaults else {}
    for name in names:
        if name not in values:
            if name not in optional:
                raise ValueError("missing device source argument: " + name)
            values[name] = optional[name]
    return [values[name] for name in names]


def _typed(value, dtype):
    if not isinstance(value, Tensor) or value.dtype != dtype:
        raise ValueError("device source operand requires " + dtype + " logical storage")
    return value.raw


def _contraction_shape(a, b):
    """[... M, K] by [... N, K]: the batch dimensions broadcast, and the result is [batch..., M, N]."""
    try:
        batch = np.broadcast_shapes(a[:-2], b[:-2])
    except ValueError as error:
        raise ValueError("contraction batch dimensions do not broadcast") from error
    return tuple(batch) + (a[-2], b[-2])


def _contraction(a, ax, b, bx, unit):
    a, b = _codes(a), _codes(b)
    if a.ndim < 2 or b.ndim < 2 or a.shape[-1] != b.shape[-1]:
        raise ValueError("contraction requires [... M,K] and [... N,K] with matching K")
    ax, bx = _row_exponents(ax, a.shape), _row_exponents(bx, b.shape)
    shape = _contraction_shape(a.shape, b.shape)
    # No per-K/block weight exponents: scales are applied after accumulation.
    exponent = ax + np.swapaxes(bx, -1, -2)
    if exponent.shape != shape:
        raise ValueError("contraction exponents differ from the broadcast result shape")
    from .evaluator import _AUDIT
    if _AUDIT.get() is not None:
        # A coverage audit checks every argument, rule, and shape, then stands in for the MXU arithmetic.
        return _AUDIT.get().contraction(shape)
    accumulator = matmul_codes(a, np.swapaxes(b, -1, -2), unit=unit)
    return readout_bf16(accumulator, exponent)


def call(target, args, kwargs, target_rules):
    """Evaluate one source Atlas op with explicit selected-unit/K32 rules.

    Floating ATen matmul/linear are source host operations and belong to the
    host adapter. This adapter never silently quantizes their operands.
    """
    if not handles(target):
        raise ValueError("no device source adapter for " + str(target))
    convolution = target == "atlas.qconv2d.default"
    linear = target == "atlas.qlinear.default" or convolution
    if target == "atlas.quantize_act.default":
        x, mode, reduce, rule, static_exp, code_dtype, tag, rows = _arguments(
            args, kwargs, ("x", "mode", "reduce", "rule", "static_exp", "code_dtype", "tag", "rows"), ("", 32))
        if not isinstance(x, Tensor) or x.dtype not in {"bf16", "f32"}:
            raise ValueError("activation quantizer requires BF16 or F32 source bytes")
        if code_dtype != "uint8" or not isinstance(tag, str):
            raise ValueError("activation quantizer requires uint8 codes and a string tag")
        if type(rows) is not int or (static_exp is not None and type(static_exp) is not int):
            raise ValueError("quantizer rows/static_exp must be source integers")
        _rules(target_rules, contraction=False)
        values = x.raw if x.dtype == "bf16" else x.raw.view(np.float32)
        codes, exponent = quantize_act(values, mode, reduce, rule, static_exp, rows)
        return Tensor(codes, "u8"), Tensor(exponent, "i32")
    names = ("x_codes", "x_exp", "w_codes", "w_exp", "bias", "out_dtype", "weight_format") if linear else (
        ("q_codes", "q_exp", "k_codes", "k_exp") if target == "atlas.qk_matmul.default" else
        ("p_codes", "p_exp", "vt_codes", "v_exp", "pop_shift"))
    if convolution:
        names += ("batch", "out_h", "out_w")
    values = _arguments(args, kwargs, names, (0,) if target == "atlas.pv_matmul.default" else ())
    a, ax, b, bx = [_typed(value, dtype) for value, dtype in zip(values[:4], ("u8", "i32", "u8", "i32"), strict=True)]
    dtype = "f32"
    if linear:
        bias, out_dtype, weight_format = values[4:7]
        if convolution:
            # Unfolded source codes: rows are ordered N, H, W; columns are K.
            batch, out_h, out_w = values[7:]
            if any(type(n) is not int or n <= 0 for n in (batch, out_h, out_w)):
                raise ValueError("qconv2d requires positive static batch/out_h/out_w")
            if a.ndim != 2 or b.ndim != 2 or a.shape[0] != batch * out_h * out_w:
                raise ValueError("qconv2d requires unfolded [batch*out_h*out_w,K] and [Cout,K]")
        if weight_format != "atlas_e4m3" or out_dtype not in {"bfloat16", "float32"}:
            raise ValueError("qlinear requires atlas_e4m3 and bfloat16/float32 output")
        dtype = "bf16" if out_dtype == "bfloat16" else "f32"
        if a.ndim < 1 or b.ndim != 2 or a.shape[-1] != b.shape[-1]:
            raise ValueError("qlinear requires [... K] and [N,K]")
        if bias is not None:
            _typed(bias, dtype)
            if bias.raw.shape != (b.shape[0],):
                raise ValueError("qlinear bias must match the source output feature axis")
        shape = a.shape[:-1]
        ax = _row_exponents(ax, a.shape).reshape(-1, 1)
        a = a.reshape(math.prod(shape), a.shape[-1])
    elif target == "atlas.pv_matmul.default" and (type(values[4]) is not int or values[4] != 0):
        raise ValueError("PV pop_shift needs a declared qualified folded arithmetic rule")
    unit = _rules(target_rules, contraction=True, k_extent=a.shape[-1])
    result = _contraction(a, ax, b, bx, unit)
    if linear:
        result = result.reshape(shape + (b.shape[0],))
    output = Tensor(result if dtype == "bf16" else result.astype(np.uint32) << np.uint32(16), dtype)
    if linear and bias is not None:
        from . import host
        output = host.call("aten.add.Tensor", [output, bias], {}, target_rules)
    if convolution:
        # Bias above acts on Cout before the source NCHW restoration.
        output = Tensor(np.ascontiguousarray(output.raw.reshape(batch, out_h, out_w, b.shape[0]).transpose(0, 3, 1, 2)),
                        output.dtype)
    return output
