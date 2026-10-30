"""Opt-in exact vector arithmetic matching atlas_tile_arithmetic's finite oracle.

The ordered MAC fast path starts at BF16 +0 and admits finite FP8 E4M3 codes
with exponent-zero flushed to +0, exactly as fp8_value. It is not a general
BF16 FMA implementation and does not accept an incoming accumulator.

Proof of exact binary64 intermediates: FP8 operands are multiples of 2**-9
and have magnitude at most 448, so products are multiples of q=2**-18 with
magnitude at most 200704 < 2**18. Starting at zero, BF16 accumulators never
leave [-2**26, 2**26]: the outward midpoint at either boundary is 2**18 away,
farther than any product. BF16 rounding preserves the q grid. Consequently
all exact pre-round sums have |sum| < 2**27 and fewer than 46 significant
binary digits on that grid, below binary64's 53-bit precision. FP8 products
are also exact. Integer bit operations perform BF16 nearest-even directly
on binary64 bits; no intermediate float32 rounding is used. Exact zero is
canonical +0; zero products preserve the existing (canonical) accumulator.

The rescale API separately retains the scalar [-126,127] exponent admission
and exact finite-overflow refusal. Decoded BF16 values and their power-of-two
rescaling remain normal binary64 values, including BF16 subnormals, so those
operations are exact too. General BF16 rounding uses integer significand
rounding for gradual subnormals and saturates finite overflow like the scalar
oracle. This module imports NumPy only; it does not import the device model.
"""

from __future__ import annotations

import operator

import numpy as np

_SIGN = np.uint64(1 << 63)
_MAGNITUDE = np.uint64((1 << 63) - 1)
_LOW45 = np.uint64((1 << 45) - 1)
_PRECISION_MASK = np.uint64(((1 << 64) - 1) ^ ((1 << 45) - 1))
_ROUND_BIAS = np.uint64((1 << 44) - 1)
_BIAS_DELTA = np.uint64((1023 - 127) << 7)
_MAX_BF16 = float(255 * 2**120)


def _codes(values, dtype, label):
    array = np.asarray(values)
    if array.dtype != np.dtype(dtype):
        raise ValueError(f"{label} must have dtype {np.dtype(dtype)}")
    return array


def _fp8_lut():
    codes = np.arange(256, dtype=np.uint16)
    exponent, fraction = (codes >> 3) & 15, codes & 7
    values = np.ldexp((8 + fraction).astype(np.float64), exponent.astype(np.int16) - 10)
    values = np.where(exponent == 0, 0.0, values)
    values = np.where((codes & 128) != 0, -values, values)
    # fp8_value discards the sign of every exponent-zero representation.
    values[exponent == 0] = 0.0
    values[[0x7F, 0xFF]] = np.nan
    return values


_FP8 = _fp8_lut()


def decode_fp8(values):
    codes = _codes(values, np.uint8, "FP8 operands")
    if np.any((codes & 127) == 127):
        raise ValueError("NaN FP8 outside admitted arithmetic")
    return _FP8[codes]


def round_bf16(values):
    """Finite binary64 values -> BF16 uint16, direct exact nearest-even.

    Negative nonzero values rounding below the least subnormal retain -0;
    input +0/-0 both become +0, matching bf16_round's exact-zero convention.
    """
    array = np.asarray(values, dtype=np.float64)
    raw = array.view(np.uint64)
    magnitude = raw & _MAGNITUDE
    exponent = (magnitude >> np.uint64(52)) & np.uint64(2047)
    if np.any(exponent == 2047):
        raise ValueError("nonfinite value outside admitted arithmetic")
    result = np.zeros(array.shape, dtype=np.uint16)
    normal = exponent >= 897  # binary64 encoding of 2**-126.
    mag = magnitude[normal]
    high, low = mag >> np.uint64(45), mag & _LOW45
    increment = (low > (1 << 44)) | ((low == (1 << 44)) & ((high & 1) != 0))
    rounded = high + increment.astype(np.uint64) - _BIAS_DELTA
    result[normal] = np.minimum(rounded, 0x7F7F).astype(np.uint16)
    subnormal = (exponent >= 889) & (exponent < 897)
    # value / 2**-133 = significand / 2**(942-biased_exponent).
    # Exponents below 889 are strictly below half the least BF16 subnormal.
    shift = np.uint64(942) - exponent[subnormal]
    significand = (magnitude[subnormal] & np.uint64((1 << 52) - 1)) | np.uint64(1 << 52)
    whole = significand >> shift
    remainder = significand & ((np.uint64(1) << shift) - np.uint64(1))
    halfway = np.uint64(1) << (shift - np.uint64(1))
    increment = (remainder > halfway) | ((remainder == halfway) & ((whole & 1) != 0))
    result[subnormal] = (whole + increment.astype(np.uint64)).astype(np.uint16)
    sign = ((raw >> np.uint64(48)) & np.uint64(0x8000)).astype(np.uint16)
    result |= np.where(magnitude != 0, sign, np.uint16(0))
    return result


def decode_bf16(values):
    codes = _codes(values, np.uint16, "BF16 operands")
    exponent, fraction = (codes >> 7) & 255, codes & 127
    if np.any(exponent == 255):
        raise ValueError("nonfinite BF16 outside admitted arithmetic")
    significand = np.where(exponent == 0, fraction, 128 + fraction).astype(np.float64)
    power = np.where(exponent == 0, -133, exponent.astype(np.int16) - 134)
    decoded = np.ldexp(significand, power)
    return np.where((codes & 0x8000) != 0, -decoded, decoded)


def rescale_bf16_array(values, exponent):
    """Exact scalar rescale_bf16 policy applied to an entire uint16 array."""
    exponent = operator.index(exponent)
    if not -126 <= exponent <= 127:
        raise ValueError("VPU rescale factor must be normal finite BF16")
    codes = _codes(values, np.uint16, "BF16 operands")
    exact = np.ldexp(decode_bf16(codes), exponent)
    if np.any(np.abs(exact) > _MAX_BF16):
        raise ValueError("VPU output rescale overflows finite BF16")
    rounded = round_bf16(exact)
    return np.where((codes & 0x7FFF) == 0, codes, rounded).astype(np.uint16)


def ordered_matmul(encoded_a, encoded_b):
    """uint8[M,K] x uint8[K,N] -> uint16[M,N], zero-init ordered BF16 MACs.

    K is visited in its original order. Rows are blocked to bound temporary
    storage; there is no reassociation, BLAS reduction, or incoming state.
    """
    a = _codes(encoded_a, np.uint8, "left FP8 operand")
    b = _codes(encoded_b, np.uint8, "right FP8 operand")
    if a.ndim != 2 or b.ndim != 2 or a.shape[1] != b.shape[0]:
        raise ValueError("operands must have compatible rank-two shapes")
    left, right = decode_fp8(a), decode_fp8(b)
    result = np.zeros((a.shape[0], b.shape[1]), dtype=np.uint16)
    for start in range(0, a.shape[0], 128):
        lhs = left[start : start + 128]
        accumulator = np.zeros((lhs.shape[0], b.shape[1]), dtype=np.float64)
        for k in range(a.shape[1]):
            exact = accumulator + lhs[:, k, None] * right[None, k, :]
            bits = exact.view(np.uint64)
            magnitude = bits & _MAGNITUDE
            rounded = (magnitude + _ROUND_BIAS + ((magnitude >> np.uint64(45)) & 1)) & _PRECISION_MASK
            rounded |= bits & _SIGN
            rounded[magnitude == 0] = 0
            accumulator = rounded.view(np.float64)
        result[start : start + lhs.shape[0]] = round_bf16(accumulator)
    return result
