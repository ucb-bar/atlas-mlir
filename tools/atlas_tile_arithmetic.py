"""Independent finite-value arithmetic for the bounded MXU0 tile experiment.

This module uses rational numbers and integer nearest-even rounding. It does
not import the execution model, Torch, an assembler, or the historical harness.
"""

import math
import struct
from fractions import Fraction
from functools import lru_cache


def power2(exponent):
    return Fraction(1 << exponent) if exponent >= 0 else Fraction(1, 1 << -exponent)


def nearest_even(value):
    quotient, remainder = divmod(value.numerator, value.denominator)
    twice = 2 * remainder
    return quotient + int(twice > value.denominator or
                          (twice == value.denominator and quotient % 2))


def floor_log2(value):
    exponent = value.numerator.bit_length() - value.denominator.bit_length()
    return exponent - int(value < power2(exponent))


@lru_cache(maxsize=65536)
def bf16_value(bits):
    exponent, fraction = (bits >> 7) & 255, bits & 127
    if exponent == 255:
        raise ValueError("nonfinite BF16 outside admitted arithmetic")
    value = (Fraction(fraction, 128) * power2(-126) if exponent == 0 else
             Fraction(128 + fraction, 128) * power2(exponent - 127))
    return -value if bits & 0x8000 else value


def bf16_round(value):
    """Nearest-even BF16, saturating finite overflow as the selected SA does."""
    if not value:
        return 0
    sign = 0x8000 if value < 0 else 0
    magnitude = abs(value)
    exponent = max(-126, floor_log2(magnitude))
    significand = nearest_even(magnitude / power2(exponent - 7))
    if significand == 256:
        exponent += 1
        significand = 128
    if exponent > 127:
        return sign | 0x7F7F
    if exponent == -126 and significand < 128:
        return sign | significand
    return sign | ((exponent + 127) << 7) | (significand - 128)


def f32_to_bf16_bits(value):
    if not math.isfinite(value):
        raise ValueError("nonfinite source")
    raw = struct.unpack("<I", struct.pack("<f", value))[0]
    high, low = raw >> 16, raw & 65535
    result = high + int(low > 32768 or (low == 32768 and high & 1))
    if result & 0x7F80 == 0x7F80:
        raise ValueError("source overflows finite BF16")
    return result


def pack_bf16_unit(bits):
    """Independent unit-scale VPU FP8Pack rule, including flushed +zero.

    Round the normalized significand to three fraction bits, then flush values
    whose rounded exponent is below -6. This is not IEEE gradual underflow.
    """
    magnitude = abs(bf16_value(bits))
    if not magnitude or (bits >> 7) & 255 == 0:
        return 0
    sign = 128 if bits & 0x8000 else 0
    exponent = floor_log2(magnitude)
    significand = nearest_even(magnitude / power2(exponent - 3))
    if significand == 16:
        exponent += 1
        significand = 8
    if exponent < -6:
        return 0
    if exponent > 8 or (exponent == 8 and significand >= 15):
        return sign | 0x7E
    return sign | ((exponent + 7) << 3) | (significand - 8)


@lru_cache(maxsize=256)
def fp8_value(bits):
    exponent, fraction = (bits >> 3) & 15, bits & 7
    if exponent == 0:
        return Fraction(0)
    if exponent == 15 and fraction == 7:
        raise ValueError("NaN FP8 outside admitted arithmetic")
    value = Fraction(8 + fraction, 8) * power2(exponent - 7)
    return -value if bits & 128 else value


@lru_cache(maxsize=65536)
def fp8_product(a, b):
    return fp8_value(a) * fp8_value(b)


def mxu0_step_uncached(a, b, accumulator):
    product = fp8_product(a, b)
    return bf16_round(bf16_value(accumulator) + product) if product else accumulator


mxu0_step = lru_cache(maxsize=262144)(mxu0_step_uncached)


def mxu0_dot(activations, weights, *, cached=True):
    """Ordered BF16 rounding after every nonzero FP8 product; reset at K=0."""
    accumulator = 0
    step = mxu0_step if cached else mxu0_step_uncached
    for a, b in zip(activations, weights, strict=True):
        accumulator = step(int(a), int(b), accumulator)
    return accumulator


def cache_statistics():
    return {name: function.cache_info()._asdict() for name, function in
            (("fp8", fp8_value), ("bf16", bf16_value),
             ("products", fp8_product), ("transitions", mxu0_step))}


def output_halves(rows):
    if len(rows) != 32 or any(len(row) != 32 for row in rows):
        raise ValueError("output must be 32 by 32")
    return b"".join(struct.pack("<H", int(rows[row][column]))
                    for half in range(2) for row in range(32)
                    for column in range(half * 16, half * 16 + 16))
