"""Source-attributed BF16 VEXP port for a bounded finite input interval.

The selected Atlas RTL at 0079c0541111197741a231c002e3843fa6f545b2 uses
ExpLane.scala, AtlasFPType.BF16 in common.scala, Qmn.scala, and ExLUT.scala
from the sp26-fp-units gitlink at
9a0cc09c41ab918a3548580185f39cca8d559e0d. This port covers normal
BF16 inputs in [-80, 80] and either signed zero. NaN, infinity, subnormal,
overflow, and underflow behavior remain outside this port.
"""

from __future__ import annotations

from fractions import Fraction

# Java Math.round(Math.pow(2.0, i/32) * 2^16), i=0..31. Each entry also
# agrees with a high-precision Decimal computation at this source revision.
EXP2_LUT_Q1_16 = (
    65536, 66971, 68438, 69936, 71468, 73032, 74632, 76266,
    77936, 79642, 81386, 83169, 84990, 86851, 88752, 90696,
    92682, 94711, 96785, 98905, 101070, 103283, 105545, 107856,
    110218, 112631, 115098, 117618, 120194, 122825, 125515, 128263,
)


def _pow2(exponent: int) -> Fraction:
    return Fraction(1 << exponent) if exponent >= 0 else Fraction(1, 1 << -exponent)


def _bf16_rne(value: Fraction) -> int:
    if value <= 0:
        raise ValueError("VEXP result must be positive")
    exponent = value.numerator.bit_length() - value.denominator.bit_length()
    if value < _pow2(exponent):
        exponent -= 1
    if not -126 <= exponent <= 127:
        raise ValueError("VEXP result is outside normal BF16")
    scaled = value / _pow2(exponent - 7)
    integer, remainder = divmod(scaled.numerator, scaled.denominator)
    rounded = integer + (
        2 * remainder > scaled.denominator
        or (2 * remainder == scaled.denominator and integer % 2 == 1)
    )
    if rounded == 256:
        rounded = 128
        exponent += 1
    if exponent > 127:
        raise ValueError("VEXP result overflows BF16")
    return ((exponent + 127) << 7) | (rounded - 128)


def vexp_bf16_bits(code: int) -> int:
    """Port the selected finite-normal VEXP path, including LUT interpolation."""
    if type(code) is not int or not 0 <= code <= 0xFFFF:
        raise ValueError("VEXP input must be a BF16 encoding")
    if code in (0, 0x8000):
        return 0x3F80
    exponent = (code >> 7) & 0xFF
    if not 1 <= exponent <= 254:
        raise ValueError("VEXP port requires finite-normal BF16 input")
    magnitude = Fraction(128 + (code & 0x7F)) * _pow2(exponent - 134)
    if magnitude > 80:
        raise ValueError("VEXP input is outside [-80, 80]")

    # qmnFromRawFloat truncates the magnitude to Q9.12, then applies sign.
    q9_12 = int(magnitude * 4096)
    if code & 0x8000:
        q9_12 = -q9_12
    # Qmn.mul with BF16's rln2Val=5909 uses an arithmetic right shift.
    q_times_log2e = (q9_12 * 5909) >> 12
    k, residual = q_times_log2e >> 12, q_times_log2e & 0xFFF
    address, low = residual >> 7, residual & 0x7F
    lower = EXP2_LUT_Q1_16[address]
    upper = EXP2_LUT_Q1_16[address + 1] if address < 31 else 0x1FFFF
    interpolated = lower + (((upper - lower) * low) >> 7)
    return _bf16_rne(Fraction(interpolated) * _pow2(k - 16))
