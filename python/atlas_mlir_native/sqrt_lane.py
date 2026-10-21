"""Bounded port of the selected SqrtLUT and LUTParams BF16 path.

Sources: Atlas Sqrt.scala at 0079c0541111197741a231c002e3843fa6f545b2
and sp26-fp-units SqrtLUT.scala/LUTParams.scala at
9a0cc09c41ab918a3548580185f39cca8d559e0d. The LUT rounds Q1.16 entries
and truncates BF16 fractions; this is not a correctly rounded square root.
"""

from math import isqrt


def _round_sqrt(numerator: int, denominator: int) -> int:
    floor = isqrt(numerator // denominator)
    return floor + (4 * numerator >= denominator * (2 * floor + 1) ** 2)


def vsqrt_bf16_bits(code: int) -> int:
    """Evaluate positive BF16 inputs in [2**-63, 2**63] using integer math."""
    if type(code) is not int or not 0x2000 <= code <= 0x5F00:
        raise ValueError("VSQRT needs positive normal BF16 in [2**-63,2**63]")
    exponent, fraction = code >> 7, code & 127
    entry = _round_sqrt((128 + fraction) << 32, 128)
    if exponent % 2 == 0:
        entry = (entry * _round_sqrt(2 << 32, 1) >> 16) & 0x1FFFF
    return ((127 + (exponent - 127) // 2) << 7) | ((entry >> 9) & 127)
