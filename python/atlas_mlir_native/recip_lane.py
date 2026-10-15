"""Source-attributed BF16 VRECIP port on positive normal inputs in [1, 32].

The selected Atlas RTL at 0079c0541111197741a231c002e3843fa6f545b2
uses Rcp.scala and the sp26-fp-units RcpLUT/LUTParams gitlink at
9a0cc09c41ab918a3548580185f39cca8d559e0d. The LUT rounds a fixed-point
reciprocal before LUTParams truncates its BF16 fraction. This is not a
correctly rounded real reciprocal.
"""

from __future__ import annotations


def vrecip_bf16_bits(code: int) -> int:
    """Replay the selected 128-entry LUT and BF16 packing in its admitted range."""
    if type(code) is not int or not 0x3F80 <= code <= 0x4200:
        raise ValueError("VRECIP requires positive normal BF16 in [1, 32]")
    exponent = (code >> 7) & 0xFF
    fraction = code & 0x7F

    # Scala Math.round((1 / (1 + fraction / 128)) * 2^16).
    denominator = 128 + fraction
    numerator = 128 << 16
    fixed = (2 * numerator + denominator) // (2 * denominator)
    highest = fixed.bit_length() - 1
    output_exponent = (254 + highest - 16 - exponent) & 0xFF
    if output_exponent in (0, 255):
        raise ValueError("VRECIP unexpectedly left the normal result domain")
    output_fraction = (fixed >> max(highest - 7, 0)) & 0x7F
    return (output_exponent << 7) | output_fraction
