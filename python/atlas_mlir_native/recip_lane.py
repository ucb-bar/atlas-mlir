"""Source-attributed BF16 VRECIP port with distinct bounded input domains.

The selected Atlas RTL at 0079c0541111197741a231c002e3843fa6f545b2
uses Rcp.scala and the sp26-fp-units RcpLUT/LUTParams gitlink at
9a0cc09c41ab918a3548580185f39cca8d559e0d. The LUT rounds a fixed-point
reciprocal before LUTParams truncates its BF16 fraction. This is not a
correctly rounded real reciprocal. The default domain stays [1, 32]; the
separate wide variant covers [1, 65536] without changing the LUT algorithm.
The positive bounded variant covers [2**-63, 2**63]; the original ranges
continue to reject inputs outside their declarations.
"""

from __future__ import annotations


def vrecip_bf16_bits(code: int, *, wide: bool = False) -> int:
    """Replay the selected 128-entry LUT and BF16 packing in its admitted range."""
    if type(wide) is not bool:
        raise ValueError("VRECIP range selection must be boolean")
    upper = 0x4780 if wide else 0x4200
    if type(code) is not int or not 0x3F80 <= code <= upper:
        raise ValueError(
            f"VRECIP requires positive normal BF16 in [1, {65536 if wide else 32}]"
        )
    return _reciprocal(code)


def vrecip_bf16_bounded_bits(code: int) -> int:
    """Separate positive normal domain [2**-63,2**63], with normal results."""
    if type(code) is not int or not 0x2000 <= code <= 0x5F00:
        raise ValueError("VRECIP requires positive normal BF16 in [2**-63,2**63]")
    return _reciprocal(code)


def _reciprocal(code: int) -> int:
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
