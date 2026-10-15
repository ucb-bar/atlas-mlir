"""Finite-normal MXU1 reset arithmetic mirrored from the selected RTL source.

This diagnostic follows E4M3Mul, AnchorAccumulationTree, FPUtils, and the
default InnerProductTreeParams at atlas-npu 0079c054 with sp26-fp-units
9a0cc09. It is not a selected-core execution result or a production fallback.
"""

from __future__ import annotations

from fractions import Fraction

from test_mxu_numerics import power2, round_bf16

_VECTOR_LENGTH = 32
_INTEGER_WIDTH = 32
_ANCHOR_HEADROOM = 7
_PRODUCT_BIAS = 13


def _product_fields(lhs: int, rhs: int) -> tuple[int, int, int] | None:
    """Return E4M3Mul's sign, unbiased exponent, and eight-bit significand."""
    if not (0 <= lhs <= 255 and 0 <= rhs <= 255):
        raise ValueError("E4M3 operand must be one byte")
    lhs_exp, rhs_exp = (lhs >> 3) & 15, (rhs >> 3) & 15
    if (lhs_exp == 15 and (lhs & 7) == 7) or (rhs_exp == 15 and (rhs & 7) == 7):
        raise ValueError("MXU1 source mirror excludes E4M3 NaNs")
    if (lhs_exp == 0 and (lhs & 7)) or (rhs_exp == 0 and (rhs & 7)):
        raise ValueError("MXU1 source mirror excludes E4M3 subnormals")
    if not lhs_exp or not rhs_exp:
        return None

    product = (8 | (lhs & 7)) * (8 | (rhs & 7))
    normalized = bool(product & 0x80)
    mantissa = (product & 0x7f) if normalized else ((product & 0x3f) << 1)
    exponent = lhs_exp + rhs_exp - 1 + int(normalized) - _PRODUCT_BIAS
    return ((lhs ^ rhs) >> 7, exponent, 0x80 | mantissa)


def mxu1_reset_lane(lhs: bytes, rhs: bytes) -> tuple[int, int]:
    """Return BF16 bits and the number of products truncated by the anchor."""
    if len(lhs) != _VECTOR_LENGTH or len(rhs) != _VECTOR_LENGTH:
        raise ValueError("MXU1 reset lane requires 32 E4M3 pairs")
    products = [_product_fields(a, b) for a, b in zip(lhs, rhs, strict=True)]
    nonzero = [product for product in products if product is not None]
    if not nonzero:
        return 0, 0

    anchor = max(product[1] for product in nonzero) + _ANCHOR_HEADROOM
    total = 0
    truncated = 0
    for sign, exponent, significand in nonzero:
        shift = anchor - exponent
        magnitude = (significand << (_INTEGER_WIDTH - 8)) >> shift
        exact_scaled = Fraction(significand << (_INTEGER_WIDTH - 8), 1 << shift)
        truncated += magnitude != exact_scaled
        total += -magnitude if sign else magnitude
    total = ((total + (1 << 31)) & 0xffffffff) - (1 << 31)
    value = Fraction(total) * power2(anchor - (_INTEGER_WIDTH - 1))
    return round_bf16(value), truncated
