"""Separate BF16 primitive domains for masked nonpositive exponential chains.

These contracts preserve the original normal-only arithmetic profiles. They
name the selected datapaths, rather than mathematical real SUB or EXP.
"""

from fractions import Fraction

from .exp_lane import EXP2_LUT_Q1_16

SUB_NONNAN_ATTRS = (
    ("input_domain", "bf16-lhs-finite-or-neginf-rhs-finite-or-posinf-v1"),
    ("numeric_semantics", "selected-f32-rne-bf16-chop-nonnan-v1"),
)
EXP_NONPOSITIVE_ATTRS = (
    ("input_domain", "bf16-finite-minus128-zero-or-neginf-v1"),
    ("numeric_semantics", "selected-rtl-bf16-exp-q9p12-lut-rne-gradual-underflow-v1"),
)
MUL_FINITE_ATTRS = (
    ("input_domain", "bf16-finite-including-subnormal-and-signed-zero-v1"),
    ("numeric_semantics", "bf16-product-rne-gradual-underflow-overflow-inf-v1"),
)


def build_bf16_primitive_request(op, *, target_identity=None):
    """Construct one exact-profile representative for native snapshot gating.

    This is a target primitive request, not a replacement for source capture.
    Callers retain the composite's source proof and operand-domain obligations.
    """
    from .binding import (TARGET_IDENTITY, BF16_BINARY_ATTRS, BF16_UNARY_ATTRS,
                          BF16_ROW_REDUCTION_ATTRS, _row_reduce_maps)
    from merlin.semantic_compiler.model import KernelRequest, SemanticNode, TensorType, pointwise_identity_maps

    attributes = {**BF16_BINARY_ATTRS, **BF16_UNARY_ATTRS, **BF16_ROW_REDUCTION_ATTRS}
    if op not in attributes:
        raise ValueError("unknown BF16 primitive profile")
    arity = 2 if op in BF16_BINARY_ATTRS else 1
    dtype = TensorType((32, 32), "bf16", "raw-bit-pattern")
    attrs = attributes[op]
    inputs = tuple(SemanticNode(f"in{i}", "input", (), dtype, attrs=(attrs[0],), effect="input")
                   for i in range(arity))
    result = SemanticNode("result", op, tuple(node.id for node in inputs), dtype, attrs=attrs,
                          index_maps=_row_reduce_maps() if op in BF16_ROW_REDUCTION_ATTRS else pointwise_identity_maps(2, arity))
    return KernelRequest((*inputs, result), (result.id,), ("external",),
                         tuple((node.id, "external") for node in inputs),
                         TARGET_IDENTITY if target_identity is None else target_identity, lowering_policy="diagnostic")


def _pow2(exponent):
    return Fraction(1 << exponent) if exponent >= 0 else Fraction(1, 1 << -exponent)


def _positive_bf16_rne_gradual(value):
    """Exact positive rounding, including gradual BF16 underflow and +zero."""
    exponent = value.numerator.bit_length() - value.denominator.bit_length()
    if value < _pow2(exponent):
        exponent -= 1
    step = _pow2(max(-126, exponent) - 7)
    scaled = value / step
    integer, remainder = divmod(scaled.numerator, scaled.denominator)
    integer += int(2 * remainder > scaled.denominator or
                   (2 * remainder == scaled.denominator and integer & 1))
    if exponent < -126:
        return integer  # 128 is the minimum normal BF16 encoding.
    if integer == 256:
        integer, exponent = 128, exponent + 1
    if exponent > 127:
        return 0x7f80
    return ((exponent + 127) << 7) | (integer - 128)


def vexp_bf16_nonpositive_bits(code):
    """Selected Q9.12/LUT EXP on [-128,0] or -Inf, with gradual underflow.

    This separate function does not widen exp_lane.vexp_bf16_bits. No ROM or
    model function is consulted; the Q9.12 path shares its attributed LUT.
    """
    if type(code) is not int or not 0 <= code <= 0xffff:
        raise ValueError("nonpositive VEXP requires a raw BF16 code")
    if code in (0, 0x8000):
        return 0x3f80
    if code == 0xff80:
        return 0
    exponent = (code >> 7) & 255
    if not code & 0x8000 or exponent == 255:
        raise ValueError("nonpositive VEXP requires finite [-128,0] or -Inf")
    magnitude = ((128 + (code & 127)) * _pow2(exponent - 134) if exponent
                 else (code & 127) * _pow2(-133))
    if magnitude > 128:
        raise ValueError("nonpositive VEXP input is below -128")
    q9_12 = -int(magnitude * 4096)
    scaled = (q9_12 * 5909) >> 12
    k, residual = scaled >> 12, scaled & 4095
    address, low = residual >> 7, residual & 127
    lower = EXP2_LUT_Q1_16[address]
    upper = EXP2_LUT_Q1_16[address + 1] if address < 31 else 0x1ffff
    interpolated = lower + (((upper - lower) * low) >> 7)
    return _positive_bf16_rne_gradual(Fraction(interpolated) * _pow2(k - 16))


def mul_bf16_finite_bits(left, right):
    """Exact IEEE BF16 finite-input product, with one nearest-even rounding."""
    values = []
    for code in (left, right):
        if type(code) is not int or not 0 <= code <= 0xffff or (code >> 7) & 255 == 255:
            raise ValueError("finite BF16 multiply rejects infinity and NaN inputs")
        exponent, fraction = (code >> 7) & 255, code & 127
        values.append((128 + fraction) * _pow2(exponent - 134) if exponent else fraction * _pow2(-133))
    sign = (left ^ right) & 0x8000
    exact = values[0] * values[1]
    return sign | (_positive_bf16_rne_gradual(exact) if exact else 0)
