"""Bounded value checks for Atlas native semantic operations.

These callbacks are for verification. The compiler does not use them to
select instructions or to produce runtime output. The MXU1 finite-reset
callback is an attributed port of the selected source arithmetic.
"""

from __future__ import annotations

import struct
from fractions import Fraction

from merlin.semantic_compiler.model import IndexMap, pointwise_identity_maps
from merlin.semantic_compiler.reference import TensorValue

from atlas_mlir_native.exp_lane import vexp_bf16_bits
from atlas_mlir_native.pack_lane import (
    ROUNDED_PACK_DOMAIN,
    ROUNDED_PACK_POLICY,
    pack_bf16_lane,
)
from atlas_mlir_native.recip_lane import vrecip_bf16_bits

_BF16_TO_FP8_UNIT = {
    0x0000: 0x00,
    0x3F00: 0x30,
    0x3F80: 0x38,
    0x4000: 0x40,
    0xBF00: 0xB0,
    0xBF80: 0xB8,
    0xC000: 0xC0,
}
_POLICY = "fp8-e4m3-unit-e8m0-seven-codes-v1"
_DOMAIN = "bf16-seven-powers-and-zero-v1"
_SCALED_POLICY = "fp8-e4m3-scaled-e8m0-exact-normal-122-128-v1"
_SCALED_DOMAIN = "bf16-positive-zero-or-exact-normal-under-scale-122-128-v1"
_ADD_ATTRS = {
    "input_domain": "finite-normal-exact-fp32-sum-v1",
    "numeric_semantics": "selected-f32-rne-bf16-chop-v1",
}
_SUB_ATTRS = {
    "input_domain": "finite-normal-exact-fp32-difference-or-zero-v1",
    "numeric_semantics": "selected-f32-rne-bf16-chop-v1",
}
_MUL_ATTRS = {
    "input_domain": "finite-normal-bf16-product-normal-v1",
    "numeric_semantics": "bf16-product-rne-v1",
}
_ROW_MAX_ATTRS = {
    "input_domain": "bf16-raw-all-codes-v1",
    "numeric_semantics": "selected-rtl-bf16-total-order-row-max-v1",
}
_ROW_SUM_ATTRS = {
    "input_domain": "bf16-normal-or-positive-zero-fp32-tree-normal-result-v1",
    "numeric_semantics": "selected-rtl-bf16-row-sum-fp32-rne-tree-bf16-rne-v1",
}
_EXP_ATTRS = {
    "input_domain": "bf16-normal-minus80-plus80-or-signed-zero-v1",
    "numeric_semantics": "selected-rtl-bf16-exp-q9p12-lut-rne-v1",
}
_RECIP_ATTRS = {
    "input_domain": "bf16-positive-normal-1-to-32-v1",
    "numeric_semantics": "selected-rtl-bf16-rcp-lut-trunc-v1",
}
_ROW_REDUCE_MAPS = (
    IndexMap(3, ((1, 0, 0), (0, 0, 1)), (0, 0)),
    IndexMap(3, ((1, 0, 0), (0, 1, 0)), (0, 0)),
)
_MXU1_BOUNDED_POLICY = "mxu1-e4m3-anchor-bf16-bounded-v1"
_MXU1_BOUNDED_CODES = {0x00: 0, 0x38: 1, 0x40: 2, 0xB8: -1, 0xC0: -2}
_MXU1_FINITE_RESET_POLICY = "mxu1-e4m3-anchor-bf16-finite-reset-v1"
_MXU_MATMUL_MAPS = (
    IndexMap(3, ((1, 0, 0), (0, 0, 1)), (0, 0)),
    IndexMap(3, ((0, 1, 0), (0, 0, 1)), (0, 0)),
    IndexMap(3, ((1, 0, 0), (0, 1, 0)), (0, 0)),
)


def evaluate_mxu1_bounded_reset(node, operands: tuple[TensorValue, ...]) -> TensorValue:
    """Check the admitted MXU1 reset domain with exact integer products.

    At atlas-npu 0079c054 / sp26-fp-units 9a0cc09, these five E4M3 codes
    yield only products 0, 1, 2, or 4 in magnitude. The anchor shifter retains
    every product bit for this exponent span, and the sum of 32 products is
    within [-128, 128], where every integer is exactly representable in BF16.
    This does not describe wider MXU1 inputs or accumulator continuation.
    """
    if (
        node.op != "matmul" or len(operands) != 2 or node.attrs
        or node.index_maps != _MXU_MATMUL_MAPS
        or node.type.shape != (32, 32) or node.type.dtype != "bf16"
        or node.type.numerical_policy != _MXU1_BOUNDED_POLICY
        or any(operand.type.shape != (32, 32)
               or operand.type.dtype != "fp8_e4m3"
               or operand.type.numerical_policy != "raw-bit-pattern"
               for operand in operands)
    ):
        raise ValueError("MXU1 bounded reset has an unsupported semantic contract")
    try:
        activation = tuple(_MXU1_BOUNDED_CODES[code] for code in operands[0].elements)
        weight = tuple(_MXU1_BOUNDED_CODES[code] for code in operands[1].elements)
    except KeyError as exc:
        raise ValueError("MXU1 input is outside the admitted five-code domain") from exc

    output = []
    for row in range(32):
        for column in range(32):
            value = sum(activation[row * 32 + reduction] * weight[column * 32 + reduction]
                        for reduction in range(32))
            bits = struct.unpack("<I", struct.pack("<f", float(value)))[0] >> 16
            output.append(bits)
    return TensorValue(node.type, tuple(output))


def _mxu1_product_fields(lhs: int, rhs: int) -> tuple[int, int, int] | None:
    """Mirror E4M3Mul's finite-normal product fields at sp26-fp-units 9a0cc09."""
    for code in (lhs, rhs):
        exponent = (code >> 3) & 15
        if code != 0 and not 1 <= exponent <= 14:
            raise ValueError("MXU1 finite reset input is outside positive-zero/finite-normal domain")
    if lhs == 0 or rhs == 0:
        return None
    left_exp, right_exp = (lhs >> 3) & 15, (rhs >> 3) & 15
    product = (8 | (lhs & 7)) * (8 | (rhs & 7))
    normalized = bool(product & 0x80)
    mantissa = (product & 0x7F) if normalized else ((product & 0x3F) << 1)
    exponent = left_exp + right_exp - 1 + int(normalized) - 13
    return ((lhs ^ rhs) >> 7, exponent, 0x80 | mantissa)


def _round_exact_bf16(value: Fraction) -> int:
    if not value:
        return 0
    magnitude = abs(value)
    exponent = magnitude.numerator.bit_length() - magnitude.denominator.bit_length()
    if magnitude < _pow2(exponent):
        exponent -= 1
    if not -126 <= exponent <= 127:
        raise ValueError("MXU1 result is outside finite normal BF16")
    scaled = magnitude / _pow2(exponent - 7)
    integer, remainder = divmod(scaled.numerator, scaled.denominator)
    rounded = integer + (
        2 * remainder > scaled.denominator
        or (2 * remainder == scaled.denominator and integer % 2 == 1)
    )
    if rounded == 256:
        rounded = 128
        exponent += 1
    if not -126 <= exponent <= 127:
        raise ValueError("MXU1 result rounds outside finite normal BF16")
    return (0x8000 if value < 0 else 0) | ((exponent + 127) << 7) | (rounded - 128)


def _mxu1_anchor_reset_lane(lhs: tuple[int, ...], rhs: tuple[int, ...]) -> int:
    """Port selected AnchorAccumulationTree reset arithmetic for 32 products.

    Source: atlas-npu 0079c054, sp26-fp-units 9a0cc09, E4M3Mul,
    AnchorAccumulationTree, and FPUtils. This does not model continuation.
    """
    if len(lhs) != 32 or len(rhs) != 32:
        raise ValueError("MXU1 reset lane requires 32 FP8 pairs")
    products = tuple(_mxu1_product_fields(left, right)
                     for left, right in zip(lhs, rhs, strict=True))
    nonzero = tuple(product for product in products if product is not None)
    if not nonzero:
        return 0
    anchor = max(product[1] for product in nonzero) + 7
    total = 0
    for sign, exponent, significand in nonzero:
        magnitude = (significand << 24) >> (anchor - exponent)
        total += -magnitude if sign else magnitude
    total = ((total + (1 << 31)) & 0xFFFFFFFF) - (1 << 31)
    return _round_exact_bf16(Fraction(total) * _pow2(anchor - 31))


def evaluate_mxu1_finite_reset(node, operands: tuple[TensorValue, ...]) -> TensorValue:
    """Verify a finite-normal MXU1 reset graph using the selected anchor path.

    This semantic primitive is a source-attributed port checked against a
    bounded Verilator lane test. It is not an AtlasCore instruction-path result.
    """
    if (
        node.op != "matmul" or len(operands) != 2 or node.attrs
        or node.index_maps != _MXU_MATMUL_MAPS
        or node.type.shape != (32, 32) or node.type.dtype != "bf16"
        or node.type.numerical_policy != _MXU1_FINITE_RESET_POLICY
        or any(operand.type.shape != (32, 32)
               or operand.type.dtype != "fp8_e4m3"
               or operand.type.numerical_policy != "raw-bit-pattern"
               for operand in operands)
    ):
        raise ValueError("MXU1 finite reset has an unsupported semantic contract")
    activation, weight = (operand.elements for operand in operands)
    output = tuple(
        _mxu1_anchor_reset_lane(
            weight[column * 32:(column + 1) * 32],
            activation[row * 32:(row + 1) * 32],
        )
        for row in range(32) for column in range(32)
    )
    return TensorValue(node.type, output)


def _normal_bf16(bits: int) -> Fraction:
    exponent = (bits >> 7) & 0xFF
    if not 0 < exponent < 255:
        raise ValueError("BF16 arithmetic requires finite normal operands")
    significand = (128 + (bits & 0x7F)) * (-1 if bits & 0x8000 else 1)
    shift = exponent - 134
    return Fraction(significand * (1 << shift)) if shift >= 0 else Fraction(significand, 1 << -shift)


def evaluate_bf16_row_max(node, operands: tuple[TensorValue, ...]) -> TensorValue:
    """Check the selected RTL's raw-bit row maximum and 32-lane broadcast."""
    if (
        node.op != "row_max_bf16" or len(operands) != 1
        or node.type.shape != (32, 32) or node.type.dtype != "bf16"
        or node.type.numerical_policy != "raw-bit-pattern"
        or operands[0].type != node.type or node.index_maps != _ROW_REDUCE_MAPS
        or dict(node.attrs) != _ROW_MAX_ATTRS
    ):
        raise ValueError("BF16 row maximum has an unsupported semantic contract")

    def key(code: int) -> int:
        if type(code) is not int or not 0 <= code <= 0xFFFF:
            raise ValueError("BF16 row maximum needs raw 16-bit encodings")
        return (~code & 0xFFFF) if code & 0x8000 else code ^ 0x8000

    result = []
    source = operands[0].elements
    for row in range(32):
        winner = max(source[32 * row:32 * (row + 1)], key=key)
        result.extend((winner,) * 32)
    return TensorValue(node.type, tuple(result))


def _round_f32_normal_or_zero(value: Fraction) -> Fraction:
    if not value:
        return Fraction(0)
    magnitude = abs(value)
    exponent = magnitude.numerator.bit_length() - magnitude.denominator.bit_length()
    if magnitude < _pow2(exponent):
        exponent -= 1
    if exponent > 127:
        raise ValueError("BF16 row sum overflows FP32")
    if exponent < -126:
        raise ValueError("BF16 row sum has a nonnormal FP32 tree value")
    step = _pow2(exponent - 23)
    scaled = magnitude / step
    integer, remainder = divmod(scaled.numerator, scaled.denominator)
    rounded = integer + (
        2 * remainder > scaled.denominator
        or (2 * remainder == scaled.denominator and integer % 2 == 1)
    )
    if rounded == 1 << 24:
        rounded = 1 << 23
        exponent += 1
    if exponent > 127:
        raise ValueError("BF16 row sum overflows FP32")
    return Fraction((-1 if value < 0 else 1) * rounded) * _pow2(exponent - 23)


def evaluate_bf16_row_sum(node, operands: tuple[TensorValue, ...]) -> TensorValue:
    """Check the selected 32-lane adjacent FP32 tree and BF16 final rounding."""
    if (
        node.op != "row_sum_bf16" or len(operands) != 1
        or node.type.shape != (32, 32) or node.type.dtype != "bf16"
        or node.type.numerical_policy != "raw-bit-pattern"
        or operands[0].type != node.type or node.index_maps != _ROW_REDUCE_MAPS
        or dict(node.attrs) != _ROW_SUM_ATTRS
    ):
        raise ValueError("BF16 row sum has an unsupported semantic contract")

    output = []
    source = operands[0].elements
    for row in range(32):
        values = []
        for code in source[row * 32:(row + 1) * 32]:
            if type(code) is not int or not 0 <= code <= 0xFFFF:
                raise ValueError("BF16 row sum needs raw 16-bit encodings")
            values.append(Fraction(0) if code == 0 else _normal_bf16(code))
        while len(values) > 1:
            values = [_round_f32_normal_or_zero(values[index] + values[index + 1])
                      for index in range(0, len(values), 2)]
        try:
            winner = _round_exact_bf16(values[0])
        except ValueError as exc:
            raise ValueError("BF16 row sum result is outside the admitted domain") from exc
        output.extend((winner,) * 32)
    return TensorValue(node.type, tuple(output))


def evaluate_bf16_exp(node, operands: tuple[TensorValue, ...]) -> TensorValue:
    """Replay the selected ExpLane Q9.12/LUT path on its qualified input domain."""
    if (
        node.op != "exp_bf16" or len(operands) != 1
        or node.type.shape != (32, 32) or node.type.dtype != "bf16"
        or node.type.numerical_policy != "raw-bit-pattern"
        or operands[0].type != node.type
        or node.index_maps != pointwise_identity_maps(2, 1)
        or dict(node.attrs) != _EXP_ATTRS
    ):
        raise ValueError("BF16 exponential has an unsupported semantic contract")
    return TensorValue(node.type, tuple(vexp_bf16_bits(code) for code in operands[0].elements))


def evaluate_bf16_recip(node, operands: tuple[TensorValue, ...]) -> TensorValue:
    """Replay the selected reciprocal LUT on positive normal BF16 in [1, 32]."""
    if (
        node.op != "recip_bf16" or len(operands) != 1
        or node.type.shape != (32, 32) or node.type.dtype != "bf16"
        or node.type.numerical_policy != "raw-bit-pattern"
        or operands[0].type != node.type
        or node.index_maps != pointwise_identity_maps(2, 1)
        or dict(node.attrs) != _RECIP_ATTRS
    ):
        raise ValueError("BF16 reciprocal has an unsupported semantic contract")
    return TensorValue(node.type, tuple(vrecip_bf16_bits(code) for code in operands[0].elements))


def evaluate_bf16_add(node, operands: tuple[TensorValue, ...]) -> TensorValue:
    """Check exact-FP32 sums and the selected VPU's final BF16 bit chop."""
    if (
        node.op != "add_bf16" or len(operands) != 2
        or node.type.dtype != "bf16" or node.type.numerical_policy != "raw-bit-pattern"
        or any(operand.type != node.type for operand in operands)
        or node.index_maps != pointwise_identity_maps(len(node.type.shape), 2)
        or dict(node.attrs) != _ADD_ATTRS
    ):
        raise ValueError("BF16 add has an unsupported semantic contract")
    output = []
    for left, right in zip(operands[0].elements, operands[1].elements):
        exact = _normal_bf16(left) + _normal_bf16(right)
        if not exact:
            raise ValueError("BF16 add zero-sum sign is outside the admitted domain")
        try:
            f32 = struct.unpack("<f", struct.pack("<f", float(exact)))[0]
        except OverflowError as exc:
            raise ValueError("BF16 add overflows FP32") from exc
        if Fraction.from_float(f32) != exact:
            raise ValueError("BF16 add sum is not exactly representable in FP32")
        chopped = struct.unpack("<I", struct.pack("<f", f32))[0] >> 16
        if not 0 < (chopped & 0x7F80) < 0x7F80:
            raise ValueError("BF16 add result is outside finite normal domain")
        output.append(chopped)
    return TensorValue(node.type, tuple(output))


def evaluate_bf16_sub(node, operands: tuple[TensorValue, ...]) -> TensorValue:
    """Check exact-FP32 differences and the selected VPU's final BF16 bit chop.

    Equal finite-normal inputs produce +0 under the selected RNE mode. The
    other signed-zero and exceptional cases remain outside this domain.
    """
    if (
        node.op != "sub_bf16" or len(operands) != 2
        or node.type.dtype != "bf16" or node.type.numerical_policy != "raw-bit-pattern"
        or any(operand.type != node.type for operand in operands)
        or node.index_maps != pointwise_identity_maps(len(node.type.shape), 2)
        or dict(node.attrs) != _SUB_ATTRS
    ):
        raise ValueError("BF16 subtract has an unsupported semantic contract")
    output = []
    for left, right in zip(operands[0].elements, operands[1].elements):
        exact = _normal_bf16(left) - _normal_bf16(right)
        if not exact:
            output.append(0)
            continue
        try:
            f32 = struct.unpack("<f", struct.pack("<f", float(exact)))[0]
        except OverflowError as exc:
            raise ValueError("BF16 subtract overflows FP32") from exc
        if Fraction.from_float(f32) != exact:
            raise ValueError("BF16 subtract difference is not exactly representable in FP32")
        chopped = struct.unpack("<I", struct.pack("<f", f32))[0] >> 16
        if not 0 < (chopped & 0x7F80) < 0x7F80:
            raise ValueError("BF16 subtract result is outside finite normal domain")
        output.append(chopped)
    return TensorValue(node.type, tuple(output))


def _pow2(exponent: int) -> Fraction:
    return Fraction(1 << exponent) if exponent >= 0 else Fraction(1, 1 << -exponent)


def _round_bf16_product(left: int, right: int) -> int:
    exact = _normal_bf16(left) * _normal_bf16(right)
    magnitude = abs(exact)
    exponent = magnitude.numerator.bit_length() - magnitude.denominator.bit_length()
    if magnitude < _pow2(exponent):
        exponent -= 1
    if not -126 <= exponent <= 126:
        raise ValueError("BF16 product is outside the admitted normal range")
    scaled = magnitude / _pow2(exponent - 7)
    integer, remainder = divmod(scaled.numerator, scaled.denominator)
    rounded = integer + (
        2 * remainder > scaled.denominator
        or (2 * remainder == scaled.denominator and integer % 2 == 1)
    )
    if rounded == 256:
        rounded = 128
        exponent += 1
    if not -126 <= exponent <= 127:
        raise ValueError("BF16 product rounds outside the admitted normal range")
    return (0x8000 if exact < 0 else 0) | ((exponent + 127) << 7) | (rounded - 128)


def evaluate_bf16_mul(node, operands: tuple[TensorValue, ...]) -> TensorValue:
    """Check selected VMUL's finite-normal, nearest-even product domain."""
    if (
        node.op != "mul_bf16" or len(operands) != 2
        or node.type.dtype != "bf16" or node.type.numerical_policy != "raw-bit-pattern"
        or any(operand.type != node.type for operand in operands)
        or node.index_maps != pointwise_identity_maps(len(node.type.shape), 2)
        or dict(node.attrs) != _MUL_ATTRS
    ):
        raise ValueError("BF16 multiply has an unsupported semantic contract")
    return TensorValue(node.type, tuple(
        _round_bf16_product(left, right)
        for left, right in zip(operands[0].elements, operands[1].elements)
    ))


def evaluate_unit_pack(node, operands: tuple[TensorValue, ...]) -> TensorValue:
    """Check the seven exact BF16 codes admitted by the unit E8M0 pack rule."""
    if (
        node.op != "pack_fp8"
        or len(operands) != 1
        or node.type.dtype != "fp8_e4m3"
        or node.type.numerical_policy != _POLICY
        or operands[0].type.dtype != "bf16"
        or operands[0].type.numerical_policy != "raw-bit-pattern"
        or operands[0].type.shape != node.type.shape
        or node.index_maps != pointwise_identity_maps(len(node.type.shape), 1)
        or dict(node.attrs) != {"input_domain": _DOMAIN, "scale_code": 127}
    ):
        raise ValueError("unit pack has an unsupported semantic contract")
    try:
        packed = tuple(_BF16_TO_FP8_UNIT[bits] for bits in operands[0].elements)
    except KeyError as exc:
        raise ValueError("BF16 input is outside the seven-code unit-pack domain") from exc
    return TensorValue(node.type, packed)


def _normal_fp8_codes() -> dict[Fraction, int]:
    codes: dict[Fraction, int] = {}
    for code in range(256):
        exponent = (code >> 3) & 15
        if not 1 <= exponent <= 14:
            continue
        significand = (8 + (code & 7)) * (-1 if code & 0x80 else 1)
        codes[Fraction(significand) * _pow2(exponent - 10)] = code
    return codes


_FP8_NORMAL_CODES = _normal_fp8_codes()


def evaluate_scaled_pack(node, operands: tuple[TensorValue, ...]) -> TensorValue:
    """Check exact normal E4M3 conversion under a declared E8M0 scale."""
    attrs = dict(node.attrs)
    code = attrs.get("scale_code")
    if (
        node.op != "pack_fp8" or len(operands) != 1
        or node.type.dtype != "fp8_e4m3"
        or node.type.numerical_policy != _SCALED_POLICY
        or operands[0].type.dtype != "bf16"
        or operands[0].type.numerical_policy != "raw-bit-pattern"
        or operands[0].type.shape != node.type.shape
        or node.index_maps != pointwise_identity_maps(len(node.type.shape), 1)
        or set(attrs) != {"input_domain", "scale_code"}
        or attrs["input_domain"] != _SCALED_DOMAIN
        or type(code) is not int or not 122 <= code <= 128
    ):
        raise ValueError("scaled pack has an unsupported semantic contract")
    scale = _pow2(code - 127)
    output = []
    for bits in operands[0].elements:
        if bits == 0:
            output.append(0)
            continue
        try:
            output.append(_FP8_NORMAL_CODES[_normal_bf16(bits) / scale])
        except (KeyError, ValueError) as exc:
            raise ValueError("BF16 input is outside the exact scaled-pack domain") from exc
    return TensorValue(node.type, tuple(output))


def evaluate_rounded_scaled_pack(node, operands: tuple[TensorValue, ...]) -> TensorValue:
    """Check VPU pack's RTL rounding, flush, and clamp behavior on BF16 bits."""
    attrs = dict(node.attrs)
    code = attrs.get("scale_code")
    if (
        node.op != "pack_fp8" or len(operands) != 1
        or node.type.dtype != "fp8_e4m3"
        or node.type.numerical_policy != ROUNDED_PACK_POLICY
        or operands[0].type.dtype != "bf16"
        or operands[0].type.numerical_policy != "raw-bit-pattern"
        or operands[0].type.shape != node.type.shape
        or node.index_maps != pointwise_identity_maps(len(node.type.shape), 1)
        or set(attrs) != {"input_domain", "scale_code"}
        or attrs["input_domain"] != ROUNDED_PACK_DOMAIN
        or type(code) is not int or not 122 <= code <= 128
    ):
        raise ValueError("rounded scaled pack has an unsupported semantic contract")
    return TensorValue(node.type, tuple(
        pack_bf16_lane(bits, code) for bits in operands[0].elements
    ))
