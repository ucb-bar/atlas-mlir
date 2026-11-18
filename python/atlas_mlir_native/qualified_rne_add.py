"""Explicit bounded BF16 RNE addition as a selected semantic expression DAG.

The VPU add primitive rounds to FP32 and then chops to BF16. This module
derives a correction from its residual, rather than relabeling that primitive.
No instruction schedule or physical register assignment is specified here.

For nonzero s=chop(f32_rne(a+b)), let x be the operand of greater magnitude
and y the other operand. Sterbenz subtraction gives s-x exactly: when s and
x differ by more than a factor of two, the original cancellation already
made a+b exactly BF16. The residual y-(s-x), however, need not be BF16:
an opposite-sign small operand immediately below half an ULP can produce
a residual immediately above half an ULP, which chopping turns into a tie.
An FP32 rounding across a BF16 boundary can make e point toward zero; it is
then strictly below half an ULP, and no correction is needed.

For M=abs(s), chop(M+M/128)-M is its exact outward ULP u. An exact reciprocal
of this power of two exposes the integer significand m=M/u in [128,255].
Since the BF16 spacing on [192,256) is one, chop(128+m/2)-128 is floor(m/2).
The parity m-2*floor(m/2) resolves midpoint ties. Saturated tanh implements
the three-way residual comparison; it never approximates a nonzero sign.
On a chopped-residual tie, small-magnitude < u/2 means round outward even
when m is even. To see why, orient the sum positive, with L>=B>=0. For
opposite signs D=L-s is an integer multiple k*u. If k>=2, B>D-u>=D/2,
so Sterbenz makes D-B exact. It is also exact for k=1 and B>=u/2. The only
inexact residual class is k=1 and B<u/2, whose true residual exceeds u/2.
For same signs the residual is at most B, so B<u/2 cannot manufacture a tie.
The additional midpoint check therefore repairs the lost residual bit without
changing real ties, exact cancellation, or an FP32 boundary overshoot.
Zero uses a positive proxy only for ULP extraction, and the final product
restores the original sum's signed zero. Inputs outside the declared bounded
normal-or-signed-zero domain are deliberately not admitted.
"""

from dataclasses import dataclass, replace
from math import prod

from merlin.semantic_compiler.model import (
    ConstantBinding, KernelRequest, SemanticNode, TensorType, pointwise_identity_maps,
)
from merlin.semantic_compiler.pointwise_tiling import PointwiseTiling
from merlin.semantic_compiler.precision_values import FloatBoundarySpec

from .binding import (
    BF16_BINARY_ATTRS, BF16_BOUNDED_RECIP_ATTRS, BF16_TANH_ATTRS,
    TARGET_IDENTITY, AtlasNativeBinding,
)

RNE_ADD_POLICY = "bounded-bf16-rne-add-residual-ulp-parity-v1"
INPUT_DOMAIN = ("input_domain", "bf16-signed-zero-or-normal-pow2-minus32-plus32-v1")
RNE_ADD_ATTRS = (INPUT_DOMAIN, ("numeric_semantics", RNE_ADD_POLICY))


def input_spec():
    return FloatBoundarySpec(
        RNE_ADD_POLICY, "bf16", "encode",
        admitted_code_ranges=((0, 0), (0x2F80, 0x4F80),
                              (0x8000, 0x8000), (0xAF80, 0xCF80)),
    )


@dataclass(frozen=True)
class RneAddPrecision:
    source: KernelRequest
    request: KernelRequest

    @property
    def operand_specs(self):
        return tuple(input_spec() for _ in self.request.nodes[:-1])

    def record(self):
        bf16 = self.source.nodes[-1].type.dtype == "bf16"
        return {
            "schema": "atlas.rne_add_precision.v1", "policy": RNE_ADD_POLICY,
            "source_request_digest": self.source.digest(),
            "request_digest": self.request.digest(),
            "inputs": list(dict.fromkeys(self.source.nodes[-1].inputs)),
            "input_conversions": [spec.record() for spec in self.operand_specs],
            "input_conversion": "bf16-bit-preserving; checked code domain" if bf16 else "f32-to-bf16-rne",
            "output_conversion": "bf16-bit-preserving" if bf16 else "bf16-to-f32-exact",
            "arithmetic": "ieee754-bf16-add-rne-v1",
            "qualification": (
                "conditional_source_equivalence; bounded finite-normal-or-signed-zero inputs"
                if bf16 else "diagnostic_precision_change; original_model_quality_unmeasured"
            ),
        }


def prepare_source(source):
    """Explicitly select RNE BF16 arithmetic for one closed add source region."""
    output = source.nodes[-1]
    dtype = output.type.dtype
    keys = tuple(dict.fromkeys(output.inputs))
    if (
        dtype not in {"f32", "bf16"} or output.op != dtype + "_add"
        or output.effect != "pure" or len(output.inputs) != 2
        or dict(output.attrs) != {"numeric_semantics": f"ieee754-{dtype}-add-rne-v1"}
        or source.target_identity != TARGET_IDENTITY or source.lowering_policy != "diagnostic"
        or source.physical_extents or source.boundary_layouts or source.output_layouts
        or source.outputs != (output.id,)
        or {node.id for node in source.nodes} != {*keys, output.id}
        or output.index_maps != pointwise_identity_maps(len(output.type.shape), 2)
        or any(source.node(key).type != output.type
               or source.node(key).effect not in {"input", "constant"}
               or source.node(key).index_maps for key in keys)
        or output.type.numerical_policy != "raw-bit-pattern"
    ):
        raise ValueError("RNE add needs one closed bounded f32 or bf16 source addition")
    tensor = TensorType(output.type.shape, "bf16", "raw-bit-pattern")
    inputs = tuple(SemanticNode(f"arg{i}", "input", (), tensor, (INPUT_DOMAIN,), "input")
                   for i, _ in enumerate(keys))
    result = SemanticNode(
        "result", "rne_add_bf16", tuple(inputs[keys.index(key)].id for key in output.inputs),
        tensor, RNE_ADD_ATTRS, index_maps=output.index_maps,
    )
    request = KernelRequest((*inputs, result), (result.id,), ("external",),
                            tuple((node.id, "external") for node in inputs),
                            TARGET_IDENTITY, "diagnostic", source.digest())
    prepare_plan(request)
    if source.constants:
        import numpy as np

        for binding in source.constants:
            if binding.encoding != dtype + "-le-raw":
                raise ValueError("RNE add parameter needs exact source bytes")
            data = bytes.fromhex(binding.data_hex)
            values = (np.frombuffer(data, dtype="<f4") if dtype == "f32" else
                      (np.frombuffer(data, dtype="<u2").astype("<u4") << 16).view("<f4"))
            input_spec().encode(values)
    return RneAddPrecision(source, request)


@dataclass(frozen=True)
class RneAddTiling(PointwiseTiling):
    def body_request(self):
        tiled = super().body_request()
        *inputs, macro = tiled.nodes
        a_input, b_input = macro.inputs
        constants = {
            "negative_one": 0xBF80, "half": 0x3F00, "one": 0x3F80,
            "unit": 0x3F80, "one_over_128": 0x3C00,
            "integer_bias_add": 0x4300, "integer_bias_sub": 0x4300,
            "two": 0x4000, "small_positive": 0x2B80, "sign_scale": 0x5880,
        }
        # small_positive=2^-40, sign_scale=2^50. The smallest nonzero sum
        # is 2^-39; the zero proxy's half ULP is 2^-48. Every nonzero comparison
        # argument, after scaling, has magnitude at least four and VTANH
        # saturates to exactly +/-1. All products remain finite and normal.
        # Boundary facts match each constant's actual primitive consumer. The
        # two bias occurrences have different add/subtract operand premises.
        constant_ops = {"one": "sub_bf16_rne", "integer_bias_add": "add_bf16_rne",
                        "integer_bias_sub": "sub_bf16_rne", "small_positive": "max_bf16"}
        nodes = [*inputs, *(SemanticNode(
            name, "constant", (), macro.type,
            (BF16_BINARY_ATTRS[constant_ops.get(name, "mul_bf16_zero")][0],), "constant",
        ) for name in constants)]

        def emit(name, operation, *operands):
            attrs = (BF16_TANH_ATTRS if operation == "tanh_bf16" else
                     BF16_BOUNDED_RECIP_ATTRS if operation == "recip_bf16_bounded" else
                     BF16_BINARY_ATTRS[operation])
            nodes.append(SemanticNode(name, operation, operands, macro.type, attrs,
                                      index_maps=pointwise_identity_maps(2, len(operands))))
            return name

        def sign(name, value):
            return emit(name, "tanh_bf16", emit(name + "_scaled", "mul_bf16_zero", value, "sign_scale"))

        def absolute(name, value):
            return emit(name, "max_bf16", value, emit(name + "_negative", "mul_bf16_zero", value, "negative_one"))

        a = emit("left_operand", "mul_bf16_zero", a_input, "unit")
        b = emit("right_operand", "mul_bf16_zero", b_input, "unit")
        s = emit("truncated_sum", "add_bf16_rne", a, b)
        sign_s = sign("sum_sign", s)
        abs_a, abs_b = absolute("abs_a", a), absolute("abs_b", b)
        large = emit("larger_magnitude", "max_bf16", abs_a, abs_b)
        negative_a = emit("negative_abs_a", "mul_bf16_zero", abs_a, "negative_one")
        negative_b = emit("negative_abs_b", "mul_bf16_zero", abs_b, "negative_one")
        small = emit("smaller_magnitude", "mul_bf16_zero",
                     emit("negative_minimum", "max_bf16", negative_a, negative_b), "negative_one")
        large_signed = emit("larger_operand", "mul_bf16_zero", large, sign_s)
        sign_product = emit("operand_sign_product", "mul_bf16_zero", sign("a_sign", a), sign("b_sign", b))
        small_sign = emit("smaller_sign", "mul_bf16_zero", sign_s, sign_product)
        small_signed = emit("smaller_operand", "mul_bf16_zero", small, small_sign)
        recovered_small = emit("recovered_smaller", "sub_bf16_rne", s, large_signed)
        residual = emit("signed_residual", "sub_bf16_rne", small_signed, recovered_small)
        remainder = emit("magnitude_residual", "mul_bf16_zero", residual, sign_s)

        magnitude = absolute("sum_magnitude", s)
        proxy = emit("positive_magnitude", "max_bf16", magnitude, "small_positive")
        step = emit("exponent_probe", "mul_bf16_zero", proxy, "one_over_128")
        next_value = emit("next_magnitude", "add_bf16_rne", proxy, step)
        ulp = emit("ulp", "sub_bf16_rne", next_value, proxy)
        inverse_ulp = emit("inverse_ulp", "recip_bf16_bounded", ulp)
        significand = emit("integer_significand", "mul_bf16_zero", proxy, inverse_ulp)
        half_significand = emit("half_significand", "mul_bf16_zero", significand, "half")
        biased = emit("truncated_biased_half", "add_bf16_rne", "integer_bias_add", half_significand)
        half_floor = emit("half_floor", "sub_bf16_rne", biased, "integer_bias_sub")
        even = emit("even_significand", "mul_bf16_zero", half_floor, "two")
        parity = emit("significand_parity", "sub_bf16_rne", significand, even)
        halfway = emit("half_ulp", "mul_bf16_zero", ulp, "half")
        comparison = sign("halfway_comparison", emit("halfway_difference", "sub_bf16_rne", remainder, halfway))
        nonzero = emit("comparison_nonzero", "mul_bf16_zero", comparison, comparison)
        positive = emit("comparison_positive", "mul_bf16_zero",
                        emit("positive_twice", "add_bf16_rne", comparison, nonzero), "half")
        equal = emit("comparison_equal", "sub_bf16_rne", "one", nonzero)
        small_half_difference = emit("small_half_difference", "sub_bf16_rne", halfway, small)
        # parity is +0 or +1, so this maximum also clamps a negative sign
        # (including -0) without a separate zero constant or maximum node.
        tie_direction = emit("tie_direction", "max_bf16", parity,
                             sign("small_half_sign", small_half_difference))
        tie_up = emit("tie_rounds_up", "mul_bf16_zero", equal, tie_direction)
        increment = emit("increment", "add_bf16_rne", positive, tie_up)
        correction = emit("correction", "mul_bf16_zero", increment, ulp)
        rounded = emit("rounded_magnitude", "add_bf16_rne", magnitude, correction)
        emit(macro.id, "mul_bf16_zero", rounded, sign_s)
        bindings = tuple(ConstantBinding(name, "bf16-le-raw", bits.to_bytes(2, "little").hex() * prod(macro.type.shape))
                         for name, bits in constants.items())
        return replace(tiled, nodes=tuple(nodes), constants=bindings)


def prepare_plan(request):
    *inputs, output = request.nodes
    if (
        not inputs or len(inputs) > 2 or output.op != "rne_add_bf16"
        or request.target_identity != TARGET_IDENTITY or request.lowering_policy != "diagnostic"
        or output.attrs != RNE_ADD_ATTRS or len(output.inputs) != 2
        or output.index_maps != pointwise_identity_maps(len(output.type.shape), 2)
        or any(node.type != output.type or node.attrs != (INPUT_DOMAIN,) for node in inputs)
        or output.type.dtype != "bf16" or output.type.numerical_policy != "raw-bit-pattern"
        or request.input_storages != tuple((node.id, "external") for node in inputs)
        or request.output_storages != ("external",)
    ):
        raise ValueError("RNE add request differs from its bounded semantic decomposition")
    descriptor = next(d for d in AtlasNativeBinding.profile().descriptors if d.computation == "max_bf16")
    shape = tuple(axis.minimum for axis in descriptor.output_axis_bounds)
    return RneAddTiling(request, shape, 2, b"\0\0")
