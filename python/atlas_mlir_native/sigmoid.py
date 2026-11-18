"""Explicit bounded sigmoid precision using selected Atlas BF16 primitives."""

from dataclasses import dataclass, replace
from math import prod

from merlin.semantic_compiler.model import (
    ConstantBinding,
    KernelRequest,
    SemanticNode,
    TensorType,
    pointwise_identity_maps,
)
from merlin.semantic_compiler.pointwise_tiling import PointwiseTiling

from .binding import (
    BF16_BOUNDED_RECIP_ATTRS,
    BF16_EXP_ATTRS,
    BF16_SIGMOID_DOMAIN,
    BF16_ZERO_BINARY_ATTRS,
    TARGET_IDENTITY,
    AtlasNativeBinding,
)

SIGMOID_POLICY = "f32-to-bf16-bounded-sigmoid-exp-recip-v1"
SOURCE_ATTRS = tuple(
    sorted(
        {
            "literal_bits": "0x3f800000",
            "numeric_semantics": "source-f32-composite-unqualified-v1",
            "operation": "sigmoid_f32",
        }.items()
    )
)
SIGMOID_ATTRS = (
    BF16_SIGMOID_DOMAIN,
    ("numeric_semantics", "selected-rtl-bf16-sigmoid-exp-recip-v1"),
)
SILU_ATTRS = (SIGMOID_ATTRS[0], ("numeric_semantics", "selected-rtl-bf16-sigmoid-then-product-v1"))


def is_source_sigmoid(node):
    return (
        node.op == "source_f32_composite_unqualified"
        and dict(node.attrs) == dict(SOURCE_ATTRS)
        and node.effect == "pure"
        and len(node.inputs) == 1
        and node.type.dtype == "f32"
        and node.type.numerical_policy == "raw-bit-pattern"
        and node.index_maps == pointwise_identity_maps(len(node.type.shape), 1)
    )


def input_spec():
    from .pointwise import input_spec as bounded_spec

    # exp(-x) is positive and finite here. Chopping 1 + exp(-x) gives a
    # denominator in [1, 2^47], inside the existing reciprocal description.
    return replace(
        bounded_spec(),
        policy=SIGMOID_POLICY,
        admitted_code_ranges=(
            (0, 0),
            (0x2000, 0x4200),
            (0x8000, 0x8000),
            (0xA000, 0xC200),
        ),
    )


@dataclass(frozen=True)
class SigmoidPrecision:
    source: KernelRequest
    request: KernelRequest

    @property
    def operand_specs(self):
        return (input_spec(),)

    def record(self):
        return {
            "schema": "atlas.sigmoid_precision.v1",
            "source_request_digest": self.source.digest(),
            "request_digest": self.request.digest(),
            "inputs": list(self.source.nodes[-1].inputs),
            "input_conversions": [input_spec().record()],
            "policy": SIGMOID_POLICY,
            "output_conversion": "bf16-to-f32-exact",
            "qualification": "diagnostic_precision_change; original_model_quality_unmeasured",
        }


@dataclass(frozen=True)
class SiluPrecision(SigmoidPrecision):
    def record(self):
        from .pointwise import POINTWISE_POLICY

        return {
            **super().record(),
            "schema": "atlas.silu_precision.v1",
            "inputs": [self.source.nodes[0].id],
            "policies": [SIGMOID_POLICY, POINTWISE_POLICY],
            "intermediate_conversion": "BF16-to-f32-to-BF16 is bit-preserving on the admitted sigmoid range",
            "arithmetic": "existing sigmoid steps, then the existing BF16 product; no reassociation",
        }


def prepare_silu_source(source):
    from merlin.semantic_compiler.silu_source import recognize_silu

    sigmoid_id, output_id = recognize_silu(source, source.outputs[0])
    sigmoid, output = source.node(sigmoid_id), source.node(output_id)
    operand = source.node(sigmoid.inputs[0])
    if (len(source.nodes) != 3 or source.nodes != (operand, sigmoid, output)
            or source.outputs != (output.id,)):
        raise ValueError("SiLU needs one closed operand, sigmoid and product")
    precision = prepare_source(replace(source, nodes=(operand, sigmoid), outputs=(sigmoid.id,)))
    arg, result = precision.request.nodes
    request = replace(precision.request, nodes=(replace(arg, attrs=(BF16_SIGMOID_DOMAIN,)),
                                               replace(result, op="silu_bf16_lut", attrs=SILU_ATTRS)),
                      source_identity=source.digest())
    prepare_plan(request)
    return SiluPrecision(source, request)


def prepare_source(source):
    output = source.nodes[-1]
    if (
        not is_source_sigmoid(output)
        or source.target_identity != TARGET_IDENTITY
        or source.lowering_policy != "diagnostic"
        or source.physical_extents
        or source.boundary_layouts
        or source.output_layouts
        or source.outputs != (output.id,)
        or len(source.nodes) != 2
        or source.nodes[0].id != output.inputs[0]
    ):
        raise ValueError("sigmoid needs its exact closed diagnostic source")
    operand = source.nodes[0]
    if operand.effect not in {"input", "constant"} or operand.index_maps or operand.type != output.type:
        raise ValueError("sigmoid needs one equally shaped f32 boundary operand")
    type_ = TensorType(output.type.shape, "bf16", "raw-bit-pattern")
    arg = SemanticNode(
        "arg0",
        "input",
        (),
        type_,
        (BF16_ZERO_BINARY_ATTRS["sub_bf16_zero"][0],),
        "input",
    )
    result = SemanticNode(
        "result",
        "sigmoid_bf16_lut",
        (arg.id,),
        type_,
        SIGMOID_ATTRS,
        index_maps=output.index_maps,
    )
    request = KernelRequest(
        (arg, result),
        (result.id,),
        ("external",),
        ((arg.id, "external"),),
        TARGET_IDENTITY,
        "diagnostic",
        source.digest(),
    )
    prepare_plan(request)
    if operand.effect == "constant":
        import numpy as np

        binding = source.constants[0]
        if binding.encoding != "f32-le-raw":
            raise ValueError("sigmoid parameter needs exact f32 bytes")
        input_spec().encode(np.frombuffer(bytes.fromhex(binding.data_hex), dtype="<f4"))
    return SigmoidPrecision(source, request)


@dataclass(frozen=True)
class SigmoidTiling(PointwiseTiling):
    def body_request(self):
        tiled = super().body_request()
        arg, macro = tiled.nodes
        binary = pointwise_identity_maps(2, 2)
        unary = pointwise_identity_maps(2, 1)
        zero = SemanticNode(
            "zero",
            "constant",
            (),
            macro.type,
            (BF16_ZERO_BINARY_ATTRS["sub_bf16_zero"][0],),
            "constant",
        )
        one = SemanticNode(
            "one",
            "constant",
            (),
            macro.type,
            (BF16_ZERO_BINARY_ATTRS["add_bf16_zero"][0],),
            "constant",
        )
        negative = SemanticNode(
            "negative",
            "sub_bf16_zero",
            (zero.id, arg.id),
            macro.type,
            BF16_ZERO_BINARY_ATTRS["sub_bf16_zero"],
            index_maps=binary,
        )
        exponential = SemanticNode(
            "exponential",
            "exp_bf16",
            (negative.id,),
            macro.type,
            BF16_EXP_ATTRS,
            index_maps=unary,
        )
        denominator = SemanticNode(
            "denominator",
            "add_bf16_zero",
            (one.id, exponential.id),
            macro.type,
            BF16_ZERO_BINARY_ATTRS["add_bf16_zero"],
            index_maps=binary,
        )
        result = replace(
            macro,
            op="recip_bf16_bounded",
            inputs=(denominator.id,),
            attrs=BF16_BOUNDED_RECIP_ATTRS,
            index_maps=unary,
        )
        constants = tuple(
            ConstantBinding(n.id, "bf16-le-raw", bits * prod(macro.type.shape))
            for n, bits in ((zero, "0000"), (one, "803f"))
        )
        if macro.op == "silu_bf16_lut":
            sigmoid = replace(result, id="sigmoid")
            product = replace(macro, op="mul_bf16_zero", inputs=(arg.id, sigmoid.id),
                              attrs=BF16_ZERO_BINARY_ATTRS["mul_bf16_zero"], index_maps=binary)
            return replace(tiled, nodes=(arg, zero, one, negative, exponential, denominator, sigmoid, product),
                           constants=constants)
        return replace(
            tiled,
            nodes=(arg, zero, one, negative, exponential, denominator, result),
            constants=constants,
        )


def prepare_plan(request):
    if (
        len(request.nodes) != 2
        or request.target_identity != TARGET_IDENTITY
        or request.lowering_policy != "diagnostic"
        or request.nodes[-1].op not in {"sigmoid_bf16_lut", "silu_bf16_lut"}
        or request.nodes[-1].attrs != (SILU_ATTRS if request.nodes[-1].op == "silu_bf16_lut" else SIGMOID_ATTRS)
        or request.nodes[-1].index_maps != pointwise_identity_maps(len(request.nodes[-1].type.shape), 1)
        or request.nodes[0].type != request.nodes[-1].type
        or request.nodes[0].attrs != ((BF16_SIGMOID_DOMAIN,) if request.nodes[-1].op == "silu_bf16_lut"
                                     else (BF16_ZERO_BINARY_ATTRS["sub_bf16_zero"][0],))
        or any(n.type.dtype != "bf16" or n.type.numerical_policy != "raw-bit-pattern" for n in request.nodes)
        or request.input_storages != ((request.nodes[0].id, "external"),)
        or request.output_storages != ("external",)
    ):
        raise ValueError("sigmoid request differs from its selected BF16 decomposition")
    descriptor = next(d for d in AtlasNativeBinding.profile().descriptors if d.computation == "recip_bf16_bounded")
    shape = tuple(axis.minimum for axis in descriptor.output_axis_bounds)
    return SigmoidTiling(request, shape, 2, b"\0\0")
