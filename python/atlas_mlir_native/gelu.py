"""Explicit bounded tanh-GELU precision using the selected Atlas VPU."""

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
from merlin.semantic_compiler.precision_values import FloatBoundarySpec

from .binding import (
    BF16_ACTIVATION_DOMAIN,
    BF16_BINARY_ATTRS,
    BF16_CUBE_ATTRS,
    BF16_TANH_ATTRS,
    TARGET_IDENTITY,
    AtlasNativeBinding,
)

GELU_POLICY = "f32-to-bf16-bounded-tanh-gelu-cube-chop-v1"
SOURCE_ATTRS = tuple(sorted({
    "operation": "tanh_gelu_f32",
    "numeric_semantics": "source-f32-composite-unqualified-v1",
    "literal_bits": "0x3f000000,0x3f800000,0x3d372713,0x3f4c422a",
}.items()))
GELU_ATTRS = (BF16_ACTIVATION_DOMAIN,
              ("numeric_semantics", "selected-rtl-bf16-tanh-gelu-cube-chop-v1"))


def is_source_gelu(node):
    return (
        node.op == "source_f32_composite_unqualified" and node.effect == "pure"
        and dict(node.attrs) == dict(SOURCE_ATTRS) and len(node.inputs) == 1
        and node.type.dtype == "f32" and node.type.numerical_policy == "raw-bit-pattern"
        and node.index_maps == pointwise_identity_maps(len(node.type.shape), 1)
    )


def input_spec():
    # |x| in [2^-32, 2^32], or signed zero. The chopped cube is normal,
    # coefficient products stay normal, and 1+tanh is zero or at least 2^-8.
    # Thus the selected binary primitives never need subnormal results.
    return FloatBoundarySpec(
        GELU_POLICY, "bf16", "encode",
        admitted_code_ranges=((0, 0), (0x2F80, 0x4F80),
                              (0x8000, 0x8000), (0xAF80, 0xCF80)),
    )


@dataclass(frozen=True)
class GeluPrecision:
    source: KernelRequest
    request: KernelRequest

    @property
    def operand_specs(self):
        return (input_spec(),)

    def record(self):
        return {
            "schema": "atlas.gelu_precision.v1", "policy": GELU_POLICY,
            "source_request_digest": self.source.digest(), "request_digest": self.request.digest(),
            "inputs": list(self.source.nodes[-1].inputs),
            "input_conversions": [input_spec().record()],
            "output_conversion": "bf16-to-f32-exact",
            "coefficient_bf16_bits": {"half": 0x3F00, "one": 0x3F80, "cubic": 0x3D37, "root": 0x3F4C},
            "qualification": "diagnostic_precision_change; original_model_quality_unmeasured",
        }


def prepare_source(source):
    output = source.nodes[-1]
    if (
        not is_source_gelu(output) or len(source.nodes) != 2
        or source.target_identity != TARGET_IDENTITY or source.lowering_policy != "diagnostic"
        or source.physical_extents or source.boundary_layouts or source.output_layouts
        or source.outputs != (output.id,) or source.nodes[0].id != output.inputs[0]
        or source.nodes[0].effect not in {"input", "constant"}
        or source.nodes[0].index_maps or source.nodes[0].type != output.type
    ):
        raise ValueError("tanh GELU needs its exact closed diagnostic f32 source")
    type_ = TensorType(output.type.shape, "bf16", "raw-bit-pattern")
    arg = SemanticNode("arg0", "input", (), type_, (BF16_ACTIVATION_DOMAIN,), "input")
    result = SemanticNode("result", "tanh_gelu_bf16_lut", (arg.id,), type_, GELU_ATTRS,
                          index_maps=output.index_maps)
    request = KernelRequest((arg, result), (result.id,), ("external",),
                            ((arg.id, "external"),), TARGET_IDENTITY, "diagnostic", source.digest())
    prepare_plan(request)
    if source.nodes[0].effect == "constant":
        import numpy as np

        binding = next(item for item in source.constants if item.node_id == source.nodes[0].id)
        if binding.encoding != "f32-le-raw":
            raise ValueError("GELU parameter needs exact f32 bytes")
        input_spec().encode(np.frombuffer(bytes.fromhex(binding.data_hex), dtype="<f4"))
    return GeluPrecision(source, request)


@dataclass(frozen=True)
class GeluTiling(PointwiseTiling):
    def body_request(self):
        tiled = super().body_request()
        arg, macro = tiled.nodes
        constants = tuple(SemanticNode(name, "constant", (), macro.type, (BF16_ACTIVATION_DOMAIN,), "constant")
                          for name in ("half", "one", "cubic", "root"))
        nodes = [arg, *constants]
        # Keep x live for both the cubic input sum and final half*x factor.
        # VCUBE and every following rounding boundary belong to this policy.
        for name, op, operands, attrs in (
            ("cube", "cube_bf16", (arg.id,), BF16_CUBE_ATTRS),
            ("scaled_cube", "mul_bf16_zero", ("cube", "cubic"), BF16_BINARY_ATTRS["mul_bf16_zero"]),
            ("inner_sum", "add_bf16_rne", (arg.id, "scaled_cube"), BF16_BINARY_ATTRS["add_bf16_rne"]),
            ("inner", "mul_bf16_zero", ("inner_sum", "root"), BF16_BINARY_ATTRS["mul_bf16_zero"]),
            ("tanh", "tanh_bf16", ("inner",), BF16_TANH_ATTRS),
            ("outer_sum", "add_bf16_rne", ("one", "tanh"), BF16_BINARY_ATTRS["add_bf16_rne"]),
            ("half_x", "mul_bf16_zero", (arg.id, "half"), BF16_BINARY_ATTRS["mul_bf16_zero"]),
            (macro.id, "mul_bf16_zero", ("half_x", "outer_sum"), BF16_BINARY_ATTRS["mul_bf16_zero"]),
        ):
            nodes.append(SemanticNode(name, op, operands, macro.type, attrs,
                                      index_maps=pointwise_identity_maps(2, len(operands))))
        bindings = tuple(ConstantBinding(node.id, "bf16-le-raw", bits * prod(macro.type.shape))
                         for node, bits in zip(constants, ("003f", "803f", "373d", "4c3f"), strict=True))
        return replace(tiled, nodes=tuple(nodes), constants=bindings)


def prepare_plan(request):
    if (
        len(request.nodes) != 2 or request.target_identity != TARGET_IDENTITY
        or request.lowering_policy != "diagnostic"
        or request.nodes[-1].op != "tanh_gelu_bf16_lut" or request.nodes[-1].attrs != GELU_ATTRS
        or request.nodes[-1].index_maps != pointwise_identity_maps(len(request.nodes[-1].type.shape), 1)
        or request.nodes[0].type != request.nodes[-1].type
        or request.nodes[0].attrs != (BF16_ACTIVATION_DOMAIN,)
        or any(n.type.dtype != "bf16" or n.type.numerical_policy != "raw-bit-pattern" for n in request.nodes)
        or request.input_storages != ((request.nodes[0].id, "external"),)
        or request.output_storages != ("external",)
    ):
        raise ValueError("GELU request differs from its selected BF16 decomposition")
    descriptor = next(d for d in AtlasNativeBinding.profile().descriptors if d.computation == "cube_bf16")
    shape = tuple(axis.minimum for axis in descriptor.output_axis_bounds)
    return GeluTiling(request, shape, 2, b"\0\0")
