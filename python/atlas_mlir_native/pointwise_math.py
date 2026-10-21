"""Explicit bounded BF16 decompositions for source division and rsqrt."""

from __future__ import annotations

from dataclasses import dataclass, replace

from merlin.semantic_compiler.model import (
    KernelRequest,
    SemanticNode,
    TensorType,
    pointwise_identity_maps,
)
from merlin.semantic_compiler.pointwise_tiling import PointwiseTiling

from .binding import (
    BF16_BOUNDED_RECIP_ATTRS,
    BF16_SQRT_ATTRS,
    BF16_ZERO_BINARY_ATTRS,
    TARGET_IDENTITY,
    AtlasNativeBinding,
)

MATH_POLICY = "f32-to-bf16-positive-denominator-lut-div-rsqrt-v1"
_OPERATIONS = {
    "f32_div": "divide_bf16_lut",
    "source_f32_rsqrt_unqualified": "rsqrt_bf16_lut",
}
_ATTRS = (
    ("input_domain", "bf16-bounded-pow2-minus63-plus63-positive-denominator-v1"),
    ("numeric_semantics", "selected-rtl-bf16-lut-div-rsqrt-v1"),
)


def input_specs(operation):
    from .pointwise import input_spec

    if operation not in _OPERATIONS.values():
        raise ValueError("unknown pointwise LUT operation")
    bounded = replace(input_spec(), policy=MATH_POLICY + "-bounded")
    positive = replace(
        bounded,
        policy=MATH_POLICY + "-positive",
        admitted_code_ranges=(bounded.admitted_code_ranges[1],),
    )
    return (bounded, positive) if operation == "divide_bf16_lut" else (positive,)


@dataclass(frozen=True)
class MathPointwisePrecision:
    source: KernelRequest
    request: KernelRequest

    @property
    def operand_specs(self):
        return input_specs(self.request.nodes[-1].op)

    def record(self):
        return {
            "schema": "atlas.pointwise_math_precision.v1",
            "source_request_digest": self.source.digest(),
            "request_digest": self.request.digest(),
            "inputs": list(self.source.nodes[-1].inputs),
            "input_conversions": [spec.record() for spec in self.operand_specs],
            "policy": MATH_POLICY,
            "output_conversion": "bf16-to-f32-exact",
            "qualification": "diagnostic_precision_change; original_model_quality_unmeasured",
        }


def prepare_source(source):
    output = source.nodes[-1]
    arity = 2 if output.op == "f32_div" else 1
    if (
        output.op not in _OPERATIONS
        or source.target_identity != TARGET_IDENTITY
        or source.lowering_policy != "diagnostic"
        or source.physical_extents
        or source.boundary_layouts
        or source.output_layouts
        or len(output.inputs) != arity
        or source.outputs != (output.id,)
        or {node.id for node in source.nodes} != {*output.inputs, output.id}
        or any(node.type.dtype != "f32" or node.type.numerical_policy != "raw-bit-pattern" for node in source.nodes)
        or output.effect != "pure"
        or dict(output.attrs)
        != {"numeric_semantics": "ieee754-f32-div-rne-v1" if arity == 2 else "source-f32-rsqrt-unqualified-v1"}
    ):
        raise ValueError("pointwise math needs a closed diagnostic division or rsqrt source")
    operands = tuple(source.node(key) for key in output.inputs)
    if any(node.effect not in {"input", "constant"} or node.index_maps for node in operands):
        raise ValueError("pointwise math operands must be explicit boundary values")
    operation = _OPERATIONS[output.op]
    inputs = tuple(
        SemanticNode(
            f"arg{i}",
            "input",
            (),
            TensorType(node.type.shape, "bf16", "raw-bit-pattern"),
            ((BF16_ZERO_BINARY_ATTRS["mul_bf16_zero"] if arity == 2 and i == 0 else BF16_BOUNDED_RECIP_ATTRS)[0],),
            "input",
        )
        for i, node in enumerate(operands)
    )
    result = SemanticNode(
        "result",
        operation,
        tuple(node.id for node in inputs),
        TensorType(output.type.shape, "bf16", "raw-bit-pattern"),
        _ATTRS,
        index_maps=output.index_maps,
    )
    request = KernelRequest(
        (*inputs, result),
        (result.id,),
        ("external",),
        tuple((node.id, "external") for node in inputs),
        TARGET_IDENTITY,
        "diagnostic",
        source.digest(),
    )
    prepare_plan(request)
    import numpy as np

    bindings = {item.node_id: item for item in source.constants}
    for node, spec in zip(operands, input_specs(operation), strict=True):
        if node.effect == "constant":
            binding = bindings[node.id]
            if binding.encoding != "f32-le-raw":
                raise ValueError("pointwise math parameter needs exact f32 bytes")
            spec.encode(np.frombuffer(bytes.fromhex(binding.data_hex), dtype="<f4"))
    return MathPointwisePrecision(source, request)


@dataclass(frozen=True)
class MathPointwiseTiling(PointwiseTiling):
    def body_request(self):
        tiled = super().body_request()
        *inputs, macro = tiled.nodes
        if macro.op == "divide_bf16_lut":
            intermediate = SemanticNode(
                "reciprocal",
                "recip_bf16_bounded",
                (inputs[1].id,),
                macro.type,
                BF16_BOUNDED_RECIP_ATTRS,
                index_maps=pointwise_identity_maps(2, 1),
            )
            result = replace(
                macro,
                op="mul_bf16_zero",
                inputs=(inputs[0].id, intermediate.id),
                attrs=BF16_ZERO_BINARY_ATTRS["mul_bf16_zero"],
                index_maps=pointwise_identity_maps(2, 2),
            )
        elif macro.op == "rsqrt_bf16_lut":
            intermediate = SemanticNode(
                "square_root",
                "sqrt_bf16_bounded",
                (inputs[0].id,),
                macro.type,
                BF16_SQRT_ATTRS,
                index_maps=pointwise_identity_maps(2, 1),
            )
            result = replace(
                macro,
                op="recip_bf16_bounded",
                inputs=(intermediate.id,),
                attrs=BF16_BOUNDED_RECIP_ATTRS,
                index_maps=pointwise_identity_maps(2, 1),
            )
        else:
            raise ValueError("unknown pointwise LUT decomposition")
        return replace(tiled, nodes=(*inputs, intermediate, result))


def prepare_plan(request):
    if (
        request.target_identity != TARGET_IDENTITY
        or request.lowering_policy != "diagnostic"
        or request.nodes[-1].op not in _OPERATIONS.values()
    ):
        raise ValueError("pointwise math request needs its selected BF16 decomposition")
    *inputs, output = request.nodes
    arity = 2 if output.op == "divide_bf16_lut" else 1
    attrs = (
        ((BF16_ZERO_BINARY_ATTRS["mul_bf16_zero"][0],), (BF16_BOUNDED_RECIP_ATTRS[0],))
        if arity == 2
        else ((BF16_BOUNDED_RECIP_ATTRS[0],),)
    )
    if (
        len(inputs) != arity
        or output.attrs != _ATTRS
        or any(n.type.dtype != "bf16" or n.type.numerical_policy != "raw-bit-pattern" for n in request.nodes)
        or tuple(node.attrs for node in inputs) != attrs
        or request.output_storages != ("external",)
        or request.input_storages != tuple((node.id, "external") for node in inputs)
    ):
        raise ValueError("pointwise math types, attributes or storage differ")
    descriptor = next(d for d in AtlasNativeBinding.profile().descriptors if d.computation == "recip_bf16_bounded")
    shape = tuple(axis.minimum for axis in descriptor.output_axis_bounds)
    return MathPointwiseTiling(
        request,
        shape,
        2,
        b"\0\0",
        (b"\0\0", b"\x80\x3f") if arity == 2 else (b"\x80\x3f",),
    )


def selected_steps(plan, selected):
    from .binding import BF16_BINARY_MODES, BF16_UNARY_MODES

    profile = {
        d.computation: d
        for d in AtlasNativeBinding.profile().descriptors
        if d.name in BF16_UNARY_MODES or d.name in BF16_BINARY_MODES
    }
    steps = [node for node in plan.body_request().nodes if node.effect == "pure"]
    expected = [profile[node.op].name for node in steps]
    actual = [
        v
        for v in selected["values"]
        if v["descriptor"] is not None and v["descriptor"] not in {"atlas.load_bf16_tile", "atlas.store_bf16_tile"}
    ]
    if (
        selected.get("engine") != "merlin_native"
        or selected.get("target_identity") != TARGET_IDENTITY
        or selected.get("request_digest") != plan.body_request().digest()
        # A branched pure DAG may be selected in another legal topological
        # order. Compare source instances, retaining multiplicity so duplicated
        # or dropped computation cannot pass this handoff check.
        or sorted((v["source_node"], v["descriptor"]) for v in actual)
        != sorted((n.id, d) for n, d in zip(steps, expected, strict=True))
    ):
        raise ValueError("pointwise math differs from its native selected decomposition")
    return expected


def virtual_steps(plan, descriptors):
    from .binding import BF16_BINARY_MODES, BF16_UNARY_MODES

    body = plan.body_request()
    names = {node.id: name for node, name in zip(body.nodes, ("%a", "%b")) if node.effect == "input"}
    result = []
    constants = {item.node_id: item for item in body.constants}
    for node in body.nodes:
        if node.effect != "constant":
            continue
        binding = constants[node.id]
        payload = bytes.fromhex(binding.data_hex)
        if binding.encoding != "bf16-le-raw" or not payload or payload != payload[:2] * (len(payload) // 2):
            raise ValueError("pointwise splat needs a checked uniform BF16 constant")
        name = f"%constant_{len(names)}"
        bits = int.from_bytes(payload[:2], "little")
        result.append(f'    {name} = "atlas.virtual_vpu_splat"() {{bits = {bits} : i32}} : () -> !atlas.virtual_bf16\n')
        names[node.id] = name
    steps = [node for node in body.nodes if node.effect == "pure"]
    for index, (node, descriptor) in enumerate(zip(steps, descriptors, strict=True)):
        name = "%r" if index == len(steps) - 1 else f"%math_{index}"
        operands = ",".join(names[key] for key in node.inputs)
        unary = descriptor in BF16_UNARY_MODES
        kind = BF16_UNARY_MODES[descriptor] if unary else BF16_BINARY_MODES[descriptor]
        tensor = "!atlas.virtual_bf16"
        types = ",".join([tensor] * len(node.inputs))
        result.append(
            f'    {name} = "atlas.virtual_vpu_{"unary" if unary else "binary"}"({operands}) {{kind = "{kind}"}} : ({types}) -> {tensor}\n'
        )
        names[node.id] = name
    return "".join(result)
