"""Selected precision for source regions in a diagnostic Atlas application."""

from __future__ import annotations

from dataclasses import replace
from hashlib import sha256

from merlin.compile.semantic_regions import extract_semantic_region
from merlin.semantic_compiler.contraction_precision import lower_semantic_contraction
from merlin.semantic_compiler.model import SemanticNode, pointwise_identity_maps
from merlin.semantic_compiler.precision_intake import SourceValueRef
from merlin.semantic_compiler.precision_values import contraction_restore_exponent
from merlin.semantic_compiler.target_binding import NativeCompilationError

from .binding import (
    BF16_RELU_ATTRS,
    _positive_zero_boundary_profile,
    diagnostic_mxu0_f32_profile,
)

CONTRACTION_POLICY = "f32-e4m3-rne-positive-zero-ordered-bf16-mxu0-v1"
SCALED_CONTRACTION_POLICY = "f32-scaled-e4m3-rne-positive-zero-ordered-bf16-mxu0-v1"


def is_source_relu(node):
    return (
        node.effect == "pure"
        and len(node.inputs) == 1
        and node.type.dtype == "f32"
        and node.type.numerical_policy == "raw-bit-pattern"
        and node.index_maps == pointwise_identity_maps(len(node.type.shape), 1)
        and (
            (
                node.op == "f32_relu_ordered_select"
                and dict(node.attrs) == {"numeric_semantics": "f32-ordered-lt-positive-zero-select-v1"}
            )
            or (
                node.op == "source_f32_composite_unqualified"
                and dict(node.attrs)
                == {
                    "numeric_semantics": "source-f32-composite-unqualified-v1",
                    "operation": "clamp_lower_literal",
                    "literal_bits": "0x00000000",
                }
            )
        )
    )


def prepare_contraction(source, *, operand_scale_exponents=None, epilogue=None):
    import numpy as np

    profile = _positive_zero_boundary_profile(diagnostic_mxu0_f32_profile())
    if operand_scale_exponents is not None:
        if not isinstance(operand_scale_exponents, (list, tuple)) or len(operand_scale_exponents) != 2:
            raise ValueError("scaled contraction needs two constant operand exponents")
        restore = contraction_restore_exponent(*operand_scale_exponents)
        inputs = tuple(
            replace(
                spec,
                policy=f"{spec.policy}-scale-{exponent}",
                pre_encode_scale_exponent=exponent,
            )
            for spec, exponent in zip(profile.conversion_specs[:2], operand_scale_exponents, strict=True)
        )
        output = replace(
            profile.conversion_specs[2],
            policy=f"bf16-to-f32-restore-scale-{restore}",
            post_decode_scale_exponent=restore,
        )
        profile = replace(
            profile,
            operand_conversion_policies=tuple(spec.policy for spec in inputs),
            result_conversion_policy=output.policy,
            conversion_specs=(*inputs, output),
        )
    contraction_source = source
    if epilogue is not None:
        pure = [node for node in source.nodes if node.effect == "pure"]
        if (
            epilogue != "relu"
            or len(pure) != 2
            or not is_source_relu(pure[-1])
            or pure[-1].inputs != (pure[0].id,)
            or pure[-1].type != pure[0].type
            or source.outputs != (pure[-1].id,)
        ):
            raise ValueError("contraction epilogue needs one closed, unobserved matmul followed by ReLU")
        contraction_source = extract_semantic_region(source, (pure[0].id,))
        if {node.id for node in source.nodes} != {node.id for node in contraction_source.nodes} | {pure[-1].id}:
            raise ValueError("contraction epilogue has unaccounted source nodes")
        profile = replace(
            profile,
            selected_decision=(*profile.selected_decision, ("epilogue", epilogue)),
        )
    if len(contraction_source.node(contraction_source.outputs[0]).type.shape) > 2:
        from merlin.semantic_compiler.batched_contraction import (
            lower_batched_contraction,
        )

        if epilogue is not None:
            raise ValueError("batched contraction epilogues require a separate checked lowering")
        result = lower_batched_contraction(contraction_source, profile=profile)
    else:
        result = lower_semantic_contraction(contraction_source, profile=profile)
    if epilogue is not None:
        final = pure[-1]
        raw_type = replace(result.request.nodes[-1].type, numerical_policy="raw-bit-pattern")
        operation = SemanticNode(
            "epilogue",
            "relu",
            ("result",),
            raw_type,
            BF16_RELU_ATTRS,
            index_maps=pointwise_identity_maps(2, 1),
        )
        source_indices = {node.id: i for i, node in enumerate(source.nodes)}
        inputs = tuple(
            replace(
                item,
                source_value=SourceValueRef(
                    "semantic_node",
                    source_indices[contraction_source.nodes[item.source_value.index].id],
                ),
            )
            for item in result.inputs
        )
        result = replace(
            result,
            source=source,
            region_id=final.id,
            inputs=inputs,
            output=replace(
                result.output,
                source_value=SourceValueRef("semantic_node", source.nodes.index(final)),
                source_type=final.type,
                target_type=raw_type,
            ),
            request=replace(
                result.request,
                nodes=(*result.request.nodes, operation),
                outputs=(operation.id,),
                source_identity=sha256((result.request.source_identity + source.digest()).encode()).hexdigest(),
            ),
        )
    bindings = {item.node_id: item for item in source.constants}
    for conversion in result.inputs:
        node = source.nodes[conversion.source_value.index]
        if node.effect == "constant":
            values = np.frombuffer(bytes.fromhex(bindings[node.id].data_hex), dtype="<f4").reshape(node.type.shape)
            try:
                conversion.convert_values(values)
            except ValueError as error:
                raise NativeCompilationError(
                    "unsupported_semantics",
                    "declared parameter leaves its operand domain",
                ) from error
    return result
