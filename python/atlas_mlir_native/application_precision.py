"""Selected precision for source regions in a diagnostic Atlas application."""

from __future__ import annotations

from dataclasses import replace
from hashlib import sha256

from merlin.compiler.semantic.model import pointwise_identity_maps
from merlin.compiler.quantization.contraction.values import contraction_restore_exponent
from merlin.compiler.semantic.target_binding import NativeCompilationError

from .binding import (
    _positive_zero_boundary_profile,
    diagnostic_mxu0_f32_profile,
)

CONTRACTION_POLICY = "f32-e4m3-rne-positive-zero-ordered-bf16-mxu0-v1"
SCALED_CONTRACTION_POLICY = "f32-scaled-e4m3-rne-positive-zero-ordered-bf16-mxu0-v1"
COLUMN_SCALED_CONTRACTION_POLICY = "f32-column-scaled-e4m3-cap448-normal-round-ftz-positive-zero-ordered-bf16-mxu0-v1"


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
    from merlin.compiler.quantization.contraction.lowering import lower_semantic_contraction
    import numpy as np

    profile = _positive_zero_boundary_profile(diagnostic_mxu0_f32_profile())
    if isinstance(operand_scale_exponents, dict):
        selected = operand_scale_exponents
        if set(selected) != {"policy", "source_request_digest", "provenance_sha256", "lhs_encode_exponent", "rhs_column_encode_exponents"}:
            raise ValueError("column-scaled contraction policy fields differ")
        if selected["policy"] != COLUMN_SCALED_CONTRACTION_POLICY or selected["source_request_digest"] != source.digest():
            raise ValueError("column-scaled contraction source or policy differs")
        provenance = selected["provenance_sha256"]
        if not isinstance(provenance, str) or len(provenance) != 64 or any(c not in "0123456789abcdef" for c in provenance):
            raise ValueError("column-scaled contraction requires a provenance SHA-256")
        lhs_exponent = selected["lhs_encode_exponent"]
        panels = tuple(tuple(panel) for panel in selected["rhs_column_encode_exponents"])
        if not panels:
            raise ValueError("column-scaled contraction requires a complete nonempty partition")
        # Validation and scale partition recording belong to shared Merlin specs.
        policy_id = sha256(__import__("json").dumps(selected, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
        base = profile.conversion_specs[0]
        common = dict(max_normal_exponent_code=15, admitted_code_ranges=((0, 0), (8, 126), (136, 254)),
                      underflow_rounding="round-normal-significand-then-flush")
        lhs = replace(base, policy=f"column-policy-{policy_id}-lhs", pre_encode_scale_exponent=lhs_exponent, **common)
        rhs = replace(base, policy=f"column-policy-{policy_id}-rhs", column_scale_exponents=panels, scale_axis=0, **common)
        restore = tuple((start, stop, contraction_restore_exponent(lhs_exponent, exponent)) for start, stop, exponent in panels)
        output = replace(profile.conversion_specs[2], policy=f"column-policy-{policy_id}-restore", column_scale_exponents=restore)
        profile = replace(profile, operand_conversion_policies=(lhs.policy, rhs.policy), result_conversion_policy=output.policy,
                          conversion_specs=(lhs, rhs, output))
    elif operand_scale_exponents is not None:
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
    if epilogue is not None:
        raise ValueError("contraction epilogues require a separate checked lowering")
    if len(source.node(source.outputs[0]).type.shape) > 2:
        from merlin.compiler.quantization.contraction.batched import lower_batched_contraction

        result = lower_batched_contraction(source, profile=profile)
    else:
        result = lower_semantic_contraction(source, profile=profile)
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


def prepare_already_quantized_contraction(source):
    """Bind authored code/exponent tensors without the float precision transform."""
    from .already_quantized import prepare_source

    return prepare_source(source)
