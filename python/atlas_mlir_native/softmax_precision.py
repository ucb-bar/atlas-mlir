"""Bind parsed source softmax to the explicitly selected BF16 row policy."""

from __future__ import annotations

import hashlib
import json

from merlin.semantic_compiler.guarded_softmax_source import recognize_guarded_masked_row_softmax
from merlin.semantic_compiler.linalg_bridge import (
    LinalgBridgeError,
    translate_linalg_text,
)
from merlin.semantic_compiler.model import KernelRequest, SemanticNode, TensorType
from merlin.semantic_compiler.precision_values import FloatBoundarySpec
from merlin.semantic_compiler.softmax_source import (
    MaskedRowSoftmaxSource,
    recognize_masked_row_softmax,
    recognize_row_softmax,
)

from .binding import TARGET_IDENTITY
from .graph_precision import DiagnosticGraphPrecision
from .row_softmax import GUARDED_SOFTMAX_POLICY, MASKED_SOFTMAX_POLICY, SOFTMAX_DOMAIN, SOFTMAX_POLICY

INPUT_POLICY = "f32-to-bf16-rne-bounded-row-scores-v1"


def input_spec():
    """RNE, then require signed zero or a score admitted by the target policy.

    The ranges are encoded BF16 words, not source f32 clipping thresholds.
    Values that round outside them are rejected. No small-normal flushing is
    introduced; f32 values that naturally round to BF16 signed zero stay zero.
    """
    return FloatBoundarySpec(
        INPUT_POLICY,
        "bf16",
        "encode",
        admitted_code_ranges=(
            (0, 0),
            (0x0400, 0x4200),
            (0x8000, 0x8000),
            (0x8400, 0xC200),
        ),
    )


def lower_source_softmax(
    source: str, *, entry: str, policy: dict, original_source_sha256: str | None = None
) -> DiagnosticGraphPrecision:
    source_sha = hashlib.sha256(source.encode()).hexdigest()
    expected = {
        "schema": "atlas.diagnostic_row_softmax_precision.v1",
        "target_identity": TARGET_IDENTITY,
        "input_conversion": INPUT_POLICY,
        "row_softmax": SOFTMAX_POLICY,
        "output_conversion": "bf16-to-f32-exact-v1",
        "source_sha256": original_source_sha256 or source_sha,
    }
    if policy != expected:
        raise LinalgBridgeError("selected source row-softmax precision policy is unsupported")
    translated = translate_linalg_text(
        source,
        entry=entry,
        target_identity=TARGET_IDENTITY,
        lowering_policy="diagnostic",
    )
    graph = translated.request
    policy_sha = hashlib.sha256(json.dumps(policy, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    return _lower_graph(graph, source_sha, policy_sha, translated.source_operations)


def lower_semantic_softmax(
    graph: KernelRequest, *, allow_masked=False, allow_guarded=False
) -> DiagnosticGraphPrecision:
    """Bind a closed parsed semantic slice without regenerating or recapturing IR."""
    if graph.target_identity != TARGET_IDENTITY or graph.lowering_policy != "diagnostic":
        raise LinalgBridgeError("semantic softmax needs the selected diagnostic target")
    selected = {
        "input_conversion": INPUT_POLICY,
        "row_softmax": GUARDED_SOFTMAX_POLICY
        if allow_guarded
        else MASKED_SOFTMAX_POLICY
        if allow_masked
        else SOFTMAX_POLICY,
        "output_conversion": "bf16-to-f32-exact-v1",
    }
    policy_sha = hashlib.sha256(json.dumps(selected, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    return _lower_graph(
        graph,
        graph.digest(),
        policy_sha,
        tuple(node.op for node in graph.nodes),
        "semantic_graph",
        allow_masked=allow_masked,
        allow_guarded=allow_guarded,
    )


def _lower_graph(
    graph, source_sha, policy_sha, source_operations, source_kind=None, *, allow_masked=False, allow_guarded=False
):
    if len(graph.outputs) != 1:
        raise LinalgBridgeError("source row-softmax entry needs one accounted result")
    try:
        recognize = (
            recognize_guarded_masked_row_softmax
            if allow_guarded
            else recognize_masked_row_softmax
            if allow_masked
            else recognize_row_softmax
        )
        region = recognize(graph, graph.outputs[0])
    except (ValueError, KeyError) as error:
        raise LinalgBridgeError(str(error)) from error
    score = graph.node(region.scores)
    masked = isinstance(region, MaskedRowSoftmaxSource)
    boundaries = (region.scores, region.mask) if masked else (region.scores,)
    accounted = {*boundaries, *region.computations, *region.initializers}
    if (
        score.op != "input"
        or set(graph.input_storages) != {(key, "external") for key in boundaries}
        or any(graph.node(key).op != "input" for key in boundaries)
        or {node.id for node in graph.nodes} != accounted
        or graph.physical_extents
        or graph.boundary_layouts
        or graph.output_layouts
    ):
        raise LinalgBridgeError("source row-softmax entry has unaccounted inputs or computation")
    request = KernelRequest(
        (
            SemanticNode(
                score.id,
                "input",
                (),
                TensorType(region.matrix_shape, "bf16", "raw-bit-pattern"),
                (("input_domain", SOFTMAX_DOMAIN),),
                "input",
            ),
            *(
                (
                    SemanticNode(
                        region.mask,
                        "input",
                        (),
                        TensorType(region.matrix_shape, "bf16", "raw-bit-pattern"),
                        (("input_domain", "exact-zero-or-one"),),
                        "input",
                    ),
                )
                if masked
                else ()
            ),
            SemanticNode(
                region.result,
                "masked_softmax" if masked else "softmax",
                boundaries,
                TensorType(
                    region.matrix_shape,
                    "bf16",
                    GUARDED_SOFTMAX_POLICY if allow_guarded else MASKED_SOFTMAX_POLICY if masked else SOFTMAX_POLICY,
                ),
                (("axis", 1),),
            ),
        ),
        (region.result,),
        ("external",),
        tuple((key, "external") for key in boundaries),
        TARGET_IDENTITY,
        lowering_policy="diagnostic",
        source_identity=source_sha + ":" + policy_sha,
    )
    spec = input_spec()
    return DiagnosticGraphPrecision(
        request,
        source_sha,
        policy_sha,
        tuple((node.id, node.id if node.id in {*boundaries, region.result} else None) for node in graph.nodes),
        tuple(
            (node, "source f32 stable softmax -> selected whole-row BF16 arithmetic") for node in region.computations
        ),
        (
            {
                "source": score.id,
                "direction": "encode",
                "policy": spec.policy,
                "conversion_spec": spec.record(),
                "shape": list(region.shape),
                **({"dense_row_view": region.record()["dense_row_view"]} if len(region.shape) != 2 else {}),
            },
            *(
                (
                    {
                        "source": region.mask,
                        "direction": "mask_to_keep",
                        "policy": GUARDED_SOFTMAX_POLICY if allow_guarded else MASKED_SOFTMAX_POLICY,
                        "source_shape": list(region.mask_shape),
                        "logical_shape": list(region.shape),
                        "mask_map": region.mask_map.record(),
                        **(
                            {"admitted_i1_codes": [0, 1]}
                            if allow_guarded
                            else {"admitted_f32_codes": ["0x00000000", "0xff800000"]}
                        ),
                        "keep_bf16_codes": ["0x3f80", "0x0000"],
                        "masked_score_bf16": "0xc200",
                        "all_masked_rows": "positive-zero" if allow_guarded else "rejected_before_region_device_io",
                    },
                )
                if masked
                else ()
            ),
            {
                "source": region.result,
                "direction": "decode",
                "policy": "bf16-to-f32-exact-v1",
                "shape": list(region.shape),
                **({"dense_row_view": region.record()["dense_row_view"]} if len(region.shape) != 2 else {}),
            },
        ),
        source_operations,
        region,
        source_kind,
    )
