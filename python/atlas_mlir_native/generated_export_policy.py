"""Graph-specific export policy through Merlin and the existing Atlas adapter.

Locations identify authenticated compiler inputs; they never select arithmetic.
The reviewed exporter decoder supplies numerical grants separately from graph
matching. Host implementation choices are explicit in the published family set.
"""
from __future__ import annotations

import json
from pathlib import Path

from merlin.compiler.frontend.policy_inputs import authenticate_policy_inputs
from merlin.compiler.quantization.policy_generation import generate_capture_policy
from merlin.compiler.semantic.target_binding import NativeCompilationError

from .soc import CAPTURED_STATE_READ_BYTES

SCHEMA = "atlas.generated_application_precision.v1"
# One explicit source-exact target configuration for all admitted exports.
EXPORT_FAMILIES = (
    "atlas.ordered-quantized-contraction", "atlas.ordered-quantized-contraction-f32", "atlas.activation-quantize",
    "atlas.layout", "atlas.host-bias-add", "atlas.host-ordered-select",
    "atlas.host-window-max", "atlas.host-batch-norm", "atlas.host-sigmoid", "atlas.host-gelu",
    "atlas.host-layer-norm", "atlas.host-sign-flip", "atlas.host-bf16-contraction", "atlas.exact-bf16-add",
    "atlas.device-bf16-mul",
    "atlas.host-f32-add", "atlas.host-f32-sub", "atlas.host-f32-mul", "atlas.host-f32-div",
    "atlas.exact-bf16-to-f32", "atlas.exact-f32-to-bf16", "atlas.host-static-quantize-f32",
)


def source_translators():
    from .source_translation import (
        translate_activation_quantizer,
        translate_prequantized_contraction,
    )

    return {"atlas.quantize_act": translate_activation_quantizer,
            **{op: translate_prequantized_contraction
               for op in ("atlas.qlinear", "atlas.qk_matmul", "atlas.pv_matmul")}}


def _inputs(export_dir, capture_dir):
    from merlin.compiler.frontend.inputs import _read_member

    from .export_contract import decode_export_contract
    from .policy_derivation import _capture_bindings

    export, capture = Path(export_dir).absolute(), Path(capture_dir).absolute()
    bound = authenticate_policy_inputs(
        export, export.parent / "fx", capture,
        export_manifest_schema="atlas_torchao.export_manifest.v1",
        export_policy_schema="atlas_torchao.execution_policy.v1",
        fx_manifest_schema="atlas_torchao.fx_bundle.v1", adapter_id="atlas_fp8",
        native_program_sha256_path=("atlas", "saved_native_program", "sha256"),
    )
    receipt = json.loads(_read_member(capture, "capture_receipt.json"))
    quantization = json.loads(bound.quantization_manifest_json)
    captured = _capture_bindings(capture, receipt, quantization)
    checked, grants = decode_export_contract(json.loads(bound.export_policy_json), captured=captured)
    return bound, checked, grants


def derive_generated_export_policy(export_dir, capture_dir):
    """Derive a graph-specific policy, preserving producer checks and custody."""
    from .binding import TARGET_IDENTITY
    from .compiler_interface import describe_compiler, describe_numerical_families

    bound, checked, grants = _inputs(export_dir, capture_dir)
    from .export_readonly_state import derive_readonly_state

    state, state_evidence = derive_readonly_state(bound, checked, export_dir, capture_dir)
    generated = generate_capture_policy(
        bound, capture_dir, describe_compiler(), describe_numerical_families(),
        target_identity=TARGET_IDENTITY, source_operation_translators=source_translators(),
        authorized_precision_changes=grants, selected_family_ids=EXPORT_FAMILIES,
        bind_external_parameters=True, application_state=state, max_state_bytes=CAPTURED_STATE_READ_BYTES,
    )
    return {
        "schema": SCHEMA, "source_sha256": generated["source_sha256"], "target_identity": TARGET_IDENTITY,
        "input_locations": {"export_dir": str(Path(export_dir).absolute()),
                            "capture_dir": str(Path(capture_dir).absolute())},
        "checked_export_contract": checked.record(), "generated_policy": generated,
        "application_state": state, "readonly_state_evidence": state_evidence,
    }


def adapter_for_prepared_source(policy, *, source, staged, original_source_sha256):
    """Recheck original inputs and explicit target configuration before selection."""
    from .application import AtlasApplicationAdapter
    from .binding import TARGET_IDENTITY

    fields = {"schema", "source_sha256", "target_identity", "input_locations",
              "checked_export_contract", "generated_policy", "application_state", "readonly_state_evidence"}
    if (type(policy) is not dict or set(policy) != fields or policy["schema"] != SCHEMA
            or policy["target_identity"] != TARGET_IDENTITY
            or policy["source_sha256"] != original_source_sha256
            or original_source_sha256 != staged.original_source_sha256):
        raise NativeCompilationError("unsupported_semantics", "generated export policy fields/source differ")
    locations = policy["input_locations"]
    if (type(locations) is not dict or set(locations) != {"export_dir", "capture_dir"}
            or any(type(path) is not str or not Path(path).is_absolute() for path in locations.values())):
        raise NativeCompilationError("unsupported_semantics", "generated export input locations differ")
    bound, checked, grants = _inputs(locations["export_dir"], locations["capture_dir"])
    if checked.record() != policy["checked_export_contract"]:
        raise NativeCompilationError("unsupported_semantics", "generated checked exporter contract changed")
    from .export_readonly_state import derive_readonly_state

    state, state_evidence = derive_readonly_state(bound, checked, locations["export_dir"], locations["capture_dir"])
    if state != policy["application_state"] or state_evidence != policy["readonly_state_evidence"]:
        raise NativeCompilationError("unsupported_semantics", "generated read-only state authority changed")
    generated = policy["generated_policy"]
    identities = generated.get("authenticated_inputs") if type(generated) is dict else None
    if (type(generated) is not dict or type(identities) is not dict
            or generated.get("capture_program_sha256") != staged.program_sha256
            or identities.get("weights_sha256") != staged.weights_sha256
            or identities.get("constants_sha256") != staged.constants_sha256):
        raise NativeCompilationError("unsupported_semantics", "generated staged program/archive identity changed")
    return AtlasApplicationAdapter.from_generated_policy(
        source, generated, bound=bound, capture_dir=locations["capture_dir"],
        authorized_precision_changes=grants, selected_family_ids=EXPORT_FAMILIES,
        bind_external_parameters=True, application_state=state,
    )
