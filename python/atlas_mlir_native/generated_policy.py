"""Admit Merlin family selection through the existing Atlas application adapter.

The generated record is reviewable evidence, never an authority source. Admission
replays authenticated capture preparation using independently supplied custody,
precision grants and implementation choices before enabling the declared
routes. Legacy diagnostic policy decoding remains in application.py.
"""
from __future__ import annotations

import json

from merlin.compiler.quantization.policy_generation import generate_capture_policy
from merlin.compiler.semantic.model import KernelRequest
from merlin.compiler.semantic.target_binding import NativeCompilationError

from .pointwise import BF16_POINTWISE_POLICY
from .soc import CAPTURED_STATE_READ_BYTES

# Reviewed family identities map to each family's first existing adapter route, not a new lowerer; declared
# alternates live only in compiler_interface._ALTERNATE_ROUTES. atlas.host-static-quantize is the explicit
# host-only quantizer configuration; no export selects it beside atlas.activation-quantize.
FAMILY_KINDS = {
    "atlas.activation-quantize": "activation_quantize_device",
    "atlas.host-static-quantize": "static_quantize",
    "atlas.host-static-quantize-f32": "static_quantize",
    "atlas.ordered-quantized-contraction": "already_quantized",
    "atlas.ordered-quantized-contraction-f32": "already_quantized",
    "atlas.device-layout": "layout_device",
    "atlas.host-layout": "layout",
    "atlas.layout": "layout_device",
    "atlas.host-bias-add": "float_arithmetic",
    "atlas.host-ordered-select": "host_ordered_relu",
    "atlas.host-window-max": "host_pointwise",
    "atlas.host-batch-norm": "host_pointwise",
    "atlas.host-sigmoid": "host_pointwise",
    "atlas.host-gelu": "host_pointwise",
    "atlas.host-layer-norm": "host_pointwise",
    "atlas.host-sign-flip": "host_pointwise",
    "atlas.host-f32-silu": "host_pointwise",
    "atlas.host-f32-rsqrt": "host_pointwise",
    "atlas.host-f32-sum": "host_pointwise",
    "atlas.host-bf16-contraction": "host_pointwise",
    "atlas.exact-bf16-add": "exact_float_device",
    "atlas.device-bf16-mul": "pointwise",
    "atlas.exact-bf16-to-f32": "exact_float_device",
    "atlas.exact-f32-to-bf16": "exact_float_device",
    "atlas.host-f32-add": "float_arithmetic",
    "atlas.host-f32-sub": "float_arithmetic",
    "atlas.host-f32-mul": "float_arithmetic",
    "atlas.host-f32-div": "float_arithmetic",
}


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _refuse(reason):
    raise NativeCompilationError("unsupported_semantics", reason)


def admit_generated_policy(adapter_type, source, generated, *, bound, capture_dir,
                           authorized_precision_changes, selected_family_ids,
                           bind_external_parameters=False, application_state=None):
    """Re-authenticate the capture and compare every generated field exactly."""
    from .binding import TARGET_IDENTITY
    from .source_translation import (
        translate_activation_quantizer,
        translate_prequantized_contraction,
    )

    if type(source) is not KernelRequest or not isinstance(generated, dict):
        _refuse("generated family admission needs a typed prepared source and policy record")
    source.__post_init__()
    if type(bind_external_parameters) is not bool:
        _refuse("generated family parameter binding mode must be explicit boolean")
    if source.target_identity != TARGET_IDENTITY or source.lowering_policy != "diagnostic":
        _refuse("generated family admission requires the selected diagnostic Atlas target")
    if (type(selected_family_ids) is not tuple or not selected_family_ids
            or any(type(family) is not str for family in selected_family_ids)
            or len(set(selected_family_ids)) != len(selected_family_ids)
            or any(family not in FAMILY_KINDS for family in selected_family_ids)):
        _refuse("generated family admission requires explicit supported family identities")
    translators = {
        "atlas.quantize_act": translate_activation_quantizer,
        **{operation: translate_prequantized_contraction
           for operation in ("atlas.qlinear", "atlas.qk_matmul", "atlas.pv_matmul")},
    }
    expected = generate_capture_policy(
        bound, capture_dir, adapter_type.describe_compiler(), adapter_type.describe_numerical_families(),
        target_identity=TARGET_IDENTITY, source_operation_translators=translators,
        authorized_precision_changes=authorized_precision_changes, selected_family_ids=selected_family_ids,
        bind_external_parameters=bind_external_parameters, application_state=application_state,
        max_state_bytes=CAPTURED_STATE_READ_BYTES,
    )
    if expected["graph_sha256"] != source.digest():
        _refuse("generated family prepared source differs from the authenticated capture")
    if _canonical(expected) != _canonical(generated):
        _refuse("generated family policy differs from authenticated recomputation or explicit authority")
    claims = tuple((row["node"], row["family"]) for row in expected["selected_families"])
    if (not claims or len({key for key, _ in claims}) != len(claims)
            or {key for key, _ in claims} != {node.id for node in source.nodes if node.effect == "pure"}
            or any(family not in FAMILY_KINDS for _, family in claims)):
        _refuse("generated family claims must cover the prepared graph with supported identities")

    return adapter_for_claims(adapter_type, source, claims, source_sha256=expected["source_sha256"],
                              policy_json=_canonical(expected))



def adapter_for_claims(adapter_type, source, claims, *, source_sha256, policy_json):
    """Configure an adapter for an authenticated family cover; callers authenticate the cover first."""
    adapter = adapter_type.__new__(adapter_type)
    # Disabled defaults are deliberate: selecting a family enables only its
    # declared routes, with no broad diagnostic arithmetic menu.
    defaults = {
        "original_source_sha256": source_sha256,
        "host_softmax": False, "host_softmax_policy": None,
        "host_preparation_scope": None, "activation_quantizer_route": None,
        "host_bf16_linear_policy": None, "already_quantized": False,
        "already_quantized_mask_sentinel": None, "already_quantized_score_domain": None,
        "source_exp_policy": None, "contraction_schedules": None,
        "rne_add": None, "qualified_rne_add": None, "softmax_value_schedules": {},
        "mxu1_contraction_schedules": {}, "rmsnorm_schedules": {}, "rmsnorm": False,
        "operand_scale_exponents": None, "epilogues": (), "pointwise": False,
        "bf16_pointwise": None, "reductions": False, "reduction_policy": None,
        "pointwise_math": False, "sigmoid": False, "gelu": False,
        "masked_softmax": False, "guarded_masked_softmax": False,
        "host_predicates": (), "host_integer_ops": (), "host_float_casts": (),
        "host_float_arithmetic": (), "host_unary_math": None, "host_layouts": (),
    }
    for name, value in defaults.items():
        setattr(adapter, name, value)
    capabilities = {
        "atlas.activation-quantize": {"activation_quantizer_route": "device"},
        "atlas.host-static-quantize": {"activation_quantizer_route": "host"},
        "atlas.host-static-quantize-f32": {},
        "atlas.ordered-quantized-contraction": {"already_quantized": True},
        "atlas.ordered-quantized-contraction-f32": {"already_quantized": True},
        "atlas.device-layout": {},
        "atlas.host-layout": {"host_layouts": tuple(sorted({source.node(key).op
                                for key, family in claims if family == "atlas.host-layout"}))},
        "atlas.layout": {"host_layouts": tuple(sorted({source.node(key).op
                           for key, family in claims if family == "atlas.layout"}))},
        "atlas.host-bias-add": {"host_float_arithmetic": ("bf16_add",)},
        "atlas.host-ordered-select": {"epilogues": ("relu",)},
        "atlas.host-window-max": {},
        "atlas.host-batch-norm": {},
        "atlas.host-sigmoid": {},
        "atlas.host-gelu": {},
        "atlas.host-layer-norm": {},
        "atlas.host-sign-flip": {},
        "atlas.host-f32-silu": {},
        "atlas.host-f32-rsqrt": {},
        "atlas.host-f32-sum": {},
        "atlas.host-bf16-contraction": {},
        # A source BF16 add the exact device route refuses runs on the checked host adder.
        "atlas.exact-bf16-add": {"host_float_arithmetic": ("bf16_add",)},
        "atlas.device-bf16-mul": {"bf16_pointwise": BF16_POINTWISE_POLICY},
        "atlas.exact-bf16-to-f32": {"host_float_casts": ("bf16_to_f32",)},
        "atlas.exact-f32-to-bf16": {"host_float_casts": ("f32_to_bf16",)},
        "atlas.host-f32-add": {"host_float_arithmetic": ("f32_add",)},
        "atlas.host-f32-sub": {"host_float_arithmetic": ("f32_sub",)},
        "atlas.host-f32-mul": {"host_float_arithmetic": ("f32_mul",)},
        "atlas.host-f32-div": {"host_float_arithmetic": ("f32_div",)},
    }
    for family in sorted({family for _, family in claims}):
        for name, value in capabilities[family].items():
            if name in {"host_float_arithmetic", "host_float_casts"}:
                # Operation menus from several selected families combine; each admits only its own operation.
                value = tuple(sorted({*getattr(adapter, name), *value}))
            setattr(adapter, name, value)
    adapter._generated_source_digest = source.digest()
    adapter._generated_family_claims = claims
    adapter._generated_policy_json = policy_json
    return adapter


def _exact_device_admits(source, keys):
    """The old planner's probe of the exact device route, on the region's single node."""
    from .application import AtlasApplicationAdapter

    return len(keys) == 1 and AtlasApplicationAdapter._supports_device_source(
        source, source.node(keys[0]), "exact_float_device")


def _device_layout_admits(adapter, source, key):
    """The old planner's rule, cached per adapter because the device proof can run external tools."""
    from merlin.compiler.semantic.source_preparation import is_prequantized_bias_add

    if not hasattr(adapter, "_layout_routes"):
        adapter._layout_routes = {}
        adapter._bias_broadcasts = {node.inputs[1] for node in source.nodes if node.effect == "pure"
                                    and is_prequantized_bias_add(source, node)
                                    and source.node(node.inputs[1]).op == "gather"}
    if key not in adapter._layout_routes:
        node = source.node(key)
        adapter._layout_routes[key] = (
            adapter.already_quantized and key not in adapter._bias_broadcasts
            and not (node.op == "gather" and adapter._is_scalar_broadcast(source, node))
            and adapter._supports_device_source(source, node, "layout_device"))
    return adapter._layout_routes[key]


def selected_kind(adapter, source, keys):
    """Require every member of an existing region to keep its selected route."""
    claims = dict(adapter._generated_family_claims)
    families = {claims.get(key) for key in keys}
    if len(families) != 1 or None in families:
        _refuse("generated application region differs from its selected family cover")
    family = families.pop()
    from .compiler_interface import alternate_kinds

    kind = FAMILY_KINDS[family]
    alternates = alternate_kinds(family)
    if family == "atlas.exact-bf16-add":
        from merlin.compiler.semantic.source_preparation import is_prequantized_bias_add

        # The family's contract excludes prepared contraction bias, folded to a constant or not.
        if any(is_prequantized_bias_add(source, source.node(key)) for key in keys):
            _refuse("exact BF16 add family excludes prepared contraction bias")
    if kind == "exact_float_device" and alternates and not _exact_device_admits(source, keys):
        # The old planner's rule: the device route where its closed region is admitted, else the checked host
        # emitter of the same source operation; the device refusal is never loosened.
        (kind,) = alternates
    if family == "atlas.layout" and not (len(keys) == 1 and _device_layout_admits(adapter, source, keys[0])):
        kind = "layout"
    if family == "atlas.activation-quantize" and len(keys) == 2 and not adapter._is_vector_quantizer(source, keys):
        # The old planner's rule: the device body where it is admitted, else the checked host encoder of its mode.
        mode = dict(source.node(keys[0]).attrs).get("mode")
        if mode not in {"static", "dynamic"}:
            _refuse("generated quantizer declares no admitted mode")
        kind = "static_quantize" if mode == "static" else "dynamic_quantize"
    if kind != FAMILY_KINDS[family] and kind not in alternates:
        _refuse("generated family route is not a declared alternate of its family")
    if family == "atlas.host-bias-add":
        from merlin.compiler.semantic.source_preparation import is_prequantized_bias_add

        if len(keys) != 1 or not is_prequantized_bias_add(source, source.node(keys[0])):
            _refuse("generated host bias family requires replayed post-cast bias structure")
    return kind
