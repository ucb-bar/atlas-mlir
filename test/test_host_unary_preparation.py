"""Selected unary preparation stays on the host without moving model math."""

from dataclasses import replace
import json

import pytest

from atlas_mlir_native.application import AtlasApplicationAdapter
from atlas_mlir_native.host_unary_preparation import POLICY, SCOPE_POLICY
from merlin.compile.semantic_regions import extract_semantic_region
from merlin.semantic_compiler.model import ConstantBinding, IndexMap, SemanticNode, TensorType, pointwise_identity_maps
from merlin.semantic_compiler.target_binding import NativeCompilationError
from merlin.tests.runtime.test_unary_math_c import policy as math_policy
from test_host_k1_contraction import policy as base_policy, request as k1_request


def policy():
    return {**base_policy(), "schema": "atlas.diagnostic_application_precision.v20",
            "host_unary_math": math_policy(),
            "host_integer_ops": ["i64_add", "signed_i64_to_f32"],
            "host_float_arithmetic": ["f32_mul", "f32_neg"],
            "host_support": ["transpose", "reshape", "copy", "gather", "concat"]}


def rotary_source():
    source = k1_request(m=2, n=3)
    integer = TensorType((2, 1), "i64", "signed-i64")
    positions = SemanticNode("positions", "input", (), integer, effect="input")
    offset = SemanticNode("offset", "constant", (), integer, effect="constant")
    added = SemanticNode("position_offset", "i64_add", (positions.id, offset.id), integer,
                         (("numeric_semantics", "signed-i64-add-wrap-v1"),),
                         index_maps=pointwise_identity_maps(2, 2))
    a = SemanticNode("a", "signed_i64_to_f32", (added.id,), source.node("a").type,
                     (("numeric_semantics", "signed-i64-to-f32-rne-v1"),),
                     index_maps=pointwise_identity_maps(2, 1))
    frequencies = SemanticNode("frequencies", "input", (),
                               TensorType((1, 3), "bf16", "raw-bit-pattern"), effect="input")
    b = SemanticNode("b", "bf16_to_f32", (frequencies.id,), source.node("b").type,
                     (("numeric_semantics", "bf16-to-f32-exact-v1"),),
                     index_maps=pointwise_identity_maps(2, 1))
    angles_type = TensorType((2, 6), "f32", "raw-bit-pattern")
    angles = SemanticNode("angles", "concat", ("y", "y"), angles_type, (("axis", 1),))
    unary = tuple(SemanticNode(key, "source_f32_" + key + "_unqualified", (angles.id,), angles_type,
                               (("numeric_semantics", "source-f32-" + key + "-unqualified-v1"),),
                               index_maps=pointwise_identity_maps(2, 1)) for key in ("sin", "cos"))
    combined_type = replace(angles_type, shape=(2, 12))
    combined = SemanticNode("combined", "concat", ("sin", "cos"), combined_type, (("axis", 1),))
    # A constant-derived layout on the forward side is also preparation.
    scale_type = replace(combined_type, shape=(24,))
    scale = SemanticNode("scale", "constant", (), scale_type, effect="constant")
    scale_view = SemanticNode("scale_view", "reshape", (scale.id,), combined_type)
    scaled = SemanticNode("scaled", "f32_mul", (combined.id, scale_view.id), combined_type,
                          (("numeric_semantics", "ieee754-f32-mul-rne-v1"),),
                          index_maps=pointwise_identity_maps(2, 2))
    negative = SemanticNode("negative", "f32_neg", (scaled.id,), combined_type,
                            (("numeric_semantics", "f32-sign-bit-flip-v1"),),
                            index_maps=pointwise_identity_maps(2, 1))
    viewed_type = replace(combined_type, shape=(1, 2, 12))
    viewed = SemanticNode("viewed", "reshape", (negative.id,), viewed_type)
    output = SemanticNode("rotary", "f32_to_bf16", (viewed.id,), replace(viewed_type, dtype="bf16"),
                          (("numeric_semantics", "f32-to-bf16-rne-v1"),),
                          index_maps=pointwise_identity_maps(3, 1))
    return replace(source,
        nodes=(positions, offset, added, a, frequencies, b, source.node("seed"), source.node("y"),
               angles, *unary, combined, scale, scale_view, scaled, negative, viewed, output),
        outputs=(output.id,), input_storages=((positions.id, "external"), (frequencies.id, "external")),
        constants=(*source.constants, ConstantBinding(offset.id, "i64-le", "0000000000000000" * 2),
                   ConstantBinding(scale.id, "f32-le-raw", "0000803f" * 24)))


def adapter():
    selected = policy()
    return AtlasApplicationAdapter(selected, original_source_sha256=selected["source_sha256"])


def by_source_id(source, plan):
    pure = [node for node in source.nodes if node.effect == "pure"]
    return {pure[index].id: alternative for alternative in plan.selected for index in alternative.nodes}


def test_complete_integer_k1_layout_trig_scale_and_final_cast_preparation_is_host():
    source = rotary_source()
    before = source.record()
    selected_adapter = adapter()
    selected = selected_adapter.plan(source)
    by_id = by_source_id(source, selected)
    assert len(by_id) == sum(node.effect == "pure" for node in source.nodes)
    assert all(value.placement == "host" for value in selected.selected)
    assert dict(by_id["y"].metadata)["kind"] == "host_k1_contraction"
    assert dict(by_id["angles"].metadata)["kind"] == "layout"
    assert dict(by_id["rotary"].metadata)["kind"] == "float_cast"
    for key, alternative in by_id.items():
        region = extract_semantic_region(source, (key,))
        result = selected_adapter.compile_region(None, region, alternative, "preparation", None, None, None)
        assert result.programs == ()
        receipt = result.description["host_preparation"]
        assert receipt["policy"] == POLICY and receipt["placement"] == "host"
        assert receipt["source_request_digest"] == source.digest()
        assert set(receipt["selected_unary_nodes"]) == {"sin", "cos"}
        assert set(receipt["source_nodes"]) == by_id.keys()
        assert receipt["arithmetic_rewrite"] == "none"
    assert source.record() == before


def test_unrelated_layout_and_cast_keep_device_preference_and_model_mixing_stops_closure():
    source = rotary_source()
    type_ = source.node("combined").type
    model = SemanticNode("model", "input", (), type_, effect="input")
    unrelated = SemanticNode("unrelated", "copy", (model.id,), type_)
    unrelated_cast = SemanticNode("unrelated_cast", "f32_to_bf16", (model.id,), replace(type_, dtype="bf16"),
                                 (("numeric_semantics", "f32-to-bf16-rne-v1"),),
                                 index_maps=pointwise_identity_maps(2, 1))
    mixed = SemanticNode("mixed", "f32_mul", ("combined", model.id), type_,
                         (("numeric_semantics", "ieee754-f32-mul-rne-v1"),),
                         index_maps=pointwise_identity_maps(2, 2))
    mixed_cast = replace(unrelated_cast, id="mixed_cast", inputs=(mixed.id,))
    source = replace(source, nodes=(*source.nodes, model, unrelated, unrelated_cast, mixed, mixed_cast),
                     outputs=(*source.outputs, unrelated.id, unrelated_cast.id, mixed_cast.id),
                     output_storages=("external",) * 4,
                     input_storages=(*source.input_storages, (model.id, "external")))
    by_id = by_source_id(source, adapter().plan(source))
    assert by_id["rotary"].placement == "host"
    assert dict(by_id["unrelated"].metadata)["kind"] == "layout_device"
    assert dict(by_id["unrelated_cast"].metadata)["kind"] == "exact_float_device"
    assert dict(by_id["mixed_cast"].metadata)["kind"] == "exact_float_device"
    # Existing explicit global f32_mul host placement remains unchanged.
    assert dict(by_id["mixed"].metadata)["kind"] == "float_arithmetic"
    assert all("host_preparation" not in dict(by_id[key].metadata)
               for key in ("unrelated", "unrelated_cast", "mixed", "mixed_cast"))


def test_without_selected_unary_math_existing_device_placement_is_retained():
    from test_layout_device import request

    source = request("concat", ((2, 1, 32), (2, 1, 32)), (2, 2, 32), dtype="f32", axis=1)
    selected = {key: value for key, value in policy().items() if key != "host_unary_math"}
    selected["schema"] = "atlas.diagnostic_application_precision.v19"
    (alternative,) = AtlasApplicationAdapter(selected, original_source_sha256=selected["source_sha256"]).plan(source).selected
    assert alternative.placement == "accelerator"
    assert dict(alternative.metadata)["kind"] == "layout_device"
    assert "host_preparation" not in dict(alternative.metadata)


def test_explicit_precision_region_boundary_stops_backward_host_selection():
    source = rotary_source()
    selected_adapter = adapter()
    closure, _ = selected_adapter._host_unary_preparation(source, {"y"}, excluded={"y"})
    assert {"sin", "cos", "angles", "rotary"} <= closure.keys()
    assert not {"y", "a", "b", "position_offset"} & closure.keys()


def test_global_host_arithmetic_selection_still_refuses_unrelated_broadcast():
    source = rotary_source()
    type_ = source.node("combined").type
    model = SemanticNode("model", "input", (), type_, effect="input")
    gain = SemanticNode("gain", "input", (), replace(type_, shape=(2, 1)), effect="input")
    identity = pointwise_identity_maps(2, 1)[0]
    broadcast = IndexMap(2, ((1, 0), (0, 0)), (0, 0))
    output = SemanticNode("model_scaled", "f32_mul", (model.id, gain.id), type_,
                          (("numeric_semantics", "ieee754-f32-mul-rne-v1"),),
                          index_maps=(identity, broadcast, identity))
    source = replace(source, nodes=(*source.nodes, model, gain, output),
                     outputs=(*source.outputs, output.id), output_storages=("external", "external"),
                     input_storages=(*source.input_storages, (model.id, "external"), (gain.id, "external")))
    with pytest.raises(NativeCompilationError, match="float arithmetic source types, indexing or numerical policy differ"):
        adapter().plan(source)


def scoped_policy(source):
    # This boundary is an explicit immutable parameter contract, never inferred from
    # a floating input's shape or graph position. The caller authenticates bytes.
    return {**policy(), "host_preparation_scope": {
        "policy": SCOPE_POLICY,
        "constant_inputs": [{"node_id": "frequencies", "type": source.node("frequencies").type.record()}],
    }}


def scoped_adapter(source, selected=None):
    selected = scoped_policy(source) if selected is None else selected
    return AtlasApplicationAdapter(selected, original_source_sha256=selected["source_sha256"])


def test_scoped_preparation_admits_only_integer_and_explicit_constant_boundaries():
    source = rotary_source()
    before = source.record()
    selected = scoped_adapter(source).plan(source)
    assert all(alternative.placement == "host" for alternative in selected.selected)
    receipt = json.loads(dict(selected.selected[0].metadata)["host_preparation"])
    assert {entry["node_id"] for entry in receipt["input_boundaries"]} == {"positions", "frequencies"}
    assert receipt["scope"] == scoped_policy(source)["host_preparation_scope"]
    assert set(receipt["selected_unary_nodes"]) == {"sin", "cos"}
    assert source.record() == before


def test_scoped_host_arithmetic_leaves_unrelated_broadcast_on_existing_device_route():
    source = rotary_source()
    type_ = source.node("combined").type
    model = SemanticNode("model", "input", (), type_, effect="input")
    gain = SemanticNode("gain", "input", (), replace(type_, shape=(2, 1)), effect="input")
    identity = pointwise_identity_maps(2, 1)[0]
    broadcast = IndexMap(2, ((1, 0), (0, 0)), (0, 0))
    output = SemanticNode("model_scaled", "f32_mul", (model.id, gain.id), type_,
                          (("numeric_semantics", "ieee754-f32-mul-rne-v1"),),
                          index_maps=(identity, broadcast, identity))
    source = replace(source, nodes=(*source.nodes, model, gain, output),
                     outputs=(*source.outputs, output.id), output_storages=("external", "external"),
                     input_storages=(*source.input_storages, (model.id, "external"), (gain.id, "external")))
    by_id = by_source_id(source, scoped_adapter(source).plan(source))
    assert by_id["rotary"].placement == "host"
    assert by_id["model_scaled"].placement == "accelerator"
    assert dict(by_id["model_scaled"].metadata)["kind"] == "pointwise"
    assert "host_preparation" not in dict(by_id["model_scaled"].metadata)


@pytest.mark.parametrize("dependency", ["float-input", "unsupported-computation", "device-exclusion"])
def test_scoped_activation_dependent_trig_cannot_become_host_preparation(dependency):
    source = rotary_source()
    selected = scoped_policy(source)
    if dependency == "float-input":
        selected["host_preparation_scope"]["constant_inputs"] = []
    elif dependency == "unsupported-computation":
        # Even integer-derived unsupported computations cannot be silently
        # treated as host preparation input boundaries.
        source = replace(source, nodes=tuple(replace(node, op="unsupported_activation")
                         if node.id == "position_offset" else node for node in source.nodes))
    excluded = {"y"} if dependency == "device-exclusion" else set()
    adapter_ = scoped_adapter(source, selected)
    closure, _ = adapter_._host_unary_preparation(source, {"y"}, excluded=excluded)
    assert closure == {}
    if dependency == "float-input":
        with pytest.raises(NativeCompilationError, match="floating contractions requiring operand recoding"):
            adapter_.plan(source)


def test_scoped_direct_activation_trig_has_no_host_alternative():
    source = rotary_source()
    type_ = source.node("sin").type
    model = SemanticNode("model", "input", (), type_, effect="input")
    trig = replace(source.node("sin"), id="model_sin", inputs=(model.id,))
    source = replace(source, nodes=(*source.nodes, model, trig), outputs=(*source.outputs, trig.id),
                     output_storages=("external", "external"),
                     input_storages=(*source.input_storages, (model.id, "external")))
    adapter_ = scoped_adapter(source)
    closure, receipt = adapter_._host_unary_preparation(source, {"y"}, excluded=set())
    assert "sin" in closure and "model_sin" not in closure
    assert set(json.loads(receipt)["selected_unary_nodes"]) == {"sin", "cos"}
    with pytest.raises(NativeCompilationError, match="no compiled application route for model_sin"):
        adapter_.plan(source)


@pytest.mark.parametrize("mutation", ["unknown-policy", "duplicate", "extra-field", "type", "absent", "shape"])
def test_scope_constant_input_contract_is_validated(mutation):
    source = rotary_source()
    selected = scoped_policy(source)
    scope = selected["host_preparation_scope"]
    entry = scope["constant_inputs"][0]
    if mutation == "unknown-policy":
        scope["policy"] = "unknown"
    elif mutation == "duplicate":
        scope["constant_inputs"].append(dict(entry))
    elif mutation == "extra-field":
        entry["sha256"] = "1" * 64
    elif mutation == "type":
        entry["type"] = {"dtype": "bf16"}
    elif mutation == "absent":
        entry["node_id"] = "absent"
    else:
        entry["type"]["shape"] = [3, 1]
    with pytest.raises(NativeCompilationError, match="host preparation"):
        scoped_adapter(source, selected).plan(source)
