"""Keep authored bias arithmetic after device readout on the host."""

from dataclasses import replace
from unittest.mock import Mock

import numpy as np
import pytest

from atlas_mlir_native.already_quantized import POLICY, prepare_source
from atlas_mlir_native.application import AtlasApplicationAdapter
from atlas_mlir_native.source_translation import NUMERICS_ATTRIBUTE, contraction_numerics_attribute
from merlin.compiler.partition.semantic_regions import extract_semantic_region
from merlin.compiler.semantic.model import ConstantBinding, SemanticNode, TensorType, pointwise_identity_maps
from merlin.compiler.semantic.source_preparation import split_prequantized_bias
from merlin.compiler.semantic.target_binding import NativeCompilationError
from test_already_quantized import source_request as base_source_request
from test_application_float_arithmetic import policy as arithmetic_policy


def policy():
    return {
        **arithmetic_policy(),
        "host_support": ["transpose", "reshape", "copy", "gather"],
        "host_float_arithmetic": ["bf16_add"],
        "already_quantized": POLICY,
    }


def source_request(**kwargs):
    source = base_source_request(**kwargs)
    output = source.node("y")
    attrs = dict(output.attrs)
    attrs[NUMERICS_ATTRIBUTE] = contraction_numerics_attribute(output.op, attrs["output_dtype"])
    return replace(source, nodes=(*source.nodes[:-1], replace(output, attrs=tuple(sorted(attrs.items())))))


def biased_source(*, batch=()):
    source = source_request(m=2, n=3, k=32, batch=batch)
    bias = SemanticNode("bias", "input", (), TensorType((3,), "bf16", "raw-bit-pattern"), effect="input")
    output = source.node("y")
    attrs = dict(output.attrs)
    attrs[NUMERICS_ATTRIBUTE] = contraction_numerics_attribute(output.op, attrs["output_dtype"], bias=True)
    output = replace(output, inputs=(*output.inputs, bias.id), attrs=tuple(sorted(attrs.items())))
    return replace(source, nodes=(*source.nodes[:-1], bias, output),
                   input_storages=(*source.input_storages, (bias.id, "external")))


def prepared_source():
    return split_prequantized_bias(biased_source())[0]


def adapter(selected=None):
    selected = policy() if selected is None else selected
    return AtlasApplicationAdapter(selected, original_source_sha256=selected["source_sha256"])


def test_unsplit_bias_cannot_enter_a_device_contraction():
    with pytest.raises(ValueError, match="numerical value admission requires unbiased source operands"):
        prepare_source(biased_source())


def test_batched_bias_readout_admits_each_batch_slice():
    source, receipt = split_prequantized_bias(biased_source(batch=(2,)))
    assert source.node("y.readout").type.shape == (2, 2, 3)
    assert receipt["changes"]
    plan = adapter().plan(source)
    assert [(dict(region.metadata)["kind"], region.placement) for region in plan.selected] == [
        ("already_quantized", "accelerator"), ("layout", "host"), ("float_arithmetic", "host")]


def batched_qk(*, rhs_batch=(2,), constant=False):
    """A batched score contraction: per-head queries against per-head keys."""
    source = source_request(m=2, n=3, k=32, batch=(2,))
    attrs = dict(source.node("y").attrs, output_dtype="float32")
    attrs[NUMERICS_ATTRIBUTE] = contraction_numerics_attribute("atlas.qk_matmul", "float32")
    shapes = dict(a=(2, 2, 32), sx=(2, 2, 1), w=(*rhs_batch, 3, 32), sw=(*rhs_batch, 1, 1))
    nodes = tuple(replace(node, op="atlas.qk_matmul", type=replace(node.type, dtype="f32"),
                          attrs=tuple(sorted(attrs.items()))) if node.id == "y" else
                  replace(node, type=replace(node.type, shape=shapes[node.id]),
                          **(dict(op="constant", effect="constant") if constant else {}))
                  for node in source.nodes)
    if not constant:
        return replace(source, nodes=nodes)
    # Nonzero codes 0x38 (1.0) and 0xb8 (-1.0) with distinct signed exponents per head.
    data = dict(a=bytes([0x38, 0xb8] * 64), sx=np.array([-3, 2, 0, 5], "<i4").tobytes(),
                w=bytes([0xb8, 0x38, 0x40] * 64), sw=np.array([1, -4], "<i4").tobytes())
    constants = tuple(ConstantBinding(key, "i8" if key in "aw" else "i32-le", value.hex()) for key, value in data.items())
    return replace(source, nodes=nodes, input_storages=(), constants=constants)


def test_batched_qk_matmul_admits_identical_prefixes_slice_by_slice():
    from atlas_mlir_native.source_translation import lower_already_quantized_contraction

    runtime = lower_already_quantized_contraction(batched_qk())
    assert runtime.batch_shape == (2,)
    assert runtime.value_admission == "available-constants-checked; runtime-input-validation-required"
    checked = lower_already_quantized_contraction(batched_qk(constant=True))
    assert checked.batch_shape == (2,)
    assert checked.value_admission == "all-constant-operands-checked"


def test_batched_qk_matmul_refuses_a_broadcast_key_prefix():
    from atlas_mlir_native.source_translation import lower_already_quantized_contraction

    with pytest.raises(ValueError, match="identical batch prefixes or one shared weight"):
        lower_already_quantized_contraction(batched_qk(rhs_batch=(1,)))


def test_bias_broadcast_and_add_are_host_even_when_device_routes_are_available():
    selected = adapter()
    selected._supports_device_source = Mock(return_value=True)
    source = prepared_source()
    plan = selected.plan(source)
    assert [(dict(region.metadata)["kind"], region.placement) for region in plan.selected] == [
        ("already_quantized", "accelerator"), ("layout", "host"), ("float_arithmetic", "host")]
    selected._supports_device_source.assert_not_called()
    region = extract_semantic_region(source, ("y",))
    compiled = selected.compile_region(None, region, plan.selected[-1], "bias_add", None, None, None)
    assert compiled.inputs == ("y.readout", "y.bias")
    assert compiled.outputs == ("y",)
    assert not compiled.programs
    assert compiled.description["numeric_semantics"] == "ieee754-bf16-add-rne-v1"


@pytest.mark.parametrize("field,value,message", [
    ("host_float_arithmetic", ["f64_mul"], "explicit host bf16_add"),
    ("host_support", ["transpose", "reshape", "copy"], "explicit host gather"),
])
def test_bias_requires_explicit_host_policy(field, value, message):
    with pytest.raises(NativeCompilationError, match=message):
        adapter({**policy(), field: value}).plan(prepared_source())


def test_f32_relu_after_bias_cannot_enter_removed_device_route():
    source = prepared_source()
    type_ = replace(source.node("y").type, dtype="f32")
    cast = SemanticNode("wide", "bf16_to_f32", ("y",), type_,
        attrs=(("numeric_semantics", "bf16-to-f32-exact-v1"),),
        index_maps=pointwise_identity_maps(len(type_.shape), 1))
    relu = SemanticNode("relu", "f32_relu_ordered_select", (cast.id,), type_,
        attrs=(("numeric_semantics", "f32-ordered-lt-positive-zero-select-v1"),),
        index_maps=pointwise_identity_maps(len(type_.shape), 1))
    source = replace(source, nodes=(*source.nodes, cast, relu), outputs=(relu.id,))
    with pytest.raises(ValueError, match="pointwise source operation, attributes or accounting differ"):
        adapter().plan(source)


def test_constant_folded_bias_add_stays_on_host():
    from merlin.compiler.semantic.source_preparation import (
        prepare_source as prepare_application_source, verify_source_preparation,
    )

    original = biased_source()
    bias = replace(original.node("bias"), op="constant", effect="constant")
    original = replace(
        original, nodes=tuple(bias if node.id == bias.id else node for node in original.nodes),
        input_storages=tuple(row for row in original.input_storages if row[0] != bias.id),
        constants=(ConstantBinding(bias.id, "bf16-le-raw", "803f00c04040"),),
    )
    source, receipt = prepare_application_source(original)
    assert source.node("y.bias").effect == "constant"
    assert verify_source_preparation(source, receipt).digest() == original.digest()
    selected = adapter()
    selected._supports_device_source = Mock(return_value=True)
    plan = selected.plan(source)
    assert [(dict(region.metadata)["kind"], region.placement) for region in plan.selected] == [
        ("already_quantized", "accelerator"), ("float_arithmetic", "host")]
    selected._supports_device_source.assert_not_called()


def with_bf16_relu(source, **changes):
    type_ = source.node("y").type
    relu = SemanticNode("relu", "relu", ("y",), type_,
        attrs=(("input_domain", "all-bf16-bit-patterns"),
               ("numeric_semantics", "bf16-ordered-lt-positive-zero-select-v1")),
        index_maps=pointwise_identity_maps(len(type_.shape), 1))
    relu = replace(relu, **changes)
    return replace(source, nodes=(*source.nodes, relu), outputs=(relu.id,))


def test_bf16_relu_follows_host_bias_with_disjoint_buffers():
    source = with_bf16_relu(prepared_source())
    selected = adapter()
    plan = selected.plan(source)
    assert [(dict(row.metadata)["kind"], row.placement) for row in plan.selected] == [
        ("already_quantized", "accelerator"), ("layout", "host"),
        ("float_arithmetic", "host"), ("host_ordered_relu", "host")]
    assert plan.selected[-1].nodes != plan.selected[-2].nodes


def test_host_relu_calls_checked_pointwise_interface(monkeypatch):
    import merlin.compiler.lower.c.pointwise_host as pointwise

    source = with_bf16_relu(prepared_source())
    selected = adapter()
    plan = selected.plan(source)
    region = extract_semantic_region(source, ("relu",))
    emitter = Mock(return_value=({"inputs": ["y"], "outputs": ["relu"]},
                                {"checked.h": "", "checked.c": "/* checked emitter */"}))
    monkeypatch.setattr(pointwise, "emit_host_pointwise_c", emitter)
    compiled = selected.compile_region(None, region, plan.selected[-1], "checked", None, None, None)
    emitter.assert_called_once_with(region, "checked")
    assert compiled.inputs == ("y",)
    assert compiled.outputs == ("relu",)
    assert compiled.files["checked.c"] == b"/* checked emitter */"


def test_host_relu_refuses_missing_checked_interface(monkeypatch):
    import merlin.compiler.lower.c.pointwise_host as pointwise

    monkeypatch.delattr(pointwise, "emit_host_pointwise_c")
    source = with_bf16_relu(prepared_source())
    selected = adapter()
    plan = selected.plan(source)
    with pytest.raises(NativeCompilationError, match="checked host pointwise C emitter"):
        selected.compile_region(None, extract_semantic_region(source, ("relu",)),
                                plan.selected[-1], "checked", None, None, None)


@pytest.mark.parametrize("change", [
    {"attrs": (("numeric_semantics", "unknown"),)},
    {"index_maps": ()},
])
def test_bf16_relu_refuses_undeclared_semantics(change):
    with pytest.raises(NativeCompilationError, match="no compiled application route"):
        adapter().plan(with_bf16_relu(prepared_source(), **change))


def test_bf16_relu_does_not_hide_shared_host_producer():
    source = with_bf16_relu(prepared_source())
    source = replace(source, outputs=("y", "relu"), output_storages=("external", "external"))
    selected = adapter()
    plan = selected.plan(source)
    assert [dict(row.metadata)["kind"] for row in plan.selected][-2:] == ["float_arithmetic", "host_ordered_relu"]
    region = extract_semantic_region(source, ("relu",))
    assert region.nodes[-1].inputs == ("y",)


def test_host_relu_rule_applies_to_non_bias_host_producer():
    source = with_bf16_relu(prepared_source())
    # A general host BF16 sign flip has no contraction/bias identity.
    neg = replace(source.node("y"), op="bf16_neg", inputs=("y.readout",),
        attrs=(("numeric_semantics", "bf16-sign-bit-flip-v1"),),
        index_maps=pointwise_identity_maps(len(source.node("y").type.shape), 1))
    source = replace(source, nodes=tuple(neg if node.id == "y" else node for node in source.nodes if node.id != "y.bias"))
    selected = adapter({**policy(), "host_float_arithmetic": ["bf16_add", "bf16_neg"]})
    selected._supports_device_source = Mock(return_value=False)
    plan = selected.plan(source)
    assert dict(plan.selected[-1].metadata)["kind"] == "host_ordered_relu"
    assert dict(plan.selected[-2].metadata)["kind"] == "float_arithmetic"


def test_bf16_relu_of_a_source_input_uses_the_checked_host_route():
    source = prepared_source()
    image = SemanticNode("image", "input", (), TensorType((1, 2, 2, 2), "bf16", "raw-bit-pattern"), effect="input")
    relu = SemanticNode("image_relu", "relu", ("image",), image.type,
        attrs=(("input_domain", "all-bf16-bit-patterns"),
               ("numeric_semantics", "bf16-ordered-lt-positive-zero-select-v1")),
        index_maps=pointwise_identity_maps(4, 1))
    source = replace(source, nodes=(*source.nodes, image, relu), outputs=(*source.outputs, relu.id),
                     input_storages=(*source.input_storages, (image.id, "external")),
                     output_storages=(*source.output_storages, "external"))
    plan = adapter().plan(source)
    assert (dict(plan.selected[-1].metadata)["kind"], plan.selected[-1].placement) == ("host_ordered_relu", "host")


def test_bf16_window_max_uses_the_checked_host_route():
    from merlin.compiler.semantic.model import IndexMap

    source = prepared_source()
    image = SemanticNode("image", "input", (), TensorType((1, 2, 5, 5), "bf16", "raw-bit-pattern"), effect="input")
    pool = SemanticNode("pool", "bf16_window_max", ("image",), TensorType((1, 2, 2, 2), "bf16", "raw-bit-pattern"),
        attrs=(("input_domain", "non-nan-bf16"), ("numeric_semantics", "ordered-window-max-first-equal-v1"),
               ("window_h", 3), ("window_w", 3)),
        index_maps=(IndexMap(6, ((1, 0, 0, 0, 0, 0), (0, 1, 0, 0, 0, 0), (0, 0, 2, 0, 1, 0), (0, 0, 0, 2, 0, 1)),
                             (0, 0, 0, 0)),
                    IndexMap(6, tuple(tuple(int(a == b) for b in range(6)) for a in range(4)), (0,) * 4)))
    source = replace(source, nodes=(*source.nodes, image, pool), outputs=(*source.outputs, pool.id),
                     input_storages=(*source.input_storages, (image.id, "external")),
                     output_storages=(*source.output_storages, "external"))
    plan = adapter().plan(source)
    assert (dict(plan.selected[-1].metadata)["kind"], plan.selected[-1].placement) == ("host_pointwise", "host")


def test_bf16_batch_norm_uses_the_checked_host_route():
    from merlin.compiler.semantic.model import IndexMap

    source = prepared_source()
    image = SemanticNode("image", "input", (), TensorType((1, 2, 2, 2), "bf16", "raw-bit-pattern"), effect="input")
    stats = [SemanticNode(name, "input", (), TensorType((2,), "bf16", "raw-bit-pattern"), effect="input")
             for name in ("mean", "var")]
    channel = IndexMap(4, ((0, 1, 0, 0),), (0,))
    identity = pointwise_identity_maps(4, 1)[0]
    norm = SemanticNode("norm", "bf16_batch_norm_inference", ("image", "mean", "var"), image.type,
        attrs=(("channel_operands", "mean,var"), ("epsilon_f32_bits", 0x3727C5AC), ("input_domain", "finite-bf16"),
               ("numeric_semantics", "torch-cpu-bf16-batch-norm-inference-v1")),
        index_maps=(identity, channel, channel, identity))
    source = replace(source, nodes=(*source.nodes, image, *stats, norm), outputs=(*source.outputs, norm.id),
                     input_storages=(*source.input_storages, *((node.id, "external") for node in (image, *stats))),
                     output_storages=(*source.output_storages, "external"))
    plan = adapter().plan(source)
    assert (dict(plan.selected[-1].metadata)["kind"], plan.selected[-1].placement) == ("host_pointwise", "host")


def test_bf16_sigmoid_uses_the_checked_host_route():
    source = prepared_source()
    image = SemanticNode("image", "input", (), TensorType((1, 32), "bf16", "raw-bit-pattern"), effect="input")
    sigmoid = SemanticNode("image_sigmoid", "bf16_sigmoid", ("image",), image.type,
        attrs=(("input_domain", "non-nan-bf16"), ("numeric_semantics", "torch-cpu-bf16-sigmoid-v1")),
        index_maps=pointwise_identity_maps(2, 1))
    source = replace(source, nodes=(*source.nodes, image, sigmoid), outputs=(*source.outputs, sigmoid.id),
                     input_storages=(*source.input_storages, (image.id, "external")),
                     output_storages=(*source.output_storages, "external"))
    plan = adapter().plan(source)
    assert (dict(plan.selected[-1].metadata)["kind"], plan.selected[-1].placement) == ("host_pointwise", "host")


@pytest.mark.parametrize("device_admitted,kind,placement", [
    (True, "exact_float_device", "accelerator"),
    (False, "float_arithmetic", "host"),
])
def test_unbiased_qlinear_residual_add_uses_base_routes(device_admitted, kind, placement):
    source = source_request(m=2, n=3, k=32, batch=())
    value = SemanticNode("residual", "input", (), source.node("y").type, effect="input")
    addition = SemanticNode("sum", "bf16_add", ("y", "residual"), value.type,
        attrs=(("numeric_semantics", "ieee754-bf16-add-rne-v1"),),
        index_maps=pointwise_identity_maps(len(value.type.shape), 2))
    source = replace(source, nodes=(*source.nodes, value, addition), outputs=("sum",),
        input_storages=(*source.input_storages, (value.id, "external")))
    selected = adapter({**policy(), "host_float_arithmetic": ["bf16_add"]})
    selected._supports_device_source = Mock(return_value=device_admitted)
    # Region admission is independently tested by exact_float_application.
    # Preserve its route here without requiring its separate precision policy.
    from unittest.mock import patch
    with patch("atlas_mlir_native.exact_float_application.prepare_plan"):
        plan = selected.plan(source)
    assert [(dict(row.metadata)["kind"], row.placement) for row in plan.selected] == [
        ("already_quantized", "accelerator"), (kind, placement)]
    selected._supports_device_source.assert_called_once_with(source, addition, "exact_float_device")
