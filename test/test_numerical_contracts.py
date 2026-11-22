"""Typed provider numerical claims bind real identities and retain refusals."""
from dataclasses import replace

import pytest
from atlas_mlir_native.compiler_interface import (
    describe_compiler,
    describe_numerical_families,
)
from atlas_mlir_native.numerical_contracts import MXU1_PRECISION_AUTHORIZATION
from atlas_mlir_native.source_translation import lower_already_quantized_contraction
from merlin.compiler.quantization.policy_generation import (
    PolicyGenerationError,
    generate_policy,
)
from merlin.compiler.semantic.model import (
    IndexMap,
    KernelRequest,
    SemanticNode,
    TensorType,
    pointwise_identity_maps,
)


def generate(source, **options):
    return generate_policy(source, describe_compiler(), describe_numerical_families(),
                           original_source_sha256="a" * 64, preparation_sha256="b" * 64,
                           **options)


def contraction(k=32, *, name="result", biased=False):
    def input_(key, shape, dtype, policy):
        return SemanticNode(key, "input", (), TensorType(shape, dtype, policy), effect="input")

    inputs = (
        input_("left", (2, k), "i8", "signed-i8"),
        input_("left_exp", (), "i32", "i32-wrap-k-ascending"),
        input_("right", (3, k), "i8", "signed-i8"),
        input_("right_exp", (3, 1), "i32", "i32-wrap-k-ascending"),
    )
    if biased:
        inputs += (input_("bias", (3,), "bf16", "raw-bit-pattern"),)
    from atlas_mlir_native.source_translation import contraction_numerics_attribute

    attrs = tuple(sorted({
        "numerical.contraction": contraction_numerics_attribute("atlas.qlinear", "bfloat16", bias=biased),
        "semantic_kind": "prequantized_contraction", "code_dtype": "uint8",
        "exponent_dtype": "int32", "weight_format": "atlas_e4m3",
        "dequantization": "e4m3(code)*2^exponent", "rhs_transposed": "true",
        "output_rescale": "2^(lhs_exp+rhs_exp)", "bias_order": "after_output_cast",
        "output_dtype": "bfloat16",
    }.items()))
    result = SemanticNode(name, "atlas.qlinear", tuple(node.id for node in inputs),
                          TensorType((2, 3), "bf16", "raw-bit-pattern"), attrs)
    source = KernelRequest((*inputs, result), (name,), ("external",),
                           tuple((node.id, "external") for node in inputs), "declared-target", "diagnostic")
    # Value admission is intentionally unbiased. Biased source declarations
    # must first be split into contraction and post-cast BF16 addition.
    if not biased:
        lower_already_quantized_contraction(source)
    return source


def test_declarations_extend_every_current_family_with_immutable_contracts():
    compiler = describe_compiler()
    rows = describe_numerical_families()
    assert {row.family for row in rows} == {family.id for family in compiler.families}
    assert len(rows) == 25
    for row in rows:
        family = next(family for family in compiler.families if family.id == row.family)
        assert row.pattern in family.patterns
        assert dict(row.numerical_contract)["schema"] == '"atlas.numerical-family.v1"'
        assert row.numerical_contract

    # Distinct host operations share an emitter kind but retain their own
    # numerical identities and original checked emitters.
    from merlin.compiler.lower.c.pointwise_host import emit_host_pointwise_c

    shape = (1, 2, 3, 3)
    type_ = TensorType(shape, "bf16", "raw-bit-pattern")
    activation = SemanticNode("activation", "input", (), type_, effect="input")
    channel = IndexMap(4, ((0, 1, 0, 0),), (0,))
    identity = pointwise_identity_maps(4, 1)[0]
    terms = tuple(SemanticNode(role, "input", (), TensorType((2,), "f32", "raw-bit-pattern"), effect="input")
                  for role in ("mean", "var", "weight", "bias"))
    batch_norm = SemanticNode("result", "bf16_batch_norm_inference", ("activation", *(term.id for term in terms)),
        type_, (("channel_operands", "mean,var,weight,bias"), ("epsilon_f32_bits", 925353388),
                ("input_domain", "finite-bf16"), ("numeric_semantics", "torch-cpu-bf16-batch-norm-inference-v1")),
        index_maps=(identity, *(channel,) * 4, identity))
    window = SemanticNode("result", "bf16_window_max", ("activation",), TensorType((1, 2, 1, 1), "bf16", "raw-bit-pattern"),
        (("input_domain", "non-nan-bf16"), ("numeric_semantics", "ordered-window-max-first-equal-v1"), ("window_h", 3), ("window_w", 3)),
        index_maps=(IndexMap(6, ((1, 0, 0, 0, 0, 0), (0, 1, 0, 0, 0, 0), (0, 0, 1, 0, 1, 0), (0, 0, 0, 1, 0, 1)), (0, 0, 0, 0)),
                    IndexMap(6, ((1, 0, 0, 0, 0, 0), (0, 1, 0, 0, 0, 0), (0, 0, 1, 0, 0, 0), (0, 0, 0, 1, 0, 0)), (0, 0, 0, 0))))
    sigmoid = SemanticNode("result", "bf16_sigmoid", ("activation",), type_,
        (("input_domain", "non-nan-bf16"), ("numeric_semantics", "torch-cpu-bf16-sigmoid-v1")),
        index_maps=pointwise_identity_maps(4, 1))
    gelu = SemanticNode("result", "bf16_gelu", ("activation",), type_,
        (("approximate", "tanh"), ("input_domain", "finite-bf16"), ("numeric_semantics", "bf16-gelu-f32-opmath-v1")),
        index_maps=pointwise_identity_maps(4, 1))
    affine = tuple(SemanticNode(role, "input", (), TensorType((3,), "bf16", "raw-bit-pattern"), effect="input")
                   for role in ("weight", "bias"))
    trailing = IndexMap(4, ((0, 0, 0, 1),), (0,))
    layer_norm = SemanticNode("result", "bf16_layer_norm", ("activation", "weight", "bias"), type_,
        (("affine_operands", "weight,bias"), ("epsilon_f32_bits", 0x358637BD), ("input_domain", "finite-bf16"),
         ("normalized_rank", 1), ("numeric_semantics", "torch-2.11-cpu-fma-bf16-layer-norm-v1")),
        index_maps=(identity, trailing, trailing, identity))
    negation = SemanticNode("result", "bf16_neg", ("activation",), type_,
        (("numeric_semantics", "bf16-sign-bit-flip-v1"),), index_maps=pointwise_identity_maps(4, 1))
    for node, extra, family in ((negation, (), "atlas.host-sign-flip"), (batch_norm, terms, "atlas.host-batch-norm"),
                                (window, (), "atlas.host-window-max"), (sigmoid, (), "atlas.host-sigmoid"),
                                (gelu, (), "atlas.host-gelu"), (layer_norm, affine, "atlas.host-layer-norm")):
        inputs = (activation, *extra)
        source = KernelRequest((*inputs, node), (node.id,), ("external",),
                               tuple((value.id, "external") for value in inputs), "declared-target")
        policy = generate(source)
        assert policy["selected_families"][0]["family"] == family
        assert policy["selected_families"][0]["precision_change"] is None
        emit_host_pointwise_c(source, "cnn")
        changed = replace(node, attrs=tuple((key, "changed" if key == "numeric_semantics" else value)
                                           for key, value in node.attrs))
        with pytest.raises(PolicyGenerationError):
            generate(replace(source, nodes=(*inputs, changed)))


def test_two_k_shapes_keep_ordered_contract_and_require_separate_authority():
    for k in (32, 64):
        source = contraction(k)
        with pytest.raises(PolicyGenerationError, match="0 authorized matches"):
            generate(source)
        policy = generate(source, authorized_precision_changes=(MXU1_PRECISION_AUTHORIZATION,))
        claim = policy["selected_families"][0]
        assert claim["family"] == "atlas.ordered-quantized-contraction"
        assert claim["status"] == "implementation_claim_pending_admission"
        contract = claim["numerical_contract"]
        assert contract["accumulation"]["instruction_k"] == 32
        assert contract["accumulation"]["readout"] == "BF16 RNE after every instruction"
        assert contract["accumulation"]["unit"] == "mxu1"
        assert contract["exponents"]["operand_bounds"] == [-127, 127]
        assert contract["exponents"]["paired_sum_bounds"] == [-133, 127]
        assert contract["encoding"]["reserved_codes"] == [127, 255]
        assert contract["runtime"]["publication"].startswith("caller output unchanged")
        assert policy["compatibility_policy_fields"] == {
            "already_quantized": "atlas-already-quantized-e4m3-signed-exponents-mxu1-model47-v1"}
        renamed = generate(contraction(k, name="other_projection"),
                           authorized_precision_changes=(MXU1_PRECISION_AUTHORIZATION,))
        assert renamed["selected_families"][0]["family"] == claim["family"]
        assert renamed["graph_sha256"] != policy["graph_sha256"]


def test_bias_and_changed_representation_do_not_acquire_contraction_claim():
    with pytest.raises(PolicyGenerationError):
        generate(contraction(biased=True), authorized_precision_changes=(MXU1_PRECISION_AUTHORIZATION,))
    source = contraction()
    for field, changed in (("bias_order", "before_output_cast"), ("weight_format", "different_encoder"),
                           ("output_rescale", "2^lhs_exp"), ("rhs_transposed", "false")):
        attrs = dict(source.nodes[-1].attrs)
        attrs[field] = changed
        result = replace(source.nodes[-1], attrs=tuple(sorted(attrs.items())))
        with pytest.raises(PolicyGenerationError):
            generate(replace(source, nodes=(*source.nodes[:-1], result)),
                     authorized_precision_changes=(MXU1_PRECISION_AUTHORIZATION,))
    # A family claim for BF16 output cannot silently cover the existing f32 route.
    result = replace(source.nodes[-1], type=replace(source.nodes[-1].type, dtype="f32"))
    with pytest.raises(PolicyGenerationError):
        generate(replace(source, nodes=(*source.nodes[:-1], result)),
                 authorized_precision_changes=(MXU1_PRECISION_AUTHORIZATION,))


def test_overlapping_layouts_require_explicit_implementation_selection():
    type_ = TensorType((2, 3), "bf16", "raw-bit-pattern")
    input_ = SemanticNode("input", "input", (), type_, effect="input")
    output = SemanticNode("output", "copy", ("input",), type_)
    source = KernelRequest((input_, output), ("output",), ("external",),
                           (("input", "external"),), "declared-target")
    with pytest.raises(PolicyGenerationError, match="3 authorized matches"):
        generate(source)
    for family in ("atlas.host-layout", "atlas.device-layout", "atlas.layout"):
        selected = generate(source, selected_family_ids=(family,))
        assert selected["selected_families"][0]["family"] == family
        assert selected["authorized_precision_changes"] == []
        assert selected["compatibility_policy_fields"] == {}


def test_source_bias_add_claim_preserves_post_cast_rne_and_runtime_domain():
    from merlin.compiler.semantic.source_preparation import prepare_source

    prepared, _ = prepare_source(contraction(biased=True))
    claims = generate(
        prepared, authorized_precision_changes=(MXU1_PRECISION_AUTHORIZATION,),
        selected_family_ids=tuple(row.family for row in describe_numerical_families()
                                  if row.family not in {"atlas.device-layout", "atlas.layout"}),
    )["selected_families"]
    selected = next(row for row in claims if row["family"] == "atlas.host-bias-add")
    assert selected["implementation_policy"] == "ieee754-bf16-add-rne-v1"
    contract = selected["numerical_contract"]
    assert contract["arithmetic"]["rounding"] == "BF16 RNE"
    assert "output cast" in contract["arithmetic"]["bias_position"]
    assert "invalid infinity" in contract["runtime"]["domain"]
    assert "primitive VPU FP32 RNE then BF16 chop" in contract["arithmetic"]["forbidden_substitutions"]
    assert selected["precision_change"] is None

    # Ordinary BF16 adds cannot acquire the explicitly prepared bias contract.
    type_ = TensorType((2, 3), "bf16", "raw-bit-pattern")
    inputs = tuple(SemanticNode(key, "input", (), type_, effect="input") for key in ("readout", "bias"))
    add = SemanticNode("added", "bf16_add", ("readout", "bias"), type_)
    source = KernelRequest((*inputs, add), ("added",), ("external",),
                           tuple((node.id, "external") for node in inputs), "declared-target")
    with pytest.raises(PolicyGenerationError):
        generate(source)
    residual = replace(add, attrs=(("numeric_semantics", "ieee754-bf16-add-rne-v1"),))
    selected = generate(replace(source, nodes=(*inputs, residual)))["selected_families"]
    assert [claim["family"] for claim in selected] == ["atlas.exact-bf16-add"]
    assert selected[0]["precision_change"] is None


def test_source_bf16_multiply_claims_the_bounded_device_route_only_with_its_semantics():
    from atlas_mlir_native.generated_export_policy import EXPORT_FAMILIES
    from atlas_mlir_native.generated_policy import FAMILY_KINDS

    type_ = TensorType((1, 4, 3, 3), "bf16", "raw-bit-pattern")
    inputs = tuple(SemanticNode(key, "input", (), type_, effect="input") for key in ("x", "scale"))
    mul = SemanticNode("scaled", "bf16_mul", ("x", "scale"), type_,
                       attrs=(("numeric_semantics", "ieee754-bf16-mul-rne-v1"),),
                       index_maps=pointwise_identity_maps(4, 2))
    source = KernelRequest((*inputs, mul), ("scaled",), ("external",),
                           tuple((node.id, "external") for node in inputs), "declared-target")
    (claim,) = generate(source)["selected_families"]
    assert claim["family"] == "atlas.device-bf16-mul" and FAMILY_KINDS[claim["family"]] == "pointwise"
    assert claim["implementation_policy"] == "bf16-bounded-pointwise-rne-v1" and claim["precision_change"] is None
    assert "normal and finite" in claim["numerical_contract"]["arithmetic"]["domain"]
    assert "atlas.device-bf16-mul" in EXPORT_FAMILIES
    # Another declared multiply semantics has no family.
    with pytest.raises(PolicyGenerationError):
        generate(replace(source, nodes=(*inputs, replace(mul, attrs=(("numeric_semantics", "other-v1"),)))))


def test_exact_bf16_add_stays_on_the_device_when_admitted_and_takes_the_host_adder_when_refused():
    from types import SimpleNamespace

    from atlas_mlir_native.binding import TARGET_IDENTITY
    from atlas_mlir_native.exact_float_device import prepare_source_plan
    from atlas_mlir_native.generated_policy import selected_kind

    type_ = TensorType((2, 32), "bf16", "raw-bit-pattern")
    inputs = tuple(SemanticNode(key, "input", (), type_, effect="input") for key in ("a", "b"))

    def source(order):
        add = SemanticNode("sum", "bf16_add", order, type_, attrs=(("numeric_semantics", "ieee754-bf16-add-rne-v1"),),
                           index_maps=pointwise_identity_maps(2, 2))
        return KernelRequest((*inputs, add), ("sum",), ("external",), tuple((node.id, "external") for node in inputs),
                             TARGET_IDENTITY, "diagnostic")

    adapter = SimpleNamespace(_generated_family_claims=(("sum", "atlas.exact-bf16-add"),))
    assert selected_kind(adapter, source(("a", "b")), ("sum",)) == "exact_float_device"
    # Operands in the other order than the region's inputs: the device route refuses, unchanged.
    with pytest.raises(ValueError, match="exact float source types"):
        prepare_source_plan(source(("b", "a")))
    assert selected_kind(adapter, source(("b", "a")), ("sum",)) == "float_arithmetic"


def test_captured_state_read_is_bounded_by_declared_off_chip_memory():
    from atlas_mlir_native.soc import CAPTURED_STATE_READ_BYTES, SOC

    assert CAPTURED_STATE_READ_BYTES == SOC.memory_region("ddr").capacity_bytes == 1 << 30


def test_authored_static_quantizer_retains_paired_outputs_and_exception_rules():
    from atlas_mlir_native.static_quantize import prepare_source

    input_ = SemanticNode("activation", "input", (), TensorType((32,), "bf16", "raw-bit-pattern"), effect="input")
    attrs = tuple(sorted({
        "mode": "static", "reduce": "tensor", "rule": "ref", "static_exp": -2, "rows": 1,
        "semantic_kind": "static_activation_quantizer", "code_dtype": "uint8",
        "exponent_dtype": "int32", "weight_format": "atlas_e4m3",
        "dequantization": "e4m3(code)*2^exponent",
    }.items()))
    codes = SemanticNode("codes", "atlas.quantize_act.codes", ("activation",),
                         TensorType((32,), "i8", "signed-i8"), attrs)
    exponent = SemanticNode("exp", "atlas.quantize_act.exponent", ("activation",),
                            TensorType((), "i32", "i32-wrap-k-ascending"), attrs)
    source = KernelRequest((input_, codes, exponent), ("codes", "exp"), ("external", "external"),
                           (("activation", "external"),), "declared-target", "diagnostic")
    prepare_source(source)
    with pytest.raises(PolicyGenerationError, match="2 authorized matches"):
        generate(source)
    for family in ("atlas.activation-quantize", "atlas.host-static-quantize"):
        selected = generate(source, selected_family_ids=(family,))["selected_families"]
        assert len(selected) == 2
        assert {row["family"] for row in selected} == {family}
        contract = selected[0]["numerical_contract"]
        assert contract["encoder"]["nan"] == "positive zero"
        assert contract["encoder"]["infinity"] == "signed finite saturation"
        assert "both projections required" in contract["source_contract"]["pairing"]
        assert all(row["precision_change"] is None for row in selected)


def test_stale_compiler_identity_cannot_publish_numerical_declaration(monkeypatch, tmp_path):
    from types import SimpleNamespace

    import atlas_mlir_native.compiler_interface as interface

    labels = {label for label, _, _ in interface._source_roots()}
    assert {"merlin-host-ops", "merlin-host-math", "merlin-host-math-core", "merlin-host-math-pow"} <= labels
    helper = tmp_path / "operator.py"
    helper.write_text("version = 1\n")
    monkeypatch.setattr(interface, "_source_roots", lambda: (("host-ops", tmp_path, (".py",)),))
    alternative = SimpleNamespace(metadata=(("kind", "host_pointwise"),
                                           ("numerical_family", "atlas.host-window-max")))
    recipe = interface.lowering_recipe(alternative)
    assert dict(recipe.guards)["family"] == "atlas.host-window-max"
    helper.write_text("version = 2\n")
    with pytest.raises(ValueError, match="implementation sources changed"):
        interface.validate_recipe(recipe)

    compiler = describe_compiler()
    monkeypatch.setattr(interface, "describe_compiler", lambda: replace(
        compiler, families=compiler.families[:-1], patterns=compiler.patterns[:-1]))
    with pytest.raises(ValueError, match="family/pattern identity"):
        describe_numerical_families()


def test_ordered_select_generation_preserves_source_bits_and_generated_route():
    from atlas_mlir_native.application import AtlasApplicationAdapter
    from atlas_mlir_native.binding import TARGET_IDENTITY
    from atlas_mlir_native.generated_export_policy import EXPORT_FAMILIES
    from atlas_mlir_native.generated_policy import selected_kind
    from merlin.compiler.lower.c.host_ops.relu import SELECT_ATTRS

    type_ = TensorType((8,), "bf16", "raw-bit-pattern")
    operand = SemanticNode("input", "input", (), type_, effect="input")
    result = SemanticNode("output", "relu", ("input",), type_, tuple(sorted(SELECT_ATTRS.items())),
                          index_maps=pointwise_identity_maps(1, 1))
    source = KernelRequest((operand, result), ("output",), ("external",),
                           (("input", "external"),), TARGET_IDENTITY, "diagnostic")
    policy = generate(source, selected_family_ids=EXPORT_FAMILIES)
    claim, = policy["selected_families"]
    assert claim["family"] == "atlas.host-ordered-select"
    arithmetic = claim["numerical_contract"]["arithmetic"]
    assert arithmetic["negative_zero"] == "preserve negative zero"
    assert arithmetic["nan"] == "preserve sign and payload bits"
    assert claim["precision_change"] is None
    from atlas_mlir_native.application_precision import CONTRACTION_POLICY
    from atlas_mlir_native.row_softmax import SOFTMAX_POLICY

    adapter = AtlasApplicationAdapter({
        "schema": "atlas.diagnostic_application_precision.v1", "source_sha256": "a" * 64,
        "target_identity": TARGET_IDENTITY, "contraction": CONTRACTION_POLICY,
        "row_softmax": SOFTMAX_POLICY, "host_support": ["transpose", "reshape", "copy"],
    }, original_source_sha256="a" * 64)
    adapter._generated_source_digest = source.digest()
    adapter._generated_family_claims = (("output", "atlas.host-ordered-select"),)
    adapter._generated_policy_json = "{}"
    assert selected_kind(adapter, source, ("output",)) == "host_ordered_relu"
    plan = adapter.plan(source)
    assert dict(plan.selected[0].metadata)["kind"] == "host_ordered_relu"


def test_device_routes_name_their_host_alternates_only_for_their_own_family():
    from atlas_mlir_native.compiler_interface import _guards
    from atlas_mlir_native.generated_export_policy import EXPORT_FAMILIES

    assert "atlas.activation-quantize" in EXPORT_FAMILIES and "atlas.host-static-quantize" not in EXPORT_FAMILIES
    for family, kind in (("atlas.activation-quantize", "static_quantize"), ("atlas.activation-quantize", "dynamic_quantize"),
                         ("atlas.exact-bf16-add", "float_arithmetic")):
        assert dict(_guards(kind, family_id=family))["family"] == family
    # Without an explicit family selection the host kinds keep their single declared family.
    assert dict(_guards("static_quantize"))["family"] == "atlas.host-static-quantize"
    with pytest.raises(ValueError, match="no declared implementation family"):
        _guards("float_arithmetic")  # several host arithmetic families: a claim must name one


@pytest.mark.parametrize("operation", ["add", "sub", "mul", "div"])
def test_source_f32_arithmetic_claims_its_host_family_and_refuses_other_semantics(operation):
    type_ = TensorType((2, 3), "f32", "raw-bit-pattern")
    inputs = tuple(SemanticNode(key, "input", (), type_, effect="input") for key in ("a", "b"))
    node = SemanticNode("out", f"f32_{operation}", ("a", "b"), type_,
                        attrs=(("numeric_semantics", f"ieee754-f32-{operation}-rne-v1"),),
                        index_maps=pointwise_identity_maps(2, 2))
    source = KernelRequest((*inputs, node), ("out",), ("external",),
                           tuple((item.id, "external") for item in inputs), "declared-target")
    (claim,) = generate(source)["selected_families"]
    assert claim["family"] == f"atlas.host-f32-{operation}" and claim["precision_change"] is None
    with pytest.raises(PolicyGenerationError):
        generate(replace(source, nodes=(*inputs, replace(node, attrs=(("numeric_semantics", "other-v1"),)))))


@pytest.mark.parametrize("op,semantics,src,dst", [("bf16_to_f32", "bf16-to-f32-exact-v1", "bf16", "f32"),
                                                  ("f32_to_bf16", "f32-to-bf16-rne-v1", "f32", "bf16")])
def test_source_casts_claim_their_exact_family_with_a_host_alternate(op, semantics, src, dst):
    from types import SimpleNamespace

    from atlas_mlir_native.binding import TARGET_IDENTITY
    from atlas_mlir_native.generated_policy import selected_kind

    family = "atlas.exact-" + op.replace("_", "-")
    value = SemanticNode("x", "input", (), TensorType((2, 32), src, "raw-bit-pattern"), effect="input")
    cast = SemanticNode("y", op, ("x",), TensorType((2, 32), dst, "raw-bit-pattern"),
                        attrs=(("numeric_semantics", semantics),), index_maps=pointwise_identity_maps(2, 1))
    source = KernelRequest((value, cast), ("y",), ("external",), (("x", "external"),), "declared-target")
    (claim,) = generate(source)["selected_families"]
    assert claim["family"] == family and claim["precision_change"] is None
    with pytest.raises(PolicyGenerationError):
        generate(replace(source, nodes=(value, replace(cast, attrs=(("numeric_semantics", "other-v1"),)))))
    adapter = SimpleNamespace(_generated_family_claims=(("y", family),))
    device = replace(source, target_identity=TARGET_IDENTITY, lowering_policy="diagnostic")
    assert selected_kind(adapter, device, ("y",)) == "exact_float_device"
    # A constant input is outside the device route's closed region: the host cast takes it.
    constant = replace(value, op="constant", effect="constant")
    from merlin.compiler.semantic.model import ConstantBinding
    width = 2 if src == "bf16" else 4
    folded = replace(device, nodes=(constant, cast), input_storages=(),
                     constants=(ConstantBinding("x", "bf16-le-raw" if src == "bf16" else "f32-le-raw",
                                                "00" * (64 * width)),))
    assert selected_kind(adapter, folded, ("y",)) == "float_cast"


def test_f32_output_contraction_claims_its_own_family_with_the_same_authority():
    from atlas_mlir_native.source_translation import contraction_numerics_attribute

    bf16 = contraction()
    result = bf16.node("result")
    attrs = dict(result.attrs, output_dtype="float32")
    attrs["numerical.contraction"] = contraction_numerics_attribute("atlas.qlinear", "float32")
    widened = replace(result, type=replace(result.type, dtype="f32"), attrs=tuple(sorted(attrs.items())))
    source = replace(bf16, nodes=(*bf16.nodes[:-1], widened))
    grant = (MXU1_PRECISION_AUTHORIZATION,)
    (claim,) = generate(source, authorized_precision_changes=grant)["selected_families"]
    assert claim["family"] == "atlas.ordered-quantized-contraction-f32"
    assert claim["numerical_contract"]["boundaries"]["output_dtype"] == "f32"
    (claim,) = generate(bf16, authorized_precision_changes=grant)["selected_families"]
    assert claim["family"] == "atlas.ordered-quantized-contraction"
    # The MXU1 precision change still needs its explicit authority.
    with pytest.raises(PolicyGenerationError):
        generate(source)


def test_f32_static_quantizer_claims_its_host_family_and_refuses_a_runtime_scale():
    from types import SimpleNamespace

    from atlas_mlir_native.binding import TARGET_IDENTITY
    from atlas_mlir_native.compiler_interface import _guards
    from atlas_mlir_native.generated_export_policy import EXPORT_FAMILIES
    from atlas_mlir_native.generated_policy import selected_kind
    from atlas_mlir_native.static_quantize import prepare_source

    input_ = SemanticNode("x", "input", (), TensorType((32,), "f32", "raw-bit-pattern"), effect="input")
    fields = {"mode": "static", "reduce": "tensor", "rule": "ref", "static_exp": -6, "rows": 32,
              "semantic_kind": "static_activation_quantizer", "code_dtype": "uint8", "exponent_dtype": "int32",
              "weight_format": "atlas_e4m3", "dequantization": "e4m3(code)*2^exponent"}

    def request(values):
        attrs = tuple(sorted(values.items()))
        codes = SemanticNode("codes", "atlas.quantize_act.codes", ("x",), TensorType((32,), "i8", "signed-i8"), attrs)
        exponent = SemanticNode("exp", "atlas.quantize_act.exponent", ("x",),
                                TensorType((), "i32", "i32-wrap-k-ascending"), attrs)
        return KernelRequest((input_, codes, exponent), ("codes", "exp"), ("external", "external"),
                             (("x", "external"),), TARGET_IDENTITY, "diagnostic")

    source = request(fields)
    prepare_source(source)
    claims = generate(source, selected_family_ids=EXPORT_FAMILIES)["selected_families"]
    assert {row["family"] for row in claims} == {"atlas.host-static-quantize-f32"}
    assert all(row["precision_change"] is None for row in claims)
    assert claims[0]["numerical_contract"]["encoder"]["source_dtype"] == "f32"
    adapter = SimpleNamespace(_generated_family_claims=tuple((row["node"], row["family"]) for row in claims))
    assert selected_kind(adapter, source, ("codes", "exp")) == "static_quantize"
    assert dict(_guards("static_quantize", family_id="atlas.host-static-quantize-f32"))["selected-kind"] == "static_quantize"
    # Unnamed static routes keep their BF16 family.
    assert dict(_guards("static_quantize"))["family"] == "atlas.host-static-quantize"
    dynamic = {key: value for key, value in fields.items() if key != "static_exp"}
    dynamic.update(mode="dynamic", semantic_kind="dynamic_activation_quantizer")
    with pytest.raises(PolicyGenerationError):
        generate(request(dynamic), selected_family_ids=EXPORT_FAMILIES)


def test_layout_family_takes_the_device_only_where_the_old_planner_rule_admits_it():
    from types import SimpleNamespace

    from atlas_mlir_native.binding import TARGET_IDENTITY
    from atlas_mlir_native.compiler_interface import _guards
    from atlas_mlir_native.generated_policy import selected_kind

    type_ = TensorType((2, 32), "bf16", "raw-bit-pattern")
    input_ = SemanticNode("x", "input", (), type_, effect="input")
    copy = SemanticNode("y", "copy", ("x",), type_, index_maps=pointwise_identity_maps(2, 1))
    source = KernelRequest((input_, copy), ("y",), ("external",), (("x", "external"),), TARGET_IDENTITY, "diagnostic")
    (claim,) = generate(source, selected_family_ids=("atlas.layout",))["selected_families"]
    assert claim["family"] == "atlas.layout" and claim["precision_change"] is None

    def adapter(*, quantized=True, device=True):
        return SimpleNamespace(_generated_family_claims=(("y", "atlas.layout"),), already_quantized=quantized,
                               _is_scalar_broadcast=lambda *_: False, _supports_device_source=lambda *_: device)

    assert selected_kind(adapter(), source, ("y",)) == "layout_device"
    assert selected_kind(adapter(device=False), source, ("y",)) == "layout"
    assert selected_kind(adapter(quantized=False), source, ("y",)) == "layout"
    for kind in ("layout_device", "layout"):
        assert dict(_guards(kind, family_id="atlas.layout"))["family"] == "atlas.layout"
    # Unnamed layout routes keep their own single family.
    assert dict(_guards("layout"))["family"] == "atlas.host-layout"
    assert dict(_guards("layout_device"))["family"] == "atlas.device-layout"


@pytest.mark.parametrize("mode, count, kind", [
    ("static", 64, "activation_quantize_device"),
    ("static", 48, "static_quantize"),
    ("dynamic", 48, "dynamic_quantize"),
    ("other", 48, None),
])
def test_generated_quantizer_cover_places_each_quantizer_by_device_admission_and_mode(mode, count, kind):
    """Planner level: the device where its vector body is admitted, else the host encoder of the declared mode."""
    from atlas_mlir_native.application import AtlasApplicationAdapter
    from atlas_mlir_native.binding import TARGET_IDENTITY
    from atlas_mlir_native.generated_policy import adapter_for_claims
    from merlin.compiler.semantic.target_binding import NativeCompilationError

    fields = {"mode": mode, "reduce": "tensor", "rule": "ref", "rows": 32,
              "semantic_kind": "dynamic_activation_quantizer" if mode == "dynamic" else "static_activation_quantizer",
              "code_dtype": "uint8", "exponent_dtype": "int32", "weight_format": "atlas_e4m3",
              "dequantization": "e4m3(code)*2^exponent", "tag": ""}
    if mode != "dynamic":
        fields["static_exp"] = -6
    attrs = tuple(sorted(fields.items()))
    input_ = SemanticNode("x", "input", (), TensorType((count,), "bf16", "raw-bit-pattern"), effect="input")
    codes = SemanticNode("codes", "atlas.quantize_act.codes", ("x",), TensorType((count,), "i8", "signed-i8"), attrs)
    exponent = SemanticNode("exp", "atlas.quantize_act.exponent", ("x",),
                            TensorType((), "i32", "i32-wrap-k-ascending"), attrs)
    source = KernelRequest((input_, codes, exponent), ("codes", "exp"), ("external", "external"),
                           (("x", "external"),), TARGET_IDENTITY, "diagnostic")
    claims = (("codes", "atlas.activation-quantize"), ("exp", "atlas.activation-quantize"))
    adapter = adapter_for_claims(AtlasApplicationAdapter, source, claims, source_sha256="0" * 64, policy_json="{}")
    if kind is None:
        with pytest.raises(NativeCompilationError, match="declares no admitted mode"):
            adapter.plan(source)
        return
    (region,) = adapter.plan(source).selected
    assert dict(region.metadata)["kind"] == kind
    assert len(region.nodes) == 2


def test_exact_bf16_add_refuses_a_folded_contraction_bias_and_gather_bias_keeps_its_host_family():
    from types import SimpleNamespace

    from atlas_mlir_native.binding import TARGET_IDENTITY
    from atlas_mlir_native.generated_policy import selected_kind
    from merlin.compiler.semantic.model import ConstantBinding
    from merlin.compiler.semantic.source_preparation import is_prequantized_bias_add, prepare_source
    from merlin.compiler.semantic.target_binding import NativeCompilationError

    prepared, _ = prepare_source(replace(contraction(biased=True), target_identity=TARGET_IDENTITY))
    (bias_add,) = [node for node in prepared.nodes if is_prequantized_bias_add(prepared, node)]
    gather = prepared.node(bias_add.inputs[1])
    assert gather.op == "gather"
    adapter = SimpleNamespace(_generated_family_claims=((bias_add.id, "atlas.host-bias-add"),))
    assert selected_kind(adapter, prepared, (bias_add.id,)) == "float_arithmetic"

    # The same bias folded to one declared BF16 constant is still prepared contraction bias.
    folded_bias = SemanticNode(gather.id, "constant", (), gather.type, effect="constant")
    nodes = tuple(folded_bias if node.id == gather.id else node for node in prepared.nodes)
    folded = replace(prepared, nodes=nodes, constants=(
        *prepared.constants, ConstantBinding(gather.id, "bf16-le-raw", "00" * (2 * 2 * 3))))
    assert is_prequantized_bias_add(folded, folded.node(bias_add.id))
    adapter = SimpleNamespace(_generated_family_claims=((bias_add.id, "atlas.exact-bf16-add"),))
    with pytest.raises(NativeCompilationError, match="excludes prepared contraction bias"):
        selected_kind(adapter, folded, (bias_add.id,))


def test_alternate_routes_are_one_declared_table_of_same_contract_host_routes(monkeypatch):
    from types import SimpleNamespace

    from atlas_mlir_native import compiler_interface
    from atlas_mlir_native.binding import TARGET_IDENTITY
    from atlas_mlir_native.generated_policy import FAMILY_KINDS, selected_kind
    from merlin.compiler.semantic.target_binding import NativeCompilationError

    rows = {row.family: row for row in describe_numerical_families()}
    families = {"atlas." + family for family, _, _ in compiler_interface._ALTERNATE_ROUTES}
    assert families == {"atlas.activation-quantize", "atlas.exact-bf16-add", "atlas.exact-bf16-to-f32",
                        "atlas.exact-f32-to-bf16", "atlas.layout"}
    for family in families:
        # Same source operation and numerical contract: no precision change, and the contract names the host route.
        assert rows[family].precision_change is None
        assert "otherwise the checked host" in " ".join(value for _, value in rows[family].numerical_contract)
        assert FAMILY_KINDS[family] not in compiler_interface.alternate_kinds(family)
    # A precision-changing family can never declare an alternate.
    changed = next(family for family, row in rows.items() if row.precision_change is not None)
    monkeypatch.setattr(compiler_interface, "_ALTERNATE_ROUTES", (
        *compiler_interface._ALTERNATE_ROUTES, (changed.removeprefix("atlas."), "float_arithmetic", ("atlas.qlinear",))))
    with pytest.raises(ValueError, match="precision-changing Atlas family cannot declare an alternate"):
        describe_numerical_families()
    # The planner takes no alternate the table does not declare.
    monkeypatch.setattr(compiler_interface, "_ALTERNATE_ROUTES", tuple(
        row for row in compiler_interface._ALTERNATE_ROUTES if row[0] != "exact-bf16-add"))
    type_ = TensorType((2, 32), "bf16", "raw-bit-pattern")
    inputs = tuple(SemanticNode(key, "input", (), type_, effect="input") for key in ("a", "b"))
    add = SemanticNode("sum", "bf16_add", ("b", "a"), type_, attrs=(("numeric_semantics", "ieee754-bf16-add-rne-v1"),),
                       index_maps=pointwise_identity_maps(2, 2))
    source = KernelRequest((*inputs, add), ("sum",), ("external",), tuple((node.id, "external") for node in inputs),
                           TARGET_IDENTITY, "diagnostic")
    adapter = SimpleNamespace(_generated_family_claims=(("sum", "atlas.exact-bf16-add"),))
    assert selected_kind(adapter, source, ("sum",)) == "exact_float_device"
    monkeypatch.setattr(compiler_interface, "_ALTERNATE_ROUTES", tuple(
        row for row in compiler_interface._ALTERNATE_ROUTES if row[0] != "activation-quantize"))
    attrs = tuple(sorted({"mode": "static", "reduce": "tensor", "rule": "ref", "rows": 32, "static_exp": -6, "tag": "",
                          "semantic_kind": "static_activation_quantizer", "code_dtype": "uint8",
                          "exponent_dtype": "int32", "weight_format": "atlas_e4m3",
                          "dequantization": "e4m3(code)*2^exponent"}.items()))
    input_ = SemanticNode("x", "input", (), TensorType((48,), "bf16", "raw-bit-pattern"), effect="input")
    codes = SemanticNode("codes", "atlas.quantize_act.codes", ("x",), TensorType((48,), "i8", "signed-i8"), attrs)
    exponent = SemanticNode("exp", "atlas.quantize_act.exponent", ("x",), TensorType((), "i32", "i32-wrap-k-ascending"),
                            attrs)
    source = KernelRequest((input_, codes, exponent), ("codes", "exp"), ("external", "external"),
                           (("x", "external"),), TARGET_IDENTITY, "diagnostic")
    from atlas_mlir_native.application import AtlasApplicationAdapter

    adapter = SimpleNamespace(_generated_family_claims=(("codes", "atlas.activation-quantize"),
                                                        ("exp", "atlas.activation-quantize")),
                              _is_vector_quantizer=AtlasApplicationAdapter._is_vector_quantizer)
    with pytest.raises(NativeCompilationError, match="not a declared alternate"):
        selected_kind(adapter, source, ("codes", "exp"))
