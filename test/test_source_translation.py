"""Authored code/exponent intake is a checked byte-preserving source boundary."""
from dataclasses import replace

import pytest

from atlas_mlir_native.source_translation import (
    lower_already_quantized_contraction,
    translate_prequantized_contraction,
    translate_static_quantizer,
)
from merlin.semantic_compiler.linalg_bridge import translate_linalg_text
from merlin.semantic_compiler.model import KernelRequest

ATTRS = '''code_dtype="uint8", exponent_dtype="int32", weight_format="source_e4m3",
 dequantization="e4m3(code)*2^exponent", rhs_transposed="true", output_rescale="2^(lhs_exp+rhs_exp)",
 output_dtype="bfloat16", bias_order="after_output_cast", prov.source_node_ids=["source:0"]'''
SOURCE = '''module {
 func.func @work(%xc: tensor<2x3xi8>, %xe: tensor<i32>, %wc: tensor<4x3xi8>, %we: tensor<4x1xi32>) -> tensor<2x4xbf16> {
 %y = "source.qlinear"(%xc,%xe,%wc,%we) {ATTRS}: (tensor<2x3xi8>,tensor<i32>,tensor<4x3xi8>,tensor<4x1xi32>) -> tensor<2x4xbf16>
 func.return %y: tensor<2x4xbf16>
 }
}'''.replace('ATTRS', ATTRS)


def translate(source=SOURCE, constants=None):
    return translate_linalg_text(source, entry="work", target_identity="synthetic", lowering_policy="diagnostic",
                                 argument_constants=constants,
                                 source_operation_translators={"source.qlinear": translate_prequantized_contraction,
                                                               "source.quantize_act": translate_static_quantizer}).request


def test_exact_codes_signed_exponents_and_source_correspondence_survive_roundtrip():
    # Includes high-bit-set bytes, signed zero, and the largest finite FP8 code.
    codes = bytes([0x80, 0xfe, 0x7e, 0x38, 0x00, 0xb8] * 2)
    exponents = b"\xf9\xff\xff\xff" * 4
    source = translate(constants={2: codes, 3: exponents})
    intake = lower_already_quantized_contraction(source)
    assert (intake.m, intake.n, intake.k, intake.batch_shape) == (2, 4, 3, ())
    assert intake.operand_ids == ("arg0", "arg2") and intake.exponent_ids == ("arg1", "arg3")
    assert intake.input_ids == ("arg0", "arg1", "arg2", "arg3")
    assert bytes.fromhex(source.constants[0].data_hex) == codes
    assert bytes.fromhex(source.constants[1].data_hex) == exponents
    assert dict(intake.attrs)["prov.source_node_ids"] == '["source:0"]'
    replay = lower_already_quantized_contraction(KernelRequest.from_record(source.record()))
    assert replay.record() == intake.record()
    intake.verify_source(replay.source)
    changed = replace(source.constants[0], data_hex=(b"\x00" + codes[1:]).hex())
    with pytest.raises(ValueError, match="identity or constant bytes changed"):
        intake.verify_source(replace(source, constants=(changed, source.constants[1])))


@pytest.mark.parametrize("before,after", [
    ('code_dtype="uint8"', 'code_dtype="int8"'),
    ('exponent_dtype="int32"', 'exponent_dtype="uint32"'),
    ('rhs_transposed="true"', 'rhs_transposed="false"'),
    ('2^(lhs_exp+rhs_exp)', '2^(lhs_exp-rhs_exp)'),
    ('after_output_cast', 'before_output_cast'),
    ('output_dtype="bfloat16"', 'output_dtype="float32"'),
    ('tensor<4x1xi32>', 'tensor<4x3xi32>'),
    ('tensor<4x3xi8>', 'tensor<4x2xi8>'),
    ('tensor<2x4xbf16>', 'tensor<2x5xbf16>'),
    ('weight_format="source_e4m3"', 'weight_format="source_e4m3", recalibrate="true"'),
])
def test_changed_semantics_and_storage_types_are_refused(before, after):
    with pytest.raises(ValueError):
        translate(SOURCE.replace(before, after))


def test_unregistered_source_ops_are_not_admitted_without_provider_callback():
    with pytest.raises(ValueError):
        translate_linalg_text(SOURCE, entry="work", target_identity="synthetic", lowering_policy="diagnostic")


def test_static_quantizer_keeps_two_projections_and_negative_exponent():
    source = '''module { func.func @work(%x:tensor<2x3xbf16>) -> (tensor<2x3xi8>,tensor<i32>) {
      %c,%e = "source.quantize_act"(%x) {mode="static", reduce="tensor", rule="ref", static_exp=-7:i32,
        code_dtype="uint8", rows=32:i32, tag="activation"}: (tensor<2x3xbf16>) -> (tensor<2x3xi8>,tensor<i32>)
      func.return %c,%e: tensor<2x3xi8>,tensor<i32>
    }}'''
    result = translate(source)
    assert result.outputs == ("op0r0", "op0r1")
    assert result.nodes[1].op == "source.quantize_act.codes"
    assert result.nodes[2].op == "source.quantize_act.exponent"
    assert all(dict(node.attrs)["static_exp"] == -7 for node in result.nodes[1:])
    with pytest.raises(ValueError, match="runtime exponent production"):
        translate(source.replace('mode="static"', 'mode="dynamic"'))


def test_registered_translator_cannot_change_source_types_or_order():
    def wrong(op, ids, results):
        return (replace(translate_prequantized_contraction(op, ids, results)[0], inputs=tuple(reversed(ids))),)
    with pytest.raises(ValueError, match="changed operands"):
        translate_linalg_text(SOURCE, entry="work", target_identity="synthetic", lowering_policy="diagnostic",
                             source_operation_translators={"source.qlinear": wrong})


@pytest.mark.parametrize("op", ["qk_matmul", "pv_matmul"])
def test_attention_nt_operand_mapping_and_batch_broadcast(op):
    source = SOURCE.replace("source.qlinear", "source." + op).replace(
        "tensor<2x3xi8>", "tensor<2x1x2x3xi8>").replace(
        "tensor<4x3xi8>", "tensor<1x3x4x3xi8>").replace(
        "tensor<4x1xi32>", "tensor<1x3x4x1xi32>").replace(
        "tensor<2x4xbf16>", "tensor<2x3x2x4xbf16>")
    result = translate_linalg_text(source, entry="work", target_identity="synthetic", lowering_policy="diagnostic",
                                  source_operation_translators={"source." + op: translate_prequantized_contraction}).request
    intake = lower_already_quantized_contraction(result)
    assert (intake.m, intake.n, intake.k, intake.batch_shape) == (2, 4, 3, (2, 3))


def test_unfolded_zero_output_shift_is_preserved_and_folded_shift_refused():
    source = SOURCE.replace('weight_format="source_e4m3"', 'weight_format="source_e4m3", pop_shift=0:i32')
    assert dict(lower_already_quantized_contraction(translate(source)).attrs)["pop_shift"] == 0
    with pytest.raises(ValueError, match="unfolded zero pop_shift"):
        translate(source.replace("pop_shift=0", "pop_shift=1"))


@pytest.mark.parametrize("reduce,exp_shape", [("tensor", ""), ("head", "2x3x1x1x"),
                                               ("token", "2x3x4x1x"), ("block", "2x3x4x1x"),
                                               ("head_block", "2x3x4x1x")])
def test_dynamic_quantizer_preserves_runtime_signed_exponent_producer(reduce, exp_shape):
    source = f'''module {{ func.func @work(%x:tensor<2x3x4x8xbf16>) -> (tensor<2x3x4x8xi8>,tensor<{exp_shape}i32>) {{
      %c,%e = "source.quantize_act"(%x) {{mode="dynamic", reduce="{reduce}", rule="ref",
        code_dtype="uint8", rows=32:i32, tag="activation"}}: (tensor<2x3x4x8xbf16>) -> (tensor<2x3x4x8xi8>,tensor<{exp_shape}i32>)
      func.return %c,%e: tensor<2x3x4x8xi8>,tensor<{exp_shape}i32>
    }}}}'''
    request = translate(source)
    assert not request.constants
    for node in request.nodes[1:]:
        assert dict(node.attrs)["semantic_kind"] == "dynamic_activation_quantizer"
        assert "static_exp" not in dict(node.attrs)
        assert node.inputs == ("arg0",)
    assert request.nodes[2].type.dtype == "i32"
    with pytest.raises(ValueError, match="signed exponent shape differs"):
        translate(source.replace(f"tensor<{exp_shape}i32>", "tensor<5xi32>"))
