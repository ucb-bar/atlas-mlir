"""Exact adapter-to-source correspondence, including refusals before mutation."""
import copy
import json
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).absolute().parents[1] / "python"))

from xdsl.dialects.builtin import (ArrayAttr, DictionaryAttr, Float32Type, IntegerType,
                                   ModuleOp, StringAttr, TensorType)
from xdsl.dialects.func import CallOp, FuncOp, ReturnOp
from xdsl.ir import Block, Region

from atlas_mlir_native.source_normalize import AtlasNormalizeError, normalize
from atlas_mlir_native.source_ops import QLinearOp, QuantizeActOp


def ids(key):
    return {"prov.source_node_ids": ArrayAttr([StringAttr("prepared:" + key)]),
            "prov.origin_node_ids": ArrayAttr([StringAttr(key)])}


def fixture():
    f32 = TensorType(Float32Type(), [2, 4])
    codes = TensorType(IntegerType(8), [2, 4])
    exponent = TensorType(IntegerType(32), [])
    weight = TensorType(IntegerType(8), [3, 4])
    result = TensorType(Float32Type(), [2, 3])
    block = Block(arg_types=[f32, weight, exponent])
    arg_attrs = [ids("input"), ids("weight") | {"prov.get_attr": StringAttr("w_codes")},
                 ids("scale") | {"prov.get_attr": StringAttr("w_exp")}]
    args = ArrayAttr([DictionaryAttr(a) for a in arg_attrs])
    quant_literals = dict(mode="static", reduce="tensor", rule="ref", static_exp=-4,
                          code_dtype="uint8", tag="layer", rows=1)
    linear_literals = dict(bias=None, out_dtype="float32", weight_format="atlas_e4m3")
    quant_schema = "atlas::quantize_act(Tensor x, str mode, str reduce, str rule, int? static_exp, str code_dtype, str tag, int rows) -> (Tensor, Tensor)"
    linear_schema = "atlas::qlinear(Tensor x_codes, Tensor x_exp, Tensor w_codes, Tensor w_exp, Tensor? bias, str out_dtype, str weight_format) -> Tensor"
    def call(symbol, operands, results, key, schema, literals):
        op = CallOp(symbol, operands, results)
        op.attributes.update(ids(key))
        op.attributes["m2m.schema"] = StringAttr(schema)
        op.attributes.update({"m2m.literal." + k: StringAttr(json.dumps(v)) for k, v in literals.items()})
        return op
    quant = call("opaque_arbitrary_name_7", [block.args[0]], [codes, exponent], "quant", quant_schema, quant_literals)
    linear = call("opaque_arbitrary_name_99", [*quant.results, *block.args[1:]], [result], "linear", linear_schema, linear_literals)
    block.add_ops([quant, linear, ReturnOp(linear.results[0])])
    function = FuncOp("main", ([f32, weight, exponent], [result]), Region(block), arg_attrs=args)
    module = ModuleOp([FuncOp.external("opaque_arbitrary_name_7", [f32], [codes, exponent]),
                       FuncOp.external("opaque_arbitrary_name_99", [codes, exponent, weight, exponent], [result]), function])
    def abi(dtype, shape, **kwargs):
        return dict(dtype=dtype, shape=shape, **kwargs)
    def operand(role, node, dtype, shape, index=0, buffer=None):
        return abi(dtype, shape, role=role, node_id=node, result_index=index, buffer=buffer)
    calls = [dict(node_id="quant", site_id="site", invocation=19, target="atlas.quantize_act.default",
                  schema=quant_schema, literals=quant_literals,
                  operands=[operand("x", "input", "float32", [2, 4])],
                  results=[abi("uint8", [2, 4], index=0), abi("int32", [], index=1)]),
             dict(node_id="linear", site_id="site", invocation=3, target="atlas.qlinear.default",
                  schema=linear_schema, literals=linear_literals,
                  operands=[operand("x_codes", "quant", "uint8", [2, 4]),
                            operand("x_exp", "quant", "int32", [], index=1),
                            operand("w_codes", "weight", "uint8", [3, 4], buffer="w_codes"),
                            operand("w_exp", "scale", "int32", [], buffer="w_exp")],
                  results=[abi("float32", [2, 3], index=0)])]
    authored = {"source.w_codes": abi("uint8", [3, 4], bytes=12, sha256="a" * 64),
                "source.w_exp": abi("int32", [], bytes=4, sha256="b" * 64)}
    manifest = dict(schema="m2m.quantization_manifest.v2", adapter_id="atlas_fp8",
                    sites=[dict(site_id="site", status="quantized", execution_route="atlas.native_fp8")],
                    atlas=dict(schema="atlas.m2m_adapter.v1", storage="uint8", calls=calls,
                               buffers={"w_codes": dict(source_name="source.w_codes", **authored["source.w_codes"]),
                                        "w_exp": dict(source_name="source.w_exp", **authored["source.w_exp"])},
                               source_tensors=authored))
    return module, manifest


class SourceNormalizeTests(unittest.TestCase):
    def test_ordered_multi_result_calls_use_exact_lineage(self):
        module, manifest = fixture()
        module.attributes["prov.quantization"] = StringAttr("external:atlas_fp8")
        manifest["atlas"]["calls"].reverse()
        self.assertIs(normalize(module, manifest), module)
        ops = list(module.walk())
        quant = next(op for op in ops if isinstance(op, QuantizeActOp))
        linear = next(op for op in ops if isinstance(op, QLinearOp))
        self.assertEqual(tuple(linear.operands[:2]), tuple(quant.results))
        self.assertFalse(any(isinstance(op, CallOp) for op in ops))
        self.assertEqual(sum(isinstance(op, FuncOp) for op in ops), 1)
        self.assertEqual(linear.attributes["output_rescale"], StringAttr("2^(lhs_exp+rhs_exp)"))
        self.assertEqual(quant.attributes["static_exp"].value.data, -4)
        self.assertNotIn("prov.atlas_call", linear.attributes)
        self.assertIn("prov.quantization_manifest_sha256", module.attributes)
        self.assertEqual(module.attributes["prov.quantization"], StringAttr("external:atlas_fp8"))
        module.verify()

    def refuse(self, change, message):
        module, manifest = fixture()
        change(module, manifest)
        before = str(module)
        uses = {value: tuple(value.uses) for op in module.walk() for value in (*op.operands, *op.results)}
        with self.assertRaisesRegex((AtlasNormalizeError, ValueError), message):
            normalize(module, manifest)
        self.assertEqual(str(module), before)
        self.assertEqual({value: tuple(value.uses) for value in uses}, uses)

    def test_empty_native_census_requires_authenticated_host_boundary(self):
        _, manifest = fixture()
        manifest["atlas"]["calls"] = []
        with self.assertRaisesRegex(AtlasNormalizeError, "authenticated saved boundary"):
            normalize(ModuleOp([]), manifest)
        manifest["atlas"]["saved_native_program"] = {"no_native_calls": True, "sha256": "a" * 64}
        module = ModuleOp([])
        normalize(module, manifest)
        self.assertIn("prov.quantization_manifest_sha256", module.attributes)

    def test_operand_swap_refused(self):
        self.refuse(lambda m, f: f["atlas"]["calls"][1]["operands"].reverse(), "shape or dtype")

    def test_tuple_result_alias_preserves_both_result_indices(self):
        module, manifest = fixture()
        quant = next(op for op in module.walk() if isinstance(op, CallOp) and len(op.results) == 2)
        quant.attributes["prov.value_aliases"] = StringAttr(json.dumps([
            dict(node_ids=["exponent_cast"], result_index=1, alias_result_index=0)]))
        manifest["atlas"]["calls"][1]["operands"][1].update(node_id="exponent_cast", result_index=0)
        normalize(module, manifest)
        quant = next(op for op in module.walk() if isinstance(op, QuantizeActOp))
        linear = next(op for op in module.walk() if isinstance(op, QLinearOp))
        self.assertIs(linear.lhs_exp, quant.exponent)

    def test_alias_to_different_tuple_result_refused(self):
        def change(module, manifest):
            quant = next(op for op in module.walk() if isinstance(op, CallOp) and len(op.results) == 2)
            quant.attributes["prov.value_aliases"] = StringAttr(json.dumps([
                dict(node_ids=["exponent_cast"], result_index=0, alias_result_index=0)]))
            manifest["atlas"]["calls"][1]["operands"][1].update(node_id="exponent_cast", result_index=0)
        self.refuse(change, "ordered operand source identity")

    def test_result_order_refused(self):
        self.refuse(lambda m, f: f["atlas"]["calls"][0]["results"].reverse(), "result index")

    def test_buffer_substitution_refused(self):
        self.refuse(lambda m, f: f["atlas"]["calls"][1]["operands"][2].update(buffer="other"), "buffer binding")

    def test_changed_buffer_bytes_refused(self):
        self.refuse(lambda m, f: f["atlas"]["buffers"]["w_codes"].update(sha256="c" * 64), "immutable source inventory")

    def test_schema_literal_change_refused(self):
        self.refuse(lambda m, f: f["atlas"]["calls"][0]["literals"].update(static_exp=-3), "literals differ")

    def test_unknown_numeric_metadata_refused(self):
        def change(module, manifest):
            op = next(op for op in module.walk() if isinstance(op, CallOp))
            op.attributes["additional_scale"] = StringAttr("2")
        self.refuse(change, "unrecognized semantic metadata")

    def test_duplicate_source_identity_refused(self):
        self.refuse(lambda m, f: f["atlas"]["calls"].append(copy.deepcopy(f["atlas"]["calls"][0])), "duplicate manifest")

    def test_missing_source_identity_refused(self):
        self.refuse(lambda m, f: f["atlas"]["calls"][0].update(node_id="unrelated"), "missing or ambiguous")

    def test_extra_unmatched_call_refused(self):
        def change(module, manifest):
            main = next(op for op in module.body.block.ops if isinstance(op, FuncOp) and not op.is_declaration)
            main.body.block.insert_op_before(CallOp("unmatched", [], []), main.body.block.last_op)
        self.refuse(change, "missing or ambiguous")

    def test_qconv2d_refusal_is_preserved(self):
        self.refuse(lambda m, f: f["atlas"]["calls"][1].update(target="atlas.qconv2d.default"), "convolution lowering is not qualified")

    def test_nonzero_pv_shift_refused(self):
        def change(module, manifest):
            record = manifest["atlas"]["calls"][1]
            record.update(target="atlas.pv_matmul.default", schema="atlas::pv_matmul(fake)", literals={"pop_shift": 1})
            for item, role in zip(record["operands"], ("p_codes", "p_exp", "vt_codes", "v_exp"), strict=True):
                item["role"] = role
            op = next(op for op in module.walk() if isinstance(op, CallOp) and len(op.operands) == 4)
            op.attributes = {key: value for key, value in op.attributes.items() if not key.startswith("m2m.literal.")}
            op.attributes["m2m.schema"] = StringAttr(record["schema"])
            op.attributes["m2m.literal.pop_shift"] = StringAttr("1")
        self.refuse(change, "pop_shift must be zero")

    def test_dynamic_token_and_head_exponents_keep_authored_shape(self):
        from atlas_mlir_native.source_translation import translate_activation_quantizer
        from xdsl.dialects.builtin import IntegerAttr, i32
        for reduction, shape in (("token", [2, 3, 1]), ("head", [2, 1, 1])):
            source = Block(arg_types=[TensorType(Float32Type(), [2, 3, 4])])
            attrs = {key: StringAttr(value) for key, value in dict(code_dtype="uint8", exponent_dtype="int32",
                weight_format="atlas_e4m3", dequantization="e4m3(code)*2^exponent", mode="dynamic",
                reduce=reduction, rule="nc", tag="attention").items()}
            attrs["rows"] = IntegerAttr(32, i32)
            quant = QuantizeActOp.build(operands=[source.args[0]], result_types=[TensorType(IntegerType(8), [2, 3, 4]),
                TensorType(IntegerType(32), shape)], attributes=attrs)
            nodes = translate_activation_quantizer(quant, ("input",), ("codes", "exponent"))
            self.assertEqual(nodes[1].type.shape, tuple(shape))

    def test_semantic_metadata_matches_existing_importer(self):
        module, manifest = fixture()
        normalize(module, manifest)
        linear = next(op for op in module.walk() if isinstance(op, QLinearOp))
        self.assertEqual({k: v.data for k, v in linear.attributes.items() if not k.startswith("prov.")},
                         dict(code_dtype="uint8", exponent_dtype="int32", weight_format="atlas_e4m3",
                              dequantization="e4m3(code)*2^exponent", rhs_transposed="true",
                              output_rescale="2^(lhs_exp+rhs_exp)", output_dtype="float32", bias_order="after_output_cast"))


if __name__ == "__main__":
    unittest.main()
