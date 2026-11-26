"""Independent source storage semantics and refusal boundaries.

These tests use raw literals and hand-derived expected bits; neither framework
execution nor compiler output is an oracle.
"""

import builtins
from unittest import TestCase, main, mock

import numpy as np

from atlas_mlir_native.reference_eval import EvaluationError, Tensor, evaluate, inspect_ops


def ref(name):
    return {"node_id": name, "value_id": name + ":v0"}


def node(name, target, args=(), *, op="call_function", dtype=None, shape=None,
         kwargs=None, role=None, binding=None):
    result = {"id": name + ":v0", "dtype": dtype, "shape": shape, "device": "cpu"}
    return {"id": name, "op": op, "target": target, "args": list(args),
            "kwargs": kwargs or {}, "results": [result],
            "input_kind": role, "input_target": binding}


def input_node(name, dtype, shape, *, constant=False):
    return node(name, name, op="placeholder", dtype=dtype, shape=shape,
                role="PARAMETER" if constant else "USER_INPUT", binding=name if constant else None)


def output_node(value):
    return node("out", "output", [value], op="output")


class SourceReferenceEvalTest(TestCase):
    def test_embedding_and_shape_ops_preserve_raw_bf16_payloads(self):
        weights = np.array([[0x8000, 0x7fc1], [0x3f80, 0xff80], [0, 1]], np.uint16)
        graph = [input_node("weight", "bfloat16", [3, 2], constant=True),
                 input_node("indices", "int64", [2]),
                 node("embedded", "aten.embedding.default", [ref("weight"), ref("indices")], dtype="bfloat16", shape=[2, 2]),
                 node("view", "aten.unsqueeze.default", [ref("embedded"), -1], dtype="bfloat16", shape=[2, 2, 1]),
                 node("expanded", "aten.expand.default", [ref("view"), [-1, -1, 3]], dtype="bfloat16", shape=[2, 2, 3]),
                 node("transpose", "aten.transpose.int", [ref("expanded"), 0, 1], dtype="bfloat16", shape=[2, 2, 3]),
                 output_node(ref("transpose"))]
        original_import = builtins.__import__

        def forbid_framework(name, *args, **kwargs):
            if name == "torch" or name.startswith("torch."):
                raise AssertionError("framework import attempted")
            return original_import(name, *args, **kwargs)

        with mock.patch("builtins.__import__", side_effect=forbid_framework):
            result = evaluate(graph, {"weight": weights}, {"indices": np.array([2, 0], np.int64)}, {})
        expected = np.array([[[0, 0, 0], [0x8000, 0x8000, 0x8000]],
                             [[1, 1, 1], [0x7fc1, 0x7fc1, 0x7fc1]]], np.uint16)
        np.testing.assert_array_equal(result.outputs.raw, expected)
        np.testing.assert_array_equal(weights, [[0x8000, 0x7fc1], [0x3f80, 0xff80], [0, 1]])
        self.assertFalse(result.outputs.raw.flags.writeable)
        self.assertFalse(result.ordinary_framework_kernels_called)

    def test_f32_to_bf16_rne_ties_subnormals_signed_zero(self):
        # Half-way cases with even/odd low BF16 bit, tiny subnormals, both zeros.
        raw = np.array([0x3f808000, 0x3f818000, 0xbf808000, 0xbf818000,
                        0x00008000, 0x00018000, 0x80000000, 0], np.uint32)
        graph = [input_node("x", "float32", [8]),
                 node("cast", "aten.to.dtype", [ref("x"), {"kind": "dtype", "value": "torch.bfloat16"}], dtype="bfloat16", shape=[8]),
                 node("wide", "aten.to.dtype", [ref("cast"), {"kind": "dtype", "value": "torch.float32"}], dtype="float32", shape=[8]),
                 output_node([ref("cast"), ref("wide")])]
        cast, wide = evaluate(graph, {}, {"x": raw}, {}).outputs
        expected = np.array([0x3f80, 0x3f82, 0xbf80, 0xbf82, 0, 2, 0x8000, 0], np.uint16)
        np.testing.assert_array_equal(cast.raw, expected)
        np.testing.assert_array_equal(wide.raw, expected.astype(np.uint32) << np.uint32(16))

    def test_integer_to_f32_conversion_and_metadata(self):
        graph = [input_node("x", "int64", [4]),
                 node("cast", "aten.to.dtype_layout", [ref("x")], kwargs={"dtype": {"kind": "dtype", "value": "torch.float32"}, "device": {"kind": "device", "value": "cpu"}}, dtype="float32", shape=[4]),
                 node("check", "aten._assert_tensor_metadata.default", [ref("cast")], kwargs={"dtype": {"kind": "dtype", "value": "torch.float32"}, "size": [4], "stride": [1]}),
                 output_node(ref("cast"))]
        result = evaluate(graph, {}, {"x": np.array([0, -1, 16777217, 16777219], np.int64)}, {})
        np.testing.assert_array_equal(result.outputs.raw, np.array([0, 0xbf800000, 0x4b800000, 0x4b800002], np.uint32))

    def test_audit_names_first_source_node_and_is_not_execution(self):
        graph = [input_node("x", "float32", [2, 2]),
                 node("first", "aten.matmul.default", [ref("x"), ref("x")], dtype="float32", shape=[2, 2]),
                 node("later", "aten.cos.default", [ref("x")]),
                 node("device", "atlas.qlinear.default", [ref("x")]),
                 output_node(ref("first"))]
        rules = {"npu_model_revision": "b97b2f5", "numerics": "rtl", "selected_unit": "MXU0", "k_order": "K32"}
        audit = inspect_ops(graph, rules)
        self.assertIsNone(audit["first_unsupported"])
        self.assertEqual(audit["status"], "covered")
        self.assertFalse(audit["executed"])
        self.assertTrue(audit["audit_only"])
        self.assertEqual(len(audit["unsupported"]), 0)
        with self.assertRaises(EvaluationError) as caught:
            evaluate(graph, {}, {"x": np.full((2, 2), 0x3f800000, np.uint32)}, rules)
        self.assertEqual(caught.exception.node_id, "first")
        self.assertEqual(caught.exception.completed_node_ids, ("x",))

    def test_multiple_source_results_bind_values_and_getitem_container(self):
        quant = node("quant", "atlas.quantize_act.default", [ref("x"), "static", "tensor", "ref", 0, "uint8"])
        quant["results"] = [
            {"id": "quant:v0", "dtype": "uint8", "shape": [1]},
            {"id": "quant:v1", "dtype": "int32", "shape": []}]
        graph = [input_node("x", "bfloat16", [1]), quant,
                 node("code", "<built-in function getitem>", [{"node_id": "quant", "value_id": "quant:v0"}, 0], dtype="uint8", shape=[1]),
                 node("exp", "<built-in function getitem>", [{"node_id": "quant", "value_id": "quant:v1"}, 1], dtype="int32", shape=[]),
                 output_node([ref("code"), ref("exp")])]
        with mock.patch("atlas_mlir_native.reference_eval.device.call", return_value=(
                Tensor(np.array([0x38], np.uint8), "u8"), Tensor(np.array(0, np.int32), "i32"))):
            result = evaluate(graph, {}, {"x": np.array([0x3f80], np.uint16)})
        self.assertEqual(result.outputs[0].raw.item(), 0x38)
        self.assertEqual(result.outputs[1].raw.item(), 0)
        self.assertEqual(result.values["quant:v1"].dtype, "i32")

    def test_domain_failures_name_the_node(self):
        cases = [
            (node("bad", "aten.unsqueeze.default", [ref("x"), 2], dtype="float32", shape=[1]), "axis"),
            (node("bad", "aten.to.dtype", [ref("x"), "torch.bfloat16"], dtype="bfloat16", shape=[1]), "finite"),
            (node("bad", "aten._assert_tensor_metadata.default", [ref("x")], kwargs={"dtype": "torch.int64"}), "assertion"),
            (node("bad", "aten.to.dtype_layout", [ref("x")], kwargs={"device": "cuda"}, dtype="float32", shape=[1]), "CPU"),
            (node("bad", "aten.expand.default", [ref("x"), [-1, 1]], dtype="float32", shape=[1, 1]), "dimension"),
        ]
        for operation, message in cases:
            with self.subTest(message=message), self.assertRaises(EvaluationError) as caught:
                evaluate([input_node("x", "float32", [1]), operation, output_node(ref("bad"))], {}, {"x": np.array([0x7fc00000], np.uint32)}, {})
            self.assertEqual(caught.exception.node_id, "bad")
            self.assertIn(message, str(caught.exception))

    def test_embedding_negative_indices_are_not_numpy_wraparound(self):
        graph = [input_node("w", "bfloat16", [1, 1], constant=True), input_node("i", "int64", [1]),
                 node("bad", "aten.embedding.default", [ref("w"), ref("i")], dtype="bfloat16", shape=[1, 1])]
        with self.assertRaisesRegex(EvaluationError, "embedding index"):
            evaluate(graph, {"w": np.array([[0]], np.uint16)}, {"i": np.array([-1], np.int64)}, {})

    def test_references_and_metadata_cannot_substitute_values(self):
        graph = [input_node("x", "float32", [1]),
                 node("bad", "aten.unsqueeze.default", [{"node_id": "other", "value_id": "x:v0"}, 0], dtype="float32", shape=[1, 1])]
        with self.assertRaisesRegex(EvaluationError, "ownership mismatch"):
            evaluate(graph, {}, {"x": np.array([0], np.uint32)}, {})
        graph[1]["args"][0] = ref("x")
        graph[1]["results"][0]["shape"] = [99]
        with self.assertRaisesRegex(EvaluationError, "dtype/shape"):
            evaluate(graph, {}, {"x": np.array([0], np.uint32)}, {})

    def test_cat_reshape_slice_and_container_output(self):
        graph = [input_node("x", "int32", [2, 2]),
                 node("slice", "aten.slice.Tensor", [ref("x"), 1, 0, 2, 2], dtype="int32", shape=[2, 1]),
                 node("cat", "aten.cat.default", [[ref("slice"), ref("slice")], 1], dtype="int32", shape=[2, 2]),
                 node("reshape", "aten.reshape.default", [ref("cat"), [-1]], dtype="int32", shape=[4]),
                 output_node({"data": ref("reshape"), "empty": None})]
        result = evaluate(graph, {}, {"x": np.array([[1, 2], [3, 4]], np.int32)}, {})
        np.testing.assert_array_equal(result.outputs["data"].raw, [1, 1, 3, 3])
        self.assertIsNone(result.outputs["empty"])

    def test_missing_bindings_no_output_and_storage_dtype_fail_closed(self):
        with self.assertRaisesRegex(EvaluationError, "binding"):
            evaluate([input_node("x", "float32", [1])], {}, {}, {})
        with self.assertRaisesRegex(EvaluationError, "no output"):
            evaluate([input_node("x", "float32", [1])], {}, {"x": np.array([0], np.uint32)}, {})
        with self.assertRaisesRegex(ValueError, "requires uint16"):
            Tensor(np.array([1], np.float32), "bf16")


if __name__ == "__main__":
    main()
