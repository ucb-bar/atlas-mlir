"""In-place source ops in the independent reference: a fresh buffer written through views, and in-place arithmetic.

torch is the oracle here only for the expected bits; the evaluator under test never imports it.
"""

from unittest import TestCase, main

import numpy as np
import torch

from atlas_mlir_native.reference_eval import EvaluationError, evaluate, inspect_ops


def _ref(node):
    return {"node_id": node, "value_id": f"{node}:v0"}


def _node(name, target, args, dtype, shape, kwargs=None):
    op = "placeholder" if target is None else "call_function"
    node = {"id": name, "op": op, "target": target or name, "args": args, "kwargs": kwargs or {},
            "results": [{"id": f"{name}:v0", "dtype": dtype, "shape": list(shape), "device": "cpu"}]}
    if target is None:
        node["input_kind"] = "USER_INPUT"
    return node


def _out(name):
    return {"id": "out", "op": "output", "target": "output", "args": [_ref(name)], "kwargs": {}, "results": []}


def _f32(t):
    return t.view(torch.int32).numpy().view(np.uint32)


def _bf16(t):
    return t.view(torch.int16).numpy().view(np.uint16)


def _rotate_half(write_second=True, read_first=False):
    """torch's rotate-half as exported: halves written into an empty_like buffer, then the buffer is read."""
    shape, half = (2, 3, 8), (2, 3, 4)
    graph = [_node("x", None, [], "float32", shape), _node("c", None, [], "float32", half),
             _node("a", "aten.slice.Tensor", [_ref("x"), 2, 0, 4], "float32", half),
             _node("b", "aten.slice.Tensor", [_ref("x"), 2, 4, 9223372036854775807], "float32", half),
             _node("buf", "aten.empty_like.default", [_ref("x")], "float32", shape, {"pin_memory": False}),
             _node("lo", "aten.mul.Tensor", [_ref("b"), _ref("c")], "float32", half),
             _node("s1", "aten.slice.Tensor", [_ref("buf"), 2, 0, 4], "float32", half),
             _node("w1", "aten.copy_.default", [_ref("s1"), _ref("lo")], "float32", half)]
    if read_first:
        graph.append(_node("early", "aten.to.dtype", [_ref("s1"), "torch.bfloat16"], "bfloat16", half))
    if write_second:
        graph += [_node("hi", "aten.mul.Tensor", [_ref("a"), _ref("c")], "float32", half),
                  _node("s2", "aten.slice.Tensor", [_ref("buf"), 2, 4, 9223372036854775807], "float32", half),
                  _node("w2", "aten.copy_.default", [_ref("s2"), _ref("hi")], "float32", half)]
    graph += [_node("y", "aten.to.dtype", [_ref("buf"), "torch.bfloat16"], "bfloat16", shape), _out("y")]
    return graph


class SourceBufferTest(TestCase):
    def setUp(self):
        g = torch.Generator().manual_seed(5)
        self.x = torch.randn(2, 3, 8, generator=g) * 3 + 0.5
        self.c = torch.randn(2, 3, 4, generator=g) + 2
        self.inputs = {"x": _f32(self.x), "c": _f32(self.c)}

    def test_halves_written_into_a_fresh_buffer_match_torch(self):
        buf = torch.empty_like(self.x)
        buf[..., :4].copy_(self.x[..., 4:] * self.c)
        buf[..., 4:].copy_(self.x[..., :4] * self.c)
        expected = _bf16(buf.to(torch.bfloat16))
        self.assertTrue(np.all(expected != 0))
        self.assertEqual(inspect_ops(_rotate_half())["status"], "covered")
        np.testing.assert_array_equal(evaluate(_rotate_half(), {}, self.inputs, {}).outputs.raw, expected)

    def test_refuses_reading_an_unwritten_element(self):
        with self.assertRaisesRegex(EvaluationError, "no copy_ has written"):
            evaluate(_rotate_half(write_second=False), {}, self.inputs, {})

    def test_refuses_a_write_after_the_buffer_was_read(self):
        with self.assertRaisesRegex(EvaluationError, "copy_ after a value has read"):
            evaluate(_rotate_half(read_first=True), {}, self.inputs, {})

    def test_refuses_copy_into_a_value_that_is_not_a_fresh_buffer(self):
        graph = [_node("x", None, [], "float32", (2, 3, 8)), _node("c", None, [], "float32", (2, 3, 4)),
                 _node("s", "aten.slice.Tensor", [_ref("x"), 2, 0, 4], "float32", (2, 3, 4)),
                 _node("w", "aten.copy_.default", [_ref("s"), _ref("c")], "float32", (2, 3, 4)), _out("x")]
        with self.assertRaisesRegex(EvaluationError, "fresh empty_like buffer"):
            evaluate(graph, {}, self.inputs, {})


class SourceInPlaceArithmeticTest(TestCase):
    def graph(self, base, extra_reader=False, dtype="float32"):
        shape = (3, 5)
        graph = [_node("x", None, [], dtype, shape), _node("y", None, [], "float32", shape)]
        if base == "computed":
            graph.append(_node("m", "aten.mul.Tensor", [_ref("x"), _ref("x")], dtype, shape))
        target = "m" if base == "computed" else "x"
        graph.append(_node("a", "aten.add_.Tensor", [_ref(target), _ref("y")], dtype, shape))
        if extra_reader:
            graph.append(_node("r", "aten.mul.Tensor", [_ref(target), _ref("a")], "float32", shape))
        return graph + [_out("r" if extra_reader else "a")]

    def test_add_on_a_value_nothing_else_reads_matches_torch(self):
        g = torch.Generator().manual_seed(7)
        x, y = torch.randn(3, 5, generator=g) + 3, torch.randn(3, 5, generator=g) - 1
        expected = _f32((x * x).add_(y))
        self.assertTrue(np.all(expected != 0))
        result = evaluate(self.graph("computed"), {}, {"x": _f32(x), "y": _f32(y)}, {})
        np.testing.assert_array_equal(result.outputs.raw, expected)

    def test_add_rounds_the_promoted_sum_once_into_a_bf16_destination(self):
        g = torch.Generator().manual_seed(8)
        x, y = (torch.randn(3, 5, generator=g) + 3).to(torch.bfloat16), torch.randn(3, 5, generator=g) / 3
        # Row 0: 3 * 3 + (2**-5 + 2**-20) is just above a BF16 tie, which rounding y to BF16 first would break to even.
        x[0], y[0] = 3.0, 2**-5 + 2**-20
        expected = _bf16((x * x).add_(y))
        self.assertTrue(np.all(expected != 0))
        self.assertFalse(np.array_equal(expected, _bf16((x * x).add_(y.to(torch.bfloat16)))))
        result = evaluate(self.graph("computed", dtype="bfloat16"), {}, {"x": _bf16(x), "y": _f32(y)}, {})
        np.testing.assert_array_equal(result.outputs.raw, expected)

    def test_refuses_in_place_on_an_input_or_a_value_read_elsewhere(self):
        inputs = {"x": np.full((3, 5), 0x3f800000, np.uint32), "y": np.full((3, 5), 0x40000000, np.uint32)}
        for graph in (self.graph("input"), self.graph("computed", extra_reader=True)):
            with self.assertRaisesRegex(EvaluationError, "no other node reads"):
                evaluate(graph, {}, inputs, {})


if __name__ == "__main__":
    main()
