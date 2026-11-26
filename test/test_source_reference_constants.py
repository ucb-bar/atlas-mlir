"""Compile-time constants, the bool cast, integer indexing, and BF16 conv2d in the independent source reference.

torch is the oracle here only for expected bits; the evaluator under test never imports it.
"""

from unittest import TestCase, main

import numpy as np
import torch

from atlas_mlir_native.reference_eval import EvaluationError, evaluate
from atlas_mlir_native.reference_eval.decisions import SelectedDecisions
from atlas_mlir_native.reference_eval.evaluator import _unrepeatable, _writes

_ORDER = {"default": "lexicographic-f32", "aten.conv2d.default": "lexicographic-f32",
          "aten.conv2d.padding": "lexicographic-f32"}
_HOST = {"execution": "host", "host_order": _ORDER}


def _ref(node_id):
    return {"node_id": node_id, "value_id": node_id + ":v0"}


def _node(node_id, target, args, dtype, shape, kwargs=None):
    return {"id": node_id, "op": "call_function", "target": target, "args": args, "kwargs": kwargs or {},
            "results": [{"id": node_id + ":v0", "dtype": dtype, "shape": list(shape), "device": "cpu"}]}


def _input(node_id, kind, dtype, shape):
    return {"id": node_id, "op": "placeholder", "target": node_id, "input_kind": kind,
            "results": [{"id": node_id + ":v0", "dtype": dtype, "shape": list(shape), "device": "cpu"}]}


def _output(node_id):
    return {"id": "out", "op": "output", "target": "output", "args": [_ref(node_id)], "kwargs": {}, "results": []}


def _bits(tensor):
    if tensor.dtype == torch.bfloat16:
        return tensor.view(torch.int16).numpy().view(np.uint16)
    if tensor.dtype == torch.float32:
        return tensor.view(torch.int32).numpy().view(np.uint32)
    return tensor.numpy()


def _decisions(rows=None):
    """SelectedDecisions with an already joined selection: no region computes any node unless ``rows`` names it."""
    decisions = object.__new__(SelectedDecisions)
    decisions._by_node, decisions._bindings, decisions._softmax = dict(rows or {}), {}, {}
    decisions.record = {"applied": {}}
    return decisions


class CompileTimeConstantTest(TestCase):
    def test_a_node_no_region_computes_passes_only_when_it_reads_no_input(self):
        graph = [_input("x", "USER_INPUT", "float32", (4,)),
                 _input("table", "CONSTANT_TENSOR", "int64", (4,)),
                 _node("ones", "aten.ones.default", [[4]], "float32", (4,)),
                 _node("mask", "aten.to.device", [_ref("ones"), "cpu", "torch.bool"], "bool", (4,)),
                 _node("index", "aten.add.Tensor", [_ref("table"), 1], "int64", (4,)),
                 _node("sum", "aten.add.Tensor", [_ref("x"), _ref("ones")], "float32", (4,)),
                 _output("sum")]
        decisions = _decisions()
        x = np.arange(1, 5, dtype=np.float32).view(np.uint32)
        with self.assertRaisesRegex(EvaluationError, "no authenticated arithmetic placement for source node sum"):
            evaluate(graph, {"table": np.arange(4, dtype=np.int64)}, {"x": x}, _HOST, decisions=decisions)
        self.assertEqual(sorted(decisions.record["applied"]), ["index", "mask", "ones"])
        self.assertTrue(all(row == {"compile_time_constant": "reads no user input, parameter, or buffer"}
                            for row in decisions.record["applied"].values()))
        # A region that computes the user-input node keeps it on its authenticated placement.
        placed = _decisions({"sum": [{"id": "r", "placement": "host", "implementation": "source_float_arithmetic"}]})
        result = evaluate(graph, {"table": np.arange(4, dtype=np.int64)}, {"x": x}, _HOST, decisions=placed)
        np.testing.assert_array_equal(result.outputs.raw.view(np.float32), np.arange(2, 6, dtype=np.float32))

    def test_a_parameter_or_buffer_is_not_a_compile_time_constant(self):
        for kind in ("PARAMETER", "BUFFER"):
            graph = [_input("w", kind, "float32", (2,)),
                     _node("scaled", "aten.mul.Tensor", [_ref("w"), 2.0], "float32", (2,)),
                     _output("scaled")]
            with self.subTest(kind=kind), self.assertRaisesRegex(EvaluationError, "no authenticated arithmetic"):
                evaluate(graph, {"w": np.ones(2, np.float32).view(np.uint32)}, {}, _HOST, decisions=_decisions())

    def buffer_chain(self, reader, *, source="x", destination="buf"):
        """ones, empty_like(ones), an optional slice view, copy_ of ``source`` into ``destination``, then ``reader``."""
        graph = [_input("x", "USER_INPUT", "float32", (4,)),
                 _node("ones", "aten.ones.default", [[4]], "float32", (4,)),
                 _node("buf", "aten.empty_like.default", [_ref("ones")], "float32", (4,)),
                 _node("view", "aten.slice.Tensor", [_ref("buf"), 0, 0, 4], "float32", (4,)),
                 _node("write", "aten.copy_.default", [_ref(destination), _ref(source)], "float32", (4,)),
                 *reader, _output(reader[-1]["id"])]
        x = np.arange(1, 5, dtype=np.float32).view(np.uint32)
        return evaluate(graph, {}, {"x": x}, _HOST, decisions=_decisions())

    def test_storage_written_in_place_is_never_a_compile_time_constant(self):
        mask = _node("mask", "aten.to.device", [_ref("ones"), "cpu", "torch.bool"], "bool", (4,))
        cases = {
            "direct": dict(reader=[_node("y", "aten.mul.Tensor", [_ref("buf"), 2.0], "float32", (4,))]),
            "through a view": dict(destination="view",
                                   reader=[_node("y", "aten.add.Tensor", [_ref("buf"), 1.0], "float32", (4,))]),
            "constant source": dict(source="ones",
                                    reader=[_node("y", "aten.mul.Tensor", [_ref("buf"), 2.0], "float32", (4,))]),
            "kwarg": dict(reader=[mask, _node("y", "aten.where.self", [_ref("mask"), _ref("ones")], "float32", (4,),
                                              {"other": _ref("buf")})]),
            "nested list": dict(reader=[_node("both", "aten.cat.default", [[_ref("ones"), _ref("buf")]], "float32", (8,)),
                                        _node("y", "aten.mul.Tensor", [_ref("both"), 2.0], "float32", (8,))]),
        }
        for label, case in cases.items():
            # Any write taints its storage, even one of constant bytes: the buffer path fails closed.
            with self.subTest(label), self.assertRaisesRegex(EvaluationError,
                                                             "no authenticated arithmetic placement for source node y"):
                self.buffer_chain(**case)

    def test_in_place_writes_refuse_and_detach_writes_nothing(self):
        x = np.arange(1, 5, dtype=np.float32).view(np.uint32)
        for label, other in (("user input", _ref("x")), ("constant", 1.0)):
            graph = [_input("x", "USER_INPUT", "float32", (4,)),
                     _node("ones", "aten.ones.default", [[4]], "float32", (4,)),
                     _node("y", "aten.add_.Tensor", [_ref("ones"), other], "float32", (4,)), _output("y")]
            with self.subTest(label), self.assertRaisesRegex(EvaluationError,
                                                             "no authenticated arithmetic placement for source node y"):
                evaluate(graph, {}, {"x": x}, _HOST, decisions=_decisions())
        graph = [_input("table", "CONSTANT_TENSOR", "float32", (4,)),
                 _node("fresh", "aten.lift_fresh_copy.default", [_ref("table")], "float32", (4,)),
                 _node("held", "aten.detach_.default", [_ref("fresh")], "float32", (4,)),
                 _node("y", "aten.mul.Tensor", [_ref("held"), 2.0], "float32", (4,)), _output("y")]
        decisions = _decisions()
        table = np.arange(1, 5, dtype=np.float32).view(np.uint32)
        result = evaluate(graph, {"table": table}, {}, _HOST, decisions=decisions)
        np.testing.assert_array_equal(result.outputs.raw.view(np.float32), 2 * np.arange(1, 5, dtype=np.float32))
        self.assertIn("y", decisions.record["applied"])

    def test_writes_are_named_by_what_they_change(self):
        # Every value-changing write taints its storage; alias ops that change no value are not writes.
        for target, writes in (("aten.copy_.default", True), ("aten.add_.Tensor", True), ("aten.mul_.Tensor", True),
                               ("aten.masked_fill_.Scalar", True), ("aten.detach_.default", False),
                               ("aten.add.Tensor", False), ("aten.empty_like.default", False)):
            self.assertEqual(_writes(target), writes, target)

    def test_random_and_uninitialized_sources_are_never_compile_time_constants(self):
        for target in ("aten.rand.default", "aten.randn_like.default", "aten.bernoulli.p", "aten.normal_.default",
                       "aten.uniform_.default", "aten.multinomial.default", "aten.poisson.default",
                       "aten.empty.memory_format", "aten.empty_like.default"):
            self.assertTrue(_unrepeatable(target), target)
        for target in ("aten.ones.default", "aten.layer_norm.default", "aten.arange.default"):
            self.assertFalse(_unrepeatable(target), target)

    def test_a_region_keeps_its_placement_for_an_input_free_node(self):
        graph = [_node("ones", "aten.ones.default", [[4]], "float32", (4,)),
                 _node("y", "aten.mul.Tensor", [_ref("ones"), 2.0], "float32", (4,)), _output("y")]
        decisions = _decisions({"y": [{"id": "r", "placement": "host", "implementation": "source_float_arithmetic"}]})
        evaluate(graph, {}, {}, _HOST, decisions=decisions)
        self.assertEqual(decisions.record["applied"]["y"]["placements"], ["host"])
        self.assertNotIn("compile_time_constant", decisions.record["applied"]["y"])
        self.assertEqual(decisions.record["applied"]["ones"], {"compile_time_constant": "reads no user input, parameter, or buffer"})


class BoolCastTest(TestCase):
    def test_bool_cast_is_nonzero_with_nan_true_and_both_zeros_false(self):
        f32 = torch.tensor([0.0, -0.0, 1.0, -2.5, float("nan"), float("inf"), -float("inf"), 1e-45, -1e-45])
        f32_raw = torch.tensor([0x00000000, 0x80000000, 0xffc00000, 0x7fc00001, 0x00000001, 0x80000001],
                               dtype=torch.int64).to(torch.int32).view(torch.float32)  # NaNs of both signs, subnormals
        bf16_raw = torch.tensor([0x0000, 0x8000, 0x0001, 0x8001, 0xffc0, 0x7fc0, 0x7f80],
                                dtype=torch.int32).to(torch.int16).view(torch.bfloat16)
        for source in (f32, f32_raw, f32.to(torch.bfloat16), bf16_raw, torch.tensor([0, -3, 7, 0], dtype=torch.int64),
                       torch.tensor([0, -2147483648, 1], dtype=torch.int32), torch.tensor([0, 255, 1], dtype=torch.uint8)):
            name = {torch.float32: "float32", torch.bfloat16: "bfloat16", torch.int64: "int64", torch.int32: "int32",
                    torch.uint8: "uint8"}[source.dtype]
            graph = [_input("x", "USER_INPUT", name, source.shape),
                     _node("b", "aten.to.device", [_ref("x"), "cpu", "torch.bool"], "bool", source.shape),
                     _output("b")]
            with self.subTest(dtype=name, values=source.numel()):
                result = evaluate(graph, {}, {"x": _bits(source)}, {})
                np.testing.assert_array_equal(result.outputs.raw, source.to(torch.bool).numpy())


_NAMES = {torch.float32: "float32", torch.bfloat16: "bfloat16", torch.int64: "int64", torch.int32: "int32",
          torch.uint8: "uint8", torch.bool: "bool"}


class IndexTest(TestCase):
    def graph(self, source, indices, shape):
        graph = [_input("x", "USER_INPUT", _NAMES[source.dtype], source.shape)]
        args, bindings = [], {"x": _bits(source)}
        for i, index in enumerate(indices):
            if index is None:
                args.append(None)
                continue
            graph.append(_input(f"i{i}", "USER_INPUT", _NAMES[index.dtype], index.shape))
            args.append(_ref(f"i{i}"))
            bindings[f"i{i}"] = index.numpy()
        graph += [_node("y", "aten.index.Tensor", [_ref("x"), args], _NAMES[source.dtype], shape), _output("y")]
        return graph, bindings

    def run_index(self, source, indices):
        expected = torch.ops.aten.index.Tensor(source, indices)
        graph, bindings = self.graph(source, indices, expected.shape)
        return evaluate(graph, {}, bindings, {}), expected

    def test_integer_indexing_matches_torch(self):
        source = torch.arange(2 * 3 * 4 * 5, dtype=torch.int64).reshape(2, 3, 4, 5) - 50
        cases = [[torch.tensor([1, 0, 1])],
                 [None, torch.tensor([[2], [0]]), torch.tensor([1, 3, -1])],
                 [torch.tensor([0, 1]), None, torch.tensor([3, -4])],  # separated: indexed dimension moves to front
                 [torch.zeros(1, 1, 1, 1, dtype=torch.int64), torch.arange(3).reshape(1, 1, 1, 3)],
                 [torch.tensor([1, -1], dtype=torch.int32), torch.tensor([0, 2], dtype=torch.int32)],
                 [None, torch.tensor(1), torch.tensor(-2)]]  # 0-dimensional indices
        for source_case in (source, (source.float() / 3).to(torch.bfloat16)):
            for indices in cases:
                with self.subTest(dtype=_NAMES[source_case.dtype],
                                  indices=[None if i is None else (tuple(i.shape), str(i.dtype)) for i in indices]):
                    result, expected = self.run_index(source_case, indices)
                    np.testing.assert_array_equal(result.outputs.raw, _bits(expected))

    def test_scalar_indices_keep_the_source_rank(self):
        source = torch.arange(2 * 3 * 4, dtype=torch.int64).reshape(2, 3, 4)
        for indices in ([torch.tensor(1)], [None, torch.tensor(2)], [None, None, torch.tensor(-1)],
                        [torch.tensor(1), torch.tensor(0), torch.tensor(3)]):  # the last indexes every axis: rank 0
            with self.subTest(indices=[None if i is None else int(i) for i in indices]):
                result, expected = self.run_index(source, indices)
                self.assertEqual(result.outputs.raw.shape, tuple(expected.shape))
                np.testing.assert_array_equal(result.outputs.raw, expected.numpy())

    def test_out_of_range_and_boolean_indices_refuse(self):
        source = torch.arange(6, dtype=torch.int64)
        graph = [_input("x", "USER_INPUT", "int64", (6,)), _input("i", "USER_INPUT", "int64", (1,)),
                 _node("y", "aten.index.Tensor", [_ref("x"), [_ref("i")]], "int64", (1,)), _output("y")]
        for index in (6, -7):
            with self.subTest(index=index), self.assertRaisesRegex(EvaluationError, "outside the source dimension"):
                evaluate(graph, {}, {"x": source.numpy(), "i": np.array([index], np.int64)}, {})
        matrix = torch.arange(12, dtype=torch.int64).reshape(3, 4)
        cases = {"bool": ([torch.tensor([True, False, True])], (2, 4), "integer index tensors"),
                 # numpy reads u8 as integers and torch as a mask, so u8 refuses.
                 "u8": ([torch.tensor([1, 0, 1], dtype=torch.uint8)], (2, 4), "integer index tensors"),
                 "empty list": ([], (3, 4), "one entry per leading"),
                 "all None": ([None, None], (3, 4), "at least one index tensor"),
                 "too many": ([torch.tensor([0]), torch.tensor([0]), torch.tensor([0])], (1,), "one entry per leading"),
                 "broadcast mismatch": ([torch.tensor([0, 1]), torch.tensor([0, 1, 2])], (2,), "shape mismatch|broadcast")}
        for label, (indices, shape, message) in cases.items():
            graph, bindings = self.graph(matrix, indices, shape)
            with self.subTest(label), self.assertRaisesRegex(EvaluationError, message):
                evaluate(graph, {}, bindings, {})


def _conv_loop(x, w, b, stride, padding, dilation):
    """The declared order spelled out one scalar at a time: binary32 products and sums from +0 over (c, ky, kx)."""
    n, c, h, wd = x.shape
    o, _, kh, kw = w.shape
    oh = (h + 2 * padding[0] - dilation[0] * (kh - 1) - 1) // stride[0] + 1
    ow = (wd + 2 * padding[1] - dilation[1] * (kw - 1) - 1) // stride[1] + 1
    out = np.zeros((n, o, oh, ow), np.float32)
    for b_, co, y, x_ in np.ndindex(n, o, oh, ow):
        total = np.float32(0.0)
        for ci, ky, kx in np.ndindex(c, kh, kw):
            iy, ix = y * stride[0] - padding[0] + ky * dilation[0], x_ * stride[1] - padding[1] + kx * dilation[1]
            value = x[b_, ci, iy, ix] if 0 <= iy < h and 0 <= ix < wd else np.float32(0.0)
            total = np.float32(total + np.float32(value * w[co, ci, ky, kx]))
        out[b_, co, y, x_] = np.float32(total + b[co]) if b is not None else total
    return out


def _bf16(values, shape):
    return torch.tensor(values, dtype=torch.float64).reshape(shape).to(torch.bfloat16)


class Conv2dTest(TestCase):
    def graph(self, x, w, b, target, options, shape):
        graph = [_input("x", "USER_INPUT", _NAMES[x.dtype], x.shape), _input("w", "PARAMETER", _NAMES[w.dtype], w.shape)]
        bindings = {"w": _bits(w)}
        if b is not None:
            graph.append(_input("b", "PARAMETER", _NAMES[b.dtype], b.shape))
            bindings["b"] = _bits(b)
        graph += [_node("y", target, [_ref("x"), _ref("w"), _ref("b") if b is not None else None, *options],
                        "bfloat16", shape), _output("y")]
        return graph, bindings

    def evaluate(self, x, w, b, target, options, rules=_HOST):
        expected = (torch.ops.aten.conv2d.padding if target == "aten.conv2d.padding" else torch.ops.aten.conv2d.default)(
            x.float(), w.float(), None if b is None else b.float(), *options)
        graph, bindings = self.graph(x, w, b, target, options, expected.shape)
        return evaluate(graph, bindings, {"x": _bits(x)}, rules).outputs.raw, expected

    def refuse(self, message, x, w, b, target, options, shape, rules=_HOST):
        graph, bindings = self.graph(x, w, b, target, options, shape)
        with self.assertRaisesRegex(EvaluationError, message):
            evaluate(graph, bindings, {"x": _bits(x)}, rules)

    def test_declared_order_equals_the_scalar_loop_and_torch_on_exact_sums(self):
        generator = torch.Generator().manual_seed(7)
        for target, options, stride, padding, dilation in (
                ("aten.conv2d.padding", ([2, 2], "valid"), (2, 2), (0, 0), (1, 1)),
                ("aten.conv2d.padding", ([16, 16],), (16, 16), (0, 0), (1, 1)),
                ("aten.conv2d.default", ([1, 2], [1, 0], [2, 1]), (1, 2), (1, 0), (2, 1))):
            for biased in (True, False):
                with self.subTest(target=target, options=options, biased=biased):
                    shape = (1, 3, 32, 32) if stride == (16, 16) else (2, 3, 9, 8)
                    kernel = (4, 3, 16, 16) if stride == (16, 16) else (4, 3, 3, 2)
                    # Rounded inputs: the scalar loop checks the reference's order bit for bit.
                    x = torch.randn(shape, generator=generator).to(torch.bfloat16)
                    w = (torch.randn(kernel, generator=generator) / 4).to(torch.bfloat16)
                    b = torch.randn(kernel[0], generator=generator).to(torch.bfloat16) if biased else None
                    raw, _ = self.evaluate(x, w, b, target, options)
                    loop = _conv_loop(x.float().numpy(), w.float().numpy(), None if b is None else b.float().numpy(),
                                      stride, padding, dilation)
                    np.testing.assert_array_equal(raw, _bits(torch.from_numpy(loop).to(torch.bfloat16)))
                    self.assertTrue(np.any(raw != 0))
                    # Small integers sum exactly in any order, so torch checks the indexing, stride, padding, and bias.
                    xi = torch.randint(-3, 4, shape, generator=generator).to(torch.bfloat16)
                    wi = torch.randint(-2, 3, kernel, generator=generator).to(torch.bfloat16)
                    bi = torch.randint(-5, 6, (kernel[0],), generator=generator).to(torch.bfloat16) if biased else None
                    raw, expected = self.evaluate(xi, wi, bi, target, options)
                    np.testing.assert_array_equal(raw, _bits(expected.to(torch.bfloat16)))
                    self.assertTrue(np.any(raw != 0))

    def test_the_order_rounding_and_subnormal_products_are_pinned(self):
        # Products 2^100, -2^100, 1, 0 at (channel, column) (0,0), (0,1), (1,0), (1,1): channel-major order gives 1,
        # column-major order gives 0.
        x = _bf16([1.0] * 4, (1, 2, 1, 2))
        w = _bf16([2.0 ** 100, -(2.0 ** 100), 1.0, 0.0], (1, 2, 1, 2))
        raw, _ = self.evaluate(x, w, None, "aten.conv2d.default", ())
        self.assertEqual(raw.reshape(-1).tolist(), [0x3f80])
        products = np.array([[2.0 ** 100, -(2.0 ** 100)], [1.0, 0.0]], np.float32)
        column_major = np.float32(0)
        for column, channel in np.ndindex(2, 2):
            column_major = np.float32(column_major + products[channel, column])
        self.assertEqual(column_major, 0)
        # One rounding after the bias: 1 + 2^-8 + 2^-10 - 2^-9 rounds to 1, while rounding to BF16 before the bias
        # gives 1 + 2^-7.
        raw, _ = self.evaluate(_bf16([1.0, 1.0], (1, 1, 1, 2)), _bf16([1.0, 2.0 ** -8 + 2.0 ** -10], (1, 1, 1, 2)),
                               _bf16([-(2.0 ** -9)], (1,)), "aten.conv2d.default", ())
        self.assertEqual(raw.reshape(-1).tolist(), [0x3f80])
        # Products round in binary32 even below the normal range: -2^-151 rounds to -0, so the sum is the tie
        # 1.5 * 2^-133, which BF16 RNE rounds to 2^-132; exact products would round to 2^-133.
        raw, _ = self.evaluate(_bf16([2.0 ** -67, 2.0 ** -75], (1, 1, 1, 2)), _bf16([2.0 ** -67, -(2.0 ** -76)], (1, 1, 1, 2)),
                               _bf16([2.0 ** -133], (1,)), "aten.conv2d.default", ())
        self.assertEqual(raw.reshape(-1).tolist(), [0x0002])

    def test_refusals(self):
        x, w = torch.ones(1, 2, 4, 4, dtype=torch.bfloat16), torch.ones(2, 2, 2, 2, dtype=torch.bfloat16)
        b = torch.ones(2, dtype=torch.bfloat16)
        shape = (1, 2, 3, 3)
        self.refuse("declare 'lexicographic-f32' for aten.conv2d.padding", x, w, b, "aten.conv2d.padding", ([1, 1],),
                    shape, rules={"execution": "host", "host_order": {"default": "lexicographic-f32"}})
        self.refuse("explicitly declared host execution", x, w, b, "aten.conv2d.padding", ([1, 1],), shape, rules={})
        self.refuse("no placed-device source adapter", x, w, b, "aten.conv2d.padding", ([1, 1],), shape,
                    rules={**_HOST, "execution": "device"})
        self.refuse("only for padding 'valid'", x, w, b, "aten.conv2d.padding", ([1, 1], "same"), shape)
        self.refuse("only for padding 'valid'", x, w, b, "aten.conv2d.padding", ([1, 1], [0, 0]), shape)
        self.refuse("integer padding", x, w, b, "aten.conv2d.default", ([1, 1], "valid"), shape)
        self.refuse("grouped", x, torch.ones(2, 1, 2, 2, dtype=torch.bfloat16), b, "aten.conv2d.default",
                    ([1, 1], [0, 0], [1, 1], 2), shape)
        self.refuse("grouped", x, w, b, "aten.conv2d.default", ([1, 1], [0, 0], [1, 1], 1.0), shape)
        self.refuse("only for BF16", x.float(), w.float(), b.float(), "aten.conv2d.default", (), shape)
        self.refuse("one value per output channel", x, w, torch.ones(3, dtype=torch.bfloat16), "aten.conv2d.default",
                    (), shape)
        self.refuse("matching channels", x, torch.ones(2, 3, 2, 2, dtype=torch.bfloat16), b, "aten.conv2d.default",
                    (), shape)


if __name__ == "__main__":
    main()
