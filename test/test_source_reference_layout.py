"""Source layout operations in the independent reference keep their bits wherever the plan places them.

torch is the oracle here only for the expected bits; the evaluator under test never imports it.
"""

from unittest import TestCase, main

import numpy as np
import torch

from atlas_mlir_native.reference_eval import EvaluationError, evaluate
from atlas_mlir_native.reference_eval.decisions import SelectedDecisions


def _value(node, index=0):
    return {"node_id": node, "value_id": f"{node}:v{index}"}


def _input(name, dtype, shape):
    return {"id": name, "op": "placeholder", "target": name, "input_kind": "USER_INPUT",
            "results": [{"id": f"{name}:v0", "dtype": dtype, "shape": list(shape), "device": "cpu"}]}


def _call(name, target, args, dtype, shape):
    return {"id": name, "op": "call_function", "target": target, "args": args, "kwargs": {},
            "results": [{"id": f"{name}:v0", "dtype": dtype, "shape": list(shape), "device": "cpu"}]}


def _output(name):
    return {"id": "out", "op": "output", "target": "output", "args": [_value(name)], "kwargs": {}, "results": []}


def _layout_only(placement):
    """Selected decisions whose only row for node e is a layout in a region with the given placement."""
    decisions = SelectedDecisions.__new__(SelectedDecisions)
    decisions._by_node = {"e": [{"id": "r0", "implementation": "source_layout", "placement": placement}]}
    decisions._bindings = {}
    decisions.record = {"applied": {}}
    return decisions


class SourceExpandAsTest(TestCase):
    def check(self, x, other, placement):
        dtype = {torch.bfloat16: "bfloat16", torch.float32: "float32", torch.int64: "int64"}[x.dtype]
        expected = torch.ops.aten.expand_as.default(x, other)
        graph = [_input("x", dtype, x.shape), _input("y", "float32", other.shape),
                 _call("e", "aten.expand_as.default", [_value("x"), _value("y")], dtype, expected.shape), _output("e")]
        bits = (lambda t: t.view(torch.int16).numpy().view(np.uint16)) if x.dtype == torch.bfloat16 else (
            (lambda t: t.view(torch.int32).numpy().view(np.uint32)) if x.dtype == torch.float32 else (lambda t: t.numpy()))
        self.assertTrue(np.all(bits(x) != 0))
        result = evaluate(graph, {}, {"x": bits(x), "y": other.view(torch.int32).numpy().view(np.uint32)}, {},
                          decisions=_layout_only(placement))
        self.assertEqual(result.outputs.dtype, {"bfloat16": "bf16", "float32": "f32", "int64": "i64"}[dtype])
        np.testing.assert_array_equal(result.outputs.raw, bits(expected.contiguous()))

    def test_broadcast_placed_on_device_or_host_keeps_source_bits(self):
        g = torch.Generator().manual_seed(3)
        other = torch.randn(2, 5, 3, generator=g)
        for placement in ("accelerator", "host"):
            for x in (torch.randn(1, 1, 3, generator=g).add(4).to(torch.bfloat16),
                      torch.randn(5, 1, generator=g).add(4),
                      torch.tensor([[7], [-9], [11], [13], [-2]], dtype=torch.int64)):
                with self.subTest(placement=placement, dtype=x.dtype, shape=tuple(x.shape)):
                    self.check(x, other, placement)

    def test_refuses_shape_that_does_not_broadcast(self):
        graph = [_input("x", "bfloat16", (2, 4)), _input("y", "float32", (3, 4)),
                 _call("e", "aten.expand_as.default", [_value("x"), _value("y")], "bfloat16", (3, 4)), _output("e")]
        with self.assertRaises(EvaluationError):
            evaluate(graph, {}, {"x": np.full((2, 4), 0x3f80, np.uint16), "y": np.ones((3, 4), np.uint32)}, {},
                     decisions=_layout_only("accelerator"))


class SourceSplitTest(TestCase):
    def test_chunks_placed_on_device_or_host_keep_source_bits(self):
        x = torch.randn(1, 4, 2, 10, generator=torch.Generator().manual_seed(4)).add(5)
        chunks = torch.ops.aten.split.Tensor(x, 4, -1)
        self.assertEqual([c.shape[-1] for c in chunks], [4, 4, 2])
        split = _call("e", "aten.split.Tensor", [_value("x"), 4, -1], "float32", ())
        split["results"] = [{"id": f"e:v{i}", "dtype": "float32", "shape": list(c.shape), "device": "cpu"}
                            for i, c in enumerate(chunks)]
        graph = [_input("x", "float32", x.shape), split]
        for i, c in enumerate(chunks):
            graph.append(_call(f"g{i}", "<built-in function getitem>", [{"node_id": "e"}, i], "float32", c.shape))
        graph.append({"id": "out", "op": "output", "target": "output", "kwargs": {}, "results": [],
                      "args": [[_value(f"g{i}") for i in range(len(chunks))]]})
        bits = lambda t: t.contiguous().view(torch.int32).numpy().view(np.uint32)
        for placement in ("accelerator", "host"):
            with self.subTest(placement=placement):
                result = evaluate(graph, {}, {"x": bits(x)}, {}, decisions=_layout_only(placement))
                for value, chunk in zip(result.outputs, chunks, strict=True):
                    np.testing.assert_array_equal(value.raw, bits(chunk))


if __name__ == "__main__":
    main()
