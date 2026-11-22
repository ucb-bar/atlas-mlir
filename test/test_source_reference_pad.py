"""Constant-mode aten.pad in the independent source reference, bit for bit against torch.

torch is the oracle here only for the expected bits; the evaluator under test never imports it.
"""

from unittest import TestCase, main

import numpy as np
import torch

from atlas_mlir_native.reference_eval import EvaluationError, evaluate, inspect_ops

_TORCH = {"bfloat16": (torch.bfloat16, torch.int16, np.uint16), "float32": (torch.float32, torch.int32, np.uint32),
          "int32": (torch.int32, torch.int32, np.int32), "int64": (torch.int64, torch.int64, np.int64)}


def _graph(dtype, in_shape, out_shape, args, kwargs=None):
    x = {"id": "x", "op": "placeholder", "target": "x", "input_kind": "USER_INPUT",
         "results": [{"id": "x:v0", "dtype": dtype, "shape": list(in_shape), "device": "cpu"}]}
    pad = {"id": "pad", "op": "call_function", "target": "aten.pad.default",
           "args": [{"node_id": "x", "value_id": "x:v0"}, *args], "kwargs": kwargs or {},
           "results": [{"id": "pad:v0", "dtype": dtype, "shape": list(out_shape), "device": "cpu"}]}
    out = {"id": "out", "op": "output", "target": "output", "args": [{"node_id": "pad", "value_id": "pad:v0"}],
           "kwargs": {}, "results": []}
    return [x, pad, out]


def _source(dtype, shape, seed):
    """Nonzero source values with distinct bits, including negative values."""
    kind, _, _ = _TORCH[dtype]
    values = torch.randn(*shape, generator=torch.Generator().manual_seed(seed)) * 3
    values = torch.where(values == 0, torch.ones_like(values), values)
    return values.to(kind) if kind.is_floating_point else (values * 10).to(kind) + 1


def _bits(tensor):
    kind, view, storage = _TORCH[str(tensor.dtype).removeprefix("torch.")]
    return tensor.view(view).numpy().view(storage) if kind.is_floating_point else tensor.numpy()


class SourcePadTest(TestCase):
    def check(self, dtype, shape, pad, *options, kwargs=None, seed=0):
        source = _source(dtype, shape, seed)
        expected = torch.ops.aten.pad.default(source, pad, *options, **(kwargs or {}))
        graph = _graph(dtype, shape, expected.shape, [pad, *options], kwargs)
        result = evaluate(graph, {}, {"x": _bits(source)}, {})
        self.assertTrue(np.any(_bits(source) != 0))
        self.assertEqual(result.outputs.raw.shape, tuple(expected.shape))
        np.testing.assert_array_equal(result.outputs.raw, _bits(expected))

    def test_matches_torch_bits(self):
        cases = [
            ("bfloat16", (1, 3, 8, 8), [1, 2, 1, 2], "constant", 0.0),  # Keras 'same' 5x5 stride 2, even side
            ("bfloat16", (1, 4, 8, 8), [0, 1, 0, 1], "constant", 0.0),  # Keras 'same' 3x3 stride 2, even side
            ("bfloat16", (1, 4, 7, 7), [1, 1, 1, 1]),  # default mode and value
            ("bfloat16", (2, 3, 5), [2, -1, -2, 3], "constant", -1.75),  # crop and pad on one tensor
            ("bfloat16", (2, 4), [-4, 1], "constant", 2.5),  # crop to empty, then pad
            ("bfloat16", (2, 4), [-2, -2]),  # crop to an empty axis
            ("bfloat16", (2, 3), [1, 1, 1, 1], "constant", 1 + 2**-8 + 2**-30),  # F32 then BF16 rounding
            ("bfloat16", (2, 3), [0, 1], "constant", -0.0),
            ("bfloat16", (2, 3), [1, 0], "constant", (2 - 2**-7) * 2.0**127),
            ("bfloat16", (3,), []),
            ("float32", (2, 3, 4), [1, 0, 0, 2, 1, 1], "constant", 0.1),
            ("int32", (3, 4), [2, -1], "constant", -7.0),
            ("int64", (3, 4), [1, 1], "constant", 3),
        ]
        for seed, (dtype, shape, pad, *options) in enumerate(cases):
            with self.subTest(dtype=dtype, shape=shape, pad=pad, options=options):
                self.check(dtype, shape, pad, *options, seed=seed)

    def test_keyword_options(self):
        self.check("bfloat16", (1, 2, 3, 3), [1, 0, 0, 1], kwargs={"mode": "constant", "value": 0.5})
        self.check("bfloat16", (1, 2, 3, 3), [1, 0, 0, 1], "constant", kwargs={"value": -3.0})

    def test_static_coverage(self):
        report = inspect_ops(_graph("bfloat16", (1, 3), (1, 5), [[1, 1], "constant", 0.0]))
        self.assertEqual(report["status"], "covered")

    def test_refuses_what_it_does_not_implement(self):
        cases = [
            ((2, 4), [1, 1], ("reflect",), "pad mode 'reflect' is not implemented"),
            ((2, 4), [1, 1], ("replicate",), "pad mode 'replicate' is not implemented"),
            ((2, 4), [1, 1], ("circular",), "pad mode 'circular' is not implemented"),
            ((2, 4), [1, 1, 1], (), "even-length"),
            ((2, 4), [0] * 6, (), "even-length"),
            ((2, 4), [1.0, 1], (), "even-length"),
            ((2, 4), [-3, -2], (), "remove more"),
            ((2, 4), [1, 1], ("constant", float("nan")), "not a finite bf16"),
            ((2, 4), [1, 1], ("constant", 3.4e38), "not a finite bf16"),
            ((2, 4), [1, 1], ("constant", "0"), "must be a number"),
        ]
        for shape, pad, options, reason in cases:
            with self.subTest(pad=pad, options=options):
                graph = _graph("bfloat16", shape, shape, [pad, *options])
                with self.assertRaises(EvaluationError) as raised:
                    evaluate(graph, {}, {"x": _bits(_source("bfloat16", shape, 1))}, {})
                self.assertEqual(raised.exception.node_id, "pad")
                self.assertIn(reason, raised.exception.reason)

    def test_refusals_torch_also_rejects(self):
        source = _source("bfloat16", (2, 4), 2)
        for pad, options in (([-3, -2], ()), ([1, 1, 1], ()), ([1, 1], ("constant", 3.4e38))):
            with self.subTest(pad=pad, options=options), self.assertRaises(RuntimeError):
                torch.ops.aten.pad.default(source, pad, *options)

    def test_integer_pad_value_must_be_integral(self):
        graph = _graph("int32", (2, 4), (2, 6), [[1, 1], "constant", 2.5])
        with self.assertRaises(EvaluationError) as raised:
            evaluate(graph, {}, {"x": _bits(_source("int32", (2, 4), 3))}, {})
        self.assertIn("only as an integer", raised.exception.reason)


if __name__ == "__main__":
    main()
