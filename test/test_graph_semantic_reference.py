"""Check the parsed graph against target-operation references on public panels."""

from __future__ import annotations

import json
import os
import sys
import unittest
from pathlib import Path

import numpy as np
from safetensors.numpy import load_file

ROOT = Path(__file__).resolve().parents[1]
MERLIN = os.environ.get("MERLIN_NATIVE_ROOT")


@unittest.skipUnless(MERLIN, "set MERLIN_NATIVE_ROOT for the typed graph reference")
class GraphSemanticReferenceTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        for path in (
            str(Path(MERLIN) / "src"),
            str(ROOT / "python"),
            str(ROOT / "test"),
            str(ROOT / "test/rtl"),
        ):
            sys.path.insert(0, path)

    @classmethod
    def tearDownClass(cls) -> None:
        for path in (
            str(Path(MERLIN) / "src"),
            str(ROOT / "python"),
            str(ROOT / "test"),
            str(ROOT / "test/rtl"),
        ):
            sys.path.remove(path)

    def test_residual_add_matches_source_on_exact_panels(self) -> None:
        from atlas_mlir_native.graph_precision import (
            diagnostic_graph_input_spec,
            lower_diagnostic_graph,
        )
        from graph_semantic_checker import bf16_outputs_as_f32, evaluate_selected_graph
        from make_graph_source_fixture import _source_outputs, _tokens

        source = (ROOT / "test/examples/attention_mlp_residual_add_linalg.mlir").read_text()
        policy = json.loads((ROOT / "test/examples/attention_mlp_graph_precision.json").read_text())
        graph = lower_diagnostic_graph(source, entry="forward", policy=policy)
        weights = load_file(ROOT / "test/examples/captured_attention_mlp32_weights.safetensors")
        arguments = json.loads((ROOT / "test/examples/captured_attention_mlp32_arguments.json").read_text())
        for case in ("half", "one"):
            with self.subTest(case=case):
                tokens = _tokens(case)
                source_outputs = _source_outputs(tokens, weights)
                expected = (*source_outputs, source_outputs[0] + source_outputs[1])
                observed = bf16_outputs_as_f32(evaluate_selected_graph(
                    graph, weights, arguments, tokens,
                    diagnostic_graph_input_spec(policy),
                ))
                self.assertEqual(len(observed), 3)
                for actual, reference in zip(observed, expected, strict=True):
                    np.testing.assert_array_equal(actual, reference)

    def test_selected_graph_matches_f32_source_on_exact_panels(self) -> None:
        from atlas_mlir_native.graph_precision import (
            diagnostic_graph_input_spec,
            lower_diagnostic_graph,
        )
        from graph_semantic_checker import (
            bf16_outputs_as_f32,
            evaluate_mxu0_reset,
            evaluate_selected_graph,
        )
        from make_graph_source_fixture import _source_outputs, _tokens
        from merlin.semantic_compiler.reference import TensorValue

        source = (
            ROOT / "test/examples/captured_attention_mlp32_two_outputs_linalg.mlir"
        ).read_text()
        policy = json.loads(
            (ROOT / "test/examples/attention_mlp_graph_precision.json").read_text()
        )
        graph = lower_diagnostic_graph(source, entry="forward", policy=policy)
        weights = load_file(
            ROOT / "test/examples/captured_attention_mlp32_weights.safetensors"
        )
        arguments = json.loads(
            (ROOT / "test/examples/captured_attention_mlp32_arguments.json").read_text()
        )
        spec = diagnostic_graph_input_spec(policy)

        mxu_node = next(node for node in graph.request.nodes if node.op == "matmul")
        top_finite = (0x7E,) + (0,) * 1023
        top_value = TensorValue(graph.request.nodes[0].type, top_finite)
        self.assertNotEqual(
            evaluate_mxu0_reset(mxu_node, (top_value, top_value)).elements[0], 0
        )
        with self.assertRaisesRegex(ValueError, "finite-normal"):
            evaluate_mxu0_reset(
                mxu_node,
                (
                    TensorValue(top_value.type, (0x7F,) + top_finite[1:]),
                    top_value,
                ),
            )

        for case in ("half", "one", "striped"):
            tokens = _tokens(case)
            with self.subTest(case=case):
                observed = bf16_outputs_as_f32(
                    evaluate_selected_graph(
                        graph,
                        weights,
                        arguments,
                        tokens,
                        spec,
                    )
                )
                source_outputs = _source_outputs(tokens, weights)
                for actual, expected in zip(observed, source_outputs, strict=True):
                    np.testing.assert_array_equal(actual, expected)

        scaled_policy = {**policy, "pack_scale_code": 126}
        scaled_graph = lower_diagnostic_graph(source, entry="forward", policy=scaled_policy)
        scaled_observed = bf16_outputs_as_f32(
            evaluate_selected_graph(
                scaled_graph, weights, arguments, _tokens("half"),
                diagnostic_graph_input_spec(scaled_policy),
            )
        )
        self.assertTrue(any(
            np.any(actual != expected)
            for actual, expected in zip(scaled_observed, _source_outputs(_tokens("half"), weights), strict=True)
        ))

        rng = np.random.default_rng(0xA71A5)
        varied = np.where(rng.integers(0, 2, size=(32, 32)), 1.0, 0.5).astype(
            np.float32
        )
        with self.assertRaisesRegex(ValueError, "outside") as rejected:
            evaluate_selected_graph(graph, weights, arguments, varied, spec)
        self.assertIn("scaled-pack", str(rejected.exception))

        rounded_policy = json.loads(
            (ROOT / "test/examples/attention_mlp_graph_rounded_pack.json").read_text()
        )
        rounded_graph = lower_diagnostic_graph(
            source, entry="forward", policy=rounded_policy
        )
        rounded = bf16_outputs_as_f32(
            evaluate_selected_graph(
                rounded_graph,
                weights,
                arguments,
                varied,
                diagnostic_graph_input_spec(rounded_policy),
            )
        )
        self.assertEqual(len(rounded), 2)
        for actual, source_output in zip(
            rounded, _source_outputs(varied, weights), strict=True
        ):
            self.assertTrue(np.isfinite(actual).all())
            self.assertTrue(np.any(actual != source_output))


if __name__ == "__main__":
    unittest.main()
