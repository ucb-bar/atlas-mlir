"""Source argument staging for the parsed diagnostic attention and MLP graph."""

from __future__ import annotations

import copy
import hashlib
import json
import os
import sys
import unittest
from pathlib import Path

import numpy as np
from safetensors.numpy import load_file

ROOT = Path(__file__).resolve().parents[1]
MERLIN = os.environ.get("MERLIN_NATIVE_ROOT")


@unittest.skipUnless(MERLIN, "set MERLIN_NATIVE_ROOT to test graph input staging")
class GraphAbiTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        sys.path.insert(0, str(Path(MERLIN) / "src"))
        sys.path.insert(0, str(ROOT / "python"))
        from atlas_mlir_native.graph_precision import lower_diagnostic_graph

        cls.graph = lower_diagnostic_graph(
            (ROOT / "test/examples/captured_attention_mlp32_two_outputs_linalg.mlir").read_text(),
            entry="forward",
            policy=json.loads((ROOT / "test/examples/attention_mlp_graph_precision.json").read_text()),
        )
        cls.policy = json.loads((ROOT / "test/examples/attention_mlp_graph_precision.json").read_text())
        cls.execution = {
            "schema": "atlas.native_tile_execution_plan.v8",
            "shape": [32, 32], "dtype": "bf16", "output_bytes": 2048,
            "output_addresses": [0x90010000, 0x90010800],
            "output_physical_layouts": ["bf16_pair_halves"] * 2,
            "input_addresses": [0x90000000 + 2048 * index for index in range(7)],
            "input_payload_bytes": [1024] * 6 + [2048],
        }
        cls.manifest = {
            "engine": "merlin_native",
            "request_digest": cls.graph.request.digest(),
            "target_identity": cls.graph.request.target_identity,
            "execution_plan_sha256": hashlib.sha256(
                (json.dumps(cls.execution, sort_keys=True) + "\n").encode()
            ).hexdigest(),
        }
        cls.arguments = json.loads((ROOT / "test/examples/captured_attention_mlp32_arguments.json").read_text())
        cls.weights = load_file(ROOT / "test/examples/captured_attention_mlp32_weights.safetensors")

    @classmethod
    def tearDownClass(cls) -> None:
        sys.path.remove(str(ROOT / "python"))
        sys.path.remove(str(Path(MERLIN) / "src"))

    def prepare(self, **changes):
        from atlas_mlir_native.graph_abi import prepare_graph_inputs

        options = {
            "graph": self.graph,
            "policy": self.policy,
            "precision_plan": self.graph.record(),
            "compilation_manifest": self.manifest,
            "execution_plan": self.execution,
            "argument_manifest": self.arguments,
            "weights": self.weights,
            "invocation": {"tokens": np.full((32, 32), 0.5, dtype=np.float32)},
        }
        options.update(changes)
        return prepare_graph_inputs(**options)

    def test_captured_weights_and_new_invocation_bind_to_six_fp8_slots(self) -> None:
        from atlas_mlir_native.graph_precision import diagnostic_graph_input_spec

        result = self.prepare()
        self.assertEqual([item.source_id for item in result], [f"arg{i}" for i in range(6)])
        self.assertEqual([item.address for item in result], self.execution["input_addresses"][:6])
        self.assertEqual([item.kind for item in result], ["param"] * 5 + ["input"])
        self.assertEqual(result[-1].payload, bytes([0x30]) * 1024)
        self.assertEqual(result[0].payload.count(0x30), 32)
        self.assertEqual(result[0].payload.count(0), 992)
        self.assertEqual(result[0].payload, diagnostic_graph_input_spec(self.policy).encode(
            self.weights["query.weight"]
        ).tobytes())
        self.assertEqual(len(self.graph.record()["boundary_conversions"][0]["conversion_spec"]
                             ["admitted_code_ranges"]), 3)

    def test_boundary_rejects_changed_identity_shape_domain_and_missing_arguments(self) -> None:
        wrong = copy.deepcopy(self.graph.record())
        wrong["source_sha256"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "precision plan differs"):
            self.prepare(precision_plan=wrong)
        wrong_manifest = copy.deepcopy(self.manifest)
        wrong_manifest["request_digest"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "compiler manifest"):
            self.prepare(compilation_manifest=wrong_manifest)
        wrong_policy = copy.deepcopy(self.policy)
        wrong_policy["pack_scale_code"] = 126
        with self.assertRaisesRegex(ValueError, "precision plan differs"):
            self.prepare(policy=wrong_policy)
        wrong_abi = copy.deepcopy(self.execution)
        wrong_abi["input_payload_bytes"][0] = 2048
        with self.assertRaisesRegex(ValueError, "compiler manifest"):
            self.prepare(execution_plan=wrong_abi)
        bad = dict(self.weights)
        bad["query.weight"] = np.zeros((31, 32), dtype=np.float32)
        with self.assertRaisesRegex(ValueError, "source argument"):
            self.prepare(weights=bad)
        bad = dict(self.weights)
        bad["query.weight"] = np.full((32, 32), np.nan, dtype=np.float32)
        with self.assertRaisesRegex(ValueError, "nonfinite"):
            self.prepare(weights=bad)
        with self.assertRaisesRegex(ValueError, "missing or unexpected"):
            self.prepare(invocation={"tokens": np.ones((32, 32), dtype=np.float32),
                                     "golden": np.ones((32, 32), dtype=np.float32)})

    def test_bf16_output_decode_preserves_order_and_layout(self) -> None:
        from atlas_mlir_native.graph_abi import decode_graph_outputs
        from atlas_mlir_native.layout import pack_bf16_pair_halves

        first = pack_bf16_pair_halves((0x3E00).to_bytes(2, "little") * 1024)
        second = pack_bf16_pair_halves((0x3D80).to_bytes(2, "little") * 1024)
        outputs = decode_graph_outputs(self.graph, self.execution, (first, second))
        self.assertEqual(len(outputs), 2)
        self.assertTrue(np.all(outputs[0] == np.float32(0.125)))
        self.assertTrue(np.all(outputs[1] == np.float32(0.0625)))
        with self.assertRaisesRegex(ValueError, "output count"):
            decode_graph_outputs(self.graph, self.execution, (first,))


if __name__ == "__main__":
    unittest.main()
