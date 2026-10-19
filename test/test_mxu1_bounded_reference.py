"""Independent graph-level checks for the currently admitted MXU1 reset domain."""

from __future__ import annotations

import hashlib
import os
import random
import struct
import sys
import unittest
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MERLIN = os.environ.get("MERLIN_NATIVE_ROOT")


@unittest.skipUnless(MERLIN, "set MERLIN_NATIVE_ROOT for the shared semantic graph")
class MXU1BoundedReferenceTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        sys.path.insert(0, str(ROOT / "python"))
        sys.path.insert(0, str(Path(MERLIN) / "src"))
        sys.path.insert(0, str(ROOT / "test"))

    @classmethod
    def tearDownClass(cls) -> None:
        sys.path.remove(str(ROOT / "test"))
        sys.path.remove(str(Path(MERLIN) / "src"))
        sys.path.remove(str(ROOT / "python"))

    def request(self):
        from atlas_mlir_native.binding import diagnostic_mxu1_f32_profile
        from merlin.semantic_compiler.precision_transform import (
            build_diagnostic_matmul_tile,
        )

        source = (ROOT / "test/examples/captured_attention_mlp32_linalg.mlir").read_bytes()
        policy = (ROOT / "test/examples/atlas_attention_mlp_source_bound_policy.json").read_bytes()
        return build_diagnostic_matmul_tile(
            source.decode(), entry="forward", policy_document=policy,
            original_source_sha256=hashlib.sha256(source).hexdigest(),
            region_id="matmul_1", profile=diagnostic_mxu1_f32_profile(),
        ).request

    def test_graph_evaluator_matches_exact_rational_oracle_and_orientation(self) -> None:
        from atlas_mlir_native.binding import MXU1_POLICY
        from atlas_mlir_native.reference import evaluate_mxu1_bounded_reset
        from merlin.semantic_compiler.reference import TensorValue, evaluate_graph
        from test_mxu1_reference import _expected

        request = self.request()
        nodes = {node.id: node for node in request.nodes}
        rng = random.Random(0xA71A65)
        codes = (0x00, 0x38, 0x40, 0xB8, 0xC0)
        activation = bytes(rng.choice(codes) for _ in range(1024))
        weight = bytes(rng.choice(codes) for _ in range(1024))
        observed = evaluate_graph(
            request,
            {"lhs": TensorValue(nodes["lhs"].type, tuple(activation)),
             "rhs": TensorValue(nodes["rhs"].type, tuple(weight))},
            primitives={("matmul", MXU1_POLICY): evaluate_mxu1_bounded_reset},
        )[0].elements
        physical = _expected(weight, activation)
        expected = tuple(
            struct.unpack_from("<H", physical,
                               1024 * (column // 16) + 2 * (row * 16 + column % 16))[0]
            for row in range(32) for column in range(32)
        )
        self.assertEqual(observed, expected)
        self.assertGreater(len(set(observed)), 8)

        sparse_activation = bytearray(1024)
        sparse_weight = bytearray(1024)
        sparse_activation[2 * 32 + 7] = 0x40
        sparse_weight[20 * 32 + 7] = 0xB8
        sparse = evaluate_graph(
            request,
            {"lhs": TensorValue(nodes["lhs"].type, tuple(sparse_activation)),
             "rhs": TensorValue(nodes["rhs"].type, tuple(sparse_weight))},
            primitives={("matmul", MXU1_POLICY): evaluate_mxu1_bounded_reset},
        )[0].elements
        self.assertEqual(sparse[2 * 32 + 20], 0xC000)
        self.assertEqual(sum(code != 0 for code in sparse), 1)

    def test_wrong_domain_or_maps_are_rejected(self) -> None:
        from atlas_mlir_native.binding import MXU1_POLICY
        from atlas_mlir_native.reference import evaluate_mxu1_bounded_reset
        from merlin.semantic_compiler.reference import TensorValue, evaluate_graph

        request = self.request()
        nodes = {node.id: node for node in request.nodes}
        zero = TensorValue(nodes["lhs"].type, (0,) * 1024)
        bad = TensorValue(nodes["rhs"].type, (0x30,) + (0,) * 1023)
        primitives = {("matmul", MXU1_POLICY): evaluate_mxu1_bounded_reset}
        with self.assertRaisesRegex(ValueError, "five-code domain"):
            evaluate_graph(request, {"lhs": zero, "rhs": bad}, primitives=primitives)
        negative_zero = TensorValue(nodes["rhs"].type, (0x80,) + (0,) * 1023)
        with self.assertRaisesRegex(ValueError, "five-code domain"):
            evaluate_graph(request, {"lhs": zero, "rhs": negative_zero}, primitives=primitives)
        wrong_result = replace(nodes["result"], index_maps=())
        wrong = replace(request, nodes=(*request.nodes[:2], wrong_result))
        with self.assertRaisesRegex(ValueError, "unsupported semantic contract"):
            evaluate_graph(wrong, {"lhs": zero, "rhs": zero}, primitives=primitives)


if __name__ == "__main__":
    unittest.main()
