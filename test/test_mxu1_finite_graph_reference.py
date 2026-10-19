"""Check the source-attributed MXU1 finite-reset semantic primitive."""

from __future__ import annotations

import os
import random
import sys
import unittest
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MERLIN = os.environ.get("MERLIN_NATIVE_ROOT")
POLICY = "mxu1-e4m3-anchor-bf16-finite-reset-v1"


@unittest.skipUnless(MERLIN, "set MERLIN_NATIVE_ROOT for the shared semantic graph")
class MXU1FiniteGraphReferenceTest(unittest.TestCase):
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

    @staticmethod
    def request():
        from merlin.semantic_compiler.model import (
            IndexMap,
            KernelRequest,
            SemanticNode,
            TensorType,
        )

        operand = TensorType((32, 32), "fp8_e4m3", "raw-bit-pattern")
        result = TensorType((32, 32), "bf16", POLICY)
        maps = (
            IndexMap(3, ((1, 0, 0), (0, 0, 1)), (0, 0)),
            IndexMap(3, ((0, 1, 0), (0, 0, 1)), (0, 0)),
            IndexMap(3, ((1, 0, 0), (0, 1, 0)), (0, 0)),
        )
        return KernelRequest(
            (
                SemanticNode("lhs", "input", (), operand, effect="input"),
                SemanticNode("rhs", "input", (), operand, effect="input"),
                SemanticNode("result", "matmul", ("lhs", "rhs"), result,
                             index_maps=maps),
            ),
            ("result",), ("external",),
            (("lhs", "external"), ("rhs", "external")),
            "selected-atlas-mxu1-finite-reset-diagnostic",
            lowering_policy="diagnostic",
        )

    @staticmethod
    def evaluate(request, lhs: bytes, rhs: bytes):
        from atlas_mlir_native.reference import evaluate_mxu1_finite_reset
        from merlin.semantic_compiler.reference import TensorValue, evaluate_graph

        return evaluate_graph(
            request,
            {"lhs": TensorValue(request.node("lhs").type, tuple(lhs)),
             "rhs": TensorValue(request.node("rhs").type, tuple(rhs))},
            primitives={("matmul", POLICY): evaluate_mxu1_finite_reset},
        )[0].elements

    def test_all_finite_normal_codes_and_anchor_cancellation(self) -> None:
        from mxu1_anchor_source_model import mxu1_reset_lane
        from test_mxu_numerics import fp8, round_bf16

        palette = [0] + [sign | (exponent << 3) | mantissa
                         for sign in (0, 0x80) for exponent in range(1, 15)
                         for mantissa in range(8)]
        rng = random.Random(0xA71A67)
        lhs = bytearray((palette * 4) + [rng.choice(palette) for _ in range(124)])
        rhs = bytearray((palette * 4) + [rng.choice(palette) for _ in range(124)])
        rng.shuffle(lhs)
        rng.shuffle(rhs)
        lhs[:32] = bytes([0x77, 0x77] + [0x08] * 30)
        rhs[:32] = bytes([0x77, 0xF7] + [0x08] * 30)
        for values in (lhs, rhs):
            missing = set(palette) - set(values)
            for index, code in enumerate(sorted(missing), start=32):
                values[index] = code
            self.assertEqual(set(values), set(palette))

        request = self.request()
        observed = self.evaluate(request, lhs, rhs)
        expected = tuple(
            mxu1_reset_lane(
                bytes(rhs[column * 32:(column + 1) * 32]),
                bytes(lhs[row * 32:(row + 1) * 32]),
            )[0]
            for row in range(32) for column in range(32)
        )
        self.assertEqual(observed, expected)
        self.assertEqual(observed[0], 0)
        exact = round_bf16(sum((fp8(w) * fp8(a)
                                for w, a in zip(rhs[:32], lhs[:32], strict=True)), start=0))
        self.assertEqual(exact, 0x3BF0)
        self.assertNotEqual(observed[0], exact)

    def test_policy_domain_and_index_map_changes_are_rejected(self) -> None:
        request = self.request()
        zero = bytes(1024)
        for bad_code in (0x01, 0x7F, 0x80):
            with self.subTest(code=bad_code), self.assertRaisesRegex(ValueError, "outside positive-zero"):
                self.evaluate(request, bytes([bad_code]) + zero[1:], zero)
        wrong_node = replace(request.node("result"), index_maps=())
        wrong_maps = replace(request, nodes=(*request.nodes[:2], wrong_node))
        with self.assertRaisesRegex(ValueError, "unsupported semantic contract"):
            self.evaluate(wrong_maps, zero, zero)
        wrong_type = replace(request.node("result"),
                             type=replace(request.node("result").type,
                                          numerical_policy="mxu1-e4m3-anchor-bf16-bounded-v1"))
        wrong_policy = replace(request, nodes=(*request.nodes[:2], wrong_type))
        with self.assertRaisesRegex(ValueError, "no semantics"):
            self.evaluate(wrong_policy, zero, zero)


if __name__ == "__main__":
    unittest.main()
