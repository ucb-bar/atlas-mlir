"""Source-bound logical conversion against the selected physical tile layout."""

from __future__ import annotations

import hashlib
import json
import os
import struct
import sys
import unittest
from dataclasses import replace
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
MERLIN = os.environ.get("MERLIN_NATIVE_ROOT")


@unittest.skipUnless(MERLIN, "set MERLIN_NATIVE_ROOT for the shared precision boundary")
class PrecisionTileAbiTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        sys.path.insert(0, str(ROOT))
        sys.path.insert(0, str(ROOT / "python"))
        sys.path.insert(0, str(Path(MERLIN) / "src"))

    @classmethod
    def tearDownClass(cls) -> None:
        sys.path.remove(str(Path(MERLIN) / "src"))
        sys.path.remove(str(ROOT / "python"))
        sys.path.remove(str(ROOT))

    def plan(self, region: str):
        from atlas_mlir_native.binding import (
            diagnostic_mxu0_f32_profile,
            diagnostic_mxu1_f32_profile,
        )
        from merlin.semantic_compiler.precision_transform import (
            build_diagnostic_matmul_tile,
        )

        source = (ROOT / "test/examples/captured_attention_mlp32_linalg.mlir").read_text()
        policy = (ROOT / "test/examples/atlas_attention_mlp_source_bound_policy.json").read_bytes()
        profile = diagnostic_mxu1_f32_profile() if region == "matmul_1" else diagnostic_mxu0_f32_profile()
        return build_diagnostic_matmul_tile(
            source, entry="forward", policy_document=policy,
            original_source_sha256=hashlib.sha256(source.encode()).hexdigest(),
            region_id=region, profile=profile,
        )

    def small_plan(self):
        from atlas_mlir_native.binding import diagnostic_mxu0_f32_profile
        from merlin.semantic_compiler.precision_transform import (
            build_diagnostic_matmul_tile,
        )

        source = (ROOT / "test/examples/source_bound_small_matmul.mlir").read_text()
        profile = diagnostic_mxu0_f32_profile()
        policy = {
            **dict(profile.required_policy_fields),
            "source_sha256": hashlib.sha256(source.encode()).hexdigest(),
            "matmul_regions": {"small": dict(profile.selected_decision)},
        }
        return build_diagnostic_matmul_tile(
            source, entry="forward", policy_document=json.dumps(policy, sort_keys=True).encode(),
            original_source_sha256=policy["source_sha256"], region_id="small", profile=profile,
        )

    def test_small_source_tile_packs_head_padding_and_crops_readout(self) -> None:
        from atlas_mlir_native.precision_abi import (
            decode_source_result,
            pack_source_operand,
        )

        plan = self.small_plan()
        self.assertEqual(plan.request.node("rhs").type.shape, (32, 32))
        self.assertEqual([conversion.target_offsets for conversion in plan.inputs], [(0, 30), (0, 30)])
        lhs = np.array([[1.0, 2.0]], dtype=np.float32)
        rhs = np.array([[3.0, 4.0], [5.0, 6.0]], dtype=np.float32)
        left = np.frombuffer(pack_source_operand(plan, "lhs", lhs), dtype=np.uint8).reshape(32, 32)
        right = np.frombuffer(pack_source_operand(plan, "rhs", rhs), dtype=np.uint8).reshape(32, 32)
        self.assertEqual((left[0, 30], left[0, 31], right[0, 30], right[0, 31]),
                         (0x38, 0x40, 0x44, 0x4A))
        self.assertEqual((right[1, 30], right[1, 31]), (0x48, 0x4C))
        self.assertTrue(np.all(left[:, :30] == 0))
        self.assertTrue(np.all(right[:, :30] == 0))
        self.assertTrue(np.all(right[2:] == 0))
        payload = bytearray(2048)
        struct.pack_into("<H", payload, 0, 0x4150)
        struct.pack_into("<H", payload, 2, 0x4180)
        np.testing.assert_array_equal(
            decode_source_result(plan, bytes(payload)),
            np.array([[13.0, 16.0]], dtype=np.float32),
        )

    def test_source_output_windows_decode_and_assemble_both_physical_halves(self) -> None:
        from atlas_mlir_native.binding import diagnostic_mxu0_f32_profile
        from atlas_mlir_native.precision_abi import decode_source_results
        from merlin.semantic_compiler.precision_transform import (
            build_diagnostic_matmul_tiles,
        )

        source = (ROOT / "test/examples/source_bound_mn_matmul.mlir").read_text()
        profile = diagnostic_mxu0_f32_profile()
        policy = {
            **dict(profile.required_policy_fields),
            "source_sha256": hashlib.sha256(source.encode()).hexdigest(),
            "matmul_regions": {"large": dict(profile.selected_decision)},
        }
        plans = build_diagnostic_matmul_tiles(
            source, entry="forward", policy_document=json.dumps(policy).encode(),
            original_source_sha256=policy["source_sha256"], region_id="large",
            profile=profile,
        )
        self.assertEqual(tuple(plan.tile_origin for plan in plans), ((0, 0), (0, 32)))
        payloads = []
        for code in (0x3F80, 0x4000):
            tile = np.full((32, 32), code, dtype=np.uint16)
            physical = np.concatenate((tile[:, :16].ravel(), tile[:, 16:].ravel()))
            payloads.append(physical.astype("<u2").tobytes())
        source_output = decode_source_results(plans, payloads)
        expected = np.ones((3, 35), dtype=np.float32)
        expected[:, 32:] = 2.0
        np.testing.assert_array_equal(source_output, expected)
        with self.assertRaisesRegex(ValueError, "unwritten"):
            decode_source_results(plans[:1], payloads[:1])
        with self.assertRaisesRegex(ValueError, "matching nonempty"):
            decode_source_results(plans, payloads[:1])

    def test_batched_source_tiles_bind_each_batch_to_a_distinct_native_request(self) -> None:
        from atlas_mlir_native.binding import (
            AtlasNativeBinding,
            diagnostic_mxu0_f32_profile,
        )
        from atlas_mlir_native.precision_abi import (
            decode_source_results,
            pack_source_operand,
        )
        from merlin.frontends.compile_inputs import StagedCompileInputs
        from merlin.semantic_compiler.target_binding import NativeCompilationError

        source = (ROOT / "test/examples/source_bound_batched_matmul.mlir").read_text()
        source_sha = hashlib.sha256(source.encode()).hexdigest()
        profile = diagnostic_mxu0_f32_profile()
        policy = {
            **dict(profile.required_policy_fields),
            "schema": "atlas.oot.quantized_linalg_policy.v7",
            "source_sha256": source_sha,
            "matmul_regions": {},
            "batched_matmul_regions": {
                "batch_matmul#0": dict(profile.selected_decision),
            },
        }
        policy_bytes = json.dumps(policy, sort_keys=True).encode()
        staged = StagedCompileInputs(
            source, "forward", policy, policy_bytes, source_sha, source_sha,
            hashlib.sha256(policy_bytes).hexdigest(),
        )
        with self.assertRaisesRegex(NativeCompilationError, "explicit tile index"):
            AtlasNativeBinding.prepare_precision_tile(staged, "batch_matmul#0")
        plans = tuple(
            AtlasNativeBinding.prepare_precision_tile(staged, "batch_matmul#0", tile_index=index)
            for index in range(2)
        )
        self.assertEqual([plan.batch_index for plan in plans], [(0,), (1,)])
        self.assertNotEqual(plans[0].request.digest(), plans[1].request.digest())
        with self.assertRaisesRegex(NativeCompilationError, "exceeds output grid"):
            AtlasNativeBinding.prepare_precision_tile(staged, "batch_matmul#0", tile_index=2)
        lhs = np.zeros((2, 2, 2), dtype=np.float32)
        lhs[0, :, :] = 1.0
        lhs[1, :, :] = 2.0
        self.assertNotEqual(
            pack_source_operand(plans[0], "lhs", lhs),
            pack_source_operand(plans[1], "lhs", lhs),
        )
        payloads = []
        for code in (0x3F80, 0x4000):
            tile = np.full((32, 32), code, dtype="<u2")
            payloads.append(np.concatenate((tile[:, :16].ravel(), tile[:, 16:].ravel())).tobytes())
        np.testing.assert_array_equal(
            decode_source_results(plans, payloads),
            np.array([[[1.0] * 2] * 2, [[2.0] * 2] * 2], dtype=np.float32),
        )

    def test_batched_positive_zero_policy_canonicalizes_source_negative_zero(self) -> None:
        from atlas_mlir_native.binding import (
            AtlasNativeBinding,
            diagnostic_mxu0_f32_profile,
        )
        from atlas_mlir_native.precision_abi import pack_source_operand
        from merlin.frontends.compile_inputs import StagedCompileInputs
        from merlin.semantic_compiler.linalg_bridge import LinalgBridgeError

        source = (ROOT / "test/examples/source_bound_batched_matmul.mlir").read_text()
        source_sha = hashlib.sha256(source.encode()).hexdigest()
        profile = diagnostic_mxu0_f32_profile()
        policy = {
            **dict(profile.required_policy_fields),
            "schema": "atlas.oot.quantized_linalg_policy.v8",
            "source_sha256": source_sha,
            "zero_sign_policy": "canonicalize_encoded_negative_zero",
            "matmul_regions": {},
            "batched_matmul_regions": {
                "batch_matmul#0": dict(profile.selected_decision),
            },
        }

        def staged(selected: dict) -> StagedCompileInputs:
            raw = json.dumps(selected, sort_keys=True).encode()
            return StagedCompileInputs(
                source, "forward", selected, raw, source_sha, source_sha,
                hashlib.sha256(raw).hexdigest(),
            )

        plan = AtlasNativeBinding.prepare_precision_tile(
            staged(policy), "batch_matmul#0", tile_index=0,
        )
        self.assertIn("positive-zero", plan.inputs[0].conversion_policy)
        values = np.full((2, 2, 2), -0.0, dtype=np.float32)
        self.assertEqual(pack_source_operand(plan, "lhs", values), bytes(1024))
        with self.assertRaisesRegex(LinalgBridgeError, "missing or unknown fields"):
            AtlasNativeBinding.prepare_precision_tile(
                staged({key: value for key, value in policy.items()
                        if key != "zero_sign_policy"}),
                "batch_matmul#0", tile_index=0,
            )

    def test_input_bytes_match_separate_handwritten_packer_and_transpose(self) -> None:
        from atlas_mlir_native.precision_abi import pack_source_operand

        from tools.compile_atlas_linalg_tile import pack_fp8_tile

        plan = self.plan("matmul_0")
        lhs = np.zeros((32, 32), dtype=np.float32)
        rhs = np.zeros((32, 32), dtype=np.float32)
        lhs[2, 7] = -2.0
        rhs[3, 5] = 2.0
        rhs[10, 1] = -1.0
        lhs_before, rhs_before = lhs.copy(), rhs.copy()
        left = pack_source_operand(plan, "lhs", lhs)
        right = pack_source_operand(plan, "rhs", rhs)
        self.assertEqual(left, pack_fp8_tile(lhs))
        self.assertEqual(right, pack_fp8_tile(rhs.T))
        self.assertEqual((left[2 * 32 + 7], right[5 * 32 + 3], right[1 * 32 + 10]),
                         (0xC0, 0x40, 0xB8))
        np.testing.assert_array_equal(lhs, lhs_before)
        np.testing.assert_array_equal(rhs, rhs_before)
        with self.assertRaisesRegex(ValueError, "role"):
            pack_source_operand(plan, "weight", rhs)
        with self.assertRaisesRegex(ValueError, "input domain"):
            pack_source_operand(self.plan("matmul_1"), "rhs", np.eye(32, dtype=np.float32) * 0.5)

    def test_weight_rows_are_output_columns(self) -> None:
        from atlas_mlir_native.precision_abi import pack_source_operand

        plan = self.plan("matmul_0")
        activations = np.zeros((32, 32), dtype=np.float32)
        activations[0, 2], activations[0, 3] = 2.0, -2.0
        weights = np.zeros((32, 32), dtype=np.float32)
        for output_column in range(32):
            weights[output_column, (output_column + 1) % 32] = 0.5

        # The source graph passes a transposed weight value to linalg.matmul.
        # The physical MXU tile stores W[N,K], with one output column per row.
        lhs = np.frombuffer(pack_source_operand(plan, "lhs", activations), dtype=np.uint8).reshape(32, 32)
        rhs = np.frombuffer(pack_source_operand(plan, "rhs", weights.T), dtype=np.uint8).reshape(32, 32)
        code_values = {0x00: 0.0, 0x30: 0.5, 0x40: 2.0, 0xC0: -2.0}
        decoded_lhs = np.vectorize(code_values.__getitem__)(lhs)
        decoded_rhs = np.vectorize(code_values.__getitem__)(rhs)
        expected = activations @ weights.T
        np.testing.assert_array_equal(decoded_lhs @ decoded_rhs.T, expected)
        self.assertFalse(np.array_equal(decoded_lhs @ decoded_rhs, expected))
        self.assertEqual((expected[0, 1], expected[0, 2]), (1.0, -1.0))

    def test_bf16_readback_joins_physical_halves_and_preserves_payload(self) -> None:
        from atlas_mlir_native.precision_abi import decode_source_result

        plan = self.plan("matmul_0")
        payload = bytearray(2048)
        for row in range(32):
            for col in range(32):
                code = 0x3F80 if col < 16 else 0xC000
                offset = (col // 16) * 1024 + 2 * (row * 16 + col % 16)
                struct.pack_into("<H", payload, offset, code)
        struct.pack_into("<H", payload, 0, 0x4000)
        struct.pack_into("<H", payload, 1024 + 2 * (31 * 16 + 15), 0x3F00)
        frozen = bytes(payload)
        result = decode_source_result(plan, frozen)
        expected = np.ones((32, 32), dtype=np.float32)
        expected[:, 16:] = -2.0
        expected[0, 0], expected[31, 31] = 2.0, 0.5
        np.testing.assert_array_equal(result, expected)
        self.assertEqual(bytes(payload), frozen)
        with self.assertRaisesRegex(ValueError, "physical extent"):
            decode_source_result(plan, frozen[:-2])
        with self.assertRaisesRegex(ValueError, "selected Atlas"):
            decode_source_result(replace(plan, request=replace(plan.request, target_identity="changed")), frozen)


if __name__ == "__main__":
    unittest.main()
