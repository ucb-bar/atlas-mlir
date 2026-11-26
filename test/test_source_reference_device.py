"""Raw literal device witnesses and authentication of the selected model path."""
from pathlib import Path
import builtins
import os
from unittest import TestCase, main, mock

import numpy as np
import pytest

from atlas_mlir_native.reference_eval import Tensor
from atlas_mlir_native.reference_eval import device
from atlas_mlir_native.reference_eval.evaluator import _call


def tensor(raw, dtype):
    storage = {"u8": np.uint8, "i32": np.int32, "bf16": np.uint16, "f32": np.uint32}[dtype]
    return Tensor(np.array(raw, storage), dtype)


def rules(unit="mxu0"):
    root = os.environ.get("ATLAS_NPU_MODEL_ROOT")
    if not root:
        raise AssertionError("ATLAS_NPU_MODEL_ROOT must name the pinned model; qualification never skips")
    return {"npu_model_revision": device.MODEL_REVISION, "numerics": "rtl", "selected_unit": unit,
            "k_order": "ascending-k32", "npu_model_root": root}


def qlinear_args(a, b, *, exponent_a=0, exponent_b=0, dtype="bfloat16"):
    return [tensor(a, "u8"), tensor(exponent_a, "i32"), tensor(b, "u8"),
            tensor(exponent_b, "i32"), None, dtype, "atlas_e4m3"]


class DeviceReferenceTest(TestCase):
    @classmethod
    def setUpClass(cls):
        cls.evidence = [device.qualify(rules(unit)) for unit in ("mxu0", "mxu1")]

    def test_qualifies_each_selected_unit_with_pinned_sources(self):
        for unit, record in zip(("mxu0", "mxu1"), self.evidence, strict=True):
            self.assertEqual(record["selected_unit"], unit)
            self.assertEqual(record["revision"], device.MODEL_REVISION)
            self.assertEqual(record["numerics"], "rtl")
            self.assertEqual(record["model_mismatches"], 0)
            self.assertEqual(record["pack_encodings_checked"], 196608)
            self.assertGreater(record["contraction_cells_checked"], 0)
            self.assertFalse(record["silicon_qualified"])

    def test_only_source_atlas_schemas_are_device_operations(self):
        self.assertTrue(device.handles("atlas.qlinear.default"))
        self.assertTrue(device.handles("atlas.quantize_act.default"))
        self.assertFalse(device.handles("aten.matmul.default"))
        self.assertFalse(device.handles("aten.linear.default"))
        with self.assertRaisesRegex(ValueError, "no device source adapter"):
            device.call("aten.matmul.default", (), {}, {})

    def test_selected_unit_rounding_and_scale_after_accumulation(self):
        a, b = [[0x38, 0x18, 0x18]], [[0x38, 0x18, 0x18]]
        for unit, expected in (("mxu0", 0x3f80), ("mxu1", 0x3f81)):
            result = device.call("atlas.qlinear.default", qlinear_args(a, b), {}, rules(unit))
            np.testing.assert_array_equal(result.raw, [[expected]])
            scaled = device.call("atlas.qlinear.default", qlinear_args(
                a, b, exponent_a=-1, exponent_b=2, dtype="float32"), {}, rules(unit))
            np.testing.assert_array_equal(scaled.raw, [[(expected + 0x80) << 16]])
            self.assertEqual(scaled.dtype, "f32")
            self.assertFalse(scaled.raw.flags.writeable)

    def test_k32_continuation_and_short_final_chunk(self):
        args = qlinear_args(np.full((2, 33), 0x38), np.full((3, 33), 0x38))
        for unit in ("mxu0", "mxu1"):
            result = device.call("atlas.qlinear.default", args, {}, rules(unit))
            np.testing.assert_array_equal(result.raw, np.full((2, 3), 0x4204, np.uint16))

    def test_qlinear_vector_and_empty_reduction(self):
        result = device.call("atlas.qlinear.default", qlinear_args(
            [0x38, 0x40], [[0x38, 0x38], [0x40, 0x40]]), {}, rules())
        np.testing.assert_array_equal(result.raw, [0x4040, 0x40c0])
        result = device.call("atlas.qlinear.default", qlinear_args(
            np.empty((2, 0), np.uint8), np.empty((3, 0), np.uint8)), {}, rules())
        np.testing.assert_array_equal(result.raw, np.zeros((2, 3), np.uint16))

    def test_attention_nt_batch_broadcast_and_row_scales(self):
        args = [tensor([[[0x38, 0x38], [0x40, 0x40]]], "u8"),
                tensor([[[0], [1]]], "i32"),
                tensor([[0x38, 0x38], [0x40, 0x40], [0x30, 0x30]], "u8"),
                tensor([[0], [0], [-1]], "i32")]
        for target in ("atlas.qk_matmul.default", "atlas.pv_matmul.default"):
            result = device.call(target, args, {}, rules())
            np.testing.assert_array_equal(result.raw,
                np.array([[[0x4000, 0x4080, 0x3f00], [0x4100, 0x4180, 0x4000]]], np.uint32) << 16)

    def test_qlinear_bias_uses_host_rounding_after_output_cast(self):
        for dtype, bias, expected in (("bfloat16", tensor([0x3bc0], "bf16"), 0x3f81),
                                     ("float32", tensor([0x3bc00000], "f32"), 0x3f80c000)):
            args = qlinear_args([[0x38]], [[0x38]], dtype=dtype)
            args[4] = bias
            result = device.call("atlas.qlinear.default", args, {}, rules())
            self.assertEqual(result.raw.item(), expected)

    def test_qconv2d_reads_unfolded_rows_adds_bias_then_restores_nchw(self):
        # Unfolded rows N=1, H=1, W=2 by K=33 (one K32 chunk plus a short tail), two output channels.
        args = qlinear_args([[0x38] * 33, [0x40] * 33], [[0x38] * 33, [0x30] * 33]) + [1, 1, 2]
        args[4] = tensor([0x3f00, 0xbf80], "bf16")
        # Readouts 33, 16.5 / 66, 33, then bias 0.5, -1: channel 0 is 33.5, 66.5; channel 1 is 15.5, 32.
        expected = np.array([[[[0x4206, 0x4285]], [[0x4178, 0x4200]]]], np.uint16)
        binding = {"operands": ["x", "x_exp", "w", "w_exp", "bias"], "results": ["out"],
                   "target": "atlas.qconv2d.default", "bias_order": "after_output_cast"}
        split = [{"id": "readout", "implementation": "source_already_quantized", "placement": "accelerator",
                  "inputs": ["x", "x_exp", "w", "w_exp"], "buffers": ["readout"]},
                 {"id": "bias", "implementation": "source_layout", "placement": "host",
                  "inputs": ["bias"], "buffers": ["bias-copy"]},
                 {"id": "add", "implementation": "source_float_arithmetic", "placement": "host",
                  "inputs": ["readout", "bias-copy"], "buffers": ["out"]}]
        for unit in ("mxu0", "mxu1"):
            with self.subTest(unit=unit):
                np.testing.assert_array_equal(device.call("atlas.qconv2d.default", args, {}, rules(unit)).raw, expected)
                staged = {**rules(unit), "numerical_stages": {"stages": split, "binding": binding}}
                np.testing.assert_array_equal(_call("atlas.qconv2d.default", args, {}, staged).raw, expected)
        with self.assertRaisesRegex(ValueError, "qconv2d requires unfolded"):
            device.call("atlas.qconv2d.default", args[:7] + [1, 2, 2], {}, rules())

    def test_static_quantizer_preserves_literal_pack_behavior(self):
        x = tensor([0x3f80, 0x4000, 0xbf80, 0x8000, 0x7fc1, 0x7f80, 0x0001], "bf16")
        args = [x, "static", "tensor", "ref", 0, "uint8"]
        codes, exponent = device.call("atlas.quantize_act.default", args, {}, rules())
        np.testing.assert_array_equal(codes.raw, [0x38, 0x40, 0xb8, 0, 0, 0x7e, 0])
        self.assertEqual(exponent.raw.item(), 0)
        self.assertEqual(exponent.dtype, "i32")

    def test_dynamic_quantizer_scale_rule_and_float32_rounding(self):
        x = tensor([[0x3f800001, 0x40000000], [0x43800000, 0x43f00000]], "f32")
        args = [x, "dynamic", "token", "ref", None, "uint8"]
        codes, exponent = device.call("atlas.quantize_act.default", args, {}, rules())
        np.testing.assert_array_equal(exponent.raw, [[-7], [0]])
        np.testing.assert_array_equal(codes.raw, [[0x70, 0x78], [0x78, 0x7e]])
        args[3] = "nc"
        codes, exponent = device.call("atlas.quantize_act.default", args, {}, rules())
        np.testing.assert_array_equal(exponent.raw, [[-7], [1]])
        np.testing.assert_array_equal(codes.raw[1], [0x70, 0x77])

    def test_reference_execution_does_not_import_framework(self):
        # Qualify/cache the pack path before forbidding imports during execution.
        args = [tensor([0x3f80], "bf16"), "static", "tensor", "ref", 0, "uint8"]
        device.call("atlas.quantize_act.default", args, {}, rules())
        original = builtins.__import__
        def checked(name, *args, **kwargs):
            if name == "torch" or name.startswith("torch."):
                raise AssertionError("ordinary framework import in reference arithmetic")
            return original(name, *args, **kwargs)
        with mock.patch("builtins.__import__", side_effect=checked):
            result = device.call("atlas.qlinear.default", qlinear_args([[0x38]], [[0x38]]), {}, rules())
            device.call("atlas.quantize_act.default", args, {}, rules())
        self.assertEqual(result.raw.item(), 0x3f80)

    def test_missing_changed_rules_are_not_inferred(self):
        args = qlinear_args([[0x38]], [[0x38]])
        cases = [("selected_unit", None, "selected_unit"), ("k_order", "K32", "k_order"),
                 ("numerics", "pytorch", "numerics=rtl"),
                 ("npu_model_revision", "other", "pinned npu_model")]
        for key, value, message in cases:
            selected = rules()
            selected[key] = value
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, message):
                device.call("atlas.qlinear.default", args, {}, selected)

    def test_parent_paths_fail_before_filesystem_access(self):
        with mock.patch.object(Path, "is_symlink", side_effect=AssertionError("path accessed")):
            with self.assertRaisesRegex(ValueError, "parent traversal"):
                device._safe_path("/tmp/../uninspected/child")

    def test_rejects_invalid_bias_pop_shift_and_invalid_codes(self):
        args = qlinear_args([[0x38]], [[0x38]])
        args[4] = tensor([0x3f800000], "f32")
        with self.assertRaisesRegex(ValueError, "requires bf16"):
            device.call("atlas.qlinear.default", args, {}, rules())
        with self.assertRaisesRegex(ValueError, "pop_shift"):
            device.call("atlas.pv_matmul.default", args[:4] + [1], {}, rules())
        for code in (0x01, 0x81, 0x7f, 0xff):
            with self.subTest(code=code), self.assertRaisesRegex(ValueError, "forbidden E4M3"):
                device.call("atlas.qlinear.default", qlinear_args([[code]], [[0x38]]), {}, rules())
        args = qlinear_args([[0x38, 0x38]], [[0x38, 0x38]])
        args[1] = tensor([[0, 1]], "i32")
        with self.assertRaisesRegex(ValueError, "constant over each contraction row"):
            device.call("atlas.qlinear.default", args, {}, rules())


def test_model_source_and_revision_drift_fail_after_cached_qualification(tmp_path):
    expected = tuple((name + "\n").encode() for name in device._MODEL_SOURCES)
    for name, data in zip(device._MODEL_SOURCES, expected, strict=True):
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
    with mock.patch.object(device, "_pinned_sources", return_value=expected), mock.patch.object(
            device.subprocess, "check_output", return_value=device.MODEL_REVISION + "\n"):
        device._authenticate(tmp_path)
        (tmp_path / device._MODEL_SOURCES[-1]).write_bytes(b"modified source")
        with pytest.raises(ValueError, match="model source modified"):
            device._authenticate(tmp_path)
    with mock.patch.object(device.subprocess, "check_output", return_value="changed-head\n"):
        with pytest.raises(ValueError, match="model revision differs"):
            device._authenticate(tmp_path)


if __name__ == "__main__":
    main()
