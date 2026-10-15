"""Portable controls for the new tile oracle, without importing npu_model."""

import importlib.util
import pathlib
import random
import struct
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch
from fractions import Fraction

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "tools"))
from atlas_tile_arithmetic import (bf16_round, f32_to_bf16_bits, mxu0_dot,
                                   output_halves, pack_bf16_unit)


class TileArithmeticTest(unittest.TestCase):
    def test_bf16_and_fp8_ties(self):
        self.assertEqual(f32_to_bf16_bits(1 + 2**-8), 0x3F80)
        self.assertEqual(f32_to_bf16_bits(1 + 3 * 2**-8), 0x3F82)
        self.assertEqual(pack_bf16_unit(0x3F88), 0x38)  # 1.0625 -> 1
        self.assertEqual(pack_bf16_unit(0x3F98), 0x3A)  # 1.1875 -> 1.25
        self.assertEqual(pack_bf16_unit(0xBF88), 0xB8)

    def test_vpu_pack_boundaries(self):
        self.assertEqual(pack_bf16_unit(f32_to_bf16_bits(2**-7)), 0)
        self.assertEqual(pack_bf16_unit(f32_to_bf16_bits(2**-6)), 8)
        self.assertEqual(pack_bf16_unit(f32_to_bf16_bits(480)), 0x7E)
        self.assertEqual(pack_bf16_unit(0x8000), 0)
        with self.assertRaises(ValueError):
            pack_bf16_unit(0x7FC0)

    def test_order_and_continuation_discriminate_reset(self):
        a, b = [0x38, 0x18, 0x18], [0x38, 0x18, 0x18]
        self.assertEqual(mxu0_dot(a, b), 0x3F80)
        self.assertNotEqual(mxu0_dot(a, b), mxu0_dot(a[1:], b[1:]))
        self.assertEqual(bf16_round(Fraction(1) - Fraction(1, 256)), 0x3F7F)

    def test_distinct_output_halves(self):
        rows = [[row * 32 + col for col in range(32)] for row in range(32)]
        payload = output_halves(rows)
        self.assertEqual(len(payload), 2048)
        self.assertEqual(struct.unpack_from("<H", payload, 1024)[0], 16)
        self.assertEqual(struct.unpack_from("<H", payload, 32)[0], 32)

    @unittest.skipUnless(importlib.util.find_spec("numpy"), "admission checks require NumPy from model environment")
    def test_input_admission_and_raw_bf16_authentication(self):
        import numpy as np
        from run_model_tile import validate_arrays
        values = {"A": np.ones((32, 64), dtype=np.float32),
                  "B": np.ones((64, 32), dtype=np.float32),
                  "native_C": np.ones((32, 32), dtype=np.float32)}
        validate_arrays(values, "bfloat16")
        values["A"][0, 0] = np.float32(1.001)
        with self.assertRaises(ValueError):
            validate_arrays(values, "bfloat16")
        validate_arrays(values, "float32")
        values["A_bf16"] = np.zeros((32, 64), dtype=np.uint16)
        with self.assertRaises(ValueError):
            validate_arrays(values, "float32")
        values.pop("A_bf16")
        values["B"] = np.ones((63, 32), dtype=np.float32)
        with self.assertRaises(ValueError):
            validate_arrays(values, "float32")

    def test_cached_and_uncached_recurrences_agree(self):
        rng = random.Random(731)
        codes = [0] + [code for code in range(256)
                       if 0 < (code >> 3) & 15 < 15]
        for _ in range(32):
            a = [rng.choice(codes) for _ in range(64)]
            b = [rng.choice(codes) for _ in range(64)]
            self.assertEqual(mxu0_dot(a, b), mxu0_dot(a, b, cached=False))

    def test_model_binding_rejects_wrong_pin_and_dirty_source(self):
        from run_model_tile import MODEL_PIN, model_binding
        with patch("run_model_tile.subprocess.check_output", return_value="wrong\n"):
            with self.assertRaisesRegex(ValueError, "revision"):
                model_binding(pathlib.Path("/tmp"))
        with patch("run_model_tile.subprocess.check_output", return_value=MODEL_PIN), \
                patch("run_model_tile.subprocess.run", return_value=SimpleNamespace(returncode=1)):
            with self.assertRaisesRegex(ValueError, "tracked source"):
                model_binding(pathlib.Path("/tmp"))


if __name__ == "__main__":
    unittest.main()
