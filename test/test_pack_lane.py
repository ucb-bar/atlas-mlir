"""Directed selected-RTL VPU pack conversion cases."""

from __future__ import annotations

import unittest

from atlas_mlir_native.pack_lane import pack_bf16_lane


class PackLaneTest(unittest.TestCase):
    def test_rounding_and_scale(self) -> None:
        for bits, scale, expected in (
            (0x3f80, 127, 0x38),  # 1.0
            (0x3f80, 128, 0x30),  # 1.0 / 2
            (0x3f80, 126, 0x40),  # 1.0 * 2
            (0x3f88, 127, 0x38),  # tie to even 1.0
            (0x3f98, 127, 0x3a),  # tie to even 1.25
            (0xbf98, 127, 0xba),
            (0x3ff8, 127, 0x40),  # significand carry
            (0x3c80, 127, 0x08),  # minimum E4M3 normal
            (0x3c70, 127, 0x00),  # flush below minimum
            (0x3c78, 127, 0x08),  # carry into minimum normal
            (0x43f0, 127, 0x7e),  # would encode E4M3 NaN
            (0xc3f0, 127, 0xfe),
            (0x4400, 127, 0x7e),  # overflow clamp
        ):
            with self.subTest(bits=hex(bits), scale=scale):
                self.assertEqual(pack_bf16_lane(bits, scale), expected)

    def test_special_values_follow_selected_rtl(self) -> None:
        for bits, expected in (
            (0x0000, 0x00), (0x8000, 0x00),
            (0x0001, 0x00), (0x8001, 0x00),
            (0x7f80, 0x7e), (0xff80, 0xfe),
            (0x7fc1, 0x00), (0xffc1, 0x00),
        ):
            with self.subTest(bits=hex(bits)):
                self.assertEqual(pack_bf16_lane(bits, 127), expected)

    def test_invalid_words_and_scale_codes_fail(self) -> None:
        for bits, scale in ((-1, 127), (65536, 127), (0, 121), (0, 129),
                            (True, 127), (0, True)):
            with self.subTest(bits=bits, scale=scale), self.assertRaises(ValueError):
                pack_bf16_lane(bits, scale)

    def test_every_pack_result_is_a_legal_finite_mxu_operand(self) -> None:
        for scale in range(122, 129):
            for bits in range(65536):
                result = pack_bf16_lane(bits, scale)
                self.assertTrue(result == 0 or 1 <= (result >> 3) & 15 <= 15)
                self.assertNotEqual(result & 0x7f, 0x7f)


if __name__ == "__main__":
    unittest.main()
