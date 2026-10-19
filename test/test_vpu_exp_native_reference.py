"""Selected ExpLane finite-domain port and generated panel checks."""

from __future__ import annotations

import struct
import unittest

from atlas_mlir_native.exp_lane import vexp_bf16_bits

if __package__:
    from .rtl.make_native_bf16_exp_fixture import _panel
else:
    from rtl.make_native_bf16_exp_fixture import _panel


class VpuExpNativeReferenceTest(unittest.TestCase):
    def test_selected_fixed_point_path_keeps_fractional_approximation(self) -> None:
        self.assertEqual({
            code: vexp_bf16_bits(code)
            for code in (0, 0x8000, 0x3F00, 0xBF00, 0x3F80, 0xBF80,
                         0x3FC0, 0xBFC0, 0x42A0, 0xC2A0, 0x3C3F)
        }, {
            0: 0x3F80, 0x8000: 0x3F80,
            0x3F00: 0x3FD3, 0xBF00: 0x3F1B,
            0x3F80: 0x402E, 0xBF80: 0x3EBC,
            0x3FC0: 0x408F, 0xBFC0: 0x3E64,
            0x42A0: 0x792A, 0xC2A0: 0x05C1,
            0x3C3F: 0x3F81,
        })
        # Exact real exp rounded once to BF16 gives 0x3F82 here. The
        # selected fixed-point/LUT path instead produces 0x3F81.
        self.assertNotEqual(vexp_bf16_bits(0x3C3F), 0x3F82)
        for code in (0x0001, 0x7F80, 0x7FC1, 0x42A1, 0xC2A1, 0x10000):
            with self.subTest(code=code), self.assertRaises(ValueError):
                vexp_bf16_bits(code)

    def test_panel_covers_fractional_signs_and_both_physical_halves(self) -> None:
        for phase in (0, 1):
            source, expected = _panel(phase)
            self.assertEqual((len(source), len(expected)), (2048, 2048))
            inputs = struct.unpack("<1024H", source)
            outputs = struct.unpack("<1024H", expected)
            self.assertEqual(len(set(inputs)), 296)
            self.assertEqual(outputs,
                             tuple(vexp_bf16_bits(code) for code in inputs))
            self.assertNotEqual(source[:1024], source[1024:])
            self.assertIn(0x42A0, inputs)
            self.assertIn(0xC2A0, inputs)


if __name__ == "__main__":
    unittest.main()
