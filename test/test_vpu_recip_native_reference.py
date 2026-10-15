"""Finite-domain reciprocal LUT port and physical panel checks."""

from __future__ import annotations

import struct
import unittest

from atlas_mlir_native.recip_lane import vrecip_bf16_bits

if __package__:
    from .rtl.make_native_bf16_recip_fixture import _panel
else:
    from rtl.make_native_bf16_recip_fixture import _panel


class VpuRecipNativeReferenceTest(unittest.TestCase):
    def test_selected_lut_values_and_domain(self) -> None:
        self.assertEqual({
            code: vrecip_bf16_bits(code)
            for code in (0x3F80, 0x3FC0, 0x4000, 0x41A0, 0x41C0, 0x41E0, 0x4200)
        }, {
            0x3F80: 0x3F80,
            0x3FC0: 0x3F2A,
            0x4000: 0x3F00,
            0x41A0: 0x3D4C,
            0x41C0: 0x3D2A,
            0x41E0: 0x3D12,
            0x4200: 0x3D00,
        })
        for code in (0x0000, 0x3F7F, 0x4201, 0xBF80, 0x7F80, 0x10000):
            with self.subTest(code=code), self.assertRaises(ValueError):
                vrecip_bf16_bits(code)

    def test_panel_covers_every_admitted_encoding_and_both_halves(self) -> None:
        for phase in (0, 1):
            source, expected = _panel(phase)
            self.assertEqual((len(source), len(expected)), (2048, 2048))
            inputs = struct.unpack("<1024H", source)
            outputs = struct.unpack("<1024H", expected)
            self.assertEqual(set(inputs), set(range(0x3F80, 0x4201)))
            self.assertEqual(outputs,
                             tuple(vrecip_bf16_bits(code) for code in inputs))
            self.assertNotEqual(source[:1024], source[1024:])


if __name__ == "__main__":
    unittest.main()
