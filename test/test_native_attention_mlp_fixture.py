"""Directed two-output native attention/MLP fixtures retain feature order."""

from __future__ import annotations

import struct
import unittest
from collections import Counter

if __package__:
    from .rtl.make_native_attention_mlp_fixture import _panel
else:
    from rtl.make_native_attention_mlp_fixture import _panel


class NativeAttentionMlpFixtureTest(unittest.TestCase):
    def test_uniform_and_value_columns_preserve_both_outputs(self) -> None:
        uniform_inputs, uniform_expected = _panel("uniform")
        striped_inputs, striped_expected = _panel("value_columns")
        self.assertEqual(tuple(map(len, uniform_inputs)), (1024,) * 6)
        self.assertEqual(tuple(map(len, striped_inputs)), (1024,) * 6)
        self.assertEqual(len(uniform_expected), 4096)
        self.assertEqual(len(striped_expected), 4096)
        self.assertEqual(uniform_expected[:2048], uniform_expected[2048:])
        self.assertEqual(striped_expected[:2048], striped_expected[2048:])
        self.assertEqual(set(struct.unpack("<2048H", uniform_expected)), {0x3F80})
        self.assertEqual(Counter(struct.unpack("<2048H", striped_expected)),
                         {0x3F80: 1024, 0x3F00: 1024})
        self.assertEqual(set(striped_inputs[3]), {0x08, 0x10})
        for weight in striped_inputs[4:]:
            self.assertEqual(weight.count(0x38), 32)
            self.assertEqual(weight.count(0), 992)
        with self.assertRaises(ValueError):
            _panel("unknown")


if __name__ == "__main__":
    unittest.main()
