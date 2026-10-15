"""Directed native attention fixtures preserve MXU value orientation."""

from __future__ import annotations

import struct
import unittest
from collections import Counter

if __package__:
    from .rtl.make_native_qk_attention_fixture import _panel
else:
    from rtl.make_native_qk_attention_fixture import _panel


class NativeQkAttentionFixtureTest(unittest.TestCase):
    def test_uniform_and_value_columns_have_distinct_expected_tiles(self) -> None:
        uniform_inputs, uniform_expected = _panel("uniform")
        striped_inputs, striped_expected = _panel("value_columns")
        self.assertEqual(tuple(map(len, uniform_inputs)), (1024,) * 4)
        self.assertEqual(tuple(map(len, striped_inputs)), (1024,) * 4)
        self.assertEqual(len(uniform_expected), 2048)
        self.assertEqual(len(striped_expected), 2048)
        self.assertEqual(set(struct.unpack("<1024H", uniform_expected)), {0x3F80})
        self.assertEqual(Counter(struct.unpack("<1024H", striped_expected)),
                         {0x3F80: 512, 0x3F00: 512})
        self.assertEqual(set(striped_inputs[3]), {0x08, 0x10})
        for feature in range(32):
            self.assertEqual(set(striped_inputs[3][feature * 32:(feature + 1) * 32]),
                             {0x10 if feature % 2 == 0 else 0x08})
        with self.assertRaises(ValueError):
            _panel("unknown")


if __name__ == "__main__":
    unittest.main()
