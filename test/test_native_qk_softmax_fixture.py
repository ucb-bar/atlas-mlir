"""Directed native Q/K softmax fixtures stay in the admitted numeric domains."""

from __future__ import annotations

import struct
import unittest
from collections import Counter

if __package__:
    from .rtl.make_native_qk_softmax_fixture import _panel
else:
    from rtl.make_native_qk_softmax_fixture import _panel


class NativeQkSoftmaxFixtureTest(unittest.TestCase):
    def test_uniform_and_patterned_scores_have_distinct_probabilities(self) -> None:
        uniform_inputs, uniform_expected = _panel("uniform")
        patterned_inputs, patterned_expected = _panel("patterned")
        self.assertEqual(tuple(map(len, uniform_inputs)), (1024, 1024, 1024))
        self.assertEqual(tuple(map(len, patterned_inputs)), (1024, 1024, 1024))
        self.assertEqual(len(uniform_expected), 2048)
        self.assertEqual(len(patterned_expected), 2048)
        self.assertEqual(set(struct.unpack("<1024H", uniform_expected)), {0x3D00})
        counts = Counter(struct.unpack("<1024H", patterned_expected))
        self.assertEqual(counts, {0x363F: 256, 0x3D80: 512, 0x2F0E: 256})
        self.assertEqual(set(patterned_inputs[0]), {0x30, 0x38})
        self.assertEqual(set(patterned_inputs[1]), {0x10})
        self.assertEqual(set(patterned_inputs[2]), {0x10, 0x18})
        with self.assertRaises(ValueError):
            _panel("identity")


if __name__ == "__main__":
    unittest.main()
