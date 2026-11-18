"""Data identity checks for independently scheduled BF16 input transfers."""

import unittest

from atlas_mlir_native.bf16_stream import loaded_bf16_pairs, trace_bf16_data
from atlas_mlir_native.binding import _check_atomic_dma_schedule

from merlin.semantic_compiler.target_binding import NativeCompilationError


def load(address, pair, scratch=0):
    return [("dma", "load", address, scratch, 2048, 0), ("wait", 0),
            ("vload", pair, scratch), ("vload", pair + 1, scratch + 256)]


def store(address, pair, scratch=1024):
    return [("vstore", pair, scratch), ("vstore", pair + 1, scratch + 256),
            ("dma", "store", address, scratch, 2048, 1), ("wait", 1)]


class BF16StreamTest(unittest.TestCase):
    def check(self, actual, expected):
        _check_atomic_dma_schedule(
            actual, expected, context="BF16 test", allow_load_reordering=True
        )

    def test_load_can_move_after_an_independent_reduction(self):
        head, tail = [("config",)], [("complete",), ("trap",)]
        reduction, addition = [("row_sum", 2, 4)], [("add", 0, 4, 6)]
        expected = head + load(4096, 0) + load(8192, 2) + reduction + addition + store(12288, 6) + tail
        actual = head + load(8192, 2) + reduction + load(4096, 0) + addition + store(12288, 6) + tail
        self.check(actual, expected)
        trace = trace_bf16_data(actual, unary_modes={"row_sum"}, binary_modes={"add"})
        self.assertEqual(loaded_bf16_pairs(trace), {8192: [2], 4096: [0]})
        changed = list(actual)
        changed[changed.index(("wait", 0))] = ("wait", 1)
        with self.assertRaises(NativeCompilationError):
            self.check(changed, expected)

    def test_same_load_multiset_cannot_overwrite_an_unread_value(self):
        first, second = load(4096, 0), load(8192, 0)
        unary, binary = [("relu", 0, 2)], [("add", 0, 2, 4)]
        expected = first + unary + second + binary + store(12288, 4)
        changed = first + second + unary + binary + store(12288, 4)
        self.check(expected, expected)
        with self.assertRaisesRegex(NativeCompilationError, "data identities"):
            self.check(changed, expected)

    def test_output_dma_must_read_the_correct_register_version(self):
        first, second = load(4096, 0), load(8192, 0)
        out_a, out_b = store(12288, 0), store(16384, 0)
        expected = first + out_a + second + out_b
        changed = first + out_a[:2] + second + out_b[:2] + out_a[2:] + out_b[2:]
        self.check(expected, expected)
        with self.assertRaisesRegex(NativeCompilationError, "data identities"):
            self.check(changed, expected)

    def test_pending_dma_cannot_reuse_a_live_scratch_range(self):
        expected = load(4096, 0) + load(8192, 2, 512) + [("add", 0, 2, 4)] + store(12288, 4)
        changed = list(expected)
        changed.insert(1, ("dma", "store", 16384, 0, 1024, 1))
        with self.assertRaises(NativeCompilationError):
            self.check(changed, expected)

    def test_pair_halves_retain_address_units(self):
        expected = load(4096, 0) + [("relu", 0, 2)] + store(12288, 2)
        for half in (0, 1):
            changed = list(expected)
            changed[2 + half] = ("vload", half, 256 * (1 - half))
            with self.subTest(half=half), self.assertRaises(NativeCompilationError):
                self.check(changed, expected)


if __name__ == "__main__":
    unittest.main()
