"""Visible-result comparison, independent of the compiler and selected core."""

from __future__ import annotations

from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
from atlas_virtual_evaluator import EvaluationResult, MemoryRegion, RuntimeInputs, Tile, compare_results, evaluate  # noqa: E402
from test_virtual_evaluator_dma import BASE, Stream, repeated  # noqa: E402
from test_virtual_evaluator_mxu import Stream as MxuStream, inputs as mxu_inputs, sparse  # noqa: E402


class VirtualEvaluatorComparisonTest(unittest.TestCase):
    def diagnostic(self, expected, actual, *fragments):
        with self.assertRaises(AssertionError) as failure:
            compare_results(expected, actual)
        for fragment in fragments:
            self.assertIn(fragment, str(failure.exception))

    def test_equal_results_ignore_output_and_region_order_and_adjacent_partitions(self):
        tile = repeated((0x0000, 0x8000, 0x7FC1, 0xFFC1))
        gap_after = MemoryRegion(BASE + 32, b"gap-after")
        whole = MemoryRegion(BASE, b"abcdefgh")
        first, second = MemoryRegion(BASE, b"abc"), MemoryRegion(BASE + 3, b"defgh")
        expected = EvaluationResult({7: tile, 2: repeated((0x3F80,))}, (gap_after, whole))
        actual = EvaluationResult({2: repeated((0x3F80,)), 7: tile}, (second, gap_after, first))
        compare_results(expected, actual)
        compare_results(actual, expected)

    def test_independent_evaluator_events_can_be_issued_and_published_in_different_orders(self):
        tiles = {0: repeated((0x3F80, 0xBF80)), 1: repeated((0x7FC1, 0x8000))}
        regions = (MemoryRegion(BASE, b"?" * 2048), MemoryRegion(BASE + 0x2000, b"!" * 2048))

        def execute(reverse):
            stream = Stream()
            values = (stream.input(index=0), stream.input(index=1))
            order = (1, 0) if reverse else (0, 1)
            pending = {index: stream.store(values[index], BASE + index * 0x2000) for index in order}
            for index in reversed(order):
                stream.wait(pending[index])
            for index in order:
                stream.output(values[index], index)
            return evaluate(stream.program(), RuntimeInputs(tiles, memory=regions[::-1] if reverse else regions))

        compare_results(execute(False), execute(True))

    def test_output_indices_must_match_even_when_tiles_are_equal(self):
        tile = repeated((0,))
        expected = EvaluationResult({2: tile, 7: tile})
        for actual in (EvaluationResult({2: tile}), EvaluationResult({2: tile, 7: tile, 9: tile})):
            with self.subTest(indices=tuple(actual.outputs)):
                self.diagnostic(expected, actual, "output indices", "expected [2, 7]")

    def test_raw_signed_zero_and_nan_payload_differences_report_logical_coordinates(self):
        for row, col, wanted, observed in ((2, 19, 0x8000, 0x0000), (9, 5, 0x7FC1, 0x7FC2)):
            with self.subTest(row=row, col=col):
                original = [0] * 1024
                original[row * 32 + col] = wanted
                changed = list(original)
                changed[row * 32 + col] = observed
                expected = EvaluationResult({7: Tile("bf16", tuple(original))})
                actual = EvaluationResult({7: Tile("bf16", tuple(changed))})
                self.diagnostic(expected, actual, f"output 7 tile[{row},{col}]", f"expected 0x{wanted:04x}", f"got 0x{observed:04x}")

    def test_first_memory_byte_difference_uses_address_order_across_partitions(self):
        later, changed_later = MemoryRegion(BASE + 32, b"later"), MemoryRegion(BASE + 32, b"Later")
        whole = MemoryRegion(BASE, bytes(range(8)))
        first = MemoryRegion(BASE, bytes(range(4)))
        second = MemoryRegion(BASE + 4, b"\x04\xfe\x06\x07")
        expected = EvaluationResult({}, (later, whole))
        actual = EvaluationResult({}, (changed_later, second, first))
        self.diagnostic(expected, actual, "memory at 0x80000005", "expected 0x05", "got 0xfe")

    def test_mapping_holes_and_extra_zero_bytes_are_not_equal_to_mapped_zero_bytes(self):
        full = EvaluationResult({}, (MemoryRegion(BASE, b"\x00" * 8),))
        hole = EvaluationResult({}, (MemoryRegion(BASE + 4, b"\x00" * 4), MemoryRegion(BASE, b"\x00" * 3)))
        self.diagnostic(full, hole, "memory mapping at 0x80000003", "missing from actual")
        self.diagnostic(hole, full, "memory mapping at 0x80000003", "missing from expected")
        extra = EvaluationResult({}, (MemoryRegion(BASE, b"\x00" * 9),))
        self.diagnostic(full, extra, "memory mapping at 0x80000008", "missing from expected")
        self.diagnostic(full, EvaluationResult({}), "memory mapping at 0x80000000", "missing from actual")

    def test_comparison_detects_a_relu_to_mov_semantic_mutation(self):
        original = sparse({(2, 19): 0xBF80}, "bf16")

        def execute(kind):
            stream = Stream()
            value = stream.input()
            result = stream.pure("vpu_unary", value, "bf16", f'{{kind = "{kind}"}}')
            stream.output(result, 7)
            return evaluate(stream.program(), RuntimeInputs({0: original}))

        self.diagnostic(execute("relu"), execute("mov"), "output 7 tile[2,19]", "expected 0x0000", "got 0xbf80")

    def test_comparison_detects_mxu_reset_changed_to_seeded_accumulation(self):
        one = sparse({(0, 0): 0x38})
        seed = sparse({(0, 0): 0x3F80}, "bf16")
        runtime = mxu_inputs(one, one, seed)

        def execute(use_seed):
            stream = MxuStream()
            stream.weight()
            if use_seed:
                stream.seed()
                stream.accumulate("acc", "result_acc")
            else:
                stream.reset("result_acc")
            stream.readout("result_acc", "result")
            stream.output("result")
            return evaluate(stream.program(), runtime)

        expected, actual = execute(False), execute(True)
        self.diagnostic(expected, actual, "output 0 tile[0,0]", "expected 0x3f80", "got 0x4000")

    def test_comparison_detects_dma_loading_a_corrupted_source_into_the_destination(self):
        original = b"\x5a" * 1024
        corrupt = original[:19] + b"\x5b" + original[20:]
        original_source = MemoryRegion(BASE + 0x4000, original)
        corrupt_source = MemoryRegion(BASE + 0x6000, corrupt)
        destination = MemoryRegion(BASE, b"?" * 1024)
        runtime = RuntimeInputs(memory=(corrupt_source, destination, original_source))

        def execute(address):
            stream = Stream()
            ready = stream.await_(stream.load(address, "fp8"), "fp8")
            stream.wait(stream.store(ready, BASE, "fp8"))
            return evaluate(stream.program(), runtime)

        self.diagnostic(execute(BASE + 0x4000), execute(BASE + 0x6000), "memory at 0x80000013", "expected 0x5a", "got 0x5b")
        self.assertEqual(runtime.memory[1].data, b"?" * 1024)


if __name__ == "__main__":
    unittest.main()
