"""Run lit-style CFG timing fixtures through the public CLI using the repository's unittest runner."""

from __future__ import annotations

import unittest

from test_delay_insertion import OPT, ROOT, addi, nop, program, run


def dma(base: int = 4) -> tuple[str, str]:
    return ("dma", f'direction = "load", channel = 0 : i32, reg = {base} : i32, '
                   'dram = 7 : i32, size = 9 : i32')


def upper(dst: int, immediate: int) -> tuple[str, str]:
    return ("upper", f'kind = "lui", dst = {dst} : i32, immediate = {immediate} : i32')


def vwork(base: int) -> list[tuple[str, str]]:
    return [("vload", f'dst = 0 : i32, base = {base} : i32, offset = 0 : i32, format = "raw"'),
            ("delay", "cycles = 66 : i32"), nop()]


WAIT = ("dma_wait", "channel = 0 : i32")
HALT = ("trap", 'kind = "ecall"')


def prefix() -> list[tuple[str, str]]:
    return [addi(4, 0, 256), addi(5, 0, 0), upper(7, 0x90000), addi(9, 0, 1024)]


class CFGDMATimingTest(unittest.TestCase):
    def assert_timing_rejects(self, ops: list[tuple[str, str]], *diagnostics: str) -> None:
        source = program(ops)
        stream = run(OPT, source, "--verify-atlas-machine-stream")
        self.assertEqual(stream.returncode, 0, stream.stderr)
        result = run(OPT, source, "--verify-atlas-timing")
        self.assertNotEqual(result.returncode, 0, result.stderr)
        for diagnostic in diagnostics:
            self.assertIn(diagnostic, result.stderr)
        self.assertEqual(result.stdout, "")

    def test_relaunch_same_channel_one_wait(self) -> None:
        # One wait cannot establish completion of two launches sharing a busy bit.
        self.assert_timing_rejects(
            prefix() + [addi(6, 0, 512), dma(), dma(6), WAIT] + vwork(6) + [HALT],
            "reuses DMA channel 0", "atlas.dma_wait")

    def test_dma_size_zero(self) -> None:
        self.assert_timing_rejects(
            [addi(4, 0, 256), addi(5, 0, 256), upper(7, 0x90000), addi(9, 0, 0), dma()]
            + vwork(5) + [WAIT, HALT],
            "DMA size must be nonzero, a multiple of 32 bytes, and less than 8192 bytes")

    def test_dma_size_8192_truncates_in_rtl(self) -> None:
        # The 13-bit RTL transfer size truncates 8192 to zero.
        self.assert_timing_rejects(
            [addi(4, 0, 256), upper(5, 1), upper(7, 0x90000), upper(9, 2), dma(), WAIT]
            + vwork(5) + [HALT],
            "DMA size must be nonzero, a multiple of 32 bytes, and less than 8192 bytes")

    def test_command_ring_overwrite_two_pending(self) -> None:
        other = ("dma", 'direction = "load", channel = 1 : i32, reg = 6 : i32, '
                        'dram = 7 : i32, size = 9 : i32')
        wait_other = ("dma_wait", "channel = 1 : i32")
        ops = [addi(4, 0, 256), addi(6, 0, 512), addi(9, 0, 32), upper(7, 0x90000), dma()]
        self.assert_timing_rejects(ops + [other, wait_other] * 8 + [WAIT, HALT],
                                   "pending DMA command ring slot")
        # Drain the old slot before the ninth launch wraps the enqueue index.
        source = program(ops + [other, wait_other] * 7 + [WAIT, other, wait_other, HALT])
        result = run(OPT, source, "--verify-atlas-machine-stream", "--verify-atlas-timing")
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_dma_store_size_8160_overflows_queue(self) -> None:
        store = ("dma", 'direction = "store", channel = 0 : i32, reg = 4 : i32, '
                        'dram = 7 : i32, size = 9 : i32')
        self.assert_timing_rejects(
            [addi(4, 0, 256), upper(7, 0x90000), upper(9, 2), addi(9, 9, -32), store, WAIT, HALT],
            "DMA store size must not exceed 4096 bytes")

    def test_counted_prefetch_loop(self) -> None:
        source = (ROOT / "test/lit/cfg_dma_prefetch.mlir").read_text()
        result = run(OPT, source, "--verify-atlas-machine-stream", "--verify-atlas-timing")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertNotIn("warning:", result.stderr)

    def test_injected_prefetch_overlap(self) -> None:
        source = (ROOT / "test/lit/cfg_dma_overlap.mlir").read_text()
        # First establish that the injected program remains valid machine IR.
        stream = run(OPT, source, "--verify-atlas-machine-stream")
        self.assertEqual(stream.returncode, 0, stream.stderr)
        result = run(OPT, source, "--verify-atlas-timing")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("conflict", result.stderr)
        self.assertEqual(result.stdout, "")


if __name__ == "__main__":
    unittest.main()
