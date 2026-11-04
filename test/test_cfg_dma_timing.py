"""Run lit-style CFG timing fixtures through the public CLI using the repository's unittest runner."""

from __future__ import annotations

import unittest

from test_delay_insertion import OPT, ROOT, run


class CFGDMATimingTest(unittest.TestCase):
    def test_real_kit_prefetch_loop(self) -> None:
        source = (ROOT / "test/examples/cfg_dma_kit_prefetch.mlir").read_text()
        result = run(OPT, source, "--verify-atlas-machine-stream", "--verify-atlas-timing")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertNotIn("warning:", result.stderr)

    def test_injected_real_kit_overlap(self) -> None:
        source = (ROOT / "test/examples/cfg_dma_kit_overlap.mlir").read_text()
        # First establish that the injected program remains valid machine IR.
        stream = run(OPT, source, "--verify-atlas-machine-stream")
        self.assertEqual(stream.returncode, 0, stream.stderr)
        result = run(OPT, source, "--verify-atlas-timing")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("conflict", result.stderr)
        self.assertEqual(result.stdout, "")


if __name__ == "__main__":
    unittest.main()
