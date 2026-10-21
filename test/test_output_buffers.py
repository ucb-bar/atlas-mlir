"""Explicit output buffers bound stores in counted CFG loops."""

import unittest

from test_bounded_dma_addresses import PIPELINE
from test_virtual_ssa import ROOT, run

SOURCE = (ROOT / "test/examples/virtual_output_buffer_loop.mlir").read_text()


class OutputBufferTest(unittest.TestCase):
    def test_complete_looped_buffer_lowers_and_reparses(self):
        machine = run("atlas-opt", SOURCE, *PIPELINE)
        self.assertEqual(machine.returncode, 0, machine.stderr[:1800])
        self.assertIn("output_buffer = {base = 2416050176 : i64, bytes = 6144 : i64}", machine.stdout)
        checked = run("atlas-opt", machine.stdout, "--verify-atlas-generated-schedule", "--verify-atlas-timing")
        self.assertEqual(checked.returncode, 0, checked.stderr[:1800])

    def test_default_abi_and_invalid_buffer_ranges_refuse(self):
        mutations = [
            (", atlas.output_dram_bytes = 6144 : i64", "", "return block"),
            ("atlas.output_dram_bytes = 6144", "atlas.output_dram_bytes = 4096", "exceeds"),
            ("atlas.output_dram_bytes = 6144", "atlas.output_dram_bytes = 6143", "aligned"),
            ("atlas.output_dram_bytes = 6144", "atlas.output_dram_bytes = 0", "aligned"),
            ("atlas.output_dram_bytes = 6144 : i64", "atlas.output_dram_bytes = 6144 : i32", "aligned"),
            ("%input = arith.constant -1879048192", "%input = arith.constant -1878917120", "overlaps"),
        ]
        for old, new, diagnostic in mutations:
            with self.subTest(new=new):
                result = run("atlas-opt", SOURCE.replace(old, new), "--verify-atlas-virtual-stream")
                self.assertNotEqual(result.returncode, 0)
                self.assertIn(diagnostic, result.stderr)

    def test_changed_buffer_declarations_or_tile_footprints_fail(self):
        machine = run("atlas-opt", SOURCE, *PIPELINE)
        self.assertEqual(machine.returncode, 0, machine.stderr[:1800])
        for old, new in [
            ("atlas.output_dram_bytes = 6144", "atlas.output_dram_bytes = 4096"),
            ("bytes = 6144 : i64", "bytes = 4096 : i64"),
        ]:
            result = run("atlas-opt", machine.stdout.replace(old, new, 1), "--verify-atlas-generated-schedule")
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("output buffer", result.stderr)


if __name__ == "__main__":
    unittest.main()
