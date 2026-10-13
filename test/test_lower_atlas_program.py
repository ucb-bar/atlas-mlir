"""The pipeline driver keeps every stage's artifact and stops at the first failure."""

from __future__ import annotations

import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from test_virtual_scheduling import wide_return_block
from test_virtual_ssa import BIN, ROOT


def drive(source: Path, output: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run([sys.executable, str(ROOT / "tools/lower_atlas_program.py"),
                           str(source), "--output-dir", str(output),
                           "--atlas-bin-dir", str(BIN)],
                          text=True, capture_output=True)


class LowerAtlasProgramTest(unittest.TestCase):
    def test_artifacts_or_the_failing_stage(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            done = Path(temp) / "attention"
            result = drive(ROOT / "test/examples/virtual_attention_tile.mlir", done)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(sorted(path.name for path in done.iterdir()), [
                "1-virtual.mlir", "2-scheduled.mlir", "3-allocated.mlir", "4-timed.mlir",
                "5-map.json", "5-program.json", "5-words.hex", "commands.txt", "log.txt"])
            self.assertEqual(len((done / "commands.txt").read_text().splitlines()), 7)

            # 32 BF16 tiles live across one edge cannot fit 31 register pairs.
            source = Path(temp) / "wide.mlir"
            source.write_text(wide_return_block(32))
            failed = Path(temp) / "wide"
            result = drive(source, failed)
            self.assertEqual(result.returncode, 1)
            self.assertFalse((failed / "3-allocated.mlir").exists())
            last = (failed / "log.txt").read_text().split("\n$ ")[-1]
            self.assertIn("--lower-atlas-virtual-to-machine", last)
            self.assertIn("exceeds 31 physical pairs", last)


if __name__ == "__main__":
    unittest.main()
