"""The build hang guard aborts a step that stops making progress, never one that keeps writing its outputs."""

from pathlib import Path
import sys
import tempfile
import time
from unittest import TestCase, main

from atlas_mlir import driver, toolchain


class HangGuardTest(TestCase):
    def test_step_that_keeps_writing_outputs_runs_past_the_guard(self):
        with tempfile.TemporaryDirectory() as root:
            out = Path(root) / "out"
            # Writes one output file every 0.2 s for 2.4 s: four times the 0.6 s guard, never quiet for 0.6 s.
            script = ("import pathlib, sys, time\n"
                      "out = pathlib.Path(sys.argv[1]); out.mkdir()\n"
                      "for i in range(12):\n"
                      "    time.sleep(0.2); (out / f'part{i}').write_text(str(i))\n")
            start = time.monotonic()
            driver._run([sys.executable, "-c", script, out], log=Path(root) / "logs" / "step.log", timeout=0.6, watch=(out,))
            self.assertGreater(time.monotonic() - start, 2.4)
            self.assertEqual(len(list(out.iterdir())), 12)

    def test_step_without_progress_is_aborted(self):
        with tempfile.TemporaryDirectory() as root:
            out = Path(root) / "out"
            out.mkdir()
            start = time.monotonic()
            with self.assertRaisesRegex(toolchain.ToolFailure, "made no progress for 0.6 s"):
                driver._run([sys.executable, "-c", "import time; time.sleep(30)"], log=Path(root) / "logs" / "step.log",
                            timeout=0.6, watch=(out,))
            self.assertLess(time.monotonic() - start, 10)

    def test_failed_step_reports_its_status(self):
        with tempfile.TemporaryDirectory() as root:
            with self.assertRaisesRegex(toolchain.ToolFailure, r"step failed \(3\)"):
                driver._run([sys.executable, "-c", "raise SystemExit(3)"], log=Path(root) / "step.log", timeout=5)


if __name__ == "__main__":
    main()
