"""Exercise exact byte diagnostics, alignment, tails and inaccessible neighbors."""
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).absolute().parents[1]


class HostCheckTest(unittest.TestCase):
    def test_exact_checker(self):
        self._run_checker("host-check-test.c")

    def test_disappearing_mismatch_fails_closed(self):
        self._run_checker("host-check-unstable-test.c")

    def _run_checker(self, source):
        cc = shutil.which("cc")
        if cc is None:
            self.skipTest("native C compiler unavailable")
        with tempfile.TemporaryDirectory(prefix="atlas-host-check-") as directory:
            executable = Path(directory) / "check"
            subprocess.run([cc, "-std=gnu11", "-O2", "-Wall", "-Wextra", "-Werror",
                            "-fstrict-aliasing", "-fsanitize=undefined", "-fsanitize-undefined-trap-on-error",
                            "-I" + str(ROOT / "runtime"), str(ROOT / "test" / source),
                            "-o", str(executable)], check=True, timeout=30)
            subprocess.run([str(executable)], check=True, timeout=30)


if __name__ == "__main__":
    unittest.main()
