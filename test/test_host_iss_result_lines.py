"""Missing progress, UART failures and duplicate completion fail admission."""

from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).absolute().parents[1] / "tools/host_iss"))
from result_lines import verify


class ResultLinesTest(unittest.TestCase):
    def lines(self):
        return ("ATLAS_KERNEL_BEGIN name=tiny-full launches_completed=0 launches_expected=2\n"
                "ATLAS_KERNEL_PROGRESS name=tiny-full launches_completed=2 launches_expected=2 boundary=layer\n"
                "ATLAS_KERNEL_PASS name=tiny-full launches_completed=2 launches_expected=2\n").encode()

    def test_complete_uart_transcript(self):
        self.assertEqual(verify(self.lines(), "tiny-full", 2)["progress_records"], 1)

    def test_missing_failure_reordered_duplicate_and_wrong_workload_refused(self):
        raw = self.lines()
        for altered in (b"", raw.splitlines(keepends=True)[0], raw + raw.splitlines(keepends=True)[-1],
                        raw.replace(b"_PASS", b"_FAIL"), raw.replace(b"completed=2", b"completed=1"),
                        raw.replace(b"name=tiny-full", b"name=smol-full"),
                        b"".join(raw.splitlines(keepends=True)[::2]),
                        raw.replace(b"boundary=layer", b"boundary=layer name=tiny-full")):
            with self.subTest(altered=altered), self.assertRaises(ValueError):
                verify(altered, "tiny-full", 2)


if __name__ == "__main__":
    unittest.main()
