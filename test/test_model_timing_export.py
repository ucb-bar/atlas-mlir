"""Event sweeps require a complete progression of distinct ages."""
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from atlas_model_timing import compress


class TimingExportTest(unittest.TestCase):
    def test_event_sweeps_reject_missing_or_repeated_edges(self):
        self.assertEqual(compress([3, 1, 2]), dict(count=3, first_age=1, last_age=3, step=1))
        for ages in ([], [1, 1], [1, 2, 4]):
            with self.subTest(ages=ages), self.assertRaises(ValueError):
                compress(ages)


if __name__ == "__main__":
    unittest.main()
