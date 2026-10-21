"""Shared physical-path admission before any filesystem content is read."""

from pathlib import Path
import stat
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.atlas_path_guard import checked


class PathAdmission(unittest.TestCase):
    def test_parent_component_is_never_inspected(self):
        with patch.object(Path, "lstat", side_effect=AssertionError("must not inspect")):
            with self.assertRaises(ValueError):
                checked(Path("/tmp/../file"))

    def test_symlink_is_refused_without_reading_target(self):
        def lstat(path):
            return type("Mode", (), {"st_mode": stat.S_IFLNK if path.name == "link" else stat.S_IFDIR})()

        with patch.object(Path, "lstat", lstat), patch("os.readlink") as target:
            with self.assertRaises(ValueError):
                checked(Path("/tmp/link/file"))
            target.assert_not_called()


if __name__ == "__main__":
    unittest.main()
