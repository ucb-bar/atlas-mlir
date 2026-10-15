"""Admission and launch controls without creating unadmitted filesystem paths."""
import importlib.util
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

spec = importlib.util.spec_from_file_location("core_probe", Path(__file__).with_name("probe.py"))
probe = importlib.util.module_from_spec(spec)
spec.loader.exec_module(probe)


class AdmissionControls(unittest.TestCase):
    def test_helper_lexical_refusal_precedes_metadata(self):
        for value in ("/unread/HaMmEr-cache/helper.py", "/unread/VLSI/helper.py", "/unread/../helper.py"):
            with self.subTest(value=value), mock.patch.object(probe.os, "lstat") as access:
                with self.assertRaises(ValueError):
                    probe.helper(value)
                access.assert_not_called()

    def test_archive_member_refusal_precedes_content(self):
        archive = mock.MagicMock()
        archive.__enter__.return_value = archive
        archive.infolist.return_value = [SimpleNamespace(filename="permitted.class"),
                                        SimpleNamespace(filename="unread/HaMmErThing.class")]
        support = SimpleNamespace(safe_path=lambda value: Path(value))
        with mock.patch.object(probe.zipfile, "ZipFile", return_value=archive):
            with self.assertRaises(ValueError):
                probe.jar_metadata("admitted.jar", support)
        archive.read.assert_not_called()
        archive.open.assert_not_called()

    def test_heavy_launch_refuses_wrong_affinity(self):
        with mock.patch.object(probe.os, "sched_getaffinity", return_value={0}), \
                mock.patch.object(probe.subprocess, "Popen") as launch:
            with self.assertRaises(RuntimeError):
                probe.bounded([], Path("."), {}, Path("unused.log"), None)
            launch.assert_not_called()


if __name__ == "__main__":
    unittest.main()
