"""Interpreter link admission controls using metadata spies only."""
import stat
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock
from capture_programs import executable


class InterpreterAdmission(unittest.TestCase):
    def test_unadmitted_original_path_precedes_metadata(self):
        for value in ("/unread/HaMmEr/python", "/unread/VLSI/python", "/unread/../python"):
            with self.subTest(value=value), mock.patch("capture_programs.os.lstat") as metadata:
                with self.assertRaises(ValueError):executable(value)
                metadata.assert_not_called()

    def test_link_target_is_filtered_before_following(self):
        def metadata(path):
            return SimpleNamespace(st_mode=stat.S_IFLNK if Path(path)==Path("/approved/python") else stat.S_IFDIR)
        with mock.patch("capture_programs.os.lstat",side_effect=metadata) as access, \
                mock.patch("capture_programs.os.readlink",return_value="/unread/vLsI/python"):
            with self.assertRaises(ValueError):executable("/approved/python")
            self.assertEqual([call.args[0] for call in access.call_args_list],
                             [Path("/approved"),Path("/approved/python")])

    def test_interpreter_link_loop_is_refused(self):
        def metadata(path):
            return SimpleNamespace(st_mode=stat.S_IFLNK if Path(path)==Path("/approved/python") else stat.S_IFDIR)
        with mock.patch("capture_programs.os.lstat",side_effect=metadata), \
                mock.patch("capture_programs.os.readlink",return_value="/approved/python"):
            with self.assertRaisesRegex(ValueError,"loop"):executable("/approved/python")


if __name__=="__main__":unittest.main()
