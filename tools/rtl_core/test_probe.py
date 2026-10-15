"""Admission and launch controls without creating unadmitted filesystem paths."""
import importlib.util
import hashlib
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
        support = SimpleNamespace(preflight_archive=lambda value: Path(value))
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

    def test_jar_admission_uses_preflight_before_content(self):
        support = SimpleNamespace(preflight_archive=mock.Mock(side_effect=ValueError("unadmitted metadata")))
        with mock.patch.object(probe.zipfile, "ZipFile") as archive:
            with self.assertRaisesRegex(ValueError, "unadmitted metadata"):
                probe.jar_metadata("admitted.jar", support)
            archive.assert_not_called()
        support.preflight_archive.assert_called_once_with("admitted.jar")

    def test_expanded_helper_paths_refuse_before_metadata(self):
        spec = importlib.util.spec_from_file_location("release_rebuild_paths", Path(__file__).with_name("rebuild_runner.py"))
        rebuild = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(rebuild)
        for expanded in ("/unread/HaMmEr/helper.py", "/unread/VlSi/helper.py", "/unread/../helper.py"):
            with self.subTest(expanded=expanded), \
                    mock.patch.object(Path, "absolute", return_value=Path(expanded)), \
                    mock.patch.object(probe.os, "lstat", side_effect=AssertionError("metadata access")) as access:
                with self.assertRaises(ValueError):
                    probe.helper("helper.py")
                with self.assertRaises(ValueError):
                    rebuild.run(SimpleNamespace(support_helper=Path("helper.py")))
                access.assert_not_called()

    def test_release_component_helper_matches_source_pin(self):
        path = Path(__file__).absolute().parents[1] / "rtl_mxu0/run.py"
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        self.assertEqual(digest, probe.HELPER_SHA)
        import ast
        for name in ("run_program.py", "rebuild_runner.py"):
            tree = ast.parse(Path(__file__).with_name(name).read_text())
            pin = next(node.value.value for node in tree.body if isinstance(node, ast.Assign)
                       and any(isinstance(target, ast.Name) and target.id == "HELPER_SHA" for target in node.targets))
            self.assertEqual(pin, digest)


if __name__ == "__main__":
    unittest.main()
