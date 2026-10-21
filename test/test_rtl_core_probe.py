"""Source admission and bounded launch controls without an RTL build."""

import importlib.util
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

TOOL_ROOT = Path(__file__).absolute().parents[1] / "tools/rtl_core"
sys.path.insert(0, str(TOOL_ROOT.parent))

spec = importlib.util.spec_from_file_location("core_probe", TOOL_ROOT / "probe.py")
probe = importlib.util.module_from_spec(spec)
spec.loader.exec_module(probe)


class AdmissionControls(unittest.TestCase):
    def test_helper_lexical_refusal_precedes_metadata(self):
        for value in ("/unread/../helper.py", "/unread/branch\\helper.py"):
            with self.subTest(value=value), mock.patch.object(probe.os, "lstat") as access:
                with self.assertRaises(ValueError):
                    probe.helper(value)
                access.assert_not_called()

    def test_archive_member_refusal_precedes_content(self):
        archive = mock.MagicMock()
        archive.__enter__.return_value = archive
        archive.infolist.return_value = [SimpleNamespace(filename="permitted.class"),
            SimpleNamespace(filename="../outside.class")]
        support = SimpleNamespace(preflight_archive=lambda value: Path(value))
        with mock.patch.object(probe.zipfile, "ZipFile", return_value=archive):
            with self.assertRaises(ValueError):
                probe.jar_metadata("admitted.jar", support)
        archive.read.assert_not_called()
        archive.open.assert_not_called()

    def test_subprocess_timeout_is_recorded_without_cpu_policy(self):
        support = SimpleNamespace(safe_path=lambda path: path, write_json=mock.Mock())
        stream = mock.MagicMock()
        with mock.patch.object(Path, "open", return_value=stream), mock.patch.object(subprocess, "run",
            side_effect=subprocess.TimeoutExpired(["tool"], 1500)) as launch:
            result = probe.bounded(["tool"], Path("."), {}, Path("unused.log"), support)
        self.assertEqual(result["returncode"], 124)
        self.assertEqual(launch.call_args.kwargs["timeout"], 1500)
        support.write_json.assert_called_once()

    def test_jar_admission_uses_preflight_before_content(self):
        support = SimpleNamespace(preflight_archive=mock.Mock(side_effect=ValueError("unadmitted metadata")))
        with mock.patch.object(probe.zipfile, "ZipFile") as archive:
            with self.assertRaisesRegex(ValueError, "unadmitted metadata"):
                probe.jar_metadata("admitted.jar", support)
            archive.assert_not_called()
        support.preflight_archive.assert_called_once_with("admitted.jar")

    def test_expanded_helper_paths_refuse_before_metadata(self):
        with mock.patch.object(Path, "absolute", return_value=Path("/unread/../helper.py")), mock.patch.object(probe.os,
            "lstat", side_effect=AssertionError("metadata access")) as access:
            with self.assertRaises(ValueError):
                probe.helper("helper.py")
            access.assert_not_called()

    def test_caller_can_select_an_updated_support_helper(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "support.py"
            path.write_text("value = 1\n")
            self.assertEqual(probe.helper(path).value, 1)
            path.write_text("value = 23\n")
            self.assertEqual(probe.helper(path).value, 23)

    def test_builder_authenticates_frozen_helper_before_loading(self):
        spec = importlib.util.spec_from_file_location("core_builder", TOOL_ROOT / "build_runner.py")
        builder = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(builder)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            probe_path = root / "probe.py"
            probe_path.write_text('def helper(path):\n    raise AssertionError("changed helper was loaded")\n')
            helper_path = root / "reviewed-support.py"
            helper_path.write_text("value = 1\n")
            records = [{"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()} for path in (probe_path,
                helper_path)]
            helper_path.write_text("value = 2\n")
            receipt = root / "receipt.json"
            receipt.write_text(json.dumps({"records": records, "status": "passed_compile_elaboration_only",
                "source_artifacts_preserved": True}))
            args = SimpleNamespace(core_root=root, core_sha256=hashlib.sha256(receipt.read_bytes()).hexdigest())
            with self.assertRaisesRegex(ValueError, "Frozen support helper differs"):
                builder.run(args)


if __name__ == "__main__":
    unittest.main()
