"""Access checks run before any excluded-path or symlink lookup."""
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import atlas_derive_mxu as mxu
import atlas_derive_vpu as vpu
import atlas_model_facts as facts
import atlas_model_timing as timing
try:
    import atlas_model_pairs as pairs
except ModuleNotFoundError:
    pairs = None
try:
    import atlas_derive_curve as curve
except ModuleNotFoundError:
    curve = None


class AccessBoundaryTest(unittest.TestCase):
    def test_all_path_exclusions_precede_filesystem_access(self):
        for name in ("/tmp/HaMmEr-source/facts.json", "/tmp/vLsI/../facts.json"):
            checks = [lambda: facts.safe_path(name), lambda: mxu._read(name),
                      lambda: vpu._check_source(SimpleNamespace()),
                      lambda: timing.manifest(Path(name), "npu_model")]
            if pairs is not None:
                checks.append(lambda: pairs.safe_path(name))
            if curve is not None:
                checks.append(lambda: curve._safe(name))
            for check in checks:
                with self.subTest(name=name, check=check), \
                        patch.object(vpu.inspect, "getsourcefile", return_value=name), \
                        patch.object(Path, "is_symlink", side_effect=AssertionError("looked up excluded path")), \
                        patch.object(Path, "read_text", side_effect=AssertionError("read excluded path")), \
                        patch.object(timing.os, "scandir", side_effect=AssertionError("traversed excluded path")):
                    with self.assertRaises((ValueError, RuntimeError)):
                        check()

    def test_manifest_refuses_mock_excluded_entries_before_metadata_or_reads(self):
        # These are synthetic directory entries only, never paths on disk.
        for name in ("HaMmEr-data", "prefix-vLsI.py", "HAMMER.txt", "vlsi-cache"):
            for suffixes in (None, {".py", ".json", ".yaml"}):
                bad = SimpleNamespace(name=name,
                                      is_symlink=Mock(side_effect=AssertionError("entry lookup")),
                                      is_dir=Mock(side_effect=AssertionError("entry lookup")))
                allowed = SimpleNamespace(name="allowed.py",
                                          is_symlink=Mock(side_effect=AssertionError("premature entry lookup")),
                                          is_dir=Mock(side_effect=AssertionError("premature entry lookup")))
                with self.subTest(name=name, suffixes=suffixes), tempfile.TemporaryDirectory() as directory:
                    root = Path(directory)
                    (root / "npu_model").mkdir()
                    with patch.object(timing.os, "scandir") as scandir, \
                            patch.object(Path, "read_bytes", side_effect=AssertionError("partial manifest read")):
                        scandir.return_value.__enter__.return_value = iter((allowed, bad))
                        with self.assertRaisesRegex(RuntimeError, "prohibited entry name"):
                            timing.manifest(root, "npu_model", suffixes)
                        scandir.assert_called_once_with(root / "npu_model")
                    bad.is_symlink.assert_not_called()
                    bad.is_dir.assert_not_called()

    def test_all_path_helpers_check_symlink_ancestors_before_descendants(self):
        leaf = Path("/tmp/model-access-test/link/child.py")
        link = leaf.parent
        def is_link(path):
            if path == leaf:
                self.fail("looked through an unchecked ancestor symlink")
            return path == link
        checks = [lambda: facts.safe_path(leaf), lambda: mxu._read(leaf),
                  lambda: vpu._check_source(SimpleNamespace()),
                  lambda: timing.manifest(link.parent, "link/child.py")]
        if pairs is not None:
            checks.append(lambda: pairs.safe_path(leaf))
        if curve is not None:
            checks.append(lambda: curve._safe(leaf))
        for check in checks:
            with self.subTest(check=check), patch.object(Path, "is_symlink", is_link), \
                    patch.object(vpu.inspect, "getsourcefile", return_value=str(leaf)):
                with self.assertRaisesRegex((ValueError, RuntimeError), "symlink"):
                    check()


if __name__ == "__main__":
    unittest.main()
