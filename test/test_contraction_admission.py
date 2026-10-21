"""Pure negative controls for physical-output and metadata-path admission."""

from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import validate_model_contraction as admission
import atlas_path_guard as paths
from prepare_resnet_graph_rtl import relocation_mapper


class ContractionAdmissionTest(unittest.TestCase):
    def test_explicit_prefix_mapping_is_lexical_and_uses_longest_prefix(self):
        with patch.object(Path, "stat", side_effect=AssertionError("filesystem")):
            mapper = relocation_mapper(["/old=/new", "/old/model=/models/pinned"])
            self.assertEqual(mapper(Path("/old/model/file.py")), Path("/models/pinned/file.py"))
            self.assertEqual(mapper(Path("/old/receipt.json")), Path("/new/receipt.json"))
            self.assertEqual(mapper(Path("/older/file")), Path("/older/file"))
            self.assertIsNone(relocation_mapper([]))
            for specifications in (["missing-separator"], ["relative=/new"], ["/old=/new/../bad"],
                ["/old=/new", "/old=/other"]):
                with self.subTest(specifications=specifications), self.assertRaises(ValueError):
                    relocation_mapper(specifications)

    def test_metadata_relocation_changes_only_absolute_paths(self):
        original = {"/old/input": ["/old/model/file.py", "relative/file", "sha", 3, 1.25, True, None]}
        expected = {"/new/input": ["/new/model/file.py", "relative/file", "sha", 3, 1.25, True, None]}
        cache = {}
        mapper = lambda path: Path("/new") / path.relative_to("/old")
        with patch.object(Path, "stat", side_effect=AssertionError("filesystem")):
            self.assertEqual(paths.map_metadata_paths(original, mapper, mapped_paths=cache), expected)
        self.assertIn("/old/input", original)
        self.assertIs(paths.map_metadata_paths(original, None), original)
        self.assertEqual(len(cache), 2)
        self.assertEqual(paths.map_metadata_paths(original, mapper, mapped_paths=cache), expected)

    def test_metadata_bad_old_path_precedes_callback_and_io(self):
        for bad in ("/old/../file", "/old/a\\b", "/old/a\n"):
            with self.subTest(path=bad), patch.object(Path, "stat",
                side_effect=AssertionError("filesystem")), self.assertRaises(ValueError):
                paths.map_metadata_paths(["/old/safe", bad], lambda p: self.fail("callback"))

    def test_metadata_bad_target_and_cross_document_collisions_fail(self):
        for bad in ("relative", "/new/../file", "/new/a\\b", "/new/a\n"):
            with self.subTest(path=bad), self.assertRaises(ValueError):
                paths.map_metadata_paths("/old/file", lambda _: bad)
        cache = {}
        paths.map_metadata_paths({"/old/a": "hash"}, lambda _: Path("/new/file"), mapped_paths=cache)
        with self.assertRaisesRegex(ValueError, "collision"):
            paths.map_metadata_paths({"/old/b": "hash"}, lambda _: Path("/new/file"), mapped_paths=cache)
        self.assertEqual(cache, {"/old/a": "/new/file"})
        with self.assertRaisesRegex(ValueError, "collision"):
            paths.map_metadata_paths({"/old/a": 1, "/old/b": 2}, lambda _: Path("/new/file"))

    def test_result_archive_names_are_refused_before_member_load(self):
        required = [SimpleNamespace(filename=name, file_size=0) for name in ("target_C.npy", "actual_bits.npy",
            "expected_bits.npy")]
        with patch.object(admission, "physical_path", return_value=Path("/not-opened.npz")), patch.object(admission.zipfile,
            "ZipFile") as archive, patch("numpy.load", side_effect=AssertionError("member load")) as load:
            for name in ("actual_bits.npy", "../actual_bits.npy", "folder/"):
                archive.return_value.__enter__.return_value.infolist.return_value = required + [
                    SimpleNamespace(filename=name, file_size=0)]
                with self.subTest(name=name), self.assertRaises(ValueError):
                    admission.load_arrays("/not-opened.npz", result=True)
            load.assert_not_called()

    def test_optional_accumulator_cannot_change_output_schema(self):
        words = np.full((32, 32), 0x3F80, dtype=np.uint16)
        result = {"actual_bits": words, "expected_bits": words.copy(),
            "target_C": (words.astype(np.uint32) << 16).view(np.float32)}
        for change in (words[:-1], words.astype(np.uint32), np.full_like(words, 0x7F80)):
            with self.assertRaises(ValueError):
                admission.admit_result_arrays(dict(result, raw_accumulator_bits=change), (32, 32))
        with self.assertRaises(ValueError):
            admission.admit_result_arrays(dict(result, unknown=words), (32, 32))

    def test_path_refusal_precedes_filesystem_access(self):
        with patch.object(Path, "lstat", side_effect=AssertionError("filesystem")):
            for path in ("/not-opened/../file",):
                with self.subTest(path=path), self.assertRaises(ValueError):
                    admission.physical_path(path)

if __name__ == "__main__":
    unittest.main()
