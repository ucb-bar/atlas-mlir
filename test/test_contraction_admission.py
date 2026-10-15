"""Pure negative controls for physical-output and source-epoch admission."""

import copy
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from atlas_tile_arithmetic import output_halves
import validate_model_contraction as admission


class ContractionAdmissionTest(unittest.TestCase):
    def test_result_archive_names_are_refused_before_member_load(self):
        required = [SimpleNamespace(filename=name, file_size=0)
                    for name in ("target_C.npy", "actual_bits.npy", "expected_bits.npy")]
        with patch.object(admission, "physical_path", return_value=Path("/not-opened.npz")), \
                patch.object(admission.zipfile, "ZipFile") as archive, \
                patch("numpy.load", side_effect=AssertionError("member load")) as load:
            for name in ("actual_bits.npy", "../actual_bits.npy", "folder/"):
                archive.return_value.__enter__.return_value.infolist.return_value = required + [SimpleNamespace(filename=name, file_size=0)]
                with self.subTest(name=name), self.assertRaises(ValueError):
                    admission.load_arrays("/not-opened.npz", result=True)
            load.assert_not_called()

    def test_coherent_aggregate_corruption_still_fails_physical_binding(self):
        words = np.full((32, 32), 0x3F80, dtype=np.uint16)
        payload = output_halves(words)
        words[7, 29] ^= np.uint16(1)
        result = {"actual_bits": words, "expected_bits": words.copy(),
                  "target_C": (words.astype(np.uint32) << 16).view(np.float32)}
        admission.admit_result_arrays(result, (32, 32))
        with self.assertRaisesRegex(ValueError, "physical tile"):
            admission.admit_tile_bytes(words, {"row_range": [0, 32], "column_range": [0, 32]}, payload)

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
        with patch.object(admission, "safe_path", side_effect=AssertionError("filesystem")):
            for path in ("/not-opened/../file", "/not-opened/Mixed-HAMMER/file", "/not-opened/Mixed-VlSi/file"):
                with self.subTest(path=path), self.assertRaises(ValueError):
                    admission.physical_path(path)
        for path in ("../outside.py", "/absolute.py", "other/file.py"):
            with self.assertRaises(ValueError):
                admission.model_source_path(Path("/not-opened"), path)

    def test_changed_bookends_or_late_failure_never_admit(self):
        source = {"input": "npz", "manifest": "manifest"}
        model = {"revision": admission.MODEL_PIN, "tracked_source_diff_exit": 0,
                 "imported_source_sha256": {"npu_model/__init__.py": "sha"}, "curve_sha256": "curve"}
        code = {name: "sha" for name in ("run_model_contraction.py", "run_model_tile.py", "atlas_tile_arithmetic.py")}
        receipt = {"status": "passed_model_only", "recipe": admission.RECIPES["tensor-pow2"],
                   "native_fidelity_claim": False, "raw_image_execution_claim": False,
                   "invocations": "serialized_fresh_core_instances_one_python_process", "source_inputs": source,
                   "model_binding_before": model, "source_bookends": {"unchanged": True, "model_after": model,
                        "input_before": source, "input_after": source}, "code_before": code, "code_after": code,
                   "model": {"revision": admission.MODEL_PIN, "numerics": "rtl", "backend": "curve",
                        "curve_sha256": "curve", "imported_source_sha256": model["imported_source_sha256"]}}
        admission.admit_bookends(receipt, source)
        for change in ("model", "input", "code", "error", "status"):
            altered = copy.deepcopy(receipt)
            if change == "model":
                altered["source_bookends"]["model_after"] = dict(model, revision="wrong")
            elif change == "input":
                altered["source_bookends"]["input_after"] = dict(source, input="changed")
            elif change == "code":
                altered["code_after"] = dict(code, **{"run_model_tile.py": "changed"})
            else:
                altered[change] = "late failure"
            with self.subTest(change=change), self.assertRaises(ValueError):
                admission.admit_bookends(altered, source)


if __name__ == "__main__":
    unittest.main()
