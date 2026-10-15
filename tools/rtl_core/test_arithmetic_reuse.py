"""Fail-closed metadata controls; these do not represent model executions."""
from pathlib import Path
import copy
from types import SimpleNamespace
import unittest
from unittest import mock
import prepare_program as preparation


class ReusedEvidenceAdmission(unittest.TestCase):
    def test_control_payload_accepts_each_known_name_and_refuses_ambiguity(self):
        support = SimpleNamespace(safe_path=lambda path: path)
        names = ("actual-bf16-tiles-0.bin", "actual-bf16-halves-0.bin")
        for present in ((), names, (names[0],), (names[1],)):
            with self.subTest(present=present), mock.patch.object(Path, "exists", lambda path: path.name in present):
                if len(present) != 1:
                    with self.assertRaises(ValueError):preparation.control_payload_path(support, Path("/unread"))
                else:
                    self.assertEqual(preparation.control_payload_path(support, Path("/unread")).name, present[0])

    def test_only_two_exact_producer_profiles_are_admitted(self):
        epochs = sorted(preparation.ARITHMETIC_EPOCHS)
        for epoch in epochs:
            code = dict(zip(preparation.PRODUCER_NAMES, epoch))
            self.assertEqual(preparation.producer_epoch({"code_before": code, "code_after": code.copy()}), epoch)
        mixed = dict(zip(preparation.PRODUCER_NAMES, (epochs[0][0], epochs[1][1], epochs[0][2])))
        with self.assertRaisesRegex(ValueError, "Unknown or mixed"):
            preparation.producer_epoch({"code_before": mixed, "code_after": mixed.copy()})

    def test_code_bookend_changes_are_refused(self):
        code = dict(zip(preparation.PRODUCER_NAMES, next(iter(preparation.ARITHMETIC_EPOCHS))))
        changed = dict(code, atlas_tile_arithmetic="unrecognized")
        with self.assertRaises(ValueError):
            preparation.producer_epoch({"code_before": code, "code_after": changed})

    def test_reuse_controls_or_missing_model_root_refuse_before_io(self):
        for controls, root in ((True, Path("/unread")), (False, None)):
            with self.subTest(controls=controls), self.assertRaises(ValueError):
                preparation.prepare(None, None, None, None, None, controls,
                                    reuse_model_arithmetic=True, model_root=root)

    def test_ambiguous_actual_payload_names_are_refused_before_contents(self):
        support = SimpleNamespace(safe_path=lambda path: path)
        with mock.patch.object(Path, "exists", return_value=True), \
                mock.patch.object(Path, "read_bytes") as contents:
            with self.assertRaisesRegex(ValueError, "exactly one"):
                preparation.model_payload_path(support, Path("/unread"), {})
            contents.assert_not_called()

    def test_actual_payload_requires_a_declared_artifact_hash(self):
        support = SimpleNamespace(safe_path=lambda path: path)
        with mock.patch.object(Path, "exists", lambda path: path.name == "actual-bf16-tiles-0.bin"), \
                mock.patch.object(Path, "read_bytes") as contents:
            with self.assertRaises(ValueError):
                preparation.model_payload_path(support, Path("/unread"), {})
            contents.assert_not_called()

    def test_ambiguous_declared_names_refuse_even_if_one_file_is_missing(self):
        support = SimpleNamespace(safe_path=lambda path: path)
        with mock.patch.object(Path, "exists", lambda path: path.name == "actual-bf16-tiles-0.bin"):
            with self.assertRaises(ValueError):
                preparation.model_payload_path(support, Path("/unread"),
                                               {"actual-bf16-tiles-0.bin": "old", "actual-bf16-halves-0.bin": "new"})

    def test_foreign_backend_numerics_or_input_mode_refuse_before_model_access(self):
        root = Path("/unread")
        code = dict(zip(preparation.PRODUCER_NAMES, next(iter(preparation.ARITHMETIC_EPOCHS))))
        before = {"revision": preparation.MODEL_PIN, "tracked_source_diff_exit": 0,
                  "imported_source_sha256": {"npu_model/__init__.py": "unread"}, "curve_sha256": "unread"}
        inputs = {"input": "unread", "manifest": "unread"}
        metadata = {"code_before": code, "code_after": code.copy(), "model_binding_before": before,
                    "invocations": "one_reset_core_per_repeat_all_tiles_in_one_instruction_program",
                    "source_inputs": inputs,
                    "source_bookends": {"unchanged": True, "model_after": before, "input_before": inputs, "input_after": inputs},
                    "model_input": "new_assembler_decoded_instruction_objects",
                    "model": {"revision": preparation.MODEL_PIN, "module_file": str(root / "npu_model/__init__.py"),
                              "imported_source_sha256": before["imported_source_sha256"], "curve_sha256": "unread",
                              "backend": "curve", "numerics": "rtl"}}
        for key, value in (("backend", "foreign"), ("numerics", "foreign"), ("model_input", "foreign")):
            changed = copy.deepcopy(metadata)
            if key == "model_input":changed[key] = value
            else:changed["model"][key] = value
            with self.subTest(key=key), mock.patch("subprocess.check_output") as access:
                with self.assertRaisesRegex(ValueError, "module/curve/source"):
                    preparation.reused_arithmetic(SimpleNamespace(safe_path=lambda path: path), Path("/unread"), changed, b"", root)
                access.assert_not_called()


if __name__ == "__main__":
    unittest.main()
