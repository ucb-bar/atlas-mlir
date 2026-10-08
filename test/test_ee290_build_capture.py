"""Exercise the phase recorder with tiny Python commands, never a build."""

import importlib.util
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock


ROOT = Path(__file__).absolute().parents[1]
SPEC = importlib.util.spec_from_file_location("ee290_capture", ROOT / "tools/ee290_build_capture.py")
CAP = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(CAP)


class CaptureTest(unittest.TestCase):
    def setUp(self):
        scratch = CAP._CHECK.allowed(ROOT / "build/rtl-timing", missing=True)
        scratch.mkdir(parents=True, exist_ok=True)
        self.temporary = tempfile.TemporaryDirectory(prefix="ee290-capture-test-", dir=scratch)
        self.directory = Path(self.temporary.name)
        self.source = self.directory / "selected-source.txt"
        self.source.write_text("selected original bytes\n")
        checker = CAP._CHECK.Checker()
        self.inputs = [{"role": "tool", "identity": checker.identity(Path(sys.executable))},
                       {"role": "selected_source", "identity": checker.identity(self.source)}]
        self.outputs = [{"role": "compiled_fixture", "relative_path": "classes", "kind": "tree"}]
        self.script = "from pathlib import Path; import os,sys; p=Path(sys.argv[1])/'classes'; p.mkdir(); " \
                      "(p/'Fixture.class').write_bytes(b'fixture output'); print(os.environ.get('CAPTURE_SENTINEL')); " \
                      "print('inherited='+str('UNDECLARED_SENTINEL' in os.environ))"

    def tearDown(self):
        self.temporary.cleanup()

    def run_phase(self, script=None, **kwargs):
        values = dict(output_dir=self.directory / "phase", kind="selected_source_compile",
                      argv=[sys.executable, "-c", script or self.script, "{output}"],
                      inputs=self.inputs, outputs=self.outputs,
                      environment={"CAPTURE_SENTINEL": "explicit"}, timeout_seconds=5)
        values.update(kwargs)
        return CAP.capture_phase(**values)

    def preflight_reject(self, pattern, **kwargs):
        with self.assertRaisesRegex(CAP.CaptureError, pattern):
            self.run_phase(**kwargs)
        self.assertFalse((self.directory / "phase").exists())

    def test_successful_phase_has_exact_inputs_environment_logs_and_outputs(self):
        with mock.patch.dict(os.environ, {"UNDECLARED_SENTINEL": "not inherited", "JAVA_TOOL_OPTIONS": "not inherited"}):
            receipt = self.run_phase()
        self.assertEqual(receipt["schema"], "atlas.ee290_captured_phase.v0")
        self.assertEqual(receipt["target_config"], "EE290SimConfig")
        self.assertEqual(receipt["state"], "phase_completed")
        self.assertTrue(receipt["inputs_stable"])
        self.assertEqual(receipt["inputs_before"], receipt["inputs_after"])
        self.assertFalse(receipt["scheduling_qualified"])
        self.assertNotIn("build_linkage_complete", receipt)
        self.assertEqual(receipt["command"]["environment"], {"CAPTURE_SENTINEL": "explicit"})
        log = Path(receipt["command"]["stdout"]["path"]).read_text()
        self.assertIn("explicit\ninherited=False", log)
        self.assertEqual(receipt["outputs"][0]["files"][0]["relative_path"], "classes/Fixture.class")
        self.assertEqual(Path(receipt["outputs"][0]["files"][0]["identity"]["path"]).read_bytes(), b"fixture output")
        self.assertTrue((self.directory / "phase/phase.json").is_file())
        self.assertEqual(receipt["snapshots"]["producer"]["sha256"], receipt["producer"]["sha256"])
        self.assertEqual(receipt["snapshots"]["path_checker_dependency"]["sha256"], receipt["path_checker_dependency"]["sha256"])

    def test_nonzero_command_and_timeout_are_retained_failures(self):
        failed = self.run_phase(self.script + "; print('diagnostic',file=sys.stderr); sys.exit(7)")
        self.assertEqual(failed["state"], "phase_failed")
        self.assertEqual(failed["command"]["returncode"], 7)
        self.assertEqual(Path(failed["command"]["stderr"]["path"]).read_text(), "diagnostic\n")
        timed = self.run_phase("import time; print('started',flush=True); time.sleep(5)",
                               output_dir=self.directory / "timeout", timeout_seconds=0.1)
        self.assertEqual(timed["state"], "phase_failed")
        self.assertTrue(timed["command"]["timed_out"])
        self.assertNotEqual(timed["command"]["returncode"], 0)
        self.assertTrue((self.directory / "timeout/phase.json").is_file())

    def test_input_mutation_is_recorded_and_never_completed(self):
        script = self.script + "; Path(sys.argv[2]).write_text('changed selected source')"
        receipt = self.run_phase(script, argv=[sys.executable, "-c", script, "{output}", str(self.source)])
        self.assertEqual(receipt["command"]["returncode"], 0)
        self.assertEqual(receipt["state"], "phase_failed")
        self.assertFalse(receipt["inputs_stable"])
        self.assertNotEqual(receipt["inputs_before"][1]["identity"]["sha256"], receipt["inputs_after"][1]["identity"]["sha256"])

    def test_missing_empty_and_escaping_outputs_do_not_complete(self):
        missing = self.run_phase("print('successful command without output')")
        self.assertEqual(missing["state"], "phase_failed")
        self.assertEqual(missing["command"]["returncode"], 0)
        empty = self.run_phase("from pathlib import Path; import sys; (Path(sys.argv[1])/'classes').mkdir()",
                               output_dir=self.directory / "empty")
        self.assertEqual(empty["state"], "phase_failed")
        self.assertIn("empty", " ".join(empty["failures"]))
        script = "from pathlib import Path; import sys; (Path(sys.argv[1])/'alias').symlink_to(sys.argv[2])"
        escaped = self.run_phase(script, output_dir=self.directory / "escaped",
                                 argv=[sys.executable, "-c", script, "{output}", str(self.source)],
                                 outputs=[{"role": "fixture", "relative_path": "alias", "kind": "file"}])
        self.assertEqual(escaped["state"], "phase_failed")
        self.assertIn("escapes", " ".join(escaped["failures"]))

    def test_reused_directory_reserved_output_and_traversal_reject(self):
        phase = self.directory / "phase"
        phase.mkdir()
        prior = phase / "result"
        prior.write_bytes(b"original output")
        with self.assertRaisesRegex(CAP.CaptureError, "already exists"):
            self.run_phase()
        self.assertEqual(prior.read_bytes(), b"original output")
        self.assertFalse((phase / "phase.json").exists())
        for name in ("../outside", "phase.json", ".", "restricted-VLsI/output"):
            with self.assertRaisesRegex(CAP.CaptureError, "relative owned path|duplicate or escaping"):
                self.run_phase(output_dir=self.directory / "fresh",
                               outputs=[{"role": "fixture", "relative_path": name, "kind": "file"}])
        self.assertFalse((self.directory / "fresh").exists())

    def test_wrong_or_missing_tool_and_input_identity_reject_before_creation(self):
        bad = [dict(item) for item in self.inputs]
        bad[1] = {"role": "selected_source", "identity": {**self.inputs[1]["identity"], "sha256": "0" * 64}}
        self.preflight_reject("hash/size mismatch", inputs=bad)
        self.preflight_reject("selected tool input", inputs=self.inputs[1:])
        self.preflight_reject("absolute executable", argv=["python3", "-c", self.script, "{output}"])
        self.preflight_reject("selected executable tool", argv=[str(self.source), "argument"])

    def test_environment_injections_and_invalid_bounds_reject(self):
        for key in ("JAVA_TOOL_OPTIONS", "_JAVA_OPTIONS", "JDK_JAVA_OPTIONS", "CLASSPATH"):
            self.preflight_reject("injection", environment={key: "unrecorded option"})
        for timeout in (0, -1, float("inf"), float("nan"), True):
            self.preflight_reject("positive timeout", timeout_seconds=timeout)
        self.preflight_reject("string key/value", environment={"EXAMPLE": 1})

    def test_non_shell_argument_and_explicit_cwd_are_preserved(self):
        working = self.directory / "working"
        working.mkdir()
        script = self.script + "; print(sys.argv[2]); print(Path.cwd())"
        literal = "literal; $(not-executed)"
        receipt = self.run_phase(script, argv=[sys.executable, "-c", script, "{output}", literal], cwd=working)
        self.assertEqual(receipt["state"], "phase_completed")
        self.assertEqual(receipt["command"]["cwd"], str(working))
        self.assertEqual(receipt["command"]["argv"][-1], literal)
        self.assertIn(literal, Path(receipt["command"]["stdout"]["path"]).read_text())

    def test_restricted_inputs_and_symlink_targets_never_inspected(self):
        actual_lstat = Path.lstat

        def guarded(path, *args, **kwargs):
            self.assertFalse(CAP._CHECK.restricted(path), "restricted path was inspected")
            return actual_lstat(path, *args, **kwargs)

        symlink = self.directory / "permitted-input-link"
        os.symlink("hAmMeR-unread/input", symlink)
        cycle = self.directory / "cycle"
        os.symlink(cycle.name, cycle)
        with mock.patch.object(Path, "lstat", guarded):
            for path in (self.directory / "VLsi-unread/input", symlink, cycle):
                bad = [self.inputs[0], {"role": "source", "identity": {"path": str(path), "sha256": "0" * 64, "bytes": 0}}]
                self.preflight_reject("restricted|recursive", inputs=bad)

    def test_generated_restricted_name_is_rejected_before_stat(self):
        class Entries:
            def __enter__(self):
                return iter([SimpleNamespace(name="VLsI-never-created")])

            def __exit__(self, *_):
                return False

        with mock.patch.object(CAP.os, "scandir", return_value=Entries()):
            receipt = self.run_phase()
        self.assertEqual(receipt["state"], "phase_failed")
        self.assertIn("restricted generated output name", " ".join(receipt["failures"]))


if __name__ == "__main__":
    unittest.main()
