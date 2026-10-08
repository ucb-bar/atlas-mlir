#!/usr/bin/env python3
"""Compile reuse and emitted-word selection reject changed recorded inputs."""

import copy
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

ROOT = Path(__file__).absolute().parent.parent
SPEC = importlib.util.spec_from_file_location("replay", ROOT / "tools/replay-ee290-atlascore.py")
REPLAY = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(REPLAY)


class ReplayTests(unittest.TestCase):
    def setUp(self):
        self.scratch = tempfile.TemporaryDirectory(prefix="reuse-test-", dir=REPLAY.CHECK.allowed(ROOT / "build/rtl-timing"))
        self.base = Path(self.scratch.name)
        self.compile = self.base / "compile"
        (self.compile / "obj").mkdir(parents=True)
        def member(name, text):
            path = self.base / name
            path.write_text(text)
            return REPLAY.CHECK.Checker().identity(path)
        self.member = member
        self.tool = member("tool", "synthetic tool; never executed")
        self.rtl = member("old-rtl.sv", "module AtlasCore; endmodule")
        self.harness = member("old-harness.cpp", "synthetic harness; never compiled")
        self.binary = member("compile/obj/VAtlasCore", "synthetic output; never executed")
        self.before = [{"role": "tool", "identity": self.tool}, {"role": "selected_rtl", "identity": self.rtl},
                       {"role": "harness", "identity": self.harness}]
        self.environment = {"PATH": "/usr/bin:/bin", "LC_ALL": "C"}
        self.phase = {"schema": "atlas.ee290_captured_phase.v0", "kind": "atlascore_verilator_compile",
                      "state": "phase_completed", "inputs_stable": True,
                      "inputs_before": self.before, "inputs_after": copy.deepcopy(self.before),
                      "command": {"argv": [self.tool["path"], "--cc", "--assert", "--Mdir", str(self.compile / "obj"), self.rtl["path"], self.harness["path"]],
                                  "environment": self.environment, "returncode": 0, "timed_out": False},
                      "outputs": [{"files": [{"identity": self.binary}]}]}
        current_rtl = member("new-rtl.sv", "module AtlasCore; endmodule")
        current_harness = member("new-harness.cpp", "synthetic harness; never compiled")
        self.current = [{"role": "tool", "identity": self.tool}, {"role": "selected_rtl", "identity": current_rtl},
                        {"role": "harness", "identity": current_harness}]
        self.argv = [self.tool["path"], "--cc", "--assert", "--Mdir", "{output}/obj", current_rtl["path"], current_harness["path"]]
        self.save()

    def tearDown(self): self.scratch.cleanup()

    def save(self):
        phase = self.member("compile/phase.json", json.dumps(self.phase))
        self.receipt = self.member("report.json", json.dumps({"schema": "atlas.selected_atlascore_replay.v0",
                                                            "state": "numerical_and_boundary_replay_passed", "compile_phase": phase}))

    def check(self, digest=None):
        return REPLAY.validate_reuse(REPLAY.CHECK.Checker(), Path(self.receipt["path"]), digest or self.receipt["sha256"],
                                     self.current, self.argv, self.environment)

    def test_byte_identical_relocated_snapshots_reuse_exact_model(self):
        self.assertEqual(self.check()[2], self.binary)

    def test_changed_flags_or_environment_reject(self):
        self.argv.remove("--assert")
        with self.assertRaisesRegex(REPLAY.OBS.ObservationError, "command/environment"): self.check()
        self.argv.insert(2, "--assert")
        self.environment = {"PATH": "/usr/bin:/bin", "LC_ALL": "changed"}
        with self.assertRaisesRegex(REPLAY.OBS.ObservationError, "command/environment"): self.check()

    def test_changed_tool_harness_or_rtl_reject(self):
        for index in range(3):
            with self.subTest(role=self.current[index]["role"]):
                original = self.current[index]["identity"]
                self.current[index]["identity"] = self.member("changed" + str(index), "different bytes")
                with self.assertRaisesRegex(REPLAY.OBS.ObservationError, "inputs differ"): self.check()
                self.current[index]["identity"] = original

    def test_report_digest_and_consumed_input_mutation_reject(self):
        with self.assertRaisesRegex(REPLAY.OBS.ObservationError, "digest"): self.check("0" * 64)
        Path(self.harness["path"]).write_text("changed after recorded capture")
        with self.assertRaisesRegex(REPLAY.OBS.ObservationError, "hash/size mismatch"): self.check()

    def test_missing_or_changed_compiled_output_reject(self):
        self.phase["outputs"][0]["files"] = []
        self.save()
        with self.assertRaisesRegex(REPLAY.OBS.ObservationError, "captured compile output"): self.check()
        self.phase["outputs"][0]["files"] = [{"identity": self.binary}]
        self.save()
        Path(self.binary["path"]).write_text("different executable bytes")
        with self.assertRaisesRegex(REPLAY.OBS.ObservationError, "captured compile output"): self.check()

    def test_success_label_cannot_override_failed_or_unstable_capture(self):
        for key, value in [("returncode", 1), ("timed_out", True)]:
            with self.subTest(key=key):
                self.phase["command"][key] = value
                self.save()
                with self.assertRaisesRegex(REPLAY.OBS.ObservationError, "incomplete/unstable"): self.check()
                self.phase["command"].update(returncode=0, timed_out=False)
        self.phase["inputs_after"] = []
        self.save()
        with self.assertRaisesRegex(REPLAY.OBS.ObservationError, "incomplete/unstable"): self.check()

    def test_selected_word_shape_count_and_imem_bounds(self):
        include = "#define ATLAS_PROGRAM_WORDS 2U\nstatic const uint32_t atlas_program[] = {\n0xc1009073U,\n0x00000073U,\n};\n"
        self.assertEqual(REPLAY.selected_words(include), [0xc1009073, 0x73])
        for text in [include.replace("2U", "3U"), include.replace("0xc1009073U", "0x1c1009073U"),
                     include.replace("2U", "0U"), include.replace("uint32_t", "uint64_t")]:
            with self.assertRaises(REPLAY.OBS.ObservationError): REPLAY.selected_words(text)


if __name__ == "__main__": unittest.main()
