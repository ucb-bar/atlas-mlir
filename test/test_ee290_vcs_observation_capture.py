#!/usr/bin/env python3
"""Synthetic capture orchestration checks; no VCS license or timing evidence."""

import importlib.util
import io
import json
import os
import subprocess
import shutil
import sys
import time
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).absolute().parent.parent
SPEC = importlib.util.spec_from_file_location("capture", ROOT / "tools/capture-ee290-vcs-observation.py")
CAP = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(CAP)
FRAME_SPEC = importlib.util.spec_from_file_location("boundary_fixture", ROOT / "test/test_ee290_vls_observation.py")
FRAME = importlib.util.module_from_spec(FRAME_SPEC)
FRAME_SPEC.loader.exec_module(FRAME)


NUMERICAL = "\n".join(
    f"EE290_VLS_OBSERVED panel={i} status=5 marker=1 illegal_pc=0 observer_cycles=185 retired=9 polls=3\n"
    f"EE290_VLS_PANEL_PASSED panel={i} output_words=256 preserved_words=1280" for i in range(3)) + "\nEE290_VLS_PASSED panels=3\n"


def full_trace(omit=None):
    signals = {name: width for name, width in CAP.capture_signals().items() if name != omit}
    codes = {name: "s" + str(i) for i, name in enumerate(signals)}
    tree = {}
    for name, width in signals.items():
        branch, leaf = name.rsplit(".", 1)
        node = tree
        for component in branch.split("."):
            node = node.setdefault(component, {})
        node[leaf] = width
    lines = ["$timescale 1 ns $end", "$scope module TestDriver $end", "$scope module atlas $end"]
    def header(node, prefix=""):
        for name, value in node.items():
            if isinstance(value, dict):
                lines.append("$scope module " + name + " $end")
                header(value, prefix + name + ".")
                lines.append("$upscope $end")
            else:
                lines.append(f"$var wire {value} {codes[prefix + name]} {name} $end")
    header(tree)
    lines += ["$upscope $end", "$upscope $end", "$enddefinitions $end", "#0"]
    for name in signals:
        lines.append("b0 " + codes[name])
    frames = FRAME.frames() * 3
    for i, frame in enumerate(frames):
        stamp = i * 20 + 30
        lines.append("#" + str(stamp - 5))
        for key, value in frame["values"].items():
            name = CAP.OBS.SIGNALS[key][0]
            if key != "clock" and name in codes:
                lines.append("b" + format(value, "b") + " " + codes[name])
        lines += ["#" + str(stamp), "1" + codes["lsu.clock"],
                  "#" + str(stamp + 5), "0" + codes["lsu.clock"]]
    return "\n".join(lines) + "\n"


class CaptureTests(unittest.TestCase):
    def setUp(self):
        scratch = CAP.CHECK.allowed(ROOT / "build/rtl-timing")
        scratch.mkdir(parents=True, exist_ok=True)
        self.tmp = tempfile.TemporaryDirectory(prefix="vcs-observation-test-", dir=scratch)
        self.base = Path(self.tmp.name)
        self.checker = CAP.CHECK.Checker()

    def tearDown(self):
        self.tmp.cleanup()

    def member(self, name, content):
        path = self.base / name
        path.write_text(content)
        return CAP.CHECK.Checker().identity(path)

    def fixture(self, mode="pass", omit=None):
        trace = self.member("synthetic.vcd", full_trace(omit))
        simulator_path = self.base / "simv"
        simulator_path.write_text("#!/usr/bin/env python3\nimport pathlib,re,sys,time,subprocess\n"
            + "script=pathlib.Path(sys.argv[sys.argv.index('-do')+1])\n"
            + f"mode={mode!r}\n"
            + "if script.name=='probe.ucli':\n"
            + " print('queuing for license' if mode=='license' else '-file -type -add -depth -fid'+('\\nucli% puts ATLAS_UCLI_HELP_COMPLETED' if mode=='missing-probe-sentinel' else '\\nATLAS_UCLI_HELP_COMPLETED'))\n"
            + " sys.exit(1 if mode=='license' else 0)\n"
            + "if mode=='timeout': time.sleep(5)\n"
            + "if mode!='missing':\n"
            + " destination=re.search(r'dump -file \"([^\"]+)\"',script.read_text())[1]\n"
            + f" pathlib.Path(destination).write_bytes(pathlib.Path({trace['path']!r}).read_bytes())\n"
            + "print('ucli% '+script.read_text())\n"
            + "print('ATLAS_UCLI_CAPTURE_SETUP_OK')\n"
            + f"print({NUMERICAL!r})\n"
            + "if mode=='error': print('Error-UCLI: hidden signal')\n"
            + "if mode=='numeric-error': print('UCLI-001: VCD backend failure')\n"
            + "if mode=='orphan':\n"
            + " child=subprocess.Popen([sys.executable,'-c','import time;time.sleep(60)'],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)\n"
            + " print('OWNED_DESCENDANT_PID='+str(child.pid),flush=True)\n")
        simulator_path.chmod(0o755)
        simulator = self.checker.identity(simulator_path)
        hardware = self.member("hw.mlir", "hardware")
        manifest = self.member("manifest.json", json.dumps({"schema": "atlas.retained_hw_ir.v0", "state": "verified", "hardware_ir": hardware}))
        sources = []
        for instance, module in CAP.OBS.MODULES.items():
            names = [signal.split(".")[1] for signal, _ in CAP.OBS.SIGNALS.values() if signal.startswith(instance + ".")]
            sources.append(self.member(module + ".sv", "module " + module + ";\nwire " + ",".join(names) + ";\nendmodule\n"))
        host = self.member("host.riscv", "selected ELF bytes")
        members = {"program": self.member("program.mlir", "program"),
                   "host_source": self.member("host.c", "host"),
                   "generated_include": self.member("atlas_program.inc", "words")}
        saved = [simulator["path"], "+permissive", "+max-cycles=2000000", "+loadmem=" + host["path"],
                 "+ntb_random_seed=1", "-no_save", "+permissive-off", host["path"]]
        archive = self.member("archive.so", "runtime")
        witness = self.member("witness.json", json.dumps({"schema": "atlas.ee290_vls_witness.v0", "target_config": "EE290SimConfig",
            "state": "integrated_execution_passed", "integrated_execution_passed": True,
            "host_binary": host, "generated_include": members["generated_include"],
            "provenance": {"manifest": manifest, "hardware_ir": hardware, "simulator": simulator,
                           "program": members["program"], "host_source": members["host_source"],
                           "simulator_sources": sources, "simulator_archive_libraries": [archive]},
            "commands": [{"argv": saved, "returncode": 0, "timed_out": False, "log": self.member("old.log", NUMERICAL)}]}))
        phase = self.member("phase.json", json.dumps({"schema": "atlas.ee290_captured_phase.v0", "target_config": "EE290SimConfig",
            "kind": "vcs_compile", "state": "phase_completed", "inputs_stable": True,
            "outputs": [{"role": "simulator", "files": [{"identity": simulator}]},
                        {"role": "simulator_archive", "files": [{"identity": archive}]}]}))
        converter_path = self.base / "vpd2vcd"
        converter_path.write_text("#!/usr/bin/env python3\nimport pathlib,sys\n"
                                 + f"mode={mode!r}\n"
                                 + "if mode=='conversion-error': sys.exit(1)\n"
                                 + "if mode!='missing-converted': pathlib.Path(sys.argv[-1]).write_bytes(pathlib.Path(sys.argv[-2]).read_bytes())\n")
        converter_path.chmod(0o755)
        self.converter = converter_path
        return witness, phase

    def plan(self, mode="pass", omit=None, timeout=30):
        witness, phase = self.fixture(mode, omit)
        output = self.base / "capture"
        CAP.prepare(witness["path"], witness["sha256"], phase["path"], phase["sha256"], "TestDriver.atlas", output, timeout, self.converter)
        return self.checker.identity(output / "plan.json"), output

    def test_complete_fake_capture_reuses_bytes_and_never_qualifies(self):
        plan_id, output = self.plan()
        plan = json.loads(Path(plan_id["path"]).read_text())
        self.assertEqual(len(plan["capture_signals"]), 356)
        script = (output / "capture.ucli").read_text()
        self.assertEqual(script.count("-fid $atlas_vcd_fid"), 356)
        self.assertIn("set atlas_vcd_fid [dump -file", script)
        self.assertIn("-type VPD", script)
        self.assertNotIn("-type VCD", script)
        self.assertIn("dump -flush $atlas_vcd_fid", script)
        self.assertFalse((output / "simulation.log").exists())
        result = CAP.execute(plan_id["path"], plan_id["sha256"])
        self.assertEqual(result["state"], "numerical_and_boundary_observation_passed")
        self.assertEqual(result["visible_signal_declarations"], 356)
        self.assertTrue(result["capture_production_recorded"])
        self.assertEqual(result["commands"][-1]["argv"][1], "-full64")
        self.assertEqual(len(result["commands"]), 3)
        self.assertIn("native_trace", result)
        self.assertIn("trace", result)
        self.assertFalse(result["scheduling_qualified"])
        boundary = json.loads((output / "boundary-observation/report.json").read_text())
        self.assertFalse(boundary["physical_sram_arbitration_verified"])
        self.assertFalse(boundary["capture_execution_link_verified"])
        self.assertEqual(len(boundary["observations"]["panels"]), 3)
        with self.assertRaisesRegex(CAP.CaptureError, "already used"):
            CAP.execute(plan_id["path"], plan_id["sha256"])

    def test_license_probe_stops_without_simulation(self):
        plan_id, output = self.plan("license")
        result = CAP.execute(plan_id["path"], plan_id["sha256"])
        self.assertEqual(result["state"], "runtime_license_unavailable")
        self.assertEqual(len(result["commands"]), 1)
        self.assertFalse((output / "run").exists())

    def test_echoed_probe_puts_cannot_replace_emitted_sentinel(self):
        plan_id, output = self.plan("missing-probe-sentinel")
        result = CAP.execute(plan_id["path"], plan_id["sha256"])
        self.assertEqual(result["state"], "ucli_probe_failed")
        self.assertFalse((output / "run").exists())

    def test_missing_trace_runtime_error_and_timeout_do_not_pass(self):
        for mode in ("missing", "error", "numeric-error", "timeout"):
            with self.subTest(mode=mode):
                # A fresh fixture subdirectory prevents state collision.
                old = self.base
                self.base = old / mode
                self.base.mkdir()
                try:
                    plan_id, _ = self.plan(mode, timeout=0.03 if mode == "timeout" else 30)
                    result = CAP.execute(plan_id["path"], plan_id["sha256"])
                    self.assertEqual(result["state"], "capture_or_numerical_execution_failed")
                    self.assertFalse(result["boundary_observation_passed"])
                finally:
                    self.base = old

    def test_missing_physical_projection_is_retained_but_rejected(self):
        plan_id, output = self.plan(omit="vmem.banks_5.RW0_en")
        result = CAP.execute(plan_id["path"], plan_id["sha256"])
        self.assertEqual(result["state"], "boundary_observation_failed")
        self.assertIn("vmem.banks_5.RW0_en", result["decode_error"])
        self.assertTrue(result["capture_production_recorded"])
        self.assertTrue(result["integrated_execution_passed"])
        self.assertTrue((output / "run/trace.vpd").exists())
        self.assertTrue((output / "run/trace.vcd").exists())

    def test_mutated_snapshot_and_invocation_are_rejected_before_launch(self):
        plan_id, output = self.plan()
        (output / "host_binary").write_text("substituted ELF")
        with self.assertRaises(CAP.CaptureError):
            CAP.execute(plan_id["path"], plan_id["sha256"])
        self.assertFalse((output / "probe").exists())
        (output / "host_binary").write_text("selected ELF bytes")
        plan = json.loads(Path(plan_id["path"]).read_text())
        plan["invocation"]["argv"][0] = "/bin/true"
        Path(plan_id["path"]).write_text(json.dumps(plan))
        changed = CAP.CHECK.Checker().identity(plan_id["path"])
        with self.assertRaisesRegex(CAP.CaptureError, "invocation"):
            CAP.execute(changed["path"], changed["sha256"])
        self.assertFalse((output / "probe").exists())

    def test_wrong_selected_build_identity_writes_nothing(self):
        witness, phase = self.fixture()
        data = json.loads(Path(phase["path"]).read_text())
        data["outputs"][0]["files"][0]["identity"] = self.member("other-simv", "different simulator")
        phase = self.member("phase.json", json.dumps(data))
        with self.assertRaisesRegex(CAP.CaptureError, "build/simulator mismatch"):
            CAP.prepare(witness["path"], witness["sha256"], phase["path"], phase["sha256"], "TestDriver.atlas", self.base / "bad", converter=self.converter)
        self.assertFalse((self.base / "bad").exists())

    def test_parent_exit_descendant_fails_and_owned_child_is_killed(self):
        plan_id, output = self.plan("orphan")
        result = CAP.execute(plan_id["path"], plan_id["sha256"])
        self.assertEqual(result["state"], "capture_or_numerical_execution_failed")
        self.assertTrue(result["commands"][-1]["descendants_after_parent_exit"])
        log = (output / "simulation.log").read_text()
        pid = int(log.split("OWNED_DESCENDANT_PID=")[1].splitlines()[0])
        # An orphan may briefly remain a zombie until its new parent reaps it;
        # neither a missing process nor a zombie can continue licensed work.
        state_path = Path(f"/proc/{pid}/stat")
        deadline = time.monotonic() + 1
        while state_path.exists() and state_path.read_text().split(") ", 1)[1].split()[0] != "Z":
            if time.monotonic() > deadline:
                self.fail("owned descendant still running after cleanup")
            time.sleep(0.01)

    def test_interrupt_terminates_real_owned_process(self):
        real_popen = subprocess.Popen
        children = []
        def interrupted_popen(*args, **kwargs):
            child = real_popen(*args, **kwargs)
            children.append(child)
            real_wait = child.wait
            first = True
            def interrupted_wait(*wait_args, **wait_kwargs):
                nonlocal first
                if first:
                    first = False
                    raise KeyboardInterrupt()
                return real_wait(*wait_args, **wait_kwargs)
            child.wait = interrupted_wait
            return child
        with patch.object(CAP.subprocess, "Popen", side_effect=interrupted_popen):
            with self.assertRaises(KeyboardInterrupt):
                CAP.run_command({"argv": [sys.executable, "-c", "import time;time.sleep(60)"],
                                 "cwd": str(self.base / "interrupt"), "timeout_seconds": 10}, self.base / "interrupt.log")
        self.assertEqual(children[0].returncode, -9)
        with self.assertRaises(ProcessLookupError):
            os.killpg(children[0].pid, 0)

    def test_real_tcl_script_reproduces_vcd_backend_failure_and_selects_vpd(self):
        tclsh = shutil.which("tclsh")
        if tclsh is None:
            self.skipTest("Tcl interpreter unavailable")
        tclsh = str(CAP.CHECK.allowed(tclsh))
        # Model the observed VCS backend contract, not its permissive Tcl
        # front end: VCD opening errors, VPD opening returns an explicit fid.
        stub = """proc dump {args} {
    if {[lindex $args 0] eq "-file"} {
        set type [lindex $args [expr {[lsearch -exact $args "-type"] + 1}]]
        if {$type ne "VPD"} {error {Error-[UCLI-DUMP-UNSUPP-FORMAT] Unsupported dump format}}
        return VPD0
    }
    if {[lindex $args 0] eq "-add"} {
        set fid [lindex $args [expr {[lsearch -exact $args "-fid"] + 1}]]
        if {$fid ne "VPD0"} {error {missing explicit VPD file ID}}
    }
}
proc run {} {puts ATLAS_TEST_RUN_EXECUTED}
proc quit {} {exit 0}
"""
        script = CAP.capture_script_text("TestDriver.atlas", self.base / "trace.vpd")
        rejected = subprocess.run([tclsh], input=stub + script.replace("-type VPD]", "-type VCD]"),
                                  capture_output=True, text=True, timeout=10)
        self.assertEqual(rejected.returncode, 0)
        self.assertIn("ATLAS_UCLI_CAPTURE_SETUP_FAILED", rejected.stdout)
        self.assertIn("UCLI-DUMP-UNSUPP-FORMAT", rejected.stdout)
        self.assertNotIn("ATLAS_TEST_RUN_EXECUTED", rejected.stdout)
        self.assertIsNotNone(CAP.BAD_LOG.search(rejected.stdout))
        accepted = subprocess.run([tclsh], input=stub + script, capture_output=True, text=True, timeout=10)
        self.assertEqual(accepted.returncode, 0)
        self.assertIn("ATLAS_UCLI_CAPTURE_SETUP_OK", accepted.stdout)
        self.assertIn("ATLAS_TEST_RUN_EXECUTED", accepted.stdout)
        self.assertIsNone(CAP.BAD_LOG.search(accepted.stdout))

    def test_conversion_failure_preserves_native_trace_and_rejects_result(self):
        for mode in ("conversion-error", "missing-converted"):
            with self.subTest(mode=mode):
                old = self.base
                self.base = old / mode
                self.base.mkdir()
                try:
                    plan_id, output = self.plan(mode)
                    result = CAP.execute(plan_id["path"], plan_id["sha256"])
                    self.assertEqual(result["state"], "native_trace_conversion_failed")
                    self.assertTrue(result["integrated_execution_passed"])
                    self.assertTrue(result["capture_production_recorded"])
                    self.assertFalse(result["boundary_observation_passed"])
                    self.assertTrue((output / "run/trace.vpd").exists())
                finally:
                    self.base = old

    def test_execution_dependency_swap_is_rejected_before_launch(self):
        plan_id, output = self.plan()
        alternative = Path(self.member("alternative-dependency.py", "different implementation")["path"])
        for owner, name in [(CAP, "OBSERVER_PATH"), (CAP.OBS, "DEPENDENCY")]:
            with self.subTest(dependency=name), patch.object(owner, name, alternative):
                with self.assertRaisesRegex(CAP.CaptureError, "execution dependency changed"):
                    CAP.execute(plan_id["path"], plan_id["sha256"])
                self.assertFalse((output / "probe").exists())

    def test_restricted_path_precedes_access_and_tcl_escaping(self):
        with patch.object(Path, "lstat", side_effect=AssertionError("unexpected access")):
            with self.assertRaises(ValueError):
                CAP.dependency_path("/tmp/prohibited-hammer-name/file")
        self.assertEqual(CAP.tcl_word('/tmp/a $b[0]"\\'), '"/tmp/a \\$b\\[0\\]\\"\\\\"')
        self.assertFalse(CAP.numerical_passed(NUMERICAL.replace("panel=2", "panel=1")))
        with self.assertRaises(CAP.CaptureError):
            CAP.tcl_word("a\nb")
        with self.assertRaisesRegex(CAP.CaptureError, "wrong-width"):
            CAP.visible_projection(io.StringIO(full_trace().replace("$var wire 13", "$var wire 12", 1)),
                                   {"TestDriver.atlas." + k: v for k, v in CAP.capture_signals().items()})


if __name__ == "__main__":
    unittest.main()
