#!/usr/bin/env python3
"""Prepare/replay a selected EE290SimConfig numerical ELF with native VPD observation.

This records trace production and reuses the existing boundary decoder. It does
not implement evidence acceptance, export, or compiler qualification. Preparation
never starts VCS; execution belongs in a shell with the selected runtime/license.
"""

import argparse
from collections import deque
import importlib.util
import json
import os
from pathlib import Path
import re
import signal
import stat
import subprocess
import sys
import time


def dependency_path(path):
    """Check intermediate symlinks before importing the shared path guard."""
    denied = lambda p: any("hammer" in x.lower() or "vlsi" in x.lower() for x in p.parts)
    path = Path(path)
    if not path.is_absolute():
        path = Path.cwd() / path
    if denied(path):
        raise ValueError("restricted dependency path")
    pending, current, links = deque(path.parts[1:]), Path(path.anchor), set()
    while pending:
        part = pending.popleft()
        if part == ".":
            continue
        if part == "..":
            current = current.parent
            continue
        candidate = current / part
        if denied(candidate):
            raise ValueError("restricted dependency path")
        if stat.S_ISLNK(candidate.lstat().st_mode):
            if candidate in links or len(links) >= 64:
                raise ValueError("recursive dependency symlink")
            links.add(candidate)
            target = Path(os.readlink(candidate))
            if denied(target):
                raise ValueError("restricted dependency symlink")
            if target.is_absolute():
                current = Path(target.anchor)
                pending.extendleft(reversed(target.parts[1:]))
            else:
                pending.extendleft(reversed(target.parts))
        else:
            current = candidate
    return current


OBSERVER_PATH = dependency_path(Path(__file__).parent / "observe-ee290-vls.py")
SPEC = importlib.util.spec_from_file_location("ee290_capture_observer", OBSERVER_PATH)
OBS = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(OBS)
CHECK = OBS.CHECK
CaptureError = CHECK.ProvenanceError


def capture_signals():
    """Fixed boundary projection plus raw SRAM/competition observations.

    Supplemental signals are retained for review, not accepted as arbitration
    evidence by the existing boundary decoder.
    """
    out = {name: width for name, width in OBS.SIGNALS.values()}
    for name in ("io_tl_a_valid", "io_lsuScalarRead_valid", "io_lsuScalarWrite_valid",
                 "io_dmaRead_valid", "io_dmaWrite_valid"):
        out["vmem." + name] = 1
    for engine in ("mxu0", "mxu1", "vpu"):
        for direction in ("Read", "Write"):
            for port in (0, 1):
                out[f"mreg.io_{engine}{direction}Req{port}_valid"] = 1
    for direction in ("Read", "Write"):
        out[f"mreg.io_xlu{direction}Req_valid"] = 1
    for bank in range(6):
        for name, width in {"addr": 13, "en": 1, "clk": 1, "wmode": 1,
                            "wdata": 256, "rdata": 256, "wmask": 32}.items():
            out[f"vmem.banks_{bank}.RW0_{name}"] = width
    for bank in range(32):
        for name, width in {"R0_addr": 6, "R0_en": 1, "R0_clk": 1, "R0_data": 256,
                            "W0_addr": 6, "W0_en": 1, "W0_clk": 1, "W0_data": 256}.items():
            out[f"mreg.banks_{bank}.{name}"] = width
    return out


def write_json(path, data):
    CHECK.allowed(path, missing=True).write_text(json.dumps(data, indent=2) + "\n")


def same_bytes(a, b):
    return (a["sha256"], a["bytes"]) == (b["sha256"], b["bytes"])


def selected_json(checker, path, expected):
    member = checker.identity(path)
    if member["sha256"] != expected:
        raise CaptureError("selected receipt digest mismatch")
    return member, CHECK.strict_json(CHECK.allowed(path).read_bytes())


def tcl_word(value):
    # Every Tcl substitution metacharacter is escaped, including whitespace.
    if "\n" in value or "\r" in value or "\0" in value:
        raise CaptureError("multiline/NUL UCLI argument")
    return '"' + ''.join("\\" + c if c in '\\"$[]' else c for c in value) + '"'


def invocation(saved, simulator, host, script):
    """Retain the selected invocation's bounded plusargs; replace only paths."""
    if not isinstance(saved, list) or any(not isinstance(x, str) for x in saved):
        raise CaptureError("malformed saved simulator invocation")
    if len(saved) != 8 or saved[1] != "+permissive" or saved[-2] != "+permissive-off" or \
            saved[4:6] != ["+ntb_random_seed=1", "-no_save"] or \
            not re.fullmatch(r"\+max-cycles=[1-9][0-9]*", saved[2]):
        raise CaptureError("unsupported saved simulator invocation")
    if saved[0] != simulator["path"] or saved[3] != "+loadmem=" + host["path"] or saved[-1] != host["path"]:
        raise CaptureError("saved invocation disagrees with simulator/ELF identities")
    return [simulator["path"], *saved[1:3], "+loadmem=" + host["path"], *saved[4:6],
            "-ucli", "-do", str(script), "+permissive-off", host["path"]]


PROBE_SCRIPT = "# Bounded help probe: no simulation run command.\nhelp dump\nhelp run\nhelp quit\nputs ATLAS_UCLI_HELP_COMPLETED\nquit\n"


def capture_script_text(scope, trace):
    setup = ("set atlas_vcd_fid [dump -file " + tcl_word(str(trace)) + " -type VPD]\n"
             + "\n".join("dump -add " + scope + "." + name + " -depth 0 -fid $atlas_vcd_fid"
                         for name in capture_signals()))
    return ("# Selected EE290SimConfig projection; runtime visibility remains untested.\n"
            + "if {[catch {\n" + setup + "\n} atlas_dump_error]} {\n"
            + 'puts "ATLAS_UCLI_CAPTURE_SETUP_FAILED: $atlas_dump_error"\nquit\n}\n'
            + "puts ATLAS_UCLI_CAPTURE_SETUP_OK\nrun\ndump -flush $atlas_vcd_fid\nquit\n")


def bind_runtime_archive(phase, libraries):
    archived = {f["identity"]["path"]: f["identity"]
                for group in phase["outputs"] if group["role"] == "simulator_archive"
                for f in group["files"]}
    if len({item["path"] for item in libraries}) != len(libraries):
        raise CaptureError("duplicate selected runtime archive")
    for item in libraries:
        if item["path"] not in archived or not same_bytes(item, archived[item["path"]]):
            raise CaptureError("selected runtime archive/build output mismatch")


def prepare(witness_path, witness_sha, build_path, build_sha, scope, output, timeout=900, converter=None):
    if not 0 < timeout <= 7200:
        raise CaptureError("timeout must be positive and at most 7200 seconds")
    checker = CHECK.Checker()
    if converter is None:
        raise CaptureError("an explicit vpd2vcd converter is required")
    converter_id = checker.identity(converter)
    witness_id, witness = selected_json(checker, witness_path, witness_sha)
    phase_id, phase = selected_json(checker, build_path, build_sha)
    if witness.get("target_config") != "EE290SimConfig" or witness.get("state") != "integrated_execution_passed" or witness.get("integrated_execution_passed") is not True:
        raise CaptureError("successful selected numerical witness required")
    if phase.get("schema") != "atlas.ee290_captured_phase.v0" or phase.get("target_config") != "EE290SimConfig" or phase.get("kind") != "vcs_compile" or phase.get("state") != "phase_completed" or phase.get("inputs_stable") is not True:
        raise CaptureError("completed stable captured simulator-build phase required")
    provenance = witness["provenance"]
    simulator = checker.verify(provenance["simulator"], Path(witness_path).parent)
    builds = [f["identity"] for group in phase["outputs"] if group["role"] == "simulator" for f in group["files"]]
    if len(builds) != 1 or not same_bytes(builds[0], simulator):
        raise CaptureError("selected build/simulator mismatch")
    checker.verify(builds[0], Path(build_path).parent)
    members = {"host_binary": witness["host_binary"], "generated_include": witness["generated_include"],
               "program": provenance["program"], "host_source": provenance["host_source"]}
    for item in members.values():
        checker.verify(item, Path(witness_path).parent)
    libraries = provenance["simulator_archive_libraries"]
    if not isinstance(libraries, list) or not libraries:
        raise CaptureError("selected simulator runtime archive inventory required")
    bind_runtime_archive(phase, libraries)
    for item in libraries:
        checker.verify(item, Path(witness_path).parent)
    saved = witness["commands"][-1]
    if saved.get("returncode") != 0 or saved.get("timed_out") is not False:
        raise CaptureError("successful saved simulator command required")
    checker.verify(saved["log"], Path(witness_path).parent)
    # Validate before writing; copied ELF has identical bytes and a new path.
    invocation(saved["argv"], simulator, members["host_binary"], Path("capture.ucli"))
    output = OBS._fresh_output(output)
    checker.recheck()
    output.mkdir(parents=True)
    boundary = output / "boundary-plan"
    OBS.prepare(witness_path, witness_sha, scope, boundary)
    snapshots = {}
    for source, name in [(witness_path, "selected-witness.json"), (build_path, "selected-build-phase.json"),
                         (Path(__file__), "capture-ee290-vcs-observation.executed.py"),
                         (OBSERVER_PATH, "observe-ee290-vls.executed.py"), (OBS.DEPENDENCY, "path-guard.executed.py"),
                         (Path(converter_id["path"]), "vpd2vcd.executed"),
                         *[(Path(item["path"]), name) for name, item in members.items()]]:
        destination = output / name
        destination.write_bytes(CHECK.allowed(source).read_bytes())
        snapshots[name] = checker.identity(destination)
    host = snapshots["host_binary"]
    # The compatibility copy remains executable/readable like its selected ELF.
    CHECK.allowed(host["path"]).chmod(0o755)
    run_dir = output / "run"
    trace = run_dir / "trace.vcd"
    native_trace = run_dir / "trace.vpd"
    script = output / "capture.ucli"
    script.write_text(capture_script_text(scope, native_trace))
    probe = output / "probe.ucli"
    probe.write_text(PROBE_SCRIPT)
    # Invocation validation used the original paths above. Substitute the copy
    # only after verifying it; neither the selected simulator nor its archive is copied.
    argv = invocation(saved["argv"], simulator, members["host_binary"], script)
    argv[3], argv[-1] = "+loadmem=" + host["path"], host["path"]
    probe_argv = argv.copy()
    probe_argv[probe_argv.index("-do") + 1] = str(probe)
    plan = {"schema": "atlas.ee290_vcs_observation_capture_plan.v0", "target_config": "EE290SimConfig",
            "state": "prepared_not_executed", "scheduling_qualified": False,
            "capture_syntax_verified_by_execution": False, "signal_visibility_verified": False,
            "selected_witness": witness_id, "selected_build_phase": phase_id, "simulator": simulator,
            "simulator_archive_libraries": libraries, "selected_members": members,
            "snapshots": snapshots, "boundary_plan": checker.identity(boundary / "plan.json"),
            "capture_script": checker.identity(script), "probe_script": checker.identity(probe),
            "producer": checker.identity(Path(__file__)),
            "capture_signals": {scope + "." + name: width for name, width in capture_signals().items()},
            "supplementary_projection_validated": False,
            "invocation": {"argv": argv, "cwd": str(run_dir), "native_trace": str(native_trace), "trace": str(trace), "timeout_seconds": timeout},
            "converter": converter_id,
            "conversion_invocation": {"argv": [converter_id["path"], "-full64", str(native_trace), str(trace)],
                                      "cwd": str(output / "conversion"), "timeout_seconds": 300},
            "probe_invocation": {"argv": probe_argv, "cwd": str(output / "probe"), "timeout_seconds": 60},
            "runtime_environment_from_witness": witness.get("runtime_environment", {}),
            "limitations": ["Preparation does not start VCS or validate UCLI syntax/visibility.",
                            "The help probe checks dump flags; native VPD visibility and VPD-to-VCD conversion are checked only by the actual capture/decode.",
                            "The fixed decoder observes scalar/LSU/CSR boundaries; physical SRAM grants and competition remain outside that decoder.",
                            "This capture receipt is production linkage, not aggregate source-to-system acceptance or qualification.",
                            "License environment is inherited; runtime system dependencies are not fully pinned."]}
    checker.recheck()
    write_json(output / "plan.json", plan)
    return plan


BAD_LOG = re.compile(r"(?m)^ATLAS_UCLI_CAPTURE_SETUP_FAILED(?::|$)|EE290_VLS_FAILED|EE290_VLS_MISMATCH|assertion failed|\$fatal|fatal:|error:|Error-|UCLI-(?:[A-Z][A-Z0-9-]*|[0-9]+)|unknown command|invalid command|invalid option|unrecognized option", re.I)
LICENSE_LOG = re.compile(r"queuing for license|license checkout failed|server node is down", re.I)


def visible_projection(stream, expected):
    """Check only declared raw-signal visibility; do not infer arbitration."""
    scopes, block, found = [], [], {}
    for line in stream:
        block.extend(line.split())
        while "$end" in block:
            index = block.index("$end")
            entry, block = block[:index], block[index + 1:]
            if not entry:
                raise CaptureError("empty trace header directive")
            if entry[0] == "$scope" and len(entry) == 3:
                scopes.append(entry[2])
            elif entry[0] == "$upscope" and len(entry) == 1 and scopes:
                scopes.pop()
            elif entry[0] == "$var" and len(entry) >= 5:
                name = ".".join(scopes + [entry[4]])
                if name in expected:
                    width = int(entry[2])
                    if name in found or width != expected[name] or entry[1] not in ("reg", "wire", "logic"):
                        raise CaptureError("ambiguous/wrong-width capture signal: " + name)
                    if len(entry) > 5 and entry[5:] != [f"[{width-1}:0]"]:
                        raise CaptureError("unsupported capture signal vector range")
                    found[name] = width
            elif entry[0] == "$enddefinitions":
                if scopes or found != expected:
                    missing = sorted(set(expected) - set(found))
                    raise CaptureError("incomplete raw capture declarations: " + ", ".join(missing[:8]))
                return len(found)
    raise CaptureError("trace lacks complete VCD header")


def numerical_passed(log):
    if BAD_LOG.search(log):
        return False
    panels = re.findall(r"^EE290_VLS_PANEL_PASSED panel=(\d+) output_words=256 preserved_words=1280$", log, re.M)
    seen = re.findall(r"^EE290_VLS_OBSERVED panel=(\d+) status=5 marker=1 illegal_pc=0 observer_cycles=\d+ retired=9 polls=\d+$", log, re.M)
    return panels == ["0", "1", "2"] and seen == ["0", "1", "2"] and log.count("EE290_VLS_PASSED panels=3") == 1


def process_group_exists(pid):
    try:
        os.killpg(pid, 0)
        return True
    except ProcessLookupError:
        return False


def run_command(command, log):
    cwd = CHECK.allowed(command["cwd"], missing=True)
    cwd.mkdir()
    start = time.monotonic()
    timed_out, descendants = False, False
    with CHECK.allowed(log, missing=True).open("wb") as stream:
        child = subprocess.Popen(command["argv"], cwd=cwd, stdout=stream, stderr=subprocess.STDOUT,
                                 stdin=subprocess.DEVNULL, start_new_session=True)
        try:
            try:
                child.wait(timeout=command["timeout_seconds"])
            except subprocess.TimeoutExpired:
                timed_out = True
            if not timed_out:
                # A zero parent exit cannot leave an unbounded licensed helper.
                descendants = process_group_exists(child.pid)
        finally:
            # Also covers KeyboardInterrupt and all other BaseException paths.
            # Keep cleanup inside the open log lifetime so no owned process can
            # continue writing after its receipt identity has been recorded.
            if child.poll() is None or process_group_exists(child.pid):
                try:
                    os.killpg(child.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
            code = child.wait()
    checker = CHECK.Checker()
    return {**command, "returncode": code, "timed_out": timed_out,
            "descendants_after_parent_exit": descendants,
            "elapsed_seconds": time.monotonic() - start, "log": checker.identity(log)}


def load_plan(path, expected):
    checker = CHECK.Checker()
    member, plan = selected_json(checker, path, expected)
    if plan.get("schema") != "atlas.ee290_vcs_observation_capture_plan.v0" or plan.get("target_config") != "EE290SimConfig" or plan.get("state") != "prepared_not_executed":
        raise CaptureError("unsupported capture plan")
    for item in [plan["selected_witness"], plan["selected_build_phase"], plan["simulator"], plan["boundary_plan"],
                 plan["capture_script"], plan["probe_script"], plan["producer"], plan["converter"], *plan["selected_members"].values(),
                 *plan["snapshots"].values(), *plan["simulator_archive_libraries"]]:
        checker.verify(item, Path(path).parent)
    if not same_bytes(plan["producer"], checker.identity(Path(__file__))):
        raise CaptureError("capture producer changed since preparation")
    for dependency, snapshot in [(OBSERVER_PATH, "observe-ee290-vls.executed.py"),
                           (OBS.DEPENDENCY, "path-guard.executed.py"),
                           (Path(plan["converter"]["path"]), "vpd2vcd.executed")]:
        if not same_bytes(checker.identity(dependency), plan["snapshots"][snapshot]):
            raise CaptureError("capture execution dependency changed since preparation")
    phase = CHECK.strict_json(CHECK.allowed(plan["selected_build_phase"]["path"]).read_bytes())
    bind_runtime_archive(phase, plan["simulator_archive_libraries"])
    # Reconstruct expected commands rather than treating the plan as arbitrary argv.
    witness = CHECK.strict_json(CHECK.allowed(plan["selected_witness"]["path"]).read_bytes())
    original = invocation(witness["commands"][-1]["argv"], plan["simulator"], plan["selected_members"]["host_binary"], plan["capture_script"]["path"])
    host = plan["snapshots"]["host_binary"]
    original[3], original[-1] = "+loadmem=" + host["path"], host["path"]
    output = CHECK.allowed(path).parent
    boundary = CHECK.strict_json(CHECK.allowed(plan["boundary_plan"]["path"]).read_bytes())
    expected_signals = {boundary["atlas_scope"] + "." + name: width for name, width in capture_signals().items()}
    if plan.get("capture_signals") != expected_signals:
        raise CaptureError("capture signal projection mismatch")
    if CHECK.allowed(plan["capture_script"]["path"]).read_text() != capture_script_text(boundary["atlas_scope"], output / "run/trace.vpd") or CHECK.allowed(plan["probe_script"]["path"]).read_text() != PROBE_SCRIPT:
        raise CaptureError("capture script content mismatch")
    for key, subdir, script in [("invocation", "run", plan["capture_script"]), ("probe_invocation", "probe", plan["probe_script"])]:
        expected_argv = original.copy()
        expected_argv[expected_argv.index("-do") + 1] = script["path"]
        if plan[key]["argv"] != expected_argv or plan[key]["cwd"] != str(output / subdir):
            raise CaptureError("capture invocation/path mismatch")
        if not 0 < plan[key]["timeout_seconds"] <= (60 if key == "probe_invocation" else 7200):
            raise CaptureError("invalid capture timeout")
    if plan["invocation"]["trace"] != str(output / "run/trace.vcd") or plan["invocation"]["native_trace"] != str(output / "run/trace.vpd"):
        raise CaptureError("trace output path mismatch")
    conversion = plan["conversion_invocation"]
    if conversion["argv"] != [plan["converter"]["path"], "-full64", str(output / "run/trace.vpd"), str(output / "run/trace.vcd")] or conversion["cwd"] != str(output / "conversion") or not 0 < conversion["timeout_seconds"] <= 300:
        raise CaptureError("conversion invocation mismatch")
    # Check copied bytes are the selected witness bytes, not merely self-consistent.
    for key in plan["selected_members"]:
        if not same_bytes(plan["selected_members"][key], plan["snapshots"][key]):
            raise CaptureError("selected snapshot mismatch")
    checker.recheck()
    return checker, member, plan


def execute(plan_path, plan_sha, probe_only=False):
    checker, plan_id, plan = load_plan(plan_path, plan_sha)
    output = CHECK.allowed(plan_path).parent
    receipt_path = output / ("probe-report.json" if probe_only else "report.json")
    if receipt_path.exists() or (output / "probe").exists() or (output / "run").exists() or (output / "conversion").exists():
        raise CaptureError("capture plan already used; prepare a fresh directory")
    receipt = {"schema": "atlas.ee290_vcs_observation_capture.v0", "target_config": "EE290SimConfig",
               "state": "probe_pending", "scheduling_qualified": False, "integrated_execution_passed": False,
               "capture_production_recorded": False, "boundary_observation_passed": False,
               "inputs": {"plan": plan_id, "witness": plan["selected_witness"], "build_phase": plan["selected_build_phase"]},
               "commands": [], "runtime_environment": {k: os.environ.get(k, "") for k in ("VCS_HOME", "VCS_64", "LD_LIBRARY_PATH")},
               "license_environment_configured": bool(os.environ.get("SNPSLMD_LICENSE_FILE") or os.environ.get("LM_LICENSE_FILE")),
               "limitations": plan["limitations"]}
    write_json(receipt_path, receipt)
    probe = run_command(plan["probe_invocation"], output / "probe.log")
    receipt["commands"].append(probe)
    text = CHECK.allowed(probe["log"]["path"]).read_text(errors="replace")
    # A generic help banner or process exit alone does not establish these flags.
    supported = all(token in text for token in ("-file", "-type", "-add", "-depth", "-fid"))
    if probe["returncode"] or probe["timed_out"] or probe["descendants_after_parent_exit"] or BAD_LOG.search(text) or not supported or len(re.findall(r"^ATLAS_UCLI_HELP_COMPLETED$", text, re.M)) != 1:
        receipt["state"] = "runtime_license_unavailable" if LICENSE_LOG.search(text) else "ucli_probe_failed"
        write_json(receipt_path, receipt)
        return receipt
    receipt["state"] = "ucli_help_options_observed"
    receipt["ucli_help_options_observed"] = True
    write_json(receipt_path, receipt)
    if probe_only:
        return receipt
    checker.recheck()
    launched = run_command(plan["invocation"], output / "simulation.log")
    receipt["commands"].append(launched)
    text = CHECK.allowed(launched["log"]["path"]).read_text(errors="replace")
    passed = launched["returncode"] == 0 and not launched["timed_out"] and not launched["descendants_after_parent_exit"] and numerical_passed(text) and len(re.findall(r"^ATLAS_UCLI_CAPTURE_SETUP_OK$", text, re.M)) == 1
    receipt["integrated_execution_passed"] = passed
    receipt["observations"] = [x for x in text.splitlines() if x.startswith("EE290_VLS_")]
    native_path = CHECK.allowed(plan["invocation"]["native_trace"], missing=True)
    if native_path.is_file():
        receipt["native_trace"] = checker.identity(native_path)
        receipt["capture_production_recorded"] = True
    if not passed or "native_trace" not in receipt:
        receipt["state"] = "runtime_license_unavailable" if LICENSE_LOG.search(text) else "capture_or_numerical_execution_failed"
        checker.recheck()
        write_json(receipt_path, receipt)
        return receipt
    receipt["state"] = "numerical_execution_and_native_trace_recorded"
    write_json(receipt_path, receipt)
    converted = run_command(plan["conversion_invocation"], output / "conversion.log")
    receipt["commands"].append(converted)
    conversion_log = CHECK.allowed(converted["log"]["path"]).read_text(errors="replace")
    trace_path = CHECK.allowed(plan["invocation"]["trace"], missing=True)
    if trace_path.is_file():
        receipt["trace"] = checker.identity(trace_path)
    if converted["returncode"] or converted["timed_out"] or converted["descendants_after_parent_exit"] or BAD_LOG.search(conversion_log) or "trace" not in receipt:
        receipt["state"] = "native_trace_conversion_failed"
        checker.recheck()
        write_json(receipt_path, receipt)
        return receipt
    receipt["state"] = "numerical_execution_and_converted_trace_recorded"
    write_json(receipt_path, receipt)
    try:
        with trace_path.open() as stream:
            receipt["visible_signal_declarations"] = visible_projection(stream, plan["capture_signals"])
        receipt["signal_visibility_verified"] = True
        decoded = OBS.decode(Path(plan["boundary_plan"]["path"]), plan["boundary_plan"]["sha256"],
                             trace_path, receipt["trace"]["sha256"], output / "boundary-observation")
        receipt["boundary_observation"] = checker.identity(output / "boundary-observation/report.json")
        receipt["boundary_observation_passed"] = decoded["state"] == "bounded_boundary_events_passed"
        receipt["state"] = "numerical_and_boundary_observation_passed"
    except (ValueError, KeyError, OSError) as exc:
        receipt["state"] = "boundary_observation_failed"
        receipt["decode_error"] = str(exc)
    checker.recheck()
    write_json(receipt_path, receipt)
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    prep = sub.add_parser("prepare")
    prep.add_argument("--witness-report", type=Path, required=True)
    prep.add_argument("--expected-witness-sha256", required=True)
    prep.add_argument("--build-phase", type=Path, required=True)
    prep.add_argument("--expected-build-phase-sha256", required=True)
    prep.add_argument("--vpd2vcd", required=True, type=Path)
    prep.add_argument("--atlas-scope", required=True)
    prep.add_argument("--output", type=Path, required=True)
    prep.add_argument("--timeout-seconds", type=float, default=900)
    run = sub.add_parser("run")
    run.add_argument("--plan", type=Path, required=True)
    run.add_argument("--expected-plan-sha256", required=True)
    run.add_argument("--probe-only", action="store_true")
    args = parser.parse_args()
    try:
        if args.action == "prepare":
            result = prepare(args.witness_report, args.expected_witness_sha256, args.build_phase,
                             args.expected_build_phase_sha256, args.atlas_scope, args.output, args.timeout_seconds, args.vpd2vcd)
            print(json.dumps({"state": result["state"], "plan": str(CHECK.allowed(args.output) / "plan.json"), "scheduling_qualified": False}))
            return 0
        result = execute(args.plan, args.expected_plan_sha256, args.probe_only)
        print(json.dumps({"state": result["state"], "scheduling_qualified": False}))
        return 0 if result["state"] in ("ucli_help_options_observed", "numerical_and_boundary_observation_passed") else 1
    except (ValueError, KeyError, OSError) as exc:
        parser.exit(2, "VCS observation rejected: " + str(exc) + "\n")


if __name__ == "__main__":
    sys.exit(main())
