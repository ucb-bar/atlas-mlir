#!/usr/bin/env python3
"""Replay selected emitted words on AtlasCore and its selected memory models.

This is a license-free component replay, not EE290 system qualification.
Only explicit selected witness members and the reviewed module closure are
read. Verilator builds its own fresh generated Makefile, never Chipyard make.
"""

import argparse
import importlib.util
import json
from pathlib import Path
import re


# The dependency bootstrap rejects prohibited spellings before lstat and
# validates every symlink target. Keep this tiny bootstrap independent of the
# scripts whose identities are about to be recorded.
def bootstrap(path):
    from collections import deque
    import os
    import stat
    path = Path(path)
    denied = lambda p: any("hammer" in x.lower() or "vlsi" in x.lower() for x in p.parts)
    if not path.is_absolute():
        path = Path.cwd() / path
    if denied(path):
        raise ValueError("restricted dependency path")
    pending, current, links = deque(path.parts[1:]), Path(path.anchor), set()
    while pending:
        part = pending.popleft()
        if part == ".": continue
        if part == "..": current = current.parent; continue
        candidate = current / part
        if denied(candidate): raise ValueError("restricted dependency path")
        if stat.S_ISLNK(candidate.lstat().st_mode):
            if candidate in links or len(links) >= 64: raise ValueError("recursive dependency symlink")
            links.add(candidate)
            target = Path(os.readlink(candidate))
            if denied(target): raise ValueError("restricted dependency symlink target")
            if target.is_absolute():
                current = Path(target.anchor)
                pending.extendleft(reversed(target.parts[1:]))
            else: pending.extendleft(reversed(target.parts))
        else: current = candidate
    return current


def load(name):
    path = bootstrap(Path(__file__).parent / name)
    spec = importlib.util.spec_from_file_location(name.replace("-", "_"), path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


OBS = load("observe-ee290-vls.py")
CAPTURE = load("ee290_build_capture.py")
FINGERPRINT = load("fingerprint-rtl-modules.py")
CHECK = OBS.CHECK


def selected_words(text):
    match = re.fullmatch(r"#define ATLAS_PROGRAM_WORDS ([0-9]+)U\s+static const uint32_t atlas_program\[\] = \{\s*((?:0x[0-9a-fA-F]{8}U,\s*)+)\};\s*", text)
    if not match:
        raise OBS.ObservationError("unsupported selected emitted-word include")
    words = re.findall(r"0x([0-9a-fA-F]{8})U", match[2])
    if len(words) != int(match[1]) or not 1 <= len(words) <= 8192:
        raise OBS.ObservationError("selected emitted-word count mismatch")
    return [int(word, 16) for word in words]


def validate_reuse(checker, report_path, expected_digest, build_inputs, argv, environment):
    """Allow a new program only when the recorded compile recipe is unchanged."""
    reused_id = checker.identity(report_path)
    if reused_id["sha256"] != expected_digest:
        raise OBS.ObservationError("reused compile receipt digest mismatch")
    reused = CHECK.strict_json(CHECK.allowed(report_path).read_bytes())
    if reused.get("schema") != "atlas.selected_atlascore_replay.v0" or reused.get("state") != "numerical_and_boundary_replay_passed":
        raise OBS.ObservationError("successful reusable AtlasCore compile required")
    checker.verify(reused["compile_phase"], report_path.parent)
    build = CHECK.strict_json(CHECK.allowed(reused["compile_phase"]["path"]).read_bytes())
    if build.get("schema") != "atlas.ee290_captured_phase.v0" or build.get("kind") != "atlascore_verilator_compile" or build.get("state") != "phase_completed" or build.get("inputs_stable") is not True or build["inputs_before"] != build["inputs_after"] or build["command"].get("returncode") != 0 or build["command"].get("timed_out") is not False:
        raise OBS.ObservationError("reused compile phase is incomplete/unstable")
    def content(item): return item["sha256"], item["bytes"]
    expected_inputs = sorted((item["role"], content(item["identity"])) for item in build_inputs)
    recorded_inputs = sorted((item["role"], content(item["identity"])) for item in build["inputs_before"])
    if recorded_inputs != expected_inputs:
        raise OBS.ObservationError("reused RTL/harness/tool inputs differ")
    for item in build["inputs_before"]:
        checker.verify(item["identity"], report_path.parent)
    # Compare every command flag and environment value, replacing only
    # fresh output paths and byte-identical snapshot input paths.
    def normalized(command, members, phase_output):
        aliases = {item["identity"]["path"]: "INPUT:" + item["role"] + ":" + item["identity"]["sha256"] for item in members}
        return [aliases.get(word, word.replace(phase_output, "{output}")) for word in command]
    recorded_output = str(Path(reused["compile_phase"]["path"]).parent)
    if normalized(build["command"]["argv"], build["inputs_before"], recorded_output) != normalized(argv, build_inputs, "{output}") or build["command"]["environment"] != environment:
        raise OBS.ObservationError("reused compilation command/environment differs")
    executable = Path(reused["compile_phase"]["path"]).parent / "obj/VAtlasCore"
    binary = checker.identity(executable)
    compiled_files = [file["identity"] for entry in build["outputs"] for file in entry["files"]]
    if binary not in compiled_files:
        raise OBS.ObservationError("reused executable is not a captured compile output")
    return reused["compile_phase"], reused_id, binary


def run(args):
    checker = CHECK.Checker()
    report_id = checker.identity(args.witness_report)
    if report_id["sha256"] != args.expected_witness_sha256:
        raise OBS.ObservationError("selected witness digest mismatch")
    witness = CHECK.strict_json(CHECK.allowed(args.witness_report).read_bytes())
    if witness.get("schema") != "atlas.ee290_vls_witness.v0" or witness.get("state") != "integrated_execution_passed" or witness.get("integrated_execution_passed") is not True:
        raise OBS.ObservationError("successful selected witness required")
    provenance = witness["provenance"]
    original = [report_id]
    for key in ("manifest", "hardware_ir", "program", "atlas_emit", "simulator"):
        original.append(checker.verify(provenance[key], args.witness_report.parent))
    manifest = CHECK.strict_json(CHECK.allowed(provenance["manifest"]["path"]).read_bytes())
    if manifest.get("state") != "verified" or manifest.get("schema") != "atlas.retained_hw_ir.v0":
        raise OBS.ObservationError("verified selected retention manifest required")
    if manifest["hardware_ir"]["sha256"] != provenance["hardware_ir"]["sha256"]:
        raise OBS.ObservationError("witness/retention hardware mismatch")
    original.append(checker.verify(witness["generated_include"], args.witness_report.parent))
    words = selected_words(CHECK.allowed(witness["generated_include"]["path"]).read_text())
    if witness.get("program_word_count") != len(words):
        raise OBS.ObservationError("witness/emitted words mismatch")
    root = CHECK.allowed(Path(__file__).parent.parent)
    table_path = CHECK.allowed(root / "lib/AtlasRTLModuleFingerprints.inc")
    table = dict(re.findall(r'\{"([A-Za-z0-9_]+)", "([0-9a-f]{64})"\}', table_path.read_text()))
    measured = FINGERPRINT.fingerprints(CHECK.allowed(provenance["hardware_ir"]["path"]).read_text(), ["AtlasCore"])
    if len(table) != 123 or measured != table:
        raise OBS.ObservationError("unreviewed selected AtlasCore semantic closure")
    original.append(checker.identity(table_path))
    sources = []
    missing = []
    for module in sorted(table):
        matches = [item for item in provenance["simulator_sources"] if Path(item["path"]).name == module + ".sv"]
        if len(matches) > 1: raise OBS.ObservationError("ambiguous selected module source")
        if matches:
            sources.append((module + ".sv", checker.verify(matches[0], args.witness_report.parent)))
        else: missing.append(module)
    memories = [item for item in provenance["simulator_sources"] if Path(item["path"]).name.endswith(".top.mems.v")]
    if len(memories) != 1:
        raise OBS.ObservationError("one selected top memory implementation required")
    memory_id = checker.verify(memories[0], args.witness_report.parent)
    memory_text = CHECK.allowed(memory_id["path"]).read_text()
    if set(missing) != {"banks_0_0_ext", "banks_0_ext", "buffer0_ext", "mem_0_ext"} or not set(missing).issubset(set(re.findall(r"^module\s+([^\s(]+)", memory_text, re.M))):
        raise OBS.ObservationError("selected external memory closure is incomplete")
    sources.append(("selected-memories.v", memory_id))
    # These two seq-memory declarations lower to inline generated modules
    # rather than hw.instance definitions in the 123-module HW closure.
    for module in ("ram_128x256", "ram_32x264"):
        matches = [item for item in provenance["simulator_sources"] if Path(item["path"]).name == module + ".sv"]
        if len(matches) != 1: raise OBS.ObservationError("missing selected inline memory module")
        sources.append((module + ".sv", checker.verify(matches[0], args.witness_report.parent)))
    for _, member in sources:
        text = CHECK.allowed(member["path"]).read_text()
        if re.search(r"`include\b|\$system\b|\$readmem[bh]\b", text):
            raise OBS.ObservationError("selected RTL has an unsupported external file/command dependency")
    tools = {name: checker.identity(getattr(args, name)) for name in ("verilator", "cxx", "make", "ar")}
    runtime = CHECK.allowed(args.verilator_root)
    if not runtime.is_dir(): raise OBS.ObservationError("explicit Verilator runtime directory required")
    output = OBS._fresh_output(args.output)
    checker.recheck()
    output.mkdir(parents=True)
    snapshots = output / "inputs"
    snapshots.mkdir()
    selected = []
    for name, member in sources:
        destination = snapshots / name
        destination.write_bytes(CHECK.allowed(member["path"]).read_bytes())
        snapshot = checker.identity(destination)
        if (snapshot["sha256"], snapshot["bytes"]) != (member["sha256"], member["bytes"]):
            raise OBS.ObservationError("source changed while snapshotting")
        selected.append({"original": member, "snapshot": snapshot})
    harness = CHECK.allowed(root / "test/ee290-atlascore-replay.cpp")
    harness_id = checker.identity(harness)
    harness_copy = snapshots / harness.name
    harness_copy.write_bytes(harness.read_bytes())
    include = snapshots / "program.hex"
    include.write_text("".join(f"{word:08x}\n" for word in words))
    snapshot_words = checker.identity(include)
    for name in ("replay-ee290-atlascore.py", "observe-ee290-vls.py", "ee290_build_capture.py", "check-ee290-provenance.py", "fingerprint-rtl-modules.py", "index-retained-hw.py"):
        source = CHECK.allowed(root / "tools" / name)
        original.append(checker.identity(source))
        (snapshots / name).write_bytes(source.read_bytes())
    (snapshots / "selected-witness.json").write_bytes(CHECK.allowed(args.witness_report).read_bytes())
    environment = {"PATH": str(CHECK.allowed(args.make).parent) + ":/bin", "LC_ALL": "C", "LANG": "C",
                   "VERILATOR_ROOT": str(runtime), "CXX": tools["cxx"]["path"], "AR": tools["ar"]["path"]}
    argv = [tools["verilator"]["path"], "--cc", "--exe", "--build", "--top-module", "AtlasCore",
            "--prefix", "VAtlasCore", "--Mdir", "{output}/obj", "--trace", "--trace-depth", "3", "--assert",
            "--output-split", "20000", "--output-split-cfuncs", "500",
            "-Wno-fatal", "-j", str(args.jobs), "-CFLAGS", "-std=c++17",
            "-MAKEFLAGS", "CXX=" + tools["cxx"]["path"] + " LINK=" + tools["cxx"]["path"] + " AR=" + tools["ar"]["path"],
            *[item["snapshot"]["path"] for item in selected], str(harness_copy)]
    build_inputs = [{"role": "tool", "identity": tools["verilator"]},
                    *[{"role": name, "identity": member} for name, member in tools.items() if name != "verilator"],
                    *[{"role": "selected_rtl", "identity": item["snapshot"]} for item in selected],
                    {"role": "harness", "identity": checker.identity(harness_copy)}]
    receipt = {"schema": "atlas.selected_atlascore_replay.v0", "target_config": "EE290SimConfig",
               "scope": "selected AtlasCore and behavioral SRAM models, directly driven host TileLink; no CPU/Chipyard system",
               "state": "prepared", "scheduling_qualified": False, "ee290_system_execution_verified": False,
               "build_linkage_complete": False, "original_inputs": original, "rtl_snapshots": selected,
               "harness": harness_id, "program_words": snapshot_words, "tool_identities": tools,
               "runtime_support": {"directory": str(runtime), "scope": "opaque installed Verilator/C++ runtime support; not a hermetic input closure"},
               "limitations": OBS.ASSUMPTIONS + ["Two-state Verilator evaluation and direct full-line TileLink host accesses differ from the EE290 CPU/system witness."]}
    def save():
        (output / "report.json").write_text(json.dumps(receipt, indent=2) + "\n")
    save()
    if args.reuse_compile:
        receipt["compile_phase"], receipt["reused_compile_from"], binary = validate_reuse(
            checker, args.reuse_compile, args.expected_reuse_sha256, build_inputs, argv, environment)
    else:
        build = CAPTURE.capture_phase(output / "compile", "atlascore_verilator_compile", argv, build_inputs,
                                      [{"role": "generated_objects", "relative_path": "obj", "kind": "tree"}],
                                      environment, args.timeout_seconds)
        receipt["compile_phase"] = checker.identity(output / "compile/phase.json")
        if build["state"] != "phase_completed":
            receipt["state"] = "compile_failed"; save(); return False
        binary = checker.identity(output / "compile/obj/VAtlasCore")
    execution = CAPTURE.capture_phase(output / "execute", "atlascore_vls_execute",
                                     [binary["path"], snapshot_words["path"], "{output}/trace.vcd", str(args.max_cycles)],
                                     [{"role": "tool", "identity": binary}, {"role": "program_words", "identity": snapshot_words}],
                                     [{"role": "waveform", "relative_path": "trace.vcd", "kind": "file"}],
                                     {"PATH": "/usr/bin:/bin", "LC_ALL": "C", "LANG": "C"}, min(args.timeout_seconds, 600))
    receipt["execution_phase"] = checker.identity(output / "execute/phase.json")
    if execution["state"] != "phase_completed":
        receipt["state"] = "execution_failed"; save(); return False
    log = CHECK.allowed(execution["command"]["stdout"]["path"]).read_text()
    observed = re.findall(r"^ATLASCORE_VLS_OBSERVED panel=([0-2]) status=5 marker=1 illegal_pc=0 polls=([0-9]+)$", log, re.M)
    passed = re.findall(r"^ATLASCORE_VLS_PANEL_PASSED panel=([0-2]) output_words=256 preserved_words=1280$", log, re.M)
    if [row[0] for row in observed] != ["0", "1", "2"] or any(int(row[1]) >= 100000 for row in observed) or passed != ["0", "1", "2"] or len(re.findall(r"^ATLASCORE_VLS_PASSED panels=3 cycles=[0-9]+$", log, re.M)) != 1:
        receipt["state"] = "numerical_record_failed"; save(); return False
    trace = CHECK.allowed(output / "execute/trace.vcd")
    receipt["trace"] = checker.identity(trace)
    try:
        with trace.open() as source:
            receipt["boundary_observations"] = OBS.validate_edges(OBS.vcd_edges(source, "TOP.AtlasCore", args.max_cycles + 10), 3)
    except OBS.ObservationError as error:
        receipt["state"], receipt["boundary_error"] = "boundary_validation_failed", str(error)
        save(); return False
    receipt["state"] = "numerical_and_boundary_replay_passed"
    checker.recheck()
    save()
    return True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--witness-report", type=Path, required=True)
    parser.add_argument("--expected-witness-sha256", required=True)
    for name in ("verilator", "cxx", "make", "ar", "verilator-root", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--jobs", type=int, choices=range(1, 5), default=4)
    parser.add_argument("--timeout-seconds", type=int, default=2400)
    parser.add_argument("--max-cycles", type=int, default=200000)
    parser.add_argument("--reuse-compile", type=Path)
    parser.add_argument("--expected-reuse-sha256")
    args = parser.parse_args()
    if bool(args.reuse_compile) != bool(args.expected_reuse_sha256):
        parser.error("--reuse-compile and --expected-reuse-sha256 must be supplied together")
    try:
        if not run(args): parser.exit(1, "AtlasCore replay failed; retained report identifies the stage\n")
    except (OBS.ObservationError, ValueError, OSError, KeyError) as error:
        parser.exit(1, "AtlasCore replay rejected: " + str(error) + "\n")


if __name__ == "__main__":
    main()
