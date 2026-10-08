#!/usr/bin/env python3
"""Check finite selected AtlasCore SRAM-wrapper events; no qualification."""

import argparse
from collections import deque
import importlib.util
import json
import os
from pathlib import Path
import re
import stat


def bootstrap(path):
    path = Path(path)
    if not path.is_absolute(): path = Path.cwd() / path
    denied = lambda p: any("hammer" in x.lower() or "vlsi" in x.lower() for x in p.parts)
    if denied(path): raise ValueError("restricted dependency path")
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
                current = Path(target.anchor); pending.extendleft(reversed(target.parts[1:]))
            else: pending.extendleft(reversed(target.parts))
        else: current = candidate
    return current


def decoder(name):
    path = bootstrap(Path(__file__).parent / "observe-ee290-vls.py")
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


OBS = decoder("atlascore_boundary_decoder")
CHECK = OBS.CHECK


def projection(boundary):
    """Derive fixed SRAM-wrapper names from measured finite operands."""
    commands = [command for panel in boundary["panels"] for command in panel["commands"]]
    vmembanks = sorted({command["line"] >> 13 for command in commands})
    mregbanks = sorted({command["mreg"] & 31 for command in commands})
    selected = dict(OBS.SIGNALS)
    # Observe all enables to reject extra granted bank activity in windows.
    # Losing/suppressed same-bank requests are not visible in these ports.
    for bank in range(6):
        selected[f"v{bank}_en"] = (f"vmem.banks_{bank}.RW0_en", 1)
        selected[f"v{bank}_mode"] = (f"vmem.banks_{bank}.RW0_wmode", 1)
    for bank in range(32):
        selected[f"m{bank}_ren"] = (f"mreg.banks_{bank}.R0_en", 1)
        selected[f"m{bank}_wen"] = (f"mreg.banks_{bank}.W0_en", 1)
    for bank in vmembanks:
        for key, name, width in [("addr", "RW0_addr", 13), ("read", "RW0_rdata", 256),
                                 ("write", "RW0_wdata", 256), ("mask", "RW0_wmask", 32)]:
            selected[f"v{bank}_{key}"] = (f"vmem.banks_{bank}.{name}", width)
    for bank in mregbanks:
        for key, name, width in [("raddr", "R0_addr", 6), ("read", "R0_data", 256),
                                 ("waddr", "W0_addr", 6), ("write", "W0_data", 256)]:
            selected[f"m{bank}_{key}"] = (f"mreg.banks_{bank}.{name}", width)
    return selected


def physical_edges(stream, boundary, max_edges=200010):
    # Use a private decoder instance with its own fixed projection table. The
    # public boundary decoder remains unchanged, and no mapping is accepted
    # from the caller. This reuses its tested pre-timestamp VCD sampler.
    private = decoder("atlascore_private_sram_projection")
    private.SIGNALS = projection(boundary)
    return private.vcd_edges(stream, "TOP.AtlasCore", max_edges)


def validate_physical(samples, boundary):
    selected = projection(boundary)
    commands = [command for panel in boundary["panels"] for command in panel["commands"]]
    result = []
    current = 0
    for sample in samples:
        edge, values = sample["edge"], sample["values"]
        while current + 1 < len(commands) and edge >= commands[current + 1]["edge"]:
            current += 1
        command = commands[current]
        age = edge - command["edge"]
        if not 0 <= age <= 35: continue
        def value(key):
            value = values.get(key)
            if type(value) is not int or not 0 <= value < 1 << selected[key][1]:
                raise OBS.ObservationError("unknown/out-of-width physical value: " + key)
            return value
        store = command["op"] == 2
        vb, mb = command["line"] >> 13, command["mreg"] & 31
        highrow = (command["mreg"] >> 5) * 32
        read_age, write_age, response_age = 1 <= age <= 32, 3 <= age <= 34, 2 <= age <= 33
        v_en = int(write_age if store else read_age)
        m_ren, m_wen = int(store and read_age), int(not store and write_age)
        for bank in range(6):
            if value(f"v{bank}_en") != (v_en if bank == vb else 0):
                raise OBS.ObservationError("physical VMEM enable/competing bank mismatch")
        for bank in range(32):
            if value(f"m{bank}_ren") != (m_ren if bank == mb else 0) or value(f"m{bank}_wen") != (m_wen if bank == mb else 0):
                raise OBS.ObservationError("physical MREG enable/competing bank mismatch")
        if v_en:
            row = age - (3 if store else 1)
            if value(f"v{vb}_mode") != int(store) or value(f"v{vb}_addr") != (command["line"] & 8191) + row:
                raise OBS.ObservationError("physical VMEM mode/address mismatch")
            if store and (value(f"v{vb}_write") != value("vw_data") or value(f"v{vb}_mask") != 0xffffffff):
                raise OBS.ObservationError("physical VMEM write data/mask mismatch")
        if m_ren and value(f"m{mb}_raddr") != highrow + age - 1:
            raise OBS.ObservationError("physical MREG read address mismatch")
        if m_wen and (value(f"m{mb}_waddr") != highrow + age - 3 or value(f"m{mb}_write") != value("mw_data")):
            raise OBS.ObservationError("physical MREG write address/data mismatch")
        if response_age:
            physical, boundary_data = (f"m{mb}_read", "mresp_data") if store else (f"v{vb}_read", "vresp_data")
            if value(physical) != value(boundary_data):
                raise OBS.ObservationError("physical SRAM read data/boundary response mismatch")
        if age == 0:
            result.append({"command_edge": edge, "op": command["op"], "vmem_bank": vb,
                           "mreg_physical_bank": mb, "mreg_row_high_bit": command["mreg"] >> 5,
                           "physical_source_ages": [1, 32], "physical_destination_ages": [3, 34],
                           "physical_response_ages": [2, 33]})
    if len(result) != len(commands):
        raise OBS.ObservationError("missing physical command windows")
    return result


def run(report_path, expected, output):
    checker = CHECK.Checker()
    identity = checker.identity(report_path)
    if identity["sha256"] != expected:
        raise OBS.ObservationError("selected replay report digest mismatch")
    receipt = CHECK.strict_json(CHECK.allowed(report_path).read_bytes())
    if receipt.get("schema") != "atlas.selected_atlascore_replay.v0" or receipt.get("state") != "numerical_and_boundary_replay_passed":
        raise OBS.ObservationError("successful selected AtlasCore replay required")
    for key in ("compile_phase", "execution_phase", "program_words", "trace"):
        checker.verify(receipt[key], Path(report_path).parent)
    phase = CHECK.strict_json(CHECK.allowed(receipt["execution_phase"]["path"]).read_bytes())
    if phase.get("schema") != "atlas.ee290_captured_phase.v0" or phase.get("kind") != "atlascore_vls_execute" or phase.get("state") != "phase_completed" or phase.get("inputs_stable") is not True or phase["command"].get("returncode") != 0 or phase["command"].get("timed_out") is not False:
        raise OBS.ObservationError("incomplete captured execution phase")
    before, after = phase["inputs_before"], phase["inputs_after"]
    if before != after:
        raise OBS.ObservationError("captured execution input stability mismatch")
    for item in before:
        checker.verify(item["identity"], Path(report_path).parent)
    words = [item["identity"] for item in before if item["role"] == "program_words"]
    tools = [item["identity"] for item in before if item["role"] == "tool"]
    if len(words) != 1 or len(tools) != 1 or words[0] != receipt["program_words"] or phase["command"]["argv"][:3] != [tools[0]["path"], words[0]["path"], receipt["trace"]["path"]]:
        raise OBS.ObservationError("captured simulator/program/trace argv mismatch")
    files = [file["identity"] for entry in phase["outputs"] for file in entry["files"]]
    if receipt["trace"] not in files:
        raise OBS.ObservationError("trace is not a captured execution output")
    compilation = CHECK.strict_json(CHECK.allowed(receipt["compile_phase"]["path"]).read_bytes())
    if compilation.get("schema") != "atlas.ee290_captured_phase.v0" or compilation.get("kind") != "atlascore_verilator_compile" or compilation.get("state") != "phase_completed" or compilation.get("inputs_stable") is not True or compilation["inputs_before"] != compilation["inputs_after"] or compilation["command"].get("returncode") != 0 or compilation["command"].get("timed_out") is not False:
        raise OBS.ObservationError("incomplete captured compilation phase")
    for item in compilation["inputs_before"]:
        checker.verify(item["identity"], Path(report_path).parent)
    compiled_files = [file["identity"] for entry in compilation["outputs"] for file in entry["files"]]
    if tools[0] not in compiled_files:
        raise OBS.ObservationError("executed model is not a captured compilation output")
    log_id = phase["command"]["stdout"]
    checker.verify(log_id, Path(report_path).parent)
    log = CHECK.allowed(log_id["path"]).read_text()
    observations = re.findall(r"^ATLASCORE_VLS_OBSERVED panel=([0-2]) status=5 marker=1 illegal_pc=0 polls=([0-9]+)$", log, re.M)
    if [row[0] for row in observations] != ["0", "1", "2"] or any(int(row[1]) >= 100000 for row in observations) or len(re.findall(r"^ATLASCORE_VLS_PASSED panels=3 cycles=[0-9]+$", log, re.M)) != 1 or re.findall(r"^ATLASCORE_VLS_PANEL_PASSED panel=([0-2]) output_words=256 preserved_words=1280$", log, re.M) != ["0", "1", "2"]:
        raise OBS.ObservationError("captured numerical/guard records are incomplete")
    trace = CHECK.allowed(receipt["trace"]["path"])
    with trace.open() as stream:
        boundary = OBS.validate_edges(OBS.vcd_edges(stream, "TOP.AtlasCore"), 3)
    if boundary != receipt["boundary_observations"]:
        raise OBS.ObservationError("saved boundary observations differ from trace")
    with trace.open() as stream:
        physical = validate_physical(physical_edges(stream, boundary), boundary)
    checker.recheck()
    output = OBS._fresh_output(output)
    output.mkdir(parents=True)
    result = {"schema": "atlas.selected_atlascore_sram_observation.v0", "target_config": "EE290SimConfig",
              "scope": receipt["scope"], "state": "selected_sram_port_events_passed",
              "scheduling_qualified": False, "ee290_system_execution_verified": False,
              "selected_model_port_events_verified": True,
              "inputs": {"replay": identity, "trace": receipt["trace"]},
              "physical_observations": physical,
              "producer": checker.identity(Path(__file__)),
              "decoder": checker.identity(Path(__file__).parent / "observe-ee290-vls.py"),
              "limitations": ["Selected behavioral SRAM-wrapper ports and finite operands only; no physical macro/PVT timing or full EE290 CPU/system qualification.",
                               "All VMEM/MREG physical bank enable patterns match the admitted stream during command windows; losing/suppressed same-bank requests and traffic outside those windows are not excluded.",
                               "Two-state Verilator evaluation and direct full-line TileLink host accesses differ from the integrated EE290 witness."]}
    (output / "selected-replay.json").write_bytes(CHECK.allowed(report_path).read_bytes())
    (output / "check-atlascore-sram-observation.executed.py").write_bytes(CHECK.allowed(Path(__file__)).read_bytes())
    (output / "report.json").write_text(json.dumps(result, indent=2) + "\n")
    checker.recheck()
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replay-report", required=True, type=Path)
    parser.add_argument("--expected-report-sha256", required=True)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    try:
        run(args.replay_report, args.expected_report_sha256, args.output)
    except (OBS.ObservationError, ValueError, KeyError, OSError) as error:
        parser.exit(1, "SRAM observation rejected: " + str(error) + "\n")


if __name__ == "__main__":
    main()
