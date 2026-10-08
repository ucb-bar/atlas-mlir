#!/usr/bin/env python3
"""Decode a selected EE290SimConfig VCD; never promote it to qualification.

The fixed signal projection is an authored, reviewed LSU/frontend boundary
mapping. It does not establish physical SRAM arbitration or build provenance.
"""

import argparse
from collections import deque
import importlib.util
import json
import os
from pathlib import Path
import re
import stat
import subprocess


def _dependency_path(path):
    # Guard the guard's own import, including every intermediate symlink.
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


DEPENDENCY = _dependency_path(Path(__file__).parent / "check-ee290-provenance.py")
_spec = importlib.util.spec_from_file_location("ee290_observation_guard", DEPENDENCY)
CHECK = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(CHECK)
ObservationError = CHECK.ProvenanceError

# key: (module-relative generated signal, VCD width). No user-authored remap.
SIGNALS = {
    "clock": ("lsu.clock", 1), "reset": ("lsu.reset", 1),
    "fire": ("scalar.s1_fire", 1), "launch": ("scalar.is_lsu_launch", 1),
    "cmd": ("lsu.io_cmd_valid", 1), "op": ("lsu.io_cmd_bits_op", 2),
    "mreg": ("lsu.io_cmd_bits_mregBank", 6),
    "line": ("lsu.io_cmd_bits_vmemLineAddr", 16),
    "load_state": ("lsu.vloadState", 2), "store_state": ("lsu.vstoreState", 2),
    "load_busy": ("lsu.io_vloadBusy", 1), "store_busy": ("lsu.io_vstoreBusy", 1),
    "vr": ("lsu.io_vmemVecRead_valid", 1),
    "vr_bank": ("lsu.io_vmemVecRead_bits_bankIdx", 3),
    "vr_addr": ("lsu.io_vmemVecRead_bits_bankAddr", 13),
    "vresp": ("lsu.io_vmemVecReadData_valid", 1),
    "vresp_data": ("lsu.io_vmemVecReadData_bits", 256),
    "vw": ("lsu.io_vmemVecWrite_valid", 1),
    "vw_bank": ("lsu.io_vmemVecWrite_bits_bankIdx", 3),
    "vw_addr": ("lsu.io_vmemVecWrite_bits_bankAddr", 13),
    "vw_data": ("lsu.io_vmemVecWrite_bits_data", 256),
    "mr": ("lsu.io_mregReadReq_valid", 1),
    "mr_id": ("lsu.io_mregReadReq_bits_mregId", 6),
    "mr_row": ("lsu.io_mregReadReq_bits_row", 5),
    "mresp": ("lsu.io_mregReadResp_valid", 1),
    "mresp_data": ("lsu.io_mregReadResp_bits", 256),
    "mw": ("lsu.io_mregWriteReq_valid", 1),
    "mw_id": ("lsu.io_mregWriteReq_bits_mregId", 6),
    "mw_row": ("lsu.io_mregWriteReq_bits_row", 5),
    "mw_data": ("lsu.io_mregWriteReq_bits_data", 256),
    "csr": ("csrfile.io_csr_valid", 1), "csr_addr": ("csrfile.io_csr_addr", 12),
    "csr_op": ("csrfile.io_csr_op", 3), "csr_data": ("csrfile.io_csr_wdata", 32),
    "marker": ("csrfile.reg_dbg0", 32),
    "halt": ("csrfile.io_csr_halted", 1), "ecall": ("csrfile.io_csr_set_ecall", 1),
    "illegal": ("csrfile.io_csr_set_illegal", 1), "ebreak": ("csrfile.io_csr_set_ebreak", 1),
}
MODULES = {"lsu": "LSU", "scalar": "ScalarCore", "csrfile": "CSRFile_1"}
CONTROLS = ("fire", "launch", "cmd", "load_busy", "store_busy", "vr", "vresp",
            "vw", "mr", "mresp", "mw", "csr", "halt", "ecall", "illegal", "ebreak")
PORTS = ("vr", "vresp", "mw", "mr", "mresp", "vw")
ASSUMPTIONS = [
    "Fixed signal mapping and LSU/frontend semantics were authored from selected RTL; hashes alone do not establish those semantics.",
    "Boundary read-response valid/data is observed; physical SRAM grants, competing engines, host traffic and blackbox internals are not validated.",
    "Signals consumed at an edge have settled before its VCD timestamp; delta-cycle ordering within one timestamp is not used.",
    "Numerical/guard checks and captured build linkage require separately validated execution receipts.",
    "Only serialized aligned 32-row VLOAD then VSTORE panels are admitted; other instructions and overlap domains remain outside this validator.",
]


def _integer(value, name):
    if not isinstance(value, int) or isinstance(value, bool):
        raise ObservationError("unknown or malformed sampled value: " + name)
    width = SIGNALS[name][1]
    if not 0 <= value < 1 << width:
        raise ObservationError("out-of-width sampled value: " + name)
    return value


def vcd_edges(stream, scope, max_edges=3000000):
    """Yield pre-timestamp samples on rising clocks, independent of VCD order.

    Supports ordinary scalar/binary VCD with whole-vector declarations only.
    Dump suppression, real values, duplicate declarations and backward/repeated
    timestamps fail closed. A missing final newline is not a truncation proof;
    the validator separately requires complete release/marker/halt endpoints.
    """
    wanted = {scope + "." + signal: key for key, (signal, _) in SIGNALS.items()}
    hierarchy, codes, declarations, values = [], {}, {}, {}
    header, block, stamp, changes, edge = True, [], None, {}, 0
    timescale = None

    def flush():
        nonlocal edge
        before = values.copy()
        values.update(changes)
        clock = declarations["clock"]
        old, new = before.get(clock), values.get(clock)
        if old == 0 and new == 1:
            edge += 1
            if edge > max_edges:
                raise ObservationError("VCD edge limit exceeded")
            return {"edge": edge, "time": stamp, "values": {
                key: before.get(code) for key, code in declarations.items()}}
        return None

    for line in stream:
        tokens = line.split()
        if header:
            block.extend(tokens)
            while "$end" in block:
                end = block.index("$end")
                entry, block = block[:end], block[end + 1:]
                if not entry:
                    raise ObservationError("empty VCD header directive")
                directive = entry[0]
                if directive == "$scope" and len(entry) == 3:
                    hierarchy.append(entry[2])
                elif directive == "$upscope" and len(entry) == 1 and hierarchy:
                    hierarchy.pop()
                elif directive == "$var" and len(entry) >= 5:
                    try:
                        width = int(entry[2])
                    except ValueError as exc:
                        raise ObservationError("invalid VCD width") from exc
                    full = ".".join(hierarchy + [entry[4]])
                    if full in wanted:
                        key = wanted[full]
                        if key in declarations or width != SIGNALS[key][1] or \
                                entry[1] not in ("wire", "reg", "logic"):
                            raise ObservationError("ambiguous or incompatible VCD signal: " + key)
                        if len(entry) > 5 and entry[5:] != [f"[{width-1}:0]"]:
                            raise ObservationError("unsupported selected vector range: " + key)
                        code = entry[3]
                        if code in codes and codes[code] != width:
                            raise ObservationError("VCD alias width mismatch")
                        declarations[key], codes[code] = code, width
                elif directive == "$timescale":
                    if timescale is not None:
                        raise ObservationError("duplicate VCD timescale")
                    timescale = "".join(entry[1:])
                    if not re.fullmatch(r"(?:1|10|100)(?:s|ms|us|ns|ps|fs)", timescale):
                        raise ObservationError("unsupported VCD timescale")
                elif directive == "$enddefinitions":
                    if hierarchy or set(declarations) != set(SIGNALS) or timescale is None:
                        raise ObservationError("incomplete selected VCD declarations/timescale")
                    header = False
                elif directive not in ("$comment", "$date", "$version"):
                    raise ObservationError("unsupported VCD header directive")
            if not header and block:
                raise ObservationError("values must follow VCD header on a new line")
            continue
        i = 0
        while i < len(tokens):
            token = tokens[i]
            i += 1
            if token.startswith("#"):
                if not re.fullmatch(r"#[0-9]+", token):
                    raise ObservationError("invalid VCD timestamp")
                new_stamp = int(token[1:])
                if stamp is not None:
                    if new_stamp <= stamp:
                        raise ObservationError("non-increasing VCD timestamp")
                    sample = flush()
                    if sample is not None:
                        yield sample
                stamp, changes = new_stamp, {}
            elif token in ("$dumpvars", "$end"):
                continue
            elif token.startswith("$"):
                raise ObservationError("unsupported/suppressed VCD values")
            else:
                if stamp is None:
                    stamp = 0
                if token[0].lower() == "b":
                    if i >= len(tokens):
                        raise ObservationError("truncated VCD vector")
                    bits, code = token[1:].lower(), tokens[i]
                    i += 1
                elif token[0].lower() in "01xz":
                    bits, code = token[0].lower(), token[1:]
                else:
                    raise ObservationError("unsupported VCD value")
                if not code or not re.fullmatch(r"[01xz]+", bits):
                    raise ObservationError("malformed VCD value")
                if code in codes:
                    if len(bits) > codes[code]:
                        raise ObservationError("oversized selected VCD value")
                    if code in changes and code == declarations["clock"]:
                        raise ObservationError("multiple clock changes at one timestamp")
                    changes[code] = None if "x" in bits or "z" in bits else int(bits, 2)
    if header or block or stamp is None:
        raise ObservationError("truncated VCD header or empty trace")
    sample = flush()
    if sample is not None:
        yield sample


def validate_edges(samples, expected_panels=3):
    """Validate finite boundary events, preserving measured edges and operands."""
    if not isinstance(expected_panels, int) or isinstance(expected_panels, bool) or not 1 <= expected_panels <= 32:
        raise ObservationError("expected panel count must be in 1..32")
    active, panel, panels, commands, endpoints = None, [], [], [], []
    pending_marker, marker_edge, last_edge, last_command, was_halted = None, None, 0, None, False
    for sample in samples:
        edge, values = sample["edge"], sample["values"]
        if not isinstance(edge, int) or edge != last_edge + 1:
            raise ObservationError("sample edge sequence is incomplete")
        last_edge = edge
        v = lambda key: _integer(values.get(key), key)
        if v("reset"):
            if active or panel or pending_marker is not None:
                raise ObservationError("reset interrupted an observed panel")
            was_halted = False
            continue
        for key in CONTROLS:
            v(key)
        if v("illegal") or v("ebreak"):
            raise ObservationError("illegal instruction or EBREAK endpoint is unsupported")
        if v("launch") != v("cmd") or v("cmd") and not v("fire"):
            raise ObservationError("scalar issue/LSU command crosslink mismatch")
        if pending_marker is not None:
            if edge != pending_marker + 1 or v("marker") != 1:
                raise ObservationError("DBG0 publication did not follow marker write")
            endpoints.append({"kind": "marker_publication", "edge": edge})
            pending_marker = None
        expected = {key: 0 for key in PORTS}
        load_busy = store_busy = 0
        if active:
            age = edge - active["edge"]
            if age > 35:
                raise ObservationError("missing engine release sample")
            store = active["op"] == 2
            load_busy, store_busy = (0, 1) if store else (1, 0)
            if age == 35:
                load_busy = store_busy = 0
                active["release_edge"] = edge
                active = None
            else:
                source, response, destination = ("mr", "mresp", "vw") if store else ("vr", "vresp", "mw")
                expected[source] = int(1 <= age <= 32)
                expected[response] = int(2 <= age <= 33)
                expected[destination] = int(3 <= age <= 34)
                if expected[source]:
                    row = age - 1
                    if store:
                        operands = (v("mr_id"), v("mr_row"))
                        required = (active["mreg"], row)
                    else:
                        operands = (v("vr_bank"), v("vr_addr"))
                        required = (active["line"] >> 13, (active["line"] & 8191) + row)
                    if operands != required:
                        raise ObservationError("source row/address stream mismatch")
                    active["source_edges"].append(edge)
                # Destination consumes the preceding response, before recording
                # this edge's response (both can be valid in the same cycle).
                if expected[destination]:
                    row = age - 3
                    if store:
                        operands = (v("vw_bank"), v("vw_addr"))
                        required = (active["line"] >> 13, (active["line"] & 8191) + row)
                    else:
                        operands = (v("mw_id"), v("mw_row"))
                        required = (active["mreg"], row)
                    if operands != required or v(destination + "_data") != active.get("response_data"):
                        raise ObservationError("destination row/data stream mismatch")
                    active["destination_edges"].append(edge)
                    if not store:
                        active.setdefault("loaded_rows", []).append(v("mw_data"))
                if expected[response]:
                    response_data = v(response + "_data")
                    if store and response_data != panel[0]["loaded_rows"][age - 2]:
                        raise ObservationError("stored MREG response differs from the loaded row")
                    active["response_data"] = response_data
                    active["response_edges"].append(edge)
        for key in PORTS:
            if v(key) != expected[key]:
                raise ObservationError("unexpected or missing boundary event: " + key)
        if (v("load_busy"), v("store_busy")) != (load_busy, store_busy):
            raise ObservationError("busy interval/release mismatch")
        if v("cmd"):
            op, line, mreg = v("op"), v("line"), v("mreg")
            if active or op not in (1, 2) or v("load_state") or v("store_state"):
                raise ObservationError("LSU command was not accepted in the idle state")
            if line % 32 or line > 49120:
                raise ObservationError("unsupported aligned VMEM tile")
            if last_command is not None and edge - last_command < 35:
                raise ObservationError("VLS issue gap below conservative 35")
            if v("halt") or v("marker") or marker_edge is not None:
                raise ObservationError("panel command lacks a clean marker/halt entry")
            if (len(panel) == 0 and op != 1) or (len(panel) == 1 and (op != 2 or mreg != panel[0]["mreg"])) or len(panel) > 1:
                raise ObservationError("expected one VLOAD then same-MREG VSTORE per panel")
            if panel and abs(line - panel[0]["line"]) < 32:
                raise ObservationError("source and destination tiles overlap")
            active = {"edge": edge, "time": sample["time"], "op": op, "line": line,
                      "mreg": mreg, "source_edges": [], "response_edges": [], "destination_edges": []}
            panel.append(active)
            commands.append(active)
            last_command = edge
            was_halted = False
        if v("csr"):
            if not v("fire") or (v("csr_addr"), v("csr_op"), v("csr_data")) != (0xC10, 1, 1):
                raise ObservationError("unsupported scalar CSR event")
            if active or len(panel) != 2 or marker_edge is not None or v("marker") != 0:
                raise ObservationError("marker write preceded drain or duplicated")
            marker_edge, pending_marker = edge, edge
            endpoints.append({"kind": "marker_write", "edge": edge})
        if v("ecall"):
            if active or len(panel) != 2 or marker_edge is None or edge <= marker_edge or v("marker") != 1 or not v("halt") or v("fire"):
                raise ObservationError("ECALL/halt preceded drained marker publication")
            endpoints.append({"kind": "ecall_halt", "edge": edge})
            panels.append({"commands": panel, "marker_edge": marker_edge, "halt_edge": edge})
            panel, marker_edge, was_halted = [], None, True
        elif v("halt"):
            if active or panel or pending_marker is not None or v("fire") or v("csr"):
                raise ObservationError("unexpected halt within an observed panel")
            # Reset-to-host-START loading and post-ECALL readback/marker clear
            # are quiescent intervals. START itself is not inferred from this
            # wire: record only the observed halt assertion/deassertion.
            if not was_halted:
                endpoints.append({"kind": "quiescent_halt", "edge": edge})
            was_halted = True
        elif not v("halt"):
            if was_halted:
                endpoints.append({"kind": "halt_deasserted", "edge": edge})
            was_halted = False
    if active or panel or pending_marker is not None or len(panels) != expected_panels:
        raise ObservationError("trace lacks complete expected panels/release/marker/halt endpoints")
    for command in commands:
        command.pop("response_data", None)
        command.pop("loaded_rows", None)
    return {"sampled_edges": last_edge, "panels": panels, "endpoints": endpoints}


def _fresh_output(path):
    path = CHECK.allowed(path, missing=True)
    root = CHECK.allowed(Path(__file__).parent.parent)
    permitted = root / "build" / "rtl-timing"
    if not path.is_relative_to(permitted) or path == permitted or path.exists():
        raise ObservationError("output must be a fresh compiler build/rtl-timing directory")
    ignored = subprocess.run(["git", "check-ignore", "--quiet", str(path)], cwd=root, check=False)
    if ignored.returncode != 0:
        raise ObservationError("observation output is not ignored")
    return path


def prepare(witness_path, expected_digest, scope, output):
    if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)+", scope):
        raise ObservationError("scope must be an explicit dotted simulator hierarchy")
    checker = CHECK.Checker()
    member = checker.identity(witness_path)
    if member["sha256"] != expected_digest:
        raise ObservationError("selected witness report digest mismatch")
    report = CHECK.strict_json(CHECK.allowed(witness_path).read_bytes())
    if report.get("schema") != "atlas.ee290_vls_witness.v0" or report.get("state") != "integrated_execution_passed" or report.get("integrated_execution_passed") is not True:
        raise ObservationError("a successful selected witness report is required")
    provenance = report["provenance"]
    inputs = {"witness": member}
    for key in ("manifest", "hardware_ir", "simulator", "program"):
        checker.verify(provenance[key], Path(witness_path).parent)
        inputs[key] = provenance[key]
    manifest = CHECK.strict_json(CHECK.allowed(inputs["manifest"]["path"]).read_bytes())
    if manifest.get("schema") != "atlas.retained_hw_ir.v0" or manifest.get("state") != "verified":
        raise ObservationError("unsupported/unverified retention manifest")
    manifest_hw = checker.verify(manifest["hardware_ir"], Path(inputs["manifest"]["path"]).parent)
    if (manifest_hw["sha256"], manifest_hw["bytes"]) != (inputs["hardware_ir"]["sha256"], inputs["hardware_ir"]["bytes"]):
        raise ObservationError("witness/retention hardware identity mismatch")
    sources = {}
    for instance, module in MODULES.items():
        candidates = [item for item in provenance["simulator_sources"] if Path(item["path"]).name == module + ".sv"]
        if len(candidates) != 1:
            raise ObservationError("missing/ambiguous selected RTL module: " + module)
        item = candidates[0]
        source = checker.verify(item, Path(witness_path).parent)
        text = CHECK.allowed(source["path"]).read_text()
        if not re.search(r"\bmodule\s+" + module + r"\b", text):
            raise ObservationError("selected source lacks module declaration")
        for signal, _ in SIGNALS.values():
            inst, name = signal.split(".", 1)
            if inst == instance and not re.search(r"\b" + re.escape(name) + r"\b", text):
                raise ObservationError("selected source lacks fixed signal: " + signal)
        sources[module] = item
    output = _fresh_output(output)
    checker.recheck()
    output.mkdir(parents=True)
    # This is a reviewable candidate, not a supported command claim. Its flags
    # must first be checked with the selected simulator's UCLI help/probe.
    trace_path = str(output / "trace.vcd")
    quoted = '"' + trace_path.replace("\\", "\\\\").replace('"', '\\"').replace("$", "\\$").replace("[", "\\[").replace("]", "\\]").replace("\n", "\\n").replace("\r", "\\r") + '"'
    script = output / "capture-candidate.ucli"
    script.write_text("# Unvalidated candidate: verify selected UCLI syntax before execution.\n"
                      + "dump -file " + quoted + " -type VCD\n"
                      + "\n".join("dump -add " + scope + "." + name + " -depth 0" for name, _ in SIGNALS.values())
                      + "\nrun\nquit\n")
    snapshots = {}
    for source, name in [(witness_path, "selected-witness.json"), (Path(__file__), "observe-ee290-vls.executed.py"), (DEPENDENCY, "path-guard.executed.py")]:
        destination = output / name
        destination.write_bytes(CHECK.allowed(source).read_bytes())
        snapshots[name] = checker.identity(destination)
    plan = {"schema": "atlas.ee290_vls_observation_plan.v0", "target_config": "EE290SimConfig",
            "atlas_scope": scope, "inputs": inputs, "rtl_sources": sources,
            "signals": {key: {"path": scope + "." + name, "width": width} for key, (name, width) in SIGNALS.items()},
            "sampling": "values immediately before each rising-clock timestamp",
            "capture_command_validated": False, "scheduling_qualified": False,
            "capture_candidate": {"script": checker.identity(script), "trace_path": trace_path,
                                  "simulator_argv_suffix": ["-ucli", "-do", str(script)],
                                  "syntax_validated": False},
            "snapshots": snapshots,
            "assumptions": ASSUMPTIONS,
            "producer": checker.identity(Path(__file__)), "path_guard": checker.identity(DEPENDENCY)}
    (output / "plan.json").write_text(json.dumps(plan, indent=2) + "\n")
    (output / "signals.txt").write_text("\n".join(item["path"] for item in plan["signals"].values()) + "\n")
    checker.recheck()
    return plan


def decode(plan_path, expected_plan, trace_path, expected_trace, output, panels=3, max_edges=3000000):
    checker = CHECK.Checker()
    plan_id, trace_id = checker.identity(plan_path), checker.identity(trace_path)
    if plan_id["sha256"] != expected_plan or trace_id["sha256"] != expected_trace:
        raise ObservationError("selected plan/trace digest mismatch")
    plan = CHECK.strict_json(CHECK.allowed(plan_path).read_bytes())
    if plan.get("schema") != "atlas.ee290_vls_observation_plan.v0" or plan.get("target_config") != "EE290SimConfig":
        raise ObservationError("unsupported observation plan")
    expected_signals = {key: {"path": plan["atlas_scope"] + "." + name, "width": width} for key, (name, width) in SIGNALS.items()}
    if plan.get("signals") != expected_signals or plan.get("sampling") != "values immediately before each rising-clock timestamp":
        raise ObservationError("unsupported signal projection or sampling convention")
    for item in [*plan["inputs"].values(), *plan["rtl_sources"].values(), plan["producer"], plan["path_guard"]]:
        checker.verify(item, Path(plan_path).parent)
    for item in plan.get("snapshots", {}).values():
        checker.verify(item, Path(plan_path).parent)
    checker.verify(plan["capture_candidate"]["script"], Path(plan_path).parent)
    with CHECK.allowed(trace_path).open() as stream:
        result = validate_edges(vcd_edges(stream, plan["atlas_scope"], max_edges), panels)
    checker.recheck()
    output = _fresh_output(output)
    output.mkdir(parents=True)
    report = {"schema": "atlas.ee290_vls_boundary_observation.v0", "target_config": "EE290SimConfig",
              "state": "bounded_boundary_events_passed", "scheduling_qualified": False,
              "physical_sram_arbitration_verified": False, "capture_execution_link_verified": False,
              "inputs": {"plan": plan_id, "trace": trace_id}, "observations": result,
              "producer": checker.identity(Path(__file__)), "path_guard": checker.identity(DEPENDENCY),
              "assumptions": ASSUMPTIONS}
    for source, name in [(plan_path, "selected-plan.json"), (Path(__file__), "observe-ee290-vls.executed.py"), (DEPENDENCY, "path-guard.executed.py")]:
        (output / name).write_bytes(CHECK.allowed(source).read_bytes())
    (output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    checker.recheck()
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="action", required=True)
    prep = commands.add_parser("prepare")
    prep.add_argument("--witness-report", required=True, type=Path)
    prep.add_argument("--expected-witness-sha256", required=True)
    prep.add_argument("--atlas-scope", required=True)
    prep.add_argument("--output", required=True, type=Path)
    dec = commands.add_parser("decode")
    dec.add_argument("--plan", required=True, type=Path)
    dec.add_argument("--expected-plan-sha256", required=True)
    dec.add_argument("--trace", required=True, type=Path)
    dec.add_argument("--expected-trace-sha256", required=True)
    dec.add_argument("--output", required=True, type=Path)
    dec.add_argument("--expected-panels", type=int, default=3)
    dec.add_argument("--max-edges", type=int, default=3000000)
    args = parser.parse_args()
    try:
        if args.action == "prepare":
            prepare(args.witness_report, args.expected_witness_sha256, args.atlas_scope, args.output)
        else:
            decode(args.plan, args.expected_plan_sha256, args.trace, args.expected_trace_sha256,
                   args.output, args.expected_panels, args.max_edges)
    except (ObservationError, ValueError, KeyError, OSError) as exc:
        parser.exit(1, "observation rejected: " + str(exc) + "\n")


if __name__ == "__main__":
    main()
