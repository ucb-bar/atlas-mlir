#!/usr/bin/env python3
"""Revalidate finite EE290SimConfig component observations; never qualify scheduling."""
import argparse
from collections import deque
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import stat
import struct
import subprocess

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



def load(name, filename):
    path = bootstrap(Path(__file__).parent / filename)
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

SRAM = load("bounded_sram", "check-atlascore-sram-observation.py")
SOURCE = load("bounded_source", "check-ee290-source-correspondence.py")
CHECK, OBS = SRAM.CHECK, SRAM.OBS


def require(condition, message):
    if not condition:
        raise ValueError(message)


def select(checker, path, expected):
    require(CHECK.hash_string(expected), "explicit selection SHA-256 required")
    identity = checker.identity(path)
    data = CHECK.allowed(identity["path"]).read_bytes()
    require(identity["sha256"] == expected == CHECK.sha256(data), "selection digest mismatch")
    return identity, CHECK.object_value(CHECK.strict_json(data), "selected report")


def content(a, b, label):
    require((a["sha256"], a["bytes"]) == (b["sha256"], b["bytes"]), label + " content mismatch")


def phase(checker, identity, kind):
    actual = checker.verify(identity, Path(identity["path"]).parent)
    value = CHECK.strict_json(CHECK.allowed(actual["path"]).read_bytes())
    require(value.get("schema") == "atlas.ee290_captured_phase.v0" and value.get("target_config") == "EE290SimConfig" and value.get("kind") == kind, "captured phase schema/target/kind mismatch")
    require(value.get("state") == "phase_completed" and value.get("inputs_stable") is True and value["inputs_before"] == value["inputs_after"] and value.get("failures") == [] and value.get("scheduling_qualified") is False, "captured phase incomplete/unstable")
    require(value["command"].get("returncode") == 0 and value["command"].get("timed_out") is False, "captured command failed")
    for item in value["inputs_before"]:
        checker.verify(item["identity"], Path(actual["path"]).parent)
    # Historical producers bind their saved execution bytes, not current files.
    for key in ("producer", "path_checker_dependency"):
        saved = checker.verify(value["snapshots"][key], Path(actual["path"]).parent)
        content(value[key], saved, "executed phase producer")
    for key in ("stdout", "stderr"):
        checker.verify(value["command"][key], Path(actual["path"]).parent)
    for entry in value["outputs"]:
        for item in entry["files"]:
            checker.verify(item["identity"], Path(actual["path"]).parent)
    return value


def source_binding(checker, correspondence, manifest_id, manifest):
    require(correspondence.get("schema") == "atlas.ee290_source_correspondence.v0" and correspondence.get("target_config") == "EE290SimConfig", "source correspondence schema/target mismatch")
    require(correspondence.get("scheduling_qualified") is False, "source correspondence must remain unqualified")
    content(correspondence["selected_manifest"], manifest_id, "selected manifest")
    source_id = checker.verify(correspondence["source_report"], Path(manifest_id["path"]).parent)
    source = CHECK.strict_json(CHECK.allowed(source_id["path"]).read_bytes())
    validation = SOURCE.validate_source_receipt(checker, Path(source_id["path"]), source)
    lowering = phase(checker, correspondence["lowering_phase"], "selected_source_retention")
    outputs = [x["identity"] for row in lowering["outputs"] for x in row["files"]]
    require(correspondence["fresh_retained_manifest"] in outputs and correspondence["fresh_hardware_ir"] in outputs, "fresh retention not captured output")
    fresh_id = checker.verify(correspondence["fresh_retained_manifest"], Path(source_id["path"]).parent)
    fresh = CHECK.strict_json(CHECK.allowed(fresh_id["path"]).read_bytes())
    content(fresh["hardware_ir"], correspondence["fresh_hardware_ir"], "fresh hardware")
    content(fresh["inputs"]["firrtl"], source["firrtl"], "source/fresh FIRRTL")
    for key in ("annotations", "lowering_options", "chisel_annotations"):
        content(fresh["inputs"][key], manifest["inputs"][key], "retention policy")
    actual = SOURCE.compare_closures(CHECK.allowed(correspondence["fresh_hardware_ir"]["path"]).read_text(), CHECK.allowed(manifest["hardware_ir"]["path"]).read_text())
    require(actual == correspondence["atlas_core_correspondence"], "saved source closure differs from recomputation")
    return {"recipe": validation, "atlas_core": actual}


def validate_export(export, evidence, words):
    require(export.get("schema") == "atlas.resolved_rtl_timing.v0" and export.get("target_config") == "EE290SimConfig", "resolver export schema/target mismatch")
    require(export.get("qualification") == "conditional" and export.get("scheduling_qualified") is False, "resolver qualification promotion rejected")
    require(export.get("resolver") == {"id": "atlas.vls.conservative.v1", "version": 1}, "unreviewed resolver")
    require(export.get("evidence") == evidence, "resolver selected evidence mismatch")
    program = export["program"]
    require(program["words"] == words and program["word_count"] == len(words), "resolver program words mismatch")
    require(all(type(x) is int and 0 <= x < 2**32 for x in words), "invalid program word")
    digest = hashlib.sha256(b"".join(struct.pack("<I", x) for x in words)).hexdigest()
    require(program["words_sha256"] == digest, "resolver word digest mismatch")
    instructions = export["instructions"]
    require(len(instructions) == len(words), "incomplete resolver instruction export")
    for index, instruction in enumerate(instructions):
        require(instruction["word_index"] == index and instruction["word_u32"] == words[index], "resolver instruction/word mismatch")
        require(type(instruction["logical_issue_cycle"]) is int and instruction["logical_issue_cycle"] >= 0, "invalid resolver issue cycle")
        require(isinstance(instruction["operands"], dict) and isinstance(instruction["footprint"], dict), "missing resolved operands/footprint")
    applicability = export["applicability"]
    require(all(key in applicability for key in ("scope", "supported_operations", "operand_domain", "environment_assumptions", "unsupported")), "missing resolver applicability")
    return export



def bind_events(export, boundary):
    """Compare finite measured events directly with shared resolver streams.

    A stream element advances address and issue-relative age by its step.
    Holds end inclusively; decoder release marks the next edge. Read responses
    follow request edges by one cycle under the exported environment assumption.
    """
    instructions = export["instructions"]
    vls = [item for item in instructions if item["mnemonic"] in ("vload", "vstore")]
    marker = [item for item in instructions if item["mnemonic"] == "csrrw"]
    terminal = [item for item in instructions if item["mnemonic"] == "ecall"]
    require(len(vls) == 2 and [x["mnemonic"] for x in vls] == ["vload", "vstore"] and len(marker) == len(terminal) == 1, "finite two-VLS completion program required")
    require(terminal[0]["event_kind"] == "terminal_acceptance", "terminal acceptance convention required")
    require("sram_read_response_one_cycle_after_request" in export["applicability"]["environment_assumptions"], "missing SRAM response assumption")
    require(len(boundary["panels"]) == 3, "three finite panels required")
    for panel in boundary["panels"]:
        require(len(panel["commands"]) == len(vls), "trace command count differs from selected program")
        origin = panel["commands"][0]["edge"] - vls[0]["logical_issue_cycle"]
        for command, item in zip(panel["commands"], vls):
            store = item["mnemonic"] == "vstore"
            footprint = item["footprint"]
            vm = [x for x in footprint["accesses"] if x["resource"] == "vmem"]
            mr = [x for x in footprint["accesses"] if x["resource"] == "mreg"]
            require(len(vm) == len(mr) == 1, "finite explicit VLS access streams required")
            require(command["op"] == (2 if store else 1) and command["mreg"] == item["operands"]["rd"] and command["line"] == vm[0]["first"] and mr[0]["first"] == command["mreg"] * 32, "trace operands differ from selected resolved instruction")
            require(command["edge"] == origin + item["logical_issue_cycle"], "trace issue gap differs from selected program")
            source, destination = (mr[0], vm[0]) if store else (vm[0], mr[0])
            require(source["write"] is False and destination["write"] is True, "VLS stream direction mismatch")
            for stream, key in ((source, "source_edges"), (destination, "destination_edges")):
                require(stream["anywhere"] is False and stream["at_completion"] is False, "finite concrete stream required")
                expected = [command["edge"] + stream["age"] + i * stream["step"] for i in range(stream["count"])]
                require(command[key] == expected, "trace access ages differ from resolver stream")
            require(command["response_edges"] == [edge + 1 for edge in command["source_edges"]], "trace response ages differ from exported SRAM assumption")
            holds = [x for x in footprint["holds"] if x["unit"] in ("VLOAD path", "VSTORE path")]
            require(len(holds) == 2 and command["release_edge"] == command["edge"] + max(x["to"] for x in holds) + 1, "trace release differs from inclusive resolver path holds")
        require(panel["marker_edge"] == origin + marker[0]["logical_issue_cycle"] and panel["halt_edge"] == origin + terminal[0]["logical_issue_cycle"], "trace completion timing differs from selected program")
    return {"operand_streams": True, "issue_gaps": True, "access_ages": True, "release_and_completion_edges": True}


def replay_binding(checker, replay_id, replay, sram, manifest_id, manifest, compiler, arm, evidence, emitter, output):
    root = Path(replay_id["path"]).parent
    require(replay.get("target_config") == "EE290SimConfig" and replay.get("scheduling_qualified") is False, "replay target/qualification mismatch")
    # Bind selected witness snapshot, avoiding changed historical producer files.
    witness_path = CHECK.allowed(root / "inputs/selected-witness.json")
    witness_id = checker.identity(witness_path)
    content(witness_id, replay["original_inputs"][0], "selected witness snapshot")
    witness = CHECK.strict_json(witness_path.read_bytes())
    provenance = witness["provenance"]
    content(provenance["manifest"], manifest_id, "replay manifest")
    content(provenance["hardware_ir"], manifest["hardware_ir"], "replay hardware")
    content(provenance["program"], compiler["artifacts"][arm + ".mlir"], "compiler program")
    measured = SOURCE.FINGERPRINT.fingerprints(CHECK.allowed(manifest["hardware_ir"]["path"]).read_text(), ["AtlasCore"])
    names = set(measured) - {"banks_0_0_ext", "banks_0_ext", "buffer0_ext", "mem_0_ext"}
    names |= {"ram_128x256", "ram_32x264"}
    required_names = {name + ".sv" for name in names} | {"selected-memories.v"}
    snapshots = replay["rtl_snapshots"]
    require(len(snapshots) == len(required_names) and {Path(x["snapshot"]["path"]).name for x in snapshots} == required_names, "selected RTL/memory closure inventory mismatch")
    sources = {x["path"]: x for x in provenance["simulator_sources"]}
    for row in snapshots:
        require(row["original"]["path"] in sources and row["original"] == sources[row["original"]["path"]], "replay RTL not a selected simulator source")
        saved = checker.verify(row["snapshot"], root)
        content(saved, row["original"], "RTL snapshot")
        require(not re.search(r"`include\b|\$system\b|\$readmem[bh]\b", CHECK.allowed(saved["path"]).read_text()), "unbounded selected RTL dependency")
    memory = next(row for row in snapshots if Path(row["snapshot"]["path"]).name == "selected-memories.v")
    modules = set(re.findall(r"^module\s+([^\s(]+)", CHECK.allowed(memory["snapshot"]["path"]).read_text(), re.M))
    require({"banks_0_0_ext", "banks_0_ext", "buffer0_ext", "mem_0_ext"} <= modules, "missing behavioral memories")
    compilation = phase(checker, replay["compile_phase"], "atlascore_verilator_compile")
    phase(checker, replay["execution_phase"], "atlascore_vls_execute")
    compiled_rtl = [x["identity"] for x in compilation["inputs_before"] if x["role"] == "selected_rtl"]
    # Reused compile snapshots may live under baseline; compare both identities and order by content.
    require([(x["sha256"], x["bytes"]) for x in compiled_rtl] == [(x["snapshot"]["sha256"], x["snapshot"]["bytes"]) for x in snapshots], "compile RTL differs from selected replay")
    harness = [x["identity"] for x in compilation["inputs_before"] if x["role"] == "harness"]
    require(len(harness) == 1, "compile harness inventory mismatch")
    content(harness[0], replay["harness"], "compiled executed harness")
    require(compilation["command"]["argv"][-len(compiled_rtl)-1:] == [x["path"] for x in compiled_rtl] + [harness[0]["path"]], "compile source argv mismatch")
    words_id = checker.verify(compiler["artifacts"][arm + ".words"], root)
    text = CHECK.allowed(words_id["path"]).read_text()
    require(re.fullmatch(r"(?:[0-9a-fA-F]{8}\n)+", text) is not None, "invalid compiler word stream")
    words = [int(x, 16) for x in text.split()]
    content(replay["program_words"], words_id, "executed compiler word stream")
    program = checker.verify(compiler["artifacts"][arm + ".mlir"], root)
    result = subprocess.run([emitter["path"], "--rtl-timing-json", program["path"]], capture_output=True, timeout=60, check=False)
    require(result.returncode == 0, "compiler resolver export failed: " + result.stderr.decode(errors="replace")[:500])
    export = validate_export(CHECK.strict_json(result.stdout), evidence, words)
    resolved_path = output / (arm + "-resolved.json")
    resolved_path.write_bytes(result.stdout)
    stderr_path = output / (arm + "-resolved.stderr.log")
    stderr_path.write_bytes(result.stderr)
    export_command = {"argv": [emitter["path"], "--rtl-timing-json", program["path"]], "returncode": result.returncode, "timed_out": False, "stdout": checker.identity(resolved_path), "stderr": checker.identity(stderr_path), "program": program}
    require(sram.get("schema") == "atlas.selected_atlascore_sram_observation.v0" and sram.get("target_config") == "EE290SimConfig" and sram.get("scheduling_qualified") is False, "SRAM schema/target/qualification mismatch")
    content(sram["inputs"]["replay"], replay_id, "SRAM replay")
    content(sram["inputs"]["trace"], replay["trace"], "SRAM trace")
    recomputed = SRAM.run(Path(replay_id["path"]), replay_id["sha256"], output / (arm + "-rechecked"))
    require(recomputed["physical_observations"] == sram["physical_observations"], "saved SRAM observations differ from trace")
    event_binding = bind_events(export, replay["boundary_observations"])
    return {"replay": replay_id, "finite_commands": replay["boundary_observations"]["panels"], "physical_windows": recomputed["physical_observations"], "resolver_applicability": export["applicability"], "event_binding": event_binding, "resolver_export": export_command}



def verilog_content_binding(checker, selection, manifest_id, manifest):
    """Content/command metadata correspondence only, not execution provenance."""
    derivation_id, derivation = selection["verilog-derivation"]
    ram_id, ram = selection["implicit-ram-correspondence"]
    require(derivation.get("schema") == "atlas.local_verilog_derivation.v0" and ram.get("schema") == "atlas.local_implicit_ram_correspondence.v0", "Verilog correspondence schema mismatch")
    require(derivation.get("target_config") == ram.get("target_config") == "EE290SimConfig" and derivation.get("scheduling_qualified") is False and ram.get("scheduling_qualified") is False, "Verilog correspondence target/qualification mismatch")
    content(derivation["manifest"], manifest_id, "Verilog selected manifest")
    content(ram["verilog_derivation"], derivation_id, "implicit RAM derivation")
    content(ram["source_replay"], selection["baseline-replay"][0], "implicit RAM selected replay")
    base = Path(derivation_id["path"]).parent
    for key, value in derivation["inputs"].items():
        checker.verify(value, base)
        content(value, manifest["inputs"][key], "Verilog selected input")
    checker.verify(derivation["firtool"], base)
    content(derivation["firtool"], manifest["tools"]["firtool"], "Verilog firtool")
    prepared_id = checker.verify(derivation["prepared_annotations"], base)
    original = CHECK.strict_json(CHECK.allowed(derivation["inputs"]["annotations"]["path"]).read_bytes())
    prepared = CHECK.strict_json(CHECK.allowed(prepared_id["path"]).read_bytes())
    require(len(original) == len(prepared), "Verilog annotation inventory mismatch")
    for before, after in zip(original, prepared):
        if before.get("class") in SOURCE.HIERARCHY:
            require(set(after) == set(before) and {k:v for k,v in before.items() if k != "filename"} == {k:v for k,v in after.items() if k != "filename"}, "Verilog hierarchy annotation changed")
            target = CHECK.allowed(after["filename"], missing=True)
            require(target.parent == base and target.name == SOURCE.HIERARCHY[before["class"]], "Verilog hierarchy output mismatch")
        else: require(before == after, "Verilog annotation policy mismatch")
    command = derivation["command"]
    argv = command["argv"]
    options = CHECK.allowed(derivation["inputs"]["lowering_options"]["path"]).read_text().strip()
    require(command["returncode"] == 0 and command["cwd"] == str(base) and argv[0] == derivation["firtool"]["path"] and argv[-1] == derivation["inputs"]["firrtl"]["path"] and "--annotation-file=" + prepared_id["path"] in argv and "--lowering-options=" + options in argv and "--split-verilog" in argv, "Verilog declared command/input mismatch")
    checker.verify(command["log"], base)
    closure = SOURCE.FINGERPRINT.fingerprints(CHECK.allowed(manifest["hardware_ir"]["path"]).read_text(), ["AtlasCore"])
    rows = derivation["module_comparisons"]
    require(len(rows) == len(closure) and {row["module"] for row in rows} == set(closure), "Verilog module correspondence coverage mismatch")
    external = {"banks_0_0_ext", "banks_0_ext", "buffer0_ext", "mem_0_ext"}
    matched = {}
    for row in rows:
        if row["module"] in external:
            require(row["status"] == "external_or_not_compared" and not row["file_comparisons"], "external implementation incorrectly certified")
            continue
        require(len(row["file_comparisons"]) == 1, "ambiguous standalone module correspondence")
        pair = row["file_comparisons"][0]
        left, right = (checker.verify(pair[key], base) for key in ("previous", "derived"))
        content(left, right, "recomputed Verilog module")
        require(Path(left["path"]).name == Path(right["path"]).name == row["module"] + ".sv", "Verilog correspondence module/file mismatch")
        matched[row["module"] + ".sv"] = right
    require(len(ram["comparisons"]) == 2 and {x["name"] for x in ram["comparisons"]} == {"ram_128x256.sv", "ram_32x264.sv"}, "implicit RAM inventory mismatch")
    for row in ram["comparisons"]:
        left, right = (checker.verify(row[key], base) for key in ("selected", "derived"))
        content(left, right, "recomputed implicit RAM")
        matched[row["name"]] = right
    for arm in ("baseline", "scheduled"):
        for row in selection[arm + "-replay"][1]["rtl_snapshots"]:
            name = Path(row["snapshot"]["path"]).name
            if name != "selected-memories.v": content(row["snapshot"], matched[name], "selected compiled Verilog")
    return {"derivation": derivation_id, "implicit_ram": ram_id, "matched_standalone_modules": len(matched) - 2, "matched_implicit_rams": 2, "external_implementations": sorted(external), "content_correspondence_verified": True, "generation_execution_proven": False}


SELECTIONS = ("source-correspondence", "manifest", "conditional-report", "compiler-witness", "baseline-replay", "scheduled-replay", "baseline-sram", "scheduled-sram")



def publish_report(checker, path, result):
    # Never leave a success packet when a final input-stability check rejects.
    checker.recheck()
    path.write_text(json.dumps(result, indent=2) + "\n")


def run(args):
    checker = CHECK.Checker()
    helpers = {name: checker.identity(Path(__file__).parent / name) for name in ("check-ee290-bounded-applicability.py", "check-atlascore-sram-observation.py", "observe-ee290-vls.py", "check-ee290-source-correspondence.py", "fingerprint-rtl-modules.py", "check-ee290-provenance.py", "ee290_build_capture.py", "elaborate-ee290-source.py", "index-retained-hw.py")}
    selected = {name: select(checker, getattr(args, name.replace("-", "_")), getattr(args, "expected_" + name.replace("-", "_") + "_sha256")) for name in SELECTIONS}
    emitter = checker.identity(args.atlas_emit)
    require(emitter["sha256"] == args.expected_atlas_emit_sha256, "selected emitter digest mismatch")
    manifest_id, manifest = selected["manifest"]
    require(manifest.get("schema") == "atlas.retained_hw_ir.v0" and manifest.get("state") == "verified", "retention manifest schema/state mismatch")
    checker.verify(manifest["hardware_ir"], Path(manifest_id["path"]).parent)
    conditional_id, conditional = selected["conditional-report"]
    require(conditional.get("schema") == "atlas.conditional_vls_hw_check.v0" and conditional.get("target_config") == "EE290SimConfig", "conditional evidence schema/target mismatch")
    content(conditional["inputs"]["manifest"], manifest_id, "conditional manifest")
    content(conditional["inputs"]["hardware_ir"], manifest["hardware_ir"], "conditional hardware")
    conditional_base = Path(conditional_id["path"]).parent
    require(set(conditional["inputs"]["sources"]) == set(conditional["snapshots"]), "conditional source/snapshot inventory mismatch")
    for role, declared in conditional["inputs"]["sources"].items():
        saved = checker.verify(conditional["snapshots"][role], conditional_base)
        content(declared, saved, "conditional executed source")
    for key, value in conditional["inputs"].items():
        if key != "sources": checker.recursively_verify(value, conditional_base)
    checker.recursively_verify(conditional["artifacts"], conditional_base)
    compiler_id, compiler = selected["compiler-witness"]
    require(compiler.get("schema") == "atlas.compiler_vls_witness.v0" and compiler.get("qualification") == "conditional", "conditional compiler witness required")
    content(compiler["evidence"], conditional_id, "compiler selected evidence")
    evidence = {"evidence_sha256": conditional_id["sha256"], "manifest_sha256": manifest_id["sha256"], "hardware_ir_sha256": manifest["hardware_ir"]["sha256"]}
    source = source_binding(checker, selected["source-correspondence"][1], manifest_id, manifest)
    for name in ("verilog-derivation", "implicit-ram-correspondence"):
        path = getattr(args, name.replace("-", "_"), None)
        expected = getattr(args, "expected_" + name.replace("-", "_") + "_sha256", None)
        require(bool(path) == bool(expected), "optional correspondence path/hash must be paired")
        if path: selected[name] = select(checker, path, expected)
    require(("verilog-derivation" in selected) == ("implicit-ram-correspondence" in selected), "both Verilog/implicit RAM correspondence selections required")
    verilog = verilog_content_binding(checker, selected, manifest_id, manifest) if "verilog-derivation" in selected else {"content_correspondence_verified": False, "generation_execution_proven": False}
    output = OBS._fresh_output(args.output)
    output.mkdir(parents=True)
    helper_snapshots = {}
    for name, identity in helpers.items():
        saved = output / (name.removesuffix(".py") + ".executed.py")
        saved.write_bytes(CHECK.allowed(identity["path"]).read_bytes())
        helper_snapshots[name] = checker.identity(saved)
        content(identity, helper_snapshots[name], "validator snapshot")
    arms = {}
    for arm in ("baseline", "scheduled"):
        replay_id, replay = selected[arm + "-replay"]
        arms[arm] = replay_binding(checker, replay_id, replay, selected[arm + "-sram"][1], manifest_id, manifest, compiler, arm, evidence, emitter, output)
    checker.recheck()
    result = {"schema": "atlas.ee290_bounded_applicability.v0", "target_config": "EE290SimConfig", "state": "finite_observations_validated", "qualification": "conditional", "scheduling_qualified": False, "system_qualified": False, "operand_domain_qualified": False, "source_to_simulation_complete": False, "source_to_simulation_edge": "FIRRTL/retained-HW and selected SV content bindings are recorded separately; no captured Verilog generation execution proves the generation edge, and four external behavioral memory implementations remain conditional", "selection": {k: v[0] for k, v in selected.items()}, "atlas_emit": emitter, "source_correspondence": source, "verilog_content_correspondence": verilog, "arms": arms,
              "assumptions": ["Selected AtlasCore with two-state Verilator and behavioral SRAM wrappers; direct full-line host TileLink accesses.", "Finite operands and recorded command windows only; no universal bank/address/mreg/stride coverage.", "Granted bank enable patterns are checked; losing or suppressed same-bank competing requests and activity outside observed windows remain unexcluded.", "The full CPU/system numerical VCS pair has no internal boundary trace and does not establish fullsystem scheduling applicability.", "Physical macro/PVT timing, opaque framework dependencies, hidden subprocess inputs and system runtime closure remain conditional."],
              "producer": helpers["check-ee290-bounded-applicability.py"], "validators": helpers, "validator_snapshots": helper_snapshots}
    (output / "check-ee290-bounded-applicability.executed.py").write_bytes(CHECK.allowed(__file__).read_bytes())
    publish_report(checker, output / "report.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in SELECTIONS:
        parser.add_argument("--" + name, required=True, type=Path)
        parser.add_argument("--expected-" + name + "-sha256", required=True)
    for name in ("verilog-derivation", "implicit-ram-correspondence"):
        parser.add_argument("--" + name, type=Path)
        parser.add_argument("--expected-" + name + "-sha256")
    parser.add_argument("--atlas-emit", required=True, type=Path)
    parser.add_argument("--expected-atlas-emit-sha256", required=True)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    try:
        run(args)
    except (ValueError, KeyError, TypeError, OSError, RuntimeError, subprocess.TimeoutExpired) as error:
        parser.exit(1, "bounded applicability rejected: " + str(error) + "\n")


if __name__ == "__main__":
    main()
