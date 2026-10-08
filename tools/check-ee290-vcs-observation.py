#!/usr/bin/env python3
"""Revalidate a pinned finite EE290SimConfig VCS trace pair without simulation.

Historical producers are bound to their execution snapshots. Current authored
validators recompute the selected event evidence; saved success flags alone do
not pass. All compiler/system/domain qualification flags remain false.
"""
import argparse
from collections import deque
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
    if not path.is_absolute():
        path = Path.cwd() / path
    denied = lambda p: any("hammer" in x.lower() or "vlsi" in x.lower() for x in p.parts)
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
                raise ValueError("restricted dependency symlink target")
            if target.is_absolute():
                current = Path(target.anchor)
                pending.extendleft(reversed(target.parts[1:]))
            else:
                pending.extendleft(reversed(target.parts))
        else:
            current = candidate
    return current


def load(name, filename):
    spec = importlib.util.spec_from_file_location(name, bootstrap(Path(__file__).parent / filename))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


BOUND = load("integrated_bounded", "check-ee290-bounded-applicability.py")
CAP = load("integrated_capture", "capture-ee290-vcs-observation.py")
CHECK, OBS = BOUND.CHECK, BOUND.OBS
require, content, select = BOUND.require, BOUND.content, BOUND.select
ROOT_SELECTIONS = ("pair", "baseline-plan", "baseline-capture", "scheduled-plan", "scheduled-capture")


def read_member(checker, member, base):
    actual = checker.verify(member, base)
    return actual, CHECK.strict_json(CHECK.allowed(actual["path"]).read_bytes())


def historical(checker, declared, snapshot, base, label):
    saved = checker.verify(snapshot, base)
    content(declared, saved, label)
    return saved


def command(checker, record, expected, base, label):
    for key, value in expected.items():
        require(record.get(key) == value, label + " invocation mismatch: " + key)
    require(record.get("returncode") == 0 and record.get("timed_out") is False,
            label + " command failed/timed out")
    require(record.get("descendants_after_parent_exit") is False,
            label + " command left descendants")
    identity = checker.verify(record["log"], base)
    text = CHECK.allowed(identity["path"]).read_text(errors="replace")
    require(CAP.BAD_LOG.search(text) is None and CAP.LICENSE_LOG.search(text) is None,
            label + " log contains error/license failure")
    return {"command": record, "text": text}


def word_stream(checker, member, base):
    actual = checker.verify(member, base)
    text = CHECK.allowed(actual["path"]).read_text()
    require(re.fullmatch(r"(?:[0-9a-fA-F]{8}\n){1,32768}", text) is not None,
            "malformed emitted word stream")
    return actual, [int(value, 16) for value in text.split()]


def host_program(checker, witness, plan, compiler, arm, base):
    require(witness.get("schema") == "atlas.ee290_vls_witness.v0" and
            witness.get("target_config") == "EE290SimConfig" and
            witness.get("scheduling_qualified") is False, "selected numerical witness schema/qualification mismatch")
    require(witness.get("state") == "integrated_execution_passed" and witness.get("integrated_execution_passed") is True, "selected witness is not completed numerical execution")
    provenance = witness["provenance"]
    require(len(witness["commands"]) == 3, "selected numerical witness command inventory mismatch")
    emit, build, simulation = witness["commands"]
    emitted_id, words = word_stream(checker, emit["log"], base)
    require(witness.get("program_word_count") == len(words), "witness word count mismatch")
    program = checker.verify(plan["snapshots"]["program"], base)
    program_id = checker.verify(compiler["artifacts"][arm + ".mlir"], base)
    words_id, selected_words = word_stream(checker, compiler["artifacts"][arm + ".words"], base)
    content(program, program_id, "executed/compiler program")
    content(emitted_id, words_id, "executed/compiler words")
    require(words == selected_words, "executed/compiler words differ")
    expected_include = (f"#define ATLAS_PROGRAM_WORDS {len(words)}U\nstatic const uint32_t atlas_program[] = {{\n"
                        + "".join(f"  0x{value:08x}U,\n" for value in words) + "};\n")
    include = checker.verify(plan["snapshots"]["generated_include"], base)
    require(CHECK.allowed(include["path"]).read_text() == expected_include, "executed include differs from words")
    host = checker.verify(plan["snapshots"]["host_binary"], base)
    source = checker.verify(plan["snapshots"]["host_source"], base)
    require('#include "atlas_program.inc"' in CHECK.allowed(source["path"]).read_text(), "host source lacks selected program include")
    original_root = Path(plan["selected_witness"]["path"]).parent
    executed_program = checker.identity(original_root / "program.mlir")
    executed_source = checker.identity(original_root / "ee290-vls-host.executed.c")
    content(executed_program, program, "original executed program")
    content(executed_source, source, "original executed host source")
    emitter = checker.verify(provenance["atlas_emit"], base)
    host_cc = checker.verify(provenance["host_cc"], base)
    require(emit["argv"] == [emitter["path"], executed_program["path"]] and
            emit.get("returncode") == 0 and emit.get("timed_out") is False,
            "selected numerical emission invocation mismatch")
    expected_build = [host_cc["path"], "-std=gnu99", "-O2", "-Wall", "-Wextra", "-Werror",
                      "-fno-common", "-fno-builtin-printf", "-march=rv64imafd", "-mabi=lp64d",
                      "-mcmodel=medany", "-specs=htif_nano.specs", "-static", "-T", "htif.ld",
                      executed_source["path"], "-o", witness["host_binary"]["path"]]
    require(build["argv"] == expected_build and build.get("returncode") == 0 and
            build.get("timed_out") is False, "selected host compilation invocation mismatch")
    checker.verify(build["log"], base)
    original_log = checker.verify(simulation["log"], base)
    require(simulation.get("returncode") == 0 and simulation.get("timed_out") is False and
            CAP.numerical_passed(CHECK.allowed(original_log["path"]).read_text(errors="replace")),
            "selected original numerical result is not independently supported")
    require(host["bytes"] <= 16 * 1024 * 1024, "host ELF exceeds supported inspection bound")
    blob = CHECK.allowed(host["path"]).read_bytes()
    require(blob.startswith(b"\x7fELF\x02\x01") and
            blob.count(b"".join(struct.pack("<I", value) for value in words)) == 1,
            "selected little-endian ELF lacks unique complete program word array")
    return {"program": program, "words": words_id, "host_binary": host,
            "program_words_embedded_once": True, "word_values": words}


def verify_arm(checker, arm, plan_id, capture_id, phase, numerical_pair, compiler):
    base = Path(capture_id["path"]).parent
    _, plan = read_member(checker, plan_id, base)
    _, capture = read_member(checker, capture_id, base)
    require(plan.get("schema") == "atlas.ee290_vcs_observation_capture_plan.v0" and
            capture.get("schema") == "atlas.ee290_vcs_observation_capture.v0" and
            plan.get("target_config") == capture.get("target_config") == "EE290SimConfig",
            "capture plan/report schema or target mismatch")
    require(plan.get("scheduling_qualified") is False and capture.get("scheduling_qualified") is False,
            "capture qualification promotion rejected")
    require(plan.get("state") == "prepared_not_executed" and capture.get("state") == "numerical_and_boundary_observation_passed" and all(capture.get(key) is True for key in ("integrated_execution_passed", "capture_production_recorded", "boundary_observation_passed", "signal_visibility_verified", "ucli_help_options_observed")) and capture.get("visible_signal_declarations") == len(CAP.capture_signals()), "capture completion metadata is inconsistent")
    content(capture["inputs"]["plan"], plan_id, "capture selected plan")
    content(capture["inputs"]["build_phase"], plan["selected_build_phase"], "capture selected build")
    content(capture["inputs"]["witness"], plan["selected_witness"], "capture selected witness")
    historical(checker, plan["producer"], plan["snapshots"]["capture-ee290-vcs-observation.executed.py"], base, "capture producer snapshot")
    historical(checker, plan["converter"], plan["snapshots"]["vpd2vcd.executed"], base, "converter snapshot")
    for member in plan["snapshots"].values():
        checker.verify(member, base)
    historical(checker, plan["selected_witness"], plan["snapshots"]["selected-witness.json"], base, "selected witness snapshot")
    historical(checker, plan["selected_build_phase"], plan["snapshots"]["selected-build-phase.json"], base, "selected build snapshot")
    witness = CHECK.strict_json(CHECK.allowed(plan["snapshots"]["selected-witness.json"]["path"]).read_bytes())
    old_members = {"host_binary": witness["host_binary"], "generated_include": witness["generated_include"],
                   "program": witness["provenance"]["program"], "host_source": witness["provenance"]["host_source"]}
    require(plan["selected_members"] == old_members, "capture selected member mapping mismatch")
    for key, member in old_members.items():
        checker.verify(member, base)
        content(member, plan["snapshots"][key], "capture input snapshot")
    selected_arm = [row for row in numerical_pair["runs"] if row["arm"] == arm]
    require(len(selected_arm) == 1, "numerical pair arm missing/ambiguous")
    content(selected_arm[0]["receipt"], plan["selected_witness"], "numerical pair/witness")
    checker.verify(plan["simulator"], base)
    content(plan["simulator"], witness["provenance"]["simulator"], "numerical/capture simulator")
    compiled = [row["identity"] for entry in phase["outputs"] if entry["role"] == "simulator" for row in entry["files"]]
    require(len(compiled) == 1, "captured build simulator output inventory mismatch")
    content(compiled[0], plan["simulator"], "compiled/captured simulator")
    require(plan["simulator_archive_libraries"] == witness["provenance"]["simulator_archive_libraries"], "capture runtime inventory differs from witness")
    CAP.bind_runtime_archive(phase, plan["simulator_archive_libraries"])
    for member in plan["simulator_archive_libraries"]:
        checker.verify(member, base)
    phase_inputs = {(row["identity"]["sha256"], row["identity"]["bytes"]) for row in phase["inputs_before"]}
    for member in witness["provenance"]["simulator_sources"]:
        checker.verify(member, base)
        require((member["sha256"], member["bytes"]) in phase_inputs,
                "numerical source/filelist is not a captured build input")
    program = host_program(checker, witness, plan, compiler, arm, base)
    boundary_plan_id, boundary_plan = read_member(checker, plan["boundary_plan"], base)
    require(boundary_plan.get("schema") == "atlas.ee290_vls_observation_plan.v0" and
            boundary_plan.get("target_config") == "EE290SimConfig" and
            boundary_plan.get("sampling") == "values immediately before each rising-clock timestamp",
            "unsupported boundary plan/sampling")
    scope = boundary_plan["atlas_scope"]
    require(re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)+", scope) is not None,
            "unsupported Atlas scope")
    expected_signals = {scope + "." + name: width for name, width in CAP.capture_signals().items()}
    require(plan["capture_signals"] == expected_signals and
            boundary_plan["signals"] == {key: {"path": scope + "." + name, "width": width} for key, (name, width) in OBS.SIGNALS.items()},
            "unsupported trace signal projection")
    require(boundary_plan["inputs"]["witness"] == plan["selected_witness"], "boundary selected witness mismatch")
    for key in ("manifest", "hardware_ir", "simulator", "program"):
        content(boundary_plan["inputs"][key], witness["provenance"][key], "boundary/witness input")
        checker.verify(boundary_plan["inputs"][key], base)
    for member in boundary_plan["rtl_sources"].values():
        checker.verify(member, base)
        require(member in witness["provenance"]["simulator_sources"], "boundary RTL source not selected simulator input")
    for key, snapshot in [("producer", "observe-ee290-vls.executed.py"), ("path_guard", "path-guard.executed.py")]:
        historical(checker, boundary_plan[key], plan["snapshots"][snapshot], base, "historical boundary dependency")
    for member in boundary_plan["snapshots"].values():
        checker.verify(member, base)
    checker.verify(boundary_plan["capture_candidate"]["script"], base)
    require(CHECK.allowed(plan["capture_script"]["path"]).read_text() == CAP.capture_script_text(scope, base / "run/trace.vpd") and
            CHECK.allowed(plan["probe_script"]["path"]).read_text() == CAP.PROBE_SCRIPT,
            "executed UCLI script differs from supported fixed script")
    checker.verify(plan["capture_script"], base)
    checker.verify(plan["probe_script"], base)
    expected = CAP.invocation(witness["commands"][-1]["argv"], plan["simulator"], old_members["host_binary"], plan["capture_script"]["path"])
    expected[3], expected[-1] = "+loadmem=" + program["host_binary"]["path"], program["host_binary"]["path"]
    expected_probe = expected.copy()
    expected_probe[expected_probe.index("-do") + 1] = plan["probe_script"]["path"]
    converter = plan["snapshots"]["vpd2vcd.executed"]
    native_id = checker.verify(capture["native_trace"], base)
    trace_id = checker.verify(capture["trace"], base)
    require(native_id["path"] == str(base / "run/trace.vpd") and trace_id["path"] == str(base / "run/trace.vcd"), "capture trace paths mismatch")
    require(len(capture["commands"]) == 3, "capture command inventory incomplete")
    requirements = [("probe", {"argv": expected_probe, "cwd": str(base / "probe"), "timeout_seconds": 60}),
                    ("simulation", {"argv": expected, "cwd": str(base / "run"), "native_trace": native_id["path"], "trace": trace_id["path"], "timeout_seconds": plan["invocation"]["timeout_seconds"]}),
                    ("conversion", {"argv": [plan["converter"]["path"], "-full64", native_id["path"], trace_id["path"]], "cwd": str(base / "conversion"), "timeout_seconds": 300})]
    require(plan["probe_invocation"] == requirements[0][1] and plan["invocation"] == requirements[1][1] and
            plan["conversion_invocation"] == requirements[2][1], "prepared invocation mapping differs from execution")
    require(0 < plan["invocation"]["timeout_seconds"] <= 7200, "unsupported simulation timeout")
    logs = {label: command(checker, record, expected_command, base, label)
            for record, (label, expected_command) in zip(capture["commands"], requirements)}
    require(len(re.findall(r"^ATLAS_UCLI_HELP_COMPLETED$", logs["probe"]["text"], re.M)) == 1 and
            all(token in logs["probe"]["text"] for token in ("-file", "-type", "-add", "-depth", "-fid")), "help probe not supported by emitted output")
    require(len(re.findall(r"^ATLAS_UCLI_CAPTURE_SETUP_OK$", logs["simulation"]["text"], re.M)) == 1 and
            CAP.numerical_passed(logs["simulation"]["text"]), "capture/numerical pass not supported by emitted output")
    require(capture.get("observations") == [line for line in logs["simulation"]["text"].splitlines() if line.startswith("EE290_VLS_")], "saved numerical observations differ from log")
    require("Done writing vcd value changes." in logs["conversion"]["text"] and
            re.search(r"^Done$", logs["conversion"]["text"], re.M) is not None,
            "conversion completion absent from log")
    require(capture.get("runtime_environment") == plan.get("runtime_environment_from_witness") == witness.get("runtime_environment"), "capture runtime environment differs from selected witness")
    with CHECK.allowed(trace_id["path"]).open() as stream:
        declarations = CAP.visible_projection(stream, expected_signals)
    with CHECK.allowed(trace_id["path"]).open() as stream:
        boundary = OBS.validate_edges(OBS.vcd_edges(stream, scope), 3)
    boundary_id, saved = read_member(checker, capture["boundary_observation"], base)
    require(saved.get("schema") == "atlas.ee290_vls_boundary_observation.v0" and saved.get("target_config") == "EE290SimConfig" and saved.get("scheduling_qualified") is False, "boundary report schema/qualification mismatch")
    content(saved["inputs"]["plan"], boundary_plan_id, "boundary report plan")
    content(saved["inputs"]["trace"], trace_id, "boundary report trace")
    require(saved.get("state") == "bounded_boundary_events_passed" and saved.get("physical_sram_arbitration_verified") is False and saved.get("capture_execution_link_verified") is False, "boundary report scope/qualification mismatch")
    require(saved["observations"] == boundary, "saved boundary observations differ from recomputed trace")
    historical(checker, saved["producer"], plan["snapshots"]["observe-ee290-vls.executed.py"], base, "boundary producer snapshot")
    historical(checker, saved["path_guard"], plan["snapshots"]["path-guard.executed.py"], base, "boundary guard snapshot")
    return {"plan": plan_id, "capture": capture_id, "simulator": plan["simulator"], "build_phase": plan["selected_build_phase"],
            "selected_witness": plan["selected_witness"], "manifest": witness["provenance"]["manifest"], "hardware_ir": witness["provenance"]["hardware_ir"],
            "source_inventory": witness["provenance"]["simulator_sources"], "capture_producer_snapshot": plan["snapshots"]["capture-ee290-vcs-observation.executed.py"],
            "converter_snapshot": converter, "native_trace": native_id, "trace": trace_id, "boundary_report": boundary_id,
            "boundary_observations": boundary, "visible_signal_declarations": declarations, "program": program,
            "runtime_environment": capture["runtime_environment"], "scope": scope,
            "validated_links": {"build_runtime_and_program": True, "native_capture_and_conversion_commands": True,
                                "numerical_log": True, "recomputed_boundary_events": True}}


def bind_compared_sources(compared, arms):
    for arm in arms.values():
        inventory = {}
        for member in arm["source_inventory"]:
            inventory.setdefault(Path(member["path"]).name, []).append(member)
        for member in compared:
            name = Path(member["path"]).name
            require(name in inventory, "compared Atlas module missing from integrated build input")
            require(len(inventory[name]) == 1, "ambiguous compared Atlas module basename: " + name)
            content(member, inventory[name][0], "integrated compared Atlas module")


def source_bridge(checker, packet_id, packet, arms, compiler_id):
    require(packet.get("schema") == "atlas.ee290_bounded_applicability.v0" and packet.get("target_config") == "EE290SimConfig" and
            all(packet.get(key) is False for key in ("scheduling_qualified", "system_qualified", "operand_domain_qualified", "source_to_simulation_complete")), "bounded applicability schema/qualification mismatch")
    require(packet.get("state") == "finite_observations_validated" and
            packet.get("qualification") == "conditional", "bounded applicability must be completed and conditional")
    base = Path(packet_id["path"]).parent
    require(set(packet["validators"]) == set(packet["validator_snapshots"]), "bounded validator snapshot inventory mismatch")
    for key, declared in packet["validators"].items():
        historical(checker, declared, packet["validator_snapshots"][key], base, "bounded validator snapshot")
    selections = {key: read_member(checker, member, base) for key, member in packet["selection"].items()}
    manifest_id, manifest = selections["manifest"]
    require(manifest.get("schema") == "atlas.retained_hw_ir.v0" and manifest.get("state") == "verified", "retained HW manifest unsupported")
    content(selections["compiler-witness"][0], compiler_id, "integrated/component compiler witness")
    checker.verify(manifest["hardware_ir"], base)
    for arm in arms.values():
        content(arm["manifest"], manifest_id, "integrated/component manifest")
        content(arm["hardware_ir"], manifest["hardware_ir"], "integrated/component hardware")
    source = BOUND.source_binding(checker, selections["source-correspondence"][1], manifest_id, manifest)
    require(source == packet["source_correspondence"], "bounded saved source correspondence differs from recomputation")
    require("verilog-derivation" in selections and "implicit-ram-correspondence" in selections, "Verilog content correspondence selections required")
    verilog = BOUND.verilog_content_binding(checker, selections, manifest_id, manifest)
    require(verilog == packet["verilog_content_correspondence"], "bounded saved Verilog correspondence differs from recomputation")
    derivation = selections["verilog-derivation"][1]
    compared = [row["file_comparisons"][0]["previous"] for row in derivation["module_comparisons"] if row["file_comparisons"]]
    bind_compared_sources(compared, arms)
    conditional_id, conditional = selections["conditional-report"]
    require(conditional.get("schema") == "atlas.conditional_vls_hw_check.v0" and
            conditional.get("target_config") == "EE290SimConfig" and
            conditional.get("state") == "conditional_checks_passed" and
            conditional.get("rtl_rules_enabled") is False and
            conditional.get("resolver_bindings") == [] and
            conditional.get("conditional_replay_expectations_met") is True and
            conditional.get("compiler_comparison_matches") is True and
            conditional.get("scope", {}).get("opaque_sram_implementation_qualified") is False and
            conditional.get("scope", {}).get("integrated_target_execution_qualified") is False,
            "conditional replay schema, completion or qualification mismatch")
    content(conditional["inputs"]["manifest"], manifest_id, "conditional selected manifest")
    content(conditional["inputs"]["hardware_ir"], manifest["hardware_ir"], "conditional selected hardware")
    evidence = {"evidence_sha256": conditional_id["sha256"], "manifest_sha256": manifest_id["sha256"], "hardware_ir_sha256": manifest["hardware_ir"]["sha256"]}
    return {"packet": packet_id, "source_correspondence": source, "verilog_content_correspondence": verilog,
            "matched_integrated_module_inputs": len(compared), "evidence": evidence,
            "source_to_simulation_complete": False}


def export_binding(checker, emitter, arm, evidence, output):
    program = arm["program"]["program"]
    argv = [emitter["path"], "--rtl-timing-json", program["path"]]
    result = subprocess.run(argv, capture_output=True, timeout=60, check=False)
    require(result.returncode == 0, "fresh resolver export failed")
    export = BOUND.validate_export(CHECK.strict_json(result.stdout), evidence, arm["program"]["word_values"])
    events = BOUND.bind_events(export, arm["boundary_observations"])
    stdout, stderr = output / "resolved.json", output / "resolved.stderr.log"
    stdout.write_bytes(result.stdout)
    stderr.write_bytes(result.stderr)
    return {"event_binding": events, "applicability": export["applicability"],
            "command": {"argv": argv, "returncode": result.returncode, "timed_out": False,
                        "stdout": checker.identity(stdout), "stderr": checker.identity(stderr)}, "emitter": emitter}


def run(args):
    checker = CHECK.Checker()
    helpers = {name: checker.identity(Path(__file__).parent / name) for name in
               ("check-ee290-vcs-observation.py", "capture-ee290-vcs-observation.py", "check-ee290-bounded-applicability.py",
                "check-atlascore-sram-observation.py", "observe-ee290-vls.py", "check-ee290-provenance.py",
                "check-ee290-source-correspondence.py", "fingerprint-rtl-modules.py", "index-retained-hw.py", "elaborate-ee290-source.py", "ee290_build_capture.py")}
    selected = {name: select(checker, getattr(args, name.replace("-", "_")), getattr(args, "expected_" + name.replace("-", "_") + "_sha256")) for name in ROOT_SELECTIONS}
    pair_id, pair = selected["pair"]
    base = Path(pair_id["path"]).parent
    require(pair.get("schema") == "atlas.ee290_vcs_observation_pair.v0" and pair.get("target_config") == "EE290SimConfig" and pair.get("scheduling_qualified") is False, "pair schema/qualification mismatch")
    require(pair.get("state") == "both_numerical_and_boundary_observations_passed", "capture pair is incomplete")
    historical(checker, pair["launcher"], checker.identity(base / "launcher.executed.py"), base, "pair launcher snapshot")
    numerical_id, numerical = read_member(checker, pair["selected_pair"], base)
    require(numerical.get("target_config") == "EE290SimConfig" and numerical.get("scheduling_qualified") is False, "numerical pair target/qualification mismatch")
    require(numerical.get("state") == "both_execution_witnesses_passed", "selected numerical pair is incomplete")
    content(numerical["simulator_build"]["receipt"], pair["selected_build_phase"], "numerical/capture build phase")
    content(numerical["simulator"], pair["simulator"], "numerical/capture pair simulator")
    phase_id = checker.verify(pair["selected_build_phase"], base)
    phase = BOUND.phase(checker, phase_id, "vcs_compile")
    compiler_id, compiler = read_member(checker, numerical["compiler_receipt"], base)
    require(compiler.get("schema") == "atlas.compiler_vls_witness.v0" and compiler.get("qualification") == "conditional", "conditional compiler witness required")
    require(len(pair["arms"]) == 2 and {row["arm"] for row in pair["arms"]} == {"baseline", "scheduled"}, "pair requires exactly baseline/scheduled arms")
    arms = {}
    for row in pair["arms"]:
        name = row["arm"]
        plan_id = selected[name + "-plan"][0]
        capture_id = selected[name + "-capture"][0]
        require(row["plan"] == plan_id and row["receipt"] == capture_id, "independently selected plan/capture differs from pair")
        arms[name] = verify_arm(checker, name, plan_id, capture_id, phase, numerical, compiler)
        require(arms[name]["build_phase"] == phase_id, "pair/arm captured build mismatch")
        content(arms[name]["simulator"], pair["simulator"], "pair/arm simulator")
        content(arms[name]["capture_producer_snapshot"], pair["capture_tool"], "pair/arm capture producer")
    require(arms["baseline"]["runtime_environment"] == arms["scheduled"]["runtime_environment"], "pair runtime environments differ")
    output = OBS._fresh_output(args.output)
    output.mkdir(parents=True)
    source = None
    packet_path = getattr(args, "bounded_applicability", None)
    packet_sha = getattr(args, "expected_bounded_applicability_sha256", None)
    require(bool(packet_path) == bool(packet_sha), "bounded applicability path/hash must be paired")
    if packet_path:
        packet_id, packet = select(checker, packet_path, packet_sha)
        source = source_bridge(checker, packet_id, packet, arms, compiler_id)
    emitter_path = getattr(args, "atlas_emit", None)
    emitter_sha = getattr(args, "expected_atlas_emit_sha256", None)
    require(bool(emitter_path) == bool(emitter_sha), "emitter path/hash must be paired")
    if emitter_path:
        require(source is not None, "resolver event binding requires selected bounded source/HW evidence")
        emitter = checker.identity(emitter_path)
        require(emitter["sha256"] == emitter_sha, "selected emitter digest mismatch")
        for name, arm in arms.items():
            arm_output = output / name
            arm_output.mkdir()
            arm["resolver_export"] = export_binding(checker, emitter, arm, source["evidence"], arm_output)
    if getattr(args, "include_system_memory", False):
        memory = load("integrated_system_memory", "check-ee290-system-memory-observation.py")
        helpers["check-ee290-system-memory-observation.py"] = checker.identity(Path(memory.__file__))
        for arm in arms.values():
            recomputed = memory.analyze_trace(Path(arm["trace"]["path"]), arm["scope"])
            require(recomputed.get("schema") == "atlas.ee290_system_memory_events.v0" and recomputed.get("target_config") == "EE290SimConfig" and recomputed.get("scheduling_qualified") is False, "unsupported system memory interpretation")
            require(recomputed["boundary_observations"] == arm["boundary_observations"], "memory/boundary decoders disagree")
            arm["system_memory_observations"] = recomputed
    snapshots = {}
    for name, identity in helpers.items():
        destination = output / (name.removesuffix(".py") + ".executed.py")
        destination.write_bytes(CHECK.allowed(identity["path"]).read_bytes())
        snapshots[name] = checker.identity(destination)
        content(identity, snapshots[name], "validator snapshot")
    for arm in arms.values():
        arm.pop("source_inventory")
    result = {"schema": "atlas.ee290_integrated_observation_check.v0", "target_config": "EE290SimConfig",
              "state": "finite_integrated_observations_validated", "qualification": "finite_observation_validation",
              "scheduling_qualified": False, "system_qualified": False, "operand_domain_qualified": False,
              "source_to_simulation_complete": False,
              "coverage": {"capture_record_links": True, "boundary_events": True,
                           "source_hw_content_bridge": source is not None, "fresh_resolver_event_binding": bool(emitter_path),
                           "system_memory_windows": bool(getattr(args, "include_system_memory", False))}, "selection": {name: member for name, (member, _) in selected.items()},
              "selected_numerical_pair": numerical_id, "compiler_witness": compiler_id, "build_phase": phase_id,
              "source_bridge": source, "arms": arms, "validators": helpers, "validator_snapshots": snapshots,
              "limitations": ["Finite captured programs and three panels per arm only; no universal bank/address/MREG/domain qualification.",
                              "Captured commands and exact artifact links are validated; opaque runtime/framework dependencies remain outside complete hermeticity.",
                              "FIRRTL/HW/SV content correspondence does not prove the unrecorded Verilog-generation execution edge; external behavioral memories remain conditional.",
                              "Boundary checks use settled pre-edge values; optional SRAM/competition evidence retains its independently stated window and clock limits.",
                              "Host numerical CSR counts and process wall time are not whole-kernel timing measurements.",
                              "This packet does not change compiler/provider qualification or enable unsupported instructions/concurrency."]}
    BOUND.publish_report(checker, output / "report.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ROOT_SELECTIONS:
        parser.add_argument("--" + name, required=True, type=Path)
        parser.add_argument("--expected-" + name + "-sha256", required=True)
    parser.add_argument("--bounded-applicability", type=Path)
    parser.add_argument("--expected-bounded-applicability-sha256")
    parser.add_argument("--atlas-emit", type=Path)
    parser.add_argument("--expected-atlas-emit-sha256")
    parser.add_argument("--include-system-memory", action="store_true")
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    try:
        result = run(args)
        print(json.dumps({"state": result["state"], "report": str(args.output / "report.json"), "scheduling_qualified": False}))
    except (ValueError, KeyError, TypeError, OSError, RuntimeError, subprocess.TimeoutExpired) as exc:
        parser.exit(1, "integrated observation rejected: " + str(exc) + "\n")


if __name__ == "__main__":
    main()
