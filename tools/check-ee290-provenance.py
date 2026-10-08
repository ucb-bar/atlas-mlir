#!/usr/bin/env python3
"""Check recorded EE290SimConfig artifact links without certifying a build.

Select a verified retention manifest and successful integrated witness receipts
by their independent SHA-256 identities. An optional current source inventory
is an observation only. Saved build commands never establish source-to-build
linkage. All checks precede creation of a new ignored output directory.
"""

import argparse
from collections import deque
import copy
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import stat
import subprocess
import sys


class ProvenanceError(ValueError):
    pass


def restricted(path):
    return any(any(word in part.lower() for word in ("hammer", "vlsi"))
               for part in Path(path).parts)


def allowed(path, *, missing=False):
    """Resolve components without traversing unchecked symlink targets."""
    path = Path(path)
    if restricted(path):
        raise ProvenanceError("restricted path component")
    if not path.is_absolute():
        cwd = Path.cwd()
        if restricted(cwd):
            raise ProvenanceError("restricted working directory")
        path = cwd / path
    pending = deque(path.parts[1:])
    resolved = Path(path.anchor)
    links = set()
    while pending:
        part = pending.popleft()
        if part == ".":
            continue
        if part == "..":
            resolved = resolved.parent
            continue
        candidate = resolved / part
        if restricted(candidate):
            raise ProvenanceError("restricted path component")
        try:
            kind = candidate.lstat().st_mode
        except FileNotFoundError:
            if not missing:
                raise ProvenanceError("missing artifact: " + str(candidate))
            resolved = candidate
            continue
        if stat.S_ISLNK(kind):
            if candidate in links or len(links) >= 64:
                raise ProvenanceError("recursive artifact symlink")
            links.add(candidate)
            target = Path(os.readlink(candidate))
            if restricted(target):
                raise ProvenanceError("restricted symlink target")
            if target.is_absolute():
                resolved = Path(target.anchor)
                pending.extendleft(reversed(target.parts[1:]))
            else:
                pending.extendleft(reversed(target.parts))
        else:
            resolved = candidate
    return resolved


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def hash_string(value):
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def object_value(value, label):
    if not isinstance(value, dict):
        raise ProvenanceError("missing/malformed object: " + label)
    return value


def list_value(value, label, *, nonempty=False):
    if not isinstance(value, list) or (nonempty and not value):
        raise ProvenanceError("missing/malformed list: " + label)
    return value


def require(value, expected, label):
    if type(value) is not type(expected) or json.dumps(value, sort_keys=True) != json.dumps(expected, sort_keys=True):
        raise ProvenanceError("unsupported/mismatched " + label)


def strict_json(data):
    def unique(items):
        out = {}
        for key, value in items:
            if key in out:
                raise ProvenanceError("duplicate JSON key: " + key)
            out[key] = value
        return out
    try:
        return json.loads(data, object_pairs_hook=unique,
                          parse_constant=lambda value: (_ for _ in ()).throw(
                              ProvenanceError("nonfinite JSON value: " + value)))
    except (ValueError, UnicodeError) as error:
        raise ProvenanceError("invalid receipt JSON: " + str(error)) from error


class Checker:
    def __init__(self):
        self.checked = {}

    def identity(self, path):
        path = allowed(path)
        if path in self.checked:
            return self.checked[path]
        if not stat.S_ISREG(path.stat().st_mode):
            raise ProvenanceError("artifact must be a regular file: " + str(path))
        digest, size = hashlib.sha256(), 0
        with path.open("rb") as source:
            for block in iter(lambda: source.read(1024 * 1024), b""):
                digest.update(block)
                size += len(block)
        result = {"path": str(path), "sha256": digest.hexdigest(), "bytes": size}
        self.checked[path] = result
        return result

    def verify(self, member, base):
        member = object_value(member, "artifact identity")
        if not isinstance(member.get("path"), str) or not member["path"] or \
                not hash_string(member.get("sha256")) or \
                type(member.get("bytes")) is not int or member["bytes"] < 0:
            raise ProvenanceError("malformed artifact identity")
        path = Path(member["path"])
        actual = self.identity(path if path.is_absolute() else base / path)
        if (actual["sha256"], actual["bytes"]) != (member["sha256"], member["bytes"]):
            raise ProvenanceError("artifact hash/size mismatch: " + member["path"])
        return actual

    def recursively_verify(self, value, base):
        if isinstance(value, dict):
            if any(key in value for key in ("build_receipt", "build_receipts", "certified_build_receipt")):
                raise ProvenanceError("build receipts are unsupported; no validated build-capture workflow exists")
            if any(key in value for key in ("path", "sha256", "bytes")):
                self.verify(value, base)
            for child in value.values():
                self.recursively_verify(child, base)
        elif isinstance(value, list):
            for child in value:
                self.recursively_verify(child, base)

    def selected(self, path, expected):
        if not hash_string(expected):
            raise ProvenanceError("an explicit SHA-256 is required for each selected receipt")
        path = allowed(path)
        identity = self.identity(path)
        if identity["sha256"] != expected:
            raise ProvenanceError("selected receipt SHA-256 mismatch: " + str(path))
        data = path.read_bytes()
        if sha256(data) != expected:
            raise ProvenanceError("selected receipt changed while reading")
        value = object_value(strict_json(data), "selected receipt")
        self.recursively_verify(value, path.parent)
        return path, identity, data, value

    def matching(self, left, left_base, right, right_base, label):
        a, b = self.verify(left, left_base), self.verify(right, right_base)
        if (a["sha256"], a["bytes"]) != (b["sha256"], b["bytes"]):
            raise ProvenanceError("crosslink mismatch: " + label)

    def file_set(self, members, base, label, *, nonempty=True):
        result = []
        for member in list_value(members, label, nonempty=nonempty):
            actual = self.verify(member, base)
            result.append((actual["path"], actual["sha256"], actual["bytes"]))
        if len({member[0] for member in result}) != len(result):
            raise ProvenanceError("duplicate artifact in " + label)
        return sorted(result)

    def argument(self, value, cwd, expected, label):
        if not isinstance(value, str):
            raise ProvenanceError("malformed command argument: " + label)
        path = Path(value)
        actual = allowed(path if path.is_absolute() else cwd / path)
        selected = Path(self.verify(expected, cwd)["path"])
        if actual != selected:
            raise ProvenanceError("command does not use selected " + label)

    def recheck(self):
        prior = self.checked
        self.checked = {}
        for path, expected in prior.items():
            if self.identity(path) != expected:
                raise ProvenanceError("artifact changed during provenance checking: " + str(path))


def command(checker, item, base, label, *, timeout_field=False):
    item = object_value(item, label)
    require(item.get("returncode"), 0, label + " returncode")
    if timeout_field:
        require(item.get("timed_out"), False, label + " timeout")
    argv = list_value(item.get("argv"), label + " argv", nonempty=True)
    if not all(isinstance(value, str) and value for value in argv):
        raise ProvenanceError("malformed command argv")
    cwd_value = item.get("cwd")
    if not isinstance(cwd_value, str):
        raise ProvenanceError("missing command working directory")
    cwd_path = Path(cwd_value)
    cwd = allowed(cwd_path if cwd_path.is_absolute() else base / cwd_path)
    if not stat.S_ISDIR(cwd.stat().st_mode):
        raise ProvenanceError("command working directory is not a directory")
    checker.verify(item.get("log"), base)
    return item, argv, cwd


def retention(checker, manifest, base):
    require(manifest.get("schema"), "atlas.retained_hw_ir.v0", "retention schema")
    require(manifest.get("target_config"), "EE290SimConfig", "retention target")
    require(manifest.get("state"), "verified", "retention state")
    require(manifest.get("original_inputs_unchanged"), True, "retention source stability")
    inputs = object_value(manifest.get("inputs"), "retention inputs")
    snapshots = object_value(manifest.get("snapshots"), "retention snapshots")
    if set(inputs) != set(snapshots) or not {"firrtl", "annotations", "lowering_options"} <= set(inputs):
        raise ProvenanceError("retention input/snapshot coverage mismatch")
    for name in inputs:
        checker.matching(inputs[name], base, snapshots[name], base, "retention " + name)
    tools = object_value(manifest.get("tools"), "retention tools")
    rows = list_value(manifest.get("commands"), "retention commands", nonempty=True)
    stages = {}
    for row in rows:
        row = object_value(row, "retention command")
        stage = row.get("stage")
        if not isinstance(stage, str) or stage in stages:
            raise ProvenanceError("missing/duplicate retention command stage")
        stages[stage] = row
    if set(stages) != {"lower", "verify"}:
        raise ProvenanceError("missing recorded FIRRTL lowering or HW verification")
    _, lower, cwd = command(checker, stages["lower"], base, "lower")
    lowering_cwd = cwd
    checker.argument(lower[0], cwd, tools.get("firtool"), "firtool")
    if not {"--format=fir", "--ir-hw", "--verify-each=true"} <= set(lower):
        raise ProvenanceError("unsupported retained-HW lowering command")
    checker.argument(lower[-1], cwd, snapshots["firrtl"], "FIRRTL snapshot")
    annotation_args = [arg[len("--annotation-file="):] for arg in lower
                       if arg.startswith("--annotation-file=")]
    if len(annotation_args) != 1:
        raise ProvenanceError("missing/ambiguous prepared annotation argument")
    checker.argument(annotation_args[0], cwd, manifest.get("prepared_annotations"), "prepared annotations")
    options = checker.verify(snapshots["lowering_options"], base)
    expected_options = "--lowering-options=" + Path(options["path"]).read_text().strip()
    option_args = [arg for arg in lower if arg.startswith("--lowering-options=")]
    if option_args != [expected_options]:
        raise ProvenanceError("lowering command options differ from selected snapshot")
    original = checker.verify(snapshots["annotations"], base)
    prepared = checker.verify(manifest.get("prepared_annotations"), base)
    annotations = list_value(strict_json(Path(original["path"]).read_bytes()), "annotation sidecar")
    expected_annotations = copy.deepcopy(annotations)
    expected_redirects = []
    hierarchy = {"sifive.enterprise.firrtl.TestHarnessHierarchyAnnotation": "model_module_hierarchy.json",
                 "sifive.enterprise.firrtl.ModuleHierarchyAnnotation": "top_module_hierarchy.json"}
    for annotation in expected_annotations:
        annotation = object_value(annotation, "annotation entry")
        cls = annotation.get("class")
        if cls in hierarchy:
            before = annotation.get("filename")
            if not isinstance(before, str):
                raise ProvenanceError("malformed hierarchy annotation filename")
            after = str(allowed(cwd / hierarchy[cls], missing=True))
            expected_redirects.append({"class": cls, "before": before, "after": after})
            annotation["filename"] = after
    require(manifest.get("annotation_redirects"), expected_redirects, "recorded annotation redirects")
    require(strict_json(Path(prepared["path"]).read_bytes()), expected_annotations, "prepared annotation transformation")
    if lower.count("-o") != 1 or lower.index("-o") + 1 >= len(lower):
        raise ProvenanceError("missing lowering output")
    checker.argument(lower[lower.index("-o") + 1], cwd, manifest.get("hardware_ir"), "retained HW")
    _, verify, cwd = command(checker, stages["verify"], base, "verify")
    if cwd != lowering_cwd:
        raise ProvenanceError("retention commands have divergent working directories")
    checker.argument(verify[0], cwd, tools.get("circt_opt"), "circt-opt")
    if len(verify) < 3 or "--verify-each" not in verify:
        raise ProvenanceError("unsupported HW verification command")
    checker.argument(verify[1], cwd, manifest.get("hardware_ir"), "verified HW")


def witness(checker, report, base, manifest, manifest_base, manifest_identity):
    require(report.get("schema"), "atlas.ee290_vls_witness.v0", "witness schema")
    require(report.get("target_config"), "EE290SimConfig", "witness target")
    require(report.get("state"), "integrated_execution_passed", "witness state")
    require(report.get("integrated_execution_passed"), True, "integrated execution")
    require(report.get("scheduling_qualified"), False, "witness scheduling qualification")
    provenance = object_value(report.get("provenance"), "witness provenance")
    checker.matching(provenance.get("manifest"), base, manifest_identity, manifest_base, "selected manifest")
    checker.matching(provenance.get("hardware_ir"), base, manifest.get("hardware_ir"), manifest_base, "selected HW")
    inputs = object_value(provenance.get("inputs"), "witness inputs")
    if set(inputs) != set(manifest["inputs"]):
        raise ProvenanceError("witness/retention input coverage mismatch")
    for name in inputs:
        checker.matching(inputs[name], base, manifest["inputs"][name], manifest_base, "witness " + name)
    commands = list_value(report.get("commands"), "witness commands")
    if len(commands) != 3:
        raise ProvenanceError("expected emission, host compilation and simulation commands")
    emitted, argv, cwd = command(checker, commands[0], base, "emission", timeout_field=True)
    execution_base = cwd
    program = checker.identity(execution_base / "program.mlir")
    host_source = checker.identity(execution_base / "ee290-vls-host.executed.c")
    checker.matching(program, base, provenance.get("program"), base, "executed program snapshot")
    checker.matching(host_source, base, provenance.get("host_source"), base, "executed host snapshot")
    runner = checker.identity(execution_base / "run-ee290-vls-witness.executed.py")
    if len(argv) != 2:
        raise ProvenanceError("unsupported emission command")
    checker.argument(argv[0], cwd, provenance.get("atlas_emit"), "compiler")
    checker.argument(argv[1], cwd, program, "executed program")
    emitted_log = Path(checker.verify(emitted["log"], base)["path"])
    lines = emitted_log.read_text().splitlines()
    if not 0 < len(lines) <= 32768 or any(not re.fullmatch(r"[0-9a-fA-F]{8}", line) for line in lines):
        raise ProvenanceError("emission log is not a bounded encoded program")
    require(report.get("program_word_count"), len(lines), "program word count")
    generated = checker.verify(report.get("generated_include"), base)
    if Path(generated["path"]) != allowed(execution_base / "atlas_program.inc"):
        raise ProvenanceError("generated instruction include is not the executed host include")
    expected_include = f"#define ATLAS_PROGRAM_WORDS {len(lines)}U\nstatic const uint32_t atlas_program[] = {{\n" + \
        "".join(f"  0x{line}U,\n" for line in lines) + "};\n"
    if Path(generated["path"]).read_bytes() != expected_include.encode():
        raise ProvenanceError("generated instruction include differs from recorded emission")
    _, argv, cwd = command(checker, commands[1], base, "host compilation", timeout_field=True)
    if len(argv) < 4 or argv[-2] != "-o":
        raise ProvenanceError("unsupported host compiler output")
    if cwd != execution_base:
        raise ProvenanceError("witness commands have divergent working directories")
    checker.argument(argv[0], cwd, provenance.get("host_cc"), "host compiler")
    checker.argument(argv[-3], cwd, host_source, "executed host source")
    checker.argument(argv[-1], cwd, report.get("host_binary"), "host binary")
    launched, argv, cwd = command(checker, commands[2], base, "simulation", timeout_field=True)
    if cwd != execution_base:
        raise ProvenanceError("witness commands have divergent working directories")
    checker.argument(argv[0], cwd, provenance.get("simulator"), "simulator")
    checker.argument(argv[-1], cwd, report.get("host_binary"), "simulated host binary")
    loaded = [arg[len("+loadmem="):] for arg in argv if arg.startswith("+loadmem=")]
    if len(loaded) != 1:
        raise ProvenanceError("missing/ambiguous simulator loadmem binary")
    checker.argument(loaded[0], cwd, report.get("host_binary"), "loaded host binary")
    simulation = Path(checker.verify(launched["log"], base)["path"]).read_text(errors="replace")
    log_lines = simulation.splitlines()
    panel_lines = [line for line in log_lines if line.startswith("EE290_VLS_PANEL_PASSED")]
    panels = [re.fullmatch(r"EE290_VLS_PANEL_PASSED panel=([0-9]+) output_words=256 preserved_words=1280", line)
              for line in panel_lines]
    observed_lines = [line for line in log_lines if line.startswith("EE290_VLS_OBSERVED")]
    observed = [re.fullmatch(r"EE290_VLS_OBSERVED panel=([0-9]{1,10}) status=([0-9]{1,10}) marker=([0-9]{1,10}) "
                             r"illegal_pc=([0-9]{1,10}) observer_cycles=([0-9]{1,10}) retired=([0-9]{1,10}) polls=([0-9]{1,10})", line)
                for line in observed_lines]
    if len(observed) != 3 or not all(observed):
        raise ProvenanceError("missing/malformed simulator terminal observations")
    values = [tuple(map(int, match.groups())) for match in observed]
    if sorted(row[0] for row in values) != [0, 1, 2] or any(
            max(row) > 0xffffffff or row[1] != 5 or row[2] != 1 or row[3] != 0 or row[6] >= 100000 for row in values):
        raise ProvenanceError("simulator terminal/marker/poll observations contradict success")
    summaries = [line for line in log_lines if line.startswith("EE290_VLS_PASSED")]
    if len(panels) != 3 or not all(panels) or sorted(match[1] for match in panels) != ["0", "1", "2"] or \
            summaries != ["EE290_VLS_PASSED panels=3"] or \
            re.search(r"EE290_VLS_FAILED|EE290_VLS_MISMATCH|assertion failed|\$fatal|fatal:|error:|error-", simulation, re.I):
        raise ProvenanceError("simulator log does not establish recorded numerical/guard success")
    observations = [line for line in log_lines if line.startswith("EE290_VLS_")]
    require(report.get("observations"), observations, "recorded simulator observations")
    shared = {name: checker.verify(provenance.get(name), base) for name in
              ("simulator", "atlas_emit", "host_cc", "host_source")}
    shared["simulator_sources"] = checker.file_set(provenance.get("simulator_sources"), base, "simulator sources")
    shared["simulator_archive_libraries"] = checker.file_set(provenance.get("simulator_archive_libraries"), base, "simulator archives")
    shared["observed_build_record_direct_sources"] = checker.file_set(
        provenance.get("observed_build_record_direct_sources", []), base, "observed direct sources", nonempty=False)
    if "observed_simulator_build_record" in provenance:
        shared["observed_simulator_build_record"] = checker.verify(provenance["observed_simulator_build_record"], base)
    shared["uninterpreted_source_list_options"] = list_value(
        provenance.get("uninterpreted_source_list_options"), "uninterpreted source options")
    return shared, {"executed_program": program, "executed_host_source": host_source,
                    "observed_runner_snapshot": runner, "generated_include": generated,
                    "host_binary": checker.verify(report["host_binary"], base),
                    "program_word_count": len(lines), "observations": observations}


def check_provenance(manifest_path, expected_manifest_sha256, witnesses, output,
                     source_inventory=None, expected_source_inventory_sha256=None):
    checker = Checker()
    root = allowed(__file__).parents[1]
    artifact_root = allowed(root / "build/rtl-timing", missing=True)
    output = allowed(output, missing=True)
    if output == artifact_root or not output.is_relative_to(artifact_root):
        raise ProvenanceError("output must be a new directory beneath ignored build/rtl-timing")
    if output.exists():
        raise ProvenanceError("output already exists")
    ignored = subprocess.run(["git", "-C", str(root), "check-ignore", "--no-index", "--quiet", "--", str(output)],
                             capture_output=True)
    if ignored.returncode:
        raise ProvenanceError("output must be ignored by this repository")
    if not witnesses:
        raise ProvenanceError("at least one selected integrated witness is required")
    manifest_path, manifest_id, manifest_bytes, manifest = checker.selected(manifest_path, expected_manifest_sha256)
    retention(checker, manifest, manifest_path.parent)
    selected = []
    shared = None
    selected_paths = set()
    for path, expected in witnesses:
        path, identity, data, report = checker.selected(path, expected)
        if path in selected_paths:
            raise ProvenanceError("duplicate selected witness")
        selected_paths.add(path)
        current, arm = witness(checker, report, path.parent, manifest, manifest_path.parent, manifest_id)
        if shared is not None and current != shared:
            raise ProvenanceError("divergent witness arms: simulator/source/archive/compiler identities differ")
        shared = current
        selected.append((identity, data, arm))
    inventory = None
    if (source_inventory is None) != (expected_source_inventory_sha256 is None):
        raise ProvenanceError("source inventory and expected SHA-256 must be supplied together")
    if source_inventory is not None:
        path, identity, data, value = checker.selected(source_inventory, expected_source_inventory_sha256)
        require(value.get("schema"), "atlas.ee290_source_inventory.v0", "source inventory schema")
        require(value.get("target_config"), "EE290SimConfig", "source inventory target")
        checker.file_set(value.get("sources"), path.parent, "observed source inventory")
        if "metadata" in value:
            object_value(value["metadata"], "source inventory metadata")
        inventory = (identity, data)
    checker_path = allowed(__file__)
    checker_id = checker.identity(checker_path)
    checker_bytes = checker_path.read_bytes()
    checker.recheck()
    output.mkdir(parents=True, exist_ok=False)
    (output / "retention-manifest.json").write_bytes(manifest_bytes)
    (output / "check-ee290-provenance.executed.py").write_bytes(checker_bytes)
    receipt = {"schema": "atlas.ee290_provenance_check.v0", "target_config": "EE290SimConfig",
               "state": "recorded_links_verified", "created_utc": datetime.now(timezone.utc).isoformat(),
               "scheduling_qualified": False, "build_linkage_complete": False,
               "selected_inputs": {"retention_manifest": manifest_id, "witness_reports": [item[0] for item in selected],
                                   "source_inventory": inventory[0] if inventory else None, "checker": checker_id},
               "shared_observed_artifacts": shared, "arms": [item[2] for item in selected],
               "edges": {
                   "source_to_firrtl": {"status": "unestablished", "reason": "No captured and validated Chisel elaboration receipt."},
                   "firrtl_to_retained_hw": {"status": "verified_recorded_execution", "scope": "Selected input/snapshot equality, successful lowering/verification commands, tool and output byte identities."},
                   "source_or_firrtl_to_simulator_build": {"status": "unestablished", "reason": "Current source/archive inventories and saved build-command text do not establish a build relationship."},
                   "program_to_emitted_words": {"status": "verified_recorded_execution", "scope": "Selected compiler, executed program snapshot, emission log and generated include equality for each arm."},
                   "simulator_numerical_execution": {"status": "verified_recorded_execution", "scope": "Selected simulator and loaded host binary; three panel outputs and guards within the initialized 6144-byte VMEM window."}},
               "limitations": ["Recorded successful executions are verified from caller-selected receipts and artifacts; this checker does not repeat them.",
                               "A public configuration name and consistent artifact hashes do not establish source/config equivalence or common build lineage.",
                               "Optional source inventory and saved simulator rebuild text are observations only.",
                               "Simulator include/preprocessor state, system runtime libraries and host linker inputs are not reconstructed.",
                               "Numerical/guard success is distinct from timing, scheduling and memory-latency qualification.",
                               "Future build receipts are unsupported until a captured and validated build workflow exists."]}
    snapshots = {"retention_manifest": checker.identity(output / "retention-manifest.json"),
                 "checker": checker.identity(output / "check-ee290-provenance.executed.py"), "witness_reports": []}
    for number, (_, data, _) in enumerate(selected):
        path = output / f"witness-{number:03d}.json"
        path.write_bytes(data)
        snapshots["witness_reports"].append(checker.identity(path))
    if inventory:
        path = output / "source-inventory.json"
        path.write_bytes(inventory[1])
        snapshots["source_inventory"] = checker.identity(path)
    receipt["snapshots"] = snapshots
    (output / "report.json").write_text(json.dumps(receipt, indent=2) + "\n")
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--expected-manifest-sha256", required=True)
    parser.add_argument("--witness-report", nargs=2, action="append", required=True, metavar=("PATH", "SHA256"))
    parser.add_argument("--source-inventory", type=Path)
    parser.add_argument("--expected-source-inventory-sha256")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        receipt = check_provenance(args.manifest, args.expected_manifest_sha256, args.witness_report,
                                   args.output, args.source_inventory, args.expected_source_inventory_sha256)
    except (OSError, ProvenanceError) as error:
        print("EE290 provenance: " + str(error), file=sys.stderr)
        return 2
    print(json.dumps({"report": str(args.output / "report.json"), "state": receipt["state"],
                      "scheduling_qualified": False, "build_linkage_complete": False}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
