#!/usr/bin/env python3
"""Retain the selected EE290 FIRRTL as HW/Comb/Seq with recorded provenance.

Supply the elaborated FIRRTL, final annotation sidecar, and lowering options.
This does not invoke Make, elaborate Chisel, or execute the simulator.
Output must be a new directory outside .agents and .codex.
"""

import argparse
import hashlib
import json
import re
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path


def allowed(path):
    path = Path(path).absolute()
    if any("hammer" in p.lower() or "vlsi" in p.lower() for p in path.parts):
        raise ValueError("Prohibited path component")
    if any("hammer" in p.lower() or "vlsi" in p.lower() for p in path.resolve().parts):
        raise ValueError("Prohibited symlink target component")
    return path


def sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def identity(path):
    return {"path": str(path), "sha256": sha(path), "bytes": path.stat().st_size}


def check_annotations(annotations):
    for annotation in annotations:
        cls = annotation.get("class", "")
        if "BlackBox" in cls:
            if cls != "firrtl.transforms.BlackBoxInlineAnno":
                raise ValueError("External blackbox input requires an explicit input audit")
            name = annotation["name"]
            if Path(name).name != name or name in (".", ".."):
                raise ValueError("Inline blackbox output must be a basename")
            allowed(name)
        for key, value in annotation.items():
            if key.lower() in ("filename", "path", "directory", "targetdir"):
                if cls not in HIERARCHY:
                    raise ValueError(f"Unreviewed annotation path: {cls}/{key}")
                allowed(value)


HIERARCHY = {
    "sifive.enterprise.firrtl.TestHarnessHierarchyAnnotation": "model_module_hierarchy.json",
    "sifive.enterprise.firrtl.ModuleHierarchyAnnotation": "top_module_hierarchy.json",
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--firrtl", type=Path, required=True, help="Elaborated FIRRTL input")
    parser.add_argument("--annotations", type=Path, required=True,
                        help="Final/appended firtool annotation sidecar")
    parser.add_argument("--lowering-options", type=Path, required=True,
                        help="File containing firtool lowering options")
    parser.add_argument("--chisel-annotations", type=Path,
                        help="Optional original Chisel annotations, retained for provenance")
    parser.add_argument("--firtool", default="firtool", help="Executable path or PATH command (default: firtool)")
    parser.add_argument("--circt-opt", default="circt-opt", help="Executable path or PATH command (default: circt-opt)")
    parser.add_argument("--source-config", help="Optional provenance label for the supplied elaboration")
    parser.add_argument("--output", type=Path, required=True, help="New artifact directory; must not already exist")
    parser.add_argument("--expected-firrtl-sha256", required=True, help="Expected SHA-256 of the selected FIRRTL")
    args = parser.parse_args()
    launch_cwd = allowed(Path.cwd())
    output = allowed(args.output)
    if any(part in (".agents", ".codex")
           for path in (output, output.resolve()) for part in path.parts):
        raise ValueError("Generated evidence belongs outside agent notes/configuration")
    inputs = {"firrtl": args.firrtl, "annotations": args.annotations,
              "lowering_options": args.lowering_options}
    if args.chisel_annotations is not None:
        inputs["chisel_annotations"] = args.chisel_annotations
    inputs = {key: allowed(path) for key, path in inputs.items()}
    before = {key: identity(path) for key, path in inputs.items()}
    if before["firrtl"]["sha256"] != args.expected_firrtl_sha256:
        raise ValueError("Selected FIRRTL hash mismatch; refusing to lower a different elaboration")
    def executable(value):
        if "/" in value:
            return allowed(value)
        # Validate the command name before allowing PATH lookup.
        allowed(value)
        found = shutil.which(value)
        if found is None:
            raise ValueError(f"Executable not found on PATH: {value}")
        return allowed(found)

    firtool = executable(args.firtool)
    circt_opt = executable(args.circt_opt)
    tools = {}
    for label, tool in (("firtool", firtool), ("circt_opt", circt_opt)):
        tools[label] = identity(tool)
        tools[label]["version"] = subprocess.check_output(
            [str(tool), "--version"], text=True, cwd=launch_cwd).strip()
    sidecar = json.loads(inputs["annotations"].read_text())
    check_annotations(sidecar)
    fir = inputs["firrtl"].read_text()
    marker = fir.find("%[")
    embedded = []
    if marker >= 0:
        annotation_start = marker + 2
        while annotation_start < len(fir) and fir[annotation_start].isspace():
            annotation_start += 1
        embedded, end = json.JSONDecoder().raw_decode(fir, annotation_start)
        check_annotations(embedded)
        if fir[end:end + 1] != "]":
            raise ValueError("Unexpected embedded FIRRTL annotation boundary")
        if any(a.get("class") in HIERARCHY for a in embedded):
            raise ValueError("Embedded output path needs an explicit preparation rule")
    del fir, embedded
    output.mkdir(parents=True, exist_ok=False)
    producer = output / "retain-ee290-hw-ir.executed.py"
    shutil.copyfile(allowed(__file__), producer)
    snapshot = output / "inputs"
    snapshot.mkdir()
    saved = {}
    snapshot_names = {"firrtl": "input.fir", "annotations": "original.anno.json",
                      "chisel_annotations": "chisel.anno.json",
                      "lowering_options": "lowering-options.txt"}
    for key, path in inputs.items():
        target = snapshot / snapshot_names[key]
        shutil.copyfile(path, target)
        saved[key] = identity(target)
        if saved[key]["sha256"] != before[key]["sha256"]:
            raise ValueError("Input changed while taking the snapshot")
    sidecar = json.loads(Path(saved["annotations"]["path"]).read_text())
    redirects = []
    for annotation in sidecar:
        cls = annotation.get("class")
        if cls in HIERARCHY:
            new = str(output / HIERARCHY[cls])
            redirects.append({"class": cls, "before": annotation["filename"], "after": new})
            annotation["filename"] = new
    prepared = snapshot / "retained-hw.anno.json"
    prepared.write_text(json.dumps(sidecar, indent=2) + "\n")
    hw = output / "ee290.hw.mlir"
    cmd = [str(firtool), "--format=fir", "--export-module-hierarchy",
           "--verify-each=true", "--warn-on-unprocessed-annotations",
           "--disable-annotation-classless", "--disable-annotation-unknown",
           "--mlir-timing", "--mlir-print-debuginfo",
           "--lowering-options=" + Path(saved["lowering_options"]["path"]).read_text().strip(),
           "--repl-seq-mem", "--repl-seq-mem-file=" + str(output / "mems.conf"),
           "--annotation-file=" + str(prepared), "--ir-hw",
           "-o", str(hw), saved["firrtl"]["path"]]
    manifest = {
        "schema": "atlas.retained_hw_ir.v0",
        "target_config": "EE290SimConfig",
        "source_config": args.source_config,
        "target_reference": "https://github.com/ucb-ee194-tapeout/bringup-chipyard/blob/main/generators/chipyard/src/main/scala/config/AtlasConfigs.scala#L12-L25",
        "invocation": {"argv": [sys.executable, str(allowed(__file__)), *sys.argv[1:]],
                       "cwd": str(launch_cwd)},
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "state": "started", "producer": identity(producer),
        "inputs": before, "snapshots": saved,
        "prepared_annotations": identity(prepared), "annotation_redirects": redirects,
        "tools": tools, "commands": [],
        "differences_from_vcs_lowering": [
            "Emit --ir-hw instead of --split-verilog, retaining sequential operations.",
            "Print MLIR debug locations for evidence locators.",
            "Redirect hierarchy and memory metadata into this new artifact directory.",
            "Use byte-identical FIRRTL/annotation snapshots; hierarchy output paths alone are rewritten.",
        ],
        "limitations": [
            "Artifact derivation and IR verification do not qualify instruction timing or execution.",
            "Sequential-memory replacement matches the VCS policy; external memory/blackbox behavior is not an internal HW graph.",
            "The full original Chisel source/tool environment is not reconstructed by this FIRRTL lowering.",
        ],
    }
    manifest_path = output / "manifest.json"

    def save():
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")

    def run(label, command):
        log = output / (label + ".log")
        print(label + ": " + str(log), flush=True)
        start = time.monotonic()
        with log.open("w") as stream:
            result = subprocess.run(command, cwd=output, stdout=stream, stderr=subprocess.STDOUT)
        manifest["commands"].append({"stage": label, "argv": command, "cwd": str(output),
                                     "returncode": result.returncode,
                                     "elapsed_seconds": time.monotonic() - start,
                                     "log": identity(log)})
        save()
        if result.returncode:
            manifest["state"] = label + "_failed"
            save()
            raise RuntimeError(f"{label} failed; inspect {log}")

    save()
    run("lower", cmd)
    manifest["hardware_ir"] = identity(hw)
    run("verify", [str(circt_opt), str(hw), "--verify-each", "-o", "/dev/null"])
    text = hw.read_text()
    seq = {op: len(re.findall(r"(?<![\w.])" + re.escape(op) + r"(?=\s)", text))
           for op in ("seq.firreg", "seq.compreg", "seq.firmem")}
    modules = re.findall(r"^\s*hw\.module\s+(?:(?:private|public)\s+)?@([\w.$-]+)", text, re.MULTILINE)
    manifest["structure"] = {"sequential_op_occurrences": seq,
                             "module_count": len(modules),
                             "required_modules": {name: name in modules for name in
                                                  ("AtlasCore", "ScalarCore", "DmaEngine")}}
    if not (seq["seq.firreg"] + seq["seq.compreg"]) or not all(manifest["structure"]["required_modules"].values()):
        manifest["state"] = "structure_check_failed"
        save()
        raise RuntimeError("Missing expected sequential operations or Atlas modules")
    after = {key: identity(path) for key, path in inputs.items()}
    manifest["original_inputs_unchanged"] = after == before
    if after != before:
        manifest["state"] = "input_changed"
        save()
        raise RuntimeError("Original elaboration inputs changed during extraction")
    manifest["state"] = "verified"
    save()
    print(json.dumps({"manifest": str(manifest_path), "hardware_ir": manifest["hardware_ir"],
                      "structure": manifest["structure"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
