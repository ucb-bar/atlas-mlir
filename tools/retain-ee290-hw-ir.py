#!/usr/bin/env python3
"""Retain verified EE290 hardware IR and its manifest in a new output directory.

Use --mode debug for named-value timing IR; legacy replaces sequential memories.
"""
import argparse
import hashlib
import json
import re
import subprocess
import sys
import time
from pathlib import Path

HIERARCHY = {"sifive.enterprise.firrtl.TestHarnessHierarchyAnnotation": "model_module_hierarchy.json",
    "sifive.enterprise.firrtl.ModuleHierarchyAnnotation": "top_module_hierarchy.json",}


def lowering_command(firtool, firrtl, prepared, options, out, mode):
    hw = out / ("ee290.debug.hw.mlir" if mode == "debug" else "ee290.hw.mlir")
    command = [firtool, "--format=fir", "--export-module-hierarchy", "--verify-each=true",
               "--warn-on-unprocessed-annotations", "--disable-annotation-classless", "--disable-annotation-unknown"]
    if mode == "debug":
        command.append("--mlir-timing")
    command += ["--mlir-print-debuginfo", "--lowering-options=" + options.strip()]
    if mode == "debug":
        command += ["-O=debug", "--preserve-values=named"]
    else:
        command += ["--repl-seq-mem", "--repl-seq-mem-file=" + str(out / "mems.conf")]
    return command + ["--annotation-file=" + str(prepared), "--ir-hw", "-o", str(hw), str(firrtl)]


def prepare_annotations(sidecar, out):
    """Redirect hierarchy files, preserving embedded blackboxes."""
    for annotation in sidecar:
        if annotation.get("class") in HIERARCHY:
            annotation["filename"] = str(out / HIERARCHY[annotation["class"]])
    return sidecar


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--firrtl", type=Path, required=True, help="Elaborated FIRRTL input")
    p.add_argument("--annotations", type=Path, required=True, help="Final firtool annotation sidecar")
    p.add_argument("--lowering-options", type=Path, required=True, help="File containing the firtool lowering options")
    p.add_argument("--firtool", default="firtool")
    p.add_argument("--circt-opt", default="circt-opt")
    p.add_argument("--output", type=Path, required=True, help="New artifact directory")
    p.add_argument("--mode", choices=("legacy", "debug"), default="legacy",
                   help="debug reproduces the named-value timing IR without sequential-memory replacement")
    args = p.parse_args()
    out = args.output.absolute()
    out.mkdir(parents=True, exist_ok=False)
    sidecar = json.loads(args.annotations.read_text())
    sidecar = prepare_annotations(sidecar, out)
    prepared = out / ("retained-hw.debug.anno.json" if args.mode == "debug" else "retained-hw.anno.json")
    prepared.write_text(json.dumps(sidecar, indent=2) + "\n")
    hw = out / ("ee290.debug.hw.mlir" if args.mode == "debug" else "ee290.hw.mlir")
    manifest = {"schema": "atlas.retained_hw_ir.v0", "target_config": "EE290SimConfig", "state": "started",
                "inputs": {"firrtl": str(args.firrtl.absolute()), "annotations": str(args.annotations.absolute()),
                           "lowering_options": str(args.lowering_options.absolute())},
                "hardware_ir": str(hw), "mode": args.mode, "commands": [],
                "input_sha256": {"firrtl": sha256(args.firrtl), "annotations": sha256(args.annotations),
                                 "lowering_options": sha256(args.lowering_options)},
                "prepared_annotations_sha256": sha256(prepared)}
    for label, tool in (("firtool", args.firtool), ("circt_opt", args.circt_opt)):
        manifest[label] = {"command": tool, "version": subprocess.check_output([tool, "--version"], text=True).strip()}

    def run(stage, argv):
        start = time.monotonic()
        with (out / f"{stage}.log").open("w") as log:
            code = subprocess.run(argv, cwd=out, stdout=log, stderr=subprocess.STDOUT).returncode
        manifest["commands"].append({"stage": stage, "argv": argv, "returncode": code, "elapsed_seconds": time.monotonic() - start})
        if code:
            manifest["state"] = stage + "_failed"
            (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
            raise RuntimeError(f"{stage} failed; see {out / (stage + '.log')}")

    run("lower", lowering_command(args.firtool, args.firrtl.absolute(), prepared,
                                  args.lowering_options.read_text(), out, args.mode))
    manifest["hardware_ir_sha256"] = sha256(hw)
    run("verify", [args.circt_opt, str(hw), "--verify-each", "-o", "/dev/null"])
    text = hw.read_text()
    seq = {op: len(re.findall(r"(?<![\w.])" + re.escape(op) + r"(?=\s)", text)) for op in ("seq.firreg", "seq.compreg", "seq.firmem")}
    modules = re.findall(r"^\s*hw\.module\s+(?:(?:private|public)\s+)?@([\w.$-]+)", text, re.MULTILINE)
    required = {name: name in modules for name in ("AtlasCore", "ScalarCore", "DmaEngine")}
    manifest["structure"] = {"sequential_op_occurrences": seq, "module_count": len(modules), "required_modules": required}
    ok = seq["seq.firreg"] + seq["seq.compreg"] and all(required.values())
    manifest["state"] = "verified" if ok else "structure_check_failed"
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    if not ok:
        raise RuntimeError("missing expected sequential operations or Atlas modules")
    print(json.dumps({"manifest": str(out / "manifest.json"), "hardware_ir": str(hw), "structure": manifest["structure"]}, indent=2))


if __name__ == "__main__":
    try:
        main()
    except (OSError, RuntimeError, subprocess.SubprocessError) as error:
        sys.exit(f"retain-ee290-hw-ir: {error}")
