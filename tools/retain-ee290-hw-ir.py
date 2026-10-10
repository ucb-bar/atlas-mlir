#!/usr/bin/env python3
"""Lower the elaborated EE290 FIRRTL to retained HW/Comb/Seq IR (`firtool --ir-hw`) plus a manifest.

Supply the elaborated FIRRTL, its final annotation sidecar and the firtool lowering options.
This does not elaborate Chisel or run a simulator. `--output` must be a new directory.
"""
import argparse
import json
import re
import subprocess
import sys
import time
from pathlib import Path

HIERARCHY = {
    "sifive.enterprise.firrtl.TestHarnessHierarchyAnnotation": "model_module_hierarchy.json",
    "sifive.enterprise.firrtl.ModuleHierarchyAnnotation": "top_module_hierarchy.json",
}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--firrtl", type=Path, required=True, help="Elaborated FIRRTL input")
    p.add_argument("--annotations", type=Path, required=True, help="Final firtool annotation sidecar")
    p.add_argument("--lowering-options", type=Path, required=True, help="File containing the firtool lowering options")
    p.add_argument("--firtool", default="firtool")
    p.add_argument("--circt-opt", default="circt-opt")
    p.add_argument("--output", type=Path, required=True, help="New artifact directory")
    args = p.parse_args()
    out = args.output.absolute()
    out.mkdir(parents=True, exist_ok=False)
    sidecar = json.loads(args.annotations.read_text())
    for annotation in sidecar:  # write hierarchy side files into the artifact directory
        if annotation.get("class") in HIERARCHY:
            annotation["filename"] = str(out / HIERARCHY[annotation["class"]])
    prepared = out / "retained-hw.anno.json"
    prepared.write_text(json.dumps(sidecar, indent=2) + "\n")
    hw = out / "ee290.hw.mlir"
    manifest = {"schema": "atlas.retained_hw_ir.v0", "target_config": "EE290SimConfig", "state": "started",
                "inputs": {"firrtl": str(args.firrtl.absolute()), "annotations": str(args.annotations.absolute()),
                           "lowering_options": str(args.lowering_options.absolute())},
                "hardware_ir": str(hw), "commands": []}
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

    run("lower", [args.firtool, "--format=fir", "--export-module-hierarchy", "--verify-each=true",
                  "--warn-on-unprocessed-annotations", "--disable-annotation-classless", "--disable-annotation-unknown",
                  "--mlir-print-debuginfo", "--lowering-options=" + args.lowering_options.read_text().strip(),
                  "--repl-seq-mem", "--repl-seq-mem-file=" + str(out / "mems.conf"),
                  "--annotation-file=" + str(prepared), "--ir-hw", "-o", str(hw), str(args.firrtl)])
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
