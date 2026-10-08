#!/usr/bin/env python3
"""Capture fresh selected-source FIRRTL retention and compare AtlasCore closure.

The selected source report is a caller-pinned producer receipt. This tool checks
its artifact identities and independently derives/compares retained hardware;
it does not certify historical simulator binaries or qualify scheduling.
"""
import argparse
import importlib.util
import json
import math
from pathlib import Path
import re
import sys

# This sibling already implements guarded dependency and symlink traversal.
_here = Path(__file__).parent
# Bootstrap without traversing any unchecked dependency symlink.
from collections import deque
import os
import stat

def safe_bootstrap(path):
    path = Path(path)
    forbidden = lambda p: any(w in part.lower() for part in p.parts for w in ("hammer", "vlsi"))
    if forbidden(path): raise ValueError("restricted dependency")
    if not path.is_absolute(): path = Path.cwd() / path
    if forbidden(path): raise ValueError("restricted dependency")
    todo, current, seen = deque(path.parts[1:]), Path(path.anchor), set()
    while todo:
        part = todo.popleft()
        if part == ".": continue
        if part == "..": current = current.parent; continue
        candidate = current / part
        if forbidden(candidate): raise ValueError("restricted dependency")
        if stat.S_ISLNK(candidate.lstat().st_mode):
            if candidate in seen or len(seen) >= 64: raise ValueError("dependency symlink cycle")
            seen.add(candidate)
            target = Path(os.readlink(candidate))
            if forbidden(target): raise ValueError("restricted dependency target")
            if target.is_absolute(): current = Path(target.anchor); todo.extendleft(reversed(target.parts[1:]))
            else: todo.extendleft(reversed(target.parts))
        else: current = candidate
    return current

def load(name, path):
    path = safe_bootstrap(path)
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module, path

CAPTURE, CAPTURE_PATH = load("atlas_source_correspondence_capture", _here / "ee290_build_capture.py")
CHECK = CAPTURE._CHECK
Error = CAPTURE.CaptureError
FINGERPRINT, FINGERPRINT_PATH = load("atlas_source_correspondence_fingerprints", _here / "fingerprint-rtl-modules.py")
SOURCE, SOURCE_HELPER_PATH = load("atlas_source_correspondence_source_audit", _here / "elaborate-ee290-source.py")
HIERARCHY = {"sifive.enterprise.firrtl.TestHarnessHierarchyAnnotation": "model_module_hierarchy.json",
             "sifive.enterprise.firrtl.ModuleHierarchyAnnotation": "top_module_hierarchy.json"}


def annotation_policy(chisel, final):
    if not isinstance(chisel, list) or not isinstance(final, list) or final[:len(chisel)] != chisel or len(final) != len(chisel) + 3:
        raise Error("final annotations must preserve the exact Chisel prefix and add exactly three policy annotations")
    tail = final[len(chisel):]
    if tail[0] != {"class": "sifive.enterprise.firrtl.MarkDUTAnnotation", "target": "~TestHarness|ChipTop"}:
        raise Error("unexpected MarkDUT annotation policy")
    for row, (cls, basename) in zip(tail[1:], HIERARCHY.items()):
        if not isinstance(row, dict) or set(row) != {"class", "filename"} or row["class"] != cls:
            raise Error("unexpected hierarchy annotation policy")
        path = CHECK.allowed(row["filename"], missing=True)
        if path.name != basename: raise Error("unexpected hierarchy output basename")
    if Path(tail[1]["filename"]).parent != Path(tail[2]["filename"]).parent:
        raise Error("hierarchy outputs must share a directory")
    return {"chisel_annotation_count": len(chisel), "added_annotations": tail,
            "rule": "Exact Chisel prefix plus MarkDUT and two hierarchy-output annotations in reviewed order."}


def audit_annotations(rows, embedded=False):
    if not isinstance(rows, list): raise Error("annotations must be a list")
    for row in rows:
        if not isinstance(row, dict): raise Error("annotation must be an object")
        cls = row.get("class", "")
        if not isinstance(cls, str): raise Error("annotation class must be a string")
        if "BlackBox" in cls:
            if cls != "firrtl.transforms.BlackBoxInlineAnno": raise Error("external blackbox requires separate audit")
            name = row.get("name")
            if not isinstance(name, str) or Path(name).name != name or name in (".", ".."):
                raise Error("inline blackbox name must be a basename")
            CHECK.allowed(name, missing=True)
        if embedded and cls in HIERARCHY: raise Error("embedded hierarchy output requires separate preparation")
        def visit(value):
            if isinstance(value, dict):
                for key, member in value.items():
                    if key.lower() in ("filename", "path", "directory", "targetdir"):
                        if cls not in HIERARCHY or key != "filename" or value is not row:
                            raise Error("unreviewed annotation path")
                        CHECK.allowed(member, missing=True)
                    else: visit(member)
            elif isinstance(value, list):
                for member in value: visit(member)
        visit(row)


def audit_firrtl(path):
    text = CHECK.allowed(path).read_text()
    marker = text.find("%[")
    if marker >= 0:
        start = marker + 2
        while start < len(text) and text[start].isspace(): start += 1
        rows, end = json.JSONDecoder().raw_decode(text, start)
        if text[end:end + 1] != "]": raise Error("unexpected embedded annotation boundary")
        audit_annotations(rows, embedded=True)


def compare_closures(fresh, selected):
    actual = FINGERPRINT.fingerprints(fresh, ["AtlasCore"])
    expected = FINGERPRINT.fingerprints(selected, ["AtlasCore"])
    if len(expected) != 123: raise Error("selected AtlasCore closure must contain 123 reviewed definitions")
    missing, extra = sorted(set(expected)-set(actual)), sorted(set(actual)-set(expected))
    changed = sorted(k for k in expected.keys() & actual.keys() if expected[k] != actual[k])
    if missing or extra or changed:
        raise Error("AtlasCore closure mismatch: " + json.dumps({"missing": missing, "extra": extra, "changed": changed}))
    return {"root": "AtlasCore", "normalizer": "atlas.module.locations_only.v1", "module_count": len(actual),
            "all_modules_match": True, "modules": actual}


def same_content(a, b):
    if (a.get("sha256"), a.get("bytes")) != (b.get("sha256"), b.get("bytes")):
        raise Error("executed snapshot differs from declared producer/input identity")


def select_source_report(c, path, expected):
    if not CHECK.hash_string(expected): raise Error("explicit source report SHA-256 required")
    path = CHECK.allowed(path)
    identity = c.identity(path)
    data = path.read_bytes()
    if identity["sha256"] != expected or CHECK.sha256(data) != expected:
        raise Error("selected source report hash mismatch")
    return path, identity, CHECK.object_value(CHECK.strict_json(data), "source report")


def validate_source_receipt(c, path, report):
    """Reconstruct this bounded recipe using current pure validators only."""
    root = path.parent
    executed = c.identity(CHECK.allowed(root / "elaborate-ee290-source.executed.py"))
    same_content(report["producer"], executed)
    saved_inputs = c.identity(CHECK.allowed(root / "inputs.json"))
    same_content(report["selected_inputs"], saved_inputs)
    value = CHECK.strict_json(Path(saved_inputs["path"]).read_bytes())
    CHECK.require(value.get("schema"), "atlas.ee290_source_build_inputs.v0", "saved source input schema")
    if report.get("source_config") != value.get("source_config") or report.get("top_module") != "chipyard.harness.TestHarness":
        raise Error("source configuration/top mismatch")
    timeout=value.get("timeout_seconds",900)
    if type(timeout) not in (int,float) or not math.isfinite(timeout) or timeout<=0:
        raise Error("source timeout must be finite and positive")
    rows = report["sources"]
    if len(rows) != 83 or len(value["sources"]) != 83: raise Error("selected recipe requires 83 source snapshots")
    selected = {row["relative_path"]: row["identity"] for row in value["sources"]}
    names = [str(SOURCE.source_relative(row["relative_path"])) for row in rows]
    if len(set(names)) != 83 or sum(name.startswith(SOURCE.MAIN_PREFIX) for name in names) != 69 or sum(name.startswith(SOURCE.GLUE_PREFIX) for name in names) != 11 or {name for name in names if not name.startswith((SOURCE.MAIN_PREFIX,SOURCE.GLUE_PREFIX))} != SOURCE.CHIPYARD_SOURCES:
        raise Error("source snapshot coverage differs from bounded recipe")
    snapshots=[]
    for row in rows:
        same_content(selected[row["relative_path"]], row["identity"])
        member=c.verify(row["snapshot"],root); same_content(row["identity"],member)
        if member["path"] != str(root / "sources" / row["relative_path"]): raise Error("unexpected source snapshot location")
        snapshots.append(member)
    tools=report["tools"]
    if set(tools) != set(value["tools"]) or not SOURCE.TOOLS <= set(tools) or set(tools)-SOURCE.TOOLS-SOURCE.HELPERS:
        raise Error("source tool-role mismatch")
    for role,row in tools.items():
        same_content(value["tools"][role],row["selected"])
        member=c.verify(row["snapshot"],root); same_content(row["selected"],member)
        expected = root/"helpers"/role if role in SOURCE.HELPERS else root/"dependencies"/(role+".jar")
        if role != "java" and member["path"] != str(expected): raise Error("unexpected tool snapshot location")
    source_args=c.identity(CHECK.allowed(root/"sources.args"))
    if Path(source_args["path"]).read_text() != "".join(member["path"]+"\n" for member in snapshots):
        raise Error("source argument file differs from selected snapshot list")
    jre=report.get("jre_files",[])
    if jre != value.get("jre_files",[]): raise Error("JRE selection mismatch")
    for member in jre:c.verify(member,root)
    overlay=c.verify(report["overlay"],root)
    classes=SOURCE.overlay_classes(overlay["path"])
    if sorted(classes) != report["overlay_classes"] or len(classes) != report["overlay_class_count"]:
        raise Error("overlay class inventory mismatch")
    actual_phase={}
    if [row["kind"] for row in report["phases"]] != ["compile","elaborate"]: raise Error("source phase recipe mismatch")
    from collections import Counter
    def signatures(rows):
        return Counter((row["role"],row["identity"]["path"],row["identity"]["sha256"],row["identity"]["bytes"]) for row in rows)
    for row in report["phases"]:
        phase_identity=c.verify(row["receipt"],root)
        phase=CHECK.strict_json(Path(phase_identity["path"]).read_bytes()); kind=row["kind"]; phase_root=root/kind
        if phase_identity["path"] != str(phase_root/"phase.json"): raise Error("unexpected source phase receipt location")
        CHECK.require(phase.get("schema"),"atlas.ee290_captured_phase.v0","source phase schema")
        CHECK.require(phase.get("target_config"),"EE290SimConfig","source phase target")
        if phase.get("state") != "phase_completed" or phase.get("kind") != kind or phase.get("inputs_stable") is not True or phase.get("inputs_before") != phase.get("inputs_after") or phase.get("failures") != [] or phase.get("scheduling_qualified") is not False:
            raise Error("source phase is incomplete or unstable")
        for key, declared in [("producer",report["producer_dependencies"]["capture"]),("path_checker_dependency",report["producer_dependencies"]["path_checker"])]:
            saved=c.verify(phase["snapshots"][key],phase_root)
            same_content(phase[key],saved);same_content(declared,saved)
        inputs=[{"role":"tool","identity":tools["java"]["snapshot"]}]
        if kind=="compile":
            inputs += [{"role":role,"identity":member["snapshot"]} for role,member in tools.items() if role!="java"]
            inputs += [{"role":"source","identity":member} for member in snapshots]
            inputs += [{"role":"jre","identity":member} for member in jre]
            inputs += [{"role":"source_arguments","identity":source_args}]
            cp=":".join(tools[role]["snapshot"]["path"] for role in ("scala_compiler","scala_library","scala_reflect"))
            argv=[tools["java"]["snapshot"]["path"],"-Xmx8G","-cp",cp,"scala.tools.nsc.Main","-classpath",tools["chipyard_jar"]["snapshot"]["path"],"-Xplugin:"+tools["chisel_plugin"]["snapshot"]["path"],"-Xplugin-require:chiselplugin","-deprecation","-unchecked","-Ytasty-reader","-Ymacro-annotations","-d",str(phase_root/"overlay.jar"),"@"+source_args["path"]]
            expected_outputs=[("overlay","overlay.jar",overlay)]
        else:
            inputs += [{"role":"overlay","identity":overlay},{"role":"opaque_framework","identity":tools["chipyard_jar"]["snapshot"]}]
            inputs += [{"role":"jre","identity":member} for member in jre]
            inputs += [{"role":role,"identity":tools[role]["snapshot"]} for role in SOURCE.HELPERS if role in tools]
            argv=[tools["java"]["snapshot"]["path"],"-Xmx8G","-Xlog:class+load=info:file="+str(phase_root/"class-load.log"),"-cp",overlay["path"]+":"+tools["chipyard_jar"]["snapshot"]["path"],"chipyard.Generator","--target-dir",str(phase_root),"--name",SOURCE.PUBLIC_NAME,"--top-module","chipyard.harness.TestHarness","--legacy-configs",report["source_config"]]
            expected_outputs=[("firrtl",SOURCE.PUBLIC_NAME+".fir",report["firrtl"]),("annotations",SOURCE.PUBLIC_NAME+".anno.json",report["annotations"]),("class_origins","class-load.log",report["class_load_log"])]
        if signatures(inputs)!=signatures(phase["inputs_before"]):raise Error("source phase input crosslink mismatch")
        for member in phase["inputs_before"]: c.verify(member["identity"],phase_root)
        command=phase["command"]
        if command.get("argv")!=argv or command.get("cwd")!=str(phase_root) or command.get("environment")!=SOURCE.java_environment(root) or command.get("returncode")!=0 or command.get("timed_out") is not False or command.get("timeout_seconds")!=value.get("timeout_seconds",900):
            raise Error("source phase command recipe mismatch")
        elapsed=command.get("elapsed_seconds")
        if type(elapsed) not in (int,float) or not math.isfinite(elapsed) or elapsed<0:
            raise Error("source phase elapsed time must be finite and nonnegative")
        for label in ("stdout","stderr"):c.verify(command[label],phase_root)
        if len(phase["outputs"])!=len(expected_outputs):raise Error("source phase output inventory mismatch")
        declared=[{"role":role,"relative_path":relative,"kind":"file"} for role,relative,_ in expected_outputs]
        if phase.get("declared_outputs")!=declared:raise Error("source declared-output recipe mismatch")
        for actual,(role,relative,expected) in zip(phase["outputs"],expected_outputs):
            if actual.get("role")!=role or actual.get("kind")!="file" or actual.get("relative_path")!=relative or len(actual.get("files",[]))!=1:
                raise Error("source phase output role mismatch")
            saved=c.verify(actual["files"][0]["identity"],phase_root)
            if saved!=expected or saved["path"]!=str(phase_root/relative):raise Error("source phase output crosslink mismatch")
        actual_phase[kind]=phase_identity
    log=c.verify(report["class_load_log"],root)
    origins=SOURCE.audit_origins(Path(log["path"]).read_text(),overlay["path"],classes,report["source_config"],tools["chipyard_jar"]["snapshot"]["path"])
    if origins!=report["class_origins"]:raise Error("source class-origin summary mismatch")
    return {"status":"bounded_source_recipe_verified","source_count":83,"phase_receipts":actual_phase,
            "executed_producer":executed,"class_origins_recomputed":True,
            "historical_producer_sources_required":False}


def run(source_path, source_sha, manifest_path, manifest_sha, output):
    c = CHECK.Checker()
    # Historical producer paths may have changed. Select the receipt bytes and
    # validate its immutable execution snapshots instead of recursively opening
    # every identity in the report, including superseded producer source files.
    source_path, source_id, source = select_source_report(c, source_path, source_sha)
    manifest_path, manifest_id, _, manifest = c.selected(manifest_path, manifest_sha)
    CHECK.require(source.get("schema"), "atlas.ee290_source_elaboration.v0", "source report schema")
    CHECK.require(source.get("state"), "source_elaboration_captured", "source producer state")
    CHECK.require(manifest.get("schema"), "atlas.retained_hw_ir.v0", "retained manifest schema")
    CHECK.require(manifest.get("state"), "verified", "selected retained state")
    if source.get("selected_inputs_stable") is not True or source.get("scheduling_qualified") is not False:
        raise Error("source producer must report stable inputs and no scheduling qualification")
    source_validation = validate_source_receipt(c, source_path, source)
    fresh = c.verify(source["firrtl"], source_path.parent)
    chisel = c.verify(source["annotations"], source_path.parent)
    selected_chisel = c.verify(manifest["inputs"]["chisel_annotations"], manifest_path.parent)
    if (chisel["sha256"], chisel["bytes"]) != (selected_chisel["sha256"], selected_chisel["bytes"]):
        raise Error("fresh Chisel annotation identity differs from selected policy baseline")
    final = c.verify(manifest["inputs"]["annotations"], manifest_path.parent)
    options = c.verify(manifest["inputs"]["lowering_options"], manifest_path.parent)
    selected_hw = c.verify(manifest["hardware_ir"], manifest_path.parent)
    firtool = c.verify(manifest["tools"]["firtool"], manifest_path.parent)
    circt = c.verify(manifest["tools"]["circt_opt"], manifest_path.parent)
    policy = annotation_policy(json.loads(Path(chisel["path"]).read_text()), json.loads(Path(final["path"]).read_text()))
    audit_annotations(json.loads(Path(final["path"]).read_text()))
    audit_firrtl(fresh["path"])
    lowering = Path(options["path"]).read_text().strip()
    if not lowering or not re.fullmatch(r"[A-Za-z0-9_,=.-]+", lowering): raise Error("unreviewed lowering-option spelling")
    helper = CHECK.allowed(_here / "retain-ee290-hw-ir.py")
    dependencies = {"retention": c.identity(helper), "capture": c.identity(CAPTURE_PATH),
                    "fingerprints": c.identity(FINGERPRINT_PATH), "indexer": c.identity(CHECK.allowed(_here / "index-retained-hw.py")),
                    "path_checker": c.identity(CAPTURE._DEPENDENCY), "source_audit": c.identity(SOURCE_HELPER_PATH), "producer": c.identity(CHECK.allowed(__file__))}
    python = c.identity(CHECK.allowed(sys.executable))
    root = CHECK.allowed(_here.parent)
    output = CHECK.allowed(output, missing=True)
    base = CHECK.allowed(root / "build/rtl-timing", missing=True)
    if output == base or not output.is_relative_to(base) or output.exists(): raise Error("new output beneath ignored build/rtl-timing required")
    import subprocess
    if subprocess.run(["git", "-C", str(root), "check-ignore", "--no-index", "--quiet", "--", str(output)]).returncode:
        raise Error("correspondence output must be ignored")
    c.recheck()
    output.mkdir(parents=True)
    report = {"schema": "atlas.ee290_source_correspondence.v0", "state": "started", "target_config": "EE290SimConfig",
              "source_report": source_id, "selected_manifest": manifest_id, "producer_dependencies": dependencies,
              "annotation_policy": policy, "source_recipe_validation": source_validation, "failures": [], "scheduling_qualified": False,
              "historical_simulator_build_linked": False, "full_chipyard_source_build": False,
              "limitations": ["The exact bounded source recipe and immutable producer snapshots are checked; remaining framework/JRE/system dependencies remain explicit.",
                              "Equality covers the AtlasCore transitive closure, including external declarations; external implementations remain conditional.",
                              "Location aliases alone are ignored; no compiler trust bindings are regenerated.",
                              "This comparison does not establish historical simulator provenance or scheduling qualification."]}
    saved_report = output / "report.json"
    def save(): saved_report.write_text(json.dumps(report, indent=2)+"\n")
    save()
    try:
        inputs = [{"role": "tool", "identity": python}, {"role": "firrtl", "identity": fresh},
                  {"role": "chisel_annotations", "identity": chisel}, {"role": "final_annotations", "identity": final},
                  {"role": "lowering_options", "identity": options}, {"role": "firtool", "identity": firtool},
                  {"role": "circt_opt", "identity": circt}, *[{"role": k, "identity": v} for k,v in dependencies.items()]]
        argv = [python["path"], str(helper), "--firrtl", fresh["path"], "--annotations", final["path"],
                "--lowering-options", options["path"], "--chisel-annotations", chisel["path"], "--firtool", firtool["path"],
                "--circt-opt", circt["path"], "--expected-firrtl-sha256", fresh["sha256"], "--source-config", "EE290SimConfig",
                "--output", "{output}/retained"]
        phase_dir = output / "lowering"
        phase = CAPTURE.capture_phase(phase_dir, "selected_source_retention", argv, inputs,
                                     [{"role": "retained", "relative_path": "retained", "kind": "tree"}],
                                     {"PATH": "/usr/bin:/bin", "LANG": "C", "LC_ALL": "C", "HOME": str(output)}, 1800)
        report["lowering_phase"] = CHECK.Checker().identity(phase_dir / "phase.json")
        if phase["state"] != "phase_completed": raise Error("fresh retention phase failed; inspect captured logs")
        mpath = CHECK.allowed(phase_dir / "retained/manifest.json")
        derived = json.loads(mpath.read_text())
        CHECK.require(derived.get("state"), "verified", "fresh retained state")
        fresh_hw = CHECK.Checker().verify(derived["hardware_ir"], mpath.parent)
        report["fresh_retained_manifest"] = CHECK.Checker().identity(mpath)
        report["fresh_hardware_ir"] = fresh_hw
        with Path(fresh_hw["path"]).open(newline="") as f: a=f.read()
        with Path(selected_hw["path"]).open(newline="") as f: b=f.read()
        report["atlas_core_correspondence"] = compare_closures(a,b)
        c.recheck()
        post = CHECK.Checker()
        post.verify(report["lowering_phase"], output)
        for row in phase["outputs"]:
            for member in row["files"]: post.verify(member["identity"], output)
        report["state"] = "selected_source_correspondence_captured"
    except (OSError, ValueError, RuntimeError, KeyError, TypeError) as error:
        report["state"] = "source_correspondence_failed"; report["failures"].append(str(error))
    save()
    return report


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source-report", type=Path, required=True); p.add_argument("--expected-source-report-sha256", required=True)
    p.add_argument("--selected-manifest", type=Path, required=True); p.add_argument("--expected-selected-manifest-sha256", required=True)
    p.add_argument("--output", type=Path, required=True)
    args=p.parse_args()
    try: report=run(args.source_report,args.expected_source_report_sha256,args.selected_manifest,args.expected_selected_manifest_sha256,args.output)
    except (OSError, ValueError, KeyError, TypeError) as error: print(str(error),file=sys.stderr); return 2
    print(json.dumps({"state":report["state"],"report":str(args.output/"report.json")}))
    return 0 if report["state"]=="selected_source_correspondence_captured" else 1

if __name__ == "__main__": sys.exit(main())
