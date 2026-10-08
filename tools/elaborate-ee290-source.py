#!/usr/bin/env python3
"""Capture selected Scala sources -> an overlay -> EE290SimConfig FIRRTL.

Only caller-selected files are consumed. Remaining Chipyard implementation and
resources are an explicitly pinned binary dependency, not source-proven code.
This producer does not invoke Make/SBT or qualify timing or a simulator build.
"""

import argparse
from collections import deque
from datetime import datetime, timezone
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import shutil
import stat
import subprocess
import sys
from urllib.parse import unquote, urlsplit
import zipfile


def bootstrap_path(path):
    path = Path(path)
    denied = lambda value: any(word in part.lower() for part in value.parts
                              for word in ("hammer", "vlsi"))
    if denied(path):
        raise ValueError("restricted producer dependency")
    if not path.is_absolute():
        path = Path.cwd() / path
    if denied(path):
        raise ValueError("restricted producer dependency")
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
            raise ValueError("restricted producer dependency")
        if stat.S_ISLNK(candidate.lstat().st_mode):
            if candidate in links or len(links) >= 64:
                raise ValueError("recursive producer dependency symlink")
            links.add(candidate)
            target = Path(os.readlink(candidate))
            if denied(target):
                raise ValueError("restricted producer dependency symlink target")
            if target.is_absolute():
                current = Path(target.anchor)
                pending.extendleft(reversed(target.parts[1:]))
            else:
                pending.extendleft(reversed(target.parts))
        else:
            current = candidate
    return current


CAPTURE_PATH = bootstrap_path(Path(__file__).parent / "ee290_build_capture.py")
SPEC = importlib.util.spec_from_file_location("atlas_ee290_source_capture", CAPTURE_PATH)
CAPTURE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(CAPTURE)
CHECK = CAPTURE._CHECK
SourceError = CAPTURE.CaptureError

MAIN_PREFIX = "generators/sp26-atlas-acc/src/main/scala/"
GLUE_PREFIX = "generators/sp26-atlas-acc/chipyard/"
CHIPYARD_SOURCES = {
    "generators/chipyard/src/main/scala/config/EE290Configs.scala",
    "generators/chipyard/src/main/scala/config/AbstractConfig.scala",
    "generators/chipyard/src/main/scala/iobinders/IOBinders.scala",
}
TOOLS = {"java", "scala_compiler", "scala_library", "scala_reflect",
         "chisel_plugin", "chipyard_jar"}
HELPERS = {"espresso", "dtc"}
PUBLIC_NAME = "chipyard.harness.TestHarness.EE290SimConfig"


def source_relative(value):
    if not isinstance(value, str):
        raise SourceError("source relative_path must be a string")
    path = Path(value)
    if path.is_absolute() or not path.parts or ".." in path.parts or CHECK.restricted(path) or \
            path.as_posix() != value or path.suffix != ".scala":
        raise SourceError("source snapshot requires a normalized allowed relative Scala path")
    return path


def selected_inputs(path, expected):
    checker = CHECK.Checker()
    path, identity, data, value = checker.selected(path, expected)
    CHECK.require(value.get("schema"), "atlas.ee290_source_build_inputs.v0", "source input schema")
    if value.get("target_config", "EE290SimConfig") != "EE290SimConfig":
        raise SourceError("unsupported public target")
    sources, seen, origins = [], set(), set()
    for row in CHECK.list_value(value.get("sources"), "sources", nonempty=True):
        row = CHECK.object_value(row, "source selection")
        relative = source_relative(row.get("relative_path"))
        actual = checker.verify(row.get("identity"), path.parent)
        if relative.as_posix() in seen or actual["path"] in origins:
            raise SourceError("duplicate source selection or snapshot path")
        seen.add(relative.as_posix())
        origins.add(actual["path"])
        sources.append({"relative_path": relative.as_posix(), "identity": actual})
    if len(sources) != 83 or sum(name.startswith(MAIN_PREFIX) for name in seen) != 69 or \
            sum(name.startswith(GLUE_PREFIX) for name in seen) != 11 or \
            {name for name in seen if not name.startswith((MAIN_PREFIX, GLUE_PREFIX))} != CHIPYARD_SOURCES:
        raise SourceError("selected recipe requires 69 Atlas main, 11 Atlas glue and 3 Chipyard Scala sources")
    tools = CHECK.object_value(value.get("tools"), "tools")
    if not TOOLS <= set(tools) or set(tools) - TOOLS - HELPERS:
        raise SourceError("source recipe requires six selected roles and only optional espresso/dtc helpers")
    tools = {role: checker.verify(member, path.parent) for role, member in tools.items()}
    jre = [checker.verify(member, path.parent)
           for member in CHECK.list_value(value.get("jre_files", []), "JRE files")]
    if len({member["path"] for member in jre}) != len(jre):
        raise SourceError("duplicate JRE input")
    configuration = value.get("source_config")
    if not isinstance(configuration, str) or not re.fullmatch(
            r"[A-Za-z_$][A-Za-z0-9_.$]*:[A-Za-z_$][A-Za-z0-9_$]*", configuration) or CHECK.restricted(Path(configuration)):
        raise SourceError("one explicit package:Config source_config is required")
    top = value.get("top_module", "chipyard.harness.TestHarness")
    if top != "chipyard.harness.TestHarness":
        raise SourceError("only the selected TestHarness top module is supported")
    timeout = value.get("timeout_seconds", 900)
    if type(timeout) not in (int, float) or not math.isfinite(timeout) or timeout <= 0:
        raise SourceError("timeout_seconds must be finite and positive")
    if "metadata" in value:
        CHECK.object_value(value["metadata"], "source metadata")
    checker.recheck()
    return checker, identity, data, value, sorted(sources, key=lambda row: row["relative_path"]), tools, jre


def java_environment(output):
    """Do not inherit option, classpath, native-library or home injections."""
    return {"PATH": str(output / "helpers") + ":/usr/bin:/bin", "LANG": "C", "LC_ALL": "C",
            "TMPDIR": str(output / "tmp"), "HOME": str(output / "home")}


def overlay_classes(path):
    path = CHECK.allowed(path)
    classes = set()
    with zipfile.ZipFile(path) as archive:
        for entry in archive.infolist():
            member = Path(entry.filename)
            # Guard spelling before examining or reading an archive member.
            if CHECK.restricted(member) or member.is_absolute() or ".." in member.parts:
                raise SourceError("restricted or escaping overlay archive member")
            if entry.filename.endswith(".class"):
                name = entry.filename[:-6].replace("/", ".")
                if name in classes:
                    raise SourceError("duplicate overlay class")
                classes.add(name)
    if not classes:
        raise SourceError("compiler produced an empty overlay")
    return classes


def origin_path(value):
    if value.startswith("file:"):
        url = urlsplit(value)
        if url.netloc or url.query or url.fragment:
            raise SourceError("unsupported class origin URL")
        value = unquote(url.path)
    if not value.startswith("/"):
        raise SourceError("selected class has no concrete file origin")
    return CHECK.allowed(value)


def audit_origins(text, overlay, classes, configuration, framework):
    overlay, framework = CHECK.allowed(overlay), CHECK.allowed(framework)
    loaded, framework_loaded, generated = {}, set(), set()
    for line in text.splitlines():
        match = re.search(r"\[class,load\]\s+(\S+)\s+source:\s+(.+)$", line)
        if not match:
            continue
        name, origin = match.groups()
        if CHECK.restricted(Path(name.replace(".", "/"))):
            raise SourceError("restricted loaded class name")
        # Hidden lambda classes have no JAR entry. The VM records their defining
        # class as their source; accept only an already verified overlay owner.
        lambda_name = re.fullmatch(r"(.+)\$\$Lambda\$[0-9]+(?:/0x[0-9a-fA-F]+)?", name)
        if lambda_name and (name.startswith("atlas.") or lambda_name[1] in classes):
            owner = lambda_name[1]
            if owner not in loaded or origin != owner:
                raise SourceError("generated selected lambda lacks a verified overlay owner: " + name)
            generated.add(name)
            continue
        if name in classes or name.startswith("atlas."):
            if name not in classes or origin_path(origin) != overlay:
                raise SourceError("selected class loaded from framework instead of source overlay: " + name)
            loaded[name] = str(overlay)
        elif origin.startswith("file:") or origin.startswith("/"):
            if origin_path(origin) != framework:
                raise SourceError("unexpected file-backed class origin: " + name)
            framework_loaded.add(name)
    package, config = configuration.split(":")
    required = {"atlas.tile.AtlasCore", "atlas.scalar.ScalarCore", "atlas.lsu.LSU",
                "atlas.config.WithAtlasTile", package + "." + config,
                "chipyard.config.AbstractConfig", "chipyard.iobinders.WithUARTIOCells",
                "chipyard.iobinders.WithDebugIOCells", "chipyard.iobinders.WithSerialTLIOCells",
                "chipyard.iobinders.WithChipIdIOCells"}
    if not required <= set(loaded):
        raise SourceError("missing required source-overlay class origins: " + ", ".join(sorted(required - set(loaded))))
    if not framework_loaded:
        raise SourceError("missing observed opaque framework dependency")
    return {"required_classes": sorted(required), "loaded_overlay_classes": sorted(loaded),
            "loaded_overlay_generated_lambdas": sorted(generated),
            "loaded_framework_class_count": len(framework_loaded),
            "loaded_framework_classes": sorted(framework_loaded), "all_selected_loaded_classes_from_overlay": True}


def elaborate(inputs, expected, output):
    checker, input_id, data, value, selected, original_tools, jre = selected_inputs(inputs, expected)
    root = CHECK.allowed(__file__).parents[1]
    output = CHECK.allowed(output, missing=True)
    artifact_root = CHECK.allowed(root / "build/rtl-timing", missing=True)
    if output == artifact_root or not output.is_relative_to(artifact_root) or output.exists():
        raise SourceError("output must be a new directory beneath ignored build/rtl-timing")
    ignored = subprocess.run(["git", "-C", str(root), "check-ignore", "--no-index", "--quiet", "--", str(output)],
                             capture_output=True)
    if ignored.returncode:
        raise SourceError("source build output must be ignored")
    producer = checker.identity(CHECK.allowed(__file__))
    producer_bytes = Path(producer["path"]).read_bytes()
    dependencies = {"capture": checker.identity(CAPTURE_PATH),
                    "path_checker": checker.identity(CAPTURE._DEPENDENCY)}
    checker.recheck()
    output.mkdir(parents=True, exist_ok=False)
    (output / "inputs.json").write_bytes(data)
    (output / "elaborate-ee290-source.executed.py").write_bytes(producer_bytes)
    (output / "tmp").mkdir()
    (output / "home").mkdir()
    (output / "helpers").mkdir()
    sources = []
    tools = {}
    phases = []
    captured_outputs = []
    receipt = {"schema": "atlas.ee290_source_elaboration.v0", "target_config": "EE290SimConfig",
               "state": "started", "created_utc": datetime.now(timezone.utc).isoformat(),
               "scheduling_qualified": False, "full_chipyard_source_build": False,
               "simulator_build_linked": False, "selected_inputs": input_id,
               "producer": producer, "producer_dependencies": dependencies,
               "source_config": value["source_config"], "top_module": value.get("top_module", "chipyard.harness.TestHarness"),
               "metadata": value.get("metadata", {}), "sources": sources, "tools": tools, "jre_files": jre,
               "helper_boundary": {"pinned_helpers": sorted(HELPERS & set(original_tools)),
                                   "unpinned_system_helper_fallback_possible": sorted(HELPERS - set(original_tools))},
               "phases": phases, "failures": [],
               "limitations": ["Only the 83 explicitly selected Scala sources are compiled freshly.",
                               "The remaining Chipyard implementation/resources are pinned opaque binary dependencies.",
                               "Class origins cover actually loaded classes, not a proof of framework source history.",
                               "Selected Java executable/JRE files do not pin every native/system runtime input.",
                               "Fresh source elaboration does not bind a historical simulator or qualify scheduling/timing."]}
    report_path = output / "report.json"

    def save():
        report_path.write_text(json.dumps(receipt, indent=2) + "\n")

    def snapshot(identity, target):
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(identity["path"], target)
        actual = CHECK.Checker().identity(target)
        if (actual["sha256"], actual["bytes"]) != (identity["sha256"], identity["bytes"]):
            raise SourceError("selected input changed during snapshot")
        return actual

    def phase(kind, argv, phase_inputs, phase_outputs):
        path = output / kind
        result = CAPTURE.capture_phase(path, kind, argv, phase_inputs, phase_outputs,
                                       java_environment(output), value.get("timeout_seconds", 900))
        phases.append({"kind": kind, "receipt": CHECK.Checker().identity(path / "phase.json")})
        save()
        if result["state"] != "phase_completed":
            raise SourceError(kind + " command failed; inspect captured phase logs")
        captured_outputs.extend(member["identity"] for row in result["outputs"] for member in row["files"])

    save()
    try:
        for row in selected:
            member = snapshot(row["identity"], output / "sources" / row["relative_path"])
            sources.append({**row, "snapshot": member})
        for role, identity in original_tools.items():
            destination = output / "helpers" / role if role in HELPERS else output / "dependencies" / (role + ".jar")
            member = identity if role == "java" else snapshot(identity, destination)
            if role in HELPERS:
                destination.chmod(0o700)
            tools[role] = {"selected": identity, "snapshot": member}
        source_list = output / "sources.args"
        # Java argument-file quoting is unnecessary for this newline-free
        # Scala source recipe; reject whitespace that scalac would split.
        names = [row["snapshot"]["path"] for row in sources]
        if any(any(char.isspace() for char in name) for name in names):
            raise SourceError("source snapshot paths containing whitespace are unsupported")
        source_list.write_text("\n".join(names) + "\n")
        source_args = CHECK.Checker().identity(source_list)
        members = [{"role": "tool", "identity": tools["java"]["snapshot"]}]
        members += [{"role": role, "identity": row["snapshot"]} for role, row in tools.items() if role != "java"]
        members += [{"role": "source", "identity": row["snapshot"]} for row in sources]
        members += [{"role": "jre", "identity": member} for member in jre]
        members.append({"role": "source_arguments", "identity": source_args})
        compiler_cp = ":".join(tools[role]["snapshot"]["path"] for role in
                               ("scala_compiler", "scala_library", "scala_reflect"))
        compile_argv = [tools["java"]["snapshot"]["path"], "-Xmx8G", "-cp", compiler_cp,
                        "scala.tools.nsc.Main", "-classpath", tools["chipyard_jar"]["snapshot"]["path"],
                        "-Xplugin:" + tools["chisel_plugin"]["snapshot"]["path"], "-Xplugin-require:chiselplugin",
                        "-deprecation", "-unchecked", "-Ytasty-reader", "-Ymacro-annotations",
                        "-d", "{output}/overlay.jar", "@" + str(source_list)]
        phase("compile", compile_argv, members,
              [{"role": "overlay", "relative_path": "overlay.jar", "kind": "file"}])
        overlay = CHECK.Checker().identity(output / "compile/overlay.jar")
        classes = overlay_classes(overlay["path"])
        receipt["overlay"] = overlay
        receipt["overlay_class_count"] = len(classes)
        receipt["overlay_classes"] = sorted(classes)
        elaboration_inputs = [{"role": "tool", "identity": tools["java"]["snapshot"]},
                              {"role": "overlay", "identity": overlay},
                              {"role": "opaque_framework", "identity": tools["chipyard_jar"]["snapshot"]}]
        elaboration_inputs += [{"role": "jre", "identity": member} for member in jre]
        elaboration_inputs += [{"role": role, "identity": tools[role]["snapshot"]} for role in HELPERS if role in tools]
        elab_argv = [tools["java"]["snapshot"]["path"], "-Xmx8G",
                     "-Xlog:class+load=info:file={output}/class-load.log", "-cp",
                     overlay["path"] + ":" + tools["chipyard_jar"]["snapshot"]["path"],
                     "chipyard.Generator", "--target-dir", "{output}", "--name", PUBLIC_NAME,
                     "--top-module", receipt["top_module"], "--legacy-configs", value["source_config"]]
        phase("elaborate", elab_argv, elaboration_inputs,
              [{"role": "firrtl", "relative_path": PUBLIC_NAME + ".fir", "kind": "file"},
               {"role": "annotations", "relative_path": PUBLIC_NAME + ".anno.json", "kind": "file"},
               {"role": "class_origins", "relative_path": "class-load.log", "kind": "file"}])
        classlog = CHECK.Checker().identity(output / "elaborate/class-load.log")
        receipt["class_load_log"] = classlog
        receipt["class_origins"] = audit_origins(Path(classlog["path"]).read_text(), overlay["path"], classes,
                                                value["source_config"], tools["chipyard_jar"]["snapshot"]["path"])
        receipt["firrtl"] = CHECK.Checker().identity(output / "elaborate" / (PUBLIC_NAME + ".fir"))
        receipt["annotations"] = CHECK.Checker().identity(output / "elaborate" / (PUBLIC_NAME + ".anno.json"))
        checker.recheck()
        final = CHECK.Checker()
        for identity in captured_outputs + [row["receipt"] for row in phases] + [producer] + list(dependencies.values()):
            final.verify(identity, output)
        final.verify(overlay, output)
        final.verify(classlog, output)
        for row in sources:
            final.verify(row["snapshot"], output)
        for row in tools.values():
            final.verify(row["snapshot"], output)
        receipt["selected_inputs_stable"] = True
        receipt["source_to_firrtl"] = {"status": "captured_bounded_source_elaboration",
                                       "scope": "83 selected Scala sources compiled to an overlay; observed selected classes load from that overlay against pinned opaque Chipyard binary dependency."}
        receipt["state"] = "source_elaboration_captured"
    except (OSError, ValueError, zipfile.BadZipFile) as error:
        receipt["state"] = "source_elaboration_failed"
        receipt["failures"].append(str(error))
    save()
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--expected-inputs-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        receipt = elaborate(args.inputs, args.expected_inputs_sha256, args.output)
    except (OSError, ValueError) as error:
        print("EE290 source elaboration: " + str(error), file=sys.stderr)
        return 2
    print(json.dumps({"state": receipt["state"], "report": str(args.output / "report.json"),
                      "scheduling_qualified": False, "full_chipyard_source_build": False}))
    return 0 if receipt["state"] == "source_elaboration_captured" else 1


if __name__ == "__main__":
    sys.exit(main())
