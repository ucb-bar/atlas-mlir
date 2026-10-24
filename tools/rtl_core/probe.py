#!/usr/bin/env python3
"""Materialize manifest-bound Atlas sources and probe a real-TileLink core.

This is a standalone compile/elaboration probe, not a program execution or a
reproducible build of the cached dependency JARs. It never invokes SBT.
"""
import sys
from pathlib import Path

TOOLS_ROOT = Path(__file__).absolute().parents[1]
if (TOOLS_ROOT / "atlas_path_guard.py").is_file():
    sys.path.insert(0, str(TOOLS_ROOT))

import argparse
from atlas_tooling import run_command, bounded
from atlas_path_guard import checked, lexical_path
import importlib.util
import json
import os
import signal
import time
import zipfile
from pathlib import PurePosixPath

sys.dont_write_bytecode = True
SOURCE_ROOTS = ("src/main/scala/atlas", "dependencies/fpex/src/main/scala", "dependencies/sp26-fp-units/src/main/scala")
EXTRA = ("src/main/scala/diplomatic/memory/CSRFile.scala", "src/main/scala/diplomatic/memory/DMA.scala",
    "src/main/scala/diplomatic/memory/InstrMem.scala", "src/main/scala/diplomatic/memory/VMEM.scala",
    "src/main/scala/diplomatic/top/AtlasCore.scala")
JARS = ("rocketchip", "diplomacy", "cde")


def lexical(value):
    lexical_path(value)


def helper(path):
    path = checked(path)
    spec = importlib.util.spec_from_file_location("rtl_support", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def jar_metadata(path, support):
    path = support.preflight_archive(path)
    # ZIP metadata first: never read/hash/copy content of an unadmitted archive.
    with zipfile.ZipFile(path) as archive:
        entries = archive.infolist()
        for entry in entries:
            lexical(entry.filename)
            if PurePosixPath(entry.filename).is_absolute():
                raise ValueError("Absolute archive member refused")
    return {"members": len(entries), "source_build_identity": "unknown; binary identity only"}


def emitted_classes(directory, support):
    """List only this job's newly generated class directory, pruning before stat."""
    pending = [support.safe_path(directory)]
    while pending:
        parent = pending.pop()
        with os.scandir(parent) as entries:
            names = sorted(entry.name for entry in entries)
        for name in names:
            lexical(name)
            path = support.safe_path(parent / name)
            if path.is_dir():
                pending.append(path)
            elif path.suffix == ".class":
                yield path
            else:
                raise ValueError("Unexpected generated compiler output")


def run(args):
    started = time.monotonic()
    support = helper(args.support_helper)
    out = support.safe_path(args.out)
    own = support.safe_path(Path(__file__).absolute().parent)
    before = [support.identity(own / name) for name in ("probe.py", "EmitOwnedAtlasCore.scala")]
    out.mkdir(parents=True, exist_ok=False)
    receipt = {"schema_version": 1, "status": "preparing", "command": sys.argv, "process_id": os.getpid(),
        "claim": "Current AtlasCore compile/elaboration only; no program or timing claim", "rtl_revision": None,
        "dependency_source_epochs": "unknown", "records": [], "source_closure": []}
    records = receipt["records"]

    def bind(path, expected=None):
        value = support.identity(path, expected)
        records.append(value)
        return value

    def copy(path, dest, expected):
        bind(path, expected)
        support.copy_source(path, dest, expected)
        bind(dest, expected)

    try:
        receipt["shared_support"] = [bind(sys.modules[name].__file__) for name in ("atlas_path_guard", "atlas_tooling")]
        helper_identity = bind(args.support_helper)
        copy(args.support_helper, out / "reviewed-support.py", helper_identity["sha256"])
        # Importing was read-only; every following access uses its reviewed admission.
        rtl = support.safe_path(args.rtl_root)
        rev = run_command(["git", "-C", str(rtl), "-c", "submodule.recurse=false", "rev-parse", "HEAD"], check=True,
            capture_output=True, text=True).stdout.strip()
        receipt["rtl_revision"] = rev
        bind(args.provenance)
        manifest = json.loads(support.safe_path(args.provenance).read_text())
        selected = {}
        counts = dict.fromkeys(SOURCE_ROOTS, 0)
        for relative, sha in sorted(manifest["sources"].items()):
            # Filter names before looking up any corresponding physical path.
            lexical(relative)
            if PurePosixPath(relative).is_absolute():
                raise ValueError("Absolute source entry")
            root = next((root for root in SOURCE_ROOTS if relative.startswith(root + "/")), None)
            if root is not None:
                if not relative.endswith(".scala"):
                    raise ValueError("Non-Scala source entry")
                selected[relative] = sha
                counts[root] += 1
        if not all(counts.values()):
            raise ValueError("A declared RTL source root was empty")
        for relative in EXTRA:
            selected[relative] = manifest["sources"][relative]
        for relative, sha in sorted(selected.items()):
            copy(rtl / relative, out / "sources" / relative, sha)
            receipt["source_closure"].append({"relative": relative, "sha256": sha})
        receipt["source_counts"] = counts | {"explicit_diplomatic_files": len(EXTRA)}
        receipt["jars"] = {}
        for name in JARS:
            path = getattr(args, name + "_jar")
            receipt["jars"][name] = jar_metadata(path, support)
            copy(path, out / "lib" / (name + ".jar"), bind(path)["sha256"])
        bind(args.dependency_export)
        exported = support.safe_path(args.dependency_export).read_text().strip().split(":")
        cache = support.safe_path(args.maven_root)
        runtime, compiler = [], []
        receipt["runtime_libraries"] = []
        # The SBT export is metadata only. Select libraries under the explicit
        # Maven root lexically; exclude project JARs and never follow symlinks.
        prefix = str(cache).rstrip("/") + "/"
        for value in exported:
            lexical(value)
            if not value.startswith(prefix):
                continue
            relative = value.removeprefix(prefix)
            source = cache / relative
            info = jar_metadata(source, support)
            record = bind(source)
            dest = out / "lib/runtime" / relative
            copy(source, dest, record["sha256"])
            runtime.append(str(dest))
            receipt["runtime_libraries"].append(record | info)
        if not runtime:
            raise ValueError("Dependency export has no runtime libraries")
        for source, is_plugin in ((args.scala_compiler_jar, False), (args.scala_plugin_jar, True)):
            source = support.safe_path(source)
            info = jar_metadata(source, support)
            record = bind(source)
            dest = out / "lib/compiler" / source.name
            copy(source, dest, record["sha256"])
            if is_plugin:
                plugin = str(dest)
            else:
                compiler.append(str(dest))
            receipt["runtime_libraries"].append(record | info)
        receipt["classpath"] = {"runtime": runtime + [str(out / "lib" / (name + ".jar")) for name in JARS],
            "compiler": compiler + runtime, "plugin": plugin, "scope": "conservative cached library closure, not a minimum"}
        for record in before:
            name = Path(record["path"]).name
            dest = out / "harness" / name if name.endswith(".scala") else out / name
            copy(record["path"], dest, record["sha256"])
        for name in ("java", "firtool"):
            receipt[name] = bind(getattr(args, name))
        java_home = support.safe_path(args.java).parent.parent
        receipt["java_runtime"] = [bind(java_home / relative) for relative in ("release", "lib/modules",
            "lib/server/libjvm.so", "lib/libjava.so")]
        receipt["build_route"] = "direct selected scalac and emitter"
        receipt["status"] = "prepared"
        support.write_json(out / "receipt.json", receipt)
        if not args.prepare_only:
            env = os.environ.copy()
            for key in ("_JAVA_OPTIONS", "JDK_JAVA_OPTIONS", "CLASSPATH", "LD_PRELOAD"):
                env.pop(key, None)
            env.update({"JAVA_HOME": str(support.safe_path(args.java).parent.parent),
                "JAVA_TOOL_OPTIONS": "-Xmx8G -XX:ActiveProcessorCount=3",
                "CHISEL_FIRTOOL_PATH": str(support.safe_path(args.firtool).parent),
                "PATH": str(support.safe_path(args.java).parent) + os.pathsep + os.environ.get("PATH", ""),
                "ATLAS_CORE_RTL_DIR": str(out / "rtl")})
            receipt["environment"] = {key: env[key] for key in ("JAVA_HOME", "JAVA_TOOL_OPTIONS", "CHISEL_FIRTOOL_PATH",
                "PATH", "ATLAS_CORE_RTL_DIR")}
            java = str(support.safe_path(args.java))
            runtime_cp = ":".join(receipt["classpath"]["runtime"])
            compiler_cp = ":".join(receipt["classpath"]["compiler"])
            (out / "classes").mkdir()
            source_args = [str(out / "sources" / relative) for relative in sorted(selected)]
            command = ([java, "-cp", compiler_cp, "scala.tools.nsc.Main", "-classpath", runtime_cp, "-Xplugin:" + plugin,
                "-deprecation", "-feature", "-unchecked", "-language:reflectiveCalls", "-d",
                str(out / "classes")] + source_args + [str(out / "harness/EmitOwnedAtlasCore.scala")])
            emitter = [java, "-cp", str(out / "classes") + ":" + runtime_cp, "atlas.rtl.EmitOwnedAtlasCore"]
            receipt["build_command"] = command
            receipt["emitter_command"] = emitter
            receipt["versions"] = [support.version([str(args.java), "-version"], env=env),
                support.version([str(args.firtool), "--version"], env=env)]
            receipt["status"] = "building"
            support.write_json(out / "receipt.json", receipt)
            for stage, argv in (("compile", command), ("elaborate", emitter)):
                receipt[stage] = bounded(argv, out, env, out / (stage + ".log"), support)
                bind(out / (stage + ".log"))
                bind(out / (stage + ".resources.json"))
                support.write_json(out / "receipt.json", receipt)
                if receipt[stage]["returncode"]:
                    raise RuntimeError(f"Owned {stage} exited {receipt[stage]['returncode']}")
                if stage == "compile":
                    receipt["compiled_classes"] = [bind(path) for path in emitted_classes(out / "classes", support)]
            receipt["system_verilog"] = bind(out / "rtl" / "AtlasCore.sv")
            receipt["status"] = "passed_compile_elaboration_only"
    except BaseException as error:
        receipt["status"] = "failed"
        receipt["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        receipt["wall_seconds"] = time.monotonic() - started
        try:
            for name in ("compile.log", "compile.resources.json", "elaborate.log", "elaborate.resources.json"):
                candidate = support.safe_path(out / name)
                if candidate.exists() and not any(record["path"] == str(candidate) for record in records):
                    bind(candidate)
            for record in records + before:
                support.identity(record["path"], record["sha256"])
            receipt["source_artifacts_preserved"] = True
        except Exception as error:
            receipt["source_artifacts_preserved"] = False
            receipt["bookend_error"] = f"{type(error).__name__}: {error}"
            receipt["status"] = "failed"
        support.write_json(out / "receipt.json", receipt)
        if not receipt.get("source_artifacts_preserved"):
            raise RuntimeError("Source/tool bookends failed")


if __name__ == "__main__":

    def interrupted(number, frame):
        raise SystemExit(f"Probe received signal {number}")

    signal.signal(signal.SIGTERM, interrupted)
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("rtl-root", "provenance", "rocketchip-jar", "diplomacy-jar", "cde-jar", "java", "firtool", "support-helper",
        "dependency-export", "maven-root", "out", "scala-compiler-jar", "scala-plugin-jar"):
        parser.add_argument("--" + name, required=True, type=Path)
    parser.add_argument("--prepare-only", action="store_true")
    run(parser.parse_args())
