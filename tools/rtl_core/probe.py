#!/usr/bin/env python3
"""Materialize 129 authenticated Atlas sources and probe a real-TileLink core.

This is a standalone compile/elaboration probe, not a program execution or a
reproducible build of the cached dependency JARs. It never invokes SBT.
"""
import argparse
import importlib.util
import json
import os
import signal
import subprocess
import sys
import time
import zipfile
from pathlib import Path, PurePosixPath

sys.dont_write_bytecode = True
HELPER_SHA = "2b089e369bee6f8d540c588e089b96395d82a18b2ec22edc95983e6a1764b6c4"
RTL_SHA = "2ae0bef209df6db78c3de18e8f651bb43855cce9"
ROOT_COUNTS = {"src/main/scala/atlas": 62,
               "dependencies/fpex/src/main/scala": 20,
               "dependencies/sp26-fp-units/src/main/scala": 42}
EXTRA = {
    "src/main/scala/diplomatic/memory/CSRFile.scala": "efe634f4881e566663cd26f63333286410728aa5ddd43463565e48ada6cdcef3",
    "src/main/scala/diplomatic/memory/DMA.scala": "9b802f714f8afa08758ea97d2e92629645e99bf51d7d7af943fb87a93344dfad",
    "src/main/scala/diplomatic/memory/InstrMem.scala": "7a0c4ecffdb494325fdb768e148b43ae67ff3b00e81c8290b2b589a513fc90f3",
    "src/main/scala/diplomatic/memory/VMEM.scala": "69ece98e95dee38378edf225c9e51bb1b8fff2e37bf6511e7da04803459c38d2",
    "src/main/scala/diplomatic/top/AtlasCore.scala": "cac182a402e37a27f688a1d714b238e7b4ad5189c0b19f3f3e21580e9e60c78b"}
JARS = {"rocketchip": "bfa5d8f2351656ddea6f3bbe7fc20359dd45a135665f14df176b54aed091a07e",
        "diplomacy": "844b06f400c232847471dc8588205a8347a0dee3ad0c2285b56fd3b1d44fb706",
        "cde": "65fc938c8a28e13014b7c086e5bd4e93ad7dc9c9a2db66bbd079e5180b178ca9"}
CHISEL_SHA = "ad3163d1b88094806cfec95fec58b4ebeeaa6af6ec6b2766a9c02becdbe42778"
PLUGIN_SHA = "3260acf1b41b399974d21ba7ef970124ff35df6e5b0a164e65d5038e3b803bfb"
COMPILER_EXTRA = (
    "org/scala-lang/scala-compiler/2.13.16/scala-compiler-2.13.16.jar",
    "io/github/java-diff-utils/java-diff-utils/4.15/java-diff-utils-4.15.jar",
    "org/jline/jline/3.27.1/jline-3.27.1-jdk8.jar")
PLUGIN = "org/chipsalliance/chisel-plugin_2.13.16/6.7.0/chisel-plugin_2.13.16-6.7.0.jar"


def lexical(value):
    parts = PurePosixPath(str(value).replace("\\", "/")).parts
    if ".." in parts or any(bad in part.lower() for part in parts for bad in ("hammer", "vlsi")):
        raise ValueError("Unadmitted lexical path")


def helper(path):
    # Bootstrap admission must precede even importing the previously reviewed helper.
    import hashlib
    import stat
    lexical(path)
    path = Path(path).absolute()
    lexical(path)
    current = Path(path.anchor)
    for part in path.parts[1:]:
        current /= part
        if stat.S_ISLNK(os.lstat(current).st_mode):
            raise ValueError("Helper symlink refused")
    if hashlib.sha256(path.read_bytes()).hexdigest() != HELPER_SHA:
        raise ValueError("Reviewed helper identity differs")
    spec = importlib.util.spec_from_file_location("reviewed_rtl_support", path)
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
    return {"members": len(entries), "forbidden_members": 0,
            "source_build_epoch": "unknown; binary identity only"}


def owned_processes(pid):
    """Inspect only an owned process and descendants reported by its threads."""
    pending, seen = [pid], set()
    while pending:
        current = pending.pop()
        if current in seen:
            continue
        seen.add(current)
        task = Path(f"/proc/{current}/task")
        try:
            tids = list(task.iterdir())
        except FileNotFoundError:
            continue
        for tid in tids:
            if not tid.name.isdigit():
                raise RuntimeError("Unexpected owned thread directory")
            try:
                children = (tid / "children").read_text().split()
            except FileNotFoundError:
                continue
            pending.extend(int(child) for child in children)
    return seen


def bounded(command, cwd, env, log_path, support):
    """One owned group, <=1500 seconds and <=10GiB aggregate sampled RSS."""
    if set(os.sched_getaffinity(0)) != {0, 1, 2}:
        raise RuntimeError("Heavy probe requires outer taskset CPUs 0,1,2")
    started, peak = time.monotonic(), 0
    with support.safe_path(log_path).open("w") as log:
        process = subprocess.Popen(command, cwd=cwd, env=env, stdout=log, stderr=subprocess.STDOUT,
                                   start_new_session=True)
        usage = {"command": command, "pid": process.pid, "cpu_affinity": sorted(os.sched_getaffinity(0)),
                 "rss_limit_bytes": 10 * 1024 ** 3, "wall_limit_seconds": 1500,
                 "rss_poll_seconds": 0.5}
        support.write_json(log_path.with_suffix(".resources.json"), usage)
        try:
            while process.poll() is None:
                rss = 0
                for pid in owned_processes(process.pid) | {os.getpid()}:
                    try:
                        status = Path(f"/proc/{pid}/status").read_text().splitlines()
                    except FileNotFoundError:
                        continue
                    rss += sum(int(line.split()[1]) * 1024 for line in status if line.startswith("VmRSS:"))
                peak = max(peak, rss)
                if rss > 10 * 1024 ** 3 or time.monotonic() - started > 1500:
                    raise RuntimeError("Owned job exceeded aggregate RSS or wall limit")
                time.sleep(0.5)
        finally:
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()
            usage.update({"returncode": process.returncode, "wall_seconds": time.monotonic() - started,
                          "peak_aggregate_rss_bytes": peak})
            support.write_json(log_path.with_suffix(".resources.json"), usage)
    return usage


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
    before = [support.identity(own / name) for name in
              ("probe.py", "build.mill", "EmitOwnedAtlasCore.scala", "test_probe.py")]
    out.mkdir(parents=True, exist_ok=False)
    receipt = {"schema_version": 1, "status": "preparing", "command": sys.argv,
               "supervisor_pid": os.getpid(), "supervisor_cpu_affinity": sorted(os.sched_getaffinity(0)),
               "claim": "Current AtlasCore compile/elaboration only; no program or timing claim",
               "rtl_revision": RTL_SHA, "dependency_source_epochs": "unknown",
               "toolchain": {"scala": "2.13.16", "chisel": "6.7.0", "firtool": "1.62.1"},
               "records": [], "source_closure": []}
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
        bind(args.support_helper, HELPER_SHA)
        copy(args.support_helper, out / "reviewed-support.py", HELPER_SHA)
        # Importing was read-only; every following access uses its reviewed admission.
        rtl = support.safe_path(args.rtl_root)
        rev = subprocess.run(["git", "-C", str(rtl), "-c", "submodule.recurse=false",
                              "rev-parse", "HEAD"], check=True, capture_output=True, text=True).stdout.strip()
        if rev != RTL_SHA:
            raise ValueError("RTL revision differs")
        bind(args.provenance)
        manifest = json.loads(support.safe_path(args.provenance).read_text())
        selected = {}
        counts = dict.fromkeys(ROOT_COUNTS, 0)
        for relative, sha in sorted(manifest["sources"].items()):
            # Filter names before looking up any corresponding physical path.
            try:
                lexical(relative)
            except ValueError:
                continue
            if PurePosixPath(relative).is_absolute():
                raise ValueError("Absolute source entry")
            root = next((root for root in ROOT_COUNTS if relative.startswith(root + "/")), None)
            if root is not None:
                if not relative.endswith(".scala"):
                    raise ValueError("Non-Scala source entry")
                selected[relative] = sha
                counts[root] += 1
        if counts != ROOT_COUNTS:
            raise ValueError(f"Main source counts differ: {counts}")
        for relative, sha in EXTRA.items():
            if manifest["sources"].get(relative) != sha:
                raise ValueError("Full-core source differs from the authenticated model epoch")
            selected[relative] = sha
        if len(selected) != 129:
            raise ValueError("Expected exactly 129 current RTL sources")
        for relative, sha in sorted(selected.items()):
            copy(rtl / relative, out / "sources" / relative, sha)
            receipt["source_closure"].append({"relative": relative, "sha256": sha})
        receipt["source_counts"] = counts | {"explicit_diplomatic_files": 5}
        receipt["jars"] = {}
        for name, sha in JARS.items():
            path = getattr(args, name + "_jar")
            receipt["jars"][name] = jar_metadata(path, support)
            copy(path, out / "lib" / (name + ".jar"), sha)
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
            if Path(relative).name == "chisel_2.13-6.7.0.jar" and record["sha256"] != CHISEL_SHA:
                raise ValueError("Chisel runtime identity differs")
            dest = out / "lib/runtime" / relative
            copy(source, dest, record["sha256"])
            runtime.append(str(dest))
            receipt["runtime_libraries"].append(record | info)
        if not any(path.endswith("chisel_2.13-6.7.0.jar") for path in runtime):
            raise ValueError("Export has no pinned Chisel 6.7 runtime")
        for relative in COMPILER_EXTRA + (PLUGIN,):
            source = cache / relative
            info = jar_metadata(source, support)
            record = bind(source, PLUGIN_SHA if relative == PLUGIN else None)
            dest = out / "lib/compiler" / relative
            copy(source, dest, record["sha256"])
            if relative == PLUGIN:
                plugin = str(dest)
            else:
                compiler.append(str(dest))
            receipt["runtime_libraries"].append(record | info)
        receipt["classpath"] = {"runtime": runtime + [str(out / "lib" / (name + ".jar")) for name in JARS],
                                "compiler": compiler + runtime, "plugin": plugin,
                                "scope": "conservative cached library closure, not a minimum"}
        for record in before:
            name = Path(record["path"]).name
            dest = out / "harness" / name if name.endswith(".scala") else out / name
            copy(record["path"], dest, record["sha256"])
        for name in ("mill", "java", "firtool"):
            receipt[name] = bind(getattr(args, name))
        java_home = support.safe_path(args.java).parent.parent
        receipt["java_runtime"] = [bind(java_home / relative) for relative in
                                   ("release", "lib/modules", "lib/server/libjvm.so", "lib/libjava.so")]
        receipt["build_route"] = "direct pinned scalac then emitter; build.mill is retained but unused"
        receipt["status"] = "prepared"
        support.write_json(out / "receipt.json", receipt)
        if not args.prepare_only:
            env = os.environ.copy()
            for key in ("_JAVA_OPTIONS", "JDK_JAVA_OPTIONS", "CLASSPATH", "LD_PRELOAD"):
                env.pop(key, None)
            env.update({"JAVA_HOME": str(support.safe_path(args.java).parent.parent),
                        "JAVA_TOOL_OPTIONS": "-Xmx8G -XX:ActiveProcessorCount=3",
                        "CHISEL_FIRTOOL_PATH": str(support.safe_path(args.firtool).parent),
                        "PATH": str(support.safe_path(args.java).parent) + ":/usr/bin:/bin",
                        "ATLAS_CORE_RTL_DIR": str(out / "rtl")})
            receipt["environment"] = {key: env[key] for key in
                                      ("JAVA_HOME", "JAVA_TOOL_OPTIONS", "CHISEL_FIRTOOL_PATH", "PATH",
                                       "ATLAS_CORE_RTL_DIR")}
            java = str(support.safe_path(args.java))
            runtime_cp = ":".join(receipt["classpath"]["runtime"])
            compiler_cp = ":".join(receipt["classpath"]["compiler"])
            (out / "classes").mkdir()
            source_args = [str(out / "sources" / relative) for relative in sorted(selected)]
            command = [java, "-cp", compiler_cp, "scala.tools.nsc.Main", "-classpath", runtime_cp,
                       "-Xplugin:" + plugin, "-deprecation", "-feature", "-unchecked",
                       "-language:reflectiveCalls", "-d", str(out / "classes")] + source_args + [
                       str(out / "harness/EmitOwnedAtlasCore.scala")]
            emitter = [java, "-cp", str(out / "classes") + ":" + runtime_cp,
                       "atlas.sota.EmitOwnedAtlasCore"]
            receipt["build_command"] = command
            receipt["emitter_command"] = emitter
            receipt["versions"] = [support.version([str(args.java), "-version"], "17.", env),
                                   support.version([str(args.firtool), "--version"], "1.62.1", env)]
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
        raise SystemExit(f"Owned supervisor received signal {number}")
    signal.signal(signal.SIGTERM, interrupted)
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("rtl-root", "provenance", "rocketchip-jar", "diplomacy-jar", "cde-jar",
                 "mill", "java", "firtool", "support-helper", "dependency-export", "maven-root", "out"):
        parser.add_argument("--" + name, required=True, type=Path)
    parser.add_argument("--prepare-only", action="store_true")
    run(parser.parse_args())
