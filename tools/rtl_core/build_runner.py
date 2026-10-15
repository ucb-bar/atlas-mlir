#!/usr/bin/env python3
"""Build the raw-word driver against an immutable, authenticated AtlasCore SV."""
import argparse
import importlib.util
import json
import os
import sys
from pathlib import Path

sys.dont_write_bytecode = True
CORE_RECEIPT = "68a1fbe069fd982fcb46690a2592d35ba59256ea3182f2271e8ccf59bd9594d1"
TOOLS_RECEIPT = "534dfe2aab4cd1c0f14c749c6cd92cda0a6567e041f2b2fd6a88d43fd359ecfc"


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


def run(args):
    # This exact helper was already admitted and frozen by the current-core probe.
    import hashlib
    import stat
    def initial_path(value):
        path = Path(value).absolute()
        if ".." in path.parts or any(bad in part.lower() for part in path.parts for bad in ("hammer", "vlsi")):
            raise ValueError("Unadmitted lexical path")
        current = Path(path.anchor)
        for part in path.parts[1:]:
            current /= part
            if stat.S_ISLNK(os.lstat(current).st_mode):
                raise ValueError("Bootstrap symlink refused")
        return path
    core_root = initial_path(args.core_root)
    core_bytes = initial_path(core_root / "receipt.json").read_bytes()
    if hashlib.sha256(core_bytes).hexdigest() != CORE_RECEIPT:
        raise ValueError("Current core receipt differs")
    core = json.loads(core_bytes)
    probe_record = next(record for record in core["records"] if record["path"] == str(core_root / "probe.py"))
    probe_path = initial_path(probe_record["path"])
    if hashlib.sha256(probe_path.read_bytes()).hexdigest() != probe_record["sha256"]:
        raise ValueError("Frozen supervisor differs")
    probe = load(probe_path, "frozen_core_probe")
    support = probe.helper(core_root / "reviewed-support.py")
    if core["status"] != "passed_compile_elaboration_only" or not core["source_artifacts_preserved"]:
        raise ValueError("Core did not pass source-bound elaboration")
    out = support.safe_path(args.out)
    out.mkdir(parents=True, exist_ok=False)
    records = []
    receipt = {"schema_version": 1, "status": "preparing", "command": sys.argv,
               "claim": "Raw instruction driver build only; no execution result",
               "supervisor_pid": os.getpid(), "records": records}
    def bind(path, expected=None):
        record = support.identity(path, expected); records.append(record); return record
    try:
        bind(core_root / "receipt.json", CORE_RECEIPT)
        for record in core["records"]:
            bind(record["path"], record["sha256"])
        bind(args.tool_receipt, TOOLS_RECEIPT)
        tools = json.loads(support.safe_path(args.tool_receipt).read_text())
        selected = {name: tools["tools"][name] for name in ("verilator_binary", "cxx", "ar")}
        for record in list(selected.values()) + tools["runtime_libraries"] + [tools["verilator_runtime_header"]]:
            bind(record["path"], record["sha256"])
        make = bind("/usr/bin/make")
        own = support.safe_path(Path(__file__).absolute().parent)
        for name in ("build_runner.py", "raw_runner.cpp"):
            record = bind(own / name)
            support.copy_source(record["path"], out / name, record["sha256"])
            bind(out / name, record["sha256"])
        sv = core["system_verilog"]
        support.copy_source(sv["path"], out / "AtlasCore.sv", sv["sha256"])
        bind(out / "AtlasCore.sv", sv["sha256"])
        env = os.environ.copy()
        env.update({"VERILATOR_ROOT": tools["environment"]["VERILATOR_ROOT"],
                    "PATH": str(Path(selected["cxx"]["path"]).parent) + ":/usr/bin:/bin",
                    "LD_PRELOAD": ":".join(record["path"] for record in tools["runtime_libraries"]),
                    "MAKEFLAGS": "-j2"})
        receipt["environment"] = {key: env[key] for key in ("VERILATOR_ROOT", "PATH", "LD_PRELOAD", "MAKEFLAGS")}
        receipt["versions"] = [support.version([selected["verilator_binary"]["path"], "--version"], "5.022", env),
                               support.version([selected["cxx"]["path"], "--version"], "13.2.0", env),
                               support.version([make["path"], "--version"], "GNU Make", env)]
        generate = [selected["verilator_binary"]["path"], "--cc", "--exe", "--top-module", "AtlasCore",
                    "--Mdir", str(out / "obj_dir"), "-o", "raw_runner", "--threads", "1",
                    "--x-initial", "unique", "--x-assign", "unique", "--assert", "-Wno-fatal",
                    "--output-split", "20000", "--output-split-cfuncs", "100",
                    "-CFLAGS", "-O0 -std=c++17", str(out / "AtlasCore.sv"), str(out / "raw_runner.cpp")]
        compile = [make["path"], "-C", str(out / "obj_dir"), "-f", "VAtlasCore.mk", "-j2",
                   "CXX=" + selected["cxx"]["path"], "LINK=" + selected["cxx"]["path"],
                   "AR=" + selected["ar"]["path"], "OPT_FAST=-O0", "OPT_SLOW=-O0"]
        receipt["status"] = "building"
        support.write_json(out / "receipt.json", receipt)
        for stage, argv in (("generate", generate), ("compile", compile)):
            receipt[stage] = probe.bounded(argv, out, env, out / (stage + ".log"), support)
            bind(out / (stage + ".log")); bind(out / (stage + ".resources.json"))
            support.write_json(out / "receipt.json", receipt)
            if receipt[stage]["returncode"]:
                raise RuntimeError(f"Owned {stage} exited {receipt[stage]['returncode']}")
        receipt["executable"] = bind(out / "obj_dir/raw_runner")
        receipt["status"] = "passed_build_only"
    except BaseException as error:
        receipt["status"] = "failed"
        receipt["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        try:
            for record in records:
                support.identity(record["path"], record["sha256"])
            receipt["source_artifacts_preserved"] = True
        except Exception as error:
            receipt["source_artifacts_preserved"] = False
            receipt["bookend_error"] = f"{type(error).__name__}: {error}"
            receipt["status"] = "failed"
        support.write_json(out / "receipt.json", receipt)
        if not receipt.get("source_artifacts_preserved"):
            raise RuntimeError("Build source/tool bookends failed")


if __name__ == "__main__":
    import signal
    def interrupted(number, frame):
        raise SystemExit(f"Owned supervisor received signal {number}")
    signal.signal(signal.SIGTERM, interrupted)
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("core-root", "tool-receipt", "out"):
        parser.add_argument("--" + name, type=Path, required=True)
    run(parser.parse_args())
