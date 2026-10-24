#!/usr/bin/env python3
"""Build the raw-word driver against an immutable, authenticated AtlasCore SV."""
import sys
from pathlib import Path

TOOLS_ROOT = Path(__file__).absolute().parents[1]
if (TOOLS_ROOT / "atlas_path_guard.py").is_file():
    sys.path.insert(0, str(TOOLS_ROOT))

import argparse
from atlas_tooling import sha
from atlas_path_guard import checked
import importlib.util
import json
import os

sys.dont_write_bytecode = True


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


def run(args):

    initial_path = checked
    core_root = initial_path(args.core_root)
    core_bytes = initial_path(core_root / "receipt.json").read_bytes()
    if sha(core_bytes) != args.core_sha256:
        raise ValueError("Current core receipt differs")
    core = json.loads(core_bytes)
    probe_record = next(record for record in core["records"] if record["path"] == str(core_root / "probe.py"))
    probe_path = initial_path(probe_record["path"])
    if sha(probe_path.read_bytes()) != probe_record["sha256"]:
        raise ValueError("Frozen compile probe differs")
    if core["status"] != "passed_compile_elaboration_only" or not core["source_artifacts_preserved"]:
        raise ValueError("Core did not pass source-bound elaboration")
    support_path = initial_path(core_root / "reviewed-support.py")
    support_record = next(record for record in core["records"] if record["path"] == str(support_path))
    if sha(support_path.read_bytes()) != support_record["sha256"]:
        raise ValueError("Frozen support helper differs")
    probe = load(probe_path, "frozen_core_probe")
    support = probe.helper(support_path)
    out = support.safe_path(args.out)
    out.mkdir(parents=True, exist_ok=False)
    records = []
    receipt = {"schema_version": 1, "status": "preparing", "command": sys.argv,
        "claim": "Raw instruction driver build only; no execution result", "process_id": os.getpid(), "records": records}

    def bind(path, expected=None):
        record = support.identity(path, expected)
        records.append(record)
        return record

    try:
        receipt["shared_support"] = [bind(sys.modules[name].__file__) for name in ("atlas_path_guard", "atlas_tooling")]
        bind(core_root / "receipt.json", args.core_sha256)
        for record in core["records"]:
            bind(record["path"], record["sha256"])
        selected = {name: bind(getattr(args, name)) for name in ("verilator_binary", "cxx", "ar")}
        runtimes = [bind(path) for path in args.runtime_library]
        verilator_root = support.safe_path(args.verilator_root)
        bind(verilator_root / "include/verilated_std.sv")
        receipt["tools"] = selected
        receipt["runtime_libraries"] = runtimes
        make = bind(args.make)
        own = support.safe_path(Path(__file__).absolute().parent)
        for name in ("build_runner.py", "raw_runner.cpp"):
            record = bind(own / name)
            support.copy_source(record["path"], out / name, record["sha256"])
            bind(out / name, record["sha256"])
        layout = bind(own.parents[1] / "runtime/atlas_host_layout.h")
        support.copy_source(layout["path"], out / "atlas_host_layout.h", layout["sha256"])
        bind(out / "atlas_host_layout.h", layout["sha256"])
        sv = core["system_verilog"]
        support.copy_source(sv["path"], out / "AtlasCore.sv", sv["sha256"])
        bind(out / "AtlasCore.sv", sv["sha256"])
        env = os.environ.copy()
        env.update({"VERILATOR_ROOT": str(verilator_root),
            "PATH": str(Path(selected["cxx"]["path"]).parent) + os.pathsep + os.environ.get("PATH", ""),
            "LD_PRELOAD": ":".join(record["path"] for record in runtimes), "MAKEFLAGS": "-j2"})
        receipt["environment"] = {key: env[key] for key in ("VERILATOR_ROOT", "PATH", "LD_PRELOAD", "MAKEFLAGS")}
        receipt["versions"] = [support.version([selected["verilator_binary"]["path"], "--version"], env=env),
            support.version([selected["cxx"]["path"], "--version"], env=env),
            support.version([make["path"], "--version"], env=env)]
        generate = [selected["verilator_binary"]["path"], "--cc", "--exe", "--top-module", "AtlasCore", "--Mdir",
            str(out / "obj_dir"), "-o", "raw_runner", "--threads", "1", "--x-initial", "unique", "--x-assign", "unique",
            "--assert", "-Wno-fatal", "--output-split", "20000", "--output-split-cfuncs", "100", "-CFLAGS", "-O0 -std=c++17",
            str(out / "AtlasCore.sv"), str(out / "raw_runner.cpp")]
        compile = [make["path"], "-C", str(out / "obj_dir"), "-f", "VAtlasCore.mk", "-j2", "CXX=" + selected["cxx"]["path"],
            "LINK=" + selected["cxx"]["path"], "AR=" + selected["ar"]["path"], "OPT_FAST=-O0", "OPT_SLOW=-O0"]
        receipt["status"] = "building"
        support.write_json(out / "receipt.json", receipt)
        for stage, argv in (("generate", generate), ("compile", compile)):
            receipt[stage] = probe.bounded(argv, out, env, out / (stage + ".log"), support)
            bind(out / (stage + ".log"))
            bind(out / (stage + ".resources.json"))
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
        raise SystemExit(f"Build received signal {number}")

    signal.signal(signal.SIGTERM, interrupted)
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("core-root", "make", "out", "verilator-binary", "verilator-root", "cxx", "ar"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--core-sha256", required=True)
    parser.add_argument("--runtime-library", type=Path, action="append", default=[])
    run(parser.parse_args())
