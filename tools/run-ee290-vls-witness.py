#!/usr/bin/env python3
"""Run a supplied final VLS program through the integrated EE290SimConfig.

This runner does not schedule or certify its input. Its caller must retain the
selected applicability/schedule checks separately. Local paths and producer
labels belong to the output receipt only. Simulator-source linkage and memory
response timing remain explicit unproved obligations even when data passes.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import shutil
import subprocess
import sys
import time


def allowed(path: Path) -> Path:
    """Reject restricted components before touching a path or symlink target."""
    for part in path.parts:
        if any(name in part.lower() for name in ("ham" + "mer", "vl" + "si")):
            raise ValueError("restricted path component")
    path = Path(os.path.abspath(path))
    for part in path.parts:
        if any(name in part.lower() for name in ("ham" + "mer", "vl" + "si")):
            raise ValueError("restricted path component")
    current = Path(path.anchor)
    for part in path.parts[1:]:
        current /= part
        if current.is_symlink():
            target = Path(os.readlink(current))
            allowed(target if target.is_absolute() else current.parent / target)
    return path


def identity(path: Path) -> dict:
    path = allowed(path)
    digest = hashlib.sha256()
    count = 0
    with path.open("rb") as stream:
        while block := stream.read(1024 * 1024):
            digest.update(block)
            count += len(block)
    return {"path": str(path), "sha256": digest.hexdigest(), "bytes": count}


def verify(member: dict) -> dict:
    actual = identity(Path(member["path"]))
    if (actual["sha256"], actual["bytes"]) != (member["sha256"], member["bytes"]):
        raise ValueError("manifest member identity mismatch")
    return actual


def source_files(filelist: Path) -> tuple[list[dict], list[dict]]:
    """Read only explicit allowed file-list members; never scan the workspace."""
    seen: set[Path] = set()
    result = []
    options = []

    def visit(path: Path):
        path = allowed(path)
        if path in seen:
            return
        seen.add(path)
        result.append(identity(path))
        for number, line in enumerate(path.read_text().splitlines(), 1):
            line = line.strip()
            if not line or line.startswith(("#", "//")):
                continue
            if line.startswith("+"):
                options.append({"filelist": str(path), "line": number, "text": line,
                                "interpretation": "not interpreted or certified"})
                continue
            if line.startswith("-f "):
                member = Path(line[3:].strip())
                visit(member if member.is_absolute() else path.parent / member)
            elif line.startswith("-"):
                raise ValueError("unsupported simulator file-list option")
            else:
                member = Path(line)
                member = allowed(member if member.is_absolute() else path.parent / member)
                if member not in seen:
                    seen.add(member)
                    result.append(identity(member))

    visit(filelist)
    return result, options


def run(argv: list[str], cwd: Path, log: Path, timeout: float) -> dict:
    start = time.monotonic()
    with log.open("wb") as stream:
        child = subprocess.Popen(argv, cwd=cwd, stdout=stream, stderr=subprocess.STDOUT,
                                 stdin=subprocess.DEVNULL, start_new_session=True)
        try:
            code = child.wait(timeout=timeout)
            timed_out = False
        except subprocess.TimeoutExpired:
            import signal
            os.killpg(child.pid, signal.SIGKILL)
            code = child.wait()
            timed_out = True
    return {"argv": argv, "cwd": str(cwd), "returncode": code,
            "timed_out": timed_out, "elapsed_seconds": time.monotonic() - start,
            "log": identity(log)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("program", "atlas-emit", "simulator", "host-cc", "manifest", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--sim-source-list", type=Path, required=True)
    parser.add_argument("--max-cycles", type=int, default=2000000)
    parser.add_argument("--timeout-seconds", type=float, default=300)
    parser.add_argument("--sim-arg", action="append", default=[])
    args = parser.parse_args()
    for name in ("program", "atlas_emit", "simulator", "host_cc", "manifest", "output", "sim_source_list"):
        setattr(args, name, allowed(getattr(args, name)))
    root = allowed(Path(__file__).absolute().parents[1]).resolve()
    artifact_root = allowed(root / "build/rtl-timing").resolve()
    output = args.output.resolve()
    if output == artifact_root or not output.is_relative_to(artifact_root):
        parser.error("output must be a new directory beneath this repository's ignored build/rtl-timing")
    ignored = subprocess.run(["git", "-C", str(root), "check-ignore", "--no-index", "--quiet", "--", str(output)],
                             capture_output=True, text=True)
    if ignored.returncode:
        parser.error("output must be ignored by this repository")
    args.output = output
    if args.output.exists():
        parser.error("output directory must not already exist")
    if args.max_cycles <= 0 or args.timeout_seconds <= 0:
        parser.error("cycle and timeout bounds must be positive")
    manifest = json.loads(args.manifest.read_text())
    if manifest.get("target_config") != "EE290SimConfig" or manifest.get("state") != "verified":
        raise ValueError("a verified EE290SimConfig retention manifest is required")
    sources, source_options = source_files(args.sim_source_list)
    provenance = {"manifest": identity(args.manifest),
                  "hardware_ir": verify(manifest["hardware_ir"]),
                  "inputs": {key: verify(value) for key, value in manifest["inputs"].items()},
                  "simulator": identity(args.simulator),
                  "simulator_sources": sources,
                  "uninterpreted_source_list_options": source_options,
                  "program": identity(args.program),
                  "atlas_emit": identity(args.atlas_emit), "host_cc": identity(args.host_cc)}
    library_dir = allowed(Path(str(args.simulator) + ".daidir"))
    libraries = []
    with os.scandir(library_dir) as entries:
        for entry in entries:
            # Filter names before any stat/open, including symlink resolution.
            if any(x in entry.name.lower() for x in ("ham" + "mer", "vl" + "si")):
                continue
            if entry.name.endswith(".so"):
                libraries.append(identity(Path(entry.path)))
    provenance["simulator_archive_libraries"] = sorted(libraries, key=lambda x: x["path"])
    if not libraries:
        raise ValueError("no simulator archive libraries recorded")
    build_record = allowed(library_dir / "vcs_rebuild")
    if build_record.exists():
        provenance["observed_simulator_build_record"] = identity(build_record)
        # Treat this saved command as metadata; never execute it. Its explicit
        # package files can occur outside the -f list and must be identified.
        metadata = build_record.read_text()
        direct_sources = []
        for token in shlex.split(metadata):
            if token.startswith("/") and Path(token).suffix in (".sv", ".v", ".cc", ".cpp"):
                direct_sources.append(identity(Path(token)))
        provenance["observed_build_record_direct_sources"] = direct_sources
        provenance["build_record_relationship"] = "observed command metadata; not a certified build receipt"
    args.output.mkdir(parents=True, exist_ok=False)
    host = allowed(root / "test/ee290-vls-host.c")
    provenance["host_source"] = identity(host)
    shutil.copyfile(host, args.output / "ee290-vls-host.executed.c")
    shutil.copyfile(args.program, args.output / "program.mlir")
    shutil.copyfile(__file__, args.output / "run-ee290-vls-witness.executed.py")
    if build_record.exists():
        shutil.copyfile(build_record, args.output / "simulator-build-record.txt")
    receipt = {"schema": "atlas.ee290_vls_witness.v0", "target_config": "EE290SimConfig",
               "state": "prepared", "integrated_execution_passed": False,
               "scheduling_qualified": False, "provenance": provenance, "commands": [],
               "runtime_environment": {key: os.environ[key] for key in
                                       ("VCS_HOME", "VCS_64", "LD_LIBRARY_PATH") if key in os.environ},
               "license_environment_configured": bool(os.environ.get("SNPSLMD_LICENSE_FILE")),
               "limitations": ["Original RTL/Chisel sources and simulator build linkage are not certified.",
                               "Current generated source and archive hashes plus saved VCS command metadata do not certify their build relationship.",
                               "File-list + options are retained but not interpreted; include search and preprocessor state are not reconstructed.",
                               "VCS system runtime libraries and host linker support libraries are not fully pinned.",
                               "No internal request/response trace or independent memory-latency measurement is recorded.",
                               "Caller supplies final scheduling/applicability checks; this runner does not certify them.",
                               "Observer CSR cycle counts include host bus polling and are not kernel timing measurements.",
                               "Only the 6144-byte initialized VMEM window is checked for preserved guards."]}
    report = args.output / "report.json"

    def save():
        report.write_text(json.dumps(receipt, indent=2) + "\n")

    save()
    emitted = run([str(args.atlas_emit), str(args.output / "program.mlir")], args.output,
                  args.output / "emit.log", args.timeout_seconds)
    receipt["commands"].append(emitted)
    if emitted["returncode"]:
        receipt["state"] = "emission_failed"
        save()
        return 1
    lines = (args.output / "emit.log").read_text().splitlines()
    if not lines or len(lines) > 32768 or any(not re.fullmatch(r"[0-9a-fA-F]{8}", x) for x in lines):
        raise ValueError("atlas-emit output is not a bounded instruction word stream")
    include = args.output / "atlas_program.inc"
    include.write_text(f"#define ATLAS_PROGRAM_WORDS {len(lines)}U\nstatic const uint32_t atlas_program[] = {{\n" +
                       "".join(f"  0x{x}U,\n" for x in lines) + "};\n")
    receipt["program_word_count"] = len(lines)
    receipt["generated_include"] = identity(include)
    binary = args.output / "host.riscv"
    built = run([str(args.host_cc), "-std=gnu99", "-O2", "-Wall", "-Wextra", "-Werror",
                 "-fno-common", "-fno-builtin-printf", "-march=rv64imafd", "-mabi=lp64d",
                 "-mcmodel=medany", "-specs=htif_nano.specs", "-static", "-T", "htif.ld",
                 str(args.output / "ee290-vls-host.executed.c"), "-o", str(binary)],
                args.output, args.output / "host-build.log", args.timeout_seconds)
    receipt["commands"].append(built)
    if built["returncode"]:
        receipt["state"] = "host_build_failed"
        save()
        return 1
    receipt["host_binary"] = identity(binary)
    launched = run([str(args.simulator), "+permissive", f"+max-cycles={args.max_cycles}",
                    f"+loadmem={binary}", "+ntb_random_seed=1", *args.sim_arg,
                    "+permissive-off", str(binary)], args.output,
                   args.output / "simulation.log", args.timeout_seconds)
    receipt["commands"].append(launched)
    log = (args.output / "simulation.log").read_text(errors="replace")
    passed = (launched["returncode"] == 0 and not launched["timed_out"] and
              log.count("EE290_VLS_PANEL_PASSED panel=") == 3 and
              "EE290_VLS_PASSED panels=3" in log and
              not re.search(r"EE290_VLS_FAILED|EE290_VLS_MISMATCH|assertion failed|\$fatal|fatal:|error:|error-", log, re.I))
    receipt["state"] = "integrated_execution_passed" if passed else "integrated_execution_failed"
    if not passed and re.search(r"queuing for license|license checkout failed|server node is down", log, re.I):
        receipt["state"] = "runtime_license_unavailable"
    receipt["integrated_execution_passed"] = passed
    receipt["observations"] = [x for x in log.splitlines() if x.startswith("EE290_VLS_")]
    save()
    print(json.dumps({"state": receipt["state"], "report": str(report),
                      "scheduling_qualified": False}))
    return 0 if passed else 1


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (OSError, ValueError, KeyError) as error:
        print(f"EE290 witness: {error}", file=sys.stderr)
        sys.exit(2)
