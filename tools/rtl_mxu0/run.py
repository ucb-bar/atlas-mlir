#!/usr/bin/env python3
"""Build one source-pinned MXU0 component and replay an existing encoded tile.

Only manifest-listed Scala files in three explicit main-source roots are copied.
This does not elaborate AtlasCore or run an instruction stream, DMA, or an SoC.
"""

import argparse
import ast
import hashlib
import importlib.util
import json
import os
import resource
import signal
import stat
import subprocess
import sys
import time
from pathlib import Path, PurePosixPath

import numpy as np


RTL_REVISION = "2ae0bef209df6db78c3de18e8f651bb43855cce9"
RECEIPT_SHA = "34ea95ac17b27b1c769bc594f0ce87a6c14a430a0ce0aa28fd474a08028d8595"
SOURCE_NPZ_SHA = "3145e772d7e8f807ddf84ac7d113dcf504d546c42b499878cd544fdded56f148"
ROOTS = ("src/main/scala/atlas", "dependencies/fpex/src/main/scala",
         "dependencies/sp26-fp-units/src/main/scala")


def safe_path(value):
    path = Path(value).absolute()
    if ".." in path.parts or any(bad in part.lower() for part in path.parts for bad in ("hammer", "vlsi")):
        raise ValueError("Unadmitted path component")
    current = Path(path.anchor)
    for part in path.parts[1:]:
        current /= part
        try:
            mode = os.lstat(current).st_mode
        except FileNotFoundError:
            break
        if stat.S_ISLNK(mode):
            raise ValueError(f"Symlink component refused: {current}")
    return path


def digest(path):
    result = hashlib.sha256()
    with safe_path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def identity(path, expected=None):
    path = safe_path(path)
    value = {"path": str(path), "sha256": digest(path)}
    if expected is not None and value["sha256"] != expected:
        raise ValueError(f"Identity mismatch: {path}")
    return value


def write_json(path, value):
    safe_path(path).write_text(json.dumps(value, indent=2) + "\n")


def copy_source(source, destination, expected):
    identity(source, expected)
    destination = safe_path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(safe_path(source).read_bytes())
    identity(destination, expected)


def version(command, expected, env=None, cwd=None):
    result = subprocess.run(command, capture_output=True, text=True, timeout=60, check=True, env=env, cwd=cwd)
    text = result.stdout + result.stderr
    if expected not in text:
        raise ValueError(f"Unexpected tool version for {command}: {text}")
    return {"command": command, "text": text.strip()}


def run(args):
    started = time.monotonic()
    support = safe_path(Path(__file__).absolute().parent)
    for name in ("run.py", "verilator_wrapper.py"):
        ast.parse(safe_path(support / name).read_text())
    out = safe_path(args.out)
    out.mkdir(parents=True, exist_ok=False)
    receipt = {"status": "preparing", "claim": "MXU0 component RTL only; no full program, DMA or SoC timing",
               "rtl_revision": RTL_REVISION, "python": sys.version, "command": sys.argv,
               "cwd": str(Path.cwd()), "pid": os.getpid(), "source_closure": []}
    receipt_path = out / "receipt.json"
    write_json(receipt_path, receipt)
    try:
        receipt["model_receipt"] = identity(args.input / "receipt.json", RECEIPT_SHA)
        original = json.loads(safe_path(args.input / "receipt.json").read_text())
        if original["source_inputs"]["npz"] != SOURCE_NPZ_SHA or original["shape"] != [32, 64, 32]:
            raise ValueError("Expected the frozen ResNet M32 K64 N32 source")
        receipt["source_npz_sha256"] = SOURCE_NPZ_SHA
        selected_inputs = {}
        for name in ("encoded.npz", "result.npz", "expected-bf16-halves.bin", "program.bin", "source-manifest.json"):
            selected_inputs[name] = identity(args.input / name, original["artifacts"][name])
        if json.loads((args.input / "source-manifest.json").read_text())["npz_sha256"] != SOURCE_NPZ_SHA:
            raise ValueError("Native source manifest binding differs")
        receipt["input_artifacts"] = selected_inputs
        receipt["oracle"] = identity(args.oracle, original["oracle_sha256"])
        copy_source(args.oracle, out / "oracle.py", original["oracle_sha256"])
        receipt["oracle_copy"] = identity(out / "oracle.py", original["oracle_sha256"])
        spec = importlib.util.spec_from_file_location("frozen_tile_arithmetic", out / "oracle.py")
        arithmetic = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(arithmetic)
        with np.load(args.input / "encoded.npz", allow_pickle=False) as bundle:
            if set(bundle.files) != {"A", "B"}:
                raise ValueError("Unexpected encoded operand keys")
            a, b = bundle["A"].copy(), bundle["B"].copy()
        if a.shape != (32, 64) or b.shape != (64, 32) or a.dtype != np.uint8 or b.dtype != np.uint8:
            raise ValueError("Encoded operand shape or dtype differs")
        expected = np.array([[arithmetic.mxu0_dot(a[row], b[:, col]) for col in range(32)]
                             for row in range(32)], dtype=np.uint16)
        with np.load(args.input / "result.npz", allow_pickle=False) as bundle:
            if not np.array_equal(bundle["expected_bits"], expected):
                raise AssertionError("Saved expected words differ from the independent rational oracle")
            target = bundle["target_C"]
            if target.dtype != np.float32 or np.any(target.view(np.uint32) & 65535):
                raise ValueError("Saved model outputs are not exact BF16 float32 views")
            if not np.array_equal((target.view(np.uint32) >> 16).astype(np.uint16), expected):
                raise AssertionError("Saved model execution differs from the independent oracle")
        if arithmetic.output_halves(expected.tolist()) != (args.input / "expected-bf16-halves.bin").read_bytes():
            raise AssertionError("Raw expected output halves differ")
        write_json(out / "input.json", {"A": a.tolist(), "B": b.tolist(), "expected_bf16": expected.tolist()})
        receipt["input"] = identity(out / "input.json")
        receipt["prebuild_oracle"] = {"all_1024_model_words_match": True, "k": 64,
                                     "expected_word_sha256": hashlib.sha256(expected.tobytes()).hexdigest()}
        revision = subprocess.run(["git", "-C", str(args.rtl_root), "-c", "submodule.recurse=false",
                                   "rev-parse", "HEAD"], text=True, capture_output=True, check=True).stdout.strip()
        if revision != RTL_REVISION:
            raise ValueError("RTL checkout revision differs")
        manifest = json.loads(args.provenance.read_text())
        receipt["provenance_manifest"] = identity(args.provenance)
        source_root = out / "sources"
        counts = {root: 0 for root in ROOTS}
        for relative, sha in sorted(manifest["sources"].items()):
            parts = PurePosixPath(relative).parts
            if any(bad in part.lower() for part in parts for bad in ("hammer", "vlsi")):
                continue
            selected = next((root for root in ROOTS if relative.startswith(root + "/")), None)
            if selected is None:
                continue
            if PurePosixPath(relative).is_absolute() or ".." in parts or not relative.endswith(".scala"):
                raise ValueError("Malformed source entry")
            copy_source(args.rtl_root / relative, source_root / relative, sha)
            receipt["source_closure"].append({"relative": relative, "sha256": sha})
            counts[selected] += 1
        if not all(counts.values()):
            raise ValueError("A declared main source root was empty")
        receipt["source_counts"] = counts
        for name, destination in (("build.mill", out / "build.mill"),
                                  ("WorkloadMxuTraceTest.scala", out / "harness/WorkloadMxuTraceTest.scala"),
                                  ("verilator_wrapper.py", out / "bin/verilator")):
            source = identity(support / name)
            copy_source(support / name, destination, source["sha256"])
            receipt.setdefault("owned_support", []).append(source)
        (out / "bin/verilator").chmod(0o755)
        receipt["owned_support"].append(identity(__file__))
        receipt["tools"] = {name: identity(value) for name, value in
                            (("mill", args.mill), ("java", args.java), ("verilator", args.verilator),
                             ("firtool", args.firtool))}
        verilator_root = safe_path(args.verilator.parent.parent / "share/verilator")
        verilator_binary = safe_path(args.verilator.parent / "verilator_bin")
        receipt["tools"]["verilator_binary"] = identity(verilator_binary)
        cxx = safe_path(args.verilator.parent / "x86_64-conda-linux-gnu-c++")
        ar = safe_path(args.verilator.parent / "x86_64-conda-linux-gnu-ar")
        receipt["tools"].update(cxx=identity(cxx), ar=identity(ar))
        runtime_root = safe_path(args.verilator.parent.parent / "lib")
        runtimes = [identity(runtime_root / name) for name in ("libstdc++.so.6.0.33", "libgcc_s.so.1")]
        receipt["runtime_libraries"] = runtimes
        receipt["verilator_runtime_header"] = identity(verilator_root / "include/verilated_std.sv")
        env = dict(os.environ)
        env.update(PATH=str(out / "bin") + ":" + str(args.verilator.parent) + ":/usr/bin:/bin",
                   JAVA_HOME=str(args.java.parent.parent),
                   JAVA_TOOL_OPTIONS="-Xmx8G -Xss8M -XX:ActiveProcessorCount=3",
                   CHISEL_FIRTOOL_PATH=str(args.firtool.parent),
                   VERILATOR_ROOT=str(verilator_root),
                   ATLAS_VERILATOR_WRAPPER=str(out / "bin/verilator"),
                   ATLAS_REAL_VERILATOR=str(verilator_binary), ATLAS_SIM_WORKSPACE=str(out / "simulation"),
                   ATLAS_MAKE_OVERRIDES=f"CXX={cxx} LINK={cxx} AR={ar}",
                   LD_PRELOAD=":".join(value["path"] for value in runtimes),
                   ATLAS_TILE_INPUT=str(out / "input.json"), ATLAS_TILE_TRACE=str(out / "trace.json"),
                   OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MAKEFLAGS="-j2")
        receipt["environment"] = {key: env[key] for key in
                                  ("PATH", "JAVA_HOME", "JAVA_TOOL_OPTIONS", "CHISEL_FIRTOOL_PATH",
                                   "VERILATOR_ROOT", "LD_PRELOAD", "ATLAS_MAKE_OVERRIDES", "ATLAS_SIM_WORKSPACE", "ATLAS_TILE_INPUT", "ATLAS_TILE_TRACE", "MAKEFLAGS")}
        # Version calls do not evaluate a project build or an old compiled model.
        receipt["tool_versions"] = {
            "java": version([str(args.java), "-version"], 'version "17.'),
            "verilator": version([str(args.verilator), "--version"], "Verilator 5."),
            "firtool": version([str(args.firtool), "--version"], "1.128.0"),
            "mill": version(["/bin/bash", str(args.mill), "--version"], "0.13.0-M0", env=env, cwd=out),
            "cxx": version([str(cxx), "--version"], "13.2.0"),
        }
        allowed_cpus = sorted(os.sched_getaffinity(0))[:3]
        if len(allowed_cpus) != 3:
            raise ValueError("Three allowed CPUs are required for this build lease")
        os.sched_setaffinity(0, allowed_cpus)
        receipt["cpu_affinity"] = allowed_cpus
        command = ["/bin/bash", str(args.mill), "--no-server", "-j", "1", "atlas.test.testOnly", "atlas.sota.WorkloadMxuTraceTest"]
        receipt.update(status="building_and_running", build_command=command)
        write_json(receipt_path, receipt)
        with (out / "build.log").open("w") as log:
            process = subprocess.Popen(command, cwd=out, env=env, stdout=log,
                                       stderr=subprocess.STDOUT, start_new_session=True)
            receipt["build_process_group"] = process.pid
            write_json(receipt_path, receipt)
            try:
                exit_code = process.wait(timeout=args.timeout)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()
                raise
        receipt["build_exit_code"] = exit_code
        receipt["child_peak_rss_kib"] = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss
        if exit_code:
            raise RuntimeError(f"Mill test failed with exit {exit_code}; see {out / 'build.log'}")
        traces = json.loads((out / "trace.json").read_text())
        if len(traces) != 2:
            raise AssertionError("Expected two initialized RTL executions")
        for run in traces:
            if run["mismatches"] or run["mreg_write_rows"] != 64 or run["unique_output_rows"] != 64 or not run["quiet_after_cycle_410"]:
                raise AssertionError("RTL output, extent or guard gate failed")
            if not np.array_equal(np.asarray(run["actual_bf16"], dtype=np.uint16), expected):
                raise AssertionError("Raw RTL output differs from independent rational oracle")
        for source in receipt["source_closure"]:
            identity(args.rtl_root / source["relative"], source["sha256"])
            identity(source_root / source["relative"], source["sha256"])
        for value in list(selected_inputs.values()) + receipt["owned_support"] + list(receipt["tools"].values()) + runtimes:
            identity(value["path"], value["sha256"])
        identity(args.oracle, receipt["oracle"]["sha256"])
        for key in ("model_receipt", "provenance_manifest", "oracle_copy", "input", "verilator_runtime_header"):
            value = receipt[key]
            identity(value["path"], value["sha256"])
        receipt.update(status="passed_component_rtl", repeats=2, raw_bf16_words_per_run=1024,
                       caller_inputs_and_mreg_guards_preserved=True, source_bookends_verified=True,
                       full_program_executed=False, dma_executed=False,
                       trace=identity(out / "trace.json"), input=identity(out / "input.json"))
    except BaseException as error:
        receipt.update(status="failed", error={"type": type(error).__name__, "message": str(error)})
        raise
    finally:
        receipt["wall_seconds"] = time.monotonic() - started
        write_json(receipt_path, receipt)
        print(json.dumps({"status": receipt["status"], "out": str(out),
                          "wall_seconds": receipt["wall_seconds"]}, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("rtl-root", "provenance", "input", "oracle", "mill", "java", "verilator", "firtool", "out"):
        parser.add_argument("--" + name, type=safe_path, required=True)
    parser.add_argument("--timeout", type=int, default=1800)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
