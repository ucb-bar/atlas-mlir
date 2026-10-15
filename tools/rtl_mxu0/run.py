#!/usr/bin/env python3
"""Build a source-pinned MXU0 or VPU component and replay authenticated cases.

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
import re
import signal
import stat
import subprocess
import sys
import time
from pathlib import Path, PurePosixPath
from types import SimpleNamespace

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
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
                               env=env, cwd=cwd, start_new_session=True)
    try:
        stdout, stderr = process.communicate(timeout=60)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGTERM)
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait()
        raise
    if process.returncode:
        raise RuntimeError(f"Tool version command failed: {command}: {stdout + stderr}")
    text = stdout + stderr
    if expected not in text:
        raise ValueError(f"Unexpected tool version for {command}: {text}")
    return {"command": command, "text": text.strip()}


def prepare_case(case, out):
    """Authenticate an existing model epoch; select its PRE-rescale MXU words."""
    name = case["id"]
    if not re.fullmatch(r"[a-z0-9_-]+", name):
        raise ValueError("Malformed case id")
    source = safe_path(case["input"])
    records = [identity(source / "receipt.json", case["receipt_sha256"])]
    original = json.loads(safe_path(source / "receipt.json").read_text())
    m, k, n = original["shape"]
    if (m, n) != (32, 32) or not 0 < k <= 2048 or k % 32:
        raise ValueError("Expected M=N=32 and a positive K multiple of32, at most2048")
    if original["status"] != "passed_model_only" or original["source_inputs"]["npz"] != case["source_npz_sha256"]:
        raise ValueError("Model status or native source binding differs")
    recipe = original["recipe"]
    if recipe not in ("atlas_mxu0_fp8_unit_bf16acc_v1", "atlas_mxu0_fp8_tensor_pow2_bf16acc_v1"):
        raise ValueError("Unsupported MXU recipe")
    for filename in ("encoded.npz", "result.npz", "expected-bf16-halves.bin", "program.bin", "source-manifest.json"):
        records.append(identity(source / filename, original["artifacts"][filename]))
    if json.loads(safe_path(source / "source-manifest.json").read_text())["npz_sha256"] != case["source_npz_sha256"]:
        raise ValueError("Native source manifest binding differs")
    oracle = identity(case["oracle"], original["oracle_sha256"])
    records.append(oracle)
    destination = safe_path(out / "oracles" / (oracle["sha256"] + ".py"))
    if destination.exists():
        records.append(identity(destination, oracle["sha256"]))
    else:
        copy_source(oracle["path"], destination, oracle["sha256"])
        records.append(identity(destination, oracle["sha256"]))
    spec = importlib.util.spec_from_file_location("arithmetic_" + oracle["sha256"], destination)
    arithmetic = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(arithmetic)
    with np.load(safe_path(source / "encoded.npz"), allow_pickle=False) as bundle:
        if set(bundle.files) != {"A", "B"}:
            raise ValueError("Unexpected encoded operand keys")
        a, b = bundle["A"].copy(), bundle["B"].copy()
    if a.shape != (32, k) or b.shape != (k, 32) or a.dtype != np.uint8 or b.dtype != np.uint8:
        raise ValueError("Encoded operand shape or dtype differs")
    def dot(left, right):
        return np.array([[arithmetic.mxu0_dot(left[row], right[:, col]) for col in range(32)]
                         for row in range(32)], dtype=np.uint16)
    expected = dot(a, b)
    with np.load(safe_path(source / "result.npz"), allow_pickle=False) as bundle:
        raw_key = "expected_bits" if recipe == "atlas_mxu0_fp8_unit_bf16acc_v1" else "raw_accumulator_bits"
        if not np.array_equal(bundle[raw_key], expected):
            raise AssertionError("Saved raw accumulator words differ from the rational oracle")
        if recipe == "atlas_mxu0_fp8_unit_bf16acc_v1":
            target = bundle["target_C"]
            if target.dtype != np.float32 or np.any(target.view(np.uint32) & 65535):
                raise ValueError("Saved model outputs are not exact BF16 float32 views")
            if not np.array_equal((target.view(np.uint32) >> 16).astype(np.uint16), expected):
                raise AssertionError("Saved model execution differs from the independent oracle")
            if arithmetic.output_halves(expected.tolist()) != safe_path(source / "expected-bf16-halves.bin").read_bytes():
                raise AssertionError("Raw expected output halves differ")
    controls = {}
    if case.get("controls", False):
        if k < 64:
            raise ValueError("Reset-accumulator control requires at least two tiles")
        wrong_b = np.concatenate([b[offset:offset + 32].T for offset in range(0, k, 32)], axis=0)
        controls["wrong_transpose"] = dot(a, wrong_b)
        controls["wrong_final_reset"] = dot(a[:, -32:], b[-32:])
        if any(np.array_equal(value, expected) for value in controls.values()):
            raise AssertionError("Selected source does not witness a required negative control")
    data = {"id": name, "recipe": recipe, "A": a.tolist(), "B": b.tolist(),
            "expected_bf16": expected.tolist(),
            "controls": {key: value.tolist() for key, value in controls.items()}}
    provenance = {"id": name, "shape": [m, k, n], "recipe": recipe,
                  "source_npz_sha256": case["source_npz_sha256"], "source_records": records,
                  "raw_output_domain": "before any VPU output rescale",
                  "model_target_words_checked": recipe == "atlas_mxu0_fp8_unit_bf16acc_v1",
                  "expected_word_sha256": hashlib.sha256(expected.tobytes()).hexdigest(),
                  "negative_control_witnesses": {key: int(np.count_nonzero(value != expected))
                                                 for key, value in controls.items()}}
    return data, provenance


def run(args):
    started = time.monotonic()
    support = safe_path(Path(__file__).absolute().parent)
    vector = args.vector_from is not None
    harness = "WorkloadVectorTraceTest.scala" if vector else "WorkloadMxuTraceTest.scala"
    names = ["run.py", "verilator_wrapper.py", "build.mill", harness]
    if vector:
        names.append("prepare_vector.py")
    support_before = {name: identity(support / name) for name in names}
    for name in ("run.py", "verilator_wrapper.py"):
        ast.parse(safe_path(support / name).read_text())
    out = safe_path(args.out)
    out.mkdir(parents=True, exist_ok=False)
    receipt = {"status": "preparing", "claim": ("VPU pack/rescale component RTL only" if vector else "MXU0 component RTL only") + "; no full program, DMA or SoC timing",
               "engine": "vector" if vector else "mxu0",
               "rtl_revision": RTL_REVISION, "python": sys.version, "command": sys.argv,
               "cwd": str(Path.cwd()), "pid": os.getpid(), "source_closure": []}
    receipt_path = out / "receipt.json"
    write_json(receipt_path, receipt)
    try:
        if vector:
            if args.cases is not None or args.input is not None or args.oracle is not None or not args.mxu_receipt_sha256:
                raise ValueError("Vector mode requires only --vector-from and --mxu-receipt-sha256")
            copy_source(support / "prepare_vector.py", out / "prepare_vector.py", support_before["prepare_vector.py"]["sha256"])
            spec = importlib.util.spec_from_file_location("frozen_vector_preparation", out / "prepare_vector.py")
            vector_preparation = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(vector_preparation)
            prepared, provenance = vector_preparation.prepare(
                SimpleNamespace(safe_path=safe_path, identity=identity), args.vector_from, args.mxu_receipt_sha256)
            source_bindings = provenance["source_records"]
            receipt["vector_cases"] = provenance["cases"]
            receipt["vector_source_records"] = source_bindings
            source_bindings.append(identity(out / "prepare_vector.py", support_before["prepare_vector.py"]["sha256"]))
            write_json(out / "input.json", prepared)
            receipt["input"] = identity(out / "input.json")
        else:
            if args.cases is not None:
                receipt["case_manifest"] = identity(args.cases)
                manifest_cases = json.loads(args.cases.read_text())
                if manifest_cases["schema_version"] != 1:
                    raise ValueError("Unsupported case manifest schema")
                cases = manifest_cases["cases"]
            else:
                if args.input is None or args.oracle is None:
                    raise ValueError("Provide --cases, or both --input and --oracle")
                cases = [{"id": "resnet-unit", "input": str(args.input), "oracle": str(args.oracle),
                          "receipt_sha256": RECEIPT_SHA, "source_npz_sha256": SOURCE_NPZ_SHA,
                          "controls": True}]
            if not cases or len(cases) > 8 or len({case["id"] for case in cases}) != len(cases):
                raise ValueError("Expected one to eight uniquely named cases")
            prepared, provenance = [], []
            for case in cases:
                data, binding = prepare_case(case, out)
                prepared.append(data)
                provenance.append(binding)
            write_json(out / "input.json", {"schema_version": 2, "cases": prepared})
            receipt["input"] = identity(out / "input.json")
            receipt["cases"] = provenance
            source_bindings = [record for case in provenance for record in case["source_records"]]
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
                                  (harness, out / "harness" / harness),
                                  ("verilator_wrapper.py", out / "bin/verilator")):
            source = support_before[name]
            copy_source(support / name, destination, source["sha256"])
            receipt.setdefault("owned_support", []).append(source)
        (out / "bin/verilator").chmod(0o755)
        receipt["owned_support"].append(support_before["run.py"])
        if vector:
            receipt["owned_support"].append(support_before["prepare_vector.py"])
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
            "mill": version(["/bin/bash", str(args.mill), "--no-server", "--version"], "0.13.0-M0", env=env, cwd=out),
            "cxx": version([str(cxx), "--version"], "13.2.0"),
        }
        allowed_cpus = sorted(os.sched_getaffinity(0))[:3]
        if len(allowed_cpus) != 3:
            raise ValueError("Three allowed CPUs are required for this build lease")
        os.sched_setaffinity(0, allowed_cpus)
        receipt["cpu_affinity"] = allowed_cpus
        test_class = "atlas.sota.WorkloadVectorTraceTest" if vector else "atlas.sota.WorkloadMxuTraceTest"
        command = ["/bin/bash", str(args.mill), "--no-server", "-j", "1", "atlas.test.testOnly", test_class]
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
        if vector:
            execution_summary = vector_preparation.validate(prepared, traces)
        else:
            expected_runs = {(case["id"], "normal", repeat) for case in prepared for repeat in range(2)}
            expected_runs.update((case["id"], name, 0) for case in prepared for name in case["controls"])
            if {(trace["case_id"], trace["run_kind"], trace["repeat"]) for trace in traces} != expected_runs or len(traces) != len(expected_runs):
                raise AssertionError("Missing or duplicate initialized RTL cases")
            for trace in traces:
                case = next(case for case in prepared if case["id"] == trace["case_id"])
                declared = case["expected_bf16"] if trace["run_kind"] == "normal" else case["controls"][trace["run_kind"]]
                if trace["declared_mismatches"] or trace["mreg_write_rows"] != 64 or trace["unique_output_rows"] != 64 or not trace["quiet_after_margin"]:
                    raise AssertionError("RTL output, extent or guard gate failed")
                if not np.array_equal(np.asarray(trace["actual_bf16"], dtype=np.uint16), np.asarray(declared, dtype=np.uint16)):
                    raise AssertionError("Raw RTL words differ from independently declared arithmetic")
                if bool(trace["mismatches"]) != (trace["run_kind"] != "normal"):
                    raise AssertionError("Hardware mutation refusal did not witness the expected failure")
            execution_summary = {"repeats_per_normal_case": 2, "raw_bf16_words_per_run": 1024,
                                 "hardware_negative_controls": sum(len(case["controls"]) for case in prepared),
                                 "output_vpu_rescale_executed": False}
        for source in receipt["source_closure"]:
            identity(args.rtl_root / source["relative"], source["sha256"])
            identity(source_root / source["relative"], source["sha256"])
        bindings = receipt["owned_support"] + list(receipt["tools"].values()) + runtimes
        bindings += source_bindings
        bindings += [receipt[key] for key in ("provenance_manifest", "input", "verilator_runtime_header")]
        if "case_manifest" in receipt:
            bindings.append(receipt["case_manifest"])
        for value in bindings:
            identity(value["path"], value["sha256"])
        receipt.update(status="passed_component_rtl", caller_inputs_and_mreg_guards_preserved=True,
                       source_bookends_verified=True, full_program_executed=False, dma_executed=False,
                       trace=identity(out / "trace.json"), input=identity(out / "input.json"), **execution_summary)
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
    for name in ("rtl-root", "provenance", "mill", "java", "verilator", "firtool", "out"):
        parser.add_argument("--" + name, type=safe_path, required=True)
    for name in ("input", "oracle", "cases", "vector-from"):
        parser.add_argument("--" + name, type=safe_path)
    parser.add_argument("--mxu-receipt-sha256")
    parser.add_argument("--timeout", type=int, default=1800)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
