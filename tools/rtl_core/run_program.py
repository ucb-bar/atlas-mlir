#!/usr/bin/env python3
"""Run authenticated raw instruction images on the freshly compiled AtlasCore."""

import sys
from pathlib import Path

TOOLS_ROOT = Path(__file__).absolute().parents[1]
if (TOOLS_ROOT / "atlas_path_guard.py").is_file():
    sys.path.insert(0, str(TOOLS_ROOT))

import argparse
from atlas_path_guard import checked
import importlib.util
import json
import os

sys.dont_write_bytecode = True


def module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


def run(args):
    path = checked(args.support_helper)
    support = module(path, "raw_rtl_support")
    out = support.safe_path(args.out)
    out.mkdir(parents=True, exist_ok=False)
    records = []
    receipt = {"schema_version": 1, "status": "preparing", "command": sys.argv, "records": records,
        "claim": "Fresh current AtlasCore raw-image functional check; explicit synthetic TL response policy",
        "native_quality_claim": False, "platform_performance_claim": False, "trace_enabled": args.trace,
        "public_idle_only": True, "vpu_busy_observed": False, "quiet_cycles": 32, "runs": []}

    def bind(path, sha=None):
        record = support.identity(path, sha)
        records.append(record)
        return record

    try:
        bind(path)
        receipt["shared_support"] = [bind(sys.modules[name].__file__) for name in ("atlas_path_guard", "atlas_tooling")]
        runner = support.safe_path(args.runner_root)
        bind(runner / "receipt.json", args.runner_sha256)
        build = json.loads((runner / "receipt.json").read_text())
        if build["status"] != "passed_build_only" or not build["source_artifacts_preserved"]:
            raise ValueError("Raw runner build did not pass")
        for record in build["records"]:
            bind(record["path"], record["sha256"])
        executable = build["executable"]
        bind(executable["path"], executable["sha256"])
        own = support.safe_path(Path(__file__).absolute().parent)
        for name in ("run_program.py", "prepare_program.py", "probe.py"):
            record = bind(own / name)
            support.copy_source(record["path"], out / name, record["sha256"])
            bind(out / name, record["sha256"])
        probe = module(out / "probe.py", "raw_replay_probe")
        preparation = module(out / "prepare_program.py", "frozen_raw_preparation")
        data, inputs = preparation.prepare(support, args.input, args.input_sha256, args.oracle, out, args.controls,
            reuse_model_arithmetic=args.reuse_model_arithmetic, model_root=args.model_root)
        records.extend(inputs)
        receipt["arithmetic_evidence"] = data["arithmetic_evidence"]
        support.write_json(out / "input.json", data)
        bind(out / "input.json")
        if args.prepare_only:
            receipt["status"] = "prepared"
        else:
            env = os.environ.copy()
            env.update(build["environment"])
            for case in data["cases"]:
                for repeat in range(2 if case["id"] == "positive" else 1):
                    stem = case["id"] + "-" + str(repeat)
                    actual = out / (stem + "-actual.bin")
                    trace = out / (stem + "-trace.jsonl") if args.trace else None
                    summary = out / (stem + "-summary.json")
                    memory = data["memory"]
                    command = [executable["path"], case["program"], data["initial"], str(actual), str(trace) if trace
                        else "-", str(42 + repeat), str(args.max_cycles), str(memory["dram_base"]),
                        str(memory["input_bytes"]), str(memory["output_offset"]), str(memory["output_bytes"]), str(summary)]
                    if args.cpu is not None:
                        command = ["taskset", "-c", str(args.cpu)] + command
                    execution = probe.bounded(command, out, env, out / (stem + ".log"), support)
                    item = {"id": case["id"], "repeat": repeat, "execution": execution}
                    receipt["runs"].append(item)
                    support.write_json(out / "receipt.json", receipt)
                    for artifact in (actual, trace, summary, out / (stem + ".log"), out / (stem + ".resources.json")):
                        if artifact is not None and support.safe_path(artifact).exists():
                            bind(artifact)
                    if case["id"] == "wrong_halt":
                        if (execution["returncode"] != 1
                            or "CSR status is not halted with ECALL reason2" not in (out / (stem + ".log")).read_text()):
                            raise AssertionError("Expected explicit CSR halt refusal did not occur")
                        if trace is not None:
                            item["physical_replay"] = preparation.replay(data, case, trace, None, case["program"])
                        item["status"] = "expected_CSR_halt_refusal"
                        support.write_json(out / "receipt.json", receipt)
                        continue
                    if execution["returncode"]:
                        raise RuntimeError("Raw RTL process failed: " + stem)
                    if actual.read_bytes() != bytes.fromhex(case["actual_oracle"]):
                        raise AssertionError("Physical output differs from independent oracle")
                    verdict = (preparation.replay(data, case, trace, actual, case["program"]) if trace is not None
                        else {"requests": sum(data["requested_dma"].values()) // 32,
                        "output_evidence": "physical output file; DMA summary only"})
                    result = json.loads(summary.read_text())
                    if (any((result[key] != value for key, value in data["requested_dma"].items()))
                        or result["responses"] != verdict["requests"]):
                        raise AssertionError("RTL summary traffic differs from independently replayed events")
                    if result.get("vpu_busy_observed") is not False or not result["guards_preserved"]:
                        raise AssertionError("Incorrect completion claim or guard result")
                    if result.get("csr_status") != 5 or result.get("halt_reason") != "ecall":
                        raise AssertionError("Summary did not confirm physical ECALL status")
                    item["physical_replay"] = verdict
                    item["summary"] = result
                    support.write_json(out / "receipt.json", receipt)
            positives = [support.safe_path(out / ("positive-" + str(i) + "-actual.bin")).read_bytes() for i in range(2)]
            if positives[0] != positives[1]:
                raise AssertionError("Initialized repeats differ")
            if args.controls and support.safe_path(out / "wrong_oracle-0-actual.bin").read_bytes() != positives[0]:
                raise AssertionError("Wrong-oracle control changed physical execution")
            receipt["status"] = "passed_raw_rtl_functional"
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
            receipt["status"] = "failed"
            receipt["source_artifacts_preserved"] = False
            receipt["bookend_error"] = f"{type(error).__name__}: {error}"
        support.write_json(out / "receipt.json", receipt)
        if not receipt.get("source_artifacts_preserved"):
            raise RuntimeError("Raw run bookends failed")


if __name__ == "__main__":
    import signal

    def interrupted(number, frame):
        raise SystemExit(f"Owned runner received signal {number}")

    signal.signal(signal.SIGTERM, interrupted)
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("runner-root", "input", "oracle", "support-helper", "out"):
        parser.add_argument("--" + name, type=Path, required=True)
    for name in ("runner-sha256", "input-sha256"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--trace", action="store_true",
        help="Enable diagnostic per-cycle/DMA trace and independent event replay (off by default)")
    parser.add_argument("--controls", action="store_true")
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--max-cycles", type=int, default=100000)
    parser.add_argument("--cpu", type=int, help="Optional CPU affinity for the simulator subprocess")
    parser.add_argument("--reuse-model-arithmetic", action="store_true",
        help="Reuse an authenticated complete-model actual payload; incompatible with controls")
    parser.add_argument("--model-root", type=Path, help="Explicit source root required for arithmetic reuse bookends")
    args = parser.parse_args()
    if args.reuse_model_arithmetic and (args.controls or args.model_root is None):
        parser.error("Arithmetic reuse requires --model-root and cannot enable controls")
    run(args)
