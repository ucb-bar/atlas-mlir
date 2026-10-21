#!/usr/bin/env python3
"""Export authenticated complete-contraction evidence, never a board executable."""
from atlas_host_layout import ATLAS_DRAM_BASE
import argparse
from collections import Counter, deque
import json
from pathlib import Path
import re
import zipfile

import numpy as np

from atlas_tile_arithmetic import f32_to_bf16_bits, pack_bf16, tensor_scale_exponent
from atlas_program import assembly, code_sources
from run_model_tile import RECIPES, metrics, validate_arrays
from atlas_path_guard import checked as safe_path
from atlas_tooling import preflight_archive, sha256, write_json
from validate_model_contraction import admit_result_arrays, load_arrays
from rtl_core import prepare_program as replay


def require(condition, message):
    if not condition:
        raise ValueError(message)


class Admission:
    def __init__(self):
        self.records = {}

    def bind(self, value, digest=None):
        path = preflight_archive(value)
        actual = sha256(path)
        require(digest is None or actual == digest, f"Hash differs: {path}")
        require(str(path) not in self.records or self.records[str(path)] == actual, "Conflicting file identities")
        self.records[str(path)] = actual
        return path

    def identity(self, path, digest=None):
        path = self.bind(path, digest)
        return {"path": str(path), "sha256": self.records[str(path)]}

    safe_path = staticmethod(safe_path)

    def recheck(self):
        for path, digest in self.records.copy().items():
            self.bind(path, digest)

    def closure(self, records):
        require(isinstance(records, list) and records, "Missing source records")
        for record in records:
            require(set(record) == {"path", "sha256"}, "Unexpected source record")
            self.bind(record["path"], record["sha256"])


def arrays(path, names):
    path = preflight_archive(path)
    with zipfile.ZipFile(path) as archive:
        members = archive.infolist()
        require({m.filename for m in members} == {n + ".npy" for n in names} and len(members) == len(names)
            and sum(m.file_size for m in members) <= 32 * 1024**2, "Unexpected bounded array archive")
    with np.load(path, allow_pickle=False) as bundle:
        return {name: bundle[name].copy() for name in names}


def physical(bits):
    m, n = bits.shape
    return b"".join(bits[r : r + 32, c : c + 32].reshape(32, 2, 16).transpose(1, 0,
        2).astype("<u2").tobytes() for r in range(0, m, 32) for c in range(0, n, 32))


def model_trace(path, run, requested):
    counts, issued, last, cycle = Counter(), 0, None, -1
    with path.open() as trace:
        for line in trace:
            event = json.loads(line)
            require(type(event["cycle"]) is int and event["cycle"] > cycle, "Model trace cycles are not increasing")
            cycle, last = event["cycle"], event
            issued += int(event["s1_fire"])
            if event["dma_a"] is not None:
                counts["store_bytes" if event["dma_a"][2] else "load_bytes"] += 32
            if event["dma_d"] is not None:
                counts["response_beats"] += 1
    require(last is not None and last["halted"] is True and last["instruction"] == "ecall" and cycle == run["cycles"]
        and last["s1_pc"] == run["halt_pc_word"] and issued == run["issued_instructions"],
        "Model trace completion/counters differ")
    for key, value in {**requested, "response_beats": sum(requested.values()) // 32}.items():
        require(counts[key] == value == run["observed"][key], "Model DMA counters differ")


def admit_model(args, admission):
    receipt_path = admission.bind(args.model_receipt, args.model_receipt_sha256)
    root = receipt_path.parent
    receipt = json.loads(receipt_path.read_text())
    expected_epoch = tuple(code_sources()[name] for name in replay.PRODUCER_NAMES)
    require(replay.producer_epoch(receipt) == expected_epoch, "Require the current release producer tuple")
    require(safe_path(args.producer_root) == safe_path(Path(__file__).absolute().parents[1]),
        "Exporter must run from selected producer root")
    for name, digest in receipt["code_before"].items():
        admission.bind(root / "producer-sources" / name, digest)
        admission.bind(safe_path(args.producer_root) / "tools" / name, digest)
    source = admission.bind(args.input)
    manifest_path = admission.bind(args.manifest)
    manifest = json.loads(manifest_path.read_text())
    source_hashes = {"input": sha256(source), "manifest": sha256(manifest_path)}
    require(manifest.get("schema_version") == 1 and manifest.get("source") and manifest.get("site")
        and manifest.get("npz_sha256") == source_hashes["input"], "Native manifest differs")
    require(receipt.get("status") == "passed_model_only" and "error" not in receipt
        and receipt.get("source_inputs") == source_hashes and receipt.get("native_fidelity_claim") is False
        and receipt.get("raw_image_execution_claim") is False, "Model receipt is not diagnostic success")
    for flag, path in (("--input", source), ("--manifest", manifest_path)):
        command = receipt["command"]
        require(command.count(flag) == 1 and safe_path(command[command.index(flag) + 1]) == path,
            "Producer command references different frozen inputs")
    retained = load_arrays(source)
    a, b, native = validate_arrays(retained, manifest.get("source_dtype"), matrix=True)
    m, k, n = a.shape[0], a.shape[1], b.shape[1]
    require(receipt["shape"] == [m, k, n] and all(type(x) is int for x in receipt["shape"]), "Source/receipt shape differs")
    requested = replay.requested_traffic(receipt, m, k, n)
    names = {"source.npz", "encoded.npz", "source-manifest.json", "result.npz", "program.S", "program.bin",
        "initial-dram.bin", "expected-bf16-tiles.bin"}
    names |= {f"{stem}-{i}.{suffix}" for i in range(2) for stem,
        suffix in (("actual-bf16-halves", "bin"), ("cycles", "jsonl"), ("perfetto", "json"))}
    require(set(receipt["artifacts"]) == names, "Unexpected or incomplete model artifact set")
    for name in names:
        admission.bind(root / name, receipt["artifacts"][name])
    require(json.loads((root / "source-manifest.json").read_text()) == manifest, "Retained source manifest differs")
    copied = load_arrays(root / "source.npz")
    require(copied.keys() == retained.keys()
        and all(copied[x].dtype == retained[x].dtype and copied[x].shape == retained[x].shape
        and copied[x].tobytes() == retained[x].tobytes() for x in retained), "Retained operand bytes differ")
    encoded = arrays(root / "encoded.npz", ("A", "B"))
    exponent = 0
    for key in ("A", "B"):
        bits = np.array([f32_to_bf16_bits(float(x)) for x in retained[key].flat], dtype=np.uint16)
        scale = tensor_scale_exponent(bits) if receipt["recipe"] == RECIPES["tensor-pow2"] else 0
        expected = np.array([pack_bf16(int(x), 127 + scale) for x in bits], dtype=np.uint8)
        value = encoded[key]
        require(value.dtype == np.uint8 and value.shape == retained[key].shape and value.tobytes() == expected.tobytes()
            and receipt["conversion"][key]["scale_exponent"] == scale
            and receipt["conversion"][key]["scale_code"] == 127 + scale, "Encoded operand differs")
        exponent += scale
    require(receipt["output_rescale_exponent"] == exponent, "Output scale differs")
    inputs = b"".join(encoded["A"][r : r + 32, q : q + 32].tobytes() for r in range(0, m, 32) for q in range(0, k, 32))
    inputs += b"".join(encoded["B"][q : q + 32, c : c + 32].tobytes() for c in range(0, n, 32) for q in range(0, k, 32))
    result = arrays(root / "result.npz", ("target_C", "actual_bits", "expected_bits"))
    target = admit_result_arrays(result, (m, n))
    expected = physical(result["actual_bits"])
    require((root / "expected-bf16-tiles.bin").read_bytes() == expected, "Physical result layout differs")
    memory = {"dram_base": ATLAS_DRAM_BASE, "input_bytes": len(inputs), "output_offset": len(inputs) + 256,
        "output_bytes": len(expected), "guard_bytes_each": 256, "total_dram_bytes": len(inputs) + len(expected) + 512}
    require(receipt["memory"] == memory and receipt["requested_dma"] == requested, "Memory or requested traffic differs")
    require((root / "initial-dram.bin").read_bytes() == inputs + b"\xa5" * 256 + b"\xcd" * len(expected) + b"\x5a" * 256,
        "Initial operands/guards/sentinel differ")
    tiles = [{"row_range": [r, r + 32], "column_range": [c, c + 32], "k_range": [0, k]} for r in range(0, m,
        32) for c in range(0, n, 32)]
    require(receipt["tiles"] == tiles, "Incomplete, reordered or overlapping output tiles")
    command = receipt["command"]
    text = assembly(m, k, n, memory["output_offset"], exponent if receipt["recipe"] == RECIPES["tensor-pow2"] else None,
        resident_a="--resident-a" in command, resident_b="--resident-b" in command,
        pretranspose_resident_b="--pretranspose-resident-b" in command)
    require((root / "program.S").read_text() == text, "Program source differs from current producer")
    program = (root / "program.bin").read_bytes()
    require(0 < len(program) <= 16384 and len(program) % 4 == 0 and receipt["instruction_words"] == len(program) // 4
        and program[-4:] == b"\x73\0\0\0", "Malformed raw program or runner capacity exceeded")
    require(len(receipt["runs"]) == 2, "Require two model repeats")
    for i, run in enumerate(receipt["runs"]):
        require(run["repeat"] == i and run["initialization_seed"] == 42 + i and run["halt_reason"] == "ecall"
            and run["halt_pc_word"] == len(program) // 4 - 1
            and all(run.get(key) is True for key in ("all_output_bytes_exact", "input_and_guards_preserved",
            "drained_before_halt_observation")), "Model repeat failed")
        require((root / f"actual-bf16-halves-{i}.bin").read_bytes() == expected, "Model actual differs")
        model_trace(root / f"cycles-{i}.jsonl", run, requested)
    records, evidence = replay.reused_arithmetic(admission, root, receipt, expected, safe_path(args.model_root))
    admission.closure(records)
    return receipt, manifest, target, native, evidence


def admit_rtl(args, admission, model, arithmetic):
    path = admission.bind(args.rtl_receipt, args.rtl_receipt_sha256)
    root, receipt = path.parent, json.loads(path.read_text())
    require(receipt.get("status") == "passed_raw_rtl_functional" and "error" not in receipt
        and receipt.get("source_artifacts_preserved") is True and receipt.get("quiet_cycles") == 32
        and receipt.get("public_idle_only") is True and receipt.get("vpu_busy_observed") is False
        and receipt.get("native_quality_claim") is False and receipt.get("platform_performance_claim") is False,
        "Raw RTL receipt failed")
    admission.closure(receipt["records"])
    records = {str(safe_path(r["path"])): r["sha256"] for r in receipt["records"]}
    for name in ("run_program.py", "prepare_program.py"):
        require(sha256(Path(__file__).absolute().parent / "rtl_core" / name) in records.values(),
            "Raw replay helper is not current")

    def recorded(value):
        value = safe_path(value)
        require(str(value) in records, f"Missing RTL artifact identity: {value}")
        return admission.bind(value, records[str(value)])

    command = receipt["command"]
    require(command.count("--runner-root") == 1 and command.count("--runner-sha256") == 1, "Missing runner identity")
    build_path = recorded(safe_path(command[command.index("--runner-root") + 1]) / "receipt.json")
    require(sha256(build_path) == command[command.index("--runner-sha256") + 1], "Runner receipt differs")
    build = json.loads(build_path.read_text())
    require(build["status"] == "passed_build_only" and build["source_artifacts_preserved"] is True,
        "Raw runner build failed")
    admission.closure(build["records"])
    admission.closure([build["executable"]])
    data = json.loads(recorded(root / "input.json").read_text())
    require(data["memory"] == model["memory"] and data["requested_dma"] == model["requested_dma"],
        "RTL/model memory differs")
    require(data["arithmetic_evidence"] == receipt["arithmetic_evidence"] == {**arithmetic,
        "source_receipt": {"path": str(safe_path(args.model_receipt)), "sha256": args.model_receipt_sha256}},
        "RTL references different model arithmetic")
    for name in ("program.bin", "initial-dram.bin", "expected-bf16-tiles.bin", "encoded.npz"):
        require(sha256(recorded(root / name)) == model["artifacts"][name], "RTL copied payload differs")
    require(safe_path(data["initial"]) == root / "initial-dram.bin", "Foreign initial memory")
    positives = [case for case in data["cases"] if case["id"] == "positive"]
    require(len(positives) == 1, "Ambiguous positive RTL case")
    case = positives[0]
    expected = (root / "expected-bf16-tiles.bin").read_bytes()
    require(safe_path(case["program"]) == root / "program.bin" and bytes.fromhex(case["expected"]) == expected
        and bytes.fromhex(case["actual_oracle"]) == expected, "RTL oracle differs")
    runs = [run for run in receipt["runs"] if run["id"] == "positive"]
    require(len(runs) == 2 and [run["repeat"] for run in runs] == [0, 1], "Require two RTL repeats")
    for i, run in enumerate(runs):
        require(run["execution"]["returncode"] == 0, "RTL process failed")
        actual = recorded(root / f"positive-{i}-actual.bin")
        trace = recorded(root / f"positive-{i}-trace.jsonl")
        summary = json.loads(recorded(root / f"positive-{i}-summary.json").read_text())
        verdict = replay.replay(data, case, trace, actual, root / "program.bin")
        require(verdict == run["physical_replay"] and summary == run["summary"], "RTL replay differs")
        require(summary["csr_status"] == 5 and summary["halt_reason"] == "ecall" and summary["quiet_cycles"] == 32
            and summary["halted"] is True and summary["guards_preserved"] is True and summary["public_busy_idle"] is True
            and summary["vpu_busy_observed"] is False and summary["cycles_from_start_csr"] > 0
            and summary["responses"] == verdict["requests"]
            and all(summary[key] == value for key, value in data["requested_dma"].items()),
            "RTL summary completion/traffic differs")
        states = deque(maxlen=32)
        with trace.open() as events:
            for line in events:
                event = json.loads(line)
                if event["event"] == "state":
                    states.append(event)
        require(len(states) == 32
            and [s["cycle"] for s in states] == list(range(states[0]["cycle"], states[0]["cycle"] + 32))
            and all(not s["reset"] for s in states), "Noncontiguous RTL quiet margin")
    return receipt


def export(args):
    out = safe_path(args.out)
    require(not out.exists(), "Fresh output destination required")
    admission = Admission()
    for name in ("export_contraction_capsule.py", "validate_model_contraction.py", "rtl_core/prepare_program.py"):
        admission.bind(Path(__file__).absolute().parent / name)
    model, manifest, target, native, arithmetic = admit_model(args, admission)
    admit_rtl(args, admission, model, arithmetic)
    diagnostic = metrics(target, native)
    require(model["diagnostic_errors"]["target_vs_native_C"] == diagnostic, "Native diagnostic differs")
    native_gate = bool(np.allclose(target, native, rtol=1e-4, atol=1e-4))
    admission.recheck()
    status = {"schema": "atlas.contraction_capsule.v1", "status": "diagnostic_only", "fpga_ready": False,
        "full_model": False, "launchable_host": False, "native_fidelity_claim": False, "shape": model["shape"],
        "memory": model["memory"], "physical_gate": "two_model_and_two_raw_rtl_repeats_exact",
        "native_gate": {"passed": native_gate, "rtol": 1e-4, "atol": 1e-4, "metrics": diagnostic},
        "arithmetic_evidence": arithmetic, "records": admission.records,
        "gaps": ["64-bit DMA host adapter and address reachability", "complete graph compilation and native fidelity",
        "bitstream identity, SoC map, loader, cache visibility and completion qualification"],
        "limits": ["No execution performed by exporter; authenticated retained evidence only",
        "Raw RTL public idle excludes VPU busy; committed ECALL status 5 plus 32 quiet cycles",
        "Producer source text and fetched binary bound; no new assembler equivalence proof"]}
    out.mkdir(parents=True, exist_ok=False)
    sources = {name: safe_path(args.model_receipt).parent / name for name in ("program.S", "program.bin", "initial-dram.bin",
        "expected-bf16-tiles.bin", "source-manifest.json", "source.npz", "encoded.npz", "result.npz",
        "actual-bf16-halves-0.bin", "actual-bf16-halves-1.bin")}
    sources.update({"frozen-input.npz": safe_path(args.input), "native-manifest.json": safe_path(args.manifest),
        "model-receipt.json": safe_path(args.model_receipt), "rtl-receipt.json": safe_path(args.rtl_receipt)})
    for i in range(2):
        sources[f"rtl-actual-{i}.bin"] = safe_path(args.rtl_receipt).parent / f"positive-{i}-actual.bin"
    for name, source in sources.items():
        (out / name).write_bytes(admission.bind(source).read_bytes())
    admission.recheck()
    write_json(out / "status.json", status)
    names = sorted([*sources, "status.json"])
    (out / "SHA256SUMS").write_text("".join(f"{sha256(out/name)}  {name}\n" for name in names))
    return status


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("model-receipt", "rtl-receipt", "input", "manifest", "producer-root", "model-root", "out"):
        parser.add_argument("--" + name, required=True, type=Path)
    for name in ("model-receipt-sha256", "rtl-receipt-sha256"):
        parser.add_argument("--" + name, required=True)
    args = parser.parse_args()
    require(all(re.fullmatch("[0-9a-f]{64}", getattr(args, name)) for name in ("model_receipt_sha256",
        "rtl_receipt_sha256")), "Receipt SHA256 must be lowercase64 hex")
    print(json.dumps(export(args), sort_keys=True))


if __name__ == "__main__":
    main()
