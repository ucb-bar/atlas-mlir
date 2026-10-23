"""Export captured full Tiny arithmetic to shared immutable Atlas assets.

This assembles instructions and packs operands without constructing a device
Core. Expected words remain arithmetic references until genuine execution.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).absolute().parents[1]))
from atlas_path_guard import checked as physical_checked
from atlas_tooling import sha256
from compile_workload_inputs import prepare, pack_output
from run_model_tile import configure_model, model_binding


def checked(value):
    path = Path(value)
    if any(word in part.lower() for part in path.parts for word in ("hammer", "vlsi")):
        raise ValueError("restricted path component")
    return physical_checked(path)


class Assets:
    def __init__(self, root):
        self.root = root
        self.sums = {}

    def put(self, kind, data, suffix=".bin"):
        digest = hashlib.sha256(data).hexdigest()
        relative = f"{kind}/{digest}{suffix}"
        destination = self.root / relative
        if relative in self.sums:
            if destination.read_bytes() != data:
                raise ValueError("content-address collision")
        else:
            destination.parent.mkdir(exist_ok=True)
            destination.write_bytes(data)
            self.sums[relative] = digest
        return relative


def read_array(root, descriptor):
    name = descriptor["file"]
    if not isinstance(name, str) or Path(name).is_absolute() or ".." in Path(name).parts:
        raise ValueError("capture payload must be relative")
    path = checked(root / name)
    raw = path.read_bytes()
    dtype = np.dtype(descriptor["dtype"])
    shape = tuple(descriptor["shape"])
    if (dtype not in (np.dtype("uint16"), np.dtype("float32"), np.dtype("int64"))
            or any(type(n) is not int or n <= 0 for n in shape)
            or len(raw) != descriptor["bytes"] or len(raw) != int(np.prod(shape)) * dtype.itemsize
            or hashlib.sha256(raw).hexdigest() != descriptor["sha256"]):
        raise ValueError("capture array identity changed")
    return np.frombuffer(raw, dtype=dtype).reshape(shape), path


def float_words(words):
    if words.dtype != np.uint16:
        raise ValueError("BF16 operand words required")
    return (words.astype(np.uint32) << 16).view(np.float32)


def export(args):
    capture = checked(args.capture)
    if sha256(capture) != args.capture_sha256:
        raise ValueError("capture receipt changed")
    report = json.loads(capture.read_text())
    if (report.get("status") != "PASS_FULL_TINY_ARITHMETIC_CAPTURE_REPLAY"
            or report.get("source_bookends_verified") is not True
            or report.get("target_replay_exact") is not True
            or report.get("greedy_feedback_actual") is not True
            or report.get("all22_cache_prefixes_exact") is not True
            or report.get("native_restored_exact") is not True):
        raise ValueError("complete full Tiny arithmetic capture required")
    count = report["decode_count"]
    if type(count) is not int or not 1 <= count <= 32:
        raise ValueError("capture decode extent")
    out = checked(args.out)
    if out.exists():
        raise ValueError("fresh full bundle required")
    pins = {str(capture): args.capture_sha256}
    for record in report["source_records"]:
        path = checked(record["path"])
        digest = sha256(path)
        if digest != record["sha256"] or (str(path) in pins and pins[str(path)] != digest):
            raise ValueError("capture source changed")
        pins[str(path)] = digest
    model_root = checked(args.model_root)
    binding = model_binding(model_root)
    _, _, assemble, _ = configure_model(model_root)
    for name in ("export_capture.py",):
        path = checked(Path(__file__).with_name(name))
        pins[str(path)] = sha256(path)
    for name in ("compile_workload_inputs.py", "run_model_tile.py", "run_workload_contraction.py",
                 "atlas_program.py", "atlas_path_guard.py", "atlas_tooling.py", "atlas_tile_arithmetic.py"):
        path = checked(Path(__file__).absolute().parents[1] / name)
        pins[str(path)] = sha256(path)
    out.mkdir(parents=True)
    assets = Assets(out)
    kernels, host, references = [], {}, []

    def read(desc):
        value, path = read_array(capture.parent, desc)
        digest = sha256(path)
        if str(path) in pins and pins[str(path)] != digest:
            raise ValueError("conflicting capture payload pin")
        pins[str(path)] = digest
        return value

    for name, desc in report["host_parameters"].items():
        words = read(desc)
        host[name] = {**desc, "file": assets.put("host-data", words.tobytes())}
    for row in report["target_trace"]:
        ref = {"step": row["step"], "next_token": row["next_token"], "input_tokens": row["input_tokens"],
               "logits": {**row["logits_payload"],
                          "file": assets.put("references", read(row["logits_payload"]).tobytes())}, "cache": []}
        for desc in row["cache"]:
            ref["cache"].append({**desc, "file": assets.put("references", read(desc).tobytes())})
        references.append(ref)
    for call_index, call in enumerate(report["calls"]):
        step, index = divmod(call_index, 155)
        if (step > count or call["step"] != step or call["site_index"] != index
                or call["site"] != ([f"model.layers.{layer}.{site}" for layer in range(22)
                    for site in ("self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj",
                                 "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj")] + ["lm_head"])[index]):
            raise ValueError("capture invocation order changed")
        a = read(call["A_bf16"])
        b = read(report["parameters"][call["site"]]["B_bf16"])
        rows = 128 if step == 0 else 1
        if a.ndim != 2 or b.ndim != 2 or a.shape != (rows, b.shape[0]):
            raise ValueError("logical source geometry")
        physical_a = float_words(a)
        if rows == 1:
            physical_a = np.concatenate((physical_a, np.zeros((31, a.shape[1]), np.float32)))
        column = 0
        for panel in call["panels"]:
            lo, hi = panel["columns"]
            expected = read(panel["expected_bf16"])
            m, k, n = panel["physical_mkn"]
            if (lo != column or hi <= lo or n != hi-lo or k != a.shape[1] or m != physical_a.shape[0]
                    or expected.shape != (m, n) or panel["logical_mkn"] != [rows, k, n]):
                raise ValueError("physical panel coverage")
            prepared = prepare({"A": physical_a, "B": float_words(np.ascontiguousarray(b[:, lo:hi])),
                                "native_C": np.zeros((m, n), np.float32)}, "float32", assemble)
            conversion = prepared["conversion"]
            if (prepared["rescale_exponent"] != panel["rescale"]
                    or conversion["A"]["scale_exponent"] != panel["A_scale"]
                    or conversion["B"]["scale_exponent"] != panel["B_scale"]):
                raise ValueError("captured integer-codec scales differ from producer")
            memory = prepared["memory"]
            kernels.append({"index": len(kernels), "step": step, "site": call["site"],
                "site_index": index, "columns": [lo, hi], "logical_rows": rows, "physical_mkn": [m, k, n],
                "A_scale": panel["A_scale"], "B_scale": panel["B_scale"], "rescale": panel["rescale"],
                "source": assets.put("programs", prepared["source"].encode(), ".S"),
                "original": assets.put("programs", prepared["program"]),
                "activation": assets.put("activations", prepared["activation"]),
                "weights": assets.put("weights", prepared["weights"]),
                "expected": assets.put("expected", pack_output(expected)),
                "initial_sha256": hashlib.sha256(prepared["initial"]).hexdigest(),
                "output_offset": memory["output_offset"], "total_dram_bytes": memory["total_dram_bytes"],
                "dram_base": memory["dram_base"], "requested_dma": prepared["requested_dma"]})
            column = hi
        if column != b.shape[1]:
            raise ValueError("incomplete output columns")
    if len(report["calls"]) != 155 * (count + 1) or len(kernels) != 280 * (count + 1):
        raise ValueError("full model launch count")
    for index, name in enumerate(sorted(host)):
        host[name]["symbol"] = f"tiny_full_param_{index:03d}"
    buffers = []
    for row in references:
        step = row["step"]
        for item in row["cache"]:
            name = f"tiny_full_cache_{step:02d}_{item['layer']:02d}_{item['field']}"
            buffers.append({**item, "actual_symbol": name, "expected_symbol": name + "_expected"})
        name = f"tiny_full_logits_{step:02d}"
        buffers.append({**row["logits"], "actual_symbol": name, "expected_symbol": name + "_expected"})
    tokens = [row["next_token"] for row in references]
    consumed = [row["input_tokens"][0][0] for row in references[1:]]
    if len(tokens) != count + 1 or consumed != tokens[:-1]:
        raise ValueError("actual greedy feedback trace differs")
    for name, values in (("tiny_full_greedy_tokens", tokens), ("tiny_full_decode_inputs", consumed)):
        words = np.array(values, dtype="<u4")
        raw = words.tobytes()
        buffers.append({"actual_symbol": name, "expected_symbol": name + "_expected",
                        "file": assets.put("references", raw), "shape": list(words.shape), "dtype": "uint32",
                        "bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()})
    contract = {"schema": "atlas.tiny_full_contract.v1", "decode_count": count,
                "kernel_count": len(kernels), "projection_calls": 155 * (count + 1),
                "cycle_cap": 200000000, "kernels": kernels, "host_parameters": host,
                "reference_buffers": buffers, "greedy_tokens": tokens,
                "mailbox_symbol": "tiny_full_mailbox", "panel_symbol": "tiny_full_panels",
                "state_symbol": "tiny_full_state_global", "context_symbol": "tiny_full_projection_state"}
    contract_raw = json.dumps(contract, indent=2).encode() + b"\n"
    (out / "contract.json").write_bytes(contract_raw)
    assets.sums["contract.json"] = hashlib.sha256(contract_raw).hexdigest()
    provenance = {"schema": "atlas.full_bundle.v1", "workload": "tiny-full", "decode_count": count,
        "kernel_count": len(kernels), "kernels": kernels, "host_parameters": host, "references": references,
        "contract": "contract.json", "contract_sha256": assets.sums["contract.json"],
        "capture": {"path": str(capture), "sha256": args.capture_sha256}, "model": binding,
        "initial_layout": "activation+weights+a5*256+cd*output_bytes+5a*256",
        "atlas_instruction_execution": False, "source_sha256": pins}
    if model_binding(model_root) != binding or any(sha256(checked(path)) != digest for path, digest in pins.items()):
        raise ValueError("export source bookend changed")
    data = json.dumps(provenance, indent=2).encode() + b"\n"
    (out / "provenance.json").write_bytes(data)
    assets.sums["provenance.json"] = hashlib.sha256(data).hexdigest()
    (out / "SHA256SUMS").write_text("".join(f"{digest}  {path}\n" for path, digest in sorted(assets.sums.items())))
    return provenance


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("capture", "model-root", "out"):
        parser.add_argument("--" + name, type=checked, required=True)
    parser.add_argument("--capture-sha256", required=True)
    export(parser.parse_args())


if __name__ == "__main__":
    main()
