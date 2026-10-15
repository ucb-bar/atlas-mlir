#!/usr/bin/env python3
"""Make a source-weight, evaluator-owned fixture for the parsed Atlas graph."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

import numpy as np
from atlas_mlir_native.graph_abi import prepare_graph_inputs
from atlas_mlir_native.graph_precision import lower_diagnostic_graph
from safetensors.numpy import load_file

ROOT = Path(__file__).resolve().parents[2]


def _sha(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _tokens(case: str) -> np.ndarray:
    if case == "half":
        row = np.full(32, 0.5, dtype=np.float32)
    elif case == "one":
        row = np.ones(32, dtype=np.float32)
    elif case == "striped":
        row = np.array([0.5 if column % 2 == 0 else 1.0 for column in range(32)], dtype=np.float32)
    else:
        raise ValueError("unknown public invocation case")
    return np.tile(row, (32, 1))


def _source_outputs(tokens: np.ndarray, weights: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    """Independent f32 formula for this public captured example only."""
    query = tokens @ weights["query.weight"].T
    key = tokens @ weights["key.weight"].T
    value = tokens @ weights["value.weight"].T
    scores = (query @ key.T) / np.sqrt(np.float32(32.0))
    exponentials = np.exp(scores - scores.max(axis=-1, keepdims=True))
    probabilities = exponentials / exponentials.sum(axis=-1, keepdims=True)
    hidden = np.maximum((probabilities @ value) @ weights["fc1.weight"].T, np.float32(0))
    final = hidden @ weights["fc2.weight"].T
    return hidden, final


def _exact_physical_bf16(values: np.ndarray) -> bytes:
    if values.dtype != np.float32 or values.shape != (32, 32):
        raise ValueError("source output needs a 32x32 float32 tile")
    bits = np.ascontiguousarray(values).view(np.uint32)
    if np.any((bits & 0xffff) != 0):
        raise ValueError("public source output is not exactly representable in BF16")
    physical = bytearray(2048)
    for row in range(32):
        for column in range(32):
            address = (column // 16) * 1024 + row * 32 + (column % 16) * 2
            physical[address:address + 2] = int(bits[row, column] >> 16).to_bytes(2, "little")
    return bytes(physical)


def build(compiled: Path, output: Path, case: str) -> dict[str, object]:
    if output.exists():
        raise ValueError("fixture output must be a fresh directory")
    source = ROOT / "test/examples/captured_attention_mlp32_two_outputs_linalg.mlir"
    policy_path = ROOT / "test/examples/attention_mlp_graph_precision.json"
    arguments_path = ROOT / "test/examples/captured_attention_mlp32_arguments.json"
    weights_path = ROOT / "test/examples/captured_attention_mlp32_weights.safetensors"
    policy = json.loads(policy_path.read_text())
    graph = lower_diagnostic_graph(source.read_text(), entry="forward", policy=policy)
    weights = load_file(weights_path)
    tokens = _tokens(case)
    manifest_path = compiled / "manifest.json"
    program_path = compiled / "program.bin"
    execution_path = compiled / "execution_plan.json"
    manifest = json.loads(manifest_path.read_text())
    execution = json.loads(execution_path.read_text())
    prepared = prepare_graph_inputs(
        graph, policy, json.loads((compiled / "precision-plan.json").read_text()),
        manifest, execution, json.loads(arguments_path.read_text()),
        weights=weights, invocation={"tokens": tokens},
    )
    constant_rows = execution.get("constant_inputs")
    if (len(prepared) != 6 or not isinstance(constant_rows, list) or len(constant_rows) != 1
            or constant_rows[0].get("address") != execution["input_addresses"][6]
            or len(execution.get("output_addresses", [])) != 2
            or execution["output_addresses"][1] != execution["output_addresses"][0] + 2048
            or _sha(program_path.read_bytes()) != manifest.get("binary_sha256")):
        raise ValueError("selected program or generated constant differs from the graph fixture ABI")
    constant = constant_rows[0]
    relative = constant.get("path")
    if not isinstance(relative, str) or Path(relative).is_absolute():
        raise ValueError("generated constant needs a relative artifact path")
    constant_path = (compiled / relative).resolve(strict=True)
    if not constant_path.is_relative_to(compiled.resolve(strict=True)):
        raise ValueError("generated constant escapes the compiler artifact")
    constant_payload = constant_path.read_bytes()
    if len(constant_payload) != 2048 or _sha(constant_payload) != constant.get("sha256"):
        raise ValueError("generated BF16 scale tile differs from the compiler manifest")
    source_outputs = _source_outputs(tokens, weights)
    expected = b"".join(_exact_physical_bf16(values) for values in source_outputs)

    output.mkdir(parents=True)
    shutil.copyfile(program_path, output / "program.bin")
    for index, item in enumerate(prepared):
        (output / f"input-{index}.bin").write_bytes(item.payload)
    (output / "input-6.bin").write_bytes(constant_payload)
    (output / "expected.bin").write_bytes(expected)
    np.savez(output / "source-f32-outputs.npz", *source_outputs)
    properties = [
        "schema=atlas.rtl_tile_fixture.v1",
        f"name=parsed_graph_source_{case}",
        "input_count=7", "output_count=2",
        f"program_sha256={_sha(program_path.read_bytes())}",
        f"compiler_manifest_sha256={_sha(manifest_path.read_bytes())}",
        f"source_sha256={_sha(source.read_bytes())}",
        f"policy_sha256={graph.policy_sha256}",
        f"output.address={execution['output_addresses'][0]:#x}",
        f"reference=captured_f32_source_exact_bf16_{case}_v1",
        *(f"input.{index}.address={item.address:#x}" for index, item in enumerate(prepared)),
        f"input.6.address={constant['address']:#x}",
    ]
    (output / "manifest.properties").write_text("\n".join(properties) + "\n")
    return {
        "case": case, "source_sha256": _sha(source.read_bytes()),
        "argument_manifest_sha256": _sha(arguments_path.read_bytes()),
        "weights_sha256": _sha(weights_path.read_bytes()),
        "policy_sha256": graph.policy_sha256,
        "program_sha256": _sha(program_path.read_bytes()),
        "input_sha256": [_sha(item.payload) for item in prepared] + [_sha(constant_payload)],
        "expected_sha256": _sha(expected),
        "source_output_count": len(source_outputs),
        "source_output_cells": sum(value.size for value in source_outputs),
        "source_outputs_exact_bf16": True,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--compiled", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--case", choices=("half", "one", "striped"), required=True)
    args = parser.parse_args()
    print(json.dumps(build(args.compiled, args.out, args.case), sort_keys=True))


if __name__ == "__main__":
    main()
