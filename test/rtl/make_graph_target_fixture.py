"""Prepare public AtlasCore fixtures from a checked native graph and target reference.

This evaluator runs after compilation. It neither selects instructions nor
passes runtime values or expected outputs to the compiler.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from itertools import pairwise
from pathlib import Path

import numpy as np
from atlas_mlir_native.graph_abi import prepare_graph_inputs
from atlas_mlir_native.graph_precision import (
    diagnostic_graph_input_spec,
    lower_diagnostic_graph,
)
from evaluate_graph_quality import _panel
from graph_semantic_checker import evaluate_selected_graph
from merlin.semantic_compiler.target_binding import verify_native_publication
from safetensors.numpy import load_file

ROOT = Path(__file__).resolve().parents[2]


def _sha(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _weights(base: dict[str, np.ndarray], profile: str) -> dict[str, np.ndarray]:
    weights = {name: np.ascontiguousarray(value.copy()) for name, value in base.items()}
    if profile == "captured":
        return weights
    if profile == "fc2_gain_1p5":
        weights["fc2.weight"] = np.ascontiguousarray(
            weights["fc2.weight"] * np.float32(1.5)
        )
        return weights
    raise ValueError(f"unknown public weight profile {profile}")


def _physical_bf16(words: tuple[int, ...]) -> bytes:
    """Pack logical row-major BF16 words into two physical register halves."""
    if len(words) != 1024 or any(
        type(word) is not int or not 0 <= word <= 0xFFFF for word in words
    ):
        raise ValueError("target reference must produce 1024 BF16 words")
    physical = bytearray(2048)
    for row in range(32):
        for column in range(32):
            offset = (column // 16) * 1024 + row * 32 + (column % 16) * 2
            physical[offset : offset + 2] = words[row * 32 + column].to_bytes(
                2, "little"
            )
    return bytes(physical)


def build(
    compiled: Path, suite_path: Path, output: Path, case_name: str,
    policy_path: Path = ROOT / "test/examples/attention_mlp_graph_rounded_pack.json",
    source_path: Path = ROOT / "test/examples/captured_attention_mlp32_two_outputs_linalg.mlir",
) -> dict[str, object]:
    if output.exists():
        raise ValueError("fixture output must be a fresh directory")
    suite = json.loads(suite_path.read_text())
    if suite.get("schema") != "atlas.public_graph_rtl_suite.v1":
        raise ValueError("unsupported public RTL suite")
    cases = [row for row in suite["cases"] if row["name"] == case_name]
    if len(cases) != 1:
        raise ValueError("RTL case name is missing or repeated")
    case = cases[0]
    source = source_path
    weights_path = ROOT / "test/examples/captured_attention_mlp32_weights.safetensors"
    arguments_path = ROOT / "test/examples/captured_attention_mlp32_arguments.json"
    for name, path in (
        ("source", source),
        ("policy", policy_path),
        ("weights", weights_path),
    ):
        if _sha(path.read_bytes()) != suite[f"{name}_sha256"]:
            raise ValueError(f"frozen {name} differs from the public RTL suite")

    policy = json.loads(policy_path.read_text())
    graph = lower_diagnostic_graph(source.read_text(), entry="forward", policy=policy)
    manifest_path = compiled / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    verify_native_publication(
        compiled,
        manifest,
        engine="merlin_native",
        request_digest=graph.request.digest(),
        target_identity=graph.request.target_identity,
    )
    precision = json.loads((compiled / "precision-plan.json").read_text())
    if precision != graph.record():
        raise ValueError("compiled precision plan differs from the frozen graph")
    execution = json.loads((compiled / "execution_plan.json").read_text())
    addresses = execution.get("output_addresses")
    if (
        not isinstance(addresses, list) or not addresses
        or any(type(address) is not int for address in addresses)
        or execution.get("output_physical_layouts") != ["bf16_pair_halves"] * len(addresses)
        or any(right != left + 2048 for left, right in pairwise(addresses))
    ):
        raise ValueError("compiled outputs need adjacent BF16 physical tiles")

    weights = _weights(load_file(weights_path), case["weights"])
    tokens = _panel(case["input"])
    arguments = json.loads(arguments_path.read_text())
    prepared = prepare_graph_inputs(
        graph,
        policy,
        precision,
        manifest,
        execution,
        arguments,
        weights=weights,
        invocation={"tokens": tokens},
    )
    constant_rows = execution.get("constant_inputs")
    if (
        len(prepared) != 6
        or not isinstance(constant_rows, list)
        or len(constant_rows) != 1
    ):
        raise ValueError("compiled input ABI is not the seven-slot graph fixture ABI")
    constant = constant_rows[0]
    relative = constant.get("path")
    if not isinstance(relative, str) or Path(relative).is_absolute():
        raise ValueError("generated constant needs a relative artifact path")
    constant_path = (compiled / relative).resolve(strict=True)
    if not constant_path.is_relative_to(compiled.resolve(strict=True)):
        raise ValueError("generated constant escapes the compiler artifact")
    constant_payload = constant_path.read_bytes()
    if (
        constant.get("address") != execution["input_addresses"][6]
        or len(constant_payload) != 2048
        or _sha(constant_payload) != constant.get("sha256")
    ):
        raise ValueError("generated BF16 constant differs from the compiler manifest")

    target = evaluate_selected_graph(
        graph,
        weights,
        arguments,
        tokens,
        diagnostic_graph_input_spec(policy),
    )
    if len(target) != len(addresses) or any(value.type.dtype != "bf16" for value in target):
        raise ValueError("target reference did not return the declared BF16 tiles")
    expected = b"".join(_physical_bf16(value.elements) for value in target)
    program_path = compiled / "program.bin"
    program = program_path.read_bytes()
    if _sha(program) != manifest.get("binary_sha256"):
        raise ValueError("compiled binary differs from its manifest")

    output.mkdir(parents=True)
    shutil.copyfile(program_path, output / "program.bin")
    for index, item in enumerate(prepared):
        (output / f"input-{index}.bin").write_bytes(item.payload)
    (output / "input-6.bin").write_bytes(constant_payload)
    (output / "expected.bin").write_bytes(expected)
    properties = [
        "schema=atlas.rtl_tile_fixture.v1",
        f"name={case_name}",
        "input_count=7",
        f"output_count={len(addresses)}",
        f"program_sha256={_sha(program)}",
        f"compiler_manifest_sha256={_sha(manifest_path.read_bytes())}",
        f"suite_sha256={_sha(suite_path.read_bytes())}",
        f"source_sha256={_sha(source.read_bytes())}",
        f"policy_sha256={graph.policy_sha256}",
        f"output.address={addresses[0]:#x}",
        "reference=evaluator_typed_target_graph_v1",
        *(
            f"input.{index}.address={item.address:#x}"
            for index, item in enumerate(prepared)
        ),
        f"input.6.address={constant['address']:#x}",
    ]
    (output / "manifest.properties").write_text("\n".join(properties) + "\n")
    record = {
        "schema": "atlas.public_graph_rtl_fixture.v1",
        "name": case_name,
        "program_sha256": _sha(program),
        "suite_sha256": _sha(suite_path.read_bytes()),
        "input_sha256": [_sha(item.payload) for item in prepared]
        + [_sha(constant_payload)],
        "expected_sha256": _sha(expected),
        "expected_bf16_cells": sum(len(value.elements) for value in target),
        "reference_scope": "typed_target_operations_evaluator_only",
    }
    (output / "fixture-record.json").write_text(json.dumps(record, indent=2) + "\n")
    return record


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--compiled", required=True, type=Path)
    parser.add_argument(
        "--suite",
        type=Path,
        default=ROOT / "test/examples/attention_mlp_graph_rtl_suite.json",
    )
    parser.add_argument("--case", required=True)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument(
        "--policy", type=Path,
        default=ROOT / "test/examples/attention_mlp_graph_rounded_pack.json",
    )
    parser.add_argument(
        "--source", type=Path,
        default=ROOT / "test/examples/captured_attention_mlp32_two_outputs_linalg.mlir",
    )
    args = parser.parse_args()
    print(
        json.dumps(build(
            args.compiled, args.suite, args.out, args.case, args.policy,
            args.source,
        ), sort_keys=True)
    )


if __name__ == "__main__":
    main()
