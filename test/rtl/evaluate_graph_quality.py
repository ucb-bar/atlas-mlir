"""Measure public captured-graph precision separately from compiler correctness."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
import torch
from atlas_mlir_native.graph_abi import prepare_graph_inputs
from atlas_mlir_native.graph_precision import (
    diagnostic_graph_input_spec,
    lower_diagnostic_graph,
)
from graph_semantic_checker import bf16_outputs_as_f32, evaluate_selected_graph
from make_graph_source_fixture import _source_outputs
from merlin.semantic_compiler.target_binding import verify_native_publication
from safetensors.numpy import load_file

ROOT = Path(__file__).resolve().parents[2]


def _sha(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _panel(row: dict[str, object]) -> np.ndarray:
    kind = row["kind"]
    if kind == "uniform":
        return np.full((32, 32), row["value"], dtype=np.float32)
    values = np.asarray(row["values"], dtype=np.float32)
    if kind == "alternating_columns":
        return np.tile(values[np.arange(32) % len(values)], (32, 1))
    if kind == "checkerboard":
        indices = np.indices((32, 32)).sum(axis=0) % len(values)
        return values[indices]
    if kind == "random_choice" and type(row.get("seed")) is int:
        indices = np.random.default_rng(row["seed"]).integers(
            len(values), size=(32, 32)
        )
        return values[indices]
    raise ValueError(f"unsupported input panel {kind}")


def _weights(base: dict[str, np.ndarray], roll: int) -> dict[str, np.ndarray]:
    if type(roll) is not int or not 0 <= roll < 32:
        raise ValueError("weight column roll is outside the declared tile")
    return {
        name: np.ascontiguousarray(np.roll(value, roll, axis=1))
        for name, value in base.items()
    }


def _quality(
    actual: np.ndarray, reference: np.ndarray, floor: float
) -> dict[str, object]:
    if actual.shape != (32, 32) or reference.shape != (32, 32):
        raise ValueError("quality comparison needs two 32x32 outputs")
    if not np.isfinite(actual).all() or not np.isfinite(reference).all():
        raise ValueError("quality comparison encountered a nonfinite output")
    delta = np.abs(actual.astype(np.float64) - reference.astype(np.float64))
    eligible = np.abs(reference) >= floor
    return {
        "cells": int(delta.size),
        "mismatched_f32_cells": int(np.count_nonzero(delta)),
        "mean_abs_error": float(delta.mean()),
        "max_abs_error": float(delta.max()),
        "rmse": float(np.sqrt(np.mean(delta * delta))),
        "mean_relative_error": float(
            np.mean(delta[eligible] / np.abs(reference[eligible]))
        )
        if np.any(eligible)
        else None,
        "relative_error_cells": int(np.count_nonzero(eligible)),
        "reference_min": float(reference.min()),
        "reference_max": float(reference.max()),
    }


def _source_model(loader_path: Path, weights: dict[str, np.ndarray]):
    spec = importlib.util.spec_from_file_location(
        "atlas_public_quality_model", loader_path
    )
    if spec is None or spec.loader is None:
        raise ValueError("public model loader could not be opened")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    model, _ = module.get_model_and_inputs()
    state = model.state_dict()
    if set(state) != set(weights):
        raise ValueError("weight profile does not match the original model")
    with torch.no_grad():
        for name, tensor in state.items():
            values = weights[name]
            if values.shape != tuple(tensor.shape) or values.dtype != np.float32:
                raise ValueError(
                    f"weight {name} differs from the original model signature"
                )
            tensor.copy_(torch.from_numpy(values))
    model.eval()
    return model


def _framework_outputs(model, tokens: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    hidden: list[np.ndarray] = []

    def before_fc2(_module, inputs) -> None:
        hidden.append(inputs[0].detach().numpy().copy())

    hook = model.fc2.register_forward_pre_hook(before_fc2)
    try:
        with torch.no_grad():
            final = model(torch.from_numpy(tokens)).detach().numpy().copy()
    finally:
        hook.remove()
    if len(hidden) != 1:
        raise ValueError("source model did not expose one hidden tile")
    return hidden[0], final


def evaluate(
    compiled: Path, merlin_root: Path, suite_path: Path, output: Path
) -> dict[str, object]:
    if output.exists() or output.resolve().is_relative_to(compiled.resolve()):
        raise ValueError("quality output must be a fresh directory outside the compiled artifact")
    suite = json.loads(suite_path.read_text())
    if suite.get("schema") != "atlas.public_graph_quality_suite.v1" or (
        suite.get("quality_gate") != "none_diagnostic_measurement"
    ):
        raise ValueError("unsupported public graph quality suite")
    source = ROOT / "test/examples/captured_attention_mlp32_two_outputs_linalg.mlir"
    weight_path = ROOT / "test/examples/captured_attention_mlp32_weights.safetensors"
    policy_path = ROOT / "test/examples/attention_mlp_graph_rounded_pack.json"
    loader_path = merlin_root / "examples/workloads/atlas_smolvla_mini/loader.py"
    for name, path in (
        ("source", source),
        ("weights", weight_path),
        ("policy", policy_path),
        ("model_loader", loader_path),
    ):
        if _sha(path.read_bytes()) != suite[f"{name}_sha256"]:
            raise ValueError(f"frozen {name} differs from the quality suite")
    policy = json.loads(policy_path.read_text())
    graph = lower_diagnostic_graph(source.read_text(), entry="forward", policy=policy)
    manifest = json.loads((compiled / "manifest.json").read_text())
    verify_native_publication(
        compiled,
        manifest,
        engine="merlin_native",
        request_digest=graph.request.digest(),
        target_identity=graph.request.target_identity,
    )
    precision = json.loads((compiled / "precision-plan.json").read_text())
    if precision != graph.record():
        raise ValueError(
            "compiled precision plan differs from the frozen source and policy"
        )
    execution = json.loads((compiled / "execution_plan.json").read_text())
    arguments = json.loads(
        (ROOT / "test/examples/captured_attention_mlp32_arguments.json").read_text()
    )
    spec = diagnostic_graph_input_spec(policy)
    base = load_file(weight_path)
    floor = suite["relative_error_reference_floor"]
    if type(floor) not in (float, int) or not 0 < floor < 1:
        raise ValueError("quality suite needs a positive relative-error floor")
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)

    output.mkdir(parents=True)
    report = {
        "schema": "atlas.public_graph_quality_report.v1",
        "status": "incomplete",
        "scope": suite["scope"],
        "engine": "merlin_native",
        "platform": "typed_target_reference_and_cpu_pytorch",
        "source_sha256": suite["source_sha256"],
        "policy_sha256": graph.policy_sha256,
        "suite_sha256": _sha(suite_path.read_bytes()),
        "compiler_manifest_sha256": _sha((compiled / "manifest.json").read_bytes()),
        "program_sha256": _sha((compiled / "program.bin").read_bytes()),
        "relative_error_reference_floor": floor,
        "required_cases": len(suite["weight_profiles"]) * len(suite["inputs"]),
        "cases": [],
        "quality_gate": "none_diagnostic_measurement",
    }
    for profile in suite["weight_profiles"]:
        weights = _weights(base, profile["column_roll"])
        model = _source_model(loader_path, weights)
        weight_digest = _sha(
            b"".join(
                name.encode() + b"\0" + weights[name].tobytes()
                for name in sorted(weights)
            )
        )
        for case in suite["inputs"]:
            name = f"{profile['name']}__{case['name']}"
            row: dict[str, object] = {"name": name, "weight_sha256": weight_digest}
            try:
                tokens = _panel(case)
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
                target = bf16_outputs_as_f32(
                    evaluate_selected_graph(
                        graph,
                        weights,
                        arguments,
                        tokens,
                        spec,
                    )
                )
                original = _framework_outputs(model, tokens)
                formula = _source_outputs(tokens, weights)
                rows = []
                for index in range(2):
                    rows.append(
                        {
                            "output": index,
                            "target_vs_original": _quality(
                                target[index], original[index], floor
                            ),
                            "formula_vs_original": _quality(
                                formula[index], original[index], floor
                            ),
                            "target_sha256": _sha(target[index].tobytes()),
                            "original_sha256": _sha(original[index].tobytes()),
                        }
                    )
                case_root = output / name
                case_root.mkdir()
                np.savez(
                    case_root / "evaluation.npz",
                    tokens=tokens,
                    target_hidden=target[0],
                    target_final=target[1],
                    original_hidden=original[0],
                    original_final=original[1],
                )
                row.update(
                    {
                        "status": "measured",
                        "input_sha256": _sha(tokens.tobytes()),
                        "staged_payload_sha256": [
                            _sha(item.payload) for item in prepared
                        ],
                        "outputs": rows,
                    }
                )
            except (OSError, TypeError, ValueError) as error:
                row.update(
                    {
                        "status": "incomplete",
                        "reason": f"{type(error).__name__}: {error}",
                    }
                )
            report["cases"].append(row)
            (output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    report["status"] = (
        "measured_all"
        if all(row["status"] == "measured" for row in report["cases"])
        else "incomplete"
    )
    (output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--compiled", type=Path, required=True)
    parser.add_argument("--merlin-root", type=Path, required=True)
    parser.add_argument(
        "--suite",
        type=Path,
        default=ROOT / "test/examples/attention_mlp_graph_quality_suite.json",
    )
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    report = evaluate(
        args.compiled.resolve(strict=True),
        args.merlin_root.resolve(strict=True),
        args.suite.resolve(strict=True),
        args.out,
    )
    print(
        json.dumps(
            {
                "status": report["status"],
                "cases": len(report["cases"]),
                "out": str(args.out),
            }
        )
    )
    return 0 if report["status"] == "measured_all" else 2


if __name__ == "__main__":
    raise SystemExit(main())
