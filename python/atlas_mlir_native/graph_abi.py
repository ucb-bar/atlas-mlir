"""Bind declared f32 graph arguments to one diagnostic Atlas image.

Static weights and runtime values enter through separate arguments. This
module checks the selected graph and physical ABI before encoding either one;
it never chooses a precision policy or computes expected outputs.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np

from .graph_precision import DiagnosticGraphPrecision, diagnostic_graph_input_spec
from .layout import unpack_native_bf16_output


@dataclass(frozen=True)
class PreparedGraphInput:
    source_id: str
    kind: str
    name: str
    address: int
    payload: bytes
    source_sha256: str


def prepare_graph_inputs(
    graph: DiagnosticGraphPrecision,
    policy: dict[str, object],
    precision_plan: dict[str, object],
    compilation_manifest: dict[str, object],
    execution_plan: dict[str, object],
    argument_manifest: dict[str, object],
    *,
    weights: Mapping[str, np.ndarray],
    invocation: Mapping[str, np.ndarray],
) -> tuple[PreparedGraphInput, ...]:
    """Encode every source argument in the compiler's declared input order."""
    policy_sha = hashlib.sha256(json.dumps(
        policy, sort_keys=True, separators=(",", ":"),
    ).encode()).hexdigest()
    if precision_plan != graph.record() or policy_sha != graph.policy_sha256:
        raise ValueError("precision plan differs from the selected source graph or policy")
    execution_sha = hashlib.sha256(
        (json.dumps(execution_plan, sort_keys=True) + "\n").encode()
    ).hexdigest()
    if (
        compilation_manifest.get("engine") != "merlin_native"
        or compilation_manifest.get("request_digest") != graph.request.digest()
        or compilation_manifest.get("target_identity") != graph.request.target_identity
        or compilation_manifest.get("execution_plan_sha256") != execution_sha
        or execution_plan.get("schema") != "atlas.native_tile_execution_plan.v8"
        or execution_plan.get("shape") != [32, 32]
    ):
        raise ValueError("compiler manifest or Atlas input ABI differs from the selected graph")
    inputs = tuple(node for node in graph.request.nodes if node.op == "input")
    addresses = execution_plan.get("input_addresses")
    payload_sizes = execution_plan.get("input_payload_bytes")
    conversions = precision_plan.get("boundary_conversions")
    if (
        not inputs or not isinstance(addresses, list) or not isinstance(payload_sizes, list)
        or not isinstance(conversions, list)
        or len(addresses) != len(payload_sizes)
        or len(addresses) < len(inputs)
        or len(conversions) != len(inputs) + len(graph.request.outputs)
        or not all(type(value) is int and value > 0 for value in addresses)
        or len(set(addresses)) != len(addresses)
        or any(conversions[index].get("source") != node.id for index, node in enumerate(inputs))
        or set(argument_manifest) != {str(index) for index in range(len(inputs))}
    ):
        raise ValueError("source arguments do not match the selected Atlas input slots")
    spec = diagnostic_graph_input_spec(policy)
    expected_names: dict[str, set[str]] = {"param": set(), "input": set()}
    prepared = []
    for index, node in enumerate(inputs):
        row = argument_manifest[str(index)]
        if not isinstance(row, dict) or row.get("kind") not in expected_names:
            raise ValueError(f"argument {index} has no declared source kind")
        kind = row["kind"]
        name = row.get("weight" if kind == "param" else "name")
        if not isinstance(name, str) or not name or name in expected_names[kind]:
            raise ValueError(f"argument {index} has an invalid source name")
        if kind == "param" and (row.get("dtype") != "float32" or row.get("shape") != list(node.type.shape)):
            raise ValueError(f"declared weight {name} differs from the source argument type")
        expected_names[kind].add(name)
        values = (weights if kind == "param" else invocation).get(name)
        if not isinstance(values, np.ndarray) or values.dtype != np.float32 or values.shape != node.type.shape:
            raise ValueError(f"source argument {name} needs a {node.type.shape} float32 array")
        source_bytes = np.ascontiguousarray(values).tobytes(order="C")
        codes = np.ascontiguousarray(spec.encode(values))
        payload = codes.tobytes(order="C")
        if (node.type.dtype != "fp8_e4m3" or codes.dtype != np.uint8
                or len(payload) != payload_sizes[index]):
            raise ValueError(f"encoded argument {name} differs from the Atlas input slot")
        prepared.append(PreparedGraphInput(
            node.id, kind, name, addresses[index], payload,
            hashlib.sha256(source_bytes).hexdigest(),
        ))
    if set(weights) != expected_names["param"] or set(invocation) != expected_names["input"]:
        raise ValueError("source arguments have missing or unexpected weights or inputs")
    return tuple(prepared)


def decode_graph_outputs(
    graph: DiagnosticGraphPrecision,
    execution_plan: dict[str, object],
    physical_payloads: tuple[bytes, ...],
) -> tuple[np.ndarray, ...]:
    """Decode ordered physical BF16 tiles to the source graph's f32 view."""
    if len(physical_payloads) != len(graph.request.outputs):
        raise ValueError("physical output count differs from the source graph")
    nodes = {node.id: node for node in graph.request.nodes}
    conversions = graph.boundary_conversions[-len(graph.request.outputs):]
    outputs = []
    for index, (node_id, payload) in enumerate(zip(
        graph.request.outputs, physical_payloads, strict=True,
    )):
        if (
            nodes[node_id].type.dtype != "bf16"
            or conversions[index].get("source") != node_id
            or conversions[index].get("policy") != "bf16-to-f32-exact-v1"
        ):
            raise ValueError("source output does not declare the selected BF16 decode")
        logical = unpack_native_bf16_output(execution_plan, payload, index)
        codes = np.frombuffer(logical, dtype="<u2").astype("<u4").reshape(32, 32)
        outputs.append((codes << 16).view("<f4"))
    return tuple(outputs)
