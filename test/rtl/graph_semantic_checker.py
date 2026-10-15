"""Test-only target semantics for the public captured attention/MLP tile."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
from atlas_mlir_native.binding import (
    MXU0_POLICY,
    ROUNDED_PACK_POLICY,
    SCALED_PACK_POLICY,
    _mxu0_maps,
)
from atlas_mlir_native.graph_precision import DiagnosticGraphPrecision
from atlas_mlir_native.reference import (
    evaluate_bf16_add,
    evaluate_bf16_exp,
    evaluate_bf16_mul,
    evaluate_bf16_recip,
    evaluate_bf16_row_max,
    evaluate_bf16_row_sum,
    evaluate_bf16_sub,
    evaluate_rounded_scaled_pack,
    evaluate_scaled_pack,
)
from merlin.semantic_compiler.model import TensorType
from merlin.semantic_compiler.precision_values import FloatBoundarySpec
from merlin.semantic_compiler.reference import TensorValue, evaluate_graph
from test_mxu_numerics import dot


def evaluate_mxu0_reset(node, operands: tuple[TensorValue, ...]) -> TensorValue:
    """Independent ordered BF16 accumulation on admitted E4M3 inputs."""
    if (
        node.type.numerical_policy != MXU0_POLICY
        or node.index_maps != _mxu0_maps()
        or len(operands) != 2
        or node.attrs
        or any(
            value.type.dtype != "fp8_e4m3" or value.type.shape != (32, 32)
            for value in operands
        )
    ):
        raise ValueError("selected MXU0 reference received a different operation")
    for value in operands:
        if any(
            code != 0 and (((code >> 3) & 15) == 0 or (code & 0x7F) == 0x7F)
            for code in value.elements
        ):
            raise ValueError(
                "selected MXU0 operand is outside the finite-normal domain"
            )
    activation, weight = (value.elements for value in operands)
    output = tuple(
        dot(
            weight[column * 32 : (column + 1) * 32],
            activation[row * 32 : (row + 1) * 32],
        )
        for row in range(32)
        for column in range(32)
    )
    return TensorValue(node.type, output)


def _raw_bf16(fn):
    def check(node, operands: tuple[TensorValue, ...]) -> TensorValue:
        normalized = []
        for value in operands:
            if value.type.dtype != "bf16" or value.type.shape != (32, 32):
                raise ValueError("VPU reference received a non-BF16 tile")
            if value.type.numerical_policy not in {"raw-bit-pattern", MXU0_POLICY}:
                raise ValueError("VPU reference received an unknown BF16 producer")
            normalized.append(
                TensorValue(
                    TensorType(value.type.shape, "bf16", "raw-bit-pattern"),
                    value.elements,
                )
            )
        return fn(node, tuple(normalized))

    return check


def evaluate_selected_graph(
    graph: DiagnosticGraphPrecision,
    weights: Mapping[str, np.ndarray],
    arguments: Mapping[str, dict[str, object]],
    tokens: np.ndarray,
    input_spec: FloatBoundarySpec,
) -> tuple[TensorValue, ...]:
    """Evaluate declared target operations without importing the production emitter."""
    callbacks = {("matmul", MXU0_POLICY): evaluate_mxu0_reset}
    for op, fn in (
        ("add_bf16", evaluate_bf16_add),
        ("mul_bf16", evaluate_bf16_mul),
        ("sub_bf16", evaluate_bf16_sub),
        ("row_max_bf16", evaluate_bf16_row_max),
        ("row_sum_bf16", evaluate_bf16_row_sum),
        ("exp_bf16", evaluate_bf16_exp),
        ("recip_bf16", evaluate_bf16_recip),
    ):
        callbacks[(op, "raw-bit-pattern")] = _raw_bf16(fn)
    packs = {
        node.type.numerical_policy
        for node in graph.request.nodes
        if node.op == "pack_fp8"
    }
    if len(packs) != 1:
        raise ValueError("selected graph needs one declared pack policy")
    pack_policy = packs.pop()
    if pack_policy == ROUNDED_PACK_POLICY:
        callbacks[("pack_fp8", pack_policy)] = _raw_bf16(evaluate_rounded_scaled_pack)
    elif pack_policy == SCALED_PACK_POLICY:
        callbacks[("pack_fp8", pack_policy)] = _raw_bf16(evaluate_scaled_pack)
    else:
        raise ValueError("selected graph has an unknown pack policy")

    nodes = {node.id: node for node in graph.request.nodes}
    constants = {}
    for binding in graph.request.constants:
        node = nodes[binding.node_id]
        codes = np.frombuffer(bytes.fromhex(binding.data_hex), dtype="<u2")
        constants[node.id] = TensorValue(node.type, tuple(int(code) for code in codes))
    inputs = {}
    source_inputs = [node for node in graph.request.nodes if node.op == "input"]
    if set(arguments) != {str(index) for index in range(len(source_inputs))}:
        raise ValueError("source argument manifest does not match the selected graph")
    for index, node in enumerate(source_inputs):
        row = arguments[str(index)]
        values = weights[row["weight"]] if row["kind"] == "param" else tokens
        codes = input_spec.encode(values)
        inputs[node.id] = TensorValue(
            node.type, tuple(int(code) for code in codes.flat)
        )
    return evaluate_graph(
        graph.request, inputs, constants=constants, primitives=callbacks
    )


def bf16_outputs_as_f32(outputs: tuple[TensorValue, ...]) -> tuple[np.ndarray, ...]:
    """Expose target BF16 words as exact f32 values in logical row-major order."""
    result = []
    for output in outputs:
        if output.type.dtype != "bf16" or output.type.shape != (32, 32):
            raise ValueError("selected graph result is not a 32x32 BF16 tile")
        bits = np.array(output.elements, dtype=np.uint32)
        result.append((bits << 16).view(np.float32).reshape(32, 32))
    return tuple(result)
