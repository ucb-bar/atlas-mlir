"""Physical input and output layout for a source-bound Atlas matmul tile.

Merlin owns the numerical conversion. This module binds its logical code
elements to the selected Atlas DRAM tile layout. It does not compile a graph,
launch a device, or turn a diagnostic precision choice into an admitted one.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from merlin.semantic_compiler.precision_transform import (
    DiagnosticMatmulTile,
    assemble_diagnostic_matmul_outputs,
)

from .binding import TARGET_IDENTITY, TILE_BYTES

TILE_EDGE = 32
BF16_HALF_COLUMNS = 16


def _require_tile(plan: DiagnosticMatmulTile) -> None:
    if (
        not isinstance(plan, DiagnosticMatmulTile)
        or plan.request.target_identity != TARGET_IDENTITY
        or len(plan.inputs) < 2 or len(plan.inputs) % 2
        or tuple(item.source_role for item in plan.inputs) != (
            tuple(f"{role}__k{index}" for index in range(len(plan.inputs) // 2)
                  for role in ("lhs", "rhs"))
            if plan.ordered_k_continuation else ("lhs", "rhs")
        )
        or any(item.target_type.shape != (TILE_EDGE, TILE_EDGE) or item.target_type.dtype != "fp8_e4m3"
               for item in plan.inputs)
        or plan.output.source_role != "result"
        or plan.output.target_type.shape != (TILE_EDGE, TILE_EDGE)
        or plan.output.target_type.dtype != "bf16"
    ):
        raise ValueError("precision plan is not a selected Atlas FP8/BF16 tile")


def pack_source_operand(plan: DiagnosticMatmulTile, role: str, values: np.ndarray) -> bytes:
    """Encode one f32 source operand into the selected row-major FP8 DRAM tile."""
    _require_tile(plan)
    conversions = {item.source_role: item for item in plan.inputs}
    if role not in conversions:
        raise ValueError("source operand role differs from the tile ABI")
    conversion = conversions[role]
    codes = conversion.convert_values(values)
    if codes.dtype != np.uint8 or codes.shape != (TILE_EDGE, TILE_EDGE):
        raise ValueError("converted operand differs from the selected FP8 tile")
    packed = np.ascontiguousarray(codes).tobytes(order="C")
    if len(packed) * 2 != TILE_BYTES:
        raise ValueError("packed FP8 tile has the wrong physical extent")
    return packed


def decode_source_result(plan: DiagnosticMatmulTile, payload: bytes) -> np.ndarray:
    """Read the two BF16 register halves from the selected physical output tile."""
    _require_tile(plan)
    if not isinstance(payload, bytes) or len(payload) != TILE_BYTES:
        raise ValueError("BF16 output payload has the wrong physical extent")
    physical = np.frombuffer(payload, dtype="<u2")
    half_elements = TILE_EDGE * BF16_HALF_COLUMNS
    codes = np.empty((TILE_EDGE, TILE_EDGE), dtype=np.uint16)
    codes[:, :BF16_HALF_COLUMNS] = physical[:half_elements].reshape(TILE_EDGE, BF16_HALF_COLUMNS)
    codes[:, BF16_HALF_COLUMNS:] = physical[half_elements:].reshape(TILE_EDGE, BF16_HALF_COLUMNS)
    return plan.output.convert_values(codes)


def decode_source_results(
    plans: Sequence[DiagnosticMatmulTile], payloads: Sequence[bytes],
) -> np.ndarray:
    """Decode and assemble every physical M/N tile of one source result."""
    if not plans or len(plans) != len(payloads):
        raise ValueError("source result needs matching nonempty tile plans and payloads")
    return assemble_diagnostic_matmul_outputs(
        plans, tuple(decode_source_result(plan, payload) for plan, payload in zip(
            plans, payloads, strict=True,
        )),
    )
