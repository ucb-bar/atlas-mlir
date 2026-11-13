"""Choose bounded MXU1 panels by operand and partial-result DRAM traffic.

This is a candidate ranking, not a timing or resource proof. The selected
program must still pass native allocation, memory proofs and image limits.
"""
from dataclasses import replace
from math import prod

from .exponent_restoration import exponent_geometry
from .quantized_panel_reuse import admitted

PANEL_TILE_BUDGET = 192  # 1 KiB FP8 tiles; retain 48 KiB for other allocations.


def shared_weight_panel_plan(plan):
    """Fold only tile-aligned batch rows that share one unchanged RHS tensor.

    This is a physical traversal choice. The source graph, C packing and output
    shape stay unchanged. Attention with batched RHS and per-batch scalar
    exponents retain their original traversal because their pages are not a
    single row-contiguous stream.
    """
    batches = prod(plan.batch_shape)
    rhs = plan.source.node(plan.operand_ids[1])
    if (batches <= 1 or plan.m % 32 or len(rhs.type.shape) != 2
            or exponent_geometry(plan)[0]['mode'] == 'batch'):
        return plan
    return replace(plan, m=batches * plan.m, batch_shape=())


def panel_geometry(plan):
    """Return profitable M/K panel counts, or retain the existing row cache."""
    plan = shared_weight_panel_plan(plan)
    mc, nc, kc = ((value + 31) // 32 for value in (plan.m, plan.n, plan.k))
    candidates = []
    for rows in (2, 4, 8, 16, 32):
        if not admitted(plan, m_tiles=rows, min_k_tiles=1):
            continue
        columns = min(64, PANEL_TILE_BUDGET // (rows + 1))
        phases = (kc + columns - 1) // columns
        # Activation reads and the final store/restoration are common to these
        # schedules. Each phase adds one BF16 output write and subsequent read.
        traffic = (mc // rows) * nc * kc * 1024 + 4096 * mc * nc * phases
        candidates.append((traffic, phases, rows, columns))
    if not candidates:
        return None
    traffic, _, rows, columns = min(candidates)
    if traffic >= mc * nc * kc * 1024:
        return None
    return rows, columns
