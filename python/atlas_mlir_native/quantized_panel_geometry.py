"""Choose bounded MXU1 panels by operand and partial-result DRAM traffic.

A candidate ranking, not a proof: the program must still pass native allocation, memory proofs and image limits.
"""
from dataclasses import replace
from math import prod

from .exponent_restoration import exponent_geometry
from .quantized_panel_reuse import admitted, inline_readout_admitted

PANEL_TILE_BUDGET = 192  # 1 KiB FP8 tiles; retain 48 KiB for other allocations.


def shared_weight_panel_plan(plan):
    """Fold only tile-aligned batch rows that share one unchanged RHS tensor.

    A physical traversal choice; source graph, C packing and output shape are unchanged. Batched-RHS attention
    and per-batch scalar exponents keep their traversal because their pages are not one row-contiguous stream.
    """
    batches = prod(plan.batch_shape)
    rhs = plan.source.node(plan.operand_ids[1])
    if (batches <= 1 or plan.m % 32 or len(rhs.type.shape) != 2
            or exponent_geometry(plan)[0]['mode'] == 'batch'):
        return plan
    return replace(plan, m=batches * plan.m, batch_shape=())


def panel_geometry(plan, *, allow_batches=False):
    """Return profitable M/K panel counts, or retain the existing row cache."""
    plan = shared_weight_panel_plan(plan)
    mc, nc, kc = ((value + 31) // 32 for value in (plan.m, plan.n, plan.k))
    candidates = []
    for rows in (2, 4, 8, 16, 32):
        if not admitted(plan, m_tiles=rows, min_k_tiles=1, allow_batches=allow_batches):
            continue
        columns = min(64, PANEL_TILE_BUDGET // (rows + 1))
        # Bound the nested batch body to one resident K panel with immediate
        # final restoration. Longer reductions retain the existing row cache.
        if (prod(plan.batch_shape) > 1 and
                (kc > columns or not inline_readout_admitted(rows, plan.k))):
            continue
        phases = (kc + columns - 1) // columns
        # Only the batch policy omits the last partial round trip, which inline restoration consumes;
        # panel-cache keeps its original ranking so recorded geometries replay.
        round_trips = phases - bool(allow_batches and inline_readout_admitted(rows, plan.k))
        traffic = (mc // rows) * nc * kc * 1024 + 4096 * mc * nc * round_trips
        candidates.append((traffic, phases, rows, columns))
    if not candidates:
        return None
    traffic, _, rows, columns = min(candidates)
    if traffic >= mc * nc * kc * 1024:
        return None
    return rows, columns
