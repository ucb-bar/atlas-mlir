"""Selected-target limits for optional row-local quantized operand residency."""
from dataclasses import dataclass


POLICIES = ('ordered', 'row-cache', 'prefix-cache', 'wide-cache', 'paired-cache',
            'panel-cache', 'batch-panel-cache', 'interleaved-panel-cache')
WIDE_CACHE_TILES = 192


def reuse_policy():
    from atlas_mlir_native.options import current_codegen_options
    return current_codegen_options().quantized_operand_reuse


@dataclass(frozen=True)
class ActivationResidency:
    tile_count: int
    tile_bytes: int
    consumers: int

    @property
    def resident_bytes(self):
        return self.tile_count * self.tile_bytes

    @property
    def avoided_read_bytes(self):
        return self.resident_bytes * (self.consumers - 1)


def activation_residency(plan, policy=None):
    policy = reuse_policy() if policy is None else policy
    if policy not in POLICIES:
        raise ValueError('unsupported recorded quantized operand reuse policy')
    if policy == 'ordered':
        return None
    if policy in ('panel-cache', 'batch-panel-cache', 'interleaved-panel-cache'):
        from .quantized_panel_geometry import panel_geometry
        if panel_geometry(plan, allow_batches=policy != 'panel-cache') is not None:
            return None
        policy = 'wide-cache'
    if policy == 'paired-cache':
        from .quantized_panel_reuse import admitted
        if admitted(plan):
            return None  # Paired panels are not a single resident activation row.
        policy = 'wide-cache'

    # Static slots bound both storage and code expansion. The wider opt-in
    # leaves 48 KiB of the selected 240 KiB frame for ordinary allocation.
    # These candidate limits never replace native allocation/image proofs.
    limit = WIDE_CACHE_TILES if policy == 'wide-cache' else 64
    tiles = (plan.k + 31) // 32
    if policy in ('prefix-cache', 'wide-cache'):
        tiles = min(tiles, limit)
    residency = ActivationResidency(tiles, 1024, (plan.n + 31) // 32)
    if residency.consumers < 2 or residency.tile_count > limit:
        return None
    return residency


def schedule_record(plan, policy=None):
    requested = reuse_policy() if policy is None else policy
    if requested == 'interleaved-panel-cache':
        from .quantized_panel_reuse import MAX_INLINE_READOUT_ROWS
        record = schedule_record(plan, 'batch-panel-cache')
        record['operand_reuse_requested'] = requested
        if (record.get('operand_reuse') == 'panel-cache'
                and record['operand_reuse_panel_m_tiles'] <= MAX_INLINE_READOUT_ROWS):
            record['operand_reuse_panel_order'] = 'k-major'
        return record
    if requested in ('panel-cache', 'batch-panel-cache'):
        from .quantized_panel_geometry import panel_geometry
        geometry = panel_geometry(plan, allow_batches=requested == 'batch-panel-cache')
        if geometry is not None:
            from math import prod
            from .quantized_panel_geometry import shared_weight_panel_plan
            physical = shared_weight_panel_plan(plan)
            extra = ({'operand_reuse_flattened_batches': prod(plan.batch_shape)}
                     if physical.batch_shape != plan.batch_shape else {})
            return dict(operand_reuse_requested=requested, operand_reuse='panel-cache',
                        operand_reuse_panel_m_tiles=geometry[0], operand_reuse_panel_k_tiles=geometry[1], **extra)
        return {**schedule_record(plan, 'wide-cache'), 'operand_reuse_requested': requested}
    if requested == 'paired-cache':
        from .quantized_panel_reuse import admitted, PANEL_K_TILES
        if admitted(plan):
            return dict(operand_reuse_requested=requested, operand_reuse='paired-cache',
                        operand_reuse_panel_m_tiles=2, operand_reuse_panel_k_tiles=PANEL_K_TILES)
        return {**schedule_record(plan, 'wide-cache'), 'operand_reuse_requested': requested}
    residency = activation_residency(plan, requested)
    if requested == 'ordered':
        return {}
    if requested == 'wide-cache' and residency is not None:
        return dict(operand_reuse_requested=requested,
                    operand_reuse='row-cache' if residency.tile_count == (plan.k + 31) // 32 else 'prefix-cache',
                    operand_reuse_cached_tiles=residency.tile_count)
    if residency is not None and residency.tile_count < (plan.k + 31) // 32:
        return dict(operand_reuse_requested=requested, operand_reuse='prefix-cache',
                    operand_reuse_cached_tiles=residency.tile_count)
    return dict(operand_reuse_requested=requested,
                operand_reuse='row-cache' if residency is not None else 'ordered')
