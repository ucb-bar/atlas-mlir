"""Selected-target limits for optional row-local quantized operand residency."""
from dataclasses import dataclass
import os

from merlin.semantic_compiler.target_binding import NativeCompilationError

POLICIES = ('ordered', 'row-cache', 'prefix-cache', 'wide-cache', 'paired-cache')
WIDE_CACHE_TILES = 192


def reuse_policy():
    """Validate the selected policy before generating or caching target code."""
    policy = os.environ.get('ATLAS_QUANTIZED_OPERAND_REUSE', 'ordered')
    if policy not in POLICIES:
        raise NativeCompilationError('unsupported_semantics',
                                     'ATLAS_QUANTIZED_OPERAND_REUSE must be one of ' + ', '.join(POLICIES))
    return policy


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
