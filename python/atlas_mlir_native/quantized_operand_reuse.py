"""Selected-target limits for optional row-local quantized operand residency."""
import os

from merlin.semantic_compiler.target_binding import NativeCompilationError


def reuse_policy():
    """Validate the selected policy before generating or caching target code."""
    policy = os.environ.get('ATLAS_QUANTIZED_OPERAND_REUSE', 'ordered')
    if policy not in ('ordered', 'row-cache'):
        raise NativeCompilationError('unsupported_semantics',
                                     'ATLAS_QUANTIZED_OPERAND_REUSE must be ordered or row-cache')
    return policy


def activation_residency(plan):
    policy = reuse_policy()
    if policy == 'ordered':
        return None
    from merlin.semantic_compiler.operand_reuse import plan_operand_reuse

    # This schedule expands one static slot per K tile. Bound code expansion
    # to 64 tiles and cache storage to 64 KiB, leaving the rest of the selected
    # 240 KiB spill frame for exponent restoration and ordinary allocation.
    # These are candidate limits, not allocation or instruction-budget proofs.
    return plan_operand_reuse(tile_count=(plan.k + 31) // 32, tile_bytes=1024,
                              consumers=(plan.n + 31) // 32,
                              capacity_bytes=65536, max_tiles=64)
