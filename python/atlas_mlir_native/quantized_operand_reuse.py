"""Selected-target limits for optional row-local quantized operand residency."""
from dataclasses import dataclass
import os

from merlin.semantic_compiler.target_binding import NativeCompilationError


def reuse_policy():
    """Validate the selected policy before generating or caching target code."""
    policy = os.environ.get('ATLAS_QUANTIZED_OPERAND_REUSE', 'ordered')
    if policy not in ('ordered', 'row-cache'):
        raise NativeCompilationError('unsupported_semantics',
                                     'ATLAS_QUANTIZED_OPERAND_REUSE must be ordered or row-cache')
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
    if policy not in ('ordered', 'row-cache'):
        raise ValueError('unsupported recorded quantized operand reuse policy')
    if policy == 'ordered':
        return None

    # This schedule expands one static slot per K tile. Bound code expansion
    # to 64 tiles and cache storage to 64 KiB, leaving the rest of the selected
    # 240 KiB spill frame for exponent restoration and ordinary allocation.
    # These are candidate limits, not allocation or instruction-budget proofs.
    residency = ActivationResidency((plan.k + 31) // 32, 1024, (plan.n + 31) // 32)
    if residency.consumers < 2 or residency.tile_count > 64 or residency.resident_bytes > 65536:
        return None
    return residency


def schedule_record(plan, policy=None):
    requested = reuse_policy() if policy is None else policy
    residency = activation_residency(plan, requested)
    if requested == 'ordered':
        return {}
    return dict(operand_reuse_requested=requested,
                operand_reuse='row-cache' if residency is not None else 'ordered')
