"""Compiler-declared Atlas arithmetic routes, independent of emitted code.

Already-quantized contractions use MXU1 for every K extent, with ascending
K32 instructions and explicit BF16 seeded continuation (already_quantized.py,
record():133 and _body():160-166 in consolidated 8b5e4a5). The tile compiler selects the unit from its explicit
per-contraction policy (compile_atlas_linalg_tile.py,
check_policy():361-371 and LinalgImporter:570-591 in consolidated 8b5e4a5), never from the source K extent.
"""
import operator

K_ORDER = "ascending-k32"


def select_mxu(k_extent, *, route="already-quantized", policy_unit=None):
    """Select the compiler route's unit; K validates shape, not unit choice.

    The one-argument API follows the already-quantized route. Tile callers must
    supply their contraction's policy unit, as compiler integers or MXU names.
    """
    if isinstance(k_extent, bool):
        raise ValueError("K extent must be a nonnegative integer")
    try:
        k_extent = operator.index(k_extent)
    except TypeError as exc:
        raise ValueError("K extent must be a nonnegative integer") from exc
    if k_extent < 0:
        raise ValueError("K extent must be a nonnegative integer")
    if route not in ("already-quantized", "tile"):
        raise ValueError("unknown compiler route")
    unit = None
    if policy_unit is not None:
        if type(policy_unit) is int and policy_unit in (0, 1):
            unit = f"mxu{policy_unit}"
        elif isinstance(policy_unit, str) and policy_unit in ("mxu0", "mxu1"):
            unit = policy_unit
        else:
            raise ValueError("policy unit must be 0, 1, mxu0, or mxu1")
    if route == "tile":
        if unit is None:
            raise ValueError("tile route requires an explicit policy unit")
        return unit
    if unit not in (None, "mxu1"):
        raise ValueError("already-quantized route requires MXU1")
    return "mxu1"
