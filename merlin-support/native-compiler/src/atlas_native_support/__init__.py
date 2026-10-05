"""Atlas-specific selected bindings; no ACT or hand-reference dependency."""

from .movement import (
    MovementContract,
    compile_movement,
    movement_profile,
    verify_movement_artifact,
)
from .mxu0 import Mxu0Contract, mxu0_profile
from .mxu0_emit import compile_mxu0, verify_mxu0_artifact
from .mxu0_launch import prepare_mxu0_launch
from .mxu0_spatial_tiling import SpatialTilePlan, lower_spatial_contraction
from .mxu0_tiling import OrderedKTilePlan, lower_ordered_k_contraction

__all__ = [
    "MovementContract",
    "Mxu0Contract",
    "OrderedKTilePlan",
    "SpatialTilePlan",
    "compile_movement",
    "compile_mxu0",
    "lower_ordered_k_contraction",
    "lower_spatial_contraction",
    "movement_profile",
    "mxu0_profile",
    "prepare_mxu0_launch",
    "verify_movement_artifact",
    "verify_mxu0_artifact",
]
