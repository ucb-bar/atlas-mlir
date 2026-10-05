"""Explicit installed binding for the selected diagnostic MXU0 tile domain."""

from __future__ import annotations

from pathlib import Path

from merlin.semantic_compiler.model import KernelRequest
from merlin.semantic_compiler.search import SearchLimits
from merlin.semantic_compiler.snapshot import NativeSnapshot, NativeTargetProfile

from .mxu0 import Mxu0Contract, mxu0_profile
from .mxu0_emit import compile_mxu0
from .mxu0_program_set import compile_mxu0_program_set
from .mxu0_spatial_tiling import (
    MAX_COMBINED_SPATIAL_NODES,
    SpatialTilePlan,
    lower_tiled_contraction,
)


class Binding:
    @staticmethod
    def profile() -> NativeTargetProfile:
        return mxu0_profile(Mxu0Contract.load())

    @staticmethod
    def compile(
        snapshot: NativeSnapshot,
        request: KernelRequest,
        *,
        fixed_inputs: dict[str, int],
        fixed_outputs: tuple[int, ...],
        target_source: Path,
        destination: Path,
        limits: SearchLimits,
    ) -> dict[str, object]:
        contract = Mxu0Contract.load()
        geometry = contract.record["geometry"]
        if (
            len(request.nodes) in (3, 4)
            and request.nodes[2].op == "contraction"
            and len(request.nodes[0].type.shape) == 2
            and len(request.nodes[2].type.shape) == 2
            and (
                len(request.nodes) == 3
                or dict(request.input_storages) == {
                    request.nodes[0].id: "external_fp8_row_major",
                    request.nodes[1].id: "external_fp8_row_major",
                }
            )
            and (
                len(request.nodes) == 4
                or request.nodes[0].type.shape[1] != geometry["array_rows"]
                or request.nodes[2].type.shape[0] != geometry["array_rows"]
                or request.nodes[2].type.shape[1] != geometry["array_cols"]
            )
        ):
            tiling = lower_tiled_contraction(request, contract)
            if (
                isinstance(tiling, SpatialTilePlan)
                and len(tiling.lowered.nodes) > MAX_COMBINED_SPATIAL_NODES
            ):
                return compile_mxu0_program_set(
                    snapshot, tiling, fixed_inputs=fixed_inputs,
                    fixed_outputs=fixed_outputs, rtl_root=target_source,
                    destination=destination, limits=limits, contract=contract,
                )
            return compile_mxu0(
                snapshot,
                tiling.lowered,
                fixed_inputs=tiling.fixed_panel_inputs(fixed_inputs),
                fixed_outputs=(
                    tiling.fixed_tile_outputs(fixed_outputs)
                    if isinstance(tiling, SpatialTilePlan) else fixed_outputs
                ),
                rtl_root=target_source,
                destination=destination,
                limits=limits,
                contract=contract,
                tiling=tiling,
            )
        return compile_mxu0(
            snapshot,
            request,
            fixed_inputs=fixed_inputs,
            fixed_outputs=fixed_outputs,
            rtl_root=target_source,
            destination=destination,
            limits=limits,
            contract=contract,
        )
