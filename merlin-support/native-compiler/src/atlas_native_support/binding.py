"""Installed Merlin native target support entry point for the movement subset."""

from __future__ import annotations

from pathlib import Path

from merlin.semantic_compiler.model import KernelRequest
from merlin.semantic_compiler.search import SearchLimits
from merlin.semantic_compiler.snapshot import NativeSnapshot, NativeTargetProfile

from .movement import MovementContract, compile_movement, movement_profile


class Binding:
    @staticmethod
    def profile() -> NativeTargetProfile:
        return movement_profile(MovementContract.load())

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
        return compile_movement(
            snapshot, request, fixed_inputs=fixed_inputs, fixed_outputs=fixed_outputs,
            rtl_root=target_source, destination=destination, limits=limits,
        )
