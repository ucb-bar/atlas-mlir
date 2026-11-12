"""Atlas public binding, source custody and native candidate-selection orchestration.

Actual numerical/resource profiles, graph admission, physical stream replay,
virtual/native emission and publication have explicit owners next to this file.
The public class and compatibility names retain their existing bindings.
"""

# ruff: noqa: F401 -- intentional exact compatibility exports from former binding ownership
from __future__ import annotations

import hashlib
import json
import os
import struct
import subprocess
import sys
from collections import Counter
from dataclasses import dataclass, replace
from pathlib import Path

from merlin.compiler.frontend.inputs import StagedCompileInputs
from merlin.compiler.semantic.allocate import (
    AllocationResult,
    CandidateGraph,
    StorageBank,
    check_assignment,
)
from merlin.compiler.semantic.model import (
    IndexMap,
    KernelRequest,
    TensorType,
    pointwise_identity_maps,
)
from merlin.compiler.quantization.contraction.constants import (
    PreparedStaticInput,
    load_declared_static_operand,
)
from merlin.compiler.quantization.contraction.transform import (
    DiagnosticMatmulTile,
    MatmulTileProfile,
    build_diagnostic_batched_matmul_tiles,
    build_diagnostic_matmul_tile,
    build_diagnostic_matmul_tiles,
)
from merlin.compiler.quantization.contraction.values import FloatBoundarySpec
from merlin.compiler.semantic.rules import (
    AddressConstraint,
    AxisBound,
    InstructionDescriptor,
    IntegerAttributeBound,
)
from merlin.compiler.semantic.search import (
    MaterializationDecision,
    SearchLimits,
    SearchResult,
)
from merlin.compiler.semantic.snapshot import NativeSnapshot, NativeTargetProfile
from merlin.compiler.semantic.target_binding import NativeCompilationError

from . import native, planning, profiles, publication
from .admission import (
    _bf16_tile_request,
    _interlayer_pack_code,
    _mxu0_accumulate_maps,
    _mxu0_k_chain_request,
    _mxu0_maps,
    _mxu_bf16_binary_request,
    _mxu_bf16_dag_request,
    _row_reduce_maps,
    _scaled_pack_code,
    _transpose_maps,
)
from .artifacts import publish_program
from .bf16_stream import (
    _check_bf16_selection,
    _realized_bf16_assignment,
    check_bf16_tile_stream,
)
from .dag_selection import _mxu_binary_selection, _mxu_dag_selection
from .dag_stream import check_mxu_bf16_binary_stream, check_mxu_bf16_dag_stream
from .layout import pack_bf16_pair_halves
from .mxu_stream import (
    _mxu0_k_chain_selection,
    check_mxu0_k_chain_stream,
    check_mxu0_stream,
    check_mxu_stream,
)
from .native import _MaterializedCandidate, _native_pipeline, _run, _sha256
from .pack_lane import ROUNDED_PACK_DOMAIN, ROUNDED_PACK_POLICY
from .pack_stream import (
    _check_pack_roundtrip_words,
    _pack_roundtrip_effects,
    check_copy_transpose_stream,
    check_pack_stream,
    check_transpose_stream,
)
from .policy import (
    BF16_ADD_ATTRS,
    BF16_BINARY_ATTRS,
    BF16_BINARY_MODES,
    BF16_DRAM_PAIR_LAYOUT,
    BF16_EXP_ATTRS,
    BF16_MUL_ATTRS,
    BF16_RECIP_ATTRS,
    BF16_RELU_ATTRS,
    BF16_ROW_MAX_ATTRS,
    BF16_ROW_REDUCTION_ATTRS,
    BF16_ROW_REDUCTION_MODES,
    BF16_ROW_SUM_ATTRS,
    BF16_SUB_ATTRS,
    BF16_TENSOR_PAIR_LAYOUT,
    BF16_UNARY_ATTRS,
    BF16_UNARY_MODES,
    DRAM_BASE,
    DRAM_SLOTS,
    FINITE_ANCHOR_POLICY,
    FP8_DRAM_TILE_LAYOUT,
    FP8_TENSOR_TILE_LAYOUT,
    MXU0_ACC_TILE_LAYOUT,
    MXU0_INPUT_DOMAIN,
    MXU0_POLICY,
    MXU1_FINITE_INPUT_DOMAIN,
    MXU1_FINITE_POLICY,
    MXU1_INPUT_DOMAIN,
    MXU1_POLICY,
    NATIVE_CHECKS,
    NATIVE_PIPELINES,
    PACK_ATTRS,
    PACK_POLICY,
    PACK_RULE_BY_POLICY,
    PACK_RULE_BY_TRANSPOSE_RULE,
    PACK_RULE_NAMES,
    RTL_REVISION,
    SCALED_PACK_DOMAIN,
    SCALED_PACK_POLICY,
    SCALED_PACK_RANGE,
    STORE_RULE_BY_POLICY,
    STORE_RULE_NAMES,
    TARGET_IDENTITY,
    TILE_BYTES,
    TRANSPOSE_RULE_BY_POLICY,
    TRANSPOSE_RULE_NAMES,
)
from .profiles import (
    _canonicalized_zero_boundary_profile,
    _finite_anchor_policy_profile,
    _mixed_positive_zero_boundary_profile,
    _positive_zero_boundary_profile,
    diagnostic_mxu0_f32_profile,
    diagnostic_mxu0_scaled_f32_profile,
    diagnostic_mxu1_f32_profile,
    diagnostic_mxu1_finite_f32_profile,
)
from .transfer import (
    _check_atomic_dma_schedule,
    _check_mxu_dag_timing,
    _tile_effects,
    check_copy_stream,
    check_copy_twice_stream,
)
from .virtual import _virtual_from_selected


def implementation_sources() -> dict[str, str]:
    """Observe selected OOT compiler files, excluding independent references/evaluators.

    Native executable hashes and the shared core snapshot retain their separate
    source scope. This is no loaded-bytecode or complete dependency attestation.
    """
    root = Path(__file__).resolve().parent
    names = (
            "admission.py",
            "artifacts.py",
            "bf16_stream.py",
            "binding.py",
            "backend.py",
            "call.py",
            "regions.py",
            "dag_selection.py",
            "dag_stream.py",
            "graph_precision.py",
            "layout.py",
            "mxu_stream.py",
            "native.py",
            "pack_lane.py",
            "pack_stream.py",
            "planning.py",
            "policy.py",
            "precision_abi.py",
            "profiles.py",
            "publication.py",
            "transfer.py",
            "virtual.py",
        )
    try:
        return {name: _sha256((root / name).read_bytes()) for name in names}
    except OSError as error:
        raise NativeCompilationError("compile_error", "Atlas binding implementation source is unavailable") from error


def revalidate_implementation(original: dict[str, str]) -> None:
    if implementation_sources() != original:
        raise NativeCompilationError(
            "compile_error", "Atlas binding implementation changed during compilation"
        )


class AtlasNativeBinding:
    """Public target binding; Merlin owns search and independent acceptance remains separate."""

    @staticmethod
    def prepare_ordered_mn(request: KernelRequest, tile_m: int, tile_n: int):
        """Qualify independent MXU0 output tiles and positive-zero tail rows."""
        from merlin.compiler.transforms.contraction_mn import (
            OrderedMNTilingError,
            partition_ordered_matmul_mn,
        )

        descriptor = next(item for item in AtlasNativeBinding.profile().descriptors
                          if item.name == "atlas.mxu0_reset_acc_tile")
        declared_m, declared_n = (bound.minimum for bound in descriptor.output_axis_bounds)
        if (
            type(tile_m) is not int or type(tile_n) is not int
            or (tile_m, tile_n) != (declared_m, declared_n)
            or request.target_identity != TARGET_IDENTITY
            or request.lowering_policy != "diagnostic"
            or len(request.nodes) != 3
            or any(node.type.dtype != "fp8_e4m3"
                   or node.type.numerical_policy != "raw-bit-pattern"
                   or node.attrs != (("input_domain", MXU0_INPUT_DOMAIN),)
                   for node in request.nodes[:2])
            or request.nodes[-1].type.dtype != "bf16"
            or request.nodes[-1].type.numerical_policy != MXU0_POLICY
        ):
            raise NativeCompilationError("unsupported_semantics", "source is outside the selected MXU0 M/N tile policy")
        try:
            m_extent = request.nodes[0].type.shape[0]
            n_extent = request.nodes[1].type.shape[0]
            return partition_ordered_matmul_mn(
                request, tile_m=tile_m, tile_n=tile_n, qualified_policy=MXU0_POLICY,
                tail_pad_element=(b"\x00" if m_extent % tile_m or n_extent % tile_n else None),
            )
        except OrderedMNTilingError as exc:
            raise NativeCompilationError("unsupported_semantics", str(exc)) from exc

    @staticmethod
    def prepare_ordered_k(request: KernelRequest, tile_k: int):
        """Qualify MXU0 K windows with any short tile before real elements."""
        from merlin.compiler.transforms.contraction_k import OrderedKTilingError, partition_ordered_matmul_k

        descriptor = next(item for item in AtlasNativeBinding.profile().descriptors
                          if item.name == "atlas.mxu0_reset_acc_tile")
        declared_k = descriptor.input_axis_bounds[0][1].minimum
        output_shape = tuple(bound.minimum for bound in descriptor.output_axis_bounds)
        if (
            type(tile_k) is not int or tile_k != declared_k
            or request.target_identity != TARGET_IDENTITY
            or request.lowering_policy != "diagnostic"
            or len(request.nodes) != 3
            or any(node.type.dtype != "fp8_e4m3"
                   or node.type.numerical_policy != "raw-bit-pattern"
                   or node.attrs != (("input_domain", MXU0_INPUT_DOMAIN),)
                   for node in request.nodes[:2])
            or request.nodes[-1].type != TensorType(output_shape, "bf16", MXU0_POLICY)
        ):
            raise NativeCompilationError("unsupported_semantics", "source is outside the selected MXU0 K tile policy")
        try:
            k_extent = request.nodes[0].type.shape[-1]
            return partition_ordered_matmul_k(
                request, tile_k=tile_k, qualified_policy=MXU0_POLICY,
                head_pad_element=b"\x00" if k_extent % tile_k else None,
            )
        except OrderedKTilingError as exc:
            raise NativeCompilationError("unsupported_semantics", str(exc)) from exc

    @staticmethod
    def prepare_precision_graph(source: str, *, entry: str, policy: dict[str, object],
                                original_source_sha256: str | None = None):
        """Apply the selected diagnostic graph precision policy to parsed Linalg."""
        from .graph_precision import lower_diagnostic_graph

        return lower_diagnostic_graph(source, entry=entry, policy=policy,
                                      original_source_sha256=original_source_sha256)

    @staticmethod
    def prepare_precision_tile(
        staged: StagedCompileInputs, region_id: str, *, tile_index: int | None = None,
    ) -> DiagnosticMatmulTile:
        if staged.precision_policy_document is None:
            raise NativeCompilationError("unsupported_semantics", "source-bound tile needs a selected policy")
        try:
            policy = json.loads(staged.precision_policy_document)
            schema = policy["schema"]
            if type(schema) is not str or schema not in {
                "atlas.oot.quantized_linalg_policy.v3", "atlas.oot.quantized_linalg_policy.v4",
                "atlas.oot.quantized_linalg_policy.v5", "atlas.oot.quantized_linalg_policy.v6",
                "atlas.oot.quantized_linalg_policy.v7", "atlas.oot.quantized_linalg_policy.v8",
            }:
                raise NativeCompilationError("unsupported_semantics", "source-bound precision policy schema differs")
            named = policy["matmul_regions"]
            batched = (policy["batched_matmul_regions"] if schema in {
                "atlas.oot.quantized_linalg_policy.v7", "atlas.oot.quantized_linalg_policy.v8",
            } else {})
            if (not isinstance(named, dict) or not isinstance(batched, dict)
                    or (region_id in named and region_id in batched)):
                raise ValueError("source-bound region has ambiguous precision decisions")
            is_batched = region_id in batched
            decision = (batched if is_batched else named)[region_id]
            if not isinstance(decision, dict):
                raise TypeError("source-bound region needs a precision decision object")
        except (ValueError, TypeError, KeyError, IndexError) as exc:
            raise NativeCompilationError(
                "unsupported_semantics", "source-bound tile has no valid selected matmul decision"
            ) from exc
        if is_batched and decision.get("mxu_unit") != 0:
            raise NativeCompilationError("unqualified_target", "batched source tiles currently need MXU0")
        if schema.endswith((".v5", ".v8")) and (
            not isinstance(decision, dict) or decision.get("mxu_unit") != 0
        ):
            raise NativeCompilationError(
                "unqualified_target", "positive-zero source boundary currently needs MXU0"
            )
        if decision == {"mxu_unit": 0, "activation_pack_scale_code": 127}:
            profile = diagnostic_mxu0_f32_profile()
        elif (isinstance(decision, dict) and set(decision) == {"mxu_unit", "activation_pack_scale_code"}
              and type(decision["mxu_unit"]) is int and decision["mxu_unit"] == 0
              and type(decision["activation_pack_scale_code"]) is int
              and decision["activation_pack_scale_code"] in range(SCALED_PACK_RANGE[0], SCALED_PACK_RANGE[1] + 1)):
            profile = diagnostic_mxu0_scaled_f32_profile(decision["activation_pack_scale_code"])
        elif decision == {"mxu_unit": 1, "activation_pack_scale_code": 127}:
            profile = (diagnostic_mxu1_f32_profile() if schema.endswith(".v3")
                       else diagnostic_mxu1_finite_f32_profile(127))
        elif (isinstance(decision, dict) and set(decision) == {"mxu_unit", "activation_pack_scale_code"}
              and type(decision["mxu_unit"]) is int and decision["mxu_unit"] == 1
              and type(decision["activation_pack_scale_code"]) is int
              and decision["activation_pack_scale_code"] in range(SCALED_PACK_RANGE[0], SCALED_PACK_RANGE[1] + 1)):
            if schema.endswith(".v3"):
                raise NativeCompilationError(
                    "unqualified_target", "selected MXU1 activation scale has no qualified tile profile"
                )
            profile = diagnostic_mxu1_finite_f32_profile(decision["activation_pack_scale_code"])
        else:
            raise NativeCompilationError(
                "unsupported_semantics", "selected MXU unit or activation scale has no native tile profile"
            )
        if schema.endswith((".v4", ".v6")) and decision["mxu_unit"] == 0:
            profile = _finite_anchor_policy_profile(profile)
        if schema.endswith(".v5"):
            profile = _positive_zero_boundary_profile(profile)
        elif schema.endswith(".v6"):
            profile = _mixed_positive_zero_boundary_profile(profile)
        elif schema.endswith(".v7"):
            profile = replace(profile, required_policy_fields=tuple(
                (key, schema if key == "schema" else value)
                for key, value in profile.required_policy_fields
            ))
        elif schema.endswith(".v8"):
            profile = _canonicalized_zero_boundary_profile(profile, schema)
        args = {
            "entry": staged.entry,
            "policy_document": staged.precision_policy_document,
            "original_source_sha256": staged.original_source_sha256,
            "region_id": region_id,
            "profile": profile,
            "ordered_k_continuation": decision["mxu_unit"] == 0,
        }
        if is_batched and tile_index is None:
            raise NativeCompilationError(
                "unsupported_semantics", "batched source contraction needs an explicit tile index"
            )
        if tile_index is None:
            return build_diagnostic_matmul_tile(staged.program, **args)
        if type(tile_index) is not int or tile_index < 0:
            raise NativeCompilationError("unsupported_semantics", "source tile index must be nonnegative")
        tiles = (build_diagnostic_batched_matmul_tiles if is_batched else build_diagnostic_matmul_tiles)(
            staged.program, **args,
        )
        if tile_index >= len(tiles):
            raise NativeCompilationError("unsupported_semantics", "source tile index exceeds output grid")
        return tiles[tile_index]

    @staticmethod
    def prepare_precision_constants(
        plan: DiagnosticMatmulTile, staged_root: Path, execution_plan: dict,
    ) -> tuple[PreparedStaticInput, ...]:
        """Pack declared parameters for the selected physical input addresses."""
        from .precision_abi import pack_source_operand

        addresses = execution_plan.get("input_addresses") if isinstance(execution_plan, dict) else None
        if (
            plan.request.target_identity != TARGET_IDENTITY
            or not isinstance(addresses, list) or len(addresses) != len(plan.inputs)
            or any(type(address) is not int or address < DRAM_BASE for address in addresses)
        ):
            raise NativeCompilationError("compile_error", "precision constants need the selected tile input ABI")
        prepared = []
        for index, conversion in enumerate(plan.inputs):
            role = conversion.source_role
            source = load_declared_static_operand(staged_root, plan, role)
            if source is None:
                continue
            try:
                payload = pack_source_operand(plan, role, source.value)
            except ValueError as error:
                if "outside the selected target input domain" not in str(error):
                    raise
                prepared.append(PreparedStaticInput(role, addresses[index], source, None, str(error)))
            else:
                prepared.append(PreparedStaticInput(role, addresses[index], source, payload))
        return tuple(prepared)

    profile = staticmethod(profiles.profile)

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
        tool_directory: Path | None = None,
        pipeline: str | None = None,
    ) -> dict[str, object]:
        source_pins = implementation_sources()
        pipeline = _native_pipeline() if pipeline is None else pipeline
        if type(pipeline) is not str or pipeline not in NATIVE_PIPELINES:
            raise NativeCompilationError("unsupported_semantics", "selected native pipeline is unknown")
        plan = planning.admit_request(
            request,
            fixed_inputs=fixed_inputs,
            fixed_outputs=fixed_outputs,
            target_source=target_source,
            pipeline=pipeline,
        )
        materialized: _MaterializedCandidate | None = None

        def accept_candidate(
            selected: SearchResult,
        ) -> tuple[bool, str] | MaterializationDecision:
            nonlocal materialized
            try:
                materialized = native.prepare_candidate(
                    selected,
                    plan,
                    target_source=target_source,
                    hooks=sys.modules[__name__],
                    **({"tool_directory": tool_directory} if tool_directory is not None else {}),
                )
            except NativeCompilationError as exc:
                if exc.status not in {"unqualified_target", "unsupported_semantics"}:
                    raise
                materialized = None
                return False, exc.reason
            realized = materialized.realized_allocation
            if realized is not None:
                assert selected.graph is not None
                valid, reason = check_assignment(
                    selected.graph,
                    realized.order,
                    realized.addresses,
                    snapshot.profile.banks,
                    fixed_inputs=fixed_inputs,
                    fixed_outputs=fixed_outputs,
                )
                if not valid:
                    materialized = None
                    return (
                        False,
                        f"physical lowering changed the selected assignment: {reason}",
                    )
            return MaterializationDecision(True, realized_allocation=realized)

        selected = snapshot.select(
            request,
            fixed_inputs=fixed_inputs,
            fixed_outputs=fixed_outputs,
            limits=limits,
            accept_materialization=accept_candidate,
        )
        if selected.status != "selected":
            status = (
                selected.status
                if selected.status
                in {
                    "unsupported_semantics",
                    "unqualified_target",
                    "infeasible_candidate",
                    "search_timeout",
                    "resource_limit",
                    "tool_unavailable",
                }
                else "compile_error"
            )
            raise NativeCompilationError(
                status, selected.reason or "native selection found no checked candidate"
            )
        assert materialized is not None
        revalidate_implementation(source_pins)
        result = publication.publish_result(
            selected,
            materialized,
            plan,
            snapshot=snapshot,
            destination=destination,
            binding_source=Path(__file__),
            publish_program=publish_program,
            source_pins=source_pins,
        )
        try:
            revalidate_implementation(source_pins)
        except NativeCompilationError:
            (destination / "manifest.json").unlink(missing_ok=True)
            raise
        return result
