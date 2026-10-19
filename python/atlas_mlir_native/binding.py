"""Bounded tile compilation through Merlin selection and Atlas emission.

This native vertical slice is diagnostic. Its register assignment is
checked against the OOT lowerer; neither a bitstream nor a cache policy is
qualified here. MXU0 reset arithmetic has a finite diagnostic domain only.
"""

from __future__ import annotations

import hashlib
import json
import os
import struct
import subprocess
from collections import Counter
from dataclasses import dataclass, replace
from pathlib import Path

from merlin.frontends.compile_inputs import StagedCompileInputs
from merlin.semantic_compiler.allocate import (
    AllocationResult,
    CandidateGraph,
    StorageBank,
    check_assignment,
)
from merlin.semantic_compiler.model import (
    IndexMap,
    KernelRequest,
    TensorType,
    pointwise_identity_maps,
)
from merlin.semantic_compiler.precision_constants import (
    PreparedStaticInput,
    load_declared_static_operand,
)
from merlin.semantic_compiler.precision_transform import (
    DiagnosticMatmulTile,
    MatmulTileProfile,
    build_diagnostic_batched_matmul_tiles,
    build_diagnostic_matmul_tile,
    build_diagnostic_matmul_tiles,
)
from merlin.semantic_compiler.precision_values import FloatBoundarySpec
from merlin.semantic_compiler.rules import (
    AddressConstraint,
    AxisBound,
    InstructionDescriptor,
    IntegerAttributeBound,
)
from merlin.semantic_compiler.search import (
    MaterializationDecision,
    SearchLimits,
    SearchResult,
)
from merlin.semantic_compiler.snapshot import NativeSnapshot, NativeTargetProfile
from merlin.semantic_compiler.target_binding import NativeCompilationError

from .layout import pack_bf16_pair_halves
from .pack_lane import ROUNDED_PACK_DOMAIN, ROUNDED_PACK_POLICY

RTL_REVISION = "0079c0541111197741a231c002e3843fa6f545b2"
TARGET_IDENTITY = f"atlas-rtl-{RTL_REVISION}:movement-mxu01-vpu-pack-tile-v21"
FP8_DRAM_TILE_LAYOUT = "atlas.dram.fp8-e4m3-row-major-32x32-in-2k-slot.v1"
FP8_TENSOR_TILE_LAYOUT = "atlas.tensor.fp8-e4m3-row-major-32x32.v1"
BF16_DRAM_PAIR_LAYOUT = "atlas.dram.bf16-pair-halves-32x32-in-2k-slot.v1"
BF16_TENSOR_PAIR_LAYOUT = "atlas.tensor.bf16-pair-halves-32x32.v1"
MXU0_ACC_TILE_LAYOUT = "atlas.mxu0.accumulator-slot-32x32-bf16.v1"
MXU0_POLICY = "mxu0-e4m3-ordered-bf16-finite-v1"
MXU0_INPUT_DOMAIN = "finite-normal-e4m3-or-positive-zero"
MXU1_POLICY = "mxu1-e4m3-anchor-bf16-bounded-v1"
MXU1_INPUT_DOMAIN = "e4m3-positive-zero-or-signed-one-two-v1"
MXU1_FINITE_POLICY = "mxu1-e4m3-anchor-bf16-finite-reset-v1"
MXU1_FINITE_INPUT_DOMAIN = "finite-normal-e4m3-or-positive-zero"
FINITE_ANCHOR_POLICY = "selected-source-anchor-finite-reset-v1"
PACK_POLICY = "fp8-e4m3-unit-e8m0-seven-codes-v1"
PACK_ATTRS = (("input_domain", "bf16-seven-powers-and-zero-v1"), ("scale_code", 127))
SCALED_PACK_POLICY = "fp8-e4m3-scaled-e8m0-exact-normal-122-128-v1"
SCALED_PACK_DOMAIN = "bf16-positive-zero-or-exact-normal-under-scale-122-128-v1"
SCALED_PACK_RANGE = (122, 128)
PACK_RULE_BY_POLICY = {
    PACK_POLICY: "atlas.vpu_pack_unit_fp8_tile",
    SCALED_PACK_POLICY: "atlas.vpu_pack_scaled_fp8_tile",
    ROUNDED_PACK_POLICY: "atlas.vpu_pack_rounded_fp8_tile",
}
TRANSPOSE_RULE_BY_POLICY = {
    PACK_POLICY: "atlas.xlu_transpose_unit_fp8_tile",
    SCALED_PACK_POLICY: "atlas.xlu_transpose_scaled_fp8_tile",
    ROUNDED_PACK_POLICY: "atlas.xlu_transpose_rounded_fp8_tile",
}
STORE_RULE_BY_POLICY = {
    PACK_POLICY: "atlas.store_fp8_unit_tile",
    SCALED_PACK_POLICY: "atlas.store_fp8_scaled_tile",
    ROUNDED_PACK_POLICY: "atlas.store_fp8_rounded_tile",
}
PACK_RULE_NAMES = frozenset(PACK_RULE_BY_POLICY.values())
TRANSPOSE_RULE_NAMES = frozenset(TRANSPOSE_RULE_BY_POLICY.values())
STORE_RULE_NAMES = frozenset(STORE_RULE_BY_POLICY.values())
PACK_RULE_BY_TRANSPOSE_RULE = {
    transpose: PACK_RULE_BY_POLICY[policy]
    for policy, transpose in TRANSPOSE_RULE_BY_POLICY.items()
}
BF16_RELU_ATTRS = (("input_domain", "finite-normal-or-zero-bf16"),
                  ("numeric_semantics", "bf16-sign-clamp-v1"))
BF16_ADD_ATTRS = (("input_domain", "finite-normal-exact-fp32-sum-v1"),
                  ("numeric_semantics", "selected-f32-rne-bf16-chop-v1"))
BF16_SUB_ATTRS = (("input_domain", "finite-normal-exact-fp32-difference-or-zero-v1"),
                  ("numeric_semantics", "selected-f32-rne-bf16-chop-v1"))
BF16_MUL_ATTRS = (("input_domain", "finite-normal-bf16-product-normal-v1"),
                  ("numeric_semantics", "bf16-product-rne-v1"))
BF16_ROW_MAX_ATTRS = (("input_domain", "bf16-raw-all-codes-v1"),
                      ("numeric_semantics", "selected-rtl-bf16-total-order-row-max-v1"))
BF16_ROW_SUM_ATTRS = (
    ("input_domain", "bf16-normal-or-positive-zero-fp32-tree-normal-result-v1"),
    ("numeric_semantics", "selected-rtl-bf16-row-sum-fp32-rne-tree-bf16-rne-v1"),
)
BF16_EXP_ATTRS = (
    ("input_domain", "bf16-normal-minus80-plus80-or-signed-zero-v1"),
    ("numeric_semantics", "selected-rtl-bf16-exp-q9p12-lut-rne-v1"),
)
BF16_RECIP_ATTRS = (
    ("input_domain", "bf16-positive-normal-1-to-32-v1"),
    ("numeric_semantics", "selected-rtl-bf16-rcp-lut-trunc-v1"),
)
BF16_UNARY_ATTRS = {"exp_bf16": BF16_EXP_ATTRS, "recip_bf16": BF16_RECIP_ATTRS}
BF16_UNARY_MODES = {
    "atlas.vpu_exp_bf16_tile": "exp",
    "atlas.vpu_recip_bf16_tile": "recip",
}
BF16_ROW_REDUCTION_ATTRS = {
    "row_max_bf16": BF16_ROW_MAX_ATTRS,
    "row_sum_bf16": BF16_ROW_SUM_ATTRS,
}
BF16_ROW_REDUCTION_MODES = {
    "atlas.vpu_row_max_bf16_tile": "row_max",
    "atlas.vpu_row_sum_bf16_tile": "row_sum",
}
BF16_BINARY_ATTRS = {
    "add_bf16": BF16_ADD_ATTRS,
    "sub_bf16": BF16_SUB_ATTRS,
    "mul_bf16": BF16_MUL_ATTRS,
}
BF16_BINARY_MODES = {
    "atlas.vpu_add_bf16_tile": "add",
    "atlas.vpu_sub_bf16_tile": "sub",
    "atlas.vpu_mul_bf16_tile": "mul",
}
TILE_BYTES = 2048
DRAM_BASE = 0x90000000
DRAM_SLOTS = 128
NATIVE_PIPELINES = {
    "mandatory": (
        "--spill-atlas-virtual",
        "--lower-atlas-virtual-to-machine", "--insert-atlas-dma-waits",
        "--insert-atlas-delays",
    ),
    "optimized": (
        "--spill-atlas-mxu-state",
        "--verify-atlas-virtual-stream", "--lower-atlas-virtual-boundary-to-dma",
        "--schedule-atlas-virtual", "--verify-atlas-virtual-stream",
        "--spill-atlas-virtual",
        "--lower-atlas-virtual-to-machine", "--insert-atlas-dma-waits",
        "--verify-atlas-machine-stream", "--schedule-atlas-stream",
        "--verify-atlas-timing", "--verify-atlas-generated-schedule",
    ),
}
NATIVE_CHECKS = ("--verify-atlas-generated-schedule", "--verify-atlas-machine-stream")


def _native_pipeline() -> str:
    pipeline = os.environ.get("ATLAS_NATIVE_PIPELINE", "mandatory")
    if pipeline not in NATIVE_PIPELINES:
        raise NativeCompilationError(
            "unsupported_semantics", "ATLAS_NATIVE_PIPELINE must be mandatory or optimized"
        )
    return pipeline


def _scaled_pack_code(node) -> int | None:
    attrs = dict(node.attrs)
    code = attrs.get("scale_code")
    domain = {
        SCALED_PACK_POLICY: SCALED_PACK_DOMAIN,
        ROUNDED_PACK_POLICY: ROUNDED_PACK_DOMAIN,
    }.get(node.type.numerical_policy)
    if (
        domain is not None
        and set(attrs) == {"input_domain", "scale_code"}
        and attrs["input_domain"] == domain
        and type(code) is int
        and SCALED_PACK_RANGE[0] <= code <= SCALED_PACK_RANGE[1]
    ):
        return code
    return None


def _interlayer_pack_code(node) -> int | None:
    if node.type.numerical_policy == PACK_POLICY and node.attrs == PACK_ATTRS:
        return 127
    return _scaled_pack_code(node)


@dataclass(frozen=True)
class _MaterializedCandidate:
    selected_registers: list[int]
    virtual: str
    machine: str
    program: dict
    words: list[int]
    tool_hashes: tuple[str, str]
    input_address: int
    output_address: int
    realized_allocation: AllocationResult | None = None
    qualification_machine: str | None = None
    qualification_program: dict | None = None


def _bf16_tile_request(request: KernelRequest) -> bool:
    """Admit bounded BF16 movement, binary arithmetic, and ReLU DAGs."""
    if not request.input_storages or not request.outputs:
        return False
    nodes = {node.id: node for node in request.nodes}
    if any(nodes[output].op not in {"identity", "relu", *BF16_UNARY_ATTRS,
                                    *BF16_ROW_REDUCTION_ATTRS,
                                    *BF16_BINARY_ATTRS}
           for output in request.outputs):
        return False
    for node in request.nodes:
        if node.type.shape != (32, 32) or node.type.dtype != "bf16" or (
            node.type.numerical_policy != "raw-bit-pattern"
        ):
            return False
        if node.op == "input":
            if (node.effect != "input" or node.attrs not in (
                (), *((attrs[0],) for attrs in BF16_UNARY_ATTRS.values()),
                *((attrs[0],) for attrs in BF16_ROW_REDUCTION_ATTRS.values()),
                *((attrs[0],) for attrs in BF16_BINARY_ATTRS.values()),
            )
                    or node.index_maps):
                return False
        elif node.op == "identity":
            if (node.effect != "pure" or node.attrs or node.index_maps or len(node.inputs) != 1
                    or nodes[node.inputs[0]].type != node.type):
                return False
        elif node.op == "relu":
            if (node.effect != "pure" or dict(node.attrs) != dict(BF16_RELU_ATTRS)
                    or node.index_maps != pointwise_identity_maps(2, 1)
                    or len(node.inputs) != 1 or nodes[node.inputs[0]].type != node.type):
                return False
        elif node.op in BF16_UNARY_ATTRS:
            if (node.effect != "pure" or node.attrs != BF16_UNARY_ATTRS[node.op]
                    or node.index_maps != pointwise_identity_maps(2, 1)
                    or len(node.inputs) != 1 or nodes[node.inputs[0]].type != node.type):
                return False
        elif node.op in BF16_ROW_REDUCTION_ATTRS:
            if (node.effect != "pure" or node.attrs != BF16_ROW_REDUCTION_ATTRS[node.op]
                    or node.index_maps != _row_reduce_maps()
                    or len(node.inputs) != 1 or nodes[node.inputs[0]].type != node.type
                    or nodes[node.inputs[0]].op not in {"input", "relu", *BF16_UNARY_ATTRS,
                                                       *BF16_ROW_REDUCTION_ATTRS,
                                                       *BF16_BINARY_ATTRS}):
                return False
        elif node.op in BF16_BINARY_ATTRS:
            attributes = BF16_BINARY_ATTRS[node.op]
            sources = tuple(nodes[source_id] for source_id in node.inputs)
            # A ReLU can return zero. Its binary consumer's finite-normal
            # operand premise remains unproved until the caller establishes
            # it for the selected invocation; diagnostic compilation records
            # that obligation in the execution plan.
            sources_admitted = all(
                (source.op == "input" and source.attrs == (attributes[0],))
                or (source.op in BF16_BINARY_ATTRS
                    and source.attrs == BF16_BINARY_ATTRS[source.op])
                or (source.op == "relu" and source.attrs == BF16_RELU_ATTRS)
                or (source.op in BF16_UNARY_ATTRS
                    and source.attrs == BF16_UNARY_ATTRS[source.op])
                or (source.op in BF16_ROW_REDUCTION_ATTRS
                    and source.attrs == BF16_ROW_REDUCTION_ATTRS[source.op])
                for source in sources
            )
            if (node.effect != "pure" or node.attrs != attributes
                    or node.index_maps != pointwise_identity_maps(2, 2)
                    or len(node.inputs) != 2 or node.inputs[0] == node.inputs[1]
                    or not sources_admitted):
                return False
        else:
            return False
    return True


def _mxu_bf16_binary_request(request: KernelRequest) -> bool:
    """Admit one selected MXU0 readout consumed by a BF16 VPU binary op."""
    if len(request.nodes) != 5 or len(request.outputs) != 1:
        return False
    nodes = {node.id: node for node in request.nodes}
    if len(nodes) != 5:
        return False
    inputs = [node for node in request.nodes if node.op == "input"]
    mxus = [node for node in request.nodes if node.op == "matmul"]
    binaries = [node for node in request.nodes if node.op in {"add_bf16", "mul_bf16"}]
    if len(inputs) != 3 or len(mxus) != 1 or len(binaries) != 1:
        return False
    mxu, binary = mxus[0], binaries[0]
    if request.outputs != (binary.id,) or len(mxu.inputs) != 2 or len(binary.inputs) != 2:
        return False
    fp8 = TensorType((32, 32), "fp8_e4m3", "raw-bit-pattern")
    bf16 = TensorType((32, 32), "bf16", "raw-bit-pattern")
    mxu_type = TensorType((32, 32), "bf16", MXU0_POLICY)
    fp8_inputs = [node for node in inputs if node.type == fp8]
    bf16_inputs = [node for node in inputs if node.type == bf16]
    attributes = BF16_ADD_ATTRS if binary.op == "add_bf16" else BF16_MUL_ATTRS
    return (
        len(fp8_inputs) == 2 and len(bf16_inputs) == 1
        and all(node.effect == "input" and node.attrs == (("input_domain", MXU0_INPUT_DOMAIN),)
                and not node.index_maps for node in fp8_inputs)
        and bf16_inputs[0].effect == "input"
        and bf16_inputs[0].attrs == (attributes[0],)
        and not bf16_inputs[0].index_maps
        and mxu.inputs == tuple(node.id for node in fp8_inputs)
        and mxu.type == mxu_type and mxu.effect == "pure" and not mxu.attrs
        and mxu.index_maps == _mxu0_maps()
        and set(binary.inputs) == {mxu.id, bf16_inputs[0].id}
        and binary.type == bf16 and binary.effect == "pure"
        and binary.attrs == attributes
        and binary.index_maps == pointwise_identity_maps(2, 2)
    )


def _mxu_bf16_dag_request(request: KernelRequest) -> bool:
    """Admit MXU0 reset results and a bounded BF16 VPU DAG with ordered outputs."""
    nodes = {node.id: node for node in request.nodes}
    if (len(nodes) != len(request.nodes) or not request.outputs
            or len(set(request.outputs)) != len(request.outputs)
            or any(output not in nodes for output in request.outputs)):
        return False
    mxus = [node for node in request.nodes if node.op == "matmul"]
    vpu = [node for node in request.nodes
           if node.op in {"relu", *BF16_UNARY_ATTRS,
                          *BF16_ROW_REDUCTION_ATTRS, *BF16_BINARY_ATTRS}]
    packs = [node for node in request.nodes if node.op == "pack_fp8"]
    fp8_inputs = [node for node in request.nodes if node.op == "input"
                  and node.type.dtype == "fp8_e4m3"]
    if not mxus or not (vpu or packs) or len(fp8_inputs) < 2:
        return False
    fp8 = TensorType((32, 32), "fp8_e4m3", "raw-bit-pattern")
    packed_fp8 = {
        TensorType((32, 32), "fp8_e4m3", PACK_POLICY),
        TensorType((32, 32), "fp8_e4m3", SCALED_PACK_POLICY),
        TensorType((32, 32), "fp8_e4m3", ROUNDED_PACK_POLICY),
    }
    bf16 = TensorType((32, 32), "bf16", "raw-bit-pattern")
    mxu_type = TensorType((32, 32), "bf16", MXU0_POLICY)
    allowed_sources = {"input", "constant", "matmul", "relu", *BF16_UNARY_ATTRS,
                       *BF16_ROW_REDUCTION_ATTRS,
                       *BF16_BINARY_ATTRS}
    for node in request.nodes:
        if any(source not in nodes for source in node.inputs):
            return False
        if node.op in {"input", "constant"}:
            if (node.effect != node.op or node.inputs or node.index_maps
                    or (node.op == "constant" and node.type != bf16)):
                return False
            if node.type == fp8:
                if node.attrs != (("input_domain", MXU0_INPUT_DOMAIN),):
                    return False
            elif node.type == bf16:
                if node.attrs not in ((), *((attrs[0],) for attrs in BF16_UNARY_ATTRS.values()),
                                      *((attrs[0],) for attrs in BF16_ROW_REDUCTION_ATTRS.values()),
                                      *((attrs[0],) for attrs in BF16_BINARY_ATTRS.values())):
                    return False
            else:
                return False
        elif node.op == "matmul":
            if (len(node.inputs) != 2 or node.inputs[0] == node.inputs[1]
                    or any(
                        (nodes[source].op != "input" or nodes[source].type != fp8)
                        and (nodes[source].op != "pack_fp8" or nodes[source].type not in packed_fp8)
                        and (nodes[source].op != "transpose" or nodes[source].type not in packed_fp8)
                        for source in node.inputs
                    )
                    or node.type != mxu_type or node.effect != "pure" or node.attrs
                    or node.index_maps != _mxu0_maps()):
                return False
        elif node.op == "pack_fp8":
            producer = nodes[node.inputs[0]] if len(node.inputs) == 1 else None
            if (node.type not in packed_fp8 or node.effect != "pure"
                    or _interlayer_pack_code(node) is None
                    or producer is None
                    or producer.op not in {"matmul", "relu", *BF16_UNARY_ATTRS,
                                           *BF16_ROW_REDUCTION_ATTRS,
                                           *BF16_BINARY_ATTRS}
                    or producer.type not in {mxu_type, bf16}
                    or node.index_maps != pointwise_identity_maps(2, 1)):
                return False
        elif node.op == "transpose":
            producer = nodes[node.inputs[0]] if len(node.inputs) == 1 else None
            if (producer is None or producer.op != "pack_fp8"
                    or producer.type not in packed_fp8 or node.type != producer.type
                    or node.effect != "pure" or node.attrs
                    or node.index_maps != _transpose_maps()):
                return False
        elif node.op == "relu":
            if (node.type != bf16 or node.effect != "pure" or node.attrs != BF16_RELU_ATTRS
                    or node.index_maps != pointwise_identity_maps(2, 1)
                    or len(node.inputs) != 1
                    or nodes[node.inputs[0]].type not in {bf16, mxu_type}):
                return False
        elif node.op in BF16_UNARY_ATTRS:
            if (node.type != bf16 or node.effect != "pure"
                    or node.attrs != BF16_UNARY_ATTRS[node.op]
                    or node.index_maps != pointwise_identity_maps(2, 1)
                    or len(node.inputs) != 1
                    or nodes[node.inputs[0]].op not in allowed_sources
                    or nodes[node.inputs[0]].type not in {bf16, mxu_type}):
                return False
        elif node.op in BF16_ROW_REDUCTION_ATTRS:
            if (node.type != bf16 or node.effect != "pure"
                    or node.attrs != BF16_ROW_REDUCTION_ATTRS[node.op]
                    or node.index_maps != _row_reduce_maps() or len(node.inputs) != 1
                    or nodes[node.inputs[0]].op not in allowed_sources
                    or nodes[node.inputs[0]].type not in {bf16, mxu_type}):
                return False
        elif node.op in BF16_BINARY_ATTRS:
            attributes = BF16_BINARY_ATTRS[node.op]
            if (node.type != bf16 or node.effect != "pure" or node.attrs != attributes
                    or node.index_maps != pointwise_identity_maps(2, 2)
                    or len(node.inputs) != 2 or node.inputs[0] == node.inputs[1]):
                return False
            for source_id in node.inputs:
                source = nodes[source_id]
                if source.op not in allowed_sources or source.type not in {bf16, mxu_type}:
                    return False
                if source.op in {"input", "constant"} and source.attrs != (attributes[0],):
                    return False
        else:
            return False
    if any(nodes[output].op not in {"matmul", "relu", *BF16_UNARY_ATTRS,
                                    *BF16_ROW_REDUCTION_ATTRS,
                                    *BF16_BINARY_ATTRS}
           for output in request.outputs):
        return False
    if any(
        node.op != "mul_bf16" or node.attrs != BF16_MUL_ATTRS
        for constant in request.nodes if constant.op == "constant"
        for node in request.nodes if constant.id in node.inputs
    ):
        return False
    reachable = set(request.outputs)
    pending = list(request.outputs)
    while pending:
        for source in nodes[pending.pop()].inputs:
            if source not in reachable:
                reachable.add(source)
                pending.append(source)
    return reachable == set(nodes)


def _transpose_maps() -> tuple[IndexMap, ...]:
    return (
        IndexMap(2, ((0, 1), (1, 0)), (0, 0)),
        IndexMap(2, ((1, 0), (0, 1)), (0, 0)),
    )


def _row_reduce_maps() -> tuple[IndexMap, ...]:
    # Logical loops (row, output_column, reduction_column) broadcast each row result.
    return (
        IndexMap(3, ((1, 0, 0), (0, 0, 1)), (0, 0)),
        IndexMap(3, ((1, 0, 0), (0, 1, 0)), (0, 0)),
    )


def _mxu0_maps() -> tuple[IndexMap, ...]:
    # A[M,K], W[N,K], C[M,N]; the weight orientation is observable.
    return (
        IndexMap(3, ((1, 0, 0), (0, 0, 1)), (0, 0)),
        IndexMap(3, ((0, 1, 0), (0, 0, 1)), (0, 0)),
        IndexMap(3, ((1, 0, 0), (0, 1, 0)), (0, 0)),
    )


def _mxu0_accumulate_maps() -> tuple[IndexMap, ...]:
    maps = _mxu0_maps()
    return (maps[0], maps[1], maps[2], maps[2])


def _mxu0_k_chain_request(request: KernelRequest) -> bool:
    """An ordered MXU0 reduction bounded by declared external storage."""
    if (
        len(request.nodes) % 3 or len(request.nodes) // 3 < 2
        or 2 * (len(request.nodes) // 3) + 1 > DRAM_SLOTS
        or request.outputs != (request.nodes[-1].id,)
    ):
        return False
    tile_count = len(request.nodes) // 3
    inputs = request.nodes[:2 * tile_count]
    reductions = request.nodes[2 * tile_count:]
    return (
        all(node.op == "input" and node.effect == "input"
            and node.type == TensorType((32, 32), "fp8_e4m3", "raw-bit-pattern")
            and node.attrs == (("input_domain", MXU0_INPUT_DOMAIN),)
            and not node.index_maps for node in inputs)
        and all(
            node.op == ("matmul" if index == 0 else "matmul_accumulate")
            and node.effect == "pure" and not node.attrs
            and node.inputs == ((inputs[0].id, inputs[1].id) if index == 0 else
                                (inputs[2 * index].id, inputs[2 * index + 1].id,
                                 reductions[index - 1].id))
            and node.type == TensorType((32, 32), "bf16", MXU0_POLICY)
            and node.index_maps == (_mxu0_maps() if index == 0 else _mxu0_accumulate_maps())
            for index, node in enumerate(reductions)
        )
    )


def diagnostic_mxu0_f32_profile() -> MatmulTileProfile:
    """Describe the source-bound f32-to-MXU0 tile as an explicit approximation."""
    operand_conversion = FloatBoundarySpec(
        "f32-to-e4m3-finite-normal-rne-flush-subnormal", "fp8_e4m3", "encode",
        max_normal_exponent_code=14, flush_below_normal=True,
        admitted_code_ranges=((0, 0), (8, 119), (136, 247)),
    )
    return MatmulTileProfile(
        TARGET_IDENTITY, (32, 32, 32), _mxu0_maps(),
        ("fp8_e4m3", "fp8_e4m3"), ("raw-bit-pattern", "raw-bit-pattern"),
        (MXU0_INPUT_DOMAIN, MXU0_INPUT_DOMAIN),
        ("f32-to-e4m3-finite-normal-rne-flush-subnormal",) * 2,
        "bf16", MXU0_POLICY, "bf16-to-f32-exact", "external", (1, 1),
        (
            ("schema", "atlas.oot.quantized_linalg_policy.v3"),
            ("selected_rtl_revision", RTL_REVISION),
            ("source_dtype", "f32"),
            ("operand_encoding", "e4m3_finite_normal_rne_flush_subnormal"),
            ("bias_encoding", "bf16_rne"),
            ("vpu_add", "selected_f32_sum_bf16_chop"),
            ("precision_transform", "explicit_diagnostic_candidate"),
        ),
        (("mxu_unit", 0), ("activation_pack_scale_code", 127)),
        ("softmax_realization",),
        (operand_conversion, operand_conversion, FloatBoundarySpec("bf16-to-f32-exact", "bf16", "decode")),
    )


def diagnostic_mxu1_f32_profile() -> MatmulTileProfile:
    """Keep MXU1's anchor-tree arithmetic distinct from ordered MXU0."""
    base = diagnostic_mxu0_f32_profile()
    operand_conversion = FloatBoundarySpec(
        "f32-to-e4m3-finite-normal-rne-flush-subnormal", "fp8_e4m3", "encode",
        max_normal_exponent_code=14, flush_below_normal=True,
        admitted_code_ranges=((0, 0), (56, 56), (64, 64), (184, 184), (192, 192)),
    )
    return MatmulTileProfile(
        TARGET_IDENTITY, base.loop_shape, base.maps,
        base.operand_dtypes, base.operand_policies,
        (MXU1_INPUT_DOMAIN, MXU1_INPUT_DOMAIN),
        base.operand_conversion_policies,
        base.result_dtype, MXU1_POLICY, base.result_conversion_policy,
        base.storage, base.input_extents, base.required_policy_fields,
        (("mxu_unit", 1), ("activation_pack_scale_code", 127)),
        base.optional_policy_fields,
        (operand_conversion, operand_conversion, base.conversion_specs[2]),
    )


def diagnostic_mxu0_scaled_f32_profile(scale_code: int) -> MatmulTileProfile:
    """Quantize one source-bound tile with inverse power-of-two operand scales.

    The emitted MXU consumes already encoded FP8 operands. This boundary
    policy records the selected pack-equivalent activation scale and the
    compensating constant-weight scale; it does not insert a VPU pack.
    """
    if type(scale_code) is not int or scale_code not in range(SCALED_PACK_RANGE[0], SCALED_PACK_RANGE[1] + 1):
        raise ValueError("source-bound MXU0 scale code is outside the selected pack range")
    base = diagnostic_mxu0_f32_profile()
    activation_shift = 127 - scale_code
    policies = (
        f"f32-to-e4m3-activation-e8m0-{scale_code}-v1",
        f"f32-to-e4m3-weight-compensation-e8m0-{scale_code}-v1",
    )
    conversions = tuple(
        FloatBoundarySpec(
            policy, "fp8_e4m3", "encode", max_normal_exponent_code=14,
            flush_below_normal=True, admitted_code_ranges=base.conversion_specs[0].admitted_code_ranges,
            pre_encode_scale_exponent=shift,
        )
        for policy, shift in zip(policies, (activation_shift, -activation_shift), strict=True)
    )
    return MatmulTileProfile(
        base.target_identity, base.loop_shape, base.maps,
        base.operand_dtypes, base.operand_policies, base.operand_domains,
        policies, base.result_dtype, base.result_policy,
        base.result_conversion_policy, base.storage, base.input_extents,
        base.required_policy_fields,
        (("mxu_unit", 0), ("activation_pack_scale_code", scale_code)),
        base.optional_policy_fields,
        (*conversions, base.conversion_specs[2]),
    )


def _finite_anchor_policy_profile(profile: MatmulTileProfile) -> MatmulTileProfile:
    """Bind one diagnostic tile to a separately identified precision policy."""
    fields = tuple(
        (key, "atlas.oot.quantized_linalg_policy.v4" if key == "schema" else value)
        for key, value in profile.required_policy_fields
    )
    return replace(profile, required_policy_fields=(
        *fields, ("mxu1_reset_arithmetic", FINITE_ANCHOR_POLICY),
    ))


def _canonicalized_zero_boundary_profile(profile: MatmulTileProfile, schema: str) -> MatmulTileProfile:
    """Keep an explicit policy identity for the lossy FP8 zero-sign choice."""
    if profile.conversion_specs is None:
        raise ValueError("positive-zero boundary needs executable operand conversions")
    inputs = tuple(replace(spec, policy=f"{spec.policy}-positive-zero",
                           canonicalize_negative_zero=True)
                   for spec in profile.conversion_specs[:2])
    required = tuple(
        (key, schema if key == "schema" else value)
        for key, value in profile.required_policy_fields
    )
    return replace(
        profile,
        operand_conversion_policies=tuple(spec.policy for spec in inputs),
        conversion_specs=(*inputs, profile.conversion_specs[2]),
        required_policy_fields=(*required, ("zero_sign_policy", "canonicalize_encoded_negative_zero")),
    )


def _positive_zero_boundary_profile(profile: MatmulTileProfile) -> MatmulTileProfile:
    if dict(profile.selected_decision).get("mxu_unit") != 0:
        raise ValueError("positive-zero boundary currently needs MXU0")
    return _canonicalized_zero_boundary_profile(profile, "atlas.oot.quantized_linalg_policy.v5")


def _mixed_positive_zero_boundary_profile(profile: MatmulTileProfile) -> MatmulTileProfile:
    if (
        dict(profile.selected_decision).get("mxu_unit") not in (0, 1)
        or dict(profile.required_policy_fields).get("mxu1_reset_arithmetic") != FINITE_ANCHOR_POLICY
    ):
        raise ValueError("mixed positive-zero boundary needs the finite-anchor MXU profile")
    return _canonicalized_zero_boundary_profile(profile, "atlas.oot.quantized_linalg_policy.v6")


def diagnostic_mxu1_finite_f32_profile(scale_code: int) -> MatmulTileProfile:
    """Describe source-derived finite-normal MXU1 reset in diagnostic mode.

    The policy is separate from the five-code MXU1 profile. It permits
    finite-normal E4M3 encodings and positive zero after the selected f32
    conversion, with the same inverse power-of-two weight compensation used
    by the explicit MXU0 scale policy.
    """
    if type(scale_code) is not int or not SCALED_PACK_RANGE[0] <= scale_code <= SCALED_PACK_RANGE[1]:
        raise ValueError("finite MXU1 scale code is outside the selected range")
    base = (diagnostic_mxu0_f32_profile() if scale_code == 127
            else diagnostic_mxu0_scaled_f32_profile(scale_code))
    return _finite_anchor_policy_profile(replace(
        base,
        operand_domains=(MXU1_FINITE_INPUT_DOMAIN, MXU1_FINITE_INPUT_DOMAIN),
        result_policy=MXU1_FINITE_POLICY,
        selected_decision=(("mxu_unit", 1), ("activation_pack_scale_code", scale_code)),
    ))


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _run(command: list[str], *, input_text: str) -> str:
    try:
        result = subprocess.run(command, input=input_text, text=True,
                                capture_output=True, timeout=30, check=False)
    except subprocess.TimeoutExpired as exc:
        raise NativeCompilationError("search_timeout", f"Atlas tool timed out: {exc}") from exc
    except OSError as exc:
        raise NativeCompilationError("tool_unavailable", f"Atlas tool failed: {exc}") from exc
    if result.returncode:
        raise NativeCompilationError(
            "compile_error", f"Atlas tool rejected selected program: {result.stderr.strip()}"
        )
    return result.stdout


def _virtual_from_selected(graph, order: tuple[int, ...], names: list[str | None],
                           source_ids: tuple[str, ...], input_slots: tuple[int, ...],
                           output_slots: tuple[int, ...], *, pack_scale_code: int = 127,
                           pack_scale_codes: dict[str, int] | None = None) -> str:
    """Materialize the selected instruction DAG through admitted Atlas virtual ops."""
    input_base = DRAM_BASE + input_slots[0] * TILE_BYTES
    output_base = DRAM_BASE + output_slots[0] * TILE_BYTES
    indexed_inputs = all(slot == input_slots[0] + index for index, slot in enumerate(input_slots))
    indexed_outputs = all(slot == output_slots[0] + index for index, slot in enumerate(output_slots))
    lines = ["module {",
             f"  func.func @native_tile() -> !atlas.virtual_state attributes {{atlas.input_dram_base = {input_base} : i64, atlas.output_dram_base = {output_base} : i64}} {{",
             '    %s0 = "atlas.virtual_start"() : () -> !atlas.virtual_state']
    values: dict[int, str] = {}
    state = 0
    fp8_store_size_declared = False
    for value_id in order:
        value = graph.value(value_id)
        name = names[value_id]
        if name in {"atlas.load_bf16_tile", "atlas.load_bf16_tile_mixed", "atlas.load_fp8_tile"}:
            if (len(value.children) != 1
                    or graph.value(value.children[0]).kind not in {"input", "constant"}):
                raise NativeCompilationError("unqualified_target", "selected tile load needs a DRAM boundary value")
            source = graph.value(value.children[0]).source_node
            if source not in source_ids:
                raise NativeCompilationError("unqualified_target", "selected input is absent from the DRAM ABI")
            source_index = source_ids.index(source)
            dtype = "bf16" if name in {"atlas.load_bf16_tile", "atlas.load_bf16_tile_mixed"} else "fp8"
            if indexed_inputs:
                lines.append(
                    f'    %s{state + 1}, %v{value_id} = "atlas.virtual_input_{dtype}"(%s{state}) '
                    f'{{index = {source_index} : i32}} : (!atlas.virtual_state) -> '
                    f'(!atlas.virtual_state, !atlas.virtual_{dtype})'
                )
                state += 1
            else:
                address = DRAM_BASE + input_slots[source_index] * TILE_BYTES - (1 << 32)
                size = 2048 if dtype == "bf16" else 1024
                lines.extend((
                    f"    %addr{value_id} = arith.constant {address} : i32",
                    f"    %bytes{value_id} = arith.constant {size} : i32",
                    (f'    %s{state + 1}, %v{value_id} = "atlas.virtual_dma_load_{dtype}"'
                     f'(%s{state}, %addr{value_id}, %bytes{value_id}) : '
                     f'(!atlas.virtual_state, i32, i32) -> '
                     f'(!atlas.virtual_state, !atlas.virtual_{dtype})'),
                ))
                state += 1
            values[value_id] = f"%v{value_id}"
            continue
        try:
            operands = tuple(values[child] for child in value.children)
        except KeyError as exc:
            raise NativeCompilationError("unqualified_target", "selected instruction order omits a producer") from exc
        if name in {"atlas.xlu_transpose_fp8", *TRANSPOSE_RULE_NAMES}:
            lines.append(
                f'    %v{value_id} = "atlas.virtual_xlu_transpose"({operands[0]}) '
                ': (!atlas.virtual_fp8) -> !atlas.virtual_fp8'
            )
            values[value_id] = f"%v{value_id}"
        elif name in {"atlas.vpu_relu_bf16_tile", "atlas.vpu_relu_bf16_from_mxu0_tile"}:
            lines.append(
                f'    %v{value_id} = "atlas.virtual_vpu_unary"({operands[0]}) '
                '{kind = "relu"} : (!atlas.virtual_bf16) -> !atlas.virtual_bf16'
            )
            values[value_id] = f"%v{value_id}"
        elif name in BF16_UNARY_MODES:
            kind = BF16_UNARY_MODES[name]
            lines.append(
                f'    %v{value_id} = "atlas.virtual_vpu_unary"({operands[0]}) '
                f'{{kind = "{kind}"}} : (!atlas.virtual_bf16) -> !atlas.virtual_bf16'
            )
            values[value_id] = f"%v{value_id}"
        elif name in BF16_ROW_REDUCTION_MODES:
            kind = BF16_ROW_REDUCTION_MODES[name]
            lines.append(
                f'    %v{value_id} = "atlas.virtual_vpu_reduce"({operands[0]}) '
                f'{{kind = "{kind}"}} : (!atlas.virtual_bf16) -> !atlas.virtual_bf16'
            )
            values[value_id] = f"%v{value_id}"
        elif name in BF16_BINARY_MODES:
            kind = BF16_BINARY_MODES[name]
            lines.append(
                f'    %v{value_id} = "atlas.virtual_vpu_binary"({operands[0]}, {operands[1]}) '
                f'{{kind = "{kind}"}} : (!atlas.virtual_bf16, !atlas.virtual_bf16) -> !atlas.virtual_bf16'
            )
            values[value_id] = f"%v{value_id}"
        elif name in PACK_RULE_NAMES:
            selected_code = (pack_scale_codes or {}).get(value.source_node, pack_scale_code)
            lines.append(
                f'    %v{value_id} = "atlas.virtual_pack_fp8"({operands[0]}) '
                f'{{scale_code = {selected_code} : i32}} : (!atlas.virtual_bf16) -> !atlas.virtual_fp8'
            )
            values[value_id] = f"%v{value_id}"
        elif name in {"atlas.mxu0_reset_tile", "atlas.mxu1_reset_tile",
                      "atlas.mxu1_finite_reset_tile"}:
            unit = 0 if name == "atlas.mxu0_reset_tile" else 1
            lines.extend((
                f'    %s{state + 1}, %weight{value_id} = "atlas.virtual_mxu_load_weight"(%s{state}, {operands[1]}) '
                + f'{{unit = {unit} : i32}} : (!atlas.virtual_state, !atlas.virtual_fp8) -> '
                + f'(!atlas.virtual_state, !atlas.virtual_mxu_weight<{unit}>)',
                f'    %s{state + 2}, %acc{value_id} = "atlas.virtual_mxu_reset"(%s{state + 1}, {operands[0]}, %weight{value_id}) '
                + f': (!atlas.virtual_state, !atlas.virtual_fp8, !atlas.virtual_mxu_weight<{unit}>) -> '
                + f'(!atlas.virtual_state, !atlas.virtual_mxu_acc<{unit}>)',
                f'    %s{state + 3}, %v{value_id} = "atlas.virtual_mxu_readout_bf16"(%s{state + 2}, %acc{value_id}) '
                + f': (!atlas.virtual_state, !atlas.virtual_mxu_acc<{unit}>) -> '
                + '(!atlas.virtual_state, !atlas.virtual_bf16)',
            ))
            state += 3
            values[value_id] = f"%v{value_id}"
        elif name == "atlas.mxu0_reset_acc_tile":
            lines.extend((
                f'    %s{state + 1}, %weight{value_id} = "atlas.virtual_mxu_load_weight"(%s{state}, {operands[1]}) '
                + '{unit = 0 : i32} : (!atlas.virtual_state, !atlas.virtual_fp8) -> '
                + '(!atlas.virtual_state, !atlas.virtual_mxu_weight<0>)',
                f'    %s{state + 2}, %acc{value_id} = "atlas.virtual_mxu_reset"(%s{state + 1}, {operands[0]}, %weight{value_id}) '
                + ': (!atlas.virtual_state, !atlas.virtual_fp8, !atlas.virtual_mxu_weight<0>) -> '
                + '(!atlas.virtual_state, !atlas.virtual_mxu_acc<0>)',
            ))
            state += 2
            values[value_id] = f"%acc{value_id}"
        elif name == "atlas.mxu0_accumulate_tile":
            lines.extend((
                f'    %s{state + 1}, %weight{value_id} = "atlas.virtual_mxu_load_weight"(%s{state}, {operands[1]}) '
                + '{unit = 0 : i32} : (!atlas.virtual_state, !atlas.virtual_fp8) -> '
                + '(!atlas.virtual_state, !atlas.virtual_mxu_weight<0>)',
                f'    %s{state + 2}, %acc{value_id} = "atlas.virtual_mxu_accumulate"'
                + f'(%s{state + 1}, {operands[0]}, %weight{value_id}, {operands[2]}) '
                + ': (!atlas.virtual_state, !atlas.virtual_fp8, !atlas.virtual_mxu_weight<0>, '
                + '!atlas.virtual_mxu_acc<0>) -> (!atlas.virtual_state, !atlas.virtual_mxu_acc<0>)',
            ))
            state += 2
            values[value_id] = f"%acc{value_id}"
        elif name in {"atlas.store_bf16_tile", "atlas.store_bf16_mxu0_tile",
                      "atlas.store_bf16_mxu1_tile", "atlas.store_bf16_mxu1_finite_tile",
                      "atlas.store_bf16_mxu0_acc_tile"}:
            if value_id not in graph.outputs:
                raise NativeCompilationError("unqualified_target", "selected BF16 store is not an output root")
            if name == "atlas.store_bf16_mxu0_acc_tile":
                lines.append(
                    f'    %s{state + 1}, %readout{value_id} = "atlas.virtual_mxu_readout_bf16"'
                    f'(%s{state}, {operands[0]}) : (!atlas.virtual_state, !atlas.virtual_mxu_acc<0>) '
                    '-> (!atlas.virtual_state, !atlas.virtual_bf16)'
                )
                state += 1
                operands = (f"%readout{value_id}",)
            output_index = graph.outputs.index(value_id)
            if indexed_outputs:
                lines.append(
                    f'    %s{state + 1} = "atlas.virtual_output_bf16"(%s{state}, {operands[0]}) '
                    f'{{index = {output_index} : i32}} : (!atlas.virtual_state, !atlas.virtual_bf16) -> !atlas.virtual_state'
                )
                state += 1
            else:
                address = DRAM_BASE + output_slots[output_index] * TILE_BYTES - (1 << 32)
                lines.extend((
                    f"    %out{value_id} = arith.constant {address} : i32",
                    f"    %outbytes{value_id} = arith.constant 2048 : i32",
                    (f'    %s{state + 1} = "atlas.virtual_dma_store_bf16"'
                     f'(%s{state}, {operands[0]}, %out{value_id}, %outbytes{value_id}) : '
                     f'(!atlas.virtual_state, !atlas.virtual_bf16, i32, i32) -> !atlas.virtual_state'),
                ))
                state += 1
            values[value_id] = operands[0]
        elif name in {"atlas.store_fp8_tile", *STORE_RULE_NAMES}:
            if value_id not in graph.outputs:
                raise NativeCompilationError("unqualified_target", "selected FP8 store is not an output root")
            output_index = graph.outputs.index(value_id)
            address = DRAM_BASE + output_slots[output_index] * TILE_BYTES - (1 << 32)
            lines.append(f"    %out{value_id} = arith.constant {address} : i32")
            if not fp8_store_size_declared:
                lines.append("    %fp8_store_bytes = arith.constant 1024 : i32")
                fp8_store_size_declared = True
            lines.extend((
                f'    %s{state + 1} = "atlas.virtual_dma_store_fp8"(%s{state}, {operands[0]}, %out{value_id}, %fp8_store_bytes) '
                + ': (!atlas.virtual_state, !atlas.virtual_fp8, i32, i32) -> '
                + '!atlas.virtual_state',
            ))
            state += 1
            values[value_id] = operands[0]
        else:
            raise NativeCompilationError("unsupported_semantics", f"no Atlas virtual emission for selected {name}")
    lines.extend((f"    return %s{state} : !atlas.virtual_state", "  }", "}"))
    return "\n".join(lines) + "\n"


def _tile_effects(program: dict) -> list[tuple]:
    """Replay scalar addresses and observable operations from structured words.

    This check reads the emitted stream, independently of virtual placement and
    of the OOT schedule verifier. It admits only the diagnostic tile families.
    """
    allowed = {
        "atlas.dma_config", "atlas.alu_imm", "atlas.upper", "atlas.dma",
        "atlas.dma_wait", "atlas.vload", "atlas.vstore", "atlas.delay",
        "atlas.xlu_transpose", "atlas.vpu_unary", "atlas.vpu_binary", "atlas.vpu_reduce",
        "atlas.mxu_push", "atlas.mxu_matmul",
        "atlas.mxu_pop", "atlas.csr", "atlas.trap", "atlas.vpu_pack",
        "atlas.scalar_load", "atlas.scalar_store", "atlas.branch",
    }
    registers = {0: 0}
    effects: list[tuple] = []
    rows = program.get("instructions")
    if not isinstance(rows, list):
        raise NativeCompilationError("compile_error", "Atlas structured stream has no instructions")
    for index, row in enumerate(rows):
        op = row.get("operation")
        fields = row.get("fields")
        if op not in allowed or not isinstance(fields, dict):
            raise NativeCompilationError("unqualified_target", "tile stream contains an unmodeled instruction")
        if op == "atlas.dma_config":
            if index != 0 or fields.get("base_reg") != 0 or fields.get("channel") != 0:
                raise NativeCompilationError("unqualified_target", "movement DMA base differs from zero")
            if row.get("word_u32") != 0x7F:
                raise NativeCompilationError("unqualified_target", "movement DMA_CONFIG word differs from selected RTL")
            effects.append(("config",))
        elif op == "atlas.upper":
            if fields.get("kind") != "lui" or fields.get("dst") == 0:
                raise NativeCompilationError("unqualified_target", "movement stream has an unmodeled upper operation")
            registers[fields["dst"]] = (fields["immediate"] << 12) & 0xffffffff
        elif op == "atlas.alu_imm":
            if fields.get("kind") != "addi" or fields.get("src") not in registers or (
                fields.get("dst") == 0 and (fields.get("src"), fields.get("immediate")) != (0, 0)
            ):
                raise NativeCompilationError("unqualified_target", "movement stream has an unmodeled scalar address")
            if fields["dst"] != 0:
                registers[fields["dst"]] = (registers[fields["src"]] + fields["immediate"]) & 0xffffffff
        elif op == "atlas.dma":
            if any(fields.get(key) not in registers for key in ("dram", "reg", "size")):
                raise NativeCompilationError("unqualified_target", "movement DMA reads an unknown scalar address")
            effects.append(("dma", fields.get("direction"),
                            registers[fields["dram"]], registers[fields["reg"]],
                            registers[fields["size"]], fields.get("channel")))
        elif op == "atlas.dma_wait":
            channel = fields.get("channel")
            if type(channel) is not int or not 0 <= channel < 8 or (
                row.get("word_u32") != ((1 << 25) | (channel << 12) | 0x7F)
            ):
                raise NativeCompilationError("unqualified_target", "movement DMA_WAIT word differs from selected RTL")
            effects.append(("wait", fields.get("channel")))
        elif op == "atlas.vload":
            if fields.get("format") != "raw" or fields.get("offset") != 0 or (
                fields.get("base") not in registers
            ):
                raise NativeCompilationError("unqualified_target", "movement load changes bits or scratch address")
            effects.append(("vload", fields["dst"], registers[fields["base"]]))
        elif op == "atlas.vstore":
            if fields.get("format") != "raw" or fields.get("offset") != 0 or (
                fields.get("base") not in registers
            ):
                raise NativeCompilationError("unqualified_target", "movement store changes bits or scratch address")
            effects.append(("vstore", fields["src"], registers[fields["base"]]))
        elif op == "atlas.xlu_transpose":
            effects.append(("transpose", fields.get("src"), fields.get("dst")))
        elif op == "atlas.vpu_unary":
            if fields.get("kind") not in {"relu", *BF16_UNARY_MODES.values()}:
                raise NativeCompilationError("unqualified_target", "BF16 tile has an unadmitted VPU mode")
            effects.append((fields["kind"], fields.get("src"), fields.get("dst")))
        elif op == "atlas.vpu_reduce":
            if fields.get("kind") not in BF16_ROW_REDUCTION_MODES.values():
                raise NativeCompilationError("unqualified_target", "BF16 tile has an unadmitted reduction mode")
            effects.append((fields["kind"], fields.get("src"), fields.get("dst")))
        elif op == "atlas.vpu_binary":
            if fields.get("kind") not in BF16_BINARY_MODES.values():
                raise NativeCompilationError("unqualified_target", "BF16 tile has an unadmitted binary VPU mode")
            effects.append((fields["kind"], fields.get("lhs"), fields.get("rhs"), fields.get("dst")))
        elif op == "atlas.vpu_pack":
            effects.append(("pack", fields.get("direction"), fields.get("src"),
                            fields.get("dst"), fields.get("scale_reg")))
        elif op == "atlas.scalar_load":
            if fields.get("kind") == "seli":
                if fields.get("base") != 0 or not 0 <= fields.get("offset", -1) <= 255:
                    raise NativeCompilationError("unqualified_target", "pack scale load is invalid")
                effects.append(("seli", fields.get("dst"), fields.get("offset")))
            elif fields.get("kind") == "lw":
                if fields.get("base") not in registers or type(fields.get("offset")) is not int:
                    raise NativeCompilationError("unqualified_target", "pack loop reads an unknown address")
                effects.append(("lw", fields.get("dst"),
                                registers[fields["base"]] + fields["offset"]))
            else:
                raise NativeCompilationError("unqualified_target", "pack stream has an unadmitted scalar load")
        elif op == "atlas.scalar_store":
            if fields.get("kind") != "sw" or fields.get("base") not in registers or (
                type(fields.get("offset")) is not int
            ):
                raise NativeCompilationError("unqualified_target", "pack loop has an unknown store address")
            effects.append(("sw", fields.get("src"),
                            registers[fields["base"]] + fields["offset"]))
        elif op == "atlas.branch":
            if fields.get("kind") != "blt" or any(
                register not in registers for register in (fields.get("lhs"), fields.get("rhs"))
            ):
                raise NativeCompilationError("unqualified_target", "pack loop has an unknown branch")
            effects.append(("blt", fields.get("lhs"), fields.get("rhs"),
                            registers[fields["lhs"]], registers[fields["rhs"]],
                            fields.get("offset_bytes")))
        elif op == "atlas.mxu_push":
            effects.append(("mxu_push", fields.get("unit"), fields.get("slot"),
                            fields.get("kind"), fields.get("src")))
        elif op == "atlas.mxu_matmul":
            effects.append(("mxu_matmul", fields.get("unit"), fields.get("acc_slot"),
                            fields.get("weight_slot"), fields.get("accumulate"),
                            fields.get("src")))
        elif op == "atlas.mxu_pop":
            effects.append(("mxu_pop", fields.get("unit"), fields.get("slot"),
                            fields.get("format"), fields.get("scale_reg"),
                            fields.get("dst")))
        elif op == "atlas.csr":
            if (fields.get("kind"), fields.get("address"), fields.get("dst")) != (
                "rrw", 0xC10, 0
            ) or registers.get(fields.get("source")) != 1:
                raise NativeCompilationError("unqualified_target", "movement completion marker changed")
            effects.append(("complete",))
        elif op == "atlas.trap":
            if index != len(rows) - 1 or fields.get("kind") != "ecall":
                raise NativeCompilationError("unqualified_target", "movement stream has a nonterminal trap")
            effects.append(("trap",))
    return effects


def _check_atomic_dma_schedule(observed: list[tuple], expected: list[tuple], *, context: str) -> None:
    """Replay a physical DMA schedule against logical transfers and scratch contents."""
    def transfer(effect: tuple) -> tuple:
        return effect[1], effect[2], effect[4]

    def non_dma(effect: tuple) -> tuple:
        return effect[:2] if effect[0] in {"vload", "vstore"} else effect

    observed_dma = Counter(map(transfer, (e for e in observed if e[0] == "dma")))
    expected_dma = Counter(map(transfer, (e for e in expected if e[0] == "dma")))
    observed_order = [non_dma(e) for e in observed if e[0] not in {"dma", "wait"}]
    expected_order = [non_dma(e) for e in expected if e[0] not in {"dma", "wait"}]
    if observed_dma != expected_dma:
        raise NativeCompilationError(
            "unqualified_target", f"{context} DMA transfers differ: "
            f"missing={list((expected_dma - observed_dma).elements())!r}, "
            f"extra={list((observed_dma - expected_dma).elements())!r}"
        )
    if observed_order != expected_order:
        index = next(i for i in range(max(len(observed_order), len(expected_order)))
                     if observed_order[i:i + 1] != expected_order[i:i + 1])
        raise NativeCompilationError(
            "unqualified_target", f"{context} non-DMA effect {index} differs: "
            f"expected={expected_order[index:index + 1]!r}, observed={observed_order[index:index + 1]!r}"
        )

    def overlaps(a: tuple[int, int], b: tuple[int, int]) -> bool:
        return a[0] < b[1] and b[0] < a[1]

    def chunks(word: int, size: int) -> tuple[int, ...]:
        if type(word) is not int or type(size) is not int or size not in {1024, 2048} or word % 256:
            raise NativeCompilationError("unqualified_target", f"{context} has misaligned scratch access")
        return tuple(word * 4 + offset for offset in range(0, size, 1024))

    # The expected stream assigns a source to every VLOAD and a register to
    # every output DMA. The allocator may choose different scratch words and
    # channels, but it cannot change these data identities.
    expected_loads: list[int] = []
    expected_stores: dict[tuple[int, int], list[tuple[int, ...]]] = {}
    for index, effect in enumerate(expected):
        if effect[0] == "vload":
            source = next((old for old in reversed(expected[:index])
                           if old[0:2] == ("dma", "load")
                           and old[3] <= effect[2] < old[3] + old[4] // 4), None)
            if source is None:
                raise NativeCompilationError("unqualified_target", f"{context} has an unbound expected input")
            expected_loads.append(source[2] + (effect[2] - source[3]) * 4)
        elif effect[0:2] == ("dma", "store"):
            registers = []
            for word in chunks(effect[3], effect[4]):
                source = next((old for old in reversed(expected[:index])
                               if old[0] == "vstore" and old[2] * 4 == word), None)
                if source is None:
                    raise NativeCompilationError("unqualified_target", f"{context} has an unbound expected output")
                registers.append(source[1])
            expected_stores.setdefault((effect[2], effect[4]), []).append(tuple(registers))

    pending: dict[int, list[tuple[str, int, tuple[int, ...]]]] = {}
    scratch: dict[int, tuple[str, int]] = {}
    load_index = 0
    for effect in observed:
        kind = effect[0]
        if kind == "dma":
            direction, dram, word, size, channel = effect[1:]
            if (direction not in {"load", "store"} or type(dram) is not int
                    or type(channel) is not int
                    or not 0 <= channel < 8):
                raise NativeCompilationError("unqualified_target", f"{context} has invalid DMA")
            words = chunks(word, size)
            span = (words[0], words[-1] + 1024)
            if any(overlaps(span, active) for items in pending.values()
                   for _, _, active_words in items
                   for active in ((active_words[0], active_words[-1] + 1024),)):
                raise NativeCompilationError("unqualified_target", f"{context} reuses scratch before DMA completion")
            if direction == "store":
                planned = expected_stores.get((dram, size), [])
                if not planned or tuple(scratch.get(addr) for addr in words) != tuple(
                    ("register", reg) for reg in planned.pop(0)
                ):
                    raise NativeCompilationError("unqualified_target", f"{context} DMA reads wrong output data")
            pending.setdefault(channel, []).append((direction, dram, words))
        elif kind == "wait":
            channel = effect[1]
            if type(channel) is not int or not 0 <= channel < 8:
                raise NativeCompilationError("unqualified_target", f"{context} has invalid DMA wait")
            for direction, dram, words in pending.pop(channel, []):
                if direction == "load":
                    for offset, addr in enumerate(words):
                        scratch[addr] = ("dram", dram + offset * 1024)
        elif kind in {"vload", "vstore"}:
            word = effect[2]
            addr = chunks(word, 1024)[0]
            span = (addr, addr + 1024)
            if any(overlaps(span, active) for items in pending.values()
                   for _, _, active_words in items
                   for active in ((active_words[0], active_words[-1] + 1024),)):
                raise NativeCompilationError("unqualified_target", f"{context} accesses in-flight DMA scratch")
            if kind == "vload":
                if load_index >= len(expected_loads) or scratch.get(addr) != ("dram", expected_loads[load_index]):
                    raise NativeCompilationError("unqualified_target", f"{context} VLOAD reads wrong input data")
                load_index += 1
            else:
                scratch[addr] = ("register", effect[1])
        elif kind in {"complete", "trap"} and pending:
            raise NativeCompilationError("unqualified_target", f"{context} ends with DMA in flight")
    if pending or load_index != len(expected_loads) or any(expected_stores.values()):
        raise NativeCompilationError("unqualified_target", f"{context} leaves an incomplete transfer")


def check_copy_stream(program: dict, *, input_address: int,
                      output_address: int, pair: int) -> None:
    effects = _tile_effects(program)
    expected = [
        ("config",),
        ("dma", "load", input_address, 0, 1024, 0),
        ("wait", 0), ("vload", pair, 0),
        ("dma", "load", input_address + 1024, 256, 1024, 0),
        ("wait", 0), ("vload", pair + 1, 256),
        ("vstore", pair, 0x10000),
        ("dma", "store", output_address, 0x10000, 1024, 1),
        ("wait", 1),
        ("vstore", pair + 1, 0x10100),
        ("dma", "store", output_address + 1024, 0x10100, 1024, 1),
        ("wait", 1), ("complete",), ("trap",),
    ]
    if effects != expected:
        raise NativeCompilationError(
            "unqualified_target", "emitted copy addresses, scratch words, order, waits, or pair differ from selection"
        )


def check_copy_twice_stream(program: dict, *, input_address: int,
                            output_addresses: tuple[int, int], pair: int) -> None:
    effects = _tile_effects(program)
    expected = [
        ("config",),
        ("dma", "load", input_address, 0, 1024, 0),
        ("wait", 0), ("vload", pair, 0),
        ("dma", "load", input_address + 1024, 256, 1024, 0),
        ("wait", 0), ("vload", pair + 1, 256),
    ]
    for index, output_address in enumerate(output_addresses):
        scratch = 0x10000 + index * 0x200
        expected.extend((
            ("vstore", pair, scratch),
            ("dma", "store", output_address, scratch, 1024, 1),
            ("wait", 1),
            ("vstore", pair + 1, scratch + 256),
            ("dma", "store", output_address + 1024, scratch + 256, 1024, 1),
            ("wait", 1),
        ))
    expected.extend((("complete",), ("trap",)))
    if effects != expected:
        raise NativeCompilationError(
            "unqualified_target", "emitted two-output copy differs in sharing, addresses, order, or waits"
        )


def _check_bf16_selection(graph: CandidateGraph, order: tuple[int, ...],
                          names: list[str | None], assignment: dict[int, int],
                          source_ids: tuple[str, ...], input_slots: tuple[int, ...],
                          output_slots: tuple[int, ...]) -> list[int]:
    """Check selected BF16 graph shape and bindings before OOT materialization."""
    instructions = {value.id for value in graph.values if value.kind == "instruction"}
    if set(order) != instructions or len(order) != len(instructions) or (
        len(graph.outputs) != len(output_slots) or len(set(graph.outputs)) != len(graph.outputs)
    ):
        raise NativeCompilationError("unqualified_target", "selected BF16 order or outputs changed")
    output_indices = {value_id: index for index, value_id in enumerate(graph.outputs)}
    registers: set[int] = set()
    for value in graph.values:
        if value.kind == "input":
            if value.source_node not in source_ids or value.storage != "external" or (
                assignment[value.id] != input_slots[source_ids.index(value.source_node)]
            ):
                raise NativeCompilationError("unqualified_target", "selected BF16 input binding changed")
        elif names[value.id] == "atlas.load_bf16_tile":
            if len(value.children) != 1 or graph.value(value.children[0]).kind != "input" or (
                value.storage != "tensor_bf16"
            ):
                raise NativeCompilationError("unqualified_target", "selected BF16 load has invalid source")
            pair = assignment[value.id]
            if pair % 2 or not 0 <= pair <= 60:
                raise NativeCompilationError("unqualified_target", "selected BF16 pair is invalid")
            registers.update((pair, pair + 1))
        elif names[value.id] in {"atlas.vpu_relu_bf16_tile", *BF16_UNARY_MODES}:
            if len(value.children) != 1 or names[value.children[0]] not in {
                "atlas.load_bf16_tile", "atlas.vpu_relu_bf16_tile", *BF16_UNARY_MODES,
                *BF16_ROW_REDUCTION_MODES, *BF16_BINARY_MODES,
            } or value.storage != "tensor_bf16":
                raise NativeCompilationError("unqualified_target", "selected BF16 unary has invalid source")
            pair = assignment[value.id]
            if pair % 2 or not 0 <= pair <= 60 or pair == assignment[value.children[0]]:
                raise NativeCompilationError("unqualified_target", "selected BF16 unary pair is invalid")
            registers.update((pair, pair + 1))
        elif names[value.id] in BF16_ROW_REDUCTION_MODES:
            if (len(value.children) != 1 or names[value.children[0]] not in {
                "atlas.load_bf16_tile", "atlas.vpu_relu_bf16_tile", *BF16_UNARY_MODES,
                *BF16_ROW_REDUCTION_MODES, *BF16_BINARY_MODES,
            } or value.storage != "tensor_bf16"):
                raise NativeCompilationError("unqualified_target", "selected BF16 row reduction has invalid source")
            pair = assignment[value.id]
            if pair % 2 or not 0 <= pair <= 60 or pair == assignment[value.children[0]]:
                raise NativeCompilationError("unqualified_target", "selected BF16 row reduction pair is invalid")
            registers.update((pair, pair + 1))
        elif names[value.id] in BF16_BINARY_MODES:
            sources = {"atlas.load_bf16_tile", "atlas.vpu_relu_bf16_tile", *BF16_UNARY_MODES,
                       *BF16_ROW_REDUCTION_MODES, *BF16_BINARY_MODES}
            if len(value.children) != 2 or any(names[child] not in sources
                                               for child in value.children) or (
                value.children[0] == value.children[1] or value.storage != "tensor_bf16"
            ):
                raise NativeCompilationError("unqualified_target", "selected BF16 binary op has invalid sources")
            pair = assignment[value.id]
            sources = tuple(assignment[child] for child in value.children)
            if pair % 2 or not 0 <= pair <= 60 or pair in sources or sources[0] == sources[1]:
                raise NativeCompilationError("unqualified_target", "selected BF16 binary op pairs overlap")
            registers.update((pair, pair + 1))
        elif names[value.id] == "atlas.store_bf16_tile":
            if value.id not in output_indices or len(value.children) != 1 or (
                names[value.children[0]] not in {
                    "atlas.load_bf16_tile", "atlas.vpu_relu_bf16_tile", *BF16_UNARY_MODES,
                    *BF16_ROW_REDUCTION_MODES, *BF16_BINARY_MODES,
                }
                or value.storage != "external"
                or assignment[value.id] != output_slots[output_indices[value.id]]
            ):
                raise NativeCompilationError("unqualified_target", "selected BF16 output binding changed")
        else:
            raise NativeCompilationError("unqualified_target", "selected BF16 graph has an unadmitted instruction")
    return sorted(registers)


def _realized_bf16_assignment(program: dict, *, graph: CandidateGraph,
                              order: tuple[int, ...], names: list[str | None],
                              selected: dict[int, int]) -> dict[int, int]:
    """Recover tensor pairs from the structured physical stream for replay."""
    effects = _tile_effects(program)
    loads = [effect for effect in effects if effect[0] == "vload"]
    binaries = [effect for effect in effects if effect[0] in BF16_BINARY_MODES.values()]
    unaries = [effect for effect in effects if effect[0] in {"relu", *BF16_UNARY_MODES.values()}]
    reductions = [effect for effect in effects if effect[0] in BF16_ROW_REDUCTION_MODES.values()]
    expected_loads = sum(names[value_id] == "atlas.load_bf16_tile" for value_id in order)
    expected_binaries = sum(names[value_id] in BF16_BINARY_MODES for value_id in order)
    expected_unaries = sum(names[value_id] in {"atlas.vpu_relu_bf16_tile", *BF16_UNARY_MODES}
                           for value_id in order)
    expected_reductions = sum(names[value_id] in BF16_ROW_REDUCTION_MODES for value_id in order)
    if (len(loads), len(binaries), len(unaries), len(reductions)) != (
        2 * expected_loads, expected_binaries, expected_unaries, expected_reductions
    ):
        raise NativeCompilationError("unqualified_target", "emitted BF16 instruction count differs from selection")
    realized = dict(selected)
    load_index = binary_index = unary_index = reduction_index = 0
    for value_id in order:
        value = graph.value(value_id)
        name = names[value_id]
        if name == "atlas.load_bf16_tile":
            first, second = loads[load_index:load_index + 2]
            load_index += 2
            if type(first[1]) is not int or second[1] != first[1] + 1:
                raise NativeCompilationError("unqualified_target", "emitted BF16 load has no adjacent register pair")
            realized[value_id] = first[1]
        elif name in BF16_BINARY_MODES:
            effect = binaries[binary_index]
            binary_index += 1
            kind = BF16_BINARY_MODES[name]
            if effect[0] != kind:
                raise NativeCompilationError("unqualified_target", "emitted BF16 binary mode differs from selection")
            if (effect[1], effect[2]) != tuple(realized[child] for child in value.children):
                raise NativeCompilationError("unqualified_target", f"emitted BF16 {kind} reads different source pairs")
            realized[value_id] = effect[3]
        elif name in {"atlas.vpu_relu_bf16_tile", *BF16_UNARY_MODES}:
            effect = unaries[unary_index]
            unary_index += 1
            kind = "relu" if name == "atlas.vpu_relu_bf16_tile" else BF16_UNARY_MODES[name]
            if effect[0] != kind or effect[1] != realized[value.children[0]]:
                raise NativeCompilationError("unqualified_target", "emitted BF16 unary reads a different source pair")
            realized[value_id] = effect[2]
        elif name in BF16_ROW_REDUCTION_MODES:
            effect = reductions[reduction_index]
            reduction_index += 1
            if (effect[0] != BF16_ROW_REDUCTION_MODES[name]
                    or effect[1] != realized[value.children[0]]):
                raise NativeCompilationError("unqualified_target", "emitted BF16 row reduction changed")
            realized[value_id] = effect[2]
    return realized


def check_bf16_tile_stream(program: dict, *, graph: CandidateGraph,
                           order: tuple[int, ...], names: list[str | None],
                           assignment: dict[int, int], source_ids: tuple[str, ...],
                           input_slots: tuple[int, ...],
                           output_slots: tuple[int, ...]) -> None:
    """Replay BF16 loads, arithmetic, and stores from the selected DAG and fixed ABI."""
    expected: list[tuple] = [("config",)]
    output_indices = {value_id: index for index, value_id in enumerate(graph.outputs)}
    indexed_inputs = all(slot == input_slots[0] + index for index, slot in enumerate(input_slots))
    indexed_outputs = all(slot == output_slots[0] + index for index, slot in enumerate(output_slots))
    for value_id in order:
        value = graph.value(value_id)
        name = names[value_id]
        if name == "atlas.load_bf16_tile":
            source = graph.value(value.children[0]).source_node
            source_index = source_ids.index(source)
            address = DRAM_BASE + input_slots[source_index] * TILE_BYTES
            pair = assignment[value_id]
            if indexed_inputs:
                scratch = source_index * 512
                for half in (0, 1):
                    expected.extend((
                        ("dma", "load", address + half * 1024, scratch + half * 256, 1024, 0),
                        ("wait", 0), ("vload", pair + half, scratch + half * 256),
                    ))
            else:
                expected.extend((
                    ("dma", "load", address, 0x20000, 2048, 0),
                    ("wait", 0),
                    ("vload", pair, 0x20000),
                    ("vload", pair + 1, 0x20100),
                ))
        elif name in {"atlas.vpu_relu_bf16_tile", *BF16_UNARY_MODES}:
            kind = "relu" if name == "atlas.vpu_relu_bf16_tile" else BF16_UNARY_MODES[name]
            expected.append((kind, assignment[value.children[0]], assignment[value_id]))
        elif name in BF16_ROW_REDUCTION_MODES:
            expected.append((BF16_ROW_REDUCTION_MODES[name],
                             assignment[value.children[0]], assignment[value_id]))
        elif name in BF16_BINARY_MODES:
            expected.append((BF16_BINARY_MODES[name], assignment[value.children[0]],
                             assignment[value.children[1]], assignment[value_id]))
        elif name == "atlas.store_bf16_tile":
            output_index = output_indices[value_id]
            address = DRAM_BASE + output_slots[output_index] * TILE_BYTES
            pair = assignment[value.children[0]]
            if indexed_outputs:
                scratch = 0x10000 + output_index * 512
                for half in (0, 1):
                    expected.extend((
                        ("vstore", pair + half, scratch + half * 256),
                        ("dma", "store", address + half * 1024, scratch + half * 256, 1024, 1),
                        ("wait", 1),
                    ))
            else:
                expected.extend((
                    ("vstore", pair, 0x20000),
                    ("vstore", pair + 1, 0x20100),
                    ("dma", "store", address, 0x20000, 2048, 1),
                    ("wait", 1),
                ))
        else:
            raise NativeCompilationError("unqualified_target", "BF16 stream checker has an unknown selection")
    expected.extend((("complete",), ("trap",)))
    observed = _tile_effects(program)
    _check_atomic_dma_schedule(observed, expected, context="BF16 stream")
    cycle = 0
    ready_at: dict[int, int] = {}
    pending_stores: list[int] = []
    for row in program["instructions"]:
        op, fields = row["operation"], row["fields"]
        if op == "atlas.vload":
            ready_at[fields["dst"]] = cycle + 35
        elif op == "atlas.vpu_unary":
            source = fields["src"]
            if any(ready_at.get(source + half, 1 << 60) > cycle for half in (0, 1)):
                raise NativeCompilationError("unqualified_target", "BF16 unary reads an unavailable pair")
            ready_at[fields["dst"]] = cycle + 66
            ready_at[fields["dst"] + 1] = cycle + 66
        elif op == "atlas.vpu_reduce":
            source = fields["src"]
            if any(ready_at.get(source + half, 1 << 60) > cycle for half in (0, 1)):
                raise NativeCompilationError("unqualified_target", "BF16 row reduction reads an unavailable pair")
            # The selected row reductions issue across 32 rows. The last
            # row maximum writes at age 33; row sum writes at age 38.
            completion = 39 if fields["kind"] == "row_sum" else 34
            ready_at[fields["dst"]] = cycle + completion
            ready_at[fields["dst"] + 1] = cycle + completion
        elif op == "atlas.vpu_binary":
            for source in (fields["lhs"], fields["rhs"]):
                if any(ready_at.get(source + half, 1 << 60) > cycle for half in (0, 1)):
                    raise NativeCompilationError("unqualified_target", "BF16 binary op reads an unavailable pair")
            ready_at[fields["dst"]] = cycle + 66
            ready_at[fields["dst"] + 1] = cycle + 66
        elif op == "atlas.vstore":
            source = fields["src"]
            if ready_at.get(source, 1 << 60) > cycle:
                raise NativeCompilationError("unqualified_target", "BF16 store reads an unavailable register")
            pending_stores.append(cycle)
        elif op == "atlas.dma" and fields["direction"] == "store":
            # Indexed outputs transfer each half separately. An explicitly
            # addressed BF16 DMA transfers both halves after two VSTOREs.
            count = 1 if indexed_outputs else 2
            if len(pending_stores) < count or any(cycle - issued < 35 for issued in pending_stores[:count]):
                raise NativeCompilationError("unqualified_target", "BF16 DMA reads scratch before store completes")
            del pending_stores[:count]
        elif op == "atlas.delay":
            count = fields.get("cycles")
            if type(count) is not int or not 0 <= count <= 65535:
                raise NativeCompilationError("unqualified_target", "BF16 stream has invalid delay")
            cycle += count
        cycle += 1
    if pending_stores:
        raise NativeCompilationError("unqualified_target", "BF16 scratch store has no DMA completion")


def _pack_relayout_effects() -> list[tuple]:
    """Current selected scalar copy: two live temporaries overlap load/store."""
    def load(copy: int) -> tuple:
        return ("lw", 16 + copy % 2, 131072 + (copy % 2) * 512 + (copy // 2) * 4)

    def store(copy: int) -> tuple:
        return ("sw", 16 + copy % 2, 132096 + (copy % 2) * 16 + (copy // 2) * 4)

    effects = [load(0)]
    for copy in range(1, 8):
        effects.extend((load(copy), store(copy - 1)))
    effects.extend((store(7), ("blt", 13, 14, 1, 32, -54)))
    return effects


def check_pack_stream(program: dict, *, input_address: int, output_address: int,
                      source_pair: int = 32, relu_pair: tuple[int, int] | None = None,
                      scale_code: int = 127) -> None:
    """Replay the selected pack scale, row relayout, and output transfer.

    Scalar addresses use bytes; VLOAD/VSTORE bases use VMEM words. The loop
    structure is checked here because a linear list of memory effects cannot
    establish that every one of the 32 rows is copied.
    """
    expected: list[tuple] = [
        ("config",),
        ("dma", "load", input_address, 0, 1024, 0),
        ("wait", 0), ("vload", 32, 0),
        ("dma", "load", input_address + 1024, 256, 1024, 0),
        ("wait", 0), ("vload", 33, 256),
    ]
    if relu_pair is not None:
        expected.append(("relu", *relu_pair))
    if type(scale_code) is not int or not 0 <= scale_code <= 255:
        raise NativeCompilationError("unqualified_target", "pack scale code is outside the physical encoding")
    expected.extend((("seli", 3, scale_code), ("pack", "bf16_to_fp8", source_pair, 0, 3),
                     ("vstore", 0, 32768)))
    expected.extend(_pack_relayout_effects())
    expected.extend((
        ("vload", 0, 33024),
        ("vstore", 0, 131072),
        ("dma", "store", output_address, 131072, 1024, 1),
        ("wait", 1), ("complete",), ("trap",),
    ))
    if _tile_effects(program) != expected:
        raise NativeCompilationError("unqualified_target", "pack stream differs in layout, addresses, or transfer")
    rows = program["instructions"]
    pack_word = next(row["word_u32"] for row in rows if row["operation"] == "atlas.vpu_pack")
    if ((pack_word >> 25) & 0x7f, (pack_word >> 19) & 0x3f,
        (pack_word >> 13) & 0x3f, (pack_word >> 7) & 0x3f,
        pack_word & 0x7f) != (0x44, source_pair, 3, 0, 0x57):
        raise NativeCompilationError("unqualified_target", "pack word does not encode the selected six-bit source")
    branch = next(index for index, row in enumerate(rows) if row["operation"] == "atlas.branch")
    first_load = next(index for index, row in enumerate(rows) if row["operation"] == "atlas.scalar_load"
                      and row["fields"].get("kind") == "lw")
    # ScalarCore's PC is word-indexed. The RISC-V B immediate encodes halfword
    # units, so the selected emitter uses two, not four, per instruction word.
    if branch - first_load != 27 or rows[branch]["fields"]["offset_bytes"] != 2 * (first_load - branch) or (
        branch + 1 >= len(rows) or rows[branch + 1]["operation"] != "atlas.alu_imm"
        or any(rows[branch + 1]["fields"].get(key) != value for key, value in
               {"dst": 0, "immediate": 0, "kind": "addi", "src": 0}.items())
    ):
        raise NativeCompilationError("unqualified_target", "pack loop target or delay slot changed")
    branch_word = rows[branch]["word_u32"]
    immediate = (((branch_word >> 31) & 1) << 12) | (((branch_word >> 7) & 1) << 11) | (
        ((branch_word >> 25) & 0x3f) << 5) | (((branch_word >> 8) & 0xf) << 1)
    immediate = immediate - 8192 if immediate & 4096 else immediate
    if (branch_word & 0x7f, (branch_word >> 12) & 7,
        (branch_word >> 15) & 31, (branch_word >> 20) & 31,
        immediate) != (0x63, 4, 13, 14, -54):
        raise NativeCompilationError("unqualified_target", "pack loop word has a wrong branch target")
    # Require 32 iterations, 16-byte advancement in each source half and
    # 32-byte advancement in the destination. The one-iteration effects above
    # then cover 32*32 bytes without touching the source or adjacent slots.
    before = rows[first_load - 1]["fields"]
    if before.get("cycles", 0) < 0:
        raise NativeCompilationError("unqualified_target", "pack loop has invalid delay")
    advances = [row["fields"] for row in rows[first_load:branch]
                if row["operation"] == "atlas.alu_imm"]
    if [(row.get("kind"), row.get("dst"), row.get("src"), row.get("immediate"))
        for row in advances] != [
            ("addi", 13, 13, 1), ("addi", 10, 10, 16),
            ("addi", 11, 11, 16), ("addi", 12, 12, 32),
        ]:
        raise NativeCompilationError("unqualified_target", "pack loop does not advance every row")
    cycle = 0
    issued: dict[str, list[int]] = {}
    for row in rows:
        op, fields = row["operation"], row["fields"]
        if op == "atlas.delay":
            count = fields.get("cycles")
            if type(count) is not int or not 0 <= count <= 65535:
                raise NativeCompilationError("unqualified_target", "pack delay is invalid")
            cycle += count
        else:
            issued.setdefault(op, []).append(cycle)
        cycle += 1
    gaps = [
        ("atlas.vpu_pack", 0, "atlas.vstore", 0, 65),
        ("atlas.vstore", 0, "atlas.scalar_load", 1, 34),
        ("atlas.vload", 2, "atlas.vstore", 1, 34),
        ("atlas.vstore", 1, "atlas.dma", 2, 34),
    ]
    if relu_pair is None:
        gaps.append(("atlas.vload", 1, "atlas.vpu_pack", 0, 34))
    else:
        gaps.extend((("atlas.vload", 1, "atlas.vpu_unary", 0, 34),
                     ("atlas.vpu_unary", 0, "atlas.vpu_pack", 0, 65)))
    if any(issued[end][end_index] - issued[start][start_index] < minimum
           for start, start_index, end, end_index, minimum in gaps):
        raise NativeCompilationError("unqualified_target", "pack stream uses data before the diagnostic delay floor")


def check_transpose_stream(program: dict, *, input_address: int,
                           output_address: int, source: int, result: int) -> None:
    effects = _tile_effects(program)
    expected = [
        ("config",),
        ("dma", "load", input_address, 0, 1024, 0),
        ("wait", 0), ("vload", source, 0),
        ("transpose", source, result),
        ("vstore", result, 0x20000),
        ("dma", "store", output_address, 0x20000, 1024, 1),
        ("wait", 1), ("complete",), ("trap",),
    ]
    if effects != expected:
        raise NativeCompilationError(
            "unqualified_target", "emitted transpose addresses, scratch words, order, waits, or registers differ"
        )
    # Replay issue gaps from the final words, separately from the OOT delay
    # insertion pass. These are conservative diagnostic floors, not an RTL
    # latency qualification for an integrated core.
    issued: dict[str, int] = {}
    cycle = 0
    for row in program["instructions"]:
        op = row["operation"]
        if op in {"atlas.vload", "atlas.xlu_transpose", "atlas.vstore"} or (
            op == "atlas.dma" and row["fields"].get("direction") == "store"
        ):
            issued[op] = cycle
        if op == "atlas.delay":
            count = row["fields"].get("cycles")
            if type(count) is not int or not 0 <= count <= 65535:
                raise NativeCompilationError("unqualified_target", "movement delay has invalid cycle count")
            cycle += count
        cycle += 1
    for producer, consumer, minimum in (
        ("atlas.vload", "atlas.xlu_transpose", 34),
        ("atlas.xlu_transpose", "atlas.vstore", 65),
        ("atlas.vstore", "atlas.dma", 34),
    ):
        if issued[consumer] - issued[producer] < minimum:
            raise NativeCompilationError(
                "unqualified_target", f"emitted transpose has premature {consumer} after {producer}"
            )


def check_copy_transpose_stream(program: dict, *, input_address: int,
                                output_addresses: tuple[int, int],
                                source: int, result: int) -> None:
    effects = _tile_effects(program)
    expected = [
        ("config",),
        ("dma", "load", input_address, 0, 1024, 0),
        ("wait", 0), ("vload", source, 0),
        ("vstore", source, 0x20000),
        ("dma", "store", output_addresses[0], 0x20000, 1024, 1),
        ("wait", 1), ("transpose", source, result),
        ("vstore", result, 0x20000),
        ("dma", "store", output_addresses[1], 0x20000, 1024, 1),
        ("wait", 1), ("complete",), ("trap",),
    ]
    _check_atomic_dma_schedule(effects, expected, context="copy/transpose stream")
    issued: dict[str, list[int]] = {}
    cycle = 0
    for row in program["instructions"]:
        op = row["operation"]
        if op in {"atlas.vload", "atlas.vstore", "atlas.xlu_transpose"} or (
            op == "atlas.dma" and row["fields"].get("direction") == "store"
        ):
            issued.setdefault(op, []).append(cycle)
        if op == "atlas.delay":
            count = row["fields"].get("cycles")
            if type(count) is not int or not 0 <= count <= 65535:
                raise NativeCompilationError("unqualified_target", "copy/transpose delay has invalid cycle count")
            cycle += count
        cycle += 1
    gaps = (
        ("atlas.vload", 0, "atlas.vstore", 0, 34),
        ("atlas.vload", 0, "atlas.xlu_transpose", 0, 34),
        ("atlas.xlu_transpose", 0, "atlas.vstore", 1, 65),
        ("atlas.vstore", 0, "atlas.dma", 0, 34),
        ("atlas.vstore", 1, "atlas.dma", 1, 34),
    )
    if any(issued[end][end_index] - issued[start][start_index] < minimum
           for start, start_index, end, end_index, minimum in gaps):
        raise NativeCompilationError("unqualified_target", "emitted copy/transpose has premature use")


def check_mxu_stream(program: dict, *, unit: int, activation_address: int,
                     weight_address: int, output_address: int,
                     activation: int, weight: int, pair: int,
                     relu_pair: int | None = None) -> None:
    if unit not in (0, 1) or (unit == 1 and relu_pair is not None):
        raise NativeCompilationError("unsupported_semantics", "selected MXU stream mode is unavailable")
    effects = _tile_effects(program)
    expected = [
        ("config",),
        ("dma", "load", activation_address, 0, 1024, 0),
        ("wait", 0), ("vload", activation, 0),
        ("dma", "load", weight_address, 512, 1024, 0),
        ("wait", 0), ("vload", weight, 512),
        ("mxu_push", unit, 0, "weight_fp8", weight),
        ("mxu_matmul", unit, 0, 0, False, activation),
        ("mxu_pop", unit, 0, "bf16", 0, pair),
    ]
    if relu_pair is not None:
        if relu_pair % 2 or not 0 <= relu_pair <= 60 or relu_pair == pair:
            raise NativeCompilationError("unqualified_target", "MXU ReLU pair is invalid")
        expected.append(("relu", pair, relu_pair))
    output_pair = relu_pair if relu_pair is not None else pair
    expected.extend([
        ("vstore", output_pair, 0x10000),
        ("dma", "store", output_address, 0x10000, 1024, 1),
        ("wait", 1),
        ("vstore", output_pair + 1, 0x10100),
        ("dma", "store", output_address + 1024, 0x10100, 1024, 1),
        ("wait", 1), ("complete",), ("trap",),
    ])
    if effects != expected:
        raise NativeCompilationError(
            "unqualified_target", "emitted MXU stream differs in addresses, resource slots, order, or waits"
        )
    # Replay conservative issue gaps from the emitted words, independently of
    # the virtual schedule verifier. These selected-core timing floors do not
    # qualify host launch or dynamic memory completion.
    issued: dict[str, list[int]] = {}
    cycle = 0
    for row in program["instructions"]:
        op = row["operation"]
        if op in {"atlas.vload", "atlas.mxu_push", "atlas.mxu_matmul",
                  "atlas.mxu_pop", "atlas.vpu_unary", "atlas.vstore"} or (
            op == "atlas.dma" and row["fields"].get("direction") == "store"
        ):
            issued.setdefault(op, []).append(cycle)
        if op == "atlas.delay":
            count = row["fields"].get("cycles")
            if type(count) is not int or not 0 <= count <= 65535:
                raise NativeCompilationError("unqualified_target", "MXU delay has invalid cycle count")
            cycle += count
        cycle += 1
    gaps = [
        ("atlas.vload", 0, "atlas.mxu_matmul", 0, 34),
        ("atlas.vload", 1, "atlas.mxu_push", 0, 34),
        # MXU1's anchor tree starts accumulator writes at age 3, whereas
        # MXU0's systolic path starts at age 63. The streams can overlap
        # once each read row follows its matching write row.
        ("atlas.mxu_matmul", 0, "atlas.mxu_pop", 0, 64 if unit == 0 else 32),
        ("atlas.vstore", 0, "atlas.dma", 0, 34),
        ("atlas.vstore", 1, "atlas.dma", 1, 34),
    ]
    if unit == 1:
        gaps.append(("atlas.mxu_push", 0, "atlas.mxu_matmul", 0, 32))
    if relu_pair is None:
        gaps.append(("atlas.mxu_pop", 0, "atlas.vstore", 0, 32))
    else:
        gaps.extend((("atlas.mxu_pop", 0, "atlas.vpu_unary", 0, 32),
                     ("atlas.vpu_unary", 0, "atlas.vstore", 0, 65)))
    if any(issued[end][end_index] - issued[start][start_index] < minimum
           for start, start_index, end, end_index, minimum in gaps):
        raise NativeCompilationError("unqualified_target", "emitted MXU stream violates diagnostic issue gaps")


def check_mxu0_stream(program: dict, *, activation_address: int,
                      weight_address: int, output_address: int,
                      activation: int, weight: int, pair: int,
                      relu_pair: int | None = None) -> None:
    """Retain the MXU0 checker entry point used by existing OOT tests."""
    check_mxu_stream(
        program, unit=0, activation_address=activation_address,
        weight_address=weight_address, output_address=output_address,
        activation=activation, weight=weight, pair=pair, relu_pair=relu_pair,
    )


def _mxu0_k_chain_selection(
    graph: CandidateGraph, names: list[str | None], assignment: dict[int, int],
    source_ids: tuple[str, ...], input_slots: tuple[int, ...], output_slot: int,
) -> tuple[list[int], tuple[int, ...]]:
    """Replay the selected K chain before materializing physical state."""
    tile_count = len(source_ids) // 2
    if tile_count < 2 or 2 * tile_count + 1 > DRAM_SLOTS or len(input_slots) != 2 * tile_count or len(graph.outputs) != 1:
        raise NativeCompilationError("unqualified_target", "K chain exceeds its input or output DRAM slots")
    loads: dict[str, int] = {}
    for value in graph.values:
        if names[value.id] != "atlas.load_fp8_tile":
            continue
        if len(value.children) != 1 or graph.value(value.children[0]).kind != "input":
            raise NativeCompilationError("unqualified_target", "K-chain load has no declared source")
        source = graph.value(value.children[0]).source_node
        if source not in source_ids or source in loads or assignment[value.children[0]] != (
            input_slots[source_ids.index(source)]
        ):
            raise NativeCompilationError("unqualified_target", "K-chain load changes its DRAM source")
        loads[source] = value.id
    if set(loads) != set(source_ids):
        raise NativeCompilationError("unqualified_target", "K-chain selection omits an input load")
    reset = [value for value in graph.values if names[value.id] == "atlas.mxu0_reset_acc_tile"]
    continued = [value for value in graph.values if names[value.id] == "atlas.mxu0_accumulate_tile"]
    stores = [value for value in graph.values if names[value.id] == "atlas.store_bf16_mxu0_acc_tile"]
    chain = reset[:]
    remaining = {value.id: value for value in continued}
    if reset:
        for index in range(1, tile_count):
            children = (loads[source_ids[2 * index]],
                        loads[source_ids[2 * index + 1]], chain[-1].id)
            matches = [value for value in remaining.values() if value.children == children]
            if len(matches) != 1:
                raise NativeCompilationError("unqualified_target", "K-chain continuation is broken")
            chain.append(matches[0])
            del remaining[matches[0].id]
    if (
        len(reset) != 1 or len(continued) != tile_count - 1 or len(stores) != 1
        or reset[0].children != (loads[source_ids[0]], loads[source_ids[1]])
        or remaining
        or stores[0].children != (chain[-1].id,)
        or graph.outputs != (stores[0].id,)
        or any(assignment[node.id] != 0 for node in chain)
        or assignment[stores[0].id] != output_slot
        or sum(value.kind == "instruction" for value in graph.values) != 3 * tile_count + 1
    ):
        raise NativeCompilationError("unqualified_target", "K-chain selection changes accumulator chain")
    registers = tuple(assignment[loads[source]] for source in source_ids)
    if any(not 0 <= register < 32 for register in registers):
        raise NativeCompilationError("unqualified_target", "K-chain FP8 placement exceeds the register file")
    return sorted(set(registers)), registers


def check_mxu0_k_chain_stream(
    program: dict, *, input_addresses: tuple[int, ...],
    output_address: int, source_registers: tuple[int, ...],
) -> None:
    """Replay a reset/continue/readout stream from emitted physical fields."""
    tile_count = len(input_addresses) // 2
    if (
        tile_count < 2 or 2 * tile_count + 1 > DRAM_SLOTS
        or len(input_addresses) != 2 * tile_count
        or len(source_registers) != len(input_addresses)
        or len(set(input_addresses)) != len(input_addresses)
        or any(source_registers[2 * index] == source_registers[2 * index + 1]
               for index in range(tile_count))
    ):
        raise NativeCompilationError("unqualified_target", "K-chain ABI or register pair overlaps")
    effects = _tile_effects(program)
    if effects[:1] != [("config",)] or effects[-2:] != [("complete",), ("trap",)]:
        raise NativeCompilationError("unqualified_target", "K-chain stream has no selected entry or completion")
    kinds = [effect[0] for effect in effects]
    expected_counts = {
        "config": 1, "dma": 2 * tile_count + 2, "wait": 2 * tile_count + 2,
        "vload": 2 * tile_count,
        "mxu_push": tile_count, "mxu_matmul": tile_count, "mxu_pop": 1,
        "vstore": 2, "complete": 1, "trap": 1,
    }
    if {kind: kinds.count(kind) for kind in set(kinds) | set(expected_counts)} != expected_counts:
        raise NativeCompilationError("unqualified_target", "K-chain stream has missing or extra effects")
    source_by_reg: dict[int, int] = {}
    load_cycles: dict[int, int] = {}
    pushes = 0
    matmuls = 0
    pair: int | None = None
    store_halves = 0
    pending_store = False
    event_cycles = []
    cycle = 0
    for row in program["instructions"]:
        if row["operation"] in {
            "atlas.dma_config", "atlas.dma", "atlas.dma_wait", "atlas.vload",
            "atlas.mxu_push", "atlas.mxu_matmul", "atlas.mxu_pop",
            "atlas.vstore", "atlas.csr", "atlas.trap",
        }:
            event_cycles.append(cycle)
        if row["operation"] == "atlas.delay":
            delay = row["fields"].get("cycles")
            if type(delay) is not int or not 0 <= delay <= 65535:
                raise NativeCompilationError("unqualified_target", "K-chain delay has an invalid cycle count")
            cycle += delay
        cycle += 1
    if len(event_cycles) != len(effects):
        raise NativeCompilationError("unqualified_target", "K-chain stream has unmodeled physical effects")
    push_cycle = 0
    matmul_cycle = 0
    pop_cycle = 0
    vstore_cycle = 0
    for index, effect in enumerate(effects):
        kind = effect[0]
        now = event_cycles[index]
        if kind == "dma" and effect[1] == "load":
            if (
                effect[2] not in input_addresses or effect[4:] != (1024, 0)
                or index + 2 >= len(effects)
                or effects[index + 1] != ("wait", 0)
                or effects[index + 2][0] != "vload"
                or effects[index + 2][2] != effect[3]
            ):
                raise NativeCompilationError("unqualified_target", "K-chain input DMA, wait, or VLOAD differs")
            address = effect[2]
            register = effects[index + 2][1]
            if register != source_registers[input_addresses.index(address)]:
                raise NativeCompilationError("unqualified_target", "K-chain selected register differs from emission")
            source_by_reg[register] = address
            load_cycles[address] = event_cycles[index + 2]
        elif kind == "mxu_push":
            if (
                pushes >= tile_count or effect[:4] != ("mxu_push", 0, 0, "weight_fp8")
                or source_by_reg.get(effect[4]) != input_addresses[2 * pushes + 1]
                or now - load_cycles.get(input_addresses[2 * pushes + 1], now) < 34
                or (pushes > 0 and matmuls != pushes)
            ):
                raise NativeCompilationError("unqualified_target", "K-chain weight push differs")
            pushes += 1
            push_cycle = now
        elif kind == "mxu_matmul":
            if (
                matmuls >= tile_count or pushes != matmuls + 1
                or effect[:5] != ("mxu_matmul", 0, 0, 0, bool(matmuls))
                or source_by_reg.get(effect[5]) != input_addresses[2 * matmuls]
                or now - load_cycles.get(input_addresses[2 * matmuls], now) < 34
                or now <= push_cycle
                or (matmuls > 0 and now - matmul_cycle < 64)
            ):
                raise NativeCompilationError("unqualified_target", "K-chain reset/continue order or availability differs")
            matmuls += 1
            matmul_cycle = now
        elif kind == "mxu_pop":
            if (
                matmuls != tile_count or pair is not None
                or effect[:5] != ("mxu_pop", 0, 0, "bf16", 0)
                or type(effect[5]) is not int or effect[5] % 2
                or not 32 <= effect[5] <= 60
                or now - matmul_cycle < 64
            ):
                raise NativeCompilationError("unqualified_target", "K-chain readout differs")
            pair = effect[5]
            pop_cycle = now
        elif kind == "vstore":
            if (
                pair is None or store_halves >= 2 or pending_store
                or effect[1] != pair + store_halves
                or now - pop_cycle < 32
            ):
                raise NativeCompilationError("unqualified_target", "K-chain output half or availability differs")
            store_halves += 1
            pending_store = True
            vstore_cycle = now
        elif kind == "dma" and effect[1] == "store":
            half = store_halves - 1
            if (
                not pending_store or half not in (0, 1)
                or effect[2] != output_address + 1024 * half
                or effect[4:] != (1024, 1)
                or effect[3] != effects[index - 1][2]
                or effects[index - 1][0] != "vstore"
                or index + 1 >= len(effects) or effects[index + 1] != ("wait", 1)
                or now - vstore_cycle < 34
            ):
                raise NativeCompilationError("unqualified_target", "K-chain output DMA or wait differs")
            pending_store = False
        elif kind in {"config", "wait", "vload", "complete", "trap"}:
            continue
        else:
            raise NativeCompilationError("unqualified_target", "K-chain stream has an unadmitted effect")
    if pushes != tile_count or matmuls != tile_count or store_halves != 2 or pending_store:
        raise NativeCompilationError("unqualified_target", "K-chain stream is incomplete")


def _mxu_binary_selection(
    graph: CandidateGraph, order: tuple[int, ...], names: list[str | None],
    assignment: dict[int, int], source_ids: tuple[str, ...],
    input_slots: tuple[int, ...], output_slot: int,
) -> dict[str, int]:
    """Check the selected DAG and initial placement before Atlas lowering."""
    by_name: dict[str, list[int]] = {}
    for value in graph.values:
        if value.kind == "instruction":
            by_name.setdefault(names[value.id], []).append(value.id)
        elif value.kind == "input" and (
            value.source_node not in source_ids
            or assignment[value.id] != input_slots[source_ids.index(value.source_node)]
        ):
            raise NativeCompilationError("unqualified_target", "MXU/VPU boundary input binding changed")
    expected_names = {
        "atlas.load_fp8_tile": 2,
        "atlas.load_bf16_tile": 1,
        "atlas.mxu0_reset_tile": 1,
        "atlas.store_bf16_tile": 1,
    }
    binaries = [name for name in ("atlas.vpu_add_bf16_tile", "atlas.vpu_mul_bf16_tile")
                if name in by_name]
    if (
        len(binaries) != 1
        or any(len(by_name.get(name, ())) != count for name, count in expected_names.items())
        or len(by_name) != len(expected_names) + 1
        or len(by_name[binaries[0]]) != 1
        or set(order) != {value.id for value in graph.values if value.kind == "instruction"}
        or len(order) != 6
    ):
        raise NativeCompilationError("unqualified_target", "selected MXU/VPU instruction DAG changed")
    fp8_loads = by_name["atlas.load_fp8_tile"]
    bf16_load = by_name["atlas.load_bf16_tile"][0]
    mxu = by_name["atlas.mxu0_reset_tile"][0]
    binary = by_name[binaries[0]][0]
    store = by_name["atlas.store_bf16_tile"][0]
    if (
        any(len(graph.value(value_id).children) != 1 or
            graph.value(graph.value(value_id).children[0]).kind != "input"
            for value_id in (*fp8_loads, bf16_load))
        or graph.value(mxu).children != tuple(fp8_loads)
        or set(graph.value(binary).children) != {mxu, bf16_load}
        or graph.value(store).children != (binary,)
        or graph.outputs != (store,)
        or assignment[store] != output_slot
    ):
        raise NativeCompilationError("unqualified_target", "selected MXU/VPU producer, output, or address changed")
    loaded_source_ids = {
        graph.value(graph.value(value_id).children[0]).source_node for value_id in (*fp8_loads, bf16_load)
    }
    if loaded_source_ids != set(source_ids):
        raise NativeCompilationError("unqualified_target", "selected MXU/VPU loads omit an input")
    fp8_registers = tuple(assignment[value_id] for value_id in fp8_loads)
    mxu_pair, bf16_pair, result_pair = (assignment[value_id] for value_id in (mxu, bf16_load, binary))
    if (
        any(not 0 <= address < 32 for address in fp8_registers)
        or fp8_registers[0] == fp8_registers[1]
        or mxu_pair != 32
        or any(pair % 2 or not 0 <= pair <= 60 for pair in (bf16_pair, result_pair))
        or len({mxu_pair, bf16_pair, result_pair}) != 3
    ):
        raise NativeCompilationError("unqualified_target", "selected MXU/VPU register placement is invalid")
    return {"activation": fp8_loads[0], "weight": fp8_loads[1],
            "bf16": bf16_load, "mxu": mxu, "binary": binary, "store": store}


def _mxu_dag_selection(
    graph: CandidateGraph, order: tuple[int, ...], names: list[str | None],
    assignment: dict[int, int], source_ids: tuple[str, ...],
    mxu_inputs: dict[str, tuple[str, str]], input_slots: tuple[int, ...],
    output_slots: tuple[int, ...], *,
    transpose_inputs: dict[str, str] | None = None, physical: bool = False,
) -> list[int]:
    """Check a selected MXU0/BF16 DAG before or after physical placement."""
    instructions = {value.id for value in graph.values if value.kind == "instruction"}
    if (set(order) != instructions or len(order) != len(instructions)
            or len(graph.outputs) != len(output_slots)
            or len(set(graph.outputs)) != len(graph.outputs)):
        raise NativeCompilationError("unqualified_target", "selected MXU/VPU order or outputs changed")
    if sum(names[value_id] == "atlas.mxu0_reset_tile" for value_id in order) != len(mxu_inputs):
        raise NativeCompilationError("unqualified_target", "selected MXU/VPU DAG has a changed MXU count")
    output_indices = {value_id: index for index, value_id in enumerate(graph.outputs)}
    registers: set[int] = set()
    fp8_sources: set[str] = set()
    packed_sources: set[str] = set()
    transposed_sources: set[str] = set()
    selected_mxus: set[str] = set()
    transpose_inputs = transpose_inputs or {}
    expected_fp8_sources = {source for pair in mxu_inputs.values() for source in pair}
    external_fp8_sources = expected_fp8_sources.intersection(source_ids)
    expected_transpose_sources = set(transpose_inputs)
    expected_packed_sources = (
        expected_fp8_sources.difference(source_ids, expected_transpose_sources)
        | set(transpose_inputs.values())
    )
    if (not expected_transpose_sources <= expected_fp8_sources.difference(source_ids)
            or any(source in source_ids or source in transpose_inputs
                   for source in transpose_inputs.values())):
        raise NativeCompilationError("unqualified_target", "MXU transpose sources are inconsistent")
    for value in graph.values:
        name = names[value.id]
        if value.kind in {"input", "constant"}:
            if (value.source_node not in source_ids or value.storage != "external"
                    or assignment[value.id] != input_slots[source_ids.index(value.source_node)]):
                raise NativeCompilationError("unqualified_target", "selected MXU/VPU boundary binding changed")
        elif name in {"atlas.load_fp8_tile", "atlas.load_bf16_tile"}:
            if (len(value.children) != 1
                    or graph.value(value.children[0]).kind not in {"input", "constant"}):
                raise NativeCompilationError("unqualified_target", "selected MXU/VPU load source changed")
            source = graph.value(value.children[0]).source_node
            if name == "atlas.load_fp8_tile":
                if source not in external_fp8_sources or value.storage != "tensor_fp8":
                    raise NativeCompilationError("unqualified_target", "selected MXU/VPU FP8 source changed")
                fp8_sources.add(source)
                address = assignment[value.id]
                if not 0 <= address < 32:
                    raise NativeCompilationError("unqualified_target", "selected FP8 register is invalid")
                registers.add(address)
            else:
                if source in external_fp8_sources or value.storage != "tensor_bf16":
                    raise NativeCompilationError("unqualified_target", "selected MXU/VPU BF16 source changed")
                pair = assignment[value.id]
                if pair % 2 or not (32 if physical else 0) <= pair <= 60:
                    raise NativeCompilationError("unqualified_target", "selected BF16 load pair is invalid")
                registers.update((pair, pair + 1))
        elif name in PACK_RULE_NAMES:
            if (value.source_node not in expected_packed_sources
                    or value.source_node in packed_sources or value.storage != "tensor_fp8"
                    or len(value.children) != 1
                    or names[value.children[0]] not in {
                        "atlas.mxu0_reset_tile", "atlas.vpu_relu_bf16_tile",
                        "atlas.vpu_relu_bf16_from_mxu0_tile", "atlas.vpu_add_bf16_tile",
                        "atlas.vpu_mul_bf16_tile", "atlas.vpu_sub_bf16_tile",
                        *BF16_UNARY_MODES,
                        *BF16_ROW_REDUCTION_MODES,
                    }
                    or not 0 <= assignment[value.id] < 32):
                raise NativeCompilationError("unqualified_target", "selected interlayer pack changed")
            packed_sources.add(value.source_node)
            registers.add(assignment[value.id])
        elif name in TRANSPOSE_RULE_NAMES:
            expected_source = transpose_inputs.get(value.source_node)
            expected_pack = PACK_RULE_BY_TRANSPOSE_RULE[name]
            if (value.source_node not in expected_transpose_sources
                    or value.source_node in transposed_sources
                    or value.storage != "tensor_fp8" or len(value.children) != 1
                    or graph.value(value.children[0]).source_node != expected_source
                    or names[value.children[0]] != expected_pack
                    or not 0 <= assignment[value.id] < 32
                    or assignment[value.id] == assignment[value.children[0]]):
                raise NativeCompilationError("unqualified_target", "selected packed FP8 transpose changed")
            transposed_sources.add(value.source_node)
            registers.add(assignment[value.id])
        elif name == "atlas.mxu0_reset_tile":
            expected_pair = mxu_inputs.get(value.source_node)
            if (value.storage != "tensor_bf16" or len(value.children) != 2
                    or any(names[child] not in {
                        "atlas.load_fp8_tile", *PACK_RULE_NAMES, *TRANSPOSE_RULE_NAMES,
                    } for child in value.children)
                    or tuple(graph.value(child).source_node for child in value.children) != expected_pair
                    or value.source_node in selected_mxus
                    or (assignment[value.id] % 2 or not 32 <= assignment[value.id] <= 60)):
                raise NativeCompilationError("unqualified_target", "selected MXU0 operands or readout changed")
            selected_mxus.add(value.source_node)
            registers.update((assignment[value.id], assignment[value.id] + 1))
        elif name in {"atlas.vpu_relu_bf16_tile", "atlas.vpu_relu_bf16_from_mxu0_tile",
                      *BF16_UNARY_MODES,
                      *BF16_ROW_REDUCTION_MODES,
                      *BF16_BINARY_MODES}:
            unary = (name.startswith("atlas.vpu_relu") or name in BF16_UNARY_MODES
                     or name in BF16_ROW_REDUCTION_MODES)
            if (value.storage != "tensor_bf16" or len(value.children) != (1 if unary else 2)
                    or (not unary and value.children[0] == value.children[1])
                    or any(names[child] not in {
                        "atlas.load_bf16_tile", "atlas.mxu0_reset_tile",
                        "atlas.vpu_relu_bf16_tile", "atlas.vpu_relu_bf16_from_mxu0_tile",
                        *BF16_UNARY_MODES,
                        *BF16_ROW_REDUCTION_MODES,
                        *BF16_BINARY_MODES,
                    } for child in value.children)):
                raise NativeCompilationError("unqualified_target", "selected MXU/VPU operand graph changed")
            if name == "atlas.vpu_relu_bf16_from_mxu0_tile" and names[value.children[0]] != "atlas.mxu0_reset_tile":
                raise NativeCompilationError("unqualified_target", "MXU0-specific ReLU lost its MXU source")
            pair = assignment[value.id]
            if (pair % 2 or not (32 if physical else 0) <= pair <= 60
                    or pair in {assignment[child] for child in value.children}):
                raise NativeCompilationError("unqualified_target", "selected MXU/VPU result pair is invalid")
            registers.update((pair, pair + 1))
        elif name in {"atlas.store_bf16_tile", "atlas.store_bf16_mxu0_tile"}:
            if (value.id not in output_indices or len(value.children) != 1
                    or value.storage != "external"
                    or assignment[value.id] != output_slots[output_indices[value.id]]):
                raise NativeCompilationError("unqualified_target", "selected MXU/VPU output binding changed")
            producer = names[value.children[0]]
            if ((name == "atlas.store_bf16_mxu0_tile" and producer != "atlas.mxu0_reset_tile")
                    or (name == "atlas.store_bf16_tile" and producer not in {
                        "atlas.load_bf16_tile", "atlas.vpu_relu_bf16_tile",
                        "atlas.vpu_relu_bf16_from_mxu0_tile", "atlas.vpu_add_bf16_tile",
                        "atlas.vpu_mul_bf16_tile", "atlas.vpu_sub_bf16_tile",
                        *BF16_UNARY_MODES,
                        *BF16_ROW_REDUCTION_MODES,
                    })):
                raise NativeCompilationError("unqualified_target", "selected MXU/VPU output producer changed")
        else:
            raise NativeCompilationError("unqualified_target", "selected MXU/VPU DAG has an unadmitted instruction")
    if (selected_mxus != set(mxu_inputs) or fp8_sources != external_fp8_sources
            or packed_sources != expected_packed_sources
            or transposed_sources != expected_transpose_sources):
        raise NativeCompilationError("unqualified_target", "selected MXU/VPU loads omit an FP8 input")
    return sorted(registers)


def check_mxu_bf16_binary_stream(
    program: dict, *, graph: CandidateGraph, order: tuple[int, ...],
    names: list[str | None], assignment: dict[int, int],
    source_ids: tuple[str, ...], input_slots: tuple[int, ...], output_slot: int,
) -> dict[int, int]:
    """Replay an emitted MXU0-to-VPU tile against its selected DAG and ABI."""
    ids = _mxu_binary_selection(
        graph, order, names, assignment, source_ids, input_slots, output_slot,
    )
    effects = _tile_effects(program)
    loads = [effect for effect in effects if effect[0] == "vload"]
    pops = [effect for effect in effects if effect[0] == "mxu_pop"]
    binaries = [effect for effect in effects if effect[0] in {"add", "mul"}]
    if (len(loads), len(pops), len(binaries)) != (4, 1, 1):
        raise NativeCompilationError("unqualified_target", "MXU/VPU physical instruction counts changed")
    realized = dict(assignment)
    load_index = 0
    for value_id in order:
        name = names[value_id]
        if name == "atlas.load_fp8_tile":
            realized[value_id] = loads[load_index][1]
            load_index += 1
        elif name == "atlas.load_bf16_tile":
            first, second = loads[load_index:load_index + 2]
            if second[1] != first[1] + 1:
                raise NativeCompilationError("unqualified_target", "MXU/VPU BF16 input is not a pair")
            realized[value_id] = first[1]
            load_index += 2
        elif name == "atlas.mxu0_reset_tile":
            realized[value_id] = pops[0][5]
        elif name in {"atlas.vpu_add_bf16_tile", "atlas.vpu_mul_bf16_tile"}:
            realized[value_id] = binaries[0][3]
    ids = _mxu_binary_selection(
        graph, order, names, realized, source_ids, input_slots, output_slot,
    )
    if any(realized[ids[key]] < 32 for key in ("mxu", "bf16", "binary")):
        raise NativeCompilationError(
            "unqualified_target", "emitted BF16 pair aliases the FP8 register view"
        )
    expected: list[tuple] = [("config",)]
    for value_id in order:
        name = names[value_id]
        value = graph.value(value_id)
        if name in {"atlas.load_fp8_tile", "atlas.load_bf16_tile"}:
            source = graph.value(value.children[0]).source_node
            index = source_ids.index(source)
            address = DRAM_BASE + input_slots[index] * TILE_BYTES
            scratch = index * 512
            count = 2 if name == "atlas.load_bf16_tile" else 1
            for half in range(count):
                expected.extend((
                    ("dma", "load", address + half * 1024, scratch + half * 256, 1024, 0),
                    ("wait", 0),
                    ("vload", realized[value_id] + half, scratch + half * 256),
                ))
        elif name == "atlas.mxu0_reset_tile":
            activation, weight = (realized[child] for child in value.children)
            expected.extend((
                ("mxu_push", 0, 0, "weight_fp8", weight),
                ("mxu_matmul", 0, 0, 0, False, activation),
                ("mxu_pop", 0, 0, "bf16", 0, realized[value_id]),
            ))
        elif name in {"atlas.vpu_add_bf16_tile", "atlas.vpu_mul_bf16_tile"}:
            kind = "add" if name == "atlas.vpu_add_bf16_tile" else "mul"
            expected.append((kind, realized[value.children[0]],
                             realized[value.children[1]], realized[value_id]))
        elif name == "atlas.store_bf16_tile":
            pair = realized[value.children[0]]
            address = DRAM_BASE + output_slot * TILE_BYTES
            for half in (0, 1):
                expected.extend((
                    ("vstore", pair + half, 0x10000 + half * 256),
                    ("dma", "store", address + half * 1024,
                     0x10000 + half * 256, 1024, 1),
                    ("wait", 1),
                ))
    expected.extend((("complete",), ("trap",)))
    if effects != expected:
        raise NativeCompilationError("unqualified_target", "MXU/VPU emitted effects differ from selection")

    cycle = 0
    issued: dict[str, list[int]] = {}
    for row in program["instructions"]:
        op = row["operation"]
        if op == "atlas.delay":
            count = row["fields"].get("cycles")
            if type(count) is not int or not 0 <= count <= 65535:
                raise NativeCompilationError("unqualified_target", "MXU/VPU delay has invalid count")
            cycle += count
        elif op in {"atlas.vload", "atlas.mxu_push", "atlas.mxu_matmul",
                    "atlas.mxu_pop", "atlas.vpu_binary", "atlas.vstore"} or (
            op == "atlas.dma" and row["fields"].get("direction") == "store"
        ):
            issued.setdefault(op, []).append(cycle)
        cycle += 1
    first_load_index: dict[int, int] = {}
    cursor = 0
    for value_id in order:
        if names[value_id] in {"atlas.load_fp8_tile", "atlas.load_bf16_tile"}:
            first_load_index[value_id] = cursor
            cursor += 2 if value_id == ids["bf16"] else 1
    gaps = (
        ("atlas.vload", first_load_index[ids["activation"]], "atlas.mxu_matmul", 0, 34),
        ("atlas.vload", first_load_index[ids["weight"]], "atlas.mxu_push", 0, 34),
        ("atlas.mxu_matmul", 0, "atlas.mxu_pop", 0, 64),
        ("atlas.mxu_pop", 0, "atlas.vpu_binary", 0, 32),
        ("atlas.vload", first_load_index[ids["bf16"]], "atlas.vpu_binary", 0, 34),
        ("atlas.vload", first_load_index[ids["bf16"]] + 1, "atlas.vpu_binary", 0, 34),
        ("atlas.vpu_binary", 0, "atlas.vstore", 0, 65),
        ("atlas.vpu_binary", 0, "atlas.vstore", 1, 65),
        ("atlas.vstore", 0, "atlas.dma", 0, 34),
        ("atlas.vstore", 1, "atlas.dma", 1, 34),
    )
    try:
        valid = all(issued[end][end_index] - issued[start][start_index] >= minimum
                    for start, start_index, end, end_index, minimum in gaps)
    except (KeyError, IndexError) as exc:
        raise NativeCompilationError("unqualified_target", "MXU/VPU issue event is missing") from exc
    if not valid:
        raise NativeCompilationError("unqualified_target", "MXU/VPU issued before diagnostic availability")
    return realized


def check_mxu_bf16_dag_stream(
    program: dict, *, graph: CandidateGraph, order: tuple[int, ...],
    names: list[str | None], assignment: dict[int, int],
    source_ids: tuple[str, ...], mxu_inputs: dict[str, tuple[str, str]],
    input_slots: tuple[int, ...], output_slots: tuple[int, ...],
    pack_scale_codes: dict[str, int] | None = None,
    transpose_inputs: dict[str, str] | None = None,
) -> dict[int, int]:
    """Recover placement and replay MXU0 resets with a BF16 VPU DAG."""
    scale_codes = pack_scale_codes or {}
    transpose_inputs = transpose_inputs or {}

    def pack_code(value_id: int) -> int:
        value = graph.value(value_id)
        name = names[value_id]
        code = scale_codes.get(value.source_node)
        if name == "atlas.vpu_pack_unit_fp8_tile":
            if code not in (None, 127):
                raise NativeCompilationError("unqualified_target", "unit pack has a nonunit scale")
            return 127
        if name in {PACK_RULE_BY_POLICY[SCALED_PACK_POLICY],
                    PACK_RULE_BY_POLICY[ROUNDED_PACK_POLICY]} and type(code) is int and code in range(122, 129):
            return code
        raise NativeCompilationError("unqualified_target", "scaled interlayer pack lacks its selected code")

    _mxu_dag_selection(graph, order, names, assignment, source_ids,
                       mxu_inputs, input_slots, output_slots,
                       transpose_inputs=transpose_inputs)
    effects = _tile_effects(program)
    loads = [effect for effect in effects if effect[0] == "vload"]
    pops = [effect for effect in effects if effect[0] == "mxu_pop"]
    packs = [effect for effect in effects if effect[0] == "pack"]
    transposes = [effect for effect in effects if effect[0] == "transpose"]
    unaries = [effect for effect in effects if effect[0] in {"relu", *BF16_UNARY_MODES.values()}]
    reductions = [effect for effect in effects if effect[0] in BF16_ROW_REDUCTION_MODES.values()]
    binaries = [effect for effect in effects if effect[0] in BF16_BINARY_MODES.values()]
    expected_loads = sum(1 if names[value_id] in {
        "atlas.load_fp8_tile", *PACK_RULE_NAMES,
    } else 2 if names[value_id] == "atlas.load_bf16_tile" else 0
                         for value_id in order)
    expected_packs = sum(names[value_id] in PACK_RULE_NAMES
                         for value_id in order)
    expected_transposes = sum(names[value_id] in TRANSPOSE_RULE_NAMES for value_id in order)
    expected_unaries = sum(names[value_id] in {
        "atlas.vpu_relu_bf16_tile", "atlas.vpu_relu_bf16_from_mxu0_tile",
        *BF16_UNARY_MODES,
    } for value_id in order)
    expected_reductions = sum(names[value_id] in BF16_ROW_REDUCTION_MODES for value_id in order)
    expected_binaries = sum(names[value_id] in BF16_BINARY_MODES for value_id in order)
    if (len(loads), len(pops), len(packs), len(transposes), len(unaries),
            len(reductions), len(binaries)) != (
        expected_loads, len(mxu_inputs), expected_packs, expected_transposes, expected_unaries,
        expected_reductions, expected_binaries,
    ):
        raise NativeCompilationError("unqualified_target", "MXU/VPU physical instruction counts changed")
    realized = dict(assignment)
    load_index = pop_index = pack_index = transpose_index = unary_index = reduction_index = binary_index = 0
    for value_id in order:
        value = graph.value(value_id)
        name = names[value_id]
        if name == "atlas.load_fp8_tile":
            realized[value_id] = loads[load_index][1]
            load_index += 1
        elif name == "atlas.load_bf16_tile":
            first, second = loads[load_index:load_index + 2]
            if second[1] != first[1] + 1:
                raise NativeCompilationError("unqualified_target", "MXU/VPU BF16 input is not a pair")
            realized[value_id] = first[1]
            load_index += 2
        elif name == "atlas.mxu0_reset_tile":
            realized[value_id] = pops[pop_index][5]
            pop_index += 1
        elif name in PACK_RULE_NAMES:
            pack_code(value_id)
            effect = packs[pack_index]
            pack_index += 1
            if (effect[1] != "bf16_to_fp8" or effect[2] != realized[value.children[0]]
                    or effect[4] != 3 or loads[load_index] != ("vload", effect[3], 33024)):
                raise NativeCompilationError("unqualified_target", "interlayer pack source or reload changed")
            realized[value_id] = effect[3]
            load_index += 1
        elif name in TRANSPOSE_RULE_NAMES:
            effect = transposes[transpose_index]
            transpose_index += 1
            if effect[1] != realized[value.children[0]]:
                raise NativeCompilationError("unqualified_target", "packed FP8 transpose source changed")
            realized[value_id] = effect[2]
        elif name in {"atlas.vpu_relu_bf16_tile", "atlas.vpu_relu_bf16_from_mxu0_tile",
                      *BF16_UNARY_MODES}:
            effect = unaries[unary_index]
            unary_index += 1
            kind = BF16_UNARY_MODES.get(name, "relu")
            if effect[0] != kind or effect[1] != realized[value.children[0]]:
                raise NativeCompilationError("unqualified_target", "MXU/VPU unary source or mode changed")
            realized[value_id] = effect[2]
        elif name in BF16_ROW_REDUCTION_MODES:
            effect = reductions[reduction_index]
            reduction_index += 1
            if (effect[0] != BF16_ROW_REDUCTION_MODES[name]
                    or effect[1] != realized[value.children[0]]):
                raise NativeCompilationError("unqualified_target", "MXU/VPU row reduction changed")
            realized[value_id] = effect[2]
        elif name in BF16_BINARY_MODES:
            effect = binaries[binary_index]
            binary_index += 1
            kind = BF16_BINARY_MODES[name]
            if effect[0] != kind or effect[1:3] != tuple(realized[child] for child in value.children):
                raise NativeCompilationError("unqualified_target", "MXU/VPU binary kind or sources changed")
            realized[value_id] = effect[3]
    _mxu_dag_selection(graph, order, names, realized, source_ids,
                       mxu_inputs, input_slots, output_slots,
                       transpose_inputs=transpose_inputs, physical=True)

    output_indices = {value_id: index for index, value_id in enumerate(graph.outputs)}
    expected: list[tuple] = [("config",)]
    for value_id in order:
        value = graph.value(value_id)
        name = names[value_id]
        if name in {"atlas.load_fp8_tile", "atlas.load_bf16_tile"}:
            source = graph.value(value.children[0]).source_node
            index = source_ids.index(source)
            address = DRAM_BASE + input_slots[index] * TILE_BYTES
            scratch = index * 512
            count = 2 if name == "atlas.load_bf16_tile" else 1
            for half in range(count):
                expected.extend((
                    ("dma", "load", address + half * 1024, scratch + half * 256, 1024, 0),
                    ("wait", 0), ("vload", realized[value_id] + half, scratch + half * 256),
                ))
        elif name == "atlas.mxu0_reset_tile":
            activation, weight = (realized[child] for child in value.children)
            expected.extend((
                ("mxu_push", 0, 0, "weight_fp8", weight),
                ("mxu_matmul", 0, 0, 0, False, activation),
                ("mxu_pop", 0, 0, "bf16", 0, realized[value_id]),
            ))
        elif name in {"atlas.vpu_relu_bf16_tile", "atlas.vpu_relu_bf16_from_mxu0_tile",
                      *BF16_UNARY_MODES}:
            expected.append((BF16_UNARY_MODES.get(name, "relu"),
                             realized[value.children[0]], realized[value_id]))
        elif name in BF16_ROW_REDUCTION_MODES:
            expected.append((BF16_ROW_REDUCTION_MODES[name],
                             realized[value.children[0]], realized[value_id]))
        elif name in BF16_BINARY_MODES:
            kind = BF16_BINARY_MODES[name]
            expected.append((kind, realized[value.children[0]],
                             realized[value.children[1]], realized[value_id]))
        elif name in PACK_RULE_NAMES:
            expected.extend((
                ("seli", 3, pack_code(value_id)),
                ("pack", "bf16_to_fp8", realized[value.children[0]], realized[value_id], 3),
                ("vstore", realized[value_id], 32768),
            ))
            expected.extend(_pack_relayout_effects())
            expected.append(("vload", realized[value_id], 33024))
        elif name in TRANSPOSE_RULE_NAMES:
            expected.append(("transpose", realized[value.children[0]], realized[value_id]))
        elif name in {"atlas.store_bf16_tile", "atlas.store_bf16_mxu0_tile"}:
            index = output_indices[value_id]
            address = DRAM_BASE + output_slots[index] * TILE_BYTES
            pair = realized[value.children[0]]
            scratch = 0x10000 + index * 512
            for half in (0, 1):
                expected.extend((
                    ("vstore", pair + half, scratch + half * 256),
                    ("dma", "store", address + half * 1024,
                     scratch + half * 256, 1024, 1),
                    ("wait", 1),
                ))
    expected.extend((("complete",), ("trap",)))
    if effects != expected:
        mismatch = next((index for index, pair in enumerate(zip(effects, expected))
                         if pair[0] != pair[1]), min(len(effects), len(expected)))
        actual = effects[mismatch] if mismatch < len(effects) else None
        wanted = expected[mismatch] if mismatch < len(expected) else None
        raise NativeCompilationError(
            "unqualified_target",
            f"MXU/VPU DAG effect {mismatch} differs: selected {wanted}, emitted {actual}",
        )
    _check_interlayer_pack_words(program, tuple(
        (realized[graph.value(value_id).children[0]], realized[value_id]) for value_id in order
        if names[value_id] in PACK_RULE_NAMES
    ))
    _check_mxu_dag_timing(program)
    return realized


def _check_interlayer_pack_words(program: dict, pack_registers: tuple[tuple[int, int], ...]) -> None:
    rows = program["instructions"]
    packs = [row for row in rows if row["operation"] == "atlas.vpu_pack"]
    branches = [index for index, row in enumerate(rows) if row["operation"] == "atlas.branch"]
    if len(packs) != len(pack_registers) or len(branches) != len(pack_registers):
        raise NativeCompilationError("unqualified_target", "interlayer pack encoding count changed")
    for row, (source_pair, destination) in zip(packs, pack_registers):
        if not 0 <= destination < 32:
            raise NativeCompilationError("unqualified_target", "interlayer pack destination is outside FP8 storage")
        word = row["word_u32"]
        if ((word >> 25) & 0x7f, (word >> 19) & 0x3f,
            (word >> 13) & 0x3f, (word >> 7) & 0x3f,
            word & 0x7f) != (0x44, source_pair, 3, destination, 0x57):
            raise NativeCompilationError("unqualified_target", "interlayer pack word changed")
    for branch in branches:
        first_load = branch - 27
        if (first_load < 0 or rows[first_load]["operation"] != "atlas.scalar_load"
                or rows[first_load]["fields"].get("kind") != "lw"
                or branch + 1 >= len(rows)
                or rows[branch + 1]["operation"] != "atlas.alu_imm"
                or any(rows[branch + 1]["fields"].get(key) != value for key, value in
                       {"dst": 0, "immediate": 0, "kind": "addi", "src": 0}.items())
                or rows[branch]["fields"].get("offset_bytes") != -54):
            raise NativeCompilationError("unqualified_target", "interlayer pack loop changed")
        word = rows[branch]["word_u32"]
        immediate = (((word >> 31) & 1) << 12) | (((word >> 7) & 1) << 11) | (
            ((word >> 25) & 0x3f) << 5) | (((word >> 8) & 0xf) << 1)
        immediate = immediate - 8192 if immediate & 4096 else immediate
        if (word & 0x7f, (word >> 12) & 7, (word >> 15) & 31,
            (word >> 20) & 31, immediate) != (0x63, 4, 13, 14, -54):
            raise NativeCompilationError("unqualified_target", "interlayer pack branch word changed")
        advances = [row["fields"] for row in rows[first_load:branch]
                    if row["operation"] == "atlas.alu_imm"]
        if [(row.get("kind"), row.get("dst"), row.get("src"), row.get("immediate"))
            for row in advances] != [
                ("addi", 13, 13, 1), ("addi", 10, 10, 16),
                ("addi", 11, 11, 16), ("addi", 12, 12, 32),
        ]:
            raise NativeCompilationError("unqualified_target", "interlayer pack row increments changed")


def _check_mxu_dag_timing(program: dict) -> None:
    """Check conservative data availability against the emitted issue stream."""
    cycle = 0
    ready_at: dict[int, int] = {}
    weight_ready: int | None = None
    matmul_ready: int | None = None
    pending_stores: list[int] = []
    pack_store_due = False
    pack_scratch_ready: int | None = None
    for row in program["instructions"]:
        op, fields = row["operation"], row["fields"]
        if op == "atlas.vload":
            ready_at[fields["dst"]] = cycle + 34
        elif op == "atlas.mxu_push":
            if ready_at.get(fields["src"], 1 << 60) > cycle:
                raise NativeCompilationError("unqualified_target", "MXU weight is not available")
            weight_ready = cycle + 1
        elif op == "atlas.mxu_matmul":
            if (ready_at.get(fields["src"], 1 << 60) > cycle
                    or weight_ready is None or weight_ready > cycle):
                raise NativeCompilationError("unqualified_target", "MXU operand is not available")
            matmul_ready = cycle + 64
        elif op == "atlas.mxu_pop":
            if matmul_ready is None or matmul_ready > cycle:
                raise NativeCompilationError("unqualified_target", "MXU readout is premature")
            ready_at[fields["dst"]] = cycle + 32
            ready_at[fields["dst"] + 1] = cycle + 32
        elif op == "atlas.vpu_unary":
            pair = fields["src"]
            if any(ready_at.get(pair + half, 1 << 60) > cycle for half in (0, 1)):
                raise NativeCompilationError("unqualified_target", "VPU unary reads an unavailable pair")
            ready_at[fields["dst"]] = cycle + 66
            ready_at[fields["dst"] + 1] = cycle + 66
        elif op == "atlas.vpu_reduce":
            pair = fields["src"]
            if any(ready_at.get(pair + half, 1 << 60) > cycle for half in (0, 1)):
                raise NativeCompilationError("unqualified_target", "VPU row reduction reads an unavailable pair")
            completion = 39 if fields["kind"] == "row_sum" else 34
            ready_at[fields["dst"]] = cycle + completion
            ready_at[fields["dst"] + 1] = cycle + completion
        elif op == "atlas.vpu_binary":
            if any(ready_at.get(pair + half, 1 << 60) > cycle
                   for pair in (fields["lhs"], fields["rhs"]) for half in (0, 1)):
                raise NativeCompilationError("unqualified_target", "VPU binary reads an unavailable pair")
            ready_at[fields["dst"]] = cycle + 66
            ready_at[fields["dst"] + 1] = cycle + 66
        elif op == "atlas.vpu_pack":
            pair = fields["src"]
            if any(ready_at.get(pair + half, 1 << 60) > cycle for half in (0, 1)):
                raise NativeCompilationError("unqualified_target", "pack reads an unavailable BF16 pair")
            ready_at[fields["dst"]] = cycle + 65
            pack_store_due = True
        elif op == "atlas.xlu_transpose":
            if ready_at.get(fields["src"], 1 << 60) > cycle:
                raise NativeCompilationError("unqualified_target", "XLU reads an unavailable FP8 register")
            ready_at[fields["dst"]] = cycle + 65
        elif op == "atlas.vstore":
            if ready_at.get(fields["src"], 1 << 60) > cycle:
                raise NativeCompilationError("unqualified_target", "VPU store reads an unavailable register")
            if pack_store_due:
                pack_scratch_ready = cycle + 34
                pack_store_due = False
            else:
                pending_stores.append(cycle)
        elif op == "atlas.scalar_load" and fields.get("kind") == "lw":
            if pack_scratch_ready is None or pack_scratch_ready > cycle:
                raise NativeCompilationError("unqualified_target", "pack row read precedes VSTORE completion")
        elif op == "atlas.dma" and fields["direction"] == "store":
            if not pending_stores or cycle - pending_stores.pop(0) < 34:
                raise NativeCompilationError("unqualified_target", "DMA reads scratch before VSTORE completes")
        elif op == "atlas.delay":
            count = fields.get("cycles")
            if type(count) is not int or not 0 <= count <= 65535:
                raise NativeCompilationError("unqualified_target", "MXU/VPU DAG delay is invalid")
            cycle += count
        cycle += 1
    if pending_stores or pack_store_due:
        raise NativeCompilationError("unqualified_target", "VPU scratch store has no DMA completion")


class AtlasNativeBinding:
    """Selected RTL facts live here; Merlin owns the search algorithms."""

    @staticmethod
    def prepare_ordered_mn(request: KernelRequest, tile_m: int, tile_n: int):
        """Qualify independent MXU0 output tiles and positive-zero tail rows."""
        from merlin.semantic_compiler.ordered_mn_tiling import (
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
        from merlin.semantic_compiler.ordered_k_tiling import (
            OrderedKTilingError,
            partition_ordered_matmul_k,
        )

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
    def prepare_precision_graph(source: str, *, entry: str, policy: dict[str, object]):
        """Apply the selected diagnostic graph precision policy to parsed Linalg."""
        from .graph_precision import lower_diagnostic_graph

        return lower_diagnostic_graph(source, entry=entry, policy=policy)

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

    @staticmethod
    def profile() -> NativeTargetProfile:
        def signature(dtype: str) -> dict:
            return {
                "output_dtype": dtype, "numerical_policy": "raw-bit-pattern", "ranks": (2,),
                "input_dtypes": (dtype,), "input_numerical_policies": ("raw-bit-pattern",),
                "input_ranks": (2,),
                "output_axis_bounds": (AxisBound(0, 32, 32), AxisBound(1, 32, 32)),
                "input_axis_bounds": ((AxisBound(0, 32, 32), AxisBound(1, 32, 32)),),
            }

        descriptors = (
            InstructionDescriptor("atlas.load_bf16_tile", "identity", ("external",),
                                  "tensor_bf16", extent=2, value_preserving_copy=True,
                                  **signature("bf16")),
            InstructionDescriptor("atlas.load_bf16_tile_mixed", "identity", ("external",),
                                  "tensor_bf16", extent=2, value_preserving_copy=True,
                                  source_required_attrs=(("input_domain", PACK_ATTRS[0][1]),),
                                  validity=(AddressConstraint("at_least", "out", value=32),
                                            AddressConstraint("at_most", "out", value=32)),
                                  **signature("bf16")),
            InstructionDescriptor("atlas.vpu_relu_bf16_tile", "relu", ("tensor_bf16",),
                                  "tensor_bf16", extent=2, required_attrs=BF16_RELU_ATTRS,
                                  index_maps=pointwise_identity_maps(2, 1),
                                  input_read_offsets=(32,), completion_offset=65,
                                  **signature("bf16")),
            InstructionDescriptor("atlas.vpu_exp_bf16_tile", "exp_bf16", ("tensor_bf16",),
                                  "tensor_bf16", extent=2, required_attrs=BF16_EXP_ATTRS,
                                  index_maps=pointwise_identity_maps(2, 1),
                                  input_read_offsets=(32,), completion_offset=65,
                                  input_policy_alternatives=((MXU0_POLICY,),),
                                  **signature("bf16")),
            InstructionDescriptor("atlas.vpu_recip_bf16_tile", "recip_bf16", ("tensor_bf16",),
                                  "tensor_bf16", extent=2, required_attrs=BF16_RECIP_ATTRS,
                                  index_maps=pointwise_identity_maps(2, 1),
                                  input_read_offsets=(32,), completion_offset=65,
                                  input_policy_alternatives=((MXU0_POLICY,),),
                                  **signature("bf16")),
            InstructionDescriptor(
                "atlas.vpu_row_max_bf16_tile", "row_max_bf16", ("tensor_bf16",),
                "tensor_bf16", extent=2, required_attrs=BF16_ROW_MAX_ATTRS,
                index_maps=_row_reduce_maps(), input_read_offsets=(32,), completion_offset=65,
                input_policy_alternatives=((MXU0_POLICY,),), **signature("bf16"),
            ),
            InstructionDescriptor(
                "atlas.vpu_row_sum_bf16_tile", "row_sum_bf16", ("tensor_bf16",),
                "tensor_bf16", extent=2, required_attrs=BF16_ROW_SUM_ATTRS,
                index_maps=_row_reduce_maps(), input_read_offsets=(32,), completion_offset=65,
                input_policy_alternatives=((MXU0_POLICY,),), **signature("bf16"),
            ),
            InstructionDescriptor(
                "atlas.vpu_add_bf16_tile", "add_bf16", ("tensor_bf16", "tensor_bf16"),
                "tensor_bf16", "bf16", "raw-bit-pattern", (2,), extent=2,
                input_dtypes=("bf16", "bf16"),
                input_numerical_policies=("raw-bit-pattern", "raw-bit-pattern"),
                input_policy_alternatives=((MXU0_POLICY,), (MXU0_POLICY,)),
                input_ranks=(2, 2), required_attrs=BF16_ADD_ATTRS,
                index_maps=pointwise_identity_maps(2, 2),
                input_read_offsets=(32, 32), completion_offset=65,
                output_axis_bounds=(AxisBound(0, 32, 32), AxisBound(1, 32, 32)),
                input_axis_bounds=(
                    (AxisBound(0, 32, 32), AxisBound(1, 32, 32)),
                    (AxisBound(0, 32, 32), AxisBound(1, 32, 32)),
                ),
            ),
            InstructionDescriptor(
                "atlas.vpu_sub_bf16_tile", "sub_bf16", ("tensor_bf16", "tensor_bf16"),
                "tensor_bf16", "bf16", "raw-bit-pattern", (2,), extent=2,
                input_dtypes=("bf16", "bf16"),
                input_numerical_policies=("raw-bit-pattern", "raw-bit-pattern"),
                input_policy_alternatives=((MXU0_POLICY,), (MXU0_POLICY,)),
                input_ranks=(2, 2), required_attrs=BF16_SUB_ATTRS,
                index_maps=pointwise_identity_maps(2, 2),
                input_read_offsets=(32, 32), completion_offset=65,
                output_axis_bounds=(AxisBound(0, 32, 32), AxisBound(1, 32, 32)),
                input_axis_bounds=(
                    (AxisBound(0, 32, 32), AxisBound(1, 32, 32)),
                    (AxisBound(0, 32, 32), AxisBound(1, 32, 32)),
                ),
            ),
            InstructionDescriptor(
                "atlas.vpu_mul_bf16_tile", "mul_bf16", ("tensor_bf16", "tensor_bf16"),
                "tensor_bf16", "bf16", "raw-bit-pattern", (2,), extent=2,
                input_dtypes=("bf16", "bf16"),
                input_numerical_policies=("raw-bit-pattern", "raw-bit-pattern"),
                input_policy_alternatives=((MXU0_POLICY,), (MXU0_POLICY,)),
                input_ranks=(2, 2), required_attrs=BF16_MUL_ATTRS,
                index_maps=pointwise_identity_maps(2, 2),
                input_read_offsets=(32, 32), completion_offset=65,
                output_axis_bounds=(AxisBound(0, 32, 32), AxisBound(1, 32, 32)),
                input_axis_bounds=(
                    (AxisBound(0, 32, 32), AxisBound(1, 32, 32)),
                    (AxisBound(0, 32, 32), AxisBound(1, 32, 32)),
                ),
            ),
            InstructionDescriptor(
                "atlas.vpu_relu_bf16_from_mxu0_tile", "relu", ("tensor_bf16",),
                "tensor_bf16", "bf16", "raw-bit-pattern", (2,), extent=2,
                input_dtypes=("bf16",), input_numerical_policies=(MXU0_POLICY,),
                input_ranks=(2,), required_attrs=BF16_RELU_ATTRS,
                index_maps=pointwise_identity_maps(2, 1),
                input_read_offsets=(32,), completion_offset=65,
                output_axis_bounds=(AxisBound(0, 32, 32), AxisBound(1, 32, 32)),
                input_axis_bounds=((AxisBound(0, 32, 32), AxisBound(1, 32, 32)),),
            ),
            InstructionDescriptor("atlas.store_bf16_tile", "identity", ("tensor_bf16",),
                                  "external", value_preserving_copy=True,
                                  **signature("bf16")),
            InstructionDescriptor("atlas.load_fp8_tile", "identity", ("external",),
                                  "tensor_fp8", value_preserving_copy=True,
                                  input_layouts=(FP8_DRAM_TILE_LAYOUT,),
                                  output_layout=FP8_TENSOR_TILE_LAYOUT,
                                  **signature("fp8_e4m3")),
            InstructionDescriptor("atlas.xlu_transpose_fp8", "transpose", ("tensor_fp8",),
                                  "tensor_fp8", index_maps=_transpose_maps(),
                                  input_layouts=(FP8_TENSOR_TILE_LAYOUT,),
                                  output_layout=FP8_TENSOR_TILE_LAYOUT,
                                  **signature("fp8_e4m3")),
            InstructionDescriptor(
                "atlas.xlu_transpose_unit_fp8_tile", "transpose", ("tensor_fp8",),
                "tensor_fp8", "fp8_e4m3", PACK_POLICY, (2,),
                input_dtypes=("fp8_e4m3",), input_numerical_policies=(PACK_POLICY,),
                input_ranks=(2,), index_maps=_transpose_maps(),
                input_read_offsets=(32,), completion_offset=65,
                output_axis_bounds=(AxisBound(0, 32, 32), AxisBound(1, 32, 32)),
                input_axis_bounds=((AxisBound(0, 32, 32), AxisBound(1, 32, 32)),),
            ),
            InstructionDescriptor(
                "atlas.xlu_transpose_scaled_fp8_tile", "transpose", ("tensor_fp8",),
                "tensor_fp8", "fp8_e4m3", SCALED_PACK_POLICY, (2,),
                input_dtypes=("fp8_e4m3",), input_numerical_policies=(SCALED_PACK_POLICY,),
                input_ranks=(2,), index_maps=_transpose_maps(),
                input_read_offsets=(32,), completion_offset=65,
                output_axis_bounds=(AxisBound(0, 32, 32), AxisBound(1, 32, 32)),
                input_axis_bounds=((AxisBound(0, 32, 32), AxisBound(1, 32, 32)),),
            ),
            InstructionDescriptor(
                "atlas.xlu_transpose_rounded_fp8_tile", "transpose", ("tensor_fp8",),
                "tensor_fp8", "fp8_e4m3", ROUNDED_PACK_POLICY, (2,),
                input_dtypes=("fp8_e4m3",), input_numerical_policies=(ROUNDED_PACK_POLICY,),
                input_ranks=(2,), index_maps=_transpose_maps(),
                input_read_offsets=(32,), completion_offset=65,
                output_axis_bounds=(AxisBound(0, 32, 32), AxisBound(1, 32, 32)),
                input_axis_bounds=((AxisBound(0, 32, 32), AxisBound(1, 32, 32)),),
            ),
            InstructionDescriptor("atlas.store_fp8_tile", "identity", ("tensor_fp8",),
                                  "external", value_preserving_copy=True,
                                  input_layouts=(FP8_TENSOR_TILE_LAYOUT,),
                                  output_layout=FP8_DRAM_TILE_LAYOUT,
                                  **signature("fp8_e4m3")),
            InstructionDescriptor(
                "atlas.vpu_pack_unit_fp8_tile", "pack_fp8", ("tensor_bf16",),
                "tensor_fp8", "fp8_e4m3", PACK_POLICY, (2,),
                input_dtypes=("bf16",), input_numerical_policies=("raw-bit-pattern",),
                input_policy_alternatives=((MXU0_POLICY,),),
                input_ranks=(2,), required_attrs=PACK_ATTRS,
                index_maps=pointwise_identity_maps(2, 1),
                input_read_offsets=(34,), completion_offset=65,
                validity=(AddressConstraint("at_least", "in0", value=32),
                          AddressConstraint("at_most", "in0", value=60)),
                output_axis_bounds=(AxisBound(0, 32, 32), AxisBound(1, 32, 32)),
                input_axis_bounds=((AxisBound(0, 32, 32), AxisBound(1, 32, 32)),),
            ),
            InstructionDescriptor(
                "atlas.store_fp8_unit_tile", "identity", ("tensor_fp8",),
                "external", "fp8_e4m3", PACK_POLICY, (2,),
                input_dtypes=("fp8_e4m3",), input_numerical_policies=(PACK_POLICY,),
                input_ranks=(2,), value_preserving_copy=True,
                output_axis_bounds=(AxisBound(0, 32, 32), AxisBound(1, 32, 32)),
                input_axis_bounds=((AxisBound(0, 32, 32), AxisBound(1, 32, 32)),),
            ),
            InstructionDescriptor(
                "atlas.vpu_pack_scaled_fp8_tile", "pack_fp8", ("tensor_bf16",),
                "tensor_fp8", "fp8_e4m3", SCALED_PACK_POLICY, (2,),
                input_dtypes=("bf16",), input_numerical_policies=("raw-bit-pattern",),
                input_policy_alternatives=((MXU0_POLICY,),),
                input_ranks=(2,), required_attrs=(("input_domain", SCALED_PACK_DOMAIN),),
                parameter_attrs=(IntegerAttributeBound("scale_code", *SCALED_PACK_RANGE),),
                index_maps=pointwise_identity_maps(2, 1),
                input_read_offsets=(34,), completion_offset=65,
                validity=(AddressConstraint("at_least", "in0", value=32),
                          AddressConstraint("at_most", "in0", value=60)),
                output_axis_bounds=(AxisBound(0, 32, 32), AxisBound(1, 32, 32)),
                input_axis_bounds=((AxisBound(0, 32, 32), AxisBound(1, 32, 32)),),
            ),
            InstructionDescriptor(
                "atlas.vpu_pack_rounded_fp8_tile", "pack_fp8", ("tensor_bf16",),
                "tensor_fp8", "fp8_e4m3", ROUNDED_PACK_POLICY, (2,),
                input_dtypes=("bf16",), input_numerical_policies=("raw-bit-pattern",),
                input_policy_alternatives=((MXU0_POLICY,),),
                input_ranks=(2,), required_attrs=(("input_domain", ROUNDED_PACK_DOMAIN),),
                parameter_attrs=(IntegerAttributeBound("scale_code", *SCALED_PACK_RANGE),),
                index_maps=pointwise_identity_maps(2, 1),
                input_read_offsets=(34,), completion_offset=65,
                validity=(AddressConstraint("at_least", "in0", value=32),
                          AddressConstraint("at_most", "in0", value=60)),
                output_axis_bounds=(AxisBound(0, 32, 32), AxisBound(1, 32, 32)),
                input_axis_bounds=((AxisBound(0, 32, 32), AxisBound(1, 32, 32)),),
            ),
            InstructionDescriptor(
                "atlas.store_fp8_scaled_tile", "identity", ("tensor_fp8",),
                "external", "fp8_e4m3", SCALED_PACK_POLICY, (2,),
                input_dtypes=("fp8_e4m3",), input_numerical_policies=(SCALED_PACK_POLICY,),
                input_ranks=(2,), value_preserving_copy=True,
                output_axis_bounds=(AxisBound(0, 32, 32), AxisBound(1, 32, 32)),
                input_axis_bounds=((AxisBound(0, 32, 32), AxisBound(1, 32, 32)),),
            ),
            InstructionDescriptor(
                "atlas.store_fp8_rounded_tile", "identity", ("tensor_fp8",),
                "external", "fp8_e4m3", ROUNDED_PACK_POLICY, (2,),
                input_dtypes=("fp8_e4m3",), input_numerical_policies=(ROUNDED_PACK_POLICY,),
                input_ranks=(2,), value_preserving_copy=True,
                output_axis_bounds=(AxisBound(0, 32, 32), AxisBound(1, 32, 32)),
                input_axis_bounds=((AxisBound(0, 32, 32), AxisBound(1, 32, 32)),),
            ),
            InstructionDescriptor(
                "atlas.mxu0_reset_tile", "matmul", ("tensor_fp8", "tensor_fp8"),
                "tensor_bf16", "bf16", MXU0_POLICY, (2,), extent=2,
                input_layouts=(FP8_TENSOR_TILE_LAYOUT, FP8_TENSOR_TILE_LAYOUT),
                output_layout=BF16_TENSOR_PAIR_LAYOUT,
                input_dtypes=("fp8_e4m3", "fp8_e4m3"),
                input_numerical_policies=("raw-bit-pattern", "raw-bit-pattern"),
                input_policy_alternatives=((PACK_POLICY, SCALED_PACK_POLICY, ROUNDED_PACK_POLICY),
                                           (PACK_POLICY, SCALED_PACK_POLICY, ROUNDED_PACK_POLICY)),
                input_ranks=(2, 2),
                output_axis_bounds=(AxisBound(0, 32, 32), AxisBound(1, 32, 32)),
                input_axis_bounds=(
                    (AxisBound(0, 32, 32), AxisBound(1, 32, 32)),
                    (AxisBound(0, 32, 32), AxisBound(1, 32, 32)),
                ),
                shape_contract="bounded", index_maps=_mxu0_maps(),
                # The OOT allocator may choose any even BF16 pair in the
                # shared register file after considering later VPU uses.
                validity=(AddressConstraint("at_least", "out", value=32),
                          AddressConstraint("at_most", "out", value=60)),
            ),
            InstructionDescriptor(
                "atlas.mxu0_reset_acc_tile", "matmul", ("tensor_fp8", "tensor_fp8"),
                "tensor_mxu0_acc", "bf16", MXU0_POLICY, (2,),
                input_layouts=(FP8_TENSOR_TILE_LAYOUT, FP8_TENSOR_TILE_LAYOUT),
                output_layout=MXU0_ACC_TILE_LAYOUT,
                input_dtypes=("fp8_e4m3", "fp8_e4m3"),
                input_numerical_policies=("raw-bit-pattern", "raw-bit-pattern"),
                input_ranks=(2, 2),
                output_axis_bounds=(AxisBound(0, 32, 32), AxisBound(1, 32, 32)),
                input_axis_bounds=(
                    (AxisBound(0, 32, 32), AxisBound(1, 32, 32)),
                    (AxisBound(0, 32, 32), AxisBound(1, 32, 32)),
                ),
                shape_contract="bounded", index_maps=_mxu0_maps(),
                input_read_offsets=(32, 32), completion_offset=96,
            ),
            InstructionDescriptor(
                "atlas.mxu0_accumulate_tile", "matmul_accumulate",
                ("tensor_fp8", "tensor_fp8", "tensor_mxu0_acc"),
                "tensor_mxu0_acc", "bf16", MXU0_POLICY, (2,),
                input_layouts=(FP8_TENSOR_TILE_LAYOUT, FP8_TENSOR_TILE_LAYOUT,
                               MXU0_ACC_TILE_LAYOUT),
                output_layout=MXU0_ACC_TILE_LAYOUT,
                input_dtypes=("fp8_e4m3", "fp8_e4m3", "bf16"),
                input_numerical_policies=("raw-bit-pattern", "raw-bit-pattern", MXU0_POLICY),
                input_ranks=(2, 2, 2),
                output_axis_bounds=(AxisBound(0, 32, 32), AxisBound(1, 32, 32)),
                input_axis_bounds=(
                    (AxisBound(0, 32, 32), AxisBound(1, 32, 32)),
                    (AxisBound(0, 32, 32), AxisBound(1, 32, 32)),
                    (AxisBound(0, 32, 32), AxisBound(1, 32, 32)),
                ),
                shape_contract="bounded", index_maps=_mxu0_accumulate_maps(),
                validity=(AddressConstraint("eq_offset", "out", "in2", 0),),
                input_read_offsets=(32, 32, 94), completion_offset=96,
                in_place_inputs=(2,),
            ),
            InstructionDescriptor(
                "atlas.store_bf16_mxu0_acc_tile", "identity", ("tensor_mxu0_acc",),
                "external", "bf16", MXU0_POLICY, (2,),
                input_layouts=(MXU0_ACC_TILE_LAYOUT,),
                output_layout=BF16_DRAM_PAIR_LAYOUT,
                input_dtypes=("bf16",), input_numerical_policies=(MXU0_POLICY,),
                input_ranks=(2,), value_preserving_copy=True,
                output_axis_bounds=(AxisBound(0, 32, 32), AxisBound(1, 32, 32)),
                input_axis_bounds=((AxisBound(0, 32, 32), AxisBound(1, 32, 32)),),
                input_read_offsets=(32,), completion_offset=64,
            ),
            InstructionDescriptor(
                "atlas.store_bf16_mxu0_tile", "identity", ("tensor_bf16",),
                "external", "bf16", MXU0_POLICY, (2,),
                input_layouts=(BF16_TENSOR_PAIR_LAYOUT,),
                output_layout=BF16_DRAM_PAIR_LAYOUT,
                input_dtypes=("bf16",), input_numerical_policies=(MXU0_POLICY,),
                input_ranks=(2,),
                output_axis_bounds=(AxisBound(0, 32, 32), AxisBound(1, 32, 32)),
                input_axis_bounds=((AxisBound(0, 32, 32), AxisBound(1, 32, 32)),),
                value_preserving_copy=True,
            ),
            InstructionDescriptor(
                "atlas.mxu1_reset_tile", "matmul", ("tensor_fp8", "tensor_fp8"),
                "tensor_bf16", "bf16", MXU1_POLICY, (2,), extent=2,
                input_dtypes=("fp8_e4m3", "fp8_e4m3"),
                input_numerical_policies=("raw-bit-pattern", "raw-bit-pattern"),
                input_ranks=(2, 2),
                output_axis_bounds=(AxisBound(0, 32, 32), AxisBound(1, 32, 32)),
                input_axis_bounds=(
                    (AxisBound(0, 32, 32), AxisBound(1, 32, 32)),
                    (AxisBound(0, 32, 32), AxisBound(1, 32, 32)),
                ),
                shape_contract="bounded", index_maps=_mxu0_maps(),
                validity=(AddressConstraint("at_least", "out", value=32),
                          AddressConstraint("at_most", "out", value=60)),
            ),
            InstructionDescriptor(
                "atlas.store_bf16_mxu1_tile", "identity", ("tensor_bf16",),
                "external", "bf16", MXU1_POLICY, (2,),
                input_dtypes=("bf16",), input_numerical_policies=(MXU1_POLICY,),
                input_ranks=(2,),
                output_axis_bounds=(AxisBound(0, 32, 32), AxisBound(1, 32, 32)),
                input_axis_bounds=((AxisBound(0, 32, 32), AxisBound(1, 32, 32)),),
                value_preserving_copy=True,
            ),
            InstructionDescriptor(
                "atlas.mxu1_finite_reset_tile", "matmul", ("tensor_fp8", "tensor_fp8"),
                "tensor_bf16", "bf16", MXU1_FINITE_POLICY, (2,), extent=2,
                input_dtypes=("fp8_e4m3", "fp8_e4m3"),
                input_numerical_policies=("raw-bit-pattern", "raw-bit-pattern"),
                input_ranks=(2, 2),
                output_axis_bounds=(AxisBound(0, 32, 32), AxisBound(1, 32, 32)),
                input_axis_bounds=(
                    (AxisBound(0, 32, 32), AxisBound(1, 32, 32)),
                    (AxisBound(0, 32, 32), AxisBound(1, 32, 32)),
                ),
                shape_contract="bounded", index_maps=_mxu0_maps(),
                validity=(AddressConstraint("at_least", "out", value=32),
                          AddressConstraint("at_most", "out", value=60)),
            ),
            InstructionDescriptor(
                "atlas.store_bf16_mxu1_finite_tile", "identity", ("tensor_bf16",),
                "external", "bf16", MXU1_FINITE_POLICY, (2,),
                input_dtypes=("bf16",), input_numerical_policies=(MXU1_FINITE_POLICY,),
                input_ranks=(2,),
                output_axis_bounds=(AxisBound(0, 32, 32), AxisBound(1, 32, 32)),
                input_axis_bounds=((AxisBound(0, 32, 32), AxisBound(1, 32, 32)),),
                value_preserving_copy=True,
            ),
        )
        banks = (
            StorageBank("external", "selected_dram_window", DRAM_SLOTS, "2048_byte_tile"),
            StorageBank("tensor_bf16", "tensor_register_file", 62,
                        "1024_byte_register", alignment=2),
            StorageBank("tensor_fp8", "tensor_register_file", 32,
                        "1024_byte_register"),
            StorageBank("tensor_mxu0_acc", "mxu0_accumulator_slots", 1,
                        "one_accumulator_slot"),
        )
        return NativeTargetProfile(TARGET_IDENTITY, descriptors, banks)

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
        pipeline = _native_pipeline()
        if request.lowering_policy != "diagnostic":
            raise NativeCompilationError(
                "unqualified_target", "selected FPGA bitstream and host coherence are not qualified"
            )
        inventory_path = target_source / "docs/selected-variant-inventory.json"
        try:
            inventory = json.loads(inventory_path.read_text())
        except (OSError, ValueError) as exc:
            raise NativeCompilationError("tool_unavailable", "selected Atlas source inventory is unavailable") from exc
        if inventory.get("selected_sources", {}).get("rtl_revision") != RTL_REVISION:
            raise NativeCompilationError("unqualified_target", "Atlas source revision differs from selected RTL")
        if request.target_identity != TARGET_IDENTITY:
            raise NativeCompilationError("unsupported_semantics", "tile needs the selected target")
        if (
            not (1 <= len(request.input_storages) <= DRAM_SLOTS)
            or not (1 <= len(request.outputs) <= DRAM_SLOTS)
            or any(storage != "external" for _, storage in request.input_storages)
            or request.output_storages != ("external",) * len(request.outputs)
        ):
            raise NativeCompilationError("unsupported_semantics", "tile needs DRAM inputs and outputs")
        bf16_tile = _bf16_tile_request(request)
        mxu_k_chain = _mxu0_k_chain_request(request)
        mxu_binary = _mxu_bf16_binary_request(request)
        mxu_dag = _mxu_bf16_dag_request(request) and not mxu_binary
        if request.constants and not mxu_dag:
            raise NativeCompilationError("unsupported_semantics", "declared constants need an admitted BF16 MXU/VPU DAG")
        mxu_inputs = {node.id: node.inputs for node in request.nodes if node.op == "matmul"}
        transpose_inputs = {
            node.id: node.inputs[0] for node in request.nodes
            if mxu_dag and node.op == "transpose"
        }
        pack_scale_codes = {
            node.id: code for node in request.nodes if node.op == "pack_fp8"
            if (code := _interlayer_pack_code(node)) is not None
        } if mxu_dag else {}
        transpose = (
            len(request.nodes) == 2 and request.nodes[0].op == "input"
            and request.nodes[1].op == "transpose"
            and request.nodes[1].inputs == (request.nodes[0].id,)
            and request.outputs == (request.nodes[1].id,)
            and request.nodes[1].index_maps == _transpose_maps()
            and all(node.type.shape == (32, 32) and node.type.dtype == "fp8_e4m3"
                    and node.type.numerical_policy == "raw-bit-pattern"
                    and not node.attrs for node in request.nodes)
            and not request.nodes[0].index_maps
        )
        copy_transpose = (
            len(request.nodes) == 3
            and tuple(node.op for node in request.nodes) == ("input", "identity", "transpose")
            and request.nodes[1].inputs == request.nodes[2].inputs == (request.nodes[0].id,)
            and request.outputs == (request.nodes[1].id, request.nodes[2].id)
            and request.nodes[2].index_maps == _transpose_maps()
            and all(node.type.shape == (32, 32) and node.type.dtype == "fp8_e4m3"
                    and node.type.numerical_policy == "raw-bit-pattern"
                    and not node.attrs for node in request.nodes)
            and not request.nodes[0].index_maps and not request.nodes[1].index_maps
        )
        scaled_code = _scaled_pack_code(request.nodes[1]) if len(request.nodes) == 2 else None
        pack_domain = dict(request.nodes[1].attrs).get("input_domain") if scaled_code is not None else None
        allowed_pack_inputs = (
            (("input_domain", pack_domain),),
        ) if scaled_code is not None else ((), (("input_domain", PACK_ATTRS[0][1]),))
        direct_pack = (
            len(request.nodes) == 2
            and request.nodes[0].op == "input"
            and request.nodes[0].type.shape == (32, 32)
            and request.nodes[0].type.dtype == "bf16"
            and request.nodes[0].type.numerical_policy == "raw-bit-pattern"
            and request.nodes[0].attrs in allowed_pack_inputs
            and not request.nodes[0].index_maps
            and request.nodes[1].op == "pack_fp8"
            and request.nodes[1].inputs == (request.nodes[0].id,)
            and request.nodes[1].type.shape == (32, 32)
            and request.nodes[1].type.dtype == "fp8_e4m3"
            and (
                (scaled_code is not None and request.nodes[1].type.numerical_policy in
                 {SCALED_PACK_POLICY, ROUNDED_PACK_POLICY})
                or (request.nodes[1].type.numerical_policy == PACK_POLICY and request.nodes[1].attrs == PACK_ATTRS)
            )
            and request.nodes[1].index_maps == pointwise_identity_maps(2, 1)
            and request.outputs == (request.nodes[1].id,)
        )
        relu_pack = (
            len(request.nodes) == 3
            and tuple(node.op for node in request.nodes) == ("input", "relu", "pack_fp8")
            and request.nodes[0].type.shape == (32, 32)
            and request.nodes[0].type.dtype == "bf16"
            and request.nodes[0].type.numerical_policy == "raw-bit-pattern"
            and request.nodes[0].attrs == (("input_domain", PACK_ATTRS[0][1]),)
            and not request.nodes[0].index_maps
            and request.nodes[1].inputs == (request.nodes[0].id,)
            and request.nodes[1].type == request.nodes[0].type
            and request.nodes[1].attrs == BF16_RELU_ATTRS
            and request.nodes[1].index_maps == pointwise_identity_maps(2, 1)
            and request.nodes[2].inputs == (request.nodes[1].id,)
            and request.nodes[2].type.shape == (32, 32)
            and request.nodes[2].type.dtype == "fp8_e4m3"
            and request.nodes[2].type.numerical_policy == PACK_POLICY
            and request.nodes[2].attrs == PACK_ATTRS
            and request.nodes[2].index_maps == pointwise_identity_maps(2, 1)
            and request.outputs == (request.nodes[2].id,)
        )
        # The seven admitted BF16 codes are closed under sign-clamp ReLU:
        # each negative code becomes positive zero, and positives are retained.
        pack = direct_pack or relu_pack
        pack_scale_code = scaled_code if direct_pack and scaled_code is not None else 127
        simple_mxu_mode = (
            {
                MXU0_POLICY: (0, MXU0_INPUT_DOMAIN),
                MXU1_POLICY: (1, MXU1_INPUT_DOMAIN),
                MXU1_FINITE_POLICY: (1, MXU1_FINITE_INPUT_DOMAIN),
            }.get(request.nodes[2].type.numerical_policy)
            if len(request.nodes) >= 3 else None
        )
        mxu = (
            len(request.nodes) in (3, 4)
            and simple_mxu_mode is not None
            and tuple(node.op for node in request.nodes[:3]) == ("input", "input", "matmul")
            and request.nodes[2].inputs == (request.nodes[0].id, request.nodes[1].id)
            and request.nodes[2].index_maps == _mxu0_maps()
            and all(node.type.shape == (32, 32) for node in request.nodes[:3])
            and all(node.type.dtype == "fp8_e4m3" and node.type.numerical_policy == "raw-bit-pattern"
                    and node.attrs == (("input_domain", simple_mxu_mode[1]),)
                    and node.effect == "input" and not node.index_maps
                    for node in request.nodes[:2])
            and request.nodes[2].type.dtype == "bf16"
            and request.nodes[2].effect == "pure" and not request.nodes[2].attrs
        )
        mxu_relu = mxu and simple_mxu_mode[0] == 0 and len(request.nodes) == 4
        if mxu_relu:
            relu = request.nodes[3]
            mxu_relu = (
                relu.op == "relu" and relu.inputs == (request.nodes[2].id,)
                and relu.type.shape == (32, 32) and relu.type.dtype == "bf16"
                and relu.type.numerical_policy == "raw-bit-pattern"
                and relu.effect == "pure" and relu.attrs == BF16_RELU_ATTRS
                and relu.index_maps == pointwise_identity_maps(2, 1)
            )
        mxu = mxu and (
            (len(request.nodes) == 3 and request.outputs == (request.nodes[2].id,))
            or (mxu_relu and request.outputs == (request.nodes[3].id,))
        )
        mxu_unit = simple_mxu_mode[0] if mxu else None
        if not (bf16_tile or transpose or copy_transpose or mxu or mxu_k_chain
                or mxu_binary or mxu_dag or pack):
            raise NativeCompilationError(
                "unsupported_semantics", "only bounded BF16/FP8 tile graphs and admitted MXU tile chains are supported"
            )
        source_ids = tuple(node.id for node in request.nodes if node.op in {"input", "constant"})
        if set(fixed_inputs) != set(source_ids) or len(fixed_outputs) != len(request.outputs):
            raise NativeCompilationError("compile_error", "tile ABI must bind every input, constant, and output")
        input_slots = tuple(fixed_inputs[source_id] for source_id in source_ids)
        output_slots = tuple(fixed_outputs)
        output_slot = output_slots[0]
        if any(type(slot) is not int or slot < 0 or slot >= DRAM_SLOTS
               for slot in (*input_slots, *output_slots)) or len({*input_slots, *output_slots}) != len(input_slots) + len(output_slots):
            raise NativeCompilationError("unqualified_target", "tile DRAM slots are invalid or overlap")
        # The BF16 DAG stream checker covers explicitly addressed tile DMA.
        # Other admitted forms still use their indexed ABI and dedicated
        # stream replays, so keep their original placement restriction.
        if not bf16_tile:
            if any(slot != input_slots[0] + index for index, slot in enumerate(input_slots)):
                raise NativeCompilationError("unsupported_semantics", "input tiles need adjacent DRAM slots")
            if any(slot != output_slots[0] + index for index, slot in enumerate(output_slots)):
                raise NativeCompilationError("unsupported_semantics", "output tiles need adjacent DRAM slots")
        input_slot = input_slots[0]

        constant_payloads: dict[str, bytes] = {}
        for binding in request.constants:
            logical = bytes.fromhex(binding.data_hex)
            codes = struct.unpack("<1024H", logical)
            if any(not 1 <= ((code >> 7) & 0xff) <= 254 for code in codes):
                raise NativeCompilationError(
                    "unsupported_semantics", "declared BF16 multiplier constant is outside finite-normal domain"
                )
            constant_payloads[binding.node_id] = pack_bf16_pair_halves(logical)

        materialized: _MaterializedCandidate | None = None

        def prepare_candidate(selected: SearchResult) -> _MaterializedCandidate:
            """Check one native candidate through physical emission without publishing it."""
            assert selected.graph and selected.allocation and selected.rules and selected.candidate
            graph, assignment = selected.graph, selected.allocation.addresses
            if any(value.kind == "projection" for value in graph.values):
                raise NativeCompilationError(
                    "unqualified_target", "Atlas emission has no multi-result instruction binding"
                )
            values = graph.values
            names = [selected.rules.symbols[value.symbol].get("descriptor", {}).get("name")
                     for value in values]
            if bf16_tile:
                selected_registers = _check_bf16_selection(
                    graph, selected.allocation.order, names, assignment,
                    source_ids, input_slots, output_slots,
                )
            elif mxu_k_chain:
                selected_registers, source_registers = _mxu0_k_chain_selection(
                    graph, names, assignment, source_ids, input_slots, output_slot,
                )
            elif mxu_binary:
                ids = _mxu_binary_selection(
                    graph, selected.allocation.order, names, assignment,
                    source_ids, input_slots, output_slot,
                )
                selected_registers = sorted({
                    assignment[ids["activation"]], assignment[ids["weight"]],
                    *(assignment[ids[key]] + half
                      for key in ("mxu", "bf16", "binary") for half in (0, 1)),
                })
            elif mxu_dag:
                semantic_nodes = {node.id: node for node in request.nodes}
                for value in graph.values:
                    if value.kind != "instruction" or value.source_node not in pack_scale_codes:
                        continue
                    source = semantic_nodes[value.source_node]
                    expected_name = PACK_RULE_BY_POLICY[source.type.numerical_policy]
                    attrs = selected.rules.symbols[value.symbol].get("instantiated_attrs")
                    if (names[value.id] != expected_name
                            or (source.type.numerical_policy != PACK_POLICY
                                and attrs != dict(source.attrs))):
                        raise NativeCompilationError("unqualified_target", "selected interlayer pack rule changed")
                selected_registers = _mxu_dag_selection(
                    graph, selected.allocation.order, names, assignment,
                    source_ids, mxu_inputs, input_slots, output_slots,
                    transpose_inputs=transpose_inputs,
                )
            elif relu_pack:
                if (
                    len(values) != 5 or graph.outputs != (4,)
                    or selected.allocation.order != (1, 2, 3, 4)
                    or tuple(value.kind for value in values) != (
                        "input", "instruction", "instruction", "instruction", "instruction"
                    )
                    or names[1:] != [
                        "atlas.load_bf16_tile_mixed", "atlas.vpu_relu_bf16_tile",
                        "atlas.vpu_pack_unit_fp8_tile", "atlas.store_fp8_unit_tile",
                    ]
                    or tuple(value.children for value in values[1:]) != ((0,), (1,), (2,), (3,))
                    or (assignment[0], assignment[4]) != (input_slot, output_slot)
                    or assignment[1] != 32 or assignment[2] != 34 or assignment[3] != 0
                ):
                    raise NativeCompilationError("unqualified_target", "selected ReLU/pack graph or bindings changed")
                selected_registers = [32, 33, 34, 35, 0]
            elif pack:
                if (
                    len(values) != 4 or graph.outputs != (3,)
                    or selected.allocation.order != (1, 2, 3)
                    or tuple(value.kind for value in values) != (
                        "input", "instruction", "instruction", "instruction"
                    )
                    or names[1:] != [
                        "atlas.load_bf16_tile",
                        PACK_RULE_BY_POLICY[request.nodes[1].type.numerical_policy],
                        STORE_RULE_BY_POLICY[request.nodes[1].type.numerical_policy],
                    ]
                    or tuple(value.children for value in values[1:]) != ((0,), (1,), (2,))
                    or (assignment[0], assignment[3]) != (input_slot, output_slot)
                    or assignment[1] != 32 or assignment[2] != 0
                ):
                    raise NativeCompilationError("unqualified_target", "selected pack graph or bindings changed")
                if scaled_code is not None:
                    selected_attrs = selected.rules.symbols[values[2].symbol].get("instantiated_attrs")
                    if (
                        type(selected_attrs) is not dict
                        or type(selected_attrs.get("scale_code")) is not int
                        or selected_attrs != dict(request.nodes[1].attrs)
                    ):
                        raise NativeCompilationError("unqualified_target", "selected pack attributes changed")
                selected_registers = [32, 33, 0]
            elif copy_transpose:
                if (
                    len(values) != 5 or graph.outputs != (2, 4)
                    or selected.allocation.order != (1, 2, 3, 4)
                    or tuple(value.kind for value in values) != (
                        "input", "instruction", "instruction", "instruction", "instruction"
                    )
                    or names[1:] != [
                        "atlas.load_fp8_tile", "atlas.store_fp8_tile",
                        "atlas.xlu_transpose_fp8", "atlas.store_fp8_tile",
                    ]
                    or tuple(value.children for value in values[1:]) != ((0,), (1,), (1,), (3,))
                    or assignment[0] != input_slot
                    or (assignment[2], assignment[4]) != output_slots
                ):
                    raise NativeCompilationError("unqualified_target", "selected copy/transpose graph changed")
                source_reg, result_reg = assignment[1], assignment[3]
                if not (0 <= source_reg < 32 and 0 <= result_reg < 32 and source_reg != result_reg):
                    raise NativeCompilationError("unqualified_target", "native copy/transpose register assignment is invalid")
                selected_registers = [source_reg, result_reg]
            elif transpose:
                if (
                    len(values) != 4 or graph.outputs != (3,)
                    or selected.allocation.order != (1, 2, 3)
                    or tuple(value.kind for value in values) != (
                        "input", "instruction", "instruction", "instruction"
                    )
                    or names[1:] != [
                        "atlas.load_fp8_tile", "atlas.xlu_transpose_fp8", "atlas.store_fp8_tile"
                    ]
                    or tuple(value.children for value in values[1:]) != ((0,), (1,), (2,))
                    or assignment[0] != input_slot or assignment[3] != output_slot
                ):
                    raise NativeCompilationError("unqualified_target", "selected FP8 graph or DRAM binding changed")
                source_reg, result_reg = assignment[1], assignment[2]
                if not (0 <= source_reg < 64 and 0 <= result_reg < 64) or source_reg == result_reg:
                    raise NativeCompilationError("unqualified_target", "native FP8 register assignment is invalid")
                selected_registers = [source_reg, result_reg]
            else:
                assert mxu and mxu_unit is not None
                expected_names = [
                    "atlas.load_fp8_tile", "atlas.load_fp8_tile",
                    ("atlas.mxu1_finite_reset_tile" if request.nodes[2].type.numerical_policy == MXU1_FINITE_POLICY
                     else f"atlas.mxu{mxu_unit}_reset_tile"),
                ]
                if mxu_relu:
                    expected_names.append("atlas.vpu_relu_bf16_from_mxu0_tile")
                expected_names.append(
                    "atlas.store_bf16_tile" if mxu_relu else
                    ("atlas.store_bf16_mxu1_finite_tile"
                     if request.nodes[2].type.numerical_policy == MXU1_FINITE_POLICY
                     else f"atlas.store_bf16_mxu{mxu_unit}_tile")
                )
                expected_children = [(0,), (2,), (1, 3)]
                if mxu_relu:
                    expected_children.append((4,))
                expected_children.append((5,) if mxu_relu else (4,))
                instruction_ids = (1, 3, 4, 5, 6) if mxu_relu else (1, 3, 4, 5)
                if (
                    len(values) != len(instruction_ids) + 2
                    or graph.outputs != (instruction_ids[-1],)
                    or selected.allocation.order != instruction_ids
                    or tuple(value.kind for value in values) != (
                        "input", "instruction", "input", *("instruction",) * (len(instruction_ids) - 1)
                    )
                    or [names[index] for index in instruction_ids] != expected_names
                    or [values[index].children for index in instruction_ids] != expected_children
                    or (assignment[0], assignment[2], assignment[instruction_ids[-1]]) != (
                        input_slots[0], input_slots[1], output_slot
                    )
                ):
                    raise NativeCompilationError("unqualified_target", "selected MXU graph or DRAM binding changed")
                source_reg, weight_reg, pair = assignment[1], assignment[3], assignment[4]
                if not (0 <= source_reg < 32 and 0 <= weight_reg < 32
                        and source_reg != weight_reg and 32 <= pair <= 60 and pair % 2 == 0):
                    raise NativeCompilationError("unqualified_target", "native MXU register assignment is invalid")
                selected_registers = [source_reg, weight_reg, pair, pair + 1]
                if mxu_relu:
                    relu_pair = assignment[5]
                    if relu_pair % 2 or not 0 <= relu_pair <= 60 or relu_pair == pair:
                        raise NativeCompilationError("unqualified_target", "native MXU0 ReLU pair is invalid")
                    selected_registers.extend((relu_pair, relu_pair + 1))

            tool_dir = Path(os.environ.get("ATLAS_OOT_BIN_DIR", target_source / "build/bin"))
            atlas_opt, atlas_emit = tool_dir / "atlas-opt", tool_dir / "atlas-emit"
            if not atlas_opt.is_file() or not atlas_emit.is_file():
                raise NativeCompilationError("tool_unavailable", "Atlas OOT lowerer or emitter is unavailable")
            tool_hashes = (_sha256(atlas_opt.read_bytes()), _sha256(atlas_emit.read_bytes()))
            input_address = DRAM_BASE + input_slot * TILE_BYTES
            output_address = DRAM_BASE + output_slot * TILE_BYTES
            virtual = _virtual_from_selected(
                graph, selected.allocation.order, names, source_ids, input_slots, output_slots,
                pack_scale_code=pack_scale_code,
                pack_scale_codes=pack_scale_codes,
            )
            machine = _run([str(atlas_opt), *NATIVE_PIPELINES["mandatory"], "-"], input_text=virtual)
            _run([str(atlas_opt), *NATIVE_CHECKS, "-"], input_text=machine)
            program = json.loads(_run([str(atlas_emit), "--program-json", "-"], input_text=machine))
            if program.get("selected_rtl_revision") != RTL_REVISION:
                raise NativeCompilationError("unqualified_target", "emitter has a different selected RTL")
            realized_allocation = None
            if bf16_tile:
                assignment = _realized_bf16_assignment(
                    program, graph=graph, order=selected.allocation.order,
                    names=names, selected=assignment,
                )
                selected_registers = _check_bf16_selection(
                    graph, selected.allocation.order, names, assignment,
                    source_ids, input_slots, output_slots,
                )
                check_bf16_tile_stream(
                    program, graph=graph, order=selected.allocation.order, names=names,
                    assignment=assignment, source_ids=source_ids, input_slots=input_slots,
                    output_slots=output_slots,
                )
                realized_allocation = AllocationResult(
                    "feasible", selected.allocation.order, assignment,
                    issue_times=selected.allocation.issue_times,
                )
            elif mxu_k_chain:
                check_mxu0_k_chain_stream(
                    program,
                    input_addresses=tuple(DRAM_BASE + slot * TILE_BYTES for slot in input_slots),
                    output_address=output_address,
                    source_registers=source_registers,
                )
            elif mxu_binary:
                assignment = check_mxu_bf16_binary_stream(
                    program, graph=graph, order=selected.allocation.order,
                    names=names, assignment=assignment, source_ids=source_ids,
                    input_slots=input_slots, output_slot=output_slot,
                )
                ids = _mxu_binary_selection(
                    graph, selected.allocation.order, names, assignment,
                    source_ids, input_slots, output_slot,
                )
                selected_registers = sorted({
                    assignment[ids["activation"]], assignment[ids["weight"]],
                    *(assignment[ids[key]] + half
                      for key in ("mxu", "bf16", "binary") for half in (0, 1)),
                })
                realized_allocation = AllocationResult(
                    "feasible", selected.allocation.order, assignment,
                    issue_times=selected.allocation.issue_times,
                )
            elif mxu_dag:
                assignment = check_mxu_bf16_dag_stream(
                    program, graph=graph, order=selected.allocation.order,
                    names=names, assignment=assignment, source_ids=source_ids,
                    mxu_inputs=mxu_inputs, input_slots=input_slots,
                    output_slots=output_slots, pack_scale_codes=pack_scale_codes,
                    transpose_inputs=transpose_inputs,
                )
                selected_registers = _mxu_dag_selection(
                    graph, selected.allocation.order, names, assignment,
                    source_ids, mxu_inputs, input_slots, output_slots,
                    transpose_inputs=transpose_inputs, physical=True,
                )
                realized_allocation = AllocationResult(
                    "feasible", selected.allocation.order, assignment,
                    issue_times=selected.allocation.issue_times,
                )
            elif pack:
                check_pack_stream(program, input_address=input_address,
                                  output_address=output_address,
                                  source_pair=34 if relu_pack else 32,
                                  relu_pair=(32, 34) if relu_pack else None,
                                  scale_code=pack_scale_code)
            elif copy_transpose:
                check_copy_transpose_stream(
                    program, input_address=input_address,
                    output_addresses=tuple(DRAM_BASE + slot * TILE_BYTES for slot in output_slots),
                    source=source_reg, result=result_reg,
                )
            elif transpose:
                check_transpose_stream(program, input_address=input_address,
                                       output_address=output_address,
                                       source=source_reg, result=result_reg)
            else:
                assert mxu_unit is not None
                check_mxu_stream(program, unit=mxu_unit, activation_address=input_address,
                                 weight_address=DRAM_BASE + input_slots[1] * TILE_BYTES,
                                 output_address=output_address, activation=source_reg,
                                 weight=weight_reg, pair=pair,
                                 relu_pair=relu_pair if mxu_relu else None)
            qualification_machine = qualification_program = None
            if pipeline == "optimized":
                # Semantic checks above qualify this virtual DAG's mandatory
                # realization; generated checks qualify optimized scheduling.
                qualification_machine, qualification_program = machine, program
                machine = _run([str(atlas_opt), *NATIVE_PIPELINES[pipeline], "-"], input_text=virtual)
                _run([str(atlas_opt), *NATIVE_CHECKS, "-"], input_text=machine)
                program = json.loads(_run([str(atlas_emit), "--program-json", "-"], input_text=machine))
                if program.get("selected_rtl_revision") != RTL_REVISION:
                    raise NativeCompilationError("unqualified_target", "optimized emitter has a different selected RTL")
            instructions = program["instructions"]
            words = [row["word_u32"] for row in instructions]
            if not words or words[-1] != 0x73 or 0x73 in words[:-1] or len(words) * 4 > 0x20000:
                raise NativeCompilationError("unqualified_target", "emitted stream lacks one bounded terminal ECALL")
            emitted = [int(word, 16) for word in _run([str(atlas_emit), "-"], input_text=machine).splitlines()]
            if words != emitted or program.get("word_count") != len(words):
                raise NativeCompilationError("compile_error", "structured and flat Atlas emission disagree")
            if tool_hashes != (_sha256(atlas_opt.read_bytes()), _sha256(atlas_emit.read_bytes())):
                raise NativeCompilationError("compile_error", "Atlas tools changed during emission")

            return _MaterializedCandidate(
                selected_registers=selected_registers, virtual=virtual, machine=machine,
                program=program, words=words, tool_hashes=tool_hashes,
                input_address=input_address, output_address=output_address,
                realized_allocation=realized_allocation,
                qualification_machine=qualification_machine,
                qualification_program=qualification_program,
            )

        def accept_candidate(selected: SearchResult) -> tuple[bool, str] | MaterializationDecision:
            nonlocal materialized
            try:
                materialized = prepare_candidate(selected)
            except NativeCompilationError as exc:
                if exc.status not in {"unqualified_target", "unsupported_semantics"}:
                    raise
                materialized = None
                return False, exc.reason
            realized = materialized.realized_allocation
            if realized is not None:
                assert selected.graph is not None
                valid, reason = check_assignment(
                    selected.graph, realized.order, realized.addresses,
                    snapshot.profile.banks, fixed_inputs=fixed_inputs,
                    fixed_outputs=fixed_outputs,
                )
                if not valid:
                    materialized = None
                    return False, f"physical lowering changed the selected assignment: {reason}"
            return MaterializationDecision(
                True, realized_allocation=realized
            )

        selected = snapshot.select(
            request, fixed_inputs=fixed_inputs, fixed_outputs=fixed_outputs,
            limits=limits, accept_materialization=accept_candidate,
        )
        if selected.status != "selected":
            status = selected.status if selected.status in {
                "unsupported_semantics", "unqualified_target", "infeasible_candidate",
                "search_timeout", "resource_limit", "tool_unavailable",
            } else "compile_error"
            raise NativeCompilationError(status, selected.reason or "native selection found no checked candidate")
        assert materialized is not None
        has_relu = any(
            selected.rules.symbols[value.symbol].get("descriptor", {}).get("name")
            in {"atlas.vpu_relu_bf16_tile", "atlas.vpu_relu_bf16_from_mxu0_tile"}
            for value in selected.graph.values
        )
        has_exp = any(node.op == "exp_bf16" for node in request.nodes)
        has_recip = any(node.op == "recip_bf16" for node in request.nodes)
        has_add = any(node.op == "add_bf16" for node in request.nodes)
        has_sub = any(node.op == "sub_bf16" for node in request.nodes)
        has_mul = any(node.op == "mul_bf16" for node in request.nodes)
        has_row_max = any(node.op == "row_max_bf16" for node in request.nodes)
        has_row_sum = any(node.op == "row_sum_bf16" for node in request.nodes)
        selected_registers = materialized.selected_registers
        virtual = materialized.virtual
        machine = materialized.machine
        program = materialized.program
        words = materialized.words
        tool_hashes = materialized.tool_hashes
        input_address = materialized.input_address
        output_address = materialized.output_address

        destination.mkdir(parents=True, exist_ok=False)
        constant_rows = []
        if constant_payloads:
            (destination / "constants").mkdir()
        for index, node_id in enumerate(sorted(constant_payloads)):
            payload = constant_payloads[node_id]
            path = f"constants/constant-{index:03d}.bin"
            (destination / path).write_bytes(payload)
            constant_rows.append({
                "node_id": node_id, "address": DRAM_BASE + fixed_inputs[node_id] * TILE_BYTES,
                "bytes": len(payload), "encoding": "bf16-le-raw",
                "physical_layout": "bf16_pair_halves", "path": path,
                "sha256": _sha256(payload),
            })
        binary = b"".join(struct.pack("<I", word) for word in words)
        (destination / "program.bin").write_bytes(binary)
        (destination / "atlas-virtual.mlir").write_text(virtual)
        (destination / "atlas-machine.mlir").write_text(machine)
        (destination / "physical-program.json").write_text(json.dumps(program, indent=2) + "\n")
        semantic_nodes = {node.id: node for node in request.nodes}

        def output_numerical_policy(output_id: str) -> str:
            node = semantic_nodes[output_id]
            while node.op == "identity":
                node = semantic_nodes[node.inputs[0]]
            if node.op == "relu":
                return "bf16-sign-clamp-v1"
            if node.op in BF16_UNARY_ATTRS:
                return BF16_UNARY_ATTRS[node.op][1][1]
            if node.op == "add_bf16":
                return BF16_ADD_ATTRS[1][1]
            if node.op == "sub_bf16":
                return BF16_SUB_ATTRS[1][1]
            if node.op == "mul_bf16":
                return BF16_MUL_ATTRS[1][1]
            if node.op in BF16_ROW_REDUCTION_ATTRS:
                return BF16_ROW_REDUCTION_ATTRS[node.op][1][1]
            return node.type.numerical_policy

        output_policies = [output_numerical_policy(output) for output in request.outputs]
        boundary_domain_obligations = []
        boundary_requirements: set[tuple[str, str]] = set()

        def require_boundary_domain(source_id: str, domain: str) -> None:
            key = (source_id, domain)
            if key in boundary_requirements:
                return
            boundary_requirements.add(key)
            boundary_domain_obligations.append({
                "input": source_id,
                "required_domain": domain,
                "status": "caller_premise_unchecked",
            })

        for node in request.nodes:
            own_domain = dict(node.attrs).get("input_domain")
            if node.op == "input" and own_domain is not None:
                require_boundary_domain(node.id, own_domain)
            elif own_domain is not None:
                for source_id in node.inputs:
                    if semantic_nodes[source_id].op == "input":
                        require_boundary_domain(source_id, own_domain)
        producer_domain_obligations = [
            {
                "producer": source_id,
                "consumer": node.id,
                "required_domain": dict(node.attrs)["input_domain"],
                "status": "caller_premise_unchecked",
            }
            for node in request.nodes if "input_domain" in dict(node.attrs)
            for source_id in node.inputs if semantic_nodes[source_id].op != "input"
        ]
        if mxu_dag:
            producer_domain_obligations.extend({
                "producer": source_id,
                "consumer": node.id,
                "required_domain": MXU0_INPUT_DOMAIN,
                "status": "caller_premise_unchecked",
            } for node in request.nodes if node.op == "matmul"
              for source_id in node.inputs
              if semantic_nodes[transpose_inputs.get(source_id, source_id)].op == "pack_fp8")

        plan = {
            "schema": "atlas.native_tile_execution_plan.v8",
            "scope": "diagnostic_selected_standalone_core",
            "input_address": input_address, "output_address": output_address,
            "input_addresses": [DRAM_BASE + slot * TILE_BYTES for slot in input_slots],
            "output_addresses": [DRAM_BASE + slot * TILE_BYTES for slot in output_slots],
            "input_bytes": None if (mxu_binary or mxu_dag) else TILE_BYTES if (bf16_tile or pack) else 1024,
            "weight_address": (DRAM_BASE + fixed_inputs[next(iter(mxu_inputs.values()))[1]] * TILE_BYTES
                               if mxu_dag and len(mxu_inputs) == 1
                               and next(iter(mxu_inputs.values()))[1] in fixed_inputs else
                               DRAM_BASE + input_slots[1] * TILE_BYTES if (mxu or mxu_k_chain or mxu_binary) else None),
            "weight_addresses": ([DRAM_BASE + input_slots[index] * TILE_BYTES
                                  for index in range(1, len(input_slots), 2)]
                                 if mxu_k_chain else None),
            "weight_bytes": 1024 if (mxu or mxu_k_chain or mxu_binary or mxu_dag) else None,
            "output_bytes": TILE_BYTES if (bf16_tile or mxu or mxu_k_chain or mxu_binary or mxu_dag) else 1024,
            "shape": [32, 32], "dtype": "bf16" if (bf16_tile or mxu or mxu_k_chain or mxu_binary or mxu_dag) else "fp8_e4m3",
            "output_physical_layouts": [
                "bf16_pair_halves" if semantic_nodes[output].type.dtype == "bf16" else None
                for output in request.outputs
            ],
            "operation": ("mxu0_bf16_dag" if mxu_dag else "mxu0_bf16_binary" if mxu_binary else
                          f"mxu0_k{len(input_slots) // 2}" if mxu_k_chain else
                          "mxu0_relu" if mxu_relu else "bf16_relu_pack" if relu_pack
                          else "bf16_mixed" if sum((has_add, has_sub, has_mul, has_row_max, has_row_sum, has_exp, has_recip)) > 1
                          else "bf16_add_relu" if has_add and has_relu
                          else "bf16_add" if has_add
                          else "bf16_sub_relu" if has_sub and has_relu
                          else "bf16_sub" if has_sub
                          else "bf16_mul_relu" if has_mul and has_relu
                          else "bf16_mul" if has_mul
                          else "bf16_row_max_relu" if has_row_max and has_relu
                          else "bf16_row_max" if has_row_max
                          else "bf16_row_sum_relu" if has_row_sum and has_relu
                          else "bf16_row_sum" if has_row_sum
                          else "bf16_exp" if has_exp
                          else "bf16_recip" if has_recip
                          else "bf16_relu" if has_relu else "bf16_movement" if bf16_tile
                          else "bf16_to_fp8_scaled_pack" if pack and scaled_code is not None
                          else "bf16_to_fp8_unit_pack" if pack
                          else "copy_transpose" if copy_transpose
                          else "transpose" if transpose else f"mxu{mxu_unit}_reset"),
            "numerical_policy": ("per-operation-declared" if (mxu_binary or mxu_dag) else
                                 MXU0_POLICY if mxu_k_chain else request.nodes[2].type.numerical_policy if mxu else
                                 request.nodes[1].type.numerical_policy if pack and scaled_code is not None else
                                 PACK_POLICY if pack else
                                 output_policies[0] if len(set(output_policies)) == 1 else "per-output"),
            "output_numerical_policies": output_policies,
            "input_domain": ("per-operation-declared" if (mxu_binary or mxu_dag) else
                             MXU0_INPUT_DOMAIN if mxu_k_chain else simple_mxu_mode[1] if mxu else
                             pack_domain if pack and scaled_code is not None else
                             "bf16-seven-powers-and-zero-v1" if pack else
                             "per-operation-declared" if sum((has_add, has_sub, has_mul, has_row_max, has_row_sum, has_exp, has_recip)) > 1 else
                             BF16_ADD_ATTRS[0][1] if has_add else
                             BF16_SUB_ATTRS[0][1] if has_sub else
                             BF16_MUL_ATTRS[0][1] if has_mul else
                             BF16_ROW_MAX_ATTRS[0][1] if has_row_max else
                             BF16_ROW_SUM_ATTRS[0][1] if has_row_sum else
                             BF16_EXP_ATTRS[0][1] if has_exp else
                             BF16_RECIP_ATTRS[0][1] if has_recip else
                             "finite-normal-or-zero-bf16" if has_relu else "all-bit-patterns"),
            "runtime_domain_guard": False,
            "boundary_domain_obligations": boundary_domain_obligations,
            "producer_domain_obligations": producer_domain_obligations,
        }
        if mxu_binary or mxu_dag:
            plan["input_payload_bytes"] = [
                TILE_BYTES if node.type.dtype == "bf16" else 1024
                for node in request.nodes if node.op in {"input", "constant"}
            ]
        if constant_rows:
            plan["constant_inputs"] = constant_rows
        if mxu_dag:
            plan["weight_addresses"] = [
                DRAM_BASE + fixed_inputs[weight] * TILE_BYTES if weight in fixed_inputs else None
                for _, weight in mxu_inputs.values()
            ]
            plan["weight_source_nodes"] = [weight for _, weight in mxu_inputs.values()]
            plan["transpose_source_nodes"] = [
                {"result": result, "source": source}
                for result, source in transpose_inputs.items()
            ]
            plan["interlayer_pack_scale_codes"] = [
                {"producer": node.id, "scale_code": pack_scale_codes[node.id]}
                for node in request.nodes if node.id in pack_scale_codes
            ]
        if pack:
            plan["pack_scale_code"] = pack_scale_code
        plan_bytes = (json.dumps(plan, sort_keys=True) + "\n").encode()
        (destination / "execution_plan.json").write_bytes(plan_bytes)
        assert selected.graph is not None and selected.allocation is not None
        assert selected.rules is not None and selected.candidate is not None
        selection = {
            "schema": "merlin.native_selected_candidate.v1",
            "engine": "merlin_native",
            "request_digest": request.digest(),
            "target_identity": TARGET_IDENTITY,
            "candidate_digest": selected.candidate.digest(),
            "check_fingerprint": selected.check_fingerprint,
            "source_outputs": list(request.outputs),
            "graph_outputs": list(selected.graph.outputs),
            "order": list(selected.allocation.order),
            "issue_times": [list(item) for item in selected.allocation.issue_times],
            "schedule_scope": "native_candidate_virtual",
            "physical_assignment_scope": "mandatory_qualification",
            "candidate_attempts": selected.candidate_attempts,
            "ordering_attempts": selected.ordering_attempts,
            "rejected_materialization": selected.rejected_materialization,
            "values": [
                {
                    "id": value.id,
                    "symbol": value.symbol,
                    "descriptor": selected.rules.symbols[value.symbol].get("descriptor", {}).get("name"),
                    "source_node": value.source_node,
                    "kind": value.kind,
                    "storage": value.storage,
                    "extent": value.extent,
                    "children": list(value.children),
                    "validity": [condition.record() for condition in value.validity],
                    "input_read_offsets": list(value.input_read_offsets),
                    "completion_offset": value.completion_offset,
                    "in_place_inputs": list(value.in_place_inputs),
                    "preserve_input": value.preserve_input,
                    "address": selected.allocation.addresses[value.id],
                }
                for value in selected.graph.values
            ],
        }
        (destination / "native-selection.json").write_text(json.dumps(selection, sort_keys=True, indent=2) + "\n")
        stage_artifacts = {
            name: _sha256((destination / name).read_bytes())
            for name in ("native-selection.json", "atlas-virtual.mlir", "atlas-machine.mlir",
                         "physical-program.json")
        }
        if materialized.qualification_machine is not None:
            for name, payload in (
                ("mandatory-qualification-machine.mlir", materialized.qualification_machine),
                ("mandatory-qualification-program.json",
                 json.dumps(materialized.qualification_program, sort_keys=True, indent=2) + "\n"),
            ):
                (destination / name).write_text(payload)
                stage_artifacts[name] = _sha256((destination / name).read_bytes())
        stage_artifacts.update({row["path"]: row["sha256"] for row in constant_rows})
        manifest: dict[str, object] = {
            "schema": "atlas.native_tile_compilation.v6",
            "lowering_pipeline": pipeline,
            "semantic_qualification_pipeline": "mandatory",
            "artifact_kind": "program", "engine": "merlin_native",
            "qualification_scope": "diagnostic_only",
            "compile_mode": "diagnostic",
            "request_digest": request.digest(), "target_identity": TARGET_IDENTITY,
            "native_check_fingerprint": selected.check_fingerprint,
            "native_snapshot_sha256": _sha256((snapshot.root / "manifest.json").read_bytes()),
            "binding_source_sha256": _sha256(Path(__file__).read_bytes()),
            "target_inventory_sha256": _sha256(inventory_path.read_bytes()),
            "atlas_opt_sha256": tool_hashes[0], "atlas_emit_sha256": tool_hashes[1],
            "candidate_attempts": selected.candidate_attempts,
            "ordering_attempts": selected.ordering_attempts,
            "rejected_materialization": selected.rejected_materialization,
            "selected_registers": selected_registers,
            "selected_registers_scope": "mandatory_qualification",
            "binary_sha256": _sha256(binary),
            "execution_plan_sha256": _sha256(plan_bytes),
            "virtual_sha256": _sha256(virtual.encode()),
            "machine_sha256": _sha256(machine.encode()),
            "physical_program_sha256": _sha256((destination / "physical-program.json").read_bytes()),
            "stage_artifacts": stage_artifacts,
        }
        if constant_rows:
            manifest["constant_inputs"] = constant_rows
        (destination / "manifest.json").write_text(json.dumps(manifest, sort_keys=True) + "\n")
        return manifest
