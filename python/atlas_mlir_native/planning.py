"""Bind the original diagnostic request and explicit physical tile ABI.

Compiler-side admission/replay retains its diagnostic scope. Independent reference
and evaluator authority remain separate; no hardware execution is established.
"""

from __future__ import annotations

import json
import struct
from dataclasses import dataclass
from pathlib import Path

from merlin.compiler.semantic.model import (
    KernelRequest,
    pointwise_identity_maps,
)
from merlin.compiler.semantic.target_binding import NativeCompilationError

from .admission import (
    _bf16_tile_request,
    _interlayer_pack_code,
    _mxu0_k_chain_request,
    _mxu0_maps,
    _mxu_bf16_binary_request,
    _mxu_bf16_dag_request,
    _scaled_pack_code,
    _transpose_maps,
)
from .layout import pack_bf16_pair_halves
from .pack_lane import ROUNDED_PACK_POLICY
from .policy import (
    BF16_RELU_ATTRS,
    DRAM_SLOTS,
    MXU0_INPUT_DOMAIN,
    MXU0_POLICY,
    MXU1_FINITE_INPUT_DOMAIN,
    MXU1_FINITE_POLICY,
    MXU1_INPUT_DOMAIN,
    MXU1_POLICY,
    PACK_ATTRS,
    PACK_POLICY,
    RTL_REVISION,
    SCALED_PACK_POLICY,
    TARGET_IDENTITY,
)


@dataclass(frozen=True)
class TilePlan:
    """Original request admission and physical ABI, not an accuracy authority."""

    request: KernelRequest
    fixed_inputs: dict[str, int]
    bf16_tile: bool
    constant_payloads: dict[str, bytes]
    copy_transpose: bool
    input_slot: int
    input_slots: tuple[int, ...]
    inventory_path: Path
    mxu: bool
    mxu_binary: bool
    mxu_dag: bool
    mxu_inputs: dict[str, tuple[str, ...]]
    mxu_k_chain: bool
    mxu_relu: bool
    mxu_unit: int | None
    output_slot: int
    output_slots: tuple[int, ...]
    pack: bool
    pack_domain: str | None
    pack_scale_code: int
    pack_scale_codes: dict[str, int]
    pipeline: str
    relu_pack: bool
    scaled_code: int | None
    simple_mxu_mode: tuple[int, str] | None
    source_ids: tuple[str, ...]
    transpose: bool
    transpose_inputs: dict[str, str]


def admit_request(
    request: KernelRequest,
    *,
    fixed_inputs: dict[str, int],
    fixed_outputs: tuple[int, ...],
    target_source: Path,
    pipeline: str,
) -> TilePlan:
    if request.lowering_policy != "diagnostic":
        raise NativeCompilationError(
            "unqualified_target",
            "selected FPGA bitstream and host coherence are not qualified",
        )
    inventory_path = target_source / "docs/selected-variant-inventory.json"
    try:
        inventory = json.loads(inventory_path.read_text())
    except (OSError, ValueError) as exc:
        raise NativeCompilationError(
            "tool_unavailable", "selected Atlas source inventory is unavailable"
        ) from exc
    if inventory.get("selected_sources", {}).get("rtl_revision") != RTL_REVISION:
        raise NativeCompilationError(
            "unqualified_target", "Atlas source revision differs from selected RTL"
        )
    if request.target_identity != TARGET_IDENTITY:
        raise NativeCompilationError(
            "unsupported_semantics", "tile needs the selected target"
        )
    if (
        not (1 <= len(request.input_storages) <= DRAM_SLOTS)
        or not (1 <= len(request.outputs) <= DRAM_SLOTS)
        or any(storage != "external" for _, storage in request.input_storages)
        or request.output_storages != ("external",) * len(request.outputs)
    ):
        raise NativeCompilationError(
            "unsupported_semantics", "tile needs DRAM inputs and outputs"
        )
    bf16_tile = _bf16_tile_request(request)
    mxu_k_chain = _mxu0_k_chain_request(request)
    mxu_binary = _mxu_bf16_binary_request(request)
    mxu_dag = _mxu_bf16_dag_request(request) and not mxu_binary
    if request.constants and not mxu_dag:
        raise NativeCompilationError(
            "unsupported_semantics",
            "declared constants need an admitted BF16 MXU/VPU DAG",
        )
    mxu_inputs = {node.id: node.inputs for node in request.nodes if node.op == "matmul"}
    transpose_inputs = {
        node.id: node.inputs[0]
        for node in request.nodes
        if mxu_dag and node.op == "transpose"
    }
    pack_scale_codes = (
        {
            node.id: code
            for node in request.nodes
            if node.op == "pack_fp8"
            if (code := _interlayer_pack_code(node)) is not None
        }
        if mxu_dag
        else {}
    )
    transpose = (
        len(request.nodes) == 2
        and request.nodes[0].op == "input"
        and request.nodes[1].op == "transpose"
        and request.nodes[1].inputs == (request.nodes[0].id,)
        and request.outputs == (request.nodes[1].id,)
        and request.nodes[1].index_maps == _transpose_maps()
        and all(
            node.type.shape == (32, 32)
            and node.type.dtype == "fp8_e4m3"
            and node.type.numerical_policy == "raw-bit-pattern"
            and not node.attrs
            for node in request.nodes
        )
        and not request.nodes[0].index_maps
    )
    copy_transpose = (
        len(request.nodes) == 3
        and tuple(node.op for node in request.nodes)
        == ("input", "identity", "transpose")
        and request.nodes[1].inputs == request.nodes[2].inputs == (request.nodes[0].id,)
        and request.outputs == (request.nodes[1].id, request.nodes[2].id)
        and request.nodes[2].index_maps == _transpose_maps()
        and all(
            node.type.shape == (32, 32)
            and node.type.dtype == "fp8_e4m3"
            and node.type.numerical_policy == "raw-bit-pattern"
            and not node.attrs
            for node in request.nodes
        )
        and not request.nodes[0].index_maps
        and not request.nodes[1].index_maps
    )
    scaled_code = (
        _scaled_pack_code(request.nodes[1]) if len(request.nodes) == 2 else None
    )
    pack_domain = (
        dict(request.nodes[1].attrs).get("input_domain")
        if scaled_code is not None
        else None
    )
    allowed_pack_inputs = (
        ((("input_domain", pack_domain),),)
        if scaled_code is not None
        else ((), (("input_domain", PACK_ATTRS[0][1]),))
    )
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
            (
                scaled_code is not None
                and request.nodes[1].type.numerical_policy
                in {SCALED_PACK_POLICY, ROUNDED_PACK_POLICY}
            )
            or (
                request.nodes[1].type.numerical_policy == PACK_POLICY
                and request.nodes[1].attrs == PACK_ATTRS
            )
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
        if len(request.nodes) >= 3
        else None
    )
    mxu = (
        len(request.nodes) in (3, 4)
        and simple_mxu_mode is not None
        and tuple(node.op for node in request.nodes[:3]) == ("input", "input", "matmul")
        and request.nodes[2].inputs == (request.nodes[0].id, request.nodes[1].id)
        and request.nodes[2].index_maps == _mxu0_maps()
        and all(node.type.shape == (32, 32) for node in request.nodes[:3])
        and all(
            node.type.dtype == "fp8_e4m3"
            and node.type.numerical_policy == "raw-bit-pattern"
            and node.attrs == (("input_domain", simple_mxu_mode[1]),)
            and node.effect == "input"
            and not node.index_maps
            for node in request.nodes[:2]
        )
        and request.nodes[2].type.dtype == "bf16"
        and request.nodes[2].effect == "pure"
        and not request.nodes[2].attrs
    )
    mxu_relu = mxu and simple_mxu_mode[0] == 0 and len(request.nodes) == 4
    if mxu_relu:
        relu = request.nodes[3]
        mxu_relu = (
            relu.op == "relu"
            and relu.inputs == (request.nodes[2].id,)
            and relu.type.shape == (32, 32)
            and relu.type.dtype == "bf16"
            and relu.type.numerical_policy == "raw-bit-pattern"
            and relu.effect == "pure"
            and relu.attrs == BF16_RELU_ATTRS
            and relu.index_maps == pointwise_identity_maps(2, 1)
        )
    mxu = mxu and (
        (len(request.nodes) == 3 and request.outputs == (request.nodes[2].id,))
        or (mxu_relu and request.outputs == (request.nodes[3].id,))
    )
    mxu_unit = simple_mxu_mode[0] if mxu else None
    if not (
        bf16_tile
        or transpose
        or copy_transpose
        or mxu
        or mxu_k_chain
        or mxu_binary
        or mxu_dag
        or pack
    ):
        raise NativeCompilationError(
            "unsupported_semantics",
            "only bounded BF16/FP8 tile graphs and admitted MXU tile chains are supported",
        )
    source_ids = tuple(
        node.id for node in request.nodes if node.op in {"input", "constant"}
    )
    if set(fixed_inputs) != set(source_ids) or len(fixed_outputs) != len(
        request.outputs
    ):
        raise NativeCompilationError(
            "compile_error", "tile ABI must bind every input, constant, and output"
        )
    input_slots = tuple(fixed_inputs[source_id] for source_id in source_ids)
    output_slots = tuple(fixed_outputs)
    output_slot = output_slots[0]
    if any(
        type(slot) is not int or slot < 0 or slot >= DRAM_SLOTS
        for slot in (*input_slots, *output_slots)
    ) or len({*input_slots, *output_slots}) != len(input_slots) + len(output_slots):
        raise NativeCompilationError(
            "unqualified_target", "tile DRAM slots are invalid or overlap"
        )
    # The BF16 DAG stream checker covers explicitly addressed tile DMA.
    # Other admitted forms still use their indexed ABI and dedicated
    # stream replays, so keep their original placement restriction.
    if not bf16_tile:
        if any(
            slot != input_slots[0] + index for index, slot in enumerate(input_slots)
        ):
            raise NativeCompilationError(
                "unsupported_semantics", "input tiles need adjacent DRAM slots"
            )
        if any(
            slot != output_slots[0] + index for index, slot in enumerate(output_slots)
        ):
            raise NativeCompilationError(
                "unsupported_semantics", "output tiles need adjacent DRAM slots"
            )
    input_slot = input_slots[0]

    constant_payloads: dict[str, bytes] = {}
    for binding in request.constants:
        logical = bytes.fromhex(binding.data_hex)
        codes = struct.unpack("<1024H", logical)
        if any(not 1 <= ((code >> 7) & 0xFF) <= 254 for code in codes):
            raise NativeCompilationError(
                "unsupported_semantics",
                "declared BF16 multiplier constant is outside finite-normal domain",
            )
        constant_payloads[binding.node_id] = pack_bf16_pair_halves(logical)

    return TilePlan(
        request=request,
        fixed_inputs=fixed_inputs,
        bf16_tile=bf16_tile,
        constant_payloads=constant_payloads,
        copy_transpose=copy_transpose,
        input_slot=input_slot,
        input_slots=input_slots,
        inventory_path=inventory_path,
        mxu=mxu,
        mxu_binary=mxu_binary,
        mxu_dag=mxu_dag,
        mxu_inputs=mxu_inputs,
        mxu_k_chain=mxu_k_chain,
        mxu_relu=mxu_relu,
        mxu_unit=mxu_unit,
        output_slot=output_slot,
        output_slots=output_slots,
        pack=pack,
        pack_domain=pack_domain,
        pack_scale_code=pack_scale_code,
        pack_scale_codes=pack_scale_codes,
        pipeline=pipeline,
        relu_pack=relu_pack,
        scaled_code=scaled_code,
        simple_mxu_mode=simple_mxu_mode,
        source_ids=source_ids,
        transpose=transpose,
        transpose_inputs=transpose_inputs,
    )
