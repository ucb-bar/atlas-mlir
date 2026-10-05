"""Diagnostic physical materialization of native-selected Atlas matrix tiles.

The operation order and addresses come from Merlin's checked native result.
The selected core has no architectural MXU wait, so the conservative DELAYs
here are diagnostic and do not qualify general execution timing.
"""

from __future__ import annotations

import json
import tempfile
from dataclasses import asdict
from pathlib import Path

from merlin.semantic_compiler.model import KernelRequest
from merlin.semantic_compiler.search import SearchLimits
from merlin.semantic_compiler.snapshot import NativeSnapshot
from merlin.semantic_compiler.verify import check_selection

from .movement import (
    _canonical,
    _load_selected_assembler,
    _sha,
    _verify_native_dependencies,
)
from .mxu0 import Mxu0Contract, admits_fp8_tile, mxu0_profile
from .mxu0_spatial_tiling import SpatialTilePlan, lower_tiled_contraction
from .mxu0_tiling import OrderedKTilePlan


def _materialize(selected, contract: Mxu0Contract) -> tuple[str, tuple[int, ...]]:
    """Bind every checked graph instruction to selected Atlas assembly."""
    graph = selected.graph
    allocation = selected.allocation
    rules = selected.rules
    if graph is None or allocation is None or rules is None:
        raise ValueError("native MXU0 selection has no checked physical graph")
    address = allocation.addresses
    geometry = contract.record["geometry"]
    tile_bytes = geometry["matrix_register_bytes"]
    word_bytes = contract.movement.record["scalar_word_bytes"]
    base = contract.movement.record["dram_program_base"]
    load_channel = contract.movement.record["load_channel"]
    store_channel = contract.movement.record["store_channel"]
    rows, cols = geometry["array_rows"], geometry["array_cols"]
    result_registers = contract.record["bf16_result_layout"]["register_count"]
    conversion = contract.record["vfp8_conversion"]
    codes = tuple(conversion["admitted_scale_codes"])
    pack_names = {f"vpu_pack_e8m0_{code}" for code in codes}
    unpack_names = {f"vpu_unpack_e8m0_{code}" for code in codes}
    pack_vstore_names = {f"vstore_pack_e8m0_{code}" for code in codes}
    pack_dma_names = {f"dma_store_pack_e8m0_{code}" for code in codes}
    unpack_vstore_names = {f"vstore_unpack_e8m0_{code}_pair" for code in codes}
    unpack_dma_names = {f"dma_store_unpack_e8m0_{code}_pair" for code in codes}
    # The selected RTL documents mregRows+2 for VLOAD/VSTORE; its sequencer
    # describes a last matmul row at T+94 for the 32-row profile. Extra cycles
    # keep this diagnostic stream serial. These are not architectural waits.
    vector_delay = rows + 4
    vpu_pair_delay = 2 * rows + 8
    push_delay = cols + 4
    compute_delay = 3 * rows + 4
    lines = [
        "LI x28, 1",
        "LI x5, 0",
        f"DMA.CONFIG x5, {load_channel}",
        f"LI x2, {tile_bytes}",
    ]
    emitted: list[int] = []

    def scalar_address(register: int, value: int) -> None:
        if value < 0 or value >= 2**32:
            raise ValueError(
                "physical address cannot be materialized in Atlas scalar register"
            )
        lines.append(f"LI x{register}, {value}")

    for value_id in allocation.order:
        value = graph.value(value_id)
        if value.kind != "instruction":
            raise ValueError("MXU0 order contains a non-instruction")
        name = rules.symbols[value.symbol]["descriptor"]["name"]
        children = tuple(graph.value(child) for child in value.children)
        emitted.append(value_id)
        if (
            name in {"dma_load_fp8", "mxu1_dma_load_fp8"}
            and len(children) == 1
            and (children[0].storage, value.storage) == ("external_fp8", "vmem_fp8")
        ):
            scalar_address(6, address[value_id] * tile_bytes // word_bytes)
            scalar_address(1, base + address[children[0].id] * tile_bytes)
            lines += [
                f"DMA.LOAD x6, x1, x2, {load_channel}",
                f"DMA.WAIT {load_channel}",
            ]
        elif (
            name in {"dma_load_bf16_anchor_pair", "dma_load_bf16_raw_pair", "dma_load_bf16_exp2_pair"}
            and len(children) == 1
            and (children[0].storage, value.storage) == ("external_bf16", "vmem_bf16")
            and address[value_id] % result_registers == 0
            and address[children[0].id] % result_registers == 0
        ):
            for half in range(result_registers):
                scalar_address(6, (address[value_id] + half) * tile_bytes // word_bytes)
                scalar_address(1, base + (address[children[0].id] + half) * tile_bytes)
                lines += [
                    f"DMA.LOAD x6, x1, x2, {load_channel}",
                    f"DMA.WAIT {load_channel}",
                ]
        elif (
            name in {"vload_fp8", "mxu1_vload_fp8"}
            and len(children) == 1
            and (children[0].storage, value.storage) == ("vmem_fp8", "vrf_fp8")
        ):
            scalar_address(6, address[children[0].id] * tile_bytes // word_bytes)
            lines += [f"VLOAD {address[value_id]}, x6, 0", f"DELAY {vector_delay}"]
        elif (
            name in {"vload_bf16_anchor_pair", "vload_bf16_raw_pair", "vload_bf16_exp2_pair"}
            and len(children) == 1
            and (children[0].storage, value.storage) == ("vmem_bf16", "vrf_bf16")
            and address[value_id] % result_registers == 0
            and address[children[0].id] % result_registers == 0
        ):
            for half in range(result_registers):
                scalar_address(
                    6, (address[children[0].id] + half) * tile_bytes // word_bytes
                )
                lines += [
                    f"VLOAD {address[value_id] + half}, x6, 0",
                    f"DELAY {vector_delay}",
                ]
        elif name in pack_names | unpack_names and len(children) == 1:
            descriptor = rules.symbols[value.symbol]["descriptor"]
            attrs = descriptor["required_attrs"]
            if set(attrs) != {"scale_e8m0"} or attrs["scale_e8m0"] not in codes:
                raise ValueError("selected pack/unpack scale binding changed")
            scale_code = attrs["scale_e8m0"]
            if (
                name
                != f"vpu_{'pack' if name in pack_names else 'unpack'}_e8m0_{scale_code}"
            ):
                raise ValueError(
                    "selected pack/unpack scale does not match instruction"
                )
            if name in pack_names:
                valid = (children[0].storage, value.storage) == (
                    "vrf_bf16",
                    "vrf_fp8",
                ) and address[children[0].id] % result_registers == 0
                mnemonic = "VFP8PACK"
            else:
                valid = (children[0].storage, value.storage) == (
                    "vrf_fp8",
                    "vrf_bf16",
                ) and address[value_id] % result_registers == 0
                mnemonic = "VFP8UNPACK"
            if not valid:
                raise ValueError(
                    "selected pack/unpack storage or BF16 pair alignment changed"
                )
            lines += [
                f"SELI 0, {scale_code}",
                f"{mnemonic} {address[value_id]}, {address[children[0].id]}, 0",
                f"DELAY {vpu_pair_delay}",
            ]
        elif (
            name == "xlu_transpose_fp8"
            and len(children) == 1
            and (children[0].storage, value.storage) == ("vrf_fp8", "vrf_fp8")
        ):
            lines += [
                f"VTRPOSE.XLU {address[value_id]}, {address[children[0].id]}",
                f"DELAY {2 * rows + 4}",
            ]
        elif (
            name == "mxu0_push_weight"
            and len(children) == 1
            and (children[0].storage, value.storage) == ("vrf_fp8", "mxu0_weight")
        ):
            lines += [
                f"VMATPUSH.W.MXU0 {address[value_id]}, {address[children[0].id]}",
                f"DELAY {push_delay}",
            ]
        elif (
            name == "mxu1_push_weight"
            and len(children) == 1
            and (children[0].storage, value.storage) == ("vrf_fp8", "mxu1_weight")
        ):
            lines += [
                f"VMATPUSH.W.MXU1 {address[value_id]}, {address[children[0].id]}",
                f"DELAY {push_delay}",
            ]
        elif (
            name == "mxu0_matmul_reset"
            and len(children) == 2
            and (children[0].storage, children[1].storage, value.storage)
            == ("vrf_fp8", "mxu0_weight", "mxu0_accum")
        ):
            lines += [
                (
                    f"VMATMUL.MXU0 {address[value_id]}, {address[children[0].id]}, "
                    f"{address[children[1].id]}"
                ),
                f"DELAY {compute_delay}",
            ]
        elif (
            name == "mxu0_matmul_accumulate"
            and len(children) == 3
            and (
                children[0].storage,
                children[1].storage,
                children[2].storage,
                value.storage,
            )
            == ("vrf_fp8", "mxu0_weight", "mxu0_accum", "mxu0_accum")
            and address[value_id] == address[children[2].id]
        ):
            lines += [
                (
                    f"VMATMUL.ACC.MXU0 {address[value_id]}, "
                    f"{address[children[0].id]}, {address[children[1].id]}"
                ),
                f"DELAY {compute_delay}",
            ]
        elif (
            name in {"mxu1_matmul_reset", "mxu1_matmul_packed_e8m0_127"}
            and len(children) == 2
            and (children[0].storage, children[1].storage, value.storage)
            == ("vrf_fp8", "mxu1_weight", "mxu1_accum")
        ):
            lines += [
                (
                    f"VMATMUL.MXU1 {address[value_id]}, {address[children[0].id]}, "
                    f"{address[children[1].id]}"
                ),
                f"DELAY {compute_delay}",
            ]
        elif (
            name == "mxu0_pop_bf16"
            and len(children) == 1
            and (children[0].storage, value.storage) == ("mxu0_accum", "vrf_bf16")
        ):
            lines += [
                f"VMATPOP.BF16.MXU0 {address[value_id]}, {address[children[0].id]}",
                f"DELAY {vector_delay}",
            ]
        elif (
            name == "mxu1_pop_bf16"
            and len(children) == 1
            and (children[0].storage, value.storage) == ("mxu1_accum", "vrf_bf16")
        ):
            lines += [
                f"VMATPOP.BF16.MXU1 {address[value_id]}, {address[children[0].id]}",
                f"DELAY {vector_delay}",
            ]
        elif (
            name == "vpu_relu_bf16"
            and len(children) == 1
            and (children[0].storage, value.storage) == ("vrf_bf16", "vrf_bf16")
            and address[value_id] % result_registers == 0
            and address[children[0].id] % result_registers == 0
        ):
            lines += [
                f"VRELU {address[value_id]}, {address[children[0].id]}",
                f"DELAY {vpu_pair_delay}",
            ]
        elif (
            name == "vpu_square_bf16"
            and len(children) == 1
            and (children[0].storage, value.storage) == ("vrf_bf16", "vrf_bf16")
            and address[value_id] % result_registers == 0
            and address[children[0].id] % result_registers == 0
        ):
            lines += [
                (
                    f"{contract.record['vpu_square']['instruction']} "
                    f"{address[value_id]}, {address[children[0].id]}"
                ),
                f"DELAY {vpu_pair_delay}",
            ]
        elif (
            name in {"vpu_recip_bf16_raw", "vpu_recip_bf16_mxu0"}
            and len(children) == 1
            and (children[0].storage, value.storage) == ("vrf_bf16", "vrf_bf16")
            and address[value_id] % result_registers == 0
            and address[children[0].id] % result_registers == 0
        ):
            lines += [
                (
                    f"{contract.record['vpu_recip']['instruction']} "
                    f"{address[value_id]}, {address[children[0].id]}"
                ),
                f"DELAY {vpu_pair_delay}",
            ]
        elif (
            name in {"vpu_log2_bf16_raw", "vpu_log2_bf16_mxu0"}
            and len(children) == 1
            and (children[0].storage, value.storage) == ("vrf_bf16", "vrf_bf16")
            and address[value_id] % result_registers == 0
            and address[children[0].id] % result_registers == 0
        ):
            lines += [
                (
                    f"{contract.record['vpu_log2']['instruction']} "
                    f"{address[value_id]}, {address[children[0].id]}"
                ),
                f"DELAY {vpu_pair_delay}",
            ]
        elif (
            name in {"vpu_sqrt_bf16_raw", "vpu_sqrt_bf16_mxu0"}
            and len(children) == 1
            and (children[0].storage, value.storage) == ("vrf_bf16", "vrf_bf16")
            and address[value_id] % result_registers == 0
            and address[children[0].id] % result_registers == 0
        ):
            lines += [
                (
                    f"{contract.record['vpu_sqrt']['instruction']} "
                    f"{address[value_id]}, {address[children[0].id]}"
                ),
                f"DELAY {vpu_pair_delay}",
            ]
        elif (
            name == "vpu_exp2_bf16_bounded"
            and len(children) == 1
            and (children[0].storage, value.storage) == ("vrf_bf16", "vrf_bf16")
            and address[value_id] % result_registers == 0
            and address[children[0].id] % result_registers == 0
        ):
            lines += [
                (
                    f"{contract.record['vpu_exp2']['instruction']} "
                    f"{address[value_id]}, {address[children[0].id]}"
                ),
                f"DELAY {vpu_pair_delay}",
            ]
        elif (
            name in {"vpu_add_bf16", "vpu_mul_bf16", "vpu_sub_bf16",
                     "vpu_min_bf16", "vpu_max_bf16"}
            and len(children) == 2
            and (children[0].storage, children[1].storage, value.storage)
            == ("vrf_bf16", "vrf_bf16", "vrf_bf16")
            and all(
                address[item.id] % result_registers == 0 for item in (*children, value)
            )
        ):
            operation = {
                "vpu_add_bf16": "vpu_add",
                "vpu_mul_bf16": "vpu_mul",
                "vpu_sub_bf16": "vpu_sub",
                "vpu_min_bf16": "vpu_min",
                "vpu_max_bf16": "vpu_max",
            }[name]
            lines += [
                (
                    f"{contract.record[operation]['instruction']} {address[value_id]}, "
                    f"{address[children[0].id]}, {address[children[1].id]}"
                ),
                f"DELAY {vpu_pair_delay}",
            ]
        elif (
            name in {"vpu_col_min_bf16", "vpu_col_max_bf16", "vpu_col_sum_bf16"}
            and len(children) == 1
            and (children[0].storage, value.storage) == ("vrf_bf16", "vrf_bf16")
            and address[value_id] % result_registers == 0
            and address[children[0].id] % result_registers == 0
        ):
            kind = name.removeprefix("vpu_col_").removesuffix("_bf16")
            mode = contract.record[f"vpu_col_{kind}"]
            lines += [
                f"{mode['instruction']} {address[value_id]}, {address[children[0].id]}",
                f"DELAY {mode['diagnostic_delay_cycles']}",
            ]
        elif (
            name in {"vpu_row_min_bf16", "vpu_row_max_bf16", "vpu_row_sum_bf16"}
            and len(children) == 1
            and (children[0].storage, value.storage) == ("vrf_bf16", "vrf_bf16")
            and address[value_id] % result_registers == 0
            and address[children[0].id] % result_registers == 0
        ):
            kind = name.removeprefix("vpu_row_").removesuffix("_bf16")
            mode = contract.record[f"vpu_row_{kind}"]
            lines += [
                f"{mode['instruction']} {address[value_id]}, {address[children[0].id]}",
                f"DELAY {mode['diagnostic_delay_cycles']}",
            ]
        elif (
            name
            in {
                "vstore_bf16_pair",
                "mxu1_vstore_bf16_pair",
                "vstore_vpu_add_bf16_pair",
                "vstore_vpu_mul_bf16_pair",
                "vstore_vpu_sub_bf16_pair",
                "vstore_vpu_square_bf16_pair",
                "vstore_vpu_recip_bf16_pair",
                "vstore_vpu_log2_bf16_pair",
                "vstore_vpu_sqrt_bf16_pair",
                "vstore_vpu_exp2_bf16_pair",
                "vstore_vpu_min_bf16_pair",
                "vstore_vpu_max_bf16_pair",
                "vstore_vpu_col_min_bf16_pair",
                "vstore_vpu_col_max_bf16_pair",
                "vstore_vpu_col_sum_bf16_pair",
                "vstore_vpu_row_min_bf16_pair",
                "vstore_vpu_row_max_bf16_pair",
                "vstore_vpu_row_sum_bf16_pair",
            }
            | unpack_vstore_names
            and len(children) == 1
            and (children[0].storage, value.storage) == ("vrf_bf16", "vmem_bf16")
        ):
            for half in range(result_registers):
                scalar_address(6, (address[value_id] + half) * tile_bytes // word_bytes)
                lines += [
                    f"VSTORE {address[children[0].id] + half}, x6, 0",
                    f"DELAY {vector_delay}",
                ]
        elif (
            name in {"vstore_fp8"} | pack_vstore_names
            and len(children) == 1
            and (children[0].storage, value.storage) == ("vrf_fp8", "vmem_fp8")
        ):
            scalar_address(6, address[value_id] * tile_bytes // word_bytes)
            lines += [
                f"VSTORE {address[children[0].id]}, x6, 0",
                f"DELAY {vector_delay}",
            ]
        elif (
            name
            in {
                "dma_store_bf16_pair",
                "mxu1_dma_store_bf16_pair",
                "dma_store_vpu_add_bf16_pair",
                "dma_store_vpu_mul_bf16_pair",
                "dma_store_vpu_sub_bf16_pair",
                "dma_store_vpu_square_bf16_pair",
                "dma_store_vpu_recip_bf16_pair",
                "dma_store_vpu_log2_bf16_pair",
                "dma_store_vpu_sqrt_bf16_pair",
                "dma_store_vpu_exp2_bf16_pair",
                "dma_store_vpu_min_bf16_pair",
                "dma_store_vpu_max_bf16_pair",
                "dma_store_vpu_col_min_bf16_pair",
                "dma_store_vpu_col_max_bf16_pair",
                "dma_store_vpu_col_sum_bf16_pair",
                "dma_store_vpu_row_min_bf16_pair",
                "dma_store_vpu_row_max_bf16_pair",
                "dma_store_vpu_row_sum_bf16_pair",
            }
            | unpack_dma_names
            and len(children) == 1
            and (children[0].storage, value.storage) == ("vmem_bf16", "external_bf16")
        ):
            for half in range(result_registers):
                scalar_address(
                    6, (address[children[0].id] + half) * tile_bytes // word_bytes
                )
                scalar_address(1, base + (address[value_id] + half) * tile_bytes)
                lines += [
                    f"DMA.STORE x1, x6, x2, {store_channel}",
                    f"DMA.WAIT {store_channel}",
                ]
        elif (
            name in {"dma_store_fp8"} | pack_dma_names
            and len(children) == 1
            and (children[0].storage, value.storage) == ("vmem_fp8", "external_fp8")
        ):
            scalar_address(6, address[children[0].id] * tile_bytes // word_bytes)
            scalar_address(1, base + address[value_id] * tile_bytes)
            lines += [
                f"DMA.STORE x1, x6, x2, {store_channel}",
                f"DMA.WAIT {store_channel}",
            ]
        else:
            raise ValueError(
                f"native MXU0 instruction lacks selected physical binding: {name}"
            )
    lines += ["LI x1, 1", "CSRRW x0, 0xC10, x1", "ECALL"]
    return "\n".join(lines) + "\n", tuple(emitted)


def _materialize_constants(
    selected, request: KernelRequest, contract: Mxu0Contract
) -> tuple[bytes, list[dict[str, object]]]:
    """Place exact compiler-supplied bytes at the checked constant leaf addresses."""
    if selected.graph is None or selected.allocation is None:
        raise ValueError("native MXU0 selection has no checked constant placement")
    tile_bytes = contract.record["geometry"]["matrix_register_bytes"]
    base = contract.movement.record["dram_program_base"]
    blob = bytearray()
    plan: list[dict[str, object]] = []
    selected_leaves = [
        value for value in selected.graph.values if value.kind == "constant"
    ]
    leaves = {value.source_node: value for value in selected_leaves}
    if len(leaves) != len(selected_leaves) or set(leaves) != {
        binding.node_id for binding in request.constants
    }:
        raise ValueError(
            "selected MXU0 graph omitted or duplicated a declared constant"
        )
    for binding in sorted(request.constants, key=lambda item: item.node_id):
        if binding.encoding != "fp8_e4m3" or binding.storage != "external_fp8":
            raise ValueError(
                "MXU0 constant has unsupported encoding or boundary storage"
            )
        payload = bytes.fromhex(binding.data_hex)
        node = next(node for node in request.nodes if node.id == binding.node_id)
        if len(payload) != tile_bytes or not admits_fp8_tile(
            contract, node.type.numerical_policy, payload
        ):
            raise ValueError(
                "Atlas matrix constant is outside its declared FP8 tile domain"
            )
        leaf = leaves[binding.node_id]
        if leaf.storage != binding.storage or leaf.extent != 1:
            raise ValueError(
                "selected MXU0 constant differs from its boundary representation"
            )
        plan.append(
            {
                "source": binding.node_id,
                "byte_address": base
                + selected.allocation.addresses[leaf.id] * tile_bytes,
                "byte_length": len(payload),
                "package_offset": len(blob),
                "encoding": binding.encoding,
                "sha256": _sha(payload),
                "preload": "before_each_invocation",
                "preserve": True,
            }
        )
        blob.extend(payload)
    return bytes(blob), plan


def compile_mxu0(
    snapshot: NativeSnapshot,
    request: KernelRequest,
    *,
    fixed_inputs: dict[str, int],
    fixed_outputs: tuple[int, ...],
    rtl_root: Path,
    software_spec: Path | None = None,
    destination: Path,
    limits: SearchLimits | None = None,
    contract: Mxu0Contract | None = None,
    tiling: OrderedKTilePlan | SpatialTilePlan | None = None,
) -> dict[str, object]:
    """Emit checked native MXU0 or bounded MXU1 tile graphs for diagnostic execution."""
    if destination.exists():
        raise FileExistsError(f"fresh compilation output required: {destination}")
    contract = contract or Mxu0Contract.load()
    contract.verify_source(rtl_root, software_spec)
    if tiling is not None and (
        tiling.lowered.record() != request.record()
        or lower_tiled_contraction(tiling.source, contract).record() != tiling.record()
    ):
        raise ValueError("ordered-K lowering changed before native compilation")
    _verify_native_dependencies(snapshot)
    profile = mxu0_profile(contract)
    if (
        snapshot.profile.digest() != profile.digest()
        or request.target_identity != contract.identity
    ):
        raise ValueError("native snapshot/request differs from selected MXU0 contract")
    if request.lowering_policy != "strict-native":
        raise ValueError("MXU0 diagnostic requires strict-native mode")
    operand = contract.record["numerical_policy"]
    mxu1 = contract.record["mxu1"]
    conversion = contract.record["vfp8_conversion"]
    rows, cols = (
        contract.record["geometry"]["array_rows"],
        contract.record["geometry"]["array_cols"],
    )
    for node in request.nodes:
        if node.op in {"input", "constant"}:
            signature = (node.type.shape, node.type.dtype, node.type.numerical_policy)
            if signature not in {
                (shape, policy["operand_dtype"], policy["operand_policy"])
                for shape in {(rows, rows), (cols, rows)}
                for policy in (operand, mxu1)
            } | {
                ((rows, cols), "bf16", conversion["input_policy"]),
                ((rows, cols), "bf16", contract.record["vpu_recip"]["operand_policies"][0]),
                ((rows, cols), "bf16", contract.record["vpu_exp2"]["operand_policy"]),
            }:
                raise ValueError(
                    "Atlas matrix input shape or numerical policy is outside selected tile"
                )
            if node.op == "constant" and signature in {
                ((rows, cols), "bf16", conversion["input_policy"]),
                ((rows, cols), "bf16", contract.record["vpu_recip"]["operand_policies"][0]),
                ((rows, cols), "bf16", contract.record["vpu_exp2"]["operand_policy"]),
            }:
                raise ValueError(
                    "BF16 pair constants lack a selected package encoding"
                )
        elif node.op in {"vfp8_pack", "vfp8_unpack"}:
            code = dict(node.attrs).get("scale_e8m0")
            if (
                type(code) is not int
                or code not in conversion["admitted_scale_codes"]
                or len(node.inputs) != 1
                or len(node.attrs) != 1
                or node.index_maps
                or node.type.shape != (rows, cols)
            ):
                raise ValueError(
                    "Atlas E8M0 conversion has an unadmitted scale or signature"
                )
            policy = (
                conversion["pack_result_policies"][str(code)]
                if node.op == "vfp8_pack"
                else conversion["unpack_result_policies"][str(code)]
            )
            dtype = "fp8_e4m3" if node.op == "vfp8_pack" else "bf16"
            if (node.type.dtype, node.type.numerical_policy) != (dtype, policy):
                raise ValueError("Atlas E8M0 conversion numerical policy changed")
        elif node.op == "contraction_accumulate":
            if (node.type.shape, node.type.dtype, node.type.numerical_policy) != (
                (rows, cols),
                operand["result_dtype"],
                operand["result_policy"],
            ):
                raise ValueError(
                    "Atlas matrix continuation has an unadmitted numerical policy"
                )
        elif node.op == "transpose":
            if (node.type.shape, node.type.dtype, node.type.numerical_policy) != (
                (rows, cols),
                operand["operand_dtype"],
                operand["operand_policy"],
            ):
                raise ValueError("Atlas XLU transpose has an unadmitted type or policy")
        elif node.op == "relu":
            if (
                (node.type.shape, node.type.dtype, node.type.numerical_policy)
                != ((rows, cols), operand["result_dtype"], operand["result_policy"])
                or len(node.inputs) != 1
                or dict(node.attrs)
                or node.index_maps
            ):
                raise ValueError(
                    "Atlas VPU ReLU has an unadmitted type, policy, or signature"
                )
        elif node.op == "add":
            if (
                (node.type.shape, node.type.dtype, node.type.numerical_policy)
                != (
                    (rows, cols),
                    operand["result_dtype"],
                    contract.record["vpu_add"]["result_policy"],
                )
                or len(node.inputs) != 2
                or dict(node.attrs)
                or node.index_maps
            ):
                raise ValueError(
                    "Atlas VPU add has an unadmitted type, policy, or signature"
                )
        elif node.op == "mul":
            if (
                (node.type.shape, node.type.dtype, node.type.numerical_policy)
                != (
                    (rows, cols),
                    operand["result_dtype"],
                    contract.record["vpu_mul"]["result_policy"],
                )
                or len(node.inputs) != 2
                or dict(node.attrs)
                or node.index_maps
            ):
                raise ValueError(
                    "Atlas VPU multiply has an unadmitted type, policy, or signature"
                )
        elif node.op == "sub":
            if (
                (node.type.shape, node.type.dtype, node.type.numerical_policy)
                != (
                    (rows, cols),
                    operand["result_dtype"],
                    contract.record["vpu_sub"]["result_policy"],
                )
                or len(node.inputs) != 2
                or dict(node.attrs)
                or node.index_maps
            ):
                raise ValueError(
                    "Atlas VPU subtract has an unadmitted type, policy, or signature"
                )
        elif node.op == "square":
            if (
                (node.type.shape, node.type.dtype, node.type.numerical_policy)
                != (
                    (rows, cols),
                    operand["result_dtype"],
                    contract.record["vpu_square"]["result_policy"],
                )
                or len(node.inputs) != 1
                or dict(node.attrs)
                or node.index_maps
            ):
                raise ValueError(
                    "Atlas VPU square has an unadmitted type, policy, or signature"
                )
        elif node.op == "reciprocal":
            if (
                (node.type.shape, node.type.dtype, node.type.numerical_policy)
                != ((rows, cols), "bf16", contract.record["vpu_recip"]["result_policy"])
                or len(node.inputs) != 1
                or dict(node.attrs)
                or node.index_maps
            ):
                raise ValueError(
                    "Atlas VPU reciprocal has an unadmitted type, policy, or signature"
                )
        elif node.op == "log2":
            if (
                (node.type.shape, node.type.dtype, node.type.numerical_policy)
                != ((rows, cols), "bf16", contract.record["vpu_log2"]["result_policy"])
                or len(node.inputs) != 1
                or dict(node.attrs)
                or node.index_maps
            ):
                raise ValueError(
                    "Atlas VPU log2 has an unadmitted type, policy, or signature"
                )
        elif node.op == "sqrt":
            if (
                (node.type.shape, node.type.dtype, node.type.numerical_policy)
                != ((rows, cols), "bf16", contract.record["vpu_sqrt"]["result_policy"])
                or len(node.inputs) != 1
                or dict(node.attrs)
                or node.index_maps
            ):
                raise ValueError(
                    "Atlas VPU sqrt has an unadmitted type, policy, or signature"
                )
        elif node.op == "exp2":
            if (
                (node.type.shape, node.type.dtype, node.type.numerical_policy)
                != ((rows, cols), "bf16", contract.record["vpu_exp2"]["result_policy"])
                or len(node.inputs) != 1
                or dict(node.attrs)
                or node.index_maps
            ):
                raise ValueError(
                    "Atlas VPU exp2 has an unadmitted type, policy, or signature"
                )
        elif node.op in {"min", "max"}:
            mode = contract.record[f"vpu_{node.op}"]
            if (
                (node.type.shape, node.type.dtype, node.type.numerical_policy)
                != ((rows, cols), operand["result_dtype"], mode["result_policy"])
                or len(node.inputs) != 2
                or dict(node.attrs)
                or node.index_maps
            ):
                raise ValueError(
                    f"Atlas VPU {node.op} has an unadmitted type, policy, or signature"
                )
        elif node.op in {"physical_col_min", "physical_col_max", "physical_col_sum"}:
            kind = node.op.rsplit("_", 1)[-1]
            mode = contract.record[f"vpu_col_{kind}"]
            if (
                (node.type.shape, node.type.dtype, node.type.numerical_policy)
                != (tuple(mode["output_shape"]), operand["result_dtype"], mode["result_policy"])
                or len(node.inputs) != 1
                or dict(node.attrs)
                or node.index_maps
            ):
                raise ValueError(
                    f"Atlas VPU physical column {kind} has an unadmitted signature"
                )
        elif node.op in {"physical_row_min", "physical_row_max", "physical_row_sum"}:
            kind = node.op.rsplit("_", 1)[-1]
            mode = contract.record[f"vpu_row_{kind}"]
            if (
                (node.type.shape, node.type.dtype, node.type.numerical_policy)
                != (tuple(mode["shape"]), operand["result_dtype"], mode["result_policy"])
                or len(node.inputs) != 1
                or dict(node.attrs)
                or node.index_maps
            ):
                raise ValueError(
                    f"Atlas VPU physical row {kind} has an unadmitted signature"
                )
        elif node.op != "contraction" or (
            node.type.shape,
            node.type.dtype,
            node.type.numerical_policy,
        ) not in {
            ((rows, cols), policy["result_dtype"], policy["result_policy"])
            for policy in (operand, mxu1)
        }:
            raise ValueError(
                "Atlas matrix diagnostic has an unadmitted semantic operation"
            )
    if set(fixed_inputs) != {
        node.id for node in request.nodes if node.op == "input"
    } or (len(fixed_outputs) != len(request.outputs)):
        raise ValueError("MXU0 ABI must bind every ordered input and output")
    effective_limits = limits or SearchLimits()
    selected = snapshot.select(
        request,
        fixed_inputs=fixed_inputs,
        fixed_outputs=fixed_outputs,
        limits=effective_limits,
    )
    if selected.status != "selected" or any(
        item is None
        for item in (
            selected.graph,
            selected.allocation,
            selected.candidate,
            selected.rules,
            selected.exploration,
        )
    ):
        raise RuntimeError(
            f"native MXU0 selection failed: {selected.status}: {selected.reason}"
        )
    replay = check_selection(
        request,
        profile.descriptors,
        selected.rules,
        selected.exploration,
        selected.candidate,
        selected.graph,
        selected.allocation,
        profile.banks,
        fixed_inputs=fixed_inputs,
        fixed_outputs=fixed_outputs,
    )
    if not replay.valid or replay.fingerprint != selected.check_fingerprint:
        raise ValueError("native MXU0 graph changed before materialization")
    constants_blob, constants_plan = _materialize_constants(selected, request, contract)
    assembly, emitted = _materialize(selected, contract)
    assembler = contract.movement.verify_source(rtl_root)
    words = tuple(_load_selected_assembler(assembler).assemble(assembly))
    if not words or len(words) * 4 > 0x20000:
        raise ValueError("selected MXU0 program exceeds diagnostic instruction memory")
    binary = b"".join(word.to_bytes(4, "little") for word in words)
    tile_bytes = contract.record["geometry"]["matrix_register_bytes"]
    rows = contract.record["geometry"]["array_rows"]
    cols = contract.record["geometry"]["array_cols"]
    layout = contract.record["bf16_result_layout"]
    result_registers = layout["register_count"]
    base = contract.movement.record["dram_program_base"]
    tiling_bytes = _canonical(tiling.record()) + b"\n" if tiling is not None else None

    def output_row(index: int, node_id: str) -> dict[str, object]:
        node = request.node(node_id)
        is_fp8 = node.type.dtype == "fp8_e4m3"
        column_mode = next(
            (
                contract.record[f"vpu_col_{kind}"]
                for kind in ("min", "max", "sum")
                if node.type.numerical_policy
                == contract.record[f"vpu_col_{kind}"]["result_policy"]
            ),
            None,
        )
        count = 1 if is_fp8 else result_registers
        physical = (
            {
                "kind": (
                    conversion["pack_row_order"]
                    if node.type.numerical_policy
                    in conversion["pack_result_policies"].values()
                    else contract.record["xlu"]["physical_layout"]
                ),
                "register_count": 1,
                "rows_per_register": rows,
                "columns_per_register": cols,
                "element_bytes": 1,
            }
            if is_fp8
            else {
                "kind": (
                    column_mode["physical_layout"]
                    if column_mode is not None
                    else layout["kind"]
                ),
                "register_count": result_registers,
                "rows_per_register": rows,
                "columns_per_register": cols // result_registers,
                "element_bytes": layout["element_bytes"],
            }
        )
        return {
            "source": node_id,
            "position": index,
            "byte_address": base + fixed_outputs[index] * tile_bytes,
            "byte_length": count * tile_bytes,
            "logical_shape": list(node.type.shape),
            "physical_layout": physical,
            "domain": node.type.numerical_policy,
        }

    plan = {
        "schema": "atlas.native_tensor_diagnostic_plan.v3",
        "scope": "selected-source-linked standalone core, diagnostic static delays",
        "program": {"sha256": _sha(binary), "word_count": len(words)},
        "inputs": [
            {
                "source": node.id,
                "byte_address": base + fixed_inputs[node.id] * tile_bytes,
                "byte_length": (result_registers if node.type.dtype == "bf16" else 1)
                * tile_bytes,
                "preserve": True,
                "domain": node.type.numerical_policy,
            }
            for node in request.nodes
            if node.op == "input"
        ],
        "outputs": [
            output_row(index, node_id) for index, node_id in enumerate(request.outputs)
        ],
        "constants": constants_plan,
        "timing_qualification": "diagnostic_only_no_architectural_mxu_wait",
    }
    if tiling_bytes is not None:
        plan["source_tiling"] = {
            "sha256": _sha(tiling_bytes),
            "host_activity": tiling.host_activity,
        }
    plan_bytes = _canonical(plan) + b"\n"
    manifest: dict[str, object] = {
        "schema": "atlas.native_tensor_compilation.v3",
        "status": "diagnostic_program",
        "engine": "merlin_native",
        "scope": "selected MXU0, bounded MXU1, FP8 XLU, and bounded BF16 VPU ReLU/add/multiply/subtract/square/reciprocal/log2/sqrt/exp2/min/max/physical-column-min-max/physical-row-min-max-sum tiles on source-linked standalone core only",
        "target_identity": contract.identity,
        "request_digest": tiling.source.digest()
        if tiling is not None
        else request.digest(),
        "candidate_digest": selected.candidate.digest(),
        "check_fingerprint": replay.fingerprint,
        "profile_digest": profile.digest(),
        "source_revision": contract.record["rtl_revision"],
        "software_spec_sha256": contract.record["software_spec_sha256"],
        "binary_sha256": _sha(binary),
        "execution_plan_sha256": _sha(plan_bytes),
        "instruction_words": len(words),
        "selected_instructions": len(emitted),
        "candidate_attempts": selected.candidate_attempts,
        "ordering_attempts": selected.ordering_attempts,
        "addresses": selected.allocation.addresses,
        "selected_order": emitted,
        "fixed_inputs": fixed_inputs,
        "fixed_outputs": fixed_outputs,
        "search_limits": asdict(effective_limits),
    }
    if tiling_bytes is not None:
        manifest["lowered_request_digest"] = request.digest()
        manifest["tiling_plan_sha256"] = _sha(tiling_bytes)
    if constants_blob:
        manifest["constants_sha256"] = _sha(constants_blob)
        manifest["constant_count"] = len(constants_plan)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix="atlas-native-mxu0-", dir=destination.parent
    ) as temp:
        root = Path(temp) / "compilation"
        root.mkdir()
        (root / "program.S").write_text(assembly)
        (root / "program.bin").write_bytes(binary)
        if constants_blob:
            (root / "constants.bin").write_bytes(constants_blob)
        (root / "execution_plan.json").write_bytes(plan_bytes)
        (root / "request.json").write_bytes(_canonical(request.record()) + b"\n")
        if tiling_bytes is not None:
            (root / "source_request.json").write_bytes(
                _canonical(tiling.source.record()) + b"\n"
            )
            (root / "tiling.json").write_bytes(tiling_bytes)
        (root / "selected_graph.json").write_bytes(
            _canonical(
                {
                    "values": [asdict(value) for value in selected.graph.values],
                    "outputs": selected.graph.outputs,
                    "order": emitted,
                }
            )
            + b"\n"
        )
        (root / "manifest.json").write_bytes(_canonical(manifest) + b"\n")
        root.rename(destination)
    return manifest


def verify_mxu0_artifact(
    snapshot: NativeSnapshot,
    destination: Path,
    *,
    rtl_root: Path,
    expected_request_digest: str,
    software_spec: Path | None = None,
) -> dict[str, object]:
    """Recompile exact inputs and detect any post-check artifact mutation."""
    manifest = json.loads((destination / "manifest.json").read_text())
    request = KernelRequest.from_record(
        json.loads((destination / "request.json").read_text())
    )
    tiling = None
    source_path = destination / "source_request.json"
    if source_path.exists():
        source = KernelRequest.from_record(json.loads(source_path.read_text()))
        tiling = lower_tiled_contraction(source, Mxu0Contract.load())
        tiling_bytes = _canonical(tiling.record()) + b"\n"
        if (
            tiling.lowered.record() != request.record()
            or (destination / "tiling.json").read_bytes() != tiling_bytes
            or manifest.get("lowered_request_digest") != request.digest()
            or manifest.get("tiling_plan_sha256") != _sha(tiling_bytes)
        ):
            raise ValueError("ordered-K source or tiling changed after verification")
    elif (destination / "tiling.json").exists() or (
        "lowered_request_digest" in manifest or "tiling_plan_sha256" in manifest
    ):
        raise ValueError("unexpected ordered-K source or tiling artifact")
    if (
        tiling.source.digest() if tiling is not None else request.digest()
    ) != expected_request_digest or manifest.get(
        "request_digest"
    ) != expected_request_digest:
        raise ValueError("MXU0 compiled request changed from evaluator-selected input")
    if (
        manifest.get("engine") != "merlin_native"
        or manifest.get("status") != "diagnostic_program"
    ):
        raise ValueError("MXU0 compiled engine or stage status changed")
    inputs = manifest.get("fixed_inputs")
    outputs = manifest.get("fixed_outputs")
    if (
        not isinstance(inputs, dict)
        or not isinstance(outputs, list)
        or (
            not all(type(value) is int for value in inputs.values())
            or not all(type(value) is int for value in outputs)
        )
    ):
        raise ValueError("MXU0 compiled fixed I/O ABI changed")
    limits = SearchLimits(**manifest["search_limits"])
    with tempfile.TemporaryDirectory(
        prefix="atlas-native-mxu0-recheck-", dir=destination.parent
    ) as temp:
        replay_path = Path(temp) / "replay"
        replay = compile_mxu0(
            snapshot,
            request,
            fixed_inputs=inputs,
            fixed_outputs=tuple(outputs),
            rtl_root=rtl_root,
            software_spec=software_spec,
            destination=replay_path,
            limits=limits,
            tiling=tiling,
        )
        names = (
            "manifest.json",
            "request.json",
            "selected_graph.json",
            "program.S",
            "program.bin",
            "execution_plan.json",
        )
        expected_names = set(names) | (
            {"constants.bin"} if request.constants else set()
        )
        if tiling is not None:
            expected_names.update({"source_request.json", "tiling.json"})
        if {item.name for item in destination.iterdir()} != expected_names:
            raise ValueError(
                "compiled MXU0 artifact file set changed after verification"
            )
        for name in sorted(expected_names):
            if (destination / name).read_bytes() != (replay_path / name).read_bytes():
                raise ValueError(
                    f"compiled MXU0 artifact changed after verification: {name}"
                )
    return replay
