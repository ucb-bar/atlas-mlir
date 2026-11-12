"""Check selected mixed MXU and BF16 DAG instruction/register assignments.

Compiler-side admission/replay retains its diagnostic scope. Independent reference
and evaluator authority remain separate; no hardware execution is established.
"""

from __future__ import annotations

from merlin.compiler.semantic.allocate import (
    CandidateGraph,
)
from merlin.compiler.semantic.target_binding import NativeCompilationError

from .policy import (
    BF16_BINARY_MODES,
    BF16_ROW_REDUCTION_MODES,
    BF16_UNARY_MODES,
    PACK_RULE_BY_TRANSPOSE_RULE,
    PACK_RULE_NAMES,
    TRANSPOSE_RULE_NAMES,
)


def _mxu_binary_selection(
    graph: CandidateGraph,
    order: tuple[int, ...],
    names: list[str | None],
    assignment: dict[int, int],
    source_ids: tuple[str, ...],
    input_slots: tuple[int, ...],
    output_slot: int,
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
            raise NativeCompilationError(
                "unqualified_target", "MXU/VPU boundary input binding changed"
            )
    expected_names = {
        "atlas.load_fp8_tile": 2,
        "atlas.load_bf16_tile": 1,
        "atlas.mxu0_reset_tile": 1,
        "atlas.store_bf16_tile": 1,
    }
    binaries = [
        name
        for name in ("atlas.vpu_add_bf16_tile", "atlas.vpu_mul_bf16_tile")
        if name in by_name
    ]
    if (
        len(binaries) != 1
        or any(
            len(by_name.get(name, ())) != count
            for name, count in expected_names.items()
        )
        or len(by_name) != len(expected_names) + 1
        or len(by_name[binaries[0]]) != 1
        or set(order)
        != {value.id for value in graph.values if value.kind == "instruction"}
        or len(order) != 6
    ):
        raise NativeCompilationError(
            "unqualified_target", "selected MXU/VPU instruction DAG changed"
        )
    fp8_loads = by_name["atlas.load_fp8_tile"]
    bf16_load = by_name["atlas.load_bf16_tile"][0]
    mxu = by_name["atlas.mxu0_reset_tile"][0]
    binary = by_name[binaries[0]][0]
    store = by_name["atlas.store_bf16_tile"][0]
    if (
        any(
            len(graph.value(value_id).children) != 1
            or graph.value(graph.value(value_id).children[0]).kind != "input"
            for value_id in (*fp8_loads, bf16_load)
        )
        or graph.value(mxu).children != tuple(fp8_loads)
        or set(graph.value(binary).children) != {mxu, bf16_load}
        or graph.value(store).children != (binary,)
        or graph.outputs != (store,)
        or assignment[store] != output_slot
    ):
        raise NativeCompilationError(
            "unqualified_target",
            "selected MXU/VPU producer, output, or address changed",
        )
    loaded_source_ids = {
        graph.value(graph.value(value_id).children[0]).source_node
        for value_id in (*fp8_loads, bf16_load)
    }
    if loaded_source_ids != set(source_ids):
        raise NativeCompilationError(
            "unqualified_target", "selected MXU/VPU loads omit an input"
        )
    fp8_registers = tuple(assignment[value_id] for value_id in fp8_loads)
    mxu_pair, bf16_pair, result_pair = (
        assignment[value_id] for value_id in (mxu, bf16_load, binary)
    )
    if (
        any(not 0 <= address < 32 for address in fp8_registers)
        or fp8_registers[0] == fp8_registers[1]
        or mxu_pair != 32
        or any(pair % 2 or not 0 <= pair <= 60 for pair in (bf16_pair, result_pair))
        or len({mxu_pair, bf16_pair, result_pair}) != 3
    ):
        raise NativeCompilationError(
            "unqualified_target", "selected MXU/VPU register placement is invalid"
        )
    return {
        "activation": fp8_loads[0],
        "weight": fp8_loads[1],
        "bf16": bf16_load,
        "mxu": mxu,
        "binary": binary,
        "store": store,
    }


def _mxu_dag_selection(
    graph: CandidateGraph,
    order: tuple[int, ...],
    names: list[str | None],
    assignment: dict[int, int],
    source_ids: tuple[str, ...],
    mxu_inputs: dict[str, tuple[str, str]],
    input_slots: tuple[int, ...],
    output_slots: tuple[int, ...],
    *,
    transpose_inputs: dict[str, str] | None = None,
    physical: bool = False,
) -> list[int]:
    """Check a selected MXU0/BF16 DAG before or after physical placement."""
    instructions = {value.id for value in graph.values if value.kind == "instruction"}
    if (
        set(order) != instructions
        or len(order) != len(instructions)
        or len(graph.outputs) != len(output_slots)
        or len(set(graph.outputs)) != len(graph.outputs)
    ):
        raise NativeCompilationError(
            "unqualified_target", "selected MXU/VPU order or outputs changed"
        )
    if sum(names[value_id] == "atlas.mxu0_reset_tile" for value_id in order) != len(
        mxu_inputs
    ):
        raise NativeCompilationError(
            "unqualified_target", "selected MXU/VPU DAG has a changed MXU count"
        )
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
    expected_packed_sources = expected_fp8_sources.difference(
        source_ids, expected_transpose_sources
    ) | set(transpose_inputs.values())
    if not expected_transpose_sources <= expected_fp8_sources.difference(
        source_ids
    ) or any(
        source in source_ids or source in transpose_inputs
        for source in transpose_inputs.values()
    ):
        raise NativeCompilationError(
            "unqualified_target", "MXU transpose sources are inconsistent"
        )
    for value in graph.values:
        name = names[value.id]
        if value.kind in {"input", "constant"}:
            if (
                value.source_node not in source_ids
                or value.storage != "external"
                or assignment[value.id]
                != input_slots[source_ids.index(value.source_node)]
            ):
                raise NativeCompilationError(
                    "unqualified_target", "selected MXU/VPU boundary binding changed"
                )
        elif name in {"atlas.load_fp8_tile", "atlas.load_bf16_tile"}:
            if len(value.children) != 1 or graph.value(value.children[0]).kind not in {
                "input",
                "constant",
            }:
                raise NativeCompilationError(
                    "unqualified_target", "selected MXU/VPU load source changed"
                )
            source = graph.value(value.children[0]).source_node
            if name == "atlas.load_fp8_tile":
                if source not in external_fp8_sources or value.storage != "tensor_fp8":
                    raise NativeCompilationError(
                        "unqualified_target", "selected MXU/VPU FP8 source changed"
                    )
                fp8_sources.add(source)
                address = assignment[value.id]
                if not 0 <= address < 32:
                    raise NativeCompilationError(
                        "unqualified_target", "selected FP8 register is invalid"
                    )
                registers.add(address)
            else:
                if source in external_fp8_sources or value.storage != "tensor_bf16":
                    raise NativeCompilationError(
                        "unqualified_target", "selected MXU/VPU BF16 source changed"
                    )
                pair = assignment[value.id]
                if pair % 2 or not (32 if physical else 0) <= pair <= 60:
                    raise NativeCompilationError(
                        "unqualified_target", "selected BF16 load pair is invalid"
                    )
                registers.update((pair, pair + 1))
        elif name in PACK_RULE_NAMES:
            if (
                value.source_node not in expected_packed_sources
                or value.source_node in packed_sources
                or value.storage != "tensor_fp8"
                or len(value.children) != 1
                or names[value.children[0]]
                not in {
                    "atlas.mxu0_reset_tile",
                    "atlas.vpu_relu_bf16_tile",
                    "atlas.vpu_relu_bf16_from_mxu0_tile",
                    "atlas.vpu_add_bf16_tile",
                    "atlas.vpu_mul_bf16_tile",
                    "atlas.vpu_sub_bf16_tile",
                    *BF16_UNARY_MODES,
                    *BF16_ROW_REDUCTION_MODES,
                }
                or not 0 <= assignment[value.id] < 32
            ):
                raise NativeCompilationError(
                    "unqualified_target", "selected interlayer pack changed"
                )
            packed_sources.add(value.source_node)
            registers.add(assignment[value.id])
        elif name in TRANSPOSE_RULE_NAMES:
            expected_source = transpose_inputs.get(value.source_node)
            expected_pack = PACK_RULE_BY_TRANSPOSE_RULE[name]
            if (
                value.source_node not in expected_transpose_sources
                or value.source_node in transposed_sources
                or value.storage != "tensor_fp8"
                or len(value.children) != 1
                or graph.value(value.children[0]).source_node != expected_source
                or names[value.children[0]] != expected_pack
                or not 0 <= assignment[value.id] < 32
                or assignment[value.id] == assignment[value.children[0]]
            ):
                raise NativeCompilationError(
                    "unqualified_target", "selected packed FP8 transpose changed"
                )
            transposed_sources.add(value.source_node)
            registers.add(assignment[value.id])
        elif name == "atlas.mxu0_reset_tile":
            expected_pair = mxu_inputs.get(value.source_node)
            if (
                value.storage != "tensor_bf16"
                or len(value.children) != 2
                or any(
                    names[child]
                    not in {
                        "atlas.load_fp8_tile",
                        *PACK_RULE_NAMES,
                        *TRANSPOSE_RULE_NAMES,
                    }
                    for child in value.children
                )
                or tuple(graph.value(child).source_node for child in value.children)
                != expected_pair
                or value.source_node in selected_mxus
                or (assignment[value.id] % 2 or not 32 <= assignment[value.id] <= 60)
            ):
                raise NativeCompilationError(
                    "unqualified_target", "selected MXU0 operands or readout changed"
                )
            selected_mxus.add(value.source_node)
            registers.update((assignment[value.id], assignment[value.id] + 1))
        elif name in {
            "atlas.vpu_relu_bf16_tile",
            "atlas.vpu_relu_bf16_from_mxu0_tile",
            *BF16_UNARY_MODES,
            *BF16_ROW_REDUCTION_MODES,
            *BF16_BINARY_MODES,
        }:
            unary = (
                name.startswith("atlas.vpu_relu")
                or name in BF16_UNARY_MODES
                or name in BF16_ROW_REDUCTION_MODES
            )
            if (
                value.storage != "tensor_bf16"
                or len(value.children) != (1 if unary else 2)
                or (not unary and value.children[0] == value.children[1])
                or any(
                    names[child]
                    not in {
                        "atlas.load_bf16_tile",
                        "atlas.mxu0_reset_tile",
                        "atlas.vpu_relu_bf16_tile",
                        "atlas.vpu_relu_bf16_from_mxu0_tile",
                        *BF16_UNARY_MODES,
                        *BF16_ROW_REDUCTION_MODES,
                        *BF16_BINARY_MODES,
                    }
                    for child in value.children
                )
            ):
                raise NativeCompilationError(
                    "unqualified_target", "selected MXU/VPU operand graph changed"
                )
            if (
                name == "atlas.vpu_relu_bf16_from_mxu0_tile"
                and names[value.children[0]] != "atlas.mxu0_reset_tile"
            ):
                raise NativeCompilationError(
                    "unqualified_target", "MXU0-specific ReLU lost its MXU source"
                )
            pair = assignment[value.id]
            if (
                pair % 2
                or not (32 if physical else 0) <= pair <= 60
                or pair in {assignment[child] for child in value.children}
            ):
                raise NativeCompilationError(
                    "unqualified_target", "selected MXU/VPU result pair is invalid"
                )
            registers.update((pair, pair + 1))
        elif name in {"atlas.store_bf16_tile", "atlas.store_bf16_mxu0_tile"}:
            if (
                value.id not in output_indices
                or len(value.children) != 1
                or value.storage != "external"
                or assignment[value.id] != output_slots[output_indices[value.id]]
            ):
                raise NativeCompilationError(
                    "unqualified_target", "selected MXU/VPU output binding changed"
                )
            producer = names[value.children[0]]
            if (
                name == "atlas.store_bf16_mxu0_tile"
                and producer != "atlas.mxu0_reset_tile"
            ) or (
                name == "atlas.store_bf16_tile"
                and producer
                not in {
                    "atlas.load_bf16_tile",
                    "atlas.vpu_relu_bf16_tile",
                    "atlas.vpu_relu_bf16_from_mxu0_tile",
                    "atlas.vpu_add_bf16_tile",
                    "atlas.vpu_mul_bf16_tile",
                    "atlas.vpu_sub_bf16_tile",
                    *BF16_UNARY_MODES,
                    *BF16_ROW_REDUCTION_MODES,
                }
            ):
                raise NativeCompilationError(
                    "unqualified_target", "selected MXU/VPU output producer changed"
                )
        else:
            raise NativeCompilationError(
                "unqualified_target",
                "selected MXU/VPU DAG has an unadmitted instruction",
            )
    if (
        selected_mxus != set(mxu_inputs)
        or fp8_sources != external_fp8_sources
        or packed_sources != expected_packed_sources
        or transposed_sources != expected_transpose_sources
    ):
        raise NativeCompilationError(
            "unqualified_target", "selected MXU/VPU loads omit an FP8 input"
        )
    return sorted(registers)
