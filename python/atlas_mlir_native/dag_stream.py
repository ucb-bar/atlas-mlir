"""Replay emitted mixed MXU/BF16 DAGs against their selected assignments.

Compiler-side admission/replay retains its diagnostic scope. Independent reference
and evaluator authority remain separate; no hardware execution is established.
"""

from __future__ import annotations

from merlin.semantic_compiler.allocate import (
    CandidateGraph,
)
from merlin.semantic_compiler.target_binding import NativeCompilationError

from .dag_selection import _mxu_binary_selection, _mxu_dag_selection
from .pack_lane import ROUNDED_PACK_POLICY
from .pack_stream import _check_pack_roundtrip_words, _pack_roundtrip_effects
from .policy import (
    BF16_BINARY_MODES,
    BF16_ROW_REDUCTION_MODES,
    BF16_UNARY_MODES,
    DRAM_BASE,
    PACK_RULE_BY_POLICY,
    PACK_RULE_NAMES,
    SCALED_PACK_POLICY,
    TILE_BYTES,
    TRANSPOSE_RULE_NAMES,
)
from .transfer import _check_mxu_dag_timing, _tile_effects


def check_mxu_bf16_binary_stream(
    program: dict,
    *,
    graph: CandidateGraph,
    order: tuple[int, ...],
    names: list[str | None],
    assignment: dict[int, int],
    source_ids: tuple[str, ...],
    input_slots: tuple[int, ...],
    output_slot: int,
) -> dict[int, int]:
    """Replay an emitted MXU0-to-VPU tile against its selected DAG and ABI."""
    ids = _mxu_binary_selection(
        graph,
        order,
        names,
        assignment,
        source_ids,
        input_slots,
        output_slot,
    )
    effects = _tile_effects(program)
    loads = [effect for effect in effects if effect[0] == "vload"]
    pops = [effect for effect in effects if effect[0] == "mxu_pop"]
    binaries = [effect for effect in effects if effect[0] in {"add", "mul"}]
    if (len(loads), len(pops), len(binaries)) != (4, 1, 1):
        raise NativeCompilationError(
            "unqualified_target", "MXU/VPU physical instruction counts changed"
        )
    realized = dict(assignment)
    load_index = 0
    for value_id in order:
        name = names[value_id]
        if name == "atlas.load_fp8_tile":
            realized[value_id] = loads[load_index][1]
            load_index += 1
        elif name == "atlas.load_bf16_tile":
            first, second = loads[load_index : load_index + 2]
            if second[1] != first[1] + 1:
                raise NativeCompilationError(
                    "unqualified_target", "MXU/VPU BF16 input is not a pair"
                )
            realized[value_id] = first[1]
            load_index += 2
        elif name == "atlas.mxu0_reset_tile":
            realized[value_id] = pops[0][5]
        elif name in {"atlas.vpu_add_bf16_tile", "atlas.vpu_mul_bf16_tile"}:
            realized[value_id] = binaries[0][3]
    ids = _mxu_binary_selection(
        graph,
        order,
        names,
        realized,
        source_ids,
        input_slots,
        output_slot,
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
                expected.extend(
                    (
                        (
                            "dma",
                            "load",
                            address + half * 1024,
                            scratch + half * 256,
                            1024,
                            0,
                        ),
                        ("wait", 0),
                        ("vload", realized[value_id] + half, scratch + half * 256),
                    )
                )
        elif name == "atlas.mxu0_reset_tile":
            activation, weight = (realized[child] for child in value.children)
            expected.extend(
                (
                    ("mxu_push", 0, 0, "weight_fp8", weight),
                    ("mxu_matmul", 0, 0, 0, False, activation),
                    ("mxu_pop", 0, 0, "bf16", 0, realized[value_id]),
                )
            )
        elif name in {"atlas.vpu_add_bf16_tile", "atlas.vpu_mul_bf16_tile"}:
            kind = "add" if name == "atlas.vpu_add_bf16_tile" else "mul"
            expected.append(
                (
                    kind,
                    realized[value.children[0]],
                    realized[value.children[1]],
                    realized[value_id],
                )
            )
        elif name == "atlas.store_bf16_tile":
            pair = realized[value.children[0]]
            address = DRAM_BASE + output_slot * TILE_BYTES
            for half in (0, 1):
                expected.extend(
                    (
                        ("vstore", pair + half, 0x10000 + half * 256),
                        (
                            "dma",
                            "store",
                            address + half * 1024,
                            0x10000 + half * 256,
                            1024,
                            1,
                        ),
                        ("wait", 1),
                    )
                )
    expected.extend((("complete",), ("trap",)))
    if effects != expected:
        raise NativeCompilationError(
            "unqualified_target", "MXU/VPU emitted effects differ from selection"
        )

    cycle = 0
    issued: dict[str, list[int]] = {}
    for row in program["instructions"]:
        op = row["operation"]
        if op == "atlas.delay":
            count = row["fields"].get("cycles")
            if type(count) is not int or not 0 <= count <= 65535:
                raise NativeCompilationError(
                    "unqualified_target", "MXU/VPU delay has invalid count"
                )
            cycle += count
        elif op in {
            "atlas.vload",
            "atlas.mxu_push",
            "atlas.mxu_matmul",
            "atlas.mxu_pop",
            "atlas.vpu_binary",
            "atlas.vstore",
        } or (op == "atlas.dma" and row["fields"].get("direction") == "store"):
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
        valid = all(
            issued[end][end_index] - issued[start][start_index] >= minimum
            for start, start_index, end, end_index, minimum in gaps
        )
    except (KeyError, IndexError) as exc:
        raise NativeCompilationError(
            "unqualified_target", "MXU/VPU issue event is missing"
        ) from exc
    if not valid:
        raise NativeCompilationError(
            "unqualified_target", "MXU/VPU issued before diagnostic availability"
        )
    return realized


def check_mxu_bf16_dag_stream(
    program: dict,
    *,
    graph: CandidateGraph,
    order: tuple[int, ...],
    names: list[str | None],
    assignment: dict[int, int],
    source_ids: tuple[str, ...],
    mxu_inputs: dict[str, tuple[str, str]],
    input_slots: tuple[int, ...],
    output_slots: tuple[int, ...],
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
                raise NativeCompilationError(
                    "unqualified_target", "unit pack has a nonunit scale"
                )
            return 127
        if (
            name
            in {
                PACK_RULE_BY_POLICY[SCALED_PACK_POLICY],
                PACK_RULE_BY_POLICY[ROUNDED_PACK_POLICY],
            }
            and type(code) is int
            and code in range(122, 129)
        ):
            return code
        raise NativeCompilationError(
            "unqualified_target", "scaled interlayer pack lacks its selected code"
        )

    _mxu_dag_selection(
        graph,
        order,
        names,
        assignment,
        source_ids,
        mxu_inputs,
        input_slots,
        output_slots,
        transpose_inputs=transpose_inputs,
    )
    effects = _tile_effects(program)
    loads = [effect for effect in effects if effect[0] == "vload"]
    pops = [
        effect for effect in effects if effect[0] == "mxu_pop" and effect[3] == "bf16"
    ]
    converted = [
        effect for effect in effects if effect[0] == "mxu_pop" and effect[3] == "fp8"
    ]
    packs = [effect for effect in effects if effect[0] == "pack"]
    transposes = [effect for effect in effects if effect[0] == "transpose"]
    unaries = [
        effect
        for effect in effects
        if effect[0] in {"relu", *BF16_UNARY_MODES.values()}
    ]
    reductions = [
        effect for effect in effects if effect[0] in BF16_ROW_REDUCTION_MODES.values()
    ]
    binaries = [effect for effect in effects if effect[0] in BF16_BINARY_MODES.values()]
    expected_loads = sum(
        1
        if names[value_id] == "atlas.load_fp8_tile"
        else 2
        if names[value_id] == "atlas.load_bf16_tile"
        else 0
        for value_id in order
    )
    expected_packs = sum(names[value_id] in PACK_RULE_NAMES for value_id in order)
    expected_transposes = sum(
        names[value_id] in TRANSPOSE_RULE_NAMES for value_id in order
    )
    expected_unaries = sum(
        names[value_id]
        in {
            "atlas.vpu_relu_bf16_tile",
            "atlas.vpu_relu_bf16_from_mxu0_tile",
            *BF16_UNARY_MODES,
        }
        for value_id in order
    )
    expected_reductions = sum(
        names[value_id] in BF16_ROW_REDUCTION_MODES for value_id in order
    )
    expected_binaries = sum(names[value_id] in BF16_BINARY_MODES for value_id in order)
    if (
        len(loads),
        len(pops),
        len(packs),
        len(transposes),
        len(unaries),
        len(reductions),
        len(binaries),
    ) != (
        expected_loads,
        len(mxu_inputs),
        2 * expected_packs,
        expected_transposes,
        expected_unaries,
        expected_reductions,
        expected_binaries,
    ):
        raise NativeCompilationError(
            "unqualified_target", "MXU/VPU physical instruction counts changed"
        )
    if len(converted) != expected_packs:
        raise NativeCompilationError(
            "unqualified_target", "interlayer pack copy/pop count changed"
        )
    roundtrip_pairs: dict[int, int] = {}
    realized = dict(assignment)
    load_index = pop_index = pack_index = transpose_index = unary_index = (
        reduction_index
    ) = binary_index = 0
    for value_id in order:
        value = graph.value(value_id)
        name = names[value_id]
        if name == "atlas.load_fp8_tile":
            realized[value_id] = loads[load_index][1]
            load_index += 1
        elif name == "atlas.load_bf16_tile":
            first, second = loads[load_index : load_index + 2]
            if second[1] != first[1] + 1:
                raise NativeCompilationError(
                    "unqualified_target", "MXU/VPU BF16 input is not a pair"
                )
            realized[value_id] = first[1]
            load_index += 2
        elif name == "atlas.mxu0_reset_tile":
            realized[value_id] = pops[pop_index][5]
            pop_index += 1
        elif name in PACK_RULE_NAMES:
            pack_code(value_id)
            effect, unpack = packs[2 * pack_index : 2 * pack_index + 2]
            result = converted[pack_index]
            pack_index += 1
            if (
                effect != ("pack", "bf16_to_fp8", realized[value.children[0]], 62, 3)
                or unpack[:3] != ("pack", "fp8_to_bf16", 62)
                or unpack[4] != 3
                or result[:5] != ("mxu_pop", 0, 0, "fp8", 3)
            ):
                raise NativeCompilationError(
                    "unqualified_target", "interlayer pack source or conversion changed"
                )
            roundtrip_pairs[value_id] = unpack[3]
            realized[value_id] = result[5]
        elif name in TRANSPOSE_RULE_NAMES:
            effect = transposes[transpose_index]
            transpose_index += 1
            if effect[1] != realized[value.children[0]]:
                raise NativeCompilationError(
                    "unqualified_target", "packed FP8 transpose source changed"
                )
            realized[value_id] = effect[2]
        elif name in {
            "atlas.vpu_relu_bf16_tile",
            "atlas.vpu_relu_bf16_from_mxu0_tile",
            *BF16_UNARY_MODES,
        }:
            effect = unaries[unary_index]
            unary_index += 1
            kind = BF16_UNARY_MODES.get(name, "relu")
            if effect[0] != kind or effect[1] != realized[value.children[0]]:
                raise NativeCompilationError(
                    "unqualified_target", "MXU/VPU unary source or mode changed"
                )
            realized[value_id] = effect[2]
        elif name in BF16_ROW_REDUCTION_MODES:
            effect = reductions[reduction_index]
            reduction_index += 1
            if (
                effect[0] != BF16_ROW_REDUCTION_MODES[name]
                or effect[1] != realized[value.children[0]]
            ):
                raise NativeCompilationError(
                    "unqualified_target", "MXU/VPU row reduction changed"
                )
            realized[value_id] = effect[2]
        elif name in BF16_BINARY_MODES:
            effect = binaries[binary_index]
            binary_index += 1
            kind = BF16_BINARY_MODES[name]
            if effect[0] != kind or effect[1:3] != tuple(
                realized[child] for child in value.children
            ):
                raise NativeCompilationError(
                    "unqualified_target", "MXU/VPU binary kind or sources changed"
                )
            realized[value_id] = effect[3]
    _mxu_dag_selection(
        graph,
        order,
        names,
        realized,
        source_ids,
        mxu_inputs,
        input_slots,
        output_slots,
        transpose_inputs=transpose_inputs,
        physical=True,
    )

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
                expected.extend(
                    (
                        (
                            "dma",
                            "load",
                            address + half * 1024,
                            scratch + half * 256,
                            1024,
                            0,
                        ),
                        ("wait", 0),
                        ("vload", realized[value_id] + half, scratch + half * 256),
                    )
                )
        elif name == "atlas.mxu0_reset_tile":
            activation, weight = (realized[child] for child in value.children)
            expected.extend(
                (
                    ("mxu_push", 0, 0, "weight_fp8", weight),
                    ("mxu_matmul", 0, 0, 0, False, activation),
                    ("mxu_pop", 0, 0, "bf16", 0, realized[value_id]),
                )
            )
        elif name in {
            "atlas.vpu_relu_bf16_tile",
            "atlas.vpu_relu_bf16_from_mxu0_tile",
            *BF16_UNARY_MODES,
        }:
            expected.append(
                (
                    BF16_UNARY_MODES.get(name, "relu"),
                    realized[value.children[0]],
                    realized[value_id],
                )
            )
        elif name in BF16_ROW_REDUCTION_MODES:
            expected.append(
                (
                    BF16_ROW_REDUCTION_MODES[name],
                    realized[value.children[0]],
                    realized[value_id],
                )
            )
        elif name in BF16_BINARY_MODES:
            kind = BF16_BINARY_MODES[name]
            expected.append(
                (
                    kind,
                    realized[value.children[0]],
                    realized[value.children[1]],
                    realized[value_id],
                )
            )
        elif name in PACK_RULE_NAMES:
            expected.extend(
                _pack_roundtrip_effects(
                    realized[value.children[0]],
                    roundtrip_pairs[value_id],
                    realized[value_id],
                    pack_code(value_id),
                )
            )
        elif name in TRANSPOSE_RULE_NAMES:
            expected.append(
                ("transpose", realized[value.children[0]], realized[value_id])
            )
        elif name in {"atlas.store_bf16_tile", "atlas.store_bf16_mxu0_tile"}:
            index = output_indices[value_id]
            address = DRAM_BASE + output_slots[index] * TILE_BYTES
            pair = realized[value.children[0]]
            scratch = 0x10000 + index * 512
            for half in (0, 1):
                expected.extend(
                    (
                        ("vstore", pair + half, scratch + half * 256),
                        (
                            "dma",
                            "store",
                            address + half * 1024,
                            scratch + half * 256,
                            1024,
                            1,
                        ),
                        ("wait", 1),
                    )
                )
    expected.extend((("complete",), ("trap",)))
    if effects != expected:
        mismatch = next(
            (
                index
                for index, pair in enumerate(zip(effects, expected))
                if pair[0] != pair[1]
            ),
            min(len(effects), len(expected)),
        )
        actual = effects[mismatch] if mismatch < len(effects) else None
        wanted = expected[mismatch] if mismatch < len(expected) else None
        raise NativeCompilationError(
            "unqualified_target",
            f"MXU/VPU DAG effect {mismatch} differs: selected {wanted}, emitted {actual}",
        )
    _check_pack_roundtrip_words(program)
    _check_mxu_dag_timing(program)
    return realized
