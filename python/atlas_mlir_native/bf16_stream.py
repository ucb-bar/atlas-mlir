"""Replay selected BF16 tile allocation and emitted command streams.

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
    DRAM_BASE,
    TILE_BYTES,
)
from .transfer import _check_atomic_dma_schedule, _tile_effects


def _check_bf16_selection(
    graph: CandidateGraph,
    order: tuple[int, ...],
    names: list[str | None],
    assignment: dict[int, int],
    source_ids: tuple[str, ...],
    input_slots: tuple[int, ...],
    output_slots: tuple[int, ...],
) -> list[int]:
    """Check selected BF16 graph shape and bindings before OOT materialization."""
    instructions = {value.id for value in graph.values if value.kind == "instruction"}
    if (
        set(order) != instructions
        or len(order) != len(instructions)
        or (
            len(graph.outputs) != len(output_slots)
            or len(set(graph.outputs)) != len(graph.outputs)
        )
    ):
        raise NativeCompilationError(
            "unqualified_target", "selected BF16 order or outputs changed"
        )
    output_indices = {value_id: index for index, value_id in enumerate(graph.outputs)}
    registers: set[int] = set()
    for value in graph.values:
        if value.kind == "input":
            if (
                value.source_node not in source_ids
                or value.storage != "external"
                or (
                    assignment[value.id]
                    != input_slots[source_ids.index(value.source_node)]
                )
            ):
                raise NativeCompilationError(
                    "unqualified_target", "selected BF16 input binding changed"
                )
        elif names[value.id] == "atlas.load_bf16_tile":
            if (
                len(value.children) != 1
                or graph.value(value.children[0]).kind != "input"
                or (value.storage != "tensor_bf16")
            ):
                raise NativeCompilationError(
                    "unqualified_target", "selected BF16 load has invalid source"
                )
            pair = assignment[value.id]
            if pair % 2 or not 0 <= pair <= 60:
                raise NativeCompilationError(
                    "unqualified_target", "selected BF16 pair is invalid"
                )
            registers.update((pair, pair + 1))
        elif names[value.id] in {"atlas.vpu_relu_bf16_tile", *BF16_UNARY_MODES}:
            if (
                len(value.children) != 1
                or names[value.children[0]]
                not in {
                    "atlas.load_bf16_tile",
                    "atlas.vpu_relu_bf16_tile",
                    *BF16_UNARY_MODES,
                    *BF16_ROW_REDUCTION_MODES,
                    *BF16_BINARY_MODES,
                }
                or value.storage != "tensor_bf16"
            ):
                raise NativeCompilationError(
                    "unqualified_target", "selected BF16 unary has invalid source"
                )
            pair = assignment[value.id]
            if pair % 2 or not 0 <= pair <= 60 or pair == assignment[value.children[0]]:
                raise NativeCompilationError(
                    "unqualified_target", "selected BF16 unary pair is invalid"
                )
            registers.update((pair, pair + 1))
        elif names[value.id] in BF16_ROW_REDUCTION_MODES:
            if (
                len(value.children) != 1
                or names[value.children[0]]
                not in {
                    "atlas.load_bf16_tile",
                    "atlas.vpu_relu_bf16_tile",
                    *BF16_UNARY_MODES,
                    *BF16_ROW_REDUCTION_MODES,
                    *BF16_BINARY_MODES,
                }
                or value.storage != "tensor_bf16"
            ):
                raise NativeCompilationError(
                    "unqualified_target",
                    "selected BF16 row reduction has invalid source",
                )
            pair = assignment[value.id]
            if pair % 2 or not 0 <= pair <= 60 or pair == assignment[value.children[0]]:
                raise NativeCompilationError(
                    "unqualified_target", "selected BF16 row reduction pair is invalid"
                )
            registers.update((pair, pair + 1))
        elif names[value.id] in BF16_BINARY_MODES:
            sources = {
                "atlas.load_bf16_tile",
                "atlas.vpu_relu_bf16_tile",
                *BF16_UNARY_MODES,
                *BF16_ROW_REDUCTION_MODES,
                *BF16_BINARY_MODES,
            }
            if (
                len(value.children) != 2
                or any(names[child] not in sources for child in value.children)
                or (
                    value.children[0] == value.children[1]
                    or value.storage != "tensor_bf16"
                )
            ):
                raise NativeCompilationError(
                    "unqualified_target", "selected BF16 binary op has invalid sources"
                )
            pair = assignment[value.id]
            sources = tuple(assignment[child] for child in value.children)
            if (
                pair % 2
                or not 0 <= pair <= 60
                or pair in sources
                or sources[0] == sources[1]
            ):
                raise NativeCompilationError(
                    "unqualified_target", "selected BF16 binary op pairs overlap"
                )
            registers.update((pair, pair + 1))
        elif names[value.id] == "atlas.store_bf16_tile":
            if (
                value.id not in output_indices
                or len(value.children) != 1
                or (
                    names[value.children[0]]
                    not in {
                        "atlas.load_bf16_tile",
                        "atlas.vpu_relu_bf16_tile",
                        *BF16_UNARY_MODES,
                        *BF16_ROW_REDUCTION_MODES,
                        *BF16_BINARY_MODES,
                    }
                    or value.storage != "external"
                    or assignment[value.id] != output_slots[output_indices[value.id]]
                )
            ):
                raise NativeCompilationError(
                    "unqualified_target", "selected BF16 output binding changed"
                )
        else:
            raise NativeCompilationError(
                "unqualified_target",
                "selected BF16 graph has an unadmitted instruction",
            )
    return sorted(registers)


def _realized_bf16_assignment(
    program: dict,
    *,
    graph: CandidateGraph,
    order: tuple[int, ...],
    names: list[str | None],
    selected: dict[int, int],
) -> dict[int, int]:
    """Recover tensor pairs from the structured physical stream for replay."""
    effects = _tile_effects(program)
    loads = [effect for effect in effects if effect[0] == "vload"]
    binaries = [effect for effect in effects if effect[0] in BF16_BINARY_MODES.values()]
    unaries = [
        effect
        for effect in effects
        if effect[0] in {"relu", *BF16_UNARY_MODES.values()}
    ]
    reductions = [
        effect for effect in effects if effect[0] in BF16_ROW_REDUCTION_MODES.values()
    ]
    expected_loads = sum(
        names[value_id] == "atlas.load_bf16_tile" for value_id in order
    )
    expected_binaries = sum(names[value_id] in BF16_BINARY_MODES for value_id in order)
    expected_unaries = sum(
        names[value_id] in {"atlas.vpu_relu_bf16_tile", *BF16_UNARY_MODES}
        for value_id in order
    )
    expected_reductions = sum(
        names[value_id] in BF16_ROW_REDUCTION_MODES for value_id in order
    )
    if (len(loads), len(binaries), len(unaries), len(reductions)) != (
        2 * expected_loads,
        expected_binaries,
        expected_unaries,
        expected_reductions,
    ):
        raise NativeCompilationError(
            "unqualified_target",
            "emitted BF16 instruction count differs from selection",
        )
    realized = dict(selected)
    load_index = binary_index = unary_index = reduction_index = 0
    for value_id in order:
        value = graph.value(value_id)
        name = names[value_id]
        if name == "atlas.load_bf16_tile":
            first, second = loads[load_index : load_index + 2]
            load_index += 2
            if type(first[1]) is not int or second[1] != first[1] + 1:
                raise NativeCompilationError(
                    "unqualified_target",
                    "emitted BF16 load has no adjacent register pair",
                )
            realized[value_id] = first[1]
        elif name in BF16_BINARY_MODES:
            effect = binaries[binary_index]
            binary_index += 1
            kind = BF16_BINARY_MODES[name]
            if effect[0] != kind:
                raise NativeCompilationError(
                    "unqualified_target",
                    "emitted BF16 binary mode differs from selection",
                )
            if (effect[1], effect[2]) != tuple(
                realized[child] for child in value.children
            ):
                raise NativeCompilationError(
                    "unqualified_target",
                    f"emitted BF16 {kind} reads different source pairs",
                )
            realized[value_id] = effect[3]
        elif name in {"atlas.vpu_relu_bf16_tile", *BF16_UNARY_MODES}:
            effect = unaries[unary_index]
            unary_index += 1
            kind = (
                "relu" if name == "atlas.vpu_relu_bf16_tile" else BF16_UNARY_MODES[name]
            )
            if effect[0] != kind or effect[1] != realized[value.children[0]]:
                raise NativeCompilationError(
                    "unqualified_target",
                    "emitted BF16 unary reads a different source pair",
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
                    "unqualified_target", "emitted BF16 row reduction changed"
                )
            realized[value_id] = effect[2]
    return realized


def check_bf16_tile_stream(
    program: dict,
    *,
    graph: CandidateGraph,
    order: tuple[int, ...],
    names: list[str | None],
    assignment: dict[int, int],
    source_ids: tuple[str, ...],
    input_slots: tuple[int, ...],
    output_slots: tuple[int, ...],
) -> None:
    """Replay BF16 loads, arithmetic, and stores from the selected DAG and fixed ABI."""
    expected: list[tuple] = [("config",)]
    output_indices = {value_id: index for index, value_id in enumerate(graph.outputs)}
    indexed_inputs = all(
        slot == input_slots[0] + index for index, slot in enumerate(input_slots)
    )
    indexed_outputs = all(
        slot == output_slots[0] + index for index, slot in enumerate(output_slots)
    )
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
                            ("vload", pair + half, scratch + half * 256),
                        )
                    )
            else:
                expected.extend(
                    (
                        ("dma", "load", address, 0x20000, 2048, 0),
                        ("wait", 0),
                        ("vload", pair, 0x20000),
                        ("vload", pair + 1, 0x20100),
                    )
                )
        elif name in {"atlas.vpu_relu_bf16_tile", *BF16_UNARY_MODES}:
            kind = (
                "relu" if name == "atlas.vpu_relu_bf16_tile" else BF16_UNARY_MODES[name]
            )
            expected.append((kind, assignment[value.children[0]], assignment[value_id]))
        elif name in BF16_ROW_REDUCTION_MODES:
            expected.append(
                (
                    BF16_ROW_REDUCTION_MODES[name],
                    assignment[value.children[0]],
                    assignment[value_id],
                )
            )
        elif name in BF16_BINARY_MODES:
            expected.append(
                (
                    BF16_BINARY_MODES[name],
                    assignment[value.children[0]],
                    assignment[value.children[1]],
                    assignment[value_id],
                )
            )
        elif name == "atlas.store_bf16_tile":
            output_index = output_indices[value_id]
            address = DRAM_BASE + output_slots[output_index] * TILE_BYTES
            pair = assignment[value.children[0]]
            if indexed_outputs:
                scratch = 0x10000 + output_index * 512
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
            else:
                expected.extend(
                    (
                        ("vstore", pair, 0x20000),
                        ("vstore", pair + 1, 0x20100),
                        ("dma", "store", address, 0x20000, 2048, 1),
                        ("wait", 1),
                    )
                )
        else:
            raise NativeCompilationError(
                "unqualified_target", "BF16 stream checker has an unknown selection"
            )
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
                raise NativeCompilationError(
                    "unqualified_target", "BF16 unary reads an unavailable pair"
                )
            ready_at[fields["dst"]] = cycle + 66
            ready_at[fields["dst"] + 1] = cycle + 66
        elif op == "atlas.vpu_reduce":
            source = fields["src"]
            if any(ready_at.get(source + half, 1 << 60) > cycle for half in (0, 1)):
                raise NativeCompilationError(
                    "unqualified_target", "BF16 row reduction reads an unavailable pair"
                )
            # The selected row reductions issue across 32 rows. The last
            # row maximum writes at age 33; row sum writes at age 38.
            completion = 39 if fields["kind"] == "row_sum" else 34
            ready_at[fields["dst"]] = cycle + completion
            ready_at[fields["dst"] + 1] = cycle + completion
        elif op == "atlas.vpu_binary":
            for source in (fields["lhs"], fields["rhs"]):
                if any(ready_at.get(source + half, 1 << 60) > cycle for half in (0, 1)):
                    raise NativeCompilationError(
                        "unqualified_target", "BF16 binary op reads an unavailable pair"
                    )
            ready_at[fields["dst"]] = cycle + 66
            ready_at[fields["dst"] + 1] = cycle + 66
        elif op == "atlas.vstore":
            source = fields["src"]
            if ready_at.get(source, 1 << 60) > cycle:
                raise NativeCompilationError(
                    "unqualified_target", "BF16 store reads an unavailable register"
                )
            pending_stores.append(cycle)
        elif op == "atlas.dma" and fields["direction"] == "store":
            # Indexed outputs transfer each half separately. An explicitly
            # addressed BF16 DMA transfers both halves after two VSTOREs.
            count = 1 if indexed_outputs else 2
            if len(pending_stores) < count or any(
                cycle - issued < 35 for issued in pending_stores[:count]
            ):
                raise NativeCompilationError(
                    "unqualified_target",
                    "BF16 DMA reads scratch before store completes",
                )
            del pending_stores[:count]
        elif op == "atlas.delay":
            count = fields.get("cycles")
            if type(count) is not int or not 0 <= count <= 65535:
                raise NativeCompilationError(
                    "unqualified_target", "BF16 stream has invalid delay"
                )
            cycle += count
        cycle += 1
    if pending_stores:
        raise NativeCompilationError(
            "unqualified_target", "BF16 scratch store has no DMA completion"
        )
