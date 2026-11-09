"""Replay admitted MXU tile and ordered reduction-chain command streams.

Compiler-side admission/replay retains its diagnostic scope. Independent reference
and evaluator authority remain separate; no hardware execution is established.
"""

from __future__ import annotations

from merlin.semantic_compiler.allocate import (
    CandidateGraph,
)
from merlin.semantic_compiler.target_binding import NativeCompilationError

from .policy import DRAM_SLOTS
from .transfer import _tile_effects


def check_mxu_stream(
    program: dict,
    *,
    unit: int,
    activation_address: int,
    weight_address: int,
    output_address: int,
    activation: int,
    weight: int,
    pair: int,
    relu_pair: int | None = None,
) -> None:
    if unit not in (0, 1) or (unit == 1 and relu_pair is not None):
        raise NativeCompilationError(
            "unsupported_semantics", "selected MXU stream mode is unavailable"
        )
    effects = _tile_effects(program)
    expected = [
        ("config",),
        ("dma", "load", activation_address, 0, 1024, 0),
        ("wait", 0),
        ("vload", activation, 0),
        ("dma", "load", weight_address, 512, 1024, 0),
        ("wait", 0),
        ("vload", weight, 512),
        ("mxu_push", unit, 0, "weight_fp8", weight),
        ("mxu_matmul", unit, 0, 0, False, activation),
        ("mxu_pop", unit, 0, "bf16", 0, pair),
    ]
    if relu_pair is not None:
        if relu_pair % 2 or not 0 <= relu_pair <= 60 or relu_pair == pair:
            raise NativeCompilationError(
                "unqualified_target", "MXU ReLU pair is invalid"
            )
        expected.append(("relu", pair, relu_pair))
    output_pair = relu_pair if relu_pair is not None else pair
    expected.extend(
        [
            ("vstore", output_pair, 0x10000),
            ("dma", "store", output_address, 0x10000, 1024, 1),
            ("wait", 1),
            ("vstore", output_pair + 1, 0x10100),
            ("dma", "store", output_address + 1024, 0x10100, 1024, 1),
            ("wait", 1),
            ("complete",),
            ("trap",),
        ]
    )
    if effects != expected:
        raise NativeCompilationError(
            "unqualified_target",
            "emitted MXU stream differs in addresses, resource slots, order, or waits",
        )
    # Replay conservative issue gaps from the emitted words, independently of
    # the virtual schedule verifier. These selected-core timing floors do not
    # qualify host launch or dynamic memory completion.
    issued: dict[str, list[int]] = {}
    cycle = 0
    for row in program["instructions"]:
        op = row["operation"]
        if op in {
            "atlas.vload",
            "atlas.mxu_push",
            "atlas.mxu_matmul",
            "atlas.mxu_pop",
            "atlas.vpu_unary",
            "atlas.vstore",
        } or (op == "atlas.dma" and row["fields"].get("direction") == "store"):
            issued.setdefault(op, []).append(cycle)
        if op == "atlas.delay":
            count = row["fields"].get("cycles")
            if type(count) is not int or not 0 <= count <= 65535:
                raise NativeCompilationError(
                    "unqualified_target", "MXU delay has invalid cycle count"
                )
            cycle += count
        cycle += 1
    gaps = [
        ("atlas.vload", 0, "atlas.mxu_matmul", 0, 34),
        ("atlas.vload", 1, "atlas.mxu_push", 0, 34),
        # accumulatorRow0 in lib/AtlasTiming.cpp: the first cycle a pop may read the matmul's accumulator.
        ("atlas.mxu_matmul", 0, "atlas.mxu_pop", 0, (64, 4)[unit]),
        ("atlas.vstore", 0, "atlas.dma", 0, 34),
        ("atlas.vstore", 1, "atlas.dma", 1, 34),
    ]
    if unit == 1:
        gaps.append(("atlas.mxu_push", 0, "atlas.mxu_matmul", 0, 32))
    if relu_pair is None:
        gaps.append(("atlas.mxu_pop", 0, "atlas.vstore", 0, 32))
    else:
        gaps.extend(
            (
                ("atlas.mxu_pop", 0, "atlas.vpu_unary", 0, 32),
                ("atlas.vpu_unary", 0, "atlas.vstore", 0, 65),
            )
        )
    if any(
        issued[end][end_index] - issued[start][start_index] < minimum
        for start, start_index, end, end_index, minimum in gaps
    ):
        raise NativeCompilationError(
            "unqualified_target", "emitted MXU stream violates diagnostic issue gaps"
        )


def check_mxu0_stream(
    program: dict,
    *,
    activation_address: int,
    weight_address: int,
    output_address: int,
    activation: int,
    weight: int,
    pair: int,
    relu_pair: int | None = None,
) -> None:
    """Retain the MXU0 checker entry point used by existing OOT tests."""
    check_mxu_stream(
        program,
        unit=0,
        activation_address=activation_address,
        weight_address=weight_address,
        output_address=output_address,
        activation=activation,
        weight=weight,
        pair=pair,
        relu_pair=relu_pair,
    )


def _mxu0_k_chain_selection(
    graph: CandidateGraph,
    names: list[str | None],
    assignment: dict[int, int],
    source_ids: tuple[str, ...],
    input_slots: tuple[int, ...],
    output_slot: int,
) -> tuple[list[int], tuple[int, ...]]:
    """Replay the selected K chain before materializing physical state."""
    tile_count = len(source_ids) // 2
    if (
        tile_count < 2
        or 2 * tile_count + 1 > DRAM_SLOTS
        or len(input_slots) != 2 * tile_count
        or len(graph.outputs) != 1
    ):
        raise NativeCompilationError(
            "unqualified_target", "K chain exceeds its input or output DRAM slots"
        )
    loads: dict[str, int] = {}
    for value in graph.values:
        if names[value.id] != "atlas.load_fp8_tile":
            continue
        if len(value.children) != 1 or graph.value(value.children[0]).kind != "input":
            raise NativeCompilationError(
                "unqualified_target", "K-chain load has no declared source"
            )
        source = graph.value(value.children[0]).source_node
        if (
            source not in source_ids
            or source in loads
            or assignment[value.children[0]] != (input_slots[source_ids.index(source)])
        ):
            raise NativeCompilationError(
                "unqualified_target", "K-chain load changes its DRAM source"
            )
        loads[source] = value.id
    if set(loads) != set(source_ids):
        raise NativeCompilationError(
            "unqualified_target", "K-chain selection omits an input load"
        )
    reset = [
        value
        for value in graph.values
        if names[value.id] == "atlas.mxu0_reset_acc_tile"
    ]
    continued = [
        value
        for value in graph.values
        if names[value.id] == "atlas.mxu0_accumulate_tile"
    ]
    stores = [
        value
        for value in graph.values
        if names[value.id] == "atlas.store_bf16_mxu0_acc_tile"
    ]
    chain = reset[:]
    remaining = {value.id: value for value in continued}
    if reset:
        for index in range(1, tile_count):
            children = (
                loads[source_ids[2 * index]],
                loads[source_ids[2 * index + 1]],
                chain[-1].id,
            )
            matches = [
                value for value in remaining.values() if value.children == children
            ]
            if len(matches) != 1:
                raise NativeCompilationError(
                    "unqualified_target", "K-chain continuation is broken"
                )
            chain.append(matches[0])
            del remaining[matches[0].id]
    if (
        len(reset) != 1
        or len(continued) != tile_count - 1
        or len(stores) != 1
        or reset[0].children != (loads[source_ids[0]], loads[source_ids[1]])
        or remaining
        or stores[0].children != (chain[-1].id,)
        or graph.outputs != (stores[0].id,)
        or any(assignment[node.id] != 0 for node in chain)
        or assignment[stores[0].id] != output_slot
        or sum(value.kind == "instruction" for value in graph.values)
        != 3 * tile_count + 1
    ):
        raise NativeCompilationError(
            "unqualified_target", "K-chain selection changes accumulator chain"
        )
    registers = tuple(assignment[loads[source]] for source in source_ids)
    if any(not 0 <= register < 32 for register in registers):
        raise NativeCompilationError(
            "unqualified_target", "K-chain FP8 placement exceeds the register file"
        )
    return sorted(set(registers)), registers


def check_mxu0_k_chain_stream(
    program: dict,
    *,
    input_addresses: tuple[int, ...],
    output_address: int,
    source_registers: tuple[int, ...],
) -> None:
    """Replay a reset/continue/readout stream from emitted physical fields."""
    tile_count = len(input_addresses) // 2
    if (
        tile_count < 2
        or 2 * tile_count + 1 > DRAM_SLOTS
        or len(input_addresses) != 2 * tile_count
        or len(source_registers) != len(input_addresses)
        or len(set(input_addresses)) != len(input_addresses)
        or any(
            source_registers[2 * index] == source_registers[2 * index + 1]
            for index in range(tile_count)
        )
    ):
        raise NativeCompilationError(
            "unqualified_target", "K-chain ABI or register pair overlaps"
        )
    effects = _tile_effects(program)
    if effects[:1] != [("config",)] or effects[-2:] != [("complete",), ("trap",)]:
        raise NativeCompilationError(
            "unqualified_target", "K-chain stream has no selected entry or completion"
        )
    kinds = [effect[0] for effect in effects]
    expected_counts = {
        "config": 1,
        "dma": 2 * tile_count + 2,
        "wait": 2 * tile_count + 2,
        "vload": 2 * tile_count,
        "mxu_push": tile_count,
        "mxu_matmul": tile_count,
        "mxu_pop": 1,
        "vstore": 2,
        "complete": 1,
        "trap": 1,
    }
    if {
        kind: kinds.count(kind) for kind in set(kinds) | set(expected_counts)
    } != expected_counts:
        raise NativeCompilationError(
            "unqualified_target", "K-chain stream has missing or extra effects"
        )
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
            "atlas.dma_config",
            "atlas.dma",
            "atlas.dma_wait",
            "atlas.vload",
            "atlas.mxu_push",
            "atlas.mxu_matmul",
            "atlas.mxu_pop",
            "atlas.vstore",
            "atlas.csr",
            "atlas.trap",
        }:
            event_cycles.append(cycle)
        if row["operation"] == "atlas.delay":
            delay = row["fields"].get("cycles")
            if type(delay) is not int or not 0 <= delay <= 65535:
                raise NativeCompilationError(
                    "unqualified_target", "K-chain delay has an invalid cycle count"
                )
            cycle += delay
        cycle += 1
    if len(event_cycles) != len(effects):
        raise NativeCompilationError(
            "unqualified_target", "K-chain stream has unmodeled physical effects"
        )
    push_cycle = 0
    matmul_cycle = 0
    pop_cycle = 0
    vstore_cycle = 0
    for index, effect in enumerate(effects):
        kind = effect[0]
        now = event_cycles[index]
        if kind == "dma" and effect[1] == "load":
            if (
                effect[2] not in input_addresses
                or effect[4:] != (1024, 0)
                or index + 2 >= len(effects)
                or effects[index + 1] != ("wait", 0)
                or effects[index + 2][0] != "vload"
                or effects[index + 2][2] != effect[3]
            ):
                raise NativeCompilationError(
                    "unqualified_target", "K-chain input DMA, wait, or VLOAD differs"
                )
            address = effect[2]
            register = effects[index + 2][1]
            if register != source_registers[input_addresses.index(address)]:
                raise NativeCompilationError(
                    "unqualified_target",
                    "K-chain selected register differs from emission",
                )
            source_by_reg[register] = address
            load_cycles[address] = event_cycles[index + 2]
        elif kind == "mxu_push":
            if (
                pushes >= tile_count
                or effect[:4] != ("mxu_push", 0, 0, "weight_fp8")
                or source_by_reg.get(effect[4]) != input_addresses[2 * pushes + 1]
                or now - load_cycles.get(input_addresses[2 * pushes + 1], now) < 34
                or (pushes > 0 and matmuls != pushes)
            ):
                raise NativeCompilationError(
                    "unqualified_target", "K-chain weight push differs"
                )
            pushes += 1
            push_cycle = now
        elif kind == "mxu_matmul":
            if (
                matmuls >= tile_count
                or pushes != matmuls + 1
                or effect[:5] != ("mxu_matmul", 0, 0, 0, bool(matmuls))
                or source_by_reg.get(effect[5]) != input_addresses[2 * matmuls]
                or now - load_cycles.get(input_addresses[2 * matmuls], now) < 34
                or now <= push_cycle
                or (matmuls > 0 and now - matmul_cycle < 64)
            ):
                raise NativeCompilationError(
                    "unqualified_target",
                    "K-chain reset/continue order or availability differs",
                )
            matmuls += 1
            matmul_cycle = now
        elif kind == "mxu_pop":
            if (
                matmuls != tile_count
                or pair is not None
                or effect[:5] != ("mxu_pop", 0, 0, "bf16", 0)
                or type(effect[5]) is not int
                or effect[5] % 2
                or not 32 <= effect[5] <= 60
                or now - matmul_cycle < 64
            ):
                raise NativeCompilationError(
                    "unqualified_target", "K-chain readout differs"
                )
            pair = effect[5]
            pop_cycle = now
        elif kind == "vstore":
            if (
                pair is None
                or store_halves >= 2
                or pending_store
                or effect[1] != pair + store_halves
                or now - pop_cycle < 32
            ):
                raise NativeCompilationError(
                    "unqualified_target", "K-chain output half or availability differs"
                )
            store_halves += 1
            pending_store = True
            vstore_cycle = now
        elif kind == "dma" and effect[1] == "store":
            half = store_halves - 1
            if (
                not pending_store
                or half not in (0, 1)
                or effect[2] != output_address + 1024 * half
                or effect[4:] != (1024, 1)
                or effect[3] != effects[index - 1][2]
                or effects[index - 1][0] != "vstore"
                or index + 1 >= len(effects)
                or effects[index + 1] != ("wait", 1)
                or now - vstore_cycle < 34
            ):
                raise NativeCompilationError(
                    "unqualified_target", "K-chain output DMA or wait differs"
                )
            pending_store = False
        elif kind in {"config", "wait", "vload", "complete", "trap"}:
            continue
        else:
            raise NativeCompilationError(
                "unqualified_target", "K-chain stream has an unadmitted effect"
            )
    if (
        pushes != tile_count
        or matmuls != tile_count
        or store_halves != 2
        or pending_store
    ):
        raise NativeCompilationError(
            "unqualified_target", "K-chain stream is incomplete"
        )
