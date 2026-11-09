"""Materialize one candidate through actual Atlas lowering and binary emission.

Compiler-side admission/replay retains its diagnostic scope. Independent reference
and evaluator authority remain separate; no hardware execution is established.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path

from merlin.semantic_compiler.allocate import (
    AllocationResult,
)
from merlin.semantic_compiler.search import (
    SearchResult,
)
from merlin.semantic_compiler.target_binding import NativeCompilationError

from .bf16_stream import (
    _check_bf16_selection,
    _realized_bf16_assignment,
    check_bf16_tile_stream,
)
from .dag_selection import _mxu_binary_selection, _mxu_dag_selection
from .dag_stream import check_mxu_bf16_binary_stream, check_mxu_bf16_dag_stream
from .mxu_stream import (
    _mxu0_k_chain_selection,
    check_mxu0_k_chain_stream,
    check_mxu_stream,
)
from .pack_stream import (
    check_copy_transpose_stream,
    check_pack_stream,
    check_transpose_stream,
)
from .planning import TilePlan
from .policy import (
    DRAM_BASE,
    MXU1_FINITE_POLICY,
    NATIVE_CHECKS,
    NATIVE_PIPELINES,
    PACK_POLICY,
    PACK_RULE_BY_POLICY,
    RTL_REVISION,
    STORE_RULE_BY_POLICY,
    TILE_BYTES,
)
from .transfer import _tile_effects


def _native_pipeline() -> str:
    pipeline = os.environ.get("ATLAS_NATIVE_PIPELINE", "mandatory")
    if pipeline not in NATIVE_PIPELINES:
        raise NativeCompilationError(
            "unsupported_semantics", "ATLAS_NATIVE_PIPELINE must be mandatory or optimized"
        )
    return pipeline


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


_MaterializedCandidate.__module__ = __package__ + ".binding"


def prepare_candidate(selected: SearchResult, plan: TilePlan, *, target_source: Path, hooks,
                      tool_directory: Path | None = None) -> _MaterializedCandidate:
    """Check one native candidate through physical emission without publishing it."""
    request = plan.request
    bf16_tile = plan.bf16_tile
    copy_transpose = plan.copy_transpose
    input_slot = plan.input_slot
    input_slots = plan.input_slots
    mxu = plan.mxu
    mxu_binary = plan.mxu_binary
    mxu_dag = plan.mxu_dag
    mxu_inputs = plan.mxu_inputs
    mxu_k_chain = plan.mxu_k_chain
    mxu_relu = plan.mxu_relu
    mxu_unit = plan.mxu_unit
    output_slot = plan.output_slot
    output_slots = plan.output_slots
    pack = plan.pack
    pack_scale_code = plan.pack_scale_code
    pack_scale_codes = plan.pack_scale_codes
    pipeline = plan.pipeline
    relu_pack = plan.relu_pack
    scaled_code = plan.scaled_code
    source_ids = plan.source_ids
    transpose = plan.transpose
    transpose_inputs = plan.transpose_inputs
    _run = hooks._run
    _virtual_from_selected = hooks._virtual_from_selected
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
            or names[1] not in {"atlas.load_bf16_tile", "atlas.load_bf16_tile_mixed"}
            or names[2:] != ["atlas.vpu_relu_bf16_tile",
                              "atlas.vpu_pack_unit_fp8_tile", "atlas.store_fp8_unit_tile"]
            or values[1].storage != "tensor_bf16" or values[2].storage != "tensor_bf16"
            or values[3].storage != "tensor_fp8"
            or tuple(value.children for value in values[1:]) != ((0,), (1,), (2,), (3,))
            or (assignment[0], assignment[4]) != (input_slot, output_slot)
            or any(assignment[index] % 2 or not 0 <= assignment[index] <= 60
                   for index in (1, 2))
            or not 32 <= assignment[2] <= 60
            or assignment[1] == assignment[2] or assignment[3] != 0
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

    if tool_directory is not None and (not isinstance(tool_directory, Path) or not tool_directory.is_absolute()):
        raise NativeCompilationError("tool_unavailable", "selected tool directory must be an explicit absolute path")
    tool_dir = (Path(os.environ.get("ATLAS_OOT_BIN_DIR", target_source / "build/bin"))
                if tool_directory is None else tool_directory)
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
        effects = _tile_effects(program)
        loads = [effect for effect in effects if effect[0] == "vload"]
        relus = [effect for effect in effects if effect[0] == "relu"]
        if len(loads) != 2 or len(relus) != int(relu_pack):
            raise NativeCompilationError("unqualified_target", "pack input or ReLU count changed")
        source_pair = relus[0][2] if relu_pack else loads[0][1]
        actual_relu = (relus[0][1], relus[0][2]) if relu_pack else None
        check_pack_stream(program, input_address=input_address,
                          output_address=output_address, source_pair=source_pair,
                          relu_pair=actual_relu, scale_code=pack_scale_code)
        input_pair = loads[0][1]
        selected_registers = [input_pair, input_pair + 1]
        if relu_pack:
            selected_registers.extend((source_pair, source_pair + 1))
        selected_registers.append(0)
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
