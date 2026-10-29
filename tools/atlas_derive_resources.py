#!/usr/bin/env python3
"""Extract engine capacities, VPU groups, and MXU resource windows."""
from __future__ import annotations

import ast
from copy import deepcopy
from pathlib import Path
import sys

sys.dont_write_bytecode = True

from atlas_derive_mxu import _assignment, _method, _one
from atlas_model_schema import VOCABULARY


def _literal(node):
    value = ast.literal_eval(node)
    if type(value) is not int or not 0 <= value < 4096:
        raise ValueError("unsupported source integer")
    return value


def _named(tree, name):
    return _one((node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == name), name)


def _age_bound(node, comparison):
    """Read an age expression's literal right endpoint."""
    return _one((_literal(n.comparators[0]) for n in ast.walk(node)
                 if isinstance(n, ast.Compare) and len(n.ops) == 1
                 and isinstance(n.ops[0], comparison) and isinstance(n.left, ast.BinOp)
                 and isinstance(n.left.op, ast.Sub) and isinstance(n.comparators[0], ast.Constant)),
                "source age endpoint")


def _guard(method, diagnostic):
    matches = []
    for node in ast.walk(method):
        if isinstance(node, ast.If) and any(isinstance(stmt, ast.Raise) and any(
                isinstance(part, ast.Constant) and isinstance(part.value, str) and diagnostic in part.value
                for part in ast.walk(stmt)) for stmt in node.body):
            matches.append(node.test)
    return _one(matches, diagnostic)


def _source(root, name):
    path = (root / "npu_model" / "hardware" / name)
    source = path.read_text()
    tree = ast.parse(source)
    return source, tree


def derive_resources(model):
    """Read structural values after the engine extractors check their source digests."""
    model = Path(model)
    _, mxu = _source(model, "mxu.py")
    _, vpu = _source(model, "vpu.py")
    _, lsu = _source(model, "lsu.py")
    base = _named(mxu, "_MatrixExecutionUnit")
    accept = _method(base, "_accept")
    active = _literal(_assignment(mxu, "PORT_CYCLES").value)
    acc_push_last = _age_bound(_assignment(accept, "acc_pushes").value, ast.LtE)
    weight_drain_last = _age_bound(_guard(accept, "same slot drained"), ast.LtE)
    acc_forward = _age_bound(_guard(accept, "cycle after a push"), ast.Lt)
    # The MXU digest pins port selection and its shared single-push priority.
    choices = ast.literal_eval(_assignment(mxu, "PORT_CHOICES").value)
    fixed = ("compute", "push_bf16", "pop_bf16", "pop_fp8")
    if (set(choices) != {*fixed, "weight", "push_fp8"}
            or any(not isinstance(choices[kind], tuple) or len(choices[kind]) != 1 for kind in fixed)
            or choices["weight"] != choices["push_fp8"]
            or any(not isinstance(choice, tuple) or len(choice) != 1 for choice in choices["weight"])):
        raise ValueError("unsupported MXU sequencer port selection")
    port_roles = {kind: list(choices[kind][0]) for kind in fixed}
    priority = [choice[0] for choice in choices["weight"]]
    if any(type(port) is not int or not 0 <= port < 4
           for values in [*port_roles.values(), priority] for port in values):
        raise ValueError("unsupported MXU sequencer port identifiers")
    depths = {unit: _literal(_assignment(_named(mxu, cls), "inflight_depth").value)
              for unit, cls in (("mxu0", "MatrixExecutionUnitSystolic"), ("mxu1", "MatrixExecutionUnitInner"))}
    from atlas_derive_vpu import _normalize_slot_capacity
    slots = _normalize_slot_capacity(deepcopy(vpu))
    busy_count = len(slots)
    mnemonic_to_block = {entry["mnemonic"]: entry["block"] for entry in VOCABULARY["vpu_blocks"]}
    groups = [sorted(mnemonic_to_block[name] for name in group)
              for group in ast.literal_eval(_assignment(vpu, "_GROUPS").value)]
    double = sorted(mnemonic_to_block[name] for name in (
        ast.literal_eval(_assignment(vpu, "_TWO_INPUT").value) |
        ast.literal_eval(_assignment(vpu, "_ROW_REDUCE").value)))
    scalar_gap = _age_bound(_guard(_method(_named(lsu, "LoadStoreUnit"), "_accept"),
                                  "prior load response pending"), ast.Lt)
    contract = dict(
        capacity=dict(sequencer_cycles=active, mxu0_compute=depths["mxu0"],
                      mxu1_compute=depths["mxu1"], vpu_slots=busy_count),
        vpu=dict(logic_groups=groups, double=double),
        mxu=dict(single_push_priority=priority, ports=port_roles,
                 acc_push_last=acc_push_last, mxu0_weight_drain_last=weight_drain_last,
                 accumulator_forward_gap=acc_forward),
        scalar_lsu=dict(load_response_gap=scalar_gap))
    # Row indexing and physical bank geometry remain explicitly supported only
    # at 32. Tracker/slot literals above are source parameters, not config knobs.
    if active != 32:
        raise ValueError("unsupported MXU sequencer geometry (requires 32 cycles)")
    if any(not 1 <= depth <= 64 for depth in depths.values()) or not 1 <= busy_count <= 3:
        raise ValueError("unsupported source tracker/slot capacity")
    return contract
