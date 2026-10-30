#!/usr/bin/env python3
"""Extract engine capacities, VPU groups, and MXU resource windows."""
from __future__ import annotations

import ast
from pathlib import Path
import sys

sys.dont_write_bytecode = True

from atlas_model_source import (assignment as _assignment, guard_literal, literal as _literal,
                                method as _method, module_tree, named as _named)


def derive_resources(model):
    """Read structural values after the engine extractors check their source digests."""
    model = Path(model)
    mxu, vpu, lsu = (module_tree(model / "npu_model" / "hardware" / filename)
                     for filename in ("mxu.py", "vpu.py", "lsu.py"))
    base = _named(mxu, "_MatrixExecutionUnit")
    accept = _method(base, "_accept")
    active = _literal(_assignment(mxu, "PORT_CYCLES").value)
    acc_push_last = guard_literal(_assignment(accept, "acc_pushes").value, None, ast.LtE)
    weight_drain_last = guard_literal(accept, "same slot drained", ast.LtE)
    acc_forward = guard_literal(accept, "cycle after a push", ast.Lt)
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
    busy_count = _literal(_assignment(vpu, "ISSUE_SLOTS").value)
    groups = [sorted(group) for group in ast.literal_eval(_assignment(vpu, "_GROUPS").value)]
    double = sorted(ast.literal_eval(_assignment(vpu, "_TWO_INPUT").value) |
                    ast.literal_eval(_assignment(vpu, "_ROW_REDUCE").value))
    scalar_gap = guard_literal(_method(_named(lsu, "LoadStoreUnit"), "_accept"),
                               "prior load response pending", ast.Lt)
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
