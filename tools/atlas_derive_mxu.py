"""Derive isolated MXU row windows and logical MREG releases from model source."""
from __future__ import annotations

import ast
from copy import deepcopy
import inspect
from pathlib import Path

from atlas_model_source import (MXU_BLOCKS, SOURCE_SHAPES, UnsupportedModelTiming, age_window,
                                assignment as _assignment, literal, method as _method,
                                name as _name, one as _one, positive_int, read_source, reservation, shape,
                                timing_block)


_CLASSES = ("_Operation", "_MatrixExecutionUnit", "MatrixExecutionUnitSystolic", "MatrixExecutionUnitInner")
_UNITS = {"mxu0": "MatrixExecutionUnitSystolic", "mxu1": "MatrixExecutionUnitInner"}


def _source(cfg):
    root = Path(inspect.getfile(cfg.isa)).parent
    return read_source(root / "hardware" / "mxu.py")


def _shape(nodes):
    return shape(ast.Module(body=nodes, type_ignores=[]))


def _integer(expr, compute_first):
    return literal(expr, "MXU timing expression", env={"self.compute_first": compute_first},
                   minimum=None, maximum=None)


def _window(comparison, compute_first):
    return age_window(comparison, {"self.compute_first": compute_first}, maximum=4095)


def _placeholder(name):
    return ast.Name(id="DERIVED_" + name, ctx=ast.Load())


def _normalize_source(source):
    tree = ast.parse(source)
    classes = {n.name: n for n in tree.body if isinstance(n, ast.ClassDef)}
    if set(classes) != set(_CLASSES):
        raise UnsupportedModelTiming("unsupported MXU source classes")
    base = classes["_MatrixExecutionUnit"]
    tick = _method(base, "tick")
    comparisons = [n for n in ast.walk(tick) if isinstance(n, ast.Compare) and len(n.ops) == 2
                   and _name(n.comparators[0], "age")]
    read = _one((n for n in comparisons if isinstance(n.ops[1], ast.Lt)), "read row window")
    compute = _one((n for n in comparisons if isinstance(n.left, ast.Attribute)), "compute row window")
    transfer = _one((n for n in comparisons if n is not read and n is not compute), "transfer row window")
    release = _one((n for n in ast.walk(tick) if isinstance(n, ast.Compare) and _name(n.left, "age")
                    and len(n.ops) == 1 and isinstance(n.ops[0], ast.Eq)
                    and not _name(n.comparators[0], "last")), "MREG release")
    last = _assignment(tick, "last")
    if not isinstance(last.value, ast.IfExp):
        raise UnsupportedModelTiming("unsupported MXU completion expression")
    slots = {"read": deepcopy(read), "compute": deepcopy(compute), "transfer": deepcopy(transfer),
             "release": deepcopy(release.comparators[0]), "compute_last": deepcopy(last.value.body),
             "transfer_last": deepcopy(last.value.orelse)}
    for label, node in (("read", read), ("compute", compute), ("transfer", transfer)):
        node.left, node.comparators[1] = _placeholder(label + "_first"), _placeholder(label + "_last")
    release.comparators[0] = _placeholder("release")
    last.value.body, last.value.orelse = _placeholder("compute_last"), _placeholder("transfer_last")
    firsts = {}
    for engine, name in _UNITS.items():
        value = _assignment(classes[name], "compute_first")
        firsts[engine] = _integer(value.value, None)
        if not 0 <= firsts[engine] < 4095:
            raise UnsupportedModelTiming("unsupported MXU compute latency")
        value.value = _placeholder(engine + "_compute_first")
        # Tracker depth changes admission, not isolated row timing. Normalize
        # only this reviewed class constant; the complete admission predicate
        # and all other control flow remain covered by the AST guard.
        depth = _assignment(classes[name], "inflight_depth")
        if (not isinstance(depth.value, ast.Constant) or type(depth.value.value) is not int
                or not 1 <= depth.value.value <= 64):
            raise UnsupportedModelTiming("unsupported MXU compute tracker depth (requires 1..64)")
        depth.value = ast.Constant(value=3 if engine == "mxu0" else 2)
    _assignment(tree, "MXU_OP_LATENCIES").value = _placeholder("execute_delay_table")
    return tree, slots, firsts


def source_shape(source):
    """Return the MXU source shape using the timing planner's normalization."""
    tree, _, _ = _normalize_source(source)
    return _shape(tree.body)


def _plan(source):
    tree, slots, firsts = _normalize_source(source)
    if _shape(tree.body) != SOURCE_SHAPES["mxu.py"]:
        raise UnsupportedModelTiming("unsupported MXU control-flow semantics")
    plans = {}
    for engine, first in firsts.items():
        values = {key: _window(slots[key], first) for key in ("read", "compute", "transfer")}
        values.update(release=_integer(slots["release"], first),
                      compute_last=_integer(slots["compute_last"], first),
                      transfer_last=_integer(slots["transfer_last"], first))
        # The source indexes SRAM rows with age, age-1, and age-compute_first.
        for key, offset in (("read", 0), ("transfer", 1), ("compute", first)):
            if (values[key]["first_age"] - offset, values[key]["last_age"] - offset) != (0, 31):
                raise UnsupportedModelTiming("unsupported MXU row geometry")
        if values["release"] < max(values[key]["last_age"] for key in ("read", "transfer")):
            raise UnsupportedModelTiming("MXU reservation releases before its last MREG access")
        for kind, windows in (("compute", ("read", "compute")), ("transfer", ("read", "transfer"))):
            end = values[kind + "_last"]
            if end < max(values[key]["last_age"] for key in windows) or not 0 <= values["release"] <= end:
                raise UnsupportedModelTiming("unsupported MXU completion/release bounds")
        plans[engine] = values
    return plans


def _validated(cfg):
    geometry = cfg.arch_state_config
    for field in ("wb_width", "num_wb_registers"):
        positive_int(getattr(geometry, field), "architecture configuration " + field)
    if (geometry.mrf_depth, geometry.mrf_width, geometry.wb_width, geometry.num_wb_registers) != (32, 32, 1024, 2):
        raise UnsupportedModelTiming("unsupported MXU configuration geometry")
    if geometry.num_m_registers != 64:
        raise UnsupportedModelTiming("MXU source requires 64 matrix registers")
    operations = cfg.isa.operations
    for engine, unit in _UNITS.items():
        if unit not in cfg.execution_units.values():
            raise UnsupportedModelTiming(f"MXU configuration does not provide {unit}")
        expected = {name + "." + engine for name in MXU_BLOCKS}
        actual = {name for name, insn in operations.items() if str(insn.exu) == unit}
        if actual != expected:
            raise UnsupportedModelTiming("unsupported MXU ISA operation coverage")
        for mnemonic in MXU_BLOCKS:
            name = mnemonic + "." + engine
            insn = operations.get(name)
            if insn is None or getattr(insn, "mnemonic", None) != name or str(insn.exu) != unit:
                raise UnsupportedModelTiming(f"unsupported MXU ISA metadata for {name}")
            expected_params = ({"vd", "vs1", "vs2"} if mnemonic.startswith("vmatmul") else
                               {"vd", "vs2", "es1"} if mnemonic == "vmatpop.fp8.acc" else
                               {"vd", "vs2"} if mnemonic.startswith("vmatpop") else {"vd", "vs1"})
            if set(inspect.signature(insn).parameters) != expected_params:
                raise UnsupportedModelTiming(f"unsupported MXU ISA operands for {name}")
    # The VPU extractor checks the shared bank checker digest once.
    source = _source(cfg)
    return _plan(source)


def _blocks(plans):
    blocks = {}
    for engine, values in plans.items():
        for mnemonic in MXU_BLOCKS:
            compute, pop = mnemonic.startswith("vmatmul"), mnemonic.startswith("vmatpop")
            bf16 = ".bf16" in mnemonic
            last = values["compute_last" if compute else "transfer_last"]
            if not 0 <= last < 4095:
                raise UnsupportedModelTiming("unsupported MXU completion age")
            events = {}
            if pop:
                events["acc_store"] = values["read"]
                for bank in range(2 if bf16 else 1):
                    events[f"mreg_write{bank}"] = values["transfer"]
            else:
                groups = ("mreg_read0",) if compute else ("mreg_read0", "mreg_read1") if bf16 else ("mreg_read1",)
                events.update({group: values["read"] for group in groups})
                group = "acc_write" if compute else "weight_write" if mnemonic.endswith("weight") else "acc_load"
                events[group] = values["compute" if compute else "transfer"]
                if mnemonic == "vmatmul.acc":
                    events["acc_read"] = values["read"]
            name = mnemonic + "." + engine
            blocks[name] = timing_block(name, deepcopy(events), last)
    return blocks


def derive_mxu(cfg):
    """Derive blocks and deferred MREG releases; ACC/weights have no checker reservation."""
    plans = _validated(cfg)
    blocks = _blocks(plans)
    releases = {}
    for engine, values in plans.items():
        for mnemonic in MXU_BLOCKS:
            name = mnemonic + "." + engine
            pop = mnemonic.startswith("vmatpop")
            end = blocks[name]["first_free_age"]
            releases[name] = {
                "mreg_write" if pop else "mreg_read": reservation(values["release"]),
                "unit": reservation(end - 1),
            }
    return blocks, releases


def derive_weight_timing(cfg):
    """Project native PE use windows onto whole output-column weight rows."""
    plans = _validated(cfg)
    source = _source(cfg)
    tree = ast.parse(source)
    base = _one((n for n in tree.body if isinstance(n, ast.ClassDef)
                 and n.name == "_MatrixExecutionUnit"), "MXU base class")
    uses = _method(base, "_weight_uses")
    inner = _one((n for n in ast.walk(uses) if isinstance(n, ast.Compare)
                  and len(n.ops) == 2 and _name(n.comparators[0], "age")), "inner weight window")
    inner_window = _window(inner, None)
    row = _assignment(uses, "row")
    # Source uses age - offset - output_column - inner. Preserve the two
    # tensor axes explicitly: a push writes output-column rows, not diagonals.
    bounds = [n for n in ast.walk(uses) if isinstance(n, ast.Compare) and _name(n.left, "row")
              and len(n.ops) == 1 and isinstance(n.ops[0], (ast.GtE, ast.LtE))]
    lower = _one((n for n in bounds if isinstance(n.ops[0], ast.GtE)), "weight row lower bound")
    upper = _one((n for n in bounds if isinstance(n.ops[0], ast.LtE)), "weight row upper bound")
    offset = _integer(row.value.left.left.right, None)
    first = offset + _integer(lower.comparators[0], None)
    last = offset + _integer(upper.comparators[0], None)
    windows = {"mxu0": dict(first_age=first, last_age=last, row_step=1, column_step=1),
               "mxu1": dict(first_age=inner_window["first_age"], last_age=inner_window["last_age"],
                            row_step=0, column_step=0)}
    engines = {}
    for engine, window in windows.items():
        rows = [dict(first_age=window["first_age"] + i * window["row_step"],
                     last_age=window["last_age"] + i * window["row_step"] +
                              31 * window["column_step"]) for i in range(32)]
        if max(row["last_age"] for row in rows) > plans[engine]["compute_last"]:
            raise UnsupportedModelTiming("MXU completes before its last weight use")
        engines[engine] = dict(use_rows=rows)
    return dict(engines=engines)
