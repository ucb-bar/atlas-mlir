"""Derive isolated MXU row windows and logical MREG releases from model source.

No execution unit, architectural state, instruction execution, or tensor is created.
The reviewed AST shape is guarded after replacing the expressions derived below;
its digest is a semantics whitelist, not a timing table or a source identity pin.
"""
from __future__ import annotations

from atlas_model_diagnostics import UnsupportedModelTiming

import ast
from copy import deepcopy
import inspect
from pathlib import Path

from atlas_model_timing import MXU_BLOCKS
from atlas_model_schema import SOURCE_SHAPES, ast_digest, reservation, timing_block, validate_mxu_weight_timing


_CLASSES = ("_Operation", "_MatrixExecutionUnit", "MatrixExecutionUnitSystolic", "MatrixExecutionUnitInner")
_UNITS = {"mxu0": "MatrixExecutionUnitSystolic", "mxu1": "MatrixExecutionUnitInner"}


class UnsupportedMXU(UnsupportedModelTiming):
    """Selected source or metadata has no supported MXU timing interpretation."""


def _read(path):
    path = Path(path)
    if any(any(word in part.lower() for word in ("hammer", "vlsi")) for part in path.parts):
        raise UnsupportedMXU("unsupported model source path")
    if ".." in path.parts:
        raise ValueError("parent traversal is unsupported")
    if any(parent.is_symlink() for parent in reversed((path, *path.parents))):
        raise UnsupportedMXU("model source symlinks are unsupported")
    return path.read_text()


def _sources(cfg):
    root = Path(inspect.getfile(cfg.isa)).parent
    return (_read(root / "hardware" / "mxu.py"), _read(root / "hardware" / "bank_conflict.py"))


def _one(values, label):
    values = list(values)
    if len(values) != 1:
        raise UnsupportedMXU(f"unsupported MXU source: expected one {label}")
    return values[0]


def _name(node, name):
    return isinstance(node, ast.Name) and node.id == name


def _assignment(tree, name):
    return _one((n for n in ast.walk(tree) if isinstance(n, ast.Assign)
                 and len(n.targets) == 1 and _name(n.targets[0], name)), name)


def _method(tree, name):
    return _one((n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name), name)


def _shape(nodes):
    class StripDocstrings(ast.NodeTransformer):
        def visit_Expr(self, node):
            if isinstance(node.value, ast.Constant) and isinstance(node.value.value, str):
                return None
            return self.generic_visit(node)
    tree = StripDocstrings().visit(ast.Module(body=deepcopy(nodes), type_ignores=[]))
    return ast_digest(tree)


def _integer(expr, compute_first):
    if isinstance(expr, ast.Constant) and type(expr.value) is int:
        return expr.value
    if (isinstance(expr, ast.Attribute) and _name(expr.value, "self") and expr.attr == "compute_first"):
        return compute_first
    if isinstance(expr, ast.BinOp) and isinstance(expr.op, (ast.Add, ast.Sub, ast.Mult)):
        left, right = _integer(expr.left, compute_first), _integer(expr.right, compute_first)
        return left + right if isinstance(expr.op, ast.Add) else left - right if isinstance(expr.op, ast.Sub) else left * right
    raise UnsupportedMXU("unsupported MXU timing expression: " + ast.dump(expr, include_attributes=False))


def _window(comparison, compute_first):
    if (not isinstance(comparison, ast.Compare) or len(comparison.ops) != 2
            or not _name(comparison.comparators[0], "age")
            or not all(isinstance(op, (ast.Lt, ast.LtE)) for op in comparison.ops)):
        raise UnsupportedMXU("unsupported MXU row window")
    first = _integer(comparison.left, compute_first) + isinstance(comparison.ops[0], ast.Lt)
    last = _integer(comparison.comparators[1], compute_first) - isinstance(comparison.ops[1], ast.Lt)
    if not 0 <= first <= last < 4095:
        raise UnsupportedMXU("unsupported MXU row window bounds")
    return dict(count=last - first + 1, first_age=first, last_age=last, step=1)


def _placeholder(name):
    return ast.Name(id="DERIVED_" + name, ctx=ast.Load())


def _plan(source):
    tree = ast.parse(source)
    classes = {n.name: n for n in tree.body if isinstance(n, ast.ClassDef)}
    if set(classes) != set(_CLASSES):
        raise UnsupportedMXU("unsupported MXU source classes")
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
        raise UnsupportedMXU("unsupported MXU completion expression")
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
            raise UnsupportedMXU("unsupported MXU compute latency")
        value.value = _placeholder(engine + "_compute_first")
    _assignment(tree, "MXU_OP_LATENCIES").value = _placeholder("execute_delay_table")
    if _shape(tree.body) != SOURCE_SHAPES["mxu.py"]:
        raise UnsupportedMXU("unsupported MXU control-flow semantics")
    plans = {}
    for engine, first in firsts.items():
        values = {key: _window(slots[key], first) for key in ("read", "compute", "transfer")}
        values.update(release=_integer(slots["release"], first),
                      compute_last=_integer(slots["compute_last"], first),
                      transfer_last=_integer(slots["transfer_last"], first))
        # The source indexes SRAM rows with age, age-1, and age-compute_first.
        for key, offset in (("read", 0), ("transfer", 1), ("compute", first)):
            if (values[key]["first_age"] - offset, values[key]["last_age"] - offset) != (0, 31):
                raise UnsupportedMXU("unsupported MXU row geometry")
        if values["release"] < max(values[key]["last_age"] for key in ("read", "transfer")):
            raise UnsupportedMXU("MXU reservation releases before its last MREG access")
        for kind, windows in (("compute", ("read", "compute")), ("transfer", ("read", "transfer"))):
            end = values[kind + "_last"]
            if end < max(values[key]["last_age"] for key in windows) or not 0 <= values["release"] <= end:
                raise UnsupportedMXU("unsupported MXU completion/release bounds")
        plans[engine] = values
    return plans


def _check_checker(source):
    tree = ast.parse(source)
    _one((n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "BankConflictChecker"), "bank checker")
    if ast_digest(tree) != SOURCE_SHAPES["bank_conflict.py"]:
        raise UnsupportedMXU("unsupported MXU reservation release semantics")


def _validated(cfg):
    geometry = cfg.arch_state_config
    if (geometry.mrf_depth, geometry.mrf_width, geometry.wb_width, geometry.num_wb_registers) != (32, 32, 1024, 2):
        raise UnsupportedMXU("unsupported MXU configuration geometry")
    if geometry.num_m_registers != 64:
        raise UnsupportedMXU("MXU source requires 64 matrix registers")
    operations = cfg.isa.operations
    for engine, unit in _UNITS.items():
        if unit not in cfg.execution_units.values():
            raise UnsupportedMXU(f"MXU configuration does not provide {unit}")
        expected = {name + "." + engine for name in MXU_BLOCKS}
        actual = {name for name, insn in operations.items() if str(insn.exu) == unit}
        if actual != expected:
            raise UnsupportedMXU("unsupported MXU ISA operation coverage")
        for mnemonic in MXU_BLOCKS:
            name = mnemonic + "." + engine
            insn = operations.get(name)
            if insn is None or getattr(insn, "mnemonic", None) != name or str(insn.exu) != unit:
                raise UnsupportedMXU(f"unsupported MXU ISA metadata for {name}")
            expected_params = ({"vd", "vs1", "vs2"} if mnemonic.startswith("vmatmul") else
                               {"vd", "vs2", "es1"} if mnemonic == "vmatpop.fp8.acc" else
                               {"vd", "vs2"} if mnemonic.startswith("vmatpop") else {"vd", "vs1"})
            if set(inspect.signature(insn).parameters) != expected_params:
                raise UnsupportedMXU(f"unsupported MXU ISA operands for {name}")
    source, checker = _sources(cfg)
    _check_checker(checker)
    return _plan(source)


def _busy(last):
    if not 0 <= last < 4095:
        raise UnsupportedMXU("unsupported MXU completion age")
    return dict(count=last + 1, first_age=0, last_age=last, step=1)


def _blocks(plans):
    blocks = {}
    for engine, values in plans.items():
        for mnemonic, suffix in MXU_BLOCKS.items():
            compute, pop = mnemonic.startswith("vmatmul"), mnemonic.startswith("vmatpop")
            bf16 = ".bf16" in mnemonic
            last = values["compute_last" if compute else "transfer_last"]
            events = {"compute_busy" if compute else "data_busy": _busy(last)}
            if pop:
                events["acc_store"] = values["read"]
                for bank in range(2 if bf16 else 1):
                    events[f"mreg_write{bank}"] = values["transfer"]
            else:
                if compute:
                    groups = ("mreg_read0",)
                elif bf16:
                    groups = ("mreg_read0", "mreg_read1")
                else:
                    groups = ("mreg_read1",)
                events.update({group: values["read"] for group in groups})
                group = "acc_write" if compute else "weight_write" if mnemonic.endswith("weight") else "acc_load"
                events[group] = values["compute" if compute else "transfer"]
                if mnemonic == "vmatmul.acc":
                    events["acc_read"] = values["read"]
            name = engine + "." + suffix
            blocks[name] = timing_block(name, engine, deepcopy(events), last)
    return blocks


def derive_mxu(cfg):
    """Derive blocks and deferred MREG releases; ACC/weights have no checker reservation."""
    plans = _validated(cfg)
    blocks = _blocks(plans)
    releases = {}
    for engine, values in plans.items():
        for mnemonic, suffix in MXU_BLOCKS.items():
            name = engine + "." + suffix
            pop = mnemonic.startswith("vmatpop")
            first = 4 if pop else 0
            banks = list(range(first, first + (2 if ".bf16" in mnemonic else 1)))
            end = blocks[name]["first_free_age"]
            releases[name] = {
                "mreg_write" if pop else "mreg_read": reservation(values["release"], banks=banks),
                "unit": reservation(end - 1),
            }
    return blocks, releases


def derive_weight_timing(cfg):
    """Project native PE use windows onto whole output-column weight rows.

    The full reviewed source shape binds the sampler's (output column, inner)
    indexing, weight-slot selection and its placement before writes in tick.
    Read bounds below are extracted from that source, never from execution or
    the instruction latency table. Row projection includes every inner lane.
    """
    plans = _validated(cfg)
    source, _ = _sources(cfg)
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
    expression = ast.parse("age - 1 - _GRID[:, None] - _GRID[None, :]").body[0].value
    if ast.dump(row.value, include_attributes=False) != ast.dump(expression, include_attributes=False):
        raise UnsupportedMXU("unsupported MXU weight PE indexing")
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
            raise UnsupportedMXU("MXU completes before its last weight use")
        engines[engine] = dict(use_window=window, use_rows=rows,
                               reads_before_writes=True, resource_row_axis="output_column")
    return validate_mxu_weight_timing(dict(schema="atlas.mxu_weight_timing.v1", engines=engines))
