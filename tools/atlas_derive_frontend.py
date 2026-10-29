"""Extract the reviewed Core/S1/scalar frontend contract without execution."""
from __future__ import annotations

import ast
import inspect
from pathlib import Path

from atlas_derive_source import method, read_source
from atlas_model_diagnostics import UnsupportedModelTiming


_METHODS = (("core.py", "Core", "tick"), ("core.py", "Core", "_tick_s1"),
            ("exu.py", "ScalarExecutionUnit", "tick"), ("ifu.py", "InstructionFetch", "tick"))


def _one(values, label):
    values = list(values)
    if len(values) != 1:
        raise UnsupportedModelTiming(f"unsupported frontend: expected one {label}")
    return values[0]


def _attr(node, base, attr):
    return (isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name)
            and node.value.id == base and node.attr == attr)


def _positive(node, label, allow_zero=False):
    if (not isinstance(node, ast.Constant) or type(node.value) is not int
            or not int(not allow_zero) <= node.value < 65536):
        raise UnsupportedModelTiming(f"unsupported frontend {label}")
    return node.value


def _operand(node):
    if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name) and node.value.id == "self":
        return node.attr
    raise UnsupportedModelTiming("unsupported scalar register operand expression")


def _accesses(node):
    reads, writes = set(), set()
    for item in ast.walk(node):
        if isinstance(item, ast.Subscript) and _attr(item.value, "state", "xrf"):
            reads.add(("xreg", _operand(item.slice)))
        if isinstance(item, ast.Subscript) and _attr(item.value, "state", "erf"):
            reads.add(("ereg", _operand(item.slice)))
        if not isinstance(item, ast.Call) or not isinstance(item.func, ast.Attribute):
            continue
        for resource, suffix in (("xreg", "xrf"), ("ereg", "erf")):
            if _attr(item.func, "state", "read_" + suffix):
                reads.add((resource, _operand(item.args[0])))
            if _attr(item.func, "state", "write_" + suffix):
                writes.add((resource, _operand(item.args[0])))
    return {key: [dict(resource=resource, operand=operand) for resource, operand in sorted(values)]
            for key, values in (("reads", reads), ("writes", writes))}


def derive_frontend_sources(sources, scalar_classes, *, fetch_width):
    """Pure AST interface, also used by mutation tests; no model tick/exec calls."""
    if fetch_width != 1 or type(fetch_width) is not int:
        raise UnsupportedModelTiming("unsupported frontend fetch width: source has one S1 issue slot")
    slots = {}
    methods = {}
    for filename, cls, name in _METHODS:
        node = method(sources[filename], cls, name)
        methods[(filename, cls, name)] = node
        if (cls, name) == ("Core", "_tick_s1"):
            mask = _one((item for item in ast.walk(node) if isinstance(item, ast.Assign)
                         and len(item.targets) == 1 and _attr(item.targets[0], "self", "delay_counter")
                         and isinstance(item.value, ast.BinOp) and isinstance(item.value.op, ast.BitAnd)),
                        "DELAY mask assignment")
            slots["delay_mask"] = _positive(mask.value.right, "DELAY mask", allow_zero=True)
            decrement = _one((item for item in ast.walk(node) if isinstance(item, ast.AugAssign)
                              and _attr(item.target, "self", "delay_counter")
                              and isinstance(item.op, ast.Sub)), "DELAY countdown decrement")
            slots["decrement"] = _positive(decrement.value, "DELAY decrement")
            if slots["decrement"] != 1:
                raise UnsupportedModelTiming("unsupported DELAY decrement: only unit countdown reaches zero for all immediates")
        if (cls, name) == ("ScalarExecutionUnit", "tick"):
            delay = _one((item for item in ast.walk(node) if isinstance(item, ast.Assign)
                          and len(item.targets) == 1 and _attr(item.targets[0], "uop", "execute_delay")),
                         "scalar completion delay")
            slots["completion"] = _positive(delay.value, "scalar completion delay")
            if slots["completion"] != 1:
                raise UnsupportedModelTiming("unsupported scalar execute_delay tag: deferred completion is one tick, "
                                          "but the source tag no longer agrees")
    tick = methods[("core.py", "Core", "tick")]
    increment = _one((item for item in ast.walk(tick) if isinstance(item, ast.AugAssign)
                      and _attr(item.target, "self", "cycle_count") and isinstance(item.op, ast.Add)),
                     "Core clock increment")
    gap = _positive(increment.value, "Core clock increment")
    ifu = methods[("ifu.py", "InstructionFetch", "tick")]
    delay_slots = len([item for item in ast.walk(ifu) if isinstance(item, ast.Call)
                       and isinstance(item.func, ast.Name) and item.func.id == "Uop"])
    classes = {item.name: item for item in ast.parse(sources["isa_definition.py"]).body
               if isinstance(item, ast.ClassDef)}
    instructions = {}
    for mnemonic, class_name in sorted(scalar_classes.items()):
        if class_name not in classes:
            raise UnsupportedModelTiming(f"missing scalar ISA source {class_name}")
        cls = classes[class_name]
        execute = _one((item for item in cls.body if isinstance(item, ast.FunctionDef) and item.name == "exec"),
                       f"{class_name}.exec")
        instructions[mnemonic] = _accesses(execute)
    # The central Core/scalar method guards pin reads and writes to the issue edge.
    return dict(ordinary_issue_gap=gap,
                delay=dict(base_gap=gap, immediate_mask=slots["delay_mask"], decrement_per_tick=slots["decrement"]),
                scalar=dict(read_age=0, write_age=0),
                control=dict(delay_slots=delay_slots, operand_read_age=0),
                instructions=instructions)


def derive_frontend(cfg):
    root = Path(inspect.getfile(cfg.isa)).parent
    from npu_model.isa import EXU
    if "ScalarExecutionUnit" not in cfg.execution_units.values():
        raise UnsupportedModelTiming("configuration does not provide the source scalar execution unit")
    for name, cls in cfg.isa.operations.items():
        if cls.exu == EXU.SCALAR and (cls.mnemonic != name or
                Path(inspect.getfile(cls)) != root / "configs" / "isa_definition.py"):
            raise UnsupportedModelTiming("unsupported scalar ISA dispatch metadata")
    scalar = {name: cls.__name__ for name, cls in cfg.isa.operations.items() if cls.exu == EXU.SCALAR}
    sources = {name: read_source(root / "hardware" / name) for name in sorted({key[0] for key in _METHODS})}
    sources["isa_definition.py"] = read_source(root / "configs" / "isa_definition.py")
    return derive_frontend_sources(sources, scalar, fetch_width=cfg.fetch_width)
