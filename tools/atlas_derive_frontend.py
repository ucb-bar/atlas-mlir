"""Extract the reviewed Core/S1/scalar frontend contract without execution."""
from __future__ import annotations

import ast
import inspect
from pathlib import Path

from atlas_model_source import (GUARDS, UnsupportedModelTiming, one as _one, positive_int, read_source,
                                source_method)
from atlas_model_semantics import module_contract


_METHODS = (("core.py", "Core", "tick"), ("core.py", "Core", "_tick_s1"),
            ("exu.py", "ScalarExecutionUnit", "tick"), ("ifu.py", "InstructionFetch", "tick"))

# These issue-edge assumptions are checked by the Core, scalar, and fetch digests.
# They describe the guarded pipeline ordering, rather than counts of AST nodes.
PINNED_ASSUMPTIONS = dict(scalar_read_age=0, scalar_write_age=0,
                         control_operand_read_age=0, control_delay_slots=1)


def _attr(node, base, attr):
    return (isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name)
            and node.value.id == base and node.attr == attr)


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
    for filename in ("core.py", "exu.py", "ifu.py"):
        if module_contract(sources[filename]) != GUARDS["files"]["hardware/" + filename]:
            raise UnsupportedModelTiming(
                f"unsupported frontend pinned assumptions: {filename}: module contract changed")
    if fetch_width != 1 or type(fetch_width) is not int:
        raise UnsupportedModelTiming("unsupported frontend fetch width: source has one S1 issue slot")
    slots = {}
    methods = {}
    for filename, cls, name in _METHODS:
        node = source_method(sources[filename], cls, name)
        methods[(filename, cls, name)] = node
        if (cls, name) == ("Core", "_tick_s1"):
            mask = _one((item for item in ast.walk(node) if isinstance(item, ast.Assign)
                         and len(item.targets) == 1 and _attr(item.targets[0], "self", "delay_counter")
                         and isinstance(item.value, ast.BinOp) and isinstance(item.value.op, ast.BitAnd)),
                        "DELAY mask assignment")
            slots["delay_mask"] = positive_int(mask.value.right, "frontend DELAY mask", minimum=0, maximum=65536)
            decrement = _one((item for item in ast.walk(node) if isinstance(item, ast.AugAssign)
                              and _attr(item.target, "self", "delay_counter")
                              and isinstance(item.op, ast.Sub)), "DELAY countdown decrement")
            slots["decrement"] = positive_int(decrement.value, "frontend DELAY decrement", maximum=65536)
            if slots["decrement"] != 1:
                raise UnsupportedModelTiming(
                    "unsupported DELAY decrement: only unit countdown reaches zero for all immediates")
        if (cls, name) == ("ScalarExecutionUnit", "tick"):
            delay = _one((item for item in ast.walk(node) if isinstance(item, ast.Assign)
                          and len(item.targets) == 1 and _attr(item.targets[0], "uop", "execute_delay")),
                         "scalar completion delay")
            slots["completion"] = positive_int(delay.value, "frontend scalar completion delay",
                                                maximum=65536)
            if slots["completion"] != 1:
                raise UnsupportedModelTiming("unsupported scalar execute_delay tag: deferred completion is one tick, "
                                          "but the source tag no longer agrees")
    tick = methods[("core.py", "Core", "tick")]
    increment = _one((item for item in ast.walk(tick) if isinstance(item, ast.AugAssign)
                      and _attr(item.target, "self", "cycle_count") and isinstance(item.op, ast.Add)),
                     "Core clock increment")
    gap = positive_int(increment.value, "frontend Core clock increment", maximum=65536)
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
    return dict(ordinary_issue_gap=gap,
                delay=dict(base_gap=gap, immediate_mask=slots["delay_mask"], decrement_per_tick=slots["decrement"]),
                scalar=dict(read_age=PINNED_ASSUMPTIONS["scalar_read_age"],
                            write_age=PINNED_ASSUMPTIONS["scalar_write_age"]),
                control=dict(delay_slots=PINNED_ASSUMPTIONS["control_delay_slots"],
                             operand_read_age=PINNED_ASSUMPTIONS["control_operand_read_age"]),
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
