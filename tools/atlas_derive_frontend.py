"""Extract the reviewed Core/S1/scalar frontend contract without execution."""
from __future__ import annotations

import ast
from copy import deepcopy
import inspect
from pathlib import Path

from atlas_derive_scratchpad import method, read_source, shape, UnsupportedScratchpad
from atlas_model_diagnostics import UnsupportedModelTiming


class UnsupportedFrontend(UnsupportedModelTiming):
    """The model frontend has no supported source interpretation."""


_SHAPES = {('core.py', 'Core', 'tick'): 'bcccfaa10cdff924e81a427456f087618fd1eba266a07759082c6c89f8f08304',
 ('core.py', 'Core', '_tick_s1'): '1705edc5631443e0e3adc625898e709861c9ca5692d186601bb9ff884c5cbe98',
 ('core.py', 'Core', '_is_control_flow_instruction'): '6dd12795ccce3fc69272fe00f27e275f9e6e9b91d0e631d0cc7da15a47094532',
 ('exu.py', 'ScalarExecutionUnit', 'tick'): 'eea4b0f6d88b06bd4610ae02fdffd006b8e2eb9b29fea0819785f6a119d82141',
 ('ifu.py', 'InstructionFetch', 'tick'): 'db52d5d12a67e42ce308fddd9b50c36383b163abfbc0a7d210a60f87ca893895',
 ('arch_state.py', 'ArchState', 'set_npc'): '6d542a822d8d486f01cac57c353a0019e076c7760282f05fd90bf17279188f88'}
_SCALAR_SHAPE = 'fafc45104e28eed1a6c2808a379567a8cb8616cf7181e0bb14bde36e60005d7c'


def _one(values, label):
    values = list(values)
    if len(values) != 1:
        raise UnsupportedFrontend(f"unsupported frontend: expected one {label}")
    return values[0]


def _attr(node, base, attr):
    return (isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name)
            and node.value.id == base and node.attr == attr)


def _positive(node, label, allow_zero=False):
    if (not isinstance(node, ast.Constant) or type(node.value) is not int
            or not int(not allow_zero) <= node.value < 65536):
        raise UnsupportedFrontend(f"unsupported frontend {label}")
    return node.value


def _operand(node):
    if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name) and node.value.id == "self":
        return node.attr
    raise UnsupportedFrontend("unsupported scalar register operand expression")


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
        raise UnsupportedFrontend("unsupported frontend fetch width: source has one S1 issue slot")
    slots = {}
    methods = {}
    for filename, cls, name in _SHAPES:
        node = method(sources[filename], cls, name)
        methods[(filename, cls, name)] = node
        if (cls, name) == ("Core", "_tick_s1"):
            mask = _one((item for item in ast.walk(node) if isinstance(item, ast.Assign)
                         and len(item.targets) == 1 and _attr(item.targets[0], "self", "delay_counter")
                         and isinstance(item.value, ast.BinOp) and isinstance(item.value.op, ast.BitAnd)),
                        "DELAY mask assignment")
            slots["delay_mask"] = _positive(mask.value.right, "DELAY mask", allow_zero=True)
            # Only a mask on int(insn.imm) is supported, checked by the shape.
            mask.value.right = ast.Name(id="DERIVED_DELAY_MASK", ctx=ast.Load())
            decrement = _one((item for item in ast.walk(node) if isinstance(item, ast.AugAssign)
                              and _attr(item.target, "self", "delay_counter")
                              and isinstance(item.op, ast.Sub)), "DELAY countdown decrement")
            slots["decrement"] = _positive(decrement.value, "DELAY decrement")
            if slots["decrement"] != 1:
                raise UnsupportedFrontend("unsupported DELAY decrement: only unit countdown reaches zero for all immediates")
        if (cls, name) == ("ScalarExecutionUnit", "tick"):
            delay = _one((item for item in ast.walk(node) if isinstance(item, ast.Assign)
                          and len(item.targets) == 1 and _attr(item.targets[0], "uop", "execute_delay")),
                         "scalar completion delay")
            slots["completion"] = _positive(delay.value, "scalar completion delay")
            if slots["completion"] != 1:
                raise UnsupportedFrontend("unsupported scalar execute_delay tag: deferred completion is one tick, "
                                          "but the source tag no longer agrees")
            delay.value = ast.Name(id="DERIVED_SCALAR_COMPLETION", ctx=ast.Load())
        if shape(node) != _SHAPES[(filename, cls, name)]:
            raise UnsupportedFrontend(f"unsupported frontend edge/order semantics in {cls}.{name}")
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
    selected = []
    instructions = {}
    link_offset = None
    for mnemonic, class_name in sorted(scalar_classes.items()):
        if class_name not in classes:
            raise UnsupportedFrontend(f"missing scalar ISA source {class_name}")
        cls = classes[class_name]
        execute = _one((item for item in cls.body if isinstance(item, ast.FunctionDef) and item.name == "exec"),
                       f"{class_name}.exec")
        # Bind method to its source class and dispatch category, excluding ISA
        # encoding fields, which belong to the separate encoding adapter.
        selected.append(ast.ClassDef(name=class_name, bases=deepcopy(cls.bases),
                        keywords=[deepcopy(item) for item in cls.keywords if item.arg == "exu"],
                        body=[deepcopy(execute)], decorator_list=[], type_params=[]))
        bases = {item.id for item in cls.bases if isinstance(item, ast.Name)}
        kind = ("branch" if "SBType" in bases else "jump" if "UJType" in bases or "JalrPattern" in bases
                else "delay" if mnemonic == "delay" else "halt" if mnemonic in {"ecall", "ebreak"}
                else "control" if any("CSR" in item for item in bases) else "scalar")
        instructions[mnemonic] = dict(kind=kind, **_accesses(execute))
        if mnemonic == "jal":
            addition = _one((item for item in ast.walk(execute) if isinstance(item, ast.BinOp)
                             and _attr(item.left, "state", "execute_pc") and isinstance(item.op, ast.Add)
                             and isinstance(item.right, ast.Constant)), "JAL link-PC offset")
            link_offset = _positive(addition.right, "JAL link-PC offset")
    if shape(ast.Module(body=selected, type_ignores=[])) != _SCALAR_SHAPE:
        raise UnsupportedFrontend("unsupported scalar ISA execution/access semantics")
    if link_offset is None:
        raise UnsupportedFrontend("scalar ISA has no JAL definition")
    return dict(schema="atlas.model_frontend.v1", ordinary_issue_gap=gap,
                delay=dict(base_gap=gap, immediate_mask=slots["delay_mask"], decrement_per_tick=slots["decrement"]),
                scalar=dict(read_age=0, write_age=0, completion_delay=slots["completion"]),
                control=dict(delay_slots=delay_slots, operand_read_age=0, jal_link_pc_offset=link_offset),
                instructions=instructions)


def derive_frontend(cfg):
    root = Path(inspect.getfile(cfg.isa)).parent
    from npu_model.isa import EXU
    if "ScalarExecutionUnit" not in cfg.execution_units.values():
        raise UnsupportedFrontend("configuration does not provide the source scalar execution unit")
    for name, cls in cfg.isa.operations.items():
        if cls.exu == EXU.SCALAR and (cls.mnemonic != name or
                Path(inspect.getfile(cls)) != root / "configs" / "isa_definition.py"):
            raise UnsupportedFrontend("unsupported scalar ISA dispatch metadata")
    scalar = {name: cls.__name__ for name, cls in cfg.isa.operations.items() if cls.exu == EXU.SCALAR}
    sources = {name: read_source(root / "hardware" / name) for name in sorted({key[0] for key in _SHAPES})}
    sources["isa_definition.py"] = read_source(root / "configs" / "isa_definition.py")
    try:
        return derive_frontend_sources(sources, scalar, fetch_width=cfg.fetch_width)
    except UnsupportedScratchpad as error:
        raise UnsupportedFrontend(str(error)) from error
