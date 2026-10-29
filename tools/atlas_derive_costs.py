"""Derive representative cost operands from ISA metadata and register geometry."""
from __future__ import annotations

import ast
from dataclasses import fields
import inspect
import textwrap

from atlas_model_diagnostics import UnsupportedModelTiming


def check_timing_conditions(node, register_fields):
    """Refuse timing predicates that depend on register selectors or aliases."""
    tainted = set()

    def depends(expression):
        return any((isinstance(part, ast.Attribute) and part.attr in register_fields)
                   or (isinstance(part, ast.Name) and part.id in tainted)
                   for part in ast.walk(expression))

    def propagate(target, value):
        if (isinstance(target, (ast.Tuple, ast.List)) and isinstance(value, (ast.Tuple, ast.List))
                and len(target.elts) == len(value.elts)):
            for destination, origin in zip(target.elts, value.elts):
                propagate(destination, origin)
        elif depends(value):
            tainted.update(item.id for item in ast.walk(target)
                           if isinstance(item, ast.Name) and isinstance(item.ctx, ast.Store))

    changed = True
    while changed:
        before = set(tainted)
        for part in ast.walk(node):
            if isinstance(part, (ast.Assign, ast.AnnAssign)) and part.value is not None:
                targets = part.targets if isinstance(part, ast.Assign) else [part.target]
                for target in targets:
                    propagate(target, part.value)
        changed = tainted != before
    for part in ast.walk(node):
        condition = part.test if isinstance(part, (ast.If, ast.IfExp, ast.While)) else None
        if condition is not None and depends(condition):
            raise UnsupportedModelTiming("register-dependent timing condition in " + node.name)
    return depends


def _tree(function):
    return ast.parse(textwrap.dedent(inspect.getsource(function))).body[0]


def _invariance():
    from npu_model.hardware.vpu import VectorExecutionUnit as VPU

    selectors = {"vd", "vs1", "vs2", "es1", "rd", "rs1", "rs2"}
    for function in (VPU._reads, VPU._writes):
        check_timing_conditions(_tree(function), selectors)
    latency = _tree(VPU._execution_latency)
    depends = check_timing_conditions(latency, selectors)
    if any(depends(part.value) for part in ast.walk(latency) if isinstance(part, ast.Return)):
        raise UnsupportedModelTiming("register-dependent VPU execution latency")
    accept = _tree(VPU._accept)
    reads = [part.value for part in ast.walk(accept) if isinstance(part, ast.Assign)
             and any(isinstance(target, ast.Name) and target.id == "read_last" for target in part.targets)]
    if len(reads) != 1 or any(depends(value) for value in reads):
        raise UnsupportedModelTiming("register-dependent VPU read lifetime")
    # MXU and memory extraction accept only age/geometry expressions. The shared
    # source guard pins their control flow and these VPU planner definitions.


def _parameters(cls):
    names = dict(vd="rd", vs1="rs1", vs2="rs2", es1="rs1",
                 rd="rd", rs1="rs1", rs2="rs2", imm="imm")
    signature = inspect.signature(cls).parameters
    result = [(field.name, names.get(field.name), field.type)
              for field in fields(cls) if field.name in signature]
    if {name for name, _, _ in result} != set(signature) or any(field is None for _, field, _ in result):
        raise UnsupportedModelTiming("unsupported cost operands for " + cls.mnemonic)
    return result


def derive_costs(cfg, architecture):
    """Check the geometry-based compiler probes against the selected ISA."""
    from npu_model.isa import EXU
    from npu_model.isa_types import MatrixReg, ScalarReg, ExponentReg, WeightBuffer

    _invariance()
    stride = architecture["bf16_registers_per_tile"]
    if type(stride) is not int or stride <= 0:
        raise UnsupportedModelTiming("cost probe register span must be positive")
    arch = cfg.arch_state_config
    limits = {MatrixReg: architecture["mreg_count"], ScalarReg: arch.num_x_registers,
              ExponentReg: arch.num_e_registers, WeightBuffer: arch.num_wb_registers}
    selected = {EXU.SCALAR, EXU.VECTOR, EXU.MATRIX_SYSTOLIC, EXU.MATRIX_INNER, EXU.XLU}
    positions = dict(rd=0, rs1=1, rs2=2)
    probes = {}
    for name, cls in sorted(cfg.isa.operations.items()):
        if cls.exu not in selected and name not in {"lw", "sw", "seld", "vload", "vstore"}:
            continue
        operands = _parameters(cls)
        mxu = cls.exu in {EXU.MATRIX_SYSTOLIC, EXU.MATRIX_INNER}
        pair = [dict(rd=0, rs1=0, rs2=0, imm=0) for _ in range(2)]
        if cls.exu == EXU.SCALAR:
            pair[1]["rd"] = stride * len(positions)
        for parameter, field, kind in operands:
            if field == "imm":
                continue
            capacity = min(kind.upper_bound, limits.get(kind, kind.upper_bound))
            if kind.lower_bound != 0 or capacity <= 0:
                raise UnsupportedModelTiming("unsupported cost operand capacity " + name + "." + field)
            if mxu:
                first, second = 0, stride if kind is MatrixReg else 0 if kind is ExponentReg else 1
            else:
                first = stride * positions[field]
                second = stride * (positions[field] + len(positions))
            # Reserve a complete source pair even for instructions using one half.
            extent = stride if kind is MatrixReg else 1
            if first + extent > capacity or second + extent > capacity:
                raise UnsupportedModelTiming("cost probes exceed " + name + "." + field)
            pair[0][field], pair[1][field] = first, second
        for probe in pair:
            cls(**{parameter: probe[field] for parameter, field, _ in operands})
        probes[name] = pair
    return probes
