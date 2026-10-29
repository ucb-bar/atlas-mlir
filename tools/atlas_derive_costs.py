"""Check register invariance and capacity for representative compiler probes."""
from __future__ import annotations

import ast
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


def derive_costs(cfg, architecture):
    """Check the loaded geometry used by representativeProbes."""
    _invariance()
    stride = architecture["bf16_registers_per_tile"]
    if type(stride) is not int or stride <= 0:
        raise UnsupportedModelTiming("cost probe register span must be positive")
    # representativeProbes uses six disjoint spans and scalar destination 3*stride.
    if 6 * stride > architecture["mreg_count"] or 3 * stride + 1 > cfg.arch_state_config.num_x_registers:
        raise UnsupportedModelTiming("cost probes exceed configured registers")
