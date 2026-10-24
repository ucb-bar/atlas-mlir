"""Source-derived operands for representative compiler proposal costs.

The field-order placement algorithm is compiler policy. Register spans, limits,
instruction roles and legal constructors come from the selected model. A probe
interval describes that pair only; it is never a general overlap admission.
"""
from __future__ import annotations

import ast
import inspect
import textwrap

from atlas_model_diagnostics import UnsupportedModelTiming
from atlas_search_policy import POLICY


class UnsupportedCosts(UnsupportedModelTiming):
    """The selected model has no reviewed operand-independent timing class."""


def _tree(function):
    from atlas_derive_vocabulary import _safe
    _safe(inspect.getsourcefile(function))
    return ast.parse(textwrap.dedent(inspect.getsource(function))).body[0]


def check_timing_conditions(node, register_fields):
    """Prove planner timing predicates do not depend on register selectors.

    Source-shape contracts separately bound calls/control flow. This small
    dependency check follows local numeric aliases to reject operand-sensitive
    conditions; returned bank selectors may depend on operands as intended.
    """
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

    # Fixed point handles aliases regardless of AST walk order.
    changed = True
    while changed:
        before = set(tainted)
        for part in ast.walk(node):
            if isinstance(part, (ast.Assign, ast.AnnAssign)):
                targets = part.targets if isinstance(part, ast.Assign) else [part.target]
                for target in targets:
                    propagate(target, part.value)
        changed = tainted != before
    for part in ast.walk(node):
        condition = (part.test if isinstance(part, (ast.If, ast.IfExp, ast.While))
                     else None)
        if condition is not None and depends(condition):
            raise UnsupportedCosts("register-dependent timing condition in " + node.name)
    return depends


def _invariance(cfg):
    from atlas_derive_vpu import _check_source
    from atlas_derive_mxu import _validated
    from atlas_derive_memory import derive_memory
    from npu_model.hardware import vpu
    _check_source(vpu, latency_table=True)
    fields = {"vd", "vs1", "vs2", "es1", "rd", "rs1", "rs2"}
    for function in (vpu.VectorExecutionUnit._reads, vpu.VectorExecutionUnit._writes):
        check_timing_conditions(_tree(function), fields)
    latency = _tree(vpu.VectorExecutionUnit._execution_latency)
    depends = check_timing_conditions(latency, fields)
    if any(depends(part.value) for part in ast.walk(latency) if isinstance(part, ast.Return)):
        raise UnsupportedCosts("operand-dependent VPU execution latency")
    accept = _tree(vpu.VectorExecutionUnit._accept)
    reads = [part.value for part in ast.walk(accept) if isinstance(part, ast.Assign)
             and any(isinstance(target, ast.Name) and target.id == "read_last" for target in part.targets)]
    if len(reads) != 1 or any(depends(value) for value in reads):
        raise UnsupportedCosts("operand-dependent VPU read lifetime")
    # Both extractors evaluate only age/geometry expressions under checked full
    # control-flow shapes. An instruction/data expression in an age is refused.
    _validated(cfg)
    derive_memory(cfg)
    return dict(kind="guarded_operand_independent_isolated_ages",
                vpu="mnemonic_latency_and_register_independent_row_predicates",
                mxu="guarded_age_geometry_expressions",
                memory="guarded_age_geometry_predicates",
                interval_scope="selected_representative_pair_only")


def derive_costs(cfg, architecture, isa_metadata):
    from atlas_derive_vocabulary import _parameters
    from npu_model import isa_types
    from npu_model.isa import EXU

    proof = _invariance(cfg)
    stride = architecture["bf16_registers_per_tile"]
    if stride <= 0:
        raise UnsupportedCosts("invalid source-derived register span")
    # Fixed field order keeps the compiler's preexisting proposal trajectory.
    field_index = dict(rd=0, rs1=1, rs2=2)
    limits = dict(MReg=architecture["mreg_count"], XReg=architecture["xreg_count"],
                  EReg=architecture["ereg_count"], Weight=architecture["weight_buffer_count"])
    probes = {}
    for name, cls in sorted(cfg.isa.operations.items()):
        if name not in isa_metadata and cls.exu != EXU.SCALAR:
            continue
        metadata = isa_metadata.get(name)
        operands = metadata["operands"] if metadata else _parameters(cls)
        mxu = metadata is not None and metadata["engine"] in {"mxu0", "mxu1"}
        pair = [dict(rd=0, rs1=0, rs2=0, imm=0) for _ in range(2)]
        # Unused encoded scalar destinations retain the compiler encoding
        # adapter's field position, while constructors omit nonexistent fields.
        if cls.exu == EXU.SCALAR:
            pair[1]["rd"] = stride * len(field_index)
        for operand in operands:
            resource, field = operand["resource"], operand["compiler_operand"]
            if resource is None and field == "imm":
                continue
            if field not in field_index:
                raise UnsupportedCosts("unsupported compiler cost operand " + field)
            kind = getattr(isa_types, operand["type"])
            lower, upper = kind.lower_bound, kind.upper_bound
            capacity = min(upper, limits.get(resource, upper))
            if lower != 0 or capacity <= 0:
                raise UnsupportedCosts("unsupported operand capacity " + name + "." + field)
            if mxu:
                first, second = 0, stride if resource == "MReg" else 0 if resource == "EReg" else 1
            else:
                first = stride * field_index[field]
                second = stride * (field_index[field] + len(field_index))
            accesses = operand.get("accesses", [])
            extent = max((row["offset"] + row["registers"] for row in accesses), default=1)
            if first + extent > capacity or second + extent > capacity:
                raise UnsupportedCosts("canonical disjoint cost probes exceed " + name + "." + field)
            pair[0][field], pair[1][field] = first, second
        # Constructors enforce the selected ISA's encoded operand domains.
        for probe in pair:
            cls(**{row["name"]: probe[row["compiler_operand"]] for row in operands})
        probes[name] = pair
    return dict(schema="atlas.proposal_costs.v1", timing_class=proof, probes=probes,
                placement_policy=POLICY["algorithms"]["proposal_operand_placement"])
