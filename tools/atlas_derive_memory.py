"""Extract LSU/XLU event predicates and reservation releases from their source AST."""
from __future__ import annotations

import ast
from copy import deepcopy
import inspect

from atlas_model_source import (UnsupportedModelTiming, age_window, assignment as _assignment,
                                method as _method, module_tree, name as _name, one as _one,
                                reservation as _reservation, timing_block)


def _ages(predicate, rows):
    return age_window(predicate, {"rows": rows}, equality=True, operators=(ast.Add, ast.Sub))


def _source(module):
    path = inspect.getsourcefile(module)
    tree = module_tree(module, reviewed=True, label="LSU/XLU control-flow")
    # Only timing predicates are evaluated below. Everything else, including row
    # indexing, completion and reservation call sites, must keep its reviewed AST.
    if path.rsplit("/", 1)[-1] == "xlu.py":
        # The full AST above guards the declaration and every use. Resolve the
        # reviewed public duration before interpreting the age expressions.
        duration = _assignment(tree, "TRANSPOSE_CYCLES").value
        if not isinstance(duration, ast.Constant) or type(duration.value) is not int:
            raise UnsupportedModelTiming("unsupported XLU transpose duration")
        class ResolveDuration(ast.NodeTransformer):
            def visit_Name(self, node):
                if isinstance(node.ctx, ast.Load) and node.id == "TRANSPOSE_CYCLES":
                    return deepcopy(duration)
                return node
        tree = ResolveDuration().visit(tree)
    return tree


def _predicates(method):
    return [node for node in ast.walk(method) if isinstance(node, ast.Compare)
            and (_name(node.left, "age") or any(_name(item, "age") for item in node.comparators))]


def _block(name, events, last):
    if last < max(event["last_age"] for event in events.values()):
        raise UnsupportedModelTiming("engine completes before its final event")
    return timing_block(name, events, last)


def derive_memory(cfg):
    """Return six isolated blocks and separate reservation/port envelopes; no ticks."""
    from npu_model.hardware import lsu, xlu
    from npu_model.isa import EXU

    # The VPU extractor checks the shared bank checker digest before this phase.
    if {name for name, op in cfg.isa.operations.items() if op.exu == EXU.XLU} != {"vtrpose.xlu"}:
        raise UnsupportedModelTiming("unsupported XLU ISA operation coverage")
    geometry = cfg.arch_state_config
    if (geometry.mrf_depth, geometry.mrf_width, geometry.num_m_registers, cfg.vmem_bank_bytes) != (32, 32, 64, 262144):
        raise UnsupportedModelTiming("unsupported LSU/XLU bank geometry")
    if not {"LoadStoreUnit", "CrossLaneExecutionUnit"} <= set(cfg.execution_units.values()):
        raise UnsupportedModelTiming("configuration does not provide LSU/XLU engines")
    for name in ("vload", "vstore", "lw", "seld", "sw", "vtrpose.xlu"):
        operation = cfg.isa.operations.get(name)
        if (operation is None or operation.mnemonic != name
                or operation.exu != (EXU.XLU if name == "vtrpose.xlu" else EXU.LSU)):
            raise UnsupportedModelTiming("unsupported memory ISA metadata")
        params = ({"vd", "vs1"} if name == "vtrpose.xlu" else {"vd", "rs1", "imm"} if name.startswith("v")
                  else {"rs1", "rs2", "imm"} if name == "sw" else {"rd", "rs1", "imm"})
        if set(inspect.signature(operation).parameters) != params:
            raise UnsupportedModelTiming("unsupported memory ISA operands")
    classes = {}
    for module, name in ((lsu, "LoadStoreUnit"), (xlu, "CrossLaneExecutionUnit")):
        classes[name] = _one((node for node in _source(module).body
                              if isinstance(node, ast.ClassDef) and node.name == name), name)
    scalar = _predicates(_method(classes["LoadStoreUnit"], "_tick_scalar"))
    vector = _predicates(_method(classes["LoadStoreUnit"], "_tick_vector"))
    transpose = _predicates(_method(classes["CrossLaneExecutionUnit"], "tick"))
    if (len(scalar), len(vector), len(transpose)) != (3, 3, 4):
        raise UnsupportedModelTiming("unexpected memory timing predicates")
    rows = geometry.mrf_depth
    # AST walk visits scalar store, load request, load response in source order.
    store, load, response = (_ages(node, rows) for node in scalar)
    read, write, finish = (_ages(node, rows) for node in vector)
    xread, xrelease, xwrite, xfinish = (_ages(node, rows) for node in transpose)
    blocks = [
        _block("vload", {"vmem_read": read, "mreg_write": write}, finish["last_age"]),
        _block("vstore", {"mreg_read": read, "vmem_write": write}, finish["last_age"]),
        _block("vtrpose.xlu", {"read": xread, "write": xwrite}, xfinish["last_age"]),
        _block("lw", {"vmem_read": load, "load_write": response}, response["last_age"]),
        _block("seld", {"vmem_read": load, "scale_write": response}, response["last_age"]),
        _block("sw", {"vmem_write": store}, store["last_age"]),
    ]
    releases = {block["name"]: {"unit": _reservation(block["first_free_age"] - 1)} for block in blocks}
    releases["vload"]["mreg_write"] = _reservation(finish["last_age"])
    releases["vstore"]["mreg_read"] = _reservation(finish["last_age"])
    releases["vtrpose.xlu"].update(mreg_read=_reservation(xrelease["last_age"]),
                                      mreg_write=_reservation(xfinish["last_age"]))
    for block in blocks:
        for key in ("vmem_read", "vmem_write"):
            if key in block["events"]:
                event = block["events"][key]
                releases[block["name"]][key] = _reservation(event["last_age"], first=event["first_age"])
    return {block["name"]: deepcopy(block) for block in blocks}, releases
