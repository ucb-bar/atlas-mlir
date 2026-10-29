"""Extract LSU/XLU event predicates and reservation releases from their source AST."""
from __future__ import annotations

from atlas_model_diagnostics import UnsupportedModelTiming

import ast
from copy import deepcopy
import inspect
from atlas_derive_source import read_source

from atlas_derive_mxu import _one, _method, _name
from atlas_model_schema import SOURCE_SHAPES, ast_digest, reservation as _reservation, timing_block


def _integer(node, rows):
    if isinstance(node, ast.Constant) and type(node.value) is int:
        return node.value
    if _name(node, "rows"):
        return rows
    if isinstance(node, ast.BinOp) and isinstance(node.op, (ast.Add, ast.Sub)):
        left, right = _integer(node.left, rows), _integer(node.right, rows)
        return left + right if isinstance(node.op, ast.Add) else left - right
    raise UnsupportedModelTiming("unsupported memory timing expression: " + ast.dump(node))


def _ages(predicate, rows):
    if len(predicate.ops) == 1 and isinstance(predicate.ops[0], ast.Eq) and _name(predicate.left, "age"):
        first = last = _integer(predicate.comparators[0], rows)
    elif (len(predicate.ops) == 2 and _name(predicate.comparators[0], "age")
          and all(isinstance(op, (ast.Lt, ast.LtE)) for op in predicate.ops)):
        first = _integer(predicate.left, rows) + isinstance(predicate.ops[0], ast.Lt)
        last = _integer(predicate.comparators[1], rows) - isinstance(predicate.ops[1], ast.Lt)
    else:
        raise UnsupportedModelTiming("unsupported memory timing predicate")
    if not 0 <= first <= last < 4096:
        raise UnsupportedModelTiming("memory timing predicate is outside the supported age range")
    return dict(first_age=first, last_age=last, count=last - first + 1, step=1)


def _source(module):
    path = inspect.getsourcefile(module)
    tree = ast.parse(read_source(path))
    # Only timing predicates are evaluated below. Everything else, including row
    # indexing, completion and reservation call sites, must keep its reviewed AST.
    digest = ast_digest(tree)
    if digest != SOURCE_SHAPES[path.rsplit("/", 1)[-1]]:
        raise UnsupportedModelTiming("unsupported LSU/XLU control-flow semantics")
    if path.rsplit("/", 1)[-1] == "xlu.py":
        # The full AST above guards the declaration and every use. Resolve the
        # reviewed public duration before interpreting the age expressions.
        from atlas_derive_mxu import _assignment
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
    return timing_block(name, name.split(".")[0], events, last)


def derive_memory(cfg):
    """Return six isolated blocks and separate reservation/port envelopes; no ticks."""
    from npu_model.hardware import lsu, xlu
    from npu_model.isa import EXU

    # The VPU extractor checks the shared bank checker digest before this phase.
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
        _block("vlsu.vload", {"vmem_read": read, "mreg_write": write}, finish["last_age"]),
        _block("vlsu.vstore", {"mreg_read": read, "vmem_write": write}, finish["last_age"]),
        _block("xlu.transpose", {"read": xread, "write": xwrite}, xfinish["last_age"]),
        _block("scalar_lsu.load", {"vmem_read": load, "load_write": response}, response["last_age"]),
        _block("scalar_lsu.scale_load", {"vmem_read": load, "scale_write": response}, response["last_age"]),
        _block("scalar_lsu.store", {"vmem_write": store}, store["last_age"]),
    ]
    releases = {block["name"]: {"unit": _reservation(block["first_free_age"] - 1)} for block in blocks}
    releases["vlsu.vload"]["mreg_write"] = _reservation(finish["last_age"], banks=[4])
    releases["vlsu.vstore"]["mreg_read"] = _reservation(finish["last_age"], banks=[0])
    releases["xlu.transpose"].update(mreg_read=_reservation(xrelease["last_age"], banks=[0]),
                                      mreg_write=_reservation(xfinish["last_age"], banks=[4]))
    for block in blocks:
        for key in ("vmem_read", "vmem_write"):
            if key in block["events"]:
                event = block["events"][key]
                releases[block["name"]][key] = _reservation(event["last_age"], first=event["first_age"], banks=[0])
    return {block["name"]: deepcopy(block) for block in blocks}, releases

