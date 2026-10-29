"""Extract ISA operands and resource roles from the selected model's pure planners."""
from __future__ import annotations

import argparse
import ast
from collections import Counter, defaultdict
from copy import deepcopy
from dataclasses import fields
import inspect
import json
from pathlib import Path
import sys
from types import FunctionType, SimpleNamespace
import textwrap

from atlas_model_diagnostics import UnsupportedModelTiming
from atlas_model_schema import VOCABULARY, canonical


class UnsupportedVocabulary(UnsupportedModelTiming):
    """ISA or planner resources cannot be represented by the compiler adapter."""


# Names describe the compiler's encoding, not hardware resource/timing values.


_OPERANDS = dict(vd="rd", vs1="rs1", vs2="rs2", es1="rs1", imm="imm",
                 rd="rd", rs1="rs1", rs2="rs2")
_RESOURCES = dict(MatrixReg="MReg", WeightBuffer="Weight", Accumulator="Acc",
                  ExponentReg="EReg", ScalarReg="XReg")
_PLANNER_RESOURCES = dict(mrf="MReg", wb="Weight", acc="Acc")


def _compiler_operands(cls):
    """Translate the model's assembly parameter order into compiler tokens."""
    from npu_model.isa_types import Bundled, Imm12, ScalarReg
    prefixes = dict(MatrixReg="m", WeightBuffer="w", Accumulator="a",
                    ExponentReg="e", ScalarReg="x")
    suffixes = dict(rd="d", rs1="1", rs2="2")
    tokens = []
    for parameter in cls.params:
        if isinstance(parameter, Bundled):
            # The compiler's @ token means exactly imm12(rs1). A different
            # bundle cannot be represented by the same assembly adapter.
            if (parameter.reg.inner is not ScalarReg or parameter.reg.repr != "rs1"
                    or parameter.imm.inner is not Imm12 or parameter.imm.repr != "imm"):
                raise UnsupportedVocabulary(f"unsupported bundled assembly operand for {cls.mnemonic}")
            tokens.append("@")
            continue
        name = parameter.inner.__name__
        if name in prefixes:
            field = _OPERANDS.get(parameter.repr)
            if field not in suffixes:
                raise UnsupportedVocabulary(f"unsupported assembly operand {cls.mnemonic}.{parameter.repr}")
            tokens.append(prefixes[name] + suffixes[field])
        elif parameter.repr == "imm":
            tokens.append("i")
        else:
            raise UnsupportedVocabulary(f"unsupported assembly operand type {cls.mnemonic}: {name}")
    return " ".join(tokens)


def _isa_directions(cls):
    """Follow ISA accessors and local helpers without running numerical code."""
    directions = defaultdict(set)
    resource_files = dict(xrf="XReg", erf="EReg", mrf="MReg", wb="Weight", acc="Acc")
    operand_resources = {field.name: _RESOURCES.get(getattr(field.type, "__name__", None))
                         for field in fields(cls)}
    register_operands = {name for name, resource in operand_resources.items() if resource}
    pair_planner = str(getattr(cls, "exu", "")) in {
        "VectorExecutionUnit", "MatrixExecutionUnitSystolic", "MatrixExecutionUnitInner"}
    def origin(node, bindings):
        if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name) and node.value.id == "self":
            return node.attr
        if isinstance(node, ast.Name):
            return bindings.get(node.id)
        if isinstance(node, ast.BinOp):
            operand = origin(node.left, bindings)
            # VPU/MXU MREG pair offsets are independently represented by their
            # row planners. Other selectors must name the encoded register;
            # folding an ERF/XRF or memory-ISA shift would lose a dependency.
            if (pair_planner and operand_resources.get(operand) == "MReg"
                    and isinstance(node.op, ast.Add) and isinstance(node.right, ast.Constant)
                    and type(node.right.value) is int):
                return operand
        return None
    def visit(function, bindings, active):
        if function in active:
            raise UnsupportedVocabulary(f"recursive ISA resource helper for {cls.mnemonic}")
        tree = ast.parse(textwrap.dedent(inspect.getsource(function)))
        root = tree.body[0]
        parents = {child: node for node in ast.walk(tree) for child in ast.iter_child_nodes(node)}
        def resolved_helper(call):
            helper = function.__globals__.get(call.func.id) if isinstance(call.func, ast.Name) else None
            return helper if inspect.isfunction(helper) and helper.__module__ == cls.__module__ else None
        def refuse(detail):
            raise UnsupportedVocabulary(f"{cls.mnemonic}: unsupported ISA resource syntax: {detail}")
        for node in ast.walk(tree):
            if isinstance(node, (ast.Lambda, ast.AnnAssign, ast.NamedExpr, ast.For, ast.AsyncFor,
                                 ast.While, ast.ListComp, ast.SetComp, ast.DictComp, ast.GeneratorExp,
                                 ast.Try, ast.With, ast.AsyncWith, ast.Match, ast.Import, ast.ImportFrom)):
                refuse(type(node).__name__)
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) and node is not root:
                refuse("nested definition")
            if isinstance(node, ast.Name) and node.id == "state" and isinstance(node.ctx, ast.Load):
                parent = parents[node]
                if isinstance(parent, ast.Attribute) and parent.value is node:
                    if parent.attr in {"cfg", "math"}:
                        continue
                    if not parent.attr.startswith(("read_", "write_")):
                        refuse("direct state attribute " + parent.attr)
                    call = parents.get(parent)
                    if not isinstance(call, ast.Call) or call.func is not parent:
                        refuse("state accessor aliases")
                    if call.keywords:
                        refuse("keyword state accessor arguments")
                elif isinstance(parent, ast.Call) and node in parent.args:
                    helper = resolved_helper(parent)
                    if helper is None:
                        refuse("unresolved helper dispatch receiving state")
                    parameters = helper.__code__.co_varnames[:helper.__code__.co_argcount]
                    position = parent.args.index(node)
                    if parent.keywords or position >= len(parameters) or parameters[position] != "state":
                        refuse("helper state parameter or keyword arguments")
                else:
                    refuse("state aliases or keyword helper arguments")
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in {
                    "eval", "exec", "compile", "globals", "locals", "__import__"}:
                refuse("dynamic helper dispatch " + node.func.id)
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        value = origin(node.value, bindings)
                        if bindings.get(target.id) is not None and bindings[target.id] != value:
                            refuse("reassigned register alias " + target.id)
                        bindings[target.id] = value
            if not isinstance(node, ast.Call):
                continue
            if isinstance(node.func, ast.Attribute) and isinstance(node.func.value, ast.Name) and node.func.value.id == "state":
                parts = node.func.attr.split("_")
                if len(parts) >= 2 and parts[0] in {"read", "write"}:
                    if parts[1] not in resource_files:
                        if parts[1] not in {"vmem", "dram"}:
                            raise UnsupportedVocabulary(f"{cls.mnemonic}: unsupported ISA resource accessor {node.func.attr}")
                        continue
                    selector = 1 if parts[1] in {"wb", "acc"} else 0
                    operand = origin(node.args[selector], bindings) if len(node.args) > selector else None
                    if operand not in register_operands:
                        refuse("unresolved register selector for " + node.func.attr)
                    directions[operand].add((resource_files[parts[1]], parts[0]))
            else:
                helper = resolved_helper(node)
                if helper:
                    if node.keywords:
                        refuse("keyword helper arguments")
                    parameters = helper.__code__.co_varnames[:helper.__code__.co_argcount]
                    child = {parameter: origin(argument, bindings) for parameter, argument in zip(parameters, node.args)}
                    visit(helper, child, active | {function})
                elif any(origin(argument, bindings) in register_operands
                         for argument in [*node.args, *(keyword.value for keyword in node.keywords)]):
                    refuse("unresolved helper dispatch receiving a register operand")
    visit(cls.exec, {}, set())
    return directions


def _parameters(cls):
    try:
        result = []
        for field in fields(cls):
            if field.name not in inspect.signature(cls).parameters:
                continue
            name = getattr(field.type, "__name__", None)
            if name is None or field.name not in _OPERANDS:
                raise UnsupportedVocabulary(f"unsupported ISA operand {cls.mnemonic}.{field.name}: {field.type}")
            result.append(dict(name=field.name, compiler_operand=_OPERANDS[field.name], type=name,
                               resource=_RESOURCES.get(name), read=False, write=False, accesses=[]))
        if {row["name"] for row in result} != set(inspect.signature(cls).parameters):
            raise UnsupportedVocabulary(f"unsupported ISA constructor operands for {cls.mnemonic}")
        return result
    except TypeError as error:
        raise UnsupportedVocabulary(f"ISA operation {cls.mnemonic} lacks dataclass operand metadata") from error


def _instruction(cls, operands):
    values = {}
    for operand in operands:
        name, resource = operand["name"], operand["resource"]
        values[name] = ({"vd": 16, "vs1": 0, "vs2": 8}.get(name, 0) if resource == "MReg"
                        else 1 if name == "vs2" else 0)
    return cls(**values), values


def _issue(cfg, unit_cls, insn, engine):
    captured = {}
    def stage(uop, state, reads, writes, *args):
        captured.update(reads=reads, writes=writes)
        return None
    accept = unit_cls._accept
    issue = FunctionType(accept.__code__, dict(accept.__globals__, StagedExecution=stage),
                         accept.__name__, accept.__defaults__, accept.__closure__)
    checker = SimpleNamespace(reserve_mreg=lambda *args, **kwargs: None)
    unit = object.__new__(unit_cls)
    unit.name, unit.config, unit.lane_id = engine, cfg, 0
    unit.arch_state = SimpleNamespace(cfg=cfg.arch_state_config, conflict_checker=checker)
    unit.logger = SimpleNamespace(log_stage_start=lambda *args, **kwargs: None)
    unit.reset()
    unit.cycle = 1
    issue(unit, SimpleNamespace(insn=insn, id=0, execute_delay=0))
    plans = unit.operations if engine == "vpu" else unit._ops
    if len(plans) != 1 or set(captured) != {"reads", "writes"}:
        raise UnsupportedVocabulary(f"{insn.mnemonic}: issue did not produce one resource plan")
    return unit, plans[0], captured


def _dimensions(requests, base, rows):
    """Split simultaneous pair ports; combine sequenced banks; count row passes."""
    banks = sorted({bank for age, bank, row in requests})
    if not banks:
        return []
    if banks != list(range(base, base + len(banks))):
        raise UnsupportedVocabulary("noncontiguous or shifted MREG operand plan")
    counts = Counter((bank, row) for age, bank, row in requests)
    passes = set(counts.values())
    if len(passes) != 1 or set(counts) != {(bank, row) for bank in banks for row in range(rows)}:
        raise UnsupportedVocabulary("MREG operand planner has incomplete or nonuniform row passes")
    simultaneous = max(Counter(age for age, bank, row in requests).values()) > 1
    groups = [[bank] for bank in banks] if simultaneous else [banks]
    return [dict(offset=group[0] - base, registers=len(group), passes=next(iter(passes))) for group in groups]


def _metadata(cfg, adapter):
    from npu_model.hardware import vpu, mxu
    from npu_model.isa import EXU
    from atlas_derive_mxu import _plan
    import npu_model.configs.isa_definition  # noqa: F401
    mxu_windows = _plan(Path(inspect.getsourcefile(mxu)).read_text())
    specs = [(entry, "vpu", vpu.VectorExecutionUnit) for entry in adapter["vpu_blocks"]]
    specs += [(dict(entry, mnemonic=entry["mnemonic"] + "." + engine), engine, cls)
              for engine, cls in (("mxu0", mxu.MatrixExecutionUnitSystolic), ("mxu1", mxu.MatrixExecutionUnitInner))
              for entry in adapter["mxu_blocks"]]
    expected = {entry["mnemonic"] for entry, engine, cls in specs}
    actual = {name for name, cls in cfg.isa.operations.items() if cls.exu in {EXU.VECTOR, EXU.MATRIX_SYSTOLIC, EXU.MATRIX_INNER}}
    if actual != expected:
        raise UnsupportedVocabulary(f"unsupported ISA vocabulary coverage: missing={sorted(expected-actual)}, added={sorted(actual-expected)}")
    result = {}
    for entry, engine, unit_cls in specs:
        mnemonic = entry["mnemonic"]
        cls = cfg.isa.operations[mnemonic]
        expected_exu = EXU.VECTOR if engine == "vpu" else unit_cls.exu_type
        if cls.mnemonic != mnemonic or cls.exu != expected_exu:
            raise UnsupportedVocabulary(f"unsupported ISA dispatch for {mnemonic}: expected {expected_exu}")
        operands = _parameters(cls)
        insn, values = _instruction(cls, operands)
        unit, plan, staged = _issue(cfg, unit_cls, insn, engine)
        for direction, locations in staged.items():
            if any(location[0] not in _PLANNER_RESOURCES for location in locations):
                raise UnsupportedVocabulary(f"{mnemonic}: unrecognized {direction} resource in unit planner")
            for location in locations:
                resource, bank = _PLANNER_RESOURCES[location[0]], location[-1]
                matches = [operand for operand in operands if operand["resource"] == resource and
                           (values[operand["name"]] <= bank < values[operand["name"]] + 8 if resource == "MReg"
                            else values[operand["name"]] == bank)]
                if len(matches) != 1:
                    raise UnsupportedVocabulary(f"{mnemonic}: unmapped {direction} resource in unit planner: {location}")
        for operand in operands:
            resource, base = operand["resource"], values[operand["name"]]
            for direction, write in (("reads", False), ("writes", True)):
                locations = [location for location in staged[direction]
                             if _PLANNER_RESOURCES.get(location[0]) == resource
                             and location[-1] == base]
                if locations:
                    operand["write" if write else "read"] = True
                    if resource == "MReg":
                        if engine == "vpu":
                            planner = unit._writes if write else unit._reads
                            requests = [(age, bank, row) for age in range(plan.write_last + 1)
                                        for bank, row in planner(plan, age)
                                        if bank in {location[-1] for location in staged[direction]
                                                    if location[0] == "mrf" and base <= location[-1] < base + 8}]
                            dimensions = _dimensions(requests, base, cfg.arch_state_config.mrf_depth)
                        else:
                            window = mxu_windows[engine]["transfer" if write else "read"]
                            requests = [(row, bank, row) for row in range(window["count"]) for bank in plan.banks]
                            dimensions = _dimensions(requests, base, cfg.arch_state_config.mrf_depth)
                        operand["accesses"].extend(dict(dim, write=write) for dim in dimensions)
                    else:
                        operand["accesses"].append(dict(offset=0, registers=1, passes=1, write=write))
            # Exponent scale is an ISA source sampled at issue, outside MREG row planners.
            if resource == "EReg":
                directions = _isa_directions(cls)[operand["name"]]
                if directions != {("EReg", "read")}:
                    raise UnsupportedVocabulary(f"{mnemonic}.{operand['name']}: unsupported exponent resource directions {sorted(directions)}")
                operand["read"] = True
                operand["accesses"] = [dict(offset=0, registers=1, passes=1, write=False)]
            if resource is not None and not (operand["read"] or operand["write"]):
                raise UnsupportedVocabulary(f"{mnemonic}.{operand['name']}: ISA resource absent from unit planner")
        result[mnemonic] = dict(engine=engine, op_class=entry["op_class"], block=entry["block"],
                                compiler_operands=_compiler_operands(cls), operands=operands)
        if engine != "vpu":
            result[mnemonic]["mreg_ports"] = list(plan.ports)
    # Their numerical accessor direction is source metadata, not an observation.
    for mnemonic in ("lw", "sw", "seld", "vload", "vstore", "vtrpose.xlu"):
        cls = cfg.isa.operations[mnemonic]
        expected_exu = EXU.XLU if mnemonic == "vtrpose.xlu" else EXU.LSU
        if cls.mnemonic != mnemonic or cls.exu != expected_exu:
            raise UnsupportedVocabulary(f"unsupported ISA dispatch for {mnemonic}: expected {expected_exu}")
        operands = _parameters(cls)
        directions = _isa_directions(cls)
        for operand in operands:
            resource = operand["resource"]
            if resource:
                accesses = directions[operand["name"]]
                if not accesses or any(found != resource for found, direction in accesses):
                    raise UnsupportedVocabulary(f"{mnemonic}.{operand['name']}: ISA accessor resource disagrees with its operand type")
                operand["read"] = (resource, "read") in accesses
                operand["write"] = (resource, "write") in accesses
        result[mnemonic] = dict(engine="xlu" if cls.exu == EXU.XLU else "vlsu" if mnemonic in {"vload", "vstore"} else "scalar_lsu",
                               compiler_operands=_compiler_operands(cls), operands=operands,
                               role_source="isa_accessors")
    return result


def _events(metadata, family):
    """Translate derived resources to the compiler's named timing-event API."""
    operands = metadata["operands"]
    events = []
    if family == "vpu":
        for write in (False, True):
            index = 0
            for operand in sorted(operands, key=lambda row: (row["resource"] == "EReg", row["name"])):
                for dim in operand["accesses"]:
                    if dim["write"] != write:
                        continue
                    key = None if operand["resource"] == "EReg" else ("write" if write else "read") + str(index)
                    if key is not None:
                        index += 1
                    events.append(dict(key=key, operand=operand["compiler_operand"], **dim,
                                       two_input=metadata["op_class"] == "VpuElementwise" and operand["name"] == "vs2"))
    else:
        for operand in sorted(operands, key=lambda row: (row["name"] == "vd", row["resource"] == "EReg")):
            resource = operand["resource"]
            # Weight read windows have a dedicated footprint in mxuFootprint.
            if resource == "Weight" and operand["read"]:
                continue
            for dim in operand["accesses"]:
                write = dim["write"]
                if resource == "MReg":
                    index = dim["offset"]
                    if not write:
                        index = metadata["mreg_ports"][index]
                    key = ("mreg_write" if write else "mreg_read") + str(index)
                elif resource == "Weight":
                    key = "weight_write"
                elif resource == "Acc":
                    key = ("acc_write" if write else "acc_read") if metadata["op_class"].startswith("MatMul") else ("acc_load" if write else "acc_store")
                elif resource == "EReg":
                    key = None
                else:
                    continue
                events.append(dict(key=key, operand=operand["compiler_operand"], resource=resource,
                                   **dim, two_input=False))
        if metadata["op_class"].startswith("MatMul"):
            events.sort(key=lambda event: (event["resource"] != "MReg", event["write"]))
    return events


def derive_isa_metadata(cfg, *, template=None):
    """Return model operand types and planner read/write/row-access metadata."""
    try:
        return _metadata(cfg, VOCABULARY if template is None else template)
    except UnsupportedVocabulary:
        raise
    except (AttributeError, KeyError, TypeError, ValueError, RuntimeError) as error:
        raise UnsupportedVocabulary(f"unsupported ISA vocabulary planner: {error}") from error


def derive_vocabulary(cfg, *, template=None):
    """Rebuild the legacy compiler vocabulary, retaining only naming adapters."""
    adapter = VOCABULARY if template is None else template
    result = deepcopy(adapter)
    metadata = derive_isa_metadata(cfg, template=adapter)
    for family in ("vpu", "mxu"):
        roles = {}
        for entry in result[family + "_blocks"]:
            name = entry["mnemonic"] + (".mxu0" if family == "mxu" else "")
            operation = metadata[name]
            if family == "vpu":
                entry["two_input"] = sum(row["resource"] == "MReg" and row["read"] for row in operation["operands"]) > 1
            events = _events(operation, family)
            old = roles.setdefault(entry["op_class"], events)
            if old != events:
                # The elementwise class combines unary and binary ISA forms.
                left = [event for event in old if not event["two_input"]]
                right = [event for event in events if not event["two_input"]]
                if left != right:
                    raise UnsupportedVocabulary(f"compiler class {entry['op_class']} has incompatible model operand plans")
                if len(events) > len(old):
                    roles[entry["op_class"]] = events
            if family == "mxu" and events != _events(metadata[entry["mnemonic"] + ".mxu1"], family):
                raise UnsupportedVocabulary(f"{entry['mnemonic']}: MXU engines disagree on operand resources")
        result[family + "_operand_roles"] = roles
    return result


def validate_vocabulary(cfg, vocabulary=None):
    """Refuse a stale generated compiler snapshot before deriving timing facts."""
    vocabulary = VOCABULARY if vocabulary is None else vocabulary
    derived = derive_vocabulary(cfg, template=vocabulary)
    if canonical(derived) != canonical(vocabulary):
        for family in ("vpu_blocks", "mxu_blocks", "vpu_operand_roles", "mxu_operand_roles"):
            if derived[family] != vocabulary[family]:
                raise UnsupportedVocabulary(f"generated compiler vocabulary is stale: {family}; regenerate with atlas_derive_vocabulary.py")
    return derived


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--npu-model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    model, output = args.npu_model, args.output
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(model))
    from atlas_model_timing import configuration
    cfg, _ = configuration()
    output.write_text(json.dumps(derive_vocabulary(cfg), indent=2) + "\n")


if __name__ == "__main__":
    main()
