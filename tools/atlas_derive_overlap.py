#!/usr/bin/env python3
"""Derive bounded overlap proposals from reviewed source, without native runs.

No catalog, workload query, measured minimum or native observation is an input.
The native gate must replace validation.passed=false before consumer selection.
"""
from __future__ import annotations

import argparse
import ast
from copy import deepcopy
from functools import lru_cache
import hashlib
import inspect
import itertools
import json
import os
from pathlib import Path
import sys

sys.dont_write_bytecode = True

from atlas_derive_mxu import _plan, _check_checker, _assignment, _method, _one, _integer
from atlas_overlap_rules import RuleSet, engine, relation_class
from atlas_model_schema import SOURCE_SHAPES, VOCABULARY, canonical, sha256
from atlas_model_timing import configuration, source_digest, numerical_data, encoded

CONTRACT = "source-resource-occupancy-v1"


def safe_path(value):
    raw = Path(value)
    if any(word in part.lower() for part in raw.parts for word in ("hammer", "vlsi")):
        raise ValueError("prohibited overlap source/output path")
    path = Path(os.path.abspath(raw))
    for parent in reversed((path, *path.parents)):
        if parent.is_symlink():
            raise ValueError("overlap source/output path contains a symlink")
    return path


def load(path):
    def unique(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError("duplicate JSON key: " + key)
            result[key] = value
        return result
    return json.loads(safe_path(path).read_text(), object_pairs_hook=unique)


def _int(value):
    if type(value) is not int or not 0 <= value < 4096:
        raise ValueError("unsupported source integer")
    return value


def _literal(node):
    return _int(ast.literal_eval(node))


def _named(tree, name):
    return _one((node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == name), name)


def _age_bound(node, comparison):
    """One reviewed age expression's literal right endpoint."""
    return _one((_literal(n.comparators[0]) for n in ast.walk(node)
                 if isinstance(n, ast.Compare) and len(n.ops) == 1
                 and isinstance(n.ops[0], comparison) and isinstance(n.left, ast.BinOp)
                 and isinstance(n.left.op, ast.Sub) and isinstance(n.comparators[0], ast.Constant)),
                "source age endpoint")


def _guard(method, diagnostic):
    matches = []
    for node in ast.walk(method):
        if isinstance(node, ast.If) and any(isinstance(stmt, ast.Raise) and any(
                isinstance(part, ast.Constant) and isinstance(part.value, str) and diagnostic in part.value
                for part in ast.walk(stmt)) for stmt in node.body):
            matches.append(node.test)
    return _one(matches, diagnostic)


def _source(root, name):
    path = safe_path(root / "npu_model" / "hardware" / name)
    source = path.read_text()
    tree = ast.parse(source)
    return source, tree, sha256(path.read_bytes())


def _guard_ast(tree, name):
    reviewed = deepcopy(tree)
    if name == "vpu.py":
        reviewed.body = [n for n in reviewed.body if not (isinstance(n, ast.Assign) and any(
            isinstance(t, ast.Name) and t.id == "VPU_OP_LATENCIES" for t in n.targets))]
        if any(isinstance(n, ast.ClassDef) and n.name == "VectorExecutionUnit" for n in reviewed.body):
            from atlas_derive_vpu import _normalize_slot_capacity
            _normalize_slot_capacity(reviewed)
    if sha256(ast.dump(reviewed, include_attributes=False).encode()) != SOURCE_SHAPES[name]:
        raise ValueError("unsupported overlap control-flow semantics: " + name)


def extract_contract(model, facts):
    """Guard complete control flow, then extract its actual structural values."""
    model = safe_path(model)
    source_sha, _ = source_digest(model)
    numerical_sha, _ = numerical_data(model)
    identity = facts.get("model", {})
    if (identity.get("source_sha256") != source_sha or
            identity.get("numerical_data_sha256") != numerical_sha):
        raise ValueError("selected model differs from selected isolated facts")
    sys.path.insert(0, str(model))
    import npu_model
    if safe_path(Path(npu_model.__file__).parent.parent) != model:
        raise ValueError("imported model does not match --model")
    cfg, config = configuration()
    if identity.get("configuration") != config or identity.get("configuration_sha256") != sha256(encoded(config)):
        raise ValueError("selected configuration differs from isolated facts")
    from atlas_derive_vpu import derive_vpu
    from atlas_derive_mxu import derive_mxu
    from atlas_derive_memory import derive_memory
    # These existing guarded extractors call no tick or numerical instruction.
    derived, released = {}, {}
    for extractor in (derive_vpu, derive_mxu, derive_memory):
        blocks, releases = extractor(cfg)
        derived.update(blocks)
        released.update(releases)
    selected = {entry["name"]: entry for entry in facts["op_timing"]}
    if (facts.get("schema") != "atlas.model_timing.v1" or facts.get("age_convention") != "issue_edge_zero"
            or len(selected) != len(facts["op_timing"])):
        raise ValueError("unsupported or duplicate isolated operation facts")
    if any(selected.get(name) != value or facts.get("reservation_releases", {}).get(name) != released[name]
           for name, value in derived.items()):
        raise ValueError("isolated events/releases differ from reviewed source derivation")
    mxu_text, mxu, mxu_sha = _source(model, "mxu.py")
    _, vpu, vpu_sha = _source(model, "vpu.py")
    checker_text, _, checker_sha = _source(model, "bank_conflict.py")
    _, lsu, lsu_sha = _source(model, "lsu.py")
    _, xlu, xlu_sha = _source(model, "xlu.py")
    plans = _plan(mxu_text)
    _check_checker(checker_text)
    for name, tree in (("vpu.py", vpu), ("lsu.py", lsu), ("xlu.py", xlu)):
        _guard_ast(tree, name)
    base = _named(mxu, "_MatrixExecutionUnit")
    accept = _method(base, "_accept")
    active = _age_bound(_assignment(accept, "active").value, ast.Lt)
    acc_push_last = _age_bound(_assignment(accept, "acc_pushes").value, ast.LtE)
    weight_drain_last = _age_bound(_guard(accept, "same slot drained"), ast.LtE)
    acc_forward = _age_bound(_guard(accept, "cycle after a push"), ast.Lt)
    ports = sorted((n for n in ast.walk(accept) if isinstance(n, ast.Assign) and any(
        isinstance(t, ast.Name) and t.id == "ports" for t in n.targets)), key=lambda n: n.lineno)
    if len(ports) != 4 or not all(isinstance(n.value, ast.IfExp) for n in ports[2:]):
        raise ValueError("unsupported MXU sequencer port selection")
    port_roles = dict(compute=list(ast.literal_eval(ports[0].value)),
                      push_bf16=list(ast.literal_eval(ports[1].value)),
                      pop_bf16=list(ast.literal_eval(ports[2].value.body)),
                      pop_fp8=list(ast.literal_eval(ports[2].value.orelse)))
    priority = [ast.literal_eval(ports[3].value.body)[0], ast.literal_eval(ports[3].value.orelse)[0]]
    if any(type(port) is not int or not 0 <= port < 4 for values in [*port_roles.values(), priority] for port in values):
        raise ValueError("unsupported MXU sequencer port identifiers")
    depths = {unit: _literal(_assignment(_named(mxu, cls), "inflight_depth").value)
              for unit, cls in (("mxu0", "MatrixExecutionUnitSystolic"), ("mxu1", "MatrixExecutionUnitInner"))}
    vclass = _named(vpu, "VectorExecutionUnit")
    slot_assignment = _assignment(_method(vclass, "_accept"), "slot")
    slot_generator = slot_assignment.value.args[0]
    if not isinstance(slot_generator, ast.GeneratorExp):
        raise ValueError("unsupported VPU slot generator")
    slots = list(ast.literal_eval(slot_generator.generators[0].iter))
    if slots != list(range(len(slots))) or any(type(value) is not int for value in slots):
        raise ValueError("unsupported VPU slot identifiers")
    busy = _method(vclass, "_issue_blocked")
    busy_count = _one((_literal(n.comparators[0]) for n in ast.walk(busy)
                      if isinstance(n, ast.Compare) and isinstance(n.left, ast.Call)
                      and isinstance(n.left.func, ast.Name) and n.left.func.id == "len"), "VPU slot count")
    if busy_count != len(slots):
        raise ValueError("VPU issue capacity differs from its slots")
    mnemonic_to_block = {entry["mnemonic"]: entry["block"] for entry in VOCABULARY["vpu_blocks"]}
    groups = [sorted(mnemonic_to_block[name] for name in group)
              for group in ast.literal_eval(_assignment(vpu, "_GROUPS").value)]
    double = sorted(mnemonic_to_block[name] for name in (
        ast.literal_eval(_assignment(vpu, "_TWO_INPUT").value) |
        ast.literal_eval(_assignment(vpu, "_ROW_REDUCE").value)))
    scalar_gap = _age_bound(_guard(_method(_named(lsu, "LoadStoreUnit"), "_accept"),
                                  "prior load response pending"), ast.Lt)
    geometry = cfg.arch_state_config
    contract = dict(capacity=dict(sequencer_cycles=active, mxu0_compute=depths["mxu0"],
                                 mxu1_compute=depths["mxu1"], vpu_slots=busy_count),
        geometry=dict(mreg_count=geometry.num_m_registers, rows=geometry.mrf_depth,
                      row_bytes=geometry.mrf_width, vmem_bank_bytes=cfg.vmem_bank_bytes),
        source_sha256={"mxu.py": mxu_sha, "vpu.py": vpu_sha, "bank_conflict.py": checker_sha,
                       "lsu.py": lsu_sha, "xlu.py": xlu_sha},
        vpu=dict(slot_ids=slots, logic_groups=groups, double=double,
                 live_before_issue="age_lt_final_write", read_coalescing="same_logical_register_and_row"),
        mxu=dict(compute_first={name: values["compute"]["first_age"] for name, values in plans.items()},
                 compute_last={name: values["compute_last"] for name, values in plans.items()},
                 compute_tracker="pre_progress_compute_only_including_final_edge",
                 sequencer_active="age_lt_sequencer_cycles", single_push_priority=priority,
                 ports=port_roles,
                 push_paths={"weight": "weight", "push_fp8": "accumulator", "push_bf16": "accumulator"},
                 acc_push_last=acc_push_last, mxu0_weight_drain_last=weight_drain_last,
                 accumulator_forward_gap=acc_forward),
        scalar_lsu=dict(load_response_gap=scalar_gap),
        cross_unit=dict(engines=["vpu", "xlu"], shared_resources=["logical_mreg", "physical_mreg_ports"],
                        read_coalescing="VPU_only", composition="pairs_only"),
        mreg=dict(logical_release="next_begin_cycle", bank="reg_mod_32", row="reg_div_32_times_32_plus_row",
                  ports="one_read_one_write_different_rows", vload_may_write_during_read=True),
        vocabulary_sha256=sha256(canonical(VOCABULARY)),
        source_change_policy="reviewed_full_AST;normalized_MXU_timing_and_tracker_literals;paired_VPU_slot_literals",
        operand_domain="aligned_BF16_pairs;complete_legal_MXU_VPU_and_VPU_XLU_cross_command_relations")
    # Row indexing and physical bank geometry remain explicitly supported only
    # at 32. Tracker/slot literals above are source parameters, not config knobs.
    if active != 32:
        raise ValueError("unsupported MXU sequencer geometry (requires 32 cycles)")
    if any(not 1 <= depth <= 64 for depth in depths.values()) or not 1 <= busy_count <= 3:
        raise ValueError("unsupported source tracker/slot capacity")
    if source_digest(model)[0] != source_sha:
        raise ValueError("model source changed while deriving overlap")
    return contract


class SourceRules(RuleSet):
    """Existing event algebra with all structural values selected from source."""
    def __init__(self, facts, contract):
        super().__init__(facts)
        self.contract = contract
        self.logic_groups = tuple(set(group) for group in contract["vpu"]["logic_groups"])
        self.double = set(contract["vpu"]["double"])

    def capacity(self, unit):
        caps = self.contract["capacity"]
        return {"vpu": caps["vpu_slots"], "mxu0": caps["mxu0_compute"], "mxu1": caps["mxu1_compute"]}.get(unit)

    def capacity_conflict(self, operations, issues):
        assigned = {}
        caps, mxu = self.contract["capacity"], self.contract["mxu"]
        for j, (name, issue) in enumerate(zip(operations, issues)):
            unit = engine(name)
            if unit == "vpu":
                live = sum(engine(old) == unit and issue - issues[i] < self.first_free_age(old) - 1
                           for i, old in enumerate(operations[:j]))
                if live >= caps["vpu_slots"]:
                    return "VPU source issue slots"
            elif unit in {"mxu0", "mxu1"}:
                kind = self.kind(name)
                if kind == "compute" and sum(engine(old) == unit and self.kind(old) == "compute" and
                        issue - issues[i] <= mxu["compute_last"][unit]
                        for i, old in enumerate(operations[:j])) >= self.capacity(unit):
                    return unit + " source compute tracker depth"
                busy = {port for i, old in enumerate(operations[:j]) if engine(old) == unit and
                        issue - issues[i] < caps["sequencer_cycles"] for port in assigned[i]}
                if kind in mxu["ports"]:
                    ports = tuple(mxu["ports"][kind])
                else:
                    free = [port for port in mxu["single_push_priority"] if port not in busy]
                    if not free:
                        return unit + " source read sequencer capacity"
                    ports = (free[0],)
                if any(port in busy for port in ports):
                    return unit + " source sequencer port capacity"
                assigned[j] = ports
        return None

    def _structural_bound(self, first, second, relation):
        a, b = engine(first), engine(second)
        x, y = first.split(".", 1)[1], second.split(".", 1)[1]
        bound = 1
        if a == b == "vpu" and self.capacity(a) == 1:
            bound = self.first_free_age(first) - 1
        if a == b == "vpu" and (x == y or x in self.double or y in self.double or
                               any(x in group and y in group for group in self.logic_groups)):
            bound = self.first_free_age(first) - 1
        if a == b == "scalar_lsu" and x != "store" and y != "store":
            bound = self.contract["scalar_lsu"]["load_response_gap"]
        if a != b or a not in {"mxu0", "mxu1"}:
            return bound
        ka, kb = self.kind(first), self.kind(second)
        same_acc, same_weight = relation.get("accumulator") == "same", relation.get("weight_slot") == "same"
        mxu, sequencer = self.contract["mxu"], self.contract["capacity"]["sequencer_cycles"]
        if ka == kb == "compute" and self.capacity(a) == 1:
            bound = max(bound, mxu["compute_last"][a] + 1)
        if ka == "compute" and kb != "weight" and same_acc:
            bound = max(bound, mxu["compute_first"][a] + 1)
        reads_a = ka in {"compute", "weight", "push_fp8", "push_bf16"}
        reads_b = kb in {"compute", "weight", "push_fp8", "push_bf16"}
        if (ka.startswith("pop") and kb.startswith("pop") or
            reads_a and reads_b and (ka == "push_bf16" or kb == "push_bf16" or ka == kb == "compute") or
            ka == kb == "weight" or ka.startswith("push") and kb.startswith("push")):
            bound = max(bound, sequencer)
        if a == "mxu0" and ka == "compute" and kb == "weight" and same_weight:
            bound = max(bound, mxu["mxu0_weight_drain_last"] + 1)
        if a == "mxu1":
            if same_weight and ((ka == "weight" and kb in {"weight", "compute"}) or ka == "compute" and kb == "weight"):
                bound = max(bound, sequencer)
            if same_acc and ka.startswith("push") and (kb == "compute" or kb == "push_fp8"):
                bound = max(bound, mxu["acc_push_last"] + 1)
        if same_acc and ka.startswith("push") and (kb.startswith("pop") or y == "matmul_acc"):
            bound = max(bound, mxu["accumulator_forward_gap"])
        if same_acc:
            ea, eb = self.blocks[first]["events"], self.blocks[second]["events"]
            ar = next((ea[k] for k in ("acc_read", "acc_store") if k in ea), None)
            br = next((eb[k] for k in ("acc_read", "acc_store") if k in eb), None)
            aw = next((ea[k] for k in ("acc_write", "acc_load") if k in ea), None)
            bw = next((eb[k] for k in ("acc_write", "acc_load") if k in eb), None)
            if ar and br:
                bound = max(bound, ar["last_age"] - br["first_age"] + 1)
            if aw and bw and (ka != "compute" or kb != "compute"):
                bound = max(bound, aw["last_age"] - bw["first_age"] + 1)
        return bound


def actual_relation(a, b):
    terms = [f"{i}{'=' if x == y else '~'}{j}" for i, x in enumerate(a["registers"])
             for j, y in enumerate(b["registers"]) if x % 32 == y % 32]
    result = dict(mreg_banks=",".join(terms) or "disjoint")
    for key in ("accumulator", "weight_slot"):
        if key in a and key in b:
            result[key] = "same" if a[key] == b[key] else "other"
    return result


def operation_spec(name):
    family, block = name.split(".", 1)
    return _one((entry for entry in VOCABULARY[("vpu" if family == "vpu" else "mxu") + "_blocks"]
                 if entry["block"] == block), name)


def operand_roles(name):
    spec = operation_spec(name)
    family = "vpu" if engine(name) == "vpu" else "mxu"
    return [role for role in VOCABULARY[family + "_operand_roles"][spec["op_class"]]
            if role.get("resource", "MReg") == "MReg" and role["key"] is not None
            and (not role["two_input"] or spec.get("two_input", False))]


def command(name, values, *, acc=0, weight=0):
    spec = operation_spec(name)
    regs = [values[role["operand"]] + role["offset"] + index
            for role in operand_roles(name) for index in range(role["registers"])]
    result = dict(registers=regs)
    family = engine(name)
    if family != "vpu":
        if spec["op_class"] != "WeightPush":
            result["accumulator"] = acc
        if spec["op_class"] in {"WeightPush", "MatMul", "MatMulAcc"}:
            result["weight_slot"] = weight
    mnemonic = spec["mnemonic"] + ("." + family if family != "vpu" else "")
    if family != "vpu":
        kind = SourceRules.kind(name)
        if kind == "weight":
            args = f"w{weight}, m{values['rs1']}"
        elif kind.startswith("push"):
            args = f"acc{acc}, m{values['rs1']}"
        elif kind.startswith("pop"):
            args = f"m{values['rd']}, acc{acc}" + (", e0" if kind == "pop_fp8" else "")
        else:
            args = f"acc{acc}, m{values['rs1']}, w{weight}"
    elif mnemonic.startswith("vli."):
        args = f"m{values['rd']}, 0"
    elif spec["op_class"] in {"VpuPack", "VpuUnpack"}:
        args = f"m{values['rd']}, m{values['rs2']}, e0"
    else:
        args = f"m{values['rd']}, m{values['rs1']}" + (f", m{values['rs2']}" if spec["two_input"] else "")
    return result, mnemonic + " " + args


@lru_cache(maxsize=None)
def mreg_domain(first, second, rules=None):
    """Complete legal cross-command MREG equivalence relations."""
    result = {}
    if engine(first) != "vpu":
        ca, cb = len(operand_roles(first)), len(operand_roles(second))
        # A BF16 operand has two one-register roles. Enumerate all legal bases,
        # including m31/m32: bank and logical identity are different relations.
        keya = operand_roles(first)[0]["operand"]
        keyb = operand_roles(second)[0]["operand"]
        for a in range(64 - ca + 1):
            if ca == 2 and a % 2:
                continue
            for b in range(64 - cb + 1):
                if cb == 2 and b % 2:
                    continue
                av, bv = {keya: a}, {keyb: b}
                ac, aa = command(first, av)
                bc, ba = command(second, bv)
                relation = actual_relation(ac, bc)
                result.setdefault(relation["mreg_banks"], (av, bv, ac, bc, aa, ba))
    else:
        if rules is None:
            raise ValueError("complete VPU operand partition requires guarded source footprints")
        from atlas_vpu_partition import operand_domain
        for key, av, bv in operand_domain(operand_roles(first), operand_roles(second),
                                          rules.registers(first), rules.registers(second)):
            ac, aa = command(first, av)
            bc, ba = command(second, bv)
            if actual_relation(ac, bc)["mreg_banks"] != key:
                raise ValueError("operand partition differs from command register roles")
            result[key] = (av, bv, ac, bc, aa, ba)
    return result


def pair_domain(first, second, rules=None):
    for _, (av, bv, ac, bc, _, _) in sorted(mreg_domain(first, second, rules).items()):
        accs = (0, 1) if "accumulator" in ac and "accumulator" in bc else (0,)
        weights = (0, 1) if "weight_slot" in ac and "weight_slot" in bc else (0,)
        for acc, weight in itertools.product(accs, weights):
            left, aa = command(first, av)
            right, ba = command(second, bv, acc=acc, weight=weight)
            yield actual_relation(left, right), [left, right], [aa, ba]


def scalar_domain():
    for first, second in itertools.product(("load", "store"), repeat=2):
        fa, fb = ("lw" if name == "load" else "sw" for name in (first, second))
        modes = ["independent", "shared_base_read_only"]
        if first == second == "load":
            modes.append("same_load_destination")
        if second == "load":
            modes.append("first_address_base_equals_second_load_destination")
            if first == "store":
                modes.append("store_data_equals_load_destination")
        for mode in modes:
            for bank, window in (("same", "same"), ("same", "disjoint"), ("other", "disjoint")):
                if mode == "shared_base_read_only" and window != "same":
                    continue
                a_base, a_value, b_base, b_value = 1, 2, 3, 4
                if mode == "shared_base_read_only":
                    b_base = a_base
                elif mode == "same_load_destination":
                    b_value = a_value
                elif mode == "first_address_base_equals_second_load_destination":
                    b_value = a_base
                elif mode == "store_data_equals_load_destination":
                    b_value = a_value
                address = 0 if window == "same" else 4 if bank == "same" else 262144
                relation = dict(mreg_banks="disjoint", first_mnemonic=fa, second_mnemonic=fb,
                    scalar_address_alignment="word", scalar_registers=mode, vmem_bank=bank,
                    vmem_word_window=window)
                if mode != "independent":
                    relation["scalar_other_registers"] = "independent"
                    if mode != "shared_base_read_only":
                        relation["scalar_destination_file"] = "xrf"
                initial = {str(a_base): 0, str(b_base): address}
                if first == "store":
                    initial[str(a_value)] = 0x12345678
                if second == "store":
                    initial[str(b_value)] = 0x23456789
                assembly = [f"{fa} x{a_value}, 0(x{a_base})", f"{fb} x{b_value}, 0(x{b_base})"]
                yield "scalar_lsu." + first, "scalar_lsu." + second, relation, assembly, initial


def cross_domain(rules):
    """Complete legal VPU/XLU operands, under the guarded shared-bank model."""
    from atlas_vpu_partition import operand_domain
    transpose = "xlu.transpose"
    xroles = [dict(operand=operand, offset=0, registers=1, write=write)
              for operand, write in (("rs1", False), ("rd", True))]
    def concrete(name, values):
        if name == transpose:
            return (dict(registers=[values["rs1"], values["rd"]]),
                    f"vtrpose.xlu m{values['rd']}, m{values['rs1']}")
        return command(name, values)
    for vector in sorted(name for name in rules.blocks if engine(name) == "vpu"):
        for first, second in ((vector, transpose), (transpose, vector)):
            roles = [xroles if name == transpose else operand_roles(name) for name in (first, second)]
            for _, av, bv in operand_domain(*roles, rules.registers(first), rules.registers(second)):
                left, aa = concrete(first, av)
                right, ba = concrete(second, bv)
                yield first, second, actual_relation(left, right), [left, right], [aa, ba]


def chain_samples(rules):
    """Source mechanism controls, independent of benchmark/catalog sequences."""
    result = []
    def chain(identifier, unit, names, values, positive, negative, category, mechanism):
        commands = [command(name, value, acc=acc, weight=weight)
                    for name, (value, acc, weight) in zip(names, values)]
        for expectation, gaps in (("positive", positive), ("reject", negative)):
            result.append(dict(id=identifier + "-" + expectation, mechanism=mechanism, engine=unit,
                operations=names, operands=[item[0] for item in commands], assembly=[item[1] for item in commands],
                gaps=gaps, expectation=expectation, numerical_equivalent=True, category=category))
    for unit in ("mxu0", "mxu1"):
        cycle = rules.contract["capacity"]["sequencer_cycles"]
        chain(unit + "-read-port-history", unit,
              [unit + ".push_weight", unit + ".push_acc_fp8", unit + ".matmul"],
              [(dict(rs1=0), 0, 0), (dict(rs1=4), 0, 0), (dict(rs1=8), 1, 1)],
              [1, cycle], [1, 1], "port_history", "dynamic_read_sequencer_assignment")
        # The first push owns P1, forcing the weight push onto P0. Its
        # retirement must not erase that assignment from issue history. Choose
        # the earliest second issue making the one-short compute occur after
        # the first push has drained, rather than fitting a native minimum.
        retired = rules.first_free_age(unit + ".push_acc_fp8")
        second_issue = max(1, retired - cycle + 1)
        chain(unit + "-retired-prefix-port-history", unit,
              [unit + ".push_acc_fp8", unit + ".push_weight", unit + ".matmul"],
              [(dict(rs1=0), 0, 0), (dict(rs1=4), 0, 0), (dict(rs1=8), 1, 1)],
              [second_issue, cycle], [second_issue, cycle - 1],
              "port_history", "retired_prefix_read_sequencer_assignment")
        chain(unit + "-nonadjacent-bank", unit,
              [unit + ".matmul", unit + ".pop_acc_bf16", unit + ".push_weight"],
              [(dict(rs1=0), 0, 0), (dict(rd=4), 1, 0), (dict(rs1=0), 0, 1)],
              [1, cycle - 1], [1, cycle - 2], "nonadjacent", "nonadjacent_MREG_read_port")
        # Exercise depth when it can bind before sequencer/accumulator limits.
        # Two accumulators force alternating computes at least ceil(first/2)
        # apart, and the read sequencer imposes its own minimum interval.
        depth = rules.capacity(unit)
        first, last = rules.contract["mxu"]["compute_first"][unit], rules.contract["mxu"]["compute_last"][unit]
        interval = max(cycle, (first + 2) // 2)
        if depth * interval <= last:
            names = [unit + ".matmul"] * (depth + 1)
            values = [(dict(rs1=4 * (index % 8)), index % 2, 0) for index in range(depth + 1)]
            boundary = last + 1 - (depth - 1) * interval
            chain(unit + "-tracker-boundary", unit, names, values,
                  [interval] * (depth - 1) + [boundary],
                  [interval] * (depth - 1) + [boundary - 1],
                  "capacity", "source_compute_tracker_final_edge_reuse")
    final = rules.first_free_age("vpu.relu") - 1
    slots = rules.capacity("vpu")
    if slots == 1:
        names = ["vpu.relu", "vpu.exp2", "vpu.fp8pack"]
        values = [(dict(rs1=0, rd=4), 0, 0), (dict(rs1=8, rd=12), 0, 0), (dict(rs2=16, rd=20), 0, 0)]
        positive = [final, rules.first_free_age("vpu.exp2") - 1]
    else:
        names = ["vpu.relu", "vpu.exp2", *(["vpu.sin"] if slots == 3 else []), "vpu.fp8pack"]
        values = [(dict(**{"rs2" if name == "vpu.fp8pack" else "rs1": 8 * index}, rd=8 * index + 4), 0, 0)
                  for index, name in enumerate(names)]
        positive = [1] * (slots - 1) + [final - slots + 1]
    chain("vpu-slot-boundary", "vpu", names, values, positive,
          [*positive[:-1], positive[-1] - 1], "capacity", "source_slots_final_write_reuse")
    if slots > 1:
        # One-slot issue admission serializes this nonadjacent RAW before its
        # register lifetime can bind; do not label a slot rejection as RAW.
        release = rules.releases["vpu.relu"]["mreg_write"]["available_age"]
        chain("vpu-nonadjacent-RAW", "vpu", ["vpu.relu", "vpu.exp2", "vpu.fp8pack"],
              [(dict(rs1=0, rd=4), 0, 0), (dict(rs1=8, rd=12), 0, 0), (dict(rs2=4, rd=20), 0, 0)],
              [1, release - 1], [1, release - 2],
              "nonadjacent", "nonadjacent_logical_write_release")
    return result


def generate(model_root, per_op_path, output_path, samples_path=None):
    model, per_op, output = map(safe_path, (model_root, per_op_path, output_path))
    facts = load(per_op)
    contract = extract_contract(model, facts)
    rules = SourceRules(facts, contract)
    pairs, samples, classes = [], [], {}
    engines = {unit: sorted(name for name in rules.blocks if engine(name) == unit)
               for unit in ("mxu0", "mxu1", "vpu")}
    def add(first, second, relation, assembly, operands=None, initial_xrf=None):
        minimum, high = rules.minimum_gap(first, second, relation), rules.first_free_age(first)
        if minimum > high:
            return
        pair = dict(first=first, second=second, relation=relation, minimum=minimum, high=high)
        index = len(pairs)
        pairs.append(pair)
        unit = engine(first) if engine(first) == engine(second) else "vpu_xlu"
        klass = relation_class(relation)
        if unit in engines:
            classes[(unit, canonical(klass))] = dict(engine=unit, relation=klass)
        bounds = rules.bounds(first, second, tuple(sorted(relation.items())))
        mechanisms = [key for key, value in bounds.items() if value == minimum]
        sample = dict(id=f"pair-{index}", mechanism="+".join(mechanisms), engine=unit,
                      operations=[first, second], assembly=assembly, gaps=[minimum], relation=relation,
                      minimum=minimum, high=high, pair_indices=[index], expectation="positive",
                      numerical_equivalent=True, category="pair")
        if operands is not None:
            sample["operands"] = operands
        if initial_xrf is not None:
            sample["initial_xrf"] = initial_xrf
        samples.append(sample)
    for unit, names in engines.items():
        for first, second in itertools.product(names, repeat=2):
            for relation, operands, assembly in pair_domain(first, second, rules):
                add(first, second, relation, assembly, operands)
    for first, second, relation, assembly, initial in scalar_domain():
        add(first, second, relation, assembly, initial_xrf=initial)
    for first, second, relation, operands, assembly in cross_domain(rules):
        add(first, second, relation, assembly, operands)
    # Runtime composition still requires every exact live pair in this table.
    # Optional ACC/weight roles differ by opcode, so requiring one identical
    # class across every opcode Cartesian pair would discard valid classes.
    composition = [value for _, value in sorted(classes.items())]
    samples.extend(chain_samples(rules))
    tool_names = ("atlas_derive_overlap.py", "atlas_overlap_rules.py", "atlas_vpu_partition.py", "atlas_derive_mxu.py",
                  "atlas_derive_vpu.py", "atlas_derive_memory.py", "atlas_model_schema.py", "atlas_model_timing.py",
                  "atlas_model_observer.py", "model-timing.lock.json",
                  "model-timing-vocabulary.json")
    tools = {name: sha256(safe_path(Path(__file__).parent / name).read_bytes()) for name in tool_names}
    contract["producer_sources"] = tools
    artifact = dict(schema="atlas.model_overlap_derived.v1", age_convention="issue_edge_zero",
        per_op_sha256=sha256(per_op.read_bytes()), model=facts["model"],
        derivation=dict(contract=CONTRACT, tool_sha256=sha256(canonical(tools)),
                        source_contract_sha256=sha256(canonical(contract)), source_contract=contract),
        validation=dict(passed=False, report_sha256="0" * 64, sample_policy="source-mechanism-boundaries-v2",
                        positive_runs=0, negative_runs=0, negative_rejections=0, numerical_checks=0,
                        engines={name: dict(positive_runs=0, negative_runs=0, negative_rejections=0,
                                            numerical_checks=0, chain_positive_runs=0, chain_negative_runs=0)
                                 for name in (*engines, "scalar_lsu", "vpu_xlu")}),
        pairs=pairs, composition=dict(engines=engines, classes=composition))
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(artifact, sort_keys=True, indent=2) + "\n")
    samples_out = safe_path(samples_path) if samples_path is not None else output.with_name("samples.json")
    samples_out.parent.mkdir(parents=True, exist_ok=True)
    samples_out.write_text(json.dumps(dict(schema="atlas.overlap_samples.v1", samples=samples,
        exclusions=["No DMA or mixed-engine chain overlap admission.",
                    "Cross-engine pair coverage is limited to VPU/XLU.",
                    "Native validation is separate; this candidate has validation.passed=false."]), sort_keys=True, indent=2) + "\n")
    return artifact


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--per-op", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--samples")
    args = parser.parse_args()
    try:
        result = generate(args.model, args.per_op, args.output, args.samples)
    except (OSError, ValueError, RuntimeError) as error:
        parser.exit(1, f"overlap derivation failed: {error}\n")
    print(json.dumps(dict(pairs=len(result["pairs"]), classes=len(result["composition"]["classes"]),
                          validation_passed=False), sort_keys=True))


if __name__ == "__main__":
    main()
