"""Interpreter for a generated projected-Core transition program.

Expressions are bounded postfix integer programs. Neither consumer carries a
second copy of transition arithmetic. Graph dependency/unit serialization and
cold DMA serialization are explicit projection policies, never Core stalls.
"""
from __future__ import annotations

from copy import deepcopy
from functools import lru_cache
import ast
from heapq import heappop, heappush
import hashlib
import json

SCHEMA = "atlas.projected_core_replay.v1"
INPUTS = {"frontend", "unit_free", "dependency_end", "transfer_end", "capacity_end",
          "queued", "drains", "frontend_cost", "latency", "busy", "waiting", "makespan"}
TARGETS = {"cycle", "work", "finished", "unit_free", "waiting", "frontend", "makespan"}
OPS = {"add", "sub", "max", "select"}


def semantic_digest(program):
    return hashlib.sha256(json.dumps(program, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def instantiate_program(source):
    from atlas_search_policy import POLICY, canonical_hash
    template = POLICY["replay_projection"]
    def substitute(value):
        if isinstance(value, list):
            return [substitute(item) for item in value]
        return source["clock_step"] if value == "$clock_step" else value
    result = dict(schema=SCHEMA, source=deepcopy(source),
                  projection_policy=deepcopy(template["projection_policy"]),
                  start=substitute(template["start"]), place=substitute(template["place"]))
    result["projection_policy"]["template_sha256"] = canonical_hash(template)
    return result


def validate_program(program):
    if not isinstance(program, dict) or set(program) != {"schema", "source", "projection_policy", "start", "place"}:
        raise ValueError("unsupported projected-Core transition program")
    if program["schema"] != SCHEMA:
        raise ValueError("unsupported projected-Core transition schema")
    if not isinstance(program["source"], dict) or not isinstance(program["projection_policy"], dict):
        raise ValueError("missing projected-Core source/policy boundary")
    available = set(INPUTS)
    for stage in ("start", "place"):
        assignments = program[stage]
        if not isinstance(assignments, list) or not 1 <= len(assignments) <= 16:
            raise ValueError("unsupported transition stage")
        for assignment in assignments:
            if not isinstance(assignment, list) or len(assignment) != 2 or assignment[0] not in TARGETS:
                raise ValueError("unsupported transition assignment")
            expression = assignment[1]
            if not isinstance(expression, list) or not 1 <= len(expression) <= 64:
                raise ValueError("unsupported transition expression")
            depth = 0
            for token in expression:
                if type(token) is int and 0 <= token < 65536:
                    depth += 1
                elif isinstance(token, str) and token in OPS:
                    arity = 3 if token == "select" else 2
                    if depth < arity:
                        raise ValueError("transition stack underflow")
                    depth -= arity - 1
                elif isinstance(token, str) and token in available:
                    depth += 1
                else:
                    raise ValueError("unsupported transition token")
            if depth != 1:
                raise ValueError("invalid transition expression stack")
            available.add(assignment[0])
    if not {"cycle", "work", "finished"}.issubset(available):
        raise ValueError("missing transition outputs")
    source = program["source"]
    if (set(source) != {"schema", "clock_step", "phases", "dma_wait_observes", "implicit_exu_stalls", "checked_methods", "source_contract"}
            or source["schema"] != "atlas.core_edges.v1" or type(source["clock_step"]) is not int
            or source["clock_step"] != 1 or source["phases"] != ["s1_previous_state", "execution_units", "fetch"]
            or source["dma_wait_observes"] != "previous_edge_flags" or source["implicit_exu_stalls"] is not False
            or source["checked_methods"] != ["Core.tick", "Core._tick_s1"] or source["source_contract"] != "atlas.model_frontend.v1"):
        raise ValueError("unsupported projected-Core source edges")
    if program != instantiate_program(source):
        raise ValueError("transition program differs from declared projection policy")
    return program


def evaluate(assignments, state):
    for target, expression in assignments:
        stack = []
        for token in expression:
            if type(token) is int:
                stack.append(token)
            elif token in OPS:
                if token == "select":
                    no, yes, condition = stack.pop(), stack.pop(), stack.pop()
                    stack.append(yes if condition else no)
                else:
                    right, left = stack.pop(), stack.pop()
                    stack.append(left + right if token == "add" else left - right if token == "sub" else max(left, right))
            else:
                stack.append(state[token])
        state[target] = stack[0]


@lru_cache(maxsize=32)
def _compiled_replay(encoded):
    program = validate_program(json.loads(encoded))
    stages = {}
    for stage in ("start", "place"):
        body = []
        for target, expression in program[stage]:
            stack = []
            for token in expression:
                if type(token) is int:
                    stack.append(ast.Constant(token))
                elif token in OPS:
                    if token == "select":
                        no, yes, condition = stack.pop(), stack.pop(), stack.pop()
                        stack.append(ast.IfExp(condition, yes, no))
                    else:
                        right, left = stack.pop(), stack.pop()
                        stack.append(ast.Call(ast.Name("max", ast.Load()), [left, right], []) if token == "max"
                                     else ast.BinOp(left, ast.Add() if token == "add" else ast.Sub(), right))
                else:
                    stack.append(ast.Name(token, ast.Load()))
            body.append(ast.Assign([ast.Name(target, ast.Store())], stack.pop()))
        stages[stage] = body
    # Fixed queue/dependency adapter; all transition arithmetic is inserted from
    # validated policy bytecode. No model text or unrestricted expression is run.
    module = ast.parse('''def run(graph, order):
    proposal = graph["proposal"]
    costs = proposal["costs"]
    capacity = proposal["dma_capacity"]
    free, transfers = {}, []
    finished_nodes = [0] * len(costs)
    frontend = waiting = makespan = transfer_end = 0
    for node in order:
        cost = costs[node]
        queued, drains = cost["queued_dma"], cost["drains"]
        frontend_cost, latency, busy = cost["frontend"], cost["latency"], cost["busy"]
        unit = cost["unit"]
        unit_free = free.get(unit, 0)
        dependency_end = 0
        for pred in graph["preds"][node]:
            dependency_end = max(dependency_end, finished_nodes[pred])
        capacity_end = 0
        _start()
        if queued:
            while transfers and transfers[0] <= cycle:
                heappop(transfers)
            while len(transfers) >= capacity:
                capacity_end = heappop(transfers)
                _start()
        _place()
        finished_nodes[node] = finished
        free[unit] = unit_free
        if queued:
            heappush(transfers, unit_free)
            transfer_end = max(transfer_end, unit_free)
    return makespan, waiting
''')
    class InsertStages(ast.NodeTransformer):
        def visit_Expr(self, node):
            if isinstance(node.value, ast.Call) and isinstance(node.value.func, ast.Name):
                name = node.value.func.id.removeprefix("_")
                if name in stages:
                    return deepcopy(stages[name])
            return node
    module = InsertStages().visit(module)
    environment = {"__builtins__": {}, "max": max, "len": len, "heappop": heappop, "heappush": heappush}
    exec(compile(ast.fix_missing_locations(module), "<validated-projected-Core>", "exec"), environment)
    return environment["run"]


def prediction_error(graphs, native_cycles, scoring_profile="curve"):
    if not all(isinstance(graph, dict) and "proposal" in graph for graph in graphs):
        return dict(schema="atlas.proposal_prediction_error.v1", status="unavailable",
                    reason="graph has no proposal metadata", scoring_profile=scoring_profile)
    predicted = sum(proposal_estimate[0] for proposal_estimate in
                    (replay(graph, tuple(graph["order"])) for graph in graphs))
    difference = predicted - native_cycles
    return dict(schema="atlas.proposal_prediction_error.v1", status="available",
                scoring_profile=scoring_profile, proposal_cost_profile="curve", predicted_cycles=predicted,
                native_cycles=native_cycles, signed_error_cycles=difference,
                absolute_error_cycles=abs(difference),
                relative_error=(difference / native_cycles if native_cycles else None),
                approximation_scope="sum_of_cold_virtual_block_makespans_vs_complete_native_stream; excludes loop_execution_multiplicity_CFG_paths_and_lowering_effects")


def replay(graph, order):
    program = graph["proposal"]["replay"]
    encoded = json.dumps(program, sort_keys=True, separators=(",", ":"))
    return _compiled_replay(encoded)(graph, order)
