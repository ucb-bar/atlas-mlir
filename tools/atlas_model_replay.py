"""Fixed cold virtual-graph recurrence, declared compiler projection policy.

Source-checked Core edges supply the clock step. Graph dependency/unit and cold
DMA serialization are projection policies, not implicit native Core stalls.
"""
from __future__ import annotations

from copy import deepcopy
from functools import lru_cache
from heapq import heappop, heappush
import hashlib
import json

SCHEMA = "atlas.projected_core_replay.v2"


def semantic_digest(program):
    return hashlib.sha256(json.dumps(program, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def instantiate_program(source):
    from atlas_search_policy import POLICY, canonical_hash
    template = POLICY["replay_projection"]
    result = dict(schema=SCHEMA, source=deepcopy(source),
                  projection_policy=deepcopy(template["projection_policy"]))
    result["projection_policy"]["policy_sha256"] = canonical_hash(template)
    return result


def validate_program(program):
    if not isinstance(program, dict) or set(program) != {"schema", "source", "projection_policy"}:
        raise ValueError("unsupported projected-Core replay descriptor")
    if program["schema"] != SCHEMA:
        raise ValueError("unsupported projected-Core replay schema")
    if not isinstance(program["source"], dict) or not isinstance(program["projection_policy"], dict):
        raise ValueError("missing projected-Core source/policy boundary")
    source = program["source"]
    if (set(source) != {"schema", "clock_step", "phases", "dma_wait_observes", "implicit_exu_stalls", "checked_methods", "source_contract"}
            or source["schema"] != "atlas.core_edges.v1" or type(source["clock_step"]) is not int
            or source["clock_step"] != 1 or source["phases"] != ["s1_previous_state", "execution_units", "fetch"]
            or source["dma_wait_observes"] != "previous_edge_flags" or source["implicit_exu_stalls"] is not False
            or source["checked_methods"] != ["Core.tick", "Core._tick_s1"] or source["source_contract"] != "atlas.model_frontend.v1"):
        raise ValueError("unsupported projected-Core source edges")
    if program != instantiate_program(source):
        raise ValueError("replay descriptor differs from declared projection policy")
    return program


@lru_cache(maxsize=32)
def _clock_step(encoded):
    return validate_program(json.loads(encoded))["source"]["clock_step"]


def replay(graph, order):
    proposal = graph["proposal"]
    clock_step = _clock_step(json.dumps(proposal["replay"], sort_keys=True, separators=(",", ":")))
    costs, capacity = proposal["costs"], proposal["dma_capacity"]
    free, transfers = {}, []
    finished = [0] * len(costs)
    frontend = waiting = makespan = transfer_end = 0
    for node in order:
        cost = costs[node]
        unit_free = free.get(cost["unit"], 0)
        cycle = frontend if cost["queued_dma"] else max(frontend, unit_free)
        if cost["drains"]:
            cycle = max(cycle, transfer_end)
        for pred in graph["preds"][node]:
            cycle = max(cycle, finished[pred])
        if cost["queued_dma"]:
            while transfers and transfers[0] <= cycle:
                heappop(transfers)
            while len(transfers) >= capacity:
                cycle = max(cycle, heappop(transfers))
        work = max(cycle + cost["frontend"] - clock_step, unit_free) if cost["queued_dma"] else cycle
        finished[node] = work + cost["latency"]
        free[cost["unit"]] = work + cost["busy"]
        waiting += cycle - frontend
        frontend = cycle + cost["frontend"]
        makespan = max(makespan, finished[node])
        if cost["queued_dma"]:
            heappush(transfers, free[cost["unit"]])
            transfer_end = max(transfer_end, free[cost["unit"]])
    return makespan, waiting


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
