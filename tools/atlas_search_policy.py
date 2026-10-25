"""Versioned compiler policy; algorithm identifiers must match reviewed implementation semantics."""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path

POLICY_PATH = Path(__file__).with_name("model-search-policy.json")


def canonical_hash(value: dict) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def fixed_execution_contract(contract: dict) -> dict:
    """This target experiment supports exactly RTL numerics and fatal errors."""
    if (not isinstance(contract, dict) or set(contract) != {
            "schema", "numerics", "randomize_init", "ignore_runtime_errors"}
            or contract["schema"] != "atlas.model_execution_contract.v1"
            or contract["numerics"] != "rtl" or contract["randomize_init"] is not False
            or contract["ignore_runtime_errors"] is not False):
        raise ValueError("unsupported model execution contract; RTL numerics, deterministic initialization and fatal errors are required")
    return dict(contract)


def load_policy() -> dict:
    policy = json.loads(POLICY_PATH.read_text())
    if not isinstance(policy, dict) or policy.get("schema") != "atlas.search_policy.v1":
        raise ValueError("unsupported search policy schema")
    if not isinstance(policy.get("revision"), str) or not policy["revision"]:
        raise ValueError("missing search policy revision")
    # Labels bind implementation revisions, not arbitrary configurable algorithms.
    algorithms = {
        "proposal_operand_placement": "fixed_encoded_field_order_source_pair_stride",
        "proposal_cost_adapter": "representative-disjoint-copy-grouped-pack-setup-omitted-optimistic-filled-slot-v2",
        "selection_objective": "selected-profile-cycles-then-index-v1",
        "selection_tie": "canonical-candidate-index-v1",
        "curve_candidates_default": "model-curve-profile-only-v1",
        "proposal_replay": "atlas.projected_core_replay.v2",
        "group_neighbors": "dependency-closure-insertion-v1",
        "candidate_ranking": "stable-summed-estimates-prefix-v1",
        "search_restart": "restart-on-any-objective-improvement-v1",
        "budget_accounting": "refusals-count-patience-after-batch-v1",
        "score_cache": "invocation-profile-exact-word-stream-v1",
        "virtual_scheduler": "start-plus-spill-frontend-negative-height-source-index-v1",
        "lowering_alternatives": "compiler-verified-pack-wait-placement-v1",
    }
    if policy.get("algorithms") != algorithms:
        raise ValueError("unsupported search policy algorithm revision")
    def integers(values, minimum=0):
        return isinstance(values, list) and all(type(v) is int and v >= minimum for v in values)
    def keys(value, expected, label):
        if not isinstance(value, dict) or set(value) != set(expected.split()):
            raise ValueError("unsupported search policy fields: " + label)
    keys(policy, "schema revision algorithms presets defaults pools candidates objectives engineering dma_proposal replay_projection validation execution_contract", "definition")
    keys(policy["presets"], "selection full", "presets")
    for name, preset in policy["presets"].items():
        keys(preset, "max_candidates search_group_evaluations search_patience bank_candidates pack_reload_candidates pack_preload_candidates", name)
        for key, value in preset.items():
            if key.endswith("candidates") and key != "max_candidates":
                if type(value) is not bool:
                    raise ValueError("invalid preset boolean: " + key)
            elif type(value) is not int or value < (1 if key == "max_candidates" else 0):
                raise ValueError("invalid preset budget: " + key)
    keys(policy["defaults"], "lower_provider selection_provider model_profile builtin_profile search_budget workers guarded_pack spill spill_mxu boundary api_max_candidates api_search_group_evaluations api_search_patience api_pack_reload_candidates api_pack_preload_candidates", "defaults")
    defaults = policy["defaults"]
    for name in ("workers", "api_max_candidates", "api_search_group_evaluations", "api_search_patience"):
        if type(defaults[name]) is not int or defaults[name] < (1 if name in ("workers", "api_max_candidates") else 0):
            raise ValueError("invalid API default budget: " + name)
    for name in ("guarded_pack", "spill", "spill_mxu", "api_pack_reload_candidates", "api_pack_preload_candidates"):
        if type(defaults[name]) is not bool:
            raise ValueError("invalid API boolean: " + name)
    if (defaults["lower_provider"] not in ("builtin", "model") or defaults["selection_provider"] not in ("builtin", "model")
            or defaults["model_profile"] not in ("curve", "fifo43") or defaults["builtin_profile"] not in ("curve", "fifo43")
            or defaults["search_budget"] not in policy["presets"] or defaults["boundary"] not in ("boundary", "none")):
        raise ValueError("invalid driver default")
    fixed_execution_contract(policy["execution_contract"])
    keys(policy["validation"], "core_edges reviewed_suite", "validation suites")
    suite = policy["validation"]["reviewed_suite"]
    keys(suite, "revision scope limits", "reviewed validation suite")
    if suite["revision"] != "atlas.reviewed_model_validation.v2":
        raise ValueError("unsupported reviewed validation suite revision")
    for name in ("scope", "limits"):
        if not isinstance(suite[name], list) or not suite[name] or not all(isinstance(row, str) and row for row in suite[name]):
            raise ValueError("missing reviewed validation suite " + name)
    edges = policy["validation"]["core_edges"]
    keys(edges, "schema algorithm delay_immediates ordinary_max_cycles dma_wait_max_cycles", "Core edge validation")
    if edges["schema"] != "atlas.core_edge_validation_policy.v1" or edges["algorithm"] != "native-phase-delay-prior-flag-boundaries-v1":
        raise ValueError("unsupported Core edge validation revision")
    if not edges["delay_immediates"] or not integers(edges["delay_immediates"]) or not integers([edges["ordinary_max_cycles"], edges["dma_wait_max_cycles"]], 1):
        raise ValueError("invalid Core edge validation suite")
    pools, candidates = policy["pools"], policy["candidates"]
    keys(pools, "supported_lookaheads initial_lookaheads virtual_scheduler_lookaheads random_seeds neighbor_shuffle_seed group_max_nodes mreg_seeds mreg_rank_choices mreg_vector_length uniform_store_ranks", "pools")
    keys(candidates, "initial_knob_cap batch_width curve_ranking_fields curve_wait_placements initial_wait_placement pack_preload_wait_placement", "candidates")
    keys(policy["objectives"], "curve fifo43", "objectives")
    keys(policy["engineering"], "compiler scorer", "engineering")
    keys(policy["engineering"]["compiler"], "timing_search_cycles generated_word_limit spill_rewrite_limit", "compiler bounds")
    keys(policy["engineering"]["scorer"], "max_cycles torch_threads assertions_required", "scorer bounds")
    keys(policy["replay_projection"], "projection_policy", "replay projection")
    if policy["replay_projection"]["projection_policy"].get("algorithm") != "fixed-start-capacity-place-v1":
        raise ValueError("unsupported fixed replay recurrence")
    dma = policy["dma_proposal"]
    keys(dma, "classification intercommand_serialization_cycles", "DMA proposal")
    if (dma["classification"] != "compiler_policy"
            or type(dma["intercommand_serialization_cycles"]) is not int
            or not 1 <= dma["intercommand_serialization_cycles"] < 2**31):
        raise ValueError("invalid DMA proposal serialization policy")
    for name in ("supported_lookaheads", "initial_lookaheads", "virtual_scheduler_lookaheads",
                 "random_seeds", "mreg_seeds", "mreg_rank_choices", "uniform_store_ranks"):
        if not integers(pools[name]) or len(set(pools[name])) != len(pools[name]):
            raise ValueError("invalid search pool: " + name)
    if not pools["virtual_scheduler_lookaheads"] or 0 not in pools["supported_lookaheads"]:
        raise ValueError("missing native lookahead policy")
    if not set(pools["virtual_scheduler_lookaheads"]) <= set(pools["supported_lookaheads"]):
        raise ValueError("unsupported virtual scheduler lookahead")
    if not set(pools["initial_lookaheads"]) <= set(pools["supported_lookaheads"]):
        raise ValueError("unsupported initial lookahead")
    if not pools["mreg_rank_choices"] or not integers([pools["mreg_vector_length"]], 1):
        raise ValueError("invalid MREG pool")
    if not integers([pools["neighbor_shuffle_seed"]]) or not integers([pools["group_max_nodes"]], 2):
        raise ValueError("invalid neighborhood bound")
    if not integers([candidates["initial_knob_cap"], candidates["batch_width"]], 1):
        raise ValueError("invalid candidate batch bound")
    if sorted(candidates["curve_ranking_fields"]) != ["exposed_waiting", "makespan"]:
        raise ValueError("unsupported curve ranking fields")
    if candidates["curve_wait_placements"] not in (["default", "source"], ["source", "default"]):
        raise ValueError("unsupported wait siblings")
    if candidates["initial_wait_placement"] not in ("default", "source") or candidates["pack_preload_wait_placement"] != "preloaded":
        raise ValueError("unsupported placement policy")
    if policy["objectives"]["curve"] != ["cycles", "index"] or policy["objectives"]["fifo43"] != ["cycles", "index"]:
        raise ValueError("unsupported selection objective")
    for section in policy["engineering"].values():
        for name, value in section.items():
            if name == "assertions_required":
                if value is not True:
                    raise ValueError("assertions are required by the native scorer")
            elif type(value) is not int or not 0 < value < 2**31:
                raise ValueError("invalid engineering bound: " + name)
    return policy


def compiler_policy_hash(policy: dict) -> str:
    """Bind compiler choices, excluding run budgets and candidate seed pools."""
    return canonical_hash({"engineering": policy["engineering"]["compiler"],
                           "virtual_scheduler_lookaheads": policy["pools"]["virtual_scheduler_lookaheads"],
                           "supported_lookaheads": policy["pools"]["supported_lookaheads"],
                           "dma_proposal": policy["dma_proposal"],
                           "algorithms": {name: policy["algorithms"][name] for name in
                                          ("virtual_scheduler", "proposal_cost_adapter", "proposal_operand_placement")}})


POLICY = load_policy()
COMPILER_POLICY_SHA256 = compiler_policy_hash(POLICY)


def policy_receipt(effective: dict) -> dict:
    """Bind both the shipped definition and all effective API/CLI overrides."""
    definition = deepcopy(POLICY)
    result = {"schema": "atlas.effective_search_policy.v1", "definition": definition,
              "definition_sha256": canonical_hash(definition),
              "compiler_policy_sha256": COMPILER_POLICY_SHA256,
              "effective": deepcopy(effective)}
    result["sha256"] = canonical_hash(result)
    return result


def generated_header() -> str:
    compiler = POLICY["engineering"]["compiler"]
    lookaheads = ", ".join(str(value) + "u" for value in POLICY["pools"]["virtual_scheduler_lookaheads"])
    supported = ", ".join(str(value) + "u" for value in POLICY["pools"]["supported_lookaheads"])
    projection = json.dumps(POLICY["replay_projection"], sort_keys=True, separators=(",", ":"))
    dma = json.dumps(POLICY["dma_proposal"], sort_keys=True, separators=(",", ":"))
    return f'''// Generated from tools/model-search-policy.json; do not edit.
#ifndef ATLAS_SEARCH_POLICY_H
#define ATLAS_SEARCH_POLICY_H
namespace mlir::atlas::search_policy {{
inline constexpr int kTimingSearchCycles = {compiler["timing_search_cycles"]};
inline constexpr unsigned kGeneratedWordLimit = {compiler["generated_word_limit"]};
inline constexpr unsigned kSpillRewriteLimit = {compiler["spill_rewrite_limit"]};
inline constexpr unsigned kVirtualLookaheads[] = {{{lookaheads}}};
inline constexpr unsigned kSupportedLookaheads[] = {{{supported}}};
inline constexpr char kProposalOperandPlacement[] = "{POLICY["algorithms"]["proposal_operand_placement"]}";
inline constexpr char kCompilerPolicySHA256[] = "{COMPILER_POLICY_SHA256}";
inline constexpr int kDMAProposalSerializationCycles = {POLICY["dma_proposal"]["intercommand_serialization_cycles"]};
inline constexpr char kDMAProposalPolicyJson[] = R"atlas_policy({dma})atlas_policy";
inline constexpr char kDMAProposalPolicySHA256[] = "{canonical_hash(POLICY["dma_proposal"])}";
inline constexpr char kReplayProjectionJson[] = R"atlas_policy({projection})atlas_policy";
inline constexpr char kReplayProjectionSHA256[] = "{canonical_hash(POLICY["replay_projection"])}";
}} // namespace mlir::atlas::search_policy
#endif
'''


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Generate the compiler's policy constants from its versioned definition")
    parser.add_argument("--header", type=Path, required=True)
    args = parser.parse_args()
    args.header.parent.mkdir(parents=True, exist_ok=True)
    args.header.write_text(generated_header())
