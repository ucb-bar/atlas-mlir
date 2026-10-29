"""Compiler constants and search settings shipped with this checkout."""
from __future__ import annotations

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
        raise ValueError("unsupported model execution contract; RTL numerics, "
                         "deterministic initialization and fatal errors are required")
    return dict(contract)


def load_policy() -> dict:
    return json.loads(POLICY_PATH.read_text())


POLICY = load_policy()


def generated_header() -> str:
    compiler = POLICY["engineering"]["compiler"]
    lookaheads = ", ".join(str(value) + "u" for value in POLICY["pools"]["virtual_scheduler_lookaheads"])
    supported = ", ".join(str(value) + "u" for value in POLICY["pools"]["supported_lookaheads"])
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
inline constexpr int kDMAProposalSerializationCycles = {POLICY["dma_proposal"]["intercommand_serialization_cycles"]};
inline constexpr char kDMAProposalPolicyJson[] = R"atlas_policy({dma})atlas_policy";
inline constexpr char kDMAProposalPolicySHA256[] = "{canonical_hash(POLICY["dma_proposal"])}";
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
