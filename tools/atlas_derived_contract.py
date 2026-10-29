"""Canonical compiler timing payload shared by Python and C++."""
from __future__ import annotations

import copy
import json

from atlas_model_schema import canonical, sha256


MODEL_FIELDS = ("revision", "revision_kind", "source_sha256", "numerical_data_sha256",
                "configuration", "configuration_sha256")
PAYLOAD_FIELDS = ("geometry", "age_convention", "op_timing", "reservation_releases",
                  "vocabulary_sha256", "architecture", "operand_vocabulary", "isa_metadata")
EXTENSIONS = ("frontend_timing", "scratchpad_contract", "overlap_rules_sha256", "mxu_weight_timing", "replay_timing", "proposal_costs")
VALIDATED_FIELDS = ("dma_timing", "engine_admission", "dma_admission")


def overlap_semantics(artifact):
    """Bind every overlap decision input without receipt or producer identities."""
    contract = copy.deepcopy(artifact["derivation"]["source_contract"])
    contract.pop("producer_sources", None)
    return dict(schema=artifact["schema"], age_convention=artifact["age_convention"],
                contract=artifact["derivation"]["contract"], source_contract=contract,
                pairs=copy.deepcopy(artifact["pairs"]),
                composition=copy.deepcopy(artifact["composition"]))


def semantic_payload(facts):
    """Select the compiler timing payload."""
    result = {name: copy.deepcopy(facts[name]) for name in PAYLOAD_FIELDS}
    result["model"] = {name: copy.deepcopy(facts["model"][name]) for name in MODEL_FIELDS}
    for name in EXTENSIONS:
        if name in facts:
            result[name] = copy.deepcopy(facts[name])
    for name in VALIDATED_FIELDS:
        if name in facts:
            result[name] = {key: copy.deepcopy(value) for key, value in facts[name].items()
                            if key not in {"validation", "producer_sources"}}
    return result


def seal_derived(facts):
    """Record canonical timing bytes for the compiler's selected facts."""
    if facts.get("test_only"):
        raise ValueError("test-only observation facts cannot be sealed for production")
    payload = canonical(semantic_payload(facts))
    facts["derivation_contract"] = dict(payload_json=payload.decode(), payload_sha256=sha256(payload))
    return facts


def verify_derived(facts):
    """Check the selected timing payload without native execution."""
    if facts.get("test_only"):
        raise ValueError("test-only observation facts are not production derived facts")
    contract = facts["derivation_contract"]
    payload = contract["payload_json"]
    digest = sha256(payload.encode())
    if digest != contract["payload_sha256"] or json.loads(payload) != semantic_payload(facts):
        raise ValueError("derived semantic payload mismatch")
    return digest
