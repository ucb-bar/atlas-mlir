"""Versioned integrity boundary for source-derived production timing facts.

The payload contains scheduling semantics and model identity, not producer
receipts or elapsed times. The serialized payload is carried explicitly so
Python and C++ verify the same bytes without relying on float printer parity.
This binds a derivation/validation record; it is not a signature or an assertion
that an arbitrary external author actually ran the recorded derivation.
"""
from __future__ import annotations

import copy
import json

from atlas_model_schema import canonical, sha256, validate_mxu_weight_timing


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
    """Select exactly the values covered by atlas.derived_facts.v1."""
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
    """Seal successful derivation output, never raw observer/test fixtures."""
    if facts.get("test_only"):
        raise ValueError("test-only observation facts cannot be sealed for production")
    validation = facts.get("validation", {})
    if (validation.get("passed") is not True or
            validation.get("operation_count") != len(facts["op_timing"]) or
            any(validation.get(key, 0) <= 0 for key in ("observed_runs", "unobserved_runs"))):
        raise ValueError("derived facts require successful operation validation")
    if set(facts["reservation_releases"]) != {block["name"] for block in facts["op_timing"]}:
        raise ValueError("derived facts require complete reservation release coverage")
    for name in VALIDATED_FIELDS:
        if name in facts and facts[name].get("validation", {}).get("passed") is not True:
            raise ValueError("derived facts require successful " + name + " validation")
    if "mxu_weight_timing" in facts and validation.get("mxu_weight_validation", {}).get("passed") is not True:
        raise ValueError("derived facts require successful MXU weight validation")
    if "mxu_weight_timing" in facts:
        validate_mxu_weight_timing(facts["mxu_weight_timing"])
    encoded = canonical(semantic_payload(facts))
    digest = sha256(encoded)
    facts["derivation_contract"] = dict(
        schema="atlas.derived_facts.v1", payload_json=encoded.decode(), payload_sha256=digest,
        derivation=dict(schema="atlas.derivation.v1", kind="source_derived", payload_sha256=digest),
        validation=dict(schema="atlas.validation.v1", passed=True, payload_sha256=digest))
    validation["payload_sha256"] = digest
    for name in VALIDATED_FIELDS:
        if name in facts:
            facts[name]["validation"]["payload_sha256"] = digest
    return facts


def verify_derived(facts):
    """Validate the producer-side boundary without running any model simulation."""
    if facts.get("test_only"):
        raise ValueError("test-only observation facts are not production derived facts")
    contract = facts.get("derivation_contract", {})
    if contract.get("schema") != "atlas.derived_facts.v1":
        raise ValueError("production model facts require atlas.derived_facts.v1; raw observations are validation/test-only")
    payload = contract.get("payload_json", "")
    digest = sha256(payload.encode())
    if digest != contract.get("payload_sha256") or json.loads(payload) != semantic_payload(facts):
        raise ValueError("derived semantic payload mismatch")
    if contract.get("derivation") != dict(schema="atlas.derivation.v1", kind="source_derived", payload_sha256=digest):
        raise ValueError("missing or mismatched source derivation record")
    if contract.get("validation") != dict(schema="atlas.validation.v1", passed=True, payload_sha256=digest):
        raise ValueError("missing or mismatched derived validation record")
    for record in (facts.get("validation", {}), *(facts[name].get("validation", {})
                                                for name in VALIDATED_FIELDS if name in facts)):
        if record.get("passed") is not True or record.get("payload_sha256") != digest:
            raise ValueError("validation is not bound to the derived payload")
    if "mxu_weight_timing" in facts:
        validate_mxu_weight_timing(facts["mxu_weight_timing"])
        if facts["validation"].get("mxu_weight_validation", {}).get("passed") is not True:
            raise ValueError("derived facts require successful MXU weight validation")
    return digest
