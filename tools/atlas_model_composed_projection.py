#!/usr/bin/env python3
"""Project validated footprint rules and exact new pairs without modifying measured evidence."""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path

from atlas_model_pairs import (MeasuredPairs, canonical, live_pairs as chain_live_pairs, safe_path,
                               PER_OP_SHA256, MODEL_REVISION, CONFIGURATION_SHA256)
from atlas_model_composition import RuleSet, relation_class, rule_source_sha256
from atlas_model_chain_composition import compose


def sha(path):
    return hashlib.sha256(safe_path(path).read_bytes()).hexdigest()


def load(path):
    return json.loads(safe_path(path).read_text())


def family(engine, relation):
    return engine + ">" + engine + ":" + canonical(relation_class(relation)).decode()


def complete_relations(query):
    result = dict(query, all_pair_relations=[])
    for i, a in enumerate(query["operands"]):
        for j, b in enumerate(query["operands"][i+1:], i+1):
            banks = [f"{x}{'=' if r == s else '~'}{y}" for x, r in enumerate(a["registers"])
                     for y, s in enumerate(b["registers"]) if r % 32 == s % 32]
            rel = dict(mreg_banks=",".join(banks) or "disjoint")
            for key in ("accumulator", "weight_slot"):
                if key in a and key in b and query["operations"][i].split(".")[0] == query["operations"][j].split(".")[0]:
                    rel[key] = "same" if a[key] == b[key] else "other"
            result["all_pair_relations"].append(dict(first_index=i, second_index=j, relation=rel))
    if "all_pair_relations" in query:
        supplied = sorted(query["all_pair_relations"], key=lambda entry: (entry["first_index"], entry["second_index"]))
        if supplied != result["all_pair_relations"]:
            raise ValueError("query relation matrix differs from its actual operands")
    return result


def live_pairs(query, rules):
    yield from chain_live_pairs(query, rules.first_free_age)


def project(per_op, base_pairs, pair_audit, chain_audit, queries, holdouts, output, *, per_op_sha256=PER_OP_SHA256):
    operations, pairs, pa, ca, qs, hs = map(load, (per_op, base_pairs, pair_audit, chain_audit, queries, holdouts))
    if sha(per_op) != per_op_sha256 or operations["model"]["revision"] != MODEL_REVISION or \
            operations["model"]["configuration_sha256"] != CONFIGURATION_SHA256:
        raise ValueError("unsupported per-operation evidence")
    if pa["master_sha256"] != ca["master_sha256"] or \
            pairs["projection"]["master_sha256"] != pa["master_sha256"] or \
            pa["per_operation_sha256"] != sha(per_op) or \
            ca["per_op_sha256"] != sha(per_op) or pairs["projection"]["per_op_sha256"] != sha(per_op):
        raise ValueError("replay does not bind selected observations")
    if pa.get("rule_source_sha256") != rule_source_sha256() or ca.get("rule_source_sha256") != rule_source_sha256():
        raise ValueError("replay does not bind the current derivation rules")
    if hs.get("passed") is not True or not hs.get("results"):
        raise ValueError("successful holdouts are required")
    rules = RuleSet(operations)
    coverage, domains, counts = set(), defaultdict(set), defaultdict(Counter)
    for item in hs["results"]:
        result_path = safe_path(item["path"])
        if sha(result_path) != item["sha256"]:
            raise ValueError("holdout receipt changed")
        result = load(result_path)
        model = result.get("current_model", {})
        identity = model.get("model_identity", {})
        if result.get("confirmed") is not True or result.get("rtl_confirmed") is not True or \
                model.get("status") != "PASS" or model.get("confirmed") is not True or \
                model.get("effective_configuration_sha256") != CONFIGURATION_SHA256 or \
                model.get("observed_ignore_runtime_errors") is not False or \
                identity.get("revision") != MODEL_REVISION or \
                identity.get("source_sha256") != operations["model"]["source_sha256"] or \
                identity.get("numerical_data_sha256") != operations["model"]["numerical_data_sha256"] or \
                model.get("engines_drained") is not True or model.get("completion_marker") != 1:
            raise ValueError("holdout did not pass exact model and RTL checks")
        query = complete_relations(result["query"])
        if result.get("measured_gaps") != query["gaps"] or model.get("measured_gaps") != query["gaps"] or \
                type(result.get("full_memory_checked_bytes")) is not int or result["full_memory_checked_bytes"] <= 0:
            raise ValueError("holdout query differs from checked RTL/model issue gaps or memory coverage")
        engine = query["operations"][0].split(".")[0]
        if engine not in ("mxu0", "mxu1", "vpu") or any(op.split(".")[0] != engine for op in query["operations"]):
            raise ValueError("holdout crosses supported engine groups")
        if not compose(query, query["gaps"], rules, rules)["admitted"]:
            raise ValueError("holdout is outside the derived rule")
        domains[engine].update(query["operations"])
        counts[engine]["holdout_chains" if len(query["operations"]) > 2 else "holdout_pairs"] += 1
        for first, second, rel, gap in live_pairs(query, rules):
            coverage.add(family(engine, rel))
    engines = {}
    for engine in domains:
        entries = [ca["counts"].get("derived/"+tier+"/"+engine, {}) for tier in ("confirmed", "model_only")]
        total = sum(entry.get("positives", 0) for entry in entries)
        accepted = sum(entry.get("accepted_positives", 0) for entry in entries)
        negatives = sum(entry.get("negative_descriptors", 0) for entry in entries)
        rejected = sum(entry.get("rejected_negatives", 0) for entry in entries)
        if total != accepted or negatives != rejected or not total or not negatives or not entries[0].get("positives"):
            raise ValueError("engine did not achieve complete positive/negative replay agreement")
        engines[engine] = dict(operations=sorted(domains[engine]), chains=total, accepted_chains=accepted,
                               confirmed_chains=entries[0]["positives"], negative_points=negatives,
                               rejected_negatives=rejected, holdout_chains=counts[engine]["holdout_chains"],
                               holdout_pairs=counts[engine]["holdout_pairs"], holdout_failures=0)
    classes, allowed = [], set()
    for key in sorted(coverage):
        audit = pa["relation_classes"].get(key, {})
        engine, raw_relation = key.split(">", 1)[0], key.split(":", 1)[1]
        rel = json.loads(raw_relation)
        if not audit.get("eligible") or set(rel) - {"mreg_banks", "accumulator", "weight_slot"}:
            continue
        allowed.add(key)
        classes.append(dict(engine=engine, relation=rel, pairs=audit["pairs"], confirmed=audit["confirmed"],
                            equal=audit["equal"], negative_points=audit["negative_rejected"], negative_violations=0))
    base_index = MeasuredPairs(pairs["pairs"], False)
    derived_pairs = {}
    def covered(first, second, rel, gap):
        key = (first, second, canonical(rel))
        return base_index.admits_pair(first, second, rel, gap) or \
            (key in derived_pairs and derived_pairs[key]["minimum"] <= gap <= derived_pairs[key]["high"])
    eligible = []
    decisions = []
    for query in qs:
        query = complete_relations(query)
        names = query["operations"]
        engine = names[0].split(".")[0]
        reason = None
        if engine not in engines or not set(names) <= domains[engine]:
            reason = "outside_holdout_operation_domain"
        elif any(family(engine, rel) not in allowed for a, b, rel, gap in live_pairs(query, rules)):
            reason = "outside_witnessed_and_heldout_relation_class"
        elif not compose(query, query["gaps"], rules, rules)["admitted"]:
            reason = "derived_pair_or_capacity_conflict"
        if reason:
            decisions.append(dict(query=query, admitted=False, reason=reason))
            continue
        for first, second, rel, gap in live_pairs(query, rules):
            if not covered(first, second, rel, gap):
                minimum = rules.minimum_gap(first, second, rel)
                derived_pairs[(first, second, canonical(rel))] = dict(first=first, second=second, relation=rel,
                    minimum=minimum, high=rules.first_free_age(first))
        eligible.append(query)
        decisions.append(dict(query=query, admitted=True))
    class PairCoverage:
        def admits_pair(self, first, second, relation, gap):
            return covered(first, second, relation, gap)
    chains = {}
    for query in eligible:
        if len(query["operations"]) < 3:
            continue
        verdict = compose(query, query["gaps"], rules, PairCoverage())
        if not verdict["admitted"]:
            raise ValueError("projected chain lacks a live admitted pair")
        record = {key: query[key] for key in ("operations", "gaps", "all_pair_relations")}
        record["in_flight_count"] = verdict["peak_post_progress"]
        chains[canonical(record)] = record
    here = Path(__file__).absolute().parent
    deriver = hashlib.sha256(b"".join((here/name).read_bytes() for name in
        ("atlas_model_composition.py", "atlas_model_chain_composition.py", "atlas_model_composed_projection.py"))).hexdigest()
    artifact = dict(schema="atlas.model_composed.v1", rule="row-footprints-and-structural-capacity-v1",
        age_convention="issue_edge_zero", provenance=dict(per_op_sha256=sha(per_op), base_pairs_sha256=sha(base_pairs),
            master_sha256=pa["master_sha256"], deriver_sha256=deriver, pair_replay_sha256=sha(pair_audit),
            chain_replay_sha256=sha(chain_audit), holdouts_sha256=sha(holdouts), revision=MODEL_REVISION,
            configuration_sha256=CONFIGURATION_SHA256), pair_classes=classes, engines=engines,
        pairs=list(derived_pairs.values()), chains=list(chains.values()))
    output = safe_path(output)
    output.mkdir(parents=True, exist_ok=False)
    path = output/"model-composed.json"
    path.write_bytes(canonical(artifact)+b"\n")
    receipt = dict(schema="atlas.model_composed_projection_receipt.v1", passed=True, sha256=sha(path),
                   queries_sha256=sha(queries), rule_source_sha256=rule_source_sha256(),
                   pairs=len(derived_pairs), chains=len(chains), classes=len(classes), engines=engines,
                   decisions=Counter("admitted" if row["admitted"] else row["reason"] for row in decisions),
                   provenance=artifact["provenance"])
    (output/"receipt.json").write_text(json.dumps(receipt, indent=2)+"\n")
    (output/"query-decisions.json").write_text(json.dumps(decisions, indent=2)+"\n")
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("per-op", "base-pairs", "pair-audit", "chain-audit", "queries", "holdouts", "output"):
        parser.add_argument("--"+key, type=Path, required=True)
    parser.add_argument("--per-op-sha256", default=PER_OP_SHA256, help="explicit digest of selected derived facts with releases")
    args = parser.parse_args()
    print(json.dumps(project(args.per_op, args.base_pairs, args.pair_audit, args.chain_audit,
                             args.queries, args.holdouts, args.output, per_op_sha256=args.per_op_sha256), sort_keys=True))


if __name__ == "__main__":
    main()
