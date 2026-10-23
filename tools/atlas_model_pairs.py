#!/usr/bin/env python3
"""Project sealed confirmed overlap evidence without running models or producers.

The compact atlas.model_pairs.v1 artifact retains complete supported semantic
constraints. Original record/proof hashes bind omitted bulky evidence trees to
the immutable producer master; this projection creates no confirmation claim.
"""
from __future__ import annotations

import argparse
import codecs
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re

from atlas_model_schema import (BASE_KEYS, CONFIGURATION_SHA256, MASTER_SHA256, MODEL_REVISION,
                                PER_OP_SHA256, RELATION_KEYS, RELATION_VALUES, VALIDATION_SHA256,
                                QUALIFIED_CONSTRAINT_HASHES, canonical, sha256)

PAIR_FIELDS = set("""first second relation swept min_gap negative live_window first_completion_age
first_live_window_end max_in_flight domain_extension_scope previous_swept consumer_requirement
producer_consumer_requirements relation_semantics resource_semantics operand_order""".split())
CHAIN_FIELDS = set("""operations gaps relation all_pair_relations relation_scope result in_flight_count
negative witness_only no_stable_suffix_claim fixed_prefix_gaps swept_last_gap minimum_last_gap swept min_gap
accepted_gaps accepted_last_gaps minimum_accepted_gap maximum_witness_gap first_live_window_end
prefix_isolated_live_ends previous_isolated_live_window_end prefix_observed_live_window_end bounded_final_gap_end
prefix_issue_ages issue_ages initial_gaps searched_final_gap swept_gap_index swept_gap_range sweep_axis
admitted_before_edge_progress_peak peak_admission_in_flight in_flight_scope capacity_scope bound_reason
matching_requirement consumer_requirement producer_consumer_requirements relation_semantics producer_relation_summary""".split())
# These original fields describe observations/provenance, not extra admission
# constraints. Their hashes remain in evidence_summary and the full record pin.
EVIDENCE_FIELDS = set("""accepted_gaps_before_stable_suffix additional_evidence additional_input_check
additional_model_witnesses all_storage_exact audit_signature case_name classifications component_one_short
control coverage dependency earliest_pair_bounds_error earliest_pair_bounds_result elapsed_seconds error
event_union evidence evidence_directory evidence_pins evidence_sha256 exact_output first_result_preserved
fixture fixture_exporter full_membership_trace independent_audit independent_integer_oracle_checked_compute_rows
input_digest input_singletons input_trial_evidence_sha256 interpretation inventory_cell minimum_accepted_gap
model model_negative_observations model_negative_review name operand_fixture oracle oracle_description
pair_bounds pairwise_relations possible_execution_sources producer_fragment producer_fragment_sha256
publishable representative_commands rerun rerun_launcher_sha256 replacement second_input_digest
second_input_minimum_passed singleton_sha256 singletons source_pair source_registers trial_evidence_sha256
static_examples witness_edges unit_model_only unresolved_gaps validation_evidence validation_evidence_sha256 validation_receipt_sha256
validation_status weight_slots accumulators operand_order""".split())
PROOF_FIELDS = set("""confirmed rtl_sha256 source scope rtl_gap_points full_suffix_rtl_swept negative_basis
run negative_run result measured_gap negative_gap negative_kind negative_scope classification first second
operations gaps negative source_mregs source_mregs_disjoint source_mregs_equality accumulator_slots
accumulator_slots_equality weight_slots weight_slots_equality same_weight_slot maximum_live_commands
maximum_rtl_tracker_occupancy all_contributions_preserved current_model fragment fragment_sha256
full_memory_checked_bytes attempted_dispatch_gap relation result_sha256 same_accumulator""".split())
PROOF_EVIDENCE_FIELDS = {"evidence_pins", "independent_review_cache", "negative_confirmation",
                         "oracle_revision", "receipt_review", "additional_confirmations"}


POLICY = "confirmed-exact-scalar-relations-v2"
BASE_POLICY = "confirmed-exact-base-relations-v1"
# Complete constraint values accepted verbatim; a matching inner descriptor hash alone does not authorize new text.
TEXT_CONSTRAINTS = {
    "dependency": {"first destination feeds second source; first result remains in distinct storage"},
    "oracle_description": {"sequential architectural oracle plus prefix-initialized second singleton event/data union"},
    "coverage": {"direct chain witness and one-short negative; not a full gap sweep"},
    "control": {"accepted"},
    "resource_semantics": {"Accumulator and weight slot local to named MXU. Disjoint MREG footprint."},
    "consumer_requirement": {"Cross-engine namespace matching required. No implicit general capacity or source alias coverage."},
    "matching_requirement": {"Match the exact gaps and every all_pair_relations entry; top relation describes only the first adjacent pair."},
    "bound_reason": {"last gap reaches observed drain of both fixed-prefix commands plus one edge"},
    "sweep_axis": {"final adjacent gap; all earlier gaps fixed to fixed_prefix_gaps",
                   "final adjacent gap; three earlier gaps fixed"},
    "in_flight_scope": {
        "Persistent completed-tick exact sequence witness; five-path prefix capacity only, no universal capacity claim.",
        "actual persistent command count after each completed tick; witnessed count for this sequence, not a universal engine capacity",
        "persistent after completed tick at issue ages0,32,64,65,66 for this exact mixed operation/operand sequence; no universal capacity claim",
        "persistent after completed tick for this exact ordered alias chain, not a universal capacity limit",
        "persistent after completed tick for this exact three-command issue vector",
        "persistent commands after completed tick for this exact nonuniform issue vector",
        "persistent commands after completed tick for this exact sequence; no universal capacity claim"},
}


def issue_ages(gaps):
    issues = [0]
    for gap in gaps:
        issues.append(issues[-1] + gap)
    return issues


def live_pairs(query, first_free_age):
    issues = issue_ages(query["gaps"])
    for entry in query["all_pair_relations"]:
        i, j = entry["first_index"], entry["second_index"]
        first, second = query["operations"][i], query["operations"][j]
        gap = issues[j] - issues[i]
        if gap < first_free_age(first):
            yield first, second, entry["relation"], gap


class MeasuredPairs:
    def __init__(self, records, confirmed=True):
        self.index = defaultdict(list)
        for record in records:
            if confirmed and not record['confirmed']:
                continue
            self.index[(record['first'], record['second'], canonical(record['relation']))].append(record)

    def admits_pair(self, first, second, relation, gap):
        return any(record['min_gap'] <= gap <= record['swept'][1]
                   for record in self.index.get((first, second, canonical(relation)), ()))


def constraints_supported(record, qualified=True):
    if "publishable" in record and record["publishable"] is not True:
        return False
    if "unit_model_only" in record and type(record["unit_model_only"]) is not bool:
        return False
    for key in ("relation_semantics", "capacity_scope", "producer_consumer_requirements", "domain_extension_scope"):
        if key in record and (not qualified or sha256(canonical(record[key])) not in QUALIFIED_CONSTRAINT_HASHES.get(key, set())):
            return False
    for key, allowed in TEXT_CONSTRAINTS.items():
        if key in record and (not isinstance(record[key], str) or record[key] not in allowed):
            if not qualified or sha256(canonical(record[key])) not in QUALIFIED_CONSTRAINT_HASHES.get(key, set()):
                return False
    if "producer_relation_summary" in record:
        summary = record["producer_relation_summary"]
        if not isinstance(summary, dict) or not set(summary) <= BASE_KEYS:
            return False
        for key, value in summary.items():
            allowed = {"disjoint", "mixed", "0~0"} if key == "mreg_banks" else {"same", "other", "mixed"}
            if not isinstance(value, str) or value not in allowed:
                return False
    if "operand_order" in record:
        operands = record["operand_order"]
        if not isinstance(operands, dict) or set(operands) != {"first", "second"}:
            return False
        if any(not isinstance(regs, list) or any(type(reg) is not int or not 0 <= reg < 64 for reg in regs)
               for regs in operands.values()):
            return False
    return True


def safe_path(path):
    path = Path(path)
    if any(word in part.lower() for part in path.parts for word in ("hammer", "vlsi")):
        raise ValueError("prohibited path")
    if ".." in path.parts:
        raise ValueError("parent traversal is unsupported")
    for ancestor in reversed((path, *path.parents)):
        if ancestor.is_symlink():
            raise ValueError("input/output path must not contain symlinks")
    return path


def read_pinned(path, expected):
    data = safe_path(path).read_bytes()
    if not re.fullmatch("[0-9a-f]{64}", expected) or sha256(data) != expected:
        raise ValueError("artifact SHA-256 mismatch")
    return json.loads(data)


class RecordReader:
    """Read top-level JSON arrays one record at a time; hash exact input bytes."""
    def __init__(self, stream):
        self.stream = stream
        self.buffer, self.position, self.eof = "", 0, False
        self.decoder = json.JSONDecoder()
        self.utf8 = codecs.getincrementaldecoder("utf-8")()
        self.digest = hashlib.sha256()

    def more(self):
        if self.eof:
            raise ValueError("truncated master JSON")
        data = self.stream.read(65536)
        self.digest.update(data)
        self.eof = not data
        self.buffer = self.buffer[self.position:] + self.utf8.decode(data, final=self.eof)
        self.position = 0

    def peek(self):
        while True:
            while self.position < len(self.buffer) and self.buffer[self.position].isspace():
                self.position += 1
            if self.position < len(self.buffer):
                return self.buffer[self.position]
            self.more()

    def take(self, expected):
        if self.peek() != expected:
            raise ValueError("unexpected master JSON delimiter")
        self.position += 1

    def value(self):
        self.peek()
        while True:
            try:
                value, end = self.decoder.raw_decode(self.buffer, self.position)
                self.position = end
                return value
            except json.JSONDecodeError:
                self.more()

    def records(self):
        seen = set()
        self.take("{")
        while self.peek() != "}":
            key = self.value()
            if not isinstance(key, str) or key in seen:
                raise ValueError("duplicate or invalid master root key")
            seen.add(key)
            self.take(":")
            if key in {"pairs", "chains"}:
                self.take("[")
                index = 0
                while self.peek() != "]":
                    yield key, index, self.value()
                    index += 1
                    if self.peek() == ",":
                        self.take(",")
                        if self.peek() == "]":
                            raise ValueError("trailing master array comma")
                    elif self.peek() != "]":
                        raise ValueError("invalid master array")
                self.take("]")
            else:
                yield key, None, self.value()
            if self.peek() == ",":
                self.take(",")
                if self.peek() == "}":
                    raise ValueError("trailing master object comma")
            elif self.peek() != "}":
                raise ValueError("invalid master object")
        self.take("}")
        if self.buffer[self.position:].strip():
            raise ValueError("trailing master JSON content")
        while True:
            data = self.stream.read(65536)
            self.digest.update(data)
            if self.utf8.decode(data, final=not data).strip():
                raise ValueError("trailing master JSON content")
            if not data:
                break
        if not {"schema", "model", "age_convention", "pairs", "chains"} <= seen:
            raise ValueError("incomplete master root")


def integer(value, minimum=0):
    return type(value) is int and minimum <= value <= 1000000


def group(name):
    if name.startswith("vpu.") or name == "xlu.transpose":
        return "vector"
    if name.startswith("mxu0."):
        return "mxu0"
    if name.startswith("mxu1."):
        return "mxu1"
    if name.startswith("scalar_lsu.") or name == "scalar.seld":
        return "scalar_lsu"
    if name.startswith("vlsu."):
        return "lsu"
    return None


def relations(record, kind):
    if kind == "pairs":
        return [record["relation"]]
    return [entry["relation"] for entry in record["all_pair_relations"]]


def validate_relation(relation):
    if not isinstance(relation, dict) or not set(relation) <= RELATION_KEYS or "mreg_banks" not in relation:
        raise ValueError("unsupported or missing relation constraint")
    for key, value in relation.items():
        if key == "mreg_banks":
            if value != "disjoint" and (not isinstance(value, str) or not re.fullmatch(r"\d+[=~]\d+(,\d+[=~]\d+)*", value)):
                raise ValueError("invalid MREG relation")
        elif not isinstance(value, str) or value not in RELATION_VALUES[key]:
            raise ValueError("invalid relation qualifier")
    mode = relation.get("scalar_registers")
    if mode is None:
        if set(relation) & {"scalar_destination_file", "scalar_other_registers"}:
            raise ValueError("scalar roles lack a joint register mode")
    elif mode == "independent":
        if "scalar_destination_file" in relation:
            raise ValueError("independent roles cannot specify a destination alias")
    else:
        if relation.get("scalar_other_registers") != "independent":
            raise ValueError("scalar alias must constrain other registers")
        if mode == "shared_base_read_only":
            if "scalar_destination_file" in relation:
                raise ValueError("shared read-only base has no destination alias")
        elif relation.get("scalar_destination_file") not in ({"xrf", "erf"} if mode == "same_load_destination" else {"xrf"}):
            raise ValueError("scalar alias has an invalid destination file")
    if relation.get("scalar_address_alignment") == "explicit_byte_offsets":
        for side in ("first", "second"):
            if relation.get(side + "_mnemonic") not in {"vload", "vstore"} and side + "_scalar_byte_offset" not in relation:
                raise ValueError("explicit alignment lacks scalar byte offset")
    elif any(key.endswith("scalar_byte_offset") for key in relation):
        raise ValueError("scalar byte offsets require explicit alignment")


def boundary(record):
    forms = set(record) & {"live_window", "first_completion_age", "first_live_window_end"}
    if len(forms) != 1:
        raise ValueError("ambiguous pair live window")
    if "live_window" in forms:
        live = record["live_window"]
        if not isinstance(live, dict) or set(live) != {"first_completion_age", "high_includes_idle_boundary"}:
            raise ValueError("unsupported nested live window")
        if live["high_includes_idle_boundary"] is not True or not integer(live["first_completion_age"]):
            raise ValueError("invalid nested live window")
        return live["first_completion_age"] + 1
    key = next(iter(forms))
    if not integer(record[key], 1 if key == "first_live_window_end" else 0):
        raise ValueError("invalid pair live window")
    return record[key] + (key == "first_completion_age")


def validate_pair(record, blocks):
    validate_relation(record["relation"])
    swept, minimum = record["swept"], record["min_gap"]
    if not isinstance(swept, list) or len(swept) != 2 or not all(integer(v, 1) for v in swept):
        raise ValueError("invalid pair sweep")
    if swept[0] != 1 or not integer(minimum, 1) or not swept[0] <= minimum <= swept[1]:
        raise ValueError("invalid stable pair minimum")
    first_free = boundary(record)
    if first_free != blocks[record["first"]]["first_free_age"] or swept[1] < first_free:
        raise ValueError("pair/per-operation live boundary mismatch")
    negative = record["negative"]
    if not isinstance(negative, dict) or not integer(negative.get("gap")) or negative.get("gap") != minimum - 1:
        raise ValueError("missing exact one-short negative")
    if minimum == 1:
        if negative.get("result") != "inexpressible":
            raise ValueError("gap zero must be inexpressible")
    elif (negative.get("result") != "conflict" or not isinstance(negative.get("error"), dict)
          or not negative["error"].get("message")):
        raise ValueError("one-short negative must name a resource conflict")


def validate_chain(record, blocks):
    names, gaps, matrix = record["operations"], record["gaps"], record["all_pair_relations"]
    if len(names) < 3 or len(gaps) != len(names) - 1 or not all(integer(gap, 1) for gap in gaps):
        raise ValueError("invalid exact chain gaps")
    if record.get("result") != "accepted" or record.get("relation_scope") != "adjacent":
        raise ValueError("unsupported chain result or relation scope")
    expected = {(i, j) for i in range(len(names)) for j in range(i + 1, len(names))}
    actual = set()
    for entry in matrix:
        if not isinstance(entry, dict) or set(entry) != {"first_index", "second_index", "relation"}:
            raise ValueError("invalid chain relation entry")
        key = (entry["first_index"], entry["second_index"])
        if not all(type(v) is int for v in key) or key not in expected or key in actual:
            raise ValueError("duplicate or invalid chain relation indices")
        actual.add(key)
        validate_relation(entry["relation"])
    first_relation = next((entry["relation"] for entry in matrix
                           if (entry["first_index"], entry["second_index"]) == (0, 1)), None)
    if actual != expected or record["relation"] != first_relation:
        raise ValueError("chain needs the complete exact relation matrix")
    issues = issue_ages(gaps)
    peak = max(sum(start <= now < start + blocks[name]["first_free_age"] - 1
                   for start, name in zip(issues, names)) for now in issues)
    if not integer(record["in_flight_count"], 1) or peak != record["in_flight_count"]:
        raise ValueError("chain post-progress peak mismatch")
    for key in ("witness_only", "no_stable_suffix_claim"):
        if key in record and record[key] is not True:
            raise ValueError("unsupported chain witness flag")
    if record.get("witness_only") is True and record.get("no_stable_suffix_claim") is not True:
        raise ValueError("witness chain must not claim a suffix")
    if "fixed_prefix_gaps" in record and record["fixed_prefix_gaps"] != gaps[:-1]:
        raise ValueError("chain fixed prefix differs from witnessed gaps")
    for key in ("swept", "swept_last_gap", "swept_gap_range", "searched_final_gap"):
        if key in record:
            axis = record[key]
            if not isinstance(axis, list) or len(axis) != 2 or not all(integer(v, 1) for v in axis) or axis[0] > axis[1]:
                raise ValueError("invalid chain axis bounds")
    if "swept_last_gap" in record:
        lo, hi = record["swept_last_gap"]
        minimum = record.get("minimum_last_gap")
        if not integer(minimum, 1) or not lo <= minimum <= gaps[-1] <= hi:
            raise ValueError("chain last gap differs from its evidence domain")
    if "swept" in record:
        lo, hi = record["swept"]
        if len(set(gaps)) != 1 or not integer(record.get("min_gap"), 1) or not lo <= record["min_gap"] <= gaps[0] <= hi:
            raise ValueError("uniform chain axis differs from exact vector")


def select_record(record, kind, index, blocks, qualified=True):
    if not isinstance(record, dict):
        raise ValueError("invalid producer record")
    proof = record.get("verilator")
    if not isinstance(proof, dict) or proof.get("confirmed") is False or "confirmed" not in proof:
        return None, "unconfirmed"
    if proof["confirmed"] is not True:
        raise ValueError("confirmation must be literal boolean true")
    if ("evidence_tier" in record or set(proof) - PROOF_FIELDS - PROOF_EVIDENCE_FIELDS or
            any("qualified" in str(proof.get(key, "")) for key in ("scope", "classification"))):
        return None, "unsupported_protocol"
    if "relation" in proof and proof["relation"] != record.get("relation"):
        raise ValueError("proof relation differs from complete record relation")
    if "result_sha256" in proof and (not isinstance(proof["result_sha256"], str)
                                     or not re.fullmatch("[0-9a-f]{64}", proof["result_sha256"])):
        raise ValueError("invalid original result SHA-256")
    if "same_accumulator" in proof and (type(proof["same_accumulator"]) is not bool or
            (record.get("relation", {}).get("accumulator") == "same") != proof["same_accumulator"]):
        raise ValueError("proof accumulator alias differs from record")
    names = [record.get("first"), record.get("second")] if kind == "pairs" else record.get("operations")
    if (not isinstance(names, list) or not names
            or any(not isinstance(n, str) or n not in blocks or group(n) is None for n in names)):
        return None, "unsupported_operation"
    if not qualified and any(group(name) == "scalar_lsu" for name in names):
        return None, "unsupported_operation"
    if len({group(name) for name in names}) != 1:
        return None, "cross_group"
    if any(not isinstance(rel, dict) or not set(rel) <= (RELATION_KEYS if qualified else BASE_KEYS)
           for rel in relations(record, kind)):
        return None, "unsupported_relation"
    if group(names[0]) == "scalar_lsu":
        required = {"first_mnemonic", "second_mnemonic", "scalar_address_alignment", "scalar_registers", "vmem_bank"}
        if any(not required <= set(rel) for rel in relations(record, kind)):
            return None, "unsupported_relation"
    if not constraints_supported(record, qualified):
        return None, "unsupported_constraint_value"
    fields = PAIR_FIELDS if kind == "pairs" else CHAIN_FIELDS
    if set(record) - fields - EVIDENCE_FIELDS - {"verilator", "adjacent_relations"}:
        return None, "unsupported_record_field"
    (validate_pair if kind == "pairs" else validate_chain)(record, blocks)
    kept = {key: record[key] for key in fields if key in record}
    kept["verilator"] = {key: value for key, value in proof.items() if key in PROOF_FIELDS}
    kept["source_record"] = dict(json_pointer=f"/{kind}/{index}", sha256=sha256(canonical(record)),
                                 proof_sha256=sha256(canonical(proof)))
    omitted = {key: value for key, value in record.items() if key not in fields and key != "verilator"}
    kept["evidence_summary"] = dict(notes={key: value for key, value in omitted.items()
                                           if isinstance(value, (str, int, bool, type(None)))},
                                     model=omitted.get("model"),
                                     field_sha256={key: sha256(canonical(value)) for key, value in omitted.items()},
                                     field_counts={key: len(value) for key, value in omitted.items()
                                                   if isinstance(value, (dict, list))},
                                     provenance={key: value for key, value in omitted.items()
                                                 if isinstance(value, str) and (key.endswith("sha256")
                                                     or key.endswith("directory")
                                                     or key in {"producer_fragment", "evidence", "validation_evidence"})})
    return kept, None


def covered_chain(record, pair_index, blocks):
    for first, second, relation, gap in live_pairs(record, lambda name: blocks[name]["first_free_age"]):
        key = (first, second, canonical(relation))
        if not any(pair["min_gap"] <= gap <= pair["swept"][1] for pair in pair_index.get(key, [])):
            return False
    return True


def project(master, validation, per_op, output, *, policy=POLICY, master_sha256=None, validation_sha256=None):
    master_pin = master_sha256 or MASTER_SHA256
    validation_pin = validation_sha256 or VALIDATION_SHA256
    if policy not in {POLICY, BASE_POLICY}:
        raise ValueError("unsupported projection policy")
    qualified = policy == POLICY
    master, validation, per_op, output = map(safe_path, (master, validation, per_op, output))
    source_sha = sha256(Path(__file__).read_bytes())
    if output.exists():
        raise ValueError("output directory must be new")
    validation_value = read_pinned(validation, validation_pin)
    if validation_value.get("passed") is not True or validation_value.get("facts_sha256") != master_pin:
        raise ValueError("validation does not bind the selected master")
    operations = read_pinned(per_op, PER_OP_SHA256)
    if operations.get("schema") != "atlas.model_timing.v1" or operations.get("validation", {}).get("passed") is not True:
        raise ValueError("requires gated per-operation facts")
    blocks = {block["name"]: block for block in operations["op_timing"]}
    if len(blocks) != 49:
        raise ValueError("requires all 49 gated operations")
    result = dict(schema="atlas.model_pairs.v1", pairs=[], chains=[])
    seen, excluded = Counter(), Counter()
    with master.open("rb") as stream:
        reader = RecordReader(stream)
        for key, index, value in reader.records():
            if key not in {"pairs", "chains"}:
                if key in {"schema", "model", "age_convention", "consumer_rules", "limitations",
                           "per_operation_reference", "provenance"}:
                    result[key] = value
                continue
            seen[key] += 1
            selected, reason = select_record(value, key, index, blocks, qualified)
            if selected is None:
                excluded[key + ":" + reason] += 1
            else:
                result[key].append(selected)
        if reader.digest.hexdigest() != master_pin:
            raise ValueError("master SHA-256 mismatch")
    pair_index = MeasuredPairs(result["pairs"], False).index
    chains = result["chains"]
    result["chains"] = [chain for chain in chains if covered_chain(chain, pair_index, blocks)]
    if len(chains) != len(result["chains"]):
        excluded["chains:missing_live_pair"] += len(chains) - len(result["chains"])
    model = result.get("model", {})
    if result["schema"] != "atlas.model_pairs.v1" or result.get("age_convention") != "issue_edge_zero":
        raise ValueError("unsupported master schema or ages")
    if model.get("revision") != MODEL_REVISION or model.get("configuration_sha256") != CONFIGURATION_SHA256:
        raise ValueError("unsupported model revision/configuration")
    if (model.get("revision") != operations["model"]["revision"]
            or model.get("configuration") != operations["model"]["configuration"]):
        raise ValueError("pair and per-operation effective model identity differ")
    if sha256(canonical(model["configuration"])) != CONFIGURATION_SHA256:
        raise ValueError("effective model configuration hash mismatch")
    if any(seen[key] != validation_value.get(key) for key in ("pairs", "chains")):
        raise ValueError("validation counts do not match master")
    result["projection"] = dict(version=2 if qualified else 1, policy=policy, master_sha256=master_pin,
                                master_path=str(master.absolute()),
                                validation_sha256=validation_pin, validation_path=str(validation.absolute()),
                                per_op_sha256=PER_OP_SHA256, projector_sha256=source_sha,
                                counts=dict(source_pairs=seen["pairs"], source_chains=seen["chains"],
                                            pairs=len(result["pairs"]), chains=len(result["chains"])),
                                excluded_reasons=dict(sorted(excluded.items())))
    output.mkdir(parents=True)
    artifact = output / "model-pairs.json"
    artifact.write_bytes(canonical(result) + b"\n")
    receipt = dict(schema="atlas.model_pairs_projection_receipt.v1", passed=True,
                   facts_sha256=sha256(artifact.read_bytes()), projection=result["projection"],
                   generated_utc=datetime.now(timezone.utc).isoformat(), bytes=artifact.stat().st_size,
                   scope="Confirmed sealed producer records, exact base/scalar relations and existing serialization groups only; no new model/RTL execution.")
    (output / "receipt.json").write_text(json.dumps(receipt, sort_keys=True, indent=2) + "\n")
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("master", "validation", "per-op", "output"):
        parser.add_argument("--" + key, type=Path, required=True)
    parser.add_argument("--policy", choices=(POLICY, BASE_POLICY), default=POLICY)
    parser.add_argument("--master-sha256")
    parser.add_argument("--validation-sha256")
    args = parser.parse_args()
    print(json.dumps(project(args.master, args.validation, args.per_op, args.output, policy=args.policy,
                             master_sha256=args.master_sha256, validation_sha256=args.validation_sha256), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
