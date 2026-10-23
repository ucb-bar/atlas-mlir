#!/usr/bin/env python3
"""Derive suffix-safe pair gaps and audit them against a read-only master.

This is an offline validator, not an admission source for the compiler. A class
is eligible only when every minimum agrees and every recorded negative rejects.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import dataclass
from functools import lru_cache
import hashlib
import json
from pathlib import Path
import re

from atlas_model_pairs import RecordReader, safe_path, validate_relation


def rule_source_sha256():
    here = Path(__file__).absolute().parent
    names = ("atlas_model_pairs.py", "atlas_model_composition.py", "atlas_model_chain_composition.py")
    return hashlib.sha256(b"".join(safe_path(here/name).read_bytes() for name in names)).hexdigest()


def relation_key(relation):
    return tuple(sorted(relation.items()))


def relation_class(relation):
    result = dict(relation)
    banks = result["mreg_banks"]
    if banks != "disjoint":
        result["mreg_banks"] = ("mixed_alias" if "=" in banks and "~" in banks else
                                "register_alias" if "=" in banks else "bank_alias")
    return result


def engine(name):
    return name.split(".", 1)[0]


@dataclass(frozen=True)
class Register:
    write: bool
    requests: tuple[tuple[int, int], ...]
    release: int


class RuleSet:
    # Pinned VectorFSM shared lane-box groups, hardware/vpu.py:_GROUPS.
    logic_groups = ({"add", "sub", "rsum"}, {"exp", "exp2"}, {"sin", "cos"}, {"square", "cube"},
                    {"pairmax", "cmax"}, {"pairmin", "cmin"}, {"vliAll", "vliRow", "vliCol", "vliOne"})
    double = {"add", "sub", "mul", "pairmax", "pairmin", "rsum", "rmax", "rmin"}

    def __init__(self, per_op):
        if per_op.get("schema") != "atlas.model_timing.v1" or per_op.get("age_convention") != "issue_edge_zero":
            raise ValueError("unsupported isolated footprint schema")
        self.blocks = {block["name"]: block for block in per_op["op_timing"] if block.get("resolved")}
        self.model = per_op["model"]
        self.releases = per_op.get("reservation_releases", {})
        if not isinstance(self.releases, dict):
            raise ValueError("invalid isolated reservation releases")

    def first_free_age(self, name):
        return self.blocks[name]["first_free_age"]

    def capacity(self, unit):
        """Tracker depths, never a universal bound on all live MXU/LSU commands."""
        return {"vpu": 2, "xlu": 1, "mxu0": 3, "mxu1": 2, "vload": 1, "vstore": 1, "scalar_load": 1}.get(unit)

    @staticmethod
    def kind(name):
        part = name.split(".", 1)[1]
        if part.startswith("matmul"):
            return "compute"
        if part == "push_weight":
            return "weight"
        if part.startswith("push_acc"):
            return "push_bf16" if part.endswith("bf16") else "push_fp8"
        if part.startswith("pop_acc"):
            return "pop_bf16" if part.endswith("bf16") else "pop_fp8"
        return part

    def capacity_conflict(self, operations, issues):
        assigned = {}
        for j, (name, issue) in enumerate(zip(operations, issues)):
            unit = engine(name)
            if unit == "vpu":
                live = sum(engine(old) == unit and issue - issues[i] < self.first_free_age(old) - 1
                           for i, old in enumerate(operations[:j]))
                if live >= 2:
                    return "VPU two issue slots"
            elif unit in {"mxu0", "mxu1"}:
                kind = self.kind(name)
                if kind == "compute":
                    live = sum(engine(old) == unit and self.kind(old) == "compute" and
                               issue - issues[i] <= self.first_free_age(old) - 1
                               for i, old in enumerate(operations[:j]))
                    if live >= self.capacity(unit):
                        return unit + " compute tracker depth"
                busy = {port for i, old in enumerate(operations[:j]) if engine(old) == unit and
                        issue - issues[i] < 32 for port in assigned[i]}
                if kind == "compute":
                    ports = (0,)
                elif kind == "push_bf16":
                    ports = (0, 1)
                elif kind.startswith("pop"):
                    ports = (2, 3) if kind == "pop_bf16" else (2,)
                else:
                    ports = (1,) if 1 not in busy else (0,)
                if any(port in busy for port in ports):
                    return unit + " sequencer port capacity"
                assigned[j] = ports
        return None

    def ages(self, name, key):
        event = self.blocks[name]["events"][key]
        return tuple(range(event["first_age"], event["last_age"] + 1, event["step"]))

    @lru_cache(maxsize=None)
    def registers(self, name):
        events = self.blocks[name]["events"]
        unit, part = name.split(".", 1)
        reads, writes = [], []

        def add(key, write, count=1):
            ages = self.ages(name, key)
            direction = "mreg_write" if write else "mreg_read"
            operation = self.releases.get(name)
            release = operation.get(direction) if isinstance(operation, dict) else None
            if (not isinstance(release, dict) or type(release.get("first_age")) is not int or
                    release["first_age"] != 0 or
                    type(release.get("release_age")) is not int or
                    type(release.get("available_age")) is not int or
                    release["available_age"] != release["release_age"] + 1 or
                    not max(ages) < release["available_age"] <= self.first_free_age(name)):
                raise ValueError(f"missing or invalid {direction} reservation release for {name}")
            for bank in range(count):
                requests = tuple((age, row % 32) for row, age in enumerate(ages) if (row // 32) % count == bank)
                (writes if write else reads).append(Register(write, requests, release["available_age"]))

        if unit == "vpu":
            row_reduce = part in {"rsum", "rmax", "rmin"}
            if row_reduce:
                for key in ("read0", "read1", "write0", "write1"):
                    add(key, key.startswith("write"))
            else:
                for key in ("read0", "read1"):
                    if key in events:
                        add(key, False, 1 if part == "fp8unpack" else 2)
                add("write0", True, 1 if part in {"fp8pack", "vliCol", "vliOne"} else 2)
        elif unit in {"mxu0", "mxu1"}:
            for key in ("mreg_read0", "mreg_read1", "mreg_write0", "mreg_write1"):
                if key in events:
                    add(key, key.startswith("mreg_write"))
        elif unit == "xlu":
            add("read", False)
            add("write", True)
        elif unit == "vlsu":
            add("mreg_write" if part == "vload" else "mreg_read", part == "vload")
        elif unit != "scalar_lsu":
            raise ValueError("unmodelled engine: " + unit)
        return tuple(reads + writes)

    def _structural_bound(self, first, second, relation):
        a, b = engine(first), engine(second)
        x, y = first.split(".", 1)[1], second.split(".", 1)[1]
        bound = 1
        if a == b == "vpu":
            shared = x == y or any(x in group and y in group for group in self.logic_groups)
            if shared or x in self.double or y in self.double:
                bound = self.first_free_age(first) - 1
        if a == b == "xlu" or first == second and a == "vlsu":
            bound = self.first_free_age(first)
        if a == b == "scalar_lsu" and x != "store" and y != "store":
            bound = 3
        if a != b or a not in {"mxu0", "mxu1"}:
            return bound
        ka, kb = self.kind(first), self.kind(second)
        same_acc, same_weight = relation.get("accumulator") == "same", relation.get("weight_slot") == "same"
        compute_first = self.blocks[first]["events"].get("acc_write", {}).get("first_age")
        if ka == "compute" and kb != "weight" and same_acc:
            bound = max(bound, compute_first + 1)
        reads_a = ka in {"compute", "weight", "push_fp8", "push_bf16"}
        reads_b = kb in {"compute", "weight", "push_fp8", "push_bf16"}
        if (ka.startswith("pop") and kb.startswith("pop") or
            reads_a and reads_b and (ka == "push_bf16" or kb == "push_bf16" or ka == kb == "compute") or
            ka == kb == "weight" or ka.startswith("push") and kb.startswith("push")):
            bound = max(bound, 32)
        if a == "mxu0" and ka == "compute" and kb == "weight" and same_weight:
            bound = max(bound, 63)
        if a == "mxu1":
            if same_weight and ((ka == "weight" and kb in {"weight", "compute"}) or ka == "compute" and kb == "weight"):
                bound = max(bound, 32)
            if same_acc and ka.startswith("push") and (kb == "compute" or kb == "push_fp8"):
                bound = max(bound, 33)
        if same_acc and ka.startswith("push") and (kb.startswith("pop") or y == "matmul_acc"):
            bound = max(bound, 2)
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

    @lru_cache(maxsize=None)
    def bounds(self, first, second, relation_items):
        relation = dict(relation_items)
        if first not in self.blocks or second not in self.blocks:
            raise ValueError("operation lacks isolated footprint")
        unknown = set(relation) - {"mreg_banks", "accumulator", "weight_slot", "vmem_bank", "vmem_window",
            "vmem_word_window", "first_mnemonic", "second_mnemonic", "scalar_registers", "scalar_destination_file",
            "scalar_other_registers", "scalar_address_alignment", "first_scalar_byte_offset", "second_scalar_byte_offset"}
        if unknown:
            raise ValueError("unsupported relation keys: " + ",".join(sorted(unknown)))
        bounds = dict(structural_issue=self._structural_bound(first, second, relation),
                      logical_reservation=1, bank_or_row=1, vmem_bank=1)
        ra, rb = self.registers(first), self.registers(second)
        banks = relation.get("mreg_banks")
        if banks is None:
            raise ValueError("missing complete MREG relation")
        if banks != "disjoint":
            for term in banks.split(","):
                match = re.fullmatch(r"(\d+)([=~])(\d+)", term)
                if not match:
                    raise ValueError("unsupported MREG relation term")
                i, sign, j = int(match[1]), match[2], int(match[3])
                if i >= len(ra) or j >= len(rb):
                    raise ValueError("MREG relation operand outside isolated footprint")
                left, right = ra[i], rb[j]
                if sign == "=" and (left.write or right.write and second != "vlsu.vload"):
                    bounds["logical_reservation"] = max(bounds["logical_reservation"], left.release)
                for age, row in left.requests:
                    for later, later_row in right.requests:
                        if left.write == right.write:
                            coalesced = (not left.write and engine(first) == engine(second) == "vpu"
                                         and sign == "=" and row == later_row)
                            collision = not coalesced
                        else:
                            collision = sign == "=" and row == later_row
                        if collision and age >= later:
                            bounds["bank_or_row"] = max(bounds["bank_or_row"], age - later + 1)
        if relation.get("vmem_bank") == "same":
            ea, eb = self.blocks[first]["events"], self.blocks[second]["events"]
            aa = [age for key in ("vmem_read", "vmem_write") if key in ea for age in self.ages(first, key)]
            bb = [age for key in ("vmem_read", "vmem_write") if key in eb for age in self.ages(second, key)]
            if aa and bb:
                bounds["vmem_bank"] = max(bounds["vmem_bank"], max(aa) - min(bb) + 1)
        return bounds

    def minimum_gap(self, first, second, relation):
        """First gap of the suffix with no derived collision at any larger gap."""
        derived_relation = dict(relation)
        address_keys = {"vmem_address_delta_bytes", "first_vmem_bank_byte_offset", "second_vmem_bank_byte_offset"}
        if address_keys & relation.keys():
            if not address_keys <= relation.keys() or engine(first) != "vlsu" or engine(second) != "vlsu":
                raise ValueError("incomplete or unsupported vector-memory address relation")
            if any(not isinstance(relation[key], str) or not re.fullmatch(r"-?\d+", relation[key]) for key in address_keys):
                raise ValueError("invalid vector-memory address qualifier")
            delta, a, b = (int(relation[key]) for key in
                           ("vmem_address_delta_bytes", "first_vmem_bank_byte_offset", "second_vmem_bank_byte_offset"))
            bank_delta = delta - (b-a)
            if any(offset < 0 or offset + 1024 > 262144 or offset % 1024 for offset in (a, b)) or \
                    bank_delta % 262144 or abs(bank_delta) > 5 * 262144 or \
                    relation.get("vmem_bank") != ("same" if bank_delta == 0 else "other"):
                raise ValueError("inconsistent vector-memory bank addresses")
            if any(key in relation and relation[key] != ("same" if delta == 0 else "disjoint")
                   for key in ("vmem_window", "vmem_word_window")):
                raise ValueError("inconsistent vector-memory windows")
            for key in address_keys:
                del derived_relation[key]
        validate_relation(derived_relation)
        return max(self.bounds(first, second, relation_key(derived_relation)).values())

    def admits_pair(self, first, second, relation, gap):
        return type(gap) is int and gap >= self.minimum_gap(first, second, relation)


def negative_points(record):
    """Retain every explicit model or RTL rejection, including successful RTL/model-stricter controls."""
    seen = set()

    def walk(value, path):
        if isinstance(value, dict):
            if type(value.get("gap")) is int:
                result = str(value.get("result", value.get("classification", value.get("kind", ""))))
                if ("conflict" in result or "exception" in result or "inexpressible" in result or
                    result == "specific_assertion_prefix" or value.get("model_status") == "EXCEPTION" or
                    value.get("passed") is False or value.get("accepted") is False):
                    key = (path, value["gap"])
                    if key not in seen:
                        seen.add(key)
                        yield path, value["gap"], result
            for key, child in value.items():
                yield from walk(child, path + "/" + key)
        elif isinstance(value, list):
            for i, child in enumerate(value):
                yield from walk(child, path + "/" + str(i))
    yield from walk(record, "")


def replay(master, per_op, output, per_op_sha256):
    rules = RuleSet(per_op)
    classes, families, totals = defaultdict(Counter), defaultdict(Counter), Counter()
    examples, details = [], []
    digest = None
    with safe_path(master).open("rb") as stream:
        reader = RecordReader(stream)
        for section, index, record in reader.records():
            if section != "pairs":
                continue
            first, second, relation = record["first"], record["second"], record["relation"]
            key = engine(first) + ">" + engine(second) + ":" + json.dumps(relation, sort_keys=True, separators=(",", ":"))
            confirmed = record.get("verilator", {}).get("confirmed") is True
            totals["pairs"] += 1
            totals["confirmed_pairs"] += confirmed
            row = {"index": index, "first": first, "second": second, "relation": relation,
                   "confirmed": confirmed, "measured": record.get("min_gap")}
            try:
                validate_relation(relation)
            except ValueError as error:
                row["projection_incompatible"] = str(error)
                classes[key]["projection_incompatible"] += 1
            try:
                row["bounds"] = rules.bounds(first, second, relation_key(relation))
                gap = max(row["bounds"].values())
                row["derived"] = gap
                comparison = "equal" if gap == record["min_gap"] else "stricter" if gap > record["min_gap"] else "looser"
            except (KeyError, ValueError) as error:
                gap, comparison = None, "unmodelled"
                row["reason"] = str(error)
            row["classification"] = comparison
            classes[key]["pairs"] += 1
            classes[key]["confirmed"] += confirmed
            classes[key][comparison] += 1
            totals[comparison] += 1
            if confirmed:
                classes[key]["confirmed_" + comparison] += 1
                totals["confirmed_" + comparison] += 1
            negatives = []
            unique_negatives = set()
            for path, negative, result in negative_points(record):
                verdict = "unmodelled" if gap is None else "unsound" if negative >= gap else "rejected"
                negatives.append({"path": path, "gap": negative, "result": result, "verdict": verdict})
                totals["negative_points"] += 1
                totals["negative_" + verdict] += 1
                classes[key]["negative_" + verdict] += 1
                unique_negatives.add(negative)
            totals["unique_negative_gap_points"] += len(unique_negatives)
            classes[key]["unique_negative_gap_points"] += len(unique_negatives)
            row["negatives"] = negatives
            details.append(row)
            if comparison != "equal" and len(examples) < 30:
                examples.append(row)
        digest = reader.digest.hexdigest()
    for key, counts in classes.items():
        pair_engine, rel = key.split(":", 1)
        family = pair_engine + ":" + json.dumps(relation_class(json.loads(rel)), sort_keys=True, separators=(",", ":"))
        families[family].update(counts)
    for group in (classes, families):
        for counts in group.values():
            counts["eligible"] = bool(counts["confirmed"] and counts["equal"] == counts["pairs"] and
                                      not counts["negative_unsound"] and not counts["negative_unmodelled"] and
                                      not counts["projection_incompatible"])
    eligible = [key for key, counts in classes.items() if counts["eligible"]]
    result = {"schema": "atlas.model_composition_audit.v1", "master_sha256": digest,
              "rule_source_sha256": rule_source_sha256(),
              "per_operation_sha256": per_op_sha256,
              "model": rules.model, "totals": dict(totals), "classes": dict(sorted(classes.items())),
              "relation_classes": dict(sorted(families.items())),
              "eligible_classes": eligible, "examples": examples,
              "limitations": ["Offline audit only; does not modify compiler admissions.",
                  "Eligibility is limited to witnessed engine/relation families, pinned geometry and model; exact constraints remain.",
                  "Pair minima are suffix-safe bounds; no general total-MXU capacity is inferred."]}
    output = safe_path(output)
    output.mkdir(parents=True, exist_ok=True)
    (output / "pair-summary.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    with (output / "pair-replay.jsonl").open("w") as stream:
        for detail in details:
            stream.write(json.dumps(detail, sort_keys=True) + "\n")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--master", required=True)
    parser.add_argument("--per-op", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    raw = safe_path(args.per_op).read_bytes()
    result = replay(args.master, json.loads(raw), args.output_dir, hashlib.sha256(raw).hexdigest())
    print(json.dumps(result["totals"], sort_keys=True))


if __name__ == "__main__":
    main()
