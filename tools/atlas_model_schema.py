"""Shared model identity pins and timing vocabulary."""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
from pathlib import Path
import re

from atlas_derive_scratchpad import scratchpad_assumptions


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def ast_digest(tree):
    return sha256(ast.dump(tree, include_attributes=False).encode())


def timing_block(name, engine, events, last):
    return dict(name=name, engine=engine, resolved=True, events=dict(sorted(events.items())),
                first_free_age=last + 1, next_issue_age=last + 1,
                assumptions=scratchpad_assumptions(), observation={"accepted": True, "drained": True})


def reservation(last, *, first=0, banks=None):
    value = dict(first_age=first, release_age=last, available_age=last + 1)
    if banks is not None:
        value["banks"] = banks
    return value


def _object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate model schema key: " + key)
        result[key] = value
    return result


def _load(path):
    return json.loads(Path(path).read_text(), object_pairs_hook=_object)


def load_lock(path):
    lock = _load(path)
    names = set("model_revision source_sha256 reference_configuration_sha256 configuration_sha256 "
                "numerical_data_sha256 reference_sha256 master_sha256 validation_sha256 per_op_sha256".split())
    # Older locks predate the optional DMA curve contract.
    optional = {"curve_sha256", "dma_source_sha256"}
    if (not isinstance(lock, dict) or set(lock) - optional != names | {"schema", "source_shapes", "qualified_constraint_hashes"}
            or lock["schema"] != "atlas.model_timing.lock.v1"):
        raise ValueError("invalid model timing lock schema")
    for name in names:
        length = 40 if name == "model_revision" else 64
        if not isinstance(lock[name], str) or not re.fullmatch(r"[0-9a-f]{%d}" % length, lock[name]):
            raise ValueError("invalid model timing pin: " + name)
    if "curve_sha256" in lock and (not isinstance(lock["curve_sha256"], str) or
                                   not re.fullmatch(r"[0-9a-f]{64}", lock["curve_sha256"])):
        raise ValueError("invalid model timing pin: curve_sha256")
    if "dma_source_sha256" in lock:
        pins = lock["dma_source_sha256"]
        if (not isinstance(pins, dict) or set(pins) != {"memory_backend.py", "dma.py"} or
                any(not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{64}", value)
                    for value in pins.values())):
            raise ValueError("invalid DMA source pins")
    shapes = lock["source_shapes"]
    if not isinstance(shapes, dict) or not shapes:
        raise ValueError("invalid model source shapes")
    for name, digest in shapes.items():
        if (not re.fullmatch(r"[a-z_]+\.py", name) or not isinstance(digest, str) or
                not re.fullmatch(r"[0-9a-f]{64}", digest)):
            raise ValueError("invalid model source shape: " + name)
    constraints = lock["qualified_constraint_hashes"]
    if not isinstance(constraints, dict) or not constraints:
        raise ValueError("invalid constraint fingerprints")
    for name, values in constraints.items():
        if (not re.fullmatch(r"[a-z_]+", name) or not isinstance(values, list) or not values
                or any(not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{64}", value) for value in values)
                or len(values) != len(set(values))):
            raise ValueError("invalid constraint fingerprints: " + name)
    return lock


def load_vocabulary(path):
    vocabulary = _load(path)
    fields = {"schema", "base_relation_keys", "relation_values", "vpu_blocks", "mxu_blocks",
              "vpu_operand_roles", "mxu_operand_roles"}
    if (not isinstance(vocabulary, dict) or set(vocabulary) != fields or
            vocabulary["schema"] != "atlas.model_timing.vocabulary.v1"):
        raise ValueError("invalid model timing vocabulary schema")
    base, relations = vocabulary["base_relation_keys"], vocabulary["relation_values"]
    if not isinstance(base, list) or not base or any(not isinstance(key, str) for key in base) or len(set(base)) != len(base):
        raise ValueError("invalid base relation keys")
    if not isinstance(relations, dict) or not relations:
        raise ValueError("invalid relation vocabulary")
    for values in relations.values():
        if (not isinstance(values, list) or not values or any(not isinstance(value, str) for value in values) or
                len(set(values)) != len(values)):
            raise ValueError("invalid relation values")
    for family, extra in (("vpu_blocks", "two_input"), ("mxu_blocks", "op_class")):
        entries = vocabulary[family]
        if not isinstance(entries, list) or not entries:
            raise ValueError("invalid operation vocabulary")
        seen = set()
        for entry in entries:
            if not isinstance(entry, dict) or set(entry) != {"mnemonic", "block", extra, "op_class"}:
                raise ValueError("invalid operation vocabulary entry")
            for key in ("mnemonic", "block"):
                if not isinstance(entry[key], str) or not re.fullmatch(r"[A-Za-z0-9_.]+", entry[key]):
                    raise ValueError("invalid operation vocabulary name")
            if entry["mnemonic"] in seen:
                raise ValueError("duplicate operation mnemonic")
            seen.add(entry["mnemonic"])
            if ((extra == "two_input" and type(entry[extra]) is not bool) or
                    (extra == "op_class" and (not isinstance(entry[extra], str) or
                                              not re.fullmatch(r"[A-Za-z][A-Za-z0-9]*", entry[extra])))):
                raise ValueError("invalid operation vocabulary attribute")
    roles = vocabulary["vpu_operand_roles"] | vocabulary["mxu_operand_roles"]
    if not isinstance(roles, dict) or not roles:
        raise ValueError("invalid VPU operand roles")
    role_fields = {"key", "operand", "write", "registers", "passes", "offset", "two_input"}
    for op_class, events in roles.items():
        if not re.fullmatch(r"[A-Za-z][A-Za-z0-9]*", op_class) or not isinstance(events, list) or not events:
            raise ValueError("invalid VPU operand class")
        for event in events:
            if not isinstance(event, dict) or set(event) - {"resource"} != role_fields:
                raise ValueError("invalid VPU operand role")
            if event.get("resource", "MReg") not in {"MReg", "Acc", "Weight", "EReg"}:
                raise ValueError("invalid operand resource")
            if event["operand"] not in {"rd", "rs1", "rs2"} or (event["key"] is not None and not isinstance(event["key"], str)):
                raise ValueError("invalid VPU operand mapping")
            if any(type(event[key]) is not bool for key in ("write", "two_input")):
                raise ValueError("invalid VPU operand flags")
            if any(type(event[key]) is not int or event[key] < 1 for key in ("registers", "passes")):
                raise ValueError("invalid VPU operand dimensions")
            if type(event["offset"]) is not int or event["offset"] < 0:
                raise ValueError("invalid VPU operand offset")
    return vocabulary


_DIRECTORY = Path(__file__).resolve().parent
LOCK = load_lock(_DIRECTORY / "model-timing.lock.json")
SOURCE_SHAPES = LOCK["source_shapes"]
QUALIFIED_CONSTRAINT_HASHES = {key: set(values) for key, values in LOCK["qualified_constraint_hashes"].items()}
VOCABULARY = load_vocabulary(_DIRECTORY / "model-timing-vocabulary.json")
MODEL_REVISION = LOCK["model_revision"]
CONFIGURATION_SHA256 = LOCK["configuration_sha256"]
MASTER_SHA256 = LOCK["master_sha256"]
VALIDATION_SHA256 = LOCK["validation_sha256"]
PER_OP_SHA256 = LOCK["per_op_sha256"]
BASE_KEYS = set(VOCABULARY["base_relation_keys"])
RELATION_VALUES = {key: set(values) for key, values in VOCABULARY["relation_values"].items()}
RELATION_KEYS = BASE_KEYS | set(RELATION_VALUES)
VPU_BLOCKS = {entry["mnemonic"]: entry["block"] for entry in VOCABULARY["vpu_blocks"]}
MXU_BLOCKS = {entry["mnemonic"]: entry["block"] for entry in VOCABULARY["mxu_blocks"]}
BINARY = {entry["mnemonic"] for entry in VOCABULARY["vpu_blocks"] if entry["two_input"]}


def mreg_event(mnemonic, engine, bank, write):
    """Map a disjoint ISA probe's MREG access through the shared operand roles."""
    if engine in {"vpu", "mxu0", "mxu1"}:
        family = "vpu" if engine == "vpu" else "mxu"
        base = mnemonic if family == "vpu" else mnemonic.rsplit(".", 1)[0]
        operation, = [entry for entry in VOCABULARY[family + "_blocks"] if entry["mnemonic"] == base]
        operands = dict(rd=4, rs1=0, rs2=2 if mnemonic in BINARY else 0)
        matches = []
        for event in VOCABULARY[family + "_operand_roles"][operation["op_class"]]:
            if event["key"] is None or event.get("resource", "MReg") != "MReg" or event["write"] != write:
                continue
            if event["two_input"] and mnemonic not in BINARY:
                continue
            first = operands[event["operand"]] + event["offset"]
            if first <= bank < first + event["registers"]:
                matches.append(event["key"])
        if len(matches) != 1:
            raise ValueError(f"MREG event is outside the operand vocabulary: {mnemonic}/{bank}/{write}")
        return matches[0]
    if engine == "xlu":
        return "write" if write else "read"
    if engine == "vlsu":
        return "mreg_write" if write else "mreg_read"
    raise ValueError(f"unexpected MREG event from {engine}")


def generate(vocabulary):
    lines = ["// Generated from model-timing-vocabulary.json. Do not edit.", "#pragma once",
             '#include "Atlas/AtlasTiming.h"', "#include <set>",
             "namespace mlir::atlas::timing::model_schema {"]
    digest = sha256(canonical(vocabulary))
    lines.append('inline const char *vocabularySha256() { return "' + digest + '"; }')
    lines.append('inline const char *vocabularyJson() { return ' + json.dumps(canonical(vocabulary).decode()) + '; }')

    def table(signature, entries):
        lines.extend(["inline const " + signature + " {", "  static const " + signature.split(" &")[0] + " table = {"])
        lines.extend("      " + entry + "," for entry in entries)
        lines.extend(["  };", "  return table;", "}"])

    quote = json.dumps
    table("std::map<std::string, std::set<std::string>> &relationValues()", (
        "{" + quote(key) + ", {" + ", ".join(map(quote, values)) + "}}"
        for key, values in vocabulary["relation_values"].items()))
    table("std::map<std::string, std::set<std::string>> &qualifiedConstraintHashes()", (
        "{" + quote(key) + ", {" + ", ".join(map(quote, values)) + "}}"
        for key, values in LOCK["qualified_constraint_hashes"].items()))
    table("std::map<std::string, std::string> &compositionIdentity()", (
        "{" + quote(key) + ", " + quote(LOCK[pin]) + "}"
        for key, pin in (("revision", "model_revision"), ("source_sha256", "source_sha256"),
                         ("configuration_sha256", "configuration_sha256"))))
    dma_identity = ({"curve_sha256": LOCK["curve_sha256"]} | LOCK["dma_source_sha256"] |
                    {"shape:" + name: LOCK["source_shapes"][name] for name in LOCK["dma_source_sha256"]}
                    if "curve_sha256" in LOCK and "dma_source_sha256" in LOCK else {})
    table("std::map<std::string, std::string> &dmaIdentity()", (
        "{" + quote(key) + ", " + quote(value) + "}" for key, value in dma_identity.items()))
    table("std::map<std::string, const char *> &vpuBlocks()", (
        "{" + quote(entry["mnemonic"]) + ", " + quote("vpu." + entry["block"]) + "}"
        for entry in vocabulary["vpu_blocks"]))
    table("std::map<OpClass, const char *> &mxuBlocks()", (
        "{OpClass::" + entry["op_class"] + ", " + quote(entry["block"]) + "}"
        for entry in vocabulary["mxu_blocks"]))
    lines.extend(["struct RegisterEvent {", "  const char *key;", "  int Instr::*operand;", "  bool write;",
                  "  int registers, passes, offset;", "  bool twoInput;", "};"])
    lines.extend(["struct ResourceEvent : RegisterEvent {", "  Res resource;", "};"])
    for family, entry_type in (("vpu", "RegisterEvent"), ("mxu", "ResourceEvent")):
        entries = []
        for op_class, events in vocabulary[family + "_operand_roles"].items():
            rows = []
            for event in events:
                fields = ["nullptr" if event["key"] is None else quote(event["key"]), "&Instr::" + event["operand"],
                          str(event["write"]).lower(), str(event["registers"]), str(event["passes"]),
                          str(event["offset"]), str(event["two_input"]).lower()]
                row = "{" + ", ".join(fields) + "}"
                rows.append("{" + row + ", Res::" + event["resource"] + "}" if family == "mxu" else row)
            entries.append("{OpClass::" + op_class + ", {" + ", ".join(rows) + "}}")
        table("std::map<OpClass, std::vector<" + entry_type + ">> &" + family + "Events()", entries)
    lines.append("} // namespace mlir::atlas::timing::model_schema")
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vocabulary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    content = generate(load_vocabulary(args.vocabulary))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if not args.output.exists() or args.output.read_text() != content:
        args.output.write_text(content)


if __name__ == "__main__":
    main()
