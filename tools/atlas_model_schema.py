"""Shared model identity pins and timing vocabulary."""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
from pathlib import Path

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


def load_lock(path):
    return json.loads(Path(path).read_text())


def load_vocabulary(path):
    return json.loads(Path(path).read_text())


_DIRECTORY = Path(__file__).resolve().parent
LOCK = load_lock(_DIRECTORY / "model-timing.lock.json")
SOURCE_SHAPES = LOCK["source_shapes"]
DMA_SOURCE_SHAPES = {name: SOURCE_SHAPES[name] for name in
                     ("dma.py", "lsu.py", "bank_conflict.py", "memory_backend.py")}
DMA_SOURCE_SHAPES["core.py"] = "faaabbe3276439f901d643b8c32e1746d6c2642fd18630db1283095fce1201d5"
DMA_EDGE_ORDER = ("Core.S1", "DMA.D", "DMA.store_read", "DMA.A",
                  "DMA.retire_clear_flag", "DMA.enqueue_launch")
VOCABULARY = load_vocabulary(_DIRECTORY / "model-timing-vocabulary.json")
RELATION_VALUES = {key: set(values) for key, values in VOCABULARY["relation_values"].items()}
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
    dma_identity = ({"curve_sha256": LOCK["curve_sha256"]} | LOCK["dma_source_sha256"] |
                    {"shape:" + name: LOCK["source_shapes"][name] for name in LOCK["dma_source_sha256"]}
                    if "curve_sha256" in LOCK and "dma_source_sha256" in LOCK else {})
    table("std::map<std::string, std::string> &dmaIdentity()", (
        "{" + quote(key) + ", " + quote(value) + "}" for key, value in dma_identity.items()))
    table("std::map<std::string, std::string> &dmaSourceShapes()", (
        "{" + quote(key) + ", " + quote(value) + "}" for key, value in DMA_SOURCE_SHAPES.items()))
    table("std::vector<std::string> &dmaEdgeOrder()", map(quote, DMA_EDGE_ORDER))
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
