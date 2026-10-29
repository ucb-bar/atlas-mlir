"""Shared extraction helpers, source pins, and operand roles."""
from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def ast_digest(tree):
    return sha256(ast.dump(tree, include_attributes=False).encode())


def timing_block(name, engine, events, last):
    return dict(name=name, engine=engine, resolved=True, events=dict(sorted(events.items())),
                first_free_age=last + 1)


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
VOCABULARY = load_vocabulary(_DIRECTORY / "model-timing-vocabulary.json")
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
