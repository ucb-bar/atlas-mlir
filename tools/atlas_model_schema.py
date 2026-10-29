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


def timing_block(name, events, last):
    return dict(name=name, events=dict(sorted(events.items())),
                first_free_age=last + 1)


def reservation(last, *, first=0):
    return dict(first_age=first, release_age=last)


_DIRECTORY = Path(__file__).resolve().parent
GUARDS = json.loads((_DIRECTORY / "model-semantics.json").read_text())
SOURCE_SHAPES = GUARDS["source_shapes"]
VOCABULARY = json.loads((_DIRECTORY / "model-timing-vocabulary.json").read_text())
VPU_BLOCKS = tuple(entry["mnemonic"] for entry in VOCABULARY["vpu_blocks"])
MXU_BLOCKS = tuple(VOCABULARY["mxu_blocks"])
BINARY = {entry["mnemonic"] for entry in VOCABULARY["vpu_blocks"] if entry["two_input"]}


def mreg_event(mnemonic, bank, write):
    """Name a disjoint VPU probe's MREG access using its operand roles."""
    operation, = [entry for entry in VOCABULARY["vpu_blocks"] if entry["mnemonic"] == mnemonic]
    operands = dict(rd=4, rs1=0, rs2=2 if mnemonic in BINARY else 0)
    matches = []
    for event in VOCABULARY["vpu_operand_roles"][operation["op_class"]]:
        if event["key"] is None or event["write"] != write:
            continue
        if event["two_input"] and mnemonic not in BINARY:
            continue
        first = operands[event["operand"]] + event["offset"]
        if first <= bank < first + event["registers"]:
            matches.append(event["key"])
    if len(matches) != 1:
        raise ValueError(f"MREG event is outside the operand vocabulary: {mnemonic}/{bank}/{write}")
    return matches[0]
