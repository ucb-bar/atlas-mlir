"""Shared model source extraction, refusals, pins, and operand roles."""
from __future__ import annotations

import ast
from copy import deepcopy
from functools import wraps
import hashlib
import inspect
import json
import math
from pathlib import Path

UPDATE_GUIDANCE = "See docs/model-timing.md#updating-for-npu-model-changes."


class UnsupportedModelTiming(ValueError):
    """The selected model cannot supply supported timing facts."""


def model_timing_cli(function):
    @wraps(function)
    def checked(*args, **kwargs):
        try:
            return function(*args, **kwargs)
        except UnsupportedModelTiming as error:
            raise SystemExit(str(error) + "\n" + UPDATE_GUIDANCE) from None
        except (OSError, ValueError) as error:
            raise SystemExit(str(error)) from None
    return checked


def read_source(path):
    return Path(path).read_text()


def shape(node):
    class StripDocstrings(ast.NodeTransformer):
        def visit_Expr(self, item):
            if isinstance(item.value, ast.Constant) and isinstance(item.value.value, str):
                return None
            return self.generic_visit(item)
    node = StripDocstrings().visit(deepcopy(node))
    return hashlib.sha256(ast.dump(node, include_attributes=False).encode()).hexdigest()


def one(values, label):
    values = list(values)
    if len(values) != 1:
        raise UnsupportedModelTiming("expected one " + label)
    return values[0]


def name(node, identifier):
    return isinstance(node, ast.Name) and node.id == identifier


def named(tree, identifier):
    return one((node for node in tree.body if isinstance(node, ast.ClassDef)
                and node.name == identifier), identifier)


def method(tree, identifier):
    return one((node for node in tree.body if isinstance(node, ast.FunctionDef)
                and node.name == identifier), identifier)


def source_method(source, cls, identifier):
    return method(named(ast.parse(source), cls), identifier)


def assignment(tree, identifier, *, single_write=False):
    node = one((item for item in ast.walk(tree) if isinstance(item, ast.Assign)
                and len(item.targets) == 1 and name(item.targets[0], identifier)),
               "assignment to " + identifier)
    # A second or nested write would invalidate the extracted expression.
    if single_write and sum(name(item, identifier) and isinstance(item.ctx, ast.Store)
                            for item in ast.walk(tree)) != 1:
        raise UnsupportedModelTiming("unsupported reassignment of " + identifier + " in " + tree.name)
    return node


def positive_int(value, label, *, minimum=1, maximum=None):
    if isinstance(value, ast.Constant):
        value = value.value
    if type(value) is not int or value < minimum or (maximum is not None and value >= maximum):
        raise UnsupportedModelTiming(label + " must be an integer in the supported range")
    return value


def literal(node, label="source integer", *, env=None, integral=True, minimum=0, maximum=4096,
            operators=(ast.Add, ast.Sub, ast.Mult)):
    if env is None:
        try:
            value = ast.literal_eval(node)
        except (ValueError, TypeError, SyntaxError) as error:
            raise UnsupportedModelTiming("unsupported " + label + ": " + ast.dump(node)) from error
    elif ast.unparse(node) in env:
        value = env[ast.unparse(node)]
    elif isinstance(node, ast.Constant) and type(node.value) is int:
        value = node.value
    elif isinstance(node, ast.BinOp) and isinstance(node.op, operators):
        left = literal(node.left, label, env=env, minimum=None, maximum=None, operators=operators)
        right = literal(node.right, label, env=env, minimum=None, maximum=None, operators=operators)
        value = (left + right if isinstance(node.op, ast.Add) else
                 left - right if isinstance(node.op, ast.Sub) else left * right)
    else:
        raise UnsupportedModelTiming("unsupported " + label + ": " + ast.dump(node))
    valid_type = type(value) is int if integral else type(value) in (int, float)
    if (not valid_type or (type(value) is float and not math.isfinite(value))
            or (minimum is not None and value < minimum)
            or (maximum is not None and value >= maximum)):
        raise UnsupportedModelTiming("unsupported " + label)
    return value


def age_window(comparison, env, *, equality=False, maximum=4096, operators=(ast.Add, ast.Sub, ast.Mult)):
    def integer(node):
        return literal(node, "timing expression", env=env, minimum=None, maximum=None, operators=operators)

    if (equality and isinstance(comparison, ast.Compare) and len(comparison.ops) == 1
            and isinstance(comparison.ops[0], ast.Eq) and name(comparison.left, "age")):
        first = last = integer(comparison.comparators[0])
    elif (isinstance(comparison, ast.Compare) and len(comparison.ops) == 2
          and name(comparison.comparators[0], "age")
          and all(isinstance(op, (ast.Lt, ast.LtE)) for op in comparison.ops)):
        first = integer(comparison.left) + isinstance(comparison.ops[0], ast.Lt)
        last = integer(comparison.comparators[1]) - isinstance(comparison.ops[1], ast.Lt)
    else:
        raise UnsupportedModelTiming("unsupported age window predicate")
    if not 0 <= first <= last < maximum:
        raise UnsupportedModelTiming("age window is outside the supported range")
    return dict(count=last - first + 1, first_age=first, last_age=last, step=1)


def guard_literal(node, text, op):
    if text is not None:
        node = one((item.test for item in ast.walk(node) if isinstance(item, ast.If)
                    and any(isinstance(stmt, ast.Raise) and any(
                        isinstance(part, ast.Constant) and isinstance(part.value, str) and text in part.value
                        for part in ast.walk(stmt)) for stmt in item.body)), text)
    return one((literal(item.comparators[0]) for item in ast.walk(node)
                if isinstance(item, ast.Compare) and len(item.ops) == 1
                and isinstance(item.ops[0], op) and isinstance(item.left, ast.BinOp)
                and isinstance(item.left.op, ast.Sub) and isinstance(item.comparators[0], ast.Constant)),
               "source age endpoint")


def module_tree(module, *, reviewed=False, label="source"):
    path = str(module) if isinstance(module, (str, Path)) else inspect.getsourcefile(module)
    if path is None:
        raise UnsupportedModelTiming("source file is unavailable")
    path = Path(path)
    tree = ast.parse(read_source(path))
    if reviewed and ast_digest(tree) != SOURCE_SHAPES.get(path.name):
        raise UnsupportedModelTiming("unsupported " + label + " semantics: " + path.name)
    return tree


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
PIN_PATH = _DIRECTORY / "model-pins.json"
GUARDS = json.loads(PIN_PATH.read_text())
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
