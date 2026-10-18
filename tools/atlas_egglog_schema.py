"""Optional egglog representation of immutable Atlas value trees.

There are deliberately no state or resource constructors. Anchor identity is
validated against the enclosing Program by atlas_egglog_ir.decode_program.
Attribute strings retain raw MLIR encodings, including signed zero bits.
"""
from __future__ import annotations

import json
from egglog import Expr, String, method
from egglog.deconstruct import get_callable_args, get_literal_value
from atlas_egglog_ir import IllegalExtraction, Node


class Value(Expr):
    @method(cost=0)
    @classmethod
    def anchor(cls, id: String, type: String) -> Value: ...

    @method(cost=1)
    @classmethod
    def apply(cls, op: String, attrs: String, type: String, children: Inputs) -> Value: ...


class Inputs(Expr):
    @method(cost=0)
    @classmethod
    def empty(cls) -> Inputs: ...

    @method(cost=0)
    @classmethod
    def cons(cls, head: Value, tail: Inputs) -> Inputs: ...


def attrs_json(attrs) -> str:
    return json.dumps(sorted(attrs), separators=(",", ":"))


def to_egglog(node: Node) -> Value:
    if node.op == "anchor":
        return Value.anchor(dict(node.attrs)["id"], node.type)
    children = Inputs.empty()
    for child in reversed(node.operands):
        children = Inputs.cons(to_egglog(child), children)
    return Value.apply(node.op, attrs_json(node.attrs), node.type, children)


def from_egglog(value: Value) -> Node:
    anchor = get_callable_args(value, Value.anchor)
    if anchor is not None:
        id, type = (get_literal_value(arg) for arg in anchor)
        return Node("anchor", (), (("id", id),), type)
    args = get_callable_args(value, Value.apply)
    if args is None:
        raise IllegalExtraction("unknown egglog Value constructor")
    op, attrs, type = (get_literal_value(arg) for arg in args[:3])
    children = []
    tail = args[3]
    while get_callable_args(tail, Inputs.empty) is None:
        pair = get_callable_args(tail, Inputs.cons)
        if pair is None:
            raise IllegalExtraction("unknown egglog Inputs constructor")
        children.append(from_egglog(pair[0]))
        tail = pair[1]
    try:
        decoded = json.loads(attrs)
        if not isinstance(decoded, list) or any(
            not isinstance(pair, list) or len(pair) != 2
            or not all(isinstance(item, str) for item in pair)
            for pair in decoded
        ):
            raise ValueError("expected a list of string pairs")
        attributes = tuple(tuple(pair) for pair in decoded)
    except (TypeError, ValueError) as error:
        raise IllegalExtraction("invalid egglog attributes JSON") from error
    return Node(op, tuple(children), attributes, type)


encode_node = to_egglog
decode_node = from_egglog
