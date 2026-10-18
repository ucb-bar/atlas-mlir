"""Exact Atlas egglog families over the shared immutable value schema.

No state, accumulator handles, scheduling, engine swaps or reassociation rules.
Register facts BEFORE running EXACT_RULES. PACK may supply additional rules via
``combined_rules``; conversion/pressure admission stays outside the egraph.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping

from egglog import relation, rewrite, ruleset, vars_
from atlas_egglog_schema import Inputs, Value, attrs_json, encode_node
from atlas_egglog_facts import Contract, RealFacts, analyze_node

BF16 = "!atlas.virtual_bf16"
FP8 = "!atlas.virtual_fp8"
finite = relation("atlas-value-finite", Value)
raw_nonnegative = relation("atlas-value-raw-nonnegative", Value)
nonpositive = relation("atlas-value-nonpositive", Value)
integer_valued = relation("atlas-value-integer", Value)
ordered_le = relation("atlas-value-ordered-le", Value, Value)


def apply(op, operands=(), *, kind=None, type=BF16):
    children = Inputs.empty()
    for child in reversed(tuple(operands)):
        children = Inputs.cons(child, children)
    attrs = () if kind is None else (("kind", f'"{kind}"'),)
    return Value.apply("virtual_" + op, attrs_json(attrs), type, children)


def unary(kind, x):
    return apply("vpu_unary", (x,), kind=kind)


def binary(kind, x, y):
    return apply("vpu_binary", (x, y), kind=kind)


def transpose(x):
    return apply("xlu_transpose", (x,), type=FP8)


x, y = vars_("atlas_rule_x atlas_rule_y", Value)
# Raw total order is bijective, including NaN payloads and signed zeros.
# Therefore min/max lattice laws are bit-identical without IEEE assumptions.
EXACT_RULES = ruleset(
    rewrite(unary("mov", x)).to(x),
    rewrite(transpose(transpose(x))).to(x),
    rewrite(binary("min", x, x)).to(x),
    rewrite(binary("max", x, x)).to(x),
    rewrite(binary("min", x, binary("max", x, y))).to(x),
    rewrite(binary("max", x, binary("min", x, y))).to(x),
    rewrite(binary("min", binary("max", x, y), x)).to(x),
    rewrite(binary("max", binary("min", x, y), x)).to(x),
    rewrite(binary("min", x, y)).to(x, ordered_le(x, y)),
    rewrite(binary("max", x, y)).to(y, ordered_le(x, y)),
    rewrite(binary("min", x, y)).to(y, ordered_le(y, x)),
    rewrite(binary("max", x, y)).to(x, ordered_le(y, x)),
    # Finite ReLU maps every negative sign-bit input (including -0) to +0.
    # The raw_nonnegative guard deliberately excludes -0.
    rewrite(unary("relu", x)).to(x, raw_nonnegative(x)),
    rewrite(unary("relu", unary("relu", x))).to(unary("relu", x), finite(x)),
)


def combined_rules(*additional):
    """Library hook: combine separately qualified PACK rules with this family."""
    result = EXACT_RULES
    for extra in additional:
        result = result | extra
    return result


def register_facts(graph, roots: Iterable, *, contracts: Mapping[str, Contract] | None = None,
                   accept_contracts: bool = False, max_nodes: int = 10000, program_facts=None, program=None) -> dict:
    """Export source-derived facts; state/resource values are never invented.

    Bound comparisons are registered only at existing min/max operand pairs,
    keeping analysis linear rather than constructing all pairs of values.
    """
    memo = {}
    for root in roots:
        local = {}
        analyze_node(root, contracts=contracts, accept_contracts=accept_contracts, memo=local, program_facts=program_facts, program=program)
        memo.update(local)
        if len(memo) > max_nodes:
            raise ValueError("value fact analysis exceeds node limit")
    actions = []
    for node, facts in memo.items():
        value = encode_node(node)
        if facts.finite:
            actions.append(finite(value))
        if facts.raw_nonnegative:
            actions.append(raw_nonnegative(value))
        if facts.nonpositive:
            actions.append(nonpositive(value))
        if facts.integer_valued:
            actions.append(integer_valued(value))
        if node.op.removeprefix("atlas.").removeprefix("virtual_") == "vpu_binary" and str(dict(node.attrs).get("kind", "")).strip('"') in ("min", "max"):
            left, right = node.operands
            lf, rf = memo[left], memo[right]
            if lf.interval is not None and rf.interval is not None:
                if lf.interval[1] <= rf.interval[0]:
                    actions.append(ordered_le(encode_node(left), encode_node(right)))
                if rf.interval[1] <= lf.interval[0]:
                    actions.append(ordered_le(encode_node(right), encode_node(left)))
    if actions:
        graph.register(*actions)
    return memo


@dataclass(frozen=True)
class SharingGroup:
    constant: str
    uses: tuple[int, ...]


def bounded_constant_sharing(uses: Mapping[str, Iterable[int]], *, max_span: int = 4,
                             max_live_shared: int = 1) -> tuple[SharingGroup, ...]:
    """Conservative materialization admission, not an arithmetic rewrite.

    Split distant uses, and only merge groups when the complete plan's extra
    live constants fits the supplied budget. The decoder must rematerialize
    one splat per returned group; absence of a pressure budget disables sharing.
    This bounds additional constant lifetimes, not whole-program feasibility.
    """
    if type(max_span) is not int or max_span < 0 or type(max_live_shared) is not int or max_live_shared < 0:
        raise ValueError("sharing bounds must be nonnegative integers")
    result, occupied = [], {}
    for name in sorted(uses):
        positions = sorted(set(uses[name]))
        if any(type(p) is not int or p < 0 for p in positions):
            raise ValueError("constant use positions must be nonnegative integers")
        group = []
        for position in positions:
            can_extend = bool(group) and position - group[0] <= max_span and all(occupied.get(p, 0) < max_live_shared for p in range(group[-1], position))
            if can_extend:
                for p in range(group[-1], position):
                    occupied[p] = occupied.get(p, 0) + 1
                group.append(position)
            else:
                if group:
                    result.append(SharingGroup(name, tuple(group)))
                group = [position]
        if group:
            result.append(SharingGroup(name, tuple(group)))
    return tuple(result)
