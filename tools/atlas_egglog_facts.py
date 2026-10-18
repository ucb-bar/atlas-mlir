"""Conservative value facts for Atlas raw arithmetic; no scheduling facts.

Intervals use VectorParam's bijective raw BF16 order, NOT IEEE comparisons.
Unknown operations, opaque loads and unproved accumulator results stay unknown. A caller
contract is a premise and is used only with explicit ``accept_contracts=True``.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from fractions import Fraction
from functools import lru_cache
import hashlib
import json
import re
from typing import Mapping, Sequence


def ordered(bits: int) -> int:
    if type(bits) is not int or not 0 <= bits <= 0xFFFF:
        raise ValueError("BF16 encoding must be an unsigned 16-bit integer")
    return bits ^ (0xFFFF if bits & 0x8000 else 0x8000)


def raw_bits(key: int) -> int:
    if type(key) is not int or not 0 <= key <= 0xFFFF:
        raise ValueError("ordered BF16 key must be an unsigned 16-bit integer")
    return key ^ (0x8000 if key & 0x8000 else 0xFFFF)


def finite_value(bits: int) -> Fraction | None:
    """Exact mathematical value, only for finite BF16 encodings."""
    ordered(bits)
    exponent, fraction = (bits >> 7) & 255, bits & 127
    if exponent == 255:
        return None
    value = Fraction(fraction if exponent == 0 else 128 + fraction)
    power = -133 if exponent == 0 else exponent - 134
    value = value * (1 << power) if power >= 0 else value / (1 << -power)
    return -value if bits & 0x8000 else value


@lru_cache(maxsize=1)
def _noninteger_prefix() -> tuple[int, ...]:
    counts = [0]
    for key in range(65536):
        value = finite_value(raw_bits(key))
        counts.append(counts[-1] + int(value is None or value.denominator != 1))
    return tuple(counts)


@dataclass(frozen=True)
class RealFacts:
    """Overapproximation of each BF16 lane; ``None`` is the top interval.

    Scale is provenance for conversion operations, not an arithmetic equality.
    ``integer_valued`` may also prove a noncontiguous subset of the interval.
    """
    interval: tuple[int, int] | None = None
    integer_valued: bool = False
    scale_code: int | None = None
    provenance: tuple[str, ...] = ()

    def __post_init__(self):
        if self.interval is not None:
            lo, hi = self.interval
            if type(lo) is not int or type(hi) is not int or not 0 <= lo <= hi <= 65535:
                raise ValueError("invalid ordered raw BF16 interval")
        if self.scale_code is not None and (type(self.scale_code) is not int or not 0 <= self.scale_code <= 255):
            raise ValueError("scale code must be in [0,255]")
        if self.integer_valued and not self.finite:
            raise ValueError("integer-valued facts require a finite interval")

    @property
    def finite(self) -> bool:
        return self.interval is not None and self.interval[0] > 0x007F and self.interval[1] < 0xFF80

    @property
    def nonnegative(self) -> bool:
        return self.finite and self.interval[0] >= 0x7FFF

    @property
    def raw_nonnegative(self) -> bool:
        """Finite with no sign-bit set: excludes -0, required for ReLU identity."""
        return self.finite and self.interval[0] >= 0x8000

    @property
    def nonpositive(self) -> bool:
        return self.finite and self.interval[1] <= 0x8000

    @property
    def sign(self) -> str:
        if not self.finite:
            return "unknown"
        lo, hi = self.interval
        if lo >= 0x7FFF and hi <= 0x8000:
            return "zero"
        if lo > 0x8000:
            return "positive"
        if hi < 0x7FFF:
            return "negative"
        if self.nonnegative:
            return "nonnegative"
        if self.nonpositive:
            return "nonpositive"
        return "mixed"

    @property
    def exact_zero(self) -> str | None:
        if self.sign != "zero":
            return None
        return {0x7FFF: "negative", 0x8000: "positive"}.get(self.interval[0]) if self.interval[0] == self.interval[1] else "either"

    @property
    def unit_interval(self) -> bool:
        return self.nonnegative and self.interval[1] <= ordered(0x3F80)


def from_interval(lo: int, hi: int, *, provenance: Sequence[str] = ()) -> RealFacts:
    facts = RealFacts((lo, hi), provenance=tuple(provenance))
    if facts.finite:
        prefix = _noninteger_prefix()
        facts = replace(facts, integer_valued=prefix[hi + 1] == prefix[lo])
    return facts


def constant(bits: int) -> RealFacts:
    return from_interval(ordered(bits), ordered(bits), provenance=(f"splat:{bits:04x}",))


def join(left: RealFacts, right: RealFacts) -> RealFacts:
    """Lattice join: widen alternative values; never strengthen a proof."""
    interval = None if left.interval is None or right.interval is None else (min(left.interval[0], right.interval[0]), max(left.interval[1], right.interval[1]))
    return RealFacts(interval, left.integer_valued and right.integer_valued,
                     left.scale_code if left.scale_code == right.scale_code else None,
                     tuple(sorted(set(left.provenance + right.provenance))))


@dataclass(frozen=True)
class Contract:
    """Explicit caller premise, bound to an exact immutable expression digest."""
    subject_sha256: str
    interval: tuple[int, int]
    label: str
    program_sha256: str | None = None

    def __post_init__(self):
        if not re.fullmatch(r"[0-9a-f]{64}", self.subject_sha256) or not self.label:
            raise ValueError("contract needs a SHA-256 subject and nonempty label")
        if self.program_sha256 is not None and not re.fullmatch(r"[0-9a-f]{64}", self.program_sha256):
            raise ValueError("contract program identity must be SHA-256")
        RealFacts(self.interval)


def node_digest(node) -> str:
    def obj(value):
        return [value.op, list(value.attrs), value.type, [obj(x) for x in value.operands]]
    return hashlib.sha256(json.dumps(obj(node), separators=(",", ":")).encode()).hexdigest()


def _int(value) -> int:
    if type(value) is int:
        return value
    return int(str(value).split(":", 1)[0].strip(), 0)


def _kind(attrs: Mapping) -> str:
    return str(attrs.get("kind", "")).strip('"')


def infer_facts(op: str, operands: Sequence[RealFacts] = (), attrs: Mapping | None = None) -> RealFacts:
    """Only source-inspected transfers; no finite add/mul/exp/reciprocal rule."""
    attrs = attrs or {}
    op = op.removeprefix("atlas.virtual_").removeprefix("virtual_")
    kind = _kind(attrs)
    if op == "vpu_splat":
        return constant(_int(attrs["bits"]))
    if op == "scale_constant":
        return RealFacts(scale_code=_int(attrs["code"]), provenance=("scale_constant",))
    if op == "vpu_unary" and kind == "mov" and len(operands) == 1:
        return replace(operands[0], provenance=operands[0].provenance + ("mov",))
    if op == "vpu_unary" and kind == "relu" and len(operands) == 1 and operands[0].finite:
        src = operands[0]
        lo, hi = src.interval
        result = from_interval(max(lo, 0x8000), max(hi, 0x8000), provenance=src.provenance + ("finite_relu",))
        return replace(result, integer_valued=result.integer_valued or src.integer_valued)
    if op == "vpu_binary" and kind in ("min", "max") and len(operands) == 2:
        left, right = operands
        # Top is the full raw order. Clamps can thus prove finite output
        # even for an unknown/NaN-bearing input, without assuming finiteness.
        lr, rr = left.interval or (0, 65535), right.interval or (0, 65535)
        choose = min if kind == "min" else max
        lo, hi = (choose(lr[i], rr[i]) for i in (0, 1))
        result = from_interval(lo, hi, provenance=left.provenance + right.provenance + (f"raw_{kind}",))
        return replace(result, integer_valued=result.integer_valued or (left.integer_valued and right.integer_valued))
    if op == "vpu_reduce" and kind in ("row_min", "row_max", "col_min", "col_max") and len(operands) == 1:
        return replace(operands[0], provenance=operands[0].provenance + (kind,))
    # Conversion scale metadata never asserts a round-trip identity. FP8 has
    # no BF16 interval. UNPACK is always finite because RTL clamps overflow.
    if op == "pack_fp8":
        return RealFacts(scale_code=_int(attrs["scale_code"]), provenance=("pack_scale",))
    if op == "unpack_fp8":
        return replace(from_interval(ordered(0xFF7F), ordered(0x7F7F), provenance=("unpack_finite",)), scale_code=_int(attrs["scale_code"]))
    return RealFacts()


def _analyze_node(node, *, contracts: Mapping[str, Contract] | None = None, accept_contracts: bool = False, memo: dict | None = None) -> RealFacts:
    """Analyze encoder Nodes; explicit contracts are identified by node digest."""
    memo = {} if memo is None else memo
    if node in memo:
        return memo[node]
    facts = infer_facts(node.op, tuple(_analyze_node(x, contracts=contracts, accept_contracts=accept_contracts, memo=memo) for x in node.operands), dict(node.attrs))
    if node.op.removeprefix("atlas.").removeprefix("virtual_") == "mxu_matmul" and node.type == "!atlas.virtual_bf16" and len(node.operands) == 2 and all(x.type == "!atlas.virtual_fp8" for x in node.operands):
        facts = raw_fp8_reset_facts()
    digest = node_digest(node)
    if accept_contracts and contracts and digest in contracts:
        contract = contracts[digest]
        if contract.subject_sha256 != digest:
            raise ValueError("contract subject digest mismatch")
        lo, hi = contract.interval
        if facts.interval is not None:
            lo, hi = max(lo, facts.interval[0]), min(hi, facts.interval[1])
            if lo > hi:
                raise ValueError("caller contract contradicts derived facts")
        facts = from_interval(lo, hi, provenance=facts.provenance + (f"caller_contract:{contract.label}:{digest}",))
    memo[node] = facts
    return facts


def contract_context(contracts: Mapping[str, Contract] | None, accept_contracts: bool) -> tuple:
    """Immutable cache context; disabled contracts contribute no premises."""
    return (bool(accept_contracts), tuple(sorted((contracts or {}).items())) if accept_contracts else ())


def analyze_node(node, *, contracts: Mapping[str, Contract] | None = None,
                 accept_contracts: bool = False, memo: dict | None = None,
                 program_facts=None, program=None) -> RealFacts:
    """Analyze a Node with an invocation-owned cache and optional SSA seeds.

    ``memo`` is output-only: prior contents never supply proofs. Reusing an
    output dictionary across contract modes cannot retain a caller premise.
    ProgramFacts seeds must have exactly the current contract context.
    """
    if accept_contracts and program is not None and any(contract.program_sha256 != program.sha256 for contract in (contracts or {}).values()):
        raise ValueError("caller contract program identity mismatch")
    if program_facts is not None and program_facts.contract_context != contract_context(contracts, accept_contracts):
        raise ValueError("program facts contract context mismatch")
    working = {}
    if program_facts is not None:
        if program is None or program_facts.program_sha256 != program.sha256:
            raise ValueError("program facts source identity mismatch")
        reachable = set()
        def visit(value):
            if value not in reachable:
                reachable.add(value)
                for child in value.operands:
                    visit(child)
        visit(node)
        allowed = set(program.nodes.values())
        if any(value.op == "anchor" and value not in allowed for value in reachable):
            raise ValueError("anchor is not in the current program")
        working = {value: fact for value, fact in program_facts.nodes.items() if value in reachable}
    result = _analyze_node(node, contracts=contracts, accept_contracts=accept_contracts, memo=working)
    if memo is not None:
        memo.clear()
        memo.update(working)
    return result


@dataclass(frozen=True)
class ProgramFacts:
    """SSA and anchor facts; accumulator proof state never enters eclasses."""
    values: Mapping[str, RealFacts]
    nodes: Mapping[object, RealFacts]
    accumulators: Mapping[str, RealFacts]
    program_sha256: str
    contract_context: tuple = (False, ())


def raw_fp8_reset_facts() -> RealFacts:
    """Conservative bound for fixed 32-term raw E4M3 reset contractions.

    E4M3Mul treats all 8-bit codes as finite integer fields. Each product is
    below 2^18; repeated BF16 rounding grows the 32-term absolute bound below
    2^25. IPT's bounded integer anchor tree also lies inside this envelope.
    This does not apply after an arbitrary BF16 seed/accumulation.
    """
    return from_interval(ordered(0xCC00), ordered(0x4C00),
                         provenance=("raw_fp8_32term_reset_abs_lt_2pow25",))


def analyze_program(program, *, contracts: Mapping[str, Contract] | None = None,
                    accept_contracts: bool = False) -> ProgramFacts:
    """Analyze validated encoder Programs, retaining resource SSA outside egglog.

    Exact accumulator copies preserve source domains. Raw FP8 reset/copy
    bounds require the fixed 32x32 tile types. Every accumulation deliberately
    kills the accumulator fact, even if all of its operands were finite.
    """
    if accept_contracts and any(contract.program_sha256 != program.sha256 for contract in (contracts or {}).values()):
        raise ValueError("caller contract program identity mismatch")
    values, nodes, accumulators = {}, {}, {}
    source_nodes = program.nodes
    for op in program.operations:
        short = op.op.removeprefix("atlas.").removeprefix("virtual_")
        operands = tuple(values.get(name, RealFacts()) for name in op.operands)
        facts = infer_facts(short, operands, dict(op.attrs))
        if short == "mxu_matmul" and op.input_types == ("!atlas.virtual_fp8", "!atlas.virtual_fp8"):
            facts = raw_fp8_reset_facts()
        elif short == "mxu_load_acc_bf16":
            accumulators[op.results[1]] = replace(values.get(op.operands[1], RealFacts()), provenance=values.get(op.operands[1], RealFacts()).provenance + ("bf16_accumulator_copy",))
        elif short == "mxu_load_acc_fp8":
            accumulators[op.results[1]] = from_interval(ordered(0xC3E0), ordered(0x43E0), provenance=("raw_fp8_accumulator_expansion_flush_specials",))
        elif short == "mxu_reset":
            accumulators[op.results[1]] = raw_fp8_reset_facts()
        elif short == "mxu_accumulate":
            accumulators[op.results[1]] = RealFacts(provenance=("unproved_accumulation",))
        elif short == "mxu_readout_bf16":
            src = accumulators.get(op.operands[1], RealFacts())
            facts = replace(src, provenance=src.provenance + ("bf16_accumulator_readout",))
        elif short == "mxu_readout_fp8":
            scale = values.get(op.operands[2], RealFacts()).scale_code
            facts = RealFacts(scale_code=scale, provenance=("mxu_fp8_readout_scale",))
        for result, type in zip(op.results, op.output_types):
            if type not in ("!atlas.virtual_bf16", "!atlas.virtual_fp8", "!atlas.virtual_scale"):
                continue
            node = source_nodes[result]
            if accept_contracts and contracts and type == "!atlas.virtual_bf16":
                digest = node_digest(node)
                if digest in contracts:
                    contract = contracts[digest]
                    if contract.subject_sha256 != digest:
                        raise ValueError("contract subject digest mismatch")
                    lo, hi = contract.interval
                    if facts.interval is not None:
                        lo, hi = max(lo, facts.interval[0]), min(hi, facts.interval[1])
                        if lo > hi:
                            raise ValueError("caller contract contradicts derived facts")
                    facts = from_interval(lo, hi, provenance=facts.provenance + (f"caller_contract:{contract.label}:{digest}",))
            values[result] = facts
            nodes[node] = facts
    return ProgramFacts(values, nodes, accumulators, program.sha256, contract_context(contracts, accept_contracts))
