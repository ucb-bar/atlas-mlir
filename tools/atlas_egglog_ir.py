"""Lossless, bounded Atlas virtual IR bridge with an optional egglog backend.

Only immutable dataflow values enter eclasses. State, DMA, MXU handles and
effect order are retained as fixed operations; their tile results are unique
anchor leaves. This module establishes representation, not rewrite legality.
The supported input language is one flat module or straight-line,
argument-free function in generic Atlas syntax, with optional scalar i32
constants and comments.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import re
from typing import Mapping


class UnsupportedProgram(ValueError):
    """Unsupported syntax or malformed virtual program."""


class IllegalExtraction(ValueError):
    """Extracted dataflow violates types, anchors or dominance."""


@dataclass(frozen=True)
class Node:
    op: str
    operands: tuple[Node, ...] = ()
    attrs: tuple[tuple[str, str], ...] = ()
    type: str = "!atlas.virtual_bf16"

    def attribute(self, name: str):
        """Decode the supported MLIR attribute subset to int/string/None."""
        value = dict(self.attrs)[name]
        if value == "unit":
            return None
        if value.startswith('"'):
            return json.loads(value)
        if value in ("true", "false"):
            return value == "true"
        return int(value.split(":")[0].strip())


@dataclass(frozen=True)
class Operation:
    line: int
    op: str
    results: tuple[str, ...]
    operands: tuple[str, ...]
    attrs: tuple[tuple[str, str], ...]
    input_types: tuple[str, ...]
    output_types: tuple[str, ...]
    pure: bool


@dataclass(frozen=True)
class Program:
    lines: tuple[str, ...]
    operations: tuple[Operation, ...]
    values: tuple[tuple[str, Node], ...]
    sha256: str

    @property
    def nodes(self) -> dict[str, Node]:
        return dict(self.values)


_SSA = r"%[A-Za-z_0-9][A-Za-z_0-9.$-]*"
_OP = re.compile(
    rf'\s*(?P<results>{_SSA}(?:\s*,\s*{_SSA})*)\s*=\s*"atlas\.(?P<op>virtual_[a-z0-9_]+)"'
    rf'\((?P<operands>[^()]*)\)\s*(?:<\{{(?P<props>[^{{}}]*)\}}>\s*)?(?:\{{(?P<attrs>[^{{}}]*)\}}\s*)?'
    r':\s*\((?P<inputs>[^()]*)\)\s*->\s*(?P<outputs>.*?)\s*'
)
_CONST = re.compile(rf'\s*({_SSA})\s*=\s*arith.constant\s+(-?\d+)\s*:\s*i32\s*')
_ADDI = re.compile(rf'\s*({_SSA})\s*=\s*arith.addi\s+({_SSA}),\s*({_SSA})\s*:\s*i32\s*')
_RETURN = re.compile(rf'\s*(?:func\.)?return\s+({_SSA})\s*:\s*!atlas.virtual_state\s*')
_BF = "!atlas.virtual_bf16"
_FP = "!atlas.virtual_fp8"
_STATE = "!atlas.virtual_state"
_SCALE = "!atlas.virtual_scale"
_SIGNATURES = {
    "start": ((), ("state",), ()),
    "input_bf16": (("state",), ("state", "bf16"), ("index",)),
    "input_fp8": (("state",), ("state", "fp8"), ("index",)),
    "output_bf16": (("state", "bf16"), ("state",), ("index",)),
    "vpu_unary": (("bf16",), ("bf16",), ("kind",)),
    "vpu_binary": (("bf16", "bf16"), ("bf16",), ("kind",)),
    "vpu_reduce": (("bf16",), ("bf16",), ("kind",)),
    "vpu_splat": ((), ("bf16",), ("bits",)),
    "pack_fp8": (("bf16",), ("fp8",), ("scale_code",)),
    "xlu_transpose": (("fp8",), ("fp8",), ()),
    "mxu_matmul": (("fp8", "fp8"), ("bf16",), ("unit",)),
    "mxu_load_weight": (("state", "fp8"), ("state", "mxu_weight"), ("unit",)),
    "mxu_load_acc_fp8": (("state", "fp8"), ("state", "mxu_acc"), ("unit",)),
    "mxu_load_acc_bf16": (("state", "bf16"), ("state", "mxu_acc"), ("unit",)),
    "mxu_reset": (("state", "fp8", "mxu_weight"), ("state", "mxu_acc"), ()),
    "mxu_accumulate": (("state", "fp8", "mxu_weight", "mxu_acc"), ("state", "mxu_acc"), ()),
    "mxu_readout_bf16": (("state", "mxu_acc"), ("state", "bf16"), ()),
    "mxu_readout_fp8": (("state", "mxu_acc", "scale"), ("state", "fp8"), ()),
    "scale_constant": ((), ("scale",), ("code",)),
    "dma_load_fp8": (("state", "i32", "i32"), ("state", "fp8"), ()),
    "dma_load_bf16": (("state", "i32", "i32"), ("state", "bf16"), ()),
    "dma_store_fp8": (("state", "fp8", "i32", "i32"), ("state",), ()),
    "dma_store_bf16": (("state", "bf16", "i32", "i32"), ("state",), ()),
}
PURE_OPS = frozenset("virtual_" + name for name in (
    "vpu_unary", "vpu_binary", "vpu_reduce", "vpu_splat", "pack_fp8",
    "xlu_transpose", "mxu_matmul", "scale_constant"))


def _split(text: str) -> tuple[str, ...]:
    if not text.strip():
        return ()
    # Commas in diagnostic/provenance strings are not field separators.
    result = []
    start = 0
    quoted = escaped = False
    for index, char in enumerate(text):
        if escaped:
            escaped = False
        elif char == "\\" and quoted:
            escaped = True
        elif char == '"':
            quoted = not quoted
        elif char == "," and not quoted:
            result.append(text[start:index].strip())
            start = index + 1
    if quoted or escaped:
        raise UnsupportedProgram("unterminated attribute string")
    result.append(text[start:].strip())
    return tuple(result)


def _types(text: str) -> tuple[str, ...]:
    text = text.strip()
    return _split(text[1:-1] if text.startswith("(") and text.endswith(")") else text)


def _kind(type: str) -> str:
    if type == "i32":
        return type
    match = re.fullmatch(r'!atlas.virtual_(state|bf16|fp8|scale|mxu_weight|mxu_acc)(?:<([01])>)?', type)
    if not match or ((match[1].startswith("mxu_")) != (match[2] is not None)):
        raise UnsupportedProgram(f"unsupported type {type}")
    return match[1]


def _attrs(text: str) -> tuple[tuple[str, str], ...]:
    attrs = {}
    for part in _split(text):
        match = re.fullmatch(r'([A-Za-z_][A-Za-z_0-9.]*)(?:\s*=\s*("(?:\\.|[^"\\])*"|-?\d+\s*:\s*i32|unit|true|false))?', part)
        if not match or match[1] in attrs:
            raise UnsupportedProgram(f"unsupported/duplicate attribute {part}")
        value = match[2] or "unit"
        if ":" in value:
            value = str(int(value.split(":")[0])) + " : i32"
        attrs[match[1]] = value
    return tuple(sorted(attrs.items()))


def _signature(op: str, inputs: tuple[str, ...], outputs: tuple[str, ...], attrs):
    short = op.removeprefix("virtual_")
    if short not in _SIGNATURES:
        raise UnsupportedProgram(f"unsupported operation {op}")
    wanted_in, wanted_out, keys = _SIGNATURES[short]
    if tuple(map(_kind, inputs)) != wanted_in or tuple(map(_kind, outputs)) != wanted_out:
        raise UnsupportedProgram(f"wrong signature for {op}")
    fields = dict(attrs)
    expected = (set(keys) | {key for key in fields if key.startswith("atlas.")}
                | ({"finite_src"} if short == "pack_fp8" and "finite_src" in fields else set()))
    if set(fields) != expected or len(fields) != len(attrs):
        raise UnsupportedProgram(f"wrong attributes for {op}")
    for key, value in fields.items():
        if key.startswith("atlas."):
            # Preserve diagnostic metadata as opaque data. It supplies no
            # finiteness or caller premise to this representation bridge.
            continue
        elif key == "finite_src":
            if value != "unit":
                raise UnsupportedProgram("finite_src requires UnitAttr")
        elif key == "kind":
            allowed = {
                "vpu_unary": ("mov", "recip", "exp", "exp2", "square", "cube", "relu", "sin", "cos", "tanh", "log2", "sqrt"),
                "vpu_binary": ("add", "sub", "mul", "min", "max"),
                "vpu_reduce": ("row_sum", "row_min", "row_max", "col_sum", "col_min", "col_max"),
            }[short]
            if value not in tuple(json.dumps(item) for item in allowed):
                raise UnsupportedProgram(f"unsupported {op} kind {value}")
        else:
            if not re.fullmatch(r'-?\d+ : i32', value):
                raise UnsupportedProgram(f"{key} requires i32")
            integer = int(value.split(":")[0])
            limit = {"bits": 65535, "code": 255, "scale_code": 255, "unit": 1, "index": 2147483647}[key]
            if not 0 <= integer <= limit:
                raise UnsupportedProgram(f"{key} out of range")
    units = {int(type[-2]) for type in (*inputs, *outputs) if _kind(type).startswith("mxu_")}
    if "unit" in fields:
        units.add(int(fields["unit"].split(":")[0]))
    if len(units) > 1:
        raise UnsupportedProgram("MXU resource unit mismatch")


def parse_program(text: str) -> Program:
    """Parse and validate the supported straight-line input, with no contracts."""
    lines = tuple(text.splitlines(keepends=True))
    operations = []
    values = {}
    types = {}
    state = None
    consumed_acc = set()
    functions = starts = returns = 0
    in_function = False
    closed_function = False
    module_seen = False
    module_closed = False
    for line, original in enumerate(lines):
        body = original.split("//", 1)[0].strip()
        if not body:
            continue
        if module_closed:
            raise UnsupportedProgram("syntax after module close")
        if body == "module {" and not module_seen and not in_function:
            module_seen = True
            continue
        if body.startswith("func.func "):
            if not module_seen or functions or not re.fullmatch(r'func.func @[A-Za-z_0-9.$-]+\(\) -> !atlas.virtual_state(?: attributes \{[^{}]*\})? \{', body):
                raise UnsupportedProgram("requires one argument-free straight-line function")
            functions += 1
            in_function = True
            continue
        if body == "}":
            if in_function:
                if returns != 1:
                    raise UnsupportedProgram("missing return")
                in_function = False
                closed_function = True
            elif module_seen and (not functions or closed_function):
                module_closed = True
            else:
                raise UnsupportedProgram("unexpected closing brace")
            continue
        if (functions and not in_function) or not module_seen or returns:
            raise UnsupportedProgram(f"unexpected syntax at line {line + 1}")
        ret = _RETURN.fullmatch(body)
        if ret:
            if not in_function:
                raise UnsupportedProgram("return outside function")
            if ret[1] != state:
                raise UnsupportedProgram("return does not consume final state")
            returns += 1
            continue
        constant = _CONST.fullmatch(body)
        addition = _ADDI.fullmatch(body)
        if constant:
            if not -(1 << 31) <= int(constant[2]) < (1 << 32):
                raise UnsupportedProgram("i32 constant out of range")
            op = Operation(line, "arith.constant", (constant[1],), (), (("value", constant[2]),), (), ("i32",), False)
        elif addition:
            op = Operation(line, "arith.addi", (addition[1],), (addition[2], addition[3]), (), ("i32", "i32"), ("i32",), False)
        else:
            match = _OP.fullmatch(body)
            if not match:
                raise UnsupportedProgram(f"unsupported syntax at line {line + 1}: {body[:100]}")
            inputs, outputs = _types(match["inputs"]), _types(match["outputs"])
            props, attributes = _attrs(match["props"] or ""), _attrs(match["attrs"] or "")
            attrs = tuple(sorted(props + attributes))
            _signature(match["op"], inputs, outputs, attrs)
            op = Operation(line, match["op"], _split(match["results"]), _split(match["operands"]), attrs, inputs, outputs, match["op"] in PURE_OPS)
        if len(op.operands) != len(op.input_types) or len(op.results) != len(op.output_types):
            raise UnsupportedProgram("operand/result arity mismatch")
        for operand, expected in zip(op.operands, op.input_types):
            if types.get(operand) != expected:
                raise UnsupportedProgram(f"undefined, non-dominating or wrong-type operand {operand}")
            if expected == _STATE and operand != state:
                raise UnsupportedProgram("nonlinear or stale state")
            if _kind(expected) == "mxu_acc":
                if operand in consumed_acc:
                    raise UnsupportedProgram("accumulator version consumed twice")
                consumed_acc.add(operand)
        if op.op == "virtual_start":
            starts += 1
            if starts != 1 or operations:
                raise UnsupportedProgram("virtual_start must be first operation")
        for result, output in zip(op.results, op.output_types):
            if result in types:
                raise UnsupportedProgram(f"duplicate SSA definition {result}")
            types[result] = output
            if output == _STATE:
                state = result
            elif output in (_BF, _FP, _SCALE):
                values[result] = (Node(op.op, tuple(values[name] for name in op.operands), op.attrs, output)
                                  if op.pure else Node("anchor", (), (("id", result),), output))
        operations.append(op)
    if starts != 1 or not module_closed or (functions and (returns != 1 or not closed_function)):
        raise UnsupportedProgram("incomplete module/function/state skeleton")
    return Program(lines, tuple(operations), tuple(values.items()), hashlib.sha256(text.encode()).hexdigest())


def load_program(path: str | Path) -> Program:
    path = Path(path)
    absolute = Path(os.path.abspath(path))
    for candidate in (path, absolute):
        if any(re.search("ha" + "mmer|vl" + "si", part, re.I) for part in candidate.parts):
            raise UnsupportedProgram("restricted input path")
    # Refuse symlinks component by component before following any of them.
    # Calling resolve() first could inspect an otherwise forbidden target.
    current = Path(absolute.anchor)
    for component in absolute.parts[1:]:
        current /= component
        if current.is_symlink():
            raise UnsupportedProgram("symlink input paths are unsupported")
    return parse_program(absolute.read_text())


def decode_program(program: Program, extracted: Mapping[str, Node]) -> str:
    """Rebuild pure ops from extracted roots, keeping effects and SSA identities.

    An extraction supplies exactly one root per original pure result. Unchanged
    roots reuse their original spelling only after tree validation. A changed
    tree may reference dominating anchor leaves and is materialized near its
    original definition, with fresh SSA names for new subexpressions. This is a
    structural safety check: callers remain responsible for exact equivalence.
    """
    originals = program.nodes
    pure = {op.results[0]: op for op in program.operations if op.pure}
    if set(extracted) != set(pure):
        raise IllegalExtraction("extraction must contain every pure SSA root exactly once")
    anchors = {node: name for name, node in program.values if node.op == "anchor"}
    available_anchors = {}
    aliases = {}
    used_names = {name for op in program.operations for name in op.results}
    lines = list(program.lines)
    counter = 0

    def check(node, active):
        if not isinstance(node, Node):
            raise IllegalExtraction("extraction contains a non-Node")
        if id(node) in active:
            raise IllegalExtraction("cyclic extracted expression")
        if node.op == "anchor":
            if node not in anchors:
                raise IllegalExtraction("unknown or forged anchor")
            return
        if node.op not in PURE_OPS:
            raise IllegalExtraction("effect/resource operation in eclass")
        try:
            _signature(node.op, tuple(child.type for child in node.operands), (node.type,), node.attrs)
        except (ValueError, TypeError, AttributeError) as error:
            raise IllegalExtraction(str(error)) from error
        for child in node.operands:
            check(child, active | {id(node)})

    for name, node in extracted.items():
        check(node, set())
        if node.type != originals[name].type:
            raise IllegalExtraction("extracted root type changed")

    def render(name, node, operands):
        fields = ", ".join(key if value == "unit" else f"{key} = {value}" for key, value in node.attrs)
        attrs = f" {{{fields}}}" if fields else ""
        return f'    {name} = "atlas.{node.op}"({", ".join(operands)}){attrs} : ({", ".join(child.type for child in node.operands)}) -> {node.type}\n'

    def substituted(line):
        return re.sub(_SSA, lambda match: aliases.get(match[0], match[0]), line)

    for op in program.operations:
        if not op.pure:
            lines[op.line] = substituted(lines[op.line])
            for result in op.results:
                if result in originals and originals[result].op == "anchor":
                    available_anchors[originals[result]] = result
            continue
        name = op.results[0]
        node = extracted[name]
        added = []
        # New expression sharing is bounded to this one root. Reusing an
        # original operand preserves preexisting sharing; other pure children
        # are rematerialized here instead of extending global live ranges.
        available = dict(available_anchors)
        for operand in op.operands:
            child = extracted.get(operand, originals.get(operand))
            if child is not None:
                available[child] = aliases.get(operand, operand)

        def emit(child):
            nonlocal counter
            if child in available:
                return available[child]
            if child.op == "anchor":
                raise IllegalExtraction("anchor does not dominate extracted use")
            operands = tuple(emit(grandchild) for grandchild in child.operands)
            while True:
                fresh = f"%egg{counter}"
                counter += 1
                if fresh not in used_names:
                    break
            used_names.add(fresh)
            added.append(render(fresh, child, operands))
            available[child] = fresh
            return fresh

        if node.op == "anchor":
            aliases[name] = emit(node)
            lines[op.line] = ""
            continue
        operands = tuple(emit(child) for child in node.operands)
        # Retain source spelling only when it represents this extracted tree,
        # and every original operand still denotes the requested child.
        unchanged = node == originals[name] and all(
            (extracted.get(operand, originals.get(operand)) == child)
            for operand, child in zip(op.operands, node.operands))
        lines[op.line] = "".join(added) + (substituted(program.lines[op.line]) if unchanged else render(name, node, operands))
    result = "".join(lines)
    try:
        parse_program(result)
    except UnsupportedProgram as error:
        raise IllegalExtraction(f"decoded program failed validation: {error}") from error
    return result


def pure_roots(program: Program) -> dict[str, Node]:
    nodes = program.nodes
    return {op.results[0]: nodes[op.results[0]] for op in program.operations if op.pure}


def egglog_roundtrip(program: Program) -> tuple[str, dict[str, Node]]:
    """Register, extract and decode the entire pure forest without rewrites."""
    from egglog import EGraph
    from atlas_egglog_schema import from_egglog, to_egglog
    graph = EGraph()
    roots = pure_roots(program)
    expressions = {name: to_egglog(node) for name, node in roots.items()}
    if expressions:
        graph.register(*expressions.values())
    extracted = {name: from_egglog(graph.extract(expr)) for name, expr in expressions.items()}
    return decode_program(program, extracted), extracted
