"""schedule-atlas-virtual: pre-allocation list scheduling of virtual Atlas SSA.

Each test exercises one guarantee from docs/virtual-scheduler-plan.md: the
scheduled order verifies, keeps every block's structure, lowers whenever the
source order does, respects the dependence model, and models no worse than
the source order.
"""

from __future__ import annotations

import json
import re
import tempfile
import unittest
from pathlib import Path

from test_virtual_lowering import virtual_pressure
from test_virtual_ssa import BIN, ROOT, run


EXAMPLES = ROOT / "test/examples"
OVERLAP = EXAMPLES / "virtual_sched_overlap.mlir"
S = "!atlas.virtual_state"
T = "!atlas.virtual_bf16"
F = "!atlas.virtual_fp8"
# Room for 32 input tiles before the output window.
ABI = ("atlas.input_dram_base = 2415919104 : i64, "
       "atlas.output_dram_base = 2415984640 : i64")
SEEDS = range(8)


# --- MLIR builders -----------------------------------------------------------

def function(name: str, lines: list[str], args: str = "", attrs: str = ABI) -> str:
    return "\n".join([
        "module {",
        f"  func.func @{name}({args}) -> {S} attributes {{{attrs}}} {{",
        *("    " + line if not line.startswith("^") else "  " + line
          for line in lines),
        "  }",
        "}",
    ])


def start(out: str = "io0") -> str:
    return f'%{out} = "atlas.virtual_start"() : () -> {S}'


def inp(state: str, out: str, value: str, index: int, fmt: str = "bf16") -> str:
    kind = T if fmt == "bf16" else F
    return (f'%{out}, %{value} = "atlas.virtual_input_{fmt}"(%{state}) '
            f'{{index = {index} : i32}} : ({S}) -> ({S}, {kind})')


def outp(state: str, out: str, value: str, index: int) -> str:
    return (f'%{out} = "atlas.virtual_output_bf16"(%{state}, %{value}) '
            f'{{index = {index} : i32}} : ({S}, {T}) -> {S}')


def unary(dst: str, src: str, kind: str = "relu") -> str:
    return (f'%{dst} = "atlas.virtual_vpu_unary"(%{src}) {{kind = "{kind}"}} '
            f': ({T}) -> {T}')


def add(dst: str, lhs: str, rhs: str) -> str:
    return (f'%{dst} = "atlas.virtual_vpu_binary"(%{lhs}, %{rhs}) '
            f'{{kind = "add"}} : ({T}, {T}) -> {T}')


def const(name: str, value: int, width: int = 32) -> str:
    return f"%{name} = arith.constant {value} : i{width}"


def edge(dest: str, values: list[str], types: list[str]) -> str:
    return f"^{dest}({', '.join('%' + v for v in values)} : {', '.join(types)})"


def dma_load(state: str, out: str, event: str, addr: str, size: str, fmt: str) -> str:
    return (f'%{out}, %{event} = "atlas.virtual_dma_load_{fmt}"(%{state}, '
            f'%{addr}, %{size}) : ({S}, i32, i32) -> ({S}, '
            f'!atlas.virtual_dma_load_{fmt})')


def dma_await(state: str, out: str, value: str, event: str, fmt: str) -> str:
    kind = T if fmt == "bf16" else F
    return (f'%{out}, %{value} = "atlas.virtual_dma_await_{fmt}"(%{state}, '
            f'%{event}) : ({S}, !atlas.virtual_dma_load_{fmt}) -> ({S}, {kind})')


def mxu_chain(state: str, unit: int, weight: str, activation: str, result: str,
              tag: str) -> list[str]:
    w = f"!atlas.virtual_mxu_weight<{unit}>"
    a = f"!atlas.virtual_mxu_acc<{unit}>"
    return [
        f'%{tag}s1, %{tag}w = "atlas.virtual_mxu_load_weight"(%{state}, '
        f'%{weight}) {{unit = {unit} : i32}} : ({S}, {F}) -> ({S}, {w})',
        f'%{tag}s2, %{tag}a = "atlas.virtual_mxu_reset"(%{tag}s1, '
        f'%{activation}, %{tag}w) : ({S}, {F}, {w}) -> ({S}, {a})',
        f'%{tag}s3, %{result} = "atlas.virtual_mxu_readout_bf16"(%{tag}s2, '
        f'%{tag}a) : ({S}, {a}) -> ({S}, {T})',
    ]


# Control-flow shapes from the design's case table (§6.8). Each is one
# function with the lowering ABI, so lowering parity can be checked.
def cfg_cases() -> dict[str, str]:
    cases: dict[str, str] = {}
    cases["C1 single block"] = function("c1", [
        start(), inp("io0", "io1", "t", 0), unary("r", "t"),
        outp("io1", "io2", "r", 0), f"return %io2 : {S}"])
    cases["C2 entry branches to return block"] = function("c2", [
        start(), inp("io0", "io1", "t", 0), unary("r", "t"),
        "cf.br " + edge("bb1", ["io1", "r"], [S, T]),
        f"^bb1(%s: {S}, %v: {T}):", unary("w", "v"),
        outp("s", "o", "w", 0), f"return %o : {S}"])
    cases["C4 both edges to one block"] = function("c4", [
        start(), inp("io0", "io1", "a", 0), inp("io1", "io2", "b", 1),
        const("c", 1, 1),
        "cf.cond_br %c, " + edge("bb1", ["io2", "a"], [S, T]) + ", "
        + edge("bb1", ["io2", "b"], [S, T]),
        f"^bb1(%s: {S}, %v: {T}):", unary("w", "v"),
        outp("s", "o", "w", 0), f"return %o : {S}"])
    cases["C5 condition from a dominating block"] = function("c5", [
        start(), inp("io0", "io1", "t", 0), const("c0", 0), const("c1", 1),
        "%c = arith.cmpi slt, %c0, %c1 : i32",
        "cf.br " + edge("bb1", ["io1", "t"], [S, T]),
        f"^bb1(%s1: {S}, %u: {T}):", unary("r", "u"),
        "cf.cond_br %c, " + edge("bb2", ["s1", "r"], [S, T]) + ", "
        + edge("bb3", ["s1", "u"], [S, T]),
        f"^bb2(%s2: {S}, %v: {T}):", "cf.br " + edge("bb4", ["s2", "v"], [S, T]),
        f"^bb3(%s3: {S}, %w: {T}):", unary("m", "w", "mov"),
        "cf.br " + edge("bb4", ["s3", "m"], [S, T]),
        f"^bb4(%s4: {S}, %x: {T}):", outp("s4", "o", "x", 0),
        f"return %o : {S}"])
    cases["C6 value used only by the terminator"] = function("c6", [
        start(), inp("io0", "io1", "t", 0), const("c0", 0), const("c1", 1),
        const("c2", 2),
        "cf.br " + edge("bb1", ["io1", "t", "c0"], [S, T, "i32"]),
        f"^bb1(%s1: {S}, %a: {T}, %i: i32):",
        "%more = arith.cmpi slt, %i, %c2 : i32",
        "cf.cond_br %more, " + edge("bb2", ["s1", "a", "i"], [S, T, "i32"])
        + ", " + edge("bb3", ["s1", "a"], [S, T]),
        f"^bb2(%s2: {S}, %b: {T}, %j: i32):",
        "%next = arith.addi %j, %c1 : i32", unary("r1", "b"), unary("r2", "r1"),
        "cf.br " + edge("bb1", ["s2", "r2", "next"], [S, T, "i32"]),
        f"^bb3(%s3: {S}, %c: {T}):", outp("s3", "o", "c", 0),
        f"return %o : {S}"])
    cases["C7 live-through value"] = function("c7", [
        start(), inp("io0", "io1", "t", 0), inp("io1", "io2", "u", 1),
        "cf.br " + edge("bb1", ["io2", "t"], [S, T]),
        f"^bb1(%s1: {S}, %a: {T}):", unary("r", "a"),
        "cf.br " + edge("bb2", ["s1", "r"], [S, T]),
        f"^bb2(%s2: {S}, %b: {T}):", add("x", "b", "u"),
        outp("s2", "o", "x", 0), f"return %o : {S}"])
    cases["C11 nested loops"] = function("c11", [
        start(), inp("io0", "io1", "t", 0), const("c0", 0), const("c1", 1),
        const("c2", 2),
        "cf.br " + edge("bb1", ["io1", "t", "c0"], [S, T, "i32"]),
        f"^bb1(%s1: {S}, %a: {T}, %i: i32):",
        "%mi = arith.cmpi slt, %i, %c2 : i32",
        "cf.cond_br %mi, " + edge("bb2", ["s1", "a", "c0", "i"], [S, T, "i32", "i32"])
        + ", " + edge("bb5", ["s1", "a"], [S, T]),
        f"^bb2(%s2: {S}, %b: {T}, %j: i32, %i2: i32):",
        "%mj = arith.cmpi slt, %j, %c2 : i32",
        "cf.cond_br %mj, " + edge("bb3", ["s2", "b", "j", "i2"], [S, T, "i32", "i32"])
        + ", " + edge("bb4", ["s2", "b", "i2"], [S, T, "i32"]),
        f"^bb3(%s3: {S}, %x: {T}, %k: i32, %i3: i32):", unary("y", "x"),
        "%k1 = arith.addi %k, %c1 : i32",
        "cf.br " + edge("bb2", ["s3", "y", "k1", "i3"], [S, T, "i32", "i32"]),
        f"^bb4(%s4: {S}, %z: {T}, %i4: i32):", "%i5 = arith.addi %i4, %c1 : i32",
        "cf.br " + edge("bb1", ["s4", "z", "i5"], [S, T, "i32"]),
        f"^bb5(%s5: {S}, %w: {T}):", outp("s5", "o", "w", 0),
        f"return %o : {S}"])
    cases["C12 loop without an exit path"] = function("c12", [
        start(), inp("io0", "io1", "t", 0), const("c", 0, 1),
        "cf.cond_br %c, " + edge("bb1", ["io1", "t"], [S, T]) + ", "
        + edge("bb2", ["io1", "t"], [S, T]),
        f"^bb1(%s1: {S}, %a: {T}):", unary("r", "a"),
        "cf.br " + edge("bb1", ["s1", "r"], [S, T]),
        f"^bb2(%s2: {S}, %b: {T}):", outp("s2", "o", "b", 0),
        f"return %o : {S}"])
    cases["C13 terminator-only block"] = function("c13", [
        start(), inp("io0", "io1", "t", 0),
        "cf.br " + edge("bb1", ["io1", "t"], [S, T]),
        f"^bb1(%s1: {S}, %a: {T}):", "cf.br " + edge("bb2", ["s1", "a"], [S, T]),
        f"^bb2(%s2: {S}, %b: {T}):", unary("r", "b"),
        outp("s2", "o", "r", 0), f"return %o : {S}"])
    cases["C14 explicit DMA in a loop body"] = function("c14", [
        start(), inp("io0", "io1", "t", 0), const("addr", -1879044096),
        const("size", 2048), const("c0", 0), const("c1", 1), const("c2", 2),
        "cf.br " + edge("bb1", ["io1", "t", "c0"], [S, T, "i32"]),
        f"^bb1(%s1: {S}, %a: {T}, %i: i32):",
        "%more = arith.cmpi slt, %i, %c2 : i32",
        "cf.cond_br %more, " + edge("bb2", ["s1", "a", "i"], [S, T, "i32"])
        + ", " + edge("bb3", ["s1", "a"], [S, T]),
        f"^bb2(%s2: {S}, %x: {T}, %j: i32):",
        dma_load("s2", "l1", "ev", "addr", "size", "bf16"),
        dma_await("l1", "l2", "y", "ev", "bf16"),
        unary("r", "x"), add("z", "r", "y"),
        "%j1 = arith.addi %j, %c1 : i32",
        "cf.br " + edge("bb1", ["l2", "z", "j1"], [S, T, "i32"]),
        f"^bb3(%s3: {S}, %w: {T}):", outp("s3", "o", "w", 0),
        f"return %o : {S}"])
    cases["C15 MXU chain in a loop body"] = function("c15", [
        start(), inp("io0", "io1", "x", 0, "fp8"), inp("io1", "io2", "w", 1, "fp8"),
        inp("io2", "io3", "t", 2), const("c0", 0), const("c1", 1), const("c2", 2),
        "cf.br " + edge("bb1", ["io3", "t", "c0"], [S, T, "i32"]),
        f"^bb1(%s1: {S}, %a: {T}, %i: i32):",
        "%more = arith.cmpi slt, %i, %c2 : i32",
        "cf.cond_br %more, " + edge("bb2", ["s1", "a", "i"], [S, T, "i32"])
        + ", " + edge("bb3", ["s1", "a"], [S, T]),
        f"^bb2(%s2: {S}, %b: {T}, %j: i32):",
        *mxu_chain("s2", 0, "w", "x", "h", "m"),
        add("z", "h", "b"), "%j1 = arith.addi %j, %c1 : i32",
        "cf.br " + edge("bb1", ["ms3", "z", "j1"], [S, T, "i32"]),
        f"^bb3(%s3: {S}, %c: {T}):", outp("s3", "o", "c", 0),
        f"return %o : {S}"])
    cases["C16 pack beside an explicit DMA"] = function("c16", [
        start(), inp("io0", "io1", "t", 0), const("addr", -1879044096),
        const("size", 1024),
        dma_load("io1", "l1", "ev", "addr", "size", "fp8"),
        dma_await("l1", "l2", "x", "ev", "fp8"),
        f'%p = "atlas.virtual_pack_fp8"(%t) {{scale_code = 127 : i32}} : ({T}) -> {F}',
        *mxu_chain("l2", 0, "p", "x", "h", "m"),
        outp("ms3", "o", "h", 0), f"return %o : {S}"])
    cases["C17 return block with output and explicit store"] = function("c17", [
        start(), inp("io0", "io1", "t", 0), const("addr", -1879035904),
        const("size", 2048), unary("r", "t"), outp("io1", "io2", "r", 0),
        f'%io3, %st = "atlas.virtual_dma_store_bf16"(%io2, %t, %addr, %size) '
        f': ({S}, {T}, i32, i32) -> ({S}, !atlas.virtual_dma_store)',
        f'%io4 = "atlas.virtual_dma_wait"(%io3, %st) : ({S}, '
        f'!atlas.virtual_dma_store) -> {S}',
        f"return %io4 : {S}"])
    cases["C18 unused function arguments"] = function(
        "c18", [start(), inp("io0", "io1", "t", 0), unary("r", "t"),
                outp("io1", "io2", "r", 0), f"return %io2 : {S}"],
        args="%a: i32, %b: i1",
        attrs=ABI + ", atlas.control_dram_base = 2415927296 : i64")
    cases["C21 edges passing different tile counts"] = function("c21", [
        start(), inp("io0", "io1", "a", 0), inp("io1", "io2", "b", 1),
        const("c", 1, 1),
        "cf.cond_br %c, " + edge("bb1", ["io2", "a", "b"], [S, T, T]) + ", "
        + edge("bb2", ["io2", "a"], [S, T]),
        f"^bb1(%s1: {S}, %x: {T}, %y: {T}):", add("z", "x", "y"),
        "cf.br " + edge("bb3", ["s1", "z"], [S, T]),
        f"^bb2(%s2: {S}, %w: {T}):", "cf.br " + edge("bb3", ["s2", "w"], [S, T]),
        f"^bb3(%s3: {S}, %v: {T}):", outp("s3", "o", "v", 0),
        f"return %o : {S}"])
    return cases


def wide_return_block(tiles: int) -> str:
    """Entry passes `tiles` BF16 tiles to the return block, which uses each
    twice. Source order computes everything before writing anything."""
    lines = [start()]
    for i in range(tiles):
        lines.append(inp(f"io{i}", f"io{i + 1}", f"t{i}", i))
    lines.append("cf.br " + edge("bb1", [f"io{tiles}", *(f"t{i}" for i in range(tiles))],
                                 [S, *([T] * tiles)]))
    lines.append(f"^bb1(%s0: {S}, " + ", ".join(f"%a{i}: {T}" for i in range(tiles)) + "):")
    lines += [unary(f"r{i}", f"a{i}") for i in range(tiles)]
    lines += [add(f"y{i}", f"r{i}", f"a{i}") for i in range(tiles)]
    lines += [outp(f"s{i}", f"s{i + 1}", f"y{i}", i) for i in range(tiles)]
    lines.append(f"return %s{tiles} : {S}")
    return function("wide", lines, attrs=(
        "atlas.input_dram_base = 2415919104 : i64, "
        "atlas.output_dram_base = 2416050176 : i64"))


# --- IR inspection -----------------------------------------------------------

GENERIC_OP = re.compile(r'"([a-z_]+\.[a-z_0-9.]+)"\(')


def blocks(text: str) -> list[dict]:
    """Each function's blocks in MLIR's generic form: argument types, op
    names in order, and terminator successors."""
    generic = run("atlas-opt", text, "--mlir-print-op-generic")
    assert generic.returncode == 0, generic.stderr
    result: list[dict] = []
    current: dict | None = None
    for line in generic.stdout.splitlines():
        stripped = line.strip()
        if stripped.startswith('"func.func"'):
            current = {"header": (), "ops": [], "succ": []}
            result.append(current)
            continue
        if stripped.startswith("^bb"):
            types = tuple(re.findall(r":\s*(![\w.<>]+|i\d+)", stripped.split("//")[0]))
            if current is not None and not current["ops"]:
                current["header"] = types  # the entry block's arguments
            else:
                current = {"header": types, "ops": [], "succ": []}
                result.append(current)
            continue
        match = GENERIC_OP.search(stripped)
        if current is None or not match:
            continue
        current["ops"].append(match.group(1))
        successors = re.search(r"\)\[([^\]]*)\]", stripped)
        if successors:
            current["succ"] = re.findall(r"\^bb\d+", successors.group(1))
    return result


def structure(text: str) -> list[tuple]:
    return [(b["header"], tuple(sorted(b["ops"])), tuple(b["succ"]),
             b["ops"][-1] if b["ops"] else None) for b in blocks(text)]


def reaches(predecessors: list[list[int]], start: int, goal: int) -> bool:
    """Whether the dependence graph orders `start` before `goal`."""
    successors: dict[int, list[int]] = {}
    for node, preds in enumerate(predecessors):
        for pred in preds:
            successors.setdefault(pred, []).append(node)
    stack, seen = [start], set()
    while stack:
        node = stack.pop()
        if node == goal:
            return True
        if node not in seen:
            seen.add(node)
            stack.extend(successors.get(node, []))
    return False


def nodes_named(block: dict, name: str) -> list[int]:
    return [i for i, label in enumerate(block["labels"])
            if label.split(" (")[0] == name]


def schedule(source: str, *options: str):
    joined = " ".join(options)
    flag = f"--schedule-atlas-virtual={joined}" if joined else "--schedule-atlas-virtual"
    return run("atlas-opt", source, flag)


def lowers(source: str) -> bool:
    return run("atlas-opt", source, "--lower-atlas-virtual-to-machine").returncode == 0


class VirtualSchedulingTest(unittest.TestCase):
    def setUp(self) -> None:
        self.assertTrue((BIN / "atlas-opt").is_file(), "build atlas-opt first")

    def scheduled(self, source: str, *options: str) -> str:
        result = schedule(source, "strict=true", *options)
        self.assertEqual(result.returncode, 0, result.stderr)
        checked = run("atlas-opt", result.stdout, "--verify-atlas-virtual-stream")
        self.assertEqual(checked.returncode, 0, checked.stderr)
        return result.stdout

    def assert_preserves_program(self, source: str, name: str) -> str:
        normalized = run("atlas-opt", source)
        self.assertEqual(normalized.returncode, 0, f"{name}: {normalized.stderr}")
        out = self.scheduled(source)
        self.assertEqual(structure(normalized.stdout), structure(out), name)
        found = blocks(out)
        if found and found[0]["ops"]:
            self.assertEqual(found[0]["ops"][0], "atlas.virtual_start", name)
        if lowers(source):
            lowered = run("atlas-opt", out, "--lower-atlas-virtual-to-machine",
                          "--verify-atlas-generated-schedule")
            self.assertEqual(lowered.returncode, 0, f"{name}: {lowered.stderr}")
            emitted = run("atlas-emit", lowered.stdout)
            self.assertEqual(emitted.returncode, 0, f"{name}: {emitted.stderr}")
        return out

    def trace(self, source: str, *options: str) -> dict:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "trace.json"
            self.scheduled(source, f"trace-file={path}", *options)
            return json.loads(path.read_text())

    # §11.1
    def test_existing_fixtures_schedule_verify_and_lower(self) -> None:
        for path in sorted(EXAMPLES.glob("virtual_*.mlir")):
            with self.subTest(fixture=path.name):
                self.assert_preserves_program(path.read_text(), path.name)

    # §11.2
    def test_control_flow_shapes_keep_structure_and_lower(self) -> None:
        for name, source in cfg_cases().items():
            with self.subTest(case=name):
                self.assertEqual(run("atlas-opt", source,
                                     "--verify-atlas-virtual-stream").returncode, 0,
                                 name)
                self.assert_preserves_program(source, name)

    def test_module_level_stream_is_unchanged(self) -> None:
        source = (EXAMPLES / "virtual_bf16_ssa.mlir").read_text()
        self.assertEqual(self.scheduled(source), run("atlas-opt", source).stdout)

    def test_unverified_input_is_rejected(self) -> None:
        source = function("bad", [
            start(), const("addr", -1879048192), const("size", 1024),
            dma_load("io0", "io1", "e1", "addr", "size", "fp8"),
            dma_load("io1", "io2", "e2", "addr", "size", "fp8"),
            dma_await("io2", "io3", "x", "e1", "fp8"),
            dma_await("io3", "io4", "y", "e2", "fp8"),
            f"return %io4 : {S}"])
        result = schedule(source)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("must complete the pending DMA before another launch",
                      result.stderr)

    # §11.3
    def test_random_legal_orders_always_verify(self) -> None:
        sources = {p.name: p.read_text() for p in EXAMPLES.glob("virtual_*.mlir")}
        sources.update(cfg_cases())
        for name, source in sources.items():
            for seed in SEEDS:
                with self.subTest(case=name, seed=seed):
                    self.scheduled(source, f"random-seed={seed}")

    def test_random_orders_explore_the_graph(self) -> None:
        source = OVERLAP.read_text()
        orders = {tuple(blocks(self.scheduled(source, f"random-seed={seed}"))[0]["ops"])
                  for seed in range(12)}
        self.assertGreater(len(orders), 4)

    # §11.4
    def test_scheduling_recovers_a_block_over_the_pair_cap(self) -> None:
        source = virtual_pressure(32)
        rejected = run("atlas-opt", source, "--lower-atlas-virtual-to-machine")
        self.assertIn("exceeds 31 physical pairs", rejected.stderr)
        out = self.scheduled(source)
        self.assertTrue(lowers(out), "scheduled order should allocate")
        ops = blocks(out)[0]["ops"]
        first_output = ops.index("atlas.virtual_output_bf16")
        last_input = len(ops) - 1 - ops[::-1].index("atlas.virtual_input_bf16")
        self.assertLess(first_output, last_input, "inputs and outputs interleave")

    def test_return_block_near_the_cap_interleaves(self) -> None:
        source = wide_return_block(29)
        self.assertFalse(lowers(source))
        self.assertTrue(lowers(self.scheduled(source)))

    def test_exit_pressure_over_the_cap_keeps_source_order(self) -> None:
        source = wide_return_block(32)
        result = schedule(source, "strict=true")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("kept source order for block 0: exit BF16 pressure 32 "
                      "exceeds cap 31", result.stderr)
        self.assertIn("kept source order for block 1: entry BF16 pressure 32 "
                      "exceeds cap 31", result.stderr)
        self.assertEqual(structure(run("atlas-opt", source).stdout),
                         structure(result.stdout))

    def test_caps_follow_fp8_and_pack(self) -> None:
        mixed = schedule(OVERLAP.read_text(), "report=true")
        self.assertIn("BF16 4/15", mixed.stderr)
        packed = schedule((EXAMPLES / "virtual_fp8_two_layer_mlp.mlir").read_text(),
                          "report=true")
        self.assertRegex(packed.stderr, r"scalar \d+/9")

    # §11.5
    def test_dma_intervals_receive_independent_work(self) -> None:
        ops = blocks(self.scheduled(OVERLAP.read_text()))[0]["ops"]
        dma = [op for op in ops if "dma" in op]
        source_dma = [op for op in blocks(run("atlas-opt", OVERLAP.read_text()).stdout)[0]["ops"]
                      if "dma" in op]
        self.assertEqual(dma, source_dma, "DMA operations keep their order")
        inside = False
        moved = []
        for op in ops:
            if op.startswith("atlas.virtual_dma_load") or op.startswith("atlas.virtual_dma_store"):
                inside = True
            elif op.startswith("atlas.virtual_dma_await") or op == "atlas.virtual_dma_wait":
                inside = False
            elif inside and ("mxu" in op or "vpu" in op):
                moved.append(op)
        self.assertIn("atlas.virtual_mxu_reset", moved)

    def test_loop_body_dma_interval_receives_vpu_work(self) -> None:
        body = blocks(self.scheduled(cfg_cases()["C14 explicit DMA in a loop body"]))[2]["ops"]
        load = body.index("atlas.virtual_dma_load_bf16")
        await_ = body.index("atlas.virtual_dma_await_bf16")
        self.assertTrue(any(op == "atlas.virtual_vpu_unary"
                            for op in body[load + 1:await_]), body)

    def test_pack_never_enters_a_dma_interval(self) -> None:
        source = cfg_cases()["C16 pack beside an explicit DMA"]
        block = self.trace(source)["functions"][0]["blocks"][0]
        preds = block["predecessors"]
        [load] = nodes_named(block, "dma_load_fp8")
        [await_] = nodes_named(block, "dma_await_fp8")
        [pack] = nodes_named(block, "pack_fp8")
        self.assertTrue(reaches(preds, await_, pack) or reaches(preds, pack, load))

    def test_weight_slot_dependences_are_exact(self) -> None:
        # Two chains on MXU0. A replacement weight load may rise above the
        # previous readout, never above the reset that reads the old weight,
        # and the second chain starts after the first is read out.
        source = function("weights", [
            start(), inp("io0", "io1", "x", 0, "fp8"), inp("io1", "io2", "w1", 1, "fp8"),
            inp("io2", "io3", "w2", 2, "fp8"),
            *mxu_chain("io3", 0, "w1", "x", "h1", "a"),
            *mxu_chain("as3", 0, "w2", "x", "h2", "b"),
            add("y", "h1", "h2"), outp("bs3", "o", "y", 0), f"return %o : {S}"])
        block = self.trace(source)["functions"][0]["blocks"][0]
        preds = block["predecessors"]
        _, load2 = nodes_named(block, "mxu_load_weight u0")
        reset1, reset2 = nodes_named(block, "mxu_reset u0")
        readout1, _ = nodes_named(block, "mxu_readout_bf16 u0")
        self.assertTrue(reaches(preds, reset1, load2))
        self.assertFalse(reaches(preds, readout1, load2))
        self.assertTrue(reaches(preds, readout1, reset2))

    def test_unit_order_is_kept_per_unit(self) -> None:
        def per_unit(text: str) -> dict[str, list[str]]:
            units: dict[str, list[str]] = {}
            for line in text.splitlines():
                op = re.search(r'"atlas\.virtual_mxu_([a-z_0-9]+)"', line)
                unit = re.search(r"virtual_mxu_(?:weight|acc)<(\d)>", line)
                if op and unit:
                    units.setdefault(unit.group(1), []).append(op.group(1))
            return units

        source = OVERLAP.read_text()
        expected = per_unit(run("atlas-opt", source).stdout)
        self.assertEqual(sorted(expected), ["0", "1"])
        for seed in range(12):
            self.assertEqual(per_unit(self.scheduled(source, f"random-seed={seed}")),
                             expected, seed)

    def test_scheduling_is_deterministic_and_idempotent(self) -> None:
        for source in (OVERLAP.read_text(), cfg_cases()["C14 explicit DMA in a loop body"],
                       (EXAMPLES / "virtual_fp8_two_layer_mlp_bias.mlir").read_text()):
            once = self.scheduled(source)
            self.assertEqual(once, self.scheduled(source))
            self.assertEqual(once, self.scheduled(once))

    # §11.6 and the report
    def test_report_and_trace_describe_the_schedule(self) -> None:
        result = schedule(OVERLAP.read_text(), "report=true")
        match = re.search(r"modeled (\d+) cycles in source order, (\d+) scheduled",
                          result.stderr)
        self.assertIsNotNone(match, result.stderr)
        source_cycles, scheduled_cycles = map(int, match.groups())
        self.assertLess(scheduled_cycles, source_cycles)
        data = self.trace(OVERLAP.read_text())
        block = data["functions"][0]["blocks"][0]
        self.assertEqual(block["source"]["cycles"], source_cycles)
        self.assertEqual(block["scheduled"]["cycles"], scheduled_cycles)
        self.assertEqual(sorted(block["scheduled"]["order"]),
                         list(range(len(block["labels"]))))

    def test_trace_file_errors_are_reported(self) -> None:
        result = schedule(OVERLAP.read_text(), "trace-file=/nonexistent/trace.json")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("cannot write trace file", result.stderr)

    def test_functions_the_lowering_rejects_keep_their_order(self) -> None:
        source = function("unlowerable", [
            start(), inp("io0", "io1", "t", 0), unary("r", "t", "exp"),
            outp("io1", "io2", "r", 0), f"return %io2 : {S}"])
        result = schedule(source, "strict=true")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("kept source order for @unlowerable: the lowering cannot "
                      "handle it", result.stderr)
        self.assertIn("admission currently requires mov or relu", result.stderr)
        self.assertEqual(result.stdout, run("atlas-opt", source).stdout)

    def test_model_pressure_over_a_cap_means_allocation_fails(self) -> None:
        # The pressure model may under-approximate what greedy coloring
        # needs, never over-approximate: an order it puts over a cap must
        # fail allocation.
        over = 0
        for source in (virtual_pressure(32), wide_return_block(29)):
            for seed in SEEDS:
                ordered = self.scheduled(source, f"random-seed={seed}")
                for block in self.trace(ordered)["functions"][0]["blocks"]:
                    timeline = block["source"]
                    if timeline and any(p > c for p, c in zip(timeline["peak"], block["caps"])):
                        over += 1
                        self.assertFalse(lowers(ordered), (seed, block["index"]))
        self.assertGreater(over, 0, "some random order must exceed a cap")

    def test_never_models_worse_than_source(self) -> None:
        for path in sorted(EXAMPLES.glob("virtual_*.mlir")):
            for function_trace in self.trace(path.read_text())["functions"]:
                for block in function_trace["blocks"]:
                    with self.subTest(fixture=path.name, block=block["index"]):
                        self.assertLessEqual(block["scheduled"]["cycles"],
                                             block["source"]["cycles"])

    def test_modeled_gaps_match_the_timing_reference(self) -> None:
        # Spec 04 "Tested examples": MXU0 weight push → matmul 1, matmul → pop
        # 64; MXU1 push → matmul 32, matmul → pop 32; a VPU result → reader 66.
        for unit, push_gap, pop_gap in ((0, 1, 64), (1, 32, 32)):
            source = function("gaps", [
                start(), inp("io0", "io1", "x", 0, "fp8"),
                inp("io1", "io2", "w", 1, "fp8"),
                *mxu_chain("io2", unit, "w", "x", "h", "m"),
                unary("r", "h"), unary("q", "r"),
                outp("ms3", "o", "q", 0), f"return %o : {S}"])
            data = self.trace(source)["functions"][0]["blocks"][0]
            start_of = {data["labels"][s["node"]].split(" (")[0]: s["start"]
                        for s in data["source"]["spans"]}
            weight = start_of[f"mxu_load_weight u{unit}"]
            reset = start_of[f"mxu_reset u{unit}"]
            readout = start_of[f"mxu_readout_bf16 u{unit}"]
            with self.subTest(unit=unit):
                self.assertEqual(reset - weight, push_gap)
                self.assertEqual(readout - reset, pop_gap)
            relu_starts = sorted(s["start"] for s in data["source"]["spans"]
                                 if data["labels"][s["node"]].startswith("vpu_unary"))
            self.assertEqual(relu_starts[1] - relu_starts[0], 66)


if __name__ == "__main__":
    unittest.main()
