"""schedule-atlas-virtual: pre-allocation list scheduling of virtual Atlas SSA.

The scheduled order verifies, keeps every block's structure, lowers whenever
the source order does, keeps every SSA and implicit-state dependence, and
never needs more registers than the allocator has unless the source order
already did.
"""

from __future__ import annotations

import re
import unittest

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


# Control-flow shapes around block boundaries. Each is one function with the
# lowering ABI, so lowering parity can be checked.
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
        unary("r", "x"), unary("q", "r"), add("z", "q", "y"),
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


def wide_return_block(tiles: int, interleaved: bool = False) -> str:
    """Entry passes `tiles` BF16 tiles to the return block, which uses each
    twice. Source order computes everything before writing anything, or,
    interleaved, finishes each tile before starting the next."""
    lines = [start()]
    for i in range(tiles):
        lines.append(inp(f"io{i}", f"io{i + 1}", f"t{i}", i))
    lines.append("cf.br " + edge("bb1", [f"io{tiles}", *(f"t{i}" for i in range(tiles))],
                                 [S, *([T] * tiles)]))
    lines.append(f"^bb1(%s0: {S}, " + ", ".join(f"%a{i}: {T}" for i in range(tiles)) + "):")
    relus = [unary(f"r{i}", f"a{i}") for i in range(tiles)]
    adds = [add(f"y{i}", f"r{i}", f"a{i}") for i in range(tiles)]
    outputs = [outp(f"s{i}", f"s{i + 1}", f"y{i}", i) for i in range(tiles)]
    if interleaved:
        lines += [line for tile in zip(relus, adds, outputs) for line in tile]
    else:
        lines += relus + adds + outputs
    lines.append(f"return %s{tiles} : {S}")
    return function("wide", lines, attrs=(
        "atlas.input_dram_base = 2415919104 : i64, "
        "atlas.output_dram_base = 2416050176 : i64"))


def boundary_tiles(count: int, fp8: bool = False) -> str:
    """`count` tiles, each read, rectified, and written before the next is
    read. An unused FP8 input selects the allocator's mixed FP8 capacity."""
    lines = [start()]
    state = "io0"
    if fp8:
        lines.append(inp(state, "f", "x", count, "fp8"))
        state = "f"
    for i in range(count):
        lines += [inp(state, f"i{i}", f"t{i}", i), unary(f"r{i}", f"t{i}"),
                  outp(f"i{i}", f"o{i}", f"r{i}", i)]
        state = f"o{i}"
    lines.append(f"return %{state} : {S}")
    return function("tiles", lines)


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


def positions(ops: list[str], name: str) -> list[int]:
    return [i for i, op in enumerate(ops) if op == f"atlas.virtual_{name}"]


def line_of(source: str, text: str) -> int:
    [line] = [i + 1 for i, row in enumerate(source.splitlines()) if text in row]
    return line


def scheduled_lines(source: str, *options: str) -> list[int]:
    """The source line of each operation, in scheduled order."""
    result = schedule(source, *options, "--mlir-print-debuginfo")
    assert result.returncode == 0, result.stderr
    return [int(line) for line in re.findall(r'loc\("<stdin>":(\d+):\d+\)', result.stdout)]


def schedule(source: str, *options: str):
    """Run the pass; options starting with "--" go to atlas-opt itself."""
    joined = " ".join(o for o in options if not o.startswith("--"))
    flag = f"--schedule-atlas-virtual={joined}" if joined else "--schedule-atlas-virtual"
    printing = [o for o in options if o.startswith("--")]
    if printing:
        printing.append("--mlir-print-local-scope")
    return run("atlas-opt", source, flag, *printing)


def lowers(source: str) -> bool:
    return run("atlas-opt", source, "--lower-atlas-virtual-to-machine").returncode == 0


class VirtualSchedulingTest(unittest.TestCase):
    def setUp(self) -> None:
        self.assertTrue((BIN / "atlas-opt").is_file(), "build atlas-opt first")

    def scheduled(self, source: str, *options: str) -> str:
        result = schedule(source, *options)
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
            self.assertIn('atlas.timing_state = "untimed"', lowered.stdout)
            self.assertNotIn('"atlas.delay"', lowered.stdout)
            timed = run("atlas-opt", lowered.stdout, "--insert-atlas-delays",
                        "--verify-atlas-timing", "--verify-atlas-generated-schedule")
            self.assertEqual(timed.returncode, 0, f"{name}: {timed.stderr}")
            emitted = run("atlas-emit", timed.stdout)
            self.assertEqual(emitted.returncode, 0, f"{name}: {emitted.stderr}")
        return out

    def test_existing_fixtures_schedule_verify_and_lower(self) -> None:
        for path in sorted(EXAMPLES.glob("virtual_*.mlir")):
            with self.subTest(fixture=path.name):
                self.assert_preserves_program(path.read_text(), path.name)

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
            dma_load("io2", "io3", "e3", "addr", "size", "fp8"),
            dma_await("io3", "io4", "x", "e1", "fp8"),
            dma_await("io4", "io5", "y", "e2", "fp8"),
            dma_await("io5", "io6", "z", "e3", "fp8"),
            f"return %io6 : {S}"])
        result = schedule(source)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("must complete a pending DMA before another launch",
                      result.stderr)

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

    def test_boundary_tiles_overlap_vpu_work_with_transfers(self) -> None:
        source = boundary_tiles(4)
        ops = blocks(self.scheduled(source))[0]["ops"]
        second_input = positions(ops, "input_bf16")[1]
        self.assertEqual(second_input, positions(ops, "vpu_unary")[0] + 1,
                         "the next tile is read while the first is rectified")
        self.assertGreater(positions(ops, "output_bf16")[0], second_input)

    def test_schedule_stays_within_register_capacity(self) -> None:
        # Reading ahead stops at the allocator's capacity: 31 BF16 pairs, or
        # 15 with FP8 values present.
        for count, fp8 in ((30, False), (31, False), (32, False),
                           (14, True), (15, True), (16, True)):
            with self.subTest(count=count, fp8=fp8):
                source = boundary_tiles(count, fp8)
                result = schedule(source)
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertNotEqual(blocks(result.stdout)[0]["ops"], blocks(source)[0]["ops"])
                self.assertNotIn("kept source order", result.stderr)
                self.assertTrue(lowers(result.stdout))

    def test_scheduling_recovers_a_block_over_capacity(self) -> None:
        source = virtual_pressure(32)
        rejected = run("atlas-opt", source, "--lower-atlas-virtual-to-machine")
        self.assertIn("exceeds 31 physical pairs", rejected.stderr)
        self.assertTrue(lowers(self.scheduled(source)), "scheduled order should allocate")

    def test_schedule_the_allocator_rejects_keeps_source_order(self) -> None:
        # Within capacity, the allocator's greedy coloring can still fail on
        # an order whose source allocated.
        source = wide_return_block(27, interleaved=True)
        self.assertTrue(lowers(source))
        result = schedule(source)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("kept source order for @wide: 'func.func' op virtual BF16 "
                      "interference exceeds 31 physical pairs", result.stderr)
        self.assertEqual(result.stdout, run("atlas-opt", source).stdout)

    def test_dma_transfers_overlap_and_receive_independent_work(self) -> None:
        def launch(op: str) -> bool:
            return "dma_load" in op or "dma_store" in op

        def completion(op: str) -> bool:
            return "dma_await" in op or op == "atlas.virtual_dma_wait"

        ops = blocks(self.scheduled(OVERLAP.read_text()))[0]["ops"]
        pending = peak = 0
        moved = []
        for op in ops:
            if launch(op):
                pending += 1
                peak = max(peak, pending)
            elif completion(op):
                pending -= 1
            elif pending and ("mxu" in op or "vpu" in op):
                moved.append(op)
        self.assertEqual(peak, 2, "the next transfer launches before the current completes")
        self.assertIn("atlas.virtual_mxu_reset", moved)

    def test_stream_scheduling_keeps_every_wait_before_a_pack(self) -> None:
        # A store and a load are pending and complete load first. The PACK
        # after both, which no transfer may overlap, stays after each wait,
        # not only the last.
        source = function("pack", [
            start(), inp("io0", "io1", "t", 0), const("a", -2147483648), const("b", -2147481600),
            const("size", 2048),
            f'%io2, %st = "atlas.virtual_dma_store_bf16"(%io1, %t, %a, %size) '
            f': ({S}, {T}, i32, i32) -> ({S}, !atlas.virtual_dma_store)',
            dma_load("io2", "io3", "eb", "b", "size", "bf16"),
            dma_await("io3", "io4", "tb", "eb", "bf16"),
            f'%io5 = "atlas.virtual_dma_wait"(%io4, %st) : ({S}, !atlas.virtual_dma_store) -> {S}',
            f'%p = "atlas.virtual_pack_fp8"(%t) {{scale_code = 127 : i32}} : ({T}) -> {F}',
            outp("io5", "o", "tb", 0), f"return %o : {S}"])
        lowered = run("atlas-opt", source, "--lower-atlas-virtual-to-machine")
        self.assertEqual(lowered.returncode, 0, lowered.stderr)
        timed = run("atlas-opt", lowered.stdout, "--schedule-atlas-stream", "--verify-atlas-timing")
        self.assertEqual(timed.returncode, 0, timed.stderr)

    def test_transfers_that_may_share_dram_never_overlap(self) -> None:
        # A store to y, then a load of y and a load of z. The load of y waits
        # for the store; the load of z may run beside either.
        source = function("dram", [
            start(), inp("io0", "io1", "t", 0),
            const("y", -2147481600), const("z", -2147479552), const("size", 2048),
            f'%io2, %st = "atlas.virtual_dma_store_bf16"(%io1, %t, %y, %size) '
            f': ({S}, {T}, i32, i32) -> ({S}, !atlas.virtual_dma_store)',
            f'%io3 = "atlas.virtual_dma_wait"(%io2, %st) : ({S}, !atlas.virtual_dma_store) -> {S}',
            dma_load("io3", "io4", "ey", "y", "size", "bf16"),
            dma_await("io4", "io5", "u", "ey", "bf16"),
            dma_load("io5", "io6", "ez", "z", "size", "bf16"),
            dma_await("io6", "io7", "v", "ez", "bf16"),
            add("w", "u", "v"), outp("io7", "o", "w", 0), f"return %o : {S}"])
        role = {name: line_of(source, text) for name, text in (
            ("store y", "%st ="), ("wait y", "%io3 ="), ("load y", "%ey ="),
            ("await y", "%u ="), ("load z", "%ez ="))}
        z_overlaps = False
        for options in ([], *([f"random-seed={seed}"] for seed in range(24))):
            position = {line: i for i, line in enumerate(scheduled_lines(source, *options))}
            at = {name: position[line] for name, line in role.items()}
            self.assertLess(at["wait y"], at["load y"], options)
            z_overlaps |= at["load z"] < at["wait y"] or at["load z"] < at["await y"]
        self.assertTrue(z_overlaps, "no order ran the load of z beside another transfer")

    def test_computed_dram_addresses_are_as_exact_as_constants(self) -> None:
        # The scheduler proves addresses built by arith.addi as the verifier
        # does: after a store to y, a load of y computed as (y - 2048) + 2048
        # waits for the store, while a load of z computed as z + 0 may run
        # beside it.
        def store_then_load(base: int, offset: int) -> str:
            return function("computed", [
                start(), inp("io0", "io1", "t", 0), const("y", -2147481600),
                const("base", base), const("offset", offset),
                "%address = arith.addi %base, %offset : i32", const("size", 2048),
                f'%io2, %st = "atlas.virtual_dma_store_bf16"(%io1, %t, %y, %size) '
                f': ({S}, {T}, i32, i32) -> ({S}, !atlas.virtual_dma_store)',
                f'%io3 = "atlas.virtual_dma_wait"(%io2, %st) : ({S}, !atlas.virtual_dma_store) -> {S}',
                dma_load("io3", "io4", "e", "address", "size", "bf16"),
                dma_await("io4", "io5", "v", "e", "bf16"),
                add("w", "t", "v"), outp("io5", "o", "w", 0), f"return %o : {S}"])

        def loads_before_the_wait(source: str) -> list[bool]:
            wait, load = line_of(source, "%io3 ="), line_of(source, "%e =")
            before = []
            for options in ([], *([f"random-seed={seed}"] for seed in range(24))):
                position = {line: i for i, line in enumerate(scheduled_lines(source, *options))}
                before.append(position[load] < position[wait])
            return before

        self.assertNotIn(True, loads_before_the_wait(store_then_load(-2147483648, 2048)))
        self.assertIn(True, loads_before_the_wait(store_then_load(-2147479552, 0)))

    def test_loads_of_the_same_dram_may_overlap(self) -> None:
        # Only a store makes overlapping DRAM a hazard.
        source = function("reread", [
            start(), const("y", -2147481600), const("size", 2048),
            dma_load("io0", "io1", "e1", "y", "size", "bf16"),
            dma_await("io1", "io2", "u", "e1", "bf16"),
            dma_load("io2", "io3", "e2", "y", "size", "bf16"),
            dma_await("io3", "io4", "v", "e2", "bf16"),
            add("w", "u", "v"), outp("io4", "o", "w", 0), f"return %o : {S}"])
        first_await, second_load = line_of(source, "%u ="), line_of(source, "%e2 =")
        overlapped = False
        for options in ([], *([f"random-seed={seed}"] for seed in range(16))):
            position = {line: i for i, line in enumerate(scheduled_lines(source, *options))}
            overlapped |= position[second_load] < position[first_await]
        self.assertTrue(overlapped, "no order ran the two loads together")

    def test_legacy_matmul_keeps_its_place_among_a_units_chains(self) -> None:
        # The one-op matmul uses fixed slots of its unit, so no explicit MXU
        # operation on that unit crosses it.
        legacy = (f'%legacy = "atlas.virtual_mxu_matmul"(%x, %w2) {{unit = 0 : i32}} '
                  f': ({F}, {F}) -> {T}')
        source = function("legacy", [
            start(), inp("io0", "io1", "x", 0, "fp8"), inp("io1", "io2", "w1", 1, "fp8"),
            inp("io2", "io3", "w2", 2, "fp8"),
            *mxu_chain("io3", 0, "w1", "x", "h1", "a"), legacy,
            *mxu_chain("as3", 0, "w2", "x", "h2", "b"),
            add("y1", "h1", "h2"), add("y", "y1", "legacy"), outp("bs3", "o", "y", 0),
            f"return %o : {S}"])
        before = [line_of(source, text) for text in ("%aw =", "%aa =", "%h1 =")]
        after = [line_of(source, text) for text in ("%bw =", "%ba =", "%h2 =")]
        matmul = line_of(source, "%legacy =")
        for options in ([], *([f"random-seed={seed}"] for seed in range(16))):
            position = {line: i for i, line in enumerate(scheduled_lines(source, *options))}
            self.assertTrue(all(position[line] < position[matmul] for line in before), options)
            self.assertTrue(all(position[matmul] < position[line] for line in after), options)
        self.assertTrue(lowers(self.scheduled(source)))

    def test_loop_body_dma_interval_receives_vpu_work(self) -> None:
        body = blocks(self.scheduled(cfg_cases()["C14 explicit DMA in a loop body"]))[2]["ops"]
        load = body.index("atlas.virtual_dma_load_bf16")
        await_ = body.index("atlas.virtual_dma_await_bf16")
        self.assertTrue(any(op == "atlas.virtual_vpu_unary"
                            for op in body[load + 1:await_]), body)

    def test_pack_never_enters_a_dma_interval(self) -> None:
        source = cfg_cases()["C16 pack beside an explicit DMA"]
        for seed in [None, *SEEDS]:
            options = [] if seed is None else [f"random-seed={seed}"]
            ops = blocks(self.scheduled(source, *options))[0]["ops"]
            [load] = positions(ops, "dma_load_fp8")
            [await_] = positions(ops, "dma_await_fp8")
            [pack] = positions(ops, "pack_fp8")
            self.assertFalse(load < pack < await_, (seed, ops))

    def test_mxu_slot_dependences_are_exact(self) -> None:
        # Three chains on MXU0. Two weights and two accumulators fit at once,
        # so the second chain may start before the first is read out; the
        # third reuses the first chain's slots and waits for them.
        source = function("weights", [
            start(), inp("io0", "io1", "x", 0, "fp8"), inp("io1", "io2", "w1", 1, "fp8"),
            inp("io2", "io3", "w2", 2, "fp8"), inp("io3", "io4", "w3", 3, "fp8"),
            *mxu_chain("io4", 0, "w1", "x", "h1", "a"),
            *mxu_chain("as3", 0, "w2", "x", "h2", "b"),
            *mxu_chain("bs3", 0, "w3", "x", "h3", "c"),
            add("y1", "h1", "h2"), add("y", "y1", "h3"), outp("cs3", "o", "y", 0),
            f"return %o : {S}"])
        role = {f"{step} {tag}": line_of(source, f"%{tag}{suffix} =")
                for tag in "abc"
                for step, suffix in (("load", "w"), ("reset", "a"))}
        role.update({f"readout {tag}": line_of(source, f"%h{n} =")
                     for n, tag in enumerate("abc", 1)})
        weights_overlap = accumulators_overlap = False
        for seed in range(24):
            position = {line: i for i, line in
                        enumerate(scheduled_lines(source, f"random-seed={seed}"))}
            at = {name: position[line] for name, line in role.items()}
            self.assertLess(at["reset a"], at["load c"], seed)
            self.assertLess(at["readout a"], at["reset c"], seed)
            weights_overlap |= at["load b"] < at["reset a"]
            accumulators_overlap |= at["reset b"] < at["readout a"]
        self.assertTrue(weights_overlap, "no order loaded the second weight early")
        self.assertTrue(accumulators_overlap, "no order kept two accumulators live")

    def test_second_weight_loads_while_the_first_chain_computes(self) -> None:
        source = function("double", [
            start(), inp("io0", "io1", "x", 0, "fp8"), inp("io1", "io2", "w1", 1, "fp8"),
            inp("io2", "io3", "w2", 2, "fp8"),
            *mxu_chain("io3", 0, "w1", "x", "h1", "a"),
            *mxu_chain("as3", 0, "w2", "x", "h2", "b"),
            add("y", "h1", "h2"), outp("bs3", "o", "y", 0), f"return %o : {S}"])
        position = {line: i for i, line in enumerate(scheduled_lines(source))}
        self.assertLess(position[line_of(source, "%bw =")], position[line_of(source, "%aa =")])
        self.assertTrue(lowers(self.scheduled(source)))

    def test_scheduling_is_deterministic(self) -> None:
        for source in (OVERLAP.read_text(), cfg_cases()["C14 explicit DMA in a loop body"],
                       (EXAMPLES / "virtual_fp8_two_layer_mlp_bias.mlir").read_text()):
            self.assertEqual(self.scheduled(source), self.scheduled(source))


if __name__ == "__main__":
    unittest.main()
