"""Logical MXU and DMA handle ownership, admission and loop rebinding through the public evaluator.

The two-handle and two-transfer bounds are target-interface admission, not hardware qualification.
"""

from __future__ import annotations

from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
from atlas_virtual_evaluator import MemoryRegion, RuntimeInputs, Scalar, Tile, VirtualInterfaceError, evaluate, parse_program  # noqa: E402


S, B, F = "!atlas.virtual_state", "!atlas.virtual_bf16", "!atlas.virtual_fp8"
W, A = "!atlas.virtual_mxu_weight<0>", "!atlas.virtual_mxu_acc<0>"
PREFIX = f'''
  %s0 = "atlas.virtual_start"() : () -> {S}
  %s1, %x = "atlas.virtual_input_fp8"(%s0) {{index = 0 : i32}} : ({S}) -> ({S}, {F})
  %s2, %w = "atlas.virtual_input_fp8"(%s1) {{index = 1 : i32}} : ({S}) -> ({S}, {F})
  %s3, %v = "atlas.virtual_input_fp8"(%s2) {{index = 2 : i32}} : ({S}) -> ({S}, {F})
  %s4, %seed = "atlas.virtual_input_bf16"(%s3) {{index = 3 : i32}} : ({S}) -> ({S}, {B})
'''


def diagonal(format: str, code: int) -> Tile:
    return Tile(format, tuple(code if row == col else 0 for row in range(32) for col in range(32)))


TILES = {0: diagonal("fp8", 0x38), 1: diagonal("fp8", 0x38), 2: diagonal("fp8", 0x40), 3: diagonal("bf16", 0x4040)}


def weight(before: str, after: str, name: str, source: str = "w") -> str:
    return f'%{after}, %{name} = "atlas.virtual_mxu_load_weight"(%{before}, %{source}) {{unit = 0 : i32}} : ({S}, {F}) -> ({S}, {W})'


def seed(before: str, after: str, name: str, source: str = "seed", format: str = "bf16") -> str:
    return f'%{after}, %{name} = "atlas.virtual_mxu_load_acc_{format}"(%{before}, %{source}) {{unit = 0 : i32}} : ({S}, {B if format == "bf16" else F}) -> ({S}, {A})'


def reset(before: str, after: str, name: str, weights: str = "w0") -> str:
    return f'%{after}, %{name} = "atlas.virtual_mxu_reset"(%{before}, %x, %{weights}) : ({S}, {F}, {W}) -> ({S}, {A})'


def accumulate(before: str, after: str, name: str, previous: str, weights: str = "w0") -> str:
    return f'%{after}, %{name} = "atlas.virtual_mxu_accumulate"(%{before}, %x, %{weights}, %{previous}) : ({S}, {F}, {W}, {A}) -> ({S}, {A})'


def read(before: str, after: str, tile: str, handle: str) -> str:
    return f'%{after}, %{tile} = "atlas.virtual_mxu_readout_bf16"(%{before}, %{handle}) : ({S}, {A}) -> ({S}, {B})'


def output(before: str, after: str, tile: str = "x", index: int = 0) -> str:
    return f'%{after} = "atlas.virtual_output_bf16"(%{before}, %{tile}) {{index = {index} : i32}} : ({S}, {B}) -> {S}'


def on_unit(text: str, unit: int) -> str:
    return text.replace(W, f"!atlas.virtual_mxu_weight<{unit}>").replace(A, f"!atlas.virtual_mxu_acc<{unit}>").replace("unit = 0 : i32", f"unit = {unit} : i32")


def legacy(name: str = "product") -> str:
    return f'%{name} = "atlas.virtual_mxu_matmul"(%x, %w) {{unit = 0 : i32}} : ({F}, {F}) -> {B}'


def function(body: str, arguments: str = "") -> str:
    return f'module {{ func.func @handles({arguments}) -> {S} {{\n{PREFIX}\n{body}\n}} }}'


def program(*body: str, final_state: str, flat: bool = False) -> str:
    text = "\n".join((*body, output(final_state, "done", "seed")))
    return f'module {{\n{PREFIX}\n{text}\n}}' if flat else function(text + f'\nfunc.return %done : {S}')


def kernel(body: tuple[str, ...], final_state: str, results: tuple[str, ...]) -> str:
    operations, current = list(body), final_state
    for index, result in enumerate(results):
        operations.append(output(current, f"out{index}", result, index))
        current = f"out{index}"
    return function("\n".join((*operations, f"func.return %{current} : {S}")))


LOOP_HEADER = f'''
  %zero = arith.constant 0 : i32
  %one = arith.constant 1 : i32
  %limit = arith.constant 2 : i32'''


class VirtualEvaluatorMXUHandleTest(unittest.TestCase):
    def test_admitted_handle_lifetimes_publish_distinct_chains_on_each_unit(self) -> None:
        loop = f'''{LOOP_HEADER}
          cf.br ^loop(%s4, %zero : {S}, i32)
        ^loop(%ls: {S}, %i: i32):
          {weight("ls", "t0", "w0")}
          {reset("t0", "t1", "a0")}
          {read("t1", "t2", "y", "a0")}
          %j = arith.addi %i, %one : i32
          %again = arith.cmpi ult, %j, %limit : i32
          cf.cond_br %again, ^loop(%t2, %j : {S}, i32), ^exit(%t2, %y : {S}, {B})
        ^exit(%es: {S}, %result: {B}):'''
        cases = {
            "two_weights_reverse_readout": ((weight("s4", "t0", "w0"), weight("t0", "t1", "w1", "v"), reset("t1", "t2", "a0"), reset("t2", "t3", "a1", "w1"),
                                             read("t3", "t4", "y1", "a1"), read("t4", "t5", "y0", "a0")), "t5", ("y1", "y0"), (0x4000, 0x3F80)),
            # w0's last use frees admission for w2 while a1 is still live.
            "weight_retired_during_live_chain": ((weight("s4", "t0", "w0"), reset("t0", "t1", "a0"), weight("t1", "t2", "w1", "v"), accumulate("t2", "t3", "a1", "a0"),
                                                  weight("t3", "t4", "w2", "v"), accumulate("t4", "t5", "a2", "a1", "w1"), accumulate("t5", "t6", "a3", "a2", "w2"),
                                                  read("t6", "t7", "y", "a3")), "t7", ("y",), (0x40C0,)),
            "seeds_without_weights": ((seed("s4", "t0", "a0"), seed("t0", "t1", "a1", "v", "fp8"), read("t1", "t2", "y1", "a1"), read("t2", "t3", "y0", "a0")),
                                      "t3", ("y1", "y0"), (0x4000, 0x4040)),
            "seed_keeps_weight_for_reset": ((weight("s4", "t0", "w0"), seed("t0", "t1", "a0"), accumulate("t1", "t2", "a1", "a0"), read("t2", "t3", "first", "a1"),
                                             reset("t3", "t4", "a2"), read("t4", "t5", "second", "a2")), "t5", ("first", "second"), (0x4080, 0x3F80)),
            # Zero-use weights are not resident for admission or legacy overlap.
            "dead_weights_then_legacy": ((weight("s4", "t0", "w0"), weight("t0", "t1", "w1"), weight("t1", "t2", "w2"), legacy()), "t2", ("product",), (0x3F80,)),
            "legacy_after_readout": ((weight("s4", "t0", "w0"), reset("t0", "t1", "a0"), read("t1", "t2", "first", "a0"), legacy()), "t2",
                                     ("first", "product"), (0x3F80, 0x3F80)),
            "loop_fresh_versions": ((loop,), "es", ("result",), (0x3F80,)),
        }
        for unit in (0, 1):
            for name, (body, final, results, codes) in cases.items():
                with self.subTest(unit=unit, case=name):
                    result = evaluate(parse_program(on_unit(kernel(body, final, results), unit)), RuntimeInputs(TILES))
                    self.assertEqual((dict(result.outputs), result.memory), ({index: diagonal("bf16", code) for index, code in enumerate(codes)}, ()))

    def test_invalid_handle_lifetimes_are_rejected_on_each_unit(self) -> None:
        live = (weight("s4", "t0", "w0"), reset("t0", "t1", "a0"))
        chain = (*live, accumulate("t1", "t2", "a1", "a0"))
        cases = {
            "stale_version": ((*chain, read("t2", "t3", "old", "a0"), read("t3", "t4", "current", "a1")), "t4"),
            "forked_version": ((*chain, accumulate("t2", "t3", "fork", "a0"), read("t3", "t4", "first", "a1"), read("t4", "t5", "second", "fork")), "t5"),
            "repeated_readout": ((*live, read("t1", "t2", "first", "a0"), read("t2", "t3", "again", "a0")), "t3"),
            "repeated_seed_readout": ((seed("s4", "t0", "a0"), read("t0", "t1", "first", "a0"), read("t1", "t2", "again", "a0")), "t2"),
            "consumed_accumulation": ((*live, read("t1", "t2", "first", "a0"), accumulate("t2", "t3", "a1", "a0"), read("t3", "t4", "y", "a1")), "t4"),
            "dropped_accumulator": ((seed("s4", "t0", "a0"),), "t0"),
            "legacy_over_live_weight": ((weight("s4", "t0", "w0"), legacy(), reset("t0", "t1", "a0"), read("t1", "t2", "y", "a0")), "t2"),
            "legacy_over_live_accumulator": ((seed("s4", "t0", "a0"), legacy(), read("t0", "t1", "y", "a0")), "t1"),
        }
        captured = f'''{weight("s4", "t0", "w0")}
          cf.br ^next(%t0 : {S})
        ^next(%ns: {S}):
          {reset("ns", "t1", "a0")}
          {read("t1", "t2", "y", "a0")}
          func.return %t2 : {S}'''
        passed = captured.replace(f'^next(%t0 : {S})', f'^next(%t0, %w0 : {S}, {W})').replace(f'%ns: {S}):', f'%ns: {S}, %nw: {W}):').replace('%ns, %x, %w0', '%ns, %x, %nw')
        untaken = f'''
          cf.cond_br %choose, ^good(%s4 : {S}), ^bad(%s4 : {S})
        ^good(%gs: {S}):
          func.return %gs : {S}
        ^bad(%bs: {S}):
          {seed("bs", "t0", "a0")}
          {read("t0", "t1", "first", "a0")}
          {read("t1", "t2", "again", "a0")}
          func.return %t2 : {S}'''
        extra = {
            "captured_across_blocks": function(captured),
            "passed_across_blocks": function(passed),
            "untaken_malformed_path": function(untaken, "%choose: i1"),
        }
        for unit in (0, 1):
            sources = {f"{name}_flat_{flat}": on_unit(program(*body, final_state=final, flat=flat), unit) for name, (body, final) in cases.items() for flat in (False, True)}
            for name, source in (sources | ({} if unit else extra)).items():
                with self.subTest(unit=unit, case=name), self.assertRaises(VirtualInterfaceError):
                    evaluate(parse_program(source), RuntimeInputs(TILES, (Scalar(1, 1),) if name == "untaken_malformed_path" else ()))

    def test_third_live_weight_and_accumulator_are_rejected_on_each_unit(self) -> None:
        weights = (weight("s4", "t0", "w0"), weight("t0", "t1", "w1"), weight("t1", "t2", "w2"),
                   reset("t2", "t3", "a0"), read("t3", "t4", "y0", "a0"),
                   reset("t4", "t5", "a1", "w1"), read("t5", "t6", "y1", "a1"),
                   reset("t6", "t7", "a2", "w2"), read("t7", "t8", "y2", "a2"))
        accumulators = (seed("s4", "t0", "a0"), seed("t0", "t1", "a1"), seed("t1", "t2", "a2"),
                        read("t2", "t3", "y0", "a0"), read("t3", "t4", "y1", "a1"), read("t4", "t5", "y2", "a2"))
        for unit in (0, 1):
            for kind, operations, final in (("weight", weights, "t8"), ("accumulator", accumulators, "t5")):
                parsed = parse_program(on_unit(program(*operations, final_state=final), unit))
                with self.subTest(unit=unit, kind=kind), self.assertRaisesRegex(VirtualInterfaceError, "at most two live MXU"):
                    evaluate(parsed, RuntimeInputs(TILES))


STORE = "!atlas.virtual_dma_store"
ADDRESS, DESTINATION = 0x80000000, 0x80002000
INPUT_BASE, OUTPUT_BASE = 0x90000000, 0x90004000
BF16 = Tile("bf16", (0x3F80,) * 1024)
FP8 = Tile("fp8", (0x38,) * 1024)
RAW = b"\x80\x3f" * 1024
ABI = f"atlas.input_dram_base = {INPUT_BASE} : i64, atlas.output_dram_base = {OUTPUT_BASE} : i64"


def bf16_bytes(bits) -> bytes:
    # Two physical halves of 32 rows x 16 little-endian words; independent of the evaluator's serializers.
    return b"".join(bits[row * 32 + col].to_bytes(2, "little") for half in range(2) for row in range(32) for col in range(half * 16, half * 16 + 16))


def type_for(format: str) -> str:
    return B if format == "bf16" else F


def event(format: str) -> str:
    return f"!atlas.virtual_dma_load_{format}"


def constants(format: str = "bf16", address: int = ADDRESS, size: int | None = None, *, addr: str = "addr", length: str = "size") -> tuple[str, str]:
    return (f"%{addr} = arith.constant {address} : i32",
            f"%{length} = arith.constant {size if size is not None else 2048 if format == 'bf16' else 1024} : i32")


def load(format: str, before: str, after: str, handle: str, addr: str = "addr", size: str = "size") -> str:
    return f'%{after}, %{handle} = "atlas.virtual_dma_load_{format}"(%{before}, %{addr}, %{size}) : ({S}, i32, i32) -> ({S}, {event(format)})'


def ready(format: str, before: str, after: str, handle: str, result: str = "loaded") -> str:
    return f'%{after}, %{result} = "atlas.virtual_dma_await_{format}"(%{before}, %{handle}) : ({S}, {event(format)}) -> ({S}, {type_for(format)})'


def store(format: str, before: str, after: str, handle: str, value: str = "x", addr: str = "addr", size: str = "size") -> str:
    return f'%{after}, %{handle} = "atlas.virtual_dma_store_{format}"(%{before}, %{value}, %{addr}, %{size}) : ({S}, {type_for(format)}, i32, i32) -> ({S}, {STORE})'


def wait(before: str, after: str, handle: str) -> str:
    return f'%{after} = "atlas.virtual_dma_wait"(%{before}, %{handle}) : ({S}, {STORE}) -> {S}'


def transfer(direction: str, before: str, after: str, handle: str, completed: str, result: str = "loaded") -> tuple[str, str]:
    """BF16 launch from `before` and completion from `completed`."""
    if direction == "load":
        return load("bf16", before, after, handle), ready("bf16", completed, f"{completed}_done", handle, result)
    return store("bf16", before, after, handle), wait(completed, f"{completed}_done", handle)


def input_tile(format: str, before: str = "s0", after: str = "io", value: str = "x", index: int = 0) -> str:
    return f'%{after}, %{value} = "atlas.virtual_input_{format}"(%{before}) {{index = {index} : i32}} : ({S}) -> ({S}, {type_for(format)})'


def wrap(*body: str, final: str, flat: bool = False, arguments: str = "", attributes: str = "") -> str:
    operations = "\n".join((f'%s0 = "atlas.virtual_start"() : () -> {S}', *body))
    if flat:
        return f"module {{\n{operations}\n}}"
    attrs = f" attributes {{{attributes}}}" if attributes else ""
    return f"module {{ func.func @dma({arguments}) -> {S}{attrs} {{\n{operations}\nfunc.return %{final} : {S}\n}} }}"


def read_program(format: str = "bf16", address: int = ADDRESS, size: int | None = None, *, flat: bool = False) -> str:
    return wrap(*constants(format, address, size), load(format, "s0", "s1", "h"), ready(format, "s1", "s2", "h"), final="s2", flat=flat)


def regions(address: int = ADDRESS, size: int = 2048) -> tuple[MemoryRegion, ...]:
    return (MemoryRegion(address, RAW[:size]),)


def branch(condition_state: str, taken: str, *body: str) -> tuple[str, ...]:
    """A `%choose` branch whose true side returns at once and false side runs `body`."""
    return (f"cf.cond_br %choose, ^good(%{condition_state} : {S}), ^{taken}(%{condition_state} : {S})",
            f"^good(%gs: {S}):", f"func.return %gs : {S}", f"^{taken}(%bs: {S}):", *body)


class VirtualEvaluatorDMAHandleTest(unittest.TestCase):
    def accepted(self, source: str, inputs: RuntimeInputs | None = None):
        return evaluate(parse_program(source), inputs if inputs is not None else RuntimeInputs(memory=regions()))

    def test_signed_unsigned_addresses_and_exact_top_span_are_admitted(self) -> None:
        for format, size in (("fp8", 1024), ("bf16", 2048)):
            for address in (-2147483648, ADDRESS, (1 << 32) - size):
                for flat in (False, True):
                    with self.subTest(format=format, address=address, flat=flat):
                        memory = regions(address & 0xFFFFFFFF, size)
                        self.assertEqual(self.accepted(read_program(format, address, flat=flat), RuntimeInputs(memory=memory)).memory, memory)

    def test_recursive_unflagged_addition_proves_modulo_i32_address_and_size(self) -> None:
        proof = ("%a = arith.constant -32 : i32", "%b = arith.constant -2147483616 : i32",
                 "%zero = arith.constant 0 : i32", "%half = arith.constant 1024 : i32",
                 "%sum = arith.addi %a, %b : i32", "%addr = arith.addi %sum, %zero : i32",
                 "%size = arith.addi %half, %half : i32")
        self.accepted(wrap(*proof, load("bf16", "s0", "s1", "h"), ready("bf16", "s1", "s2", "h"), final="s2"))
        chain = ["%address0 = arith.constant 2147483648 : i32", "%zero = arith.constant 0 : i32", "%size = arith.constant 2048 : i32"]
        chain += [f"%address{index + 1} = arith.addi %address{index}, %zero : i32" for index in range(1500)]
        self.assertEqual(self.accepted(wrap(*chain, load("bf16", "s0", "s1", "h", "address1500"), ready("bf16", "s1", "s2", "h"), final="s2")).memory, regions())

    def test_invalid_dma_programs_are_rejected(self) -> None:
        base, mailbox = RuntimeInputs(memory=regions()), f"atlas.control_dram_base = {ADDRESS} : i64"
        with_tile, choose = RuntimeInputs({0: BF16}, memory=regions()), RuntimeInputs(controls=(Scalar(1, 1),))
        cases = []
        for format in ("fp8", "bf16"):
            cases += [(f"{format}_address_{address:#x}", read_program(format, address), base) for address in (0, ADDRESS - 32, ADDRESS + 1, (1 << 32) - 32)]
            cases += [(f"{format}_size_{size}", read_program(format, size=size), base) for size in (0, 32, -1, 2048 if format == "fp8" else 1024)]
        for operand in ("addr", "size"):
            body = list(constants())
            body[0 if operand == "addr" else 1] = f"%{operand} = arith.addi %dynamic, %dynamic : i32"
            source = wrap(*body, load("bf16", "s0", "s1", "h"), ready("bf16", "s1", "s2", "h"), final="s2", arguments="%dynamic: i32")
            cases.append((f"control_{operand}", source, RuntimeInputs(controls=(Scalar(32, 1024),), memory=regions())))
        cases.append(("block_argument_address", wrap(*constants(), f"cf.br ^next(%s0, %addr : {S}, i32)", f"^next(%entry: {S}, %passed: i32):",
                                                     load("bf16", "entry", "s1", "h", "passed"), ready("bf16", "s1", "s2", "h"), final="s2"), base))
        pair = (*constants(), load("bf16", "s0", "s1", "h0"), load("bf16", "s1", "s2", "h1"))
        cases.append(("third_pending", wrap(*pair, load("bf16", "s2", "s3", "h2"), ready("bf16", "s3", "s4", "h0", "v0"),
                                            ready("bf16", "s4", "s5", "h1", "v1"), ready("bf16", "s5", "s6", "h2", "v2"), final="s6"), base))
        old = (*constants(), load("bf16", "s0", "s1", "old"), ready("bf16", "s1", "s2", "old", "first"))
        cases.append(("repeated_await", wrap(*old, ready("bf16", "s2", "s3", "old", "fork"), final="s3"), base))
        cases.append(("stale_epoch", wrap(*old, load("bf16", "s2", "s3", "new"), ready("bf16", "s3", "s4", "old", "stale"), ready("bf16", "s4", "s5", "new"), final="s5"), base))
        cases.append(("repeated_wait", wrap(input_tile("bf16"), *constants(), store("bf16", "io", "s1", "h"), wait("s1", "s2", "h"), wait("s2", "s3", "h"), final="s3"), with_tile))
        for flat in (False, True):
            cases.append((f"dropped_load_flat_{flat}", wrap(*constants(), load("bf16", "s0", "s1", "h"), final="s1", flat=flat), base))
        cases.append(("dropped_store", wrap(input_tile("bf16"), *constants(), store("bf16", "io", "s1", "h"), final="s1"), with_tile))
        dropped = (*constants(), load("bf16", "s0", "s1", "h"), f"cf.br ^exit(%s1 : {S})", f"^exit(%es: {S}):")
        cases.append(("captured_across_blocks", wrap(*dropped, ready("bf16", "es", "s2", "h"), final="s2"), base))
        for malformed in ("bad_address", "dropped"):
            body = branch("s0", "bad", load("bf16", "bs", "s1", "h"), *((ready("bf16", "s1", "s2", "h"),) if malformed == "bad_address" else ()))
            source = wrap(*constants(address=0 if malformed == "bad_address" else ADDRESS), *body, final="s2" if malformed == "bad_address" else "s1", arguments="%choose: i1")
            cases.append((f"untaken_{malformed}", source, choose))
        for direction in ("load", "store"):
            for kind in ("input", "output", "pack"):
                hidden = {"input": input_tile("bf16", "s1", "hidden", "extra", 1), "output": output("s1", "hidden"),
                          "pack": f'%packed = "atlas.virtual_pack_fp8"(%x) {{scale_code = 127 : i32}} : ({B}) -> {F}'}[kind]
                before = "s1" if kind == "pack" else "hidden"
                launch, completion = transfer(direction, "io", "s1", "h", before)
                source = wrap(input_tile("bf16"), *constants(), launch, hidden, completion, final=f"{before}_done")
                cases.append((f"{direction}_pending_over_{kind}", source, RuntimeInputs({0: BF16, 1: BF16} if kind == "input" else {0: BF16}, memory=regions())))
        conflict = wrap(input_tile("bf16"), *constants(), *branch("io", "bad", load("bf16", "bs", "s1", "h0"), store("bf16", "s1", "s2", "h1"),
                                                                    ready("bf16", "s2", "s3", "h0"), wait("s3", "s4", "h1")), final="s4", arguments="%choose: i1")
        for choose in (0, 1):  # rejected whether or not the conflicting path executes
            cases.append((f"conflict_choose_{choose}", conflict, RuntimeInputs({0: BF16}, (Scalar(1, choose),), regions())))
        for direction in ("load", "store"):
            launch, completion = transfer(direction, "io", "s1", "h", "s1")
            source = wrap(input_tile("bf16"), *constants(), launch, completion, final="s1_done", arguments="%choose: i1", attributes=mailbox)
            cases.append((f"{direction}_mailbox", source, RuntimeInputs({0: BF16}, (Scalar(1, 1),), regions())))
            launch, completion = transfer(direction, "bs", "s1", "h", "s1")
            source = wrap(input_tile("bf16"), *constants(), *branch("io", "dma", launch, completion), final="s1_done", arguments="%choose: i1", attributes=mailbox)
            cases.append((f"{direction}_untaken_mailbox", source, RuntimeInputs({0: BF16}, (Scalar(1, 1),))))
        for kind in ("input", "output", "control"):
            for value in ('"invalid"', "-1 : i64", "4294967296 : i64", "4294967295 : i64"):
                attributes = f"atlas.{kind}_dram_base = {value}"
                if kind == "control":
                    cases.append((f"{kind}_base_{value}", wrap(final="s0", arguments="%choose: i1", attributes=attributes), choose))
                else:
                    cases.append((f"{kind}_base_{value}", wrap(input_tile("bf16"), output("io", "done"), final="done", attributes=attributes), RuntimeInputs({0: BF16})))
        for name, source, inputs in cases:
            with self.subTest(case=name), self.assertRaises(VirtualInterfaceError):
                evaluate(parse_program(source), inputs)

    def test_format_mismatches_and_passed_handles_fail_parsing(self) -> None:
        source = read_program()
        passed = wrap(*constants(), load("bf16", "s0", "s1", "h"), f"cf.br ^exit(%s1, %h : {S}, {event('bf16')})",
                      f"^exit(%es: {S}, %passed: {event('bf16')}):", ready("bf16", "es", "s2", "passed"), final="s2")
        for text in (source.replace(ready("bf16", "s1", "s2", "h"), ready("fp8", "s1", "s2", "h")),
                     source.replace(ready("bf16", "s1", "s2", "h"), wait("s1", "s2", "h")),
                     wrap(input_tile("bf16"), *constants(), store("bf16", "io", "s1", "h"), ready("bf16", "s1", "s2", "h"), final="s2"),
                     wrap(input_tile("bf16"), *constants("fp8"), store("fp8", "io", "s1", "h"), wait("s1", "s2", "h"), final="s2"), passed):
            with self.subTest(source=text), self.assertRaises(VirtualInterfaceError):
                parse_program(text)

    def test_two_pending_transfers_complete_in_either_order(self) -> None:
        other = MemoryRegion(DESTINATION, b"\x00\x40" * 1024)
        for order in (("h0", "h1"), ("h1", "h0")):
            source = wrap(*constants(), f"%dst = arith.constant {DESTINATION} : i32", load("bf16", "s0", "s1", "h0"), load("bf16", "s1", "s2", "h1", "dst"),
                          ready("bf16", "s2", "s3", order[0], f"v{order[0][1]}"), ready("bf16", "s3", "s4", order[1], f"v{order[1][1]}"),
                          output("s4", "o0", "v0"), output("o0", "done", "v1", 1), final="done")
            with self.subTest(order=order):
                result = self.accepted(source, RuntimeInputs(memory=(*regions(), other)))
                self.assertEqual((dict(result.outputs), result.memory), ({0: BF16, 1: Tile("bf16", (0x4000,) * 1024)}, (*regions(), other)))
        self.accepted(wrap(*constants(), "%small = arith.constant 1024 : i32", load("bf16", "s0", "s1", "h0"), load("fp8", "s1", "s2", "h1", size="small"),
                           ready("fp8", "s2", "s3", "h1", "small_tile"), ready("bf16", "s3", "s4", "h0"), final="s4"))
        for order in (0, 1):
            completion = (ready("bf16", "s2", "s3", "read"), wait("s3", "s4", "write")) if order == 0 else (wait("s2", "s3", "write"), ready("bf16", "s3", "s4", "read"))
            source = wrap(input_tile("bf16"), *constants(), *constants(address=DESTINATION, addr="dst", length="store_size"),
                          load("bf16", "io", "s1", "read"), store("bf16", "s1", "s2", "write", addr="dst", size="store_size"), *completion, final="s4")
            with self.subTest(load_store_order=order):
                result = self.accepted(source, RuntimeInputs({0: BF16}, memory=(*regions(), MemoryRegion(DESTINATION, bytes(2048)))))
                self.assertEqual(result.memory, (*regions(), MemoryRegion(DESTINATION, RAW)))
        for order in (("wide", "narrow"), ("narrow", "wide")):
            source = wrap(input_tile("bf16"), input_tile("fp8", "io", "io2", "small_tile", 1), *constants(), *constants("fp8", DESTINATION, addr="dst", length="small"),
                          store("bf16", "io2", "s1", "wide"), store("fp8", "s1", "s2", "narrow", "small_tile", "dst", "small"),
                          wait("s2", "s3", order[0]), wait("s3", "s4", order[1]), final="s4")
            with self.subTest(store_order=order):
                result = self.accepted(source, RuntimeInputs({0: BF16, 1: FP8}, memory=(MemoryRegion(ADDRESS, bytes(2048)), MemoryRegion(DESTINATION, bytes(1024)))))
                self.assertEqual(result.memory, (MemoryRegion(ADDRESS, RAW), MemoryRegion(DESTINATION, bytes(FP8.bits))))
        prefix = (*constants(), load("bf16", "s0", "s1", "h0"), load("bf16", "s1", "s2", "h1"))
        self.accepted(wrap(*prefix, ready("bf16", "s2", "s3", "h0", "v0"), load("bf16", "s3", "s4", "h2"),
                           ready("bf16", "s4", "s5", "h2", "v2"), ready("bf16", "s5", "s6", "h1", "v1"), final="s6"))

    def test_other_vpu_and_mxu_work_are_allowed_while_dma_is_pending(self) -> None:
        relu = f'%positive = "atlas.virtual_vpu_unary"(%x) {{kind = "relu"}} : ({B}) -> {B}'
        add = f'%sum = "atlas.virtual_vpu_binary"(%positive, %x) {{kind = "add"}} : ({B}, {B}) -> {B}'
        matmul = f'%product = "atlas.virtual_mxu_matmul"(%w, %w) {{unit = 1 : i32}} : ({F}, {F}) -> {B}'
        source = wrap(input_tile("bf16"), input_tile("fp8", "io", "io2", "w", 1), *constants(), load("bf16", "io2", "s1", "h"), relu, add, matmul,
                      ready("bf16", "s1", "s2", "h"), output("s2", "done", "sum"), final="done")
        result = self.accepted(source, RuntimeInputs({0: BF16, 1: Tile("fp8", (0,) * 1024)}, memory=regions()))
        self.assertEqual(dict(result.outputs), {0: Tile("bf16", (0x4000,) * 1024)})
        for unit in (0, 1):
            source = wrap(input_tile("bf16"), *constants(), load("bf16", "io", "s1", "h"), on_unit(seed("s1", "m1", "acc", "x"), unit), on_unit(read("m1", "m2", "copied", "acc"), unit),
                          ready("bf16", "m2", "s2", "h"), output("s2", "done", "copied"), final="done")
            with self.subTest(unit=unit):
                self.assertEqual(dict(self.accepted(source, RuntimeInputs({0: BF16}, memory=regions())).outputs), {0: BF16})

    def test_completed_transfers_and_boundary_aliases_share_memory(self) -> None:
        source = wrap(input_tile("bf16"), *constants(address=OUTPUT_BASE), load("bf16", "io", "s1", "h"), ready("bf16", "s1", "s2", "h"), output("s2", "done"), final="done", attributes=ABI)
        self.assertEqual(dict(self.accepted(source, RuntimeInputs({0: BF16}, memory=regions(OUTPUT_BASE))).outputs), {0: BF16})
        # Without declared bases, no boundary address is invented.
        source = wrap(input_tile("bf16"), *constants(address=INPUT_BASE), store("bf16", "io", "s1", "h"), wait("s1", "s2", "h"), output("s2", "done"), final="done")
        self.accepted(source, RuntimeInputs({0: BF16}, memory=regions(INPUT_BASE)))
        memory = (MemoryRegion(INPUT_BASE, bytes(2048)),)
        result = self.accepted(wrap(input_tile("bf16"), output("io", "done"), final="done"), RuntimeInputs({0: BF16}, memory=memory))
        self.assertEqual((dict(result.outputs), result.memory), ({0: BF16}, memory))

    def test_known_input_aliases_require_matching_initial_memory_bytes(self) -> None:
        for format, tile, payload in (("bf16", BF16, RAW), ("fp8", FP8, bytes(FP8.bits))):
            attributes = f"atlas.input_dram_base = {INPUT_BASE} : i64"
            dma = wrap(input_tile(format), *constants(format, INPUT_BASE), load(format, "io", "s1", "h"), ready(format, "s1", "s2", "h"), final="s2", attributes=attributes)
            implicit = wrap(input_tile(format), final="io", attributes=attributes)
            # A partial snapshot is checked even without a DMA access.
            for name, source, address, data, flip in (("dma", dma, INPUT_BASE, payload, 0), ("partial", implicit, INPUT_BASE + 13, payload[13:94], -1)):
                with self.subTest(format=format, case=name):
                    memory = (MemoryRegion(address, data),)
                    self.assertEqual(self.accepted(source, RuntimeInputs({0: tile}, memory=memory)).memory, memory)
                    inconsistent = bytearray(data)
                    inconsistent[flip] ^= 1
                    with self.assertRaises(VirtualInterfaceError):
                        evaluate(parse_program(source), RuntimeInputs({0: tile}, memory=(MemoryRegion(address, inconsistent),)))


class VirtualEvaluatorResourceTest(unittest.TestCase):
    def test_dma_loop_rebinds_static_store_and_load_handles_to_changed_tiles(self) -> None:
        # The second visit stores B at A's address; final bytes and the awaited tile must come from it.
        first = Tile("bf16", tuple((row * 2039 + col * 137) & 0xFFFF for row in range(32) for col in range(32)))
        second = Tile("bf16", tuple(bits ^ 0x8155 for bits in first.bits))
        source = wrap(input_tile("bf16"), input_tile("bf16", "io", "io2", "other", 1), *constants(), LOOP_HEADER,
                      f"cf.br ^loop(%io2, %x, %other, %zero : {S}, {B}, {B}, i32)", f"^loop(%ls: {S}, %a: {B}, %b: {B}, %i: i32):",
                      store("bf16", "ls", "s1", "write", "a"), wait("s1", "s2", "write"), load("bf16", "s2", "s3", "read"), ready("bf16", "s3", "s4", "read"),
                      "%next = arith.addi %i, %one : i32", "%again = arith.cmpi ult, %next, %limit : i32",
                      f"cf.cond_br %again, ^loop(%s4, %b, %a, %next : {S}, {B}, {B}, i32), ^exit(%s4, %loaded : {S}, {B})",
                      f"^exit(%es: {S}, %last: {B}):", output("es", "done", "last"), final="done")
        guard = MemoryRegion(ADDRESS - 13, b"before!before" + b"?" * 2048 + b"after!")
        result = evaluate(parse_program(source), RuntimeInputs({0: first, 1: second}, memory=(guard,)))
        self.assertEqual(dict(result.outputs), {0: second})
        self.assertEqual(result.memory, (MemoryRegion(guard.address, b"before!before" + bf16_bytes(second.bits) + b"after!"),))

    def test_mxu_loop_rebinds_weight_and_accumulator_versions_from_carried_tiles(self) -> None:
        # Identity activations contract with packed diagonal A, seeded by A; swapping to B changes both: 6 -> 4.
        body = "\n".join((input_tile("bf16", "s4", "io5", "other", 4), LOOP_HEADER,
                          f"cf.br ^loop(%io5, %seed, %other, %zero : {S}, {B}, {B}, i32)", f"^loop(%ls: {S}, %a: {B}, %b: {B}, %i: i32):",
                          f'%packed = "atlas.virtual_pack_fp8"(%a) {{scale_code = 127 : i32}} : ({B}) -> {F}',
                          weight("ls", "ws", "w0", "packed"), seed("ws", "as", "a0", "a"), accumulate("as", "ms", "a1", "a0"), read("ms", "rs", "result", "a1"),
                          "%next = arith.addi %i, %one : i32", "%again = arith.cmpi ult, %next, %limit : i32",
                          f"cf.cond_br %again, ^loop(%rs, %b, %a, %next : {S}, {B}, {B}, i32), ^exit(%rs, %result : {S}, {B})",
                          f"^exit(%es: {S}, %last: {B}):", output("es", "done", "last"), f"func.return %done : {S}"))
        tiles = {**TILES, 4: diagonal("bf16", 0x4000)}
        for unit in (0, 1):
            with self.subTest(unit=unit):
                result = evaluate(parse_program(on_unit(function(body), unit)), RuntimeInputs(tiles))
                self.assertEqual((dict(result.outputs), result.memory), ({0: diagonal("bf16", 0x4080)}, ()))


if __name__ == "__main__":
    unittest.main()
