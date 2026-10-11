"""DMA byte effects and shared external bytes over independent raw snapshots, without Atlas tools.

SSA tiles stay immutable while explicit DMA completion and boundary I/O update shared memory.
"""

from __future__ import annotations

from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
from atlas_virtual_evaluator import MemoryRegion, RuntimeInputs, Scalar, Tile, VirtualInterfaceError, evaluate, parse_program  # noqa: E402
from test_virtual_evaluator_arithmetic import repeated  # noqa: E402
from test_virtual_evaluator_resources import (  # noqa: E402
    S as STATE, STORE, bf16_bytes, branch, constants, input_tile, load, output, ready, store, type_for as tile_type, wait, wrap,
)


BASE = 0x80000000
INPUT, OUTPUT = 0x90000000, 0x90004000


class Stream:
    """Textual state-chain builder, with no numerical or memory oracle."""

    def __init__(self):
        self.lines = [f'%s0 = "atlas.virtual_start"() : () -> {STATE}']
        self.state = "%s0"
        self.count = 0

    def fresh(self):
        self.count += 1
        return f"%v{self.count}"

    def effect(self, name, args=(), types=(), result_type=None, attrs=""):
        state = self.fresh()
        result = self.fresh() if result_type else None
        lhs = f"{state}, {result}" if result else state
        returns = f"({STATE}, {result_type})" if result else STATE
        self.lines.append(f'{lhs} = "atlas.virtual_{name}"({", ".join((self.state, *args))}) '
                          f'{attrs} : ({", ".join((STATE, *types))}) -> {returns}')
        self.state = state
        return result

    def constant(self, value):
        result = self.fresh()
        self.lines.append(f"{result} = arith.constant {value} : i32")
        return result

    def input(self, fmt="bf16", index=0):
        return self.effect(f"input_{fmt}", result_type=tile_type(fmt), attrs=f"{{index = {index} : i32}}")

    def load(self, address, fmt="bf16"):
        addr, size = self.constant(address), self.constant(1024 if fmt == "fp8" else 2048)
        return self.effect(f"dma_load_{fmt}", (addr, size), ("i32", "i32"), f"!atlas.virtual_dma_load_{fmt}")

    def await_(self, handle, fmt="bf16"):
        return self.effect(f"dma_await_{fmt}", (handle,), (f"!atlas.virtual_dma_load_{fmt}",), tile_type(fmt))

    def store(self, value, address, fmt="bf16"):
        addr, size = self.constant(address), self.constant(1024 if fmt == "fp8" else 2048)
        return self.effect(f"dma_store_{fmt}", (value, addr, size), (tile_type(fmt), "i32", "i32"), STORE)

    def wait(self, handle):
        self.effect("dma_wait", (handle,), (STORE,))

    def output(self, value, index=0):
        self.effect("output_bf16", (value,), (tile_type("bf16"),), attrs=f"{{index = {index} : i32}}")

    def pure(self, name, value, fmt, attrs):
        result = self.fresh()
        self.lines.append(f'{result} = "atlas.virtual_{name}"({value}) {attrs} : ({tile_type("bf16")}) -> {tile_type(fmt)}')
        return result

    def copy(self, source, destination, fmt="fp8"):
        self.wait(self.store(self.await_(self.load(source, fmt), fmt), destination, fmt))

    def program(self, *, input_base=None, output_base=None):
        """Flat IR, or a function carrying the given boundary DRAM bases."""
        if input_base is None and output_base is None:
            return parse_program("module {\n" + "\n".join(self.lines) + "\n}")
        attributes = ", ".join(f"atlas.{kind}_dram_base = {address} : i64" for kind, address in (("input", input_base), ("output", output_base)) if address is not None)
        body = "\n".join((*self.lines, f"func.return {self.state} : {STATE}"))
        return parse_program(f"module {{ func.func @aliases() -> {STATE} attributes {{{attributes}}} {{\n{body}\n}} }}")


class VirtualEvaluatorDmaTest(unittest.TestCase):
    def test_all_raw_fp8_codes_copy_with_guards_and_fresh_owned_snapshots(self):
        stream = Stream()
        stream.copy(-2147483648, BASE + 0x2000)
        program = stream.program()
        payload = bytes(range(256)) * 4
        source_buffer = bytearray(payload)
        source = MemoryRegion(BASE, source_buffer)
        guarded = MemoryRegion(BASE + 0x2000 - 32, b"L" * 32 + b"?" * 1024 + b"R" * 31)
        runtime = RuntimeInputs(memory=(guarded, source))
        source_buffer[:] = b"!" * 1024
        expected = (MemoryRegion(guarded.address, b"L" * 32 + payload + b"R" * 31), source)
        self.assertEqual(evaluate(program, runtime).memory, expected)
        alternate = RuntimeInputs(memory=(guarded, MemoryRegion(BASE, payload[::-1])))
        self.assertEqual(evaluate(program, alternate).memory[0].data, b"L" * 32 + payload[::-1] + b"R" * 31)
        self.assertEqual(evaluate(program, runtime).memory, expected)

    def test_bf16_load_and_store_independently_check_half_pair_layout_and_raw_bits(self):
        bits = tuple((row * 2039 + col * 137) & 0xFFFF for row in range(32) for col in range(32))
        original = Tile("bf16", (0x0000, 0x8000, 0x0001, 0x8001, 0x7F80, 0xFF80, 0x7FC1, 0xFFA1) + bits[8:])
        payload = bf16_bytes(original.bits)
        load = Stream()
        load.output(load.await_(load.load(BASE)))
        runtime = RuntimeInputs(memory=(MemoryRegion(BASE, payload),))
        result = evaluate(load.program(), runtime)
        self.assertEqual((dict(result.outputs), result.memory), ({0: original}, runtime.memory))
        store = Stream()
        store.wait(store.store(store.input(), BASE + 0x2000))
        destination = MemoryRegion(BASE + 0x2000 - 7, b"before!" + b"?" * 2048 + b"after")
        inputs = RuntimeInputs({0: original}, memory=(destination,))
        self.assertEqual(evaluate(store.program(), inputs).memory, (MemoryRegion(destination.address, b"before!" + payload + b"after"),))

    def test_adjacent_regions_cover_a_transfer_and_preserve_supplied_order_and_extents(self):
        payload = bytes(range(256)) * 4
        destination = BASE + 0x2000
        tail = MemoryRegion(destination + 777, b"?" * 247 + b"tail")
        source = MemoryRegion(BASE, payload)
        head = MemoryRegion(destination - 32, b"head" * 8 + b"?" * 777)
        untouched = MemoryRegion(BASE + 0x4000, b"unrelated")
        stream = Stream()
        stream.copy(BASE, destination)
        runtime = RuntimeInputs(memory=(tail, source, head, untouched))
        self.assertEqual(evaluate(stream.program(), runtime).memory,
                         (MemoryRegion(tail.address, payload[777:] + b"tail"), source, MemoryRegion(head.address, b"head" * 8 + payload[:777]), untouched))
        # A load can cross the same unaligned region boundary, too.
        split = RuntimeInputs(memory=(MemoryRegion(BASE + 13, payload[13:]), MemoryRegion(BASE, payload[:13]), MemoryRegion(destination, b"?" * 1024)))
        self.assertEqual(evaluate(stream.program(), split).memory[-1], MemoryRegion(destination, payload))

    def test_overlapping_reads_can_complete_in_reverse_issue_order(self):
        payload = bytes(range(256)) * 4
        stream = Stream()
        first, second = stream.load(BASE, "fp8"), stream.load(BASE, "fp8")
        second_ready, first_ready = stream.await_(second, "fp8"), stream.await_(first, "fp8")
        stream.wait(stream.store(second_ready, BASE + 0x2000, "fp8"))
        stream.wait(stream.store(first_ready, BASE + 0x4000, "fp8"))
        runtime = RuntimeInputs(memory=(MemoryRegion(BASE, payload), MemoryRegion(BASE + 0x2000, b"?" * 1024), MemoryRegion(BASE + 0x4000, b"!" * 1024)))
        self.assertEqual(evaluate(stream.program(), runtime).memory,
                         (runtime.memory[0], MemoryRegion(BASE + 0x2000, payload), MemoryRegion(BASE + 0x4000, payload)))

    def test_ready_load_snapshot_survives_later_store_and_serial_reload_sees_new_bytes(self):
        old = bytes(range(256)) * 4
        new = old[::-1]
        stream = Stream()
        replacement = stream.input("fp8")
        original = stream.await_(stream.load(BASE, "fp8"), "fp8")
        stream.wait(stream.store(replacement, BASE, "fp8"))
        reloaded = stream.await_(stream.load(BASE, "fp8"), "fp8")
        # Keep both stores pending and finish them in reverse order.
        old_store = stream.store(original, BASE + 0x2000, "fp8")
        stream.wait(stream.store(reloaded, BASE + 0x4000, "fp8"))
        stream.wait(old_store)
        runtime = RuntimeInputs({0: Tile("fp8", tuple(new))}, memory=(MemoryRegion(BASE, old),
                                MemoryRegion(BASE + 0x2000, b"?" * 1024), MemoryRegion(BASE + 0x4000, b"!" * 1024)))
        self.assertEqual(evaluate(stream.program(), runtime).memory,
                         (MemoryRegion(BASE, new), MemoryRegion(BASE + 0x2000, old), MemoryRegion(BASE + 0x4000, new)))

    def test_store_captures_ready_source_before_a_later_pure_value_is_produced(self):
        tile = repeated((0xBF80, 0x3F80, 0x8000, 0x0001, 0xFFC1, 0x7FC1))
        relu = Tile("bf16", tuple(0 if bits & 0x8000 else bits for bits in tile.bits))
        stream = Stream()
        source = stream.input()
        pending = stream.store(source, BASE)
        changed = stream.pure("vpu_unary", source, "bf16", '{kind = "relu"}')
        stream.wait(pending)
        stream.wait(stream.store(changed, BASE + 0x2000))
        runtime = RuntimeInputs({0: tile}, memory=(MemoryRegion(BASE, b"?" * 2048), MemoryRegion(BASE + 0x2000, b"!" * 2048)))
        self.assertEqual(evaluate(stream.program(), runtime).memory,
                         (MemoryRegion(BASE, bf16_bytes(tile.bits)), MemoryRegion(BASE + 0x2000, bf16_bytes(relu.bits))))

    def test_unmapped_load_is_checked_only_on_the_executed_cfg_path(self):
        body = branch("s0", "load", load("bf16", "bs", "s1", "pending"), ready("bf16", "s1", "s2", "pending"))
        program = parse_program(wrap(*constants(address=-2147483648), *body, final="s2", arguments="%choose: i1"))
        untouched = MemoryRegion(BASE + 0x4000, b"guard")
        result = evaluate(program, RuntimeInputs(controls=(Scalar(1, 1),), memory=(untouched,)))
        self.assertEqual((dict(result.outputs), result.memory), ({}, (untouched,)))
        with self.assertRaisesRegex(VirtualInterfaceError, "unmapped memory"):
            evaluate(program, RuntimeInputs(controls=(Scalar(1, 0),), memory=(untouched,)))

    def test_holes_and_partial_load_or_store_spans_fail(self):
        cases = ((MemoryRegion(BASE, b"?" * 1023),), (MemoryRegion(BASE + 1, b"?" * 1023),),
                 (MemoryRegion(BASE, b"?" * 512), MemoryRegion(BASE + 513, b"!" * 511)))
        tile = repeated((0x7F, 0xFF, 0x01, 0x81), "fp8")
        for direction in ("load", "store"):
            stream = Stream()
            if direction == "load":
                stream.await_(stream.load(BASE, "fp8"), "fp8")
            else:
                stream.wait(stream.store(stream.input("fp8"), BASE, "fp8"))
            for regions in cases:
                with self.subTest(direction=direction, addresses=tuple(r.address for r in regions)):
                    with self.assertRaisesRegex(VirtualInterfaceError, "unmapped memory"):
                        evaluate(stream.program(), RuntimeInputs({0: tile} if direction == "store" else {}, memory=regions))


class VirtualEvaluatorAliasesTest(unittest.TestCase):
    def test_completed_store_updates_repeated_input_but_preserves_old_ssa_tile(self):
        old, new = repeated((0x7FC1, 0x8001, 0x3F80)), repeated((0xFFA1, 0x007F, 0xBF80))
        stream = Stream()
        before, replacement = stream.input(), stream.input(index=1)
        stream.wait(stream.store(replacement, INPUT))
        after = stream.input()
        stream.output(before)
        stream.output(after, 1)
        guard = MemoryRegion(INPUT + 4096, b"guard")
        runtime = RuntimeInputs({0: old, 1: new}, memory=(MemoryRegion(INPUT - 3, b"old" + bf16_bytes(old.bits)), guard))
        result = evaluate(stream.program(input_base=INPUT, output_base=OUTPUT), runtime)
        self.assertEqual(dict(result.outputs), {0: old, 1: new})
        self.assertEqual(result.memory, (MemoryRegion(INPUT - 3, b"old" + bf16_bytes(new.bits)), guard))

    def test_boundary_output_initializes_bytes_for_a_later_explicit_load(self):
        original = repeated((0x8000, 0x0001, 0x7FC1, 0xFFA1))
        stream = Stream()
        stream.output(stream.input())
        stream.output(stream.await_(stream.load(OUTPUT)), 1)
        result = evaluate(stream.program(output_base=OUTPUT), RuntimeInputs({0: original}))
        self.assertEqual((dict(result.outputs), result.memory), ({0: original, 1: original}, ()))

    def test_boundary_output_is_the_last_writer_after_a_completed_explicit_store(self):
        first, last = repeated((0x3F80,)), repeated((0xBF80,))
        stream = Stream()
        first_value, last_value = stream.input(), stream.input(index=1)
        stream.wait(stream.store(first_value, OUTPUT))
        stream.output(last_value)
        regions = (MemoryRegion(OUTPUT + 1500, b"?" * 548 + b"tail"), MemoryRegion(OUTPUT - 4, b"head" + b"?" * 1500))
        result = evaluate(stream.program(output_base=OUTPUT), RuntimeInputs({0: first, 1: last}, memory=regions))
        self.assertEqual(dict(result.outputs), {0: last})
        self.assertEqual(result.memory, (MemoryRegion(OUTPUT + 1500, bf16_bytes(last.bits)[1500:] + b"tail"),
                                         MemoryRegion(OUTPUT - 4, b"head" + bf16_bytes(last.bits)[:1500])))

    def test_completed_partial_fp8_store_changes_final_host_output_without_changing_ssa(self):
        original = repeated((0x3F80,))
        replacement = Tile("fp8", (0x34, 0x12) * 512)
        stream = Stream()
        source, fp8 = stream.input(), stream.input("fp8", 1)
        stream.output(source)
        stream.wait(stream.store(fp8, OUTPUT + 32, "fp8"))
        stream.output(source, 1)
        # The write covers first-half rows 1..31 and second-half row 0, as 0x1234 words.
        wanted = Tile("bf16", tuple(0x1234 if (col < 16 and row >= 1) or (col >= 16 and row == 0) else 0x3F80 for row in range(32) for col in range(32)))
        payload = bf16_bytes(original.bits)
        final = payload[:32] + bytes(replacement.bits) + payload[1056:]
        runtime = RuntimeInputs({0: original, 1: replacement}, memory=(MemoryRegion(OUTPUT - 7, b"before!" + payload + b"?" * 2048 + b"after"),))
        result = evaluate(stream.program(output_base=OUTPUT), runtime)
        self.assertEqual(dict(result.outputs), {0: wanted, 1: original})
        self.assertEqual(result.memory, (MemoryRegion(OUTPUT - 7, b"before!" + final + payload + b"after"),))

    def test_input_and_output_boundaries_can_share_an_address(self):
        old, new = repeated((0x0001, 0x8000)), repeated((0x7FC1, 0xBF80))
        stream = Stream()
        before, replacement = stream.input(), stream.input(index=1)
        stream.output(replacement)
        after = stream.input()
        stream.output(before, 1)
        # Output1 also overwrites input1; already loaded values keep their immutable bits.
        stream.output(after, 2)
        result = evaluate(stream.program(input_base=INPUT, output_base=INPUT), RuntimeInputs({0: old, 1: new}))
        self.assertEqual(dict(result.outputs), {0: new, 1: old, 2: new})

    def test_output_snapshots_receive_only_executed_write_bytes(self):
        original = repeated((0x8001, 0x7FC1, 0x3F80))
        stream = Stream()
        stream.output(stream.input())
        supplied = (MemoryRegion(OUTPUT + 13, b"?" * 81), MemoryRegion(OUTPUT + 2048, b"untouched"))
        result = evaluate(stream.program(output_base=OUTPUT), RuntimeInputs({0: original}, memory=supplied))
        self.assertEqual(result.memory, (MemoryRegion(OUTPUT + 13, bf16_bytes(original.bits)[13:94]), supplied[1]))
        source = wrap(input_tile("bf16"), *branch("io", "write", output("bs", "done")), final="done", arguments="%choose: i1",
                      attributes=f"atlas.output_dram_base = {OUTPUT} : i64")
        memory = (MemoryRegion(OUTPUT, b"?" * 2048),)
        result = evaluate(parse_program(source), RuntimeInputs({0: original}, (Scalar(1, 1),), memory))
        self.assertEqual((dict(result.outputs), result.memory), ({}, memory))

    def test_uninitialized_output_and_fp8_input_padding_bytes_are_not_defined(self):
        stream = Stream()
        source = stream.input()
        loaded = stream.await_(stream.load(OUTPUT))
        stream.output(source)
        stream.output(loaded, 1)
        for memory in ((), (MemoryRegion(OUTPUT, b"?" * 1024),)):
            with self.subTest(partial=bool(memory)), self.assertRaisesRegex(VirtualInterfaceError, "undefined bytes"):
                evaluate(stream.program(output_base=OUTPUT), RuntimeInputs({0: repeated((0x3F80,))}, memory=memory))
        stream = Stream()
        stream.input("fp8")
        stream.output(stream.await_(stream.load(INPUT)))
        original = Tile("fp8", (0x38,) * 1024)
        with self.assertRaisesRegex(VirtualInterfaceError, "unmapped memory"):
            evaluate(stream.program(input_base=INPUT), RuntimeInputs({0: original}))
        padding = MemoryRegion(INPUT + 1024, b"\x34\x12" * 512)
        result = evaluate(stream.program(input_base=INPUT), RuntimeInputs({0: original}, memory=(padding,)))
        wanted = Tile("bf16", tuple(0x3838 if col < 16 else 0x1234 for row in range(32) for col in range(32)))
        self.assertEqual((dict(result.outputs), result.memory), ({0: wanted}, (padding,)))

    def test_pending_partial_write_conflicts_do_not_depend_on_wait_order(self):
        for first, second in (("load", "store"), ("store", "load"), ("store", "store")):
            launches = [load("bf16", before, f"s{index + 1}", f"h{index}", addr) if direction == "load" else store("bf16", before, f"s{index + 1}", f"h{index}", addr=addr)
                        for index, (direction, before, addr) in enumerate(((first, "io", "addr"), (second, "s1", "second")))]
            for reverse in (False, True):
                completions = [ready("bf16", f"s{position + 2}", f"s{position + 3}", f"h{index}", f"tile{index}") if (first, second)[index] == "load"
                               else wait(f"s{position + 2}", f"s{position + 3}", f"h{index}") for position, index in enumerate((1, 0) if reverse else (0, 1))]
                source = wrap(input_tile("bf16"), *constants(address=INPUT), f"%second = arith.constant {INPUT + 1024} : i32", *launches, *completions, final="s4")
                runtime = RuntimeInputs({0: repeated((0x3F80,))}, memory=(MemoryRegion(INPUT, b"?" * 3072),))
                with self.subTest(first=first, second=second, reverse=reverse), self.assertRaisesRegex(VirtualInterfaceError, "conflicts"):
                    evaluate(parse_program(source), runtime)


if __name__ == "__main__":
    unittest.main()
