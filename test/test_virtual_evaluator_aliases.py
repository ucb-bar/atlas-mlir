"""Shared external bytes with immutable SSA tiles and explicit DMA completion."""

from __future__ import annotations

import unittest

from test_virtual_evaluator_dma import STATE, Stream, bf16_bytes, repeated
from test_virtual_evaluator_dma_handles import constants, input_tile, load, output, ready, store, wait, wrap
from atlas_virtual_evaluator import MemoryRegion, RuntimeInputs, Scalar, Tile, VirtualInterfaceError, evaluate, parse_program


INPUT, OUTPUT = 0x90000000, 0x90004000


def program(stream: Stream, *, input_base=None, output_base=None):
    attributes = []
    for kind, address in (("input", input_base), ("output", output_base)):
        if address is not None:
            attributes.append(f"atlas.{kind}_dram_base = {address} : i64")
    attrs = " attributes {" + ", ".join(attributes) + "}" if attributes else ""
    body = "\n".join((*stream.lines, f"func.return {stream.state} : {STATE}"))
    return parse_program(f"module {{ func.func @aliases() -> {STATE}{attrs} {{\n{body}\n}} }}")


class VirtualEvaluatorAliasesTest(unittest.TestCase):
    def run_owned(self, source, inputs):
        original_tiles, original_memory = dict(inputs.tiles), inputs.memory
        result = evaluate(source, inputs)
        self.assertEqual(dict(inputs.tiles), original_tiles)
        self.assertEqual(inputs.memory, original_memory)
        return result

    def test_completed_store_updates_repeated_input_but_preserves_old_ssa_tile(self):
        old, new = repeated((0x7FC1, 0x8001, 0x3F80)), repeated((0xFFA1, 0x007F, 0xBF80))
        stream = Stream()
        before, replacement = stream.input(), stream.input(index=1)
        stream.wait(stream.store(replacement, INPUT))
        after = stream.input()
        stream.output(before)
        stream.output(after, 1)
        original = MemoryRegion(INPUT - 3, b"old" + bf16_bytes(old.bits))
        guard = MemoryRegion(INPUT + 4096, b"guard")
        runtime = RuntimeInputs({0: old, 1: new}, memory=(original, guard))
        result = self.run_owned(program(stream, input_base=INPUT, output_base=OUTPUT), runtime)
        self.assertEqual(dict(result.outputs), {0: old, 1: new})
        self.assertEqual(result.memory, (MemoryRegion(INPUT - 3, b"old" + bf16_bytes(new.bits)), guard))

    def test_boundary_output_initializes_bytes_for_a_later_explicit_load(self):
        original = repeated((0x8000, 0x0001, 0x7FC1, 0xFFA1))
        stream = Stream()
        source = stream.input()
        stream.output(source)
        copied = stream.await_(stream.load(OUTPUT))
        stream.output(copied, 1)
        result = self.run_owned(program(stream, output_base=OUTPUT), RuntimeInputs({0: original}))
        self.assertEqual(dict(result.outputs), {0: original, 1: original})
        self.assertEqual(result.memory, ())

    def test_boundary_output_is_the_last_writer_after_a_completed_explicit_store(self):
        first, last = repeated((0x3F80,)), repeated((0xBF80,))
        stream = Stream()
        first_value, last_value = stream.input(), stream.input(index=1)
        stream.wait(stream.store(first_value, OUTPUT))
        stream.output(last_value)
        regions = (MemoryRegion(OUTPUT + 1500, b"?" * 548 + b"tail"), MemoryRegion(OUTPUT - 4, b"head" + b"?" * 1500))
        result = self.run_owned(program(stream, output_base=OUTPUT), RuntimeInputs({0: first, 1: last}, memory=regions))
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
        # The write starts at physical first-half row 1, spans that half's
        # remaining 31 rows, and ends after second-half row 0. Each word is 1234.
        wanted = Tile("bf16", tuple(0x1234 if (col < 16 and row >= 1) or (col >= 16 and row == 0) else 0x3F80
                                     for row in range(32) for col in range(32)))
        payload = bf16_bytes(original.bits)
        final = payload[:32] + bytes(replacement.bits) + payload[1056:]
        runtime = RuntimeInputs({0: original, 1: replacement}, memory=(MemoryRegion(OUTPUT - 7, b"before!" + payload + b"?" * 2048 + b"after"),))
        result = self.run_owned(program(stream, output_base=OUTPUT), runtime)
        self.assertEqual(dict(result.outputs), {0: wanted, 1: original})
        self.assertEqual(result.memory, (MemoryRegion(OUTPUT - 7, b"before!" + final + payload + b"after"),))

    def test_input_and_output_boundaries_can_share_an_address(self):
        old, new = repeated((0x0001, 0x8000)), repeated((0x7FC1, 0xBF80))
        stream = Stream()
        before, replacement = stream.input(), stream.input(index=1)
        stream.output(replacement)
        after = stream.input()
        stream.output(before, 1)
        # Output1 aliases input1 and overwrites it too. The already loaded
        # replacement and reread input0 retain their immutable new bits.
        stream.output(after, 2)
        result = self.run_owned(program(stream, input_base=INPUT, output_base=INPUT), RuntimeInputs({0: old, 1: new}))
        self.assertEqual(dict(result.outputs), {0: new, 1: old, 2: new})

    def test_partial_output_snapshot_receives_only_executed_write_bytes(self):
        original = repeated((0x8001, 0x7FC1, 0x3F80))
        stream = Stream()
        stream.output(stream.input())
        supplied = (MemoryRegion(OUTPUT + 13, b"?" * 81), MemoryRegion(OUTPUT + 2048, b"untouched"))
        result = self.run_owned(program(stream, output_base=OUTPUT), RuntimeInputs({0: original}, memory=supplied))
        self.assertEqual(result.memory, (MemoryRegion(OUTPUT + 13, bf16_bytes(original.bits)[13:94]), supplied[1]))

    def test_output_mapping_does_not_define_uninitialized_bytes(self):
        stream = Stream()
        source = stream.input()
        loaded = stream.await_(stream.load(OUTPUT))
        stream.output(source)
        stream.output(loaded, 1)
        source = program(stream, output_base=OUTPUT)
        original = repeated((0x3F80,))
        for memory in ((), (MemoryRegion(OUTPUT, b"?" * 1024),)):
            with self.subTest(partial=bool(memory)), self.assertRaisesRegex(VirtualInterfaceError, "undefined bytes"):
                evaluate(source, RuntimeInputs({0: original}, memory=memory))

    def test_fp8_input_padding_requires_supplied_bytes(self):
        stream = Stream()
        stream.input("fp8")
        stream.output(stream.await_(stream.load(INPUT)))
        source = program(stream, input_base=INPUT)
        original = Tile("fp8", (0x38,) * 1024)
        with self.assertRaisesRegex(VirtualInterfaceError, "unmapped memory"):
            evaluate(source, RuntimeInputs({0: original}))
        padding = MemoryRegion(INPUT + 1024, b"\x34\x12" * 512)
        wanted = Tile("bf16", tuple(0x3838 if col < 16 else 0x1234 for row in range(32) for col in range(32)))
        result = self.run_owned(source, RuntimeInputs({0: original}, memory=(padding,)))
        self.assertEqual(dict(result.outputs), {0: wanted})
        self.assertEqual(result.memory, (padding,))

    def test_untaken_boundary_write_preserves_supplied_output_snapshot(self):
        source = wrap(input_tile("bf16"), f"cf.cond_br %choose, ^write(%io : {STATE}), ^skip(%io : {STATE})",
                      f"^write(%ws: {STATE}):", output("ws", "done"), f"func.return %done : {STATE}",
                      f"^skip(%ss: {STATE}):", final="ss", arguments="%choose: i1", attributes=f"atlas.output_dram_base = {OUTPUT} : i64")
        original = repeated((0x3F80,))
        memory = (MemoryRegion(OUTPUT, b"?" * 2048),)
        result = self.run_owned(parse_program(source), RuntimeInputs({0: original}, (Scalar(1, 0),), memory))
        self.assertEqual(dict(result.outputs), {})
        self.assertEqual(result.memory, memory)

    def test_pending_partial_write_conflicts_do_not_depend_on_wait_order(self):
        for first, second in (("load", "store"), ("store", "load"), ("store", "store")):
            for reverse in (False, True):
                launches = []
                completions = []
                for index, direction in enumerate((first, second)):
                    before = "io" if index == 0 else "s1"
                    addr = "addr" if index == 0 else "second"
                    launches.append(load("bf16", before, f"s{index + 1}", f"h{index}", addr) if direction == "load"
                                    else store("bf16", before, f"s{index + 1}", f"h{index}", addr=addr))
                for position, index in enumerate((1, 0) if reverse else (0, 1)):
                    completions.append(ready("bf16", f"s{position + 2}", f"s{position + 3}", f"h{index}", f"tile{index}") if (first, second)[index] == "load"
                                       else wait(f"s{position + 2}", f"s{position + 3}", f"h{index}"))
                source = wrap(input_tile("bf16"), *constants(address=INPUT), f"%second = arith.constant {INPUT + 1024} : i32",
                              *launches, *completions, final="s4")
                runtime = RuntimeInputs({0: repeated((0x3F80,))}, memory=(MemoryRegion(INPUT, b"?" * 3072),))
                with self.subTest(first=first, second=second, reverse=reverse), self.assertRaisesRegex(VirtualInterfaceError, "conflicts"):
                    evaluate(parse_program(source), runtime)


if __name__ == "__main__":
    unittest.main()
