"""DMA reference effects over independent raw byte snapshots, without Atlas tools."""

from __future__ import annotations

from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
from atlas_virtual_evaluator import MemoryRegion, RuntimeInputs, Scalar, Tile, VirtualInterfaceError, evaluate, parse_program  # noqa: E402

STATE = "!atlas.virtual_state"
STORE = "!atlas.virtual_dma_store"
BASE = 0x80000000


def tile_type(fmt):
    return f"!atlas.virtual_{fmt}"


def bf16_bytes(bits):
    # The two physical halves each hold 32 rows of 16 little-endian words.
    # This fixture intentionally does not call the evaluator's serializers.
    return b"".join(bits[row * 32 + col].to_bytes(2, "little")
                    for half in range(2) for row in range(32)
                    for col in range(half * 16, half * 16 + 16))


def repeated(values, fmt="bf16"):
    return Tile(fmt, tuple(values[i % len(values)] for i in range(1024)))


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
        self.lines.append(f'{result} = "atlas.virtual_{name}"({value}) {attrs} '
                          f': ({tile_type("bf16")}) -> {tile_type(fmt)}')
        return result

    def program(self):
        return parse_program("module {\n" + "\n".join(self.lines) + "\n}")


class VirtualEvaluatorDmaTest(unittest.TestCase):
    def test_all_raw_fp8_codes_copy_with_guards_and_fresh_owned_snapshots(self):
        stream = Stream()
        ready = stream.await_(stream.load(-2147483648, "fp8"), "fp8")
        stream.wait(stream.store(ready, BASE + 0x2000, "fp8"))
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
        self.assertEqual(runtime.memory, (guarded, source))

    def test_bf16_load_and_store_independently_check_half_pair_layout_and_raw_bits(self):
        bits = tuple((row * 2039 + col * 137) & 0xFFFF for row in range(32) for col in range(32))
        bits = (0x0000, 0x8000, 0x0001, 0x8001, 0x7F80, 0xFF80, 0x7FC1, 0xFFA1) + bits[8:]
        original = Tile("bf16", bits)
        payload = bf16_bytes(bits)
        load = Stream()
        load.output(load.await_(load.load(BASE)))
        runtime = RuntimeInputs(memory=(MemoryRegion(BASE, payload),))
        result = evaluate(load.program(), runtime)
        self.assertEqual(dict(result.outputs), {0: original})
        self.assertEqual(result.memory, runtime.memory)
        store = Stream()
        store.wait(store.store(store.input(), BASE + 0x2000))
        destination = MemoryRegion(BASE + 0x2000 - 7, b"before!" + b"?" * 2048 + b"after")
        inputs = RuntimeInputs({0: original}, memory=(destination,))
        self.assertEqual(evaluate(store.program(), inputs).memory,
                         (MemoryRegion(destination.address, b"before!" + payload + b"after"),))
        self.assertEqual(inputs.tiles[0], original)
        self.assertEqual(inputs.memory, (destination,))

    def test_adjacent_regions_cover_a_transfer_and_preserve_supplied_order_and_extents(self):
        payload = bytes(range(256)) * 4
        destination = BASE + 0x2000
        tail = MemoryRegion(destination + 777, b"?" * 247 + b"tail")
        source = MemoryRegion(BASE, payload)
        head = MemoryRegion(destination - 32, b"head" * 8 + b"?" * 777)
        untouched = MemoryRegion(BASE + 0x4000, b"unrelated")
        stream = Stream()
        ready = stream.await_(stream.load(BASE, "fp8"), "fp8")
        stream.wait(stream.store(ready, destination, "fp8"))
        runtime = RuntimeInputs(memory=(tail, source, head, untouched))
        self.assertEqual(evaluate(stream.program(), runtime).memory,
                         (MemoryRegion(tail.address, payload[777:] + b"tail"), source,
                          MemoryRegion(head.address, b"head" * 8 + payload[:777]), untouched))
        # A load can cross the same unaligned region boundary, too.
        split = RuntimeInputs(memory=(MemoryRegion(BASE + 13, payload[13:]), MemoryRegion(BASE, payload[:13]),
                                     MemoryRegion(destination, b"?" * 1024)))
        self.assertEqual(evaluate(stream.program(), split).memory[-1], MemoryRegion(destination, payload))
        self.assertEqual(runtime.memory, (tail, source, head, untouched))

    def test_overlapping_reads_can_complete_in_reverse_issue_order(self):
        payload = bytes(range(256)) * 4
        stream = Stream()
        first = stream.load(BASE, "fp8")
        second = stream.load(BASE, "fp8")
        second_ready = stream.await_(second, "fp8")
        first_ready = stream.await_(first, "fp8")
        stream.wait(stream.store(second_ready, BASE + 0x2000, "fp8"))
        stream.wait(stream.store(first_ready, BASE + 0x4000, "fp8"))
        runtime = RuntimeInputs(memory=(MemoryRegion(BASE, payload),
                                        MemoryRegion(BASE + 0x2000, b"?" * 1024),
                                        MemoryRegion(BASE + 0x4000, b"!" * 1024)))
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
        new_store = stream.store(reloaded, BASE + 0x4000, "fp8")
        stream.wait(new_store)
        stream.wait(old_store)
        runtime = RuntimeInputs({0: Tile("fp8", tuple(new))}, memory=(MemoryRegion(BASE, old),
                                MemoryRegion(BASE + 0x2000, b"?" * 1024), MemoryRegion(BASE + 0x4000, b"!" * 1024)))
        self.assertEqual(evaluate(stream.program(), runtime).memory,
                         (MemoryRegion(BASE, new), MemoryRegion(BASE + 0x2000, old), MemoryRegion(BASE + 0x4000, new)))
        self.assertEqual(runtime.memory[0].data, old)
        self.assertEqual(runtime.tiles[0].bits, tuple(new))

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
        self.assertEqual(runtime.tiles[0], tile)

    def test_fp8_store_exposes_converter_literals_including_reserved_mxu_codes(self):
        raw = (0x3F80, 0xBF80, 0x43E8, 0x43E9, 0x43F0, 0xC3E9, 0x7F80, 0xFF80, 0x7FC1, 0x8000, 0x0001)
        for converter, expected in (
            ("pack", (0x38, 0xB8, 0x7E, 0x7E, 0x7E, 0xFE, 0x7E, 0xFE, 0, 0, 0)),
            ("mxu", (0x38, 0xB8, 0x7E, 0x7F, 0x7F, 0xFF, 0x7E, 0xFE, 0, 0, 0)),
        ):
            with self.subTest(converter=converter):
                stream = Stream()
                source = stream.input()
                if converter == "pack":
                    packed = stream.pure("pack_fp8", source, "fp8", "{scale_code = 127 : i32}")
                else:
                    acc = stream.effect("mxu_load_acc_bf16", (source,), (tile_type("bf16"),),
                                        "!atlas.virtual_mxu_acc<0>", "{unit = 0 : i32}")
                    scale = stream.fresh()
                    stream.lines.append(f'{scale} = "atlas.virtual_scale_constant"() {{code = 127 : i32}} : () -> !atlas.virtual_scale')
                    packed = stream.effect("mxu_readout_fp8", (acc, scale), ("!atlas.virtual_mxu_acc<0>", "!atlas.virtual_scale"), tile_type("fp8"))
                stream.wait(stream.store(packed, BASE, "fp8"))
                runtime = RuntimeInputs({0: repeated(raw)}, memory=(MemoryRegion(BASE, b"?" * 1024),))
                payload = bytes(expected[i % len(expected)] for i in range(1024))
                self.assertEqual(evaluate(stream.program(), runtime).memory, (MemoryRegion(BASE, payload),))
                self.assertEqual(runtime.memory[0].data, b"?" * 1024)

    def test_unmapped_load_is_checked_only_on_the_executed_cfg_path(self):
        source = f'''module {{
          func.func @choose(%take: i1) -> {STATE} {{
            %s0 = "atlas.virtual_start"() : () -> {STATE}
            %addr = arith.constant -2147483648 : i32
            %size = arith.constant 2048 : i32
            cf.cond_br %take, ^load(%s0 : {STATE}), ^done(%s0 : {STATE})
          ^load(%ls: {STATE}):
            %s1, %pending = "atlas.virtual_dma_load_bf16"(%ls, %addr, %size) : ({STATE}, i32, i32) -> ({STATE}, !atlas.virtual_dma_load_bf16)
            %s2, %tile = "atlas.virtual_dma_await_bf16"(%s1, %pending) : ({STATE}, !atlas.virtual_dma_load_bf16) -> ({STATE}, !atlas.virtual_bf16)
            cf.br ^done(%s2 : {STATE})
          ^done(%ds: {STATE}):
            return %ds : {STATE}
          }}
        }}'''
        program = parse_program(source)
        untouched = MemoryRegion(BASE + 0x4000, b"guard")
        result = evaluate(program, RuntimeInputs(controls=(Scalar(1, 0),), memory=(untouched,)))
        self.assertEqual(dict(result.outputs), {})
        self.assertEqual(result.memory, (untouched,))
        with self.assertRaisesRegex(VirtualInterfaceError, "unmapped memory"):
            evaluate(program, RuntimeInputs(controls=(Scalar(1, 1),), memory=(untouched,)))

    def test_holes_and_partial_load_or_store_spans_fail_without_mutating_inputs(self):
        cases = (
            (MemoryRegion(BASE, b"?" * 1023),),
            (MemoryRegion(BASE + 1, b"?" * 1023),),
            (MemoryRegion(BASE, b"?" * 512), MemoryRegion(BASE + 513, b"!" * 511)),
        )
        for direction in ("load", "store"):
            stream = Stream()
            if direction == "load":
                stream.await_(stream.load(BASE, "fp8"), "fp8")
            else:
                stream.wait(stream.store(stream.input("fp8"), BASE, "fp8"))
            program = stream.program()
            for regions in cases:
                with self.subTest(direction=direction, addresses=tuple(r.address for r in regions)):
                    tile = repeated((0x7F, 0xFF, 0x01, 0x81), "fp8")
                    runtime = RuntimeInputs({0: tile} if direction == "store" else {}, memory=regions)
                    with self.assertRaisesRegex(VirtualInterfaceError, "unmapped memory"):
                        evaluate(program, runtime)
                    self.assertEqual(runtime.memory, regions)
                    if direction == "store":
                        self.assertEqual(runtime.tiles[0], tile)


if __name__ == "__main__":
    unittest.main()
