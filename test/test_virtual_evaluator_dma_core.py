"""Explicit DMA memory effects versus literals and the existing selected core."""

from __future__ import annotations

import unittest

from test_virtual_evaluator_core import compare_result, selected_core, tile_bytes
from atlas_virtual_evaluator import MemoryRegion, RuntimeInputs, Tile, evaluate, parse_program
from test_virtual_dma import STATE, copy, dma_load, dma_await, dma_store, dma_wait, wrap
from test_virtual_dma_lowering import MARKER
from test_virtual_lowering import ROOT, emitted, lower, object_words
from test_virtual_mxu_lowering import instructions


EXAMPLES = ROOT / "test/examples"
FP8_CODES = (0x00, 0x30, 0x38, 0x40, 0xB8)
EIGHT_FP8 = {0x00: 0x00, 0x30: 0x48, 0x38: 0x50, 0x40: 0x58, 0xB8: 0xD0}
TWICE_BF16 = {0x00: 0x0000, 0x30: 0x3F80, 0x38: 0x4000, 0x40: 0x4080, 0xB8: 0xC000}
READY_CODES = (0xBF80, 0x0000, 0x3F80, 0x4000, 0xBF00, 0x3F00)
READY_SUM = {0xBF80: 0x3F00, 0x0000: 0x3F00, 0x3F80: 0x3FC0,
             0x4000: 0x4020, 0xBF00: 0x3F00, 0x3F00: 0x3F80}


def pending_work() -> str:
    return wrap([
        f'%io0 = "atlas.virtual_start"() : () -> {STATE}',
        "%first_addr = arith.constant -2147483648 : i32",
        "%later_addr = arith.constant -2147481600 : i32",
        "%out_addr = arith.constant -2147475456 : i32",
        "%copy_addr = arith.constant -2147473408 : i32",
        "%size = arith.constant 2048 : i32",
        dma_load("bf16", addr="first_addr", handle="first_event"),
        dma_await("bf16", handle="first_event", result="ready"),
        dma_load("bf16", before="io2", after="io3", addr="later_addr", handle="later_event"),
        '%positive = "atlas.virtual_vpu_unary"(%ready) {kind = "relu"} : (!atlas.virtual_bf16) -> !atlas.virtual_bf16',
        dma_await("bf16", before="io3", after="io4", handle="later_event", result="later"),
        '%sum = "atlas.virtual_vpu_binary"(%positive, %later) {kind = "add"} : (!atlas.virtual_bf16, !atlas.virtual_bf16) -> !atlas.virtual_bf16',
        dma_store("bf16", before="io4", after="io5", value="sum", handle="first_store").replace("%addr,", "%out_addr,"),
        '%copied = "atlas.virtual_vpu_unary"(%sum) {kind = "mov"} : (!atlas.virtual_bf16) -> !atlas.virtual_bf16',
        dma_wait(before="io5", after="io6", handle="first_store"),
        dma_store("bf16", before="io6", after="io7", value="copied", handle="last_store").replace("%addr,", "%copy_addr,"),
        dma_wait(before="io7", after="io8", handle="last_store"),
    ], final="io8")


SOURCES = {
    "bf16_relu": (EXAMPLES / "virtual_dma_tiles.mlir").read_text(),
    "fp8_copy": copy("fp8").replace("%size = arith.constant 1024 : i32", "%size = arith.constant 1024 : i32\n%destination = arith.constant -2147481600 : i32")
                           .replace('"atlas.virtual_dma_store_fp8"(%io2, %tile, %addr,', '"atlas.virtual_dma_store_fp8"(%io2, %tile, %destination,'),
    "pending_work": pending_work(),
}
for fmt, filename in (("fp8", "virtual_dma_mxu.mlir"), ("bf16", "virtual_dma_mxu_bf16.mlir")):
    for unit in (0, 1):
        source = (EXAMPLES / filename).read_text().replace("unit = 0 : i32", f"unit = {unit} : i32")
        for handle in ("weight", "acc"):
            source = source.replace(f"virtual_mxu_{handle}<0>", f"virtual_mxu_{handle}<{unit}>")
        SOURCES[f"mxu_{fmt}_{unit}"] = source


def patterned(codes, phase: int, fmt: str = "bf16") -> Tile:
    return Tile(fmt, tuple(codes[(row * 7 + col * 3 + col // 16 + phase) % len(codes)] for row in range(32) for col in range(32)))


def mapped(buffers: dict[int, bytes], writes: dict[int, bytes]):
    regions = [MemoryRegion(address, data) for address, data in buffers.items()]
    blocks = []
    for address, data in sorted(buffers.items()):
        end = address + len(data)
        if blocks and blocks[-1][1] == address:
            blocks[-1] = (blocks[-1][0], end)
        else:
            blocks.append((address, end))
    for start, end in blocks:
        regions.extend((MemoryRegion(start - 64, b"\x51" * 64), MemoryRegion(end, b"\x62" * 64)))
    expected = tuple(MemoryRegion(region.address, writes[region.address] + region.data[len(writes[region.address]):])
                     if region.address in writes else region for region in regions)
    return RuntimeInputs(memory=tuple(regions)), expected


def case(name: str, phase: int):
    if name == "bf16_relu":
        tile = patterned((0xBF80, 0x3F80, 0x8000, 0x0001, 0x8001, 0x7FC1, 0xFFC1, 0x4000), phase)
        positive = Tile("bf16", tuple(0 if bits & 0x8000 else bits for bits in tile.bits))
        inputs, expected = mapped({0x80000800: tile_bytes(tile), 0x80001000: b"\xA5" * 2048}, {0x80001000: tile_bytes(positive)})
        return inputs, expected, (64, 64)
    if name == "fp8_copy":
        # Raw DMA transports every FP8 encoding, including signed zeros/NaNs.
        tile = Tile("fp8", tuple((index + phase) % 256 for index in range(1024)))
        inputs, expected = mapped({0x80000000: bytes(tile.bits) + b"\xC3" * 1024,
                                   0x80000800: b"\xA5" * 1024 + b"\xD4" * 1024}, {0x80000800: bytes(tile.bits)})
        return inputs, expected, (32, 32)
    if name == "pending_work":
        ready = patterned(READY_CODES, phase)
        result = tile_bytes(Tile("bf16", tuple(READY_SUM[bits] for bits in ready.bits)))
        inputs, expected = mapped({0x80000000: tile_bytes(ready), 0x80000800: tile_bytes(Tile("bf16", (0x3F00,) * 1024)),
                                   0x80002000: b"\xA5" * 2048, 0x80002800: b"\xB6" * 2048},
                                  {0x80002000: result, 0x80002800: result})
        return inputs, expected, (128, 128)
    fmt = name.split("_")[1]
    x, shift = patterned(FP8_CODES, phase, "fp8"), phase % 7
    weight = bytes(0x38 if k == (col + shift) % 32 else 0 for col in range(32) for k in range(32))
    codes = tuple(x.bits[row * 32 + (col + shift) % 32] for row in range(32) for col in range(32))
    # Reset+continue gives 2X. MXU readout code129 multiplies by four,
    # giving 8X in FP8; BF16 readout preserves the unscaled 2X.
    payload = bytes(EIGHT_FP8[code] for code in codes) if fmt == "fp8" else tile_bytes(Tile("bf16", tuple(TWICE_BF16[code] for code in codes)))
    initial = b"\xA5" * len(payload) + (b"\xD4" * 1024 if fmt == "fp8" else b"")
    inputs, expected = mapped({0x90000000: bytes(x.bits), 0x90000400: weight, 0x90001000: initial}, {0x90001000: payload})
    return inputs, expected, (64, len(payload) // 32)


class VirtualEvaluatorDMACoreTest(unittest.TestCase):
    def literal_result(self, name: str, phase: int):
        inputs, memory, traffic = case(name, phase)
        original = inputs.memory
        result = evaluate(parse_program(SOURCES[name]), inputs)
        self.assertEqual(dict(result.outputs), {})
        self.assertEqual(result.memory, memory)
        self.assertEqual(inputs.memory, original)
        return inputs, result, traffic

    def test_original_sources_match_independent_final_memory_on_fresh_inputs(self) -> None:
        for name in SOURCES:
            for phase in (0, 3):
                with self.subTest(name=name, phase=phase):
                    self.literal_result(name, phase)

    def test_lowering_emission_and_issue_completion_separation(self) -> None:
        for name, source in SOURCES.items():
            with self.subTest(name=name):
                machine = lower(source)
                self.assertNotIn("atlas.virtual_", machine.replace(MARKER, ""))
                self.assertTrue(emitted(machine))
        entries = instructions(lower(SOURCES["pending_work"]))
        launches = [index for index, entry in enumerate(entries) if entry["operation"] == "atlas.dma" and MARKER in entry["fields"]]
        for launch, kind in ((launches[1], "relu"), (launches[2], "mov")):
            marker = entries[launch]["fields"][MARKER]
            completion = next(index for index in range(launch + 1, len(entries))
                              if entries[index]["operation"] == "atlas.dma_wait" and entries[index]["fields"].get(MARKER) == marker)
            self.assertTrue(any(entry["operation"] == "atlas.vpu_unary" and entry["fields"]["kind"] == kind for entry in entries[launch + 1:completion]))
            self.assertEqual(entries[launch]["fields"]["channel"], entries[completion]["fields"]["channel"])

    def test_llvm_object_words_match_emission(self) -> None:
        for name, source in SOURCES.items():
            with self.subTest(name=name):
                machine = lower(source)
                self.assertEqual(object_words(machine), emitted(machine))

    def test_selected_core_matches_final_memory_and_preserves_sources_padding_guards(self) -> None:
        with selected_core(max_cycles=100000) as run:
            for name, source in SOURCES.items():
                machine = lower(source)
                words = object_words(machine)
                self.assertEqual(words, emitted(machine))
                for phase in (0, 3):
                    with self.subTest(name=name, phase=phase):
                        inputs, expected, traffic = self.literal_result(name, phase)
                        observed = run(words, [(region.address, region.data) for region in inputs.memory])
                        self.assertTrue(observed.halted, "selected core did not halt")
                        self.assertEqual((observed.reads, observed.writes), traffic)
                        compare_result(expected, observed.slave.captured)


if __name__ == "__main__":
    unittest.main()
