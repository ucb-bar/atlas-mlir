"""Both MXUs' virtual semantics versus independent literals and selected core."""

from __future__ import annotations

from itertools import product
import unittest

from test_virtual_evaluator_core import compare_result, selected_core, tile_bytes
from atlas_virtual_evaluator import MemoryRegion, RuntimeInputs, Tile, evaluate, parse_program
from test_virtual_lowering import emitted, lower, object_words
from test_virtual_mxu_extended import seeded_chain
from test_virtual_mxu_handles import STATE, FP8, BF16, load, reset, accumulate, readout


INPUT_BASE, OUTPUT_BASE = 0x90000000, 0x90004000
FP8_TO_BF16 = {0x00: 0x0000, 0x30: 0x3F00, 0x38: 0x3F80, 0x40: 0x4000, 0xB8: 0xBF80}
TRIPLE_BF16 = {0x00: 0x0000, 0x30: 0x3FC0, 0x38: 0x4040, 0x40: 0x40C0, 0xB8: 0xC040}


def program(body: str, *, extra_input: bool = False) -> str:
    extra = (f'%io3, %next_x = "atlas.virtual_input_fp8"(%io2) {{index = 2 : i32}} : ({STATE}) -> ({STATE}, {FP8})' if extra_input else "")
    return f'''module {{
      func.func @mxu0() -> {STATE} attributes {{atlas.input_dram_base = 2415919104 : i64, atlas.output_dram_base = 2415935488 : i64}} {{
        %io0 = "atlas.virtual_start"() : () -> {STATE}
        %io1, %x = "atlas.virtual_input_fp8"(%io0) {{index = 0 : i32}} : ({STATE}) -> ({STATE}, {FP8})
        %io2, %w = "atlas.virtual_input_fp8"(%io1) {{index = 1 : i32}} : ({STATE}) -> ({STATE}, {FP8})
        {extra}
        {body}
      }}
    }}'''


LEGACY = program(f'''
    %first = "atlas.virtual_mxu_matmul"(%x, %w) {{unit = 0 : i32}} : ({FP8}, {FP8}) -> {BF16}
    %positive = "atlas.virtual_vpu_unary"(%first) {{kind = "relu"}} : ({BF16}) -> {BF16}
    %packed = "atlas.virtual_pack_fp8"(%positive) {{scale_code = 127 : i32}} : ({BF16}) -> {FP8}
    %second = "atlas.virtual_mxu_matmul"(%packed, %w) {{unit = 0 : i32}} : ({FP8}, {FP8}) -> {BF16}
    %out0 = "atlas.virtual_output_bf16"(%io2, %first) {{index = 0 : i32}} : ({STATE}, {BF16}) -> {STATE}
    %out1 = "atlas.virtual_output_bf16"(%out0, %second) {{index = 1 : i32}} : ({STATE}, {BF16}) -> {STATE}
    return %out1 : {STATE}
''')
CONTINUED = program("\n".join((
    load("io3", "s0", "weight"), reset("s0", "s1", "a0"),
    accumulate("s1", "s2", "a1", "a0").replace("%x,", "%next_x,"),
    readout("s2", "s3", "y", "a1"),
    f'%out = "atlas.virtual_output_bf16"(%s3, %y) {{index = 0 : i32}} : ({STATE}, {BF16}) -> {STATE}',
    f"return %out : {STATE}",
)), extra_input=True)
SOURCES = {"legacy_pack": LEGACY, "continuation": CONTINUED,
           "seed_bf16": seeded_chain(0, "bf16", 127), "seed_fp8": seeded_chain(0, "fp8", 127)}


def source_for_unit(name: str, unit: int) -> str:
    return (SOURCES[name].replace("unit = 0 : i32", f"unit = {unit} : i32")
            .replace("virtual_mxu_weight<0>", f"virtual_mxu_weight<{unit}>")
            .replace("virtual_mxu_acc<0>", f"virtual_mxu_acc<{unit}>"))


def inputs_with_guards(tiles: dict[int, Tile], output_count: int) -> RuntimeInputs:
    regions = []
    for index, tile in tiles.items():
        data = tile_bytes(tile) if tile.format == "bf16" else bytes(tile.bits) + bytes([0xC1 + index]) * 1024
        regions.append(MemoryRegion(INPUT_BASE + index * 2048, data))
    regions.extend(MemoryRegion(address, bytes([value]) * 64) for address, value in (
        (INPUT_BASE - 64, 0x51), (INPUT_BASE + (max(tiles) + 1) * 2048, 0x62),
        (OUTPUT_BASE - 64, 0x73), (OUTPUT_BASE + output_count * 2048, 0x84),
    ))
    return RuntimeInputs(tiles, memory=tuple(regions))


def panel(phase: int) -> Tile:
    codes = tuple(FP8_TO_BF16)
    return Tile("fp8", tuple(codes[(row * 7 + col * 3 + col // 16 + phase) % len(codes)] for row in range(32) for col in range(32)))


def identity_weight(shift: int = 0) -> Tile:
    return Tile("fp8", tuple(0x38 if k == (column + shift) % 32 else 0 for column in range(32) for k in range(32)))


def case(name: str, phase: int, unit: int = 0):
    if name == "legacy_pack":
        x, shift = panel(phase), phase % 7
        first = Tile("bf16", tuple(FP8_TO_BF16[x.bits[row * 32 + (col + shift) % 32]] for row in range(32) for col in range(32)))
        final = Tile("bf16", tuple(0 if (code := x.bits[row * 32 + (col + 2 * shift) % 32]) == 0xB8 else FP8_TO_BF16[code] for row in range(32) for col in range(32)))
        return inputs_with_guards({0: x, 1: identity_weight(shift)}, 2), {0: first, 1: final}, (64, 128)
    if name == "continuation":
        # MXU0 rounds each FMA: +1, +half-ULP, -half-ULP becomes 0x3f7f.
        # MXU1 rounds once per operation, preserving +1 (0x3f80) here.
        k0, columns = (phase % 10) * 3, (3 + phase, 20 + phase)
        first, next_x, weight, expected = ([0] * 1024 for _ in range(4))
        for row in range(32):
            negative = (row * 5 + phase) % 3 == 0
            first[row * 32 + k0] = 0xB8 if negative else 0x38
            next_x[row * 32 + k0 + 1] = 0x98 if negative else 0x18
            next_x[row * 32 + k0 + 2] = 0x18 if negative else 0x98
            for col in columns:
                expected[row * 32 + col] = (0x3F7F if unit == 0 else 0x3F80) | (0x8000 if negative else 0)
        for col in columns:
            weight[col * 32 + k0:col * 32 + k0 + 3] = (0x38, 0x18, 0x18)
        tiles = {0: Tile("fp8", first), 1: Tile("fp8", weight), 2: Tile("fp8", next_x)}
        return inputs_with_guards(tiles, 1), {0: Tile("bf16", expected)}, (96, 64)
    x = panel(phase)
    seed = Tile("bf16", tuple(FP8_TO_BF16[code] for code in x.bits))
    # Identity products give 2X before FP8 readout/reseed and exactly 3X
    # afterwards. All intermediate values have exact FP8/BF16 encodings.
    expected = Tile("bf16", tuple(TRIPLE_BF16[code] for code in x.bits))
    return inputs_with_guards({0: x, 1: identity_weight(), 2: seed}, 1), {0: expected}, (128, 64)


class VirtualEvaluatorMXUCoreTest(unittest.TestCase):
    def literal_result(self, name: str, phase: int, unit: int = 0):
        inputs, outputs, traffic = case(name, phase, unit)
        original = dict(inputs.tiles)
        result = evaluate(parse_program(source_for_unit(name, unit)), inputs)
        self.assertEqual(dict(result.outputs), outputs)
        self.assertEqual(result.memory, inputs.memory)
        self.assertEqual(dict(inputs.tiles), original)
        return inputs, result, traffic

    def test_evaluator_matches_whole_tile_literals_on_fresh_panels(self) -> None:
        for unit, name in product((0, 1), SOURCES):
            for phase in (0, 3):
                with self.subTest(unit=unit, name=name, phase=phase):
                    self.literal_result(name, phase, unit)

    def test_lowering_and_emission_cover_pack_continuation_and_seeds(self) -> None:
        for unit, name in product((0, 1), SOURCES):
            with self.subTest(unit=unit, name=name):
                machine = lower(source_for_unit(name, unit))
                self.assertNotIn('"atlas.virtual_', machine)
                self.assertTrue(emitted(machine))
                if name == "legacy_pack":
                    self.assertEqual(machine.count('"atlas.mxu_matmul"'), 2)
                    self.assertIn('"atlas.vpu_pack"', machine)
                elif name == "continuation":
                    self.assertIn("accumulate = false", machine)
                    self.assertIn("accumulate = true", machine)
                else:
                    self.assertIn(f'kind = "acc_{name.removeprefix("seed_")}"', machine)
                    self.assertIn('kind = "acc_fp8"', machine)
                    self.assertIn('format = "fp8"', machine)
                    self.assertIn('format = "bf16"', machine)

    def test_llvm_object_words_match_emission_for_all_forms(self) -> None:
        for unit, name in product((0, 1), SOURCES):
            with self.subTest(unit=unit, name=name):
                machine = lower(source_for_unit(name, unit))
                self.assertEqual(object_words(machine), emitted(machine))

    def test_selected_core_matches_evaluator_and_preserves_inputs_padding_guards(self) -> None:
        with selected_core(max_cycles=100000) as run:
            for unit, name in product((0, 1), SOURCES):
                machine = lower(source_for_unit(name, unit))
                words = object_words(machine)
                self.assertEqual(words, emitted(machine))
                for phase in (0, 3):
                    with self.subTest(unit=unit, name=name, phase=phase):
                        inputs, expected, traffic = self.literal_result(name, phase, unit)
                        preload = [(region.address, region.data) for region in inputs.memory]
                        preload.append((OUTPUT_BASE, b"\xA5" * (len(expected.outputs) * 2048)))
                        observed = run(words, preload)
                        self.assertTrue(observed.halted, "selected core did not halt")
                        self.assertEqual((observed.reads, observed.writes), traffic)
                        compare_result(expected, observed.slave.captured, output_base=OUTPUT_BASE)


if __name__ == "__main__":
    unittest.main()
