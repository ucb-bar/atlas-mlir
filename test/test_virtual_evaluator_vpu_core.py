"""Original virtual MOV/ADD/ReLU versus selected core, plus a physical PACK probe.

The PACK probe checks converter bits and register/DRAM transport against an
independent physical permutation. It does not qualify virtual PACK lowering or
an FP8 consumer. LLVM word agreement and unused-input/pack lowering are separate
checks; selected-core checks use the existing runner and its prerequisite policy.
"""

from __future__ import annotations

from pathlib import Path
import unittest

from test_virtual_evaluator_core import compare_result, selected_core, tile_bytes
from atlas_virtual_evaluator import MemoryRegion, RuntimeInputs, Tile, evaluate, evaluate_tile_operation, operation_name, parse_program
from test_virtual_lowering import emitted, lower, object_words


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "test/examples/virtual_bf16_vpu_program.mlir"
PACK_SOURCE = ROOT / "test/examples/vpu_pack_reference.mlir"
INPUT_BASE, OUTPUT_BASE = 0x90000000, 0x90004000

# Exact BF16 additions: selected FP32 RNE addition followed by BF16 truncation,
# with HardFloat's canonical positive quiet NaN. These literal expectations do
# not call RtlNumerics or a production quantizer. 1 + 3/512 distinguishes chop
# from BF16 nearest-even. Cancellation and exceptional encodings check raw I/O.
ADD_CASES = (
    (0x3F80, 0x3BC0, 0x3F80), (0xBF80, 0xBBC0, 0xBF80),
    (0x3F80, 0x3C40, 0x3F81), (0x3F80, 0x8001, 0x3F80),
    (0x0080, 0x8081, 0x8001), (0x0081, 0x8080, 0x0001),
    (0x0001, 0x0001, 0x0002), (0x007F, 0x0001, 0x0080),
    (0x0001, 0x8001, 0x0000), (0x8000, 0x8000, 0x8000),
    (0x0000, 0x8000, 0x0000), (0x3F80, 0xBF80, 0x0000),
    (0x7F7F, 0x7F7F, 0x7F80), (0xFF7F, 0xFF7F, 0xFF80),
    (0x7F80, 0xFF80, 0x7FC0), (0xFFA1, 0x3F80, 0x7FC0),
)

# Source-derived unit-scale FP8Pack expectations: even/odd RNE ties, exponent
# carry, flush before/after the minimum-normal boundary, saturation and specials.
# In particular rounded 480 is 0x7e here, unlike the MXU converter's 0x7f.
PACK_CASES = (
    (0x0000, 0x00), (0x8000, 0x00), (0x0001, 0x00), (0x007F, 0x00),
    (0x8001, 0x00), (0x807F, 0x00), (0x7F81, 0x00), (0x7FC0, 0x00),
    (0xFF81, 0x00), (0xFFC0, 0x00), (0x7F80, 0x7E), (0xFF80, 0xFE),
    (0x3F80, 0x38), (0xBF80, 0xB8), (0x3F87, 0x38), (0x3F88, 0x38),
    (0x3F89, 0x39), (0x3F98, 0x3A), (0xBF98, 0xBA), (0x3FF7, 0x3F),
    (0x3FF8, 0x40), (0xBFF8, 0xC0), (0x3C60, 0x00), (0x3C77, 0x00),
    (0x3C78, 0x08), (0x3C80, 0x08), (0xBC77, 0x00), (0xBC78, 0x88),
    (0x43E0, 0x7E), (0x43E8, 0x7E), (0x43E9, 0x7E), (0x43F0, 0x7E),
    (0x7F7F, 0x7E), (0xC3E0, 0xFE), (0xC3E8, 0xFE), (0xC3E9, 0xFE),
    (0xC3F0, 0xFE), (0xFF7F, 0xFE),
)

PACK_VIRTUAL = '''module {
  %s = "atlas.virtual_start"() : () -> !atlas.virtual_state
  %s1, %a = "atlas.virtual_input_bf16"(%s) {index = 0 : i32} : (!atlas.virtual_state) -> (!atlas.virtual_state, !atlas.virtual_bf16)
  %packed = "atlas.virtual_pack_fp8"(%a) {scale_code = 127 : i32} : (!atlas.virtual_bf16) -> !atlas.virtual_fp8
}'''

# FP8 values deliberately have no consumer in this smoke check. Consumer and
# completed-store comparisons belong to the later MXU/DMA execution slices.
FP8_SMOKE = '''module {
  func.func @unused_fp8() -> !atlas.virtual_state attributes {atlas.input_dram_base = 2415919104 : i64, atlas.output_dram_base = 2415935488 : i64} {
    %s = "atlas.virtual_start"() : () -> !atlas.virtual_state
    %s1, %a = "atlas.virtual_input_bf16"(%s) {index = 0 : i32} : (!atlas.virtual_state) -> (!atlas.virtual_state, !atlas.virtual_bf16)
    %s2, %f = "atlas.virtual_input_fp8"(%s1) {index = 1 : i32} : (!atlas.virtual_state) -> (!atlas.virtual_state, !atlas.virtual_fp8)
    %packed = "atlas.virtual_pack_fp8"(%a) {scale_code = 127 : i32} : (!atlas.virtual_bf16) -> !atlas.virtual_fp8
    %s3 = "atlas.virtual_output_bf16"(%s2, %a) {index = 0 : i32} : (!atlas.virtual_state, !atlas.virtual_bf16) -> !atlas.virtual_state
    return %s3 : !atlas.virtual_state
  }
}'''


def panel_indices(count: int, phase: int) -> tuple[int, ...]:
    # The additional column-half term breaks identical left/right patterns.
    return tuple((row * 5 + col * 3 + col // 16 + phase) % count for row in range(32) for col in range(32))


def guards(input_size: int, output_size: int) -> tuple[MemoryRegion, ...]:
    return tuple(MemoryRegion(address, bytes([value]) * 64) for address, value in (
        (INPUT_BASE - 64, 0x51), (INPUT_BASE + input_size, 0x62),
        (OUTPUT_BASE - 64, 0x73), (OUTPUT_BASE + output_size, 0x84),
    ))


def vpu_inputs(phase: int) -> tuple[RuntimeInputs, Tile, Tile]:
    indices = panel_indices(len(ADD_CASES), phase)
    tiles = {operand: Tile("bf16", tuple(ADD_CASES[index][operand] for index in indices)) for operand in (0, 1)}
    summed = Tile("bf16", tuple(ADD_CASES[index][2] for index in indices))
    # The selected ReLU returns raw positive encodings unchanged and +0 for
    # every sign-set encoding, including -0, negative subnormals and -infinity.
    positive = Tile("bf16", tuple(0 if bits & 0x8000 else bits for bits in summed.bits))
    memory = tuple(MemoryRegion(INPUT_BASE + operand * 2048, tile_bytes(tile)) for operand, tile in tiles.items()) + guards(4096, 8192)
    return RuntimeInputs(tiles, memory=memory), summed, positive


def pack_input(phase: int) -> tuple[Tile, Tile]:
    indices = panel_indices(len(PACK_CASES), phase)
    return (Tile("bf16", tuple(PACK_CASES[index][0] for index in indices)),
            Tile("fp8", tuple(PACK_CASES[index][1] for index in indices)))


def physical_pack_bytes(logical: Tile) -> bytes:
    # VectorFSM streams all rows of the left BF16 register, then the right.
    # FP8Pack pairs successive 16-lane rows. Derive this mapping from those
    # source rules rather than the compiler's production relayout or allocator.
    return bytes(logical.bits[((p // 16) % 32) * 32 + (p // 512) * 16 + p % 16] for p in range(1024))


def pack_operation():
    program = parse_program(PACK_VIRTUAL)
    return next(op for op in program.operations if operation_name(op) == "atlas.virtual_pack_fp8")


class VirtualEvaluatorVPUCoreTest(unittest.TestCase):
    def assert_vpu_reference(self, program, inputs, summed, positive):
        expected = evaluate(program, inputs)
        self.assertEqual(dict(expected.outputs), {
            0: inputs.tiles[0], 1: inputs.tiles[0], 2: summed, 3: positive,
        })
        self.assertEqual(expected.memory, inputs.memory)
        return expected

    def test_original_vpu_evaluation_matches_independent_raw_literals(self) -> None:
        program = parse_program(SOURCE.read_text())
        for phase in (0, 5):
            with self.subTest(phase=phase):
                self.assert_vpu_reference(program, *vpu_inputs(phase))

    def test_pack_operation_matches_independent_raw_literals_and_layout(self) -> None:
        op = pack_operation()
        for phase in (0, 11):
            with self.subTest(phase=phase):
                source, literal = pack_input(phase)
                actual = evaluate_tile_operation(op, (source,))
                self.assertEqual(actual, literal)
                self.assertNotEqual(physical_pack_bytes(actual)[:32], bytes(actual.bits[:32]))

    def test_lowering_and_emission_for_vpu_and_unused_fp8_forms(self) -> None:
        machine = lower(SOURCE.read_text())
        for kind in ("mov", "add", "relu"):
            self.assertIn(f'kind = "{kind}"', machine)
        self.assertTrue(emitted(machine))
        smoke = lower(FP8_SMOKE)
        self.assertIn('"atlas.vpu_pack"', smoke)
        self.assertTrue(emitted(smoke))
        probe = PACK_SOURCE.read_text()
        self.assertNotIn('"atlas.mxu_', probe)
        self.assertNotIn('direction = "fp8_to_bf16"', probe)
        self.assertTrue(emitted(probe))

    def test_llvm_words_match_emission_for_both_fixtures_and_fp8_smoke(self) -> None:
        for name, machine in (
            ("vpu", lower(SOURCE.read_text())),
            ("physical_pack", PACK_SOURCE.read_text()),
            ("unused_fp8", lower(FP8_SMOKE)),
        ):
            with self.subTest(name=name):
                self.assertEqual(object_words(machine), emitted(machine))

    def test_selected_core_matches_original_vpu_program_on_fresh_panels(self) -> None:
        with selected_core() as run:
            program = parse_program(SOURCE.read_text())
            words = object_words(lower(SOURCE.read_text()))
            for phase in (0, 5):
                with self.subTest(phase=phase):
                    inputs, summed, positive = vpu_inputs(phase)
                    expected = self.assert_vpu_reference(program, inputs, summed, positive)
                    preload = [(region.address, region.data) for region in inputs.memory]
                    preload.append((OUTPUT_BASE, b"\xA5" * 8192))
                    result = run(words, preload)
                    self.assertTrue(result.halted, "selected core did not halt")
                    self.assertEqual((result.reads, result.writes), (128, 256))
                    compare_result(expected, result.slave.captured, output_base=OUTPUT_BASE)

    def test_selected_core_unit_scale_physical_pack_converter_and_transport(self) -> None:
        # This hand-authored machine probe is deliberately separate from
        # full-stream virtual PACK lowering and from MXU/DMA evaluator execution.
        with selected_core() as run:
            words = object_words(PACK_SOURCE.read_text())
            op = pack_operation()
            for phase in (0, 11):
                with self.subTest(phase=phase):
                    source, literal = pack_input(phase)
                    packed = evaluate_tile_operation(op, (source,))
                    self.assertEqual(packed, literal)
                    preserved = (MemoryRegion(INPUT_BASE, tile_bytes(source)),
                                 MemoryRegion(OUTPUT_BASE + 1024, b"\xC3" * 1024),
                                 *guards(2048, 2048))
                    preload = [(region.address, region.data) for region in preserved]
                    preload.append((OUTPUT_BASE, b"\xA5" * 1024))
                    result = run(words, preload)
                    self.assertTrue(result.halted, "physical PACK probe did not halt")
                    self.assertEqual((result.reads, result.writes), (64, 32))
                    self.assertEqual(result.slave.captured(OUTPUT_BASE, 1024), physical_pack_bytes(packed))
                    for region in preserved:
                        self.assertEqual(result.slave.captured(region.address, len(region.data)), region.data,
                                         f"preserved memory changed at 0x{region.address:08x}")


if __name__ == "__main__":
    unittest.main()
