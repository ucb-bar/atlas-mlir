"""Compare virtual tile semantics with the existing selected-core machine runner."""

from __future__ import annotations

from contextlib import contextmanager
import os
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
from atlas_virtual_evaluator import EvaluationResult, MemoryRegion, RuntimeInputs, Tile, evaluate, parse_program  # noqa: E402
from test_virtual_lowering import emitted, lower, object_words  # noqa: E402


SOURCE = ROOT / "test/examples/virtual_bf16_shared_relu.mlir"
INPUT_BASE = 0x90000000
OUTPUT_BASE = 0x90001000


def tile_bytes(tile: Tile) -> bytes:
    # A BF16 register pair carries the left 16 columns, then the right 16.
    return b"".join(tile.bits[row * 32 + col].to_bytes(2, "little") for half in (0, 16) for row in range(32) for col in range(half, half + 16))


def compare_result(expected: EvaluationResult, captured) -> None:
    for index, tile in expected.outputs.items():
        observed = captured(OUTPUT_BASE + index * 2048, 2048)
        if len(observed) != 2048:
            raise AssertionError(f"output {index}: expected 2048 bytes, got {len(observed)}")
        for row in range(32):
            for col in range(32):
                offset = (col // 16) * 1024 + (row * 16 + col % 16) * 2
                bits = int.from_bytes(observed[offset:offset + 2], "little")
                wanted = tile.bits[row * 32 + col]
                if bits != wanted:
                    raise AssertionError(f"output {index} tile[{row},{col}]: expected 0x{wanted:04x}, got 0x{bits:04x}")
    for region in expected.memory:
        if captured(region.address, len(region.data)) != region.data:
            raise AssertionError(f"preserved memory changed at 0x{region.address:08x}")


def runtime_inputs(phase: int) -> RuntimeInputs:
    bits = tuple(0 if (i + phase) % 19 == 0 else (0x3E00 + i + phase) | (0x8000 if (i // 32 + i + phase) % 2 else 0) for i in range(1024))
    tile = Tile("bf16", bits)
    guards = tuple(MemoryRegion(address, bytes([value]) * 64) for address, value in (
        (INPUT_BASE - 64, 0x5A), (INPUT_BASE + 2048, 0x6B), (OUTPUT_BASE - 64, 0x7C), (OUTPUT_BASE + 4096, 0x8D),
    ))
    return RuntimeInputs({0: tile}, memory=(MemoryRegion(INPUT_BASE, tile_bytes(tile)), *guards))


@contextmanager
def selected_core():
    keys = ("ATLAS_ARC_MODEL", "ATLAS_ARC_STATE", "ATLAS_MODELIR_ROOT", "ATLAS_LLVM_BIN")
    missing = [key for key in keys if not os.environ.get(key)]
    if missing:
        message = "selected-core comparison requires " + ", ".join(missing)
        if os.environ.get("ATLAS_REQUIRE_VIRTUAL_CORE") == "1":
            raise RuntimeError(message)
        raise unittest.SkipTest(message)
    model, state, modelir = (Path(os.environ[key]).resolve(strict=True) for key in keys[:3])
    previous = Path.cwd()
    sys.path.insert(0, str(modelir))
    try:
        os.chdir(modelir)
        from mlc.backends import cosim_atlas

        original = cosim_atlas.CosimCore

        class SelectedCore(original):
            def peek(self, name: str) -> int:
                return super().peek("scalar/halt_now" if name == "io_halted" else name)

            def poke(self, name: str, value: int) -> None:
                if name in ("io_dmaTL_d_bits_opcode", "io_dmaTL_d_bits_size"):
                    if name in self._S:
                        raise AssertionError(f"unexpected live ARC input {name}")
                    return
                super().poke(name, value)

        cosim_atlas.CosimCore = SelectedCore
        try:
            yield lambda words, preload: cosim_atlas.run_program(model, state, words, preload=preload, max_cycles=20000)
        finally:
            cosim_atlas.CosimCore = original
    finally:
        os.chdir(previous)
        sys.path.remove(str(modelir))


class VirtualEvaluatorCoreTest(unittest.TestCase):
    def test_lowering_and_llvm_preserve_emitted_words_for_both_modes(self) -> None:
        machine = lower(SOURCE.read_text())
        self.assertEqual(machine.count('kind = "relu"'), 1)
        changed = machine.replace('kind = "relu"', 'kind = "mov"')
        words, changed_words = emitted(machine), emitted(changed)
        self.assertEqual(object_words(machine), words)
        self.assertEqual(object_words(changed), changed_words)
        self.assertEqual(len(words), len(changed_words))
        self.assertEqual(sum(left != right for left, right in zip(words, changed_words)), 1)

    def test_comparison_reports_tile_coordinates_and_preservation_failures(self) -> None:
        inputs = runtime_inputs(0)
        expected = evaluate(parse_program(SOURCE.read_text()), inputs)
        buffers = {region.address: region.data for region in inputs.memory}
        buffers.update({OUTPUT_BASE + index * 2048: tile_bytes(tile) for index, tile in expected.outputs.items()})
        captured = lambda address, size: buffers[address][:size]
        compare_result(expected, captured)
        original = buffers[OUTPUT_BASE + 2048]
        changed = bytearray(original)
        # Row 2, column 19 is in the second physical register half.
        offset = 1024 + (2 * 16 + 3) * 2
        changed[offset] ^= 1
        buffers[OUTPUT_BASE + 2048] = bytes(changed)
        with self.assertRaisesRegex(AssertionError, r"output 1 tile\[2,19\]"):
            compare_result(expected, captured)
        buffers[OUTPUT_BASE + 2048] = original
        for region in inputs.memory:
            buffers[region.address] = b"\x00" * len(region.data)
            with self.assertRaisesRegex(AssertionError, "preserved memory changed"):
                compare_result(expected, captured)
            buffers[region.address] = region.data

    def test_selected_core_matches_both_outputs_and_detects_relu_to_mov(self) -> None:
        with selected_core() as run:
            program = parse_program(SOURCE.read_text())
            machine = lower(SOURCE.read_text())
            self.assertEqual(machine.count('kind = "relu"'), 1)
            words = object_words(machine)
            changed_words = object_words(machine.replace('kind = "relu"', 'kind = "mov"'))
            for phase in (0, 257):
                with self.subTest(phase=phase):
                    inputs = runtime_inputs(phase)
                    expected = evaluate(program, inputs)
                    preload = [(region.address, region.data) for region in inputs.memory]
                    preload += [(OUTPUT_BASE, b"\xA5" * 4096)]
                    result = run(words, preload)
                    self.assertTrue(result.halted, "selected core did not halt")
                    self.assertEqual((result.reads, result.writes), (64, 128))
                    compare_result(expected, result.slave.captured)
                    corrupted = run(changed_words, preload)
                    self.assertTrue(corrupted.halted, "mutated program did not halt")
                    self.assertEqual((corrupted.reads, corrupted.writes), (64, 128))
                    with self.assertRaisesRegex(AssertionError, "output 1 tile"):
                        compare_result(expected, corrupted.slave.captured)
                    # The mutation should produce X twice and preserve the guards.
                    moved = EvaluationResult({0: inputs.tiles[0], 1: inputs.tiles[0]}, inputs.memory)
                    compare_result(moved, corrupted.slave.captured)


if __name__ == "__main__":
    unittest.main()
