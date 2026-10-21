"""Boundary ABI conversion exposes transfers without changing tile addresses."""
import sys
import unittest

from test_virtual_ssa import ROOT, run
from test_virtual_scheduling import ABI, function, start, inp, outp

sys.path.insert(0, str(ROOT / "tools"))

PASS = "--lower-atlas-virtual-boundary-to-dma"
INPUT_BASE, OUTPUT_BASE = 0x90000000, 0x90010000


def signed(value: int) -> int:
    """An i32 constant as MLIR prints it."""
    return value - (1 << 32) if value >= 1 << 31 else value


def tile_bytes(tile) -> bytes:
    """A BF16 tile in DRAM: two halves of 16 columns, each 32 rows, little-endian."""
    return b"".join(tile.bits[row * 32 + col].to_bytes(2, "little")
                    for half in (0, 16) for row in range(32) for col in range(half, half + 16))


class VirtualBoundaryToDMATest(unittest.TestCase):
    def checked(self, source, *options):
        result = run("atlas-opt", source, *options)
        self.assertEqual(result.returncode, 0, result.stderr)
        return result.stdout

    def test_mixed_formats_preserve_slot_stride_size_and_state(self):
        source = function("mixed", [start(), inp("io0", "io1", "a", 3, "fp8"),
                                    inp("io1", "io2", "b", 5),
                                    outp("io2", "io3", "b", 1),
                                    "return %io3 : !atlas.virtual_state"])
        converted = self.checked(source, PASS, "--verify-atlas-virtual-stream")
        self.assertNotIn('"atlas.virtual_input_', converted)
        self.assertNotIn('"atlas.virtual_output_', converted)
        self.assertEqual(converted.count('"atlas.virtual_dma_load_fp8"'), 1)
        self.assertEqual(converted.count('"atlas.virtual_dma_load_bf16"'), 1)
        self.assertEqual(converted.count('"atlas.virtual_dma_store_bf16"'), 1)
        # Each boundary slot is 2 KiB; FP8 moves 1 KiB of it.
        for value in (INPUT_BASE + 3 * 2048, INPUT_BASE + 5 * 2048, OUTPUT_BASE + 1 * 2048, 1024, 2048):
            self.assertRegex(converted, rf"arith.constant {signed(value)} : i32")
        self.assertEqual(self.checked(converted, PASS), converted)
        self.checked(converted, "--schedule-atlas-virtual",
                     "--lower-atlas-virtual-to-machine", "--insert-atlas-dma-waits",
                     "--schedule-atlas-stream", "--verify-atlas-timing",
                     "--verify-atlas-generated-schedule", "--verify-atlas-machine-stream")

    def test_bf16_payload_and_output_match_before_and_after_scheduling(self):
        from atlas_virtual_evaluator import (
            MemoryRegion, RuntimeInputs, Tile, evaluate, parse_program,
        )
        source = function("copy", [start(), inp("io0", "io1", "a", 2),
                                   outp("io1", "io2", "a", 0),
                                   "return %io2 : !atlas.virtual_state"])
        tile = Tile("bf16", tuple((i * 37) & 65535 for i in range(1024)))
        payload = MemoryRegion(INPUT_BASE + 2 * 2048, tile_bytes(tile))
        expected = evaluate(parse_program(source), RuntimeInputs(tiles={2: tile}, memory=(payload,)))
        # The converted output is a DMA store into the output slot it would have filled.
        converted = self.checked(source, PASS, "--schedule-atlas-virtual")
        output = MemoryRegion(OUTPUT_BASE, bytes(2048))
        actual = evaluate(parse_program(converted), RuntimeInputs(memory=(payload, output)))
        self.assertEqual(actual.outputs, {})
        stored = next(region for region in actual.memory if region.address == OUTPUT_BASE)
        self.assertEqual(stored.data, tile_bytes(expected.outputs[0]))

    def test_invalid_boundary_abi_cannot_be_hidden_by_conversion(self):
        source = function("bad", [start(), inp("io0", "io1", "a", 0),
                                  outp("io1", "io2", "a", 0),
                                  "return %io2 : !atlas.virtual_state"])
        for attrs, diagnostic in [
            ("", "requires explicit"),
            (ABI.replace("2415984640", "2415919104"), "overlap or overflow"),
            (ABI.replace("2415919104", "4294966272"), "overlap or overflow"),
            (ABI.replace("2415919104", "2415919105"), "aligned"),
        ]:
            with self.subTest(attrs=attrs):
                result = run("atlas-opt", source.replace(ABI, attrs), PASS)
                self.assertNotEqual(result.returncode, 0)
                self.assertIn(diagnostic, result.stderr)

    def test_captured_program_schedules_explicit_transfers_and_lowers(self):
        source = (ROOT / "examples/handoff/captured_attention_mlp/00-atlas-virtual-ssa.mlir").read_text()
        converted = self.checked(source, PASS, "--schedule-atlas-virtual")
        self.assertNotIn('"atlas.virtual_input_', converted)
        self.assertNotIn('"atlas.virtual_output_', converted)
        self.assertIn('"atlas.virtual_dma_load_bf16"', converted)
        self.assertEqual(converted.count('"atlas.virtual_dma_store_bf16"'), 2)
        self.checked(converted, "--lower-atlas-virtual-to-machine",
                     "--insert-atlas-dma-waits", "--schedule-atlas-stream",
                     "--verify-atlas-timing", "--verify-atlas-generated-schedule")


if __name__ == "__main__":
    unittest.main()
