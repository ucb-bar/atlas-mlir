"""Guarded virtual softmax/attention checks; scheduling uses ATLAS_OOT_BIN_DIR."""

from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
from atlas_virtual_evaluator import (  # noqa: E402
    EvaluationResult, MemoryRegion, RuntimeInputs, Tile, compare_results, evaluate, parse_program,
)
from test_virtual_evaluator_resources import bf16_bytes, diagonal  # noqa: E402

EXAMPLES = ROOT / "test/examples"
BIN = Path(os.environ.get("ATLAS_OOT_BIN_DIR", ROOT / "build/bin"))
BASE = 0x90000000


def guarded_memory(inputs: bytes, output_bits: tuple[int, ...], output_offset: int):
    prefix = b"L" * 32 + inputs + b"G" * (output_offset - len(inputs))
    initial = MemoryRegion(BASE - 32, prefix + b"?" * 2048 + b"R" * 32)
    expected = MemoryRegion(initial.address, prefix + bf16_bytes(output_bits) + b"R" * 32)
    return initial, expected


def softmax_cases():
    # Scaled lows round to -1; row sums 20/24/28/32 use source-derived RcpLUT bits.
    counts, reciprocals = (8, 16, 24, 32), (0x3D4C, 0x3D2A, 0x3D12, 0x3D00)
    high = tuple((col + 3 * row) % 32 < counts[row % 4] for row in range(32) for col in range(32))
    patterned = tuple(0 if selected else 0xC07A for selected in high)
    probabilities = tuple(reciprocals[i // 32 % 4] - (0 if selected else 0x80) for i, selected in enumerate(high))
    for name, scores, output in (("uniform", (0,) * 1024, (0x3D00,) * 1024),
                                 ("asymmetric", patterned, probabilities)):
        initial, expected = guarded_memory(bf16_bytes(scores), output, 0x1000)
        runtime = RuntimeInputs({0: Tile("bf16", scores)}, memory=(initial,))
        yield name, runtime, EvaluationResult({0: Tile("bf16", output)}, (expected,))


def attention_cases():
    # Stored V[N,K] has N twos: output column N is 1+N/32.
    varying_v = bytes(0x40 if (k + 3 * n) % 32 < n else 0x38 for n in range(32) for k in range(32))
    varying_output = tuple(0x3F80 + 4 * col for _ in range(32) for col in range(32))
    for name, qk, v, output in (("uniform", bytes(1024), b"\x38" * 1024, (0x3F80,) * 1024),
                              ("asymmetric", b"\x38" * 1024, varying_v, varying_output)):
        initial, expected = guarded_memory(qk + qk + v, output, 0xC00)
        yield name, RuntimeInputs(memory=(initial,)), EvaluationResult({}, (expected,))


def key_orientation_inputs():
    identity = bytes(diagonal("fp8", 0x38).bits)
    key = bytearray(1024)
    for row, col, bits in ((2, 19, 0x38), (11, 3, 0x40), (25, 7, 0x48)):
        key[row * 32 + col] = bits
    rows = tuple(key[start:start + 32] for start in range(0, 1024, 32))
    stored_key = bytes(bits for column in zip(*rows) for bits in column)
    inputs = []
    for layout in (stored_key, bytes(key)):
        initial, _ = guarded_memory(identity + layout + identity, (0,) * 1024, 0xC00)
        inputs.append(RuntimeInputs(memory=(initial,)))
    return tuple(inputs)


class VirtualEvaluatorAttentionTest(unittest.TestCase):
    def test_softmax_matches_independent_bits_and_preserves_memory(self) -> None:
        program = parse_program((EXAMPLES / "virtual_softmax_tile.mlir").read_text())
        cases = tuple(softmax_cases())
        for name, inputs, expected in (*cases, cases[0]):
            with self.subTest(case=name):
                compare_results(expected, evaluate(program, inputs))

    def test_attention_matches_independent_bits_and_preserves_memory(self) -> None:
        for fixture in ("virtual_attention_tile.mlir", "virtual_attention_tile_transposed_key.mlir"):
            program = parse_program((EXAMPLES / fixture).read_text())
            cases = tuple(attention_cases())
            for name, inputs, expected in (*cases, cases[0]):
                with self.subTest(fixture=fixture, case=name):
                    compare_results(expected, evaluate(program, inputs))

    def attention_output(self, inputs: RuntimeInputs, result: EvaluationResult) -> bytes:
        self.assertEqual(dict(result.outputs), {})
        self.assertEqual(len(result.memory), 1)
        before, after = inputs.memory[0], result.memory[0]
        self.assertEqual(after.address, before.address)
        self.assertEqual(len(after.data), len(before.data))
        start = BASE + 0xC00 - before.address
        end = start + 2048
        self.assertEqual(after.data[:start], before.data[:start])
        self.assertEqual(after.data[end:], before.data[end:])
        return after.data[start:end]

    def test_transposed_key_matches_prepared_key_and_changes_orientation(self) -> None:
        ordinary = parse_program((EXAMPLES / "virtual_attention_tile.mlir").read_text())
        transposed = parse_program((EXAMPLES / "virtual_attention_tile_transposed_key.mlir").read_text())
        prepared_inputs, logical_inputs = key_orientation_inputs()
        expected = self.attention_output(prepared_inputs, evaluate(ordinary, prepared_inputs))
        actual = self.attention_output(logical_inputs, evaluate(transposed, logical_inputs))
        wrong = self.attention_output(logical_inputs, evaluate(ordinary, logical_inputs))
        self.assertEqual(actual, expected)
        self.assertNotEqual(actual, wrong)

    def test_mlp_bias_baseline_still_produces_three(self) -> None:
        program = parse_program((EXAMPLES / "virtual_fp8_two_layer_mlp_bias.mlir").read_text())
        inputs = RuntimeInputs({0: Tile("fp8", (0x38,) * 1024), 1: diagonal("fp8", 0x38),
                                2: diagonal("fp8", 0x38), 3: Tile("bf16", (0x3F80,) * 1024),
                                4: Tile("bf16", (0x3F80,) * 1024)})
        self.assertEqual(evaluate(program, inputs).outputs[0], Tile("bf16", (0x4040,) * 1024))

    def test_scheduled_examples_match_independent_bits_and_source(self) -> None:
        tool = BIN / "atlas-opt"
        if not tool.is_file():
            self.skipTest("set ATLAS_OOT_BIN_DIR to the matching atlas-opt build")
        for fixture, cases in (("virtual_softmax_tile.mlir", softmax_cases),
                               ("virtual_attention_tile.mlir", attention_cases),
                               ("virtual_attention_tile_transposed_key.mlir", attention_cases)):
            source = (EXAMPLES / fixture).read_text()
            original = parse_program(source)
            for seed in (None, 0, 5):
                option = "--schedule-atlas-virtual" + (f"=random-seed={seed}" if seed is not None else "")
                run = subprocess.run([str(tool), "-", option, "--verify-atlas-virtual-stream"],
                                     input=source, text=True, capture_output=True, timeout=60)
                self.assertEqual(run.returncode, 0, run.stderr)
                scheduled = parse_program(run.stdout)
                for name, inputs, expected in cases():
                    with self.subTest(fixture=fixture, seed=seed, case=name):
                        reference, actual = evaluate(original, inputs), evaluate(scheduled, inputs)
                        compare_results(expected, actual)
                        compare_results(reference, actual)
                if fixture == "virtual_attention_tile_transposed_key.mlir":
                    _, inputs = key_orientation_inputs()
                    with self.subTest(fixture=fixture, seed=seed, case="key_orientation"):
                        reference, actual = evaluate(original, inputs), evaluate(scheduled, inputs)
                        self.attention_output(inputs, actual)
                        compare_results(reference, actual)


if __name__ == "__main__":
    unittest.main()
