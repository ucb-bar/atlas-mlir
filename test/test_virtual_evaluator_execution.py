"""Execution semantics for the initial BF16 input/ReLU/output subset."""

from __future__ import annotations

from dataclasses import FrozenInstanceError
from pathlib import Path
import sys
import unittest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
from atlas_virtual_evaluator import MemoryRegion, RuntimeInputs, Scalar, Tile, VirtualInterfaceError, evaluate, parse_program  # noqa: E402


OPERATIONS = '''
  %s0 = "atlas.virtual_start"() : () -> !atlas.virtual_state
  %s1, %original = "atlas.virtual_input_bf16"(%s0) {index = 11 : i32} : (!atlas.virtual_state) -> (!atlas.virtual_state, !atlas.virtual_bf16)
  %rectified = "atlas.virtual_vpu_unary"(%original) {kind = "relu"} : (!atlas.virtual_bf16) -> !atlas.virtual_bf16
  %s2 = "atlas.virtual_output_bf16"(%s1, %original) {index = 23 : i32} : (!atlas.virtual_state, !atlas.virtual_bf16) -> !atlas.virtual_state
  %s3 = "atlas.virtual_output_bf16"(%s2, %rectified) {index = 29 : i32} : (!atlas.virtual_state, !atlas.virtual_bf16) -> !atlas.virtual_state
'''
FLAT = "module {" + OPERATIONS + "}"
FUNCTION = "module { func.func @relu_pair() -> !atlas.virtual_state {" + OPERATIONS + "func.return %s3 : !atlas.virtual_state } }"
SPECIAL_BF16 = (0x8000, 0x0001, 0x007F, 0x8001, 0x7F80, 0xFF80, 0x7F81, 0x7FC1, 0xFFC1)


def patterned_tile(offset: int = 0) -> Tile:
    bits = tuple((0x3E00 + offset + index) | (0x8000 if (index // 32 + index % 32) % 2 else 0) for index in range(1024))
    return Tile("bf16", bits)


class VirtualEvaluatorExecutionTest(unittest.TestCase):
    def test_shared_input_remains_original_after_relu_and_each_call_uses_fresh_inputs(self) -> None:
        program = parse_program(FLAT)
        previous = None
        for offset in (0, 0x400):
            tile = patterned_tile(offset)
            inputs = RuntimeInputs({11: tile})
            result = evaluate(program, inputs)
            self.assertEqual(set(result.outputs), {23, 29})
            self.assertEqual(result.outputs[23], tile)
            expected = tuple(0 if bits & 0x8000 else bits for bits in tile.bits)
            self.assertEqual(result.outputs[29], Tile("bf16", expected))
            self.assertEqual(inputs.tiles[11], tile)
            if previous is not None:
                self.assertNotEqual(result.outputs[23], previous.outputs[23])
                self.assertNotEqual(result.outputs[29], previous.outputs[29])
            previous = result

    def test_single_block_function_and_ssa_renaming_have_same_semantics(self) -> None:
        tile = patterned_tile()
        inputs = RuntimeInputs({11: tile})
        expected = evaluate(parse_program(FLAT), inputs)
        renamed = FUNCTION.replace("%s", "%event_").replace("%original", "%logical_input").replace("%rectified", "%logical_output")
        self.assertEqual(evaluate(parse_program(renamed), inputs), expected)

    def test_positive_zero_and_extreme_finite_normals_are_admitted_exactly(self) -> None:
        # Smallest/largest mantissas at the first and last normal exponents,
        # with both signs, also exercise values outside ordinary kernel ranges.
        encodings = (0x0000, 0x0080, 0x00FF, 0x8080, 0x80FF, 0x7F00, 0x7F7F, 0xFF00, 0xFF7F)
        bits = tuple(encodings[index % len(encodings)] for index in range(1024))
        result = evaluate(parse_program(FLAT), RuntimeInputs({11: Tile("bf16", bits)}))
        self.assertEqual(result.outputs[23].bits, bits)
        self.assertEqual(result.outputs[29].bits, tuple(0 if value & 0x8000 else value for value in bits))

    def test_boundary_only_passthrough_preserves_special_bf16_encodings(self) -> None:
        passthrough = '''module {
          %s0 = "atlas.virtual_start"() : () -> !atlas.virtual_state
          %s1, %tile = "atlas.virtual_input_bf16"(%s0) {index = 11 : i32} : (!atlas.virtual_state) -> (!atlas.virtual_state, !atlas.virtual_bf16)
          %s2 = "atlas.virtual_output_bf16"(%s1, %tile) {index = 23 : i32} : (!atlas.virtual_state, !atlas.virtual_bf16) -> !atlas.virtual_state
        }'''
        bits = tuple(SPECIAL_BF16[index % len(SPECIAL_BF16)] for index in range(1024))
        inputs = RuntimeInputs({11: Tile("bf16", bits)})
        result = evaluate(parse_program(passthrough), inputs)
        self.assertEqual(result.outputs[23].bits, bits)
        self.assertEqual(inputs.tiles[11].bits, bits)

    def test_relu_preserves_positive_encodings_and_clears_negative_encodings(self) -> None:
        program = parse_program(FLAT)
        for encoding in SPECIAL_BF16:
            with self.subTest(encoding=hex(encoding)):
                bits = [0x3F80] * 1024
                bits[731] = encoding
                result = evaluate(program, RuntimeInputs({11: Tile("bf16", bits)}))
                self.assertEqual(result.outputs[23].bits, tuple(bits))
                self.assertEqual(result.outputs[29].bits[731], 0 if encoding & 0x8000 else encoding)

    def test_execution_preserves_memory_inputs_and_publishes_immutable_outputs(self) -> None:
        tile = patterned_tile()
        memory = (MemoryRegion(0x90000020, b"right guard"), MemoryRegion(0x90000000, b"left guard"))
        inputs = RuntimeInputs({11: tile}, memory=memory)
        result = evaluate(parse_program(FLAT), inputs)
        self.assertEqual(result.memory, memory)
        self.assertEqual(inputs.memory, memory)
        self.assertEqual(dict(inputs.tiles), {11: tile})
        with self.assertRaises(TypeError):
            result.outputs[23] = Tile("bf16", (0,) * 1024)
        with self.assertRaises(FrozenInstanceError):
            result.outputs[29].bits = (0,) * 1024
        with self.assertRaises(FrozenInstanceError):
            result.memory = ()

    def test_stale_state_cannot_be_consumed_by_input_output_or_return(self) -> None:
        second_input = '  %s4, %extra = "atlas.virtual_input_bf16"(%s0) {index = 11 : i32} : (!atlas.virtual_state) -> (!atlas.virtual_state, !atlas.virtual_bf16)\n'
        sources = (
            FLAT.replace('  %rectified =', second_input + '  %rectified ='),
            FLAT.replace('(%s2, %rectified)', '(%s1, %rectified)'),
            FUNCTION.replace('func.return %s3', 'func.return %s1'),
        )
        for index, source in enumerate(sources):
            with self.subTest(consumer=index):
                with self.assertRaises(VirtualInterfaceError):
                    evaluate(parse_program(source), RuntimeInputs({11: patterned_tile()}))

    def test_execution_requires_exactly_one_virtual_start(self) -> None:
        duplicate = '  %extra_start = "atlas.virtual_start"() : () -> !atlas.virtual_state\n'
        cases = (
            ("module {}", RuntimeInputs()),
            (FLAT.replace('  %s1, %original =', duplicate + '  %s1, %original ='), RuntimeInputs({11: patterned_tile()})),
        )
        for index, (source, inputs) in enumerate(cases):
            with self.subTest(case=index):
                with self.assertRaisesRegex(VirtualInterfaceError, "virtual_start"):
                    evaluate(parse_program(source), inputs)

    def test_operand_from_another_program_has_no_runtime_binding(self) -> None:
        program, other = parse_program(FLAT), parse_program(FLAT)
        # A same-typed, similarly named value is still a distinct SSA identity.
        program.operations[2].operands = (other.operations[1].results[1],)
        with self.assertRaises(VirtualInterfaceError):
            evaluate(program, RuntimeInputs({11: patterned_tile()}))

    def test_evaluate_enforces_exact_input_indices_formats_and_controls(self) -> None:
        program, tile = parse_program(FLAT), patterned_tile()
        cases = (
            RuntimeInputs(),
            RuntimeInputs({11: tile, 12: tile}),
            RuntimeInputs({0: tile}),
            RuntimeInputs({11: Tile("fp8", (0,) * 1024)}),
            RuntimeInputs({11: tile}, (Scalar(1, 0),)),
            {11: tile},
        )
        for index, inputs in enumerate(cases):
            with self.subTest(case=index):
                with self.assertRaises(VirtualInterfaceError):
                    evaluate(program, inputs)


if __name__ == "__main__":
    unittest.main()
