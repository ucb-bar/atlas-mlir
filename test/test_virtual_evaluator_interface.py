"""Virtual parser, runtime interface, BF16 execution and scalar/CFG semantics, without Atlas tools."""

from __future__ import annotations

from dataclasses import FrozenInstanceError
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
from atlas_virtual_evaluator import (  # noqa: E402
    EvaluationResult, MemoryRegion, RuntimeInputs, Scalar, Tile, UnsupportedVirtualMode, VirtualInterfaceError, compare_results, evaluate,
    operation_name, parse_program,
)
from test_virtual_evaluator_resources import load, ready, wrap  # noqa: E402


def example(name: str) -> str:
    return (ROOT / "test/examples" / name).read_text()


FLAT = example("virtual_bf16_ssa.mlir")
CFG = example("virtual_bf16_cfg.mlir")


class VirtualParserInterfaceTest(unittest.TestCase):
    def test_shared_values_output_order_and_renaming_preserve_ssa_identity(self) -> None:
        program = parse_program(FLAT)
        self.assertIsNone(program.function)
        self.assertEqual(dict(program.input_formats), {0: "bf16"})
        self.assertEqual(program.output_indices, (0, 1))
        start, boundary, unary, binary, output0, output1 = program.operations
        self.assertIs(boundary.operands[0], start.results[0])
        self.assertIs(output0.operands[1], unary.results[0])
        self.assertIs(output1.operands[1], binary.results[0])
        renamed = parse_program(FLAT.replace("%t", "%logical_tensor_").replace("%io", "%event_"))
        self.assertEqual((renamed.input_formats, renamed.output_indices), (program.input_formats, program.output_indices))
        self.assertEqual(tuple(map(operation_name, renamed.operations)), tuple(map(operation_name, program.operations)))
        for parsed in (program, renamed):
            boundary, unary, binary = parsed.operations[1:4]
            self.assertIs(unary.operands[0], boundary.results[1])
            self.assertIs(binary.operands[0], boundary.results[1])
            self.assertIs(binary.operands[1], unary.results[0])
        self.assertIsNot(renamed.operations[1].results[1], program.operations[1].results[1])

    def test_cfg_selection_retains_branch_and_block_argument_identity(self) -> None:
        program = parse_program(CFG, function="choose_tile")
        self.assertEqual(program.function.sym_name.data, "choose_tile")
        self.assertEqual(program.control_widths, (1,))
        entry, left, right, join = program.blocks
        branch = tuple(entry.ops)[-1]
        self.assertIs(branch.operands[0], entry.args[0])
        self.assertEqual(tuple(branch.successors), (left, right))
        self.assertIs(tuple(left.ops)[0].operands[0], left.args[1])
        self.assertIs(tuple(join.ops)[0].operands[1], join.args[1])
        loop = parse_program(CFG, function="two_relu_steps")
        self.assertEqual(loop.control_widths, ())
        self.assertIn("arith.addi", tuple(map(operation_name, loop.operations)))

    def test_existing_virtual_families_parse_without_machine_tools(self) -> None:
        for fixture in ("bf16_loop_program", "bf16_swap_loop_program", "bf16_dynamic_branch_program", "fp8_matmul_program", "fp8_two_layer_mlp_bias",
                        "mxu_accumulation", "mxu_seeded_fp8", "dma_tiles", "dma_mxu", "dma_mxu_bf16"):
            with self.subTest(fixture=fixture):
                program = parse_program(example(f"virtual_{fixture}.mlir"))
                self.assertIsNotNone(program.function)
                self.assertTrue(program.operations)

    def test_two_pending_dma_loads_retain_reverse_completion_identities(self) -> None:
        program = parse_program(wrap(
            "%a = arith.constant -1879048192 : i32", "%b = arith.constant -1879047168 : i32", "%size = arith.constant 1024 : i32",
            load("fp8", "s0", "s1", "first", "a"), load("fp8", "s1", "s2", "second", "b"),
            ready("fp8", "s2", "s3", "second", "tile_second"), ready("fp8", "s3", "s4", "first", "tile_first"),
            '%product = "atlas.virtual_mxu_matmul"(%tile_first, %tile_second) {unit = 0 : i32} : (!atlas.virtual_fp8, !atlas.virtual_fp8) -> !atlas.virtual_bf16',
            '%s5 = "atlas.virtual_output_bf16"(%s4, %product) {index = 0 : i32} : (!atlas.virtual_state, !atlas.virtual_bf16) -> !atlas.virtual_state',
            final="s5", flat=True))
        first, second, await_second, await_first, contraction, output = program.operations[4:]
        self.assertIsNot(first.results[1], second.results[1])
        self.assertIs(await_second.operands[1], second.results[1])
        self.assertIs(await_first.operands[1], first.results[1])
        self.assertIsNot(await_first.results[1], await_second.results[1])
        self.assertEqual((contraction.operands[0], contraction.operands[1]), (await_first.results[1], await_second.results[1]))
        self.assertIs(output.operands[1], contraction.results[0])
        self.assertEqual(program.output_indices, (0,))

    def test_same_unit_independent_mxu_chains_retain_handle_identities(self) -> None:
        source = example("virtual_mxu_accumulation.mlir").replace("unit = 1 : i32", "unit = 0 : i32")
        program = parse_program(source.replace("virtual_mxu_weight<1>", "virtual_mxu_weight<0>").replace("virtual_mxu_acc<1>", "virtual_mxu_acc<0>"))
        weight0, weight1, reset0, reset1, next0, next1, read0, read1 = program.operations[3:11]
        self.assertIsNot(weight0.results[1], weight1.results[1])
        self.assertIsNot(reset0.results[1], reset1.results[1])
        for weight, reset, accumulation, readout in ((weight0, reset0, next0, read0), (weight1, reset1, next1, read1)):
            self.assertIs(reset.operands[2], weight.results[1])
            self.assertIs(accumulation.operands[2], weight.results[1])
            self.assertIs(accumulation.operands[3], reset.results[1])
            self.assertIs(readout.operands[1], accumulation.results[1])
        self.assertEqual(program.output_indices, (0, 1))

    def test_declarations_on_both_cfg_paths_require_runtime_inputs(self) -> None:
        source = example("virtual_bf16_dynamic_branch_program.mlir").replace(
            '    %right_result = "atlas.virtual_vpu_unary"(%right_tile)',
            '    %right_next, %extra = "atlas.virtual_input_bf16"(%right_io) {index = 1 : i32} : (!atlas.virtual_state) -> (!atlas.virtual_state, !atlas.virtual_bf16)\n'
            '    %right_result = "atlas.virtual_vpu_unary"(%extra)',
        ).replace("cf.br ^join(%right_io, %right_result", "cf.br ^join(%right_next, %right_result")
        program = parse_program(source)
        self.assertEqual(dict(program.input_formats), {0: "bf16", 1: "bf16"})
        with self.assertRaisesRegex(VirtualInterfaceError, "input indices"):
            program.validate_inputs(RuntimeInputs({0: Tile("bf16", (0,) * 1024)}, (Scalar(1, 0),)))

    def test_valid_generic_and_canonical_property_forms_are_admitted(self) -> None:
        source = 'module { %a = arith.constant 0 : i32 %c = "arith.cmpi"(%a, %a) <{predicate = 2 : i64}> : (i32, i32) -> i1 }'
        constant, comparison = parse_program(source).operations
        self.assertEqual(operation_name(comparison), "arith.cmpi")
        self.assertEqual(tuple(comparison.operands), (constant.results[0],) * 2)
        relu = example("virtual_bf16_shared_relu.mlir")
        canonical = relu.replace('{index = 0 : i32}', '<{index = 0 : i32}>').replace('{index = 1 : i32}', '<{index = 1 : i32}>').replace('{kind = "relu"}', '<{kind = "relu"}>')
        inputs = RuntimeInputs({0: Tile("bf16", (0xBF80, 0x3F80) * 512)})
        compare_results(evaluate(parse_program(relu), inputs), evaluate(parse_program(canonical), inputs))

    def test_parser_rejects_unsupported_and_malformed_programs(self) -> None:
        canonical = example("virtual_bf16_shared_relu.mlir").replace('{index = 0 : i32}', '<{index = 0 : i32}>')
        seeded, cmpi = example("virtual_mxu_seeded_fp8.mlir"), 'module { %a = arith.constant 0 : i32 %c = "arith.cmpi"(%a, %a) <{predicate = 2 : i64}> : (i32, i32) -> i1 }'
        unsupported = (
            (FLAT.replace("atlas.virtual_start", "atlas.start"), "unsupported virtual operation"),
            (FLAT.replace("atlas.virtual_vpu_unary", "atlas.virtual_invented"), "unsupported virtual operation"),
            (FLAT.replace('kind = "relu"', 'kind = "sqrt"'), "supported kinds"),
            (FLAT.replace('kind = "add"', 'kind = "mul"'), "supported kinds"),
            (example("virtual_fp8_two_layer_mlp.mlir").replace("scale_code = 127", "scale_code = 128"), "scale_code 127"),
        )
        malformed = (
            (FLAT.replace("index = 0 : i32", "index = 0 : i64", 1), "index : i32"),
            (FLAT.replace("index = 0 : i32", "index = -1 : i32", 1), "nonnegative"),
            (FLAT.replace("index = 0 : i32", "index = 0 : i32, invented = 1 : i32", 1), "expected attributes"),
            (FLAT.replace('{kind = "relu"}', "{}"), "expected attributes"),
            (FLAT.replace("atlas.virtual_vpu_unary", "atlas.virtual_vpu_binary"), "signature"),
            (FLAT.replace("index = 1 : i32", "index = 0 : i32"), "duplicate output"),
            ("module { %a = arith.constant 0 : i32 %b = arith.addi %a, %a overflow<nsw> : i32 }", "wrapping i32"),
            ("module { %a = arith.constant 0 : i64 %b = arith.addi %a, %a : i64 }", "i1/i32 integer"),
            (cmpi.replace("predicate = 2", "predicate = 10"), "invalid virtual MLIR|ten predicates"),
            (cmpi.replace("predicate = 2 : i64", "predicate = 2 : i32"), "predicates encoded as i64"),
            (seeded.replace("unit = 0 : i32", "unit = 2 : i32", 1), "unit must be"),
            (seeded.replace("unit = 0 : i32", "unit = 1 : i32", 1), "handle units must agree"),
            (seeded.replace("code = 129 : i32", "code = 256 : i32"), "scale code"),
            (seeded.replace("virtual_mxu_weight<0>", "virtual_mxu_weight<2>"), "unsupported virtual type"),
            ('module { %a = "arith.constant"() <{value = 0 : i32}> {invented = 1 : i32} : () -> i32 }', "unsupported attributes/properties"),
            ('module { %a = arith.constant 0 : i32 %b = "arith.addi"(%a, %a) {invented = 1 : i32} : (i32, i32) -> i32 }', "unsupported attributes/properties"),
            # xDSL rejects unknown properties before interface admission.
            ('module { %a = "arith.constant"() <{value = 0 : i32, invented = 1 : i32}> : () -> i32 }', "invalid virtual MLIR: property 'invented' is not defined"),
            ('module { %a = arith.constant 0 : i32 %b = "arith.addi"(%a, %a) <{invented = 1 : i32}> : (i32, i32) -> i32 }', "invalid virtual MLIR: property 'invented' is not defined"),
            (canonical.replace('<{index = 0 : i32}>', '<{index = 0 : i32}> {index = 1 : i32}', 1), "duplicate attribute/property"),
            (canonical.replace('<{index = 0 : i32}>', '<{index = 0 : i32, invented = 1 : i32}>', 1), "expected attributes"),
            ("module { %x =", "invalid virtual MLIR"),
            (FLAT.replace("(%t0)", "(%undefined)", 1), "invalid virtual MLIR"),
        )
        selections = ((CFG, None, "select a function"), (CFG, "absent", "not found"), (FLAT, "anything", "not found in flat"))
        cases = ([(source, None, UnsupportedVirtualMode, message) for source, message in unsupported]
                 + [(source, None, VirtualInterfaceError, message) for source, message in malformed]
                 + [(source, function, VirtualInterfaceError, message) for source, function, message in selections])
        for index, (source, function, error, message) in enumerate(cases):
            with self.subTest(case=index, message=message), self.assertRaisesRegex(error, message):
                parse_program(source, function=function)


class VirtualRuntimeInterfaceTest(unittest.TestCase):
    def test_tiles_copy_bits_and_preserve_special_encodings(self) -> None:
        for format, special in (("bf16", (0x8000, 0x0001, 0x7F80, 0x7FC1)), ("fp8", (0x80, 0x01, 0x7F, 0xFF))):
            with self.subTest(format=format):
                bits = list(special) + [0] * 1020
                tile = Tile(format, bits)
                bits[0] = 0
                self.assertEqual((tile.bits[:4], tile.shape), (special, (32, 32)))
                self.assertIsInstance(tile.bits, tuple)
                with self.assertRaises(FrozenInstanceError):
                    tile.format = "fp8"

    def test_values_reject_inexact_encodings_and_extents(self) -> None:
        self.assertEqual((Scalar(1, 1).bits, Scalar(32, 0xFFFFFFFF).bits), (1, 0xFFFFFFFF))
        data = bytearray(b"input and guard")
        region = MemoryRegion(0xFFFFFFFF - len(data) + 1, memoryview(data))
        data[:] = b"X" * len(data)
        self.assertEqual(region.data, b"input and guard")
        cases = [(Tile, format, bits) for format, bits in (("f32", [0] * 1024), ("bf16", [0] * 1023), ("fp8", [256] * 1024),
                                                           ("bf16", [-1] * 1024), ("bf16", [True] * 1024), ("fp8", [1.0] * 1024))]
        cases += [(Scalar, width, bits) for width, bits in ((True, 0), (8, 0), (1, 2), (32, -1), (32, 1 << 32), (1, True), (32, 0.0))]
        cases += [(MemoryRegion, address, payload) for address, payload in ((-1, b"x"), (1 << 32, b"x"), (True, b"x"), (0xFFFFFFFF, b"xx"), (0, b""), (0, [1]))]
        for constructor, first, second in cases:
            with self.subTest(constructor=constructor.__name__, first=first, second=str(second)[:12]), self.assertRaises(VirtualInterfaceError):
                constructor(first, second)

    def test_snapshots_allow_adjacent_regions_and_reject_overlap(self) -> None:
        first, adjacent = MemoryRegion(32, b"ab"), MemoryRegion(34, b"cd")
        self.assertEqual(RuntimeInputs(memory=[adjacent, first]).memory, (adjacent, first))
        overlap = MemoryRegion(33, b"xy")
        for constructor in (lambda: RuntimeInputs(memory=(first, overlap)), lambda: EvaluationResult({}, memory=(overlap, first))):
            with self.assertRaisesRegex(VirtualInterfaceError, "overlap"):
                constructor()

    def test_runtime_collections_own_immutable_copies(self) -> None:
        tile = Tile("bf16", (0,) * 1024)
        tiles, controls, memory = {0: tile}, [Scalar(1, 0)], [MemoryRegion(0x90000000, b"guard")]
        inputs = RuntimeInputs(tiles, controls, memory)
        tiles.clear()
        controls.clear()
        memory.clear()
        self.assertEqual((dict(inputs.tiles), inputs.controls, len(inputs.memory)), ({0: tile}, (Scalar(1, 0),), 1))
        with self.assertRaises(TypeError):
            inputs.tiles[1] = tile
        outputs = {7: tile}
        result = EvaluationResult(outputs, inputs.memory)
        outputs.clear()
        self.assertEqual(dict(result.outputs), {7: tile})
        with self.assertRaises(TypeError):
            result.outputs[0] = tile

    def test_collections_require_typed_tiles_controls_and_memory(self) -> None:
        tile = Tile("bf16", (0,) * 1024)
        constructors = (
            lambda: RuntimeInputs({True: tile}), lambda: RuntimeInputs({-1: tile}), lambda: RuntimeInputs({1 << 31: tile}),
            lambda: RuntimeInputs({0: tile.bits}), lambda: RuntimeInputs(controls=(1,)), lambda: RuntimeInputs(memory=(b"bytes",)),
            lambda: EvaluationResult({0: Tile("fp8", (0,) * 1024)}), lambda: EvaluationResult({0: object()}), lambda: EvaluationResult({}, memory=(b"bytes",)),
        )
        for index, constructor in enumerate(constructors):
            with self.subTest(case=index), self.assertRaises(VirtualInterfaceError):
                constructor()

    def test_program_validates_exact_boundary_indices_formats_and_controls(self) -> None:
        program = parse_program(CFG, function="choose_tile")
        bf16, fp8 = Tile("bf16", (0,) * 1024), Tile("fp8", (0,) * 1024)
        for choice in (0, 1):
            program.validate_inputs(RuntimeInputs({0: bf16}, (Scalar(1, choice),)))
        invalid = (
            RuntimeInputs({}, (Scalar(1, 0),)), RuntimeInputs({0: bf16, 1: bf16}, (Scalar(1, 0),)), RuntimeInputs({0: fp8}, (Scalar(1, 0),)),
            RuntimeInputs({0: bf16}), RuntimeInputs({0: bf16}, (Scalar(32, 0),)), RuntimeInputs({0: bf16}, (Scalar(1, 0), Scalar(1, 1))), {0: bf16},
        )
        for index, inputs in enumerate(invalid):
            with self.subTest(case=index), self.assertRaises(VirtualInterfaceError):
                program.validate_inputs(inputs)
        with self.assertRaises(VirtualInterfaceError):
            evaluate(parse_program(RELU_FLAT), {11: bf16})


RELU_OPERATIONS = '''
  %s0 = "atlas.virtual_start"() : () -> !atlas.virtual_state
  %s1, %original = "atlas.virtual_input_bf16"(%s0) {index = 11 : i32} : (!atlas.virtual_state) -> (!atlas.virtual_state, !atlas.virtual_bf16)
  %rectified = "atlas.virtual_vpu_unary"(%original) {kind = "relu"} : (!atlas.virtual_bf16) -> !atlas.virtual_bf16
  %s2 = "atlas.virtual_output_bf16"(%s1, %original) {index = 23 : i32} : (!atlas.virtual_state, !atlas.virtual_bf16) -> !atlas.virtual_state
  %s3 = "atlas.virtual_output_bf16"(%s2, %rectified) {index = 29 : i32} : (!atlas.virtual_state, !atlas.virtual_bf16) -> !atlas.virtual_state
'''
RELU_FLAT = "module {" + RELU_OPERATIONS + "}"
RELU_FUNCTION = "module { func.func @relu_pair() -> !atlas.virtual_state {" + RELU_OPERATIONS + "func.return %s3 : !atlas.virtual_state } }"
SPECIAL_BF16 = (0x8000, 0x0001, 0x007F, 0x8001, 0x7F80, 0xFF80, 0x7F81, 0x7FC1, 0xFFC1)


def patterned_tile(offset: int = 0) -> Tile:
    return Tile("bf16", tuple((0x3E00 + offset + index) | (0x8000 if (index // 32 + index % 32) % 2 else 0) for index in range(1024)))


def relu(bits: tuple[int, ...]) -> tuple[int, ...]:
    return tuple(0 if value & 0x8000 else value for value in bits)


class VirtualEvaluatorExecutionTest(unittest.TestCase):
    def test_shared_input_remains_original_after_relu_and_each_call_uses_fresh_inputs(self) -> None:
        program = parse_program(RELU_FLAT)
        previous = None
        for offset in (0, 0x400):
            tile = patterned_tile(offset)
            inputs = RuntimeInputs({11: tile})
            result = evaluate(program, inputs)
            self.assertEqual(dict(result.outputs), {23: tile, 29: Tile("bf16", relu(tile.bits))})
            self.assertEqual(inputs.tiles[11], tile)
            if previous is not None:
                self.assertNotEqual(result.outputs[23], previous.outputs[23])
                self.assertNotEqual(result.outputs[29], previous.outputs[29])
            previous = result

    def test_single_block_function_and_ssa_renaming_have_same_semantics(self) -> None:
        inputs = RuntimeInputs({11: patterned_tile()})
        renamed = RELU_FUNCTION.replace("%s", "%event_").replace("%original", "%logical_input").replace("%rectified", "%logical_output")
        self.assertEqual(evaluate(parse_program(renamed), inputs), evaluate(parse_program(RELU_FLAT), inputs))

    def test_special_and_extreme_encodings_pass_through_and_relu_clears_only_negative_ones(self) -> None:
        # Signed zeros, subnormals, infinities, NaN payloads, and the smallest/largest mantissas of the extreme normal exponents.
        encodings = SPECIAL_BF16 + (0x0000, 0x0080, 0x00FF, 0x8080, 0x80FF, 0x7F00, 0x7F7F, 0xFF00, 0xFF7F)
        bits = tuple(encodings[index % len(encodings)] for index in range(1024))
        inputs = RuntimeInputs({11: Tile("bf16", bits)})
        passthrough = "module {" + "\n".join(line for line in RELU_OPERATIONS.splitlines() if "rectified" not in line) + "}"
        self.assertEqual(dict(evaluate(parse_program(passthrough), inputs).outputs), {23: inputs.tiles[11]})
        result = evaluate(parse_program(RELU_FLAT), inputs)
        self.assertEqual((result.outputs[23].bits, result.outputs[29].bits, inputs.tiles[11].bits), (bits, relu(bits), bits))

    def test_execution_preserves_memory_inputs_and_publishes_immutable_outputs(self) -> None:
        tile = patterned_tile()
        memory = (MemoryRegion(0x90000020, b"right guard"), MemoryRegion(0x90000000, b"left guard"))
        inputs = RuntimeInputs({11: tile}, memory=memory)
        result = evaluate(parse_program(RELU_FLAT), inputs)
        self.assertEqual((result.memory, inputs.memory, dict(inputs.tiles)), (memory, memory, {11: tile}))
        with self.assertRaises(TypeError):
            result.outputs[23] = Tile("bf16", (0,) * 1024)
        with self.assertRaises(FrozenInstanceError):
            result.outputs[29].bits = (0,) * 1024
        with self.assertRaises(FrozenInstanceError):
            result.memory = ()

    def test_stale_state_duplicate_start_and_foreign_operands_are_rejected(self) -> None:
        second_input = '  %s4, %extra = "atlas.virtual_input_bf16"(%s0) {index = 11 : i32} : (!atlas.virtual_state) -> (!atlas.virtual_state, !atlas.virtual_bf16)\n'
        duplicate = '  %extra_start = "atlas.virtual_start"() : () -> !atlas.virtual_state\n'
        tile = RuntimeInputs({11: patterned_tile()})
        cases = (
            (RELU_FLAT.replace('  %rectified =', second_input + '  %rectified ='), tile, ""),
            (RELU_FLAT.replace('(%s2, %rectified)', '(%s1, %rectified)'), tile, ""),
            (RELU_FUNCTION.replace('func.return %s3', 'func.return %s1'), tile, ""),
            ("module {}", RuntimeInputs(), "virtual_start"),
            (RELU_FLAT.replace('  %s1, %original =', duplicate + '  %s1, %original ='), tile, "virtual_start"),
        )
        for index, (source, inputs, message) in enumerate(cases):
            with self.subTest(case=index), self.assertRaisesRegex(VirtualInterfaceError, message):
                evaluate(parse_program(source), inputs)
        program, other = parse_program(RELU_FLAT), parse_program(RELU_FLAT)
        # A same-typed, similarly named value is still a distinct SSA identity.
        program.operations[2].operands = (other.operations[1].results[1],)
        with self.assertRaises(VirtualInterfaceError):
            evaluate(program, tile)


INPUTS = '''
  %s0 = "atlas.virtual_start"() : () -> !atlas.virtual_state
  %s1, %a = "atlas.virtual_input_bf16"(%s0) {index = 0 : i32} : (!atlas.virtual_state) -> (!atlas.virtual_state, !atlas.virtual_bf16)
  %s2, %b = "atlas.virtual_input_bf16"(%s1) {index = 1 : i32} : (!atlas.virtual_state) -> (!atlas.virtual_state, !atlas.virtual_bf16)
'''
SELECT = '''
  cf.cond_br %choose, ^exit(%s2, %a : !atlas.virtual_state, !atlas.virtual_bf16), ^exit(%s2, %b : !atlas.virtual_state, !atlas.virtual_bf16)
^exit(%state: !atlas.virtual_state, %tile: !atlas.virtual_bf16):
  %done = "atlas.virtual_output_bf16"(%state, %tile) {index = 7 : i32} : (!atlas.virtual_state, !atlas.virtual_bf16) -> !atlas.virtual_state
  func.return %done : !atlas.virtual_state
'''
EXIT = '''
^exit(%es: !atlas.virtual_state, %et: !atlas.virtual_bf16):
  %done = "atlas.virtual_output_bf16"(%es, %et) {index = 7 : i32} : (!atlas.virtual_state, !atlas.virtual_bf16) -> !atlas.virtual_state
  func.return %done : !atlas.virtual_state
'''
TILES = {0: Tile("bf16", (0xBF80,) * 1024), 1: Tile("bf16", (0x4000,) * 1024)}


def function(body: str, arguments: str = "") -> str:
    return "module { func.func @test(" + arguments + ") -> !atlas.virtual_state {" + body + "} }"


class VirtualEvaluatorCFGTest(unittest.TestCase):
    def assert_selection(self, source: str, controls: tuple[Scalar, ...], selected: int) -> None:
        result = evaluate(parse_program(source), RuntimeInputs(TILES, controls))
        self.assertEqual(dict(result.outputs), {7: TILES[selected]})

    def test_all_comparisons_distinguish_signed_and_unsigned_i32_boundaries(self) -> None:
        # Explicit answers catch sign extension or predicate swaps.
        predicates = ("eq", "ne", "slt", "sle", "sgt", "sge", "ult", "ule", "ugt", "uge")
        cases = (
            (0xFFFFFFFF, 0, (0, 1, 1, 1, 0, 0, 0, 0, 1, 1)),
            (0x80000000, 0x7FFFFFFF, (0, 1, 1, 1, 0, 0, 0, 0, 1, 1)),
            (0x7FFFFFFF, 0x80000000, (0, 1, 0, 0, 1, 1, 1, 1, 0, 0)),
            (0x80000000, 0x80000000, (1, 0, 0, 1, 0, 1, 0, 1, 0, 1)),
            (0, 1, (0, 1, 1, 1, 0, 0, 1, 1, 0, 0)),
        )
        for position, predicate in enumerate(predicates):
            source = function(INPUTS + f"%choose = arith.cmpi {predicate}, %lhs, %rhs : i32\n" + SELECT, "%lhs: i32, %rhs: i32")
            for lhs, rhs, expected in cases:
                with self.subTest(predicate=predicate, lhs=hex(lhs), rhs=hex(rhs)):
                    self.assert_selection(source, (Scalar(32, lhs), Scalar(32, rhs)), 0 if expected[position] else 1)

    def test_addition_wraps_before_comparison_and_negative_constants_keep_bits(self) -> None:
        body = INPUTS + "%sum = arith.addi %lhs, %rhs : i32\n%choose = arith.cmpi eq, %sum, %expected : i32\n" + SELECT
        source = function(body, "%lhs: i32, %rhs: i32, %expected: i32")
        for lhs, rhs, expected in ((0xFFFFFFFF, 1, 0), (0x7FFFFFFF, 1, 0x80000000), (0x80000000, 0xFFFFFFFF, 0x7FFFFFFF)):
            with self.subTest(lhs=hex(lhs), rhs=hex(rhs)):
                self.assert_selection(source, tuple(Scalar(32, bits) for bits in (lhs, rhs, expected)), 0)
                self.assert_selection(source, tuple(Scalar(32, bits) for bits in (lhs, rhs, expected ^ 1)), 1)
        source = function(INPUTS + "%negative = arith.constant -1 : i32\n%choose = arith.cmpi eq, %negative, %expected : i32\n" + SELECT, "%expected: i32")
        self.assert_selection(source, (Scalar(32, 0xFFFFFFFF),), 0)
        self.assert_selection(source, (Scalar(32, 0x7FFFFFFF),), 1)

    def test_i1_constants_and_controls_select_both_paths(self) -> None:
        for literal, selected in (("true", 0), ("false", 1)):
            self.assert_selection(function(INPUTS + f"%choose = arith.constant {literal}\n" + SELECT), (), selected)
        source = function(INPUTS + SELECT, "%choose: i1")
        for choice in (0, 1):
            self.assert_selection(source, (Scalar(1, choice),), 0 if choice else 1)

    def test_diamond_passes_state_bf16_i1_and_i32_arguments(self) -> None:
        source = function(INPUTS + '''
          %ten = arith.constant 10 : i32
          %twenty = arith.constant 20 : i32
          cf.cond_br %choose, ^left(%s2, %a : !atlas.virtual_state, !atlas.virtual_bf16), ^right(%s2, %b : !atlas.virtual_state, !atlas.virtual_bf16)
        ^left(%ls: !atlas.virtual_state, %lt: !atlas.virtual_bf16):
          %yes = arith.constant true
          cf.br ^join(%ls, %lt, %yes, %ten : !atlas.virtual_state, !atlas.virtual_bf16, i1, i32)
        ^right(%rs: !atlas.virtual_state, %rt: !atlas.virtual_bf16):
          %no = arith.constant false
          cf.br ^join(%rs, %rt, %no, %twenty : !atlas.virtual_state, !atlas.virtual_bf16, i1, i32)
        ^join(%js: !atlas.virtual_state, %jt: !atlas.virtual_bf16, %flag: i1, %number: i32):
          %correct = arith.cmpi eq, %number, %expected : i32
          cf.cond_br %correct, ^check(%js, %jt, %flag : !atlas.virtual_state, !atlas.virtual_bf16, i1), ^exit(%js, %b : !atlas.virtual_state, !atlas.virtual_bf16)
        ^check(%cs: !atlas.virtual_state, %ct: !atlas.virtual_bf16, %carried: i1):
          cf.cond_br %carried, ^exit(%cs, %ct : !atlas.virtual_state, !atlas.virtual_bf16), ^exit(%cs, %a : !atlas.virtual_state, !atlas.virtual_bf16)''' + EXIT,
                          "%choose: i1, %expected: i32")
        for choice, expected, selected in ((1, 10, 0), (1, 20, 1), (0, 20, 0), (0, 10, 1)):
            self.assert_selection(source, (Scalar(1, choice), Scalar(32, expected)), selected)
        renamed = source.replace("%a", "%first").replace("%b", "%second").replace("%js", "%joined_state")
        inputs = RuntimeInputs(TILES, (Scalar(1, 1), Scalar(32, 10)))
        self.assertEqual(evaluate(parse_program(source), inputs), evaluate(parse_program(renamed), inputs))

    def test_untaken_paths_have_no_outputs_but_still_declare_inputs(self) -> None:
        source = function('''
          %s0 = "atlas.virtual_start"() : () -> !atlas.virtual_state
          cf.cond_br %choose, ^take(%s0 : !atlas.virtual_state), ^skip(%s0 : !atlas.virtual_state)
        ^take(%ts: !atlas.virtual_state):
          %next, %tile = "atlas.virtual_input_bf16"(%ts) {index = 0 : i32} : (!atlas.virtual_state) -> (!atlas.virtual_state, !atlas.virtual_bf16)
          %relu = "atlas.virtual_vpu_unary"(%tile) {kind = "relu"} : (!atlas.virtual_bf16) -> !atlas.virtual_bf16
          %out = "atlas.virtual_output_bf16"(%next, %relu) {index = 7 : i32} : (!atlas.virtual_state, !atlas.virtual_bf16) -> !atlas.virtual_state
          func.return %out : !atlas.virtual_state
        ^skip(%ss: !atlas.virtual_state):
          func.return %ss : !atlas.virtual_state
        ''', "%choose: i1")
        special = Tile("bf16", (0x7FC1,) * 1024)
        for choice, outputs in ((0, {}), (1, {7: special})):
            self.assertEqual(dict(evaluate(parse_program(source), RuntimeInputs({0: special}, (Scalar(1, choice),))).outputs), outputs)
        with self.assertRaises(VirtualInterfaceError):
            evaluate(parse_program(source), RuntimeInputs(controls=(Scalar(1, 0),)))

    def test_loop_reexecutes_local_definitions_and_carries_changed_tiles(self) -> None:
        program = parse_program(example("virtual_bf16_loop_program.mlir"))
        bits = tuple(0xBF80 if index % 2 else 0x4000 for index in range(1024))
        inputs = RuntimeInputs({0: Tile("bf16", bits)})
        self.assertEqual(evaluate(program, inputs).outputs[0].bits, relu(bits))
        # Six entry operations, two five-operation iterations, the final two-operation check, output and return.
        self.assertEqual(evaluate(program, inputs, max_steps=20), evaluate(program, inputs))
        with self.assertRaises(VirtualInterfaceError):
            evaluate(program, inputs, max_steps=19)

    def test_loop_backedges_bind_swapped_tiles_and_scalars_simultaneously(self) -> None:
        source = example("virtual_bf16_swap_loop_program.mlir")
        program = parse_program(source.replace("@swap_once()", "@swap_once(%limit: i32)").replace("arith.cmpi slt, %i, %c1", "arith.cmpi slt, %i, %limit"))
        for limit, selected in ((0, 0), (1, 1), (2, 0), (3, 1)):
            with self.subTest(limit=limit):
                self.assertEqual(dict(evaluate(program, RuntimeInputs(TILES, (Scalar(32, limit),))).outputs), {0: TILES[selected]})
        scalar_swap = function(INPUTS + '''
          %zero = arith.constant 0 : i32
          %one = arith.constant 1 : i32
          cf.br ^loop(%s2, %zero, %zero, %one : !atlas.virtual_state, i32, i32, i32)
        ^loop(%ls: !atlas.virtual_state, %i: i32, %x: i32, %y: i32):
          %more = arith.cmpi slt, %i, %limit : i32
          %next = arith.addi %i, %one : i32
          cf.cond_br %more, ^loop(%ls, %next, %y, %x : !atlas.virtual_state, i32, i32, i32), ^check(%ls, %x : !atlas.virtual_state, i32)
        ^check(%cs: !atlas.virtual_state, %carried: i32):
          %choose = arith.cmpi eq, %carried, %expected : i32
          cf.cond_br %choose, ^exit(%cs, %a : !atlas.virtual_state, !atlas.virtual_bf16), ^exit(%cs, %b : !atlas.virtual_state, !atlas.virtual_bf16)''' + EXIT,
                               "%limit: i32, %expected: i32")
        for limit, expected in ((1, 1), (2, 0), (3, 1)):
            with self.subTest(scalar_swaps=limit):
                self.assert_selection(scalar_swap, (Scalar(32, limit), Scalar(32, expected)), 0)
                self.assert_selection(scalar_swap, (Scalar(32, limit), Scalar(32, expected ^ 1)), 1)

    def test_repeated_effects_need_fresh_dynamic_state_tokens(self) -> None:
        source = example("virtual_bf16_loop_program.mlir").replace(
            '    %next_tile =', '    %body_next, %unused = "atlas.virtual_input_bf16"(%body_io) {index = 0 : i32} : (!atlas.virtual_state) -> (!atlas.virtual_state, !atlas.virtual_bf16)\n    %next_tile =')
        source = source.replace("cf.br ^loop(%body_io, %next_tile", "cf.br ^loop(%body_next, %next_tile")
        inputs = RuntimeInputs({0: TILES[0]})
        self.assertEqual(evaluate(parse_program(source), inputs).outputs[0], Tile("bf16", (0,) * 1024))
        for stale in ("%body_io", "%io1"):
            with self.subTest(stale=stale), self.assertRaises(VirtualInterfaceError):
                evaluate(parse_program(source.replace("cf.br ^loop(%body_next, %next_tile", f"cf.br ^loop({stale}, %next_tile")), inputs)

    def test_invalid_control_flow_and_dominance_are_rejected(self) -> None:
        source = function(INPUTS + SELECT, "%choose: i1")
        handle = function('''
          %s = "atlas.virtual_start"() : () -> !atlas.virtual_state
          %address = arith.constant 0 : i32
          %size = arith.constant 2048 : i32
          %next, %handle = "atlas.virtual_dma_load_bf16"(%s, %address, %size) : (!atlas.virtual_state, i32, i32) -> (!atlas.virtual_state, !atlas.virtual_dma_load_bf16)
          cf.br ^exit(%next, %handle : !atlas.virtual_state, !atlas.virtual_dma_load_bf16)
        ^exit(%state: !atlas.virtual_state, %pending: !atlas.virtual_dma_load_bf16):
          func.return %state : !atlas.virtual_state''')
        forward = function('''
          %s = "atlas.virtual_start"() : () -> !atlas.virtual_state
          %sum = arith.addi %later, %later : i32
          %later = arith.constant 1 : i32
          func.return %s : !atlas.virtual_state''')
        # The chosen path visits body before exit, but SSA dominance requires %late to cross the edge as a block argument.
        visited = function('''
          %s = "atlas.virtual_start"() : () -> !atlas.virtual_state
          %zero = arith.constant 0 : i32
          %one = arith.constant 1 : i32
          cf.br ^head(%s, %zero : !atlas.virtual_state, i32)
        ^head(%hs: !atlas.virtual_state, %i: i32):
          %first = arith.cmpi eq, %i, %zero : i32
          cf.cond_br %first, ^body(%hs, %i : !atlas.virtual_state, i32), ^exit(%hs : !atlas.virtual_state)
        ^body(%bs: !atlas.virtual_state, %j: i32):
          %late = arith.addi %j, %one : i32
          cf.br ^head(%bs, %late : !atlas.virtual_state, i32)
        ^exit(%es: !atlas.virtual_state):
          %invalid = arith.cmpi eq, %late, %one : i32
          func.return %es : !atlas.virtual_state''')
        untaken = "^exit(%s2, %b : !atlas.virtual_state, !atlas.virtual_bf16)"
        selected = (
            (source.replace("^exit(%s2, %a", "^exit(%s1, %a"), ""),
            (source.replace("func.return %done", "func.return %state"), ""),
            # Both values hold the same dynamic token, but the successor must consume its own state argument.
            (source.replace('"atlas.virtual_output_bf16"(%state, %tile)', '"atlas.virtual_output_bf16"(%s2, %tile)'), "current state token"),
            (source.replace(untaken, "^exit(%s2 : !atlas.virtual_state)"), ""),
            (source.replace(untaken, "^exit(%s2, %choose : !atlas.virtual_state, i1)"), ""),
            (source.replace("^exit(%s2, %b :", "^exit(%s1, %b :"), ""),
            (source.replace("%tile: !atlas.virtual_bf16", "%tile: i32"), ""),
            (source.replace("%state: !atlas.virtual_state", "%state: !atlas.virtual_dma_store"), ""),
        )
        cases = [(changed, message, RuntimeInputs(TILES, (Scalar(1, 1),))) for changed, message in selected]
        cases += [(handle, "non-entry block", RuntimeInputs()), (forward, "does not dominate", RuntimeInputs()), (visited, "", RuntimeInputs())]
        for index, (changed, message, inputs) in enumerate(cases):
            with self.subTest(case=index), self.assertRaisesRegex(VirtualInterfaceError, message):
                evaluate(parse_program(changed), inputs)
        program, other = parse_program(source), parse_program(source)
        branch = tuple(program.blocks[0].ops)[-1]
        # The same spelling and type in another parse do not identify a value.
        branch.operands = (other.blocks[0].args[0], *branch.operands[1:])
        with self.assertRaises(VirtualInterfaceError):
            evaluate(program, RuntimeInputs(TILES, (Scalar(1, 1),)))

    def test_step_budget_counts_branches_and_returns_and_bounds_infinite_loops(self) -> None:
        program, inputs = parse_program(function(INPUTS + SELECT, "%choose: i1")), RuntimeInputs(TILES, (Scalar(1, 0),))
        self.assertEqual(evaluate(program, inputs, max_steps=6), evaluate(program, inputs))
        for budget in (5, 1, 0, -1, True, 1.5, None):
            with self.subTest(budget=budget), self.assertRaises(VirtualInterfaceError):
                evaluate(program, inputs, max_steps=budget)
        infinite = function('''
          %s = "atlas.virtual_start"() : () -> !atlas.virtual_state
          cf.br ^loop(%s : !atlas.virtual_state)
        ^loop(%state: !atlas.virtual_state):
          cf.br ^loop(%state : !atlas.virtual_state)
        ''')
        with self.assertRaises(VirtualInterfaceError):
            evaluate(parse_program(infinite), RuntimeInputs(), max_steps=17)


if __name__ == "__main__":
    unittest.main()
