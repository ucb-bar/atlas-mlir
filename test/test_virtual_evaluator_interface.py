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
from test_virtual_evaluator_arithmetic import INPUTS, relu  # noqa: E402


def example(name: str) -> str:
    return (ROOT / "test/examples" / name).read_text()


FLAT = example("virtual_bf16_ssa.mlir")
CFG = example("virtual_bf16_cfg.mlir")
CMPI = 'module { %a = arith.constant 0 : i32 %c = "arith.cmpi"(%a, %a) <{predicate = 2 : i64}> : (i32, i32) -> i1 }'


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

    def test_valid_generic_and_canonical_property_forms_are_admitted(self) -> None:
        constant, comparison = parse_program(CMPI).operations
        self.assertEqual(operation_name(comparison), "arith.cmpi")
        self.assertEqual(tuple(comparison.operands), (constant.results[0],) * 2)
        relu = example("virtual_bf16_shared_relu.mlir")
        canonical = relu.replace('{index = 0 : i32}', '<{index = 0 : i32}>').replace('{index = 1 : i32}', '<{index = 1 : i32}>').replace('{kind = "relu"}', '<{kind = "relu"}>')
        inputs = RuntimeInputs({0: Tile("bf16", (0xBF80, 0x3F80) * 512)})
        compare_results(evaluate(parse_program(relu), inputs), evaluate(parse_program(canonical), inputs))

    def test_parser_rejects_unsupported_and_malformed_programs(self) -> None:
        canonical = example("virtual_bf16_shared_relu.mlir").replace('{index = 0 : i32}', '<{index = 0 : i32}>')
        seeded = example("virtual_mxu_seeded_fp8.mlir")
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
            (CMPI.replace("predicate = 2", "predicate = 10"), "invalid virtual MLIR|ten predicates"),
            (CMPI.replace("predicate = 2 : i64", "predicate = 2 : i32"), "predicates encoded as i64"),
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
        for snapshot in (inputs, result):
            with self.assertRaises(FrozenInstanceError):
                snapshot.memory = ()

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


RELU_OPERATIONS = '''
  %s0 = "atlas.virtual_start"() : () -> !atlas.virtual_state
  %s1, %original = "atlas.virtual_input_bf16"(%s0) {index = 11 : i32} : (!atlas.virtual_state) -> (!atlas.virtual_state, !atlas.virtual_bf16)
  %rectified = "atlas.virtual_vpu_unary"(%original) {kind = "relu"} : (!atlas.virtual_bf16) -> !atlas.virtual_bf16
  %s2 = "atlas.virtual_output_bf16"(%s1, %original) {index = 23 : i32} : (!atlas.virtual_state, !atlas.virtual_bf16) -> !atlas.virtual_state
  %s3 = "atlas.virtual_output_bf16"(%s2, %rectified) {index = 29 : i32} : (!atlas.virtual_state, !atlas.virtual_bf16) -> !atlas.virtual_state
'''
RELU_FLAT = "module {" + RELU_OPERATIONS + "}"
RELU_FUNCTION = "module { func.func @relu_pair() -> !atlas.virtual_state {" + RELU_OPERATIONS + "func.return %s3 : !atlas.virtual_state } }"


def patterned_tile(offset: int = 0) -> Tile:
    return Tile("bf16", tuple((0x3E00 + offset + index) | (0x8000 if (index // 32 + index % 32) % 2 else 0) for index in range(1024)))


class VirtualEvaluatorExecutionTest(unittest.TestCase):
    def test_shared_input_remains_original_after_relu_and_each_call_uses_fresh_inputs(self) -> None:
        program = parse_program(RELU_FLAT)
        previous = None
        for offset in (0, 0x400):
            tile = patterned_tile(offset)
            result = evaluate(program, RuntimeInputs({11: tile}))
            self.assertEqual(dict(result.outputs), {23: tile, 29: relu(tile)})
            if previous is not None:
                self.assertNotEqual(result.outputs[23], previous.outputs[23])
                self.assertNotEqual(result.outputs[29], previous.outputs[29])
            previous = result

    def test_single_block_function_and_ssa_renaming_have_same_semantics(self) -> None:
        inputs = RuntimeInputs({11: patterned_tile()})
        renamed = RELU_FUNCTION.replace("%s", "%event_").replace("%original", "%logical_input").replace("%rectified", "%logical_output")
        self.assertEqual(evaluate(parse_program(renamed), inputs), evaluate(parse_program(RELU_FLAT), inputs))

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
        with self.assertRaisesRegex(VirtualInterfaceError, "input indices"):
            evaluate(parse_program(source), RuntimeInputs(controls=(Scalar(1, 0),)))

    def test_loop_reexecutes_local_definitions_and_carries_changed_tiles(self) -> None:
        program = parse_program(example("virtual_bf16_loop_program.mlir"))
        tile = Tile("bf16", tuple(0xBF80 if index % 2 else 0x4000 for index in range(1024)))
        inputs = RuntimeInputs({0: tile})
        self.assertEqual(evaluate(program, inputs).outputs[0], relu(tile))
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
            # Both values hold the same dynamic token, but the successor must consume its own state argument.
            (source.replace('"atlas.virtual_output_bf16"(%state, %tile)', '"atlas.virtual_output_bf16"(%s2, %tile)'), "current state token"),
            (source.replace(untaken, "^exit(%s2 : !atlas.virtual_state)"), ""),
            (source.replace(untaken, "^exit(%s2, %choose : !atlas.virtual_state, i1)"), ""),
            (source.replace("^exit(%s2, %b :", "^exit(%s1, %b :"), ""),
            (source.replace("%state: !atlas.virtual_state", "%state: !atlas.virtual_dma_store"), ""),
        )
        cases = [(changed, message, RuntimeInputs(TILES, (Scalar(1, 1),))) for changed, message in selected]
        cases += [(handle, "non-entry block", RuntimeInputs()), (forward, "does not dominate", RuntimeInputs()), (visited, "", RuntimeInputs())]
        for index, (changed, message, inputs) in enumerate(cases):
            with self.subTest(case=index), self.assertRaisesRegex(VirtualInterfaceError, message):
                evaluate(parse_program(changed), inputs)

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
