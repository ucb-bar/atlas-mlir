"""Virtual parser, runtime interface, BF16 execution and scalar/CFG semantics, without Atlas tools."""

from __future__ import annotations

from dataclasses import FrozenInstanceError
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
from atlas_virtual_evaluator import (  # noqa: E402
    EvaluationResult, MemoryRegion, RuntimeInputs, Scalar, Tile, UnsupportedVirtualMode, VirtualInterfaceError, evaluate, operation_name,
    parse_program,
)


def example(name: str) -> str:
    return (ROOT / "test/examples" / name).read_text()


FLAT = example("virtual_bf16_ssa.mlir")
CFG = example("virtual_bf16_cfg.mlir")


class VirtualParserInterfaceTest(unittest.TestCase):
    def test_shared_value_and_output_order_preserve_ssa_identity(self) -> None:
        program = parse_program(FLAT)
        self.assertIsNone(program.function)
        self.assertEqual(dict(program.input_formats), {0: "bf16"})
        self.assertEqual(program.output_indices, (0, 1))
        start, boundary, unary, binary, output0, output1 = program.operations
        self.assertIs(boundary.operands[0], start.results[0])
        self.assertIs(unary.operands[0], boundary.results[1])
        self.assertIs(binary.operands[0], boundary.results[1])
        self.assertIs(binary.operands[1], unary.results[0])
        self.assertIs(output0.operands[1], unary.results[0])
        self.assertIs(output1.operands[1], binary.results[0])

    def test_ssa_renaming_preserves_connections_and_boundary_contract(self) -> None:
        renamed = FLAT.replace("%t", "%logical_tensor_").replace("%io", "%event_")
        original, changed = parse_program(FLAT), parse_program(renamed)
        self.assertEqual(original.input_formats, changed.input_formats)
        self.assertEqual(original.output_indices, changed.output_indices)
        self.assertEqual(tuple(map(operation_name, original.operations)), tuple(map(operation_name, changed.operations)))
        boundary, unary, binary = changed.operations[1:4]
        self.assertIs(unary.operands[0], boundary.results[1])
        self.assertIs(binary.operands[0], boundary.results[1])
        self.assertIs(binary.operands[1], unary.results[0])
        self.assertIsNot(boundary.results[1], original.operations[1].results[1])

    def test_cfg_selection_retains_branch_and_block_argument_identity(self) -> None:
        with self.assertRaisesRegex(VirtualInterfaceError, "select a function"):
            parse_program(CFG)
        with self.assertRaisesRegex(VirtualInterfaceError, "not found"):
            parse_program(CFG, function="absent")
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
        fixtures = (
            "virtual_bf16_loop_program.mlir",
            "virtual_bf16_swap_loop_program.mlir",
            "virtual_bf16_dynamic_branch_program.mlir",
            "virtual_fp8_matmul_program.mlir",
            "virtual_fp8_two_layer_mlp_bias.mlir",
            "virtual_mxu_accumulation.mlir",
            "virtual_mxu_seeded_fp8.mlir",
            "virtual_dma_tiles.mlir",
            "virtual_dma_mxu.mlir",
            "virtual_dma_mxu_bf16.mlir",
        )
        for fixture in fixtures:
            with self.subTest(fixture=fixture):
                program = parse_program(example(fixture))
                self.assertIsNotNone(program.function)
                self.assertTrue(program.operations)

    def test_two_pending_dma_loads_retain_reverse_completion_identities(self) -> None:
        program = parse_program('''module {
          %s = "atlas.virtual_start"() : () -> !atlas.virtual_state
          %a = arith.constant -1879048192 : i32
          %b = arith.constant -1879047168 : i32
          %size = arith.constant 1024 : i32
          %s1, %first = "atlas.virtual_dma_load_fp8"(%s, %a, %size) : (!atlas.virtual_state, i32, i32) -> (!atlas.virtual_state, !atlas.virtual_dma_load_fp8)
          %s2, %second = "atlas.virtual_dma_load_fp8"(%s1, %b, %size) : (!atlas.virtual_state, i32, i32) -> (!atlas.virtual_state, !atlas.virtual_dma_load_fp8)
          %s3, %tile_second = "atlas.virtual_dma_await_fp8"(%s2, %second) : (!atlas.virtual_state, !atlas.virtual_dma_load_fp8) -> (!atlas.virtual_state, !atlas.virtual_fp8)
          %s4, %tile_first = "atlas.virtual_dma_await_fp8"(%s3, %first) : (!atlas.virtual_state, !atlas.virtual_dma_load_fp8) -> (!atlas.virtual_state, !atlas.virtual_fp8)
          %product = "atlas.virtual_mxu_matmul"(%tile_first, %tile_second) {unit = 0 : i32} : (!atlas.virtual_fp8, !atlas.virtual_fp8) -> !atlas.virtual_bf16
          %s5 = "atlas.virtual_output_bf16"(%s4, %product) {index = 0 : i32} : (!atlas.virtual_state, !atlas.virtual_bf16) -> !atlas.virtual_state
        }''')
        first, second, await_second, await_first, contraction, output = program.operations[4:]
        self.assertIsNot(first.results[1], second.results[1])
        self.assertIs(await_second.operands[1], second.results[1])
        self.assertIs(await_first.operands[1], first.results[1])
        self.assertIsNot(await_first.results[1], await_second.results[1])
        self.assertIs(contraction.operands[0], await_first.results[1])
        self.assertIs(contraction.operands[1], await_second.results[1])
        self.assertIs(output.operands[1], contraction.results[0])
        self.assertEqual(program.output_indices, (0,))

    def test_same_unit_independent_mxu_chains_retain_handle_identities(self) -> None:
        source = example("virtual_mxu_accumulation.mlir")
        source = source.replace("unit = 1 : i32", "unit = 0 : i32")
        source = source.replace("virtual_mxu_weight<1>", "virtual_mxu_weight<0>")
        source = source.replace("virtual_mxu_acc<1>", "virtual_mxu_acc<0>")
        program = parse_program(source)
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
        source = example("virtual_bf16_dynamic_branch_program.mlir")
        source = source.replace(
            '    %right_result = "atlas.virtual_vpu_unary"(%right_tile)',
            '    %right_next, %extra = "atlas.virtual_input_bf16"(%right_io) {index = 1 : i32} : (!atlas.virtual_state) -> (!atlas.virtual_state, !atlas.virtual_bf16)\n'
            '    %right_result = "atlas.virtual_vpu_unary"(%extra)',
        ).replace("cf.br ^join(%right_io, %right_result", "cf.br ^join(%right_next, %right_result")
        program = parse_program(source)
        self.assertEqual(dict(program.input_formats), {0: "bf16", 1: "bf16"})
        with self.assertRaisesRegex(VirtualInterfaceError, "input indices"):
            program.validate_inputs(RuntimeInputs({0: Tile("bf16", (0,) * 1024)}, (Scalar(1, 0),)))

    def test_unsupported_operations_and_numerical_modes_fail_explicitly(self) -> None:
        for source, message in (
            (FLAT.replace("atlas.virtual_start", "atlas.start"), "unsupported virtual operation"),
            (FLAT.replace("atlas.virtual_vpu_unary", "atlas.virtual_invented"), "unsupported virtual operation"),
            (FLAT.replace('kind = "relu"', 'kind = "sqrt"'), "supported kinds"),
            (FLAT.replace('kind = "add"', 'kind = "mul"'), "supported kinds"),
            (example("virtual_fp8_two_layer_mlp.mlir").replace("scale_code = 127", "scale_code = 128"), "scale_code 127"),
        ):
            with self.subTest(message=message, source=source[:40]):
                with self.assertRaisesRegex(UnsupportedVirtualMode, message):
                    parse_program(source)

    def test_attributes_and_signatures_are_checked_after_real_parsing(self) -> None:
        for source, message in (
            (FLAT.replace("index = 0 : i32", "index = 0 : i64", 1), "index : i32"),
            (FLAT.replace("index = 0 : i32", "index = -1 : i32", 1), "nonnegative"),
            (FLAT.replace("index = 0 : i32", "index = 0 : i32, invented = 1 : i32", 1), "expected attributes"),
            (FLAT.replace('{kind = "relu"}', "{}"), "expected attributes"),
            (FLAT.replace("atlas.virtual_vpu_unary", "atlas.virtual_vpu_binary"), "signature"),
            (FLAT.replace("index = 1 : i32", "index = 0 : i32"), "duplicate output"),
            ("module { %a = arith.constant 0 : i32 %b = arith.addi %a, %a overflow<nsw> : i32 }", "wrapping i32"),
            ("module { %a = arith.constant 0 : i64 %b = arith.addi %a, %a : i64 }", "i1/i32 integer"),
            ('module { %a = arith.constant 0 : i32 %c = "arith.cmpi"(%a, %a) <{predicate = 10 : i64}> : (i32, i32) -> i1 }', "invalid virtual MLIR|ten predicates"),
        ):
            with self.subTest(message=message):
                with self.assertRaisesRegex(VirtualInterfaceError, message):
                    parse_program(source)

    def test_mxu_units_and_scale_encoding_are_checked(self) -> None:
        source = example("virtual_mxu_seeded_fp8.mlir")
        for changed, message in (
            (source.replace("unit = 0 : i32", "unit = 2 : i32", 1), "unit must be"),
            (source.replace("unit = 0 : i32", "unit = 1 : i32", 1), "handle units must agree"),
            (source.replace("code = 129 : i32", "code = 256 : i32"), "scale code"),
            (source.replace("virtual_mxu_weight<0>", "virtual_mxu_weight<2>"), "unsupported virtual type"),
        ):
            with self.subTest(message=message):
                with self.assertRaisesRegex(VirtualInterfaceError, message):
                    parse_program(changed)

    def test_generic_cmpi_requires_i64_predicate_without_rejecting_valid_form(self) -> None:
        source = 'module { %a = arith.constant 0 : i32 %c = "arith.cmpi"(%a, %a) <{predicate = 2 : i64}> : (i32, i32) -> i1 }'
        program = parse_program(source)
        constant, comparison = program.operations
        self.assertEqual(operation_name(comparison), "arith.cmpi")
        self.assertIs(comparison.operands[0], constant.results[0])
        self.assertIs(comparison.operands[1], constant.results[0])
        with self.assertRaisesRegex(VirtualInterfaceError, "predicates encoded as i64"):
            parse_program(source.replace("predicate = 2 : i64", "predicate = 2 : i32"))

    def test_standard_operations_reject_unknown_attributes_and_properties(self) -> None:
        sources = (
            'module { %a = "arith.constant"() <{value = 0 : i32}> {invented = 1 : i32} : () -> i32 }',
            'module { %a = arith.constant 0 : i32 %b = "arith.addi"(%a, %a) {invented = 1 : i32} : (i32, i32) -> i32 }',
            'module { %a = "arith.constant"() <{value = 0 : i32, invented = 1 : i32}> : () -> i32 }',
            'module { %a = arith.constant 0 : i32 %b = "arith.addi"(%a, %a) <{invented = 1 : i32}> : (i32, i32) -> i32 }',
        )
        for index, source in enumerate(sources):
            with self.subTest(case=index):
                # xDSL rejects unknown properties before interface admission.
                message = "unsupported attributes/properties" if index < 2 else "invalid virtual MLIR: property 'invented' is not defined"
                with self.assertRaisesRegex(VirtualInterfaceError, message):
                    parse_program(source)

    def test_atlas_canonical_properties_preserve_fields_and_reject_duplicates(self) -> None:
        from atlas_virtual_evaluator import compare_results, evaluate

        source = example("virtual_bf16_shared_relu.mlir")
        canonical = source.replace('{index = 0 : i32}', '<{index = 0 : i32}>').replace('{index = 1 : i32}', '<{index = 1 : i32}>').replace('{kind = "relu"}', '<{kind = "relu"}>')
        inputs = RuntimeInputs({0: Tile("bf16", (0xBF80, 0x3F80) * 512)})
        compare_results(evaluate(parse_program(source), inputs), evaluate(parse_program(canonical), inputs))
        duplicate = canonical.replace('<{index = 0 : i32}>', '<{index = 0 : i32}> {index = 1 : i32}', 1)
        with self.assertRaisesRegex(VirtualInterfaceError, "duplicate attribute/property"):
            parse_program(duplicate)
        unknown = canonical.replace('<{index = 0 : i32}>', '<{index = 0 : i32, invented = 1 : i32}>', 1)
        with self.assertRaisesRegex(VirtualInterfaceError, "expected attributes"):
            parse_program(unknown)

    def test_malformed_mlir_has_interface_diagnostic(self) -> None:
        for source in ("module { %x =", FLAT.replace("(%t0)", "(%undefined)", 1)):
            with self.subTest(source=source[:30]):
                with self.assertRaisesRegex(VirtualInterfaceError, "invalid virtual MLIR"):
                    parse_program(source)

    def test_function_name_cannot_select_flat_ir(self) -> None:
        with self.assertRaisesRegex(VirtualInterfaceError, "not found in flat"):
            parse_program(FLAT, function="anything")


class VirtualRuntimeInterfaceTest(unittest.TestCase):
    def test_tiles_copy_bits_and_preserve_special_encodings(self) -> None:
        for format, special in (("bf16", (0x8000, 0x0001, 0x7F80, 0x7FC1)), ("fp8", (0x80, 0x01, 0x7F, 0xFF))):
            with self.subTest(format=format):
                bits = list(special) + [0] * 1020
                tile = Tile(format, bits)
                bits[0] = 0
                self.assertEqual(tile.bits[:4], special)
                self.assertEqual(tile.shape, (32, 32))
                self.assertIsInstance(tile.bits, tuple)
                with self.assertRaises(FrozenInstanceError):
                    tile.format = "fp8"

    def test_tile_values_require_exact_raw_integer_encodings(self) -> None:
        for format, bits in (
            ("f32", [0] * 1024), ("bf16", [0] * 1023), ("fp8", [256] * 1024),
            ("bf16", [-1] * 1024), ("bf16", [True] * 1024), ("fp8", [1.0] * 1024),
        ):
            with self.subTest(format=format, value=bits[0]):
                with self.assertRaises(VirtualInterfaceError):
                    Tile(format, bits)

    def test_scalars_preserve_unsigned_bits_and_reject_implicit_conversion(self) -> None:
        self.assertEqual(Scalar(1, 1).bits, 1)
        self.assertEqual(Scalar(32, 0xFFFFFFFF).bits, 0xFFFFFFFF)
        for width, bits in ((True, 0), (8, 0), (1, 2), (32, -1), (32, 1 << 32), (1, True), (32, 0.0)):
            with self.subTest(width=width, bits=bits):
                with self.assertRaises(VirtualInterfaceError):
                    Scalar(width, bits)

    def test_memory_snapshots_copy_buffers_and_check_address_extent(self) -> None:
        data = bytearray(b"input and guard")
        region = MemoryRegion(0xFFFFFFFF - len(data) + 1, memoryview(data))
        data[:] = b"X" * len(data)
        self.assertEqual(region.data, b"input and guard")
        for address, payload in ((-1, b"x"), (1 << 32, b"x"), (True, b"x"), (0xFFFFFFFF, b"xx"), (0, b""), (0, [1])):
            with self.subTest(address=address, payload=payload):
                with self.assertRaises(VirtualInterfaceError):
                    MemoryRegion(address, payload)

    def test_snapshots_allow_adjacent_regions_and_reject_overlap(self) -> None:
        first, adjacent = MemoryRegion(32, b"ab"), MemoryRegion(34, b"cd")
        inputs = RuntimeInputs(memory=[adjacent, first])
        self.assertEqual(inputs.memory, (adjacent, first))
        overlap = MemoryRegion(33, b"xy")
        for constructor in (lambda: RuntimeInputs(memory=(first, overlap)), lambda: EvaluationResult({}, memory=(overlap, first))):
            with self.assertRaisesRegex(VirtualInterfaceError, "overlap"):
                constructor()

    def test_runtime_collections_own_immutable_copies(self) -> None:
        tile = Tile("bf16", (0,) * 1024)
        tiles, controls = {0: tile}, [Scalar(1, 0)]
        memory = [MemoryRegion(0x90000000, b"guard")]
        inputs = RuntimeInputs(tiles, controls, memory)
        tiles.clear()
        controls.clear()
        memory.clear()
        self.assertEqual(dict(inputs.tiles), {0: tile})
        self.assertEqual(inputs.controls, (Scalar(1, 0),))
        self.assertEqual(len(inputs.memory), 1)
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
            lambda: RuntimeInputs({True: tile}),
            lambda: RuntimeInputs({-1: tile}),
            lambda: RuntimeInputs({1 << 31: tile}),
            lambda: RuntimeInputs({0: tile.bits}),
            lambda: RuntimeInputs(controls=(1,)),
            lambda: RuntimeInputs(memory=(b"bytes",)),
            lambda: EvaluationResult({0: Tile("fp8", (0,) * 1024)}),
            lambda: EvaluationResult({0: object()}),
            lambda: EvaluationResult({}, memory=(b"bytes",)),
        )
        for index, constructor in enumerate(constructors):
            with self.subTest(case=index):
                with self.assertRaises(VirtualInterfaceError):
                    constructor()

    def test_program_validates_exact_boundary_indices_formats_and_controls(self) -> None:
        program = parse_program(CFG, function="choose_tile")
        bf16, fp8 = Tile("bf16", (0,) * 1024), Tile("fp8", (0,) * 1024)
        for choice in (0, 1):
            program.validate_inputs(RuntimeInputs({0: bf16}, (Scalar(1, choice),)))
        invalid = (
            RuntimeInputs({}, (Scalar(1, 0),)),
            RuntimeInputs({0: bf16, 1: bf16}, (Scalar(1, 0),)),
            RuntimeInputs({0: fp8}, (Scalar(1, 0),)),
            RuntimeInputs({0: bf16}),
            RuntimeInputs({0: bf16}, (Scalar(32, 0),)),
            RuntimeInputs({0: bf16}, (Scalar(1, 0), Scalar(1, 1))),
            {0: bf16},
        )
        for index, inputs in enumerate(invalid):
            with self.subTest(case=index):
                with self.assertRaises(VirtualInterfaceError):
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
    bits = tuple((0x3E00 + offset + index) | (0x8000 if (index // 32 + index % 32) % 2 else 0) for index in range(1024))
    return Tile("bf16", bits)


class VirtualEvaluatorExecutionTest(unittest.TestCase):
    def test_shared_input_remains_original_after_relu_and_each_call_uses_fresh_inputs(self) -> None:
        program = parse_program(RELU_FLAT)
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
        expected = evaluate(parse_program(RELU_FLAT), inputs)
        renamed = RELU_FUNCTION.replace("%s", "%event_").replace("%original", "%logical_input").replace("%rectified", "%logical_output")
        self.assertEqual(evaluate(parse_program(renamed), inputs), expected)

    def test_positive_zero_and_extreme_finite_normals_are_admitted_exactly(self) -> None:
        # Smallest/largest mantissas at the first and last normal exponents,
        # with both signs, also exercise values outside ordinary kernel ranges.
        encodings = (0x0000, 0x0080, 0x00FF, 0x8080, 0x80FF, 0x7F00, 0x7F7F, 0xFF00, 0xFF7F)
        bits = tuple(encodings[index % len(encodings)] for index in range(1024))
        result = evaluate(parse_program(RELU_FLAT), RuntimeInputs({11: Tile("bf16", bits)}))
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
        program = parse_program(RELU_FLAT)
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
        result = evaluate(parse_program(RELU_FLAT), inputs)
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
            RELU_FLAT.replace('  %rectified =', second_input + '  %rectified ='),
            RELU_FLAT.replace('(%s2, %rectified)', '(%s1, %rectified)'),
            RELU_FUNCTION.replace('func.return %s3', 'func.return %s1'),
        )
        for index, source in enumerate(sources):
            with self.subTest(consumer=index):
                with self.assertRaises(VirtualInterfaceError):
                    evaluate(parse_program(source), RuntimeInputs({11: patterned_tile()}))

    def test_execution_requires_exactly_one_virtual_start(self) -> None:
        duplicate = '  %extra_start = "atlas.virtual_start"() : () -> !atlas.virtual_state\n'
        cases = (
            ("module {}", RuntimeInputs()),
            (RELU_FLAT.replace('  %s1, %original =', duplicate + '  %s1, %original ='), RuntimeInputs({11: patterned_tile()})),
        )
        for index, (source, inputs) in enumerate(cases):
            with self.subTest(case=index):
                with self.assertRaisesRegex(VirtualInterfaceError, "virtual_start"):
                    evaluate(parse_program(source), inputs)

    def test_operand_from_another_program_has_no_runtime_binding(self) -> None:
        program, other = parse_program(RELU_FLAT), parse_program(RELU_FLAT)
        # A same-typed, similarly named value is still a distinct SSA identity.
        program.operations[2].operands = (other.operations[1].results[1],)
        with self.assertRaises(VirtualInterfaceError):
            evaluate(program, RuntimeInputs({11: patterned_tile()}))


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
TILES = {0: Tile("bf16", (0xBF80,) * 1024), 1: Tile("bf16", (0x4000,) * 1024)}


def function(body: str, arguments: str = "") -> str:
    return "module { func.func @test(" + arguments + ") -> !atlas.virtual_state {" + body + "} }"


class VirtualEvaluatorCFGTest(unittest.TestCase):
    def assert_selection(self, source: str, controls: tuple[Scalar, ...], selected: int) -> None:
        result = evaluate(parse_program(source), RuntimeInputs(TILES, controls))
        self.assertEqual(dict(result.outputs), {7: TILES[selected]})

    def test_all_comparisons_distinguish_signed_and_unsigned_i32_boundaries(self) -> None:
        # Expected answers are explicit to catch sign extension or predicate swaps.
        predicates = ("eq", "ne", "slt", "sle", "sgt", "sge", "ult", "ule", "ugt", "uge")
        cases = (
            (0xFFFFFFFF, 0, (0, 1, 1, 1, 0, 0, 0, 0, 1, 1)),
            (0x80000000, 0x7FFFFFFF, (0, 1, 1, 1, 0, 0, 0, 0, 1, 1)),
            (0x7FFFFFFF, 0x80000000, (0, 1, 0, 0, 1, 1, 1, 1, 0, 0)),
            (0x80000000, 0x80000000, (1, 0, 0, 1, 0, 1, 0, 1, 0, 1)),
            (0, 1, (0, 1, 1, 1, 0, 0, 1, 1, 0, 0)),
        )
        for predicate, position in zip(predicates, range(10)):
            source = function(INPUTS + f"%choose = arith.cmpi {predicate}, %lhs, %rhs : i32\n" + SELECT, "%lhs: i32, %rhs: i32")
            for lhs, rhs, expected in cases:
                with self.subTest(predicate=predicate, lhs=hex(lhs), rhs=hex(rhs)):
                    self.assert_selection(source, (Scalar(32, lhs), Scalar(32, rhs)), 0 if expected[position] else 1)

    def test_addition_wraps_before_comparison_and_negative_constants_keep_bits(self) -> None:
        body = INPUTS + '''
          %sum = arith.addi %lhs, %rhs : i32
          %choose = arith.cmpi eq, %sum, %expected : i32
        ''' + SELECT
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
          cf.cond_br %carried, ^exit(%cs, %ct : !atlas.virtual_state, !atlas.virtual_bf16), ^exit(%cs, %a : !atlas.virtual_state, !atlas.virtual_bf16)
        ^exit(%es: !atlas.virtual_state, %et: !atlas.virtual_bf16):
          %done = "atlas.virtual_output_bf16"(%es, %et) {index = 7 : i32} : (!atlas.virtual_state, !atlas.virtual_bf16) -> !atlas.virtual_state
          func.return %done : !atlas.virtual_state
        ''', "%choose: i1, %expected: i32")
        self.assert_selection(source, (Scalar(1, 1), Scalar(32, 10)), 0)
        self.assert_selection(source, (Scalar(1, 1), Scalar(32, 20)), 1)
        self.assert_selection(source, (Scalar(1, 0), Scalar(32, 20)), 0)
        self.assert_selection(source, (Scalar(1, 0), Scalar(32, 10)), 1)
        renamed = source.replace("%a", "%first").replace("%b", "%second").replace("%js", "%joined_state")
        self.assertEqual(evaluate(parse_program(source), RuntimeInputs(TILES, (Scalar(1, 1), Scalar(32, 10)))), evaluate(parse_program(renamed), RuntimeInputs(TILES, (Scalar(1, 1), Scalar(32, 10)))))

    def test_untaken_paths_have_no_outputs(self) -> None:
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
        self.assertEqual(dict(evaluate(parse_program(source), RuntimeInputs({0: special}, (Scalar(1, 0),))).outputs), {})
        self.assertEqual(dict(evaluate(parse_program(source), RuntimeInputs({0: special}, (Scalar(1, 1),))).outputs), {7: special})
        # Declarations still define the input interface even when not executed.
        with self.assertRaises(VirtualInterfaceError):
            evaluate(parse_program(source), RuntimeInputs(controls=(Scalar(1, 0),)))

    def test_loop_reexecutes_local_definitions_and_carries_changed_tiles(self) -> None:
        source = example("virtual_bf16_loop_program.mlir")
        program = parse_program(source)
        bits = tuple(0xBF80 if index % 2 else 0x4000 for index in range(1024))
        inputs = RuntimeInputs({0: Tile("bf16", bits)})
        self.assertEqual(evaluate(program, inputs).outputs[0].bits, tuple(0 if value & 0x8000 else value for value in bits))
        # Six entry operations, two full five-operation iterations, the final
        # two-operation condition check, output, and return total twenty steps.
        self.assertEqual(evaluate(program, inputs, max_steps=20), evaluate(program, inputs))
        with self.assertRaises(VirtualInterfaceError):
            evaluate(program, inputs, max_steps=19)

    def test_loop_backedges_bind_swapped_tiles_and_scalars_simultaneously(self) -> None:
        source = example("virtual_bf16_swap_loop_program.mlir")
        source = source.replace("@swap_once()", "@swap_once(%limit: i32)").replace("arith.cmpi slt, %i, %c1", "arith.cmpi slt, %i, %limit")
        program = parse_program(source)
        for limit, selected in ((0, 0), (1, 1), (2, 0), (3, 1)):
            with self.subTest(limit=limit):
                result = evaluate(program, RuntimeInputs(TILES, (Scalar(32, limit),)))
                self.assertEqual(dict(result.outputs), {0: TILES[selected]})
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
          cf.cond_br %choose, ^exit(%cs, %a : !atlas.virtual_state, !atlas.virtual_bf16), ^exit(%cs, %b : !atlas.virtual_state, !atlas.virtual_bf16)
        ^exit(%es: !atlas.virtual_state, %et: !atlas.virtual_bf16):
          %done = "atlas.virtual_output_bf16"(%es, %et) {index = 7 : i32} : (!atlas.virtual_state, !atlas.virtual_bf16) -> !atlas.virtual_state
          func.return %done : !atlas.virtual_state
        ''', "%limit: i32, %expected: i32")
        for limit, expected in ((1, 1), (2, 0), (3, 1)):
            with self.subTest(scalar_swaps=limit):
                self.assert_selection(scalar_swap, (Scalar(32, limit), Scalar(32, expected)), 0)
                self.assert_selection(scalar_swap, (Scalar(32, limit), Scalar(32, expected ^ 1)), 1)

    def test_repeated_effects_need_fresh_dynamic_state_tokens(self) -> None:
        source = example("virtual_bf16_loop_program.mlir")
        source = source.replace('    %next_tile =', '    %body_next, %unused = "atlas.virtual_input_bf16"(%body_io) {index = 0 : i32} : (!atlas.virtual_state) -> (!atlas.virtual_state, !atlas.virtual_bf16)\n    %next_tile =')
        source = source.replace("cf.br ^loop(%body_io, %next_tile", "cf.br ^loop(%body_next, %next_tile")
        inputs = RuntimeInputs({0: TILES[0]})
        self.assertEqual(evaluate(parse_program(source), inputs).outputs[0], Tile("bf16", (0,) * 1024))
        for stale in ("%body_io", "%io1"):
            with self.subTest(stale=stale):
                changed = source.replace("cf.br ^loop(%body_next, %next_tile", f"cf.br ^loop({stale}, %next_tile")
                with self.assertRaises(VirtualInterfaceError):
                    evaluate(parse_program(changed), inputs)

    def test_stale_branch_tokens_and_foreign_ssa_identities_fail(self) -> None:
        source = function(INPUTS + SELECT, "%choose: i1")
        for changed in (source.replace("^exit(%s2, %a", "^exit(%s1, %a"), source.replace("func.return %done", "func.return %state")):
            with self.assertRaises(VirtualInterfaceError):
                evaluate(parse_program(changed), RuntimeInputs(TILES, (Scalar(1, 1),)))
        program, other = parse_program(source), parse_program(source)
        branch = tuple(program.blocks[0].ops)[-1]
        # The same spelling and type in another parse do not identify a value.
        branch.operands = (other.blocks[0].args[0], *branch.operands[1:])
        with self.assertRaises(VirtualInterfaceError):
            evaluate(program, RuntimeInputs(TILES, (Scalar(1, 1),)))

    def test_dominating_state_alias_cannot_replace_the_current_block_argument(self) -> None:
        source = function(INPUTS + SELECT, "%choose: i1")
        source = source.replace('"atlas.virtual_output_bf16"(%state, %tile)', '"atlas.virtual_output_bf16"(%s2, %tile)')
        # Both values would hold the same dynamic token, but the successor must
        # consume its own current state argument rather than a predecessor alias.
        with self.assertRaisesRegex(VirtualInterfaceError, "current state token"):
            evaluate(parse_program(source), RuntimeInputs(TILES, (Scalar(1, 1),)))

    def test_same_block_forward_scalar_reference_does_not_dominate_its_use(self) -> None:
        source = function('''
          %s = "atlas.virtual_start"() : () -> !atlas.virtual_state
          %sum = arith.addi %later, %later : i32
          %later = arith.constant 1 : i32
          func.return %s : !atlas.virtual_state
        ''')
        with self.assertRaisesRegex(VirtualInterfaceError, "does not dominate"):
            evaluate(parse_program(source), RuntimeInputs())

    def test_step_budget_counts_branches_and_returns_and_bounds_infinite_loops(self) -> None:
        source = function(INPUTS + SELECT, "%choose: i1")
        program, inputs = parse_program(source), RuntimeInputs(TILES, (Scalar(1, 0),))
        self.assertEqual(evaluate(program, inputs, max_steps=6), evaluate(program, inputs))
        for budget in (5, 1, 0, -1, True, 1.5, None):
            with self.subTest(budget=budget):
                with self.assertRaises(VirtualInterfaceError):
                    evaluate(program, inputs, max_steps=budget)
        infinite = function('''
          %s = "atlas.virtual_start"() : () -> !atlas.virtual_state
          cf.br ^loop(%s : !atlas.virtual_state)
        ^loop(%state: !atlas.virtual_state):
          cf.br ^loop(%state : !atlas.virtual_state)
        ''')
        with self.assertRaises(VirtualInterfaceError):
            evaluate(parse_program(infinite), RuntimeInputs(), max_steps=17)

    def test_parser_rejects_handle_crossings_and_branch_type_mismatches(self) -> None:
        source = function(INPUTS + SELECT, "%choose: i1")
        for changed in (source.replace("%tile: !atlas.virtual_bf16", "%tile: i32"), source.replace("%state: !atlas.virtual_state", "%state: !atlas.virtual_dma_store")):
            with self.assertRaises(VirtualInterfaceError):
                parse_program(changed)
        handle = function('''
          %s = "atlas.virtual_start"() : () -> !atlas.virtual_state
          %address = arith.constant 0 : i32
          %size = arith.constant 2048 : i32
          %next, %handle = "atlas.virtual_dma_load_bf16"(%s, %address, %size) : (!atlas.virtual_state, i32, i32) -> (!atlas.virtual_state, !atlas.virtual_dma_load_bf16)
          cf.br ^exit(%next, %handle : !atlas.virtual_state, !atlas.virtual_dma_load_bf16)
        ^exit(%state: !atlas.virtual_state, %pending: !atlas.virtual_dma_load_bf16):
          func.return %state : !atlas.virtual_state
        ''')
        with self.assertRaisesRegex(VirtualInterfaceError, "non-entry block"):
            parse_program(handle)

    def test_malformed_untaken_edges_are_rejected(self) -> None:
        source = function(INPUTS + SELECT, "%choose: i1")
        cases = (
            source.replace("^exit(%s2, %b : !atlas.virtual_state, !atlas.virtual_bf16)", "^exit(%s2 : !atlas.virtual_state)"),
            source.replace("^exit(%s2, %b : !atlas.virtual_state, !atlas.virtual_bf16)", "^exit(%s2, %choose : !atlas.virtual_state, i1)"),
            source.replace("^exit(%s2, %b :", "^exit(%s1, %b :"),
        )
        for changed in cases:
            with self.assertRaises(VirtualInterfaceError):
                evaluate(parse_program(changed), RuntimeInputs(TILES, (Scalar(1, 1),)))

    def test_a_previous_visit_does_not_make_a_nondominating_definition_valid(self) -> None:
        source = function('''
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
          func.return %es : !atlas.virtual_state
        ''')
        # The chosen path visits body before exit, but SSA dominance requires
        # this value to cross the edge explicitly as a block argument.
        with self.assertRaises(VirtualInterfaceError):
            evaluate(parse_program(source), RuntimeInputs())


if __name__ == "__main__":
    unittest.main()
