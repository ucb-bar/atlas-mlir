"""Virtual parser and runtime interface tests; no execution or Atlas tools."""

from __future__ import annotations

from dataclasses import FrozenInstanceError
from pathlib import Path
import sys
import unittest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
from atlas_virtual_evaluator import EvaluationResult, MemoryRegion, RuntimeInputs, Scalar, Tile, UnsupportedVirtualMode, VirtualInterfaceError, operation_name, parse_program  # noqa: E402


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


if __name__ == "__main__":
    unittest.main()
