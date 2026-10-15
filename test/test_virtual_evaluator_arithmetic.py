"""Raw-bit VPU, MXU0 and MXU1 arithmetic with independent literals and no Atlas tools.

VPU literals distinguish FP32 nearest-even plus BF16 chopping from BF16 rounding. MXU expectations are raw literals from
the selected FMA/anchor sources, exact rational witnesses and the pinned RTL arithmetic fixture, not model numerics.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import platform
import subprocess
import sys
import textwrap
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
from atlas_virtual_evaluator import (  # noqa: E402
    RuntimeInputs, Tile, UnsupportedVirtualMode, VirtualInterfaceError, _mxu_tile, evaluate, evaluate_tile_operation, parse_program,
)
from test_virtual_evaluator_resources import (  # noqa: E402
    A, S, TILES, accumulate, diagonal, kernel, legacy, on_unit, output, read, reset, seed, weight,
)
from xdsl.dialects import builtin  # noqa: E402


INPUTS = '''
  %s0 = "atlas.virtual_start"() : () -> !atlas.virtual_state
  %s1, %a = "atlas.virtual_input_bf16"(%s0) {index = 0 : i32} : (!atlas.virtual_state) -> (!atlas.virtual_state, !atlas.virtual_bf16)
  %s2, %b = "atlas.virtual_input_bf16"(%s1) {index = 1 : i32} : (!atlas.virtual_state) -> (!atlas.virtual_state, !atlas.virtual_bf16)
'''
MOV = '%m = "atlas.virtual_vpu_unary"(%a) {kind = "mov"} : (!atlas.virtual_bf16) -> !atlas.virtual_bf16'
RELU = MOV.replace('kind = "mov"', 'kind = "relu"')
ADD = '%sum = "atlas.virtual_vpu_binary"(%a, %b) {kind = "add"} : (!atlas.virtual_bf16, !atlas.virtual_bf16) -> !atlas.virtual_bf16'
SUB = ADD.replace('kind = "add"', 'kind = "sub"')
MUL = ADD.replace('kind = "add"', 'kind = "mul"')
EXP2 = MOV.replace('kind = "mov"', 'kind = "exp2"')
RECIP = MOV.replace('kind = "mov"', 'kind = "recip"')
PACK = '%packed = "atlas.virtual_pack_fp8"(%a) {scale_code = 127 : i32} : (!atlas.virtual_bf16) -> !atlas.virtual_fp8'

FP8_INPUTS = INPUTS.replace("input_bf16", "input_fp8").replace("virtual_bf16", "virtual_fp8")
TRANSPOSE = '%transposed = "atlas.virtual_xlu_transpose"(%a) : (!atlas.virtual_fp8) -> !atlas.virtual_fp8'

# FP32 nearest-even plus BF16 chopping, gradual underflow, signed zeros and canonical NaNs.
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
# SUB uses FP32 arithmetic and BF16 chopping.
SUB_CASES = (
    (0x3E80, 0x3EC0, 0xBE00), (0x3E8F, 0x3ECF, 0xBE00),
    (0x3F80, 0x3B40, 0x3F7F), (0xBF80, 0x3BC0, 0xBF80),
    (0x3F80, 0xBBC0, 0x3F80), (0x3F80, 0xBC40, 0x3F81),
    (0x0081, 0x0080, 0x0001), (0x0080, 0x0081, 0x8001),
    (0x007F, 0x8001, 0x0080), (0x0001, 0x0001, 0x0000),
    (0x8000, 0x0000, 0x8000), (0x8000, 0x8000, 0x0000),
    (0x7F7F, 0xFF7F, 0x7F80), (0xFF7F, 0x7F7F, 0xFF80),
    (0x7F80, 0x7F80, 0x7FC0), (0xFFA1, 0x3F80, 0x7FC0),
)
# MUL uses BF16 nearest-even rounding, including subnormals.
MUL_CASES = (
    (0x3E80, 0x3EC0, 0x3DC0), (0x3E81, 0x3EC1, 0x3DC3),
    (0x3E82, 0x3EC2, 0x3DC5), (0x3E8F, 0x3ECF, 0x3DE7),
    (0x3F81, 0x3F81, 0x3F82), (0x3F81, 0x3FC0, 0x3FC2),
    (0x3F83, 0x3FC0, 0x3FC4), (0xBF81, 0x3FC0, 0xBFC2),
    (0x3FC0, 0x3FC0, 0x4010), (0x0080, 0x3F00, 0x0040),
    (0x0080, 0x3F7F, 0x0080), (0x0001, 0x3F00, 0x0000),
    (0x0003, 0x3F00, 0x0002), (0x0005, 0x3F00, 0x0002),
    (0x8001, 0x3F00, 0x8000), (0x8003, 0x3F00, 0x8002),
    (0x8000, 0xBF80, 0x0000), (0x0000, 0xBF80, 0x8000),
    (0x7F7F, 0x4000, 0x7F80), (0xFF7F, 0x4000, 0xFF80),
    (0x7F80, 0x0000, 0x7FC0), (0xFFC1, 0x3F80, 0x7FC0),
)
# Captured VPU bits: npu-model@0c4a1f9, tests/rtl/vector_traces.json, bank 4/row 0.
# SHA256: 05fccec50691381c3cf1f3c693f71e2b976bc501879e100af2492c6eabdeed17.
FRACTIONAL_INPUTS = tuple(range(0x3E80, 0x3E90))
EXP2_FRACTIONAL = (0x3F98, 0x3F98, 0x3F99, 0x3F99, 0x3F99, 0x3F99, 0x3F99, 0x3F9A,
                   0x3F9A, 0x3F9A, 0x3F9A, 0x3F9B, 0x3F9B, 0x3F9B, 0x3F9B, 0x3F9B)
RECIP_FRACTIONAL = (0x4080, 0x407E, 0x407C, 0x407A, 0x4078, 0x4076, 0x4074, 0x4072,
                    0x4070, 0x406F, 0x406D, 0x406B, 0x406A, 0x4068, 0x4066, 0x4065)
EXP2_CASES = tuple(zip(FRACTIONAL_INPUTS, EXP2_FRACTIONAL)) + (
    (0x0000, 0x3F80), (0x8000, 0x3F80), (0x3F80, 0x4000),
    (0xBF80, 0x3F00), (0x4000, 0x4080), (0xC000, 0x3E80),
)
# RcpLUT corners, including exponent wrap: diagnostics/characterize_vpu_recip.py.
RECIP_CASES = tuple(zip(FRACTIONAL_INPUTS, RECIP_FRACTIONAL)) + (
    (0x0000, 0x7F80), (0x8000, 0xFF80), (0x0001, 0x7F80), (0x8001, 0xFF80),
    (0x007F, 0x7F80), (0x807F, 0xFF80), (0x7F80, 0x0000), (0xFF80, 0x8000),
    (0x7FC1, 0x0000), (0xFFA1, 0x8000), (0x3F80, 0x3F80), (0xBF80, 0xBF80),
    (0x4000, 0x3F00), (0xC000, 0xBF00), (0x0080, 0x7E80), (0x7F00, 0x0000),
    (0x7F7F, 0x7F80), (0xFF7F, 0xFF80),
)
# Source-derived unit-scale FP8Pack expectations: even/odd RNE ties, exponent carry, flush around the
# minimum normal, saturation and specials. Rounded 480 is 0x7e here, unlike the MXU converter's 0x7f.
PACK_CASES = (
    (0x0000, 0x00), (0x8000, 0x00), (0x0001, 0x00), (0x007F, 0x00), (0x8001, 0x00), (0x807F, 0x00), (0x7F81, 0x00),
    (0x7FC0, 0x00), (0xFF81, 0x00), (0xFFC0, 0x00), (0x7F80, 0x7E), (0xFF80, 0xFE), (0x3F80, 0x38), (0xBF80, 0xB8),
    (0x3F87, 0x38), (0x3F88, 0x38), (0x3F89, 0x39), (0x3F98, 0x3A), (0xBF98, 0xBA), (0x3FF7, 0x3F), (0x3FF8, 0x40),
    (0xBFF8, 0xC0), (0x3C60, 0x00), (0x3C77, 0x00), (0x3C78, 0x08), (0x3C80, 0x08), (0xBC77, 0x00), (0xBC78, 0x88),
    (0x43E0, 0x7E), (0x43E8, 0x7E), (0x43E9, 0x7E), (0x43F0, 0x7E), (0x7F7F, 0x7E), (0xC3E0, 0xFE), (0xC3E8, 0xFE),
    (0xC3E9, 0xFE), (0xC3F0, 0xFE), (0xFF7F, 0xFE),
)


def operation(source: str):
    return parse_program("module {" + INPUTS + source + "}").operations[-1]


def transpose_operation():
    return parse_program("module {" + FP8_INPUTS + TRANSPOSE + "}").operations[-1]


def repeated(values, format="bf16") -> Tile:
    return Tile(format, tuple(values[i % len(values)] for i in range(1024)))


def relu(tile: Tile) -> Tile:
    return Tile("bf16", tuple(0 if bits & 0x8000 else bits for bits in tile.bits))


SPLAT = '%splat = "atlas.virtual_vpu_splat"() {bits = 0 : i32} : () -> !atlas.virtual_bf16'
REDUCE = '%reduced = "atlas.virtual_vpu_reduce"(%a) {kind = "row_sum"} : (!atlas.virtual_bf16) -> !atlas.virtual_bf16'
# Each row is padded with +0. Expected sum/min/max are independent raw literals.
REDUCTION_CASES = (
    ((0x3F80,) * 32, 0x4200, 0x3F80, 0x3F80),
    ((0x4000,) * 32, 0x4280, 0x4000, 0x4000),
    # Final BF16 rounding retains two half-ULP terms; BF16 intermediate rounding loses them.
    ((0x3F80, 0x3B80, 0x3B80), 0x3F81, 0x0000, 0x3F80),
    # Halfway sums select the even BF16 mantissa, rounding down or up as needed.
    ((0x3F80, 0x3B80), 0x3F80, 0x0000, 0x3F80),
    ((0x3F81, 0x3B80), 0x3F82, 0x0000, 0x3F81),
    # Final BF16 RNE rounds up here; BF16 bit chopping would retain 0x3F80.
    ((0x3F80,) + (0,) * 15 + (0x3BC0,), 0x3F81, 0x0000, 0x3F80),
    ((0xBF80, 0xC000, 0x4040), 0x0000, 0xC000, 0x4040),
    # Adjacent FP32 pairs lose both ones: tree=0, sequential=1, exact=2.
    ((0x4C00, 0x3F80, 0xCC00, 0x3F80), 0x0000, 0xCC00, 0x4C00),
    ((0x0000, 0x8000), 0x0000, 0x8000, 0x0000),
    ((0x8000,) * 32, 0x8000, 0x8000, 0x8000),
    ((0x7FC1, 0x3F80), 0x7FC0, 0x0000, 0x7FC1),
    ((0xFFC1, 0xBF80), 0x7FC0, 0xFFC1, 0x0000),
    ((0x7F80, 0xFF80), 0x7FC0, 0xFF80, 0x7F80),
    ((0x0001, 0x0001), 0x0002, 0x0000, 0x0001),
)


def row_reduction_fixture():
    rows = tuple(REDUCTION_CASES[row % len(REDUCTION_CASES)] for row in range(32))
    tile = Tile("bf16", tuple(bits for values, *_ in rows for bits in values + (0,) * (32 - len(values))))
    expected = {
        kind: Tile("bf16", tuple(case[index] for case in rows for _ in range(32)))
        for index, kind in enumerate(("row_sum", "row_min", "row_max"), 1)
    }
    return tile, expected


class VirtualEvaluatorTransposeTest(unittest.TestCase):
    def test_transpose_preserves_all_raw_fp8_codes_without_numerics(self) -> None:
        rows = tuple(tuple((row * 37 + col * 11) & 255 for col in range(32)) for row in range(32))
        source = Tile("fp8", tuple(bits for row in rows for bits in row))
        self.assertEqual(set(source.bits), set(range(256)))
        expected = Tile("fp8", tuple(bits for column in zip(*rows) for bits in column))
        original = source.bits
        with patch("atlas_virtual_evaluator._numerics", side_effect=AssertionError("transpose needs no numerics")):
            transposed = evaluate_tile_operation(transpose_operation(), (source,))
            self.assertEqual(transposed, expected)
            self.assertEqual(evaluate_tile_operation(transpose_operation(), (transposed,)), source)
        self.assertEqual(source.bits, original)
        self.assertIsNot(transposed, source)

    def test_transpose_moves_asymmetric_special_bytes_across_halves_and_edges(self) -> None:
        bits = [0x38] * 1024
        placements = ((0, 31, 0x80), (31, 0, 0x00), (15, 16, 0x7F), (16, 15, 0xFF),
                      (1, 30, 0x01), (30, 1, 0x81), (7, 23, 0x7E), (23, 7, 0xFE))
        for row, col, value in placements:
            bits[row * 32 + col] = value
        result = evaluate_tile_operation(transpose_operation(), (Tile("fp8", tuple(bits)),))
        targets = ((31, 0, 0x80), (0, 31, 0x00), (16, 15, 0x7F), (15, 16, 0xFF),
                   (30, 1, 0x01), (1, 30, 0x81), (23, 7, 0x7E), (7, 23, 0xFE))
        for row, col, value in targets:
            with self.subTest(row=row, col=col):
                self.assertEqual(result.bits[row * 32 + col], value)
        self.assertEqual(sum(value != 0x38 for value in result.bits), len(targets))

    def test_transpose_helper_checks_runtime_operands_and_revalidates_operation(self) -> None:
        fp8, bf16 = repeated((0x7F,), "fp8"), repeated((0,))
        for operands in ((), (fp8, fp8), (bf16,), [fp8], (None,), None):
            with self.subTest(operands=operands), self.assertRaises(VirtualInterfaceError):
                evaluate_tile_operation(transpose_operation(), operands)
        for mutation in ("kind", "property", "operands", "input_type", "result_type"):
            op = transpose_operation()
            if mutation == "kind":
                op.attributes["kind"] = builtin.StringAttr("mov")
            elif mutation == "property":
                op.properties["invented"] = builtin.IntegerAttr(0, builtin.i32)
            elif mutation == "operands":
                op.operands = ()
            elif mutation == "input_type":
                op.operands[0]._type = operation(MOV).operands[0].type
            else:
                op.results[0]._type = operation(MOV).results[0].type
            with self.subTest(mutation=mutation), self.assertRaises(VirtualInterfaceError):
                evaluate_tile_operation(op, (fp8,))


class VirtualEvaluatorReductionTest(unittest.TestCase):
    def test_splat_preserves_raw_bf16_encodings(self) -> None:
        for bits in (0x0000, 0x8000, 0x0001, 0x3F80, 0x7F80, 0x7F81, 0x7FC1, 0xFFFF):
            source = SPLAT.replace("bits = 0", f"bits = {bits}")
            with self.subTest(bits=hex(bits)), patch("atlas_virtual_evaluator._numerics", side_effect=AssertionError("splat needs no numerics")):
                self.assertEqual(evaluate_tile_operation(operation(source), ()), repeated((bits,)))

    def test_reductions_round_and_broadcast_rows(self) -> None:
        tile, expected = row_reduction_fixture()
        for kind, wanted in expected.items():
            source = REDUCE.replace('"row_sum"', f'"{kind}"')
            with self.subTest(kind=kind):
                self.assertEqual(evaluate_tile_operation(operation(source), (tile,)), wanted)

    def test_reductions_preserve_all_32_row_identities(self) -> None:
        # Rows contain 1, 2, 4, ...; each 0x80 step increments the BF16 exponent.
        codes = tuple(0x3F80 + (row << 7) for row in range(32))
        tile = Tile("bf16", tuple(code for code in codes for _ in range(32)))
        # Summing 32 equal powers of two increments the exponent by five.
        for kind, offset in (("row_sum", 0x0280), ("row_min", 0), ("row_max", 0)):
            wanted = Tile("bf16", tuple(code + offset for code in codes for _ in range(32)))
            with self.subTest(kind=kind):
                self.assertEqual(evaluate_tile_operation(operation(REDUCE.replace("row_sum", kind)), (tile,)), wanted)

    def test_full_evaluation_publishes_reductions_and_raw_splats(self) -> None:
        tile, reductions = row_reduction_fixture()
        operations = [
            REDUCE.replace("%reduced", f"%{kind}").replace('"row_sum"', f'"{kind}"')
            for kind in reductions
        ]
        values = list(reductions.items())
        for name, bits in (("negative_zero", 0x8000), ("nan_payload", 0x7FC1)):
            operations.append(SPLAT.replace("%splat", f"%{name}").replace("bits = 0", f"bits = {bits}"))
            values.append((name, repeated((bits,))))
        states = ("s2",) + tuple(f"out{index}" for index in range(len(values)))
        operations.extend(
            output(states[index], states[index + 1], name, index)
            for index, (name, _) in enumerate(values)
        )
        for canonical in (False, True):
            body = operations if not canonical else [text.replace(" {", " <{").replace("} :", "}> :") for text in operations]
            source = "module {" + INPUTS + "\n".join(body) + "\n}"
            with self.subTest(canonical=canonical):
                result = evaluate(parse_program(source), RuntimeInputs({0: tile, 1: repeated((0,))}))
                self.assertEqual(dict(result.outputs), {index: wanted for index, (_, wanted) in enumerate(values)})

    def test_helper_checks_operands_and_revalidates_fields(self) -> None:
        tile, fp8 = repeated((0,)), repeated((0,), "fp8")
        for source, operands in ((SPLAT, (tile,)), (SPLAT, []), (REDUCE, ()), (REDUCE, (tile, tile)), (REDUCE, (fp8,))):
            with self.subTest(source=source, operands=operands), self.assertRaises(VirtualInterfaceError):
                evaluate_tile_operation(operation(source), operands)
        for bits in (-1, 65536):
            op = operation(SPLAT)
            op.attributes["bits"] = builtin.IntegerAttr(bits, builtin.i32)
            with self.subTest(bits=bits), self.assertRaisesRegex(VirtualInterfaceError, "bits must be in"):
                evaluate_tile_operation(op, ())
        op = operation(REDUCE)
        op.attributes["kind"] = builtin.StringAttr("col_sum")
        with self.assertRaises(UnsupportedVirtualMode):
            evaluate_tile_operation(op, (tile,))


class VirtualEvaluatorVpuTest(unittest.TestCase):
    def test_mov_and_relu_exhaust_every_raw_bf16_encoding(self) -> None:
        mov, rectify = operation(MOV), operation(RELU)
        for first in range(0, 65536, 1024):
            with self.subTest(first=hex(first)):
                tile = Tile("bf16", tuple(range(first, first + 1024)))
                self.assertEqual(evaluate_tile_operation(mov, (tile,)), tile)
                self.assertEqual(evaluate_tile_operation(rectify, (tile,)), relu(tile))

    def test_add_matches_independent_raw_bit_literals(self) -> None:
        left, right, expected = (repeated([case[i] for case in ADD_CASES]) for i in range(3))
        self.assertEqual(evaluate_tile_operation(operation(ADD), (left, right)), expected)

    def test_sub_and_mul_match_independent_rounding_and_special_literals(self) -> None:
        for source, cases in ((SUB, SUB_CASES), (MUL, MUL_CASES)):
            left, right, expected = (repeated([case[i] for case in cases]) for i in range(3))
            with self.subTest(source=source):
                self.assertEqual(evaluate_tile_operation(operation(source), (left, right)), expected)

    def test_exp2_and_recip_match_independent_bounded_literals(self) -> None:
        for source, cases in ((EXP2, EXP2_CASES), (RECIP, RECIP_CASES)):
            tile, expected = (repeated([case[i] for case in cases]) for i in range(2))
            with self.subTest(source=source):
                self.assertEqual(evaluate_tile_operation(operation(source), (tile,)), expected)

    def test_attention_helpers_revalidate_operand_arity_and_format(self) -> None:
        tile, fp8 = repeated((0,)), repeated((0,), "fp8")
        for source, arity in ((SUB, 2), (MUL, 2), (EXP2, 1), (RECIP, 1)):
            for operands in ((tile,) * (arity - 1), (tile,) * (arity + 1), (fp8,) * arity):
                with self.subTest(source=source, operands=operands), self.assertRaises(VirtualInterfaceError):
                    evaluate_tile_operation(operation(source), operands)
            op = operation(source)
            op.operands = ()
            with self.subTest(source=source), self.assertRaisesRegex(VirtualInterfaceError, "signature"):
                evaluate_tile_operation(op, (tile,) * arity)

    def test_integer_unary_tables_do_not_require_the_host_arithmetic_probe(self) -> None:
        from npu_model.configs.numerics import RtlNumerics

        # Unary tables bypass the host arithmetic probe.
        with patch.object(RtlNumerics, "add", side_effect=AssertionError("integer tables need no FP probe")):
            for source, cases in ((EXP2, EXP2_CASES), (RECIP, RECIP_CASES)):
                tile, expected = (repeated([case[i] for case in cases]) for i in range(2))
                with self.subTest(source=source):
                    self.assertEqual(evaluate_tile_operation(operation(source), (tile,)), expected)

    def test_pack_literals_and_asymmetric_logical_row_major_layout(self) -> None:
        cases = PACK_CASES + ((0x7FC1, 0x00), (0xFFC1, 0x00))
        # Unequal row/column strides cross both halves of every register pair.
        indices = tuple((row * 7 + col * 11) % len(cases) for row in range(32) for col in range(32))
        packed = evaluate_tile_operation(operation(PACK), (Tile("bf16", tuple(cases[i][0] for i in indices)),))
        self.assertEqual(packed, Tile("fp8", tuple(cases[i][1] for i in indices)))

    def test_mixed_stream_publishes_independent_original_mov_add_relu_outputs(self) -> None:
        source = "module {" + INPUTS + "\n".join((MOV, ADD, RELU.replace("%m", "%r").replace("(%a)", "(%sum)"))) + "\n"
        states = ("s2", "out0", "out1", "out2", "out3")
        source += "\n".join(output(states[index], states[index + 1], value, index) for index, value in enumerate(("a", "m", "sum", "r"))) + "\n"
        left, right, summed = (repeated([case[i] for case in ADD_CASES]) for i in range(3))
        outputs = evaluate(parse_program(source + "}"), RuntimeInputs({0: left, 1: right})).outputs
        self.assertEqual(dict(outputs), {0: left, 1: left, 2: summed, 3: relu(summed)})
        with self.assertRaises(TypeError):
            outputs[0] = right

    def test_helper_rejects_malformed_operands_and_revalidates_admission(self) -> None:
        tile, fp8 = repeated((0,)), repeated((0,), "fp8")
        for source, operands in ((MOV, ()), (MOV, (tile, tile)), (ADD, (tile,)), (MOV, (fp8,)), (ADD, (tile, fp8)), (PACK, (fp8,)),
                                 (MOV, (None,)), (MOV, [tile]), (MOV, None)):
            with self.subTest(source=source, operands=operands), self.assertRaises(VirtualInterfaceError):
                evaluate_tile_operation(operation(source), operands)
        for source, name, attribute in ((MOV, "kind", builtin.StringAttr("sqrt")), (ADD, "kind", builtin.StringAttr("div")),
                                         (PACK, "scale_code", builtin.IntegerAttr(128, builtin.i32))):
            op = operation(source)
            op.attributes[name] = attribute
            with self.subTest(attribute=attribute), self.assertRaises(UnsupportedVirtualMode):
                evaluate_tile_operation(op, (tile, tile) if source == ADD else (tile,))
        for mutation in ("attribute", "signature", "scale_type", "result_type"):
            op = operation(PACK)
            if mutation == "attribute":
                op.attributes["invented"] = builtin.IntegerAttr(0, builtin.i32)
            elif mutation == "signature":
                op.operands = ()
            elif mutation == "scale_type":
                op.attributes["scale_code"] = builtin.IntegerAttr(127, builtin.i64)
            else:
                op.results[0]._type = builtin.i32
            with self.subTest(mutation=mutation), self.assertRaises(VirtualInterfaceError):
                evaluate_tile_operation(op, (tile,))
        for op in parse_program("module {" + INPUTS + "}").operations:
            with self.assertRaises(UnsupportedVirtualMode):
                evaluate_tile_operation(op, ())
        op = operation(MOV)
        op.attributes["op_name__"] = builtin.StringAttr("atlas.virtual_invented")
        with self.assertRaises(UnsupportedVirtualMode):
            evaluate_tile_operation(op, (tile,))
        with self.assertRaises(VirtualInterfaceError):
            evaluate_tile_operation(None, (tile,))

    def run_isolated(self, script: str, source: str = ADD) -> None:
        prelude = f'''
            import sys
            sys.path.insert(0, "tools")
            from atlas_virtual_evaluator import Tile, UnsupportedVirtualMode, evaluate_tile_operation, parse_program
            op = parse_program({"module {" + INPUTS + source + "}"!r}).operations[-1]
            tile = Tile("bf16", (0x3f80,) * 1024)
            operands = (tile,) * len(op.operands)
        '''
        child = subprocess.run([sys.executable, "-c", textwrap.dedent(prelude) + textwrap.dedent(script)], cwd=ROOT, capture_output=True, text=True, timeout=60)
        if child.returncode == 77:
            self.skipTest("host cannot select the required floating-point mode")
        self.assertEqual(child.returncode, 0, child.stdout + child.stderr)

    def test_fp32_operations_reject_host_flush_to_zero_in_an_isolated_process(self) -> None:
        script = '''
            import torch
            if not torch.set_flush_denormal(True):
                sys.exit(77)
            try:
                evaluate_tile_operation(op, operands)
            except UnsupportedVirtualMode:
                sys.exit(0)
            raise AssertionError("operation accepted a host that flushes subnormals")
        '''
        for source in (ADD, SUB, MUL, REDUCE):
            with self.subTest(source=source):
                self.run_isolated(script, source)

    def test_fp32_operations_reject_directed_rounding_in_an_isolated_process(self) -> None:
        if platform.machine() not in ("x86_64", "AMD64", "i386", "i686") or platform.libc_ver()[0] != "glibc":
            self.skipTest("requires the verified x86 glibc fenv ABI")
        # FE_DOWNWARD/UPWARD/TOWARDZERO verified in /usr/include/bits/fenv.h.
        script = '''
            import ctypes
            try:
                setround = ctypes.CDLL(None).fesetround
            except AttributeError:
                sys.exit(77)
            setround.argtypes = (ctypes.c_int,)
            setround.restype = ctypes.c_int
            evaluate_tile_operation(op, operands)
            for mode in (0x400, 0x800, 0xC00):
                if setround(mode) != 0:
                    sys.exit(77)
                try:
                    evaluate_tile_operation(op, operands)
                except UnsupportedVirtualMode:
                    continue
                raise AssertionError(f"operation accepted directed rounding mode {mode:#x}")
        '''
        for source in (ADD, SUB, MUL, REDUCE):
            with self.subTest(source=source):
                self.run_isolated(script, source)


ZERO8, ZERO16 = Tile("fp8", (0,) * 1024), Tile("bf16", (0,) * 1024)
LEGACY = kernel((legacy("y"),), "s4", ("y",))
RESET = kernel((weight("s4", "t0", "w0"), reset("t0", "t1", "a0"), read("t1", "t2", "y", "a0")), "t2", ("y",))
SEEDED = kernel((weight("s4", "t0", "w0"), seed("t0", "t1", "a0"), accumulate("t1", "t2", "a1", "a0"), read("t2", "t3", "y", "a1")), "t3", ("y",))
COPY = kernel((seed("s4", "t0", "a0"), read("t0", "t1", "y", "a0")), "t1", ("y",))
FP8_COPY = kernel((seed("s4", "t0", "a0", "x", "fp8"), read("t0", "t1", "y", "a0")), "t1", ("y",))


def readout_fp8(code: int) -> str:
    """Reseed the FP8 readout to observe its decoded bits through BF16 boundary I/O."""
    return kernel((seed("s4", "t0", "a0"), f'%scale = "atlas.virtual_scale_constant"() {{code = {code} : i32}} : () -> !atlas.virtual_scale',
                   f'%t1, %packed = "atlas.virtual_mxu_readout_fp8"(%t0, %a0, %scale) : ({S}, {A}, !atlas.virtual_scale) -> ({S}, !atlas.virtual_fp8)',
                   seed("t1", "t2", "a1", "packed", "fp8"), read("t2", "t3", "y", "a1")), "t3", ("y",))


def sparse(values: dict[tuple[int, int], int], format="fp8") -> Tile:
    return Tile(format, tuple(values.get((row, col), 0) for row in range(32) for col in range(32)))


def leading(format: str, codes: tuple[int, ...]) -> Tile:
    return Tile(format, codes + (0,) * (1024 - len(codes)))


def mxu_tiles(x: Tile = ZERO8, w: Tile = ZERO8, initial: Tile = ZERO16) -> dict[int, Tile]:
    return {0: x, 1: w, 2: ZERO8, 3: initial}


def rtl_arithmetic_cases():
    import npu_model.configs.numerics as numerics
    data = (Path(numerics.__file__).resolve().parents[2] / "tests/rtl/arithmetic.json").read_bytes()
    if hashlib.sha256(data).hexdigest() != "8b2ebb3cb98b4c9e7c7c0f4cb71e86d8a6675c698abec0c3e0525bc693736840":
        raise AssertionError("pinned MXU RTL arithmetic fixture changed")
    return tuple(json.loads(data))


class VirtualEvaluatorMxuTest(unittest.TestCase):
    def assert_outputs(self, source, tiles: dict[int, Tile], expected: dict[int, Tile]) -> None:
        result = evaluate(parse_program(source) if isinstance(source, str) else source, RuntimeInputs(tiles))
        self.assertEqual((dict(result.outputs), result.memory), (expected, ()))

    def test_legacy_and_reset_use_n_by_k_weights_and_fresh_inputs(self) -> None:
        x = sparse({(0, 0): 0x38, (0, 1): 0x40, (1, 0): 0x44, (1, 1): 0x48})
        w = sparse({(0, 0): 0x38, (0, 1): 0x44, (1, 0): 0x40, (1, 1): 0x48})
        expected = sparse({(0, 0): 0x40E0, (0, 1): 0x4120, (1, 0): 0x4170, (1, 1): 0x41B0}, "bf16")
        # Weight rows become output columns.
        acts, weights = sparse({(2, 3): 0x38, (2, 7): 0x40}), sparse({(5, 3): 0x44, (20, 7): 0x48})
        columns = sparse({(2, 5): 0x4040, (2, 20): 0x4100}, "bf16")
        for unit in (0, 1):
            for name, source in (("legacy", LEGACY), ("reset", RESET)):
                program = parse_program(on_unit(source, unit))
                for tiles, wanted in ((mxu_tiles(x, w), expected), (mxu_tiles(w=w), ZERO16), (mxu_tiles(x, w), expected), (mxu_tiles(acts, weights), columns)):
                    with self.subTest(unit=unit, program=name):
                        self.assert_outputs(program, tiles, {0: wanted})

    def test_rounding_order_and_anchor_truncation_literals(self) -> None:
        cases = (
            # Products 1, 1/256, 1/256: MXU0's half-ULP steps tie back to 1, while MXU1 rounds once.
            ("two_half_ulp_terms", (0x38, 0x18, 0x18), (0x38, 0x18, 0x18), 0x3F80, 0x3F81),
            ("half_ulp_terms_first", (0x18, 0x18, 0x38), (0x18, 0x18, 0x38), 0x3F81, 0x3F81),
            ("above_midpoint", (0x38, 0x18, 0x08), (0x38, 0x18, 0x08), 0x3F80, 0x3F81),
            # +/-448^2 cancel. With anchor 24, the 2^-12 term shifts by 36 and vanishes from the 32-bit IPT integer.
            ("anchor_discards_residual", (0x7E, 0xFE, 0x08), (0x7E, 0x7E, 0x08), 0x3980, 0x0000),
            # IPT is unchanged by product permutation; MXU0's K order matters.
            ("residual_before_cancellation", (0x08, 0x7E, 0xFE), (0x08, 0x7E, 0x7E), 0x0000, 0x0000),
        )
        for name, acts, weights, *bits in cases:
            for unit in (0, 1):
                for program, source in (("legacy", LEGACY), ("reset", RESET)):
                    with self.subTest(case=name, unit=unit, program=program):
                        self.assert_outputs(on_unit(source, unit), {**TILES, 0: leading("fp8", acts), 1: leading("fp8", weights)}, {0: leading("bf16", (bits[unit],))})

    def test_instruction_boundary_rounding_has_no_persistent_anchor(self) -> None:
        # First tile is 1+1/256 -> 1. Continuing with 1/256 ties back to 1; exact state across instructions would give 3f81.
        tiles = {**TILES, 0: leading("fp8", (0x38, 0x18)), 1: leading("fp8", (0x38, 0x18)), 2: leading("fp8", (0x18,))}
        second = accumulate("t3", "t4", "a1", "a0", "w1").replace("%x,", "%v,")
        body = (legacy("first"), weight("s4", "t0", "w0"), reset("t0", "t1", "a0"), weight("t1", "t3", "w1", "v"), second,
                read("t4", "t5", "continued", "a1"), legacy("second_reset").replace("%x, %w", "%v, %v"))
        source = on_unit(kernel(body, "t5", ("first", "continued", "second_reset")), 1)
        self.assert_outputs(source, tiles, {0: leading("bf16", (0x3F80,)), 1: leading("bf16", (0x3F80,)), 2: leading("bf16", (0x3B80,))})

    def test_continuation_rounds_per_mac_on_mxu0_and_once_per_instruction_on_mxu1(self) -> None:
        tiles = {**TILES, 0: leading("fp8", (0x38,)), 1: leading("fp8", (0x38,)), 2: leading("fp8", (0x18, 0x18))}
        second = accumulate("t2", "t3", "a1", "a0", "w1").replace("%x,", "%v,")
        body = (weight("s4", "t0", "w0"), reset("t0", "t1", "a0"), weight("t1", "t2", "w1", "v"), second, read("t3", "t4", "y", "a1"))
        small = sparse({(0, 0): 0x18, (0, 1): 0x18})
        for unit, bits in ((0, 0x3F80), (1, 0x3F81)):
            with self.subTest(unit=unit):
                self.assert_outputs(on_unit(kernel(body, "t4", ("y",)), unit), tiles, {0: leading("bf16", (bits,))})
                self.assert_outputs(on_unit(SEEDED, unit), mxu_tiles(small, small, leading("bf16", (0x3F80,))), {0: leading("bf16", (bits,))})

    def test_reset_and_continuation_store_bf16_and_reuse_resident_weights(self) -> None:
        body = (weight("s4", "t0", "w0"), reset("t0", "t1", "first"), read("t1", "t2", "thirty_two", "first"),
                reset("t2", "t3", "second"), accumulate("t3", "t4", "continued", "second"), read("t4", "t5", "sixty_four", "continued"),
                reset("t5", "t6", "replacement"), read("t6", "t7", "reset_again", "replacement"))
        ones = repeated((0x38,), "fp8")
        for unit in (0, 1):
            with self.subTest(unit=unit):
                self.assert_outputs(on_unit(kernel(body, "t7", ("thirty_two", "sixty_four", "reset_again")), unit), mxu_tiles(ones, ones),
                                    {0: repeated((0x4200,)), 1: repeated((0x4280,)), 2: repeated((0x4200,))})

    def test_zero_products_keep_signed_zero_seeds_on_mxu0_only(self) -> None:
        zero_codes = repeated(tuple(range(8)) + tuple(range(0x80, 0x88)), "fp8")
        # MXU0 reset starts from +0 even with negative weights; a seeded continuation keeps each seed's sign.
        body = (weight("s4", "t0", "w0"), reset("t0", "t1", "reset"), read("t1", "t2", "reset_result", "reset"),
                seed("t2", "t3", "acc"), accumulate("t3", "t4", "continued", "acc"), read("t4", "t5", "seed_result", "continued"))
        signed = repeated((0x0000, 0x8000))
        self.assert_outputs(kernel(body, "t5", ("reset_result", "seed_result")), mxu_tiles(zero_codes, repeated((0xB8,), "fp8"), signed), {0: ZERO16, 1: signed})
        negative_zero = Tile("bf16", (0x8000,) * 1024)
        tiles = {**TILES, 0: zero_codes, 1: Tile("fp8", (0x38,) * 1024), 3: negative_zero}
        body = (on_unit(seed("s4", "t0", "copy"), 1), on_unit(read("t0", "t1", "raw", "copy"), 1),
                weight("t1", "t2", "sw"), seed("t2", "t3", "sa_acc0"), accumulate("t3", "t4", "sa_acc1", "sa_acc0", "sw"), read("t4", "t5", "sa", "sa_acc1"),
                on_unit(weight("t5", "t6", "iw"), 1), on_unit(seed("t6", "t7", "i0"), 1),
                on_unit(accumulate("t7", "t8", "i1", "i0", "iw"), 1), on_unit(read("t8", "t9", "ipt", "i1"), 1))
        self.assert_outputs(kernel(body, "t9", ("raw", "sa", "ipt")), tiles, {0: negative_zero, 1: negative_zero, 2: ZERO16})

    def test_raw_fp8_products_flush_exponent_zero_but_reserved_codes_multiply_as_480(self) -> None:
        codes = tuple(range(8)) + tuple(range(0x80, 0x88)) + (0x7F, 0xFF)
        acts = sparse({(row, 0): code for row, code in enumerate(codes)})
        weights = sparse({(col, 0): code for col, code in enumerate((0x38, 0xB8, 0x7F, 0xFF))})
        # +/-480 * (+1, -1, +480, -480), including exact +/-230400.
        wanted = {(16, 0): 0x43F0, (16, 1): 0xC3F0, (16, 2): 0x4861, (16, 3): 0xC861,
                  (17, 0): 0xC3F0, (17, 1): 0x43F0, (17, 2): 0xC861, (17, 3): 0x4861}
        for unit in (0, 1):
            for name, source in (("legacy", LEGACY), ("reset", RESET)):
                for x, w, expected in ((acts, weights, wanted), (weights, acts, {(col, row): bits for (row, col), bits in wanted.items()})):
                    with self.subTest(unit=unit, program=name, swapped=x is weights):
                        self.assert_outputs(on_unit(source, unit), mxu_tiles(x, w), {0: sparse(expected, "bf16")})

    def test_seeds_copy_raw_bf16_and_decode_fp8_specials_to_signed_zero(self) -> None:
        raw = (0x00, 0x80, 0x01, 0x81, 0x07, 0x87, 0x7F, 0xFF, 0x08, 0x18, 0x38, 0xB8, 0x7E, 0xFE)
        decoded = (0, 0x8000, 0, 0x8000, 0, 0x8000, 0, 0x8000, 0x3C80, 0x3D80, 0x3F80, 0xBF80, 0x43E0, 0xC3E0)
        for unit in (0, 1):
            copy = parse_program(on_unit(COPY, unit))
            for first in range(0, 65536, 1024):
                with self.subTest(unit=unit, first=hex(first)):
                    initial = Tile("bf16", tuple(range(first, first + 1024)))
                    self.assert_outputs(copy, mxu_tiles(initial=initial), {0: initial})
            with self.subTest(unit=unit, seed="fp8"):
                self.assert_outputs(on_unit(FP8_COPY, unit), mxu_tiles(repeated(raw, "fp8")), {0: repeated(decoded)})

    def test_fp8_readout_uses_mxu_scaling_and_reserved_480_encoding(self) -> None:
        # Rounded +/-480 produces reserved 7f/ff, which reseeding decodes to +/-0.
        cases = (
            (126, (0x3F80, 0xBF80), (0x3F00, 0xBF00)),
            (127, (0x3F80, 0xBF80, 0x3F88, 0x3F98, 0x43E8, 0x43E9, 0x43F0, 0xC3E9, 0xC3F0, 0x7F80, 0xFF80, 0xFFA1, 0x0001, 0x8000),
                  (0x3F80, 0xBF80, 0x3F80, 0x3FA0, 0x43E0, 0, 0, 0x8000, 0x8000, 0x43E0, 0xC3E0, 0, 0, 0)),
            (128, (0x3F80, 0xBF80, 0x0000), (0x4000, 0xC000, 0x0000)),
            (0, (0x3F80, 0x7F7F, 0xFF7F, 0x7F80, 0xFF80), (0, 0x4000, 0xC000, 0x43E0, 0xC3E0)),
            (254, (0x0080, 0x8080, 0x0001, 0x8001, 0x7FC1, 0xFFC1, 0x8000), (0x4000, 0xC000, 0, 0, 0, 0, 0)),
            (255, (0x0080, 0x8080, 0x3F80), (0x4000, 0xC000, 0x43E0)),
        )
        for unit in (0, 1):
            for code, raw, expected in cases:
                with self.subTest(unit=unit, code=code):
                    self.assert_outputs(on_unit(readout_fp8(code), unit), mxu_tiles(initial=repeated(raw)), {0: repeated(expected)})

    def test_all_pinned_rtl_mxu_converter_cases_cover_every_scale(self) -> None:
        cases = rtl_arithmetic_cases()
        self.assertEqual({case["scale"] for case in cases}, set(range(256)))
        for code in range(256):
            batch = tuple(case for case in cases if case["scale"] == code)
            with self.subTest(scale=code):
                # Inspect the raw adapter result: reseeding would erase reserved 7f/ff encodings.
                result = _mxu_tile("readout_fp8", (repeated(tuple(case["partial"] for case in batch)),), code)
                self.assertEqual(result, repeated(tuple(case["quant"] for case in batch), "fp8"))

    def test_exceptional_bf16_seeds_follow_each_units_zero_and_nonzero_product_paths(self) -> None:
        raw = (0x0001, 0x007F, 0x8001, 0x807F, 0x7F80, 0xFF80, 0x7FC1, 0xFFA1, 0x7FFF, 0xFFFF)
        initial = sparse({(row, 0): code for row, code in enumerate(raw)}, "bf16")
        weights = sparse({(0, 0): 0x38})
        large = (0x7F7F, 0xFF7F, 0x7F7F, 0xFF7F, 0x7F7F, 0xFF7F)
        # A zero product bypasses MXU0 even for Inf/NaN. MXU1 sanitizes subnormal seeds to +0 and clamps exponent-255
        # seeds by sign. Any nonzero product replaces an exponent-zero addend; exponent-255 addends clamp finite.
        bypass = (initial, sparse({(row + 4, 0): code for row, code in enumerate(large)}, "bf16"))
        for unit in (0, 1):
            program = parse_program(on_unit(SEEDED, unit))
            with self.subTest(unit=unit, activation=None):
                self.assert_outputs(program, mxu_tiles(w=weights, initial=initial), {0: bypass[unit]})
            for act, product in ((0x38, 0x3F80), (0xB8, 0xBF80)):
                with self.subTest(unit=unit, activation=hex(act)):
                    acts = sparse({(row, 0): act for row in range(len(raw))})
                    wanted = sparse({(row, 0): code for row, code in enumerate((product,) * 4 + large)}, "bf16")
                    self.assert_outputs(program, mxu_tiles(acts, weights, initial), {0: wanted})

    def test_all_pinned_rtl_mac_and_anchor_cases_including_raw_bf16_seeds(self) -> None:
        cases = rtl_arithmetic_cases()
        # MXU0 cases are single MACs, MXU1 cases full anchor dot products. Unrelated cases share a batch on the
        # diagonal; off-diagonal outputs are intentionally unconstrained.
        for unit, key, depth in ((0, "fma", 1), (1, "ipt", None)):
            program = parse_program(on_unit(SEEDED, unit))
            for first in range(0, len(cases), 32):
                batch = cases[first:first + 32]
                acts = sparse({(row, k): code for row, case in enumerate(batch) for k, code in enumerate(case["a"][:depth])})
                weights = sparse({(row, k): code for row, case in enumerate(batch) for k, code in enumerate(case["w"][:depth])})
                initial = sparse({(row, row): case["partial"] for row, case in enumerate(batch)}, "bf16")
                result = evaluate(program, RuntimeInputs(mxu_tiles(acts, weights, initial))).outputs[0]
                for row, case in enumerate(batch):
                    with self.subTest(unit=unit, case=first + row, seed=hex(case["partial"])):
                        self.assertEqual(result.bits[row * 32 + row], case[key])

    def test_two_weights_and_accumulators_are_independent_on_each_unit(self) -> None:
        body = (
            weight("s4", "t0", "sw0"), weight("t0", "t1", "sw1", "v"),
            on_unit(weight("t1", "t2", "iw0"), 1), on_unit(weight("t2", "t3", "iw1", "v"), 1),
            reset("t3", "t4", "sa0", "sw0"), reset("t4", "t5", "sa1", "sw1"),
            on_unit(reset("t5", "t6", "ia0", "iw0"), 1), on_unit(reset("t6", "t7", "ia1", "iw1"), 1),
            on_unit(read("t7", "t8", "iy1", "ia1"), 1), read("t8", "t9", "sy1", "sa1"),
            on_unit(read("t9", "t10", "iy0", "ia0"), 1), read("t10", "t11", "sy0", "sa0"),
        )
        self.assert_outputs(kernel(body, "t11", ("iy1", "sy1", "iy0", "sy0")), TILES,
                            {0: diagonal("bf16", 0x4000), 1: diagonal("bf16", 0x4000), 2: diagonal("bf16", 0x3F80), 3: diagonal("bf16", 0x3F80)})
        # Legacy MXU0 work may overlap live MXU1 handles; same-unit overlap is rejected in the resource tests.
        body = (on_unit(weight("s4", "t0", "w0"), 1), on_unit(reset("t0", "t1", "a0"), 1), legacy("sa"), on_unit(read("t1", "t2", "ipt", "a0"), 1))
        self.assert_outputs(kernel(body, "t2", ("sa", "ipt")), TILES, {0: diagonal("bf16", 0x3F80), 1: diagonal("bf16", 0x3F80)})


if __name__ == "__main__":
    unittest.main()
