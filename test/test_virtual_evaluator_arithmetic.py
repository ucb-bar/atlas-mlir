"""Raw-bit VPU, MXU0 and MXU1 arithmetic with independent literals and no Atlas tools.

VPU literals distinguish FP32 nearest-even plus BF16 chopping from BF16 rounding. MXU expectations are raw
literals from the selected FMA/anchor sources, exact rational witnesses and the pinned RTL arithmetic fixture,
not calls to model numerics or compiler lowering.
"""

from __future__ import annotations

from dataclasses import FrozenInstanceError
import hashlib
import json
from pathlib import Path
import platform
import subprocess
import sys
import textwrap
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
from atlas_virtual_evaluator import (  # noqa: E402
    RuntimeInputs, Scalar, Tile, UnsupportedVirtualMode, VirtualInterfaceError, _mxu_tile, evaluate, evaluate_tile_operation, parse_program,
)
from test_virtual_evaluator_resources import (  # noqa: E402
    A, S, TILES, accumulate, diagonal, function, legacy, on_unit, output, program, read, reset, seed, weight,
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
PACK = '%packed = "atlas.virtual_pack_fp8"(%a) {scale_code = 127 : i32} : (!atlas.virtual_bf16) -> !atlas.virtual_fp8'

# These literals distinguish FP32 nearest-even plus BF16 chopping from BF16
# rounding, and preserve gradual underflow, signed zeros, and canonical NaNs.
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
PACK_CASES = (
    (0x0000, 0x00), (0x8000, 0x00), (0x0001, 0x00), (0x807F, 0x00),
    (0x7F81, 0x00), (0x7FC1, 0x00), (0xFF81, 0x00), (0xFFC1, 0x00),
    (0x7F80, 0x7E), (0xFF80, 0xFE), (0x3F80, 0x38), (0xBF80, 0xB8),
    (0x3F87, 0x38), (0x3F88, 0x38), (0x3F89, 0x39),
    (0x3F98, 0x3A), (0xBF98, 0xBA), (0x3FF7, 0x3F),
    (0x3FF8, 0x40), (0xBFF8, 0xC0), (0x3C60, 0x00), (0x3C77, 0x00),
    (0x3C78, 0x08), (0x3C80, 0x08), (0xBC77, 0x00), (0xBC78, 0x88),
    (0x43E0, 0x7E), (0x43E8, 0x7E), (0x43E9, 0x7E),
    (0x43F0, 0x7E), (0x7F7F, 0x7E), (0xC3E0, 0xFE),
    (0xC3E8, 0xFE), (0xC3E9, 0xFE), (0xC3F0, 0xFE), (0xFF7F, 0xFE),
)


def operation(source: str):
    return parse_program("module {" + INPUTS + source + "}").operations[-1]


def repeated(values, format="bf16") -> Tile:
    return Tile(format, tuple(values[i % len(values)] for i in range(1024)))


class VirtualEvaluatorVpuTest(unittest.TestCase):
    def test_mov_and_relu_exhaust_every_raw_bf16_encoding(self) -> None:
        mov, relu = operation(MOV), operation(RELU)
        for first in range(0, 65536, 1024):
            with self.subTest(first=hex(first)):
                tile = Tile("bf16", tuple(range(first, first + 1024)))
                self.assertEqual(evaluate_tile_operation(mov, (tile,)), tile)
                self.assertEqual(evaluate_tile_operation(relu, (tile,)).bits,
                                 tuple(0 if value & 0x8000 else value for value in tile.bits))
                self.assertEqual(tile.bits, tuple(range(first, first + 1024)))

    def test_add_matches_independent_raw_bit_literals(self) -> None:
        left, right, expected = (repeated([case[i] for case in ADD_CASES]) for i in range(3))
        self.assertEqual(evaluate_tile_operation(operation(ADD), (left, right)), expected)
        self.assertEqual(left.bits, repeated([case[0] for case in ADD_CASES]).bits)
        self.assertEqual(right.bits, repeated([case[1] for case in ADD_CASES]).bits)

    def test_pack_literals_and_asymmetric_logical_row_major_layout(self) -> None:
        # Unequal row/column strides cross both halves of every register pair.
        indices = tuple((row * 7 + col * 11) % len(PACK_CASES) for row in range(32) for col in range(32))
        original = tuple(PACK_CASES[i][0] for i in indices)
        tile = Tile("bf16", original)
        packed = evaluate_tile_operation(operation(PACK), (tile,))
        self.assertEqual(packed, Tile("fp8", tuple(PACK_CASES[i][1] for i in indices)))
        self.assertEqual(tile.bits, original)
        with self.assertRaises(FrozenInstanceError):
            packed.bits = (0,) * 1024
        # VPU pack saturates 480 to 448 (0x7e); MXU rounding to 480 emits
        # reserved encoding 0x7f, so their converters are not interchangeable.
        self.assertEqual(evaluate_tile_operation(operation(PACK), (repeated((0x43F0,)),)).bits, (0x7E,) * 1024)

    def test_mixed_stream_publishes_independent_original_mov_add_relu_outputs(self) -> None:
        source = "module {" + INPUTS + "\n".join((MOV, ADD, RELU.replace("%m", "%r").replace("(%a)", "(%sum)"))) + "\n"
        state = "%s2"
        for index, value in enumerate(("%a", "%m", "%sum", "%r")):
            next_state = f"%out{index}"
            source += f'{next_state} = "atlas.virtual_output_bf16"({state}, {value}) {{index = {index} : i32}} : (!atlas.virtual_state, !atlas.virtual_bf16) -> !atlas.virtual_state\n'
            state = next_state
        left, right, summed = (repeated([case[i] for case in ADD_CASES]) for i in range(3))
        inputs = RuntimeInputs({0: left, 1: right})
        outputs = evaluate(parse_program(source + "}"), inputs).outputs
        self.assertEqual(dict(outputs), {0: left, 1: left, 2: summed, 3: Tile("bf16", tuple(0 if v & 0x8000 else v for v in summed.bits))})
        self.assertEqual(dict(inputs.tiles), {0: left, 1: right})
        with self.assertRaises(TypeError):
            outputs[0] = right

    def test_fp8_boundary_and_unused_pack_execute_with_no_fp8_output_boundary(self) -> None:
        # Until MXU/DMA support, the helper supplies the observable FP8 result.
        source = "module {" + INPUTS + PACK + '''
          %s3, %fp8 = "atlas.virtual_input_fp8"(%s2) {index = 2 : i32} : (!atlas.virtual_state) -> (!atlas.virtual_state, !atlas.virtual_fp8)
          %s4 = "atlas.virtual_output_bf16"(%s3, %a) {index = 4 : i32} : (!atlas.virtual_state, !atlas.virtual_bf16) -> !atlas.virtual_state
        }'''
        program = parse_program(source)
        bf16, fp8 = repeated((0x43F0,)), repeated((0x7F, 0xFF, 0x80, 0x01), "fp8")
        inputs = RuntimeInputs({0: bf16, 1: bf16, 2: fp8})
        self.assertEqual(dict(evaluate(program, inputs).outputs), {4: bf16})
        self.assertEqual(inputs.tiles[2], fp8)
        for bad in (RuntimeInputs({0: bf16, 1: bf16}), RuntimeInputs({0: bf16, 1: bf16, 2: bf16})):
            with self.assertRaises(VirtualInterfaceError):
                evaluate(program, bad)
        with self.assertRaises(VirtualInterfaceError):
            evaluate(parse_program(source.replace('(%s3, %a)', '(%s2, %a)')), inputs)

    def test_helper_rejects_malformed_runtime_operands(self) -> None:
        tile, fp8 = repeated((0,)), repeated((0,), "fp8")
        for source, operands in ((MOV, ()), (MOV, (tile, tile)), (ADD, (tile,)),
                                 (MOV, (fp8,)), (ADD, (tile, fp8)), (PACK, (fp8,)),
                                 (MOV, (None,)), (MOV, [tile]), (MOV, None)):
            with self.subTest(source=source, operands=operands):
                with self.assertRaises(VirtualInterfaceError):
                    evaluate_tile_operation(operation(source), operands)

    def test_helper_revalidates_operation_admission(self) -> None:
        tile = repeated((0,))
        for source, name, attribute in (
            (MOV, "kind", builtin.StringAttr("sqrt")),
            (ADD, "kind", builtin.StringAttr("mul")),
            (PACK, "scale_code", builtin.IntegerAttr(128, builtin.i32)),
        ):
            op = operation(source)
            op.attributes[name] = attribute
            with self.assertRaises(UnsupportedVirtualMode):
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

    def test_add_rejects_host_flush_to_zero_in_an_isolated_process(self) -> None:
        source = "module {" + INPUTS + ADD + "}"
        script = textwrap.dedent(f'''\
            import sys
            sys.path.insert(0, "tools")
            import torch
            from atlas_virtual_evaluator import Tile, UnsupportedVirtualMode, evaluate_tile_operation, parse_program
            if not torch.set_flush_denormal(True):
                sys.exit(77)
            op = parse_program({source!r}).operations[-1]
            tile = Tile("bf16", (0x3f80,) * 1024)
            try:
                evaluate_tile_operation(op, (tile, tile))
            except UnsupportedVirtualMode:
                sys.exit(0)
            raise AssertionError("normal add accepted a host that flushes subnormals")
        ''')
        child = subprocess.run([sys.executable, "-c", script], cwd=ROOT, capture_output=True, text=True, timeout=60)
        if child.returncode == 77:
            self.skipTest("host does not support enabling flush to zero")
        self.assertEqual(child.returncode, 0, child.stdout + child.stderr)

    def test_add_rejects_directed_rounding_in_an_isolated_process(self) -> None:
        if platform.machine() not in ("x86_64", "AMD64", "i386", "i686") or platform.libc_ver()[0] != "glibc":
            self.skipTest("requires the verified x86 glibc fenv ABI")
        # FE_DOWNWARD/UPWARD/TOWARDZERO verified in /usr/include/bits/fenv.h.
        modes = (0x400, 0x800, 0xC00)
        source = "module {" + INPUTS + ADD + "}"
        script = textwrap.dedent(f'''\
            import ctypes
            import sys
            sys.path.insert(0, "tools")
            from atlas_virtual_evaluator import Tile, UnsupportedVirtualMode, evaluate_tile_operation, parse_program
            try:
                setround = ctypes.CDLL(None).fesetround
            except AttributeError:
                sys.exit(77)
            setround.argtypes = (ctypes.c_int,)
            setround.restype = ctypes.c_int
            op = parse_program({source!r}).operations[-1]
            tile = Tile("bf16", (0x3f80,) * 1024)
            evaluate_tile_operation(op, (tile, tile))
            for mode in {modes!r}:
                if setround(mode) != 0:
                    sys.exit(77)
                try:
                    evaluate_tile_operation(op, (tile, tile))
                except UnsupportedVirtualMode:
                    continue
                raise AssertionError(f"normal add accepted directed rounding mode {{mode:#x}}")
        ''')
        child = subprocess.run([sys.executable, "-c", script], cwd=ROOT, capture_output=True, text=True, timeout=60)
        if child.returncode == 77:
            self.skipTest("host does not expose the required fenv rounding control")
        self.assertEqual(child.returncode, 0, child.stdout + child.stderr)


STATE = "!atlas.virtual_state"
FP8 = "!atlas.virtual_fp8"
BF16 = "!atlas.virtual_bf16"
WEIGHT = "!atlas.virtual_mxu_weight<0>"
ACC = "!atlas.virtual_mxu_acc<0>"
SCALE = "!atlas.virtual_scale"

PREFIX = f'''
  %s0 = "atlas.virtual_start"() : () -> {STATE}
  %s1, %x = "atlas.virtual_input_fp8"(%s0) {{index = 0 : i32}} : ({STATE}) -> ({STATE}, {FP8})
  %s2, %w = "atlas.virtual_input_fp8"(%s1) {{index = 1 : i32}} : ({STATE}) -> ({STATE}, {FP8})
  %s3, %seed = "atlas.virtual_input_bf16"(%s2) {{index = 2 : i32}} : ({STATE}) -> ({STATE}, {BF16})
'''


def sparse(values: dict[tuple[int, int], int], format="fp8") -> Tile:
    return Tile(format, tuple(values.get((row, col), 0) for row in range(32) for col in range(32)))


def inputs(x=None, w=None, seed=None, *, controls=()) -> RuntimeInputs:
    return RuntimeInputs({0: x if x is not None else repeated((0,), "fp8"),
                          1: w if w is not None else repeated((0,), "fp8"),
                          2: seed if seed is not None else repeated((0,))}, controls)


class Stream:
    """Small textual fixture builder; expected values never use this builder."""

    def __init__(self):
        self.lines = [PREFIX]
        self.state = "%s3"
        self.count = 0

    def effect(self, name, args, types, result, result_type, attributes=""):
        self.count += 1
        next_state = f"%event{self.count}"
        self.lines.append(f'{next_state}, %{result} = "atlas.virtual_{name}"({self.state}, {args}) '
                          f'{attributes} : ({STATE}, {types}) -> ({STATE}, {result_type})')
        self.state = next_state

    def weight(self):
        self.effect("mxu_load_weight", "%w", FP8, "weight", WEIGHT, "{unit = 0 : i32}")

    def reset(self, result):
        self.effect("mxu_reset", "%x, %weight", f"{FP8}, {WEIGHT}", result, ACC)

    def accumulate(self, previous, result):
        self.effect("mxu_accumulate", f"%x, %weight, %{previous}", f"{FP8}, {WEIGHT}, {ACC}", result, ACC)

    def seed(self, source="seed", format="bf16", result="acc"):
        type = BF16 if format == "bf16" else FP8
        self.effect(f"mxu_load_acc_{format}", f"%{source}", type, result, ACC, "{unit = 0 : i32}")

    def readout(self, accumulator, result):
        self.effect("mxu_readout_bf16", f"%{accumulator}", ACC, result, BF16)

    def fp8_readout(self, code):
        self.lines.append(f'%scale = "atlas.virtual_scale_constant"() {{code = {code} : i32}} : () -> {SCALE}')
        self.effect("mxu_readout_fp8", "%acc, %scale", f"{ACC}, {SCALE}", "packed", FP8)
        self.seed("packed", "fp8", "decoded_acc")
        self.readout("decoded_acc", "decoded")

    def output(self, value, index=0):
        self.count += 1
        next_state = f"%event{self.count}"
        self.lines.append(f'{next_state} = "atlas.virtual_output_bf16"({self.state}, %{value}) '
                          f'{{index = {index} : i32}} : ({STATE}, {BF16}) -> {STATE}')
        self.state = next_state

    def program(self):
        return parse_program("module {" + "\n".join(self.lines) + "}")


def legacy_program():
    stream = Stream()
    stream.lines.append(f'%result = "atlas.virtual_mxu_matmul"(%x, %w) {{unit = 0 : i32}} '
                        f': ({FP8}, {FP8}) -> {BF16}')
    stream.output("result")
    return stream.program()


def reset_program():
    stream = Stream()
    stream.weight()
    stream.reset("acc")
    stream.readout("acc", "result")
    stream.output("result")
    return stream.program()


def rtl_arithmetic_cases():
    import npu_model.configs.numerics as numerics
    path = Path(numerics.__file__).resolve().parents[2] / "tests/rtl/arithmetic.json"
    data = path.read_bytes()
    if hashlib.sha256(data).hexdigest() != "8b2ebb3cb98b4c9e7c7c0f4cb71e86d8a6675c698abec0c3e0525bc693736840":
        raise AssertionError("pinned MXU RTL arithmetic fixture changed")
    cases = json.loads(data)
    return tuple(cases)


class VirtualEvaluatorMxuTest(unittest.TestCase):
    def test_legacy_and_explicit_reset_use_n_by_k_weights_and_fresh_inputs(self):
        x = sparse({(0, 0): 0x38, (0, 1): 0x40, (1, 0): 0x44, (1, 1): 0x48})
        w = sparse({(0, 0): 0x38, (0, 1): 0x44, (1, 0): 0x40, (1, 1): 0x48})
        expected = sparse({(0, 0): 0x40E0, (0, 1): 0x4120, (1, 0): 0x4170, (1, 1): 0x41B0}, "bf16")
        original = inputs(x, w)
        for program in (legacy_program(), reset_program()):
            with self.subTest(explicit=len(program.operations) > 6):
                self.assertEqual(evaluate(program, original).outputs[0], expected)
                self.assertEqual(evaluate(program, inputs(w=w)).outputs[0], repeated((0,)))
                self.assertEqual(evaluate(program, original).outputs[0], expected)
                self.assertEqual(dict(original.tiles), {0: x, 1: w, 2: repeated((0,))})

    def test_each_mac_rounds_bf16_in_increasing_k_order_and_from_the_seed(self):
        # Products 1, 1/256, 1/256: both half-ULP steps tie back to 1.
        forward = sparse({(0, 0): 0x38, (0, 1): 0x18, (0, 2): 0x18})
        reverse = sparse({(0, 0): 0x18, (0, 1): 0x18, (0, 2): 0x38})
        for program in (legacy_program(), reset_program()):
            self.assertEqual(evaluate(program, inputs(forward, forward)).outputs[0], sparse({(0, 0): 0x3F80}, "bf16"))
            self.assertEqual(evaluate(program, inputs(reverse, reverse)).outputs[0], sparse({(0, 0): 0x3F81}, "bf16"))
        small = sparse({(0, 0): 0x18, (0, 1): 0x18})
        seed = sparse({(0, 0): 0x3F80}, "bf16")
        stream = Stream()
        stream.weight()
        stream.seed()
        stream.accumulate("acc", "continued")
        stream.readout("continued", "result")
        stream.output("result")
        self.assertEqual(evaluate(stream.program(), inputs(small, small, seed)).outputs[0], seed)

    def test_reset_and_continuation_store_bf16_and_reuse_resident_weights(self):
        stream = Stream()
        stream.weight()
        stream.reset("first")
        stream.readout("first", "thirty_two")
        stream.output("thirty_two", 0)
        stream.reset("second")
        stream.accumulate("second", "continued")
        stream.readout("continued", "sixty_four")
        stream.output("sixty_four", 1)
        stream.reset("replacement")
        stream.readout("replacement", "reset_again")
        stream.output("reset_again", 2)
        ones = repeated((0x38,), "fp8")
        self.assertEqual(dict(evaluate(stream.program(), inputs(ones, ones)).outputs),
                         {0: repeated((0x4200,)), 1: repeated((0x4280,)), 2: repeated((0x4200,))})

    def test_zero_products_preserve_signed_seed_but_reset_starts_positive_zero(self):
        stream = Stream()
        stream.weight()
        stream.reset("reset")
        stream.readout("reset", "reset_result")
        stream.output("reset_result", 0)
        stream.seed()
        stream.accumulate("acc", "continued")
        stream.readout("continued", "seed_result")
        stream.output("seed_result", 1)
        seed = repeated((0x0000, 0x8000))
        zero = repeated(tuple(range(8)) + tuple(range(0x80, 0x88)), "fp8")
        self.assertEqual(dict(evaluate(stream.program(), inputs(zero, repeated((0xB8,), "fp8"), seed)).outputs),
                         {0: repeated((0,)), 1: seed})

    def test_raw_fp8_products_flush_exponent_zero_but_reserved_codes_multiply_as_480(self):
        codes = tuple(range(8)) + tuple(range(0x80, 0x88)) + (0x7F, 0xFF)
        x = sparse({(row, 0): code for row, code in enumerate(codes)})
        w = sparse({(col, 0): code for col, code in enumerate((0x38, 0xB8, 0x7F, 0xFF))})
        # +/-480 * (+1, -1, +480, -480), including exact +/-230400.
        wanted = {(16, 0): 0x43F0, (16, 1): 0xC3F0, (16, 2): 0x4861, (16, 3): 0xC861,
                  (17, 0): 0xC3F0, (17, 1): 0x43F0, (17, 2): 0xC861, (17, 3): 0x4861}
        for program in (legacy_program(), reset_program()):
            for acts, weights, expected in ((x, w, wanted), (w, x, {(col, row): bits for (row, col), bits in wanted.items()})):
                with self.subTest(explicit=len(program.operations) > 6, swapped=acts is w):
                    runtime = inputs(acts, weights)
                    self.assertEqual(evaluate(program, runtime).outputs[0], sparse(expected, "bf16"))
                    self.assertEqual(dict(runtime.tiles), dict(inputs(acts, weights).tiles))

    def test_bf16_seed_and_readout_preserve_every_raw_encoding(self):
        stream = Stream()
        stream.seed()
        stream.readout("acc", "result")
        stream.output("result")
        program = stream.program()
        for first in range(0, 65536, 1024):
            with self.subTest(first=hex(first)):
                seed = Tile("bf16", tuple(range(first, first + 1024)))
                runtime = inputs(seed=seed)
                self.assertEqual(evaluate(program, runtime).outputs[0], seed)
                self.assertEqual(runtime.tiles[2], seed)

    def test_fp8_seed_decodes_specials_to_signed_zero_without_a_scale(self):
        raw = (0x00, 0x80, 0x01, 0x81, 0x07, 0x87, 0x7F, 0xFF, 0x08, 0x18, 0x38, 0xB8, 0x7E, 0xFE)
        expected = (0, 0x8000, 0, 0x8000, 0, 0x8000, 0, 0x8000, 0x3C80, 0x3D80, 0x3F80, 0xBF80, 0x43E0, 0xC3E0)
        stream = Stream()
        stream.seed("x", "fp8")
        stream.readout("acc", "result")
        stream.output("result")
        runtime = inputs(x=repeated(raw, "fp8"))
        self.assertEqual(evaluate(stream.program(), runtime).outputs[0], repeated(expected))
        self.assertEqual(runtime.tiles[0], repeated(raw, "fp8"))

    def test_fp8_readout_uses_mxu_scaling_and_reserved_480_encoding(self):
        # Expected values are BF16 after the FP8 result is seeded back. In
        # particular rounded +/-480 produces reserved 7f/ff, decoded to +/-0.
        cases = (
            (126, (0x3F80, 0xBF80), (0x3F00, 0xBF00)),
            (127, (0x3F80, 0xBF80, 0x3F88, 0x3F98, 0x43E8, 0x43E9, 0x43F0, 0xC3E9, 0xC3F0),
                  (0x3F80, 0xBF80, 0x3F80, 0x3FA0, 0x43E0, 0, 0, 0x8000, 0x8000)),
            (128, (0x3F80, 0xBF80), (0x4000, 0xC000)),
            (0, (0x3F80, 0x7F7F, 0xFF7F, 0x7F80, 0xFF80), (0, 0x4000, 0xC000, 0x43E0, 0xC3E0)),
            (254, (0x0080, 0x8080, 0x0001, 0x8001, 0x7FC1, 0xFFC1, 0x8000), (0x4000, 0xC000, 0, 0, 0, 0, 0)),
            (255, (0x0080, 0x8080, 0x3F80), (0x4000, 0xC000, 0x43E0)),
        )
        for code, raw, expected in cases:
            with self.subTest(code=code):
                stream = Stream()
                stream.seed()
                stream.fp8_readout(code)
                stream.output("decoded")
                seed = repeated(raw)
                runtime = inputs(seed=seed)
                self.assertEqual(evaluate(stream.program(), runtime).outputs[0], repeated(expected))
                self.assertEqual(runtime.tiles[2], seed)

    def test_all_pinned_rtl_mxu_converter_cases_cover_every_scale(self):
        cases = rtl_arithmetic_cases()
        self.assertEqual({case["scale"] for case in cases}, set(range(256)))
        for code in range(256):
            batch = tuple(case for case in cases if case["scale"] == code)
            raw = tuple(case["partial"] for case in batch)
            wanted = tuple(case["quant"] for case in batch)
            with self.subTest(scale=code):
                # Inspect the raw adapter result: reseeding would erase reserved
                # 7f/ff encodings and could hide an incorrect converter result.
                result = _mxu_tile("readout_fp8", (repeated(raw),), code)
                self.assertEqual(result, repeated(wanted, "fp8"))

    def test_exceptional_bf16_seeds_follow_custom_mac_zero_and_nonzero_paths(self):
        stream = Stream()
        stream.weight()
        stream.seed()
        stream.accumulate("acc", "continued")
        stream.readout("continued", "result")
        stream.output("result")
        program = stream.program()
        raw = (0x0001, 0x007F, 0x8001, 0x807F, 0x7F80, 0xFF80, 0x7FC1, 0xFFA1, 0x7FFF, 0xFFFF)
        seed = sparse({(row, 0): code for row, code in enumerate(raw)}, "bf16")
        weights = sparse({(0, 0): 0x38})
        # A zero product bypasses even Inf/NaN unchanged. Any nonzero product
        # replaces an exponent-zero addend; exponent-255 addends clamp finite.
        self.assertEqual(evaluate(program, inputs(w=weights, seed=seed)).outputs[0], seed)
        for act, product in ((0x38, 0x3F80), (0xB8, 0xBF80)):
            with self.subTest(activation=hex(act)):
                expected = (product,) * 4 + (0x7F7F, 0xFF7F, 0x7F7F, 0xFF7F, 0x7F7F, 0xFF7F)
                acts = sparse({(row, 0): act for row in range(len(raw))})
                wanted = sparse({(row, 0): code for row, code in enumerate(expected)}, "bf16")
                runtime = inputs(acts, weights, seed)
                self.assertEqual(evaluate(program, runtime).outputs[0], wanted)
                self.assertEqual(runtime.tiles[2], seed)

    def test_all_pinned_rtl_single_mac_cases_including_raw_bf16_seeds(self):
        stream = Stream()
        stream.weight()
        stream.seed()
        stream.accumulate("acc", "continued")
        stream.readout("continued", "result")
        stream.output("result")
        program = stream.program()
        cases = rtl_arithmetic_cases()
        # Batch unrelated harness cases on the diagonal; off-diagonal outputs
        # are intentionally unconstrained and do not provide expected values.
        for first in range(0, len(cases), 32):
            batch = cases[first:first + 32]
            acts = sparse({(row, 0): case["a"][0] for row, case in enumerate(batch)})
            weights = sparse({(row, 0): case["w"][0] for row, case in enumerate(batch)})
            seed = sparse({(row, row): case["partial"] for row, case in enumerate(batch)}, "bf16")
            runtime = inputs(acts, weights, seed)
            result = evaluate(program, runtime).outputs[0]
            for row, case in enumerate(batch):
                with self.subTest(case=first + row, seed=hex(case["partial"])):
                    self.assertEqual(result.bits[row * 32 + row], case["fma"])
            self.assertEqual(runtime.tiles[2], seed)

    def test_cfg_executes_only_selected_raw_seed_copy_or_contraction(self):
        source = f'''module {{
          func.func @choose(%take: i1) -> {STATE} {{
            {PREFIX}
            cf.cond_br %take, ^compute(%s3 : {STATE}), ^copy(%s3 : {STATE})
          ^compute(%cs: {STATE}):
            %c1, %weight = "atlas.virtual_mxu_load_weight"(%cs, %w) {{unit = 0 : i32}} : ({STATE}, {FP8}) -> ({STATE}, {WEIGHT})
            %c2, %acc = "atlas.virtual_mxu_load_acc_bf16"(%c1, %seed) {{unit = 0 : i32}} : ({STATE}, {BF16}) -> ({STATE}, {ACC})
            %c3, %continued = "atlas.virtual_mxu_accumulate"(%c2, %x, %weight, %acc) : ({STATE}, {FP8}, {WEIGHT}, {ACC}) -> ({STATE}, {ACC})
            %c4, %result = "atlas.virtual_mxu_readout_bf16"(%c3, %continued) : ({STATE}, {ACC}) -> ({STATE}, {BF16})
            %out = "atlas.virtual_output_bf16"(%c4, %result) {{index = 0 : i32}} : ({STATE}, {BF16}) -> {STATE}
            return %out : {STATE}
          ^copy(%ss: {STATE}):
            %copy_out = "atlas.virtual_output_bf16"(%ss, %seed) {{index = 1 : i32}} : ({STATE}, {BF16}) -> {STATE}
            return %copy_out : {STATE}
          }}
        }}'''
        program = parse_program(source)
        acts = repeated((0x7F,), "fp8")
        seed = repeated((0x7FC1, 0x8001))
        self.assertEqual(dict(evaluate(program, inputs(acts, seed=seed, controls=(Scalar(1, 0),))).outputs), {1: seed})
        # Default weights are zero: the custom MAC's bypass preserves raw seeds.
        self.assertEqual(dict(evaluate(program, inputs(acts, seed=seed, controls=(Scalar(1, 1),))).outputs), {0: seed})


def leading(format: str, codes: tuple[int, ...]) -> Tile:
    return Tile(format, codes + (0,) * (1024 - len(codes)))


def kernel(body: tuple[str, ...], final_state: str, results: tuple[str, ...]) -> str:
    operations = list(body)
    current = final_state
    for index, result in enumerate(results):
        after = f"out{index}"
        operations.append(output(current, after, result, index))
        current = after
    operations.append(f"func.return %{current} : {S}")
    return function("\n".join(operations))


class VirtualEvaluatorMXU1Test(unittest.TestCase):
    def assert_outputs(self, source: str, tiles: dict[int, Tile], expected: dict[int, Tile]) -> None:
        inputs = RuntimeInputs(tiles)
        result = evaluate(parse_program(source), inputs)
        self.assertEqual(dict(result.outputs), expected)
        self.assertEqual(dict(inputs.tiles), tiles)
        self.assertEqual(result.memory, inputs.memory)

    def reject(self, source: str, tiles: dict[int, Tile], error=VirtualInterfaceError) -> None:
        with self.assertRaises(error):
            evaluate(parse_program(source), RuntimeInputs(tiles))

    def test_reset_literals_distinguish_rounding_and_anchor_truncation(self) -> None:
        cases = (
            ("two_half_ulp_terms", (0x38, 0x18, 0x18), (0x38, 0x18, 0x18), 0x3F80, 0x3F81),
            ("above_midpoint", (0x38, 0x18, 0x08), (0x38, 0x18, 0x08), 0x3F80, 0x3F81),
            # +/-448^2 cancel. With anchor 24, the 2^-12 term shifts by
            # 36 and vanishes from the 32-bit IPT integer representation.
            ("anchor_discards_residual", (0x7E, 0xFE, 0x08), (0x7E, 0x7E, 0x08), 0x3980, 0x0000),
            # IPT is unchanged by product permutation; MXU0's K order matters.
            ("residual_before_cancellation", (0x08, 0x7E, 0xFE), (0x08, 0x7E, 0x7E), 0x0000, 0x0000),
        )
        source = kernel((legacy("sa"), on_unit(legacy("ipt"), 1)), "s4", ("sa", "ipt"))
        for name, acts, weights, sa_bits, ipt_bits in cases:
            with self.subTest(case=name):
                tiles = {**TILES, 0: leading("fp8", acts), 1: leading("fp8", weights)}
                self.assert_outputs(source, tiles, {0: leading("bf16", (sa_bits,)), 1: leading("bf16", (ipt_bits,))})

    def test_instruction_boundary_rounding_has_no_persistent_anchor(self) -> None:
        # First tile is 1+1/256 -> 1. Continuing with 1/256 ties back to 1;
        # retaining exact state across instructions would instead produce 3f81.
        tiles = {**TILES, 0: leading("fp8", (0x38, 0x18)), 1: leading("fp8", (0x38, 0x18)),
                 2: leading("fp8", (0x18,))}
        second = accumulate("t3", "t4", "a1", "a0", "w1").replace("%x,", "%v,")
        second_reset = legacy("second_reset").replace("%x, %w", "%v, %v")
        body = (legacy("first"), weight("s4", "t0", "w0"), reset("t0", "t1", "a0"),
                weight("t1", "t3", "w1", "v"), second, read("t4", "t5", "continued", "a1"), second_reset)
        source = on_unit(kernel(body, "t5", ("first", "continued", "second_reset")), 1)
        self.assert_outputs(source, tiles, {0: leading("bf16", (0x3F80,)), 1: leading("bf16", (0x3F80,)), 2: leading("bf16", (0x3B80,))})

    def test_continuation_rounds_once_per_mxu1_instruction(self) -> None:
        tiles = {**TILES, 0: leading("fp8", (0x38,)), 1: leading("fp8", (0x38,)),
                 2: leading("fp8", (0x18, 0x18))}
        second = accumulate("t2", "t3", "a1", "a0", "w1").replace("%x,", "%v,")
        body = (weight("s4", "t0", "w0"), reset("t0", "t1", "a0"), weight("t1", "t2", "w1", "v"),
                second, read("t3", "t4", "y", "a1"))
        for unit, bits in ((0, 0x3F80), (1, 0x3F81)):
            with self.subTest(unit=unit):
                self.assert_outputs(on_unit(kernel(body, "t4", ("y",)), unit), tiles, {0: leading("bf16", (bits,))})

    def test_both_units_use_weight_rows_as_output_columns(self) -> None:
        acts, weights, wanted = [0] * 1024, [0] * 1024, [0] * 1024
        acts[2 * 32 + 3], acts[2 * 32 + 7] = 0x38, 0x40
        weights[5 * 32 + 3], weights[20 * 32 + 7] = 0x44, 0x48
        wanted[2 * 32 + 5], wanted[2 * 32 + 20] = 0x4040, 0x4100
        expected = Tile("bf16", tuple(wanted))
        tiles = {**TILES, 0: Tile("fp8", tuple(acts)), 1: Tile("fp8", tuple(weights))}
        source = kernel((legacy("sa"), on_unit(legacy("ipt"), 1)), "s4", ("sa", "ipt"))
        self.assert_outputs(source, tiles, {0: expected, 1: expected})

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
                            {0: diagonal("bf16", 0x4000), 1: diagonal("bf16", 0x4000),
                             2: diagonal("bf16", 0x3F80), 3: diagonal("bf16", 0x3F80)})

    def test_legacy_other_unit_is_allowed_but_own_unit_overlap_is_rejected(self) -> None:
        body = (on_unit(weight("s4", "t0", "w0"), 1), on_unit(reset("t0", "t1", "a0"), 1),
                legacy("sa"), on_unit(read("t1", "t2", "ipt", "a0"), 1))
        self.assert_outputs(kernel(body, "t2", ("sa", "ipt")), TILES,
                            {0: diagonal("bf16", 0x3F80), 1: diagonal("bf16", 0x3F80)})
        for body, state in (
            ((weight("s4", "t0", "w0"), legacy(), reset("t0", "t1", "a0"), read("t1", "t2", "y", "a0")), "t2"),
            ((seed("s4", "t0", "a0"), legacy(), read("t0", "t1", "y", "a0")), "t1"),
        ):
            with self.subTest(state=state):
                self.reject(on_unit(program(*body, final_state=state), 1), TILES)

    def test_mxu1_stale_forked_and_repeated_readout_versions_are_rejected(self) -> None:
        prefix = (weight("s4", "t0", "w0"), reset("t0", "t1", "a0"), accumulate("t1", "t2", "a1", "a0"))
        cases = (
            (*prefix, read("t2", "t3", "old", "a0"), read("t3", "t4", "current", "a1")),
            (*prefix, accumulate("t2", "t3", "fork", "a0"), read("t3", "t4", "first", "a1"), read("t4", "t5", "second", "fork")),
            (seed("s4", "t0", "a0"), read("t0", "t1", "first", "a0"), read("t1", "t2", "again", "a0")),
        )
        for body, state in zip(cases, ("t4", "t5", "t2")):
            with self.subTest(state=state):
                self.reject(on_unit(program(*body, final_state=state), 1), TILES)

    def test_signed_zero_seed_copy_and_zero_product_continuation_differ(self) -> None:
        negative_zero = Tile("bf16", (0x8000,) * 1024)
        zero_codes = tuple(range(8)) + tuple(range(0x80, 0x88))
        tiles = {**TILES, 0: Tile("fp8", zero_codes * 64), 1: Tile("fp8", (0x38,) * 1024), 3: negative_zero}
        body = (on_unit(seed("s4", "t0", "copy"), 1), on_unit(read("t0", "t1", "raw", "copy"), 1),
                weight("t1", "t2", "sw"), seed("t2", "t3", "sa_acc0"), accumulate("t3", "t4", "sa_acc1", "sa_acc0", "sw"), read("t4", "t5", "sa", "sa_acc1"),
                on_unit(weight("t5", "t6", "iw"), 1), on_unit(seed("t6", "t7", "i0"), 1),
                on_unit(accumulate("t7", "t8", "i1", "i0", "iw"), 1), on_unit(read("t8", "t9", "ipt", "i1"), 1))
        self.assert_outputs(kernel(body, "t9", ("raw", "sa", "ipt")), tiles,
                            {0: negative_zero, 1: negative_zero, 2: Tile("bf16", (0,) * 1024)})

    def test_raw_seed_and_readout_preserve_their_conversion_rules(self) -> None:
        raw_bf16 = (0x8000, 0xFFA1, 0x7F80, 0x0001)
        raw_fp8 = (0x80, 0x81, 0xFF, 0x38, 0xB8, 0x7E, 0xFE, 0x7F)
        decoded = (0x8000, 0x8000, 0x8000, 0x3F80, 0xBF80, 0x43E0, 0xC3E0, 0x0000)
        tiles = {**TILES, 0: Tile("fp8", raw_fp8 * 128), 3: Tile("bf16", raw_bf16 * 256)}
        body = (seed("s4", "t0", "b0"), seed("t0", "t1", "f0", "x", "fp8"),
                read("t1", "t2", "fp8_seed", "f0"), read("t2", "t3", "bf16_seed", "b0"))
        self.assert_outputs(on_unit(kernel(body, "t3", ("fp8_seed", "bf16_seed")), 1), tiles,
                            {0: Tile("bf16", decoded * 128), 1: tiles[3]})

    def test_mxu1_scaled_fp8_readout_uses_the_shared_mxu_converter(self) -> None:
        # Seed the FP8 readout into a fresh accumulator to observe its decoded
        # bits through BF16 boundary I/O, without inventing an FP8 output op.
        source = f'''{seed("s4", "t0", "a0")}
          %scale = "atlas.virtual_scale_constant"() {{code = 127 : i32}} : () -> !atlas.virtual_scale
          %t1, %packed = "atlas.virtual_mxu_readout_fp8"(%t0, %a0, %scale) : ({S}, {A}, !atlas.virtual_scale) -> ({S}, !atlas.virtual_fp8)
          {seed("t1", "t2", "a1", "packed", "fp8")}
          {read("t2", "t3", "y", "a1")}
        '''
        raw = (0x43F0, 0xC3F0, 0x7F80, 0xFF80, 0xFFA1, 0x0001, 0x8000, 0x3F80)
        wanted = (0x0000, 0x8000, 0x43E0, 0xC3E0, 0x0000, 0x0000, 0x0000, 0x3F80)
        tiles = {**TILES, 3: Tile("bf16", raw * 128)}
        self.assert_outputs(on_unit(kernel((source,), "t3", ("y",)), 1), tiles, {0: Tile("bf16", wanted * 128)})
        tiles = {**TILES, 3: diagonal("bf16", 0x3F80)}
        scaled = on_unit(kernel((source.replace("code = 127", "code = 128"),), "t3", ("y",)), 1)
        self.assert_outputs(scaled, tiles, {0: diagonal("bf16", 0x4000)})

    def test_raw_fp8_products_use_multiplier_semantics_on_mxu1(self) -> None:
        codes = tuple(range(8)) + tuple(range(0x80, 0x88)) + (0x7F, 0xFF)
        acts = sparse({(row, 0): code for row, code in enumerate(codes)})
        weights = sparse({(col, 0): code for col, code in enumerate((0x38, 0xB8, 0x7F, 0xFF))})
        wanted = {(16, 0): 0x43F0, (16, 1): 0xC3F0, (16, 2): 0x4861, (16, 3): 0xC861,
                  (17, 0): 0xC3F0, (17, 1): 0x43F0, (17, 2): 0xC861, (17, 3): 0x4861}
        programs = (kernel((legacy("y"),), "s4", ("y",)),
                    kernel((weight("s4", "t0", "w0"), reset("t0", "t1", "a0"), read("t1", "t2", "y", "a0")), "t2", ("y",)))
        for index, program in enumerate(programs):
            source = on_unit(program, 1)
            for x, w, expected in ((acts, weights, wanted), (weights, acts, {(col, row): bits for (row, col), bits in wanted.items()})):
                with self.subTest(explicit=bool(index), swapped=x is weights):
                    self.assert_outputs(source, {**TILES, 0: x, 1: w}, {0: sparse(expected, "bf16")})

    def test_mxu1_exceptional_bf16_seeds_are_anchor_integers(self) -> None:
        continued = on_unit(kernel((weight("s4", "t0", "w0"), seed("t0", "t1", "a0"),
                                   accumulate("t1", "t2", "a1", "a0"), read("t2", "t3", "y", "a1")), "t3", ("y",)), 1)
        raw = (0x0001, 0x007F, 0x8001, 0x807F, 0x7F80, 0xFF80, 0x7FC1, 0xFFA1, 0x7FFF, 0xFFFF)
        initial = sparse({(row, 0): code for row, code in enumerate(raw)}, "bf16")
        weights = sparse({(0, 0): 0x38})
        # With zero products, subnormal seeds convert below the normal range
        # and are sanitized to +0. Raw exponent-255 seeds clamp by sign.
        large = (0x7F7F, 0xFF7F, 0x7F7F, 0xFF7F, 0x7F7F, 0xFF7F)
        wanted = sparse({(row + 4, 0): code for row, code in enumerate(large)}, "bf16")
        self.assert_outputs(continued, {**TILES, 0: sparse({}), 1: weights, 3: initial}, {0: wanted})
        for act, product in ((0x38, 0x3F80), (0xB8, 0xBF80)):
            with self.subTest(activation=hex(act)):
                acts = sparse({(row, 0): act for row in range(len(raw))})
                expected = (product,) * 4 + large
                wanted = sparse({(row, 0): code for row, code in enumerate(expected)}, "bf16")
                self.assert_outputs(continued, {**TILES, 0: acts, 1: weights, 3: initial}, {0: wanted})

    def test_all_pinned_rtl_anchor_cases_including_raw_bf16_seeds(self) -> None:
        body = (weight("s4", "t0", "w0"), seed("t0", "t1", "a0"),
                accumulate("t1", "t2", "a1", "a0"), read("t2", "t3", "y", "a1"))
        source = parse_program(on_unit(kernel(body, "t3", ("y",)), 1))
        cases = rtl_arithmetic_cases()
        for first in range(0, len(cases), 32):
            batch = cases[first:first + 32]
            acts = sparse({(row, k): code for row, case in enumerate(batch) for k, code in enumerate(case["a"])})
            weights = sparse({(row, k): code for row, case in enumerate(batch) for k, code in enumerate(case["w"])})
            initial = sparse({(row, row): case["partial"] for row, case in enumerate(batch)}, "bf16")
            runtime = RuntimeInputs({**TILES, 0: acts, 1: weights, 3: initial})
            result = evaluate(source, runtime).outputs[0]
            for row, case in enumerate(batch):
                with self.subTest(case=first + row, seed=hex(case["partial"])):
                    self.assertEqual(result.bits[row * 32 + row], case["ipt"])
            self.assertEqual(runtime.tiles[3], initial)


if __name__ == "__main__":
    unittest.main()
