"""Raw-bit VPU semantics and pure-tile helper admission, without Atlas tools."""

from __future__ import annotations

from dataclasses import FrozenInstanceError
from pathlib import Path
import platform
import subprocess
import sys
import textwrap
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
from atlas_virtual_evaluator import RuntimeInputs, Tile, UnsupportedVirtualMode, VirtualInterfaceError, evaluate, evaluate_tile_operation, parse_program  # noqa: E402
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


if __name__ == "__main__":
    unittest.main()
