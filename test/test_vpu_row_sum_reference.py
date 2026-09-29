"""Independent bounded row-sum tree check for the hand Atlas OOT dialect."""

from __future__ import annotations

from fractions import Fraction
import importlib.util
import os
import pathlib
import struct
import subprocess
import sys
import unittest

from test_mxu_reference import _object_words

ROOT = pathlib.Path(__file__).resolve().parents[1]
BIN = pathlib.Path(os.environ.get("ATLAS_OOT_BIN_DIR", ROOT / "build/bin"))
SOURCE = ROOT / "test/examples/vpu_row_sum_pair.mlir"
ASSEMBLY = ROOT / "test/examples/vpu_row_sum_pair.S"


def _pow2(exponent: int) -> Fraction:
    return Fraction(1 << exponent) if exponent >= 0 else Fraction(1, 1 << -exponent)


def _decode(code: int) -> Fraction:
    if code == 0:
        return Fraction(0)
    exponent = (code >> 7) & 0xFF
    if not 1 <= exponent <= 254:
        raise ValueError(f"outside finite-normal/positive-zero domain: {code:#06x}")
    value = Fraction(128 + (code & 0x7F)) * _pow2(exponent - 134)
    return -value if code & 0x8000 else value


def _exponent(value: Fraction) -> int:
    magnitude = abs(value)
    exponent = magnitude.numerator.bit_length() - magnitude.denominator.bit_length()
    if magnitude < _pow2(exponent):
        exponent -= 1
    return exponent


def _round_nearest_even(value: Fraction, fractional_bits: int) -> Fraction:
    if value == 0:
        return Fraction(0)
    exponent = _exponent(value)
    if not -126 <= exponent <= 127:
        raise ValueError("outside normal binary32/BF16 result domain")
    step = _pow2(exponent - fractional_bits)
    scaled = abs(value) / step
    integer, remainder = divmod(scaled.numerator, scaled.denominator)
    twice = 2 * remainder
    rounded = integer + (twice > scaled.denominator or
                         (twice == scaled.denominator and integer % 2 == 1))
    return (-1 if value < 0 else 1) * rounded * step


def _encode_bf16(value: Fraction) -> int:
    rounded = _round_nearest_even(value, 7)
    if rounded == 0:
        return 0
    exponent = _exponent(rounded)
    significand = abs(rounded) / _pow2(exponent - 7)
    if significand.denominator != 1:
        raise AssertionError("BF16 reference did not land on a representable value")
    return ((0x8000 if rounded < 0 else 0) |
            ((exponent + 127) << 7) | (significand.numerator - 128))


def _tree_row(codes: list[int]) -> int:
    if len(codes) != 32:
        raise ValueError("selected row sum requires 32 lanes")
    values = [_decode(code) for code in codes]
    while len(values) > 1:
        values = [_round_nearest_even(values[i] + values[i + 1], 23)
                  for i in range(0, len(values), 2)]
    return _encode_bf16(values[0])


def _sequential_row(codes: list[int]) -> int:
    value = Fraction(0)
    for code in codes:
        value = _round_nearest_even(value + _decode(code), 23)
    return _encode_bf16(value)


def _panel(phase: int) -> tuple[bytes, bytes]:
    rows: list[list[int]] = []
    for row in range(32):
        if row % 8 == 0:
            codes = [0x4C00, 0x3F80, 0xCC00, 0x3F80, 0x4000] + [0] * 27
        elif row % 8 == 1:
            codes = [0x3F80] + [0] * 15 + [0x3BC0] + [0] * 15
        elif row % 8 == 2:
            codes = [0xBF80] + [0] * 15 + [0xBBC0] + [0] * 15
        else:
            codes = [(0x3D80, 0x3E80, 0x3F00, 0x3F80)
                     [(col * 3 + row + phase) % 4] for col in range(32)]
        rows.append(codes)
    input_halves = ([row[:16] for row in rows], [row[16:] for row in rows])
    source = b"".join(struct.pack("<H", code)
                      for half in input_halves for row in half for code in row)
    sums = [_tree_row(row) for row in rows]
    expected = b"".join(struct.pack("<H", value)
                        for _half in range(2) for value in sums for _col in range(16))
    return source, expected


class VpuRowSumReferenceTest(unittest.TestCase):
    def test_directed_tree_and_final_rounding(self) -> None:
        witness = [0x4C00, 0x3F80, 0xCC00, 0x3F80, 0x4000] + [0] * 27
        self.assertEqual(_tree_row(witness), 0x4000)  # adjacent tree gives 2
        self.assertEqual(_sequential_row(witness), 0x4040)  # serial gives 3
        self.assertEqual(_tree_row([0x3F80] + [0] * 15 + [0x3BC0] + [0] * 15),
                         0x3F81)  # final BF16 nearest-even, unlike VADD chop
        self.assertEqual(_tree_row([0xBF80] + [0] * 15 + [0xBBC0] + [0] * 15),
                         0xBF81)
        for phase in (0, 2):
            source, expected = _panel(phase)
            self.assertEqual((len(source), len(expected)), (2048, 2048))
            self.assertNotEqual(source, expected)

    def test_selected_assembler_and_llvm_object_words(self) -> None:
        root = os.environ.get("ATLAS_ASSEMBLER_ROOT")
        if not root:
            self.skipTest("set ATLAS_ASSEMBLER_ROOT for selected assembler")
        spec = importlib.util.spec_from_file_location(
            "selected_vpu_reduce_assembler", pathlib.Path(root) / "assembler.py")
        assert spec and spec.loader
        assembler = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(assembler)
        emitted = tuple(int(line, 16) for line in subprocess.check_output(
            [str(BIN / "atlas-emit"), str(SOURCE)], text=True).splitlines())
        self.assertEqual(len(emitted), 36)
        self.assertEqual(emitted, tuple(assembler.assemble(ASSEMBLY.read_text())))
        self.assertEqual(emitted[17], assembler.VREDSUM_ROW_BF16(4, 0))
        self.assertEqual(emitted, _object_words(SOURCE))
        printed = subprocess.run([str(BIN / "atlas-opt"), str(SOURCE)],
                                 capture_output=True, text=True, check=True)
        self.assertEqual(printed.stdout.count('"atlas.vpu_reduce"'), 1)
        reparsed = subprocess.run([str(BIN / "atlas-opt"), "-"],
                                  input=printed.stdout, capture_output=True,
                                  text=True, check=True)
        self.assertEqual(reparsed.stdout, printed.stdout)

    def test_invalid_pair_and_kind_are_rejected(self) -> None:
        source = SOURCE.read_text()
        original = 'kind = "row_sum", dst = 4 : i32, src = 0 : i32'
        for replacement in (
            'kind = "row_sum", dst = 5 : i32, src = 0 : i32',
            'kind = "row_sum", dst = 4 : i32, src = 63 : i32',
            'kind = "unknown", dst = 4 : i32, src = 0 : i32',
        ):
            with self.subTest(replacement=replacement):
                changed = source.replace(original, replacement)
                self.assertNotEqual(changed, source)
                run = subprocess.run([str(BIN / "atlas-opt"), "-"], input=changed,
                                     capture_output=True, text=True, check=False)
                self.assertNotEqual(run.returncode, 0)
                self.assertTrue(run.stderr)

    def test_selected_core_matches_tree_and_pair_broadcast(self) -> None:
        keys = ("ATLAS_ARC_MODEL", "ATLAS_ARC_STATE", "ATLAS_MODELIR_ROOT", "ATLAS_LLVM_BIN")
        if not all(os.environ.get(key) for key in keys):
            self.skipTest("set selected-source-linked ARC, ModeLIR, and LLVM paths")
        model, state, modelir = (pathlib.Path(os.environ[key]).resolve(strict=True)
                                 for key in keys[:3])
        old_cwd = pathlib.Path.cwd()
        sys.path.insert(0, str(modelir))
        try:
            os.chdir(modelir)
            from mlc.backends import cosim_atlas

            original = cosim_atlas.CosimCore

            class SelectedCore(original):
                def peek(self, name: str) -> int:
                    return super().peek("scalar/halt_now" if name == "io_halted" else name)

                def poke(self, name: str, value: int) -> None:
                    if name in ("io_dmaTL_d_bits_opcode", "io_dmaTL_d_bits_size"):
                        if name in self._S:
                            raise AssertionError(f"unexpected live ARC input {name}")
                        return
                    super().poke(name, value)

            cosim_atlas.CosimCore = SelectedCore
            try:
                for phase in (0, 2):
                    with self.subTest(phase=phase):
                        source, expected = _panel(phase)
                        result = cosim_atlas.run_program(
                            model, state, _object_words(SOURCE),
                            preload=[(0x90000000, source),
                                     (0x90000800, b"\xA5" * 2048),
                                     (0x90001000, b"\x5A" * 32)],
                            max_cycles=8000,
                        )
                        observed = result.slave.captured(0x90000800, 2048)
                        self.assertTrue(result.halted)
                        self.assertEqual((result.reads, result.writes), (64, 64))
                        self.assertEqual(observed, expected)
                        self.assertEqual(result.slave.captured(0x90000000, 2048), source)
                        self.assertEqual(result.slave.captured(0x90001000, 32), b"\x5A" * 32)
            finally:
                cosim_atlas.CosimCore = original
        finally:
            os.chdir(old_cwd)
            sys.path.remove(str(modelir))


if __name__ == "__main__":
    unittest.main()
