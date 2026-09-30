"""Bounded selected-RTL BF16 column-sum check for the hand Atlas OOT dialect."""

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
SOURCE = ROOT / "test/examples/vpu_col_sum_pair.mlir"
ASSEMBLY = ROOT / "test/examples/vpu_col_sum_pair.S"


def _pow2(exponent: int) -> Fraction:
    return Fraction(1 << exponent) if exponent >= 0 else Fraction(1, 1 << -exponent)


def _decode_bf16(code: int) -> Fraction:
    if code == 0:
        return Fraction(0)
    exponent = (code >> 7) & 0xFF
    if not 1 <= exponent <= 254:
        raise ValueError(f"outside finite-normal/positive-zero test domain: {code:#06x}")
    significand = 128 + (code & 0x7F)
    value = significand * _pow2(exponent - 134)
    return -value if code & 0x8000 else value


def _floor_log2(value: Fraction) -> int:
    magnitude = abs(value)
    exponent = magnitude.numerator.bit_length() - magnitude.denominator.bit_length()
    return exponent - (magnitude < _pow2(exponent))


def _round_binary32(value: Fraction) -> Fraction:
    if value == 0:
        return Fraction(0)
    exponent = _floor_log2(value)
    if not -126 <= exponent <= 127:
        raise ValueError("outside normal binary32 test domain")
    quantum = _pow2(exponent - 23)
    scaled = abs(value) / quantum
    integer, remainder = divmod(scaled.numerator, scaled.denominator)
    rounded = integer + (2 * remainder > scaled.denominator or
                         (2 * remainder == scaled.denominator and integer & 1))
    return (-rounded if value < 0 else rounded) * quantum


def _binary32_bits(value: Fraction) -> int:
    if value == 0:
        return 0
    exponent = _floor_log2(value)
    significand = abs(value) / _pow2(exponent - 23)
    if significand.denominator != 1:
        raise AssertionError("binary32 reference has an unrepresentable value")
    return ((0x80000000 if value < 0 else 0) |
            ((exponent + 127) << 23) |
            (significand.numerator - (1 << 23)))


def _column(rows: list[list[int]], lane: int) -> int:
    if len(rows) != 64 or any(len(row) != 16 for row in rows):
        raise ValueError("selected physical column is 64 rows by 16 lanes")
    accumulator = Fraction(0)
    for row in rows:
        accumulator = _round_binary32(accumulator + _decode_bf16(row[lane]))
    # The RTL takes the upper 16 bits of the widened binary32 result.
    return _binary32_bits(accumulator) >> 16


def _panel(phase: int) -> tuple[bytes, bytes, tuple[int, ...]]:
    rows = [[0] * 16 for _ in range(64)]
    if phase == 0:
        rows[0][0], rows[1][0], rows[2][0] = 0x3F00, 0x3B00, 0x3A80
        rows[0][1], rows[1][1], rows[2][1] = 0x3F80, 0x3B80, 0x3B80
        rows[0][2] = 0x4780
        for row in range(1, 31):
            rows[row][2] = 0x3B80
        rows[31][2] = 0xC780
        rows[0][3], rows[32][3] = 0xBF80, 0x3F00
        rows[63][15] = 0x3F80
    elif phase == 1:
        for lane in range(16):
            rows[(3 * lane) % 64][lane] = 0x4000
            rows[(3 * lane + 31) % 64][lane] = 0xBF80
            rows[(3 * lane + 47) % 64][lane] = 0x3C00 if lane % 2 == 0 else 0x3B80
    else:
        raise ValueError("unknown directed panel")
    source = b"".join(struct.pack("<H", code) for row in rows for code in row)
    result = tuple(_column(rows, lane) for lane in range(16))
    expected = b"".join(struct.pack("<H", code)
                        for _row in range(64) for code in result)
    return source, expected, result


class VpuColSumReferenceTest(unittest.TestCase):
    def test_exact_widened_steps_final_chop_and_physical_layout(self) -> None:
        source, expected, result = _panel(0)
        self.assertEqual((len(source), len(expected)), (2048, 2048))
        self.assertEqual(result[:4], (0x3F00, 0x3F81, 0, 0xBF00))
        self.assertEqual(result[15], 0x3F80)
        self.assertEqual(_decode_bf16(0x3F00) + _decode_bf16(0x3B00) +
                         _decode_bf16(0x3A80), Fraction(515, 1024))
        self.assertEqual(_binary32_bits(_round_binary32(Fraction(1) +
                         _decode_bf16(0x3B80))), 0x3F808000)
        other_source, other_expected, other_result = _panel(1)
        self.assertEqual((len(other_source), len(other_expected)), (2048, 2048))
        self.assertEqual(other_result, tuple(0x3F81 if lane % 2 == 0 else 0x3F80
                                             for lane in range(16)))
        self.assertNotEqual(expected, other_expected)

    def test_selected_assembler_and_llvm_object_words(self) -> None:
        root = os.environ.get("ATLAS_ASSEMBLER_ROOT")
        if not root:
            self.skipTest("set ATLAS_ASSEMBLER_ROOT for selected assembler")
        spec = importlib.util.spec_from_file_location(
            "selected_vpu_col_sum_assembler", pathlib.Path(root) / "assembler.py")
        assert spec and spec.loader
        assembler = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(assembler)
        emitted = tuple(int(line, 16) for line in subprocess.check_output(
            [str(BIN / "atlas-emit"), str(SOURCE)], text=True).splitlines())
        self.assertEqual(len(emitted), 36)
        self.assertEqual(emitted, tuple(assembler.assemble(ASSEMBLY.read_text())))
        self.assertEqual(emitted[17], assembler.VREDSUM_BF16(4, 0))
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
        original = 'kind = "col_sum", dst = 4 : i32, src = 0 : i32'
        for replacement in (
            'kind = "col_sum", dst = 5 : i32, src = 0 : i32',
            'kind = "col_sum", dst = 4 : i32, src = 63 : i32',
            'kind = "unknown", dst = 4 : i32, src = 0 : i32',
        ):
            with self.subTest(replacement=replacement):
                changed = source.replace(original, replacement)
                self.assertNotEqual(changed, source)
                run = subprocess.run([str(BIN / "atlas-opt"), "-"], input=changed,
                                     capture_output=True, text=True, check=False)
                self.assertNotEqual(run.returncode, 0)
                self.assertTrue(run.stderr)

    def test_selected_core_matches_physical_column_and_preserves_input(self) -> None:
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
                for phase in (0, 1):
                    with self.subTest(phase=phase):
                        source, expected, _ = _panel(phase)
                        result = cosim_atlas.run_program(
                            model, state, _object_words(SOURCE),
                            preload=[(0x90000000, source),
                                     (0x90000800, b"\xA5" * 2048),
                                     (0x90001000, b"\x5A" * 32)],
                            max_cycles=8000,
                        )
                        self.assertTrue(result.halted)
                        self.assertEqual((result.reads, result.writes), (64, 64))
                        self.assertEqual(result.slave.captured(0x90000800, 2048), expected)
                        self.assertEqual(result.slave.captured(0x90000000, 2048), source)
                        self.assertEqual(result.slave.captured(0x90001000, 32), b"\x5A" * 32)
            finally:
                cosim_atlas.CosimCore = original
        finally:
            os.chdir(old_cwd)
            sys.path.remove(str(modelir))


if __name__ == "__main__":
    unittest.main()
