"""Static hand-OOT row-min qualification; selected-core run remains pending."""

from __future__ import annotations

from fractions import Fraction
import importlib.util
import os
import pathlib
import struct
import subprocess
import unittest

from test_mxu_reference import _object_words

ROOT = pathlib.Path(__file__).resolve().parents[1]
BIN = pathlib.Path(os.environ.get("ATLAS_OOT_BIN_DIR", ROOT / "build/bin"))
SOURCE = ROOT / "test/examples/vpu_row_min_pair.mlir"
ASSEMBLY = ROOT / "test/examples/vpu_row_min_pair.S"


def _value(code: int) -> Fraction:
    exponent = (code >> 7) & 0xFF
    if not 1 <= exponent <= 254:
        raise ValueError(f"outside finite-normal BF16 test domain: {code:#06x}")
    significand = Fraction(128 + (code & 0x7F))
    scale = exponent - 134
    magnitude = significand * (Fraction(1 << scale) if scale >= 0
                               else Fraction(1, 1 << -scale))
    return -magnitude if code & 0x8000 else magnitude


def _panel(phase: int) -> tuple[bytes, bytes, list[int]]:
    ordinary = (0x3F80, 0xBF80, 0x4000, 0xC000,
                0x3E80, 0xBE80, 0x4040, 0xC040)
    winners = (0xC080, 0xC100, 0xC180, 0xC200,
               0xC280, 0xC300, 0xC380, 0xC400)
    rows: list[list[int]] = []
    minima: list[int] = []
    for row in range(32):
        codes = [ordinary[(row * 3 + col * 5 + phase) % len(ordinary)]
                 for col in range(32)]
        winner_at = (row * 7 + phase) % 32
        codes[winner_at] = winners[row % len(winners)]
        minimum = min(codes, key=_value)
        assert minimum == codes[winner_at]
        rows.append(codes)
        minima.append(minimum)
    source = b"".join(struct.pack("<H", code)
                      for half in (0, 1) for row in rows
                      for code in row[half * 16:(half + 1) * 16])
    expected = b"".join(struct.pack("<H", minimum)
                        for half in (0, 1) for minimum in minima
                        for _lane in range(16))
    return source, expected, minima


class VpuRowMinReferenceTest(unittest.TestCase):
    def test_independent_finite_min_and_pair_layout(self) -> None:
        self.assertLess(_value(0xC100), _value(0xBF80))
        self.assertLess(_value(0xBE80), _value(0x3E80))
        for phase in (0, 2):
            source, expected, minima = _panel(phase)
            self.assertEqual((len(source), len(expected), len(minima)),
                             (2048, 2048, 32))
            self.assertNotEqual(source, expected)
            self.assertEqual(minima[0], 0xC080)
            self.assertEqual(minima[1], 0xC100)
            for row, minimum in enumerate(minima):
                first = expected[32 * row:32 * (row + 1)]
                second = expected[1024 + 32 * row:1024 + 32 * (row + 1)]
                self.assertEqual(first, struct.pack("<H", minimum) * 16)
                self.assertEqual(second, first)
            # A wrong per-half reduction must differ when the unique minimum
            # resides in the second half of a logical row.
            rows = [struct.unpack("<16H", source[32 * row:32 * (row + 1)])
                    for row in range(32)]
            first_half_only = [min(row, key=_value) for row in rows]
            self.assertTrue(any(a != b for a, b in zip(first_half_only, minima)))

    def test_selected_assembler_and_llvm_object_words(self) -> None:
        root = os.environ.get("ATLAS_ASSEMBLER_ROOT")
        if not root:
            self.skipTest("set ATLAS_ASSEMBLER_ROOT for selected assembler")
        spec = importlib.util.spec_from_file_location(
            "selected_vpu_row_min_assembler", pathlib.Path(root) / "assembler.py")
        assert spec and spec.loader
        assembler = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(assembler)
        emitted = tuple(int(line, 16) for line in subprocess.check_output(
            [str(BIN / "atlas-emit"), str(SOURCE)], text=True).splitlines())
        self.assertEqual(len(emitted), 36)
        self.assertEqual(emitted, tuple(assembler.assemble(ASSEMBLY.read_text())))
        self.assertEqual(emitted[17], assembler.VREDMIN_ROW_BF16(4, 0))
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
        original = 'kind = "row_min", dst = 4 : i32, src = 0 : i32'
        for replacement in (
            'kind = "row_min", dst = 5 : i32, src = 0 : i32',
            'kind = "row_min", dst = 4 : i32, src = 63 : i32',
            'kind = "unknown", dst = 4 : i32, src = 0 : i32',
        ):
            with self.subTest(replacement=replacement):
                changed = source.replace(original, replacement)
                self.assertNotEqual(changed, source)
                run = subprocess.run([str(BIN / "atlas-opt"), "-"], input=changed,
                                     capture_output=True, text=True, check=False)
                self.assertNotEqual(run.returncode, 0)
                self.assertTrue(run.stderr)


if __name__ == "__main__":
    unittest.main()
