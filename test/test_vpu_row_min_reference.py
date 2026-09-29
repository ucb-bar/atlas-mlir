"""Bounded row-min check through the hand OOT and selected AtlasCore."""

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

    def test_selected_core_matches_full_row_minimum_and_preserves_input(self) -> None:
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
                        source, expected, minima = _panel(phase)
                        first_only = [min(struct.unpack(
                            "<16H", source[32 * row:32 * (row + 1)]), key=_value)
                            for row in range(32)]
                        wrong = b"".join(struct.pack("<H", value)
                                         for _half in range(2) for value in first_only
                                         for _lane in range(16))
                        self.assertNotEqual(wrong, expected)
                        self.assertTrue(any(a != b for a, b in zip(first_only, minima)))
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
                        self.assertNotEqual(observed, wrong)
                        self.assertEqual(result.slave.captured(0x90000000, 2048), source)
                        self.assertEqual(result.slave.captured(0x90001000, 32), b"\x5A" * 32)
            finally:
                cosim_atlas.CosimCore = original
        finally:
            os.chdir(old_cwd)
            sys.path.remove(str(modelir))


if __name__ == "__main__":
    unittest.main()
