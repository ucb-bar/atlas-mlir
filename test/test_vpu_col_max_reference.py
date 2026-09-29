"""Bounded column-maximum check through the hand OOT and selected AtlasCore."""

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
SOURCE = ROOT / "test/examples/vpu_col_max_pair.mlir"
ASSEMBLY = ROOT / "test/examples/vpu_col_max_pair.S"


def _value(code: int) -> Fraction:
    exponent = (code >> 7) & 0xFF
    if not 1 <= exponent <= 254:
        raise ValueError(f"outside finite-normal BF16 test domain: {code:#06x}")
    significand = Fraction(128 + (code & 0x7F))
    scale = exponent - 134
    magnitude = significand * (Fraction(1 << scale) if scale >= 0
                               else Fraction(1, 1 << -scale))
    return -magnitude if code & 0x8000 else magnitude


def _panel(phase: int) -> tuple[bytes, bytes, bytes, list[int]]:
    # Each logical column has a distinct positive winner in a different row.
    # Phase zero makes the high register bank win the physical 64-row lane;
    # phase two makes the low bank win instead. All values are finite normal.
    rows = [[0xBF80 + ((row * 3 + col * 5 + phase) % 32)
             for col in range(32)] for row in range(32)]
    for col in range(32):
        winner = ((0x4000 + col if col < 16 else 0x4040 + col - 16)
                  if phase == 0 else
                  (0x4040 + col if col < 16 else 0x4000 + col - 16))
        rows[(col * 7 + phase) % 32][col] = winner
    maxima = [max((rows[row][col] for row in range(32)), key=_value)
              for col in range(32)]
    assert maxima == [(0x4000 + col if col < 16 else 0x4040 + col - 16)
                      if phase == 0 else
                      (0x4040 + col if col < 16 else 0x4000 + col - 16)
                      for col in range(32)]
    source = b"".join(struct.pack("<H", code)
                      for half in (0, 1) for row in rows
                      for code in row[half * 16:(half + 1) * 16])
    architectural = b"".join(struct.pack("<H", maxima[half * 16 + col])
                              for half in (0, 1) for _row in range(32)
                              for col in range(16))
    selected = [max((maxima[col], maxima[16 + col]), key=_value)
                for col in range(16)]
    expected = b"".join(struct.pack("<H", selected[col])
                        for _half in (0, 1) for _row in range(32)
                        for col in range(16))
    return source, expected, architectural, maxima


class VpuColMaxReferenceTest(unittest.TestCase):
    def test_independent_selected_column_max_and_architectural_discrepancy(self) -> None:
        self.assertLess(_value(0xBF80), _value(0x4000))
        self.assertGreater(_value(0x404F), _value(0x4000))
        for phase in (0, 2):
            source, expected, architectural, maxima = _panel(phase)
            self.assertEqual((len(source), len(expected), len(maxima)),
                             (2048, 2048, 32))
            self.assertNotEqual(source, expected)
            self.assertNotEqual(expected, architectural)
            selected = maxima[16:] if phase == 0 else maxima[:16]
            self.assertEqual(struct.unpack("<16H", expected[:32]), tuple(selected))
            for half in (0, 1):
                expected_row = struct.pack("<16H", *selected)
                for row in range(32):
                    base = half * 1024 + row * 32
                    self.assertEqual(expected[base:base + 32], expected_row)
            self.assertNotEqual(architectural[:32], architectural[1024:1056])

    def test_selected_assembler_and_llvm_object_words(self) -> None:
        root = os.environ.get("ATLAS_ASSEMBLER_ROOT")
        if not root:
            self.skipTest("set ATLAS_ASSEMBLER_ROOT for selected assembler")
        spec = importlib.util.spec_from_file_location(
            "selected_vpu_col_max_assembler", pathlib.Path(root) / "assembler.py")
        assert spec and spec.loader
        assembler = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(assembler)
        emitted = tuple(int(line, 16) for line in subprocess.check_output(
            [str(BIN / "atlas-emit"), str(SOURCE)], text=True).splitlines())
        self.assertEqual(len(emitted), 36)
        self.assertEqual(emitted, tuple(assembler.assemble(ASSEMBLY.read_text())))
        self.assertEqual(emitted[17], assembler.VREDMAX_BF16(4, 0))
        self.assertEqual(emitted[17] >> 25, 0x07)
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
        original = 'kind = "col_max", dst = 4 : i32, src = 0 : i32'
        for replacement in (
            'kind = "col_max", dst = 5 : i32, src = 0 : i32',
            'kind = "col_max", dst = 4 : i32, src = 63 : i32',
            'kind = "unknown", dst = 4 : i32, src = 0 : i32',
        ):
            with self.subTest(replacement=replacement):
                changed = source.replace(original, replacement)
                self.assertNotEqual(changed, source)
                run = subprocess.run([str(BIN / "atlas-opt"), "-"], input=changed,
                                     capture_output=True, text=True, check=False)
                self.assertNotEqual(run.returncode, 0)
                self.assertTrue(run.stderr)

    def test_selected_core_matches_64_row_maximum_and_exposes_layout_mismatch(self) -> None:
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
                        source, expected, architectural, maxima = _panel(phase)
                        self.assertNotEqual(expected, architectural)
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
                        self.assertNotEqual(observed, architectural)
                        self.assertEqual(result.slave.captured(0x90000000, 2048), source)
                        self.assertEqual(result.slave.captured(0x90001000, 32), b"\x5A" * 32)
            finally:
                cosim_atlas.CosimCore = original
        finally:
            os.chdir(old_cwd)
            sys.path.remove(str(modelir))


if __name__ == "__main__":
    unittest.main()
