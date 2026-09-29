"""Bounded row-max check through the hand OOT and selected AtlasCore."""

from __future__ import annotations

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
SOURCE = ROOT / "test/examples/vpu_row_max_pair.mlir"
ASSEMBLY = ROOT / "test/examples/vpu_row_max_pair.S"


def _ordered(code: int) -> int:
    """Unsigned key for the selected RTL's raw BF16 total ordering."""
    if not 0 <= code <= 0xFFFF:
        raise ValueError("BF16 encoding is outside 16 bits")
    return (~code & 0xFFFF) if code & 0x8000 else code ^ 0x8000


def _panel(phase: int) -> tuple[bytes, bytes, list[int]]:
    # Every winner exceeds its row's ordinary values in the RTL bit order.
    # Cases 0-3 distinguish signed zero, positive NaN, negative NaN,
    # and infinity from a conventional floating-point maximum.
    cases = (
        ((0xBF80, 0xC000, 0x8000), 0x0000),
        ((0x3F80, 0x4000, 0x4040), 0x7F80),
        ((0x3F80, 0x4000, 0x7F80), 0x7FC1),
        ((0xFFC1, 0xFF80, 0xC040), 0xC000),
        ((0xC040, 0x0000, 0x3F80, 0x4000), 0x4080),
        ((0x3F00, 0x3F80, 0x4000, 0x4040), 0x4100),
        ((0xC100, 0xC080, 0xC040, 0xC000), 0xBF80),
        ((0x3F80, 0x4000, 0x4040, 0x4080), 0x4180),
    )
    rows: list[list[int]] = []
    maxima: list[int] = []
    for row in range(32):
        ordinary, winner = cases[row % len(cases)]
        codes = [ordinary[(row * 3 + col * 5 + phase) % len(ordinary)]
                 for col in range(32)]
        winner_at = (row * 7 + phase) % 32
        codes[winner_at] = winner
        maximum = max(codes, key=_ordered)
        assert maximum == codes[winner_at]
        rows.append(codes)
        maxima.append(maximum)
    source = b"".join(struct.pack("<H", code)
                      for half in (0, 1) for row in rows
                      for code in row[half * 16:(half + 1) * 16])
    expected = b"".join(struct.pack("<H", maximum)
                        for half in (0, 1) for maximum in maxima
                        for _lane in range(16))
    return source, expected, maxima


class VpuRowMaxReferenceTest(unittest.TestCase):
    def test_raw_order_and_pair_layout(self) -> None:
        self.assertGreater(_ordered(0x0000), _ordered(0x8000))
        self.assertGreater(_ordered(0x7FC1), _ordered(0x7F80))
        self.assertGreater(_ordered(0xC000), _ordered(0xFFC1))
        with self.assertRaises(ValueError):
            _ordered(0x10000)
        for phase in (0, 2):
            source, expected, maxima = _panel(phase)
            self.assertEqual((len(source), len(expected), len(maxima)),
                             (2048, 2048, 32))
            self.assertNotEqual(source, expected)
            self.assertEqual(maxima[:4], [0x0000, 0x7F80, 0x7FC1, 0xC000])
            for row, maximum in enumerate(maxima):
                first = expected[32 * row:32 * (row + 1)]
                second = expected[1024 + 32 * row:1024 + 32 * (row + 1)]
                self.assertEqual(first, struct.pack("<H", maximum) * 16)
                self.assertEqual(second, first)
            # A wrong per-half reduction must differ when the unique maximum
            # resides in the second half of a logical row.
            rows = [struct.unpack("<16H", source[32 * row:32 * (row + 1)])
                    for row in range(32)]
            first_half_only = [max(row, key=_ordered) for row in rows]
            self.assertTrue(any(a != b for a, b in zip(first_half_only, maxima)))

    def test_selected_assembler_and_llvm_object_words(self) -> None:
        root = os.environ.get("ATLAS_ASSEMBLER_ROOT")
        if not root:
            self.skipTest("set ATLAS_ASSEMBLER_ROOT for selected assembler")
        spec = importlib.util.spec_from_file_location(
            "selected_vpu_row_max_assembler", pathlib.Path(root) / "assembler.py")
        assert spec and spec.loader
        assembler = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(assembler)
        emitted = tuple(int(line, 16) for line in subprocess.check_output(
            [str(BIN / "atlas-emit"), str(SOURCE)], text=True).splitlines())
        self.assertEqual(len(emitted), 36)
        self.assertEqual(emitted, tuple(assembler.assemble(ASSEMBLY.read_text())))
        self.assertEqual(emitted[17], assembler.VREDMAX_ROW_BF16(4, 0))
        self.assertEqual(emitted[17] >> 25, 0x26)
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
        original = 'kind = "row_max", dst = 4 : i32, src = 0 : i32'
        for replacement in (
            'kind = "row_max", dst = 5 : i32, src = 0 : i32',
            'kind = "row_max", dst = 4 : i32, src = 63 : i32',
            'kind = "unknown", dst = 4 : i32, src = 0 : i32',
        ):
            with self.subTest(replacement=replacement):
                changed = source.replace(original, replacement)
                self.assertNotEqual(changed, source)
                run = subprocess.run([str(BIN / "atlas-opt"), "-"], input=changed,
                                     capture_output=True, text=True, check=False)
                self.assertNotEqual(run.returncode, 0)
                self.assertTrue(run.stderr)

    def test_selected_core_matches_full_row_maximum_and_preserves_input(self) -> None:
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
                        source, expected, maxima = _panel(phase)
                        first_only = [max(struct.unpack(
                            "<16H", source[32 * row:32 * (row + 1)]), key=_ordered)
                            for row in range(32)]
                        wrong = b"".join(struct.pack("<H", value)
                                         for _half in range(2) for value in first_only
                                         for _lane in range(16))
                        self.assertNotEqual(wrong, expected)
                        self.assertTrue(any(a != b for a, b in zip(first_only, maxima)))
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
