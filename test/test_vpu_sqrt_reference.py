"""Bounded BF16 VSQRT raw-bit check against selected AtlasCore, not npu_model."""

from __future__ import annotations

import importlib.util
import math
import os
import pathlib
import struct
import subprocess
import sys
import unittest

from test_vpu_relu_reference import _emitted, _object_words

ROOT = pathlib.Path(__file__).resolve().parents[1]
BIN = pathlib.Path(os.environ.get("ATLAS_OOT_BIN_DIR", ROOT / "build/bin"))
SOURCE = ROOT / "test/examples/vpu_sqrt_pair.mlir"
ASSEMBLY = ROOT / "test/examples/vpu_sqrt_pair.S"
BASE_OP = 'kind = "sqrt", dst = 4 : i32, src = 0 : i32'


def _rounded_fixed_sqrt(numerator: int, denominator: int) -> int:
    """Round sqrt(numerator / denominator) to nearest integer, ties upward."""
    floor = math.isqrt(numerator // denominator)
    return floor + ((2 * floor + 1) ** 2 * denominator <= 4 * numerator)


def _sqrt_bits(raw: int) -> int:
    """Exact integer transcription of the selected 17-bit SqrtLUT path.

    The LUT samples sqrt(1 + fraction/128) in Q1.16 with ties upward. For
    even encoded exponents the selected circuit multiplies by its rounded
    sqrt(2) table endpoint and truncates to Q1.16. It then chops the top
    seven fraction bits and adjusts the exponent. Neither Torch nor host
    floating-point sqrt supplies expected output bits.
    """
    if not 0 <= raw <= 0xFFFF:
        raise ValueError("BF16 code must fit 16 bits")
    exponent = (raw >> 7) & 0xFF
    fraction = raw & 0x7F
    if exponent == 0:
        return 0
    if exponent == 0xFF:
        return 0x7F80 if fraction == 0 else 0
    entry = _rounded_fixed_sqrt((128 + fraction) << 32, 128)
    if exponent % 2 == 0:
        sqrt_two = _rounded_fixed_sqrt(2 << 32, 1)
        entry = (entry * sqrt_two >> 16) & 0x1FFFF
    # All selected normal inputs leave the 17-bit fixed value in [2^16, 2^17).
    assert 1 << 16 <= entry < 1 << 17
    result_exponent = 127 + ((exponent - 127) // 2)
    return (result_exponent << 7) | ((entry >> 9) & 0x7F)


def _panel(phase: int) -> tuple[bytes, bytes]:
    codes = (0x0000, 0x8000, 0x0001, 0x807F,
             0x7F80, 0xFF80, 0x7FC1, 0xFFC1,
             0x3F80, 0xBF80, 0x4000, 0xC000,
             0x3F00, 0xBF00, 0x4080, 0xC080,
             0x3FC0, 0xBFC0, 0x4010, 0xC010,
             0x0080, 0x8080, 0x7F7F, 0xFF7F)
    values = [codes[(i * 7 + i // 16 + phase) % len(codes)] for i in range(1024)]
    source = b"".join(struct.pack("<H", code) for code in values)
    expected = b"".join(struct.pack("<H", _sqrt_bits(code)) for code in values)
    return source, expected


class VpuSqrtReferenceTest(unittest.TestCase):
    def test_source_derived_integer_oracle_distinguishes_sign_and_specials(self) -> None:
        self.assertEqual(_sqrt_bits(0x7FC1), 0)
        self.assertEqual(_sqrt_bits(0xFFC1), 0)
        self.assertEqual(_sqrt_bits(0x8000), 0)
        self.assertEqual(_sqrt_bits(0x807F), 0)
        self.assertEqual(_sqrt_bits(0xFF80), 0x7F80)
        self.assertEqual(_sqrt_bits(0xBF80), 0x3F80)  # input sign is ignored
        self.assertEqual(_sqrt_bits(0x4000), 0x3FB5)  # sqrt(2), LUT chop
        self.assertEqual(_sqrt_bits(0x4080), 0x4000)  # sqrt(4) = 2
        self.assertEqual(_sqrt_bits(0x3F00), 0x3F35)  # sqrt(1/2)
        self.assertEqual(_sqrt_bits(0xBFC0), _sqrt_bits(0x3FC0))
        self.assertNotEqual(_sqrt_bits(0x3FC0), 0x3FC0)  # sqrt is not VMOV
        with self.assertRaisesRegex(ValueError, "fit 16 bits"):
            _sqrt_bits(0x10000)
        for phase in (0, 5):
            source, expected = _panel(phase)
            self.assertEqual((len(source), len(expected)), (2048, 2048))
            self.assertNotEqual(source, expected)
            self.assertIn(b"\x00\x00", expected)
            self.assertIn(b"\x80\x7F", expected)

    def test_typed_stream_matches_selected_assembler_and_llvm_object(self) -> None:
        selected_root = os.environ.get("ATLAS_ASSEMBLER_ROOT")
        if not selected_root:
            self.skipTest("set ATLAS_ASSEMBLER_ROOT for independent selected assembler")
        path = pathlib.Path(selected_root) / "assembler.py"
        spec = importlib.util.spec_from_file_location("selected_sqrt_assembler", path)
        assert spec and spec.loader
        assembler = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(assembler)
        emitted = _emitted(SOURCE)
        self.assertEqual(len(emitted), 36)
        self.assertEqual(emitted, tuple(assembler.assemble(ASSEMBLY.read_text())))
        self.assertEqual(emitted[17], assembler.VSQRT(4, 0))
        self.assertEqual(emitted[17] >> 25, 0x4D)
        self.assertNotEqual(emitted[17] >> 25, 0x46)  # VSQUARE mode
        self.assertEqual(_object_words(SOURCE), emitted)
        lowered = subprocess.run([str(BIN / "atlas-opt"), "--convert-atlas-to-llvm",
                                  str(SOURCE)], text=True, capture_output=True, check=True)
        self.assertIn("llvm.inline_asm has_side_effects", lowered.stdout)
        self.assertIn('"~{memory}"', lowered.stdout)
        self.assertIn(f".word 0x{emitted[17]:08x}", lowered.stdout)
        parsed = subprocess.run([str(BIN / "atlas-opt"), str(SOURCE)],
                                text=True, capture_output=True, check=True)
        self.assertEqual(parsed.stdout.count('"atlas.vpu_unary"'), 1)
        reparsed = subprocess.run([str(BIN / "atlas-opt"), "-"], input=parsed.stdout,
                                  text=True, capture_output=True, check=True)
        self.assertEqual(reparsed.stdout, parsed.stdout)

    def test_invalid_pairs_and_unknown_mode_are_rejected(self) -> None:
        source = SOURCE.read_text()
        for mutation in (
            'kind = "sqrt", dst = 5 : i32, src = 0 : i32',
            'kind = "sqrt", dst = 63 : i32, src = 0 : i32',
            'kind = "sqrt", dst = 4 : i32, src = 1 : i32',
            'kind = "sqrt", dst = 4 : i32, src = 63 : i32',
            'kind = "unqualified", dst = 4 : i32, src = 0 : i32',
        ):
            with self.subTest(mutation=mutation):
                changed = source.replace(BASE_OP, mutation)
                self.assertNotEqual(changed, source)
                run = subprocess.run([str(BIN / "atlas-opt"), "-"], input=changed,
                                     text=True, capture_output=True, check=False)
                self.assertNotEqual(run.returncode, 0)
                self.assertTrue(run.stderr)

    def test_selected_core_matches_source_derived_sqrt(self) -> None:
        keys = ("ATLAS_ARC_MODEL", "ATLAS_ARC_STATE", "ATLAS_MODELIR_ROOT",
                "ATLAS_LLVM_BIN")
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
                words = _object_words(SOURCE)
                self.assertEqual(words, _emitted(SOURCE))
                for phase in (0, 5):
                    with self.subTest(phase=phase):
                        source, expected = _panel(phase)
                        result = cosim_atlas.run_program(
                            model, state, words,
                            preload=[(0x90000000, source[:1024]),
                                     (0x90000400, source[1024:]),
                                     (0x90000800, b"\xA5" * 1024),
                                     (0x90000C00, b"\xA5" * 1024),
                                     (0x90001000, b"\x5A" * 32)],
                            max_cycles=8000,
                        )
                        observed = (result.slave.captured(0x90000800, 1024) +
                                    result.slave.captured(0x90000C00, 1024))
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
