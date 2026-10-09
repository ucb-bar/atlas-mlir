"""Bounded BF16 VEXP2 raw-bit check against selected AtlasCore, not npu_model."""

from __future__ import annotations

import importlib.util
import os
import pathlib
import struct
import subprocess
import sys
import unittest

from test_vpu_relu_reference import _emitted, _object_words

from selected_core_runtime import run_selected_program

ROOT = pathlib.Path(__file__).resolve().parents[1]
BIN = pathlib.Path(os.environ.get("ATLAS_OOT_BIN_DIR", ROOT / "build/bin"))
SOURCE = ROOT / "test/examples/vpu_exp2_pair.mlir"
ASSEMBLY = ROOT / "test/examples/vpu_exp2_pair.S"
BASE_OP = 'kind = "exp2", dst = 4 : i32, src = 0 : i32'


def _exp2_bits(raw: int) -> int:
    """Bounded exact-power oracle for the selected ExpLane integer path.

    Integer inputs in [-100, 88] have exact Q9.12 conversion, integer k,
    residual r=0, and ExLUT[0]=1. The selected threshold marks positive
    integers from 89 as overflow before this path. Zero/infinity use explicit
    early outputs. Fractional, subnormal, and NaN inputs are outside this
    oracle; no Torch or model output is used as a golden.
    """
    if not 0 <= raw <= 0xFFFF:
        raise ValueError("BF16 code must fit 16 bits")
    sign = -1 if raw & 0x8000 else 1
    exponent = (raw >> 7) & 0xFF
    fraction = raw & 0x7F
    if exponent == 0:
        if fraction:
            raise ValueError("subnormal is outside bounded oracle")
        return 0x3F80
    if exponent == 0xFF:
        if fraction:
            raise ValueError("NaN is outside bounded oracle")
        return 0 if sign < 0 else 0x7F80
    shift = exponent - 127 - 7
    significand = 128 + fraction
    numerator, denominator = ((significand << shift, 1) if shift >= 0
                              else (significand, 1 << -shift))
    if numerator % denominator:
        raise ValueError("fractional input is outside bounded oracle")
    integer = sign * (numerator // denominator)
    if not -100 <= integer <= 100:
        raise ValueError("integer outside tested Q9.12 range")
    if integer >= 89:
        return 0x7F80
    return (127 + integer) << 7


def _panel(phase: int) -> tuple[bytes, bytes]:
    codes = (0x0000, 0x8000, 0x7F80, 0xFF80,
             0x3F80, 0xBF80, 0x4000, 0xC000,
             0x4080, 0xC080, 0x4100, 0xC100,
             0x42B0, 0x42B2, 0x42C8, 0xC2B2,
             0xC2C8)
    values = [codes[(i * 7 + i // 16 + phase) % len(codes)] for i in range(1024)]
    source = b"".join(struct.pack("<H", code) for code in values)
    expected = b"".join(struct.pack("<H", _exp2_bits(code)) for code in values)
    return source, expected


class VpuExp2ReferenceTest(unittest.TestCase):
    def test_integer_oracle_exposes_selected_overflow_and_specials(self) -> None:
        self.assertEqual(_exp2_bits(0x0000), 0x3F80)
        self.assertEqual(_exp2_bits(0x8000), 0x3F80)
        self.assertEqual(_exp2_bits(0x7F80), 0x7F80)
        self.assertEqual(_exp2_bits(0xFF80), 0)
        self.assertEqual(_exp2_bits(0x3F80), 0x4000)  # 2^1 = 2
        self.assertEqual(_exp2_bits(0xBF80), 0x3F00)  # 2^-1 = 1/2
        self.assertEqual(_exp2_bits(0x42B0), 0x6B80)  # 2^88 finite
        self.assertEqual(_exp2_bits(0x42B2), 0x7F80)  # 89 trips RTL guard
        self.assertEqual(_exp2_bits(0xC2B2), 0x1300)  # 2^-89 finite
        self.assertNotEqual(_exp2_bits(0x42B2), 0x6C00)  # exact 2^89
        with self.assertRaisesRegex(ValueError, "fit 16 bits"):
            _exp2_bits(0x10000)
        for unsupported in (0x0001, 0x7FC1, 0x3F00):
            with self.assertRaisesRegex(ValueError, "outside bounded oracle"):
                _exp2_bits(unsupported)
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
        spec = importlib.util.spec_from_file_location("selected_exp2_assembler", path)
        assert spec and spec.loader
        assembler = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(assembler)
        emitted = _emitted(SOURCE)
        self.assertEqual(len(emitted), 36)
        self.assertEqual(emitted, tuple(assembler.assemble(ASSEMBLY.read_text())))
        self.assertEqual(emitted[17], assembler.VEXP2(4, 0))
        self.assertEqual(emitted[17] >> 25, 0x43)
        self.assertNotEqual(emitted[17] >> 25, 0x42)  # VEXP mode
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
            'kind = "exp2", dst = 5 : i32, src = 0 : i32',
            'kind = "exp2", dst = 63 : i32, src = 0 : i32',
            'kind = "exp2", dst = 4 : i32, src = 1 : i32',
            'kind = "exp2", dst = 4 : i32, src = 63 : i32',
            'kind = "unqualified", dst = 4 : i32, src = 0 : i32',
        ):
            with self.subTest(mutation=mutation):
                changed = source.replace(BASE_OP, mutation)
                self.assertNotEqual(changed, source)
                run = subprocess.run([str(BIN / "atlas-opt"), "-"], input=changed,
                                     text=True, capture_output=True, check=False)
                self.assertNotEqual(run.returncode, 0)
                self.assertTrue(run.stderr)

    def test_selected_core_matches_source_derived_exp2(self) -> None:
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
                        result = run_selected_program(cosim_atlas,
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
