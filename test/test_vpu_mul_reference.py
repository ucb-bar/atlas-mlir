"""Bounded exact BF16 VMUL check through hand OOT and selected AtlasCore."""

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
SOURCE = ROOT / "test/examples/vpu_mul_pair.mlir"
ASSEMBLY = ROOT / "test/examples/vpu_mul_pair.S"


def _pow2(exponent: int) -> Fraction:
    return Fraction(1 << exponent) if exponent >= 0 else Fraction(1, 1 << -exponent)


def _decode_normal(code: int) -> Fraction:
    exponent = (code >> 7) & 0xFF
    if not 1 <= exponent <= 254:
        raise ValueError(f"outside finite-normal test domain: {code:#06x}")
    magnitude = Fraction(128 + (code & 0x7F)) * _pow2(exponent - 134)
    return -magnitude if code & 0x8000 else magnitude


def _binary_exponent(value: Fraction) -> int:
    magnitude = abs(value)
    exponent = magnitude.numerator.bit_length() - magnitude.denominator.bit_length()
    if magnitude < _pow2(exponent):
        exponent -= 1
    return exponent


def _product_rne(lhs: int, rhs: int) -> int:
    exact = _decode_normal(lhs) * _decode_normal(rhs)
    if exact == 0:
        raise ValueError("zero products are outside this bounded check")
    exponent = _binary_exponent(exact)
    if not -126 <= exponent <= 126:
        raise ValueError("result outside bounded normal BF16 range")
    scaled = abs(exact) / _pow2(exponent - 7)
    integer, remainder = divmod(scaled.numerator, scaled.denominator)
    twice = 2 * remainder
    rounded = integer + (twice > scaled.denominator or
                         (twice == scaled.denominator and integer % 2 == 1))
    if rounded == 256:
        rounded = 128
        exponent += 1
    return ((0x8000 if exact < 0 else 0) |
            ((exponent + 127) << 7) | (rounded - 128))


def _panel(phase: int) -> tuple[bytes, bytes, bytes]:
    pairs = (
        (0x3F82, 0x3FA0),  # halfway: nearest-even chooses lower 0x3fa2
        (0x3F81, 0x3FC0),  # halfway: nearest-even chooses upper 0x3fc2
        (0xBF82, 0x3FA0), (0xBF81, 0xBFC0),
        (0x3F80, 0xBF80), (0xBF80, 0xBF80),
        (0x3E80, 0x4000), (0x4000, 0x3E80),
        (0x3F40, 0x3FC0), (0xC000, 0x3F80),
        (0x3D80, 0x4040), (0xBD80, 0x3FC0),
    )
    chosen = [pairs[(i * 7 + i // 16 + phase) % len(pairs)] for i in range(1024)]
    chosen[0], chosen[1], chosen[512] = pairs[0], pairs[1], pairs[2]
    lhs = b"".join(struct.pack("<H", a) for a, _ in chosen)
    rhs = b"".join(struct.pack("<H", b) for _, b in chosen)
    expected = b"".join(struct.pack("<H", _product_rne(a, b))
                        for a, b in chosen)
    return lhs, rhs, expected


class VpuMulReferenceTest(unittest.TestCase):
    def test_exact_product_rounding_and_sign(self) -> None:
        self.assertEqual(_product_rne(0x3F82, 0x3FA0), 0x3FA2)
        self.assertEqual(_product_rne(0x3F81, 0x3FC0), 0x3FC2)
        self.assertEqual(_product_rne(0xBF82, 0x3FA0), 0xBFA2)
        self.assertEqual(_product_rne(0xBF80, 0xBF80), 0x3F80)
        for phase in (0, 3):
            lhs, rhs, expected = _panel(phase)
            self.assertEqual((len(lhs), len(rhs), len(expected)), (2048, 2048, 2048))
            self.assertNotEqual(expected, lhs)
            self.assertNotEqual(expected, rhs)

    def test_selected_assembler_and_llvm_object_words(self) -> None:
        root = os.environ.get("ATLAS_ASSEMBLER_ROOT")
        if not root:
            self.skipTest("set ATLAS_ASSEMBLER_ROOT for selected assembler")
        spec = importlib.util.spec_from_file_location(
            "selected_vpu_mul_assembler", pathlib.Path(root) / "assembler.py")
        assert spec and spec.loader
        assembler = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(assembler)
        emitted = tuple(int(line, 16) for line in subprocess.check_output(
            [str(BIN / "atlas-emit"), str(SOURCE)], text=True).splitlines())
        self.assertEqual(len(emitted), 49)
        self.assertEqual(emitted, tuple(assembler.assemble(ASSEMBLY.read_text())))
        self.assertEqual(emitted[31], assembler.VMUL_BF16(4, 0, 2))
        self.assertEqual(emitted, _object_words(SOURCE))
        printed = subprocess.run([str(BIN / "atlas-opt"), str(SOURCE)],
                                 capture_output=True, text=True, check=True)
        self.assertEqual(printed.stdout.count('"atlas.vpu_binary"'), 1)
        reparsed = subprocess.run([str(BIN / "atlas-opt"), "-"],
                                  input=printed.stdout, capture_output=True,
                                  text=True, check=True)
        self.assertEqual(reparsed.stdout, printed.stdout)

    def test_invalid_pair_and_kind_are_rejected(self) -> None:
        source = SOURCE.read_text()
        original = 'kind = "mul", dst = 4 : i32, lhs = 0 : i32, rhs = 2 : i32'
        for replacement in (
            'kind = "mul", dst = 5 : i32, lhs = 0 : i32, rhs = 2 : i32',
            'kind = "mul", dst = 4 : i32, lhs = 63 : i32, rhs = 2 : i32',
            'kind = "mul", dst = 4 : i32, lhs = 0 : i32, rhs = 3 : i32',
            'kind = "invalid", dst = 4 : i32, lhs = 0 : i32, rhs = 2 : i32',
        ):
            with self.subTest(replacement=replacement):
                changed = source.replace(original, replacement)
                self.assertNotEqual(changed, source)
                run = subprocess.run([str(BIN / "atlas-opt"), "-"], input=changed,
                                     capture_output=True, text=True, check=False)
                self.assertNotEqual(run.returncode, 0)
                self.assertTrue(run.stderr)

    def test_selected_core_matches_exact_bf16_multiply(self) -> None:
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
                for phase in (0, 3):
                    with self.subTest(phase=phase):
                        lhs, rhs, expected = _panel(phase)
                        result = cosim_atlas.run_program(
                            model, state, _object_words(SOURCE),
                            preload=[(0x90000000, lhs), (0x90000800, rhs),
                                     (0x90001000, b"\xA5" * 2048),
                                     (0x90001800, b"\x5A" * 32)],
                            max_cycles=10000,
                        )
                        observed = result.slave.captured(0x90001000, 2048)
                        self.assertTrue(result.halted)
                        self.assertEqual((result.reads, result.writes), (128, 64))
                        self.assertEqual(observed, expected)
                        self.assertEqual(result.slave.captured(0x90000000, 2048), lhs)
                        self.assertEqual(result.slave.captured(0x90000800, 2048), rhs)
                        self.assertEqual(result.slave.captured(0x90001800, 32), b"\x5A" * 32)
            finally:
                cosim_atlas.CosimCore = original
        finally:
            os.chdir(old_cwd)
            sys.path.remove(str(modelir))


if __name__ == "__main__":
    unittest.main()
