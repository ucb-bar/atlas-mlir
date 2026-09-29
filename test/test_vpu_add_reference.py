"""Bounded exact VADD check through the hand OOT and selected AtlasCore."""

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
SOURCE = ROOT / "test/examples/vpu_add_pair.mlir"
ASSEMBLY = ROOT / "test/examples/vpu_add_pair.S"


def _pow2(exponent: int) -> Fraction:
    return Fraction(1 << exponent) if exponent >= 0 else Fraction(1, 1 << -exponent)


def _decode_normal_bf16(code: int) -> Fraction:
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


def _add_then_chop(lhs: int, rhs: int) -> int:
    # AddSubSumVec widens each BF16 input, performs FP32 recoded addition,
    # converts to FP32, then takes bits [31:16]. In this declared domain the
    # rational sum is exactly representable as FP32, so no FP32 rounding is
    # needed; the final BF16 step is a bit chop, not BF16 nearest-even.
    exact = _decode_normal_bf16(lhs) + _decode_normal_bf16(rhs)
    if exact == 0:
        raise ValueError("signed-zero cases are outside this bounded check")
    exponent = _binary_exponent(exact)
    if not -126 <= exponent <= 127:
        raise ValueError("result outside normal FP32/BF16 domain")
    fp32_significand = abs(exact) / _pow2(exponent - 23)
    if fp32_significand.denominator != 1:
        raise ValueError("sum requires FP32 rounding; outside exact subset")
    bf16_significand = abs(exact) / _pow2(exponent - 7)
    fraction = bf16_significand.numerator // bf16_significand.denominator - 128
    return ((0x8000 if exact < 0 else 0) | ((exponent + 127) << 7) | fraction)


def _panel(phase: int) -> tuple[bytes, bytes, bytes]:
    pairs = (
        (0x3F80, 0x3BC0),  # 1 + 3/512: chop=0x3f80, BF16 RNE=0x3f81
        (0xBF80, 0xBBC0),  # corresponding negative result
        (0x3F80, 0xBF00), (0xBF80, 0x3F00),
        (0x4000, 0x3F00), (0xC000, 0xBF00),
        (0x3F00, 0x3E80), (0xBF00, 0xBE80),
        (0x3E80, 0x3C80), (0xBE80, 0x3C80),
        (0x3D80, 0x3B80), (0xBD80, 0xBB80),
    )
    chosen = [pairs[(i * 7 + i // 17 + phase) % len(pairs)] for i in range(1024)]
    chosen[0] = pairs[0]
    chosen[512] = pairs[1]
    lhs = b"".join(struct.pack("<H", a) for a, _ in chosen)
    rhs = b"".join(struct.pack("<H", b) for _, b in chosen)
    expected = b"".join(struct.pack("<H", _add_then_chop(a, b)) for a, b in chosen)
    return lhs, rhs, expected


class VpuAddReferenceTest(unittest.TestCase):
    def test_exact_subset_and_rounding_discriminator(self) -> None:
        self.assertEqual(_decode_normal_bf16(0x3BC0), Fraction(3, 512))
        self.assertEqual(_add_then_chop(0x3F80, 0x3BC0), 0x3F80)
        self.assertEqual(_add_then_chop(0xBF80, 0xBBC0), 0xBF80)
        self.assertNotEqual(_add_then_chop(0x3F80, 0x3BC0), 0x3F81)
        for phase in (0, 3):
            lhs, rhs, expected = _panel(phase)
            self.assertEqual((len(lhs), len(rhs), len(expected)), (2048, 2048, 2048))
            self.assertNotEqual(expected, lhs)
            self.assertNotEqual(expected, rhs)

    def test_selected_assembler_and_llvm_object_words(self) -> None:
        selected_root = os.environ.get("ATLAS_ASSEMBLER_ROOT")
        if not selected_root:
            self.skipTest("set ATLAS_ASSEMBLER_ROOT for selected assembler")
        spec = importlib.util.spec_from_file_location(
            "selected_vadd_assembler", pathlib.Path(selected_root) / "assembler.py")
        assert spec and spec.loader
        assembler = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(assembler)
        emitted = tuple(int(line, 16) for line in subprocess.check_output(
            [str(BIN / "atlas-emit"), str(SOURCE)], text=True).splitlines())
        self.assertEqual(len(emitted), 49)
        self.assertEqual(emitted, tuple(assembler.assemble(ASSEMBLY.read_text())))
        self.assertEqual(emitted[31], assembler.VADD_BF16(4, 0, 2))
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
        original = 'kind = "add", dst = 4 : i32, lhs = 0 : i32, rhs = 2 : i32'
        for mutated in (
            'kind = "add", dst = 5 : i32, lhs = 0 : i32, rhs = 2 : i32',
            'kind = "add", dst = 4 : i32, lhs = 63 : i32, rhs = 2 : i32',
            'kind = "add", dst = 4 : i32, lhs = 0 : i32, rhs = 3 : i32',
            'kind = "unknown", dst = 4 : i32, lhs = 0 : i32, rhs = 2 : i32',
        ):
            with self.subTest(mutated=mutated):
                changed = source.replace(original, mutated)
                self.assertNotEqual(changed, source)
                run = subprocess.run([str(BIN / "atlas-opt"), "-"], input=changed,
                                     capture_output=True, text=True, check=False)
                self.assertNotEqual(run.returncode, 0)
                self.assertTrue(run.stderr)

    def test_selected_core_matches_exact_sum_then_bf16_chop(self) -> None:
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
