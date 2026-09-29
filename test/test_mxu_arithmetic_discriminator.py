"""Bit-level MXU0/MXU1 witnesses against the inspected FP16 model path.

The tiny oracle uses exact rational E4M3 products and integer ties-to-even
rounding. It does not call npu_model, Torch, the OOT emitter, or the core to
calculate expected output bits.
"""

from __future__ import annotations

from fractions import Fraction
import importlib.util
import os
import pathlib
import subprocess
import sys
import unittest

from test_mxu_reference import _object_words

ROOT = pathlib.Path(__file__).resolve().parents[1]
BIN = pathlib.Path(os.environ.get("ATLAS_OOT_BIN_DIR", ROOT / "build/bin"))
SOURCES = {0: ROOT / "test/examples/mxu0_pair.mlir",
           1: ROOT / "test/examples/mxu1_pair.mlir"}
ONE = 0x38
SIXTEENTH = 0x18
SIXTY_FOURTH = 0x08


def _power2(exponent: int) -> Fraction:
    return Fraction(1 << exponent) if exponent >= 0 else Fraction(1, 1 << -exponent)


def _fp8(bits: int) -> Fraction:
    if bits == 0:
        return Fraction(0)
    exponent = (bits >> 3) & 15
    assert 1 <= exponent <= 14 and bits < 128
    return Fraction(8 + (bits & 7), 8) * _power2(exponent - 7)


def _round_ieee(value: Fraction, *, exponent_bits: int, fraction_bits: int,
                bias: int) -> int:
    """Round a finite normal result by exact integer quotient/remainder."""
    if value == 0:
        return 0
    sign = 1 if value < 0 else 0
    positive = abs(value)
    exponent = positive.numerator.bit_length() - positive.denominator.bit_length()
    if positive < _power2(exponent):
        exponent -= 1
    scaled = positive * _power2(fraction_bits - exponent)
    quotient, remainder = divmod(scaled.numerator, scaled.denominator)
    if 2 * remainder > scaled.denominator or (
            2 * remainder == scaled.denominator and quotient & 1):
        quotient += 1
    if quotient == 2 << fraction_bits:
        exponent += 1
        quotient >>= 1
    biased = exponent + bias
    if not 1 <= biased < (1 << exponent_bits) - 1:
        raise ValueError("witness escaped finite normal format domain")
    return ((sign << (exponent_bits + fraction_bits)) |
            (biased << fraction_bits) | (quotient - (1 << fraction_bits)))


def _decode_ieee(bits: int, *, exponent_bits: int, fraction_bits: int,
                 bias: int) -> Fraction:
    if bits == 0:
        return Fraction(0)
    exponent = (bits >> fraction_bits) & ((1 << exponent_bits) - 1)
    if not 1 <= exponent < (1 << exponent_bits) - 1:
        raise ValueError("outside finite normal format domain")
    magnitude = Fraction((1 << fraction_bits) + (bits & ((1 << fraction_bits) - 1)),
                         1 << fraction_bits) * _power2(exponent - bias)
    return -magnitude if bits >> (exponent_bits + fraction_bits) else magnitude


def _bf16(value: Fraction) -> int:
    return _round_ieee(value, exponent_bits=8, fraction_bits=7, bias=127)


def _ordered_bf16(products: tuple[Fraction, ...]) -> int:
    accumulator = Fraction(0)
    for product in products:
        bits = _bf16(accumulator + product)
        accumulator = _decode_ieee(bits, exponent_bits=8, fraction_bits=7, bias=127)
    return _bf16(accumulator)


def _fp16_then_bf16(products: tuple[Fraction, ...]) -> int:
    fp16_bits = _round_ieee(sum(products, start=Fraction(0)),
                           exponent_bits=5, fraction_bits=10, bias=15)
    fp16_value = _decode_ieee(fp16_bits, exponent_bits=5, fraction_bits=10, bias=15)
    return _bf16(fp16_value)


def _witness(name: str) -> tuple[tuple[int, ...], tuple[int, ...]]:
    if name == "two_half_ulp_products":
        return (ONE, SIXTEENTH, SIXTEENTH), (ONE, SIXTEENTH, SIXTEENTH)
    if name == "quarter_fp16_ulp_above_bf16_midpoint":
        return (ONE, SIXTEENTH, SIXTY_FOURTH), (ONE, SIXTEENTH, SIXTY_FOURTH)
    raise ValueError(name)


def _panel(values: tuple[int, ...]) -> bytes:
    result = bytearray(1024)
    result[:len(values)] = bytes(values)
    return bytes(result)


def _emitted(source: pathlib.Path) -> tuple[int, ...]:
    return tuple(int(word, 16) for word in subprocess.check_output(
        [str(BIN / "atlas-emit"), str(source)], text=True).splitlines())


class MXUArithmeticDiscriminatorTest(unittest.TestCase):
    def test_exact_bit_witnesses_separate_three_arithmetic_policies(self) -> None:
        expected = {
            "two_half_ulp_products": (0x3F80, 0x3F81, 0x3F81),
            "quarter_fp16_ulp_above_bf16_midpoint": (0x3F80, 0x3F80, 0x3F81),
        }
        for name, (ordered, fp16_model, exact_once) in expected.items():
            with self.subTest(name=name):
                weights, activations = _witness(name)
                products = tuple(_fp8(w) * _fp8(a)
                                 for w, a in zip(weights, activations, strict=True))
                self.assertEqual(_ordered_bf16(products), ordered)
                self.assertEqual(_fp16_then_bf16(products), fp16_model)
                self.assertEqual(_bf16(sum(products, start=Fraction(0))), exact_once)
        self.assertEqual(_fp8(SIXTEENTH) ** 2, Fraction(1, 256))
        self.assertEqual(_fp8(SIXTY_FOURTH) ** 2, Fraction(1, 4096))

    def test_both_typed_streams_match_llvm_words_and_selected_matmul_encoding(self) -> None:
        selected_root = os.environ.get("ATLAS_ASSEMBLER_ROOT")
        if not selected_root:
            self.skipTest("set ATLAS_ASSEMBLER_ROOT for selected assembler")
        spec = importlib.util.spec_from_file_location(
            "selected_mxu_arithmetic_assembler", pathlib.Path(selected_root) / "assembler.py")
        assert spec and spec.loader
        assembler = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(assembler)
        for unit, source in SOURCES.items():
            with self.subTest(unit=unit):
                words = _emitted(source)
                self.assertEqual(len(words), 42)
                self.assertEqual(words, _object_words(source))
                matmul = (assembler.VMATMUL_MXU0 if unit == 0 else
                          assembler.VMATMUL_MXU1)
                self.assertEqual(words[20], matmul(0, 0, 0))

    def test_selected_core_distinguishes_mxu0_mxu1_and_fp16_model(self) -> None:
        keys = ("ATLAS_ARC_MODEL", "ATLAS_ARC_STATE", "ATLAS_MODELIR_ROOT", "ATLAS_LLVM_BIN")
        if not all(os.environ.get(key) for key in keys):
            self.skipTest("set selected-source-linked ARC, ModeLIR, and LLVM paths")
        model, state, modelir = (pathlib.Path(os.environ[key]).resolve(strict=True)
                                 for key in keys[:3])
        words = {unit: _object_words(source) for unit, source in SOURCES.items()}
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
                for name in ("two_half_ulp_products",
                             "quarter_fp16_ulp_above_bf16_midpoint"):
                    weights, activations = _witness(name)
                    weight_bytes, activation_bytes = _panel(weights), _panel(activations)
                    products = tuple(_fp8(w) * _fp8(a)
                                     for w, a in zip(weights, activations, strict=True))
                    for unit in (0, 1):
                        with self.subTest(name=name, unit=unit):
                            expected_bits = (_ordered_bf16(products) if unit == 0 else
                                             _bf16(sum(products, start=Fraction(0))))
                            expected = (expected_bits.to_bytes(2, "little") +
                                        bytes(1022) + bytes(1024))
                            result = cosim_atlas.run_program(
                                model, state, words[unit],
                                preload=[(0x90000000, weight_bytes),
                                         (0x90000400, activation_bytes),
                                         (0x90000800, b"\xA5" * 1024),
                                         (0x90001000, b"\x5A" * 1024),
                                         (0x90001400, b"\xC3" * 32)],
                                max_cycles=5000)
                            observed = (result.slave.captured(0x90000800, 1024) +
                                        result.slave.captured(0x90001000, 1024))
                            self.assertTrue(result.halted)
                            self.assertEqual((result.reads, result.writes), (64, 64))
                            self.assertEqual(observed, expected)
                            if (unit, name) in (
                                    (0, "two_half_ulp_products"),
                                    (1, "quarter_fp16_ulp_above_bf16_midpoint")):
                                model_bits = _fp16_then_bf16(products)
                                self.assertNotEqual(observed[:2],
                                                    model_bits.to_bytes(2, "little"))
                            self.assertEqual(result.slave.captured(0x90000000, 1024),
                                             weight_bytes)
                            self.assertEqual(result.slave.captured(0x90000400, 1024),
                                             activation_bytes)
                            self.assertEqual(result.slave.captured(0x90001400, 32),
                                             b"\xC3" * 32)
            finally:
                cosim_atlas.CosimCore = original
        finally:
            os.chdir(old_cwd)
            sys.path.remove(str(modelir))


if __name__ == "__main__":
    unittest.main()
