"""Bounded exact-rational MXU0 check on a selected-source-linked core.

Only one output lane and the listed finite normal FP8 encodings are tested.
The oracle is independent Python arithmetic, not npu_model or the emitter.
"""

import bisect
import os
import pathlib
import random
import struct
import sys
import unittest
from fractions import Fraction

from test_mxu_reference import _object_words, _tile

from selected_core_runtime import run_selected_program


def power2(exponent):
    return Fraction(2**exponent) if exponent >= 0 else Fraction(1, 2 ** -exponent)


def fp8(bits):
    exp = (bits >> 3) & 15
    mant = bits & 7
    if exp == 0:
        return Fraction(0)
    if exp == 15 and mant == 7:
        raise ValueError("NaN outside diagnostic domain")
    return (-1 if bits & 128 else 1) * Fraction(8 + mant, 8) * power2(exp - 7)


BF16_BITS = [0] + [bits for bits in range(0x0080, 0x7F80)]
BF16_VALUES = [Fraction(0)] + [Fraction(128 + (bits & 127), 128) * power2(((bits >> 7) & 255) - 127)
                              for bits in BF16_BITS[1:]]


def round_bf16(value):
    if value == 0:
        return 0
    sign = 0x8000 if value < 0 else 0
    target = abs(value)
    index = bisect.bisect_left(BF16_VALUES, target)
    if index == len(BF16_VALUES):
        return sign | BF16_BITS[-1]
    if BF16_VALUES[index] == target:
        return sign | BF16_BITS[index]
    lower = index - 1
    low_distance = target - BF16_VALUES[lower]
    high_distance = BF16_VALUES[index] - target
    if low_distance < high_distance or (low_distance == high_distance and BF16_BITS[lower] % 2 == 0):
        return sign | BF16_BITS[lower]
    return sign | BF16_BITS[index]


def bf16(bits):
    magnitude = bits & 0x7FFF
    if magnitude == 0:
        return Fraction(0)
    exponent = (magnitude >> 7) & 255
    if exponent == 0 or exponent == 255:
        raise ValueError("outside normal finite BF16 domain")
    value = Fraction(128 + (magnitude & 127), 128) * power2(exponent - 127)
    return -value if bits & 0x8000 else value


def dot(weights, acts):
    accum = 0
    for weight, act in zip(weights, acts):
        accum = round_bf16(bf16(accum) + fp8(weight) * fp8(act))
    return accum


class MXUNumericsTest(unittest.TestCase):
    def test_exact_oracle_tie_and_order(self):
        one = fp8(0x38)
        half_ulp_product = fp8(0x18) * fp8(0x18)
        self.assertEqual(round_bf16(one + half_ulp_product), 0x3F80)
        self.assertEqual(round_bf16(one - half_ulp_product), 0x3F7F)
        self.assertEqual(dot((0x38, 0x18, 0x18), (0x38, 0x18, 0x18)), 0x3F80)
        self.assertEqual(dot((0x38, 0x18, 0x18), (0x38, 0x98, 0x98)), 0x3F7E)

    def test_seeded_finite_normal_vectors_on_selected_core(self):
        keys = ("ATLAS_ARC_MODEL", "ATLAS_ARC_STATE", "ATLAS_MODELIR_ROOT", "ATLAS_LLVM_BIN")
        if not all(os.environ.get(key) for key in keys):
            self.skipTest("set selected-source-linked ARC model and LLVM paths")
        modelir = pathlib.Path(os.environ["ATLAS_MODELIR_ROOT"]).resolve(strict=True)
        model = pathlib.Path(os.environ["ATLAS_ARC_MODEL"]).resolve(strict=True)
        state = pathlib.Path(os.environ["ATLAS_ARC_STATE"]).resolve(strict=True)
        old_cwd = pathlib.Path.cwd()
        os.chdir(modelir)
        sys.path.insert(0, str(modelir))
        try:
            from mlc.backends import cosim_atlas

            original = cosim_atlas.CosimCore

            class SelectedCore(original):
                def peek(self, name):
                    return super().peek("scalar/halt_now" if name == "io_halted" else name)

                def poke(self, name, value):
                    if name in ("io_dmaTL_d_bits_opcode", "io_dmaTL_d_bits_size"):
                        if name in self._S:
                            raise AssertionError(f"unexpected live ARC input {name}")
                        return
                    super().poke(name, value)

            cosim_atlas.CosimCore = SelectedCore
            rng = random.Random(0xA71A5)
            values = tuple(bits for bits in range(256) if ((bits >> 3) & 15) > 0 and
                           not (((bits >> 3) & 15) == 15 and (bits & 7) == 7))
            cases = []
            for _ in range(40):
                k = rng.randint(2, 32)
                cases.append((tuple(rng.choice(values) for _ in range(k)),
                              tuple(rng.choice(values) for _ in range(k))))
            covered = {bit for weights, acts in cases for bit in (*weights, *acts)}
            for bit in sorted(set(values) - covered):
                cases.append(((bit, 0), (0x38, 0x38)))
            self.assertEqual({bit for weights, acts in cases for bit in (*weights, *acts)} & set(values),
                             set(values))
            try:
                for case, (weights, acts) in enumerate(cases):
                    expected = dot(weights, acts)
                    with self.subTest(case=case, weights=weights, acts=acts):
                        result = run_selected_program(cosim_atlas,
                            model, state, _object_words(),
                            preload=[(0x90000000, _tile(tuple((0, j, value) for j, value in enumerate(weights)))),
                                     (0x90000400, _tile(tuple((0, j, value) for j, value in enumerate(acts)))),
                                     (0x90000800, b"\xA5" * 1024)],
                            max_cycles=5000,
                        )
                        observed = struct.unpack_from("<H", result.slave.captured(0x90000800, 1024))[0]
                        self.assertTrue(result.halted)
                        self.assertEqual(observed, expected)
            finally:
                cosim_atlas.CosimCore = original
        finally:
            os.chdir(old_cwd)
            sys.path.remove(str(modelir))
