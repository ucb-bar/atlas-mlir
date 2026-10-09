"""Bounded FP8 signed-zero check through the hand OOT and selected AtlasCore.

Every weight is a finite normal E4M3 value.  The expected BF16 bits are
constructed independently from the core and the npu_model implementation.
"""

from __future__ import annotations

import os
import pathlib
import struct
import sys
import unittest

from test_mxu_reference import PAIR_SOURCE, _object_words

from selected_core_runtime import run_selected_program


ONE = 0x38
MINUS_ONE = 0xB8
PLUS_ZERO = 0x00
MINUS_ZERO = 0x80
OUTPUT_BASES = (0x90000800, 0x90001000)


def _activation(first_row: tuple[int, ...], fill: int) -> bytes:
    assert len(first_row) <= 32
    result = bytearray([fill] * 1024)
    result[:len(first_row)] = bytes(first_row)
    return bytes(result)


def _expected(first_row_value: int) -> bytes:
    """Physical BF16 pair: columns 0-15, then columns 16-31."""
    half = b"".join(struct.pack("<H", first_row_value if row == 0 else 0)
                    for row in range(32) for _ in range(16))
    assert len(half) == 1024
    return half + half


class MXU0SignedZeroReferenceTest(unittest.TestCase):
    def test_selected_core_zero_sign_with_finite_normal_weights(self) -> None:
        keys = ("ATLAS_ARC_MODEL", "ATLAS_ARC_STATE", "ATLAS_MODELIR_ROOT", "ATLAS_LLVM_BIN")
        if not all(os.environ.get(key) for key in keys):
            self.skipTest("set selected-source-linked ARC, ModeLIR, and LLVM paths")
        model, state, modelir = (pathlib.Path(os.environ[key]).resolve(strict=True)
                                 for key in keys[:3])
        words = _object_words(PAIR_SOURCE)
        self.assertEqual(len(words), 42)
        cases = (
            ("positive_weight_positive_zero", ONE, _activation((), PLUS_ZERO), 0x0000),
            ("positive_weight_negative_zero", ONE, _activation((), MINUS_ZERO), 0x0000),
            ("negative_weight_positive_zero", MINUS_ONE, _activation((), PLUS_ZERO), 0x0000),
            ("negative_weight_negative_zero", MINUS_ONE, _activation((), MINUS_ZERO), 0x0000),
            ("one_then_positive_zero", ONE, _activation((ONE,), PLUS_ZERO), 0x3F80),
            ("one_then_negative_zero", ONE, _activation((ONE,), MINUS_ZERO), 0x3F80),
            ("negative_zero_before_one", ONE,
             _activation((MINUS_ZERO, ONE), MINUS_ZERO), 0x3F80),
            ("exact_cancellation_then_negative_zero", ONE,
             _activation((ONE, MINUS_ONE), MINUS_ZERO), 0x0000),
        )
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
                for name, weight, acts, expected_first_row in cases:
                    with self.subTest(name=name):
                        weights = bytes([weight]) * 1024
                        result = run_selected_program(cosim_atlas,
                            model, state, words,
                            preload=[(0x90000000, weights), (0x90000400, acts),
                                     (OUTPUT_BASES[0], b"\xA5" * 1024),
                                     (OUTPUT_BASES[1], b"\x5A" * 1024),
                                     (0x90001400, b"\xC3" * 32)],
                            max_cycles=5000)
                        observed = b"".join(result.slave.captured(address, 1024)
                                            for address in OUTPUT_BASES)
                        self.assertTrue(result.halted)
                        self.assertEqual((result.reads, result.writes), (64, 64))
                        self.assertEqual(observed, _expected(expected_first_row))
                        self.assertEqual(result.slave.captured(0x90000000, 1024), weights)
                        self.assertEqual(result.slave.captured(0x90000400, 1024), acts)
                        self.assertEqual(result.slave.captured(0x90001400, 32), b"\xC3" * 32)
            finally:
                cosim_atlas.CosimCore = original
        finally:
            os.chdir(old_cwd)
            sys.path.remove(str(modelir))


if __name__ == "__main__":
    unittest.main()
