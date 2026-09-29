"""Bounded MXU1 two-K-tile continuation on selected standalone AtlasCore."""

from __future__ import annotations

import importlib.util
import os
import pathlib
import random
import struct
import subprocess
import sys
import tempfile
import unittest

from test_mxu_numerics import bf16, fp8, round_bf16
from test_mxu_reference import _object_words

ROOT = pathlib.Path(__file__).resolve().parents[1]
BIN = pathlib.Path(os.environ.get("ATLAS_OOT_BIN_DIR", ROOT / "build/bin"))
SOURCE = ROOT / "test/examples/mxu1_k2.mlir"
ASSEMBLY = ROOT / "test/examples/mxu1_k2.S"
ONE = 0x38
SMALL = 0x18


def _emitted() -> tuple[int, ...]:
    result = subprocess.run([str(BIN / "atlas-emit"), str(SOURCE)],
                            capture_output=True, text=True, check=True)
    return tuple(int(line, 16) for line in result.stdout.splitlines())


def _panel() -> tuple[bytes, bytes, bytes, bytes]:
    """One plus two half-ULP products, with another witness in the high half."""
    a0, a1, w0, w1 = (bytearray(1024) for _ in range(4))
    for row, col in ((0, 0), (5, 20)):
        a0[32 * row] = w0[32 * col] = ONE
        a0[32 * row + 1] = w0[32 * col + 1] = SMALL
        a1[32 * row + 2] = w1[32 * col + 2] = SMALL
    # A separate negative continuation exercises the accumulator sign path.
    a0[32 * 12] = w0[32 * 31] = ONE
    a1[32 * 12] = SMALL | 0x80
    w1[32 * 31] = SMALL
    return bytes(a0), bytes(a1), bytes(w0), bytes(w1)


def _mixed_panel() -> tuple[bytes, bytes, bytes, bytes]:
    rng = random.Random(0xA71A62)
    pool = (0, ONE, ONE | 0x80, SMALL, SMALL | 0x80)
    return tuple(bytes(rng.choice(pool) for _ in range(1024)) for _ in range(4))


def _expected(inputs: tuple[bytes, bytes, bytes, bytes], *, continue_acc: bool) -> bytes:
    a0, a1, w0, w1 = inputs
    cells: list[int] = []
    for row in range(32):
        for col in range(32):
            first = sum((fp8(a0[32 * row + k]) * fp8(w0[32 * col + k])
                         for k in range(32)), start=0)
            second = sum((fp8(a1[32 * row + k]) * fp8(w1[32 * col + k])
                          for k in range(32)), start=0)
            prior = bf16(round_bf16(first)) if continue_acc else 0
            cells.append(round_bf16(prior + second))
    return (b"".join(struct.pack("<H", cells[32 * row + col])
                     for row in range(32) for col in range(16)) +
            b"".join(struct.pack("<H", cells[32 * row + col])
                     for row in range(32) for col in range(16, 32)))


class MXU1ContinuationReferenceTest(unittest.TestCase):
    def test_typed_llvm_words_match_selected_assembler(self) -> None:
        root = os.environ.get("ATLAS_ASSEMBLER_ROOT")
        if not root:
            self.skipTest("set ATLAS_ASSEMBLER_ROOT for selected assembler")
        path = pathlib.Path(root) / "assembler.py"
        spec = importlib.util.spec_from_file_location("selected_mxu1_k2_assembler", path)
        assert spec and spec.loader
        assembler = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(assembler)
        words = _emitted()
        self.assertEqual(len(words), 57)
        self.assertEqual(words, tuple(assembler.assemble(ASSEMBLY.read_text())))
        self.assertEqual(words[19], assembler.VMATMUL_MXU1(0, 0, 0))
        self.assertEqual(words[37], assembler.VMATMUL_ACC_MXU1(0, 2, 0))
        self.assertEqual(_object_words(SOURCE), words)
        printed = subprocess.run([str(BIN / "atlas-opt"), str(SOURCE)],
                                 capture_output=True, text=True, check=True)
        self.assertEqual(printed.stdout.count('"atlas.mxu_matmul"'), 2)
        reparsed = subprocess.run([str(BIN / "atlas-opt"), "-"], input=printed.stdout,
                                  capture_output=True, text=True, check=True)
        self.assertEqual(reparsed.stdout, printed.stdout)

    def test_invalid_unit_and_accumulator_slot_are_rejected(self) -> None:
        source = SOURCE.read_text()
        for original, replacement in (
            ('unit = 1 : i32, src = 2 : i32, weight_slot',
             'unit = 2 : i32, src = 2 : i32, weight_slot'),
            ('weight_slot = 0 : i32, acc_slot = 0 : i32, accumulate = true',
             'weight_slot = 0 : i32, acc_slot = 2 : i32, accumulate = true'),
        ):
            with self.subTest(replacement=replacement):
                changed = source.replace(original, replacement)
                self.assertNotEqual(changed, source)
                result = subprocess.run([str(BIN / "atlas-opt"), "-"], input=changed,
                                        capture_output=True, text=True, check=False)
                self.assertNotEqual(result.returncode, 0)
                self.assertTrue(result.stderr)

    def test_oracle_distinguishes_tile_rounding_reset_and_global_rounding(self) -> None:
        inputs = _panel()
        continued = _expected(inputs, continue_acc=True)
        reset = _expected(inputs, continue_acc=False)
        self.assertEqual(struct.unpack_from("<H", continued, 0)[0], 0x3F80)
        self.assertEqual(struct.unpack_from("<H", reset, 0)[0], 0x3B80)
        self.assertEqual(round_bf16(1 + fp8(SMALL) * fp8(SMALL) * 2), 0x3F81)
        self.assertEqual(struct.unpack_from("<H", continued, 1024 + 2 * (5 * 16 + 4))[0],
                         0x3F80)
        self.assertEqual(struct.unpack_from("<H", continued, 1024 + 2 * (12 * 16 + 15))[0],
                         0x3F7F)
        dense = (bytes([ONE]) * 1024,) * 4
        self.assertEqual(set(struct.iter_unpack("<H", _expected(dense, continue_acc=True))),
                         {(0x4280,)})
        self.assertEqual(set(struct.iter_unpack("<H", _expected(dense, continue_acc=False))),
                         {(0x4200,)})

    def test_selected_core_continues_then_rejects_reset_mutation(self) -> None:
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
                cases = (("witness", _panel()), ("dense", (bytes([ONE]) * 1024,) * 4),
                         ("mixed", _mixed_panel()))
                for name, inputs in cases:
                  for continue_acc in (True, False):
                    with self.subTest(name=name, continue_acc=continue_acc):
                        if continue_acc:
                            words = _object_words(SOURCE)
                        else:
                            with tempfile.TemporaryDirectory() as temporary:
                                changed = pathlib.Path(temporary) / "reset.mlir"
                                original_source = SOURCE.read_text()
                                mutated = original_source.replace(
                                    'unit = 1 : i32, src = 2 : i32, weight_slot = 0 : i32, acc_slot = 0 : i32, accumulate = true',
                                    'unit = 1 : i32, src = 2 : i32, weight_slot = 0 : i32, acc_slot = 0 : i32, accumulate = false')
                                self.assertNotEqual(mutated, original_source)
                                changed.write_text(mutated)
                                words = _object_words(changed)
                        preload = [(0x90000000 + 1024 * i, value)
                                   for i, value in enumerate(inputs)]
                        preload += [(0x90001000, b"\xA5" * 2048),
                                    (0x90001800, b"\x5A" * 32)]
                        result = cosim_atlas.run_program(model, state, words,
                                                         preload=preload, max_cycles=10000)
                        self.assertTrue(result.halted)
                        self.assertEqual((result.reads, result.writes), (128, 64))
                        self.assertEqual(result.slave.captured(0x90001000, 2048),
                                         _expected(inputs, continue_acc=continue_acc))
                        for address, value in preload[:4]:
                            self.assertEqual(result.slave.captured(address, 1024), value)
                        self.assertEqual(result.slave.captured(0x90001800, 32), b"\x5A" * 32)
            finally:
                cosim_atlas.CosimCore = original
        finally:
            os.chdir(old_cwd)
            sys.path.remove(str(modelir))


if __name__ == "__main__":
    unittest.main()
