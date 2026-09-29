"""Bounded MXU1 anchor-tree arithmetic checks on selected standalone AtlasCore."""

from __future__ import annotations

import importlib.util
import os
import pathlib
import random
import struct
import subprocess
import sys
import unittest

from test_mxu_numerics import dot, fp8, round_bf16
from test_mxu_reference import PAIR_SOURCE, _object_words

ROOT = pathlib.Path(__file__).resolve().parents[1]
BIN = pathlib.Path(os.environ.get("ATLAS_OOT_BIN_DIR", ROOT / "build/bin"))
SOURCE = ROOT / "test/examples/mxu1_pair.mlir"
ASSEMBLY = ROOT / "test/examples/mxu1_pair.S"
ONE = 0x38
SMALL = 0x18


def _emitted() -> tuple[int, ...]:
    run = subprocess.run([str(BIN / "atlas-emit"), str(SOURCE)],
                         capture_output=True, text=True, check=True)
    return tuple(int(line, 16) for line in run.stdout.splitlines())


def _tie_panel() -> tuple[bytes, bytes]:
    weights, acts = bytearray(1024), bytearray(1024)
    weights[0] = acts[0] = ONE
    weights[1] = weights[2] = acts[1] = acts[2] = SMALL
    weights[20 * 32] = 0xB8  # -1 in the high BF16 register half
    acts[5 * 32] = ONE
    return bytes(weights), bytes(acts)


def _mixed_panel() -> tuple[bytes, bytes]:
    rng = random.Random(0xA71A61)
    # Bounded finite normals, with zero. Exponent ratios are modest enough for
    # the selected anchor integer width; the execution test checks that claim.
    values = (0, ONE, ONE | 0x80, SMALL, SMALL | 0x80)
    return (bytes(rng.choice(values) for _ in range(1024)),
            bytes(rng.choice(values) for _ in range(1024)))


def _expected(weights: bytes, acts: bytes, *, ordered_steps: bool = False) -> bytes:
    cells = []
    for row in range(32):
        for col in range(32):
            w = weights[32 * col:32 * col + 32]
            a = acts[32 * row:32 * row + 32]
            value = dot(w, a) if ordered_steps else round_bf16(
                sum((fp8(x) * fp8(y) for x, y in zip(w, a, strict=True)), start=0))
            cells.append(value)
    return (b"".join(struct.pack("<H", cells[32 * row + col])
                     for row in range(32) for col in range(16)) +
            b"".join(struct.pack("<H", cells[32 * row + col])
                     for row in range(32) for col in range(16, 32)))


class MXU1ReferenceTest(unittest.TestCase):
    def test_selected_assembler_encoding_and_llvm_object(self) -> None:
        root = os.environ.get("ATLAS_ASSEMBLER_ROOT")
        if not root:
            self.skipTest("set ATLAS_ASSEMBLER_ROOT for selected assembler")
        path = pathlib.Path(root) / "assembler.py"
        spec = importlib.util.spec_from_file_location("selected_mxu1_assembler", path)
        assert spec and spec.loader
        assembler = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(assembler)
        emitted = _emitted()
        self.assertEqual(len(emitted), 42)
        self.assertEqual(emitted, tuple(assembler.assemble(ASSEMBLY.read_text())))
        self.assertEqual(emitted[18], assembler.VMATPUSH_W_MXU1(0, 4))
        self.assertEqual(emitted[20], assembler.VMATMUL_MXU1(0, 0, 0))
        self.assertEqual(emitted[22], assembler.VMATPOP_BF16_MXU1(2, 0))
        self.assertEqual(_object_words(SOURCE), emitted)
        printed = subprocess.run([str(BIN / "atlas-opt"), str(SOURCE)],
                                 capture_output=True, text=True, check=True)
        self.assertEqual(printed.stdout.count('"atlas.mxu_matmul"'), 1)
        reparsed = subprocess.run([str(BIN / "atlas-opt"), "-"],
                                  input=printed.stdout, capture_output=True,
                                  text=True, check=True)
        self.assertEqual(reparsed.stdout, printed.stdout)

    def test_invalid_mxu_unit_and_bf16_pair_are_rejected(self) -> None:
        source = SOURCE.read_text()
        for original, replacement in (
            ('unit = 1 : i32, src = 0 : i32, weight_slot',
             'unit = 2 : i32, src = 0 : i32, weight_slot'),
            ('unit = 1 : i32, dst = 2 : i32, slot',
             'unit = 1 : i32, dst = 63 : i32, slot'),
        ):
            with self.subTest(replacement=replacement):
                changed = source.replace(original, replacement)
                self.assertNotEqual(changed, source)
                run = subprocess.run([str(BIN / "atlas-opt"), "-"], input=changed,
                                     capture_output=True, text=True, check=False)
                self.assertNotEqual(run.returncode, 0)
                self.assertTrue(run.stderr)

    def test_exact_reference_distinguishes_mxu1_from_ordered_mxu0(self) -> None:
        weights, acts = _tie_panel()
        once = _expected(weights, acts)
        ordered = _expected(weights, acts, ordered_steps=True)
        self.assertEqual(struct.unpack_from("<H", once, 0)[0], 0x3F81)
        self.assertEqual(struct.unpack_from("<H", ordered, 0)[0], 0x3F80)
        self.assertEqual(struct.unpack_from("<H", once, 1024 + 2 * (5 * 16 + 4))[0], 0xBF80)

    def test_selected_core_executes_mxu1_and_distinguishes_mxu0(self) -> None:
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
                cases = (("ones", bytes([ONE]) * 1024, bytes([ONE]) * 1024),
                         ("tie", *_tie_panel()), ("mixed", *_mixed_panel()))
                for name, weights, acts in cases:
                    for unit in ((1, 0) if name == "tie" else (1,)):
                        with self.subTest(name=name, unit=unit):
                            source = SOURCE if unit == 1 else PAIR_SOURCE
                            result = cosim_atlas.run_program(
                                model, state, _object_words(source),
                                preload=[(0x90000000, weights), (0x90000400, acts),
                                         (0x90000800, b"\xA5" * 1024),
                                         (0x90001000, b"\xA5" * 1024),
                                         (0x90001400, b"\x5A" * 32)],
                                max_cycles=5000,
                            )
                            observed = (result.slave.captured(0x90000800, 1024) +
                                        result.slave.captured(0x90001000, 1024))
                            expected = _expected(weights, acts, ordered_steps=(unit == 0))
                            self.assertTrue(result.halted)
                            self.assertEqual((result.reads, result.writes), (64, 64))
                            self.assertEqual(observed, expected)
                            self.assertEqual(result.slave.captured(0x90000000, 1024), weights)
                            self.assertEqual(result.slave.captured(0x90000400, 1024), acts)
                            self.assertEqual(result.slave.captured(0x90001400, 32), b"\x5A" * 32)
            finally:
                cosim_atlas.CosimCore = original
        finally:
            os.chdir(old_cwd)
            sys.path.remove(str(modelir))


if __name__ == "__main__":
    unittest.main()
