"""Two-K-tile MXU0 reset/continuation observation on selected AtlasCore."""

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

from test_mxu_numerics import dot
from test_mxu_reference import _object_words

from selected_core_runtime import run_selected_program

ROOT = pathlib.Path(__file__).resolve().parents[1]
BIN = pathlib.Path(os.environ.get("ATLAS_OOT_BIN_DIR", ROOT / "build/bin"))
SOURCE = ROOT / "test/examples/mxu0_k2.mlir"
ASSEMBLY = ROOT / "test/examples/mxu0_k2.S"
ONE = 0x38
SMALL = 0x18


def _emitted(source: pathlib.Path = SOURCE) -> tuple[int, ...]:
    result = subprocess.run([str(BIN / "atlas-emit"), str(source)],
                            capture_output=True, text=True, check=True)
    return tuple(int(line, 16) for line in result.stdout.splitlines())


def _mixed_inputs() -> tuple[bytes, bytes, bytes, bytes]:
    rng = random.Random(0xA71A60)
    pool = (0x00, 0x38, 0xB8, 0x18, 0x98, 0x40, 0xC0)
    a0, a1, w0, w1 = ([rng.choice(pool) for _ in range(1024)] for _ in range(4))

    # Two output cells isolate continuation at both register halves. These
    # assignments leave many other cells dense and differently valued.
    for row, column in ((0, 0), (5, 20)):
        for k in range(32):
            a0[32 * row + k] = a1[32 * row + k] = 0
            w0[32 * column + k] = w1[32 * column + k] = 0
    a0[0] = w0[0] = ONE
    a1[0] = a1[1] = SMALL
    w1[0] = SMALL
    w1[1] = SMALL | 0x80
    a0[32 * 5 + 2] = ONE
    w0[32 * 20 + 2] = 0x40  # two
    a1[32 * 5 + 2] = a1[32 * 5 + 3] = SMALL
    w1[32 * 20 + 2] = w1[32 * 20 + 3] = SMALL | 0x80
    return bytes(a0), bytes(a1), bytes(w0), bytes(w1)


def _expected(inputs: tuple[bytes, bytes, bytes, bytes], *, continue_acc: bool) -> bytes:
    a0, a1, w0, w1 = inputs
    cells = []
    for row in range(32):
        for col in range(32):
            acts = a0[32 * row:32 * row + 32] + a1[32 * row:32 * row + 32]
            weights = w0[32 * col:32 * col + 32] + w1[32 * col:32 * col + 32]
            cells.append(dot(weights if continue_acc else weights[32:],
                             acts if continue_acc else acts[32:]))
    # Each physical BF16 mreg holds the same 16-column half of every row.
    return (b"".join(struct.pack("<H", cells[32 * row + col])
                     for row in range(32) for col in range(16)) +
            b"".join(struct.pack("<H", cells[32 * row + col])
                     for row in range(32) for col in range(16, 32)))


class MXUK2ReferenceTest(unittest.TestCase):
    def test_typed_k_chain_matches_selected_assembler_and_llvm_object(self) -> None:
        root = os.environ.get("ATLAS_ASSEMBLER_ROOT")
        if not root:
            self.skipTest("set ATLAS_ASSEMBLER_ROOT for selected assembler")
        path = pathlib.Path(root) / "assembler.py"
        spec = importlib.util.spec_from_file_location("selected_mxu_k2_assembler", path)
        assert spec and spec.loader
        assembler = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(assembler)
        emitted = _emitted()
        self.assertEqual(len(emitted), 57)
        self.assertEqual(emitted, tuple(assembler.assemble(ASSEMBLY.read_text())))
        self.assertEqual(emitted[19], assembler.VMATMUL_MXU0(0, 0, 0))
        self.assertEqual(emitted[37], assembler.VMATMUL_ACC_MXU0(0, 2, 0))
        self.assertEqual(_object_words(SOURCE), emitted)
        printed = subprocess.run([str(BIN / "atlas-opt"), str(SOURCE)],
                                 capture_output=True, text=True, check=True)
        self.assertEqual(printed.stdout.count('"atlas.mxu_matmul"'), 2)
        reparsed = subprocess.run([str(BIN / "atlas-opt"), "-"],
                                  input=printed.stdout, capture_output=True,
                                  text=True, check=True)
        self.assertEqual(reparsed.stdout, printed.stdout)

    def test_exact_oracle_distinguishes_reset_and_continuation(self) -> None:
        dense = (bytes([ONE]) * 1024,) * 4
        continued = _expected(dense, continue_acc=True)
        reset = _expected(dense, continue_acc=False)
        self.assertEqual(set(struct.iter_unpack("<H", continued)), {(0x4280,)})  # 64
        self.assertEqual(set(struct.iter_unpack("<H", reset)), {(0x4200,)})  # 32
        mixed = _mixed_inputs()
        continued = _expected(mixed, continue_acc=True)
        reset = _expected(mixed, continue_acc=False)
        self.assertNotEqual(continued, reset)
        # +half-ULP ties to 1, then -half-ULP reaches the next lower BF16.
        # A single final rounding of both products would instead yield 1.
        self.assertEqual(struct.unpack_from("<H", continued, 0)[0], 0x3F7F)
        self.assertNotEqual(struct.unpack_from("<H", continued, 1024 + 2 * (5 * 16 + 4))[0],
                            struct.unpack_from("<H", reset, 1024 + 2 * (5 * 16 + 4))[0])

    def test_selected_core_matches_full_ordered_bf16_result_and_reset_mutation(self) -> None:
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
                cases = (("dense", (bytes([ONE]) * 1024,) * 4),
                         ("mixed", _mixed_inputs()))
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
                                        'src = 2 : i32, weight_slot = 0 : i32, acc_slot = 0 : i32, accumulate = true',
                                        'src = 2 : i32, weight_slot = 0 : i32, acc_slot = 0 : i32, accumulate = false')
                                    self.assertNotEqual(mutated, original_source)
                                    changed.write_text(mutated)
                                    words = _object_words(changed)
                            preload = [(0x90000000 + 1024 * i, value)
                                       for i, value in enumerate(inputs)]
                            preload += [(0x90001000, b"\xA5" * 2048),
                                        (0x90001800, b"\x5A" * 32)]
                            result = run_selected_program(cosim_atlas, model, state, words,
                                                             preload=preload, max_cycles=10000)
                            observed = result.slave.captured(0x90001000, 2048)
                            expected = _expected(inputs, continue_acc=continue_acc)
                            self.assertTrue(result.halted)
                            self.assertEqual((result.reads, result.writes), (128, 64))
                            self.assertEqual(observed, expected)
                            for address, value in preload[:4]:
                                self.assertEqual(result.slave.captured(address, 1024), value)
                            self.assertEqual(result.slave.captured(0x90001800, 32), b"\x5A" * 32)
                self.assertNotEqual(_expected(cases[1][1], continue_acc=True),
                                    _expected(cases[1][1], continue_acc=False))
            finally:
                cosim_atlas.CosimCore = original
        finally:
            os.chdir(old_cwd)
            sys.path.remove(str(modelir))


if __name__ == "__main__":
    unittest.main()
