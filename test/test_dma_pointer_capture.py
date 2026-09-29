"""Bounded DMA scalar-pointer capture check on the selected standalone core."""

from __future__ import annotations

import importlib.util
import os
import pathlib
import subprocess
import sys
import unittest

from test_mxu_reference import _object_words

ROOT = pathlib.Path(__file__).resolve().parents[1]
BIN = pathlib.Path(os.environ.get("ATLAS_OOT_BIN_DIR", ROOT / "build/bin"))
SOURCE = ROOT / "test/examples/dma_pointer_capture.mlir"
ASSEMBLY = ROOT / "test/examples/dma_pointer_capture.S"


def _emitted() -> tuple[int, ...]:
    return tuple(int(line, 16) for line in subprocess.check_output(
        [str(BIN / "atlas-emit"), str(SOURCE)], text=True).splitlines())


class DMAPointerCaptureTest(unittest.TestCase):
    def test_selected_assembler_and_llvm_object_words(self) -> None:
        selected_root = os.environ.get("ATLAS_ASSEMBLER_ROOT")
        if not selected_root:
            self.skipTest("set ATLAS_ASSEMBLER_ROOT for selected assembler")
        spec = importlib.util.spec_from_file_location(
            "selected_dma_pointer_assembler", pathlib.Path(selected_root) / "assembler.py")
        assert spec and spec.loader
        assembler = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(assembler)
        emitted = _emitted()
        self.assertEqual(len(emitted), 16)
        self.assertEqual(emitted, tuple(assembler.assemble(ASSEMBLY.read_text())))
        self.assertEqual(emitted, _object_words(SOURCE))
        self.assertNotEqual(emitted[4], emitted[7])  # distinct pointer values
        printed = subprocess.run([str(BIN / "atlas-opt"), str(SOURCE)],
                                 capture_output=True, text=True, check=True)
        reparsed = subprocess.run([str(BIN / "atlas-opt"), "-"],
                                  input=printed.stdout, capture_output=True,
                                  text=True, check=True)
        self.assertEqual(reparsed.stdout, printed.stdout)

    def test_illegal_dma_register_and_channel_are_rejected(self) -> None:
        source = SOURCE.read_text()
        original = ('direction = "load", channel = 0 : i32, reg = 6 : i32, '
                    'dram = 1 : i32, size = 2 : i32')
        for mutated in (
            'direction = "load", channel = 8 : i32, reg = 6 : i32, dram = 1 : i32, size = 2 : i32',
            'direction = "load", channel = 0 : i32, reg = 32 : i32, dram = 1 : i32, size = 2 : i32',
            'direction = "load", channel = 0 : i32, reg = 6 : i32, dram = 32 : i32, size = 2 : i32',
            'direction = "load", channel = 0 : i32, reg = 6 : i32, dram = 1 : i32, size = 32 : i32',
        ):
            with self.subTest(mutated=mutated):
                changed = source.replace(original, mutated)
                self.assertNotEqual(changed, source)
                run = subprocess.run([str(BIN / "atlas-opt"), "-"], input=changed,
                                     capture_output=True, text=True, check=False)
                self.assertNotEqual(run.returncode, 0)
                self.assertTrue(run.stderr)

    def test_selected_core_captures_pointer_at_dma_launch(self) -> None:
        keys = ("ATLAS_ARC_MODEL", "ATLAS_ARC_STATE", "ATLAS_MODELIR_ROOT", "ATLAS_LLVM_BIN")
        if not all(os.environ.get(key) for key in keys):
            self.skipTest("set selected-source-linked ARC, ModeLIR, and LLVM paths")
        model, state, modelir = (pathlib.Path(os.environ[key]).resolve(strict=True)
                                 for key in keys[:3])
        words = _object_words(SOURCE)
        source_a = bytes((i * 37 + 11) & 0xFF for i in range(128))
        source_b = bytes((i * 53 + 79) & 0xFF for i in range(128))
        self.assertNotEqual(source_a, source_b)
        guards = ((0x90000080, b"\x91" * 16),
                  (0x900003F0, b"\xA2" * 16),
                  (0x90000480, b"\xB3" * 16),
                  (0x90001080, b"\xC4" * 16))
        preload = [(0x90000000, source_a), (0x90001000, source_b),
                   (0x90000400, b"\xA5" * 128), *guards]

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
                for clobber_before_launch in (False, True):
                    with self.subTest(clobber_before_launch=clobber_before_launch):
                        program = list(words)
                        if clobber_before_launch:
                            # Change the first LUI to B. The later LUI already
                            # clobbers x1 to B after the DMA issue point.
                            program[4] = program[7]
                        result = cosim_atlas.run_program(
                            model, state, program, preload=preload, max_cycles=5000)
                        expected = source_b if clobber_before_launch else source_a
                        self.assertTrue(result.halted)
                        self.assertEqual((result.reads, result.writes), (4, 4))
                        self.assertEqual(result.slave.captured(0x90000400, 128), expected)
                        self.assertEqual(result.slave.captured(0x90000000, 128), source_a)
                        self.assertEqual(result.slave.captured(0x90001000, 128), source_b)
                        for address, value in guards:
                            self.assertEqual(result.slave.captured(address, len(value)), value)
            finally:
                cosim_atlas.CosimCore = original
        finally:
            os.chdir(old_cwd)
            sys.path.remove(str(modelir))


if __name__ == "__main__":
    unittest.main()
