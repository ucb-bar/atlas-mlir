"""Source-linked DMA launch snapshot and wait/reuse probe on one AtlasCore."""

from __future__ import annotations

import importlib.util
import os
import pathlib
import subprocess
import sys
import unittest

from test_mxu_reference import _object_words

from selected_core_runtime import run_selected_program

ROOT = pathlib.Path(__file__).resolve().parents[1]
BIN = pathlib.Path(os.environ.get("ATLAS_OOT_BIN_DIR", ROOT / "build/bin"))
SOURCE = ROOT / "test/examples/dma_pointer_lifetime.mlir"
ASSEMBLY = ROOT / "test/examples/dma_pointer_lifetime.S"
ADDRESS_A = 0x90000000
ADDRESS_B = 0x90001000
OUTPUT_A = 0x90000400
OUTPUT_B = 0x90000600
TRANSFER_BYTES = 128


def _emitted() -> tuple[int, ...]:
    return tuple(int(line, 16) for line in subprocess.check_output(
        [str(BIN / "atlas-emit"), str(SOURCE)], text=True).splitlines())


class DMAPointerLifetimeTest(unittest.TestCase):
    def test_typed_stream_matches_selected_assembler_and_llvm_object(self) -> None:
        root = os.environ.get("ATLAS_ASSEMBLER_ROOT")
        if not root:
            self.skipTest("set ATLAS_ASSEMBLER_ROOT for selected assembler")
        spec = importlib.util.spec_from_file_location(
            "selected_dma_lifetime_assembler", pathlib.Path(root) / "assembler.py")
        assert spec and spec.loader
        assembler = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(assembler)
        emitted = _emitted()
        self.assertEqual(len(emitted), 25)
        self.assertEqual(emitted, tuple(assembler.assemble(ASSEMBLY.read_text())))
        self.assertEqual(emitted, _object_words(SOURCE))
        self.assertNotEqual(emitted[4], emitted[7])  # disjoint source pointers
        printed = subprocess.run([str(BIN / "atlas-opt"), str(SOURCE)],
                                 capture_output=True, text=True, check=True)
        reparsed = subprocess.run([str(BIN / "atlas-opt"), "-"],
                                  input=printed.stdout, capture_output=True,
                                  text=True, check=True)
        self.assertEqual(reparsed.stdout, printed.stdout)

    def test_invalid_register_channel_and_immediate_are_rejected(self) -> None:
        source = SOURCE.read_text()
        substitutions = (
            ('direction = "load", channel = 0 : i32, reg = 6 : i32, dram = 1 : i32, size = 2 : i32',
             'direction = "load", channel = 8 : i32, reg = 6 : i32, dram = 1 : i32, size = 2 : i32'),
            ('direction = "store", channel = 1 : i32, reg = 6 : i32, dram = 3 : i32, size = 2 : i32',
             'direction = "store", channel = 1 : i32, reg = 6 : i32, dram = 32 : i32, size = 2 : i32'),
            ('kind = "addi", dst = 4 : i32, src = 4 : i32, immediate = 1536 : i32',
             'kind = "addi", dst = 4 : i32, src = 4 : i32, immediate = 2048 : i32'),
        )
        for original, replacement in substitutions:
            with self.subTest(replacement=replacement):
                self.assertIn(original, source)
                run = subprocess.run([str(BIN / "atlas-opt"), "-"],
                                     input=source.replace(original, replacement),
                                     capture_output=True, text=True, check=False)
                self.assertNotEqual(run.returncode, 0)
                self.assertTrue(run.stderr)

    def test_selected_core_captures_address_at_launch_and_waits_before_reuse(self) -> None:
        keys = ("ATLAS_ARC_MODEL", "ATLAS_ARC_STATE", "ATLAS_MODELIR_ROOT", "ATLAS_LLVM_BIN",
                "ATLAS_RTL_ROOT")
        if not all(os.environ.get(key) for key in keys):
            self.skipTest("set selected-source-linked ARC, RTL, ModeLIR, and LLVM paths")
        model, state, modelir, _, rtl = (pathlib.Path(os.environ[key]).resolve(strict=True)
                                        for key in keys)
        scalar = (rtl / "src/main/scala/atlas/scalar/ScalarCore.scala").read_text()
        dma = (rtl / "src/main/scala/diplomatic/memory/DMA.scala").read_text()
        self.assertIn("io.dmaCmd.bits.addr", scalar)
        self.assertIn("dma_wait_stall", scalar)
        self.assertIn("commandQueue(enqueueIdx)               := io.command.bits", dma)
        source_a = bytes((37 * i + 11) & 0xFF for i in range(TRANSFER_BYTES))
        source_b = bytes((53 * i + 79) & 0xFF for i in range(TRANSFER_BYTES))
        self.assertNotEqual(source_a, source_b)
        guards = ((ADDRESS_A + TRANSFER_BYTES, b"\x91" * 16),
                  (OUTPUT_A - 16, b"\xA2" * 16),
                  (OUTPUT_A + TRANSFER_BYTES, b"\xB3" * 16),
                  (OUTPUT_B - 16, b"\xC4" * 16),
                  (OUTPUT_B + TRANSFER_BYTES, b"\xD5" * 16),
                  (ADDRESS_B + TRANSFER_BYTES, b"\xE6" * 16))
        preload = [(ADDRESS_A, source_a), (ADDRESS_B, source_b),
                   (OUTPUT_A, b"\xA5" * TRANSFER_BYTES),
                   (OUTPUT_B, b"\x5A" * TRANSFER_BYTES), *guards]
        object_words = _object_words(SOURCE)

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
                        words = list(object_words)
                        if clobber_before_launch:
                            words[4] = words[7]  # B is present before both launches.
                        trace: list[tuple[int, int, int, int, int, int]] = []

                        def on_cycle(core: SelectedCore) -> None:
                            trace.append(tuple(core.peek(name) & 0xFFFFFFFF for name in (
                                "scalar.io_dma_busy_0", "scalar.io_dma_busy_1",
                                "scalar/regfile/regs_1", "scalar/regfile/regs_10",
                                "scalar/regfile/regs_11", "scalar/regfile/regs_12")))

                        result = run_selected_program(cosim_atlas,
                            model, state, words, preload=preload, max_cycles=5000,
                            on_cycle=on_cycle)
                        self.assertTrue(result.halted)
                        self.assertEqual((result.reads, result.writes), (8, 8))
                        first_expected = source_b if clobber_before_launch else source_a
                        self.assertEqual(result.slave.captured(OUTPUT_A, TRANSFER_BYTES),
                                         first_expected)
                        self.assertEqual(result.slave.captured(OUTPUT_B, TRANSFER_BYTES),
                                         source_b)
                        self.assertEqual(result.slave.captured(ADDRESS_A, TRANSFER_BYTES),
                                         source_a)
                        self.assertEqual(result.slave.captured(ADDRESS_B, TRANSFER_BYTES),
                                         source_b)
                        for address, value in guards:
                            self.assertEqual(result.slave.captured(address, len(value)), value)

                        first_load_done = next(i for i, row in enumerate(trace) if row[3] == 1)
                        first_store_done = next(i for i, row in enumerate(trace) if row[4] == 1)
                        second_store_done = next(i for i, row in enumerate(trace) if row[5] == 1)
                        self.assertLess(first_load_done, first_store_done)
                        self.assertLess(first_store_done, second_store_done)
                        self.assertIn(1, [row[1] for row in
                                          trace[first_store_done:second_store_done]])
                        for channel, marker in ((0, first_load_done),
                                                (1, first_store_done),
                                                (1, second_store_done)):
                            self.assertIn(1, [row[channel] for row in trace[:marker]])
                            self.assertEqual(trace[marker][channel], 0)
                        self.assertTrue(any(row[2] == ADDRESS_B and row[0] == 1
                                            for row in trace[:first_load_done]))
            finally:
                cosim_atlas.CosimCore = original
        finally:
            os.chdir(old_cwd)
            sys.path.remove(str(modelir))


if __name__ == "__main__":
    unittest.main()
