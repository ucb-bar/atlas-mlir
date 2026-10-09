"""Canonical selected FENCE word and bounded scalar control observation."""

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
SOURCE = ROOT / "test/examples/fence_scalar.mlir"
ASSEMBLY = ROOT / "test/examples/fence_scalar.S"
RTL_REVISION = "0079c0541111197741a231c002e3843fa6f545b2"


def _assembler():
    root = os.environ.get("ATLAS_ASSEMBLER_ROOT")
    if not root:
        raise unittest.SkipTest("set ATLAS_ASSEMBLER_ROOT for selected assembler")
    spec = importlib.util.spec_from_file_location(
        "selected_fence_assembler", pathlib.Path(root) / "assembler.py")
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _words() -> tuple[int, ...]:
    return tuple(int(line, 16) for line in subprocess.check_output(
        [str(BIN / "atlas-emit"), str(SOURCE)], text=True).splitlines())


class FenceReferenceTest(unittest.TestCase):
    def test_typed_words_match_selected_assembler_and_llvm_object(self) -> None:
        assembler = _assembler()
        words = _words()
        self.assertEqual(len(words), 6)
        self.assertEqual(words, tuple(assembler.assemble(ASSEMBLY.read_text())))
        self.assertEqual(words, _object_words(SOURCE))
        self.assertEqual(words[2], assembler.FENCE())
        printed = subprocess.run([str(BIN / "atlas-opt"), str(SOURCE)],
                                 text=True, capture_output=True, check=True)
        reparsed = subprocess.run([str(BIN / "atlas-opt"), "-"],
                                  input=printed.stdout, text=True,
                                  capture_output=True, check=True)
        self.assertEqual(reparsed.stdout, printed.stdout)

    def test_selected_core_fence_has_no_scalar_frontend_stall(self) -> None:
        keys = ("ATLAS_ARC_MODEL", "ATLAS_ARC_STATE", "ATLAS_MODELIR_ROOT",
                "ATLAS_LLVM_BIN", "ATLAS_RTL_ROOT", "ATLAS_ASSEMBLER_ROOT")
        if not all(os.environ.get(key) for key in keys):
            self.skipTest("set selected-source-linked ARC, LLVM, ModeLIR, RTL and assembler paths")
        model, state, modelir, _, rtl, _ = (
            pathlib.Path(os.environ[key]).resolve(strict=True) for key in keys)
        self.assertEqual(subprocess.check_output(
            ["git", "-C", str(rtl), "rev-parse", "HEAD"], text=True).strip(), RTL_REVISION)
        core_source = (rtl / "src/main/scala/atlas/scalar/ScalarCore.scala").read_text()
        decode_source = (rtl / "src/main/scala/atlas/scalar/IDecode.scala").read_text()
        self.assertIn("val stall   = delay_stall || dma_wait_stall", core_source)
        self.assertIn("FENCE  -> List(Y, ALU_X, BR_X, OP1_X, OP2_X, IMM_X", decode_source)
        for name in ("scalar/pc_ctrl/io_s1_pc", "scalar/s1_fire",
                     "scalar/delayCounter", "csrfile/reg_dbg0", "csrfile/reg_dbg1"):
            self.assertIn(f'"name": "{name}"', state.read_text())
        assembler = _assembler()
        words = _object_words(SOURCE)
        guards = ((0x90000000, b"\xA5" * 32),
                  (0x90000400, b"\x5A" * 32))
        old_cwd = pathlib.Path.cwd()
        sys.path.insert(0, str(modelir))
        try:
            os.chdir(modelir)
            from mlc.backends import cosim_atlas

            original = cosim_atlas.CosimCore

            class SelectedCore(original):
                def peek(self, name: str) -> int:
                    return super().peek("scalar/halt_now" if name == "io_halted" else name)

            cosim_atlas.CosimCore = SelectedCore
            try:
                observed: dict[str, tuple[int, list[tuple[int, ...]]]] = {}
                for name, opcode in (("fence", assembler.FENCE()),
                                     ("nop", assembler.NOP())):
                    with self.subTest(name=name):
                        program = list(words)
                        program[2] = opcode
                        trace: list[tuple[int, ...]] = []

                        def on_cycle(core: SelectedCore) -> None:
                            trace.append((
                                core.peek("scalar/pc_ctrl/io_s1_pc"),
                                core.peek("scalar/s1_fire"),
                                core.peek("scalar/delayCounter"),
                                core.peek("csrfile/reg_dbg0"),
                                core.peek("csrfile/reg_dbg1"),
                                core.peek("scalar/regfile/regs_1"),
                                core.peek("scalar/regfile/regs_2"),
                            ))

                        result = run_selected_program(cosim_atlas,
                            model, state, program, preload=guards,
                            max_cycles=100, on_cycle=on_cycle)
                        self.assertTrue(result.halted)
                        self.assertEqual((result.reads, result.writes), (0, 0))
                        for pc in (1, 2, 3, 4):
                            self.assertEqual(sum(row[0] == pc and row[1] == 1 for row in trace), 1)
                        fence_fire = next(i for i, row in enumerate(trace)
                                          if row[0] == 2 and row[1] == 1)
                        successor_fire = next(i for i, row in enumerate(trace)
                                              if row[0] == 3 and row[1] == 1)
                        self.assertEqual(successor_fire - fence_fire, 1)
                        self.assertTrue(all(row[2] == 0 for row in trace))
                        self.assertEqual(trace[fence_fire][2:7], (0, 7, 0, 7, 0))
                        self.assertEqual(trace[-1][3:7], (7, 8, 7, 8))
                        for address, value in guards:
                            self.assertEqual(result.slave.captured(address, len(value)), value)
                        observed[name] = (result.cycles, trace)
                self.assertEqual(observed["fence"][0], observed["nop"][0])
            finally:
                cosim_atlas.CosimCore = original
        finally:
            os.chdir(old_cwd)
            sys.path.remove(str(modelir))


if __name__ == "__main__":
    unittest.main()
