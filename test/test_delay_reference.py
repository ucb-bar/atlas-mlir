"""Bounded selected-core DELAY counter and frontend-effect observation."""

from __future__ import annotations

import importlib.util
import os
import pathlib
import subprocess
import sys
import tempfile
import unittest

from test_mxu_reference import _object_words

ROOT = pathlib.Path(__file__).resolve().parents[1]
BIN = pathlib.Path(os.environ.get("ATLAS_OOT_BIN_DIR", ROOT / "build/bin"))
SOURCE = ROOT / "test/examples/delay_timing.mlir"
ASSEMBLY = ROOT / "test/examples/delay_timing.S"
RTL_REVISION = "0079c0541111197741a231c002e3843fa6f545b2"


def _assembler():
    root = os.environ.get("ATLAS_ASSEMBLER_ROOT")
    if not root:
        raise unittest.SkipTest("set ATLAS_ASSEMBLER_ROOT for selected assembler")
    spec = importlib.util.spec_from_file_location(
        "selected_delay_assembler", pathlib.Path(root) / "assembler.py")
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _words() -> tuple[int, ...]:
    return tuple(int(line, 16) for line in subprocess.check_output(
        [str(BIN / "atlas-emit"), str(SOURCE)], text=True).splitlines())


class DelayReferenceTest(unittest.TestCase):
    def test_typed_words_match_selected_assembler_and_llvm_object(self) -> None:
        assembler = _assembler()
        with tempfile.TemporaryDirectory() as temporary:
            for count in (0, 4):
                with self.subTest(count=count):
                    path = pathlib.Path(temporary) / f"delay{count}.mlir"
                    path.write_text(SOURCE.read_text().replace(
                        'cycles = 4 : i32', f'cycles = {count} : i32'))
                    words = tuple(int(line, 16) for line in subprocess.check_output(
                        [str(BIN / "atlas-emit"), str(path)], text=True).splitlines())
                    assembly = ASSEMBLY.read_text().replace("DELAY 4", f"DELAY {count}")
                    self.assertEqual(len(words), 6)
                    self.assertEqual(words, tuple(assembler.assemble(assembly)))
                    self.assertEqual(words, _object_words(path))
                    self.assertEqual(words[2], assembler.DELAY_INSN(count))
        printed = subprocess.run([str(BIN / "atlas-opt"), str(SOURCE)],
                                 text=True, capture_output=True, check=True)
        reparsed = subprocess.run([str(BIN / "atlas-opt"), "-"],
                                  input=printed.stdout, text=True,
                                  capture_output=True, check=True)
        self.assertEqual(reparsed.stdout, printed.stdout)

    def test_typed_verifier_rejects_out_of_domain_counts(self) -> None:
        source = SOURCE.read_text()
        original = '"atlas.delay"(%s2) {cycles = 4 : i32}'
        self.assertIn(original, source)
        for count in (-1, 4096):
            with self.subTest(count=count):
                run = subprocess.run([str(BIN / "atlas-opt"), "-"],
                                     input=source.replace(original,
                                         f'"atlas.delay"(%s2) {{cycles = {count} : i32}}'),
                                     text=True, capture_output=True, check=False)
                self.assertNotEqual(run.returncode, 0)
                self.assertIn("cycles", run.stderr)

    def test_selected_core_stalls_successor_for_exact_counter_span(self) -> None:
        keys = ("ATLAS_ARC_MODEL", "ATLAS_ARC_STATE", "ATLAS_MODELIR_ROOT",
                "ATLAS_LLVM_BIN", "ATLAS_RTL_ROOT", "ATLAS_ASSEMBLER_ROOT")
        if not all(os.environ.get(key) for key in keys):
            self.skipTest("set selected-source-linked ARC, LLVM, ModeLIR, RTL and assembler paths")
        model, state, modelir, _, rtl, _ = (
            pathlib.Path(os.environ[key]).resolve(strict=True) for key in keys)
        self.assertEqual(subprocess.check_output(
            ["git", "-C", str(rtl), "rev-parse", "HEAD"], text=True).strip(), RTL_REVISION)
        core_source = (rtl / "src/main/scala/atlas/scalar/ScalarCore.scala").read_text()
        pc_source = (rtl / "src/main/scala/atlas/scalar/PcControl.scala").read_text()
        self.assertIn("val delay_stall    = delayCounter =/= 0.U", core_source)
        self.assertIn("delayCounter := dec.imm(11, 0).asUInt", core_source)
        self.assertIn("delayCounter := delayCounter - 1.U", core_source)
        self.assertIn("}.elsewhen(io.stall) {", pc_source)
        for name in ("scalar/delayCounter", "scalar/pc_ctrl/io_s1_pc",
                     "scalar/s1_fire", "csrfile/reg_dbg0", "csrfile/reg_dbg1"):
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
                observations: dict[int, tuple[int, list[tuple[int, ...]]]] = {}
                for count in (0, 4):
                    with self.subTest(count=count):
                        program = list(words)
                        program[2] = assembler.DELAY_INSN(count)
                        trace: list[tuple[int, ...]] = []

                        def on_cycle(core: SelectedCore) -> None:
                            trace.append((
                                core.peek("scalar/pc_ctrl/io_s1_pc"),
                                core.peek("scalar/s1_fire"),
                                core.peek("scalar/delayCounter"),
                                core.peek("csrfile/reg_dbg0"),
                                core.peek("csrfile/reg_dbg1"),
                                core.peek("scalar/regfile/regs_2"),
                            ))

                        result = cosim_atlas.run_program(
                            model, state, program, preload=guards,
                            max_cycles=100, on_cycle=on_cycle)
                        self.assertTrue(result.halted)
                        self.assertEqual((result.reads, result.writes), (0, 0))
                        for pc in (2, 3, 4):
                            self.assertEqual(sum(row[0] == pc and row[1] == 1 for row in trace), 1)
                        delay_fire = next(i for i, row in enumerate(trace)
                                          if row[0] == 2 and row[1] == 1)
                        successor_fire = next(i for i, row in enumerate(trace)
                                              if row[0] == 3 and row[1] == 1)
                        self.assertEqual(successor_fire - delay_fire, count + 1)
                        held = trace[delay_fire + 1:successor_fire]
                        self.assertEqual([row[2] for row in held], list(range(count, 0, -1)))
                        self.assertTrue(all(row[0] == 3 and row[1] == 0 for row in held))
                        self.assertTrue(all(row[3:5] == (7, 0) for row in held))
                        self.assertEqual(trace[-1][3:6], (7, 8, 8))
                        for address, value in guards:
                            self.assertEqual(result.slave.captured(address, len(value)), value)
                        observations[count] = (result.cycles, trace)
                self.assertEqual(observations[4][0] - observations[0][0], 4)
            finally:
                cosim_atlas.CosimCore = original
        finally:
            os.chdir(old_cwd)
            sys.path.remove(str(modelir))


if __name__ == "__main__":
    unittest.main()
