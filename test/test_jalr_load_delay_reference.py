"""Bounded selected-core scalar LW to JALR scheduling discriminator."""

from __future__ import annotations

import importlib.util
import os
import pathlib
import subprocess
import sys
import tempfile
import unittest

from selected_core_runtime import run_selected_program

ROOT = pathlib.Path(__file__).resolve().parents[1]
BIN = pathlib.Path(os.environ.get("ATLAS_OOT_BIN_DIR", ROOT / "build/bin"))
SOURCE = ROOT / "test/examples/jalr_load_delay.mlir"
ASSEMBLY = ROOT / "test/examples/jalr_load_delay.S"


def _assembler():
    root = os.environ.get("ATLAS_ASSEMBLER_ROOT")
    if not root:
        raise unittest.SkipTest("set ATLAS_ASSEMBLER_ROOT for selected assembler")
    spec = importlib.util.spec_from_file_location(
        "selected_load_jalr_assembler", pathlib.Path(root) / "assembler.py")
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _words() -> tuple[int, ...]:
    result = subprocess.run([str(BIN / "atlas-emit"), str(SOURCE)],
                            text=True, capture_output=True, check=True)
    return tuple(int(line, 16) for line in result.stdout.splitlines())


class JALRLoadDelayReferenceTest(unittest.TestCase):
    def test_typed_words_and_selected_source_timing(self) -> None:
        assembler = _assembler()
        words = _words()
        self.assertEqual(len(words), 14)
        self.assertEqual(words, tuple(assembler.assemble(ASSEMBLY.read_text())))
        self.assertEqual(words[5], assembler.DELAY_INSN(8))
        self.assertEqual(words[6], assembler.JALR(10, 1, 0))
        printed = subprocess.run([str(BIN / "atlas-opt"), str(SOURCE)],
                                 text=True, capture_output=True, check=True)
        reparsed = subprocess.run([str(BIN / "atlas-opt"), "-"],
                                  input=printed.stdout, text=True,
                                  capture_output=True, check=True)
        self.assertEqual(reparsed.stdout, printed.stdout)

        rtl_root = os.environ.get("ATLAS_RTL_ROOT")
        if not rtl_root:
            self.skipTest("set ATLAS_RTL_ROOT for selected timing source")
        core = (pathlib.Path(rtl_root) /
                "src/main/scala/atlas/scalar/ScalarCore.scala").read_text()
        lsu = (pathlib.Path(rtl_root) /
               "src/main/scala/atlas/lsu/LSU.scala").read_text()
        self.assertIn("Scalar loads have fixed 3-cycle latency", core)
        self.assertIn("val stall   = delay_stall || dma_wait_stall", core)
        self.assertIn("val memRespCapture = memLoadPending && io.scalarMemResp.valid", core)
        self.assertIn("val memRespArrived = memLoadRespValid", core)
        self.assertIn("io.scalarResp.valid := io.vmemScalarReadData.valid && scalarLoadPending", lsu)

    def test_llvm_rejects_memory_target_and_fence_does_not_discharge_proof(self) -> None:
        source = SOURCE.read_text()
        printed = subprocess.run([str(BIN / "atlas-opt"),
                                  "--convert-atlas-to-llvm", "-"],
                                 input=source, text=True, capture_output=True)
        self.assertNotEqual(printed.returncode, 0)
        self.assertIn("JALR target proof", printed.stderr)
        for replacement in ('cycles = 0 : i32', None):
            with self.subTest(replacement=replacement):
                if replacement is None:
                    candidate = source.replace(
                        '"atlas.delay"(%s5) {cycles = 8 : i32}',
                        '"atlas.fence"(%s5)')
                else:
                    candidate = source.replace('cycles = 8 : i32', replacement)
                self.assertNotEqual(candidate, source)
                run = subprocess.run([str(BIN / "atlas-opt"),
                                      "--convert-atlas-to-llvm", "-"],
                                     input=candidate, text=True, capture_output=True)
                self.assertNotEqual(run.returncode, 0)
                self.assertIn("JALR target proof", run.stderr)

    def test_selected_core_distinguishes_delay_from_fence_and_zero_delay(self) -> None:
        keys = ("ATLAS_ARC_MODEL", "ATLAS_ARC_STATE", "ATLAS_MODELIR_ROOT",
                "ATLAS_RTL_ROOT")
        if not all(os.environ.get(key) for key in keys):
            self.skipTest("set selected-source-linked ARC, ModeLIR, and RTL paths")
        model, state, modelir, _ = (pathlib.Path(os.environ[key]).resolve(strict=True)
                                   for key in keys)
        assembler = _assembler()
        words = _words()
        fence_source = SOURCE.read_text().replace(
            '"atlas.delay"(%s5) {cycles = 8 : i32}',
            '"atlas.fence"(%s5)')
        self.assertNotEqual(fence_source, SOURCE.read_text())
        with tempfile.TemporaryDirectory() as temporary:
            fence_path = pathlib.Path(temporary) / "typed_fence.mlir"
            fence_path.write_text(fence_source)
            fence_words = tuple(int(line, 16) for line in subprocess.check_output(
                [str(BIN / "atlas-emit"), str(fence_path)], text=True).splitlines())
        self.assertEqual(fence_words,
                         tuple(assembler.assemble(ASSEMBLY.read_text().replace(
                             "DELAY 8", "FENCE"))))
        self.assertEqual(fence_words[5], assembler.FENCE())
        self.assertEqual(fence_words[:5] + fence_words[6:], words[:5] + words[6:])
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
                for name, opcode, target, marker in (
                    ("delay8", assembler.DELAY_INSN(8), 11, 3),
                    ("delay0", assembler.DELAY_INSN(0), 8, 2),
                    ("fence", assembler.FENCE(), 8, 2),
                ):
                    with self.subTest(name=name):
                        program = list(fence_words if name == "fence" else words)
                        if name != "fence":
                            program[5] = opcode
                        trace: list[tuple[int, ...]] = []

                        def on_cycle(core: SelectedCore) -> None:
                            trace.append((
                                core.peek("scalar/pc_ctrl/io_s1_pc"),
                                core.peek("scalar/s1_fire"),
                                core.peek("scalar/regfile/regs_1"),
                                core.peek("scalar/memLoadPending"),
                                core.peek("lsu/io_scalarResp_valid"),
                                core.peek("scalar/memLoadRespValid"),
                                core.peek("scalar/regfile/regs_10"),
                                core.peek("scalar/regfile/regs_12"),
                                core.peek("csrfile/reg_dbg0"),
                            ))

                        result = run_selected_program(cosim_atlas,
                            model, state, program, preload=guards,
                            max_cycles=150, on_cycle=on_cycle)
                        self.assertTrue(result.halted)
                        self.assertEqual((result.reads, result.writes), (0, 0))
                        fired = [row for row in trace if row[0] == 6 and row[1] == 1]
                        self.assertEqual(len(fired), 1)
                        self.assertEqual(fired[0][2], target)
                        self.assertEqual(fired[0][3:6], (0, 0, 0) if name == "delay8"
                                         else (1, 1, 0))
                        self.assertEqual(trace[-1][2], 11)
                        self.assertEqual(trace[-1][6:9], (7, marker, marker))
                        for address, value in guards:
                            self.assertEqual(result.slave.captured(address, len(value)), value)
                        # The two paths halt at different program words. The
                        # stale path executes index 8; the loaded path starts
                        # at index 11 after the architectural delay slot.
                        self.assertEqual(trace[-1][0], 13 if marker == 3 else 10)
            finally:
                cosim_atlas.CosimCore = original
        finally:
            os.chdir(old_cwd)
            sys.path.remove(str(modelir))


if __name__ == "__main__":
    unittest.main()
