"""Selected word-index JALR target, link, and delay-slot observation."""

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
SOURCE = ROOT / "test/examples/jalr_word_target.mlir"
ASSEMBLY = ROOT / "test/examples/jalr_word_target.S"


def _emitted() -> tuple[int, ...]:
    return tuple(int(line, 16) for line in subprocess.check_output(
        [str(BIN / "atlas-emit"), str(SOURCE)], text=True).splitlines())


class JALRReferenceTest(unittest.TestCase):
    def test_typed_words_match_selected_assembler_and_llvm_object(self) -> None:
        root = os.environ.get("ATLAS_ASSEMBLER_ROOT")
        if not root:
            self.skipTest("set ATLAS_ASSEMBLER_ROOT for selected assembler")
        spec = importlib.util.spec_from_file_location(
            "selected_jalr_assembler", pathlib.Path(root) / "assembler.py")
        assert spec and spec.loader
        assembler = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(assembler)
        words = _emitted()
        self.assertEqual(len(words), 8)
        self.assertEqual(words, tuple(assembler.assemble(ASSEMBLY.read_text())))
        self.assertEqual(words, _object_words(SOURCE))
        self.assertEqual(words[1], assembler.JALR(10, 1, -1))
        printed = subprocess.run([str(BIN / "atlas-opt"), str(SOURCE)],
                                 text=True, capture_output=True, check=True)
        reparsed = subprocess.run([str(BIN / "atlas-opt"), "-"],
                                  input=printed.stdout, text=True,
                                  capture_output=True, check=True)
        self.assertEqual(reparsed.stdout, printed.stdout)

    def test_typed_legality_and_target_proof_reject_invalid_cases(self) -> None:
        source = SOURCE.read_text()
        for original, replacement in (
            ('kind = "jalr", dst = 10 : i32, base = 1 : i32, offset = -1 : i32',
             'kind = "jalr", dst = 32 : i32, base = 1 : i32, offset = -1 : i32'),
            ('kind = "jalr", dst = 10 : i32, base = 1 : i32, offset = -1 : i32',
             'kind = "jalr", dst = 10 : i32, base = 32 : i32, offset = -1 : i32'),
            ('kind = "jalr", dst = 10 : i32, base = 1 : i32, offset = -1 : i32',
             'kind = "jalr", dst = 10 : i32, base = 1 : i32, offset = -2049 : i32'),
        ):
            with self.subTest(replacement=replacement):
                self.assertIn(original, source)
                run = subprocess.run([str(BIN / "atlas-opt"), "-"],
                                     input=source.replace(original, replacement),
                                     text=True, capture_output=True, check=False)
                self.assertNotEqual(run.returncode, 0)
                self.assertTrue(run.stderr)
        for original, replacement, message in (
            ('dst = 1 : i32, src = 0 : i32, immediate = 6 : i32',
             'dst = 2 : i32, src = 0 : i32, immediate = 6 : i32',
             'no proven constant'),
            ('dst = 1 : i32, src = 0 : i32, immediate = 6 : i32',
             'dst = 1 : i32, src = 0 : i32, immediate = 20 : i32',
             'escapes'),
        ):
            with self.subTest(message=message):
                self.assertIn(original, source)
                run = subprocess.run([str(BIN / "atlas-opt"),
                                      "--convert-atlas-to-llvm", "-"],
                                     input=source.replace(original, replacement),
                                     text=True, capture_output=True, check=False)
                self.assertNotEqual(run.returncode, 0)
                self.assertIn(message, run.stderr)
        async_load = source.replace(
            '"atlas.alu_imm"(%s0) {kind = "addi", dst = 1 : i32, src = 0 : i32, immediate = 6 : i32}',
            '"atlas.scalar_load"(%s0) {kind = "lw", dst = 1 : i32, base = 0 : i32, offset = 0 : i32}')
        self.assertNotEqual(async_load, source)
        run = subprocess.run([str(BIN / "atlas-opt"), "--convert-atlas-to-llvm", "-"],
                             input=async_load, text=True, capture_output=True, check=False)
        self.assertNotEqual(run.returncode, 0)
        self.assertIn("asynchronous scalar load", run.stderr)

        first = ('  %s1 = "atlas.alu_imm"(%s0) {kind = "addi", dst = 1 : i32, '
                 'src = 0 : i32, immediate = 6 : i32}')
        redirected = source.replace(first,
            '  %branch = "atlas.branch"(%s0) {kind = "beq", lhs = 0 : i32, '
            'rhs = 0 : i32, offset_bytes = 6 : i32} : (!atlas.state) -> !atlas.state\n'
            '  %slot = "atlas.alu_imm"(%branch) {kind = "addi", dst = 2 : i32, '
            'src = 0 : i32, immediate = 1 : i32} : (!atlas.state) -> !atlas.state\n'
            '  %s1 = "atlas.alu_imm"(%slot) {kind = "addi", dst = 1 : i32, '
            'src = 0 : i32, immediate = 6 : i32}')
        self.assertNotEqual(redirected, source)
        run = subprocess.run([str(BIN / "atlas-opt"), "--convert-atlas-to-llvm", "-"],
                             input=redirected, text=True, capture_output=True, check=False)
        self.assertNotEqual(run.returncode, 0)
        self.assertIn("earlier control flow", run.stderr)

    def test_selected_core_executes_odd_word_target_link_and_one_slot(self) -> None:
        keys = ("ATLAS_ARC_MODEL", "ATLAS_ARC_STATE", "ATLAS_MODELIR_ROOT",
                "ATLAS_LLVM_BIN", "ATLAS_RTL_ROOT")
        if not all(os.environ.get(key) for key in keys):
            self.skipTest("set selected-source-linked ARC, LLVM, ModeLIR, and RTL paths")
        model, state, modelir, _, rtl = (pathlib.Path(os.environ[key]).resolve(strict=True)
                                        for key in keys)
        scalar = (rtl / "src/main/scala/atlas/scalar/ScalarCore.scala").read_text()
        pc = (rtl / "src/main/scala/atlas/scalar/PcControl.scala").read_text()
        self.assertIn("val jalr_target   = (rs1_data.asSInt + dec.imm).asUInt", scalar)
        self.assertIn("val link_value = s1_pc + 1.U", scalar)
        self.assertIn("val delay_slot_pending", pc)
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
                for odd_target in (True, False):
                    with self.subTest(odd_target=odd_target):
                        program = list(words)
                        if not odd_target:
                            # x1=5 with offset -1 reaches even word 4;
                            # executing that word changes the checked marker.
                            program[0] = 0x00500093
                        observed: dict[str, int] = {}

                        def on_cycle(core: SelectedCore) -> None:
                            for name in ("scalar/regfile/regs_10",
                                         "scalar/regfile/regs_12",
                                         "scalar/regfile/regs_11",
                                         "csrfile/reg_dbg0"):
                                observed[name] = core.peek(name) & 0xFFFFFFFF

                        result = run_selected_program(cosim_atlas,
                            model, state, program, preload=guards,
                            max_cycles=100, on_cycle=on_cycle)
                        self.assertTrue(result.halted)
                        self.assertEqual((result.reads, result.writes), (0, 0))
                        self.assertEqual(observed["scalar/regfile/regs_10"], 2)
                        self.assertEqual(observed["scalar/regfile/regs_12"],
                                         1 if odd_target else 3)
                        self.assertEqual(observed["scalar/regfile/regs_11"],
                                         3 if odd_target else 5)
                        self.assertEqual(observed["csrfile/reg_dbg0"],
                                         3 if odd_target else 5)
                        for address, value in guards:
                            self.assertEqual(result.slave.captured(address, len(value)), value)
            finally:
                cosim_atlas.CosimCore = original
        finally:
            os.chdir(old_cwd)
            sys.path.remove(str(modelir))


if __name__ == "__main__":
    unittest.main()
