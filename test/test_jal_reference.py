"""Selected direct JAL target, link, and one-delay-slot observation."""

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
SOURCE = ROOT / "test/examples/jal_direct_target.mlir"
ASSEMBLY = ROOT / "test/examples/jal_direct_target.S"
RTL_REVISION = "0079c0541111197741a231c002e3843fa6f545b2"


def _emitted() -> tuple[int, ...]:
    return tuple(int(line, 16) for line in subprocess.check_output(
        [str(BIN / "atlas-emit"), str(SOURCE)], text=True).splitlines())


class JALReferenceTest(unittest.TestCase):
    def test_typed_words_match_selected_assembler_and_llvm_object(self) -> None:
        root = os.environ.get("ATLAS_ASSEMBLER_ROOT")
        if not root:
            self.skipTest("set ATLAS_ASSEMBLER_ROOT for selected assembler")
        spec = importlib.util.spec_from_file_location(
            "selected_jal_assembler", pathlib.Path(root) / "assembler.py")
        assert spec and spec.loader
        assembler = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(assembler)
        words = _emitted()
        self.assertEqual(len(words), 8)
        self.assertEqual(words, tuple(assembler.assemble(ASSEMBLY.read_text())))
        self.assertEqual(words, _object_words(SOURCE))
        self.assertEqual(words[1], assembler.JAL(10, 4))
        printed = subprocess.run([str(BIN / "atlas-opt"), str(SOURCE)],
                                 text=True, capture_output=True, check=True)
        reparsed = subprocess.run([str(BIN / "atlas-opt"), "-"],
                                  input=printed.stdout, text=True,
                                  capture_output=True, check=True)
        self.assertEqual(reparsed.stdout, printed.stdout)

    def test_typed_legality_and_llvm_block_reject_invalid_cases(self) -> None:
        source = SOURCE.read_text()
        operation = 'kind = "jal", dst = 10 : i32, base = 0 : i32, offset = 8 : i32'
        self.assertIn(operation, source)
        for replacement, tool, message in (
            ('kind = "jal", dst = 32 : i32, base = 0 : i32, offset = 8 : i32',
             "atlas-opt", "dst"),
            ('kind = "jal", dst = 10 : i32, base = 1 : i32, offset = 8 : i32',
             "atlas-opt", "base"),
            ('kind = "jal", dst = 10 : i32, base = 0 : i32, offset = 7 : i32',
             "atlas-opt", "even"),
            ('kind = "jal", dst = 10 : i32, base = 0 : i32, offset = 1048576 : i32',
             "atlas-opt", "offset"),
            ('kind = "jal", dst = 10 : i32, base = 0 : i32, offset = 16 : i32',
             "atlas-opt --convert-atlas-to-llvm", "escapes"),
        ):
            with self.subTest(replacement=replacement):
                command = [str(BIN / "atlas-opt")]
                if "--convert-atlas-to-llvm" in tool:
                    command.append("--convert-atlas-to-llvm")
                command.append("-")
                run = subprocess.run(command, input=source.replace(operation, replacement),
                                     text=True, capture_output=True, check=False)
                self.assertNotEqual(run.returncode, 0)
                self.assertIn(message, run.stderr)

    def test_selected_core_executes_direct_target_link_and_one_slot(self) -> None:
        keys = ("ATLAS_ARC_MODEL", "ATLAS_ARC_STATE", "ATLAS_MODELIR_ROOT",
                "ATLAS_LLVM_BIN", "ATLAS_RTL_ROOT", "ATLAS_ASSEMBLER_ROOT")
        if not all(os.environ.get(key) for key in keys):
            self.skipTest("set selected-source-linked ARC, LLVM, ModeLIR, RTL and assembler paths")
        model, state, modelir, _, rtl, assembler_root = (
            pathlib.Path(os.environ[key]).resolve(strict=True) for key in keys)
        self.assertEqual(subprocess.check_output(
            ["git", "-C", str(rtl), "rev-parse", "HEAD"], text=True).strip(), RTL_REVISION)
        scalar = (rtl / "src/main/scala/atlas/scalar/ScalarCore.scala").read_text()
        pc = (rtl / "src/main/scala/atlas/scalar/PcControl.scala").read_text()
        self.assertIn("val jal_target    = (s1_pc.asSInt + (dec.imm >> 1)).asUInt", scalar)
        self.assertIn("val link_value = s1_pc + 1.U", scalar)
        self.assertIn("val delay_slot_pending", pc)
        self.assertIn("scalar/regfile/regs_10", state.read_text())
        spec = importlib.util.spec_from_file_location(
            "selected_jal_assembler_runtime", assembler_root / "assembler.py")
        assert spec and spec.loader
        assembler = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(assembler)
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
                for displacement_words, expected_marker in ((4, 3), (2, 6)):
                    with self.subTest(displacement_words=displacement_words):
                        program = list(words)
                        program[1] = assembler.JAL(10, displacement_words)
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
                                         1 if displacement_words == 4 else 4)
                        self.assertEqual(observed["scalar/regfile/regs_11"], expected_marker)
                        self.assertEqual(observed["csrfile/reg_dbg0"], expected_marker)
                        for address, value in guards:
                            self.assertEqual(result.slave.captured(address, len(value)), value)
            finally:
                cosim_atlas.CosimCore = original
        finally:
            os.chdir(old_cwd)
            sys.path.remove(str(modelir))


if __name__ == "__main__":
    unittest.main()
