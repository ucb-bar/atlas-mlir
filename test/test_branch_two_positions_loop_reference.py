"""Selected Atlas one-slot branch and backward-loop source-linked evidence."""

from __future__ import annotations

import importlib.util
import json
import os
import pathlib
import subprocess
import sys
import tempfile
import unittest

from test_mxu_reference import _object_words

from selected_core_runtime import run_selected_program

ROOT = pathlib.Path(__file__).resolve().parents[1]
BIN = pathlib.Path(os.environ.get("ATLAS_OOT_BIN_DIR", ROOT / "build/bin"))
SOURCE = ROOT / "test/examples/branch_two_positions_loop.mlir"
ASSEMBLY = ROOT / "test/examples/branch_two_positions_loop.S"


def _emitted(source: pathlib.Path = SOURCE) -> tuple[int, ...]:
    run = subprocess.run([str(BIN / "atlas-emit"), str(source)],
                         text=True, capture_output=True, check=True)
    return tuple(int(line, 16) for line in run.stdout.splitlines())


class BranchTwoPositionsLoopReferenceTest(unittest.TestCase):
    def test_typed_parser_assembler_and_llvm_object_agree(self) -> None:
        root = os.environ.get("ATLAS_ASSEMBLER_ROOT")
        if not root:
            self.skipTest("set ATLAS_ASSEMBLER_ROOT for selected assembler")
        path = pathlib.Path(root) / "assembler.py"
        spec = importlib.util.spec_from_file_location("selected_loop_assembler", path)
        assert spec and spec.loader
        assembler = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(assembler)
        words = _emitted()
        self.assertEqual(len(words), 14)
        self.assertEqual(words, tuple(assembler.assemble(ASSEMBLY.read_text())))
        self.assertEqual(words[2], assembler.BEQ(0, 0, 3))
        self.assertEqual(words[10], assembler.BLT(12, 14, -1))
        self.assertEqual(_object_words(SOURCE), words)
        parsed = subprocess.run([str(BIN / "atlas-opt"), str(SOURCE)],
                                text=True, capture_output=True, check=True)
        self.assertEqual(parsed.stdout.count('"atlas.branch"'), 2)
        reparsed = subprocess.run([str(BIN / "atlas-opt"), "-"], input=parsed.stdout,
                                  text=True, capture_output=True, check=True)
        self.assertEqual(reparsed.stdout, parsed.stdout)

    def test_invalid_offsets_and_redirect_in_slot_are_rejected(self) -> None:
        source = SOURCE.read_text()
        for original, replacement, command, diagnostic in (
            ('offset_bytes = 6 : i32', 'offset_bytes = 5 : i32', "atlas-opt", "even"),
            ('offset_bytes = -2 : i32', 'offset_bytes = -3 : i32', "atlas-opt", "even"),
            ('address = 3089 : i32', 'address = 3090 : i32', "atlas-opt", "address"),
            ('offset_bytes = 6 : i32', 'offset_bytes = 2000 : i32',
             "convert", "escapes"),
            ('offset_bytes = -2 : i32', 'offset_bytes = -2000 : i32',
             "convert", "escapes"),
            ('"atlas.csr"(%s3) {kind = "rrw", dst = 0 : i32, source = 10 : i32, address = 3088 : i32}',
             '"atlas.branch"(%s3) {kind = "beq", lhs = 0 : i32, rhs = 0 : i32, offset_bytes = 2 : i32}',
             "convert", "delay slot"),
        ):
            with self.subTest(replacement=replacement):
                changed = source.replace(original, replacement)
                self.assertNotEqual(changed, source)
                cmd = [str(BIN / "atlas-opt")]
                if command == "convert":
                    cmd.append("--convert-atlas-to-llvm")
                result = subprocess.run(cmd + ["-"], input=changed,
                                        text=True, capture_output=True, check=False)
                self.assertNotEqual(result.returncode, 0)
                self.assertIn(diagnostic, result.stderr)

    def test_selected_core_observes_first_slot_skips_second_and_loops(self) -> None:
        keys = ("ATLAS_ARC_MODEL", "ATLAS_ARC_STATE", "ATLAS_MODELIR_ROOT", "ATLAS_LLVM_BIN")
        if not all(os.environ.get(key) for key in keys):
            self.skipTest("set selected-source-linked ARC, ModeLIR, and LLVM paths")
        model, state, modelir = (pathlib.Path(os.environ[key]).resolve(strict=True)
                                 for key in keys[:3])
        names = {entry["name"] for entry in json.loads(state.read_text())[0]["states"]}
        signals = ("csrfile/reg_dbg0", "csrfile/reg_dbg1",
                   "scalar/regfile/regs_10", "scalar/regfile/regs_11",
                   "scalar/regfile/regs_12", "scalar/regfile/regs_13",
                   "scalar/regfile/regs_14", "scalar/regfile/regs_15")
        self.assertTrue(set(signals).issubset(names))
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
            observed: dict[str, int] = {}

            def on_cycle(core: SelectedCore) -> None:
                for name in signals:
                    observed[name] = core.peek(name)

            preload = [(0x90000000, bytes(range(64))),
                       (0x90000400, b"\x5A" * 64)]
            try:
                words = _object_words(SOURCE)
                result = run_selected_program(cosim_atlas, model, state, words,
                                                 preload=preload, max_cycles=200,
                                                 on_cycle=on_cycle)
                selected = dict(observed)
                with tempfile.TemporaryDirectory() as temporary:
                    changed = pathlib.Path(temporary) / "redirect_to_second.mlir"
                    mutated = SOURCE.read_text().replace(
                        'offset_bytes = 6 : i32', 'offset_bytes = 4 : i32')
                    self.assertNotEqual(mutated, SOURCE.read_text())
                    changed.write_text(mutated)
                    changed_words = _object_words(changed)
                    self.assertNotEqual(changed_words[2], words[2])
                observed.clear()
                mutant = run_selected_program(cosim_atlas, model, state, changed_words,
                                                 preload=preload, max_cycles=200,
                                                 on_cycle=on_cycle)
                changed_state = dict(observed)
            finally:
                cosim_atlas.CosimCore = original
        finally:
            os.chdir(old_cwd)
            sys.path.remove(str(modelir))
        self.assertTrue(result.halted)
        self.assertLess(result.cycles, 200)
        self.assertEqual((result.reads, result.writes), (0, 0))
        self.assertEqual(selected["csrfile/reg_dbg0"], 17)
        self.assertEqual(selected["csrfile/reg_dbg1"], 0)
        self.assertEqual((selected["scalar/regfile/regs_12"],
                          selected["scalar/regfile/regs_13"],
                          selected["scalar/regfile/regs_15"]), (3, 3, 1))
        for address, value in preload:
            self.assertEqual(result.slave.captured(address, len(value)), value)
        self.assertTrue(mutant.halted)
        self.assertEqual(changed_state["csrfile/reg_dbg0"], 17)
        self.assertEqual(changed_state["csrfile/reg_dbg1"], 34)
        for address, value in preload:
            self.assertEqual(mutant.slave.captured(address, len(value)), value)


if __name__ == "__main__":
    unittest.main()
