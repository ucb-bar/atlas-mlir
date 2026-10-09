"""Bounded selected-core checks for LUI and AUIPC word-PC semantics."""

from __future__ import annotations

import importlib.util
import os
import pathlib
import subprocess
import sys
import tempfile
import unittest

from test_scalar_alu_reference import MASK, _emitted, _object_words

from selected_core_runtime import run_selected_program


PANELS = (0x90000, 0xFFFFF)


def _program(kind: str, immediate: int) -> tuple[str, str]:
    source = "\n".join((
        "module {",
        '  %s0 = "atlas.start"() : () -> !atlas.state',
        '  %s1 = "atlas.alu_imm"(%s0) {kind = "addi", dst = 5 : i32, '
        'src = 0 : i32, immediate = 7 : i32} : (!atlas.state) -> !atlas.state',
        f'  %s2 = "atlas.upper"(%s1) {{kind = "{kind}", dst = 12 : i32, '
        f'immediate = {immediate} : i32}} : (!atlas.state) -> !atlas.state',
        '  %s3 = "atlas.trap"(%s2) {kind = "ecall"} : (!atlas.state) -> !atlas.state',
        "}",
        "",
    ))
    assembly = f"ADDI x5, x0, 7\n{kind.upper()} x12, {immediate}\nECALL\n"
    return source, assembly


class ScalarUpperReferenceTest(unittest.TestCase):
    def test_typed_words_match_selected_assembler_and_llvm_object(self) -> None:
        assembler_root = os.environ.get("ATLAS_ASSEMBLER_ROOT")
        if not assembler_root:
            self.skipTest("set ATLAS_ASSEMBLER_ROOT for selected assembler")
        spec = importlib.util.spec_from_file_location(
            "atlas_selected_assembler", pathlib.Path(assembler_root) / "assembler.py",
        )
        assert spec is not None and spec.loader is not None
        assembler = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(assembler)
        with tempfile.TemporaryDirectory() as temporary:
            directory = pathlib.Path(temporary)
            path = directory / "upper.mlir"
            for kind in ("lui", "auipc"):
                for immediate in PANELS:
                    with self.subTest(kind=kind, immediate=immediate):
                        source, assembly = _program(kind, immediate)
                        path.write_text(source)
                        words = _emitted(path)
                        self.assertEqual(words, tuple(assembler.assemble(assembly)))
                        self.assertEqual(words, _object_words(path, len(words), directory))

    def test_selected_core_matches_independent_upper_and_pc_semantics(self) -> None:
        keys = ("ATLAS_ARC_MODEL", "ATLAS_ARC_STATE", "ATLAS_MODELIR_ROOT", "ATLAS_RTL_ROOT")
        if not all(os.environ.get(key) for key in keys):
            self.skipTest("set selected-source-linked ARC model, state, ModeLIR and RTL paths")
        model, state, modelir, rtl = (pathlib.Path(os.environ[key]).resolve(strict=True) for key in keys)
        self.assertEqual(
            subprocess.check_output(["git", "-C", str(rtl), "rev-parse", "HEAD"], text=True).strip(),
            "0079c0541111197741a231c002e3843fa6f545b2",
        )
        self.assertIn("scalar/regfile/regs_12", state.read_text())
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
                with tempfile.TemporaryDirectory() as temporary:
                    path = pathlib.Path(temporary) / "upper.mlir"
                    for kind in ("lui", "auipc"):
                        for immediate in PANELS:
                            with self.subTest(kind=kind, immediate=immediate):
                                path.write_text(_program(kind, immediate)[0])
                                observed: dict[str, int] = {}

                                def on_cycle(core: SelectedCore) -> None:
                                    observed["x12"] = core.peek("scalar/regfile/regs_12")

                                result = run_selected_program(cosim_atlas,
                                    model, state, _emitted(path), max_cycles=80, on_cycle=on_cycle,
                                )
                                self.assertTrue(result.halted)
                                # ScalarCore feeds AUIPC the execute-stage IMEM word index.
                                # The preceding ADDI occupies word zero, so AUIPC adds one.
                                expected = ((immediate << 12) + (1 if kind == "auipc" else 0)) & MASK
                                self.assertEqual(observed["x12"] & MASK, expected)
            finally:
                cosim_atlas.CosimCore = original
        finally:
            os.chdir(old_cwd)
            sys.path.remove(str(modelir))


if __name__ == "__main__":
    unittest.main()
