"""Bounded, bit-exact VLI.ALL check independent of npu_model execution."""

from __future__ import annotations

import importlib.util
import os
import pathlib
import subprocess
import sys
import tempfile
import unittest

from test_vpu_relu_reference import _emitted, _object_words

from selected_core_runtime import run_selected_program

ROOT = pathlib.Path(__file__).resolve().parents[1]
BIN = pathlib.Path(os.environ.get("ATLAS_OOT_BIN_DIR", ROOT / "build/bin"))
SOURCE = ROOT / "test/examples/vli_all_pair.mlir"
ASSEMBLY = ROOT / "test/examples/vli_all_pair.S"
DEFAULT_IMMEDIATE = 0x3F80


def _with_immediate(code: int) -> str:
    source = SOURCE.read_text()
    old = f'mode = "all", dst = 4 : i32, immediate = {DEFAULT_IMMEDIATE} : i32'
    new = f'mode = "all", dst = 4 : i32, immediate = {code} : i32'
    assert old in source
    return source.replace(old, new)


def _expected_raw_pair(code: int) -> bytes:
    # 32 x 32 BF16 storage cells. This is a raw-bit fill, not a FP conversion.
    return code.to_bytes(2, "little") * 1024


class VliAllReferenceTest(unittest.TestCase):
    def test_raw_reference_covers_both_halves_without_float_conversion(self) -> None:
        for code in (0x3F80, 0xBF80, 0x8000, 0x7FC1):
            with self.subTest(code=f"0x{code:04x}"):
                expected = _expected_raw_pair(code)
                self.assertEqual(len(expected), 2048)
                self.assertEqual(expected[:2], code.to_bytes(2, "little"))
                self.assertEqual(expected[1024:1026], code.to_bytes(2, "little"))
                self.assertEqual(expected[-2:], code.to_bytes(2, "little"))

    def test_typed_stream_matches_selected_assembler_and_llvm_object(self) -> None:
        selected_root = os.environ.get("ATLAS_ASSEMBLER_ROOT")
        if not selected_root:
            self.skipTest("set ATLAS_ASSEMBLER_ROOT for independent selected assembler")
        path = pathlib.Path(selected_root) / "assembler.py"
        spec = importlib.util.spec_from_file_location("selected_vli_assembler", path)
        assert spec and spec.loader
        assembler = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(assembler)
        emitted = _emitted(SOURCE)
        self.assertEqual(len(emitted), 36)
        self.assertEqual(emitted, tuple(assembler.assemble(ASSEMBLY.read_text())))
        self.assertEqual(emitted[17], assembler.VLI_ALL(4, DEFAULT_IMMEDIATE))
        self.assertEqual(_object_words(SOURCE), emitted)
        parsed = subprocess.run([str(BIN / "atlas-opt"), str(SOURCE)],
                                text=True, capture_output=True, check=True)
        self.assertEqual(parsed.stdout.count('"atlas.vli"'), 1)
        reparsed = subprocess.run([str(BIN / "atlas-opt"), "-"],
                                  input=parsed.stdout, text=True,
                                  capture_output=True, check=True)
        self.assertEqual(reparsed.stdout, parsed.stdout)

    def test_typed_legality_rejects_wrong_pair_or_immediate(self) -> None:
        source = SOURCE.read_text()
        old = f'mode = "all", dst = 4 : i32, immediate = {DEFAULT_IMMEDIATE} : i32'
        for replacement in (
            'mode = "all", dst = 5 : i32, immediate = 16256 : i32',
            'mode = "all", dst = 63 : i32, immediate = 16256 : i32',
            'mode = "all", dst = 4 : i32, immediate = 65536 : i32',
            'mode = "unknown", dst = 4 : i32, immediate = 16256 : i32',
        ):
            with self.subTest(replacement=replacement):
                mutated = source.replace(old, replacement)
                self.assertNotEqual(mutated, source)
                result = subprocess.run([str(BIN / "atlas-opt"), "-"],
                                        input=mutated, text=True,
                                        capture_output=True, check=False)
                self.assertNotEqual(result.returncode, 0)
                self.assertTrue(result.stderr)

    def test_selected_core_fills_both_halves_as_raw_bits(self) -> None:
        keys = ("ATLAS_ARC_MODEL", "ATLAS_ARC_STATE", "ATLAS_MODELIR_ROOT",
                "ATLAS_LLVM_BIN")
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
                for phase, code in enumerate((0x3F80, 0xBF80, 0x8000, 0x7FC1)):
                    with self.subTest(code=f"0x{code:04x}"):
                        noise = bytes((i * 17 + phase * 29) & 0xFF
                                      for i in range(2048))
                        expected = _expected_raw_pair(code)
                        with tempfile.TemporaryDirectory() as temporary:
                            source = pathlib.Path(temporary) / "vli_all.mlir"
                            source.write_text(_with_immediate(code))
                            words = _object_words(source)
                            self.assertEqual(words, _emitted(source))
                        result = run_selected_program(cosim_atlas,
                            model, state, words,
                            preload=[(0x90000000, noise[:1024]),
                                     (0x90000400, noise[1024:]),
                                     (0x90000800, b"\xA5" * 1024),
                                     (0x90000C00, b"\xA5" * 1024),
                                     (0x90001000, b"\x5A" * 32)],
                            max_cycles=8000,
                        )
                        observed = (result.slave.captured(0x90000800, 1024) +
                                    result.slave.captured(0x90000C00, 1024))
                        self.assertTrue(result.halted)
                        self.assertEqual((result.reads, result.writes), (64, 64))
                        self.assertEqual(observed, expected)
                        self.assertEqual(result.slave.captured(0x90000000, 2048), noise)
                        self.assertEqual(result.slave.captured(0x90001000, 32), b"\x5A" * 32)
            finally:
                cosim_atlas.CosimCore = original
        finally:
            os.chdir(old_cwd)
            sys.path.remove(str(modelir))


if __name__ == "__main__":
    unittest.main()
