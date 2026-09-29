"""Bit-preserving XLU transpose through typed OOT, LLVM, and selected core."""

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
SOURCE = ROOT / "test/examples/xlu_transpose.mlir"
ASSEMBLY = ROOT / "test/examples/xlu_transpose.S"


def _emitted(source: pathlib.Path = SOURCE) -> tuple[int, ...]:
    run = subprocess.run([str(BIN / "atlas-emit"), str(source)],
                         capture_output=True, text=True, check=True)
    return tuple(int(line, 16) for line in run.stdout.splitlines())


def _transpose(source: bytes) -> bytes:
    if len(source) != 1024:
        raise ValueError("XLU reference requires exactly one 32x32 byte tile")
    return bytes(source[32 * row + col] for col in range(32) for row in range(32))


def _finite_tile() -> bytes:
    values = (0x38, 0xB8, 0x40, 0xC0, 0x18, 0x98, 0x3F, 0xBF)
    return bytes(values[(row * 5 + col * 11 + row * col) % len(values)]
                 for row in range(32) for col in range(32))


class XLUReferenceTest(unittest.TestCase):
    def test_selected_assembler_and_llvm_object_words(self) -> None:
        root = os.environ.get("ATLAS_ASSEMBLER_ROOT")
        if not root:
            self.skipTest("set ATLAS_ASSEMBLER_ROOT for selected assembler")
        path = pathlib.Path(root) / "assembler.py"
        spec = importlib.util.spec_from_file_location("selected_xlu_assembler", path)
        assert spec and spec.loader
        assembler = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(assembler)
        emitted = _emitted()
        self.assertEqual(len(emitted), 29)
        self.assertEqual(emitted, tuple(assembler.assemble(ASSEMBLY.read_text())))
        self.assertEqual(emitted[10], assembler.VTRPOSE_XLU(8, 4))
        self.assertEqual(emitted[10] & (0x3F << 19), 0)  # reserved vs2 field
        self.assertEqual(_object_words(SOURCE), emitted)
        printed = subprocess.run([str(BIN / "atlas-opt"), str(SOURCE)],
                                 capture_output=True, text=True, check=True)
        self.assertEqual(printed.stdout.count('"atlas.xlu_transpose"'), 1)
        reparsed = subprocess.run([str(BIN / "atlas-opt"), "-"],
                                  input=printed.stdout, capture_output=True,
                                  text=True, check=True)
        self.assertEqual(reparsed.stdout, printed.stdout)

    def test_invalid_register_fields_are_rejected(self) -> None:
        source = SOURCE.read_text()
        for original, replacement in (
            ('dst = 8 : i32, src = 4 : i32', 'dst = 64 : i32, src = 4 : i32'),
            ('dst = 8 : i32, src = 4 : i32', 'dst = 8 : i32, src = -1 : i32'),
        ):
            with self.subTest(replacement=replacement):
                changed = source.replace(original, replacement)
                self.assertNotEqual(changed, source)
                run = subprocess.run([str(BIN / "atlas-opt"), "-"], input=changed,
                                     capture_output=True, text=True, check=False)
                self.assertNotEqual(run.returncode, 0)
                self.assertTrue(run.stderr)

    def test_selected_core_transposes_non_symmetric_bytes_and_preserves_source(self) -> None:
        keys = ("ATLAS_ARC_MODEL", "ATLAS_ARC_STATE", "ATLAS_MODELIR_ROOT", "ATLAS_LLVM_BIN")
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
                cases = (("all_bytes", bytes(range(256)) * 4, False),
                         ("finite", _finite_tile(), False),
                         ("in_place", _finite_tile(), True))
                for name, source, in_place in cases:
                    with self.subTest(name=name):
                        expected = _transpose(source)
                        self.assertNotEqual(expected, source)
                        if in_place:
                            with tempfile.TemporaryDirectory() as temporary:
                                changed = pathlib.Path(temporary) / "in_place.mlir"
                                original_source = SOURCE.read_text()
                                mutated = original_source.replace(
                                    '"atlas.xlu_transpose"(%s10) {dst = 8 : i32, src = 4 : i32}',
                                    '"atlas.xlu_transpose"(%s10) {dst = 4 : i32, src = 4 : i32}')
                                mutated = mutated.replace(
                                    '"atlas.vstore"(%s13) {src = 8 : i32, base = 8 : i32',
                                    '"atlas.vstore"(%s13) {src = 4 : i32, base = 8 : i32')
                                self.assertNotEqual(mutated, original_source)
                                changed.write_text(mutated)
                                words = _object_words(changed)
                        else:
                            words = _object_words(SOURCE)
                        result = cosim_atlas.run_program(
                            model, state, words,
                            preload=[(0x90000000, source), (0x90000400, b"\xA5" * 1024),
                                     (0x90000800, b"\xA5" * 1024),
                                     (0x90000C00, b"\x5A" * 32)],
                            max_cycles=5000,
                        )
                        self.assertTrue(result.halted)
                        self.assertEqual((result.reads, result.writes), (32, 64))
                        self.assertEqual(result.slave.captured(0x90000400, 1024), expected)
                        self.assertEqual(result.slave.captured(0x90000800, 1024),
                                         expected if in_place else source)
                        self.assertEqual(result.slave.captured(0x90000000, 1024), source)
                        self.assertEqual(result.slave.captured(0x90000C00, 32), b"\x5A" * 32)
            finally:
                cosim_atlas.CosimCore = original
        finally:
            os.chdir(old_cwd)
            sys.path.remove(str(modelir))


if __name__ == "__main__":
    unittest.main()
