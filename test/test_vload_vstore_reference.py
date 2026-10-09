"""Bounded raw-byte VLOAD/VSTORE evidence on the selected standalone core."""

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
SOURCE = ROOT / "test/examples/vload_vstore.mlir"
ASSEMBLY = ROOT / "test/examples/vload_vstore.S"


def _emitted() -> tuple[int, ...]:
    run = subprocess.run([str(BIN / "atlas-emit"), str(SOURCE)],
                         capture_output=True, text=True, check=True)
    return tuple(int(line, 16) for line in run.stdout.splitlines())


class VLoadVStoreReferenceTest(unittest.TestCase):
    def test_selected_assembler_and_llvm_object_words(self) -> None:
        root = os.environ.get("ATLAS_ASSEMBLER_ROOT")
        if not root:
            self.skipTest("set ATLAS_ASSEMBLER_ROOT for selected assembler")
        path = pathlib.Path(root) / "assembler.py"
        spec = importlib.util.spec_from_file_location("selected_vls_assembler", path)
        assert spec and spec.loader
        assembler = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(assembler)
        emitted = _emitted()
        self.assertEqual(len(emitted), 29)
        self.assertEqual(emitted, tuple(assembler.assemble(ASSEMBLY.read_text())))
        self.assertEqual(emitted[8], assembler.VLOAD(4, 6, 0))
        self.assertEqual(emitted[13], assembler.VSTORE(4, 8, 0))
        self.assertEqual(_object_words(SOURCE), emitted)
        printed = subprocess.run([str(BIN / "atlas-opt"), str(SOURCE)],
                                 capture_output=True, text=True, check=True)
        self.assertEqual(printed.stdout.count('"atlas.vload"'), 1)
        self.assertEqual(printed.stdout.count('"atlas.vstore"'), 2)
        reparsed = subprocess.run([str(BIN / "atlas-opt"), "-"],
                                  input=printed.stdout, capture_output=True,
                                  text=True, check=True)
        self.assertEqual(reparsed.stdout, printed.stdout)

    def test_invalid_register_and_offset_fields_are_rejected(self) -> None:
        source = SOURCE.read_text()
        for original, replacement in (
            ('dst = 4 : i32, base = 6 : i32, offset = 0 : i32',
             'dst = 64 : i32, base = 6 : i32, offset = 0 : i32'),
            ('src = 4 : i32, base = 8 : i32, offset = 0 : i32',
             'src = -1 : i32, base = 8 : i32, offset = 0 : i32'),
            ('dst = 4 : i32, base = 6 : i32, offset = 0 : i32',
             'dst = 4 : i32, base = 6 : i32, offset = 2048 : i32'),
            ('src = 4 : i32, base = 8 : i32, offset = 0 : i32',
             'src = 4 : i32, base = 8 : i32, offset = -2049 : i32'),
        ):
            with self.subTest(replacement=replacement):
                changed = source.replace(original, replacement)
                self.assertNotEqual(changed, source)
                run = subprocess.run([str(BIN / "atlas-opt"), "-"], input=changed,
                                     capture_output=True, text=True, check=False)
                self.assertNotEqual(run.returncode, 0)
                self.assertTrue(run.stderr)

    def test_selected_core_preserves_all_bytes_and_guards(self) -> None:
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
                panels = (
                    bytes(range(256)) * 4,
                    bytes((37 * i + 19) & 255 for i in range(1024)),
                    bytes((0x80 if i % 2 else 0x00) ^ (i // 32) for i in range(1024)),
                )
                for source in panels:
                    with self.subTest(first=source[:8].hex()):
                        result = run_selected_program(cosim_atlas,
                            model, state, _object_words(SOURCE),
                            preload=[(0x90000000, source),
                                     (0x90000400, b"\xA5" * 1024),
                                     (0x90000800, b"\xA5" * 1024),
                                     (0x90000C00, b"\x5A" * 32)],
                            max_cycles=5000,
                        )
                        self.assertTrue(result.halted)
                        self.assertEqual((result.reads, result.writes), (32, 64))
                        for address in (0x90000400, 0x90000800, 0x90000000):
                            self.assertEqual(result.slave.captured(address, 1024), source)
                        self.assertEqual(result.slave.captured(0x90000C00, 32), b"\x5A" * 32)
            finally:
                cosim_atlas.CosimCore = original
        finally:
            os.chdir(old_cwd)
            sys.path.remove(str(modelir))


if __name__ == "__main__":
    unittest.main()
