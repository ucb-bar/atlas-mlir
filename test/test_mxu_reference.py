"""Hand-authored typed MXU0 stream through LLVM object code and selected core."""

from __future__ import annotations

import importlib.util
import os
import pathlib
import struct
import subprocess
import sys
import tempfile
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]
SOURCE = ROOT / "test/examples/mxu0_full.mlir"
ASSEMBLY = "assembly/mxu0_push_pop.S"


def _emitted() -> tuple[int, ...]:
    run = subprocess.run([str(ROOT / "build/bin/atlas-emit"), str(SOURCE)],
                         capture_output=True, text=True, check=True)
    return tuple(int(line, 16) for line in run.stdout.splitlines())


def _object_words() -> tuple[int, ...]:
    selected = os.environ.get("ATLAS_LLVM_BIN")
    if not selected:
        raise unittest.SkipTest("set ATLAS_LLVM_BIN for LLVM object lowering")
    tools = pathlib.Path(selected)
    lowered = subprocess.run([str(ROOT / "build/bin/atlas-opt"), "--convert-atlas-to-llvm", str(SOURCE)],
                             capture_output=True, text=True, check=True)
    translated = subprocess.run([str(tools / "mlir-translate"), "--mlir-to-llvmir"],
                                input=lowered.stdout, capture_output=True, text=True, check=True)
    with tempfile.TemporaryDirectory() as temporary:
        obj = pathlib.Path(temporary) / "mxu.o"
        section = pathlib.Path(temporary) / "text.bin"
        subprocess.run([str(tools / "llc"), "-mtriple=riscv32-unknown-elf", "-filetype=obj", "-o", str(obj)],
                       input=translated.stdout.encode(), capture_output=True, check=True)
        subprocess.run([str(tools / "llvm-objcopy"), "--dump-section", f".text={section}", str(obj)],
                       capture_output=True, check=True)
        data = section.read_bytes()
    expected = _emitted()
    if len(data) < 4 * len(expected):
        raise AssertionError("LLVM object is shorter than the typed MXU0 stream")
    return tuple(int.from_bytes(data[4 * i : 4 * i + 4], "little") for i in range(len(expected)))


def _tile(entries: tuple[tuple[int, int, int], ...]) -> bytes:
    tile = bytearray(32 * 32)
    for row, col, encoding in entries:
        tile[32 * row + col] = encoding
    return bytes(tile)


class MXUReferenceTest(unittest.TestCase):
    def test_typed_mxu_matches_selected_assembler_and_llvm_object(self) -> None:
        root = os.environ.get("ATLAS_ASSEMBLER_ROOT")
        if not root:
            self.skipTest("set ATLAS_ASSEMBLER_ROOT for selected assembler")
        selected = pathlib.Path(root) / "assembler.py"
        spec = importlib.util.spec_from_file_location("selected_atlas_mxu_assembler", selected)
        assert spec and spec.loader
        assembler = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(assembler)
        reference = assembler.assemble((pathlib.Path(root) / ASSEMBLY).read_text())
        emitted = _emitted()
        self.assertEqual(len(emitted), 35)
        self.assertEqual(emitted, tuple(reference[:len(emitted)]))
        self.assertEqual(_object_words(), emitted)

    def test_selected_core_executes_typed_mxu_with_sparse_orientation_and_rounding(self) -> None:
        keys = ("ATLAS_ARC_MODEL", "ATLAS_ARC_STATE", "ATLAS_MODELIR_ROOT", "ATLAS_LLVM_BIN")
        if not all(os.environ.get(key) for key in keys):
            self.skipTest("set explicit selected-source-linked ARC and LLVM paths")
        model, state, modelir = (pathlib.Path(os.environ[key]).resolve(strict=True) for key in keys[:3])
        one, small = 0x38, 0x18
        weights_nk = _tile(((0, 0, one), (0, 1, small), (0, 2, small)))
        weights_kn = _tile(((0, 0, one), (1, 0, small), (2, 0, small)))
        activation = _tile(((0, 0, one), (0, 1, small), (0, 2, small)))
        negative = _tile(((0, 0, one), (0, 1, small | 0x80), (0, 2, small | 0x80)))
        cases = {
            "ones": (bytes([one]) * 1024, bytes([one]) * 1024, 0x4200),
            "sparse_nk": (weights_nk, activation, 0x3F80),
            "negative_nk": (weights_nk, negative, 0x3F7E),
            "sparse_kn": (weights_kn, activation, 0x3F80),
        }
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
                for name, (weights, acts, expected) in cases.items():
                    with self.subTest(name=name):
                        result = cosim_atlas.run_program(
                            model, state, _object_words(),
                            preload=[(0x90000000, weights), (0x90000400, acts),
                                     (0x90000800, b"\xA5" * 1024), (0x90000C00, b"\x5A" * 32)],
                            max_cycles=5000,
                        )
                        output = result.slave.captured(0x90000800, 1024)
                        self.assertTrue(result.halted)
                        self.assertEqual((result.reads, result.writes), (64, 32))
                        self.assertEqual(struct.unpack_from("<H", output)[0], expected)
                        self.assertEqual(result.slave.captured(0x90000000, 1024), weights)
                        self.assertEqual(result.slave.captured(0x90000400, 1024), acts)
                        self.assertEqual(result.slave.captured(0x90000C00, 32), b"\x5A" * 32)
                        if name == "sparse_kn":
                            self.assertEqual([struct.unpack_from("<H", output, 2 * i)[0] for i in (1, 2)],
                                             [0x3D80, 0x3D80])
            finally:
                cosim_atlas.CosimCore = original
        finally:
            os.chdir(old_cwd)
            sys.path.remove(str(modelir))


if __name__ == "__main__":
    unittest.main()
