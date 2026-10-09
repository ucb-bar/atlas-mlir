"""Selected RTL raw-bit VLI.ROW/COL/ONE check, independent of npu_model."""

from __future__ import annotations

import importlib.util
import os
import pathlib
import struct
import subprocess
import sys
import tempfile
import unittest

from test_vpu_relu_reference import _emitted, _object_words

from selected_core_runtime import run_selected_program

ROOT = pathlib.Path(__file__).resolve().parents[1]
BIN = pathlib.Path(os.environ.get("ATLAS_OOT_BIN_DIR", ROOT / "build/bin"))
SOURCE = ROOT / "test/examples/vli_selective_pair.mlir"
ASSEMBLY = ROOT / "test/examples/vli_selective_pair.S"
BASE_OP = 'mode = "row", dst = 4 : i32, immediate = 16384 : i32'
CASES = (("row", 4), ("col", 5), ("one", 5))
CODES = (0x4000, 0x8000)


def _source(mode: str, dst: int, code: int) -> str:
    text = SOURCE.read_text()
    assert text.count(BASE_OP) == 1
    return text.replace(BASE_OP, f'mode = "{mode}", dst = {dst} : i32, immediate = {code} : i32')


def _assembly(mode: str, dst: int, code: int) -> str:
    text = ASSEMBLY.read_text()
    old = "VLI.ROW 4, 0x4000"
    assert text.count(old) == 1
    return text.replace(old, f"VLI.{mode.upper()} {dst}, 0x{code:04X}")


def _expected(mode: str, code: int, noise: bytes) -> bytes:
    """Two 32-row mregs, each row holding 16 little-endian BF16 bit cells."""
    assert len(noise) == 2048 and mode in ("row", "col", "one")
    result = bytearray(2048 if mode == "row" else noise[:1024] + bytes(1024))
    raw = code.to_bytes(2, "little")
    if mode == "row":
        for base in (0, 1024):
            for col in range(16):
                offset = base + 2 * col
                result[offset:offset + 2] = raw
    elif mode == "col":
        for row in range(32):
            offset = 1024 + 32 * row
            result[offset:offset + 2] = raw
    else:
        result[1024:1026] = raw
    return bytes(result)


class VliSelectiveReferenceTest(unittest.TestCase):
    def test_raw_reference_distinguishes_numeric_assignment_and_scope(self) -> None:
        # The model routines assign integer imm into a BF16 tensor, then view
        # its bits. For imm=0x4000, that would mean BF16(16384)=0x4680.
        numeric_bf16 = int.from_bytes(struct.pack("<f", float(0x4000))[2:], "little")
        self.assertEqual(numeric_bf16, 0x4680)
        self.assertNotEqual(numeric_bf16, 0x4000)
        noise = bytes((i * 17 + 29) & 0xFF for i in range(2048))
        row = _expected("row", 0x4000, noise)
        col = _expected("col", 0x8000, noise)
        one = _expected("one", 0x4000, noise)
        self.assertEqual(row[:32], b"\x00\x40" * 16)
        self.assertEqual(row[32:1024], bytes(992))
        self.assertEqual(row[1024:1056], b"\x00\x40" * 16)
        self.assertEqual(row[1056:], bytes(992))
        self.assertEqual(col[:1024], noise[:1024])
        self.assertEqual(col[1024:1026], b"\x00\x80")
        self.assertEqual(col[1056:1058], b"\x00\x80")
        self.assertEqual(col[1026:1056], bytes(30))
        self.assertEqual(one[:1024], noise[:1024])
        self.assertEqual(one[1024:1026], b"\x00\x40")
        self.assertEqual(one[1026:], bytes(1022))

    def test_typed_modes_match_selected_assembler_and_llvm_object(self) -> None:
        selected_root = os.environ.get("ATLAS_ASSEMBLER_ROOT")
        if not selected_root:
            self.skipTest("set ATLAS_ASSEMBLER_ROOT for independent selected assembler")
        path = pathlib.Path(selected_root) / "assembler.py"
        spec = importlib.util.spec_from_file_location("selected_vli_selective_assembler", path)
        assert spec and spec.loader
        assembler = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(assembler)
        for mode, dst in CASES:
            for code in CODES:
                with self.subTest(mode=mode, code=f"0x{code:04x}"):
                    with tempfile.TemporaryDirectory() as temporary:
                        source = pathlib.Path(temporary) / "vli_selective.mlir"
                        source.write_text(_source(mode, dst, code))
                        emitted = _emitted(source)
                        self.assertEqual(len(emitted), 36)
                        self.assertEqual(emitted, tuple(assembler.assemble(
                            _assembly(mode, dst, code))))
                        self.assertEqual(emitted[17], getattr(assembler, f"VLI_{mode.upper()}")(
                            dst, code))
                        self.assertEqual(_object_words(source), emitted)
                        lowered = subprocess.run(
                            [str(BIN / "atlas-opt"), "--convert-atlas-to-llvm", str(source)],
                            text=True, capture_output=True, check=True)
                        self.assertIn("llvm.inline_asm has_side_effects", lowered.stdout)
                        self.assertIn('"~{memory}"', lowered.stdout)
                        self.assertIn(f".word 0x{emitted[17]:08x}", lowered.stdout)
                        parsed = subprocess.run([str(BIN / "atlas-opt"), str(source)],
                                                text=True, capture_output=True, check=True)
                        self.assertEqual(parsed.stdout.count('"atlas.vli"'), 1)
                        reparsed = subprocess.run([str(BIN / "atlas-opt"), "-"],
                                                  input=parsed.stdout, text=True,
                                                  capture_output=True, check=True)
                        self.assertEqual(reparsed.stdout, parsed.stdout)

    def test_typed_legality_rejects_wrong_pair_register_or_immediate(self) -> None:
        source = SOURCE.read_text()
        replacements = (
            'mode = "row", dst = 5 : i32, immediate = 16384 : i32',
            'mode = "row", dst = 63 : i32, immediate = 16384 : i32',
            'mode = "col", dst = 64 : i32, immediate = 16384 : i32',
            'mode = "one", dst = -1 : i32, immediate = 16384 : i32',
            'mode = "one", dst = 5 : i32, immediate = -1 : i32',
            'mode = "col", dst = 5 : i32, immediate = 65536 : i32',
            'mode = "other", dst = 5 : i32, immediate = 16384 : i32',
        )
        for replacement in replacements:
            with self.subTest(replacement=replacement):
                changed = source.replace(BASE_OP, replacement)
                self.assertNotEqual(changed, source)
                result = subprocess.run([str(BIN / "atlas-opt"), "-"], input=changed,
                                        text=True, capture_output=True, check=False)
                self.assertNotEqual(result.returncode, 0)
                self.assertTrue(result.stderr)

    def test_selected_core_executes_raw_selective_fill(self) -> None:
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
                for mode, dst in CASES:
                    for phase, code in enumerate(CODES):
                        with self.subTest(mode=mode, code=f"0x{code:04x}"):
                            noise = bytes((i * 17 + phase * 29 + dst * 7) & 0xFF
                                          for i in range(2048))
                            expected = _expected(mode, code, noise)
                            with tempfile.TemporaryDirectory() as temporary:
                                source = pathlib.Path(temporary) / "vli_selective.mlir"
                                source.write_text(_source(mode, dst, code))
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
