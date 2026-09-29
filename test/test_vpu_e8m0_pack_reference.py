"""Restricted raw-bit E8M0 pack/unpack and downstream FP8 MXU1 checks."""

from __future__ import annotations

import importlib.util
import os
import pathlib
import struct
import subprocess
import sys
import tempfile
import unittest
from fractions import Fraction

from test_mxu_numerics import bf16, fp8, round_bf16
from test_mxu_reference import _object_words

ROOT = pathlib.Path(__file__).resolve().parents[1]
BIN = pathlib.Path(os.environ.get("ATLAS_OOT_BIN_DIR", ROOT / "build/bin"))
SOURCE = ROOT / "test/examples/vpu_e8m0_pack_chain.mlir"
ASSEMBLY = ROOT / "test/examples/vpu_e8m0_pack_chain.S"
BASE = 0x90000000
SELI128 = 'kind = "seli", dst = 3 : i32, base = 0 : i32, offset = 128 : i32'


def _emitted(source: pathlib.Path = SOURCE) -> tuple[int, ...]:
    result = subprocess.run([str(BIN / "atlas-emit"), str(source)],
                            capture_output=True, text=True, check=True)
    return tuple(int(line, 16) for line in result.stdout.splitlines())


def _panel(phase: int) -> bytes:
    # The selected powers of two are exactly representable after both tested
    # E8M0 exponent shifts. No torch/model cast is used as an oracle.
    codes = (0x0000, 0x3F00, 0x3F80, 0x4000, 0xBF00, 0xBF80, 0xC000)
    halves = []
    for half in range(2):
        values = [codes[(row * 5 + col * 3 + half * 2 + phase) % len(codes)]
                  for row in range(32) for col in range(16)]
        halves.append(b"".join(struct.pack("<H", code) for code in values))
    return b"".join(halves)


def _factor(scale_code: int) -> Fraction:
    exponent = scale_code - 127
    return Fraction(1 << exponent) if exponent >= 0 else Fraction(1, 1 << -exponent)


def _fp8_codes() -> dict[Fraction, int]:
    result = {Fraction(0): 0}
    for code in range(256):
        if (code >> 3) & 15 == 0 or ((code >> 3) & 15 == 15 and code & 7 == 7):
            continue
        result[fp8(code)] = code
    return result


FP8_CODES = _fp8_codes()


def _expected(source: bytes, scale_code: int) -> tuple[bytes, bytes, bytes]:
    assert len(source) == 2048
    halves = (struct.unpack("<512H", source[:1024]),
              struct.unpack("<512H", source[1024:]))
    factor = _factor(scale_code)
    packed = bytearray()
    unpacked_rows: list[bytes] = []
    logical_rows: list[bytes] = []
    # VectorFSM reads all 32 physical rows of the first BF16 register, then
    # all 32 of the second. FP8Pack joins consecutive read rows, so packed
    # row 0 is physical BF16 rows 0 and 1, not columns 0..31 of logical row 0.
    physical_rows = [half[16 * row:16 * row + 16]
                     for half in halves for row in range(32)]
    for row in range(32):
        logical = physical_rows[2 * row] + physical_rows[2 * row + 1]
        fp8_row = bytes(FP8_CODES[bf16(code) / factor] for code in logical)
        logical_rows.append(fp8_row)
        packed.extend(fp8_row)
        for half in range(2):
            unpacked_rows.append(
                b"".join(struct.pack("<H", round_bf16(fp8(code) * factor))
                         for code in fp8_row[16 * half:16 * half + 16]))
    cells = [round_bf16(sum((fp8(a) * fp8(w) for a, w in
                            zip(logical_rows[row], logical_rows[col], strict=True)), start=0))
             for row in range(32) for col in range(32)]
    mxu = (b"".join(struct.pack("<H", cells[32 * row + col])
                    for row in range(32) for col in range(16)) +
           b"".join(struct.pack("<H", cells[32 * row + col])
                    for row in range(32) for col in range(16, 32)))
    return bytes(packed), b"".join(unpacked_rows), mxu


class VpuE8M0PackReferenceTest(unittest.TestCase):
    def test_typed_stream_matches_selected_assembler_and_llvm_object(self) -> None:
        root = os.environ.get("ATLAS_ASSEMBLER_ROOT")
        if not root:
            self.skipTest("set ATLAS_ASSEMBLER_ROOT for selected assembler")
        path = pathlib.Path(root) / "assembler.py"
        spec = importlib.util.spec_from_file_location("selected_e8m0_assembler", path)
        assert spec and spec.loader
        assembler = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(assembler)
        words = _emitted()
        self.assertEqual(len(words), 65)
        self.assertEqual(words, tuple(assembler.assemble(ASSEMBLY.read_text())))
        self.assertEqual(words[17], assembler.SELI(3, 128))
        self.assertEqual(words[18], assembler.VFP8PACK(2, 0, 3))
        self.assertEqual(words[20], assembler.VFP8UNPACK(4, 2, 3))
        self.assertEqual(_object_words(SOURCE), words)
        with tempfile.TemporaryDirectory() as temporary:
            changed = pathlib.Path(temporary) / "scale126.mlir"
            mutated = SOURCE.read_text().replace(SELI128, SELI128.replace("128", "126"))
            self.assertNotEqual(mutated, SOURCE.read_text())
            changed.write_text(mutated)
            scaled_words = _emitted(changed)
            self.assertEqual(scaled_words[17], assembler.SELI(3, 126))
            self.assertEqual(scaled_words[:17] + scaled_words[18:], words[:17] + words[18:])
            self.assertEqual(_object_words(changed), scaled_words)
        parsed = subprocess.run([str(BIN / "atlas-opt"), str(SOURCE)],
                                capture_output=True, text=True, check=True)
        self.assertEqual(parsed.stdout.count('"atlas.vpu_pack"'), 2)
        reparsed = subprocess.run([str(BIN / "atlas-opt"), "-"], input=parsed.stdout,
                                  capture_output=True, text=True, check=True)
        self.assertEqual(reparsed.stdout, parsed.stdout)

    def test_invalid_pair_and_scale_register_are_rejected(self) -> None:
        source = SOURCE.read_text()
        for original, replacement in (
            ('direction = "bf16_to_fp8", dst = 2 : i32, src = 0 : i32',
             'direction = "bf16_to_fp8", dst = 2 : i32, src = 63 : i32'),
            ('direction = "fp8_to_bf16", dst = 4 : i32, src = 2 : i32',
             'direction = "fp8_to_bf16", dst = 5 : i32, src = 2 : i32'),
            ('direction = "bf16_to_fp8", dst = 2 : i32, src = 0 : i32, scale_reg = 3 : i32',
             'direction = "bf16_to_fp8", dst = 2 : i32, src = 0 : i32, scale_reg = 32 : i32'),
        ):
            with self.subTest(replacement=replacement):
                changed = source.replace(original, replacement)
                self.assertNotEqual(changed, source)
                result = subprocess.run([str(BIN / "atlas-opt"), "-"], input=changed,
                                        capture_output=True, text=True, check=False)
                self.assertNotEqual(result.returncode, 0)
                self.assertTrue(result.stderr)

    def test_nonunit_scale_oracle_and_pair_layout(self) -> None:
        source = _panel(0)
        for code in (126, 128):
            with self.subTest(scale_code=code):
                packed, unpacked, mxu = _expected(source, code)
                self.assertEqual(len(packed), 1024)
                self.assertEqual(unpacked, source)
                self.assertEqual(len(mxu), 2048)
                self.assertNotEqual(packed, source[:1024])
        self.assertNotEqual(_expected(source, 126)[0], _expected(source, 128)[0])
        self.assertNotEqual(_expected(source, 126)[2], _expected(source, 128)[2])
        first_half = struct.unpack("<512H", source[:1024])
        second_half = struct.unpack("<512H", source[1024:])
        naive_row_zero = bytes(FP8_CODES[bf16(value) / _factor(128)]
                               for value in first_half[:16] + second_half[:16])
        self.assertNotEqual(_expected(source, 128)[0][:32], naive_row_zero)
        self.assertEqual(FP8_CODES[bf16(0x3F80) / _factor(128)], 0x30)  # 1 / 2
        self.assertEqual(FP8_CODES[bf16(0x3F80) / _factor(126)], 0x40)  # 1 / 0.5

    def test_selected_core_pack_unpack_and_mxu_fp8_consumer(self) -> None:
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
                for code, phase in ((128, 0), (126, 2)):
                    with self.subTest(scale_code=code, phase=phase):
                        source = _panel(phase)
                        expected_pack, expected_unpack, expected_mxu = _expected(source, code)
                        with tempfile.TemporaryDirectory() as temporary:
                            changed = pathlib.Path(temporary) / "scale.mlir"
                            text = SOURCE.read_text().replace(SELI128, SELI128.replace("128", str(code)))
                            if code != 128:
                                self.assertNotEqual(text, SOURCE.read_text())
                            changed.write_text(text)
                            words = _object_words(changed)
                        preload = [(BASE, source[:1024]), (BASE + 0x400, source[1024:])]
                        preload += [(BASE + 0x800 + 0x400 * i, b"\xA5" * 1024)
                                    for i in range(5)]
                        preload += [(BASE + 0x1C00, b"\x5A" * 32)]
                        result = cosim_atlas.run_program(model, state, words,
                                                         preload=preload, max_cycles=15000)
                        self.assertTrue(result.halted)
                        self.assertEqual((result.reads, result.writes), (64, 160))
                        self.assertEqual(result.slave.captured(BASE + 0x800, 1024), expected_pack)
                        self.assertEqual(result.slave.captured(BASE + 0xC00, 2048), expected_unpack)
                        self.assertEqual(result.slave.captured(BASE + 0x1400, 2048), expected_mxu)
                        self.assertEqual(result.slave.captured(BASE, 2048), source)
                        self.assertEqual(result.slave.captured(BASE + 0x1C00, 32), b"\x5A" * 32)
            finally:
                cosim_atlas.CosimCore = original
        finally:
            os.chdir(old_cwd)
            sys.path.remove(str(modelir))


if __name__ == "__main__":
    unittest.main()
