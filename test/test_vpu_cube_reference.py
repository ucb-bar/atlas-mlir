"""Bounded BF16 VCUBE raw-bit check against selected AtlasCore, not npu_model."""

from __future__ import annotations

import importlib.util
import os
import pathlib
import struct
import subprocess
import sys
import unittest

from test_vpu_relu_reference import _emitted, _object_words

from selected_core_runtime import run_selected_program

ROOT = pathlib.Path(__file__).resolve().parents[1]
BIN = pathlib.Path(os.environ.get("ATLAS_OOT_BIN_DIR", ROOT / "build/bin"))
SOURCE = ROOT / "test/examples/vpu_cube_pair.mlir"
ASSEMBLY = ROOT / "test/examples/vpu_cube_pair.S"
BASE_OP = 'kind = "cube", dst = 4 : i32, src = 0 : i32'


def _cube_bits(raw: int) -> int:
    """Restricted independent oracle: BF16 powers of two and explicit specials.

    Selected VectorParam.scala flushes zero, subnormal and NaN inputs to
    signed zero; infinity maps to signed infinity. Normal inputs have no fraction, so
    their mathematical cube is an exact power of two. No RTL mantissa
    algorithm or model routine is called to compute the expected bits.
    """
    sign = raw & 0x8000
    exponent = (raw >> 7) & 0xFF
    fraction = raw & 0x7F
    if exponent == 0:
        return sign
    if exponent == 0xFF:
        return (sign | 0x7F80) if fraction == 0 else sign
    if fraction:
        raise ValueError("oracle admits only normal powers of two")
    result_exponent = 3 * (exponent - 127) + 127
    if result_exponent <= 0:
        return sign
    if result_exponent >= 255:
        return sign | 0x7F80
    return sign | (result_exponent << 7)


def _panel(phase: int) -> tuple[bytes, bytes]:
    codes = (0x0000, 0x8000, 0x0001, 0x807F,
             0x7F80, 0xFF80, 0x7FC1, 0xFFC1,
             0x3F80, 0xBF80, 0x4000, 0xC000,
             0x3F00, 0xBF00, 0x0080, 0x7F00)
    values = [codes[(i * 7 + i // 16 + phase) % len(codes)] for i in range(1024)]
    source = b"".join(struct.pack("<H", code) for code in values)
    expected = b"".join(struct.pack("<H", _cube_bits(code)) for code in values)
    return source, expected


class VpuCubeReferenceTest(unittest.TestCase):
    def test_restricted_oracle_distinguishes_nan_and_signed_specials(self) -> None:
        self.assertEqual(_cube_bits(0x7FC1), 0)
        self.assertEqual(_cube_bits(0xFFC1), 0x8000)
        self.assertEqual(_cube_bits(0x8000), 0x8000)
        self.assertEqual(_cube_bits(0x807F), 0x8000)
        self.assertEqual(_cube_bits(0xFF80), 0xFF80)
        self.assertEqual(_cube_bits(0xC000), 0xC100)  # (-2)^3 = -8
        self.assertEqual(_cube_bits(0x3F00), 0x3E00)  # (1/2)^3 = 1/8
        self.assertEqual(_cube_bits(0xBF00), 0xBE00)  # (-1/2)^3 = -1/8
        self.assertEqual(_cube_bits(0x0080), 0)       # underflow
        self.assertEqual(_cube_bits(0x7F00), 0x7F80)  # overflow
        with self.assertRaisesRegex(ValueError, "only normal powers"):
            _cube_bits(0x3FC0)
        for phase in (0, 5):
            source, expected = _panel(phase)
            self.assertEqual((len(source), len(expected)), (2048, 2048))
            self.assertNotEqual(source, expected)
            self.assertIn(b"\x00\x00", expected)
            self.assertIn(b"\x00\x80", expected)
            self.assertIn(b"\x80\x7F", expected)

    def test_typed_stream_matches_selected_assembler_and_llvm_object(self) -> None:
        selected_root = os.environ.get("ATLAS_ASSEMBLER_ROOT")
        if not selected_root:
            self.skipTest("set ATLAS_ASSEMBLER_ROOT for independent selected assembler")
        path = pathlib.Path(selected_root) / "assembler.py"
        spec = importlib.util.spec_from_file_location("selected_cube_assembler", path)
        assert spec and spec.loader
        assembler = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(assembler)
        emitted = _emitted(SOURCE)
        self.assertEqual(len(emitted), 36)
        self.assertEqual(emitted, tuple(assembler.assemble(ASSEMBLY.read_text())))
        self.assertEqual(emitted[17], assembler.VCUBE_BF16(4, 0))
        self.assertEqual(emitted[17] >> 25, 0x47)
        self.assertNotEqual(emitted[17] >> 25, 0x4F)  # inspected model class
        self.assertEqual(_object_words(SOURCE), emitted)
        lowered = subprocess.run([str(BIN / "atlas-opt"), "--convert-atlas-to-llvm",
                                  str(SOURCE)], text=True, capture_output=True, check=True)
        self.assertIn("llvm.inline_asm has_side_effects", lowered.stdout)
        self.assertIn('"~{memory}"', lowered.stdout)
        self.assertIn(f".word 0x{emitted[17]:08x}", lowered.stdout)
        parsed = subprocess.run([str(BIN / "atlas-opt"), str(SOURCE)],
                                text=True, capture_output=True, check=True)
        self.assertEqual(parsed.stdout.count('"atlas.vpu_unary"'), 1)
        reparsed = subprocess.run([str(BIN / "atlas-opt"), "-"], input=parsed.stdout,
                                  text=True, capture_output=True, check=True)
        self.assertEqual(reparsed.stdout, parsed.stdout)

    def test_invalid_pairs_and_unknown_mode_are_rejected(self) -> None:
        source = SOURCE.read_text()
        for mutation in (
            'kind = "cube", dst = 5 : i32, src = 0 : i32',
            'kind = "cube", dst = 63 : i32, src = 0 : i32',
            'kind = "cube", dst = 4 : i32, src = 1 : i32',
            'kind = "cube", dst = 4 : i32, src = 63 : i32',
            'kind = "unqualified", dst = 4 : i32, src = 0 : i32',
        ):
            with self.subTest(mutation=mutation):
                changed = source.replace(BASE_OP, mutation)
                self.assertNotEqual(changed, source)
                run = subprocess.run([str(BIN / "atlas-opt"), "-"], input=changed,
                                     text=True, capture_output=True, check=False)
                self.assertNotEqual(run.returncode, 0)
                self.assertTrue(run.stderr)

    def test_selected_core_matches_specials_and_exact_powers(self) -> None:
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
                words = _object_words(SOURCE)
                self.assertEqual(words, _emitted(SOURCE))
                for phase in (0, 5):
                    with self.subTest(phase=phase):
                        source, expected = _panel(phase)
                        result = run_selected_program(cosim_atlas,
                            model, state, words,
                            preload=[(0x90000000, source[:1024]),
                                     (0x90000400, source[1024:]),
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
                        self.assertEqual(result.slave.captured(0x90000000, 2048), source)
                        self.assertEqual(result.slave.captured(0x90001000, 32), b"\x5A" * 32)
            finally:
                cosim_atlas.CosimCore = original
        finally:
            os.chdir(old_cwd)
            sys.path.remove(str(modelir))


if __name__ == "__main__":
    unittest.main()
