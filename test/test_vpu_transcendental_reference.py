"""Bounded selected-core checks for five approximate BF16 VPU modes."""

from __future__ import annotations

import importlib.util
import math
import os
import pathlib
import struct
import subprocess
import sys
import tempfile
import unittest

from test_vpu_exp2_reference import ASSEMBLY, SOURCE
from test_vpu_relu_reference import _emitted, _object_words


ROOT = pathlib.Path(__file__).resolve().parents[1]
BIN = pathlib.Path(os.environ.get("ATLAS_OOT_BIN_DIR", ROOT / "build/bin"))
RTL_REVISION = "0079c0541111197741a231c002e3843fa6f545b2"
MODES = {"exp": "VEXP", "sin": "VSIN", "cos": "VCOS", "tanh": "VTANH", "log2": "VLOG2"}
CODES = (0x0000, 0x3F00, 0x3F80, 0x4000, 0x4080, 0x3FC0, 0x4040, 0x40A0)
EXACT = {
    "exp": {0x0000: 0x3F80},
    "sin": {0x0000: 0x0000},
    "cos": {0x0000: 0x3F80},
    "tanh": {0x0000: 0x0000},
    "log2": {0x0000: 0xFF80, 0x3F00: 0xBF80, 0x3F80: 0x0000,
             0x4000: 0x3F80, 0x4080: 0x4000},
}


def _source(mode: str) -> str:
    source = SOURCE.read_text()
    selected = source.replace('kind = "exp2", dst = 4 : i32, src = 0 : i32',
                              f'kind = "{mode}", dst = 4 : i32, src = 0 : i32')
    if selected == source or selected.count(f'kind = "{mode}"') != 1:
        raise AssertionError("unary fixture changed")
    return selected


def _panel(phase: int) -> tuple[bytes, tuple[int, ...]]:
    values = tuple(CODES[(i * 7 + i // 16 + phase) % len(CODES)] for i in range(1024))
    return b"".join(struct.pack("<H", code) for code in values), values


def _bf16_value(code: int) -> float:
    return struct.unpack(">f", struct.pack(">I", code << 16))[0]


def _rounded_math(mode: str, code: int) -> int:
    x = _bf16_value(code)
    y = float("-inf") if mode == "log2" and x == 0 else getattr(math, mode)(x)
    bits = struct.unpack(">I", struct.pack(">f", y))[0]
    upper, lower = bits >> 16, bits & 0xFFFF
    return (upper + (lower > 0x8000 or (lower == 0x8000 and bool(upper & 1)))) & 0xFFFF


def _ordered_bf16(code: int) -> int:
    return ((~code) & 0xFFFF) if code & 0x8000 else code | 0x8000


class VpuTranscendentalReferenceTest(unittest.TestCase):
    def test_independent_math_anchors_and_panel(self) -> None:
        for mode, anchors in EXACT.items():
            for source, expected in anchors.items():
                with self.subTest(mode=mode, source=source):
                    self.assertEqual(_rounded_math(mode, source), expected)
        for phase in (0, 5):
            source, values = _panel(phase)
            self.assertEqual(len(source), 2048)
            self.assertEqual(len(values), 1024)
            self.assertEqual(set(values), set(CODES))

    def test_typed_words_match_selected_assembler_and_llvm(self) -> None:
        assembler_root = os.environ.get("ATLAS_ASSEMBLER_ROOT")
        if not assembler_root:
            self.skipTest("set ATLAS_ASSEMBLER_ROOT for selected assembler")
        spec = importlib.util.spec_from_file_location(
            "selected_transcendental_assembler", pathlib.Path(assembler_root) / "assembler.py"
        )
        assert spec is not None and spec.loader is not None
        assembler = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(assembler)
        with tempfile.TemporaryDirectory() as temporary:
            path = pathlib.Path(temporary) / "unary.mlir"
            for mode, mnemonic in MODES.items():
                with self.subTest(mode=mode):
                    path.write_text(_source(mode))
                    assembly = ASSEMBLY.read_text().replace("VEXP2 4, 0", f"{mnemonic} 4, 0")
                    self.assertIn(f"{mnemonic} 4, 0", assembly)
                    words = _emitted(path)
                    self.assertEqual(len(words), 36)
                    self.assertEqual(words[17], getattr(assembler, mnemonic)(4, 0))
                    self.assertEqual(words, tuple(assembler.assemble(assembly)))
                    self.assertEqual(words, _object_words(path))
                    parsed = subprocess.run([str(BIN / "atlas-opt"), str(path)],
                                            capture_output=True, text=True, check=True)
                    reparsed = subprocess.run([str(BIN / "atlas-opt"), "-"],
                                              input=parsed.stdout, capture_output=True,
                                              text=True, check=True)
                    self.assertEqual(parsed.stdout, reparsed.stdout)

    def test_invalid_pairs_are_rejected_for_each_mode(self) -> None:
        for mode in MODES:
            for bad in (f'kind = "{mode}", dst = 5 : i32, src = 0 : i32',
                        f'kind = "{mode}", dst = 4 : i32, src = 1 : i32'):
                with self.subTest(mode=mode, bad=bad):
                    changed = _source(mode).replace(
                        f'kind = "{mode}", dst = 4 : i32, src = 0 : i32', bad)
                    run = subprocess.run([str(BIN / "atlas-opt"), "-"], input=changed,
                                         text=True, capture_output=True, check=False)
                    self.assertNotEqual(run.returncode, 0)
                    self.assertTrue(run.stderr)

    def test_selected_core_matches_bounded_math_and_preserves_memory(self) -> None:
        keys = ("ATLAS_ARC_MODEL", "ATLAS_ARC_STATE", "ATLAS_MODELIR_ROOT", "ATLAS_RTL_ROOT")
        if not all(os.environ.get(key) for key in keys):
            self.skipTest("set selected ARC model, state, ModeLIR, and RTL paths")
        model, state, modelir, rtl = (pathlib.Path(os.environ[key]).resolve(strict=True)
                                      for key in keys)
        self.assertEqual(
            subprocess.check_output(["git", "-C", str(rtl), "rev-parse", "HEAD"], text=True).strip(),
            RTL_REVISION,
        )
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
                with tempfile.TemporaryDirectory() as temporary:
                    path = pathlib.Path(temporary) / "unary.mlir"
                    for mode in MODES:
                        path.write_text(_source(mode))
                        words = _emitted(path)
                        for phase in (0, 5):
                            with self.subTest(mode=mode, phase=phase):
                                source, codes = _panel(phase)
                                result = cosim_atlas.run_program(
                                    model, state, words,
                                    preload=[(0x90000000, source[:1024]),
                                             (0x90000400, source[1024:]),
                                             (0x90000800, b"\xA5" * 1024),
                                             (0x90000C00, b"\xA5" * 1024),
                                             (0x90001000, b"\x5A" * 32)],
                                    max_cycles=8000,
                                )
                                output = (result.slave.captured(0x90000800, 1024) +
                                          result.slave.captured(0x90000C00, 1024))
                                actual = struct.unpack("<1024H", output)
                                self.assertTrue(result.halted)
                                self.assertEqual((result.reads, result.writes), (64, 64))
                                self.assertEqual(result.slave.captured(0x90000000, 2048), source)
                                self.assertEqual(result.slave.captured(0x90001000, 32), b"\x5A" * 32)
                                for index, (input_code, output_code) in enumerate(zip(codes, actual)):
                                    expected = _rounded_math(mode, input_code)
                                    with self.subTest(index=index):
                                        self.assertLessEqual(
                                            abs(_ordered_bf16(output_code) - _ordered_bf16(expected)), 1)
                                        if input_code in EXACT[mode]:
                                            self.assertEqual(output_code, EXACT[mode][input_code])
            finally:
                cosim_atlas.CosimCore = original
        finally:
            os.chdir(old_cwd)
            sys.path.remove(str(modelir))


if __name__ == "__main__":
    unittest.main()
