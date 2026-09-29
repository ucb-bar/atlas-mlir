"""Bounded raw-bit VMIN check through the hand OOT and selected AtlasCore."""

from __future__ import annotations

import importlib.util
import os
import pathlib
import struct
import subprocess
import sys
import unittest

from test_mxu_reference import _object_words

ROOT = pathlib.Path(__file__).resolve().parents[1]
BIN = pathlib.Path(os.environ.get("ATLAS_OOT_BIN_DIR", ROOT / "build/bin"))
SOURCE = ROOT / "test/examples/vpu_min_pair.mlir"
ASSEMBLY = ROOT / "test/examples/vpu_min_pair.S"


def _ordered(code: int) -> int:
    """The selected RTL's unsigned ordering of raw 16-bit BF16 encodings."""
    if not 0 <= code <= 0xFFFF:
        raise ValueError("BF16 encoding is outside 16 bits")
    return (~code & 0xFFFF) if code & 0x8000 else code ^ 0x8000


def _min_code(lhs: int, rhs: int) -> int:
    return lhs if _ordered(lhs) < _ordered(rhs) else rhs


def _panel(phase: int) -> tuple[bytes, bytes, bytes]:
    directed = (
        (0x0000, 0x8000),  # -0 ranks below +0
        (0x8000, 0x0000),  # operand order does not reverse that choice
        (0x7FC1, 0x3F80),  # finite positive ranks below positive NaN bits
        (0xFFC1, 0x3F80),  # negative NaN bits rank below finite positive
        (0x7F80, 0x7FC0),  # +infinity ranks below positive NaN bits
        (0xFF80, 0xFFC0),  # negative NaN bits rank below -infinity
        (0x3F80, 0xBF80),
        (0xBF80, 0x3F80),
        (0x3555, 0xB555),
        (0xB555, 0x3555),
    )
    state = 0xA71A5 ^ phase
    chosen: list[tuple[int, int]] = []
    for index in range(1024):
        state = (1103515245 * state + 12345) & 0xFFFFFFFF
        lhs = (state >> 8) & 0xFFFF
        state = (1103515245 * state + 12345) & 0xFFFFFFFF
        rhs = (state >> 12) & 0xFFFF
        chosen.append((lhs, rhs))
    for index, pair in enumerate(directed):
        chosen[index] = pair
        chosen[512 + index] = directed[(index + phase) % len(directed)]
    lhs_bytes = b"".join(struct.pack("<H", lhs) for lhs, _ in chosen)
    rhs_bytes = b"".join(struct.pack("<H", rhs) for _, rhs in chosen)
    expected = b"".join(struct.pack("<H", _min_code(lhs, rhs))
                        for lhs, rhs in chosen)
    return lhs_bytes, rhs_bytes, expected


class VpuMinReferenceTest(unittest.TestCase):
    def test_bit_order_and_directed_discriminators(self) -> None:
        self.assertEqual(_min_code(0x0000, 0x8000), 0x8000)
        self.assertEqual(_min_code(0x8000, 0x0000), 0x8000)
        self.assertEqual(_min_code(0x7FC1, 0x3F80), 0x3F80)
        self.assertEqual(_min_code(0xFFC1, 0x3F80), 0xFFC1)
        self.assertEqual(_min_code(0xFF80, 0xFFC0), 0xFFC0)
        with self.assertRaises(ValueError):
            _ordered(0x10000)
        for phase in (0, 3):
            lhs, rhs, expected = _panel(phase)
            self.assertEqual((len(lhs), len(rhs), len(expected)),
                             (2048, 2048, 2048))
            self.assertNotEqual(expected, lhs)
            self.assertNotEqual(expected, rhs)

    def test_selected_assembler_and_llvm_object_words(self) -> None:
        selected_root = os.environ.get("ATLAS_ASSEMBLER_ROOT")
        if not selected_root:
            self.skipTest("set ATLAS_ASSEMBLER_ROOT for selected assembler")
        spec = importlib.util.spec_from_file_location(
            "selected_vmin_assembler", pathlib.Path(selected_root) / "assembler.py")
        assert spec and spec.loader
        assembler = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(assembler)
        emitted = tuple(int(line, 16) for line in subprocess.check_output(
            [str(BIN / "atlas-emit"), str(SOURCE)], text=True).splitlines())
        self.assertEqual(len(emitted), 49)
        self.assertEqual(emitted, tuple(assembler.assemble(ASSEMBLY.read_text())))
        self.assertEqual(emitted[31], assembler.VMIN_BF16(4, 0, 2))
        self.assertEqual(emitted[31] >> 25, 0x04)
        self.assertEqual(emitted, _object_words(SOURCE))
        printed = subprocess.run([str(BIN / "atlas-opt"), str(SOURCE)],
                                 capture_output=True, text=True, check=True)
        self.assertEqual(printed.stdout.count('"atlas.vpu_binary"'), 1)
        reparsed = subprocess.run([str(BIN / "atlas-opt"), "-"],
                                  input=printed.stdout, capture_output=True,
                                  text=True, check=True)
        self.assertEqual(reparsed.stdout, printed.stdout)

    def test_invalid_pair_and_kind_are_rejected(self) -> None:
        source = SOURCE.read_text()
        original = 'kind = "min", dst = 4 : i32, lhs = 0 : i32, rhs = 2 : i32'
        for mutated in (
            'kind = "min", dst = 5 : i32, lhs = 0 : i32, rhs = 2 : i32',
            'kind = "min", dst = 4 : i32, lhs = 63 : i32, rhs = 2 : i32',
            'kind = "min", dst = 4 : i32, lhs = 0 : i32, rhs = 3 : i32',
            'kind = "unknown", dst = 4 : i32, lhs = 0 : i32, rhs = 2 : i32',
        ):
            with self.subTest(mutated=mutated):
                changed = source.replace(original, mutated)
                self.assertNotEqual(changed, source)
                run = subprocess.run([str(BIN / "atlas-opt"), "-"], input=changed,
                                     capture_output=True, text=True, check=False)
                self.assertNotEqual(run.returncode, 0)
                self.assertTrue(run.stderr)

    def test_selected_core_matches_bit_order_and_preserves_inputs(self) -> None:
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
                for phase in (0, 3):
                    with self.subTest(phase=phase):
                        lhs, rhs, expected = _panel(phase)
                        result = cosim_atlas.run_program(
                            model, state, _object_words(SOURCE),
                            preload=[(0x90000000, lhs), (0x90000800, rhs),
                                     (0x90001000, b"\xA5" * 2048),
                                     (0x90001800, b"\x5A" * 32)],
                            max_cycles=10000,
                        )
                        observed = result.slave.captured(0x90001000, 2048)
                        self.assertTrue(result.halted)
                        self.assertEqual((result.reads, result.writes), (128, 64))
                        self.assertEqual(observed, expected)
                        self.assertEqual(result.slave.captured(0x90000000, 2048), lhs)
                        self.assertEqual(result.slave.captured(0x90000800, 2048), rhs)
                        self.assertEqual(result.slave.captured(0x90001800, 32), b"\x5A" * 32)
            finally:
                cosim_atlas.CosimCore = original
        finally:
            os.chdir(old_cwd)
            sys.path.remove(str(modelir))


if __name__ == "__main__":
    unittest.main()
