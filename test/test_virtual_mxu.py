"""A virtual FP8 contraction reaches LLVM object words and selected AtlasCore."""

from __future__ import annotations

import os
from pathlib import Path
import json
import re
import shutil
import struct
import subprocess
import sys
import tempfile
import unittest

from test_virtual_lowering import BIN, EXAMPLES, ROOT, emitted, lower, object_words, run


def panel() -> bytes:
    codes = (0x00, 0x30, 0x38, 0x40, 0xB8)
    return bytes(codes[(row * 7 + col * 3) % len(codes)]
                 for row in range(32) for col in range(32))


def weight(shift: int) -> bytes:
    data = bytearray(1024)
    for output_column in range(32):
        data[output_column * 32 + (output_column + shift) % 32] = 0x38
    return bytes(data)


def bf16_cell(data: bytes, row: int, col: int) -> int:
    return struct.unpack_from(
        "<H", data, (col // 16) * 1024 + (row * 16 + col % 16) * 2
    )[0]


def bf16_bits(value: float) -> int:
    bits = struct.unpack("<I", struct.pack("<f", value))[0]
    assert bits & 0xFFFF == 0, "directed reference value must be exact BF16"
    return bits >> 16


def bf16_bias_tile(by_column: tuple[float, ...]) -> bytes:
    assert len(by_column) == 32
    tile = bytearray(2048)
    for row in range(32):
        for col in range(32):
            offset = (col // 16) * 1024 + (row * 16 + col % 16) * 2
            struct.pack_into("<H", tile, offset, bf16_bits(by_column[col]))
    return bytes(tile)


class VirtualMXUTest(unittest.TestCase):
    def source(self, unit: int = 0) -> str:
        text = (EXAMPLES / "virtual_fp8_matmul_program.mlir").read_text()
        return text.replace("unit = 0 : i32", f"unit = {unit} : i32")

    def test_one_layer_has_checked_words_and_physical_alias_partition(self) -> None:
        machine = lower(self.source())
        self.assertIn('"atlas.mxu_push"', machine)
        self.assertIn('"atlas.mxu_matmul"', machine)
        self.assertIn('"atlas.mxu_pop"', machine)
        self.assertIn('atlas.generated_from_virtual', machine)
        checked = run("atlas-opt", machine, "--verify-atlas-generated-schedule")
        self.assertEqual(checked.returncode, 0, checked.stderr)
        self.assertEqual(object_words(machine), emitted(machine))
        fp8_registers = [int(reg) for reg in re.findall(
            r'"atlas.vload"[^\n]*dst = (\d+) : i32', machine)]
        readout = re.search(r'"atlas.mxu_pop"[^\n]*dst = (\d+) : i32', machine)
        self.assertEqual(len(fp8_registers), 2)
        self.assertTrue(all(reg < 32 for reg in fp8_registers))
        self.assertIsNotNone(readout)
        self.assertGreaterEqual(int(readout.group(1)), 32)
        bad_unit = run("atlas-opt", self.source(2),
                       "--lower-atlas-virtual-to-machine")
        self.assertNotEqual(bad_unit.returncode, 0)
        self.assertIn("unit must be in [0, 1]", bad_unit.stderr)
        mlp_source = (EXAMPLES / "virtual_fp8_two_layer_mlp.mlir").read_text()
        nonunit = run(
            "atlas-opt", mlp_source.replace("scale_code = 127 : i32",
                                             "scale_code = 128 : i32"),
            "--lower-atlas-virtual-to-machine",
        )
        self.assertNotEqual(nonunit.returncode, 0)
        self.assertIn("unit E8M0 scale code 127 only", nonunit.stderr)
        scratch_overlap = run(
            "atlas-opt", mlp_source.replace("index = 2 : i32",
                                             "index = 64 : i32"),
            "--lower-atlas-virtual-to-machine",
        )
        self.assertNotEqual(scratch_overlap.returncode, 0)
        self.assertIn("input indexes below 64", scratch_overlap.stderr)
        biased_source = (
            EXAMPLES / "virtual_fp8_two_layer_mlp_bias.mlir"
        ).read_text()
        unqualified_add = run(
            "atlas-opt", biased_source.replace('kind = "add"',
                                               'kind = "sub"', 1),
            "--lower-atlas-virtual-to-machine",
        )
        self.assertNotEqual(unqualified_add.returncode, 0)
        self.assertIn("binary VPU admission currently requires add",
                      unqualified_add.stderr)
        changed = machine.replace("cycles = 256 : i32", "cycles = 1 : i32", 1)
        rejected = run("atlas-emit", changed)
        self.assertNotEqual(rejected.returncode, 0)
        self.assertIn("DELAY >= 256", rejected.stderr)

    def test_virtual_mlp_handoff_generates_all_stages(self) -> None:
        llvm = os.environ.get("ATLAS_LLVM_BIN")
        linker = os.environ.get("ATLAS_LLD") or shutil.which("ld.lld")
        if not llvm or not linker:
            self.skipTest("set ATLAS_LLVM_BIN and an ld.lld path")
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "handoff"
            subprocess.run([
                sys.executable, str(ROOT / "tools/export_llvm_handoff.py"),
                "--atlas-bin-dir", str(BIN), "--llvm-bin-dir", llvm,
                "--linker", linker, "--output-dir", str(output),
            ], check=True, capture_output=True, text=True)
            manifest = json.loads((output / "manifest.json").read_text())
            name = "virtual_fp8_two_layer_mlp"
            self.assertEqual(manifest["examples"][name]["source_stage"],
                             "virtual_ssa")
            self.assertEqual(manifest["examples"][name]["word_count"], 109)
            self.assertEqual(
                (output / f"{name}.virtual.mlir").read_bytes(),
                (EXAMPLES / "virtual_fp8_two_layer_mlp.mlir").read_bytes(),
            )
            self.assertEqual(
                (ROOT / "examples/handoff/virtual_mlp/00-atlas-virtual-ssa.mlir")
                .read_bytes(),
                (EXAMPLES / "virtual_fp8_two_layer_mlp.mlir").read_bytes(),
            )
            self.assertEqual(
                (output / f"{name}.atlas.mlir").read_text(),
                lower((EXAMPLES / "virtual_fp8_two_layer_mlp.mlir").read_text()).rstrip()
                + "\n",
            )
            for suffix in ("llvm-structured.mlir", "llvm.mlir", "ll", "s",
                           "o", "elf", "disasm.txt", "words.txt",
                           "word-map.json", "physical-program.json"):
                with self.subTest(stage=suffix):
                    self.assertTrue((output / f"{name}.{suffix}").is_file())

    def test_both_mxus_execute_generated_matmul_and_preserve_memory(self) -> None:
        required = ("ATLAS_ARC_MODEL", "ATLAS_ARC_STATE",
                    "ATLAS_MODELIR_ROOT", "ATLAS_LLVM_BIN")
        if not all(os.environ.get(key) for key in required):
            self.skipTest("set selected-source-linked ARC, ModeLIR, and LLVM paths")
        model, state, modelir = (Path(os.environ[key]).resolve(strict=True)
                                 for key in required[:3])
        previous = Path.cwd()
        sys.path.insert(0, str(modelir))
        try:
            os.chdir(modelir)
            from mlc.backends import cosim_atlas

            original = cosim_atlas.CosimCore

            class SelectedCore(original):
                def peek(self, name: str) -> int:
                    return super().peek("scalar/halt_now" if name == "io_halted"
                                        else name)

                def poke(self, name: str, value: int) -> None:
                    if name in ("io_dmaTL_d_bits_opcode", "io_dmaTL_d_bits_size"):
                        if name in self._S:
                            raise AssertionError(f"unexpected ARC input {name}")
                        return
                    super().poke(name, value)

            cosim_atlas.CosimCore = SelectedCore
            try:
                source_panel = panel()
                expected_codes = {0x00: 0x0000, 0x30: 0x3F00,
                                  0x38: 0x3F80, 0x40: 0x4000,
                                  0xB8: 0xBF80}
                base = 0x90000000
                for unit, shift in ((0, 0), (1, 3)):
                    with self.subTest(unit=unit):
                        weights = weight(shift)
                        words = object_words(lower(self.source(unit)))
                        result = cosim_atlas.run_program(
                            model, state, words,
                            preload=[(base, source_panel),
                                     (base + 0x800, weights),
                                     (base + 0x1000, b"\xA5" * 2048),
                                     (base + 0x1800, b"\x5A" * 64)],
                            max_cycles=20000,
                        )
                        output = result.slave.captured(base + 0x1000, 2048)
                        self.assertTrue(result.halted)
                        self.assertEqual((result.reads, result.writes), (64, 64))
                        for row in range(32):
                            for col in range(32):
                                source = source_panel[row * 32 + (col + shift) % 32]
                                self.assertEqual(
                                    bf16_cell(output, row, col),
                                    expected_codes[source],
                                    (unit, row, col),
                                )
                        self.assertEqual(result.slave.captured(base, 1024),
                                         source_panel)
                        self.assertEqual(result.slave.captured(base + 0x800, 1024),
                                         weights)
                        self.assertEqual(result.slave.captured(base + 0x1800, 64),
                                         b"\x5A" * 64)
                mlp_source = (EXAMPLES / "virtual_fp8_two_layer_mlp.mlir").read_text()
                mlp_machine = lower(mlp_source)
                self.assertEqual(mlp_machine.count('"atlas.mxu_matmul"'), 2)
                self.assertIn('"atlas.vpu_pack"', mlp_machine)
                mlp_words = object_words(mlp_machine)
                self.assertEqual(mlp_words, emitted(mlp_machine))
                for first_shift, second_shift in ((0, 0), (3, 5)):
                    with self.subTest(mlp=(first_shift, second_shift)):
                        first_weight = weight(first_shift)
                        second_weight = weight(second_shift)
                        result = cosim_atlas.run_program(
                            model, state, mlp_words,
                            preload=[(base, source_panel),
                                     (base + 0x800, first_weight),
                                     (base + 0x1000, second_weight),
                                     (base + 0x2000, b"\xA5" * 2048),
                                     (base + 0x2800, b"\x5A" * 64)],
                            max_cycles=100000,
                        )
                        output = result.slave.captured(base + 0x2000, 2048)
                        self.assertTrue(result.halted)
                        self.assertEqual((result.reads, result.writes), (96, 64))
                        for row in range(32):
                            for col in range(32):
                                source = source_panel[
                                    row * 32 + (col + first_shift + second_shift) % 32
                                ]
                                expected = 0 if source == 0xB8 else expected_codes[source]
                                self.assertEqual(
                                    bf16_cell(output, row, col), expected,
                                    (first_shift, second_shift, row, col),
                                )
                        self.assertEqual(result.slave.captured(base, 1024),
                                         source_panel)
                        self.assertEqual(result.slave.captured(base + 0x800, 1024),
                                         first_weight)
                        self.assertEqual(result.slave.captured(base + 0x1000, 1024),
                                         second_weight)
                        self.assertEqual(result.slave.captured(base + 0x2800, 64),
                                         b"\x5A" * 64)
                biased_source = (
                    EXAMPLES / "virtual_fp8_two_layer_mlp_bias.mlir"
                ).read_text()
                biased_machine = lower(biased_source)
                self.assertEqual(biased_machine.count('"atlas.vpu_binary"'), 2)
                biased_words = object_words(biased_machine)
                self.assertEqual(biased_words, emitted(biased_machine))
                b0_values = tuple(0.5 if col % 2 == 0 else 1.0
                                  for col in range(32))
                b1_values = tuple(0.25 if col % 2 == 0 else 0.5
                                  for col in range(32))
                b0, b1 = bf16_bias_tile(b0_values), bf16_bias_tile(b1_values)
                float_codes = {0x00: 0.0, 0x30: 0.5, 0x38: 1.0,
                               0x40: 2.0, 0xB8: -1.0}
                for first_shift, second_shift in ((0, 0), (3, 5)):
                    with self.subTest(biased_mlp=(first_shift, second_shift)):
                        first_weight = weight(first_shift)
                        second_weight = weight(second_shift)
                        result = cosim_atlas.run_program(
                            model, state, biased_words,
                            preload=[(base, source_panel),
                                     (base + 0x800, first_weight),
                                     (base + 0x1000, second_weight),
                                     (base + 0x1800, b0),
                                     (base + 0x2000, b1),
                                     (base + 0x3000, b"\xA5" * 2048),
                                     (base + 0x3800, b"\x5A" * 64)],
                            max_cycles=100000,
                        )
                        output = result.slave.captured(base + 0x3000, 2048)
                        self.assertTrue(result.halted)
                        self.assertEqual((result.reads, result.writes), (224, 64))
                        for row in range(32):
                            for col in range(32):
                                middle_col = (col + second_shift) % 32
                                source = source_panel[
                                    row * 32 +
                                    (middle_col + first_shift) % 32
                                ]
                                value = max(float_codes[source] +
                                            b0_values[middle_col], 0.0)
                                expected = bf16_bits(value + b1_values[col])
                                self.assertEqual(
                                    bf16_cell(output, row, col), expected,
                                    (first_shift, second_shift, row, col),
                                )
                        for address, data in ((base, source_panel),
                                              (base + 0x800, first_weight),
                                              (base + 0x1000, second_weight),
                                              (base + 0x1800, b0),
                                              (base + 0x2000, b1),
                                              (base + 0x3800, b"\x5A" * 64)):
                            self.assertEqual(result.slave.captured(address, len(data)),
                                             data)
            finally:
                cosim_atlas.CosimCore = original
        finally:
            os.chdir(previous)
            sys.path.remove(str(modelir))


if __name__ == "__main__":
    unittest.main()
