"""Reproducible Atlas/LLVM MLIR handoff and bounded composition diagnostics."""

from __future__ import annotations

import json
import hashlib
import os
import pathlib
import shutil
import struct
import subprocess
import sys
import tempfile
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[1]
BIN = pathlib.Path(os.environ.get("ATLAS_OOT_BIN_DIR", ROOT / "build/bin"))
NAMES = ("handoff_mlp_tile", "handoff_attention_tile")
BASE = 0x90000000


def words(name: str) -> tuple[int, ...]:
    result = subprocess.run([str(BIN / "atlas-emit"),
                             str(ROOT / "test/examples" / f"{name}.mlir")],
                            text=True, capture_output=True, check=True)
    return tuple(int(line, 16) for line in result.stdout.splitlines())


def run_core(name: str, tiles: tuple[bytes, bytes, bytes],
             program_words: tuple[int, ...] | None = None):
    for key in ("ATLAS_ARC_MODEL", "ATLAS_ARC_STATE", "ATLAS_MODELIR_ROOT"):
        if not os.environ.get(key):
            raise unittest.SkipTest(f"set {key} for selected standalone core")
    model, state, modelir = (pathlib.Path(os.environ[key]).resolve(strict=True)
                             for key in ("ATLAS_ARC_MODEL", "ATLAS_ARC_STATE",
                                         "ATLAS_MODELIR_ROOT"))
    previous = pathlib.Path.cwd()
    sys.path.insert(0, str(modelir))
    try:
        os.chdir(modelir)
        from mlc.backends import cosim_atlas

        original = cosim_atlas.CosimCore

        class SelectedCore(original):
            def peek(self, signal: str) -> int:
                return super().peek("scalar/halt_now" if signal == "io_halted"
                                    else signal)

            def poke(self, signal: str, value: int) -> None:
                if signal in ("io_dmaTL_d_bits_opcode", "io_dmaTL_d_bits_size"):
                    if signal in self._S:
                        raise AssertionError(f"unexpected live ARC input {signal}")
                    return
                super().poke(signal, value)

        cosim_atlas.CosimCore = SelectedCore
        try:
            preloads = [(BASE + 0x400 * index, tile)
                        for index, tile in enumerate(tiles)]
            preloads += [(BASE + 0xC00, b"\xA5" * 1024),
                         (BASE + 0x1000, b"\xA5" * 1024),
                         (BASE + 0x1400, b"\x5A" * 32)]
            result = cosim_atlas.run_program(model, state,
                                             words(name) if program_words is None
                                             else program_words,
                                             preload=preloads, max_cycles=20000)
            for index, tile in enumerate(tiles):
                assert result.slave.captured(BASE + 0x400 * index, 1024) == tile
            assert result.slave.captured(BASE + 0x1400, 32) == b"\x5A" * 32
            output = (result.slave.captured(BASE + 0xC00, 1024) +
                      result.slave.captured(BASE + 0x1000, 1024))
            return result, output
        finally:
            cosim_atlas.CosimCore = original
    finally:
        os.chdir(previous)
        sys.path.remove(str(modelir))


def cell(output: bytes, row: int, col: int) -> int:
    half = col // 16
    offset = half * 1024 + (row * 16 + col % 16) * 2
    return struct.unpack_from("<H", output, offset)[0]


class HandoffExamplesTest(unittest.TestCase):
    def test_linked_elf_text_executes_on_selected_core(self) -> None:
        llvm = os.environ.get("ATLAS_LLVM_BIN")
        if not llvm:
            self.skipTest("set ATLAS_LLVM_BIN for ELF text extraction")
        cases = (
            (NAMES[0], "mlp_tile", (bytes([0x38]) * 1024,) * 3, 0x4480),
            (NAMES[1], "attention_tile",
             (bytes(1024), bytes(1024), bytes([0x38]) * 1024), 0x3F80),
        )
        with tempfile.TemporaryDirectory() as temporary:
            for name, folder, tiles, expected in cases:
                with self.subTest(name=name):
                    elf = ROOT / "examples/handoff" / folder / "07-riscv-linked.elf"
                    text_file = pathlib.Path(temporary) / f"{name}.text"
                    subprocess.run([
                        str(pathlib.Path(llvm) / "llvm-objcopy"),
                        "--dump-section", f".text={text_file}", str(elf),
                    ], capture_output=True, check=True)
                    raw = text_file.read_bytes()
                    linked_words = struct.unpack(f"<{len(raw) // 4}I", raw)
                    self.assertEqual(linked_words, words(name) + (0x00008067,))
                    result, output = run_core(name, tiles, linked_words)
                    self.assertTrue(result.halted)
                    self.assertEqual(struct.unpack("<1024H", output),
                                     (expected,) * 1024)

    def test_llvm_mlir_and_object_words_match_emitter(self) -> None:
        llvm = os.environ.get("ATLAS_LLVM_BIN")
        if not llvm:
            self.skipTest("set ATLAS_LLVM_BIN for LLVM object handoff")
        linker = os.environ.get("ATLAS_LLD") or shutil.which("ld.lld")
        if not linker:
            self.skipTest("set ATLAS_LLD or install ld.lld for linked ELF handoff")
        with tempfile.TemporaryDirectory() as temporary:
            output = pathlib.Path(temporary) / "handoff"
            subprocess.run([
                sys.executable, str(ROOT / "tools/export_llvm_handoff.py"),
                "--atlas-bin-dir", str(BIN), "--llvm-bin-dir", llvm,
                "--linker", linker,
                "--output-dir", str(output),
            ], check=True, capture_output=True, text=True)
            manifest = json.loads((output / "manifest.json").read_text())
            self.assertEqual(manifest["schema"], "atlas-llvm-handoff-v3")
            stale = subprocess.run([
                sys.executable, str(ROOT / "tools/export_llvm_handoff.py"),
                "--atlas-bin-dir", str(BIN), "--llvm-bin-dir", llvm,
                "--linker", linker,
                "--output-dir", str(output),
            ], capture_output=True, text=True)
            self.assertNotEqual(stale.returncode, 0)
            self.assertIn("must be empty", stale.stderr)
            self.assertEqual(json.loads((output / "manifest.json").read_text()),
                             manifest)
            for name in NAMES:
                with self.subTest(name=name):
                    folder = (ROOT / "examples/handoff" /
                              ("mlp_tile" if name == NAMES[0] else "attention_tile"))
                    stages = {
                        "atlas.mlir": "01-atlas-machine.mlir",
                        "llvm-structured.mlir": "02-llvm-structured.mlir",
                        "llvm.mlir": "03-llvm-encoded.mlir",
                        "ll": "04-llvm-ir.ll",
                        "s": "05-riscv-words.s",
                        "o": "06-riscv-relocatable.o",
                        "elf": "07-riscv-linked.elf",
                        "disasm.txt": "08-riscv-disassembly.txt",
                        "words.txt": "atlas-words.txt",
                        "word-map.json": "atlas-word-map.json",
                    }
                    for suffix, snapshot in stages.items():
                        self.assertEqual(
                            (output / f"{name}.{suffix}").read_bytes(),
                            (folder / snapshot).read_bytes(),
                            f"{name}: stage {snapshot} differs from fresh LLVM output",
                        )
                    source = (ROOT / "test/examples" / f"{name}.mlir").read_text()
                    self.assertEqual(source.count('"atlas.branch"'), 1)
                    self.assertEqual(source.count('kind = "lw"'), 8)
                    self.assertEqual(source.count('"atlas.scalar_store"'), 8)
                    self.assertEqual(manifest["examples"][name]["word_count"],
                                     len(words(name)))
                    self.assertTrue(manifest["examples"][name]
                                    ["object_prefix_matches_emitter"])
                    llvm_mlir = (output / f"{name}.llvm.mlir").read_text()
                    snapshot = (folder / "03-llvm-encoded.mlir").read_text()
                    self.assertEqual(llvm_mlir.rstrip(), snapshot.rstrip())
                    self.assertEqual(llvm_mlir.count("llvm.inline_asm"), 1)
                    self.assertIn("has_side_effects", llvm_mlir)
                    structured = (output / f"{name}.llvm-structured.mlir").read_text()
                    self.assertEqual(structured.count("llvm.call @atlas_emit_"),
                                     len(words(name)))
                    self.assertNotIn("llvm.inline_asm", structured)
                    self.assertEqual(structured.count('atlas.availability = "unknown"'),
                                     len(words(name)))
                    object_text = (output / f"{name}.text.bin").read_bytes()
                    expected = b"".join(struct.pack("<I", word)
                                        for word in words(name))
                    self.assertTrue(object_text.startswith(expected))
                    self.assertEqual(
                        (output / f"{name}.elf.text.bin").read_bytes(),
                        object_text,
                    )
                    self.assertTrue(manifest["examples"][name]
                                    ["linked_text_matches_object"])
                    map_bytes = (output / f"{name}.word-map.json").read_bytes()
                    self.assertEqual(
                        map_bytes,
                        (folder / "atlas-word-map.json").read_bytes(),
                    )
                    self.assertEqual(hashlib.sha256(map_bytes).hexdigest(),
                                     manifest["examples"][name]["word_map_sha256"])
                    word_map = json.loads(map_bytes)
                    self.assertEqual(word_map["schema"], "atlas.machine_word_map.v1")
                    mapped = word_map["operations"]
                    self.assertEqual(len(mapped), len(words(name)))
                    self.assertTrue(all(
                        (row["word_index"], row["byte_offset"], row["word_hex"])
                        == (index, 4 * index, f"{word:08x}")
                        for index, (row, word) in enumerate(zip(mapped, words(name)))
                    ))
                    self.assertTrue(all(
                        row["effect_scope"] == "conservative_physical_state_read_write"
                        and row["availability"] == "unknown"
                        for row in mapped
                    ))
                    self.assertTrue(any(row["operation"] == "atlas.mxu_matmul"
                                        for row in mapped))
                    self.assertTrue(any(row["operation"] == "atlas.delay"
                                        and row["explicit_stall_cycles"] > 0
                                        for row in mapped))
                    branches = [row for row in mapped
                                if row["operation"] == "atlas.branch"]
                    self.assertEqual(len(branches), 1)
                    branch = branches[0]
                    self.assertLess(branch["direct_target_word_index"],
                                    branch["word_index"])
                    self.assertEqual(
                        mapped[branch["word_index"] + 1]
                        ["delay_slot_for_word_index"], branch["word_index"]
                    )

    def test_structured_llvm_rechecks_fields_and_words(self) -> None:
        source = ROOT / "test/examples/handoff_mlp_tile.mlir"
        structured = subprocess.run([
            str(BIN / "atlas-opt"), "--convert-atlas-to-llvm-calls", str(source),
        ], capture_output=True, text=True, check=True).stdout
        direct = subprocess.run([
            str(BIN / "atlas-opt"), "--convert-atlas-to-llvm", str(source),
        ], capture_output=True, text=True, check=True).stdout
        checked = subprocess.run([
            str(BIN / "atlas-opt"), "--finalize-atlas-llvm-calls", "-",
        ], input=structured, capture_output=True, text=True, check=True).stdout
        self.assertEqual(checked, direct)
        self.assertIn("llvm.call @atlas_emit_mxu_matmul", structured)
        for original, replacement in (
            ("dst = 28 : i32", "dst = 27 : i32"),
            ("atlas.word = 1052179 : i32", "atlas.word = 1052178 : i32"),
            ("atlas.word_index = 0 : i32", "atlas.word_index = 1 : i32"),
        ):
            with self.subTest(original=original):
                self.assertIn(original, structured)
                changed = structured.replace(original, replacement, 1)
                result = subprocess.run([
                    str(BIN / "atlas-opt"), "--finalize-atlas-llvm-calls", "-",
                ], input=changed, capture_output=True, text=True)
                self.assertNotEqual(result.returncode, 0)

    def test_uniform_directed_programs_on_selected_core(self) -> None:
        cases = (
            (NAMES[0], (bytes([0x38]) * 1024,) * 3, 0x4480),
            (NAMES[1], (bytes(1024), bytes(1024),
                        bytes([0x38]) * 1024), 0x3F80),
        )
        for name, tiles, expected in cases:
            with self.subTest(name=name):
                result, output = run_core(name, tiles)
                self.assertTrue(result.halted)
                self.assertEqual((result.reads, result.writes), (96, 64))
                self.assertEqual(struct.unpack("<1024H", output),
                                 (expected,) * 1024)

    def test_sparse_mlp_relayout_preserves_logical_position(self) -> None:
        activation = bytearray(1024)
        activation[20] = 0x38
        identity = bytearray(1024)
        for index in range(32):
            identity[32 * index + index] = 0x38
        result, output = run_core(NAMES[0],
                                  (bytes(activation), bytes(identity),
                                   bytes(identity)))
        self.assertTrue(result.halted)
        # The selected PACK joins physical rows; the scalar loop restores the
        # logical row before the second MXU consumes the FP8 tile.
        self.assertEqual(cell(output, 0, 20), 0x3F80)
        self.assertEqual(cell(output, 16, 4), 0)
        self.assertEqual(sum(code != 0 for code in struct.unpack("<1024H", output)), 1)

    def test_mlp_relayout_covers_all_rows_and_columns(self) -> None:
        fp8 = (0x00, 0x30, 0x38, 0x40)
        bf16 = (0x0000, 0x3F00, 0x3F80, 0x4000)
        activation = bytes(fp8[(row * 7 + col * 3) % len(fp8)]
                           for row in range(32) for col in range(32))
        identity = bytearray(1024)
        for index in range(32):
            identity[32 * index + index] = 0x38
        result, output = run_core(NAMES[0],
                                  (activation, bytes(identity), bytes(identity)))
        self.assertTrue(result.halted)
        for row in range(32):
            for col in range(32):
                with self.subTest(row=row, col=col):
                    self.assertEqual(cell(output, row, col),
                                     bf16[(row * 7 + col * 3) % len(bf16)])

    def test_sparse_attention_relayout_preserves_logical_position(self) -> None:
        query = bytearray(1024)
        query[0] = 0x38
        query[17 * 32 + 1] = 0x38
        key = bytearray(1024)
        key[20 * 32] = 0x38
        key[3 * 32 + 1] = 0x38
        value = bytearray(1024)
        for index in range(32):
            value[32 * index + index] = 0x38
        result, output = run_core(NAMES[1],
                                  (bytes(query), bytes(key), bytes(value)))
        self.assertTrue(result.halted)
        # The elevated key-20 probability remains in row 0, column 20
        # after the packed FP8 tile has been reordered for the PV MXU.
        self.assertEqual(cell(output, 0, 20), 0x3D20)
        self.assertEqual(cell(output, 16, 4), 0x3D00)
        self.assertEqual(cell(output, 17, 3), 0x3D20)
        self.assertEqual(struct.unpack("<1024H", output).count(0x3D20), 2)


if __name__ == "__main__":
    unittest.main()
