"""Bounded ACT-independent Atlas reset-entry package and core launch tests."""

from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import os
import pathlib
import struct
import subprocess
import sys
import tempfile
import unittest

from test_elf_entry_reference import _compile_elf
from test_vpu_relu_reference import _emitted
from test_vpu_square_reference import SOURCE, _panel

from selected_core_runtime import run_selected_program

ROOT = pathlib.Path(__file__).resolve().parents[1]
BIN = pathlib.Path(os.environ.get("ATLAS_OOT_BIN_DIR", ROOT / "build/bin"))
LAYOUT = ROOT / "test/examples/vpu_square_boot_layout.json"
PACKER = BIN / "atlas-boot-pack"

spec = importlib.util.spec_from_file_location(
    "atlas_boot_pack_source", ROOT / "tools/atlas_boot_pack.py")
assert spec and spec.loader
boot = importlib.util.module_from_spec(spec)
spec.loader.exec_module(boot)


def _compile_object(path: pathlib.Path) -> None:
    llvm = os.environ.get("ATLAS_LLVM_BIN")
    if not llvm:
        raise unittest.SkipTest("set ATLAS_LLVM_BIN for LLVM object package")
    tools = pathlib.Path(llvm)
    lowered = subprocess.run([str(BIN / "atlas-opt"), "--convert-atlas-to-llvm",
                              str(SOURCE)], capture_output=True, text=True, check=True)
    translated = subprocess.run([str(tools / "mlir-translate"), "--mlir-to-llvmir"],
                                input=lowered.stdout, capture_output=True,
                                text=True, check=True)
    subprocess.run([str(tools / "llc"), "-mtriple=riscv32-unknown-elf",
                    "-mattr=-c", "-filetype=obj", "-o", str(path)],
                   input=translated.stdout.encode(), capture_output=True, check=True)


def _pack(obj: pathlib.Path, output: pathlib.Path,
          source: pathlib.Path = SOURCE, layout: pathlib.Path = LAYOUT,
          check: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [str(PACKER), "--object", str(obj), "--source", str(source),
         "--atlas-emit", str(BIN / "atlas-emit"), "--layout", str(layout),
         "--out", str(output), "--llvm-bin", os.environ["ATLAS_LLVM_BIN"]],
        capture_output=True, text=True, check=check)


class BootCapsuleTest(unittest.TestCase):
    def test_package_binds_checked_ir_object_and_declared_memory(self) -> None:
        if not os.environ.get("ATLAS_LLVM_BIN"):
            self.skipTest("set ATLAS_LLVM_BIN")
        with tempfile.TemporaryDirectory() as temporary:
            directory = pathlib.Path(temporary)
            obj, output = directory / "atlas.o", directory / "capsule"
            _compile_object(obj)
            run = _pack(obj, output)
            self.assertEqual(json.loads(run.stdout)["status"], "PASS")
            manifest = json.loads((output / "manifest.json").read_text())
            program = (output / "program.bin").read_bytes()
            words = struct.unpack(f"<{len(program) // 4}I", program)
            emitted = _emitted(SOURCE)
            self.assertEqual(words, emitted + (boot.RET,))
            self.assertEqual((len(words), manifest["program_words"]), (37, 37))
            self.assertEqual(manifest["program_sha256"], hashlib.sha256(program).hexdigest())
            self.assertEqual(manifest["packer_sha256"], hashlib.sha256(PACKER.read_bytes()).hexdigest())
            self.assertEqual(manifest["source_mlir_sha256"],
                             hashlib.sha256(SOURCE.read_bytes()).hexdigest())
            self.assertEqual(manifest["source_object_sha256"],
                             hashlib.sha256(obj.read_bytes()).hexdigest())
            self.assertEqual((manifest["entry_pc_word"], manifest["imem_tl_byte_base"]),
                             (0, 0x20000))
            self.assertEqual((manifest["start_csr_tl_byte_address"],
                              manifest["start_csr_value"]), (0x18, 1))
            self.assertEqual(manifest["completion"]["ecall_pc_word"], 35)
            self.assertEqual(manifest["completion"]["unreachable_llvm_ret_pc_word"], 36)
            self.assertFalse(manifest["register_initialization_proved"])
            self.assertEqual((manifest["arguments"], manifest["returns"]), ([], []))
            self.assertIn("not a callable ABI", manifest["scope"])
            regions = {row["name"]: row for row in manifest["dram"]["regions"]}
            self.assertEqual(set(regions), {"input", "output", "guard"})
            self.assertEqual((regions["input"]["address"], regions["input"]["size_bytes"]),
                             (0x90000000, 2048))
            self.assertEqual((regions["output"]["address"], regions["output"]["size_bytes"]),
                             (0x90000800, 2048))
            self.assertNotIn(_panel(5)[0], program)

            # Stale success destinations, a different source, and bad regions
            # fail without publishing another successful capsule.
            self.assertNotEqual(_pack(obj, output, check=False).returncode, 0)
            wrong_source = ROOT / "test/examples/vpu_cube_pair.mlir"
            wrong = _pack(obj, directory / "wrong", source=wrong_source, check=False)
            self.assertNotEqual(wrong.returncode, 0)
            self.assertIn("differs from checked Atlas source", wrong.stderr)
            self.assertFalse((directory / "wrong").exists())
            bad_layout = json.loads(LAYOUT.read_text())
            bad_layout["regions"][1]["address"] = "0x90000020"
            bad_path = directory / "overlap.json"
            bad_path.write_text(json.dumps(bad_layout))
            invalid = _pack(obj, directory / "overlap", layout=bad_path, check=False)
            self.assertNotEqual(invalid.returncode, 0)
            self.assertIn("overlapping DRAM regions", invalid.stderr)
            self.assertFalse((directory / "overlap").exists())

    def test_packager_rejects_moved_entry_relocation_and_truncation(self) -> None:
        metadata, text = _compile_elf()
        self.assertEqual(len(boot.validate_text(metadata, text)), 37)
        moved = copy.deepcopy(metadata)
        next(row["Symbol"] for row in moved["Symbols"]
             if row["Symbol"]["Name"]["Name"] == "atlas_program")["Value"] = 4
        with self.assertRaisesRegex(ValueError, "entry zero"):
            boot.validate_text(moved, text)
        relocated = copy.deepcopy(metadata)
        group = relocated["Relocations"][0]
        sections = {row["Section"]["Index"]: row["Section"]
                    for row in relocated["Sections"]}
        sections[group["SectionIndex"]]["Info"] = 2
        with self.assertRaisesRegex(ValueError, "unapplied .text relocation"):
            boot.validate_text(relocated, text)
        with self.assertRaisesRegex(ValueError, "complete .text"):
            boot.validate_text(metadata, text[:-4])

    def test_packaged_complete_function_executes_on_selected_core(self) -> None:
        keys = ("ATLAS_ARC_MODEL", "ATLAS_ARC_STATE", "ATLAS_MODELIR_ROOT",
                "ATLAS_LLVM_BIN")
        if not all(os.environ.get(key) for key in keys):
            self.skipTest("set selected-source-linked ARC, ModeLIR, and LLVM paths")
        model, state, modelir = (pathlib.Path(os.environ[key]).resolve(strict=True)
                                 for key in keys[:3])
        with tempfile.TemporaryDirectory() as temporary:
            directory = pathlib.Path(temporary)
            obj, output = directory / "atlas.o", directory / "capsule"
            _compile_object(obj)
            _pack(obj, output)
            manifest = json.loads((output / "manifest.json").read_text())
            program = (output / "program.bin").read_bytes()
            words = struct.unpack(f"<{len(program) // 4}I", program)
            regions = {row["name"]: row for row in manifest["dram"]["regions"]}
            source, expected = _panel(5)
            seen_pcs: set[int] = set()
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

                def observe(core) -> None:
                    if core.peek("scalar/pc_ctrl/io_s1_valid"):
                        seen_pcs.add(core.peek("scalar/pc_ctrl/io_s1_pc"))

                cosim_atlas.CosimCore = SelectedCore
                try:
                    result = run_selected_program(cosim_atlas,
                        model, state, words,
                        preload=[(regions["input"]["address"], source),
                                 (regions["output"]["address"], b"\xA5" * 2048),
                                 (regions["guard"]["address"], b"\x5A" * 32)],
                        imem_base=manifest["imem_tl_byte_base"], max_cycles=8000,
                        on_cycle=observe)
                finally:
                    cosim_atlas.CosimCore = original
            finally:
                os.chdir(old_cwd)
                sys.path.remove(str(modelir))
            self.assertTrue(result.halted)
            self.assertEqual((result.reads, result.writes), (64, 64))
            self.assertEqual(result.slave.captured(regions["output"]["address"], 2048),
                             expected)
            self.assertEqual(result.slave.captured(regions["input"]["address"], 2048),
                             source)
            self.assertEqual(result.slave.captured(regions["guard"]["address"], 32),
                             b"\x5A" * 32)
            self.assertIn(manifest["completion"]["ecall_pc_word"], seen_pcs)
            self.assertNotIn(manifest["completion"]["unreachable_llvm_ret_pc_word"],
                             seen_pcs)


if __name__ == "__main__":
    unittest.main()
