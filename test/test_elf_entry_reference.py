"""Execute a complete LLVM-produced Atlas function as a bounded boot entry.

This is a diagnostic baremetal entry contract, not a C-callable Atlas ABI:
reset starts at symbol offset zero, DRAM is preloaded at fixed addresses,
and the source stream halts with ECALL before LLVM's appended RET.
"""

from __future__ import annotations

import copy
import json
import os
import pathlib
import struct
import subprocess
import sys
import tempfile
import unittest

from test_vpu_relu_reference import _emitted
from test_vpu_square_reference import SOURCE, _panel

from selected_core_runtime import run_selected_program

ROOT = pathlib.Path(__file__).resolve().parents[1]
BIN = pathlib.Path(os.environ.get("ATLAS_OOT_BIN_DIR", ROOT / "build/bin"))


def _compile_elf() -> tuple[dict, bytes]:
    llvm = os.environ.get("ATLAS_LLVM_BIN")
    if not llvm:
        raise unittest.SkipTest("set ATLAS_LLVM_BIN for LLVM object entry test")
    tools = pathlib.Path(llvm)
    lowered = subprocess.run([str(BIN / "atlas-opt"), "--convert-atlas-to-llvm",
                              str(SOURCE)], capture_output=True, text=True, check=True)
    translated = subprocess.run([str(tools / "mlir-translate"), "--mlir-to-llvmir"],
                                input=lowered.stdout, capture_output=True,
                                text=True, check=True)
    with tempfile.TemporaryDirectory() as temporary:
        obj = pathlib.Path(temporary) / "atlas_program.o"
        section = pathlib.Path(temporary) / "text.bin"
        subprocess.run([str(tools / "llc"), "-mtriple=riscv32-unknown-elf",
                        "-mattr=-c", "-filetype=obj", "-o", str(obj)],
                       input=translated.stdout.encode(), capture_output=True, check=True)
        inspected = subprocess.run(
            [str(tools / "llvm-readobj"), "--elf-output-style=JSON",
             "--sections", "--symbols", "--relocations", str(obj)],
            capture_output=True, text=True, check=True)
        subprocess.run([str(tools / "llvm-objcopy"), "--dump-section",
                        f".text={section}", str(obj)], capture_output=True, check=True)
        return json.loads(inspected.stdout)[0], section.read_bytes()


def _entry_words(metadata: dict, text: bytes) -> tuple[int, ...]:
    if metadata["FileSummary"]["Format"] != "elf32-littleriscv":
        raise ValueError("entry requires ELF32 little-endian RISC-V")
    sections = {row["Section"]["Index"]: row["Section"]
                for row in metadata["Sections"]}
    executable = [section for section in sections.values()
                  if section["Name"]["Name"] == ".text"]
    if len(executable) != 1 or executable[0]["Size"] != len(text):
        raise ValueError("one complete .text section required")
    text_section = executable[0]
    flags = {flag["Name"] for flag in text_section["Flags"]["Flags"]}
    if not {"SHF_ALLOC", "SHF_EXECINSTR"} <= flags:
        raise ValueError(".text must be allocated executable code")
    for group in metadata["Relocations"]:
        relocation_section = sections[group["SectionIndex"]]
        if relocation_section["Info"] == text_section["Index"] and group["Relocs"]:
            raise ValueError("unapplied .text relocation")
    functions = [row["Symbol"] for row in metadata["Symbols"]
                 if row["Symbol"]["Name"]["Name"] == "atlas_program"]
    if len(functions) != 1:
        raise ValueError("one atlas_program symbol required")
    function = functions[0]
    if (function["Type"]["Name"] != "Function" or
            function["Section"]["Name"] != ".text" or
            function["Value"] != 0 or function["Size"] != len(text)):
        raise ValueError("atlas_program must occupy .text from entry zero")
    if len(text) % 4:
        raise ValueError("entry requires full 32-bit instruction words")
    words = struct.unpack(f"<{len(text) // 4}I", text)
    if any(word & 3 != 3 for word in words):
        raise ValueError("entry contains compressed or misaligned instruction")
    if len(words) < 2 or words[-2:] != (0x00000073, 0x00008067):
        raise ValueError("entry must halt by ECALL before LLVM RET")
    return words


class ElfEntryReferenceTest(unittest.TestCase):
    def test_complete_llvm_function_has_checked_boot_entry_layout(self) -> None:
        metadata, text = _compile_elf()
        words = _entry_words(metadata, text)
        emitted = _emitted(SOURCE)
        self.assertEqual(len(emitted), 36)
        self.assertEqual(len(words), 37)
        self.assertEqual(words[:36], emitted)
        self.assertEqual(words[-1], 0x00008067)
        self.assertEqual(len(text), 148)

        moved = copy.deepcopy(metadata)
        next(row["Symbol"] for row in moved["Symbols"]
             if row["Symbol"]["Name"]["Name"] == "atlas_program")["Value"] = 4
        with self.assertRaisesRegex(ValueError, "entry zero"):
            _entry_words(moved, text)
        relocated = copy.deepcopy(metadata)
        group = relocated["Relocations"][0]
        sections = {row["Section"]["Index"]: row["Section"]
                    for row in relocated["Sections"]}
        sections[group["SectionIndex"]]["Info"] = 2  # .text
        with self.assertRaisesRegex(ValueError, "unapplied .text relocation"):
            _entry_words(relocated, text)
        with self.assertRaisesRegex(ValueError, "complete .text"):
            _entry_words(metadata, text[:-4])

    def test_complete_llvm_function_boot_entry_runs_on_selected_core(self) -> None:
        keys = ("ATLAS_ARC_MODEL", "ATLAS_ARC_STATE", "ATLAS_MODELIR_ROOT",
                "ATLAS_LLVM_BIN")
        if not all(os.environ.get(key) for key in keys):
            self.skipTest("set selected-source-linked ARC, ModeLIR, and LLVM paths")
        model, state, modelir = (pathlib.Path(os.environ[key]).resolve(strict=True)
                                 for key in keys[:3])
        metadata, text = _compile_elf()
        words = _entry_words(metadata, text)
        emitted = _emitted(SOURCE)
        self.assertEqual(words[:len(emitted)], emitted)
        # The selected standalone PcControl exposes instruction-word indices.
        ret_pc = len(emitted)
        ecall_pc = ret_pc - 1
        source, expected = _panel(5)
        seen_stage1_pcs: set[int] = set()
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
                    seen_stage1_pcs.add(core.peek("scalar/pc_ctrl/io_s1_pc"))

            cosim_atlas.CosimCore = SelectedCore
            try:
                result = run_selected_program(cosim_atlas,
                    model, state, words,
                    preload=[(0x90000000, source[:1024]),
                             (0x90000400, source[1024:]),
                             (0x90000800, b"\xA5" * 1024),
                             (0x90000C00, b"\xA5" * 1024),
                             (0x90001000, b"\x5A" * 32)],
                    max_cycles=8000, on_cycle=observe,
                )
            finally:
                cosim_atlas.CosimCore = original
        finally:
            os.chdir(old_cwd)
            sys.path.remove(str(modelir))
        observed = (result.slave.captured(0x90000800, 1024) +
                    result.slave.captured(0x90000C00, 1024))
        self.assertTrue(result.halted)
        self.assertEqual((result.reads, result.writes), (64, 64))
        self.assertEqual(observed, expected)
        self.assertEqual(result.slave.captured(0x90000000, 2048), source)
        self.assertEqual(result.slave.captured(0x90001000, 32), b"\x5A" * 32)
        self.assertIn(ecall_pc, seen_stage1_pcs)
        self.assertNotIn(ret_pc, seen_stage1_pcs)


if __name__ == "__main__":
    unittest.main()
