"""Bounded runtime-pointer mailbox call on the selected standalone AtlasCore.

The same LLVM-lowered instruction words execute twice in one core instance.
The host changes a DRAM descriptor and input between launches; Atlas scalar
loads supply the DMA addresses. This is not a C/RISC-V function ABI.
"""

from __future__ import annotations

import importlib.util
import hashlib
import json
import os
import pathlib
import struct
import subprocess
import sys
import tempfile
import unittest

from test_elf_entry_reference import _entry_words
from test_vpu_relu_reference import _emitted
from test_vpu_square_reference import _panel

ROOT = pathlib.Path(__file__).resolve().parents[1]
BIN = pathlib.Path(os.environ.get("ATLAS_OOT_BIN_DIR", ROOT / "build/bin"))
SOURCE = ROOT / "test/examples/vpu_square_mailbox.mlir"
ASSEMBLY = ROOT / "test/examples/vpu_square_mailbox.S"
MAILBOX = 0x90000000
INPUT_A, OUTPUT_A = 0x90002000, 0x90006000
INPUT_B, OUTPUT_B = 0x90003000, 0x90007000
GUARD = 0x9000A000
LAYOUT = ROOT / "test/examples/vpu_square_mailbox_layout.json"

spec = importlib.util.spec_from_file_location(
    "atlas_mailbox_boot_pack_source", ROOT / "tools/atlas_boot_pack.py")
assert spec and spec.loader
boot = importlib.util.module_from_spec(spec)
spec.loader.exec_module(boot)
ABI = boot.validate_layout(json.loads(LAYOUT.read_text()))


def _compile_object(obj: pathlib.Path) -> None:
    llvm_bin = os.environ.get("ATLAS_LLVM_BIN")
    if not llvm_bin:
        raise unittest.SkipTest("set ATLAS_LLVM_BIN for LLVM object check")
    llvm = pathlib.Path(llvm_bin)
    lowered = subprocess.run(
        [str(BIN / "atlas-opt"), "--convert-atlas-to-llvm", str(SOURCE)],
        text=True, capture_output=True, check=True)
    translated = subprocess.run(
        [str(llvm / "mlir-translate"), "--mlir-to-llvmir"],
        input=lowered.stdout, text=True, capture_output=True, check=True)
    subprocess.run(
        [str(llvm / "llc"), "-mtriple=riscv32-unknown-elf", "-mattr=-c",
         "-filetype=obj", "-o", str(obj)], input=translated.stdout.encode(),
        capture_output=True, check=True)


def _object_words() -> tuple[int, ...]:
    llvm = pathlib.Path(os.environ["ATLAS_LLVM_BIN"])
    with tempfile.TemporaryDirectory() as temporary:
        obj = pathlib.Path(temporary) / "atlas_program.o"
        section = pathlib.Path(temporary) / "text.bin"
        _compile_object(obj)
        inspected = subprocess.run(
            [str(llvm / "llvm-readobj"), "--elf-output-style=JSON",
             "--sections", "--symbols", "--relocations", str(obj)],
            text=True, capture_output=True, check=True)
        subprocess.run(
            [str(llvm / "llvm-objcopy"), "--dump-section",
             f".text={section}", str(obj)], capture_output=True, check=True)
        return _entry_words(json.loads(inspected.stdout)[0], section.read_bytes())


def _descriptor(input_address: int, output_address: int) -> bytes:
    return boot.mailbox_descriptor(ABI, input_address, output_address)


class MailboxCallReferenceTest(unittest.TestCase):
    def test_typed_llvm_words_match_selected_assembler(self) -> None:
        selected_root = os.environ.get("ATLAS_ASSEMBLER_ROOT")
        if not selected_root:
            self.skipTest("set ATLAS_ASSEMBLER_ROOT")
        if not os.environ.get("ATLAS_LLVM_BIN"):
            self.skipTest("set ATLAS_LLVM_BIN")
        path = pathlib.Path(selected_root) / "assembler.py"
        spec = importlib.util.spec_from_file_location("selected_mailbox_assembler", path)
        assert spec and spec.loader
        assembler = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(assembler)
        emitted = _emitted(SOURCE)
        self.assertEqual(len(emitted), 39)
        self.assertEqual(emitted, tuple(assembler.assemble(ASSEMBLY.read_text())))
        self.assertEqual(_object_words(), emitted + (0x00008067,))

    def test_descriptor_rejects_bad_runtime_addresses(self) -> None:
        self.assertEqual(len(_descriptor(INPUT_A, OUTPUT_A)), 32)
        self.assertEqual(struct.unpack_from("<II", _descriptor(INPUT_A, OUTPUT_A)),
                         (INPUT_A, OUTPUT_A))
        for pair in ((INPUT_A + 1, OUTPUT_A), (INPUT_A, INPUT_A + 1024),
                     (MAILBOX, OUTPUT_A), (INPUT_A, 0x90100000),
                     (True, OUTPUT_A)):
            with self.subTest(pair=pair), self.assertRaises(ValueError):
                _descriptor(*pair)

        mutated = json.loads(LAYOUT.read_text())
        mutated["call"]["output_pointer_offset_bytes"] = 0
        with self.assertRaisesRegex(ValueError, "fields overlap"):
            boot.validate_layout(mutated)
        mutated = json.loads(LAYOUT.read_text())
        mutated["regions"][2]["address"] = "0x90002000"
        with self.assertRaisesRegex(ValueError, "overlapping DRAM"):
            boot.validate_layout(mutated)

    def test_capsule_records_runtime_pointer_contract_without_sample_data(self) -> None:
        if not os.environ.get("ATLAS_LLVM_BIN"):
            self.skipTest("set ATLAS_LLVM_BIN")
        with tempfile.TemporaryDirectory() as temporary:
            directory = pathlib.Path(temporary)
            obj = directory / "atlas_program.o"
            output = directory / "capsule"
            _compile_object(obj)
            run = subprocess.run(
                [str(BIN / "atlas-boot-pack"), "--object", str(obj),
                 "--source", str(SOURCE), "--atlas-emit", str(BIN / "atlas-emit"),
                 "--layout", str(LAYOUT), "--out", str(output),
                 "--llvm-bin", os.environ["ATLAS_LLVM_BIN"]],
                text=True, capture_output=True, check=True)
            self.assertEqual(json.loads(run.stdout)["status"], "PASS")
            manifest = json.loads((output / "manifest.json").read_text())
            code = (output / "program.bin").read_bytes()
            self.assertEqual(struct.unpack(f"<{len(code) // 4}I", code), _object_words())
            self.assertEqual(manifest["dram"], ABI)
            self.assertEqual(manifest["arguments"][0]["mailbox_offset_bytes"], 0)
            self.assertEqual(manifest["returns"][0]["mailbox_offset_bytes"], 4)
            self.assertEqual(manifest["completion"]["ecall_pc_word"], 38)
            self.assertFalse(manifest["program_mailbox_binding_proved_by_packer"])
            self.assertNotIn(_panel(0)[0], code)
            self.assertNotIn(_descriptor(INPUT_A, OUTPUT_A), code)

    def test_same_core_restarts_with_changed_runtime_pointers_and_data(self) -> None:
        keys = ("ATLAS_ARC_MODEL", "ATLAS_ARC_STATE", "ATLAS_MODELIR_ROOT",
                "ATLAS_LLVM_BIN")
        if not all(os.environ.get(key) for key in keys):
            self.skipTest("set selected-source-linked ARC, ModeLIR, and LLVM paths")
        model, state, modelir = (pathlib.Path(os.environ[key]).resolve(strict=True)
                                 for key in keys[:3])
        words = _object_words()
        self.assertEqual(words[:-1], _emitted(SOURCE))
        source_a, expected_a = _panel(0)
        source_b, expected_b = _panel(5)
        self.assertNotEqual(source_a, source_b)
        self.assertNotEqual(expected_a, expected_b)
        old_cwd = pathlib.Path.cwd()
        sys.path.insert(0, str(modelir))
        try:
            os.chdir(modelir)
            from mlc.backends import cosim_atlas
            from mlc.backends.protocols import TileLinkAdapter, TileLinkSlave
            from mlc.backends.cosim_core import large_stack_call

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

            def execute_twice():
                core = SelectedCore(model, state)
                core.reset()
                imem = TileLinkAdapter(core, "imemTL")
                csr = TileLinkAdapter(core, "csrTL")
                slave = TileLinkSlave(core, "dmaTL", size_bytes=1 << 20, beat_bytes=32)
                for index, word in enumerate(words):
                    imem.put(0x20000 + 4 * index, word, size=2)
                slave.preload(INPUT_A, source_a)
                slave.preload(INPUT_B, b"\xA3" * 2048)
                slave.preload(OUTPUT_A, b"\xA5" * 2048)
                slave.preload(OUTPUT_B, b"\xA5" * 2048)
                slave.preload(GUARD, b"\x5A" * 32)

                observations = []
                for input_address, output_address, replacement in (
                        (INPUT_A, OUTPUT_A, None),
                        (INPUT_B, OUTPUT_B, source_b)):
                    if replacement is not None:
                        slave.preload(input_address, replacement)
                    slave.preload(MAILBOX, _descriptor(input_address, output_address))
                    old_reads, old_writes = slave.reads, slave.writes
                    csr.put(0x18, 1, size=2)
                    started = False
                    halted = False
                    seen_pcs = set()
                    for cycle in range(9000):
                        slave.step()
                        if core.peek("scalar/pc_ctrl/io_s1_valid"):
                            seen_pcs.add(core.peek("scalar/pc_ctrl/io_s1_pc"))
                        h = core.peek("io_halted") == 1
                        if not h:
                            started = True
                        core.tick()
                        if started and h:
                            halted = True
                            break
                    observations.append({
                        "halted": halted, "cycle": cycle,
                        "reads": slave.reads - old_reads,
                        "writes": slave.writes - old_writes,
                        "pcs": seen_pcs,
                        "output_a": slave.captured(OUTPUT_A, 2048),
                        "output_b": slave.captured(OUTPUT_B, 2048),
                        "input_a": slave.captured(INPUT_A, 2048),
                        "input_b": slave.captured(INPUT_B, 2048),
                        "guard": slave.captured(GUARD, 32),
                    })
                return observations

            first, second = large_stack_call(execute_twice)
        finally:
            os.chdir(old_cwd)
            sys.path.remove(str(modelir))

        self.assertTrue(first["halted"])
        self.assertTrue(second["halted"])
        self.assertEqual((first["reads"], first["writes"]), (65, 64))
        self.assertEqual((second["reads"], second["writes"]), (65, 64))
        self.assertEqual(first["output_a"], expected_a)
        self.assertEqual(first["output_b"], b"\xA5" * 2048)
        self.assertEqual(second["output_a"], expected_a)
        self.assertEqual(second["output_b"], expected_b)
        self.assertEqual(first["input_a"], source_a)
        self.assertEqual(second["input_a"], source_a)
        self.assertEqual(second["input_b"], source_b)
        self.assertEqual(first["guard"], b"\x5A" * 32)
        self.assertEqual(second["guard"], b"\x5A" * 32)
        for run in (first, second):
            self.assertIn(38, run["pcs"])  # ECALL
            self.assertNotIn(39, run["pcs"])  # LLVM RET
        receipt_path = os.environ.get("ATLAS_MAILBOX_RECEIPT")
        if receipt_path:
            receipt = {
                "status": "PASS", "execution_tier": "selected standalone AtlasCore ARC",
                "core_instance_count": 1, "imem_load_count": 1, "csr_start_count": 2,
                "program_words": len(words),
                "program_sha256": hashlib.sha256(struct.pack(f"<{len(words)}I", *words)).hexdigest(),
                "source_mlir_sha256": hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
                "arc_model_sha256": hashlib.sha256(model.read_bytes()).hexdigest(),
                "arc_state_sha256": hashlib.sha256(state.read_bytes()).hexdigest(),
                "runs": [
                    {"input_address": input_address, "output_address": output_address,
                     "cycles": observed["cycle"], "dma_reads": observed["reads"],
                     "dma_writes": observed["writes"],
                     "output_sha256": hashlib.sha256(expected).hexdigest(),
                     "input_preserved": True, "guard_preserved": True}
                    for observed, input_address, output_address, expected in (
                        (first, INPUT_A, OUTPUT_A, expected_a),
                        (second, INPUT_B, OUTPUT_B, expected_b))],
            }
            path = pathlib.Path(receipt_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    unittest.main()
