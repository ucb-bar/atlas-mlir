"""Hand-authored branch path through encoding, LLVM lowering and optional ARC."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import pathlib
import subprocess
import sys
import tempfile
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]
SOURCE = ROOT / "test/examples/branch_delay.mlir"
ASSEMBLY = ROOT / "test/examples/branch_delay.S"
EXPECTED = (
    0x00000513, 0x00000363, 0x0AA00513, 0x0BB00513,
    0xF5750093, 0xC1009073, 0x00000073,
)


def _emitted() -> tuple[int, ...]:
    run = subprocess.run([str(ROOT / "build/bin/atlas-emit"), str(SOURCE)],
                         text=True, capture_output=True, check=False)
    if run.returncode:
        raise AssertionError(run.stderr)
    return tuple(int(line, 16) for line in run.stdout.splitlines())


def _object_words() -> tuple[int, ...]:
    llvm_bin = os.environ.get("ATLAS_LLVM_BIN")
    if not llvm_bin:
        raise unittest.SkipTest("set ATLAS_LLVM_BIN for RISC-V object lowering")
    tools = pathlib.Path(llvm_bin)
    for name in ("mlir-translate", "llc", "llvm-objcopy"):
        if not (tools / name).is_file():
            raise AssertionError(f"selected LLVM installation lacks {name}")
    lowered = subprocess.run([str(ROOT / "build/bin/atlas-opt"), "--convert-atlas-to-llvm", str(SOURCE)],
                             text=True, capture_output=True, check=True)
    translated = subprocess.run([str(tools / "mlir-translate"), "--mlir-to-llvmir"],
                                input=lowered.stdout, text=True, capture_output=True, check=True)
    with tempfile.TemporaryDirectory() as temp:
        obj = pathlib.Path(temp) / "branch.o"
        section = pathlib.Path(temp) / "text.bin"
        subprocess.run([str(tools / "llc"), "-mtriple=riscv32-unknown-elf", "-filetype=obj",
                        "-o", str(obj)], input=translated.stdout.encode(), capture_output=True, check=True)
        subprocess.run([str(tools / "llvm-objcopy"), "--dump-section", f".text={section}", str(obj)],
                       capture_output=True, check=True)
        data = section.read_bytes()
    if len(data) < len(EXPECTED) * 4:
        raise AssertionError("LLVM object contains fewer words than the Atlas program")
    return tuple(int.from_bytes(data[index : index + 4], "little")
                 for index in range(0, len(EXPECTED) * 4, 4))


class BranchReferenceTest(unittest.TestCase):
    def test_typed_branch_matches_audited_words_and_lowers_to_llvm(self) -> None:
        self.assertEqual(_emitted(), EXPECTED)
        run = subprocess.run([str(ROOT / "build/bin/atlas-opt"), "--convert-atlas-to-llvm", str(SOURCE)],
                             text=True, capture_output=True, check=False)
        self.assertEqual(run.returncode, 0, run.stderr)
        self.assertEqual(run.stdout.count("llvm.inline_asm"), 1)
        for word in EXPECTED:
            self.assertIn(f".word 0x{word:08x}", run.stdout)

    def test_words_match_independent_selected_assembler(self) -> None:
        root = os.environ.get("ATLAS_ASSEMBLER_ROOT")
        if not root:
            self.skipTest("set ATLAS_ASSEMBLER_ROOT for the independent assembler")
        assembler = pathlib.Path(root) / "assembler.py"
        self.assertTrue(assembler.is_file())
        spec = importlib.util.spec_from_file_location("atlas_reference_assembler", assembler)
        self.assertIsNotNone(spec)
        module = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(module)
        self.assertEqual(tuple(module.assemble(ASSEMBLY.read_text())), _emitted())

    def test_riscv_object_program_words_match_independent_emitter(self) -> None:
        self.assertEqual(_object_words(), _emitted())

    def test_selected_core_executes_one_delay_slot(self) -> None:
        keys = ("ATLAS_ARC_MODEL", "ATLAS_ARC_STATE", "ATLAS_MODELIR_ROOT", "ATLAS_RTL_ROOT",
                "ATLAS_LLVM_BIN")
        if not all(os.environ.get(key) for key in keys):
            self.skipTest("set explicit ARC model, manifest, ModeLIR and RTL paths")
        model, state, modelir, rtl = (pathlib.Path(os.environ[key]).resolve(strict=True) for key in keys[:4])
        manifest = json.loads(state.read_text())[0]
        names = {entry["name"] for entry in manifest["states"]}
        for name in ("scalar/halt_now", "csrfile/reg_dbg0", "scalar/regfile/regs_10", "scalar/regfile/regs_1"):
            self.assertIn(name, names)
        scalar = (rtl / "src/main/scala/atlas/scalar/ScalarCore.scala").read_text()
        self.assertIn("io.halted    := halt_now", scalar)
        self.assertIn("val halt_now = halted || hostStop || illegal_detected || ecall_ebreak", scalar)

        # The selected optimized core omits io_halted. Map that proven wire
        # while using ModeLIR's existing TileLink/ARC driver unchanged.
        old_cwd = pathlib.Path.cwd()
        sys.path.insert(0, str(modelir))
        try:
            os.chdir(modelir)  # legacy ModeLIR bootstrap reads its discovered-interface cache here
            from mlc.backends import cosim_atlas

            original = cosim_atlas.CosimCore

            class SelectedCore(original):
                def peek(self, name: str) -> int:
                    return super().peek("scalar/halt_now" if name == "io_halted" else name)

            cosim_atlas.CosimCore = SelectedCore
            observed: dict[str, int] = {}

            def on_cycle(core: SelectedCore) -> None:
                for name in ("csrfile/reg_dbg0", "scalar/regfile/regs_10", "scalar/regfile/regs_1"):
                    observed[name] = core.peek(name)

            try:
                words = _object_words()
                result = cosim_atlas.run_program(model, state, words, max_cycles=100, on_cycle=on_cycle)
                correct = dict(observed)
                # A different in-block branch target executes the second
                # post-branch write, so the checked dbg0 value must change.
                changed = list(words)
                changed[1] = 0x00000263  # BEQ displacement 4 instead of 6.
                observed.clear()
                mutant = cosim_atlas.run_program(model, state, changed, max_cycles=100, on_cycle=on_cycle)
                wrong_target = dict(observed)
            finally:
                cosim_atlas.CosimCore = original
        finally:
            os.chdir(old_cwd)
            sys.path.remove(str(modelir))
        self.assertTrue(result.halted)
        self.assertLess(result.cycles, 100)
        self.assertEqual(correct["scalar/regfile/regs_10"], 0xAA)
        self.assertEqual(correct["scalar/regfile/regs_1"], 1)
        self.assertEqual(correct["csrfile/reg_dbg0"], 1)
        self.assertTrue(mutant.halted)
        self.assertEqual(wrong_target["scalar/regfile/regs_10"], 0xBB)
        self.assertEqual(wrong_target["csrfile/reg_dbg0"], 18)
        if receipt_name := os.environ.get("ATLAS_ARC_RECEIPT"):
            receipt = pathlib.Path(receipt_name)
            if receipt.exists():
                raise AssertionError("ARC diagnostic receipt destination must be fresh")
            receipt.parent.mkdir(parents=True, exist_ok=True)
            identity = {
                "model_sha256": hashlib.sha256(model.read_bytes()).hexdigest(),
                "state_sha256": hashlib.sha256(state.read_bytes()).hexdigest(),
                "mlir_sha256": hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
                "assembler_source_sha256": hashlib.sha256(ASSEMBLY.read_bytes()).hexdigest(),
                "mode_lir_driver_sha256": hashlib.sha256((modelir / "mlc/backends/cosim_atlas.py").read_bytes()).hexdigest(),
            }
            receipt.write_text(json.dumps({
                "schema": "atlas.hand_reference_branch_diagnostic.v1",
                "scope": "selected AtlasCore ARC model supplied externally; not a frozen RTL certificate",
                "identity": identity,
                "object_words": [f"{word:08x}" for word in words],
                "selected": {"halted": result.halted, "cycles": result.cycles,
                             "x10": correct["scalar/regfile/regs_10"], "dbg0": correct["csrfile/reg_dbg0"]},
                "mutant": {"halted": mutant.halted, "cycles": mutant.cycles,
                           "x10": wrong_target["scalar/regfile/regs_10"],
                           "dbg0": wrong_target["csrfile/reg_dbg0"]},
            }, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    unittest.main()
