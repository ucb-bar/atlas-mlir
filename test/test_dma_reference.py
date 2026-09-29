"""Hand-authored DMA stream, independent word assembly, and optional ARC execution."""

from __future__ import annotations

import importlib.util
import json
import os
import pathlib
import subprocess
import sys
import tempfile
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]
BIN = pathlib.Path(os.environ.get("ATLAS_OOT_BIN_DIR", ROOT / "build/bin"))
SOURCE = ROOT / "test/examples/dma_loopback.mlir"
EXPECTED = (
    0x00100E13, 0x00000293, 0x0002807F, 0x00000313, 0x900000B7,
    0x08000113, 0x0020837B, 0x0200007F, 0x900001B7, 0x40018193,
    0x022311FB, 0x0200107F, 0x00100093, 0xC1009073, 0x00000073,
)


def _emitted() -> tuple[int, ...]:
    run = subprocess.run([str(BIN / "atlas-emit"), str(SOURCE)],
                         text=True, capture_output=True, check=True)
    return tuple(int(line, 16) for line in run.stdout.splitlines())


def _assembler():
    root = os.environ.get("ATLAS_ASSEMBLER_ROOT")
    if not root:
        raise unittest.SkipTest("set ATLAS_ASSEMBLER_ROOT for selected public assembler")
    path = pathlib.Path(root) / "assembler.py"
    if not path.is_file():
        raise AssertionError(f"missing selected assembler: {path}")
    spec = importlib.util.spec_from_file_location("atlas_dma_reference_assembler", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assembly = (pathlib.Path(root) / "assembly/dma_loopback.S").read_text()
    return module, assembly


def _object_words() -> tuple[int, ...]:
    root = os.environ.get("ATLAS_LLVM_BIN")
    if not root:
        raise unittest.SkipTest("set ATLAS_LLVM_BIN for RISC-V object lowering")
    bin_root = pathlib.Path(root)
    lowered = subprocess.run([str(BIN / "atlas-opt"),
                             "--convert-atlas-to-llvm", str(SOURCE)],
                             text=True, capture_output=True, check=True)
    translated = subprocess.run([str(bin_root / "mlir-translate"), "--mlir-to-llvmir"],
                                input=lowered.stdout, text=True, capture_output=True, check=True)
    with tempfile.TemporaryDirectory() as temp:
        obj = pathlib.Path(temp) / "dma.o"
        section = pathlib.Path(temp) / "text.bin"
        subprocess.run([str(bin_root / "llc"), "-mtriple=riscv32-unknown-elf", "-filetype=obj",
                        "-o", str(obj)], input=translated.stdout.encode(), capture_output=True, check=True)
        subprocess.run([str(bin_root / "llvm-objcopy"), "--dump-section", f".text={section}", str(obj)],
                       capture_output=True, check=True)
        data = section.read_bytes()
    if len(data) < len(EXPECTED) * 4:
        raise AssertionError("LLVM object has fewer words than typed DMA program")
    return tuple(int.from_bytes(data[i : i + 4], "little")
                 for i in range(0, len(EXPECTED) * 4, 4))


class DMAReferenceTest(unittest.TestCase):
    def test_typed_dma_matches_audited_words_and_llvm_lowering(self) -> None:
        self.assertEqual(_emitted(), EXPECTED)
        run = subprocess.run([str(BIN / "atlas-opt"), "--convert-atlas-to-llvm",
                              str(SOURCE)], text=True, capture_output=True, check=False)
        self.assertEqual(run.returncode, 0, run.stderr)
        self.assertEqual(run.stdout.count("llvm.inline_asm"), 1)
        self.assertIn("~{memory}", run.stdout)

    def test_words_match_independent_selected_assembler(self) -> None:
        assembler, assembly = _assembler()
        self.assertEqual(tuple(assembler.assemble(assembly)[:len(EXPECTED)]), _emitted())

    def test_riscv_object_program_words_match_independent_emitter(self) -> None:
        self.assertEqual(_object_words(), _emitted())

    def test_selected_core_executes_dma_and_preserves_guards(self) -> None:
        keys = ("ATLAS_ARC_MODEL", "ATLAS_ARC_STATE", "ATLAS_MODELIR_ROOT", "ATLAS_RTL_ROOT",
                "ATLAS_ASSEMBLER_ROOT", "ATLAS_LLVM_BIN")
        if not all(os.environ.get(key) for key in keys):
            self.skipTest("set explicit ARC, ModeLIR, RTL, assembler and LLVM paths")
        model, state, modelir, rtl = (pathlib.Path(os.environ[key]).resolve(strict=True)
                                       for key in keys[:4])
        names = {entry["name"] for entry in json.loads(state.read_text())[0]["states"]}
        self.assertIn("scalar/halt_now", names)
        self.assertIn("csrfile/reg_dbg0", names)
        self.assertNotIn("io_halted", names)
        scalar = (rtl / "src/main/scala/atlas/scalar/ScalarCore.scala").read_text()
        self.assertIn("io.halted    := halt_now", scalar)
        dma = (rtl / "src/main/scala/diplomatic/memory/DMA.scala").read_text()
        self.assertNotIn("io.dmaTL.d.bits.opcode", dma)
        self.assertNotIn("io.dmaTL.d.bits.size", dma)

        assembler, assembly = _assembler()
        parsed = assembler._parse_inline_golden(assembly, 32)
        assert parsed is not None
        base, preload_addrs, preload_words, check_addrs, check_words = parsed
        self.assertEqual(base, 0x90000000)
        self.assertEqual(len(preload_addrs), 32)
        self.assertEqual(len(check_addrs), 32)
        preload = [(address, value.to_bytes(4, "little"))
                   for address, value in zip(preload_addrs, preload_words, strict=True)]
        guards = {base + offset: bytes([value] * 4)
                  for offset, value in ((128, 0xA5), (1020, 0x5A), (1152, 0xC3))}
        preload.extend(guards.items())

        old_cwd = pathlib.Path.cwd()
        sys.path.insert(0, str(modelir))
        try:
            os.chdir(modelir)  # legacy ModeLIR bootstrap uses a local interface cache
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
            observed: dict[str, int] = {}

            def on_cycle(core: SelectedCore) -> None:
                observed["dbg0"] = core.peek("csrfile/reg_dbg0")

            try:
                result = cosim_atlas.run_program(model, state, _object_words(),
                                                 preload=preload, max_cycles=5000,
                                                 on_cycle=on_cycle)
            finally:
                cosim_atlas.CosimCore = original
        finally:
            os.chdir(old_cwd)
            sys.path.remove(str(modelir))
        self.assertTrue(result.halted)
        self.assertEqual(result.reads, 4)
        self.assertEqual(result.writes, 4)
        self.assertEqual(observed["dbg0"], 1)
        for address, value in zip(check_addrs, check_words, strict=True):
            self.assertEqual(result.slave.captured(address, 4), value.to_bytes(4, "little"),
                             f"output at {address:#x}")
        for address, value in zip(preload_addrs, preload_words, strict=True):
            self.assertEqual(result.slave.captured(address, 4), value.to_bytes(4, "little"),
                             f"input at {address:#x}")
        for address, value in guards.items():
            self.assertEqual(result.slave.captured(address, 4), value,
                             f"guard at {address:#x}")


if __name__ == "__main__":
    unittest.main()
