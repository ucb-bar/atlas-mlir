"""Bounded selected-core evidence for SELI, SELD, ECALL, and EBREAK."""

from __future__ import annotations

import importlib.util
import os
import pathlib
import subprocess
import sys
import tempfile
import unittest

from test_scalar_alu_reference import _emitted, _object_words


ROOT = pathlib.Path(__file__).resolve().parents[1]
PANELS = ((0x7B, 0x86, 0), (0xE1, 0x24, 4))
TRAPS = (("ecall", 2), ("ebreak", 3))


def _program(scale_immediate: int, scale_memory: int, base_address: int, trap: str) -> tuple[str, str]:
    """Write a nonzero word to VMEM, then read only its low byte into e4."""
    operations = [
        ("alu_imm", {"kind": "addi", "dst": 9, "src": 0, "immediate": 11}, "ADDI x9, x0, 11"),
        ("alu_imm", {"kind": "addi", "dst": 6, "src": 0, "immediate": base_address},
         f"ADDI x6, x0, {base_address}"),
        ("upper", {"kind": "lui", "dst": 2, "immediate": 0x12345}, "LUI x2, 74565"),
        ("alu_imm", {"kind": "addi", "dst": 2, "src": 2, "immediate": scale_memory},
         f"ADDI x2, x2, {scale_memory}"),
        ("scalar_store", {"kind": "sw", "src": 2, "base": 6, "offset": 8}, "SW x2, x6, 8"),
        ("delay", {"cycles": 4}, "DELAY 4"),
        ("scalar_load", {"kind": "seli", "dst": 3, "base": 0, "offset": scale_immediate},
         f"SELI 3, {scale_immediate}"),
        ("scalar_load", {"kind": "seli", "dst": 5, "base": 0, "offset": 0x55}, "SELI 5, 85"),
        ("scalar_load", {"kind": "seld", "dst": 4, "base": 6, "offset": 8}, "SELD 4, x6, 8"),
        ("delay", {"cycles": 8}, "DELAY 8"),
        ("alu_imm", {"kind": "addi", "dst": 0, "src": 0, "immediate": 0}, "ADDI x0, x0, 0"),
        ("trap", {"kind": trap}, trap.upper()),
        ("alu_imm", {"kind": "addi", "dst": 9, "src": 0, "immediate": 99}, "ADDI x9, x0, 99"),
    ]
    lines = ["module {", '  %s0 = "atlas.start"() : () -> !atlas.state']
    assembly = []
    for index, (name, attrs, asm) in enumerate(operations, start=1):
        fields = ", ".join(
            f'{key} = "{value}"' if isinstance(value, str) else f"{key} = {value} : i32"
            for key, value in attrs.items()
        )
        lines.append(
            f'  %s{index} = "atlas.{name}"(%s{index - 1}) '
            f'{{{fields}}} : (!atlas.state) -> !atlas.state'
        )
        assembly.append(asm)
    lines.extend(("}", ""))
    return "\n".join(lines), "\n".join(assembly) + "\n"


class ScaleTrapReferenceTest(unittest.TestCase):
    def test_typed_words_match_selected_assembler_and_llvm_object(self) -> None:
        assembler_root = os.environ.get("ATLAS_ASSEMBLER_ROOT")
        if not assembler_root:
            self.skipTest("set ATLAS_ASSEMBLER_ROOT for the selected assembler")
        spec = importlib.util.spec_from_file_location(
            "atlas_selected_scale_trap_assembler", pathlib.Path(assembler_root) / "assembler.py"
        )
        assert spec is not None and spec.loader is not None
        assembler = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(assembler)
        with tempfile.TemporaryDirectory() as temporary:
            directory = pathlib.Path(temporary)
            source_path = directory / "scale_trap.mlir"
            for scale_immediate, scale_memory, base_address in PANELS:
                for trap, _ in TRAPS:
                    with self.subTest(scale=scale_immediate, memory=scale_memory, base=base_address, trap=trap):
                        source, assembly = _program(scale_immediate, scale_memory, base_address, trap)
                        source_path.write_text(source)
                        words = _emitted(source_path)
                        self.assertEqual(words, tuple(assembler.assemble(assembly)))
                        self.assertEqual(words, _object_words(source_path, len(words), directory))

    def test_selected_core_scale_registers_and_trap_reasons(self) -> None:
        keys = ("ATLAS_ARC_MODEL", "ATLAS_ARC_STATE", "ATLAS_MODELIR_ROOT", "ATLAS_RTL_ROOT")
        if not all(os.environ.get(key) for key in keys):
            self.skipTest("set selected-source-linked ARC model, state, ModeLIR and RTL paths")
        model, state, modelir, rtl = (
            pathlib.Path(os.environ[key]).resolve(strict=True) for key in keys
        )
        self.assertEqual(
            subprocess.check_output(["git", "-C", str(rtl), "rev-parse", "HEAD"], text=True).strip(),
            "0079c0541111197741a231c002e3843fa6f545b2",
        )
        for register in (3, 4, 5):
            self.assertIn(f"scalar/scaleRF/reg_{register}", state.read_text())
        old_cwd = pathlib.Path.cwd()
        sys.path.insert(0, str(modelir))
        try:
            os.chdir(modelir)
            from mlc.backends import cosim_atlas

            original = cosim_atlas.CosimCore

            class SelectedCore(original):
                def peek(self, name: str) -> int:
                    return super().peek("scalar/halt_now" if name == "io_halted" else name)

            cosim_atlas.CosimCore = SelectedCore
            try:
                with tempfile.TemporaryDirectory() as temporary:
                    source_path = pathlib.Path(temporary) / "scale_trap.mlir"
                    for scale_immediate, scale_memory, base_address in PANELS:
                        for trap, reason in TRAPS:
                            with self.subTest(scale=scale_immediate, memory=scale_memory, base=base_address,
                                              trap=trap):
                                source, _ = _program(scale_immediate, scale_memory, base_address, trap)
                                source_path.write_text(source)
                                observed: dict[str, int] = {}

                                def on_cycle(core: SelectedCore) -> None:
                                    for register in (2, 3, 4, 5):
                                        observed[f"e{register}"] = core.peek(f"scalar/scaleRF/reg_{register}")
                                    observed["x9"] = core.peek("scalar/regfile/regs_9")

                                result = cosim_atlas.run_program(
                                    model, state, _emitted(source_path), max_cycles=250, on_cycle=on_cycle,
                                )
                                self.assertTrue(result.halted)
                                self.assertEqual(result.halt_reason, reason)
                                self.assertEqual(observed["e3"], scale_immediate)
                                self.assertEqual(observed["e4"], scale_memory)
                                self.assertEqual(observed["e5"], 0x55)
                                self.assertEqual(observed["e2"], 0)
                                self.assertEqual(observed["x9"], 11)
            finally:
                cosim_atlas.CosimCore = original
        finally:
            os.chdir(old_cwd)
            sys.path.remove(str(modelir))


if __name__ == "__main__":
    unittest.main()
