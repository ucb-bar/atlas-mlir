"""Bounded selected-core semantics for RV32 scalar VMEM loads and stores."""

from __future__ import annotations

import importlib.util
import os
import pathlib
import subprocess
import sys
import tempfile
import unittest

from test_scalar_alu_reference import MASK, _emitted, _object_words

from selected_core_runtime import run_selected_program


ROOT = pathlib.Path(__file__).resolve().parents[1]
BIN = pathlib.Path(os.environ.get("ATLAS_OOT_BIN_DIR", ROOT / "build/bin"))
PANELS = ((0x80FF0, 0x17F), (0x7F008, 0x1FE))


def _signed(value: int, bits: int) -> int:
    value &= (1 << bits) - 1
    return (value - (1 << bits) if value & (1 << (bits - 1)) else value) & MASK


def _program(
    upper: int, low: int, *, drain_before_halt: bool = True
) -> tuple[str, str, dict[int, int]]:
    # The selected scalar LSU addresses VMEM in bytes. Initialize two words,
    # overwrite only byte 1 and halfword 1 of the first, and observe both.
    operations = [
        ("upper", {"kind": "lui", "dst": 2, "immediate": upper}, f"LUI x2, {upper}"),
        ("alu_imm", {"kind": "addi", "dst": 2, "src": 2, "immediate": low},
         f"ADDI x2, x2, {low}"),
        ("scalar_store", {"kind": "sw", "src": 2, "base": 0, "offset": 0},
         "SW x2, x0, 0"),
        ("upper", {"kind": "lui", "dst": 5, "immediate": 0x5A5A6},
         "LUI x5, 370086"),
        ("alu_imm", {"kind": "addi", "dst": 5, "src": 5, "immediate": -1446},
         "ADDI x5, x5, -1446"),
        ("scalar_store", {"kind": "sw", "src": 5, "base": 0, "offset": 4},
         "SW x5, x0, 4"),
        ("delay", {"cycles": 4}, "DELAY 4"),
        ("scalar_load", {"kind": "lw", "dst": 11, "base": 0, "offset": 0},
         "LW x11, x0, 0"),
        ("delay", {"cycles": 8}, "DELAY 8"),
        ("alu_imm", {"kind": "addi", "dst": 3, "src": 0, "immediate": 0xA5},
         "ADDI x3, x0, 165"),
        ("scalar_store", {"kind": "sb", "src": 3, "base": 0, "offset": 1},
         "SB x3, x0, 1"),
        ("upper", {"kind": "lui", "dst": 4, "immediate": 9}, "LUI x4, 9"),
        ("alu_imm", {"kind": "addi", "dst": 4, "src": 4, "immediate": -1075},
         "ADDI x4, x4, -1075"),
        ("scalar_store", {"kind": "sh", "src": 4, "base": 0, "offset": 2},
         "SH x4, x0, 2"),
        ("delay", {"cycles": 4}, "DELAY 4"),
    ]
    for kind, register, offset in (
        ("lb", 12, 3), ("lbu", 13, 1), ("lh", 14, 2),
        ("lhu", 15, 2), ("lw", 16, 0), ("lw", 17, 4),
    ):
        operations.append((
            "scalar_load", {"kind": kind, "dst": register, "base": 0, "offset": offset},
            f"{kind.upper()} x{register}, x0, {offset}",
        ))
        operations.append(("delay", {"cycles": 8}, "DELAY 8"))
    # On the selected core, ECALL asserts halt_now as soon as it is decoded,
    # even while the preceding DELAY is stalling. Keep a harmless instruction
    # between that delay and ECALL so the final load can retire.
    if drain_before_halt:
        operations.append((
            "alu_imm", {"kind": "addi", "dst": 0, "src": 0, "immediate": 0},
            "ADDI x0, x0, 0",
        ))
    operations.append(("trap", {"kind": "ecall"}, "ECALL"))

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
    initial = ((upper << 12) + low) & MASK
    final = (initial & 0xFF) | (0xA5 << 8) | (0x8BCD << 16)
    expected = {
        11: initial,
        12: _signed(final >> 24, 8),
        13: 0xA5,
        14: _signed(final >> 16, 16),
        15: 0x8BCD,
        16: final,
        17: 0x5A5A5A5A,
    }
    return "\n".join(lines), "\n".join(assembly) + "\n", expected


class ScalarMemoryReferenceTest(unittest.TestCase):
    def test_typed_words_match_selected_assembler_and_llvm_object(self) -> None:
        root = os.environ.get("ATLAS_ASSEMBLER_ROOT")
        if not root:
            self.skipTest("set ATLAS_ASSEMBLER_ROOT for selected assembler")
        spec = importlib.util.spec_from_file_location(
            "atlas_selected_scalar_memory_assembler", pathlib.Path(root) / "assembler.py"
        )
        assert spec is not None and spec.loader is not None
        assembler = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(assembler)
        with tempfile.TemporaryDirectory() as temporary:
            directory = pathlib.Path(temporary)
            source_path = directory / "scalar_memory.mlir"
            for upper, low in PANELS:
                with self.subTest(upper=upper, low=low):
                    source, assembly, _ = _program(upper, low)
                    source_path.write_text(source)
                    words = _emitted(source_path)
                    self.assertEqual(words, tuple(assembler.assemble(assembly)))
                    self.assertEqual(words, _object_words(source_path, len(words), directory))

    def test_selected_core_matches_byte_halfword_and_word_semantics(self) -> None:
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
                    path = pathlib.Path(temporary) / "scalar_memory.mlir"
                    for upper, low in PANELS:
                        with self.subTest(upper=upper, low=low):
                            source, _, expected = _program(upper, low)
                            path.write_text(source)
                            observed: dict[int, int] = {}

                            def on_cycle(core: SelectedCore) -> None:
                                observed.update({
                                    register: core.peek(f"scalar/regfile/regs_{register}")
                                    for register in expected
                                })

                            result = run_selected_program(cosim_atlas,
                                model, state, _emitted(path), max_cycles=250, on_cycle=on_cycle,
                            )
                            self.assertTrue(result.halted)
                            self.assertEqual({key: value & MASK for key, value in observed.items()}, expected)
            finally:
                cosim_atlas.CosimCore = original
        finally:
            os.chdir(old_cwd)
            sys.path.remove(str(modelir))

    def test_ecall_immediately_after_delay_discards_pending_last_load(self) -> None:
        keys = ("ATLAS_ARC_MODEL", "ATLAS_ARC_STATE", "ATLAS_MODELIR_ROOT")
        if not all(os.environ.get(key) for key in keys):
            self.skipTest("set selected-source-linked ARC model, state and ModeLIR paths")
        model, state, modelir = (
            pathlib.Path(os.environ[key]).resolve(strict=True) for key in keys
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

            cosim_atlas.CosimCore = SelectedCore
            try:
                with tempfile.TemporaryDirectory() as temporary:
                    path = pathlib.Path(temporary) / "early_halt.mlir"
                    source, _, expected = _program(
                        *PANELS[0], drain_before_halt=False
                    )
                    path.write_text(source)
                    final: dict[str, int] = {}

                    def on_cycle(core: SelectedCore) -> None:
                        final["pending"] = core.peek("scalar/memLoadPending")
                        final["x17"] = core.peek("scalar/regfile/regs_17")

                    result = run_selected_program(cosim_atlas,
                        model, state, _emitted(path), max_cycles=250, on_cycle=on_cycle,
                    )
                    self.assertTrue(result.halted)
                    self.assertEqual(result.halt_reason, 2)
                    self.assertEqual(final, {"pending": 1, "x17": 0})
                    self.assertNotEqual(final["x17"], expected[17])
            finally:
                cosim_atlas.CosimCore = original
        finally:
            os.chdir(old_cwd)
            sys.path.remove(str(modelir))


if __name__ == "__main__":
    unittest.main()
