"""Bounded selected-core check of DMA.CONFIG's global upper-address register."""

from __future__ import annotations

import importlib.util
import os
import pathlib
import subprocess
import sys
import tempfile
import unittest

from test_vpu_relu_reference import _emitted, _object_words


ROOT = pathlib.Path(__file__).resolve().parents[1]
BIN = pathlib.Path(os.environ.get("ATLAS_OOT_BIN_DIR", ROOT / "build/bin"))
RTL_REVISION = "0079c0541111197741a231c002e3843fa6f545b2"
PANELS = ((5, 0, 0x12345, 0x345), (5, 7, 0x23456, 0x123),
          (6, 0, 0x34567, 0x234), (6, 7, 0x45678, 0x456))


def _program(source_reg: int, channel: int, upper: int, lower: int) -> tuple[str, str, int]:
    value = (upper << 12) + lower
    operations = [
        ("upper", {"kind": "lui", "dst": source_reg, "immediate": upper},
         f"LUI x{source_reg}, {upper}"),
        ("alu_imm", {"kind": "addi", "dst": source_reg, "src": source_reg,
                     "immediate": lower}, f"ADDI x{source_reg}, x{source_reg}, {lower}"),
        ("dma_config", {"channel": channel, "base_reg": source_reg},
         f"DMA.CONFIG x{source_reg}, {channel}"),
        ("alu_imm", {"kind": "addi", "dst": source_reg, "src": 0, "immediate": 0},
         f"ADDI x{source_reg}, x0, 0"),
        ("delay", {"cycles": 4}, "DELAY 4"),
        ("alu_imm", {"kind": "addi", "dst": 9, "src": 0, "immediate": 11},
         "ADDI x9, x0, 11"),
        ("trap", {"kind": "ecall"}, "ECALL"),
        ("alu_imm", {"kind": "addi", "dst": 9, "src": 0, "immediate": 99},
         "ADDI x9, x0, 99"),
    ]
    lines = ["module {", '  %s0 = "atlas.start"() : () -> !atlas.state']
    assembly = []
    for index, (op, attrs, asm) in enumerate(operations, start=1):
        fields = ", ".join(
            f'{key} = "{item}"' if isinstance(item, str) else f"{key} = {item} : i32"
            for key, item in attrs.items()
        )
        lines.append(f'  %s{index} = "atlas.{op}"(%s{index - 1}) '
                     f'{{{fields}}} : (!atlas.state) -> !atlas.state')
        assembly.append(asm)
    lines.extend(("}", ""))
    return "\n".join(lines), "\n".join(assembly) + "\n", value


class DmaConfigReferenceTest(unittest.TestCase):
    def test_typed_words_match_assembler_and_llvm_object(self) -> None:
        assembler_root = os.environ.get("ATLAS_ASSEMBLER_ROOT")
        if not assembler_root:
            self.skipTest("set ATLAS_ASSEMBLER_ROOT for selected assembler")
        spec = importlib.util.spec_from_file_location(
            "selected_dma_config_assembler", pathlib.Path(assembler_root) / "assembler.py"
        )
        assert spec is not None and spec.loader is not None
        assembler = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(assembler)
        with tempfile.TemporaryDirectory() as temporary:
            path = pathlib.Path(temporary) / "dma_config.mlir"
            for source_reg, channel, upper, lower in PANELS:
                with self.subTest(reg=source_reg, channel=channel):
                    source, assembly, _ = _program(source_reg, channel, upper, lower)
                    path.write_text(source)
                    words = _emitted(path)
                    self.assertEqual(words[2], assembler.DMA_CONFIG(source_reg, channel))
                    self.assertEqual(words, tuple(assembler.assemble(assembly)))
                    self.assertEqual(words, _object_words(path))
                    parsed = subprocess.run([str(BIN / "atlas-opt"), str(path)], text=True,
                                            capture_output=True, check=True)
                    reparsed = subprocess.run([str(BIN / "atlas-opt"), "-"], input=parsed.stdout,
                                              text=True, capture_output=True, check=True)
                    self.assertEqual(parsed.stdout, reparsed.stdout)

    def test_invalid_channel_and_source_register_are_rejected(self) -> None:
        source, _, _ = _program(*PANELS[0])
        for bad in ('channel = 8 : i32, base_reg = 5 : i32',
                    'channel = 0 : i32, base_reg = 32 : i32'):
            with self.subTest(bad=bad):
                changed = source.replace('channel = 0 : i32, base_reg = 5 : i32', bad)
                self.assertNotEqual(changed, source)
                run = subprocess.run([str(BIN / "atlas-opt"), "-"], input=changed, text=True,
                                     capture_output=True, check=False)
                self.assertNotEqual(run.returncode, 0)
                self.assertTrue(run.stderr)

    def test_selected_core_captures_rs1_value_not_later_register_contents(self) -> None:
        keys = ("ATLAS_ARC_MODEL", "ATLAS_ARC_STATE", "ATLAS_MODELIR_ROOT", "ATLAS_RTL_ROOT")
        if not all(os.environ.get(key) for key in keys):
            self.skipTest("set selected ARC model, state, ModeLIR, and RTL paths")
        model, state, modelir, rtl = (pathlib.Path(os.environ[key]).resolve(strict=True)
                                      for key in keys)
        self.assertEqual(
            subprocess.check_output(["git", "-C", str(rtl), "rev-parse", "HEAD"], text=True).strip(),
            RTL_REVISION,
        )
        self.assertIn('"name": "scalar/dmaBaseReg"', state.read_text())
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
                    path = pathlib.Path(temporary) / "dma_config.mlir"
                    for source_reg, channel, upper, lower in PANELS:
                        with self.subTest(reg=source_reg, channel=channel):
                            source, _, expected = _program(source_reg, channel, upper, lower)
                            path.write_text(source)
                            observed: dict[str, int] = {}

                            def on_cycle(core: SelectedCore) -> None:
                                observed["base"] = core.peek("scalar/dmaBaseReg")
                                observed["source"] = core.peek(f"scalar/regfile/regs_{source_reg}")
                                observed["x9"] = core.peek("scalar/regfile/regs_9")

                            result = cosim_atlas.run_program(
                                model, state, _emitted(path), max_cycles=250, on_cycle=on_cycle,
                            )
                            self.assertTrue(result.halted)
                            self.assertEqual(result.halt_reason, 2)
                            self.assertEqual((result.reads, result.writes), (0, 0))
                            self.assertEqual(observed, {"base": expected, "source": 0, "x9": 11})
            finally:
                cosim_atlas.CosimCore = original
        finally:
            os.chdir(old_cwd)
            sys.path.remove(str(modelir))


if __name__ == "__main__":
    unittest.main()
