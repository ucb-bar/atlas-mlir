"""Bounded selected-core evidence for the direct scalar ALU ISA modes."""

from __future__ import annotations

import importlib.util
import os
import pathlib
import subprocess
import sys
import tempfile
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]
BIN = pathlib.Path(os.environ.get("ATLAS_OOT_BIN_DIR", ROOT / "build/bin"))
REGISTER_MODES = ("add", "sub", "sll", "slt", "sltu", "xor", "srl", "sra", "or", "and")
IMMEDIATE_MODES = ("addi", "slti", "sltiu", "xori", "ori", "andi", "slli", "srli", "srai")
PANELS = ((-17, 3), (17, 31))
MASK = (1 << 32) - 1


def _signed(value: int) -> int:
    bits = value & MASK
    return bits - (1 << 32) if bits & (1 << 31) else bits


def _expected(kind: str, lhs: int, rhs: int) -> int:
    left, right = lhs & MASK, rhs & MASK
    shift = right & 31
    if kind in {"add", "addi"}:
        return (left + right) & MASK
    if kind == "sub":
        return (left - right) & MASK
    if kind in {"sll", "slli"}:
        return (left << shift) & MASK
    if kind in {"slt", "slti"}:
        return int(_signed(left) < _signed(right))
    if kind in {"sltu", "sltiu"}:
        return int(left < right)
    if kind in {"xor", "xori"}:
        return left ^ right
    if kind in {"srl", "srli"}:
        return left >> shift
    if kind in {"sra", "srai"}:
        return (_signed(left) >> shift) & MASK
    if kind in {"or", "ori"}:
        return left | right
    if kind in {"and", "andi"}:
        return left & right
    raise AssertionError(f"unhandled scalar ALU mode {kind}")


def _program(kind: str, lhs: int, rhs: int) -> tuple[str, str]:
    if kind in REGISTER_MODES:
        selected = (
            f'%s3 = "atlas.alu_reg"(%s2) {{kind = "{kind}", dst = 12 : i32, '
            'lhs = 10 : i32, rhs = 11 : i32} : (!atlas.state) -> !atlas.state'
        )
        assembly = f"{kind.upper()} x12, x10, x11"
    else:
        selected = (
            f'%s3 = "atlas.alu_imm"(%s2) {{kind = "{kind}", dst = 12 : i32, '
            f'src = 10 : i32, immediate = {rhs} : i32}} : (!atlas.state) -> !atlas.state'
        )
        assembly = f"{kind.upper()} x12, x10, {rhs}"
    source = "\n".join((
        "module {",
        '  %s0 = "atlas.start"() : () -> !atlas.state',
        f'  %s1 = "atlas.alu_imm"(%s0) {{kind = "addi", dst = 10 : i32, src = 0 : i32, '
        f'immediate = {lhs} : i32}} : (!atlas.state) -> !atlas.state',
        f'  %s2 = "atlas.alu_imm"(%s1) {{kind = "addi", dst = 11 : i32, src = 0 : i32, '
        f'immediate = {rhs} : i32}} : (!atlas.state) -> !atlas.state',
        f"  {selected}",
        '  %s4 = "atlas.trap"(%s3) {kind = "ecall"} : (!atlas.state) -> !atlas.state',
        "}",
        "",
    ))
    transcription = f"ADDI x10, x0, {lhs}\nADDI x11, x0, {rhs}\n{assembly}\nECALL\n"
    return source, transcription


def _emitted(path: pathlib.Path) -> tuple[int, ...]:
    run = subprocess.run([str(BIN / "atlas-emit"), str(path)], text=True, capture_output=True, check=True)
    return tuple(int(line, 16) for line in run.stdout.splitlines())


def _object_words(path: pathlib.Path, count: int, directory: pathlib.Path) -> tuple[int, ...]:
    llvm_bin = os.environ.get("ATLAS_LLVM_BIN")
    if not llvm_bin:
        raise unittest.SkipTest("set ATLAS_LLVM_BIN for RISC-V object lowering")
    tools = pathlib.Path(llvm_bin)
    lowered = subprocess.run(
        [str(BIN / "atlas-opt"), "--convert-atlas-to-llvm", str(path)],
        text=True, capture_output=True, check=True,
    )
    assert lowered.stdout.count("llvm.inline_asm") == 1
    translated = subprocess.run(
        [str(tools / "mlir-translate"), "--mlir-to-llvmir"],
        input=lowered.stdout, text=True, capture_output=True, check=True,
    )
    object_path, section = directory / "program.o", directory / "text.bin"
    subprocess.run(
        [str(tools / "llc"), "-mtriple=riscv32-unknown-elf", "-filetype=obj", "-o", str(object_path)],
        input=translated.stdout.encode(), capture_output=True, check=True,
    )
    subprocess.run(
        [str(tools / "llvm-objcopy"), "--dump-section", f".text={section}", str(object_path)],
        capture_output=True, check=True,
    )
    data = section.read_bytes()
    assert len(data) >= count * 4
    return tuple(int.from_bytes(data[index * 4 : (index + 1) * 4], "little") for index in range(count))


class ScalarALUReferenceTest(unittest.TestCase):
    def test_typed_words_match_selected_assembler_and_llvm_object(self) -> None:
        assembler_root = os.environ.get("ATLAS_ASSEMBLER_ROOT")
        if not assembler_root:
            self.skipTest("set ATLAS_ASSEMBLER_ROOT for the selected assembler")
        assembler_file = pathlib.Path(assembler_root) / "assembler.py"
        spec = importlib.util.spec_from_file_location("atlas_selected_assembler", assembler_file)
        assert spec is not None and spec.loader is not None
        assembler = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(assembler)
        with tempfile.TemporaryDirectory() as temporary:
            directory = pathlib.Path(temporary)
            source_path = directory / "program.mlir"
            for kind in (*REGISTER_MODES, *IMMEDIATE_MODES):
                for lhs, rhs in PANELS:
                    with self.subTest(kind=kind, lhs=lhs, rhs=rhs):
                        source, transcription = _program(kind, lhs, rhs)
                        source_path.write_text(source)
                        words = _emitted(source_path)
                        self.assertEqual(words, tuple(assembler.assemble(transcription)))
                        self.assertEqual(words, _object_words(source_path, len(words), directory))

    def test_selected_core_matches_independent_32_bit_semantics(self) -> None:
        keys = ("ATLAS_ARC_MODEL", "ATLAS_ARC_STATE", "ATLAS_MODELIR_ROOT", "ATLAS_RTL_ROOT")
        if not all(os.environ.get(key) for key in keys):
            self.skipTest("set selected-source-linked ARC model, state, ModeLIR and RTL paths")
        model, state, modelir, rtl = (pathlib.Path(os.environ[key]).resolve(strict=True) for key in keys)
        self.assertEqual(subprocess.check_output(["git", "-C", str(rtl), "rev-parse", "HEAD"], text=True).strip(),
                         "0079c0541111197741a231c002e3843fa6f545b2")
        self.assertIn("scalar/regfile/regs_12", state.read_text())
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
                    path = pathlib.Path(temporary) / "scalar_alu.mlir"
                    for kind in (*REGISTER_MODES, *IMMEDIATE_MODES):
                        for lhs, rhs in PANELS:
                            with self.subTest(kind=kind, lhs=lhs, rhs=rhs):
                                path.write_text(_program(kind, lhs, rhs)[0])
                                observed: dict[str, int] = {}

                                def on_cycle(core: SelectedCore) -> None:
                                    observed["x12"] = core.peek("scalar/regfile/regs_12")

                                result = cosim_atlas.run_program(
                                    model, state, _emitted(path), max_cycles=80, on_cycle=on_cycle,
                                )
                                self.assertTrue(result.halted)
                                self.assertEqual(observed["x12"] & MASK, _expected(kind, lhs, rhs))
            finally:
                cosim_atlas.CosimCore = original
        finally:
            os.chdir(old_cwd)
            sys.path.remove(str(modelir))


if __name__ == "__main__":
    unittest.main()
