"""Selected-source-linked BF16 VRELU pair check for the hand OOT dialect."""

from __future__ import annotations

import importlib.util
import os
import pathlib
import struct
import subprocess
import sys
import tempfile
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]
BIN = pathlib.Path(os.environ.get("ATLAS_OOT_BIN_DIR", ROOT / "build/bin"))
SOURCE = ROOT / "test/examples/vpu_relu_pair.mlir"
ASSEMBLY = ROOT / "test/examples/vpu_relu_pair.S"


def _emitted(source: pathlib.Path = SOURCE) -> tuple[int, ...]:
    result = subprocess.run([str(BIN / "atlas-emit"), str(source)],
                            capture_output=True, text=True, check=True)
    return tuple(int(line, 16) for line in result.stdout.splitlines())


def _object_words(source: pathlib.Path = SOURCE) -> tuple[int, ...]:
    llvm = os.environ.get("ATLAS_LLVM_BIN")
    if not llvm:
        raise unittest.SkipTest("set ATLAS_LLVM_BIN for object lowering")
    tools = pathlib.Path(llvm)
    lowered = subprocess.run([str(BIN / "atlas-opt"), "--convert-atlas-to-llvm", str(source)],
                             capture_output=True, text=True, check=True)
    translated = subprocess.run([str(tools / "mlir-translate"), "--mlir-to-llvmir"],
                                input=lowered.stdout, capture_output=True, text=True, check=True)
    with tempfile.TemporaryDirectory() as temporary:
        obj = pathlib.Path(temporary) / "vpu.o"
        section = pathlib.Path(temporary) / "text.bin"
        subprocess.run([str(tools / "llc"), "-mtriple=riscv32-unknown-elf",
                        "-filetype=obj", "-o", str(obj)], input=translated.stdout.encode(),
                       capture_output=True, check=True)
        subprocess.run([str(tools / "llvm-objcopy"), "--dump-section", f".text={section}",
                        str(obj)], capture_output=True, check=True)
        data = section.read_bytes()
    expected = _emitted(source)
    if len(data) < 4 * len(expected):
        raise AssertionError("LLVM object shorter than the typed VPU stream")
    return tuple(int.from_bytes(data[4 * i:4 * i + 4], "little")
                 for i in range(len(expected)))


def _panel(phase: int) -> tuple[bytes, bytes]:
    # Finite BF16 values only. Both halves contain positive and negative values;
    # no approximation or torch/npu_model routine determines the expected bits.
    codes = (0x0000, 0x3F80, 0xBF80, 0x4000, 0xC000,
             0x3E80, 0xBE80, 0x4280, 0xC280)
    values = [codes[(i * 7 + phase + i // 16) % len(codes)] for i in range(1024)]
    source = b"".join(struct.pack("<H", code) for code in values)
    expected = b"".join(struct.pack("<H", 0 if code & 0x8000 else code)
                        for code in values)
    return source, expected


class VpuReluReferenceTest(unittest.TestCase):
    def test_typed_stream_matches_selected_assembler_and_llvm_object(self) -> None:
        selected_root = os.environ.get("ATLAS_ASSEMBLER_ROOT")
        if not selected_root:
            self.skipTest("set ATLAS_ASSEMBLER_ROOT for independent selected assembler")
        path = pathlib.Path(selected_root) / "assembler.py"
        spec = importlib.util.spec_from_file_location("selected_vpu_assembler", path)
        assert spec and spec.loader
        assembler = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(assembler)
        emitted = _emitted()
        self.assertEqual(len(emitted), 36)
        self.assertEqual(emitted, tuple(assembler.assemble(ASSEMBLY.read_text())))
        self.assertEqual(emitted[17], assembler.VRELU(4, 0))
        self.assertEqual(_object_words(), emitted)
        parsed = subprocess.run([str(BIN / "atlas-opt"), str(SOURCE)],
                                capture_output=True, text=True, check=True)
        self.assertEqual(parsed.stdout.count('"atlas.vpu_unary"'), 1)
        reparsed = subprocess.run([str(BIN / "atlas-opt"), "-"], input=parsed.stdout,
                                  capture_output=True, text=True, check=True)
        self.assertEqual(reparsed.stdout, parsed.stdout)

    def test_invalid_pair_and_unknown_mode_are_rejected(self) -> None:
        source = SOURCE.read_text()
        for mutation in (
            'kind = "relu", dst = 5 : i32, src = 0 : i32',
            'kind = "relu", dst = 4 : i32, src = 63 : i32',
            'kind = "unqualified", dst = 4 : i32, src = 0 : i32',
        ):
            with self.subTest(mutation=mutation):
                changed = source.replace('kind = "relu", dst = 4 : i32, src = 0 : i32', mutation)
                self.assertNotEqual(changed, source)
                run = subprocess.run([str(BIN / "atlas-opt"), "-"], input=changed,
                                     capture_output=True, text=True, check=False)
                self.assertNotEqual(run.returncode, 0)
                self.assertTrue(run.stderr)

    def test_selected_core_executes_relu_across_both_bf16_halves(self) -> None:
        keys = ("ATLAS_ARC_MODEL", "ATLAS_ARC_STATE", "ATLAS_MODELIR_ROOT", "ATLAS_LLVM_BIN")
        if not all(os.environ.get(key) for key in keys):
            self.skipTest("set selected-source-linked ARC, ModeLIR, and LLVM paths")
        model, state, modelir = (pathlib.Path(os.environ[key]).resolve(strict=True)
                                 for key in keys[:3])
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

            cosim_atlas.CosimCore = SelectedCore
            try:
                for phase in (0, 4):
                    with self.subTest(phase=phase):
                        source, expected = _panel(phase)
                        self.assertNotEqual(source, expected)
                        result = cosim_atlas.run_program(
                            model, state, _object_words(),
                            preload=[(0x90000000, source[:1024]),
                                     (0x90000400, source[1024:]),
                                     (0x90000800, b"\xA5" * 1024),
                                     (0x90000C00, b"\xA5" * 1024),
                                     (0x90001000, b"\x5A" * 32)],
                            max_cycles=8000,
                        )
                        observed = (result.slave.captured(0x90000800, 1024) +
                                    result.slave.captured(0x90000C00, 1024))
                        self.assertTrue(result.halted)
                        self.assertEqual((result.reads, result.writes), (64, 64))
                        self.assertEqual(observed, expected)
                        self.assertEqual(result.slave.captured(0x90000000, 2048), source)
                        self.assertEqual(result.slave.captured(0x90001000, 32), b"\x5A" * 32)
                # A different selected VPU mode must produce a distinguishable
                # output for the same inputs; this checks the semantic oracle.
                source, expected = _panel(0)
                with tempfile.TemporaryDirectory() as temporary:
                    changed = pathlib.Path(temporary) / "vpu_mov.mlir"
                    changed.write_text(SOURCE.read_text().replace(
                        'kind = "relu", dst = 4 : i32, src = 0 : i32',
                        'kind = "mov", dst = 4 : i32, src = 0 : i32'))
                    result = cosim_atlas.run_program(
                        model, state, _object_words(changed),
                        preload=[(0x90000000, source[:1024]),
                                 (0x90000400, source[1024:]),
                                 (0x90000800, b"\xA5" * 1024),
                                 (0x90000C00, b"\xA5" * 1024)],
                        max_cycles=8000,
                    )
                observed = (result.slave.captured(0x90000800, 1024) +
                            result.slave.captured(0x90000C00, 1024))
                self.assertTrue(result.halted)
                self.assertNotEqual(source, expected)
                self.assertEqual(observed, source)
                self.assertNotEqual(observed, expected)
            finally:
                cosim_atlas.CosimCore = original
        finally:
            os.chdir(old_cwd)
            sys.path.remove(str(modelir))


if __name__ == "__main__":
    unittest.main()
