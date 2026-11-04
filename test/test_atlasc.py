"""atlasc: one command from a captured or PyTorch model to a program, host headers and ELF."""

from __future__ import annotations

import contextlib
import importlib.util
import io
import json
import struct
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
from safetensors.numpy import save_file

from atlas_mlir import toolchain
from atlas_mlir.cli import main
from atlas_mlir.capture import Capture
from atlas_mlir.driver import build

ROOT = Path(__file__).resolve().parents[1]
EXAMPLES = ROOT / "test/examples"
POLICY = EXAMPLES / "atlas_mlp_numeric_policy.json"
CAPTURED = ["--linalg", str(EXAMPLES / "captured_mlp32_linalg.mlir"),
            "--arguments", str(EXAMPLES / "captured_mlp32_arguments.json"),
            "--weights", str(EXAMPLES / "captured_mlp32_weights.safetensors")]


def directed_input() -> np.ndarray:
    """The example input of test/models/directed_mlp32.py, built without torch."""
    values = (0.0, 0.5, 1.0, 2.0, -1.0)
    return np.array([[values[(row * 7 + col * 3) % len(values)] for col in range(32)]
                     for row in range(32)], dtype=np.float32)


def atlasc(*arguments: str) -> int:
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        return main(list(arguments))


class AtlascBuildTest(unittest.TestCase):
    def setUp(self) -> None:
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.inputs = self.root / "inputs.safetensors"
        save_file({"input": directed_input()}, str(self.inputs))

    def test_captured_linalg_builds_the_importer_program_and_host_files(self) -> None:
        out = self.root / "build"
        self.assertEqual(atlasc("build", *CAPTURED, "--inputs", str(self.inputs),
                                "--policy", str(POLICY), "-o", str(out)), 0)
        direct = self.root / "direct"
        subprocess.run([sys.executable, str(ROOT / "tools/compile_atlas_linalg_tile.py"),
                        *CAPTURED[:2], "--argument-manifest", CAPTURED[3],
                        *CAPTURED[4:], "--policy", str(POLICY),
                        "--atlas-bin-dir", str(toolchain.bin_dir()),
                        "--output-dir", str(direct)], check=True, capture_output=True)
        program = (out / "compiled/program.bin").read_bytes()
        self.assertEqual(program, (direct / "program.bin").read_bytes())
        report = json.loads((out / "report.json").read_text())
        self.assertEqual(report["program"]["words"], len(program) // 4)
        self.assertEqual(report["mxu_units"], [0, 1])
        self.assertEqual(report["inputs"], [{"name": "input", "dram_address": "0x90000000",
                                             "role": "activation_fp8"}])
        regions = json.loads((out / "layout.json").read_text())["regions"]
        self.assertEqual([(row["kind"], row["address"], row["size_bytes"]) for row in regions],
                         [("input", "0x90000000", 5 * 2048), ("output", "0x90010000", 2048),
                          ("guard", "0x90010800", 32)])
        header = (out / "host/atlas_run_inputs.h").read_text()
        self.assertIn("#define ATLAS_RUN_INPUT_COUNT 1u", header)
        self.assertIn("{0x90000000u, 1024u, atlas_run_input_0}", header)
        for name in ("atlas_captured_data.h", "atlas_program.h"):
            self.assertTrue((out / "host" / name).is_file(), name)
        elf = out / "host/atlas-host.elf"
        if toolchain.host_cc() is None:
            self.assertFalse(elf.exists())
            self.assertIn("no RISC-V compiler", report["host"]["elf_skipped"])
            return
        data = elf.read_bytes()
        self.assertEqual(data[:6], b"\x7fELF\x02\x01")  # 64-bit, little-endian.
        self.assertEqual(struct.unpack_from("<H", data, 18)[0], 243)  # RISC-V.
        self.assertEqual(struct.unpack_from("<Q", data, 24)[0], 0x80000000)

    def test_fp8_preset_places_every_contraction_on_mxu0(self) -> None:
        out = self.root / "build"
        self.assertEqual(atlasc("build", *CAPTURED, "--no-host-elf", "-o", str(out)), 0)
        policy = json.loads((out / "policy.json").read_text())
        self.assertEqual(policy["mxu_units"], [0, 0])
        self.assertNotIn("softmax_realization", policy)
        report = json.loads((out / "report.json").read_text())
        self.assertEqual(report["mxu_units"], [0, 0])
        self.assertIsNone(report["host"]["elf"])

    def test_output_directory_is_fresh_or_an_earlier_build(self) -> None:
        out = self.root / "build"
        arguments = ("build", *CAPTURED, "--no-host-elf", "-o", str(out))
        self.assertEqual(atlasc(*arguments), 0)
        self.assertEqual(atlasc(*arguments), 1)
        self.assertEqual(atlasc(*arguments, "--force"), 0)
        foreign = self.root / "foreign"
        foreign.mkdir()
        (foreign / "keep.txt").write_text("not an atlasc build\n")
        self.assertEqual(atlasc("build", *CAPTURED, "--no-host-elf", "--force",
                                "-o", str(foreign)), 1)
        self.assertTrue((foreign / "keep.txt").is_file())

    def test_capture_inputs_are_used_by_python_api(self) -> None:
        result = build(out=self.root / "api", captured=Capture(
            Path(CAPTURED[1]), Path(CAPTURED[3]), Path(CAPTURED[5]), self.inputs),
            host_elf=False)
        self.assertTrue((result.out / "host/atlas_run_inputs.h").is_file())

    def test_bundled_demo_and_missing_file_diagnostic(self) -> None:
        self.assertEqual(atlasc("demo", "-o", str(self.root / "demo")), 0)
        self.assertEqual(atlasc("build", "--linalg", str(self.root / "missing.mlir"),
                                *CAPTURED[2:], "-o", str(self.root / "missing")), 1)

    @unittest.skipUnless(importlib.util.find_spec("torch") and importlib.util.find_spec("torch_mlir"),
                         "capturing a PyTorch model needs torch and torch-mlir")
    def test_pytorch_loader_compiles_to_the_model2mlir_program(self) -> None:
        captured = self.root / "captured"
        self.assertEqual(atlasc("build", *CAPTURED, "--policy", str(POLICY), "--no-host-elf",
                                "-o", str(captured)), 0)
        out = self.root / "pytorch"
        self.assertEqual(atlasc("build", str(ROOT / "test/models/directed_mlp32.py"),
                                "--policy", str(POLICY), "--no-host-elf", "-o", str(out)), 0)
        self.assertEqual(json.loads((out / "capture/arguments.json").read_text()),
                         json.loads((EXAMPLES / "captured_mlp32_arguments.json").read_text()))
        self.assertEqual((out / "compiled/program.bin").read_bytes(),
                         (captured / "compiled/program.bin").read_bytes())


if __name__ == "__main__":
    unittest.main()
