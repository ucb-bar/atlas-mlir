"""Optional fresh PyTorch capture -> Linalg -> Atlas ELF -> selected-core test."""

from __future__ import annotations

import json
import hashlib
import os
from pathlib import Path
import shutil
import struct
import subprocess
import sys
import tempfile
import unittest

from test_captured_mlp_import import ROOT
from test_virtual_mxu import bf16_bits, bf16_cell

sys.path.insert(0, str(ROOT / "tools"))
from compile_atlas_linalg_tile import pack_fp8_tile  # noqa: E402

from selected_core_runtime import run_selected_program


REQUIRED = (
    "ATLAS_M2M_PYTHON", "ATLAS_M2M_ROOT", "ATLAS_MERLIN_CAPTURE_WORKER",
    "ATLAS_LLVM_BIN", "ATLAS_ARC_MODEL", "ATLAS_ARC_STATE",
    "ATLAS_MODELIR_ROOT", "ATLAS_OOT_BIN_DIR",
)


class CapturedMLPEndToEndTest(unittest.TestCase):
    def test_fresh_capture_compiles_once_and_executes_two_inputs(self) -> None:
        if not all(os.environ.get(key) for key in REQUIRED):
            self.skipTest("set M2M, Merlin capture worker, LLVM, and selected ARC paths")
        linker = os.environ.get("ATLAS_LLD") or shutil.which("ld.lld")
        if not linker:
            self.skipTest("set ATLAS_LLD or install ld.lld")
        # Preserve executable basenames: a venv Python and ld.lld may be
        # symlinks whose invocation name selects their environment/flavor.
        paths = {key: Path(os.environ[key]).absolute() for key in REQUIRED}
        for path in paths.values():
            self.assertTrue(path.exists(), path)
        with tempfile.TemporaryDirectory(dir=ROOT / "out") as temporary:
            temporary_root = Path(temporary)
            capture = temporary_root / "capture"
            env = dict(os.environ, TMPDIR=str(temporary_root))
            captured = subprocess.run([
                str(paths["ATLAS_M2M_PYTHON"]),
                str(paths["ATLAS_MERLIN_CAPTURE_WORKER"]),
                "--m2m-dir", str(paths["ATLAS_M2M_ROOT"]),
                "--loader", str(ROOT / "test/models/directed_mlp32.py"),
                "--dtype", "fp32", "--seed", "0", "--out", str(capture),
            ], check=True, capture_output=True, text=True, env=env)
            self.assertIn('"ok": true', captured.stdout)
            compiled = temporary_root / "compiled"
            compile_command = [
                str(paths["ATLAS_M2M_PYTHON"]),
                str(ROOT / "tools/compile_atlas_linalg_tile.py"),
                "--linalg", str(capture / "linalg.mlir"),
                "--argument-manifest",
                str(capture / "weights.safetensors.manifest.json"),
                "--weights", str(capture / "weights.safetensors"),
                "--policy", str(ROOT / "test/examples/atlas_mlp_numeric_policy.json"),
                "--atlas-bin-dir", str(paths["ATLAS_OOT_BIN_DIR"]),
                "--llvm-bin-dir", str(paths["ATLAS_LLVM_BIN"]),
                "--linker", str(linker), "--output-dir", str(compiled),
            ]
            subprocess.run(compile_command, check=True, capture_output=True,
                           text=True, env=env)
            manifest = json.loads((compiled / "import-manifest.json").read_text())
            stale = subprocess.run(compile_command, capture_output=True,
                                   text=True, env=env)
            self.assertNotEqual(stale.returncode, 0)
            self.assertIn("must be empty", stale.stderr)
            self.assertEqual(json.loads((compiled / "import-manifest.json").read_text()),
                             manifest)
            source = (capture / "linalg.mlir").read_text()
            declared = str(capture / "weights.safetensors")
            self.assertIn(declared, source)
            changed = temporary_root / "wrong-weights-source.mlir"
            changed.write_text(source.replace(declared, str(temporary_root / "other.safetensors")))
            wrong_command = list(compile_command)
            wrong_command[wrong_command.index("--linalg") + 1] = str(changed)
            wrong_command[wrong_command.index("--output-dir") + 1] = str(
                temporary_root / "rejected"
            )
            wrong = subprocess.run(wrong_command, capture_output=True,
                                   text=True, env=env)
            self.assertNotEqual(wrong.returncode, 0)
            self.assertIn("source-declared weights file differs", wrong.stderr)
            self.assertEqual(manifest["source_operations_accounted"], 19)
            self.assertEqual(manifest["mxu_units"], [0, 1])
            self.assertEqual(len(manifest["constants"]), 4)
            self.assertEqual([row["physical_role"] for row in manifest["input_slots"]],
                             ["activation_fp8", "weight_fp8_nk", "bias_bf16",
                              "weight_fp8_nk", "bias_bf16"])
            self.assertTrue(manifest["program"]["linked_text_matches_object"])
            raw = (compiled / "program.elf.text.bin").read_bytes()
            words = struct.unpack(f"<{len(raw) // 4}I", raw)
            checked = tuple(int(line, 16) for line in
                            (compiled / "program.words.txt").read_text().splitlines())
            self.assertEqual(words[:len(checked)], checked)
            physical = json.loads((compiled / "physical-program.json").read_text())
            self.assertEqual(physical["schema"], "atlas.physical_program.v1")
            self.assertEqual([row["word_u32"] for row in physical["instructions"]],
                             list(checked))
            self.assertEqual(
                hashlib.sha256((compiled / "physical-program.json").read_bytes()).hexdigest(),
                manifest["program"]["physical_program_sha256"],
            )
            self.assertEqual(len(checked), 143)

            modelir = paths["ATLAS_MODELIR_ROOT"]
            previous = Path.cwd()
            sys.path.insert(0, str(modelir))
            try:
                os.chdir(modelir)
                from mlc.backends import cosim_atlas

                original = cosim_atlas.CosimCore

                class SelectedCore(original):
                    def peek(self, signal: str) -> int:
                        return super().peek("scalar/halt_now" if signal == "io_halted"
                                            else signal)

                    def poke(self, signal: str, value: int) -> None:
                        if signal in ("io_dmaTL_d_bits_opcode", "io_dmaTL_d_bits_size"):
                            if signal in self._S:
                                raise AssertionError(f"unexpected ARC input {signal}")
                            return
                        super().poke(signal, value)

                cosim_atlas.CosimCore = SelectedCore
                try:
                    executions = self.execute_variants(compiled, manifest, words,
                                                       paths, cosim_atlas, env)
                finally:
                    cosim_atlas.CosimCore = original
            finally:
                os.chdir(previous)
                sys.path.remove(str(modelir))
            report_path = os.environ.get("ATLAS_MLP_RESULT_JSON")
            if report_path:
                destination = Path(report_path).absolute()
                destination.parent.mkdir(parents=True, exist_ok=True)
                destination.write_text(json.dumps({
                    "schema": "atlas.oot.captured_mlp_diagnostic.v1",
                    "platform": "selected_source_linked_standalone_AtlasCore_ARC",
                    "capture_source_sha256": manifest["source_sha256"],
                    "compiler_source_sha256": manifest["compiler_source_sha256"],
                    "program_elf_sha256": manifest["program"]["elf_sha256"],
                    "selected_rtl_revision": manifest["selected_rtl_revision"],
                    "source_precision": "f32",
                    "target_precision": "E4M3_operands_BF16_output",
                    "precision_policy_status": "explicit_diagnostic_candidate",
                    "compile_count": 1,
                    "executions": executions,
                    "scope": "32x32 two-biased-Linear ReLU MLP; two finite exact inputs",
                }, indent=2) + "\n")

    def execute_variants(self, compiled, manifest, words, paths, cosim_atlas,
                         env) -> list[dict]:
        import numpy as np

        executions = []
        for variant in (0, 1):
            with self.subTest(runtime_input=variant):
                evaluated = subprocess.run([
                    str(paths["ATLAS_M2M_PYTHON"]),
                    str(ROOT / "test/models/evaluate_directed_mlp32.py"),
                    "--variant", str(variant),
                ], check=True, capture_output=True, text=True, env=env)
                reference = json.loads(evaluated.stdout)
                runtime_input = np.asarray(reference["input"], dtype="float32")
                packed_input = pack_fp8_tile(runtime_input)
                self.assertEqual(len(packed_input), 1024)
                allowed = {0x00, 0x30, 0x38, 0x40, 0xB8}
                self.assertEqual(set(packed_input), allowed)
                input_slot = manifest["input_slots"][0]
                preload = [(input_slot["dram_address"], packed_input)]
                preload.extend(
                    (row["dram_address"],
                     (compiled / row["payload_file"]).read_bytes())
                    for row in manifest["constants"]
                )
                output_base = manifest["output"]["dram_address"]
                guard = b"\x5A" * 64
                preload += [(output_base, b"\xA5" * 2048),
                            (output_base + 0x800, guard)]
                result = run_selected_program(cosim_atlas,
                    paths["ATLAS_ARC_MODEL"], paths["ATLAS_ARC_STATE"],
                    words, preload=preload, max_cycles=100000,
                )
                self.assertTrue(result.halted)
                self.assertEqual((result.reads, result.writes), (224, 64))
                output = result.slave.captured(output_base, 2048)
                for row in range(32):
                    for col in range(32):
                        expected = bf16_bits(reference["original_output"][row][col])
                        self.assertEqual(bf16_cell(output, row, col), expected,
                                         (variant, row, col))
                for address, data in preload:
                    if address != output_base:
                        self.assertEqual(result.slave.captured(address, len(data)),
                                         data)
                executions.append({
                    "runtime_input_variant": variant,
                    "reference": "original_PyTorch_f32_output",
                    "matching_output_cells": 1024,
                    "required_output_cells": 1024,
                    "inputs_constants_and_guard_preserved": True,
                    "halted": True,
                    "cycles": result.cycles,
                    "dma_reads": result.reads,
                    "dma_writes": result.writes,
                })
        return executions


if __name__ == "__main__":
    unittest.main()
