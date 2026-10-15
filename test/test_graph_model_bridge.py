"""Check the native graph to MLIR C-interface handoff with real artifacts."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
from atlas_graph_model_bridge import prepare


@unittest.skipUnless(
    all(os.environ.get(name) for name in (
        "MERLIN_NATIVE_ROOT", "ATLAS_GRAPH_SOURCE", "ATLAS_GRAPH_ARTIFACT",
        "ATLAS_GRAPH_HOST_BRIDGE",
    )),
    "select a Merlin checkout, source, native graph, and host bridge",
)
class GraphModelBridgeTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        sys.path.insert(0, str(Path(os.environ["MERLIN_NATIVE_ROOT"]) / "src"))
        cls.source = Path(os.environ["ATLAS_GRAPH_SOURCE"])
        cls.compiled = Path(os.environ["ATLAS_GRAPH_ARTIFACT"])
        cls.bridge = Path(os.environ["ATLAS_GRAPH_HOST_BRIDGE"])

    @classmethod
    def tearDownClass(cls) -> None:
        sys.path.remove(str(Path(os.environ["MERLIN_NATIVE_ROOT"]) / "src"))

    def test_generated_callback_compiles_with_selected_c_interface(self) -> None:
        compiler = shutil.which("cc")
        nm = shutil.which("nm")
        if not compiler or not nm:
            self.skipTest("C compiler and nm are required")
        record, files = prepare(self.source, self.bridge, self.compiled)
        self.assertEqual(record["status"], "generated_unlinked")
        self.assertEqual(record["input_count"], 6)
        self.assertIn(record["output_count"], (2, 3))
        temporary_root = Path(os.environ.get("TMPDIR", ROOT / "out"))
        temporary_root.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=temporary_root) as temporary:
            directory = Path(temporary)
            for name, content in files.items():
                (directory / name).write_text(content)
            output = directory / "callback.o"
            command = [
                compiler, "-std=c11", "-O2", "-Wall", "-Wextra", "-Werror",
                "-DATLAS_DIAGNOSTIC_GRAPH_PRECISION=1",
                "-I", str(directory), "-I", str(ROOT / "runtime"),
                "-I", str(Path(os.environ["MERLIN_NATIVE_ROOT"]) / "merlin/runtime/c"),
                "-c", str(directory / "atlas_graph_model_wrapper.c"),
                "-o", str(output),
            ]
            subprocess.run(command, check=True, capture_output=True, text=True)
            symbols = subprocess.run([nm, "-P", str(output)], check=True,
                                     capture_output=True, text=True).stdout
            self.assertIn("_mlir_ciface_" + record["device_symbol"] + " T ", symbols)
            host_entry = directory / "host-entry.o"
            subprocess.run([
                compiler, "-std=c11", "-O2", "-Wall", "-Wextra", "-Werror",
                "-I", str(directory), "-I", str(ROOT / "runtime"),
                "-I", str(Path(os.environ["MERLIN_NATIVE_ROOT"]) / "merlin/runtime/c"),
                "-c", str(directory / "atlas_graph_host_entry.c"),
                "-o", str(host_entry),
            ], check=True, capture_output=True, text=True)
            entry_symbols = subprocess.run([nm, "-P", str(host_entry)], check=True,
                                           capture_output=True, text=True).stdout
            self.assertIn("atlas_graph_invoke_host T ", entry_symbols)

    def test_changed_source_or_bridge_is_refused(self) -> None:
        temporary_root = Path(os.environ.get("TMPDIR", ROOT / "out"))
        temporary_root.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=temporary_root) as temporary:
            directory = Path(temporary)
            changed_source = directory / "source.mlir"
            changed_source.write_bytes(self.source.read_bytes() + b"\n")
            with self.assertRaisesRegex(ValueError, "differs from the checked native program"):
                prepare(changed_source, self.bridge, self.compiled)
            bridge = directory / "bridge"
            shutil.copytree(self.bridge, bridge)
            with (bridge / "host-bridge.mlir").open("ab") as stream:
                stream.write(b"\n")
            with self.assertRaisesRegex(ValueError, "differs from the checked native program"):
                prepare(self.source, bridge, self.compiled)

    def test_linked_mlir_host_entry_stages_and_returns_ordered_outputs(self) -> None:
        llvm_ir = os.environ.get("ATLAS_GRAPH_HOST_LLVM_IR")
        llvm_clang = os.environ.get("ATLAS_GRAPH_LLVM_CLANG")
        host_clang = shutil.which("clang")
        if not llvm_ir or not llvm_clang or not host_clang:
            self.skipTest("select the lowered host LLVM IR and matching LLVM clang")
        _, files = prepare(self.source, self.bridge, self.compiled)
        temporary_root = Path(os.environ.get("TMPDIR", ROOT / "out"))
        temporary_root.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=temporary_root) as temporary:
            directory = Path(temporary)
            for name, content in files.items():
                (directory / name).write_text(content)
            model_object = directory / "model.o"
            subprocess.run([
                llvm_clang, "-O1", "-Wno-override-module", "-c", llvm_ir,
                "-o", str(model_object),
            ], check=True, capture_output=True, text=True)
            executable = directory / "host-entry-test"
            subprocess.run([
                host_clang, "-std=c11", "-O1", "-g", "-Wall", "-Wextra",
                "-Werror", "-fsanitize=address,undefined", "-fno-omit-frame-pointer",
                "-DATLAS_DIAGNOSTIC_GRAPH_PRECISION=1",
                "-I", str(directory), "-I", str(ROOT / "runtime"),
                "-I", str(Path(os.environ["MERLIN_NATIVE_ROOT"]) / "merlin/runtime/c"),
                str(model_object), str(directory / "atlas_graph_model_wrapper.c"),
                str(directory / "atlas_graph_host_entry.c"),
                str(ROOT / "runtime/atlas_graph_memref.c"),
                str(ROOT / "runtime/atlas_host.c"),
                str(ROOT / "test/graph-model-bridge-host-test.c"),
                "-o", str(executable),
            ], check=True, capture_output=True, text=True)
            subprocess.run([str(executable)], check=True, capture_output=True,
                           text=True)

    def test_rv64_mailbox_build_is_source_bound_and_fresh(self) -> None:
        llvm_clang = os.environ.get("ATLAS_GRAPH_LLVM_CLANG")
        if not llvm_clang:
            self.skipTest("select the matching LLVM clang")
        llvm_bin = Path(llvm_clang).parent
        script = ROOT / "examples/handoff/captured_attention_mlp/build_graph_mailbox.sh"
        temporary_root = Path(os.environ.get("TMPDIR", ROOT / "out"))
        temporary_root.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=temporary_root) as temporary:
            output = Path(temporary) / "handoff"
            command = [
                str(script), os.environ["MERLIN_NATIVE_ROOT"], sys.executable,
                str(llvm_bin), str(self.source), str(self.compiled), str(output),
            ]
            subprocess.run(command, check=True, capture_output=True, text=True)
            status_path = output / "host/handoff-status.json"
            status_bytes = status_path.read_bytes()
            status = json.loads(status_bytes)
            elf = (output / "host/shuttle-graph-host.elf").read_bytes()
            self.assertEqual(status["status"], "diagnostic_unbound")
            _, files = prepare(self.source, self.bridge, self.compiled)
            self.assertIn("atlas_graph_host_entry.c", files)
            expected_outputs = len(json.loads(
                (self.bridge / "manifest.json").read_text()
            )["result_types"])
            self.assertEqual(status["output_count"], expected_outputs)
            self.assertEqual(status["mailbox_bytes"], 24600 + 4096 * expected_outputs)
            self.assertEqual(status["mailbox_offsets"]["guard_after"],
                             24596 + 4096 * expected_outputs)
            self.assertEqual(status["elf_sha256"], hashlib.sha256(elf).hexdigest())
            symbols = subprocess.run([
                str(llvm_bin / "llvm-nm"), "--undefined-only",
                str(output / "host/shuttle-graph-host.elf"),
            ], check=True, capture_output=True, text=True).stdout
            self.assertFalse(symbols.strip())
            again = subprocess.run(command, check=False, capture_output=True, text=True)
            self.assertNotEqual(again.returncode, 0)
            self.assertEqual(status_path.read_bytes(), status_bytes)


if __name__ == "__main__":
    unittest.main()
