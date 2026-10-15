"""Compile the source-graph memref runtime against a fake Atlas launch."""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


class GraphMemrefTest(unittest.TestCase):
    def test_graph_call_preserves_inputs_and_ordered_results(self) -> None:
        compiler = shutil.which("cc")
        merlin = os.environ.get("MERLIN_NATIVE_ROOT")
        if compiler is None or not merlin:
            self.skipTest("a C compiler and MERLIN_NATIVE_ROOT are required")
        temporary_root = Path(os.environ.get("TMPDIR", ROOT / "out"))
        temporary_root.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=temporary_root) as temporary:
            executable = Path(temporary) / "graph-memref-test"
            command = [
                compiler, "-std=c11", "-O2", "-Wall", "-Wextra", "-Werror",
                "-I", str(ROOT / "runtime"),
                "-I", str(Path(merlin) / "merlin/runtime/c"),
                str(ROOT / "runtime/atlas_graph_memref.c"),
                str(ROOT / "runtime/atlas_host.c"),
                str(ROOT / "test/graph-memref-test.c"),
                "-o", str(executable),
            ]
            subprocess.run(command, check=True, capture_output=True, text=True)
            subprocess.run([str(executable)], check=True, capture_output=True, text=True)

    def test_freestanding_rv64_object(self) -> None:
        compiler = shutil.which("clang")
        merlin = os.environ.get("MERLIN_NATIVE_ROOT")
        if compiler is None or not merlin:
            self.skipTest("clang and MERLIN_NATIVE_ROOT are required")
        temporary_root = Path(os.environ.get("TMPDIR", ROOT / "out"))
        temporary_root.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=temporary_root) as temporary:
            output = Path(temporary) / "atlas_graph_memref.o"
            command = [
                compiler, "--target=riscv64-unknown-elf", "-march=rv64gc",
                "-mabi=lp64d", "-mcmodel=medany", "-std=c11", "-ffreestanding",
                "-DATLAS_FREESTANDING", "-Wall", "-Wextra", "-Werror",
                "-I", str(ROOT / "runtime"),
                "-I", str(Path(merlin) / "merlin/runtime/c"),
                "-c", str(ROOT / "runtime/atlas_graph_memref.c"),
                "-o", str(output),
            ]
            subprocess.run(command, check=True, capture_output=True, text=True)
            header = output.read_bytes()[:20]
            self.assertEqual(header[:4], b"\x7fELF")
            self.assertEqual(int.from_bytes(header[18:20], "little"), 243)


if __name__ == "__main__":
    unittest.main()
