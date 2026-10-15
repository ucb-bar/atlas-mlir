"""Compile the source-matmul memref adapter against a checked fake tile ABI."""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


class SourceMNMemrefTest(unittest.TestCase):
    def test_strided_rank_two_and_batched_views(self) -> None:
        compiler = shutil.which("cc")
        if compiler is None:
            self.skipTest("a C compiler is required")
        temporary_root = Path(os.environ.get("TMPDIR", ROOT / "out"))
        temporary_root.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=temporary_root) as temporary:
            executable = Path(temporary) / "source-mn-memref-test"
            command = [
                compiler, "-std=c11", "-O2", "-Wall", "-Wextra", "-Werror",
                "-I", str(ROOT / "runtime"),
                str(ROOT / "runtime/atlas_source_mn_memref.c"),
                str(ROOT / "test/source-mn-memref-test.c"),
                "-o", str(executable),
            ]
            subprocess.run(command, check=True, capture_output=True, text=True)
            subprocess.run([str(executable)], check=True, capture_output=True, text=True)

    def test_freestanding_rv64_object(self) -> None:
        compiler = shutil.which("clang")
        if compiler is None:
            self.skipTest("clang is required for the RV64 object check")
        temporary_root = Path(os.environ.get("TMPDIR", ROOT / "out"))
        temporary_root.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=temporary_root) as temporary:
            object_path = Path(temporary) / "atlas_source_mn_memref.o"
            command = [
                compiler, "--target=riscv64-unknown-elf", "-march=rv64gc",
                "-mabi=lp64d", "-mcmodel=medany", "-std=c11", "-ffreestanding",
                "-DATLAS_FREESTANDING", "-Wall", "-Wextra", "-Werror",
                "-I", str(ROOT / "runtime"), "-c",
                str(ROOT / "runtime/atlas_source_mn_memref.c"),
                "-o", str(object_path),
            ]
            subprocess.run(command, check=True, capture_output=True, text=True)
            header = object_path.read_bytes()[:20]
            self.assertEqual(header[:4], b"\x7fELF")
            self.assertEqual(int.from_bytes(header[18:20], "little"), 243)


if __name__ == "__main__":
    unittest.main()
