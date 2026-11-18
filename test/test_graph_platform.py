"""Graph handoff checks against a synthetic, explicitly bounded SoC map."""

from __future__ import annotations

import hashlib
import json
import os
import struct
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
from atlas_graph_platform import check, render_map_c


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _row(base: int, size: int, *, executable: bool = False,
         cached: bool = False) -> dict:
    return {"base": [base], "size": [size], "r": [True], "w": [True],
            "x": [executable], "c": [cached], "names": [f"synthetic-{base:x}"]}


class GraphPlatformTest(unittest.TestCase):
    def setUp(self) -> None:
        temporary_root = Path(os.environ.get("TMPDIR", ROOT / "out"))
        temporary_root.mkdir(parents=True, exist_ok=True)
        self.temporary = tempfile.TemporaryDirectory(dir=temporary_root)
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.compiled = self.root / "compiled"
        self.compiled.mkdir()
        (self.compiled / "constants").mkdir()
        self.host_map_path = self.root / "host-map.json"
        self.memmap_path = self.root / "memmap.json"
        self.elf_path = self.root / "host.elf"
        self.program = b"\x13\x00\x00\x00"
        (self.compiled / "program.bin").write_bytes(self.program)
        constant = b"\x00" * 64
        (self.compiled / "constants/constant-000.bin").write_bytes(constant)
        self.plan = {
            "schema": "atlas.native_tile_execution_plan.v8",
            "input_addresses": [0x90000000, 0x90000100],
            "input_payload_bytes": [64, 64],
            "output_addresses": [0x90010000], "output_bytes": 64,
        }
        self._write_plan()
        self.memmap = {"mapping": [
            _row(0x20000, 0x20000), _row(0x40000, 0x1000),
            _row(0x80000000, 0x10000, executable=True, cached=True),
            _row(0x90000000, 0x20000, cached=True),
        ]}
        self._write_memmap()
        self.host_map = json.loads((ROOT / "runtime/selected-host-map.json").read_text())
        self.host_map["bitstream_identity"] = "synthetic-test-build"
        self._write_host_map()
        elf = bytearray(0x104)
        elf[:7] = b"\x7fELF\x02\x01\x01"
        struct.pack_into("<HHIQQQIHHHHHH", elf, 16,
                         2, 243, 1, 0x80000000, 64, 0, 0, 64, 56, 1, 0, 0, 0)
        struct.pack_into("<IIQQQQQQ", elf, 64,
                         1, 5, 0x100, 0x80000000, 0x80000000, 4, 8, 0x100)
        elf[0x100:] = self.program
        self.elf_path.write_bytes(elf)

    def _write_plan(self) -> None:
        plan_bytes = json.dumps(self.plan).encode()
        (self.compiled / "execution_plan.json").write_bytes(plan_bytes)
        manifest = {
            "schema": "atlas.native_tile_compilation.v6",
            "engine": "merlin_native", "binary_sha256": _sha(self.program),
            "execution_plan_sha256": _sha(plan_bytes),
            "target_identity": "atlas-rtl-0079c0541111197741a231c002e3843fa6f545b2:synthetic",
            "constant_inputs": [{"address": 0x90000100, "bytes": 64,
                                 "path": "constants/constant-000.bin",
                                 "sha256": _sha(b"\x00" * 64)}],
        }
        (self.compiled / "manifest.json").write_text(json.dumps(manifest))

    def _write_memmap(self) -> None:
        self.memmap_path.write_text(json.dumps(self.memmap))

    def _write_host_map(self) -> None:
        self.host_map["soc_memmap_sha256"] = _sha(self.memmap_path.read_bytes())
        self.host_map_path.write_text(json.dumps(self.host_map))

    def test_checked_platform_map_and_final_elf(self) -> None:
        preflight, host_map = check(self.compiled, self.host_map_path,
                                    self.memmap_path)
        self.assertEqual(preflight["status"], "platform_inputs_checked")
        self.assertEqual(len(preflight["mapped_regions"]["dram"]), 3)
        self.assertIn("atlas_selected_host_map", render_map_c(host_map))
        linked, _ = check(self.compiled, self.host_map_path, self.memmap_path,
                          self.elf_path)
        self.assertEqual(linked["status"], "platform_elf_checked")
        self.assertEqual(linked["host_elf"]["sha256"], _sha(self.elf_path.read_bytes()))

    def test_c_handoff_checks_each_serial_image_and_its_bytes(self) -> None:
        handoff = self.root / "handoff"
        (handoff / "programs").mkdir(parents=True)
        (handoff / "programs/a.bin").write_bytes(self.program)
        second = b"\x73\x00\x00\x00"
        (handoff / "programs/b.bin").write_bytes(second)
        other_plan = {**self.plan, "output_addresses": [0x90018000]}
        abi = {"schema": "merlin.c_handoff.v1",
               "target_identity": json.loads((self.compiled / "manifest.json").read_text())["target_identity"],
               "files": {"programs/a.bin": _sha(self.program), "programs/b.bin": _sha(second)},
               "programs": [{"path": "programs/a.bin", "sha256": _sha(self.program), "bytes": 4, "execution": self.plan},
                            {"path": "programs/b.bin", "sha256": _sha(second), "bytes": 4, "execution": other_plan}]}
        (handoff / "abi.json").write_text(json.dumps(abi))
        report, _ = check(handoff, self.host_map_path, self.memmap_path, self.elf_path)
        self.assertEqual(len(report["program_sha256s"]), 2)
        abi["programs"][1]["execution"]["output_addresses"] = [0x91000000]
        (handoff / "abi.json").write_text(json.dumps(abi))
        with self.assertRaisesRegex(ValueError, "DRAM output_0"):
            check(handoff, self.host_map_path, self.memmap_path)
        (handoff / "programs/b.bin").write_bytes(self.program)
        with self.assertRaisesRegex(ValueError, "member identity"):
            check(handoff, self.host_map_path, self.memmap_path)

    def test_changed_map_or_target_fails_before_build(self) -> None:
        self.memmap_path.write_text(self.memmap_path.read_text() + " ")
        with self.assertRaisesRegex(ValueError, "memmap hash"):
            check(self.compiled, self.host_map_path, self.memmap_path)
        self._write_memmap()
        self.host_map["rtl_revision"] = "0" * 40
        self._write_host_map()
        with self.assertRaisesRegex(ValueError, "different Atlas sources"):
            check(self.compiled, self.host_map_path, self.memmap_path)
        self.host_map["rtl_revision"] = "0079c0541111197741a231c002e3843fa6f545b2"
        self.host_map["bitstream_identity"] = None
        self._write_host_map()
        with self.assertRaisesRegex(ValueError, "unbound"):
            check(self.compiled, self.host_map_path, self.memmap_path)

    def test_missing_dram_constant_and_bad_elf_fail(self) -> None:
        self.memmap["mapping"][-1]["size"] = [0x10000]
        self._write_memmap()
        self._write_host_map()
        with self.assertRaisesRegex(ValueError, "DRAM output_0"):
            check(self.compiled, self.host_map_path, self.memmap_path)
        self.memmap["mapping"][-1]["size"] = [0x20000]
        self._write_memmap()
        self._write_host_map()
        (self.compiled / "constants/constant-000.bin").write_bytes(b"\x01" * 64)
        with self.assertRaisesRegex(ValueError, "constant placement or identity"):
            check(self.compiled, self.host_map_path, self.memmap_path)
        (self.compiled / "constants/constant-000.bin").write_bytes(b"\x00" * 64)
        self.elf_path.write_bytes(b"not an ELF")
        with self.assertRaisesRegex(ValueError, "ELF64"):
            check(self.compiled, self.host_map_path, self.memmap_path, self.elf_path)

    def test_host_elf_overlapping_device_buffers_is_refused(self) -> None:
        elf = bytearray(self.elf_path.read_bytes())
        struct.pack_into("<IIQQQQQQ", elf, 64,
                         1, 7, 0x100, 0x90000000, 0x90000000, 4, 8, 0x100)
        struct.pack_into("<Q", elf, 24, 0x90000000)
        self.elf_path.write_bytes(elf)
        self.memmap["mapping"][-1]["x"] = [True]
        self._write_memmap()
        self._write_host_map()
        with self.assertRaisesRegex(ValueError, "overlaps Atlas input_0"):
            check(self.compiled, self.host_map_path, self.memmap_path, self.elf_path)

    def test_malformed_target_identity_is_a_value_error(self) -> None:
        manifest = json.loads((self.compiled / "manifest.json").read_text())
        manifest["target_identity"] = 7
        (self.compiled / "manifest.json").write_text(json.dumps(manifest))
        with self.assertRaisesRegex(ValueError, "different Atlas sources"):
            check(self.compiled, self.host_map_path, self.memmap_path)


if __name__ == "__main__":
    unittest.main()
