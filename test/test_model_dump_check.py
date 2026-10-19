"""Fault checks for the external Atlas model dump handoff."""

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
from atlas_check_model_dump import check, main


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


class ModelDumpCheckTest(unittest.TestCase):
    def setUp(self) -> None:
        temporary_root = Path(os.environ.get("TMPDIR", ROOT / "out"))
        temporary_root.mkdir(parents=True, exist_ok=True)
        self.temporary = tempfile.TemporaryDirectory(dir=temporary_root)
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.handoff = self.root / "handoff"
        (self.handoff / "host").mkdir(parents=True)
        elf = b"public diagnostic ELF fixture"
        (self.handoff / "host/full-graph-diagnostic.elf").write_bytes(elf)
        self.package = {
            "schema": "atlas.diagnostic_seven_region_handoff.v1",
            "status": "diagnostic_unbound",
            "bitstream_bound": False,
            "elf_sha256": sha(elf),
            "files": {"host/full-graph-diagnostic.elf": {"sha256": sha(elf)}},
        }
        self._write_package()
        self.status = self.root / "status.bin"
        self.status.write_bytes(struct.pack("<I", 1))
        self.outputs = self.root / "outputs.bin"
        self.outputs.write_bytes(struct.pack("<2048f", *range(2048)))
        self.expected = self.root / "expected.bin"
        self.expected.write_bytes(self.outputs.read_bytes())
        self.receipt_path = self.root / "receipt.json"
        self.receipt = {
            "schema": "atlas.external_model_dump.v1",
            "run_id": "public-fixture-run-1",
            "bitstream_sha256": sha(b"reported bitstream fixture"),
            "elf_sha256": sha(elf),
        }
        self._write_receipt()

    def _write_package(self) -> None:
        (self.handoff / "handoff-status.json").write_text(json.dumps(self.package))

    def _write_receipt(self) -> None:
        self.receipt["status_sha256"] = sha(self.status.read_bytes())
        self.receipt["outputs_sha256"] = sha(self.outputs.read_bytes())
        self.receipt_path.write_text(json.dumps(self.receipt))

    def _check(self) -> dict:
        return check(self.handoff, self.receipt_path, self.status,
                     self.outputs, self.expected)

    def test_reported_match_keeps_hardware_qualification_explicit(self) -> None:
        result = self._check()
        self.assertEqual(result["status"], "reported_match_unqualified")
        self.assertEqual(result["matching_output_bytes"], 8192)
        self.assertTrue(result["limitations"])

    def test_output_mutation_is_not_a_match(self) -> None:
        damaged = bytearray(self.outputs.read_bytes())
        damaged[100] ^= 1
        self.outputs.write_bytes(damaged)
        self._write_receipt()
        result = self._check()
        self.assertEqual(result["status"], "numerical_mismatch")
        self.assertEqual(result["matching_output_bytes"], 8191)

    def test_device_failure_status_is_reported_before_output_comparison(self) -> None:
        self.status.write_bytes(struct.pack("<I", 0x40000201))
        self._write_receipt()
        result = self._check()
        self.assertEqual(result["status"], "execution_failure")
        self.assertEqual((result["tile_status"], result["device_status"]), (2, 1))

    def test_stale_receipt_or_changed_elf_is_rejected(self) -> None:
        self.outputs.write_bytes(bytes(8192))
        with self.assertRaisesRegex(ValueError, "differs from its receipt"):
            self._check()
        self.outputs.write_bytes(self.expected.read_bytes())
        (self.handoff / "host/full-graph-diagnostic.elf").write_bytes(b"changed")
        with self.assertRaisesRegex(ValueError, "ELF identity"):
            self._check()

    def test_wrong_dump_length_and_existing_report_are_rejected(self) -> None:
        self.outputs.write_bytes(bytes(4))
        self._write_receipt()
        with self.assertRaisesRegex(ValueError, "invalid size"):
            self._check()
        self.outputs.write_bytes(self.expected.read_bytes())
        self._write_receipt()
        report = self.root / "report.json"
        arguments = ["--handoff", str(self.handoff), "--receipt", str(self.receipt_path),
                     "--status", str(self.status), "--outputs", str(self.outputs),
                     "--expected", str(self.expected), "--out", str(report)]
        self.assertEqual(main(arguments), 0)
        self.assertEqual(main(arguments), 2)
        self.assertEqual(json.loads(report.read_text())["status"],
                         "reported_match_unqualified")


if __name__ == "__main__":
    unittest.main()
