"""Check repeated mailbox responses without trusting a saved runner dump."""

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
from atlas_check_mailbox_dump import check, main


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


class MailboxDumpCheckTest(unittest.TestCase):
    def setUp(self) -> None:
        temporary_root = Path(os.environ.get("TMPDIR", ROOT / "out"))
        temporary_root.mkdir(parents=True, exist_ok=True)
        self.temporary = tempfile.TemporaryDirectory(dir=temporary_root)
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.handoff = self.root / "handoff"
        (self.handoff / "host").mkdir(parents=True)
        self.elf = b"public mailbox ELF fixture"
        (self.handoff / "host/full-graph-mailbox.elf").write_bytes(self.elf)
        status = {
            "schema": "atlas.attention_mlp_mailbox_handoff.v1",
            "status": "diagnostic_unbound",
            "elf_sha256": sha(self.elf),
            "mailbox_offsets": {"request_id": 0, "response_id": 64, "status": 68,
                                "input": 128, "output": 4224},
            "mailbox_bytes": 8320,
            "mailbox_address": 0x80010000,
        }
        (self.handoff / "host/handoff-status.json").write_text(json.dumps(status))
        self.package = {
            "schema": "atlas.attention_mlp_mailbox_package.v1",
            "status": "diagnostic_unbound",
            "engine": "merlin_native",
            "bitstream_bound": False,
            "mailbox": status["mailbox_offsets"],
            "mailbox_address": status["mailbox_address"],
            "files": {},
        }
        self._write_package()
        self.cases_path = self.root / "cases.json"
        self.receipt_path = self.root / "receipt.json"
        self.cases = []
        self.receipt = {
            "schema": "atlas.external_mailbox_dump.v1",
            "run_id": "public-challenge-001",
            "bitstream_sha256": sha(b"reported bitstream fixture"),
            "elf_sha256": sha(self.elf),
            "cases": [],
        }
        for index in range(2):
            request = index + 1
            source = bytes([request]) * 4096
            expected = bytes([request + 2]) * 4096
            dump = bytearray(8320)
            struct.pack_into("<I", dump, 0, request)
            struct.pack_into("<I", dump, 64, request)
            struct.pack_into("<I", dump, 68, 1)
            dump[128:4224] = source
            dump[4224:] = expected
            (self.root / f"input-{index}.bin").write_bytes(source)
            (self.root / f"expected-{index}.bin").write_bytes(expected)
            (self.root / f"dump-{index}.bin").write_bytes(dump)
            self.cases.append({"request_id": request, "input": f"input-{index}.bin",
                               "expected": f"expected-{index}.bin", "dump": f"dump-{index}.bin"})
            self.receipt["cases"].append({"request_id": request,
                                          "input_sha256": sha(source),
                                          "dump_sha256": sha(dump)})
        self._write_cases()

    def _write_package(self) -> None:
        self.package["files"] = {
            str(path.relative_to(self.handoff)):
                {"bytes": path.stat().st_size, "sha256": sha(path.read_bytes())}
            for path in self.handoff.rglob("*") if path.is_file() and path.name != "manifest.json"
        }
        (self.handoff / "manifest.json").write_text(json.dumps(self.package))

    def _write_cases(self) -> None:
        self.cases_path.write_text(json.dumps({"schema": "atlas.mailbox_evaluation_cases.v1",
                                               "cases": self.cases}))
        self.receipt_path.write_text(json.dumps(self.receipt))

    def _check(self) -> dict:
        return check(self.handoff, self.receipt_path, self.cases_path)

    def test_two_distinct_runtime_challenges_match(self) -> None:
        result = self._check()
        self.assertEqual(result["status"], "reported_match_unqualified")
        self.assertEqual((result["passing_cases"], result["total_cases"]), (2, 2))
        self.assertEqual([row["matching_output_bytes"] for row in result["cases"]], [4096, 4096])

    def test_replayed_dump_for_new_input_refuses_even_with_updated_hash(self) -> None:
        replay = self.root / "dump-0.bin"
        (self.root / "dump-1.bin").write_bytes(replay.read_bytes())
        self.receipt["cases"][1]["dump_sha256"] = sha(replay.read_bytes())
        self._write_cases()
        with self.assertRaisesRegex(ValueError, "evaluator challenge"):
            self._check()

    def test_wrong_output_and_device_failure_are_separate(self) -> None:
        damaged = bytearray((self.root / "dump-1.bin").read_bytes())
        damaged[-1] ^= 1
        (self.root / "dump-1.bin").write_bytes(damaged)
        self.receipt["cases"][1]["dump_sha256"] = sha(damaged)
        self._write_cases()
        self.assertEqual(self._check()["status"], "numerical_mismatch")
        struct.pack_into("<I", damaged, 68, 0x40000201)
        (self.root / "dump-1.bin").write_bytes(damaged)
        self.receipt["cases"][1]["dump_sha256"] = sha(damaged)
        self._write_cases()
        self.assertEqual(self._check()["status"], "execution_failure")

    def test_missing_identity_changed_package_and_stale_response_refuse(self) -> None:
        self.receipt["bitstream_sha256"] = "missing"
        self._write_cases()
        with self.assertRaisesRegex(ValueError, "receipt"):
            self._check()
        self.receipt["bitstream_sha256"] = sha(b"reported bitstream fixture")
        self._write_cases()
        (self.handoff / "host/full-graph-mailbox.elf").write_bytes(b"changed")
        with self.assertRaisesRegex(ValueError, "member changed"):
            self._check()
        (self.handoff / "host/full-graph-mailbox.elf").write_bytes(self.elf)
        damaged = bytearray((self.root / "dump-1.bin").read_bytes())
        struct.pack_into("<I", damaged, 64, 1)
        (self.root / "dump-1.bin").write_bytes(damaged)
        self.receipt["cases"][1]["dump_sha256"] = sha(damaged)
        self._write_cases()
        with self.assertRaisesRegex(ValueError, "stale or incomplete"):
            self._check()

    def test_distinct_challenges_and_fresh_report_required(self) -> None:
        (self.root / "input-1.bin").write_bytes((self.root / "input-0.bin").read_bytes())
        self.receipt["cases"][1]["input_sha256"] = self.receipt["cases"][0]["input_sha256"]
        changed = bytearray((self.root / "dump-1.bin").read_bytes())
        changed[128:4224] = (self.root / "input-0.bin").read_bytes()
        (self.root / "dump-1.bin").write_bytes(changed)
        self.receipt["cases"][1]["dump_sha256"] = sha(changed)
        self._write_cases()
        with self.assertRaisesRegex(ValueError, "distinct runtime inputs"):
            self._check()
        (self.root / "input-1.bin").write_bytes(bytes([2]) * 4096)
        self.receipt["cases"][1]["input_sha256"] = sha(bytes([2]) * 4096)
        changed[128:4224] = bytes([2]) * 4096
        (self.root / "dump-1.bin").write_bytes(changed)
        self.receipt["cases"][1]["dump_sha256"] = sha(changed)
        self._write_cases()
        report = self.root / "report.json"
        argv = ["--handoff", str(self.handoff), "--receipt", str(self.receipt_path),
                "--cases", str(self.cases_path), "--out", str(report)]
        self.assertEqual(main(argv), 0)
        self.assertEqual(main(argv), 2)


if __name__ == "__main__":
    unittest.main()
