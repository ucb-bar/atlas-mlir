"""Check graph mailbox receipts, including stale and corrupted responses."""

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
from check_graph_mailbox_dump import check


def _write(path: Path, payload: bytes) -> Path:
    path.write_bytes(payload)
    return path


def _sha(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


class GraphMailboxCheckerTest(unittest.TestCase):
    def setUp(self) -> None:
        temporary_root = Path(os.environ.get("TMPDIR", ROOT / "out"))
        temporary_root.mkdir(parents=True, exist_ok=True)
        self.temporary = tempfile.TemporaryDirectory(dir=temporary_root)
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)

    def _case(self, outputs: int) -> dict[str, Path | int]:
        elf = _write(self.root / "host.elf", b"checked host elf")
        program = _write(self.root / "program.bin", b"\x00\x00\x00\x00")
        source = _write(self.root / "inputs.bin", bytes(range(256)) * 96)
        expected = _write(self.root / "expected.bin", bytes(outputs * 2048))
        after = 24596 + outputs * 4096
        payload = bytearray(after + 4)
        struct.pack_into("<III", payload, 0, 7, 7, 0)
        struct.pack_into("<I", payload, 16, 0x76A31C5B)
        struct.pack_into("<I", payload, after, 0x76A31C5B)
        payload[20:24596] = source.read_bytes()
        dump = _write(self.root / "dump.bin", payload)
        status = {
            "schema": "atlas.diagnostic_graph_mailbox_handoff.v1",
            "status": "diagnostic_unbound", "engine": "merlin_native",
            "bitstream_bound": False,
            "elf_sha256": _sha(elf.read_bytes()),
            "program_sha256": _sha(program.read_bytes()),
            "input_count": 6, "output_count": outputs,
            "mailbox_offsets": {
                "request_id": 0, "response_id": 4, "status": 8,
                "guard_before": 16, "inputs": 20, "outputs": 24596,
                "guard_after": after,
            },
            "mailbox_bytes": after + 4,
        }
        plan = {
            "shape": [32, 32], "dtype": "bf16",
            "output_physical_layouts": ["bf16_pair_halves"] * outputs,
            "output_bytes": 2048,
        }
        handoff_status = _write(self.root / "status.json", json.dumps(status).encode())
        execution_plan = _write(self.root / "plan.json", json.dumps(plan).encode())
        return {
            "handoff_status": handoff_status, "execution_plan": execution_plan,
            "elf": elf, "program": program, "dump": dump,
            "inputs": source, "expected": expected, "request_id": 7,
        }

    def test_two_and_three_ordered_outputs(self) -> None:
        for outputs in (2, 3):
            with self.subTest(outputs=outputs):
                paths = self._case(outputs)
                result = check(**paths)
                self.assertEqual(result["status"], "reported_match_unqualified")
                self.assertEqual(result["matching_output_bytes"], outputs * 4096)

    def test_pair_half_outputs_decode_in_row_major_order(self) -> None:
        paths = self._case(2)
        physical = bytearray(2 * 2048)
        dump = bytearray(paths["dump"].read_bytes())
        for output in range(2):
            for row in range(32):
                for column in range(32):
                    # Distinct BF16 bits per cell expose any transposition or
                    # half-swap between the device layout and the f32 window.
                    bits = 0x3F80 + output * 0x400 + row * 32 + column
                    half, lane = divmod(column, 16)
                    offset = output * 2048 + half * 1024 + row * 32 + lane * 2
                    physical[offset:offset + 2] = bits.to_bytes(2, "little")
                    struct.pack_into("<I", dump,
                                     24596 + output * 4096 + (row * 32 + column) * 4,
                                     bits << 16)
        paths["expected"].write_bytes(physical)
        paths["dump"].write_bytes(dump)
        result = check(**paths)
        self.assertEqual(result["status"], "reported_match_unqualified")
        self.assertEqual(result["expected_sha256"], _sha(bytes(physical)))
        self.assertEqual(result["inputs_sha256"], _sha(paths["inputs"].read_bytes()))
        self.assertEqual(result["execution_plan_sha256"],
                         _sha(paths["execution_plan"].read_bytes()))
        dump[24596:24600], dump[24600:24604] = dump[24600:24604], dump[24596:24600]
        paths["dump"].write_bytes(dump)
        self.assertEqual(check(**paths)["status"], "execution_failure")

    def test_platform_checked_handoff_keeps_reported_bytes_unqualified(self) -> None:
        paths = self._case(3)
        status_path = paths["handoff_status"]
        status = json.loads(status_path.read_text())
        status.update({
            "schema": "atlas.graph_mailbox_handoff.v2",
            "status": "platform_map_checked_execution_unqualified",
            "bitstream_bound": True,
            "bitstream_identity": "synthetic-test-build",
            "cache_policy": "callback_supplied_unqualified",
        })
        status_path.write_text(json.dumps(status))
        self.assertEqual(check(**paths)["status"], "reported_match_unqualified")
        status["bitstream_identity"] = ""
        status_path.write_text(json.dumps(status))
        with self.assertRaisesRegex(ValueError, "graph handoff"):
            check(**paths)

    def test_stale_response_corruption_and_guard_are_detected(self) -> None:
        for offset, value in ((4, 6), (24596, 1), (16, 0), (20, 255)):
            with self.subTest(offset=offset):
                paths = self._case(3)
                dump = bytearray(paths["dump"].read_bytes())
                dump[offset] = value
                paths["dump"].write_bytes(dump)
                self.assertEqual(check(**paths)["status"], "execution_failure")

    def test_modified_program_or_layout_is_refused(self) -> None:
        paths = self._case(2)
        paths["program"].write_bytes(b"\x01\x00\x00\x00")
        with self.assertRaisesRegex(ValueError, "handoff, program"):
            check(**paths)
        paths = self._case(2)
        plan = json.loads(paths["execution_plan"].read_text())
        plan["output_physical_layouts"][1] = "row_major"
        paths["execution_plan"].write_text(json.dumps(plan))
        with self.assertRaisesRegex(ValueError, "physical output layout"):
            check(**paths)

    @unittest.skipUnless(os.environ.get("ATLAS_GRAPH_HANDOFF"),
                         "select the local two-output handoff archive")
    def test_reported_spike_model_mailboxes(self) -> None:
        handoff = Path(os.environ["ATLAS_GRAPH_HANDOFF"])
        for request_id, name in ((1, "checkerboard"), (2, "ternary")):
            with self.subTest(request_id=request_id):
                result = check(
                    handoff_status=handoff / "host/handoff-status.json",
                    execution_plan=handoff / "device/execution_plan.json",
                    elf=handoff / "host/shuttle-graph-host.elf",
                    program=handoff / "device/program.bin",
                    dump=handoff / f"model-spike/mailbox-{request_id}.bin",
                    inputs=handoff / f"model-spike/source-{request_id - 1}.bin",
                    expected=handoff / f"fixtures/{name}/expected.bin",
                    request_id=request_id,
                )
                self.assertEqual(result["status"], "reported_match_unqualified")


if __name__ == "__main__":
    unittest.main()
