"""Segment admission rejects incomplete live state and incorrect device evidence."""
import copy
from pathlib import Path
import struct
import sys
from types import SimpleNamespace
import unittest
from unittest import mock

TOOLS = Path(__file__).absolute().parents[1]/"tools"
sys.path[:0] = [str(TOOLS), str(TOOLS/"host_iss")]
import smol_segment_contract as contract


class SmolSegmentTest(unittest.TestCase):
    def test_qemu_exit_race_is_reaped_and_timeout_escalates(self):
        import run_smol_segment as runner
        for race in (True, False):
            process = mock.Mock(pid=123)
            process.poll.side_effect = [None, 0]
            if not race:
                process.wait.side_effect = [runner.subprocess.TimeoutExpired("qemu", 10), 0]
            with mock.patch.object(runner.os, "killpg", side_effect=ProcessLookupError if race else None) as kill:
                self.assertTrue(runner.reap_qemu(process))
                self.assertEqual(kill.call_count, 1 if race else 2)
                self.assertEqual(process.wait.call_count, 1 if race else 2)

    def test_phase_partition_covers_all_calls_in_source_order(self):
        phases = contract.partitions()
        self.assertEqual([sum(p["identity"][0] == kind for p in phases) for kind in range(3)], [24, 16, 160])
        self.assertEqual([i for p in phases for i in range(p["call_start"], p["call_end"])], list(range(1421)))
        self.assertEqual(phases[0]["identity"], (0, 0, 0))
        self.assertEqual(phases[-1]["identity"], (2, 9, 15))

    def evidence(self):
        item = dict(program=struct.pack("<2I", 0x13, 0x73), expected_sha256="e"*64,
                    initial_sha256="a"*64, requested_dma=dict(load_bytes=128, store_bytes=64))
        summary = dict(index=39, sequence=40, reset_count=1, cycles=1234, exact=True, drained=True,
            program_sha256=contract.full.digest(item["program"][:-4]+contract.full.EPILOGUE),
            host_input_sha256="a"*64, actual_output_sha256="e"*64, expected_output_sha256="e"*64,
            observed_dma=dict(load_bytes=128, store_bytes=64, response_beats=6))
        return item, summary

    def test_actual_global_ledger_and_payload_binding(self):
        item, summary = self.evidence()
        contract.verify_kernel(summary, item, 39)
        for key, value in (("index", 0), ("sequence", 1), ("reset_count", 0), ("cycles", 0),
                           ("exact", False), ("drained", False), ("cycles", True),
                           ("host_input_sha256", "b"*64), ("actual_output_sha256", "c"*64),
                           ("program_sha256", "d"*64), ("observed_dma", {})):
            altered = copy.deepcopy(summary)
            altered[key] = value
            with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                contract.verify_kernel(altered, item, 39)

    def test_readonly_tables_require_file_backing(self):
        image = SimpleNamespace(segments=[dict(address=1024, memory_bytes=64, file_bytes=32, flags=4)])
        contract.span(image, 1024, 32, False)
        with self.assertRaises(ValueError):
            contract.span(image, 1024, 33, False)
        with self.assertRaises(ValueError):
            contract.span(image, 1024, 32, True)


if __name__ == "__main__":
    unittest.main()
