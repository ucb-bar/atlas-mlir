"""Source-only frontend derivation and the explicit SRAM response limitation."""
from pathlib import Path
import os
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import atlas_derive_scratchpad as scratchpad
import atlas_model_timing as timing
from model_evidence import ModelSourceFixture


class ResponseBoundaryTest(unittest.TestCase):
    def test_validator_does_not_treat_assumed_response_latency_as_observed(self):
        block = dict(name="probe", engine="probe", events={}, first_free_age=2,
                     assumptions={"scratchpad_read_latency": 1})
        candidate = dict(block, assumptions={"scratchpad_read_latency": 99})
        raw = {key: [] for key in ("architectural_states", "control_states", "pre_edges")}
        raw.update(initial_state_sha256="a", final_state_sha256="b", first_free_age=2,
                   complete_count=1, total_instructions=1, busy_cycles=2, instruction_word=0,
                   configuration={}, reservations=[])
        with tempfile.TemporaryDirectory() as directory, \
                patch.object(timing, "isa_cases", return_value=[("probe", "probe", "nop")]), \
                patch.object(timing, "observe_case", return_value=(block, raw)), \
                patch.object(timing, "observed_releases", return_value={}):
            report, _ = timing.validate_derived(object(), {"probe": candidate}, {"probe": {}}, Path(directory))
        self.assertTrue(report["passed"])
        self.assertEqual(report["assumption_fields_not_observed"], ["scratchpad_read_latency"])
        self.assertTrue(all(item["observable_fields_byte_identical"] for item in report["checks"]))
        self.assertTrue(all("byte_identical" not in item for item in report["checks"]))



@unittest.skipUnless(os.environ.get("ATLAS_DERIVE_MODEL_ROOT"), "set ATLAS_DERIVE_MODEL_ROOT for native source contracts")
class SourceContractTest(ModelSourceFixture, unittest.TestCase):
    source_shapes = scratchpad._SHAPES

    def test_response_edge_is_explicitly_not_derived(self):
        value = scratchpad.derive_scratchpad_sources(self.sources)
        self.assertEqual(scratchpad.derive_scratchpad(self.cfg), value)
        self.assertFalse(value["response_latency_derived"])
        self.assertEqual(value["status"], "explicit-target-assumption")
        self.assertEqual(value["dma_store_read_to_request_cycles"], 2)
        self.assertIn("No model-owned executable field separates SRAM response latency", value["limitation"])

    def test_dma_combined_store_staging_change_propagates_without_inventing_response_latency(self):
        source = self.mutated("dma.py", "_StoreBeat(head, self._vmem_read_beat, cycle + 2)",
                              "_StoreBeat(head, self._vmem_read_beat, cycle + 3)")
        expected = scratchpad.derive_scratchpad_sources(self.sources)
        expected["dma_store_read_to_request_cycles"] = 3
        self.assertEqual(scratchpad.derive_scratchpad_sources(source), expected)

    def test_response_pipeline_mutations_fail_at_explicit_boundary(self):
        cases = [
            ("vpu.py", "op.staged.sample((\"mrf\", bank), row, self._read_cache[key])", "op.pending_response = (bank, row)"),
            ("mxu.py", "self._sample(op, age)", "self._sample(op, age - 1)"),
            ("lsu.py", "age == 3", "age == 4"),
            ("xlu.py", "age - 34", "age - 35"),
            ("dma.py", "self._store_queue[0].available <= cycle", "self._store_queue[0].available < cycle"),
        ]
        for filename, old, new in cases:
            with self.subTest(filename=filename), self.assertRaisesRegex(scratchpad.UnsupportedScratchpad, "no separate SRAM response edge"):
                scratchpad.derive_scratchpad_sources(self.mutated(filename, old, new))

if __name__ == "__main__":
    unittest.main()
