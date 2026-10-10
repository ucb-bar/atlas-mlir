"""The op_timing document: Merlin named blocks, null for unknown, minimal identity (no hardware needed)."""
import copy
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from rtl_extract import facts  # noqa: E402

SPEC = {"engine": "toy", "module": "Toy", "reset_cycles": 4, "flush_cycles": 4, "limit": 32,
        "events": {"read": {"valid": "rd", "response": {"input": "rsp", "latency": 2}}},
        "operations": {"pulse": {"commands": {0: {"go": 1}}, "next_issue": {"signal": "ready"}}, "plain": {"commands": {}}}}
EVENTS = {"read": {"first_age": 1, "last_age": 3, "count": 3, "step": 1, "values": {}, "row_contiguous": True}}
RESULT = {"events": EVENTS, "first_free_age": 7, "next_issue_age": 5}


def blocks(*records):
    with tempfile.TemporaryDirectory() as work:
        ir = Path(work) / "toy.hw.mlir"
        ir.write_text("hw.module @Toy()\n")
        path = Path(work) / "out.json"
        facts.write(path, facts.document(ir, records))
        return json.loads(path.read_text()), hashlib.sha256(ir.read_bytes()).hexdigest(), str(ir.resolve())


class OpTimingDocument(unittest.TestCase):
    def test_document_identity_is_schema_and_ir_only(self):
        document, digest, path = blocks(facts.record(SPEC, "pulse", result=RESULT, evidence="simulated"))
        self.assertEqual(set(document), {"schema", "hw_ir", "op_timing"})
        self.assertEqual(document["schema"], "merlin.op_timing.v1")
        self.assertEqual(document["hw_ir"], {"path": path, "sha256": digest})

    def test_computed_block_follows_the_timing_conventions(self):
        block = blocks(facts.record(SPEC, "pulse", "slow", RESULT, evidence="simulated"))[0]["op_timing"][0]
        self.assertEqual((block["name"], block["module"], block["engine"], block["operation"], block["variant"]),
                         ("toy.pulse/slow", "Toy", "toy", "pulse", "slow"))
        self.assertEqual((block["source"], block["evidence"]), ("control_simulation", "simulated"))
        self.assertEqual((block["events"], block["first_free_age"], block["next_issue_age"]), (EVENTS, 7, 5))
        self.assertEqual(block["assumptions"], {"scratchpad_read_latency": 2, "reset_cycles": 4, "flush_cycles": 4, "limit": 32})
        self.assertNotIn("status", block)
        self.assertNotIn("reason", block)

    def test_variantless_engine_without_next_issue_omits_both_keys(self):
        spec = {**SPEC, "operations": {"plain": SPEC["operations"]["plain"]}}
        block = blocks(facts.record(spec, "plain", result={"events": EVENTS, "first_free_age": 2}))[0]["op_timing"][0]
        self.assertEqual(block["name"], "toy.plain")
        self.assertNotIn("variant", block)
        self.assertNotIn("next_issue_age", block)

    def test_unresolved_block_is_null_with_the_reason_as_evidence(self):
        block = blocks(facts.record(SPEC, "pulse", reason="Busy did not clear within 32 ages"))[0]["op_timing"][0]
        self.assertEqual((block["events"], block["first_free_age"], block["next_issue_age"]), (None, None, None))
        self.assertEqual(block["evidence"], "Busy did not clear within 32 ages")

    def test_failed_check_unresolves_the_block(self):
        record = facts.record(SPEC, "pulse", result=copy.deepcopy(RESULT), evidence="simulated")
        facts.unresolve(record, "Check failed: a == b")
        block = facts.block(record)
        self.assertEqual((block["events"], block["first_free_age"], block["next_issue_age"]), (None, None, None))
        self.assertEqual(block["evidence"], "Check failed: a == b")


if __name__ == "__main__":
    unittest.main()
