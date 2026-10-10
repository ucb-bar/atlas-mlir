"""Resolved exports must describe exactly the checked final instruction stream."""

import hashlib
import json
import os
from pathlib import Path
import re
import struct
import unittest

from test_delay_insertion import OPT, EMIT, ROOT, run, without_delays
from test_rtl_timing import FACTS, RESOLVER, block, facts, selection, sha256


@unittest.skipUnless(FACTS and OPT.is_file() and EMIT.is_file(),
                     "requires explicit RTL timing facts and Atlas tools")
class RTLTimingExportTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        source = (ROOT / "test/examples/ee290_vls_copy.mlir").read_text()
        cls.selected = {}
        for consumer in ("--insert-atlas-delays", "--schedule-atlas-stream"):
            result = run(OPT, source, selection(), consumer)
            if result.returncode:
                raise AssertionError(result.stderr)
            cls.selected[consumer] = result.stdout

    def export(self, source):
        return run(EMIT, source, "--rtl-timing-json")

    def test_shared_provider_export_matches_both_consumers_and_words(self):
        document = facts()
        load, store = block(document, "vlsu.vload"), block(document, "vlsu.vstore")
        for consumer, source in self.selected.items():
            with self.subTest(consumer=consumer):
                result = self.export(source)
                self.assertEqual(result.returncode, 0, result.stderr)
                exported = json.loads(result.stdout)
                emitted = run(EMIT, source)
                self.assertEqual(emitted.returncode, 0, emitted.stderr)
                words = [int(w, 16) for w in emitted.stdout.split()]
                self.assertEqual(exported["program"]["words"], words)
                self.assertEqual(exported["program"]["word_count"], len(words))
                self.assertEqual(exported["program"]["words_sha256"],
                                 hashlib.sha256(b"".join(struct.pack("<I", w)
                                                         for w in words)).hexdigest())
                self.assertEqual([i["word_u32"] for i in exported["instructions"]], words)
                vector = [i for i in exported["instructions"]
                          if i["mnemonic"] in ("vload", "vstore")]
                self.assertEqual([i["mnemonic"] for i in vector], ["vload", "vstore"])
                self.assertGreaterEqual(vector[1]["logical_issue_cycle"] -
                                        vector[0]["logical_issue_cycle"], load["first_free_age"])
                for entry in vector:
                    accesses = {a["resource"]: a for a in entry["footprint"]["accesses"]}
                    is_load = entry["mnemonic"] == "vload"
                    events = (load if is_load else store)["events"]
                    self.assertEqual(accesses["vmem"]["age"],
                                     events["vmem_read" if is_load else "vmem_write"]["first_age"])
                    self.assertEqual(accesses["mreg"]["age"],
                                     events["mreg_write" if is_load else "mreg_read"]["first_age"])
                    self.assertEqual(accesses["vmem"]["write"], not is_load)
                    self.assertEqual(accesses["mreg"]["write"], is_load)
                    self.assertEqual(accesses["vmem"]["count"], 32)
                    self.assertEqual(entry["footprint"]["done_age"], load["first_free_age"] - 1)
                self.assertEqual(exported["instructions"][-1]["event_kind"],
                                 "terminal_acceptance")

    def test_applicability_and_qualification_remain_explicit(self):
        result = self.export(next(iter(self.selected.values())))
        self.assertEqual(result.returncode, 0, result.stderr)
        value = json.loads(result.stdout)
        document = facts()
        self.assertEqual(value["schema"], "atlas.resolved_rtl_timing.v0")
        self.assertEqual(value["target_config"], "EE290SimConfig")
        self.assertEqual(value["qualification"], "conditional")
        self.assertIs(value["scheduling_qualified"], False)
        self.assertEqual(value["resolver"], {"id": RESOLVER, "version": 1, "dma_policy": "none"})
        self.assertEqual(value["evidence"], {"op_timing_schema": "merlin.op_timing.v1",
                                             "op_timing_sha256": sha256(FACTS),
                                             "hardware_ir_sha256": document["hw_ir"]["sha256"]})
        assumptions = value["applicability"]["environment_assumptions"]
        self.assertIn("other_engines_and_competing_memory_requesters_quiescent", assumptions)
        latency = block(document, "vlsu.vload")["assumptions"]["scratchpad_read_latency"]
        self.assertIn(f"sram_read_response_{latency}_cycles_after_request", assumptions)
        admission = value["applicability"]["admission"]
        self.assertEqual(admission["minimum_vls_issue_gap"], block(document, "vlsu.vload")["first_free_age"])
        self.assertEqual(admission["xlu_first_free_age"], block(document, "xlu.transpose")["first_free_age"])
        domain = value["applicability"]["operand_domain"]
        self.assertEqual(domain["vmem_banks"], 6)
        self.assertEqual(domain["mlir_offset_min"], -2048)
        self.assertEqual(domain["mlir_offset_max"], 2047)
        self.assertIn("vmul.bf16", value["applicability"]["supported_operations"])
        self.assertNotIn(str(ROOT), result.stdout)
        self.assertNotIn(str(Path(FACTS)), result.stdout)

    def test_missing_selection_cannot_export_model_fallback(self):
        source = (ROOT / "test/examples/ee290_vls_copy.mlir").read_text()
        result = self.export(source)
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(result.stdout, "")
        self.assertIn("requires selected evidence", result.stderr)

    def test_export_rechecks_actual_delays_and_emits_nothing_on_failure(self):
        for source in self.selected.values():
            result = self.export(without_delays(source))
            self.assertNotEqual(result.returncode, 0)
            self.assertEqual(result.stdout, "")
            self.assertIn("serialized engine admission", result.stderr)

    def test_export_rejects_claimed_qualification_and_changed_evidence(self):
        source = next(iter(self.selected.values()))
        for changed in (
                source.replace('atlas.rtl_qualification = "conditional"',
                               'atlas.rtl_qualification = "qualified"'),
                re.sub(r'op_timing_sha256 = "[0-9a-f]{64}"',
                       'op_timing_sha256 = "' + "0" * 64 + '"', source)):
            self.assertNotEqual(changed, source)
            result = self.export(changed)
            self.assertNotEqual(result.returncode, 0)
            self.assertEqual(result.stdout, "")


if __name__ == "__main__":
    unittest.main()
