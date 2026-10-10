"""VPU hardware regressions: computed timing compared with tools/rtl_extract/targets/atlas/vpu.yaml ``expect:``.

Set ATLAS_HW_IR (CIRCT HW IR, preferably the -O=debug lowering) and ATLAS_HW_EXPORTER
(hw_ir_export built from tools/rtl_extract/export); the tests skip otherwise.
"""
import copy
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from rtl_extract.runner import extract
from test_rtl_extract_atlas import EngineCase


class VpuTest(EngineCase):
    engine = "vpu"
    # Instance-level stage under test: bypassing its valid register must advance the writes.
    stage_module, stage_in, stage_out = "Relu", "io_req_valid", "io_resp_valid"
    stage_operation = "relu"

    def test_all_operations_present(self):
        self.assertEqual(len(self.records), 29)
        self.assertNotIn("fp8", self.records)

    def test_expected_events_use_declared_groups(self):
        for name, expected in self.spec["expect"].items():
            with self.subTest(name):
                self.assertLessEqual(set(expected["events"]), set(self.spec["events"]))

    def test_busy_release_not_before_next_issue(self):
        for name, record in self.records.items():
            with self.subTest(name):
                self.assertGreaterEqual(record["first_free_age"], record["next_issue_age"])

    def bypassed(self):
        document = copy.deepcopy(self.document)
        module = self.module(document, self.stage_module)
        ports = {p["name"]: p for p in module["ports"]}
        ports[self.stage_out]["value"] = ports[self.stage_in]["value"]
        return document

    def test_bypassed_valid_stage_advances_writes(self):
        spec = {**self.spec, "checks": []}  # the family-equality checks would otherwise reject the mutant
        mutated = {r["name"].split(".", 1)[1]: r for r in extract(self.bypassed(), spec)}[self.stage_operation]
        base = self.records[self.stage_operation]
        self.assertEqual(mutated["status"], "computed", mutated.get("reason"))
        for group in ("write0", "write1"):
            if base["events"][group]["count"]:
                self.assertLess(mutated["events"][group]["first_age"], base["events"][group]["first_age"])

    def test_bypassed_valid_stage_violates_family_check(self):
        mutated = self.extract(self.bypassed())[self.stage_operation]
        self.assertEqual(mutated["status"], "unresolved")
        self.assertIn("Check failed", mutated["reason"])

if __name__ == "__main__":
    unittest.main()
