"""Atlas hardware regressions: computed timing compared with each spec's ``expect:`` block.

Set ATLAS_HW_IR (CIRCT HW IR, preferably the -O=debug lowering) and ATLAS_HW_EXPORTER
(hw_ir_export built from tools/rtl_extract/export); the tests skip otherwise.
"""
import copy
import os
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from rtl_extract.ir import load_document
from rtl_extract.runner import extract, modules
from rtl_extract.spec import load_target

HW_IR = os.environ.get("ATLAS_HW_IR")
EXPORTER = os.environ.get("ATLAS_HW_EXPORTER")


def mismatches(expected, actual, path=""):
    if not isinstance(actual, dict):
        return [f"{path}: missing"]
    found = []
    for key, value in expected.items():
        if isinstance(value, dict):
            found += mismatches(value, actual.get(key), f"{path}.{key}" if path else key)
        elif actual.get(key) != value:
            found.append(f"{path}.{key}: expected {value!r}, computed {actual.get(key)!r}")
    return found


@unittest.skipUnless(HW_IR and EXPORTER, "requires ATLAS_HW_IR and ATLAS_HW_EXPORTER")
class EngineCase(unittest.TestCase):
    engine = None

    @classmethod
    def setUpClass(cls):
        if cls.engine is None:
            raise unittest.SkipTest("engine base class")
        cls.spec = load_target("atlas", [cls.engine])[cls.engine]
        cls.document = load_document(HW_IR, modules(cls.spec), EXPORTER)
        cls.records = cls.extract(cls.document)

    @classmethod
    def extract(cls, document):
        return {r["name"].split(".", 1)[1]: r for r in extract(document, cls.spec)}

    def module(self, document, name):
        return next(m for m in document["modules"] if m["name"] == name)

    def test_records_computed(self):
        for name, record in self.records.items():
            with self.subTest(name):
                self.assertEqual(record["status"], "computed", record.get("reason"))

    def test_regression_values_match_expect(self):
        self.assertTrue(self.spec["expect"])
        for name, expected in self.spec["expect"].items():
            with self.subTest(name):
                self.assertEqual(mismatches(expected, self.records[name]), [])


class XluTest(EngineCase):
    engine = "xlu"

    def test_payload_dependent_busy_rejected(self):
        document = copy.deepcopy(self.document)
        ports = {p["name"]: p for p in self.module(document, self.spec["module"])["ports"]}
        ports[self.spec["busy"]]["value"] = ports["io_mregReadResp_bits"]["value"]
        records = self.extract(document)
        self.assertTrue(all(r["status"] == "unresolved" and "unapproved input" in r["reason"] for r in records.values()))

    def test_extra_write_valid_stage_delays_writes(self):
        document = copy.deepcopy(self.document)
        module = self.module(document, self.spec["module"])
        ports = {p["name"]: p for p in module["ports"]}
        valid = ports[self.spec["events"]["write"]["valid"]]
        module["operations"].append({"id": "mutation", "kind": "seq.firreg", "operands": [valid["value"], ports["clock"]["value"]],
                                     "results": ["mutation.r0"], "result_types": ["i1"], "attributes": {}, "has_regions": False})
        valid["value"] = "mutation.r0"
        mutated = self.extract(document)["transpose"]
        base = self.records["transpose"]
        self.assertEqual(mutated["events"]["write"]["first_age"], base["events"]["write"]["first_age"] + 1)
        self.assertEqual(mutated["events"]["read"], base["events"]["read"])


if __name__ == "__main__":
    unittest.main()
