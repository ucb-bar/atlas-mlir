"""Atlas hardware regressions: computed timing against each spec's ``expect:`` block, plus netlist mutations.

Set ATLAS_HW_IR (CIRCT HW IR, preferably the -O=debug lowering) and ATLAS_HW_EXPORTER
(hw_ir_export built from tools/rtl_extract/export); the tests skip otherwise.
"""
import copy
import os
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from rtl_extract import derive
from rtl_extract.ir import load_document
from rtl_extract.runner import extract, modules
from rtl_extract.spec import load_target

HW_IR = os.environ.get("ATLAS_HW_IR")
EXPORTER = os.environ.get("ATLAS_HW_EXPORTER")
TIMING = ("first_age", "last_age", "count", "step")


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


def find(document, name):
    return next(m for m in document["modules"] if m["name"] == name)


def add_stage(module, port):
    """Insert one register between ``port``'s driver and the port."""
    ports = {p["name"]: p for p in module["ports"]}
    result = f"mutation.{port}"
    module["operations"].append({"id": result, "kind": "seq.firreg", "operands": [ports[port]["value"], ports["clock"]["value"]],
                                 "results": [result], "result_types": ["i1"], "attributes": {}, "has_regions": False})
    ports[port]["value"] = result


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
    def extract(cls, document, spec=None):
        return {r["name"].split(".", 1)[1]: r for r in extract(document, spec or cls.spec)}

    def mutated(self, module, port):
        document = copy.deepcopy(self.document)
        add_stage(find(document, module), port)
        return document

    def assert_delayed(self, mutated, base, group, others=()):
        """``group`` starts and ends exactly one age later with the same count; ``others`` are unchanged."""
        self.assertEqual(mutated["status"], "computed", mutated.get("reason"))
        for key in ("first_age", "last_age"):
            self.assertEqual(mutated["events"][group][key], base["events"][group][key] + 1)
        self.assertEqual(mutated["events"][group]["count"], base["events"][group]["count"])
        for other in others:
            self.assertEqual(mutated["events"][other], base["events"][other])

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
        ports = {p["name"]: p for p in find(document, self.spec["module"])["ports"]}
        ports[self.spec["busy"]]["value"] = ports["io_mregReadResp_bits"]["value"]
        self.assertTrue(all(r["status"] == "unresolved" and "unapproved input" in r["reason"] for r in self.extract(document).values()))

    def test_extra_write_valid_stage_delays_writes(self):
        mutated = self.extract(self.mutated(self.spec["module"], self.spec["events"]["write"]["valid"]))["transpose"]
        self.assert_delayed(mutated, self.records["transpose"], "write", ["read"])


class Mxu0Test(EngineCase):
    engine = "mxu0"


class Mxu1Test(EngineCase):
    engine = "mxu1"

    def test_extra_tree_output_stage_delays_writes_by_one(self):
        mutated = self.extract(self.mutated("InnerProductTrees", "io_out_valid"))
        for op in ("matmul", "matmul_acc"):
            with self.subTest(op):
                self.assertEqual(mutated[op]["status"], "computed", mutated[op].get("reason"))
                self.assertEqual(mutated[op]["events"]["acc_write"]["first_age"],
                                 self.records[op]["events"]["acc_write"]["first_age"] + 1)
                self.assertEqual(mutated[op]["events"]["mreg_read0"], self.records[op]["events"]["mreg_read0"])


class ScalarLsuTest(EngineCase):
    engine = "scalar_lsu"

    def test_extra_response_stage_in_partner_delays_writeback(self):
        mutated = self.extract(self.mutated(self.spec["partner"]["module"], "io_scalarResp_valid"))["load"]
        base = self.records["load"]
        self.assertEqual(mutated["status"], "computed", mutated.get("reason"))
        self.assertEqual(mutated["events"]["load_write"]["first_age"], base["events"]["load_write"]["first_age"] + 1)
        self.assertEqual(mutated["events"]["vmem_read"], base["events"]["vmem_read"])


class VlsuTest(EngineCase):
    engine = "vlsu"

    @staticmethod
    def timing(record):
        return ({g: {k: e[k] for k in TIMING} for g, e in record["events"].items()},
                record["first_free_age"], record["next_issue_age"])

    def test_operands_do_not_move_timing(self):
        for op in self.spec["operations"]:
            with self.subTest(op):
                self.assertEqual(self.timing(self.records[f"{op}/operands"]), self.timing(self.records[op]))

    def test_extra_write_valid_stage_delays_writes(self):
        for op, group in (("vload", "mreg_write"), ("vstore", "vmem_write")):
            with self.subTest(op):
                mutated = self.extract(self.mutated(self.spec["module"], self.spec["events"][group]["valid"]))[op]
                base = self.records[op]
                self.assert_delayed(mutated, base, group, [g for g in base["events"] if g != group])


class VpuTest(EngineCase):
    engine = "vpu"

    def test_all_operations_present(self):
        self.assertEqual(len(self.records), 29)
        self.assertNotIn("fp8", self.records)

    def test_busy_release_not_before_next_issue(self):
        for name, record in self.records.items():
            with self.subTest(name):
                self.assertGreaterEqual(record["first_free_age"], record["next_issue_age"])

    def bypassed(self):
        """Relu's valid register bypassed: its response valid follows the request directly."""
        document = copy.deepcopy(self.document)
        ports = {p["name"]: p for p in find(document, "Relu")["ports"]}
        ports["io_resp_valid"]["value"] = ports["io_req_valid"]["value"]
        return document

    def test_bypassed_valid_stage_advances_writes(self):
        mutated = self.extract(self.bypassed(), {**self.spec, "checks": []})["relu"]  # the family check would reject the mutant
        base = self.records["relu"]
        self.assertEqual(mutated["status"], "computed", mutated.get("reason"))
        for group in ("write0", "write1"):
            if base["events"][group]["count"]:
                self.assertLess(mutated["events"][group]["first_age"], base["events"][group]["first_age"])

    def test_bypassed_valid_stage_violates_family_check(self):
        mutated = self.extract(self.bypassed())["relu"]
        self.assertEqual(mutated["status"], "unresolved")
        self.assertIn("Check failed", mutated["reason"])


@unittest.skipUnless(HW_IR and EXPORTER, "requires ATLAS_HW_IR and ATLAS_HW_EXPORTER")
class ResponseLatencyTest(unittest.TestCase):
    def test_every_memory_response_is_one_cycle(self):
        specs = load_target("atlas")
        document = load_document(HW_IR, sorted({m for s in specs.values() for m in modules(s)}), EXPORTER)
        for name, spec in specs.items():
            resolved, notes = derive.resolve(document, spec)
            for group, event in resolved["events"].items():
                if "response" in event:
                    with self.subTest(f"{name}.{group}"):
                        note = notes["scratchpad_read_derivation"][group]
                        self.assertEqual((note["measured"], note["memory_read_latency"], event["response"]["latency"]), (1, 1, 1))


if __name__ == "__main__":
    unittest.main()
