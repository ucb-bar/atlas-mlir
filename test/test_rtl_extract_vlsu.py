"""Vector LSU regressions: computed VLOAD/VSTORE timing against ``expect:``, operand independence, a mutation.

Set ATLAS_HW_IR (CIRCT HW IR, preferably the -O=debug lowering) and ATLAS_HW_EXPORTER
(hw_ir_export built from tools/rtl_extract/export); the tests skip otherwise.
"""
import copy
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import test_rtl_extract_atlas as atlas

TIMING = ("first_age", "last_age", "count", "step")


class VlsuTest(atlas.EngineCase):
    engine = "vlsu"

    def timing(self, record):
        return ({g: {k: e[k] for k in TIMING} for g, e in record["events"].items()},
                record["first_free_age"], record["next_issue_age"])

    def test_operands_do_not_move_timing(self):
        for op in self.spec["operations"]:
            with self.subTest(op):
                self.assertEqual(self.timing(self.records[f"{op}/operands"]), self.timing(self.records[op]))

    def delay_output(self, document, port_name):
        """Insert one extra register stage on a module output port."""
        module = self.module(document, self.spec["module"])
        ports = {p["name"]: p for p in module["ports"]}
        port = ports[port_name]
        result = f"mutation.{port_name}"
        module["operations"].append({"id": result, "kind": "seq.firreg", "operands": [port["value"], ports["clock"]["value"]],
                                     "results": [result], "result_types": ["i1"], "attributes": {}, "has_regions": False})
        port["value"] = result

    def test_extra_write_valid_stage_delays_writes(self):
        for op, group in (("vload", "mreg_write"), ("vstore", "vmem_write")):
            with self.subTest(op):
                document = copy.deepcopy(self.document)
                self.delay_output(document, self.spec["events"][group]["valid"])
                mutated, base = self.extract(document)[op], self.records[op]
                self.assertEqual(mutated["status"], "computed", mutated.get("reason"))
                for key in ("first_age", "last_age"):
                    self.assertEqual(mutated["events"][group][key], base["events"][group][key] + 1)
                self.assertEqual(mutated["events"][group]["count"], base["events"][group]["count"])
                others = {g: e for g, e in base["events"].items() if g != group}
                self.assertEqual({g: mutated["events"][g] for g in others}, others)


if __name__ == "__main__":
    unittest.main()
