"""Atlas MXU0, MXU1 and scalar LSU regressions against each spec's ``expect:`` block.

Set ATLAS_HW_IR (the -O=debug lowering) and ATLAS_HW_EXPORTER; the tests skip otherwise.
"""
import copy
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import test_rtl_extract_atlas as atlas


def add_stage(module, port):
    """Insert one register between ``port``'s driver and the port."""
    ports = {p["name"]: p for p in module["ports"]}
    module["operations"].append({"id": "mutation", "kind": "seq.firreg", "operands": [ports[port]["value"], ports["clock"]["value"]],
                                 "results": ["mutation.r0"], "result_types": ["i1"], "attributes": {}, "has_regions": False})
    ports[port]["value"] = "mutation.r0"


class Mxu0Test(atlas.EngineCase):
    engine = "mxu0"


class Mxu1Test(atlas.EngineCase):
    engine = "mxu1"

    def test_extra_tree_output_stage_delays_writes_by_one(self):
        document = copy.deepcopy(self.document)
        add_stage(self.module(document, "InnerProductTrees"), "io_out_valid")
        mutated = self.extract(document)
        for op in ("matmul", "matmul_acc"):
            with self.subTest(op):
                self.assertEqual(mutated[op]["status"], "computed", mutated[op].get("reason"))
                self.assertEqual(mutated[op]["events"]["acc_write"]["first_age"],
                                 self.records[op]["events"]["acc_write"]["first_age"] + 1)
                self.assertEqual(mutated[op]["events"]["mreg_read0"], self.records[op]["events"]["mreg_read0"])


class ScalarLsuTest(atlas.EngineCase):
    engine = "scalar_lsu"

    def test_extra_response_stage_in_partner_delays_writeback(self):
        document = copy.deepcopy(self.document)
        add_stage(self.module(document, self.spec["partner"]["module"]), "io_scalarResp_valid")
        mutated, base = self.extract(document)["load"], self.records["load"]
        self.assertEqual(mutated["status"], "computed", mutated.get("reason"))
        self.assertEqual(mutated["events"]["load_write"]["first_age"], base["events"]["load_write"]["first_age"] + 1)
        self.assertEqual(mutated["events"]["vmem_read"], base["events"]["vmem_read"])


if __name__ == "__main__":
    unittest.main()
