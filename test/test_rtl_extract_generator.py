"""Atlas pair generator regression, with optional current hardware facts."""
import importlib.util
import json
import os
from pathlib import Path
import unittest

import yaml

TARGET = Path(__file__).resolve().parents[1] / "tools/rtl_extract/targets/atlas"
MODULE = importlib.util.spec_from_file_location("atlas_gen_core_vpu", TARGET / "gen_core_vpu.py")
GENERATOR = importlib.util.module_from_spec(MODULE)
MODULE.loader.exec_module(GENERATOR)


class GeneratorTests(unittest.TestCase):
    def setUp(self):
        self.text = (TARGET / "pairs/core_vpu.yaml").read_text()
        self.spec = yaml.safe_load(self.text)
        words = yaml.safe_load((TARGET / "vpu.yaml").read_text())["decode"]["words"]
        self.facts = {"op_timing": [
            {"engine": "vpu", "operation": name,
             "next_issue_age": self.spec["expect"][f"{name}_{name}"]["next_issue_age"]}
            for name in words
        ]}
        self.sweep = {"op_pairs": [
            {"name": f"core.{name}", **regression}
            for name, regression in self.spec["expect"].items()
        ]}

    def test_committed_yaml_reproduces_byte_for_byte(self):
        self.assertEqual(GENERATOR.generate(self.facts, self.sweep), self.text)

    def test_without_sweep_has_all_pairs_and_no_regressions(self):
        spec = yaml.safe_load(GENERATOR.generate(self.facts))
        self.assertEqual(len(spec["pairs"]), 841)
        self.assertNotIn("expect", spec)
        self.assertEqual(spec["pairs"], self.spec["pairs"])

    def test_missing_facts_or_pair_fails(self):
        self.facts["op_timing"].pop()
        with self.assertRaisesRegex(ValueError, "29 VPU"):
            GENERATOR.generate(self.facts)
        self.setUp()
        self.sweep["op_pairs"].pop()
        with self.assertRaisesRegex(ValueError, "every ordered"):
            GENERATOR.generate(self.facts, self.sweep)

    @unittest.skipUnless(os.environ.get("ATLAS_OP_TIMING"), "set ATLAS_OP_TIMING for hardware facts")
    def test_current_hardware_facts_reproduce_pair_commands(self):
        facts = json.loads(Path(os.environ["ATLAS_OP_TIMING"]).read_text())
        spec = yaml.safe_load(GENERATOR.generate(facts))
        self.assertEqual(spec["pairs"], self.spec["pairs"])


if __name__ == "__main__":
    unittest.main()
