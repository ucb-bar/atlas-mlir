"""MXU generator coverage and byte regressions, independent of hardware execution."""
import contextlib
import importlib.util
import io
import json
import os
from pathlib import Path
import re
import tempfile
import unittest

import yaml

TARGET = Path(__file__).resolve().parents[1] / "tools/rtl_extract/targets/atlas"


def load_generator(name):
    spec = importlib.util.spec_from_file_location(f"atlas_gen_core_mxu_{name}",
                                                TARGET / f"gen_core_mxu_{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


SAME = load_generator("same")
SHARED = load_generator("shared")
CHAINS = load_generator("chains")


class MXUGeneratorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.texts = {name: (TARGET / path).read_bytes() for name, path in (
            ("same", "pairs/core_mxu_same.yaml"), ("shared", "pairs/core_mxu_shared.yaml"),
            ("chains", "sequences/core_mxu.yaml"))}
        cls.specs = {name: yaml.safe_load(text) for name, text in cls.texts.items()}

    def facts(self):
        # Recover only the event ages needed to reproduce the committed gap ranges.
        ages = {}
        for pair in self.specs["same"]["pairs"].values():
            ages[pair["as"]["first"]] = int(pair["gaps"].split("..")[1]) - 6
        return {"op_timing": [{"name": name, "engine": name.split(".")[0], "events": {
            "active": {"count": 1, "last_age": age},
            "inactive": {"count": 0, "last_age": 1000},
        }} for name, age in ages.items()]}

    def pair_sweep(self, name):
        records = []
        for pair, expected in self.specs[name]["expect"].items():
            stable = expected["stable_gap"]
            # rejected_by is a regex; restore its literal first failure message.
            failure = re.sub(r"\\(.)", r"\1", expected.get("rejected_by", "accepted"))
            records.append({"name": f"core.{pair}", "min_gap": expected["min_gap"],
                            "stable_gap": stable, "results": [
                                {"from": 1, "to": stable - 1, "result": failure + "; later failure"},
                                {"from": stable, "to": stable + 5, "result": "accepted"}]})
        return {"op_pairs": records}

    def chain_sweep(self):
        return {"op_sequences": [{"sequence": name, "length": length, "stable_gap": gap}
                for name, lengths in self.specs["chains"]["expect"].items()
                for length, gap in lengths.items()]}

    def test_committed_yaml_reproduces_byte_for_byte(self):
        for name, generator in (("same", SAME), ("shared", SHARED)):
            with self.subTest(name=name):
                self.assertEqual(generator.generate(self.facts(), self.pair_sweep(name)).encode(),
                                 self.texts[name])
        self.assertEqual(CHAINS.generate(self.chain_sweep()).encode(), self.texts["chains"])

    def test_without_sweeps_preserves_commands_and_coverage(self):
        for name, generator, count in (("same", SAME, 196), ("shared", SHARED, 536)):
            with self.subTest(name=name):
                generated = yaml.safe_load(generator.generate(self.facts()))
                self.assertEqual(len(generated["pairs"]), count)
                self.assertEqual(generated["pairs"], self.specs[name]["pairs"])
                self.assertNotIn("expect", generated)
        generated = yaml.safe_load(CHAINS.generate())
        self.assertEqual(len(generated["sequences"]), 50)
        self.assertEqual(generated["sequences"], self.specs["chains"]["sequences"])
        self.assertEqual(len(self.chain_sweep()["op_sequences"]), 172)
        self.assertNotIn("expect", generated)

    def test_gap_bound_uses_latest_active_event(self):
        for generator in (SAME, SHARED):
            facts = self.facts()
            facts["op_timing"][0]["events"]["active"]["last_age"] = 73
            generated = yaml.safe_load(generator.generate(facts))
            operation = facts["op_timing"][0]["name"]
            for name, pair in generated["pairs"].items():
                expected = "1..79" if pair["as"]["first"] == operation else self.specs[
                    "same" if generator is SAME else "shared"]["pairs"][name]["gaps"]
                self.assertEqual(pair["gaps"], expected)

    def test_operand_encodings_and_overlap_placements(self):
        # Explicit instruction constants cover scale, paired destinations, accumulator and slot fields.
        self.assertEqual(SAME.word("pop_acc_fp8", 1, 1, 1, 0), 0x0e082c77)
        self.assertEqual(SAME.word("matmul_acc", 0, 1, 1, 1), 0x180a00f7)
        self.assertEqual(SHARED.placements("push_acc_bf16", "pop_acc_bf16"),
                         {"0=0,1=1": (12, 12), "0~0,1~1": (12, 44)})
        self.assertEqual(SHARED.placements("push_weight", "pop_acc_bf16"),
                         {"0=0": (12, 12), "0~0": (12, 44),
                          "0=1": (13, 12), "0~1": (13, 44)})
        self.assertEqual(SHARED.pattern([12, 13], [44, 45]), "0~0,1~1")
        self.assertEqual(SHARED.pattern([12], [16]), "disjoint")
        self.assertEqual(CHAINS.word("pop_acc_bf16", 1, 5, 1), 0x12080b77)

    def test_missing_facts_and_incomplete_or_duplicate_sweeps_fail(self):
        for name, generator in (("same", SAME), ("shared", SHARED)):
            facts = self.facts()
            facts["op_timing"].pop()
            with self.subTest(name=name), self.assertRaisesRegex(ValueError, "14 MXU"):
                generator.generate(facts)
            for duplicate in (False, True):
                sweep = self.pair_sweep(name)
                if duplicate:
                    sweep["op_pairs"].append(sweep["op_pairs"][0])
                else:
                    sweep["op_pairs"].pop()
                with self.assertRaisesRegex(ValueError, "exactly every"):
                    generator.generate(self.facts(), sweep)
        for duplicate in (False, True):
            sweep = self.chain_sweep()
            if duplicate:
                sweep["op_sequences"].append(sweep["op_sequences"][0])
            else:
                sweep["op_sequences"].pop()
            with self.assertRaisesRegex(ValueError, "every MXU chain length"):
                CHAINS.generate(sweep)

    def test_cli_preserves_positional_output_and_supports_stdout(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            facts = directory / "facts.json"
            facts.write_text(json.dumps(self.facts()))
            for name, generator in (("same", SAME), ("shared", SHARED), ("chains", CHAINS)):
                sweep = directory / f"{name}.json"
                sweep.write_text(json.dumps(self.chain_sweep() if name == "chains" else self.pair_sweep(name)))
                output = directory / name / "output.yaml"
                arguments = [] if name == "chains" else [str(facts)]
                self.assertEqual(generator.main(arguments + [str(output), str(sweep)]), 0)
                self.assertEqual(output.read_bytes(), self.texts[name])
                stream = io.StringIO()
                with contextlib.redirect_stdout(stream):
                    self.assertEqual(generator.main(arguments), 0)
                generated = yaml.safe_load(stream.getvalue())
                self.assertNotIn("expect", generated)
                self.assertEqual(generator.main(arguments + ["--output", str(output)]), 0)
                self.assertEqual(output.read_text(), stream.getvalue())

    @unittest.skipUnless(os.environ.get("ATLAS_OP_TIMING"), "set ATLAS_OP_TIMING for hardware facts")
    def test_current_hardware_facts_reproduce_pair_commands(self):
        facts = json.loads(Path(os.environ["ATLAS_OP_TIMING"]).read_text())
        for name, generator in (("same", SAME), ("shared", SHARED)):
            with self.subTest(name=name):
                self.assertEqual(yaml.safe_load(generator.generate(facts))["pairs"],
                                 self.specs[name]["pairs"])


if __name__ == "__main__":
    unittest.main()
