"""Pair recipe (tools/rtl_extract/pairs.py): synthetic sweeps, pair-file validation, and Atlas hardware regressions.

Hardware tests set ATLAS_HW_IR and ATLAS_HW_EXPORTER and check a narrow window around each pair file's expected
``stable_gap`` (the gap below it rejected, it and the next accepted); full sweeps run through
``python3 -m rtl_extract.pairs``.
"""
import copy
import os
from pathlib import Path
import sys
import tempfile
import unittest

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import test_rtl_extract_synthetic as synthetic
from rtl_extract import pairs
from rtl_extract.ir import load_document
from rtl_extract.runner import modules

HW_IR = os.environ.get("ATLAS_HW_IR")
EXPORTER = os.environ.get("ATLAS_HW_EXPORTER")
op = synthetic.operation


def counter(accept_while_busy=False):
    """``go`` loads a 2-bit counter with 3; ``out``/``busy`` are high while it counts down. A busy engine ignores ``go``
    unless ``accept_while_busy``, in which case a second ``go`` restarts the count."""
    return synthetic.module("Toy", ["go"], [("out", "busy"), ("busy", "busy")], [
        op("zero", "hw.constant", [], "i2", value="0 : i2"),
        op("one", "hw.constant", [], "i2", value="1 : i2"),
        op("three", "hw.constant", [], "i2", value="3 : i2"),
        op("cnt", "seq.firreg", ["next", "clock", "reset", "zero"], "i2", name="cnt"),
        op("idle", "comb.icmp", ["cnt", "zero"], predicate="0 : i64"),
        op("busy", "comb.icmp", ["cnt", "zero"], predicate="1 : i64"),
        op("dec", "comb.sub", ["cnt", "one"], "i2"),
        op("start", "comb.and", ["go", "idle"]) if not accept_while_busy else op("start", "hw.wire", ["go"]),
        op("hold", "comb.mux", ["busy", "dec", "zero"], "i2"),
        op("next", "comb.mux", ["start", "three", "hold"], "i2")])


SPEC = {"engine": "toy", "module": "Toy", "inputs": {"go": 0}, "busy": "busy", "limit": 24,
        "events": {"out": {"valid": "out"}},
        "operations": {"go": {"commands": {0: {"go": 1}}, "next_issue": {"signal": "busy"}}}}


def sweep(module, pair, spec=SPEC, extra=None):
    spec_ = synthetic.load(spec)
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "toy.yaml"
        path.write_text(yaml.safe_dump({"engine": "toy", "pairs": {"p": pair}, **(extra or {})}))
        raw = pairs.load_file(path, spec_)
    return pairs.extract({"modules": [module]}, raw, spec_)[0]


class SyntheticPairTests(unittest.TestCase):
    def test_ignored_launch_is_rejected_until_busy_clears(self):
        record = sweep(counter(), {"first": "go", "second": "go", "gaps": "1..6"})
        self.assertEqual((record["min_gap"], record["stable_gap"]), (4, 4))
        self.assertEqual(record["results"][0], {"from": 1, "to": 3, "result": "guard busy set; missing out events"})

    def test_event_check_alone_rejects_an_ignored_launch(self):
        record = sweep(counter(), {"first": "go", "second": "go", "gaps": "1..6", "guards": []})
        self.assertEqual(record["stable_gap"], 4)
        self.assertEqual(record["results"][0]["result"], "missing out events")

    def test_restart_merges_events_until_streams_abut(self):
        # A restart at gap 1 or 2 overlaps the first stream; at 3 the streams abut and match the isolated runs.
        record = sweep(counter(True), {"first": "go", "second": "go", "gaps": "1..6", "guards": []})
        self.assertEqual((record["min_gap"], record["stable_gap"]), (3, 3))
        guarded = sweep(counter(True), {"first": "go", "second": "go", "gaps": "1..6"})
        self.assertEqual(guarded["results"][1], {"from": 3, "to": 3, "result": "guard busy set; events unchanged"})
        self.assertEqual(guarded["stable_gap"], 4)

    def test_equivalent_groups_compare_as_one_resource(self):
        # A command served on the other port of an interchangeable pair matches its isolated run on the first port.
        alone = [{"age": 3, "events": {"a": {"row": 0}}}]
        paired = [{"age": 5, "events": {"b": {"row": 0}}}]
        self.assertNotEqual(pairs.events(alone, {}, shift=2), pairs.events(paired, {}))
        same = {"port": ["a", "b"]}
        self.assertEqual(pairs.events(alone, same, shift=2), pairs.events(paired, same))

    def test_pair_files_fail_closed(self):
        spec = synthetic.load(SPEC)
        for label, raw, message in (
                ("unknown operation", {"engine": "toy", "pairs": {"p": {"first": "go", "second": "stop", "gaps": "1..2"}}}, "unknown operation"),
                ("undeclared input", {"engine": "toy", "pairs": {"p": {"first": "go", "second": "go", "gaps": "1..2", "second_inputs": {"x": 1}}}}, "undeclared inputs"),
                ("gap zero", {"engine": "toy", "pairs": {"p": {"first": "go", "second": "go", "gaps": "0..2"}}}, "gaps start at 1"),
                ("unknown guard", {"engine": "toy", "pairs": {"p": {"first": "go", "second": "go", "gaps": "1..2", "guards": [{"signal": "nope"}]}}}, "not an observed"),
                ("unknown key", {"engine": "toy", "pairs": {"p": {"first": "go", "second": "go", "gaps": "1..2", "extra": 1}}}, "unknown keys"),
                ("wrong engine", {"engine": "other", "pairs": {"p": {"first": "go", "second": "go", "gaps": "1..2"}}}, "does not match"),
                ("bad equivalent", {"engine": "toy", "equivalent": {"x": ["missing"]}, "pairs": {"p": {"first": "go", "second": "go", "gaps": "1..2"}}}, "equivalent")):
            with self.subTest(label), self.assertRaisesRegex(ValueError, message):
                pairs.validate(copy.deepcopy(raw), spec, "toy.yaml")

    def test_target_pair_files_validate(self):
        loaded = pairs.load("atlas")
        self.assertTrue(loaded)
        for name, (raw, spec) in loaded.items():
            self.assertEqual(set(raw.get("expect", {})), set(raw["pairs"]), name)


@unittest.skipUnless(HW_IR and EXPORTER, "requires ATLAS_HW_IR and ATLAS_HW_EXPORTER")
class HardwarePairTests(unittest.TestCase):
    def check(self, name):
        raw, spec = pairs.load("atlas", [name])[name]
        document = load_document(HW_IR, modules(spec), EXPORTER)
        window = copy.deepcopy(raw)
        for pair, expected in raw["expect"].items():
            stable = expected["stable_gap"]
            window["pairs"][pair]["gaps"] = f"{max(1, stable - 1)}..{stable + 1}"
        for record in pairs.extract(document, window, spec):
            pair = record["name"].split(".", 1)[1]
            stable = raw["expect"][pair]["stable_gap"]
            with self.subTest(record["name"]):
                self.assertEqual(record["stable_gap"], stable, record["results"])
                if stable > 1:
                    self.assertEqual(record["min_gap"], stable, record["results"])

    def test_vlsu(self):
        self.check("vlsu")

    def test_vpu(self):
        self.check("vpu")

    def test_mxu0(self):
        self.check("mxu0")

    def test_mxu1(self):
        self.check("mxu1")


if __name__ == "__main__":
    unittest.main()
