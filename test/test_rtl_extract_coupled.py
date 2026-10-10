"""Synthetic checks of the generic ``coupled`` recipe (two circuits, same-cycle links); no hardware IR is needed."""
import copy
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import test_rtl_extract_control as synthetic
from rtl_extract.runner import extract, modules
from rtl_extract.spec import load_spec

operation = synthetic.operation


def module(name, inputs, outputs, operations):
    p = synthetic.ports([("clock", "!seq.clock"), ("reset", "i1"), *[(n, "i1") for n in inputs]], outputs)
    return {"name": name, "ports": p, "operations": operations,
            "arguments": [{"id": q["value"], "type": q["type"]} for q in p if q["direction"] == "input"]}


def document(hold=False):
    """A registers ``cmd`` into ``req`` and passes ``ack`` straight to ``seen``; B echoes ``in`` and registers it."""
    zero, one = operation("zero", "hw.constant", [], value="false"), operation("one", "hw.constant", [], value="true")
    a = module("A", ["cmd", "ack"], [("req", "req"), ("seen", "seen")],
               [zero, operation("req", "seq.firreg", ["cmd", "clock", "reset", "zero"]), operation("seen", "hw.wire", ["ack"])])
    resp = (operation("next", "comb.mux", ["in", "one", "resp"]), operation("resp", "seq.firreg", ["next", "clock"])) if hold else \
        (operation("resp", "seq.firreg", ["in", "clock", "reset", "zero"]),)
    b = module("B", ["in"], [("echo", "echo"), ("resp", "resp")], [zero, one, operation("echo", "hw.wire", ["in"]), *resp])
    return {"modules": [a, b]}


SPEC = {"engine": "pair", "module": "A", "recipe": "coupled", "limit": 8,
        "inputs": {"cmd": 0}, "partner": {"module": "B", "inputs": []},
        "links": {"in": "req", "ack": "echo"},
        "events": {"req": {"valid": "req"}, "seen": {"valid": "seen"}, "resp": {"valid": "resp"}},
        "operations": {"pulse": {"commands": {0: {"cmd": 1}}}}}


class CoupledTests(unittest.TestCase):
    def records(self, value, doc=None):
        with tempfile.TemporaryDirectory() as directory:
            spec = load_spec(synthetic.write_spec(directory, value, "pair"))
        return {r["name"]: r for r in extract(doc or document(), spec)}

    def test_links_resolve_within_the_cycle(self):
        record = self.records(SPEC)["pair.pulse"]
        self.assertEqual(record["status"], "computed", record.get("reason"))
        self.assertEqual({g: e["first_age"] for g, e in record["events"].items()}, {"req": 1, "seen": 1, "resp": 2})
        self.assertEqual(modules(SPEC), ["A", "B"])

    def test_link_order_does_not_matter(self):
        value = copy.deepcopy(SPEC)
        value["links"] = dict(reversed(list(value["links"].items())))
        self.assertEqual(self.records(value)["pair.pulse"]["events"], self.records(SPEC)["pair.pulse"]["events"])

    def test_combinational_loop_is_unresolved(self):
        value = copy.deepcopy(SPEC)
        value["links"] = {"in": "seen", "ack": "echo"}
        record = self.records(value)["pair.pulse"]
        self.assertEqual(record["status"], "unresolved")
        self.assertIn("unknown", record["reason"])

    def test_unknown_partner_state_reached_through_link_is_unresolved(self):
        value = copy.deepcopy(SPEC)
        value["links"] = {"in": "req", "ack": "resp"}
        value["events"] = {"seen": {"valid": "seen"}}
        record = self.records(value, document(hold=True))["pair.pulse"]
        self.assertEqual(record["status"], "unresolved")
        self.assertIn("unknown after reset/flush", record["reason"])

    def test_invalid_wiring_fails_closed(self):
        cases = {"link to an output": {"links": {"echo": "req"}},
                 "link within one circuit": {"links": {"ack": "seen", "in": "req"}},
                 "unknown partner key": {"partner": {"module": "B", "inputs": [], "wiring": {}}},
                 "partner input not top-level": {"partner": {"module": "B", "inputs": ["in"]}}}
        for label, overlay in cases.items():
            with self.subTest(label):
                record = self.records({**copy.deepcopy(SPEC), **overlay})["pair.pulse"]
                self.assertEqual(record["status"], "unresolved")
                self.assertIn("Circuit construction failed", record["reason"])

    def test_partner_payload_dependence_is_rejected(self):
        doc = document()
        b = doc["modules"][1]
        b["ports"].insert(2, {"name": "data", "direction": "input", "value": "data", "type": "i1"})
        b["arguments"].append({"id": "data", "type": "i1"})
        b["operations"][2]["operands"] = ["data"]
        record = self.records(SPEC, doc)["pair.pulse"]
        self.assertEqual(record["status"], "unresolved")
        self.assertIn("unapproved input: data", record["reason"])


class UnresetTests(unittest.TestCase):
    """``unreset`` allowlists registers that stay unknown after reset/flush; stale, unmatched or already-known entries fail."""
    META = {"path": "", "name": "meta"}

    def masked(self, name, inp):
        """Module whose output ``flag`` is a reset valid ANDed with an unreset hold register ``meta``."""
        zero, one = operation("zero", "hw.constant", [], value="false"), operation("one", "hw.constant", [], value="true")
        return module(name, [inp], [("flag", "flag")], [
            zero, one, operation("valid", "seq.firreg", [inp, "clock", "reset", "zero"], name="valid"),
            operation("next", "comb.mux", [inp, "one", "meta"]), operation("meta", "seq.firreg", ["next", "clock"], name="meta"),
            operation("flag", "comb.and", ["valid", "meta"])])

    def extracted(self, doc, value):
        value = {"engine": "meta", "limit": 6, "operations": {"pulse": {"commands": {0: {"cmd": 1}}}}, **value}
        with tempfile.TemporaryDirectory() as directory:
            return extract(doc, load_spec(synthetic.write_spec(directory, value, "meta")))[0]

    def records(self, unreset=()):
        return self.extracted({"modules": [self.masked("M", "cmd")]}, {
            "module": "M", "inputs": {"cmd": 0}, "unreset": list(unreset), "events": {"flag": {"valid": "flag"}}})

    def coupled(self, unreset=(), partner_unreset=()):
        doc = document()
        doc["modules"].append(self.masked("P", "in"))
        return self.extracted(doc, {
            "module": "A", "recipe": "coupled", "inputs": {"cmd": 0}, "unreset": list(unreset),
            "partner": {"module": "P", "inputs": [], "unreset": list(partner_unreset)}, "links": {"in": "req", "ack": "flag"},
            "events": {"seen": {"valid": "seen"}}})

    def test_unlisted_unknown_register_is_unresolved(self):
        record = self.records()
        self.assertEqual(record["status"], "unresolved")
        self.assertIn("1 control registers remain unknown after reset/flush", record["reason"])

    def test_listed_register_passes(self):
        record = self.records([self.META])
        self.assertEqual((record["status"], record["events"]["flag"]["first_age"]), ("computed", 1))

    def test_listed_register_outside_cone_or_misspelled_fails(self):
        for label, selector in {"misspelled": {"path": "", "name": "meta2"}, "wrong path": {"path": "x", "name": "meta"}}.items():
            with self.subTest(label):
                record = self.records([self.META, selector])
                self.assertEqual(record["status"], "unresolved")
                self.assertIn("matches 0 registers", record["reason"])

    def test_listed_register_that_is_known_is_rejected(self):
        record = self.records([self.META, {"path": "", "name": "valid"}])
        self.assertEqual(record["status"], "unresolved")
        self.assertIn("1 unreset registers are known after reset/flush", record["reason"])

    def test_malformed_unreset_fails_spec_validation(self):
        for bad in ("meta", [{"name": "meta"}], [{"path": "", "name": "meta", "extra": 1}]):
            with self.subTest(bad=bad), tempfile.TemporaryDirectory() as directory:
                value = {"engine": "meta", "module": "M", "inputs": {}, "unreset": bad, "events": {},
                         "operations": {"pulse": {"commands": {0: {}}}}}
                with self.assertRaises(ValueError):
                    load_spec(synthetic.write_spec(directory, value, "meta"))

    def test_state_check_is_gone(self):
        with tempfile.TemporaryDirectory() as directory:
            value = {"engine": "meta", "module": "M", "inputs": {}, "state_check": "outputs", "events": {},
                     "operations": {"pulse": {"commands": {0: {}}}}}
            with self.assertRaisesRegex(ValueError, "unknown keys"):
                load_spec(synthetic.write_spec(directory, value, "meta"))

    def test_coupled_partner_entries_are_per_circuit(self):
        self.assertIn("remain unknown after reset/flush", self.coupled()["reason"])
        self.assertIn("matches 0 registers", self.coupled(unreset=[self.META])["reason"])
        record = self.coupled(partner_unreset=[self.META])
        self.assertEqual((record["status"], record["events"]["seen"]["first_age"]), ("computed", 2), record.get("reason"))


if __name__ == "__main__":
    unittest.main()
