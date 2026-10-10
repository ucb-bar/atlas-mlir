"""Synthetic ControlCircuit, spec-loader and runner checks; no hardware IR is needed."""
import copy
from pathlib import Path
import sys
import tempfile
import unittest

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from rtl_extract.control import ControlCircuit
from rtl_extract.runner import extract
from rtl_extract.spec import load_spec, load_target, variant


def operation(ident, kind, operands, typ="i1", **attrs):
    return {"id": ident, "kind": kind, "operands": operands, "results": [ident],
            "result_types": [typ], "attributes": attrs, "has_regions": False}


def ports(inputs, outputs):
    items = [{"name": n, "direction": "input", "value": n, "type": t} for n, t in inputs]
    return items + [{"name": n, "direction": "output", "value": v, "type": "i1"} for n, v in outputs]


def module(name="Pipe"):
    p = ports([("clock", "!seq.clock"), ("reset", "i1"), ("req", "i1"), ("payload", "i8")], [("out", "reg")])
    return {"name": name, "ports": p,
            "arguments": [{"id": q["value"], "type": q["type"]} for q in p if q["direction"] == "input"],
            "operations": [operation("zero", "hw.constant", [], value="false"),
                           operation("reg", "seq.firreg", ["req", "clock", "reset", "zero"], name="valid")]}


def build(m):
    return ControlCircuit({"modules": [m]}, m["name"], ["out"], {"clock", "reset", "req"})


def trace(c):
    c.cycle({"req": 0, "reset": 1})
    return [c.cycle({"req": int(t == 0), "reset": 0})["out"] for t in range(5)]


def child(operands):
    return operation("reg", "hw.instance", operands, moduleName="@Pipe", instanceName="child",
                     argNames='["clock", "reset", "req", "payload"]', resultNames='["out"]')


class ControlTests(unittest.TestCase):
    def test_pipeline_register_addition_changes_timing(self):
        m = module()
        self.assertEqual(trace(build(m)), [0, 1, 0, 0, 0])
        m["operations"].append(operation("stage2", "seq.firreg", ["reg", "clock", "reset", "zero"], name="valid2"))
        m["ports"][-1]["value"] = "stage2"
        self.assertEqual(trace(build(m)), [0, 0, 1, 0, 0])

    def test_payload_control_dependency_rejected(self):
        m = module()
        m["operations"].insert(1, operation("select", "comb.extract", ["payload"], lowBit="0 : i32"))
        m["operations"][-1]["operands"][0] = "select"
        with self.assertRaisesRegex(ValueError, "unapproved input: payload"):
            build(m)

    def test_enabled_feedback_is_executed(self):
        m = module()
        m["operations"].insert(1, operation("next", "comb.mux", ["req", "req", "reg"]))
        m["operations"][-1]["operands"][0] = "next"
        self.assertEqual(trace(build(m)), [0, 1, 1, 1, 1])

    def test_alternate_clock_rejected(self):
        m = module()
        m["operations"][-1]["operands"][1] = "req"
        with self.assertRaisesRegex(ValueError, "clock"):
            build(m)

    def test_async_reset_rejected(self):
        m = module()
        m["operations"][-1]["attributes"]["isAsync"] = "unit"
        with self.assertRaisesRegex(ValueError, "register attribute"):
            build(m)

    def test_unavailable_instance_rejected(self):
        m = module()
        m["operations"][-1] = operation("reg", "hw.instance", ["req"], moduleName="@Missing",
                                        instanceName="child", argNames='["req"]', resultNames='["out"]')
        with self.assertRaisesRegex(ValueError, "Missing module: Missing"):
            build(m)

    def test_instance_binding_follows_actual_inputs(self):
        m = module("Top")
        m["operations"][-1] = child(["clock", "reset", "req", "payload"])
        c = ControlCircuit({"modules": [m, module()]}, "Top", ["out"], {"clock", "reset", "req"})
        self.assertEqual(trace(c), [0, 1, 0, 0, 0])
        m["operations"][-1]["operands"][2] = "payload"
        with self.assertRaisesRegex(ValueError, "unapproved input"):
            ControlCircuit({"modules": [m, module()]}, "Top", ["out"], {"clock", "reset", "req"})

    def test_unknown_operation_rejected(self):
        m = module()
        m["operations"][-1]["kind"] = "seq.unknown_register"
        with self.assertRaisesRegex(ValueError, "Unsupported timing operation"):
            build(m)

    def test_signed_and_unsigned_comparisons_differ(self):
        m = module()
        m["operations"] = [operation("lhs", "hw.constant", [], "i8", value="255 : i8"),
                           operation("rhs", "hw.constant", [], "i8", value="1 : i8"),
                           operation("reg", "comb.icmp", ["lhs", "rhs"], predicate="2 : i64")]
        self.assertEqual(build(m).cycle({})["out"], 1)
        m["operations"][-1]["attributes"]["predicate"] = "6 : i64"
        self.assertEqual(build(m).cycle({})["out"], 0)

    def test_unknown_state_requires_reset_or_flush(self):
        m = module()
        m["operations"][-1]["operands"] = ["reg", "clock"]
        c = build(m)
        self.assertIsNone(c.cycle({"req": 0, "reset": 1})["out"])
        self.assertIsNone(c.cycle({"req": 0, "reset": 0})["out"])
        m["operations"][-1]["operands"] = ["req", "clock"]
        c = build(m)
        c.cycle({"req": 0, "reset": 0})
        self.assertEqual(c.cycle({"req": 0, "reset": 0})["out"], 0)

    def test_child_clock_binding_rejected(self):
        m = module("Top")
        m["operations"][-1] = child(["req", "reset", "req", "payload"])
        with self.assertRaisesRegex(ValueError, "clock"):
            ControlCircuit({"modules": [m, module()]}, "Top", ["out"], {"clock", "reset", "req"})

    def test_array_indices_follow_hw_order(self):
        m = module()
        m["operations"] = [operation("zero", "hw.constant", [], value="false"),
                           operation("one", "hw.constant", [], value="true"),
                           operation("array", "hw.array_create", ["one", "zero"], "!hw.array<2xi1>"),
                           operation("reg", "hw.array_get", ["array", "zero"])]
        self.assertEqual(build(m).cycle({})["out"], 0)

    def test_renamed_clock_and_reset_ports(self):
        m = module()
        for p in m["ports"]:
            p["name"] = {"clock": "clk", "reset": "rst"}.get(p["name"], p["name"])
        c = ControlCircuit({"modules": [m]}, "Pipe", ["out"], {"clk", "rst", "req"}, clock="clk", reset="rst")
        c.cycle({"req": 0, "rst": 1})
        self.assertEqual([c.cycle({"req": int(t == 0), "rst": 0})["out"] for t in range(3)], [0, 1, 0])
        with self.assertRaisesRegex(ValueError, "clock"):
            build(m)

    def test_instance_result_probe_and_named_cut(self):
        top = module("Top")
        top["ports"][-1]["value"] = "zero"
        top["operations"].append(child(["clock", "reset", "req", "payload"]))
        top["operations"][-1]["results"] = ["inner"]
        doc = {"modules": [top, module()]}
        c = ControlCircuit(doc, "Top", [], {"clock", "reset", "req"}, probes={"inner": {"instance": "child", "result": "out"}})
        c.cycle({"req": 0, "reset": 1})
        self.assertEqual([c.cycle({"req": int(t == 0), "reset": 0})["inner"] for t in range(3)], [0, 1, 0])
        c = ControlCircuit(doc, "Top", [], {"clock", "reset"}, cuts={"forced": {"path": "child", "value": "valid"}},
                           probes={"inner": {"instance": "child", "result": "out"}})
        self.assertEqual(c.cycle({"forced": 1})["inner"], 1)
        self.assertEqual(c.registers, {})
        with self.assertRaisesRegex(ValueError, "Missing named value"):
            ControlCircuit(doc, "Top", [], {"clock", "reset"}, probes={"x": {"path": "child", "value": "absent"}})


def engine():
    """Command starts busy and a request; the response retires the request one cycle later."""
    p = ports([("clock", "!seq.clock"), ("reset", "i1"), ("cmd", "i1"), ("resp", "i1"), ("payload", "i8")],
              [("req", "req"), ("done", "done"), ("busy", "busy")])
    return {"modules": [{"name": "Engine", "ports": p,
                         "arguments": [{"id": q["value"], "type": q["type"]} for q in p if q["direction"] == "input"],
                         "operations": [operation("zero", "hw.constant", [], value="false"),
                                        operation("one", "hw.constant", [], value="true"),
                                        operation("req", "seq.firreg", ["cmd", "clock", "reset", "zero"]),
                                        operation("done", "seq.firreg", ["resp", "clock", "reset", "zero"]),
                                        operation("hold", "comb.mux", ["done", "zero", "busy"]),
                                        operation("next", "comb.mux", ["cmd", "one", "hold"]),
                                        operation("busy", "seq.firreg", ["next", "clock", "reset", "zero"])]}]}


SPEC = {"engine": "toy", "module": "Engine", "limit": 40, "busy": "busy",
        "inputs": {"cmd": 0, "resp": 0},
        "events": {"request": {"valid": "req", "response": {"input": "resp", "latency": 1}}, "done": {"valid": "done"}},
        "operations": {"pulse": {"commands": {0: {"cmd": 1}}}},
        "variants": {"slow": {"events": {"request": {"response": {"latency": 3}}}}}}


def write_spec(directory, value, name="toy"):
    path = Path(directory) / f"{name}.yaml"
    path.write_text(yaml.safe_dump(value))
    return path


class SpecTests(unittest.TestCase):
    def load(self, value):
        with tempfile.TemporaryDirectory() as directory:
            return load_spec(write_spec(directory, value))

    def test_valid_spec_and_variant(self):
        spec = self.load(SPEC)
        self.assertEqual(spec["reset_cycles"], 16)
        self.assertEqual(variant(spec, "slow")["events"]["request"]["response"]["latency"], 3)

    def test_fail_closed(self):
        cases = {"missing module": lambda s: s.pop("module"),
                 "unknown top-level key": lambda s: s.update(stimulus={}),
                 "unknown event key": lambda s: s["events"]["done"].update(ready="x"),
                 "undeclared command input": lambda s: s["operations"]["pulse"]["commands"][0].update(payload=1),
                 "response without latency or memory": lambda s: s["events"]["request"]["response"].pop("latency"),
                 "bad response latency": lambda s: s["events"]["request"]["response"].update(latency=0),
                 "unknown recipe": lambda s: s.update(recipe="magic"),
                 "variant overrides module": lambda s: s["variants"].update(other={"module": "X"}),
                 "invalid variant result": lambda s: s["variants"].update(other={"events": {"done": {"bogus": 1}}}),
                 "bad age range": lambda s: s["operations"]["pulse"]["commands"].update({"5..2": {"cmd": 1}}),
                 "unknown row field": lambda s: s["events"]["done"].update(row="row"),
                 "unknown check key": lambda s: s.update(checks=[{"equal": [], "within": 1}])}
        for label, mutate in cases.items():
            with self.subTest(label):
                value = copy.deepcopy(SPEC)
                mutate(value)
                with self.assertRaises(ValueError):
                    self.load(value)

    def test_operation_table_expands_template(self):
        value = copy.deepcopy(SPEC)
        value["operation_table"] = {"codes": {"a": 1, "b": 0}, "template": {"commands": {0: {"cmd": "$code"}}}}
        self.assertEqual(self.load(value)["operations"]["b"]["commands"][0], {"cmd": 0})

    def test_unknown_engine_rejected(self):
        with self.assertRaisesRegex(ValueError, "Unknown engines"):
            load_target("atlas", ["not_an_engine"])


class RunnerTests(unittest.TestCase):
    def records(self, value, document=None):
        with tempfile.TemporaryDirectory() as directory:
            spec = load_spec(write_spec(directory, value))
        return {r["name"]: r for r in extract(document or engine(), spec)}

    def test_responses_drive_events_and_release(self):
        records = self.records(SPEC)
        base, slow = records["toy.pulse"], records["toy.pulse/slow"]
        self.assertEqual((base["status"], base["events"]["request"]["first_age"], base["events"]["done"]["first_age"],
                          base["first_free_age"]), ("computed", 1, 3, 4))
        self.assertEqual((slow["events"]["done"]["first_age"], slow["first_free_age"]), (5, 6))
        self.assertEqual(slow["assumptions"]["scratchpad_read_latency"], 3)

    def test_missing_release_is_unresolved(self):
        value = copy.deepcopy(SPEC)
        del value["events"]["request"]["response"]
        value["inputs"].pop("resp")
        value["variants"] = {}
        doc = engine()
        doc["modules"][0]["operations"][3]["operands"][0] = "zero"
        record = self.records(value, doc)["toy.pulse"]
        self.assertEqual((record["status"], record["first_free_age"]), ("unresolved", None))
        self.assertIn("Busy did not clear", record["reason"])

    def test_payload_dependence_is_unresolved(self):
        doc = engine()
        doc["modules"][0]["operations"].insert(2, operation("bit", "comb.extract", ["payload"], lowBit="0 : i32"))
        doc["modules"][0]["operations"][3]["operands"][0] = "bit"
        records = self.records(SPEC, doc)
        self.assertTrue(all(r["status"] == "unresolved" and "unapproved input" in r["reason"] for r in records.values()))

    def test_failed_check_unresolves_records(self):
        value = copy.deepcopy(SPEC)
        value["checks"] = [{"equal": ["pulse.first_free_age", "pulse/slow.first_free_age"]}]
        records = self.records(value)
        self.assertTrue(all(r["status"] == "unresolved" and r["first_free_age"] is None for r in records.values()))

    def test_unknown_control_state_is_unresolved(self):
        doc = engine()
        doc["modules"][0]["operations"][6]["operands"] = ["next", "clock"]
        doc["modules"][0]["operations"][5]["operands"] = ["cmd", "one", "busy"]
        record = self.records(SPEC, doc)["toy.pulse"]
        self.assertEqual(record["status"], "unresolved")
        self.assertIn("unknown after reset/flush", record["reason"])


if __name__ == "__main__":
    unittest.main()
