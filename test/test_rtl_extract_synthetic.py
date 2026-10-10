"""Synthetic checks (no hardware IR): ControlCircuit, spec loading, runner, derivation, coupled recipe, unreset rules, op_timing document."""
import copy
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from rtl_extract import derive, facts
from rtl_extract.control import ControlCircuit
from rtl_extract.runner import extract, modules
from rtl_extract.spec import load_spec, load_target, variant


def operation(ident, kind, operands, typ="i1", **attrs):
    return {"id": ident, "kind": kind, "operands": operands, "results": [ident],
            "result_types": [typ], "attributes": attrs, "has_regions": False}


def module(name, inputs, outputs, operations, clocked=True):
    """``inputs``: names (i1) or (name, type); ``outputs``: (name, driving value[, type])."""
    ins = [("clock", "!seq.clock"), ("reset", "i1")] * clocked + [(i, "i1") if isinstance(i, str) else i for i in inputs]
    outs = [(o[0], o[1], o[2] if len(o) > 2 else "i1") for o in outputs]
    return {"name": name, "operations": operations, "arguments": [{"id": n, "type": t} for n, t in ins],
            "ports": [{"name": n, "direction": "input", "value": n, "type": t} for n, t in ins]
                     + [{"name": n, "direction": "output", "value": v, "type": t} for n, v, t in outs]}


def instance(name, target, pins, results):
    """``pins``: {port: value}; ``results``: [(port, type)], result ids are ``name.port``."""
    return {"id": name, "kind": "hw.instance", "operands": list(pins.values()), "results": [f"{name}.{p}" for p, _ in results],
            "result_types": [t for _, t in results], "has_regions": False,
            "attributes": {"moduleName": f"@{target}", "instanceName": name, "argNames": json.dumps(list(pins)),
                           "resultNames": json.dumps([p for p, _ in results])}}


def write_spec(directory, value):
    path = Path(directory) / f"{value['engine']}.yaml"
    path.write_text(yaml.safe_dump(value))
    return path


def load(value):
    with tempfile.TemporaryDirectory() as directory:
        return load_spec(write_spec(directory, value))


def records(value, document):
    return {r["name"]: r for r in extract(document, load(value))}


def zero():
    return operation("zero", "hw.constant", [], value="false")


def one():
    return operation("one", "hw.constant", [], value="true")


def pipe(name="Pipe"):
    return module(name, ["req", ("payload", "i8")], [("out", "reg")], [
        zero(), operation("reg", "seq.firreg", ["req", "clock", "reset", "zero"], name="valid")])


def build(m):
    return ControlCircuit({"modules": [m]}, m["name"], ["out"], {"clock", "reset", "req"})


def trace(c):
    c.cycle({"req": 0, "reset": 1})
    return [c.cycle({"req": int(t == 0), "reset": 0})["out"] for t in range(5)]


def child(operands, target="Pipe"):
    """Instance named ``child`` of ``target`` whose single result is the id ``reg``."""
    pins = dict(zip(["clock", "reset", "req", "payload"], operands))
    item = instance("child", target, pins, [("out", "i1")])
    return item | {"id": "reg", "results": ["reg"]}


class ControlTests(unittest.TestCase):
    def test_pipeline_register_addition_changes_timing(self):
        m = pipe()
        self.assertEqual(trace(build(m)), [0, 1, 0, 0, 0])
        m["operations"].append(operation("stage2", "seq.firreg", ["reg", "clock", "reset", "zero"], name="valid2"))
        m["ports"][-1]["value"] = "stage2"
        self.assertEqual(trace(build(m)), [0, 0, 1, 0, 0])

    def test_payload_control_dependency_rejected(self):
        m = pipe()
        m["operations"].insert(1, operation("select", "comb.extract", ["payload"], lowBit="0 : i32"))
        m["operations"][-1]["operands"][0] = "select"
        with self.assertRaisesRegex(ValueError, "unapproved input: payload"):
            build(m)

    def test_enabled_feedback_is_executed(self):
        m = pipe()
        m["operations"].insert(1, operation("next", "comb.mux", ["req", "req", "reg"]))
        m["operations"][-1]["operands"][0] = "next"
        self.assertEqual(trace(build(m)), [0, 1, 1, 1, 1])

    def test_unsupported_register_forms_rejected(self):
        for label, mutate, message in (
                ("alternate clock", lambda m: m["operations"][-1]["operands"].__setitem__(1, "req"), "clock"),
                ("async reset", lambda m: m["operations"][-1]["attributes"].update(isAsync="unit"), "register attribute"),
                ("unknown operation", lambda m: m["operations"][-1].update(kind="seq.unknown_register"), "Unsupported timing operation"),
                ("missing instance", lambda m: m["operations"].__setitem__(-1, child(["req"], "Missing")),
                 "Missing module: Missing")):
            with self.subTest(label):
                m = pipe()
                mutate(m)
                with self.assertRaisesRegex(ValueError, message):
                    build(m)

    def test_instance_binding_follows_actual_inputs(self):
        top = pipe("Top")
        top["operations"][-1] = child(["clock", "reset", "req", "payload"])
        run = lambda: ControlCircuit({"modules": [top, pipe()]}, "Top", ["out"], {"clock", "reset", "req"})
        self.assertEqual(trace(run()), [0, 1, 0, 0, 0])
        top["operations"][-1]["operands"][2] = "payload"
        with self.assertRaisesRegex(ValueError, "unapproved input"):
            run()
        top["operations"][-1]["operands"] = ["req", "reset", "req", "payload"]
        with self.assertRaisesRegex(ValueError, "clock"):
            run()

    def test_signed_and_unsigned_comparisons_differ(self):
        m = pipe()
        m["operations"] = [operation("lhs", "hw.constant", [], "i8", value="255 : i8"),
                           operation("rhs", "hw.constant", [], "i8", value="1 : i8"),
                           operation("reg", "comb.icmp", ["lhs", "rhs"], predicate="2 : i64")]
        self.assertEqual(build(m).cycle({})["out"], 1)
        m["operations"][-1]["attributes"]["predicate"] = "6 : i64"
        self.assertEqual(build(m).cycle({})["out"], 0)

    def test_unknown_state_requires_reset_or_flush(self):
        m = pipe()
        m["operations"][-1]["operands"] = ["reg", "clock"]
        c = build(m)
        self.assertIsNone(c.cycle({"req": 0, "reset": 1})["out"])
        self.assertIsNone(c.cycle({"req": 0, "reset": 0})["out"])
        m["operations"][-1]["operands"] = ["req", "clock"]
        c = build(m)
        c.cycle({"req": 0, "reset": 0})
        self.assertEqual(c.cycle({"req": 0, "reset": 0})["out"], 0)

    def test_array_indices_follow_hw_order(self):
        m = pipe()
        m["operations"] = [zero(), one(), operation("array", "hw.array_create", ["one", "zero"], "!hw.array<2xi1>"),
                           operation("reg", "hw.array_get", ["array", "zero"])]
        self.assertEqual(build(m).cycle({})["out"], 0)

    def test_renamed_clock_and_reset_ports(self):
        m = pipe()
        for p in m["ports"]:
            p["name"] = {"clock": "clk", "reset": "rst"}.get(p["name"], p["name"])
        c = ControlCircuit({"modules": [m]}, "Pipe", ["out"], {"clk", "rst", "req"}, clock="clk", reset="rst")
        c.cycle({"req": 0, "rst": 1})
        self.assertEqual([c.cycle({"req": int(t == 0), "rst": 0})["out"] for t in range(3)], [0, 1, 0])
        with self.assertRaisesRegex(ValueError, "clock"):
            build(m)

    def test_instance_result_probe_and_named_cut(self):
        top = pipe("Top")
        top["ports"][-1]["value"] = "zero"
        top["operations"].append(child(["clock", "reset", "req", "payload"]))
        top["operations"][-1]["results"] = ["inner"]
        doc = {"modules": [top, pipe()]}
        probes = {"inner": {"instance": "child", "result": "out"}}
        c = ControlCircuit(doc, "Top", [], {"clock", "reset", "req"}, probes=probes)
        c.cycle({"req": 0, "reset": 1})
        self.assertEqual([c.cycle({"req": int(t == 0), "reset": 0})["inner"] for t in range(3)], [0, 1, 0])
        c = ControlCircuit(doc, "Top", [], {"clock", "reset"}, cuts={"forced": {"path": "child", "value": "valid"}}, probes=probes)
        self.assertEqual(c.cycle({"forced": 1})["inner"], 1)
        self.assertEqual(c.registers, {})
        with self.assertRaisesRegex(ValueError, "Missing named value"):
            ControlCircuit(doc, "Top", [], {"clock", "reset"}, probes={"x": {"path": "child", "value": "absent"}})


def engine():
    """Command starts busy and a request; the response retires the request one cycle later."""
    return {"modules": [module("Engine", ["cmd", "resp", ("payload", "i8")], [("req", "req"), ("done", "done"), ("busy", "busy")], [
        zero(), one(),
        operation("req", "seq.firreg", ["cmd", "clock", "reset", "zero"]),
        operation("done", "seq.firreg", ["resp", "clock", "reset", "zero"]),
        operation("hold", "comb.mux", ["done", "zero", "busy"]),
        operation("next", "comb.mux", ["cmd", "one", "hold"]),
        operation("busy", "seq.firreg", ["next", "clock", "reset", "zero"])])]}


TOY = {"engine": "toy", "module": "Engine", "limit": 40, "busy": "busy", "inputs": {"cmd": 0, "resp": 0},
       "events": {"request": {"valid": "req", "response": {"input": "resp", "latency": 1}}, "done": {"valid": "done"}},
       "operations": {"pulse": {"commands": {0: {"cmd": 1}}}},
       "variants": {"slow": {"events": {"request": {"response": {"latency": 3}}}}}}


class SpecTests(unittest.TestCase):
    def test_valid_spec_and_variant(self):
        spec = load(TOY)
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
                 "unknown check key": lambda s: s.update(checks=[{"equal": [], "within": 1}]),
                 "state_check removed": lambda s: s.update(state_check="outputs"),
                 "unreset not a list": lambda s: s.update(unreset="meta"),
                 "unreset missing path": lambda s: s.update(unreset=[{"name": "meta"}]),
                 "unreset extra key": lambda s: s.update(unreset=[{"path": "", "name": "meta", "extra": 1}])}
        for label, mutate in cases.items():
            with self.subTest(label):
                value = copy.deepcopy(TOY)
                mutate(value)
                with self.assertRaises(ValueError):
                    load(value)

    def test_operation_table_expands_template(self):
        value = copy.deepcopy(TOY)
        value["operation_table"] = {"codes": {"a": 1, "b": 0}, "template": {"commands": {0: {"cmd": "$code"}}}}
        self.assertEqual(load(value)["operations"]["b"]["commands"][0], {"cmd": 0})

    def test_unknown_engine_rejected(self):
        with self.assertRaisesRegex(ValueError, "Unknown engines"):
            load_target("atlas", ["not_an_engine"])


class RunnerTests(unittest.TestCase):
    def test_responses_drive_events_and_release(self):
        found = records(TOY, engine())
        base, slow = found["toy.pulse"], found["toy.pulse/slow"]
        self.assertEqual((base["status"], base["events"]["request"]["first_age"], base["events"]["done"]["first_age"],
                          base["first_free_age"]), ("computed", 1, 3, 4))
        self.assertEqual((slow["events"]["done"]["first_age"], slow["first_free_age"]), (5, 6))
        self.assertEqual(slow["assumptions"]["scratchpad_read_latency"], 3)

    def test_missing_release_is_unresolved(self):
        value = copy.deepcopy(TOY)
        del value["events"]["request"]["response"]
        value["inputs"].pop("resp")
        value["variants"] = {}
        doc = engine()
        doc["modules"][0]["operations"][3]["operands"][0] = "zero"
        record = records(value, doc)["toy.pulse"]
        self.assertEqual((record["status"], record["first_free_age"]), ("unresolved", None))
        self.assertIn("Busy did not clear", record["reason"])

    def test_payload_dependence_is_unresolved(self):
        doc = engine()
        doc["modules"][0]["operations"].insert(2, operation("bit", "comb.extract", ["payload"], lowBit="0 : i32"))
        doc["modules"][0]["operations"][3]["operands"][0] = "bit"
        self.assertTrue(all(r["status"] == "unresolved" and "unapproved input" in r["reason"] for r in records(TOY, doc).values()))

    def test_failed_check_unresolves_records(self):
        value = copy.deepcopy(TOY)
        value["checks"] = [{"equal": ["pulse.first_free_age", "pulse/slow.first_free_age"]}]
        self.assertTrue(all(r["status"] == "unresolved" and r["first_free_age"] is None for r in records(value, engine()).values()))

    def test_unknown_control_state_is_unresolved(self):
        doc = engine()
        doc["modules"][0]["operations"][6]["operands"] = ["next", "clock"]
        doc["modules"][0]["operations"][5]["operands"] = ["cmd", "one", "busy"]
        record = records(TOY, doc)["toy.pulse"]
        self.assertEqual(record["status"], "unresolved")
        self.assertIn("unknown after reset/flush", record["reason"])


def system(stages=1, firmem=(1,)):
    """Sys instantiates a decoder (instruction bits 26:25 are the op), an engine and a memory answering after ``stages``."""
    eng = module("Eng", ["io_cmd_valid", ("io_cmd_bits_op", "i2"), ("io_cmd_bits_tag", "i3"), "io_resp_valid", ("io_resp_bits", "i8")],
                 [("io_req_valid", "req"), ("io_req_bits_tag", "tag", "i3"), ("io_req_bits_data", "io_resp_bits", "i8"), ("io_busy", "busy")],
                 [zero(), one(), operation("go", "hw.constant", [], "i2", value="1 : i2"),
                  operation("isgo", "comb.icmp", ["io_cmd_bits_op", "go"], predicate="0 : i64"),
                  operation("start", "comb.and", ["io_cmd_valid", "isgo"]),
                  operation("req", "seq.firreg", ["start", "clock", "reset", "zero"]),
                  operation("tag", "seq.firreg", ["io_cmd_bits_tag", "clock"], "i3"),
                  operation("hold", "comb.mux", ["io_resp_valid", "zero", "busy"]),
                  operation("next", "comb.mux", ["start", "one", "hold"]),
                  operation("busy", "seq.firreg", ["next", "clock", "reset", "zero"])])
    chain = [operation(f"stage{i}", "seq.firreg", [f"stage{i - 1}" if i else "io_eng_valid", "clock", "reset", "zero"]) for i in range(stages)]
    memory = module("Mem", ["io_eng_valid", ("io_eng_bits_row", "i3"), "io_other_valid"], [("io_engResp_valid", f"stage{stages - 1}")],
                    [zero(), *chain, *[operation(f"bank{i}", "seq.firmem", [], "!seq.firmem<8 x 8>", readLatency=f"{n} : i32")
                                       for i, n in enumerate(firmem)]])
    decoder = module("Dec", [("io_instr", "i32")], [("io_op", "op", "i2")],
                     [operation("op", "comb.extract", ["io_instr"], "i2", lowBit="25 : i32")], clocked=False)
    top = module("Sys", [("instr", "i32")], [], [
        zero(), operation("three", "hw.constant", [], "i3", value="3 : i3"),
        operation("s1_instr", "hw.wire", ["instr"], "i32", name="s1_instr"),
        instance("dec", "Dec", {"io_instr": "s1_instr"}, [("io_op", "i2")]),
        instance("eng", "Eng", {"clock": "clock", "reset": "reset", "io_cmd_valid": "zero", "io_cmd_bits_op": "dec.io_op",
                                "io_cmd_bits_tag": "three", "io_resp_valid": "mem.io_engResp_valid", "io_resp_bits": "zero"},
                 [("io_req_valid", "i1"), ("io_req_bits_tag", "i3"), ("io_req_bits_data", "i8"), ("io_busy", "i1")]),
        instance("mem", "Mem", {"clock": "clock", "reset": "reset", "io_eng_valid": "eng.io_req_valid",
                                "io_eng_bits_row": "eng.io_req_bits_tag", "io_other_valid": "zero"}, [("io_engResp_valid", "i1")])])
    return {"modules": [top, eng, memory, decoder]}


DERIVED = {"engine": "toy", "module": "Eng", "system": "Sys", "command": "io_cmd", "limit": 40, "busy": "io_busy",
           "inputs": {"io_cmd_bits_tag": 5},
           "decode": {"input": "io_cmd_bits_op", "instruction": {"instance": "dec", "port": "io_instr"},
                      "words": {"go": 1 << 25, "stop": 2 << 25}},
           "events": {"req": {"bundle": "io_req", "response": {"input": "io_resp_valid", "memory": "Mem"}}},
           "operations": {"go": {"commands": {0: {"io_cmd_valid": 1, "io_cmd_bits_op": "$go", "io_cmd_bits_tag": 2}}}},
           "variants": {"slow": {"events": {"req": {"response": {"latency": 3}}}}}}


class DeriveTests(unittest.TestCase):
    def resolve(self, value=DERIVED, doc=None):
        return derive.resolve(doc or system(), load(value))

    def test_ports_expand_from_bundles(self):
        spec, _ = self.resolve()
        self.assertEqual(spec["inputs"], {"io_cmd_valid": 0, "io_cmd_bits_op": 0, "io_cmd_bits_tag": 5, "io_resp_valid": 0})
        self.assertEqual(spec["events"]["req"]["valid"], "io_req_valid")
        self.assertEqual(spec["events"]["req"]["fields"], {"tag": "io_req_bits_tag"})  # data depends on io_resp_bits: payload
        value = copy.deepcopy(DERIVED)
        value["events"]["req"]["fields"] = {"t": "io_req_bits_tag"}
        self.assertEqual(self.resolve(value)[0]["events"]["req"]["fields"], {"t": "io_req_bits_tag"})

    def test_missing_bundle_fails(self):
        value = copy.deepcopy(DERIVED)
        value["events"]["req"]["bundle"] = "io_absent"
        with self.assertRaisesRegex(ValueError, "exactly one module"):
            self.resolve(value)
        doc = system()
        doc["modules"][1]["ports"][2]["name"] = "io_start"
        with self.assertRaisesRegex(ValueError, "Missing command port"):
            self.resolve(doc=doc)

    def test_response_latency_measured(self):
        spec, notes = self.resolve()
        self.assertEqual(spec["events"]["req"]["response"]["latency"], 1)
        self.assertEqual(notes["scratchpad_read_derivation"]["req"],
                         {"memory": "Sys/mem", "request": "io_eng_valid", "response": "io_engResp_valid",
                          "measured": 1, "memory_read_latency": 1, "used": 1})

    def test_extra_response_register_wins_and_both_are_reported(self):
        spec, notes = self.resolve(doc=system(stages=2))
        self.assertEqual(spec["events"]["req"]["response"]["latency"], 2)
        self.assertEqual({k: notes["scratchpad_read_derivation"]["req"][k] for k in ("measured", "memory_read_latency")},
                         {"measured": 2, "memory_read_latency": 1})

    def test_declared_latency_overrides_and_reports_measured(self):
        resolved, notes = derive.resolve(system(), variant(load(DERIVED), "slow"))
        note = notes["scratchpad_read_derivation"]["req"]
        self.assertEqual((resolved["events"]["req"]["response"]["latency"], note["measured"], note["used"]), (3, 1, 3))

    def test_response_fails_closed(self):
        cases = {"precedes its read data": system(firmem=(2,)), "mixes memory read latencies": system(firmem=(1, 2))}
        doc = system()
        doc["modules"][0]["operations"][-1]["operands"][-1] = "eng.io_req_valid"  # the request valid also drives io_other_valid
        cases["drives 2"] = doc
        doc = system()
        doc["modules"][0]["operations"][4]["operands"][5] = "zero"  # engine response not driven by the memory
        cases["driven by 0"] = doc
        doc = system()
        operations = doc["modules"][0]["operations"]
        operations.append(copy.deepcopy(operations[-1]) | {"id": "mem2", "results": ["mem2.io_engResp_valid"]})
        operations[-1]["attributes"]["instanceName"] = "mem2"
        cases["2 times"] = doc
        for message, doc in cases.items():
            with self.subTest(message), self.assertRaisesRegex(ValueError, message):
                self.resolve(doc=doc)

    def test_codes_decoded_from_instruction_words(self):
        spec, _ = self.resolve()
        self.assertEqual(spec["operations"]["go"]["commands"][0]["io_cmd_bits_op"], 1)
        self.assertEqual(derive.decode(system(), load(DERIVED), {}), {"go": 1, "stop": 2})

    def test_decode_fails_closed(self):
        clash = copy.deepcopy(DERIVED)
        clash["decode"]["words"]["again"] = 1 << 25
        idle = copy.deepcopy(DERIVED)
        idle["decode"]["words"]["nop"] = 0x13
        cases = {"decode to the same code": (clash, system()), "all-zero word": (idle, system())}
        doc = system()
        doc["modules"][0]["operations"][3]["operands"] = ["instr"]  # decoder input driven straight by a port
        cases["no nameable driver"] = (DERIVED, doc)
        doc = system()
        decoder = doc["modules"][3]
        decoder["operations"].append(operation("late", "seq.firreg", ["op", "clock"], "i2"))
        decoder["ports"][-1]["value"] = "late"
        decoder["ports"].insert(0, {"name": "clock", "direction": "input", "value": "clock", "type": "!seq.clock"})
        doc["modules"][0]["operations"][3]["operands"].insert(0, "clock")
        doc["modules"][0]["operations"][3]["attributes"]["argNames"] = '["clock", "io_instr"]'
        cases["depends on state"] = (DERIVED, doc)
        for message, (value, doc) in cases.items():
            with self.subTest(message), self.assertRaisesRegex(ValueError, message):
                derive.decode(doc, load(value), {})

    def test_records_computed_with_derived_inputs(self):
        found = records(DERIVED, system())
        base, slow = found["toy.go"], found["toy.go/slow"]
        self.assertEqual((base["status"], base["events"]["req"]["first_age"], base["first_free_age"]), ("computed", 1, 3))
        self.assertEqual(base["events"]["req"]["values"], {"tag": [2]})
        self.assertEqual(base["assumptions"]["scratchpad_read_latency"], 1)
        self.assertEqual((slow["first_free_age"], slow["assumptions"]["scratchpad_read_latency"]), (5, 3))

    def test_derivation_failure_is_unresolved(self):
        doc = system()
        doc["modules"][0]["operations"][4]["operands"][5] = "zero"
        self.assertTrue(all(r["status"] == "unresolved" and "driven by 0" in r["reason"] for r in extract(doc, load(DERIVED))))

    def test_modules_include_system(self):
        self.assertEqual(modules(load(DERIVED)), ["Eng", "Sys"])

    def test_spec_validation(self):
        cases = {"valid and bundle": lambda s: s["events"]["req"].update(valid="io_req_valid"),
                 "latency or memory": lambda s: s["events"]["req"]["response"].pop("memory"),
                 "need a system": lambda s: s.pop("system"),
                 "absent from decode.words": lambda s: s["operations"]["go"]["commands"][0].update(io_cmd_bits_op="$halt"),
                 "needs codes or decode.words": lambda s: (s.pop("decode"), s["operations"]["go"]["commands"][0].pop("io_cmd_bits_op"),
                                                           s["events"]["req"]["response"].update(latency=1),
                                                           s.update(operation_table={"template": {"commands": {0: {"io_cmd_valid": 1}}}})),
                 "undeclared inputs": lambda s: s["operations"]["go"]["commands"][0].update(other=1)}
        for message, mutate in cases.items():
            value = copy.deepcopy(DERIVED)
            mutate(value)
            with self.subTest(message), self.assertRaisesRegex(ValueError, message):
                load(value)

    def test_operation_table_from_decode_words(self):
        value = copy.deepcopy(DERIVED)
        value["operations"] = {}
        value["operation_table"] = {"template": {"commands": {0: {"io_cmd_valid": 1, "io_cmd_bits_op": "$code"}}}}
        spec = load(value)
        self.assertEqual(spec["operations"]["stop"]["commands"][0]["io_cmd_bits_op"], "$stop")
        self.assertEqual(derive.resolve(system(), spec)[0]["operations"]["stop"]["commands"][0]["io_cmd_bits_op"], 2)


def pair(hold=False):
    """A registers ``cmd`` into ``req`` and passes ``ack`` straight to ``seen``; B echoes ``in`` and registers it."""
    a = module("A", ["cmd", "ack"], [("req", "req"), ("seen", "seen")],
               [zero(), operation("req", "seq.firreg", ["cmd", "clock", "reset", "zero"]), operation("seen", "hw.wire", ["ack"])])
    resp = (operation("next", "comb.mux", ["in", "one", "resp"]), operation("resp", "seq.firreg", ["next", "clock"])) if hold else \
        (operation("resp", "seq.firreg", ["in", "clock", "reset", "zero"]),)
    b = module("B", ["in"], [("echo", "echo"), ("resp", "resp")], [zero(), one(), operation("echo", "hw.wire", ["in"]), *resp])
    return {"modules": [a, b]}


PAIR = {"engine": "pair", "module": "A", "recipe": "coupled", "limit": 8,
        "inputs": {"cmd": 0}, "partner": {"module": "B", "inputs": []}, "links": {"in": "req", "ack": "echo"},
        "events": {"req": {"valid": "req"}, "seen": {"valid": "seen"}, "resp": {"valid": "resp"}},
        "operations": {"pulse": {"commands": {0: {"cmd": 1}}}}}


class CoupledTests(unittest.TestCase):
    def pulse(self, value=PAIR, doc=None):
        return records(value, doc or pair())["pair.pulse"]

    def test_links_resolve_within_the_cycle(self):
        record = self.pulse()
        self.assertEqual(record["status"], "computed", record.get("reason"))
        self.assertEqual({g: e["first_age"] for g, e in record["events"].items()}, {"req": 1, "seen": 1, "resp": 2})
        self.assertEqual(modules(load(PAIR)), ["A", "B"])

    def test_link_order_does_not_matter(self):
        value = copy.deepcopy(PAIR)
        value["links"] = dict(reversed(list(value["links"].items())))
        self.assertEqual(self.pulse(value)["events"], self.pulse()["events"])

    def test_combinational_loop_is_unresolved(self):
        record = self.pulse({**PAIR, "links": {"in": "seen", "ack": "echo"}})
        self.assertEqual(record["status"], "unresolved")
        self.assertIn("unknown", record["reason"])

    def test_unknown_partner_state_reached_through_link_is_unresolved(self):
        value = {**PAIR, "links": {"in": "req", "ack": "resp"}, "events": {"seen": {"valid": "seen"}}}
        record = self.pulse(value, pair(hold=True))
        self.assertEqual(record["status"], "unresolved")
        self.assertIn("unknown after reset/flush", record["reason"])

    def test_invalid_wiring_fails_closed(self):
        cases = {"link to an output": {"links": {"echo": "req"}},
                 "link within one circuit": {"links": {"ack": "seen", "in": "req"}},
                 "unknown partner key": {"partner": {"module": "B", "inputs": [], "wiring": {}}},
                 "partner input not top-level": {"partner": {"module": "B", "inputs": ["in"]}}}
        for label, overlay in cases.items():
            with self.subTest(label):
                record = self.pulse({**copy.deepcopy(PAIR), **overlay})
                self.assertEqual(record["status"], "unresolved")
                self.assertIn("Circuit construction failed", record["reason"])

    def test_partner_payload_dependence_is_rejected(self):
        doc = pair()
        b = doc["modules"][1]
        b["ports"].insert(2, {"name": "data", "direction": "input", "value": "data", "type": "i1"})
        b["arguments"].append({"id": "data", "type": "i1"})
        b["operations"][2]["operands"] = ["data"]
        record = self.pulse(doc=doc)
        self.assertEqual(record["status"], "unresolved")
        self.assertIn("unapproved input: data", record["reason"])


class UnresetTests(unittest.TestCase):
    """``unreset`` allowlists registers that stay unknown after reset/flush; stale, unmatched or already-known entries fail."""
    META = {"path": "", "name": "meta"}

    def masked(self, name, inp):
        """Module whose output ``flag`` is a reset valid ANDed with an unreset hold register ``meta``."""
        return module(name, [inp], [("flag", "flag")], [
            zero(), one(), operation("valid", "seq.firreg", [inp, "clock", "reset", "zero"], name="valid"),
            operation("next", "comb.mux", [inp, "one", "meta"]), operation("meta", "seq.firreg", ["next", "clock"], name="meta"),
            operation("flag", "comb.and", ["valid", "meta"])])

    def extracted(self, doc, value):
        value = {"engine": "meta", "limit": 6, "operations": {"pulse": {"commands": {0: {"cmd": 1}}}}, **value}
        return extract(doc, load(value))[0]

    def single(self, unreset=()):
        return self.extracted({"modules": [self.masked("M", "cmd")]}, {
            "module": "M", "inputs": {"cmd": 0}, "unreset": list(unreset), "events": {"flag": {"valid": "flag"}}})

    def coupled(self, unreset=(), partner_unreset=()):
        doc = pair()
        doc["modules"].append(self.masked("P", "in"))
        return self.extracted(doc, {
            "module": "A", "recipe": "coupled", "inputs": {"cmd": 0}, "unreset": list(unreset),
            "partner": {"module": "P", "inputs": [], "unreset": list(partner_unreset)}, "links": {"in": "req", "ack": "flag"},
            "events": {"seen": {"valid": "seen"}}})

    def test_unlisted_unknown_register_is_unresolved(self):
        record = self.single()
        self.assertEqual(record["status"], "unresolved")
        self.assertIn("1 control registers remain unknown after reset/flush", record["reason"])

    def test_listed_register_passes(self):
        record = self.single([self.META])
        self.assertEqual((record["status"], record["events"]["flag"]["first_age"]), ("computed", 1))

    def test_listed_register_outside_cone_or_misspelled_fails(self):
        for label, selector in {"misspelled": {"path": "", "name": "meta2"}, "wrong path": {"path": "x", "name": "meta"}}.items():
            with self.subTest(label):
                record = self.single([self.META, selector])
                self.assertEqual(record["status"], "unresolved")
                self.assertIn("matches 0 registers", record["reason"])

    def test_listed_register_that_is_known_is_rejected(self):
        record = self.single([self.META, {"path": "", "name": "valid"}])
        self.assertEqual(record["status"], "unresolved")
        self.assertIn("1 unreset registers are known after reset/flush", record["reason"])

    def test_coupled_partner_entries_are_per_circuit(self):
        self.assertIn("remain unknown after reset/flush", self.coupled()["reason"])
        self.assertIn("matches 0 registers", self.coupled(unreset=[self.META])["reason"])
        record = self.coupled(partner_unreset=[self.META])
        self.assertEqual((record["status"], record["events"]["seen"]["first_age"]), ("computed", 2), record.get("reason"))


class OpTimingDocument(unittest.TestCase):
    SPEC = {"engine": "toy", "module": "Toy", "reset_cycles": 4, "flush_cycles": 4, "limit": 32,
            "events": {"read": {"valid": "rd", "response": {"input": "rsp", "latency": 2}}},
            "operations": {"pulse": {"commands": {0: {"go": 1}}, "next_issue": {"signal": "ready"}}, "plain": {"commands": {}}}}
    EVENTS = {"read": {"first_age": 1, "last_age": 3, "count": 3, "step": 1, "values": {}, "row_contiguous": True}}
    RESULT = {"events": EVENTS, "first_free_age": 7, "next_issue_age": 5}

    def blocks(self, *items):
        with tempfile.TemporaryDirectory() as work:
            ir = Path(work) / "toy.hw.mlir"
            ir.write_text("hw.module @Toy()\n")
            path = Path(work) / "out.json"
            facts.write(path, facts.document(ir, items))
            return json.loads(path.read_text()), hashlib.sha256(ir.read_bytes()).hexdigest(), str(ir.resolve())

    def test_document_identity_is_schema_and_ir_only(self):
        document, digest, path = self.blocks(facts.record(self.SPEC, "pulse", result=self.RESULT, evidence="simulated"))
        self.assertEqual(set(document), {"schema", "hw_ir", "op_timing"})
        self.assertEqual(document["schema"], "merlin.op_timing.v1")
        self.assertEqual(document["hw_ir"], {"path": path, "sha256": digest})

    def test_computed_block_follows_the_timing_conventions(self):
        block = self.blocks(facts.record(self.SPEC, "pulse", "slow", self.RESULT, evidence="simulated"))[0]["op_timing"][0]
        self.assertEqual((block["name"], block["module"], block["engine"], block["operation"], block["variant"]),
                         ("toy.pulse/slow", "Toy", "toy", "pulse", "slow"))
        self.assertEqual((block["source"], block["evidence"]), ("control_simulation", "simulated"))
        self.assertEqual((block["events"], block["first_free_age"], block["next_issue_age"]), (self.EVENTS, 7, 5))
        self.assertEqual(block["assumptions"], {"scratchpad_read_latency": 2, "reset_cycles": 4, "flush_cycles": 4, "limit": 32})
        self.assertFalse({"status", "reason"} & set(block))

    def test_variantless_engine_without_next_issue_omits_both_keys(self):
        spec = {**self.SPEC, "operations": {"plain": self.SPEC["operations"]["plain"]}}
        block = self.blocks(facts.record(spec, "plain", result={"events": self.EVENTS, "first_free_age": 2}))[0]["op_timing"][0]
        self.assertEqual(block["name"], "toy.plain")
        self.assertFalse({"variant", "next_issue_age"} & set(block))

    def test_unresolved_block_is_null_with_the_reason_as_evidence(self):
        block = self.blocks(facts.record(self.SPEC, "pulse", reason="Busy did not clear within 32 ages"))[0]["op_timing"][0]
        self.assertEqual((block["events"], block["first_free_age"], block["next_issue_age"]), (None, None, None))
        self.assertEqual(block["evidence"], "Busy did not clear within 32 ages")

    def test_failed_check_unresolves_the_block(self):
        record = facts.record(self.SPEC, "pulse", result=copy.deepcopy(self.RESULT), evidence="simulated")
        facts.unresolve(record, "Check failed: a == b")
        block = facts.block(record)
        self.assertEqual((block["events"], block["first_free_age"], block["next_issue_age"]), (None, None, None))
        self.assertEqual(block["evidence"], "Check failed: a == b")


if __name__ == "__main__":
    unittest.main()
