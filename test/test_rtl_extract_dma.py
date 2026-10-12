"""DMA concurrency evidence (tools/rtl_extract/targets/atlas/system/core_dma.yaml, pairs/core_dma.yaml).

Synthetic tests cover the responder runner (a declared external memory serving a circuit's request/response
handshakes) and ``unreset`` regex selectors. Hardware tests (ATLAS_HW_IR, ATLAS_HW_EXPORTER) check a window around
each expected pair result and the DMA-specific claims: a launch on a second channel is queued, never dropped or
merged; VLS on other lines keeps its isolated events while the transfer's accesses are only delayed; channel
completion is independent; same-channel reuse is not protected by the RTL; and the disjoint-bank results hold under
other response patterns.
"""
from collections import Counter
import copy
import os
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import test_rtl_extract_synthetic as synthetic
from rtl_extract import derive, pairs, runner
from rtl_extract.ir import load_document
from rtl_extract.runner import modules
from rtl_extract.spec import load_spec

HW_IR = os.environ.get("ATLAS_HW_IR")
EXPORTER = os.environ.get("ATLAS_HW_EXPORTER")
op = synthetic.operation


def client():
    """``go`` raises ``req_valid`` (tag 5) until accepted, then waits for the response; ``resp_ready`` toggles every
    cycle, so a response offered on an even age is held to the next odd one. ``done`` fires only for tag 5."""
    return synthetic.module("Client", ["go", "req_ready", "resp_valid", ("resp_tag", "i3")],
                            [("req_valid", "want"), ("req_tag", "five", "i3"), ("resp_ready", "rdy"), ("done", "done"), ("busy", "busy")], [
        synthetic.zero(), synthetic.one(), op("five", "hw.constant", [], "i3", value="5 : i3"),
        op("want", "seq.firreg", ["want_next", "clock", "reset", "zero"], name="want"),
        op("waiting", "seq.firreg", ["wait_next", "clock", "reset", "zero"], name="waiting"),
        op("rdy", "seq.firreg", ["rdy_next", "clock", "reset", "zero"], name="rdy"),
        op("rdy_next", "comb.xor", ["rdy", "one"]),
        op("fire", "comb.and", ["want", "req_ready"]),
        op("resp_fire", "comb.and", ["resp_valid", "rdy"]),
        op("hold", "comb.mux", ["fire", "zero", "want"]),
        op("want_next", "comb.mux", ["go", "one", "hold"]),
        op("wait_hold", "comb.mux", ["resp_fire", "zero", "waiting"]),
        op("wait_next", "comb.mux", ["fire", "one", "wait_hold"]),
        op("tagged", "comb.icmp", ["resp_tag", "five"], predicate="0 : i64"),
        op("done", "comb.and", ["resp_fire", "tagged"]),
        op("busy", "comb.or", ["want", "waiting"])])


SPEC = {"engine": "client", "module": "Client", "inputs": {"go": 0}, "busy": "busy", "limit": 24,
        "events": {"done": {"valid": "done"}},
        "responders": {"mem": {"request": {"valid": "req_valid", "ready": "req_ready", "tag": "req_tag"},
                               "response": {"valid": "resp_valid", "ready": "resp_ready", "tag": "resp_tag"},
                               "latency": 3, "accept": [0, 1]}},
        "operations": {"go": {"commands": {0: {"go": 1}}}}}


def events_of(trace):
    return sorted((e["age"], group, fields) for e in trace for group, fields in e["events"].items())


class ResponderTests(unittest.TestCase):
    def run_client(self, **overlay):
        spec = synthetic.load({**SPEC, "responders": {"mem": {**SPEC["responders"]["mem"], **overlay}}})
        document = {"modules": [client()]}
        circuit = runner.build_single(document, spec)
        return runner.run_single(circuit, spec, "go"), spec

    def test_request_waits_for_accept_and_response_is_held_until_ready(self):
        trace, _ = self.run_client()
        # go at 0; the request is valid from age 1 and accepted at the first odd age; due at 4, ready only at odd ages.
        self.assertEqual(events_of(trace), [(1, "mem.request", {"tag": 5}), (5, "done", {}), (5, "mem.response", {"tag": 5})])
        self.assertEqual([e["signals"]["resp_ready"] for e in trace[:6]], [0, 1, 0, 1, 0, 1])
        self.assertEqual(runner.summarize(trace, _, "go")["first_free_age"], 6)

    def test_accept_pattern_and_latency_move_the_handshakes(self):
        trace, _ = self.run_client(accept=[0, 0, 0, 1], latency=2)
        self.assertEqual([(a, g) for a, g, _ in events_of(trace)], [(3, "mem.request"), (5, "done"), (5, "mem.response")])

    def test_request_fields_are_recorded_and_must_be_known(self):
        trace, _ = self.run_client(request={"valid": "req_valid", "ready": "req_ready", "tag": "req_tag", "fields": {"copy": "req_tag"}})
        self.assertEqual(events_of(trace)[0], (1, "mem.request", {"copy": 5, "tag": 5}))

    def test_responder_signals_are_observed_and_control(self):
        spec = synthetic.load(SPEC)
        self.assertIn("req_valid", runner.control_signals(spec))
        self.assertNotIn("resp_ready", runner.control_signals(spec))
        self.assertTrue({"req_valid", "req_tag", "resp_ready"} <= set(runner.observed(spec)))
        self.assertEqual(spec["inputs"], {"go": 0, "req_ready": 0, "resp_valid": 0, "resp_tag": 0})
        self.assertEqual(runner.group_valid(spec, "mem.request"), "req_valid")
        self.assertEqual(runner.group_valid(spec, "mem.response"), "resp_ready")
        self.assertEqual(runner.group_valid(spec, "done"), "done")
        with self.assertRaises(ValueError):
            runner.group_valid(spec, "other.request")

    def test_invalid_responders_are_rejected(self):
        mem = SPEC["responders"]["mem"]
        bad = {"latency zero": {**mem, "latency": 0},
               "accept level": {**mem, "accept": [2]},
               "accept empty": {**mem, "accept": []},
               "unknown key": {**mem, "order": "any"},
               "response fields": {**mem, "response": {**mem["response"], "fields": {"x": "req_tag"}}},
               "field named tag": {**mem, "request": {**mem["request"], "fields": {"tag": "req_tag"}}},
               "missing ready": {**mem, "request": {"valid": "req_valid"}}}
        for label, responder in bad.items():
            with self.subTest(label), self.assertRaises(ValueError):
                synthetic.load({**SPEC, "responders": {"mem": responder}})
        with self.assertRaises(ValueError):
            synthetic.load({**SPEC, "responders": {"done": mem}})

    def test_pair_sweep_compares_handshakes(self):
        # The client against itself: a second go at an even gap requests at g+1 (odd, accepted) and is answered at
        # g+5 as in isolation; at an odd gap the request slips to g+2, which the handshake comparison rejects.
        spec = synthetic.load(SPEC)
        document = {"modules": [client()]}
        import tempfile
        import yaml
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "client.yaml"
            path.write_text(yaml.safe_dump({"engine": "client", "pairs": {"p": {"first": "go", "second": "go", "gaps": "1..6"}}}))
            raw = pairs.load_file(path, spec)
        record = pairs.extract(document, raw, spec)[0]
        self.assertEqual([(r["from"], r["result"]) for r in record["results"]],
                         [(g, "accepted" if g % 2 == 0 else "missing mem.request events") for g in range(1, 7)])
        self.assertEqual((record["min_gap"], record["stable_gap"]), (2, 6))


class UnresetMatchTests(unittest.TestCase):
    def masked(self):
        """``flag`` is a reset valid ANDed with two hold registers that stay unknown until the first command."""
        return synthetic.module("M", ["cmd"], [("flag", "flag")], [
            synthetic.zero(), synthetic.one(), op("valid", "seq.firreg", ["cmd", "clock", "reset", "zero"], name="valid"),
            op("next_a", "comb.mux", ["cmd", "one", "meta_a"]), op("meta_a", "seq.firreg", ["next_a", "clock"], name="meta_a"),
            op("next_b", "comb.mux", ["cmd", "one", "meta_b"]), op("meta_b", "seq.firreg", ["next_b", "clock"], name="meta_b"),
            op("both", "comb.and", ["meta_a", "meta_b"]), op("flag", "comb.and", ["valid", "both"])])

    def record(self, unreset):
        value = {"engine": "meta", "module": "M", "limit": 6, "inputs": {"cmd": 0}, "unreset": unreset,
                 "events": {"flag": {"valid": "flag"}}, "operations": {"pulse": {"commands": {0: {"cmd": 1}}}}}
        return runner.extract({"modules": [self.masked()]}, synthetic.load(value))[0]

    def test_match_selects_every_register_it_names(self):
        record = self.record([{"path": "", "match": "meta_[ab]"}])
        self.assertEqual((record["status"], record["events"]["flag"]["first_age"]), ("computed", 1), record.get("reason"))
        self.assertEqual(self.record([{"path": "", "match": "meta_a"}, {"path": "", "name": "meta_b"}])["status"], "computed")

    def test_match_without_registers_or_with_a_known_register_fails(self):
        self.assertIn("matches 0 registers", self.record([{"path": "", "match": "meta_[ab]"}, {"path": "", "match": "nothing.*"}])["reason"])
        self.assertIn("known after reset/flush", self.record([{"path": "", "match": "(meta_[ab]|valid)"}])["reason"])
        self.assertIn("matches 0 registers", self.record([{"path": "", "match": "meta"}])["reason"])  # whole-name match

    def test_selector_takes_one_of_name_and_match(self):
        for selector in ({"path": "", "name": "meta_a", "match": "meta_b"}, {"path": ""}):
            with self.subTest(selector), self.assertRaises(ValueError):
                synthetic.load({"engine": "meta", "module": "M", "inputs": {"cmd": 0}, "unreset": [selector],
                                "events": {"flag": {"valid": "flag"}}, "operations": {"pulse": {"commands": {0: {"cmd": 1}}}}})


def dma_groups(trace):
    """Multiset of DMA events without ages or source tags (tags count up across transfers): (group, fields)."""
    return Counter((g, tuple(sorted((k, v) for k, v in f.items() if k != "tag"))) for e in trace for g, f in e["events"].items()
                   if g.startswith("dma_") or g.startswith("dram."))


def vls_events(trace, shift=0):
    return sorted((g, e["age"] - shift, tuple(sorted(f.items()))) for e in trace for g, f in e["events"].items()
                  if not (g.startswith("dma_") or g.startswith("dram.")))


def span(trace, signal):
    ages = [e["age"] for e in trace if e["signals"][signal]]
    return (ages[0], ages[-1]) if ages else None


@unittest.skipUnless(HW_IR and EXPORTER, "requires ATLAS_HW_IR and ATLAS_HW_EXPORTER")
class HardwareDMATests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.raw, cls.spec = pairs.load("atlas", ["core_dma"])["core_dma"]
        cls.document = load_document(HW_IR, modules(cls.spec), EXPORTER, True)
        cls.concrete, _ = derive.resolve(cls.document, cls.spec, {})
        cls.circuit = runner.build_single(cls.document, cls.concrete)

    def paired(self, first, second, gap, spec=None, second_inputs=None):
        spec = spec or self.concrete
        timeline = {**pairs.commands(spec, first, {}), **pairs.commands(spec, second, second_inputs or {}, gap)}
        trace, problem = pairs.run(self.circuit, spec, timeline)
        self.assertIsNone(problem)
        self.assertEqual(runner.failures(self.circuit, trace), [])
        return trace

    def alone(self, name, spec=None):
        spec = spec or self.concrete
        trace, problem = pairs.run(self.circuit, spec, pairs.commands(spec, name, {}))
        self.assertIsNone(problem)
        return trace

    def test_pair_windows(self):
        window = copy.deepcopy(self.raw)
        for pair, expected in self.raw["expect"].items():
            stable = expected["stable_gap"]
            window["pairs"][pair]["gaps"] = "1..2" if stable is None else f"{max(1, stable - 1)}..{stable + 1}"
        for record in pairs.extract(self.document, window, self.spec, jobs=8):
            pair = record["name"].split(".", 1)[1]
            expected = self.raw["expect"][pair]
            with self.subTest(record["name"]):
                self.assertEqual(record["invariants"]["checked"], 177)
                self.assertEqual(record["stable_gap"], expected["stable_gap"], record["results"])
                self.assertEqual(record["min_gap"], expected["stable_gap"], record["results"])  # first accepted in the window
                if "rejected_by" in expected:
                    self.assertRegex(record["results"][0]["result"], expected["rejected_by"])
                if "disjoint" in expected:
                    self.assertEqual(record["shared_control"]["disjoint"], expected["disjoint"], record["shared_control"])

    def test_second_channel_launch_is_queued_not_dropped(self):
        first, second = self.alone("dma_load0"), self.alone("dma_load1")
        trace = self.paired("dma_load0", "dma_load1", 1)
        # Every beat of both transfers happens exactly once: 64 requests with distinct addresses, 64 distinct writes.
        self.assertEqual(dma_groups(trace), dma_groups(first) + dma_groups(second))
        requests = [f for e in trace for g, f in e["events"].items() if g == "dram.request"]
        self.assertEqual(len({f["address"] for f in requests}), 64)
        # Requests are dispatched in launch order: the second transfer's first beat follows the first's last.
        ages = [e["age"] for e in trace for g in e["events"] if g == "dram.request"]
        self.assertEqual(ages, sorted(ages))
        self.assertEqual([f["address"] >> 12 & 1 for f in requests], [0] * 32 + [1] * 32)
        # Channel busy is per channel: channel 0 retires while channel 1 is still in flight; channel 1 busy from launch.
        self.assertEqual(span(trace, "dma_channel0"), span(first, "dma_channel0"))
        self.assertEqual(span(trace, "dma_channel1")[0], 2)
        self.assertGreater(span(trace, "dma_channel1")[1], span(trace, "dma_channel0")[1])
        self.assertEqual(span(trace, "dma_launched1"), (2, 2))

    def test_vls_on_other_lines_keeps_its_events_while_the_dma_waits(self):
        first, second = self.alone("dma_load0"), self.alone("vload_bank3")
        gap = 40
        trace = self.paired("dma_load0", "vload_bank3", gap)
        self.assertEqual(vls_events(trace, gap), vls_events(second))
        self.assertEqual(dma_groups(trace), dma_groups(first))
        self.assertNotEqual(pairs.events(trace, {}), sorted(pairs.events(first, {}) + pairs.events(second, {}, gap)))
        writes = lambda t: [e["age"] for e in t if "dma_vmem_write" in e["events"]]
        self.assertGreater(writes(trace)[-1], writes(first)[-1])
        # On another bank nothing moves.
        trace = self.paired("dma_load0", "vload", gap)
        self.assertEqual(pairs.events(trace, {}), sorted(pairs.events(first, {}) + pairs.events(self.alone("vload"), {}, gap)))

    def test_dma_store_reads_wait_for_the_lsu(self):
        first, second = self.alone("dma_store0"), self.alone("vload_bank3")
        trace = self.paired("dma_store0", "vload_bank3", 1)
        self.assertEqual(vls_events(trace, 1), vls_events(second))
        self.assertEqual(dma_groups(trace), dma_groups(first))
        reads = lambda t: [e["age"] for e in t if "dma_vmem_read" in e["events"]]
        self.assertEqual(reads(first), list(range(1, 33)))
        self.assertGreater(reads(trace)[-1], 33)

    def test_same_channel_reuse_is_not_protected(self):
        first = self.alone("dma_load0")
        trace = self.paired("dma_load0", "dma_load0", 10, second_inputs={"dma_vmem_addr": 197120})
        # Channel 0 busy falls when the first transfer retires although the second transfer's beats are outstanding.
        self.assertEqual(span(trace, "dma_channel0"), span(first, "dma_channel0"))
        later = [e["age"] for e in trace if "dma_vmem_write" in e["events"] and e["age"] > span(first, "dma_channel0")[1]]
        self.assertEqual(len(later), 32)

    def test_disjoint_bank_results_hold_under_other_response_patterns(self):
        patterns = (("fast", {"accept": [1], "latency": 1}, 450), ("stalled", {"accept": [1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0], "latency": 43}, 450),
                    ("slow", {"accept": [1] + [0] * 9, "latency": 61}, 900))
        for label, overlay, limit in patterns:
            spec = copy.deepcopy(self.spec)
            spec["responders"]["dram"].update(overlay)
            spec["limit"] = limit
            window = copy.deepcopy(self.raw)
            window["pairs"] = {"load0_vload": {**self.raw["pairs"]["load0_vload"], "gaps": "1..2"},
                               "load0_add": {**self.raw["pairs"]["load0_add"], "gaps": "1..2"},
                               "load0_load1": {**self.raw["pairs"]["load0_load1"], "gaps": "1..1"}}
            results = {r["name"]: r for r in pairs.extract(self.document, window, spec, jobs=5)}
            with self.subTest(label):
                self.assertEqual(results["core_dma.load0_vload"]["stable_gap"], 1, results["core_dma.load0_vload"]["results"])
                self.assertEqual(results["core_dma.load0_add"]["stable_gap"], 1, results["core_dma.load0_add"]["results"])
                self.assertTrue(results["core_dma.load0_vload"]["shared_control"]["disjoint"])
                self.assertIn("missing dram.request events", results["core_dma.load0_load1"]["results"][0]["result"])


if __name__ == "__main__":
    unittest.main()
