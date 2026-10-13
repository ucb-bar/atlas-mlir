"""Pair, sequence and pair-fact evidence: validation, synthetic cases and hardware windows."""
import contextlib
import copy
import hashlib
import importlib.util
import io
from functools import partial
import re
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
import yaml
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import test_rtl_extract_synthetic as synthetic
from rtl_extract import derive, pairfacts, pairs, sequences
from rtl_extract.ir import load_document
from rtl_extract.runner import RECIPES, modules
from rtl_extract.spec import TARGETS, ages
from rtl_extract.spec import read_spec as resolved_spec
HW_IR = os.environ.get("ATLAS_HW_IR")
EXPORTER = os.environ.get("ATLAS_HW_EXPORTER")
op = synthetic.operation

def counter(accept_while_busy=False):
    """A busy counter ignores go, or restarts when accept_while_busy is set."""
    return synthetic.module("Toy", ["go"], [("out", "busy"), ("busy", "busy")], [
        op("zero", "hw.constant", [], "i2", value="0 : i2"), op("one", "hw.constant", [], "i2", value="1 : i2"),
        op("three", "hw.constant", [], "i2", value="3 : i2"),
        op("cnt", "seq.firreg", ["next", "clock", "reset", "zero"], "i2", name="cnt"),
        op("idle", "comb.icmp", ["cnt", "zero"], predicate="0 : i64"),
        op("busy", "comb.icmp", ["cnt", "zero"], predicate="1 : i64"), op("dec", "comb.sub", ["cnt", "one"], "i2"),
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

    def check(self, name, jobs=8):
        raw, spec = pairs.load("atlas", [name])[name]
        document = load_document(HW_IR, modules(spec), EXPORTER, pairs.needs_assertions(raw, spec))
        window = copy.deepcopy(raw)
        for pair, expected in raw["expect"].items():
            stable = expected["stable_gap"]
            window["pairs"][pair]["gaps"] = f"{max(1, stable - 1)}..{stable + 1}"
        for record in pairs.extract(document, window, spec, jobs=jobs):
            pair = record["name"].split(".", 1)[1]
            expected = raw["expect"][pair]
            stable = expected["stable_gap"]
            with self.subTest(record["name"]):
                self.assertEqual(record["stable_gap"], stable, record["results"])
                if stable > 1:
                    self.assertEqual(record["min_gap"], stable, record["results"])
                if "rejected_by" in expected:
                    self.assertRegex(record["results"][0]["result"], expected["rejected_by"])
                if "disjoint" in expected:
                    self.assertEqual(record["shared_control"]["disjoint"], expected["disjoint"], record["shared_control"])



    def test_pair_windows(self):
        for name in ("vlsu", "vpu", "core", "mxu0", "mxu1", "core_mxu", "core_mxu_cross", "core_mxu_units", "core_mxu_shared", "core_mxu_same"):
            with self.subTest(spec=name):
                self.check(name)

    CORE_VPU_SAMPLE = ("exp_relu relu_exp exp_exp2 exp_add add_relu sin_cos rmax_exp rsum_relu csum_relu csum_csum csum_add "
                       "cmax_pairmax fp8pack_relu fp8unpack_relu fp8unpack_fp8unpack vliAll_relu vliAll_vliCol vliOne_vliRow "
                       "relu_vliOne mov_vliAll").split()

    def test_core_vpu_sample(self):
        """One window per VPU timing class, bound by issue-busy assertions and next_issue_age."""
        raw, spec = pairs.load("atlas", ["core_vpu"])["core_vpu"]
        document = load_document(HW_IR, modules(spec), EXPORTER, True)
        window = copy.deepcopy(raw)
        window["pairs"] = {n: window["pairs"][n] for n in self.CORE_VPU_SAMPLE}
        for name, pair in window["pairs"].items():
            stable = raw["expect"][name]["stable_gap"]
            pair["gaps"] = f"{max(1, stable - 1)}..{stable + 1}"
        for record in pairs.extract(document, window, spec, jobs=8):
            name = record["name"].split(".", 1)[1]
            expected = raw["expect"][name]
            with self.subTest(name):
                self.assertEqual(record["invariants"], {"checked": 177, "excluded": []})
                self.assertEqual(record["stable_gap"], expected["stable_gap"], record["results"])
                self.assertLessEqual(record["stable_gap"], expected["next_issue_age"])
                if expected["stable_gap"] > 1:
                    self.assertEqual(record["min_gap"], expected["stable_gap"], record["results"])
                    self.assertIn("issue-busy bit is set", record["results"][0]["result"])








def queue():
    """Two commands may occupy the four-stage queue; further launches are ignored."""
    stages = [op(f"sr{i}", "seq.firreg", [("accept" if i == 0 else f"sr{i - 1}"), "clock", "reset", "zero"], name=f"sr{i}") for i in range(4)]
    return synthetic.module("Queue", ["go"], [("out", "sr3"), ("busy", "busy")], [
        synthetic.zero(), synthetic.one(), *stages,
        op("ab", "comb.and", ["sr0", "sr1"]), op("ac", "comb.and", ["sr0", "sr2"]), op("bc", "comb.and", ["sr1", "sr2"]),
        op("m", "comb.or", ["ab", "ac"]), op("full", "comb.or", ["m", "bc"]), op("free", "comb.xor", ["full", "one"]),
        op("accept", "comb.and", ["go", "free"]),
        op("lo", "comb.or", ["sr0", "sr1"]), op("hi", "comb.or", ["sr2", "sr3"]), op("busy", "comb.or", ["lo", "hi"])])
QUEUE_SPEC = {"engine": "toy", "module": "Queue", "inputs": {"go": 0}, "busy": "busy", "limit": 24,
              "events": {"out": {"valid": "out"}}, "operations": {"go": {"commands": {0: {"go": 1}}}}}

def sequence_sweep(module, sequence, spec=QUEUE_SPEC, extra=None):
    """``{length: record}`` for one sequence of a synthetic engine."""
    spec_ = synthetic.load(spec)
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "toy.yaml"
        path.write_text(yaml.safe_dump({"engine": "toy", "sequences": {"s": sequence}, **(extra or {})}))
        raw = sequences.load_file(path, spec_)
    return {r["length"]: r for r in sequences.extract({"modules": [module]}, raw, spec_)}

def chains(module, sequence, invariants=None, spec=synthetic.TWO):
    """``{length: record}`` for one sequence of the two-counter engine, whose one assertion fires when B is issued while A is busy."""
    spec_ = synthetic.load(spec)
    raw = {"engine": "two", "sequences": {"s": sequence}, **({} if invariants is None else {"invariants": invariants})}
    return {r["length"]: r for r in sequences.extract({"modules": [module]}, sequences.validate(raw, spec_, "two.yaml"), spec_)}

class InvariantSequenceTests(unittest.TestCase):
    ABA = {"ops": [{"op": "a"}, {"op": "b"}], "lengths": "2..3", "gaps": "1..6"}

    def test_firing_invariant_rejects_chain_spacings(self):
        records = chains(synthetic.engines(), self.ABA, {})
        self.assertEqual({n: (r["min_gap"], r["stable_gap"]) for n, r in records.items()}, {2: (4, 4), 3: (4, 4)})
        self.assertEqual(records[2]["results"][0], {"from": 1, "to": 3, "result": "invariant B issued while A busy (/) fired"})
        self.assertEqual(records[3]["invariants"], {"checked": 1, "excluded": []})
        self.assertNotIn("invariants", chains(synthetic.engines(), self.ABA)[2])
        self.assertEqual(chains(synthetic.engines(), self.ABA)[2]["stable_gap"], 1)

    def test_excluded_invariant_is_accepted_with_its_reason(self):
        record = chains(synthetic.engines(), self.ABA, {"exclude": [{"message": "B issued", "reason": "independent counters"}]})[2]
        self.assertEqual(record["stable_gap"], 1)
        self.assertEqual(record["invariants"]["excluded"], [{"path": "", "message": "B issued while A busy", "reason": "independent counters"}])

    def test_chain_and_pair_agree_on_the_outcome(self):
        """One acceptance implementation: a two-operation chain and the pair of the same operations give the same reasons."""
        record = chains(synthetic.engines(), self.ABA, {})[2]
        pair = synthetic.two_pairs(synthetic.engines(), {})
        self.assertEqual((record["results"], record["min_gap"], record["stable_gap"]), (pair["results"], pair["min_gap"], pair["stable_gap"]))

    def test_unknown_invariant_fails_closed(self):
        with self.assertRaisesRegex(ValueError, r"a alone: invariant B issued while A busy \(/\) unknown"):
            chains(synthetic.engines(unknown=True), self.ABA, {})

    def test_invariants_fail_closed(self):
        spec = synthetic.load(synthetic.TWO)
        for label, raw, message in ( ("missing reason", {"exclude": [{"message": "x"}]}, "missing keys"), ("unknown key", {"only": []}, "unknown keys")):
            with self.subTest(label), self.assertRaisesRegex(ValueError, message):
                sequences.validate({"engine": "two", "invariants": raw, "sequences": {"s": self.ABA}}, spec, "two.yaml")
        with self.assertRaisesRegex(ValueError, "match no assertion"):
            chains(synthetic.engines(), self.ABA, {"exclude": [{"message": "nothing", "reason": "r"}]})
        with self.assertRaisesRegex(ValueError, "exported without assertions"):
            chains({k: v for k, v in synthetic.engines().items() if k != "assertions"}, self.ABA, {})

    def test_prepared_spec_replaces_the_spec_invariants(self):
        spec = synthetic.load({**synthetic.TWO, "invariants": {}})
        raw = sequences.validate({"engine": "two", "invariants": {"exclude": [{"message": "B issued", "reason": "r"}]}, "sequences": {"s": self.ABA}}, spec, "two.yaml")
        self.assertTrue(pairs.needs_assertions(raw, spec))
        records = sequences.extract({"modules": [synthetic.engines()]}, raw, spec)
        self.assertEqual(records[0]["stable_gap"], 1)

class SyntheticSequenceTests(unittest.TestCase):

    def test_two_deep_queue_rejects_the_third_command(self):
        records = sequence_sweep(queue(), {"ops": [{"op": "go"}], "lengths": "2..4", "gaps": "1..5"})
        self.assertEqual({n: r["stable_gap"] for n, r in records.items()}, {2: 1, 3: 2, 4: 2})
        self.assertEqual(records[2]["results"], [{"from": 1, "to": 5, "result": "accepted"}])
        self.assertEqual(records[3]["results"][0], {"from": 1, "to": 1, "result": "missing out x1 (ops [2])"})
        self.assertEqual(records[4]["results"][0], {"from": 1, "to": 1, "result": "missing out x2 (ops [2, 3])"})

    def test_restart_merges_events_until_streams_abut(self):
        spec = SPEC
        records = sequence_sweep(counter(True), {"ops": [{"op": "go", "guards": []}], "lengths": "2..3", "gaps": "1..6"}, spec)
        self.assertEqual({n: (r["min_gap"], r["stable_gap"]) for n, r in records.items()}, {2: (3, 3), 3: (3, 3)})
        self.assertRegex(records[3]["results"][0]["result"], "^missing out")

    def test_guard_applies_to_every_operation_after_the_first(self):
        records = sequence_sweep(counter(True), {"ops": [{"op": "go"}], "lengths": 3, "gaps": "1..6"}, SPEC)
        self.assertEqual(records[3]["stable_gap"], 4)
        self.assertEqual(records[3]["results"][0]["result"], "guard busy set at op 1; guard busy set at op 2; missing out x4 (ops [1, 2])")
        self.assertEqual(records[3]["results"][2], {"from": 3, "to": 3, "result": "guard busy set at op 1; guard busy set at op 2; events unchanged"})

    def test_command_ages_collide(self):
        spec = copy.deepcopy(QUEUE_SPEC)
        spec["operations"]["go"]["commands"] = {0: {"go": 1}, 1: {"go": 1}}
        records = sequence_sweep(queue(), {"ops": [{"op": "go"}], "lengths": 2, "gaps": "1..3"}, spec)
        self.assertEqual(records[2]["results"][0], {"from": 1, "to": 1, "result": "command ages collide"})

    def test_lost_positions_and_difference(self):
        a, b = ("out", 1, ()), ("out", 3, ())
        self.assertEqual(sequences.lost([[a], [b], [b]], [a, b]), [2])
        self.assertEqual(sequences.lost([[a], [b]], [a, b]), [])
        self.assertEqual(sequences.difference([a, b], [a, ("x", 2, ())]), "missing out x1; unexpected x x1")

    def test_cycle_and_timeline(self):
        spec = synthetic.load(QUEUE_SPEC)
        entries = [{"op": "go", "inputs": {"go": 1}}, {"op": "go"}]
        steps = sequences.chain({"ops": entries}, 3)
        self.assertEqual([s is e for s, e in zip(steps, [entries[0], entries[1], entries[0]])], [True] * 3)
        merged, schedules = sequences.timeline(spec, steps, 4)
        self.assertEqual(sorted(merged), [0, 4, 8])
        self.assertIsNone(sequences.timeline(spec, steps, 0)[0])

    def test_sequence_files_fail_closed(self):
        spec = synthetic.load(QUEUE_SPEC)
        base = {"ops": [{"op": "go"}], "lengths": "2..3", "gaps": "1..4"}
        for label, raw, message in (
                ("unknown operation", {"engine": "toy", "sequences": {"s": {**base, "ops": [{"op": "stop"}]}}}, "unknown operation"),
                ("undeclared input", {"engine": "toy", "sequences": {"s": {**base, "ops": [{"op": "go", "inputs": {"x": 1}}]}}}, "undeclared inputs"),
                ("non-integer input", {"engine": "toy", "sequences": {"s": {**base, "ops": [{"op": "go", "inputs": {"go": "1"}}]}}}, "map inputs to integers"),
                ("unknown guard", {"engine": "toy", "sequences": {"s": {**base, "ops": [{"op": "go", "guards": [{"signal": "nope"}]}]}}}, "not an observed"),
                ("empty ops", {"engine": "toy", "sequences": {"s": {**base, "ops": []}}}, "non-empty list"),
                ("missing ops", {"engine": "toy", "sequences": {"s": {"lengths": "2", "gaps": "1..4"}}}, "missing keys"),
                ("unknown op key", {"engine": "toy", "sequences": {"s": {**base, "ops": [{"op": "go", "at": 1}]}}}, "unknown keys"),
                ("unknown sequence key", {"engine": "toy", "sequences": {"s": {**base, "extra": 1}}}, "unknown keys"),
                ("length one", {"engine": "toy", "sequences": {"s": {**base, "lengths": "1..3"}}}, "lengths start at 2"),
                ("gap zero", {"engine": "toy", "sequences": {"s": {**base, "gaps": "0..4"}}}, "gaps start at 1"),
                ("bad range", {"engine": "toy", "sequences": {"s": {**base, "gaps": "4"}}}, "Unsupported command age"),
                ("bad name", {"engine": "toy", "sequences": {"a/b": base}}, "invalid sequence name"),
                ("wrong engine", {"engine": "other", "sequences": {"s": base}}, "does not match"),
                ("no sequences", {"engine": "toy", "sequences": {}}, "no sequences"),
                ("bad equivalent", {"engine": "toy", "equivalent": {"x": ["missing"]}, "sequences": {"s": base}}, "equivalent"),
                ("bad ignore", {"engine": "toy", "ignore": ["missing"], "sequences": {"s": base}}, "ignore"),
                ("expect unknown length", {"engine": "toy", "expect": {"s": {9: 3}}, "sequences": {"s": base}}, "must map chain lengths"),
                ("expect non-integer", {"engine": "toy", "expect": {"s": {2: "3"}}, "sequences": {"s": base}}, "must map chain lengths"),
                ("expect unknown sequence", {"engine": "toy", "expect": {"t": {}}, "sequences": {"s": base}}, "unknown sequences")):
            with self.subTest(label), self.assertRaisesRegex(ValueError, message):
                sequences.validate(copy.deepcopy(raw), spec, "toy.yaml")

    def test_target_sequence_files_validate(self):
        loaded = sequences.load("atlas")
        self.assertTrue(loaded)
        for name, (raw, spec) in loaded.items():
            for sequence, lengths in raw["expect"].items():
                with self.subTest(f"{name}.{sequence}"):
                    self.assertLessEqual(set(lengths), set(ages(raw["sequences"][sequence]["lengths"])))
                    self.assertTrue(all(isinstance(g, int) and g <= ages(raw["sequences"][sequence]["gaps"])[-1] for g in lengths.values()))

@unittest.skipUnless(HW_IR and EXPORTER, "requires ATLAS_HW_IR and ATLAS_HW_EXPORTER")
class HardwareSequenceTests(unittest.TestCase):
    JOBS = min(os.cpu_count() or 1, 8)

    def check(self, name):
        raw, spec = sequences.load("atlas", [name])[name]
        document = load_document(HW_IR, modules(spec), EXPORTER, pairs.needs_assertions(raw, spec))
        self.assertTrue(raw["expect"])
        for sequence, lengths in raw["expect"].items():
            for length, expected in lengths.items():
                window = copy.deepcopy(raw)
                window["sequences"] = {sequence: {**raw["sequences"][sequence], "lengths": length, "gaps": f"{max(1, expected - 1)}..{expected + 1}"}}
                record = sequences.extract(document, window, spec, self.JOBS)[0]
                with self.subTest(record["name"]):
                    self.assertEqual(record["stable_gap"], expected, record["results"])
                    if expected > record["gaps"][0]:
                        self.assertEqual(record["min_gap"], expected, record["results"])

    def test_core_mxu(self):
        """Check each MXU window through the core decoder with all assertions enabled."""
        raw, spec = sequences.load("atlas", ["core_mxu"])["core_mxu"]
        document = load_document(HW_IR, modules(spec), EXPORTER, True)
        window = copy.deepcopy(raw)
        window["sequences"] = {f"{name}_{length}": {**raw["sequences"][name], "lengths": length, "gaps": f"{max(1, stable - 1)}..{stable + 1}"}
                               for name, lengths in raw["expect"].items() for length, stable in lengths.items()}
        for record in sequences.extract(document, window, spec, self.JOBS):
            name, length = record["sequence"].rsplit("_", 1)
            expected = raw["expect"][name][int(length)]
            with self.subTest(record["name"]):
                self.assertEqual(record["invariants"], {"checked": 177, "excluded": []})
                self.assertEqual(record["stable_gap"], expected, record["results"])
                if expected > 1:
                    self.assertEqual(record["min_gap"], expected, record["results"])

    def test_mxu0(self):
        self.check("mxu0")

    def test_mxu1(self):
        self.check("mxu1")

    def test_mxu_assertions_bind_chains(self):
        """Accumulator write/load invariants reject gap 64, which event comparison alone accepts."""
        raw, spec = sequences.load("atlas", ["mxu0"])["mxu0"]
        document = load_document(HW_IR, modules(spec), EXPORTER, True)
        window = copy.deepcopy(raw)
        window["sequences"] = {"s": {**raw["sequences"]["push_acc_matmul_acc_same_acc"], "lengths": 3, "gaps": "63..64"}}
        bare = {k: v for k, v in window.items() if k != "invariants"}
        self.assertEqual(sequences.extract(document, bare, spec, self.JOBS)[0]["stable_gap"], 64)
        record = sequences.extract(document, window, spec, self.JOBS)[0]
        self.assertIsNone(record["stable_gap"])
        self.assertRegex(record["results"][-1]["result"], "invariant AccumulationBuffers: compute write and load target the same buffer")
        self.assertEqual(record["invariants"]["excluded"], [])

    def test_mxu1_boundary_block_is_the_chain_at_gap_32(self):
        """matmul_acc_at_32 matches the rejected boundary chain, including its missing accumulator writes."""
        raw, spec = sequences.load("atlas", ["mxu1"])["mxu1"]
        document = load_document(HW_IR, modules(spec), EXPORTER)
        concrete, _ = derive.resolve(document, spec, {})
        circuit = RECIPES[concrete["recipe"]][0](document, concrete)
        sequence = raw["sequences"]["push_acc_matmul_acc_same_acc"]
        steps = sequences.chain(sequence, 2)
        stable = raw["expect"]["push_acc_matmul_acc_same_acc"][2]
        alone = [sequences.isolated(circuit, concrete, raw, s) for s in steps]
        boundary = stable - 1
        self.assertIn("missing", sequences.trial(circuit, concrete, raw, steps, alone, boundary))
        self.assertIsNone(sequences.trial(circuit, concrete, raw, steps, alone, stable))
        merged, _ = sequences.timeline(concrete, steps, boundary)
        chain_trace, _ = pairs.run(circuit, concrete, merged)
        block_trace, _ = pairs.run(circuit, concrete, pairs.commands(concrete, "matmul_acc_at_32", {}))
        self.assertEqual(boundary, 32)
        self.assertEqual(pairs.events(chain_trace, raw["equivalent"], ignore=raw["ignore"]), pairs.events(block_trace, raw["equivalent"], ignore=raw["ignore"]))
FACTS = os.environ.get("ATLAS_OP_TIMING")
SEQUENCES = os.environ.get("ATLAS_VPU_SEQUENCES")
HW = {"path": "ir.mlir", "sha256": "a" * 64}
CHECKED = {"checked": 7, "excluded": []}

def block(name, next_issue):
    return {"name": name, "engine": name.split(".")[0], "operation": name.split(".")[1],
            "events": {"write": {"first_age": 2, "last_age": next_issue, "count": next_issue - 1}},
            "first_free_age": next_issue, "next_issue_age": next_issue}
OP_TIMING = {"schema": "merlin.op_timing.v1", "hw_ir": HW, "op_timing": [block("vpu.exp", 65), block("vpu.relu", 65), block("vpu.sin", 65)]}

def runs(accepted_from, last):
    out = [{"from": 1, "to": accepted_from - 1, "result": "guard set"}] if accepted_from > 1 else []
    return out + [{"from": accepted_from, "to": last, "result": "accepted"}]

def pair(name, stable, invariants=CHECKED):
    record = {"name": f"core.{name}", "engine": "core", "module": "AtlasCore", "first": "add", "second": "add_hi",
              "gaps": [1, 71], "min_gap": stable, "stable_gap": stable, "results": runs(stable, 71)}
    return {**record, "invariants": invariants} if invariants else record

def chain(stable, length=3):
    ops = [{"op": "exp"}, {"op": "relu", "inputs": {"io_cmd_bits_vs1": 18, "io_cmd_bits_vs2": 22, "io_cmd_bits_vd": 26}},
           {"op": "exp", "inputs": {"io_cmd_bits_vs1": 34, "io_cmd_bits_vs2": 38, "io_cmd_bits_vd": 42}},
           {"op": "relu", "inputs": {"io_cmd_bits_vs1": 50, "io_cmd_bits_vs2": 54, "io_cmd_bits_vd": 58}}][:length]
    return {"name": f"vpu.exp_relu_exp/{length}", "engine": "vpu", "module": "VectorEngineTop", "sequence": "exp_relu_exp",
            "length": length, "ops": ops, "gaps": [1, 135], "min_gap": stable, "stable_gap": stable,
            "results": runs(stable, 135), "invariants": CHECKED}

class PairFactsTests(unittest.TestCase):

    def build(self, records, chains=(), op_timing=OP_TIMING, hw=HW):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "facts.json").write_text(json.dumps(op_timing))
            (root / "pairs-core_vpu.json").write_text(json.dumps({"schema": "rtl_extract.op_pairs.v1", "hw_ir": hw, "op_pairs": records}))
            sequences = []
            if chains:
                sequences = [root / "sequences-core_vpu.json"]
                sequences[0].write_text(json.dumps({"schema": "rtl_extract.op_sequences.v1", "hw_ir": hw, "op_sequences": list(chains)}))
            return pairfacts.build(root / "facts.json", [root / "pairs-core_vpu.json"], sequences)

    def test_records_name_blocks_and_carry_the_pair_file_relation(self):
        document = self.build([pair("exp_relu", 1), pair("exp_exp", 65)])
        self.assertEqual(document["schema"], "merlin.op_pairs.v1")
        self.assertEqual(document["hw_ir"], HW)
        first, second = document["op_pairs"]
        self.assertEqual((first["first"], first["second"], first["engine"]), ("vpu.exp", "vpu.relu", "vpu"))
        self.assertEqual(first["relation"], {"mreg_banks": "disjoint"})
        self.assertEqual((first["stable_gap"], first["gaps"], first["assertions"]), (1, [1, 71], {"checked": 7, "excluded": 0}))
        self.assertEqual(second["source"], "core.exp_exp")
        self.assertNotIn("results", first)

    def test_capacity_is_what_the_runs_show(self):
        self.assertEqual(self.build([pair("exp_exp", 65)])["capacity"][0]["live"], 1)
        two = self.build([pair("exp_exp", 65), pair("exp_relu", 1)])["capacity"]
        self.assertEqual(two, [{"engine": "vpu", "live": 2, "window": "next_issue_age", "witness": {"source": "core.exp_relu", "gap": 1}}])
        # exp, relu, exp at 33: the third issues as the first retires, so two stay live; a run accepted at 1 shows three.
        self.assertEqual(self.build([pair("exp_exp", 65)], [chain(33)])["capacity"][0]["live"], 2)
        self.assertEqual(self.build([pair("exp_exp", 65)], [chain(1)])["capacity"][0]["live"], 3)

    def test_inputs_fail_closed(self):
        unchecked = pair("exp_relu", 1, invariants=None)
        cases = {"assertion-checked": [unchecked], "does not match": [{**pair("exp_relu", 1), "first": "sub"}],
            "a second record": [pair("exp_relu", 1), pair("exp_relu", 1)],
            "not a resolved op_timing block": [pair("exp_cos", 1)],}
        for message, records in cases.items():
            with self.subTest(message), self.assertRaisesRegex(ValueError, message):
                self.build(records)
        with self.assertRaisesRegex(ValueError, "different hardware IR"):
            self.build([pair("exp_relu", 1)], hw={"path": "x", "sha256": "b" * 64})
        unresolved = copy.deepcopy(OP_TIMING)
        unresolved["op_timing"][1]["events"] = None
        with self.assertRaisesRegex(ValueError, "not a resolved op_timing block"):
            self.build([pair("exp_relu", 1)], op_timing=unresolved)
        wrong = chain(33)
        wrong["ops"][1]["op"] = "sin"
        with self.assertRaisesRegex(ValueError, "does not match sequences/core_vpu.yaml"):
            self.build([pair("exp_relu", 1)], [wrong])

    def test_written_document_keeps_one_record_per_line(self):
        document = self.build([pair("exp_relu", 1)])
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "op-pairs.json"
            pairfacts.write(path, document)
            self.assertEqual(json.loads(path.read_text()), document)
            self.assertIn('    {"first": "vpu.exp", "second": "vpu.relu"', path.read_text())

def mxu_block(name, last):
    """An MXU block: no next_issue_age, live through its last event age."""
    return {"name": name, "engine": name.split(".")[0], "operation": name.split(".")[1],
            "events": {"compute_busy": {"first_age": 1, "last_age": last, "count": last}, "acc_store": {"count": 0}}}
MXU_TIMING = {"schema": "merlin.op_timing.v1", "hw_ir": HW,
              "op_timing": [block("vpu.exp", 65), block("vpu.relu", 65), block("vpu.sin", 65),
                            mxu_block("mxu0.matmul", 94), mxu_block("mxu0.pop_acc_bf16", 32), mxu_block("mxu0.push_weight", 32)]}

def mxu_pair(name, stable, last=100):
    return {"name": f"core.{name}", "engine": "core", "module": "AtlasCore", "first": "mxu0_matmul", "second": "mxu0_matmul",
            "gaps": [1, last], "min_gap": stable, "stable_gap": stable, "results": runs(stable, last), "invariants": CHECKED}

def mxu_chain(stable, length=3):
    return {"name": f"core.mxu0_matmul_other_acc_same_slot/{length}", "engine": "core", "module": "AtlasCore",
            "sequence": "mxu0_matmul_other_acc_same_slot", "length": length,
            "ops": [{"op": "mxu0_matmul", "inputs": {"instruction": w}} for w in (0x14000077, 0x140080f7, 0x14010077)][:length],
            "gaps": [1, 100], "min_gap": stable, "stable_gap": stable, "results": runs(stable, 100), "invariants": CHECKED}

class MxuPairFactsTests(unittest.TestCase):
    """Same-MXU sweeps (pairs/core_mxu_same.yaml, sequences/core_mxu.yaml): relations and the last-event window."""

    def build(self, files, op_timing=MXU_TIMING):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "facts.json").write_text(json.dumps(op_timing))
            paths = {"pairs": [], "sequences": []}
            for n, (stem, records) in enumerate(files):
                kind = "pairs" if stem.startswith("pairs-") else "sequences"
                schema = "rtl_extract.op_pairs.v1" if kind == "pairs" else "rtl_extract.op_sequences.v1"
                path = root / str(n) / f"{stem}.json"
                path.parent.mkdir()
                path.write_text(json.dumps({"schema": schema, "hw_ir": HW, ("op_pairs" if kind == "pairs" else "op_sequences"): records}))
                paths[kind].append(path)
            return pairfacts.build(root / "facts.json", paths["pairs"], paths["sequences"])

    def test_records_carry_accumulator_and_weight_slot_relations(self):
        document = self.build([("pairs-core_mxu_same", [mxu_pair("mxu0_mm_pop16_acc_other", 1), mxu_pair("mxu0_mm_pop16_acc_same", 64), mxu_pair("mxu0_mm_pw_slot_other", 1)])])
        relations = {(r["second"], r["source"]): (r["engine"], r["relation"]) for r in document["op_pairs"]}
        self.assertEqual(relations[("mxu0.pop_acc_bf16", "core.mxu0_mm_pop16_acc_other")], ("mxu0", {"mreg_banks": "disjoint", "accumulator": "other"}))
        self.assertEqual(relations[("mxu0.pop_acc_bf16", "core.mxu0_mm_pop16_acc_same")][1]["accumulator"], "same")
        self.assertEqual(relations[("mxu0.push_weight", "core.mxu0_mm_pw_slot_other")][1], {"mreg_banks": "disjoint", "weight_slot": "other"})

    def test_mxu_capacity_counts_commands_live_through_their_last_event(self):
        late = self.build([("pairs-core_mxu_same", [mxu_pair("mxu0_mm_pop16_acc_same", 95)])])["capacity"]
        self.assertEqual(late, [{"engine": "mxu0", "live": 1, "window": "last_event_age", "witness": {"source": "core.mxu0_mm_pop16_acc_same", "gap": 95}}])
        early = self.build([("pairs-core_mxu_same", [mxu_pair("mxu0_mm_pop16_acc_same", 94)])])["capacity"]
        self.assertEqual((early[0]["live"], early[0]["witness"]["gap"]), (2, 94))
        # Three matmuls alternating accumulators at 32 stay live together; at 48 the third issues after the first.
        pairs = ("pairs-core_mxu_same", [mxu_pair("mxu0_mm_pop16_acc_same", 64)])
        self.assertEqual(self.build([pairs, ("sequences-core_mxu", [mxu_chain(32)])])["capacity"][0]["live"], 3)
        self.assertEqual(self.build([pairs, ("sequences-core_mxu", [mxu_chain(48)])])["capacity"][0]["live"], 2)

    def test_duplicate_records_across_files_fail(self):
        records = [mxu_pair("mxu0_mm_pop16_acc_other", 1)]
        with self.assertRaisesRegex(ValueError, "a second record"):
            self.build([("pairs-core_mxu_same", records), ("pairs-core_mxu_same", records)])

    def test_an_engine_takes_one_window(self):
        mixed = copy.deepcopy(MXU_TIMING)
        mixed["op_timing"][3]["next_issue_age"] = 95
        with self.assertRaisesRegex(ValueError, "take different live windows"):
            self.build([("pairs-core_mxu_same", [mxu_pair("mxu0_mm_pop16_acc_other", 1)])], op_timing=mixed)

@unittest.skipUnless(FACTS and (Path(FACTS).parent / "pairs-core_vpu.json").is_file(), "requires ATLAS_OP_TIMING with pairs-core_vpu.json beside it")
class AtlasPairFactsTests(unittest.TestCase):

    def test_vpu_pairs_and_chains(self):
        sweeps = [Path(FACTS).parent / "pairs-core_vpu.json"]
        document = pairfacts.build(FACTS, sweeps, [SEQUENCES] if SEQUENCES else [])
        records = document["op_pairs"]
        self.assertEqual(len(records), 29 * 29)
        self.assertEqual(sum(r["stable_gap"] == 1 for r in records), 402)
        self.assertTrue(all(r["assertions"]["checked"] == 177 and r["relation"] == {"mreg_banks": "disjoint"} for r in records))
        self.assertEqual([c["live"] for c in document["capacity"] if c["engine"] == "vpu"], [2])
TARGET = TARGETS / "atlas"
MODULE = importlib.util.spec_from_file_location("atlas_gen_core", TARGET / "gen_core.py")
GENERATOR = importlib.util.module_from_spec(MODULE)
MODULE.loader.exec_module(GENERATOR)
SPEC_DIGESTS = {'pairs/core.yaml': 'caece41795d3b6ce93d0c37c6abdb7f16495b8229a356bb212f7b7de1c6fabea',
    'pairs/core_dma.yaml': 'f0f75d54938e13113eb891a8e53a4a7ff2c0a7a10d0ea8de916358d5dc7a28ce',
    'pairs/core_dma_mxu.yaml': '39e3ee2cebdb7907ed233bc935e334f2d16aa4fa32eea9f5fb4774ab8b94b1fe',
    'pairs/core_mxu.yaml': '18bbd1a353bdb88b22a97b60c66f06dad3f803fd51be7febfb1b5a4631db0875',
    'pairs/core_mxu_cross.yaml': 'f95963c4f3e034963ac2a36c91d5a239328427ae3aab255ded7cd137ec38980d',
    'pairs/core_mxu_same.yaml': '07735078c591cd5bafb1d0e88caaf51919d999551b1906d5cd18878063204413',
    'pairs/core_mxu_shared.yaml': '1af02709040d0aa8495c1a6816d7a7a2f180cf1063a167329016833210f84be2',
    'pairs/core_mxu_units.yaml': 'f5d5be30b0113e4d0088f6da2b519815fb135d34646782abfc85e602e6852a7f',
    'pairs/core_vpu.yaml': 'f92c80de22c3aaff1b7eb8629e2c1c81b0a3e0ff6240db7fa7e97fefed6072f6',
    'sequences/core_mxu.yaml': 'c7c388bb53eb9dd2a8ba7632f51cf27e13fd40a05370fe64e7d00f21bb7ca177',
    'sequences/core_vpu.yaml': '98004014df10489a0ffc9ae0eb448605f5ce52424e079338c11eb890ab2c3f44',
    'system/core.yaml': '044782994f4c40324d4f2ed6f6d97b60e63bcd27601b60bb67eee29aa29da0e8',
    'system/core_dma.yaml': '9704652d1f09331021e91168bf205a3ded003098a75c3c094a7f459f31a07386',}


SPEC_ORDER_DIGESTS = {'pairs/core.yaml': '35ea71f0e7cb75783f6ff82bdd11ff14f1e3fc158de2e4737890000f47e0483f',
    'pairs/core_dma.yaml': '3b042e2655ffa655c06aca1fdb9d17cd3389152fcf7139a0e3e061b6eb23dd00',
    'pairs/core_dma_mxu.yaml': '313051a3d85516de3cb4baeff9ae5be229f7c97c1017d6df1b2c334fc393d9c6',
    'pairs/core_mxu.yaml': 'd6d6b8d99fac5a79ee3393b9108c4bfd34c4c7520f83fc3fa12a170271667a98',
    'pairs/core_mxu_cross.yaml': 'f69b8f6267e5f0683c0f279c142723698b07fdc98caed0fe09b0aa8c448a9fb3',
    'pairs/core_mxu_same.yaml': '89faf49321463e4c20b8b724eec6ab63ced1ecc25880fe37bd638977f53362ba',
    'pairs/core_mxu_shared.yaml': '0d38eabab7db61e16081b728c915c9e608c177203823c8bbbeab5430ec03c76a',
    'pairs/core_mxu_units.yaml': '98ec8908929c1fe6676c2f75a8c1f0c611a2ce7abbe39dd92b102e47116625b0',
    'pairs/core_vpu.yaml': '6fa4d311023577883dd10715ba5ed37ea64951efc6cb1c7df51b838f08eddb5e',
    'sequences/core_mxu.yaml': '9ac8b0a6c1462c5640faccd884b18d8bd0a771ea9fc75cf79e74fe72366b6445',
    'sequences/core_vpu.yaml': '8fd8535380559a2f3293e0527856fff0d4b409be8646068039173a6f00bc0674',
    'system/core.yaml': 'fd94a0d853783561813b9397cd8b4e7722b81826e93e51e51a2bebc6a31b7cb8',
    'system/core_dma.yaml': 'd16cf75075c8a0e4948b160685dc2e05500dc0b638a3ceef5f2ccf7e31689617',}

class GeneratorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.specs = {name: resolved_spec(TARGET / path) for name, path in (
            ("vpu", "pairs/core_vpu.yaml"), ("mxu-same", "pairs/core_mxu_same.yaml"),
            ("mxu-shared", "pairs/core_mxu_shared.yaml"), ("mxu-chains", "sequences/core_mxu.yaml"))}
        cls.generators = {"vpu": GENERATOR.vpu_pairs, "mxu-same": GENERATOR.mxu_pairs, "mxu-shared": partial(GENERATOR.mxu_pairs, shared=True), "mxu-chains": GENERATOR.mxu_chains}

    def facts(self, name):
        if name == "vpu":
            words = yaml.safe_load((TARGET / "vpu.yaml").read_text())["decode"]["words"]
            return {"op_timing": [{"engine": "vpu", "operation": op, "next_issue_age": self.specs[name]["expect"][f"{op}_{op}"]["next_issue_age"]} for op in words]}
        ages = {pair["as"]["first"]: int(pair["gaps"].split("..")[1]) - 6 for pair in self.specs["mxu-same"]["pairs"].values()}
        return {"op_timing": [{"name": op, "engine": op.split(".")[0], "events": {
            "active": {"count": 1, "last_age": age}, "inactive": {"count": 0, "last_age": 1000}}} for op, age in ages.items()]}

    def sweep(self, name):
        expected = self.specs[name]["expect"]
        if name == "mxu-chains":
            return {"op_sequences": [{"sequence": seq, "length": length, "stable_gap": gap} for seq, lengths in expected.items() for length, gap in lengths.items()]}
        records = [{"name": f"core.{pair}", **regression} for pair, regression in expected.items()]
        if name != "vpu":
            for record in records:
                gap = record["stable_gap"]
                failure = re.sub(r"\\(.)", r"\1", record.get("rejected_by", "accepted"))
                record["results"] = [{"from": 1, "to": gap - 1, "result": failure + "; later failure"}, {"from": gap, "to": gap + 5, "result": "accepted"}]
        return {"op_pairs": records}

    def generated(self, name, sweep=True, facts=None):
        args = [] if name == "mxu-chains" else [self.facts(name) if facts is None else facts]
        return yaml.safe_load(self.generators[name](*args, *([self.sweep(name)] if sweep else [])))

    def test_canonical_spec_digests(self):
        for path, expected in SPEC_DIGESTS.items():
            with self.subTest(path=path):
                document = resolved_spec(TARGET / path)
                canonical = json.dumps(document, sort_keys=True).encode()
                self.assertEqual(hashlib.sha256(canonical).hexdigest(), expected)
                records = document.get("pairs", document.get("sequences", document.get("operations")))
                order = hashlib.sha256(json.dumps(document).encode()).hexdigest()
                self.assertEqual(order, SPEC_ORDER_DIGESTS[path])
                document.setdefault("invariants", {}).setdefault("exclude", []).append({"message": "mutated", "reason": "test"})
                next(iter(records.values()))["mutated"] = True
                canonical = json.dumps(resolved_spec(TARGET / path), sort_keys=True).encode()
                self.assertEqual(hashlib.sha256(canonical).hexdigest(), expected)

    def test_generators_with_and_without_sweeps(self):
        for name, count in (("vpu", 841), ("mxu-same", 196), ("mxu-shared", 536), ("mxu-chains", 50)):
            with self.subTest(name=name):
                self.assertEqual(self.generated(name), self.specs[name])
                raw = self.generated(name, False)
                key = "sequences" if name == "mxu-chains" else "pairs"
                self.assertEqual(len(raw[key]), count)
                self.assertEqual(raw[key], self.specs[name][key])
                self.assertNotIn("expect", raw)
        self.assertEqual(len(self.sweep("mxu-chains")["op_sequences"]), 172)

    def test_missing_facts_and_incomplete_or_duplicate_sweeps_fail(self):
        for name, generator in self.generators.items():
            args = [] if name == "mxu-chains" else [self.facts(name)]
            if args:
                missing = self.facts(name)
                missing["op_timing"].pop()
                with self.subTest(name=name), self.assertRaisesRegex(ValueError, "29 VPU" if name == "vpu" else "14 MXU"):
                    generator(missing)
                with self.assertRaises((KeyError, ValueError)):
                    GENERATOR.generate(f"pairs/core_{name.replace('-', '_')}.yaml", facts={})
            # VPU historically rejects an incomplete sweep; MXU also rejects duplicates.
            for duplicate in ((False,) if name == "vpu" else (False, True)):
                sweep = self.sweep(name)
                records = sweep["op_sequences" if name == "mxu-chains" else "op_pairs"]
                records.append(records[0]) if duplicate else records.pop()
                message = "every ordered" if name == "vpu" else "every MXU chain length" if name == "mxu-chains" else "exactly every"
                with self.subTest(name=name, duplicate=duplicate), self.assertRaisesRegex(ValueError, message):
                    generator(*args, sweep)

    def test_gap_bound_uses_latest_active_event(self):
        for name in ("mxu-same", "mxu-shared"):
            facts = self.facts(name)
            facts["op_timing"][0]["events"]["active"]["last_age"] = 73
            operation = facts["op_timing"][0]["name"]
            for pair, entry in self.generated(name, False, facts)["pairs"].items():
                expected = "1..79" if entry["as"]["first"] == operation else self.specs[name]["pairs"][pair]["gaps"]
                self.assertEqual(entry["gaps"], expected)

    def test_operand_encodings_and_overlap_placements(self):
        for args, expected in ((("pop_acc_fp8", 1, 24, 1, 0, 1), 0x0e082c77), (("matmul_acc", 0, 16, 1, 1), 0x180a00f7), (("pop_acc_bf16", 1, 22, 1), 0x12080b77)):
            self.assertEqual(GENERATOR.word(*args), expected)
        self.assertEqual(GENERATOR.placements("push_acc_bf16", "pop_acc_bf16"), {"0=0,1=1": (12, 12), "0~0,1~1": (12, 44)})
        self.assertEqual(GENERATOR.placements("push_weight", "pop_acc_bf16"), {"0=0": (12, 12), "0~0": (12, 44), "0=1": (13, 12), "0~1": (13, 44)})
        self.assertEqual(GENERATOR.pattern([12, 13], [44, 45]), "0~0,1~1")
        self.assertEqual(GENERATOR.pattern([12], [16]), "disjoint")

    def test_cli_files_stdout_and_wrong_argument_counts(self):
        with self.assertRaisesRegex(ValueError, "Unknown generated spec"):
            GENERATOR.generate("not-a-spec")
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "nested/output.yaml"
            for name in SPEC_DIGESTS:
                expected = resolved_spec(TARGET / name)
                self.assertEqual(GENERATOR.main([name, "--output", str(output)]), 0)
                self.assertEqual(yaml.safe_load(output.read_text()), expected)
                stream = io.StringIO()
                with contextlib.redirect_stdout(stream):
                    self.assertEqual(GENERATOR.main([name]), 0)
                self.assertEqual(yaml.safe_load(stream.getvalue()), expected)
                self.assertEqual(output.read_text(), stream.getvalue())
        for args in ([], ["not-a-spec"], ["pairs/core_vpu.yaml", "extra.json"]):
            with self.subTest(arguments=args), contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as error:
                GENERATOR.main(args)
            self.assertEqual(error.exception.code, 2)

    @unittest.skipUnless(FACTS, "requires ATLAS_OP_TIMING")
    def test_current_hardware_facts_reproduce_commands_and_bound_gaps(self):
        facts = json.loads(Path(FACTS).read_text())
        blocks = {b["name"]: b for b in facts["op_timing"]}
        for name in ("vpu", "mxu-same", "mxu-shared"):
            generated = self.generated(name, False, facts)
            self.assertEqual(generated["pairs"], self.specs[name]["pairs"])
            for pair, entry in generated["pairs"].items():
                first = blocks[entry["as"]["first"]]
                low, high = map(int, entry["gaps"].split(".."))
                self.assertEqual(low, 1)
                expected = self.specs[name]["expect"][pair]
                if name == "vpu":
                    self.assertGreaterEqual(high, first["next_issue_age"] + 6)
                    self.assertEqual(expected["next_issue_age"], first["next_issue_age"])
                    self.assertLessEqual(expected["min_gap"], expected["stable_gap"])
                    self.assertLessEqual(expected["stable_gap"], first["next_issue_age"])
                else:
                    last = max(e["last_age"] for e in first["events"].values() if e["count"])
                    self.assertGreater(high, last)
                    self.assertLessEqual(expected["stable_gap"], last + 1)
if __name__ == "__main__":
    unittest.main()
