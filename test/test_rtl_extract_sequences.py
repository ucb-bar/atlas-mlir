"""Sequence recipe (tools/rtl_extract/sequences.py): synthetic chains, sequence-file validation, and Atlas hardware regressions.

Hardware tests set ATLAS_HW_IR and ATLAS_HW_EXPORTER and check a narrow window around each sequence file's expected
per-length ``stable_gap`` (the spacing below it rejected, it and the next accepted); full sweeps run through
``python3 -m rtl_extract.sequences``.
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
import test_rtl_extract_pairs as pair_tests
import test_rtl_extract_synthetic as synthetic
from rtl_extract import derive, pairs, sequences
from rtl_extract.ir import load_document
from rtl_extract.runner import RECIPES, modules
from rtl_extract.spec import ages

HW_IR = os.environ.get("ATLAS_HW_IR")
EXPORTER = os.environ.get("ATLAS_HW_EXPORTER")
op = synthetic.operation


def queue():
    """``go`` is accepted unless two earlier commands are still in the first three stages of a 4-stage shift register;
    ``out`` pulses 4 ages after an accepted ``go`` (a 2-deep queue: the third command in flight is ignored)."""
    stages = [op(f"sr{i}", "seq.firreg", [("accept" if i == 0 else f"sr{i - 1}"), "clock", "reset", "zero"], name=f"sr{i}") for i in range(4)]
    return synthetic.module("Queue", ["go"], [("out", "sr3"), ("busy", "busy")], [
        synthetic.zero(), synthetic.one(), *stages,
        op("ab", "comb.and", ["sr0", "sr1"]), op("ac", "comb.and", ["sr0", "sr2"]), op("bc", "comb.and", ["sr1", "sr2"]),
        op("m", "comb.or", ["ab", "ac"]), op("full", "comb.or", ["m", "bc"]), op("free", "comb.xor", ["full", "one"]),
        op("accept", "comb.and", ["go", "free"]),
        op("lo", "comb.or", ["sr0", "sr1"]), op("hi", "comb.or", ["sr2", "sr3"]), op("busy", "comb.or", ["lo", "hi"])])


QUEUE_SPEC = {"engine": "toy", "module": "Queue", "inputs": {"go": 0}, "busy": "busy", "limit": 24,
              "events": {"out": {"valid": "out"}}, "operations": {"go": {"commands": {0: {"go": 1}}}}}


def sweep(module, sequence, spec=QUEUE_SPEC, extra=None):
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
        for label, raw, message in (
                ("missing reason", {"exclude": [{"message": "x"}]}, "missing keys"),
                ("unknown key", {"only": []}, "unknown keys")):
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
        records = sweep(queue(), {"ops": [{"op": "go"}], "lengths": "2..4", "gaps": "1..5"})
        self.assertEqual({n: r["stable_gap"] for n, r in records.items()}, {2: 1, 3: 2, 4: 2})
        self.assertEqual(records[2]["results"], [{"from": 1, "to": 5, "result": "accepted"}])
        self.assertEqual(records[3]["results"][0], {"from": 1, "to": 1, "result": "missing out x1 (ops [2])"})
        self.assertEqual(records[4]["results"][0], {"from": 1, "to": 1, "result": "missing out x2 (ops [2, 3])"})

    def test_restart_merges_events_until_streams_abut(self):
        spec = pair_tests.SPEC
        records = sweep(pair_tests.counter(True), {"ops": [{"op": "go", "guards": []}], "lengths": "2..3", "gaps": "1..6"}, spec)
        self.assertEqual({n: (r["min_gap"], r["stable_gap"]) for n, r in records.items()}, {2: (3, 3), 3: (3, 3)})
        self.assertRegex(records[3]["results"][0]["result"], "^missing out")

    def test_guard_applies_to_every_operation_after_the_first(self):
        records = sweep(pair_tests.counter(True), {"ops": [{"op": "go"}], "lengths": 3, "gaps": "1..6"}, pair_tests.SPEC)
        self.assertEqual(records[3]["stable_gap"], 4)
        self.assertEqual(records[3]["results"][0]["result"], "guard busy set at op 1; guard busy set at op 2; missing out x4 (ops [1, 2])")
        self.assertEqual(records[3]["results"][2], {"from": 3, "to": 3, "result": "guard busy set at op 1; guard busy set at op 2; events unchanged"})

    def test_command_ages_collide(self):
        spec = copy.deepcopy(QUEUE_SPEC)
        spec["operations"]["go"]["commands"] = {0: {"go": 1}, 1: {"go": 1}}
        records = sweep(queue(), {"ops": [{"op": "go"}], "lengths": 2, "gaps": "1..3"}, spec)
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
    JOBS = min(os.cpu_count() or 1, 16)

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
        """Same-MXU chains on AtlasCore (sequences/core_mxu.yaml), every assertion checked: each chain length is rejected
        one spacing below its stable gap and accepted at it and one above. One extraction runs every window."""
        raw, spec = sequences.load("atlas", ["core_mxu"])["core_mxu"]
        document = load_document(HW_IR, modules(spec), EXPORTER, True)
        window = copy.deepcopy(raw)
        window["sequences"] = {f"{name}_{length}": {**raw["sequences"][name], "lengths": length, "gaps": f"{max(1, stable - 1)}..{stable + 1}"}
                               for name, lengths in raw["expect"].items() for length, stable in lengths.items()}
        for record in sequences.extract(document, window, spec, min(64, os.cpu_count() or 1)):
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
        """Without invariants the event comparison alone accepts MXU0's push_acc, matmul_acc, push_acc chain from gap 64;
        the AccumulationBuffers check (the load targets the buffer the compute is still writing) moves it to 94."""
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
        """The facts block ``matmul_acc_at_32`` (a push_acc_fp8 then a matmul_acc at age 32, accepted but writing no
        accumulator rows) is the chain event stream at the stable gap minus one; the chain rule rejects it."""
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
        self.assertEqual(pairs.events(chain_trace, raw["equivalent"], ignore=raw["ignore"]),
                         pairs.events(block_trace, raw["equivalent"], ignore=raw["ignore"]))


if __name__ == "__main__":
    unittest.main()
