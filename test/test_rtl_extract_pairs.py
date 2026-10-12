"""Pair recipe (tools/rtl_extract/pairs.py): synthetic sweeps, pair-file validation, and Atlas hardware regressions.

Hardware tests set ATLAS_HW_IR and ATLAS_HW_EXPORTER (built with ``--assertions`` support) and check a narrow window
around each pair file's expected ``stable_gap`` (the gap below it rejected, by the invariant or guard matching
``rejected_by``; it and the next accepted) and the expected ``disjoint`` shared-control verdict; full sweeps run
through ``python3 -m rtl_extract.pairs``.
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
from rtl_extract.spec import TARGETS

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


class CoreVpuPairFileTests(unittest.TestCase):
    """The all-pairs VPU file: 29x29 ordered pairs with disjoint physical banks, and the compiler's VPU->VPU gap rule."""

    @classmethod
    def setUpClass(cls):
        cls.raw, cls.spec = pairs.load("atlas", ["core_vpu"])["core_vpu"]
        vpu = yaml.safe_load((TARGETS / "atlas" / "vpu.yaml").read_text())
        cls.words = vpu["decode"]["words"]
        cls.next_issue = {op: v["next_issue_age"] for op, v in vpu["expect"].items()}

    def split(self, name):
        first, second = name.split("_")
        return first, second

    def test_every_ordered_pair_is_declared(self):
        self.assertEqual(len(self.words), 29)
        self.assertEqual(set(self.raw["pairs"]), {f"{a}_{b}" for a in self.words for b in self.words})

    def test_instruction_words_encode_the_named_operations(self):
        for name, pair in self.raw["pairs"].items():
            for side, op in zip(("first", "second"), self.split(name)):
                word = pair[f"{side}_inputs"]["instruction"]
                mask = 0x607F if op.startswith("vli") else 0xFE00007F      # vli: mode and opcode; others: funct7 and opcode
                with self.subTest(name, side=side):
                    self.assertEqual(word & mask, self.words[op] & mask)

    def test_operand_banks_are_disjoint_modulo_32(self):
        def banks(op, word):
            vd, vs1, vs2 = word >> 7 & 63, word >> 13 & 63, word >> 19 & 63
            if op.startswith("vli"):
                return {vd % 32, (vd + 1) % 32}
            return {r % 32 for r in (vd, vd + 1, vs1, vs1 + 1, vs2, vs2 + 1)}
        for name, pair in self.raw["pairs"].items():
            first, second = self.split(name)
            with self.subTest(name):
                self.assertFalse(banks(first, pair["first_inputs"]["instruction"]) & banks(second, pair["second_inputs"]["instruction"]))

    def test_gaps_cover_the_first_operations_first_free_age(self):
        for name, pair in self.raw["pairs"].items():
            low, high = pair["gaps"].split("..")
            with self.subTest(name):
                self.assertEqual(int(low), 1)
                self.assertGreaterEqual(int(high), self.next_issue[self.split(name)[0]] + 1 + 5)

    def test_stable_gap_never_exceeds_the_first_operations_next_issue_age(self):
        for name, expected in self.raw["expect"].items():
            first = self.split(name)[0]
            with self.subTest(name):
                self.assertEqual(expected["next_issue_age"], self.next_issue[first])
                self.assertLessEqual(expected["min_gap"], expected["stable_gap"])
                self.assertLessEqual(expected["stable_gap"], expected["next_issue_age"])


@unittest.skipUnless(HW_IR and EXPORTER, "requires ATLAS_HW_IR and ATLAS_HW_EXPORTER")
class HardwarePairTests(unittest.TestCase):
    def check(self, name):
        raw, spec = pairs.load("atlas", [name])[name]
        document = load_document(HW_IR, modules(spec), EXPORTER, pairs.needs_assertions(raw, spec))
        window = copy.deepcopy(raw)
        for pair, expected in raw["expect"].items():
            stable = expected["stable_gap"]
            window["pairs"][pair]["gaps"] = f"{max(1, stable - 1)}..{stable + 1}"
        for record in pairs.extract(document, window, spec, jobs=8):
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

    def test_vlsu(self):
        self.check("vlsu")

    def test_vpu(self):
        self.check("vpu")

    def test_core(self):
        self.check("core")

    CORE_VPU_SAMPLE = ("exp_relu relu_exp exp_exp2 exp_add add_relu sin_cos rmax_exp rsum_relu csum_relu csum_csum csum_add "
                       "cmax_pairmax fp8pack_relu fp8unpack_relu fp8unpack_fp8unpack vliAll_relu vliAll_vliCol vliOne_vliRow "
                       "relu_vliOne mov_vliAll").split()

    def test_core_vpu_sample(self):
        """Windows of stable_gap - 1 .. stable_gap + 1 for one pair per timing class; the gap below must fire the VPU
        issue-busy assertions, and the stable gap never exceeds the first operation's next_issue_age."""
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

    def test_mxu0(self):
        self.check("mxu0")

    def test_mxu1(self):
        self.check("mxu1")

    def test_core_mxu(self):
        """MXU commands through the RTL decoder on AtlasCore: MXU against MXU (the sequencer's issue assertions bind) and
        against VPU, VLS and XLU commands on disjoint banks (accepted at gap 1 with disjoint control)."""
        self.check("core_mxu")


if __name__ == "__main__":
    unittest.main()
