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

    def test_pairs_name_their_blocks_and_relation(self):
        self.assertEqual(self.raw["relation"], {"mreg_banks": "disjoint"})
        for name, pair in self.raw["pairs"].items():
            first, second = self.split(name)
            self.assertEqual(pair["as"], {"first": f"vpu.{first}", "second": f"vpu.{second}"}, name)

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


# Same-MXU commands: funct7 = base + unit (lib/AtlasEncoding.cpp); (accumulator field, weight-slot field) or None.
MXU_KINDS = {"push_weight": (0, None, "vd"), "push_acc_fp8": (2, "vd", None), "push_acc_bf16": (4, "vd", None),
             "pop_acc_fp8": (6, "vs2", None), "pop_acc_bf16": (8, "vs2", None), "matmul": (10, "vd", "vs2"),
             "matmul_acc": (12, "vd", "vs2")}


def mxu_fields(word):
    return {"vd": word >> 7 & 63, "vs1": word >> 13 & 63, "vs2": word >> 19 & 63, "f7": word >> 25}


def mxu_mreg_banks(kind, word):
    f = mxu_fields(word)
    if kind.startswith("pop"):
        regs = [f["vd"], f["vd"] + 1] if kind.endswith("bf16") else [f["vd"]]
    else:
        regs = [f["vs1"], f["vs1"] + 1] if kind.endswith("bf16") else [f["vs1"]]
    return {r % 32 for r in regs}


class CoreMxuSamePairFileTests(unittest.TestCase):
    """Every ordered pair of commands on one MXU (pairs/core_mxu_same.yaml): blocks, relations and instruction words."""

    @classmethod
    def setUpClass(cls):
        cls.raw, cls.spec = pairs.load("atlas", ["core_mxu_same"])["core_mxu_same"]

    def test_every_ordered_pair_and_relation_is_declared(self):
        expected = set()
        for unit in (0, 1):
            for a, (_, acc_a, slot_a) in MXU_KINDS.items():
                for b, (_, acc_b, slot_b) in MXU_KINDS.items():
                    accs = ("same", "other") if acc_a and acc_b else (None,)
                    slots = ("same", "other") if slot_a and slot_b else (None,)
                    expected |= {(f"mxu{unit}.{a}", f"mxu{unit}.{b}", acc, slot) for acc in accs for slot in slots}
        declared = {(p["as"]["first"], p["as"]["second"], p.get("relation", {}).get("accumulator"),
                     p.get("relation", {}).get("weight_slot")) for p in self.raw["pairs"].values()}
        self.assertEqual(len(declared), len(self.raw["pairs"]))
        self.assertEqual(declared, expected)
        self.assertEqual(self.raw["relation"], {"mreg_banks": "disjoint"})

    def test_instruction_words_encode_the_pair_and_its_relation(self):
        for name, pair in self.raw["pairs"].items():
            words = [pair["first_inputs"]["instruction"], pair["second_inputs"]["instruction"]]
            kinds = [pair["as"][side].split(".")[1] for side in ("first", "second")]
            unit = int(pair["as"]["first"][3])
            with self.subTest(name):
                self.assertTrue(all(w & 0x7F == 0x77 for w in words))
                for word, kind in zip(words, kinds):
                    self.assertEqual(mxu_fields(word)["f7"], MXU_KINDS[kind][0] + unit)
                fields = [mxu_fields(w) for w in words]
                for key, index in (("accumulator", 1), ("weight_slot", 2)):
                    relation = pair.get("relation", {}).get(key)
                    used = [MXU_KINDS[k][index] for k in kinds]
                    self.assertEqual(relation is not None, all(used))
                    if relation:
                        same = fields[0][used[0]] == fields[1][used[1]]
                        self.assertEqual(same, relation == "same")
                self.assertFalse(mxu_mreg_banks(kinds[0], words[0]) & mxu_mreg_banks(kinds[1], words[1]))

    @unittest.skipUnless(os.environ.get("ATLAS_OP_TIMING"), "requires ATLAS_OP_TIMING")
    def test_gaps_reach_past_the_first_commands_last_event(self):
        import json
        blocks = {b["name"]: b for b in json.loads(Path(os.environ["ATLAS_OP_TIMING"]).read_text())["op_timing"]}
        for name, pair in self.raw["pairs"].items():
            last = max(g["last_age"] for g in blocks[pair["as"]["first"]]["events"].values() if g["count"])
            low, high = pair["gaps"].split("..")
            with self.subTest(name):
                self.assertEqual(int(low), 1)
                self.assertGreater(int(high), last)
                self.assertLessEqual(self.raw["expect"][name]["stable_gap"], last + 1)


def mxu_registers(kind, word):
    """MREG registers of an MXU command in operand order (a BF16 pair as base, base+1)."""
    f = mxu_fields(word)
    base = f["vd"] if kind.startswith("pop") else f["vs1"]
    return [base, base + 1] if kind.endswith("bf16") else [base]


def mxu_overlap(first, second):
    """The pair files' MREG overlap pattern: "i=j" same register, "i~j" same physical bank, else "disjoint"."""
    parts = [f"{i}={j}" if a == b else f"{i}~{j}" for i, a in enumerate(first) for j, b in enumerate(second) if a % 32 == b % 32]
    return ",".join(parts) or "disjoint"


class CoreMxuSharedPairFileTests(unittest.TestCase):
    """Same-MXU pairs whose MREG operands share a register or a bank (pairs/core_mxu_shared.yaml): every overlap the two
    commands' operands can have, for every accumulator and weight-slot relation, with words that encode it."""

    @classmethod
    def setUpClass(cls):
        cls.raw, cls.spec = pairs.load("atlas", ["core_mxu_shared"])["core_mxu_shared"]

    def test_every_overlap_and_relation_is_declared(self):
        expected = set()
        for unit in (0, 1):
            for a, (_, acc_a, slot_a) in MXU_KINDS.items():
                for b, (_, acc_b, slot_b) in MXU_KINDS.items():
                    shapes = lambda kind: [[r, r + 1] for r in range(0, 64, 2)] if kind.endswith("bf16") else [[r] for r in range(64)]
                    overlaps = {mxu_overlap(x, y) for x in shapes(a) for y in shapes(b)} - {"disjoint"}
                    accs = ("same", "other") if acc_a and acc_b else (None,)
                    slots = ("same", "other") if slot_a and slot_b else (None,)
                    expected |= {(f"mxu{unit}.{a}", f"mxu{unit}.{b}", o, acc, slot) for o in overlaps for acc in accs for slot in slots}
        declared = {(p["as"]["first"], p["as"]["second"], p["relation"]["mreg_banks"], p["relation"].get("accumulator"),
                     p["relation"].get("weight_slot")) for p in self.raw["pairs"].values()}
        self.assertEqual(len(declared), len(self.raw["pairs"]))
        self.assertEqual(declared, expected)

    def test_instruction_words_encode_the_overlap_and_relation(self):
        for name, pair in self.raw["pairs"].items():
            words = [pair["first_inputs"]["instruction"], pair["second_inputs"]["instruction"]]
            kinds = [pair["as"][side].split(".")[1] for side in ("first", "second")]
            unit = int(pair["as"]["first"][3])
            with self.subTest(name):
                for word, kind in zip(words, kinds):
                    self.assertEqual((word & 0x7F, mxu_fields(word)["f7"]), (0x77, MXU_KINDS[kind][0] + unit))
                self.assertEqual(mxu_overlap(*(mxu_registers(k, w) for k, w in zip(kinds, words))), pair["relation"]["mreg_banks"])
                fields = [mxu_fields(w) for w in words]
                for key, index in (("accumulator", 1), ("weight_slot", 2)):
                    relation, used = pair["relation"].get(key), [MXU_KINDS[k][index] for k in kinds]
                    self.assertEqual(relation is not None, all(used))
                    if relation:
                        self.assertEqual(fields[0][used[0]] == fields[1][used[1]], relation == "same")

    @unittest.skipUnless(os.environ.get("ATLAS_OP_TIMING"), "requires ATLAS_OP_TIMING")
    def test_gaps_reach_past_the_first_commands_last_event(self):
        import json
        blocks = {b["name"]: b for b in json.loads(Path(os.environ["ATLAS_OP_TIMING"]).read_text())["op_timing"]}
        for name, pair in self.raw["pairs"].items():
            last = max(g["last_age"] for g in blocks[pair["as"]["first"]]["events"].values() if g["count"])
            with self.subTest(name):
                self.assertGreater(int(pair["gaps"].split("..")[1]), last)
                self.assertLessEqual(self.raw["expect"][name]["stable_gap"], last + 1)


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

    def test_core_mxu_cross(self):
        """Every MXU command against VPU, VLS and XLU commands: gap 1 on disjoint operands with disjoint control; a shared
        register binds at the MREG bank tracker's RAW/WAR assertions and a shared physical bank at MregFile's port checks."""
        self.check("core_mxu_cross")

    def test_core_mxu_units(self):
        """Every ordered pair of commands across MXU0 and MXU1: gap 1 on disjoint operands with disjoint control."""
        self.check("core_mxu_units")

    def test_core_mxu_shared(self):
        """Same-MXU pairs sharing an MREG register or bank: one cycle below each stable gap the MREG bank tracker, MregFile's
        ports, an MXU sequencer register check or the issue checks of the disjoint pair bind."""
        self.check("core_mxu_shared", jobs=min(64, os.cpu_count() or 8))

    def test_core_mxu_same(self):
        """Every ordered pair of commands on one MXU, per accumulator and weight-slot relation: the MXU sequencer's issue
        assertions, the accumulation buffer's read check or a moved MREG port event bind one cycle below each stable gap."""
        self.check("core_mxu_same", jobs=min(64, os.cpu_count() or 8))


if __name__ == "__main__":
    unittest.main()
