"""Commands on one MXU at once under selected pair facts (ATLAS_OP_TIMING, ATLAS_PAIR_FACTS, ATLAS_OOT_BIN_DIR).

An MXU command is live from its issue through its busy end (its last event; MREG reads through their response). With
pair facts that cover its unit selected, a command may issue while another on the same MXU is live exactly where a
pair fact for the two (in issue order) covers the gap and their operands satisfy its relation (disjoint physical MREG
banks, and the same or other accumulator and weight slot wherever both name one), and at most the facts' live
capacity run together; otherwise it waits for the live command's busy end, as without pair facts. The accumulator and
weight-slot access rules still bind, so a pair issues at the larger of its stable gap and its data distance. Chains
check the accounting against the chain sweeps (tools/rtl_extract/targets/atlas/sequences/core_mxu.yaml).
"""
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import tempfile
import unittest

import yaml

from test_delay_insertion import OPT, ROOT, nop, program, run, without_delays
from test_rtl_gap_coverage import issue_cycles, issue_gap
from test_rtl_mxu_timing import busy, mxu, spacer
from test_rtl_timing import CONSUMERS, FACTS, HALT, PAIRS, VERIFY, block, delay, export, facts, selection

KINDS = ("push_weight", "push_acc_fp8", "push_acc_bf16", "pop_acc_fp8", "pop_acc_bf16", "matmul", "matmul_acc")
USES = {"push_weight": (False, True), "push_acc_fp8": (True, False), "push_acc_bf16": (True, False),
        "pop_acc_fp8": (True, False), "pop_acc_bf16": (True, False), "matmul": (True, True), "matmul_acc": (True, True)}
# MREG operands of the first and second command (sources, pop destinations), in disjoint physical banks.
REGS = ({"src": 12, "dst": 20}, {"src": 16, "dst": 24})
MATCHED_CHAINS = 109  # of the 172 chain records, those whose stable gap the compiler reproduces exactly


def command(kind, unit, side, acc=0, slot=0, regs=None):
    reg = (regs or REGS[side])["dst" if kind.startswith("pop") else "src"]
    return mxu(kind, unit, reg=reg, acc=acc, slot=slot, scale=side)


def accesses(document, kind, unit, acc, slot):
    """(resource, write, first element, age, step) of the accumulator and weight rows a command touches."""
    events = block(document, f"mxu{unit}.{kind}")["events"]
    a, w = (unit * 2 + acc) * 32, (unit * 2 + slot) * 32
    out = []
    if kind == "push_weight":
        out.append(("weight", True, w, events["weight_write"]["first_age"], 1))
    elif kind.startswith("push_acc"):
        out.append(("acc", True, a, events["acc_load"]["first_age"], 1))
    elif kind.startswith("pop"):
        out.append(("acc", False, a, events["acc_store"]["first_age"], 1))
    else:
        out.append(("acc", True, a, events["acc_write"]["first_age"], 1))
        if kind == "matmul_acc":
            out.append(("acc", False, a, events["acc_read"]["first_age"], 1))
        out += [("weight", False, w, age, 0) for age in (0, events["acc_write"]["last_age"])]
    return out


def data_gap(document, first, second):
    """The compiler's accumulator and weight access distance from `first` to `second` ((kind, unit, acc, slot))."""
    gap = 1
    for res, write, element, age, step in accesses(document, *first):
        for res2, write2, element2, age2, step2 in accesses(document, *second):
            if res == res2 and element == element2 and (write or write2):
                gap = max(gap, max(age - age2, age + 31 * step - (age2 + 31 * step2)) + 1)
    return gap


def same_mxu_facts(overlap="disjoint"):
    """(unit, first, second, accumulator relation, weight-slot relation) -> the pair fact record whose MREG operands
    overlap as `overlap` states (pairs/core_mxu_shared.yaml)."""
    out = {}
    for record in json.loads(Path(PAIRS).read_text())["op_pairs"]:
        if record["engine"] in ("mxu0", "mxu1") and record["relation"]["mreg_banks"] == overlap:
            relation = record["relation"]
            key = (int(record["engine"][3]), record["first"].split(".")[1], record["second"].split(".")[1],
                   relation.get("accumulator"), relation.get("weight_slot"))
            out[key] = record
    return out


def operands(relation):
    """Second command's accumulator and slot for a relation (the first uses accumulator 0 and slot 0)."""
    return 0 if relation in (None, "same") else 1


# funct7 - unit -> operation (lib/AtlasEncoding.cpp).
WORDS = {0: "push_weight", 2: "push_acc_fp8", 4: "push_acc_bf16", 6: "pop_acc_fp8", 8: "pop_acc_bf16", 10: "matmul", 12: "matmul_acc"}
CHAINS = ROOT / "tools/rtl_extract/targets/atlas/sequences/core_mxu.yaml"
SHARED = ROOT / "tools/rtl_extract/targets/atlas/pairs/core_mxu_shared.yaml"


def decoded(word):
    """The compiler operation of an MXU instruction word of the chain file."""
    vd, vs1, vs2, f7 = word >> 7 & 63, word >> 13 & 63, word >> 19 & 63, word >> 25
    kind, unit = WORDS[f7 & ~1], f7 & 1
    if kind == "push_weight":
        return mxu(kind, unit, reg=vs1, slot=vd)
    if kind.startswith("push_acc"):
        return mxu(kind, unit, reg=vs1, acc=vd)
    if kind.startswith("pop"):
        return mxu(kind, unit, reg=vd, acc=vs2, scale=vs1)
    return mxu(kind, unit, reg=vs1, acc=vd, slot=vs2)


def spaced(ops, gap, drain=400):
    out = []
    for i, op in enumerate(ops):
        out += ([] if i == 0 else spacer(gap)) + [op]
    return program(out + [delay(drain), nop(), HALT])


def pair_facts_variant(directory, mutate, name="op-pairs.json"):
    document = json.loads(Path(PAIRS).read_text())
    mutate(document)
    path = Path(directory) / name
    path.write_text(json.dumps(document))
    return path


def record_of(document, unit, first, second, acc=None, slot=None):
    return next(r for r in document["op_pairs"]
                if (r["first"], r["second"]) == (f"mxu{unit}.{first}", f"mxu{unit}.{second}")
                and r["relation"].get("accumulator") == acc and r["relation"].get("weight_slot") == slot
                and r["relation"]["mreg_banks"] == "disjoint")


@unittest.skipUnless(FACTS and PAIRS and OPT.is_file(), "requires RTL timing facts, pair facts and built tools")
class MxuPipelineTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.facts, cls.pairs = facts(), same_mxu_facts()
        cls.with_pairs, cls.plain = selection(pairs=PAIRS), selection()
        if not cls.pairs:
            raise unittest.SkipTest("the selected pair facts carry no MXU records")

    def window(self, kind, unit):
        return busy(block(self.facts, f"mxu{unit}.{kind}")) + 1

    def boundary(self, unit, first, second, acc, slot, pairs=True):
        """First accepted gap: the stable gap (or, without a fact, the busy end) and the data distance."""
        data = data_gap(self.facts, (first, unit, 0, 0), (second, unit, operands(acc), operands(slot)))
        record = self.pairs.get((unit, first, second, acc, slot)) if pairs else None
        hold = record["stable_gap"] if record else self.window(first, unit)
        return max(data, min(hold, self.window(first, unit)))

    def verify(self, ops, option=None):
        return run(OPT, ops if isinstance(ops, str) else program(ops), option or self.with_pairs, VERIFY)

    def assert_boundary(self, a, b, gap, option=None, drain=200):
        accepted = self.verify(spaced([a, b], gap, drain), option)
        self.assertEqual(accepted.returncode, 0, accepted.stderr)
        if gap > 1:
            early = self.verify(spaced([a, b], gap - 1, drain), option)
            self.assertNotEqual(early.returncode, 0)
            self.assertRegex(early.stderr, f"first permitted cycle {gap}|requires {gap}, got {gap - 1}")
            return early.stderr
        return ""

    def test_every_fact_admits_exactly_its_gap(self):
        """Every same-MXU pair fact: accepted at max(stable gap, data distance) and rejected one cycle earlier."""
        self.assertEqual(len(self.pairs), 196)

        def check(key):
            unit, first, second, acc, slot = key
            a = command(first, unit, 0)
            b = command(second, unit, 1, operands(acc), operands(slot))
            gap = self.boundary(*key)
            accepted = self.verify(spaced([a, b], gap, 200))
            early = self.verify(spaced([a, b], gap - 1, 200)) if gap > 1 else None
            return key, gap, accepted, early

        with ThreadPoolExecutor(16) as pool:
            for key, gap, accepted, early in pool.map(check, sorted(self.pairs, key=str)):
                with self.subTest(key, gap=gap):
                    self.assertEqual(accepted.returncode, 0, accepted.stderr)
                    if early is not None:
                        self.assertNotEqual(early.returncode, 0)
                        self.assertRegex(early.stderr, f"first permitted cycle {gap}|requires {gap}, got {gap - 1}")

    def test_shared_operand_facts_hold_with_the_mreg_rules(self):
        """Every same-MXU pair fact for operands sharing an MREG register or bank (pairs/core_mxu_shared.yaml), on the
        swept operands: accepted at the first gap the compiler admits and rejected one cycle earlier, never below the
        fact's stable gap; where the two commands only read the shared registers the MREG rules add nothing, so the
        pair issues exactly at the larger of its stable gap and its accumulator and weight access distance."""
        raw = yaml.safe_load(SHARED.read_text())
        records = [r for r in json.loads(Path(PAIRS).read_text())["op_pairs"]
                   if r["engine"] in ("mxu0", "mxu1") and r["relation"]["mreg_banks"] != "disjoint"]
        self.assertEqual(len(records), len(raw["pairs"]))

        def check(record):
            pair = raw["pairs"][record["source"].split(".", 1)[1]]
            a, b = (decoded(pair[f"{side}_inputs"]["instruction"]) for side in ("first", "second"))
            gap = next(g for g in range(1, 200) if self.verify(spaced([a, b], g, 200)).returncode == 0)
            return record, a, b, gap, self.verify(spaced([a, b], gap + 1, 200)), self.verify(spaced([a, b], gap - 1, 200)) if gap > 1 else None

        with ThreadPoolExecutor(16) as pool:
            for record, a, b, gap, later, early in pool.map(check, records):
                first, second = record["first"].split(".")[1], record["second"].split(".")[1]
                unit, relation = int(record["engine"][3]), record["relation"]
                with self.subTest(record["source"], gap=gap):
                    self.assertGreaterEqual(gap, record["stable_gap"])
                    self.assertEqual(later.returncode, 0, later.stderr)
                    if early is not None:
                        self.assertNotEqual(early.returncode, 0)
                    reads = not first.startswith("pop") and not second.startswith("pop")
                    if reads and "~" not in relation["mreg_banks"]:
                        data = data_gap(self.facts, (first, unit, 0, 0), (second, unit, operands(relation.get("accumulator")),
                                                                          operands(relation.get("weight_slot"))))
                        self.assertEqual(gap, max(record["stable_gap"], data))

    def test_overlap_beats_the_busy_rule(self):
        """Representative pairs issue well inside the first command's busy window, as the sweeps show."""
        cases = [(0, "matmul", "pop_acc_bf16", "same", None, 64), (0, "matmul", "pop_acc_bf16", "other", None, 1),
                 (0, "matmul", "matmul", "other", "same", 32), (0, "matmul", "push_weight", None, "other", 1),
                 (0, "push_weight", "matmul", None, "other", 1), (0, "push_acc_fp8", "matmul_acc", "other", None, 1),
                 (0, "pop_acc_bf16", "matmul", "same", None, 1), (1, "matmul", "pop_acc_bf16", "same", None, 4),
                 (1, "matmul", "pop_acc_fp8", "other", None, 1), (1, "matmul", "matmul", "same", "same", 32),
                 (1, "push_weight", "matmul", None, "other", 1), (1, "matmul", "push_weight", None, "other", 1)]
        for unit, first, second, acc, slot, gap in cases:
            key = (unit, first, second, acc, slot)
            with self.subTest(key):
                self.assertEqual(self.boundary(*key), gap)
                self.assertLess(gap, self.window(first, unit))
                a, b = command(first, unit, 0), command(second, unit, 1, operands(acc), operands(slot))
                self.assert_boundary(a, b, gap)
                self.assert_boundary(a, b, self.window(first, unit) if gap < self.window(first, unit) else gap, self.plain)
                for consumer in CONSUMERS:
                    timed = run(OPT, program([a, b, HALT]), self.with_pairs, consumer, VERIFY)
                    self.assertEqual(timed.returncode, 0, timed.stderr)
                    placed = issue_gap(timed.stdout, a, b)
                    expected = gap if placed > 0 else self.boundary(unit, second, first, acc, slot)
                    self.assertEqual(abs(placed), expected, consumer)

    def test_operands_must_satisfy_the_relation(self):
        """The pop writes m44:45, physical banks 12 and 13 of the matmul's source m12 ("0~0"): the shared-bank record for
        that overlap applies (pairs/core_mxu_shared.yaml); without it, or under the disjoint record alone, the pop waits
        for the matmul's busy end."""
        for unit in (0, 1):
            first = command("matmul", unit, 0)
            second = command("pop_acc_bf16", unit, 1, acc=1, regs={"dst": 44})
            window = self.window("matmul", unit)
            shared = same_mxu_facts("0~0")[(unit, "matmul", "pop_acc_bf16", "other", None)]
            with self.subTest(unit=unit):
                self.assert_boundary(first, second, shared["stable_gap"])
                with tempfile.TemporaryDirectory() as directory:
                    disjoint = pair_facts_variant(directory, lambda d: d.update(
                        op_pairs=[r for r in d["op_pairs"] if r["relation"]["mreg_banks"] in ("disjoint",)]))
                    stderr = self.assert_boundary(first, second, window, selection(pairs=disjoint))
                self.assertIn(f"no pair fact admits vmatpop.bf16.acc.mxu{unit} {window - 1} cycles after "
                              f"vmatmul.mxu{unit} with these operands", stderr)
                self.assertIn(f"MXU in-flight matmuls {unit} held by vmatmul.mxu{unit}", stderr)

    def test_export_reports_the_pipelined_units(self):
        source = program([command("matmul", 0, 0), command("pop_acc_bf16", 0, 1), HALT])
        for option, pipelined in ((self.with_pairs, ["mxu0", "mxu1"]), (self.plain, None)):
            timed = run(OPT, source, option, CONSUMERS[0], VERIFY)
            self.assertEqual(timed.returncode, 0, timed.stderr)
            applicability = export(timed.stdout)["applicability"]
            admission = applicability["admission"]
            self.assertEqual(admission.get("mxu_pipelined_units"), pipelined)
            self.assertEqual(admission["mxu_commands_sequenced_per_unit"], pipelined is None)
            self.assertEqual("same_mxu_pipelining" in applicability["unsupported"], pipelined is None)
            if pipelined:
                self.assertEqual(applicability["pair_facts"]["live_window"],
                                 {"mxu0": "last_event_age", "mxu1": "last_event_age", "vpu": "next_issue_age"})

    def test_missing_facts_and_unknown_relations_do_not_apply(self):
        """A missing record, a relation in an unknown vocabulary, one that leaves the accumulator unstated where both
        commands name one, or a sweep that stops short of the busy end each fall back to the busy rule."""
        first, second = command("matmul", 0, 0), command("pop_acc_bf16", 0, 1, acc=1)
        window = self.window("matmul", 0)
        mutations = {
            "missing": lambda d: d["op_pairs"].remove(record_of(d, 0, "matmul", "pop_acc_bf16", "other")),
            "unknown value": lambda d: record_of(d, 0, "matmul", "pop_acc_bf16", "other")["relation"].update(accumulator="different"),
            "unknown key": lambda d: record_of(d, 0, "matmul", "pop_acc_bf16", "other")["relation"].update(port="read0"),
            "unstated accumulator": lambda d: record_of(d, 0, "matmul", "pop_acc_bf16", "other")["relation"].pop("accumulator"),
            "short sweep": lambda d: record_of(d, 0, "matmul", "pop_acc_bf16", "other")["gaps"].__setitem__(1, window - 2),
        }
        with tempfile.TemporaryDirectory() as directory:
            for name, mutate in mutations.items():
                with self.subTest(name):
                    path = pair_facts_variant(directory, mutate, f"{name.replace(' ', '-')}.json")
                    option = selection(pairs=path)
                    self.assertEqual(self.verify(spaced([first, second], 1), option).returncode != 0, True)
                    self.assert_boundary(first, second, window, option)
                    # The same-accumulator record still applies.
                    same = command("pop_acc_bf16", 0, 1, acc=0)
                    self.assert_boundary(first, same, 64, option)

    def test_capacity_bounds_a_third_command(self):
        """Three MXU0 matmuls alternating accumulators: the pairwise facts admit each at 32, but the facts' capacity
        (two, all the pair runs show) holds the third until the first's busy end; a capacity of three admits it."""
        ops = [command("matmul", 0, 0, acc=0), command("matmul", 0, 1, acc=1), mxu("matmul", 0, reg=4, acc=0)]
        window = self.window("matmul", 0)
        capacity = {c["engine"]: c["live"] for c in json.loads(Path(PAIRS).read_text())["capacity"]}
        self.assertEqual(capacity["mxu0"], 2)
        full = self.verify(spaced(ops, 32))
        self.assertIn("all 2 mxu0 slots busy", full.stderr)
        self.assertIn(f"first permitted cycle {window}", full.stderr)
        for consumer in CONSUMERS:
            timed = run(OPT, program([*ops, HALT]), self.with_pairs, consumer, VERIFY)
            self.assertEqual(timed.returncode, 0, timed.stderr)
            cycles = [c for op, _, c in issue_cycles(timed.stdout) if op == "mxu_matmul"]
            self.assertEqual(cycles, [0, 32, window], consumer)
        with tempfile.TemporaryDirectory() as directory:
            three = pair_facts_variant(directory, lambda d: next(c for c in d["capacity"] if c["engine"] == "mxu0").update(live=3))
            self.assertEqual(self.verify(spaced(ops, 32), selection(pairs=three)).returncode, 0)

    def test_chains_never_admit_what_the_sweeps_reject(self):
        """Every chain of sequences/core_mxu.yaml at uniform spacing: the compiler accepts no spacing below the chain's
        stable gap (each chain is rejected below it and accepted from it), and exactly the stable gap unless the facts'
        capacity of two or an accumulator or weight access rule binds, which the rejection at the stable gap names."""
        raw = yaml.safe_load(CHAINS.read_text())
        records = [(name, length, stable) for name, lengths in raw["expect"].items() for length, stable in lengths.items()]

        def scan(record):
            name, length, stable = record
            ops = [decoded(entry["inputs"]["instruction"]) for entry in raw["sequences"][name]["ops"][:length]]
            verdicts = {}
            for gap in range(1, 200):
                verdicts[gap] = self.verify(spaced(ops, gap))
                if gap >= stable and verdicts[gap].returncode == 0 and all(
                        self.verify(spaced(ops, g)).returncode == 0 for g in range(gap + 1, gap + 3)):
                    break
            return record, verdicts

        with ThreadPoolExecutor(32) as pool:
            scanned = list(pool.map(scan, records))
        matched = 0
        for (name, length, stable), verdicts in scanned:
            accepted = [g for g, r in verdicts.items() if r.returncode == 0]
            with self.subTest(f"{name}/{length}", stable=stable):
                self.assertTrue(accepted)
                self.assertGreaterEqual(min(accepted), stable)
                if min(accepted) == stable:
                    matched += 1
                else:
                    self.assertRegex(verdicts[stable].stderr, r"mxu\d slots busy|requires \d+, got \d+")
        self.assertGreaterEqual(matched, MATCHED_CHAINS)

    def test_without_mxu_facts_nothing_changes(self):
        """Pair facts without MXU records (the VPU document) and no pair facts time MXU programs identically; MXU programs
        compare against the 38393c5 build in the word comparison (docs/rtl-timing/verification.md)."""
        programs = {name: without_delays((ROOT / f"test/examples/{name}.mlir").read_text())
                    for name in ("mxu0_full", "mxu0_k2", "mxu1_k2")}
        tiles = {f"tile{u}": program([command("push_weight", u, 0), command("matmul", u, 0), command("pop_acc_bf16", u, 0), HALT])
                 for u in (0, 1)}
        with tempfile.TemporaryDirectory() as directory:
            vpu_only = pair_facts_variant(directory, lambda d: (
                d.update(op_pairs=[r for r in d["op_pairs"] if r["engine"] not in ("mxu0", "mxu1")]),
                d.update(capacity=[c for c in d["capacity"] if c["engine"] not in ("mxu0", "mxu1")])))
            for name, source in {**programs, **tiles}.items():
                for consumer in CONSUMERS:
                    with self.subTest(name, consumer=consumer):
                        timed = [run(OPT, source, option, consumer, VERIFY)
                                 for option in (selection(dma="wait"), selection(pairs=vpu_only, dma="wait"))]
                        self.assertEqual(timed[0].returncode, 0, timed[0].stderr)
                        self.assertEqual(issue_cycles(timed[0].stdout), issue_cycles(timed[1].stdout))


if __name__ == "__main__":
    unittest.main()
