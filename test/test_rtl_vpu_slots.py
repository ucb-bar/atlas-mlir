"""Two VPU commands at once under selected pair facts (ATLAS_OP_TIMING, ATLAS_PAIR_FACTS, ATLAS_OOT_BIN_DIR).

A VPU command is live from its issue until its next_issue_age. With pair facts selected, a command may issue while
another is live exactly where a pair fact for the two (in issue order) covers the gap and their operands satisfy its
relation (disjoint physical MREG banks), and at most the facts' live capacity run together; without a fact the next
command waits for next_issue_age. Chains check the slot accounting against the chain sweeps
(tools/rtl_extract/targets/atlas/sequences/core_vpu.yaml). Without pair facts nothing changes.
"""
import json
from pathlib import Path
import tempfile
import unittest

import yaml

from test_delay_insertion import OPT, ROOT, nop, program, run, stream
from test_rtl_gap_coverage import gapped, issue_cycles, issue_gap
from test_rtl_overlap import KERNELS
from test_rtl_timing import CONSUMERS, FACTS, HALT, PAIRS, VERIFY, block, delay, facts, selection
from test_rtl_vmul_timing import pack, reduce, unary, vli, vmul
from test_rtl_xlu_timing import transpose

# vs1, vs2, vd per chain position: every register pair in its own physical bank.
SETS = [(0, 2, 4), (8, 10, 12), (16, 18, 20), (24, 26, 28)]
BINARY = {"add": "add", "sub": "sub", "mul": "mul", "pairmax": "max", "pairmin": "min"}
UNARY = {"mov": "mov", "rcp": "recip", "log": "log2",
         **{k: k for k in ("exp", "exp2", "relu", "sin", "cos", "tanh", "sqrt", "square", "cube")}}
REDUCE = {"rsum": "row_sum", "rmax": "row_max", "rmin": "row_min", "csum": "col_sum", "cmax": "col_max", "cmin": "col_min"}
VLI = {"vliAll": "all", "vliRow": "row", "vliCol": "col", "vliOne": "one"}
OVERLAP = [("exp", "relu"), ("relu", "exp"), ("rcp", "sqrt"), ("mov", "exp"), ("fp8unpack", "relu"),
           ("vliAll", "relu"), ("csum", "relu")]
SERIAL = [("exp", "exp"), ("exp", "exp2"), ("relu", "add"), ("add", "relu"), ("rsum", "relu"), ("exp", "rsum"),
          ("vliAll", "vliCol")]


def vpu(name, regs):
    vs1, vs2, vd = regs
    if name in BINARY:
        return vmul(vs1, vs2, vd, BINARY[name])
    if name in UNARY:
        return unary(UNARY[name], vs1, vd)
    if name in REDUCE:
        return reduce(REDUCE[name], vs1, vd)
    if name in VLI:
        return vli(VLI[name], vd)
    return pack("bf16_to_fp8" if name == "fp8pack" else "fp8_to_bf16", vs1, vd, 3)


def pair_facts():
    return {(r["first"], r["second"]): r for r in json.loads(Path(PAIRS).read_text())["op_pairs"]}


def spaced(ops, gap):
    """`ops` issued `gap` cycles apart, then a drain and the terminal."""
    out = []
    for i, op in enumerate(ops):
        out += ([] if i == 0 or gap == 1 else [nop()] if gap == 2 else [delay(gap - 2)]) + [op]
    return program(out + [delay(400), nop(), HALT])


@unittest.skipUnless(FACTS and PAIRS and OPT.is_file(), "requires RTL timing facts, pair facts and built tools")
class VpuSlotTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.facts, cls.pairs = facts(), pair_facts()
        cls.with_pairs, cls.plain = selection(pairs=PAIRS), selection()

    def accepts(self, source, option=None):
        return run(OPT, source, option or self.with_pairs, VERIFY)

    def next_issue(self, name):
        return block(self.facts, f"vpu.{name}")["next_issue_age"]

    def test_pairs_admit_exactly_their_stable_gap(self):
        for first, second in OVERLAP + SERIAL:
            stable = self.pairs[(f"vpu.{first}", f"vpu.{second}")]["stable_gap"]
            a, b = vpu(first, SETS[0]), vpu(second, SETS[1])
            with self.subTest(f"{first}->{second}", stable=stable):
                self.assertEqual((stable == 1), (first, second) in OVERLAP)
                result = self.accepts(gapped([], a, b, stable))
                self.assertEqual(result.returncode, 0, result.stderr)
                if stable > 1:
                    early = self.accepts(gapped([], a, b, stable - 1))
                    self.assertIn(f"first permitted cycle {stable}", early.stderr)
                    self.assertIn("no pair fact admits", early.stderr)
                else:
                    plain = self.accepts(gapped([], a, b, 1), self.plain)
                    self.assertIn(f"first permitted cycle {self.next_issue(first)}", plain.stderr)
                reverse = self.pairs[(f"vpu.{second}", f"vpu.{first}")]["stable_gap"]
                for consumer in CONSUMERS:
                    timed = run(OPT, program([a, b, HALT]), self.with_pairs, consumer, VERIFY)
                    self.assertEqual(timed.returncode, 0, timed.stderr)
                    gap = issue_gap(timed.stdout, a, b)
                    # The scheduler may issue the second first; it then keeps the reverse pair's gap.
                    self.assertEqual(gap if gap > 0 else -gap, stable if gap > 0 else reverse, consumer)

    def test_operands_must_satisfy_the_relation(self):
        # relu reads m48 and m49, physical banks 16 and 17 of exp's destination pair: no fact applies.
        first, second = unary("exp", 8, 16), unary("relu", 48, 28)
        shared = self.accepts(gapped([], first, second, 1))
        self.assertIn("no pair fact admits vrelu.bf16 1 cycles after vexp.bf16 with these operands", shared.stderr)
        self.assertIn(f"first permitted cycle {self.next_issue('exp')}", shared.stderr)
        self.assertEqual(self.accepts(gapped([], first, second, self.next_issue("exp"))).returncode, 0)

    def test_missing_facts_and_unknown_relations_do_not_apply(self):
        first, second = unary("exp", 8, 16), unary("relu", 20, 28)
        with tempfile.TemporaryDirectory() as directory:
            for relation in (None, {}, {"mreg_banks": "shared"}, {"mreg_banks": "disjoint", "accumulator": "other"}):
                document = json.loads(Path(PAIRS).read_text())
                target = next(r for r in document["op_pairs"] if (r["first"], r["second"]) == ("vpu.exp", "vpu.relu"))
                if relation is None:
                    document["op_pairs"].remove(target)
                else:
                    target["relation"] = relation
                path = Path(directory) / "op-pairs.json"
                path.write_text(json.dumps(document))
                with self.subTest(relation=relation):
                    result = self.accepts(gapped([], first, second, 1), selection(pairs=path))
                    self.assertIn(f"first permitted cycle {self.next_issue('exp')}", result.stderr)

    def test_chains_match_the_sequence_sweeps(self):
        raw = yaml.safe_load((ROOT / "tools/rtl_extract/targets/atlas/sequences/core_vpu.yaml").read_text())
        for name, lengths in raw["expect"].items():
            cycle = [entry["op"] for entry in raw["sequences"][name]["ops"]]
            for length, stable in lengths.items():
                ops = [vpu(cycle[i % len(cycle)], SETS[i % len(SETS)]) for i in range(length)]
                accepted = lambda gap: self.accepts(spaced(ops, gap)).returncode == 0
                with self.subTest(name, length=length, stable=stable):
                    self.assertEqual(min(g for g in range(1, stable + 2) if accepted(g)), stable)
                    self.assertTrue(all(accepted(g) for g in range(stable, stable + 4)))

    def test_slot_capacity_bounds_a_third_command(self):
        # exp and relu fill both slots; sin shares no logic with either, so only capacity holds it back.
        ops = [vpu("exp", SETS[0]), vpu("relu", SETS[1]), vpu("sin", SETS[2])]
        full = self.accepts(program([*ops, delay(400), nop(), HALT]))
        self.assertIn("VPU slots busy", full.stderr)
        self.assertIn(f"first permitted cycle {self.next_issue('exp')}", full.stderr)
        for consumer in CONSUMERS:
            timed = run(OPT, program([*ops, HALT]), self.with_pairs, consumer, VERIFY)
            self.assertEqual(timed.returncode, 0, timed.stderr)
            cycles = [c for op, _, c in issue_cycles(timed.stdout) if op == "vpu_unary"]
            self.assertEqual(cycles, [0, 1, self.next_issue("exp")], consumer)

    def test_xlu_waits_for_every_live_vpu_command(self):
        ops = [vpu("exp", SETS[0]), vpu("relu", SETS[1])]
        free = 1 + block(self.facts, "vpu.relu")["first_free_age"]
        late = lambda gap: program([*ops, *([] if gap == 1 else [nop()] if gap == 2 else [delay(gap - 2)]),
                                    transpose(40, 42), delay(400), nop(), HALT])
        self.assertEqual(self.accepts(late(free - 1)).returncode, 0, self.accepts(late(free - 1)).stderr)
        self.assertIn("busy", self.accepts(late(free - 2)).stderr)

    def test_without_pair_facts_nothing_changes(self):
        ops = [vpu("exp", SETS[0]), vpu("relu", SETS[1]), vpu("exp", SETS[2])]
        for consumer in CONSUMERS:
            plain = run(OPT, program([*ops, HALT]), self.plain, consumer, VERIFY)
            cycles = [c for op, _, c in issue_cycles(plain.stdout) if op == "vpu_unary"]
            self.assertEqual(cycles, [0, self.next_issue("exp"), self.next_issue("exp") + self.next_issue("relu")])
        # Programs without two VPU commands time identically with and without pair facts.
        for name, kernel in KERNELS.items():
            for consumer in CONSUMERS:
                with self.subTest(name, consumer=consumer):
                    timed = [run(OPT, program(kernel()), option, consumer, VERIFY) for option in (self.plain, self.with_pairs)]
                    self.assertEqual(stream(timed[0].stdout), stream(timed[1].stdout))


if __name__ == "__main__":
    unittest.main()
