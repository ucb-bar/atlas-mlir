"""Selected issue gaps between VLS, VPU and XLU operations (ATLAS_OOT_BIN_DIR, ATLAS_OP_TIMING).

Every gap comes from the facts: a VPU admits the next VPU command at its next_issue_age and keeps the
other engines out until its first_free_age; VLS and XLU keep every engine out until their first_free_age.
An MXU command excludes every engine through its last event. Operands are disjoint (MREG physical banks, VMEM banks), so only engine admission decides.
"""
import functools
import tempfile
import unittest

from test_delay_insertion import OPT, EMIT, nop, program, run, stream
from test_rtl_gap_coverage import gapped, issue_cycles, issue_gap
from test_rtl_timing import (CONSUMERS, FACTS, HALT, RESOLVER, VERIFY, block, export, facts, facts_variant,
                             selection, vls, word_base)
from test_rtl_vmul_timing import reduce, unary, vmul
from test_rtl_xlu_timing import transpose

BANK_WORDS = 8192 * 8


def matmul(unit, src):
    return ("mxu_matmul", f"unit = {unit} : i32, src = {src} : i32, weight_slot = 0 : i32, acc_slot = 0 : i32, "
                          "accumulate = false")


# name: (facts block, setup, operation); first and second operations use disjoint resources.
FIRST = {
    "vload": ("vlsu.vload", word_base(6, 0), vls("vload", 4, 6)),
    "vstore": ("vlsu.vstore", word_base(8, 2 * BANK_WORDS), vls("vstore", 9, 8)),
    "add": ("vpu.add", [], vmul(16, 20, 24, "add")),
    "rmax": ("vpu.rmax", [], reduce("row_max", 12, 14)),
    "xlu": ("xlu.transpose", [], transpose(30, 40)),
    "mxu": ("mxu0.matmul", [], matmul(0, 50)),
}
SECOND = {
    "vload": ("vlsu.vload", word_base(7, 3 * BANK_WORDS), vls("vload", 5, 7)),
    "vstore": ("vlsu.vstore", word_base(9, 4 * BANK_WORDS), vls("vstore", 11, 9)),
    "relu": ("vpu.relu", [], unary("relu", 0, 6)),
    "exp": ("vpu.exp", [], unary("exp", 2, 28)),
    "xlu": ("xlu.transpose", [], transpose(31, 42)),
    "mxu": ("mxu1.matmul", [], matmul(1, 51)),
}


@functools.lru_cache(maxsize=None)
def mxu_free(name):
    """One past the last held age of an MXU command's exported footprint."""
    op = next(op for fa, _, op in (*FIRST.values(), *SECOND.values()) if fa == name)
    result = run(OPT, program([op, HALT]), selection(), CONSUMERS[0], VERIFY)
    holds = [h for i in export(result.stdout)["instructions"] for h in i["footprint"]["holds"]]
    return max(h["to"] for h in holds) + 1


def first_free(document, name):
    return mxu_free(name) if name.startswith("mxu") else block(document, name)["first_free_age"]


def expected_gap(document, first, second):
    if first.startswith("mxu") or second.startswith("mxu"):
        return first_free(document, first)
    if first.startswith("vpu.") and second.startswith("vpu."):
        return block(document, first)["next_issue_age"]
    return first_free(document, first)


def pairs():
    for x, (fa, sa, a) in FIRST.items():
        for y, (fb, sb, b) in SECOND.items():
            yield f"{x}->{y}", fa, fb, sa + sb, a, b


def cycles(printed):
    """Issue cycle of the terminal."""
    return next(cycle for op, _, cycle in issue_cycles(printed) if op == "trap")


def tile_kernel(tiles=4):
    """Per tile: load two BF16 pairs, add them, store the sum pair. Tile t uses m(8t)..m(8t+5), reads VMEM
    bank t%3 and writes bank 3+t%3 (a later tile on the same bank uses the next lines)."""
    ops = []
    for t in range(tiles):
        r, src, dst, line = 8 * t, 10 + 2 * t, 11 + 2 * t, 128 * (t // 3)
        ops += [*word_base(src, (t % 3) * BANK_WORDS + line * 8), *word_base(dst, (3 + t % 3) * BANK_WORDS + line * 8)]
        ops += [vls("vload", r + k, src, 32 * k) for k in range(4)]
        ops += [vmul(r, r + 2, r + 4, "add"), vls("vstore", r + 4, dst), vls("vstore", r + 5, dst, 32)]
    return ops + [HALT]


@unittest.skipUnless(FACTS and OPT.is_file() and EMIT.is_file(),
                     "requires explicit RTL timing facts and built tools")
class SelectedOverlapTest(unittest.TestCase):
    def verify(self, source, path=None):
        options = [selection(path)] if path else [selection()]
        return run(OPT, source, *options, VERIFY)

    def test_issue_gap_boundaries_come_from_the_facts(self):
        document = facts()
        for name, fa, fb, setup, a, b in pairs():
            gap = expected_gap(document, fa, fb)
            with self.subTest(pair=name, gap=gap):
                accepted = self.verify(gapped(setup, a, b, gap))
                self.assertEqual(accepted.returncode, 0, accepted.stderr)
                if gap > 1:
                    rejected = self.verify(gapped(setup, a, b, gap - 1))
                    self.assertNotEqual(rejected.returncode, 0)
                    # MXU commands are admitted by the same holds through the reservation check.
                    self.assertRegex(rejected.stderr, "serialized engine admission: .* busy; issue cycle .*, first "
                                                      "permitted cycle|selected RTL reservation: .* busy")

    def test_both_consumers_issue_at_the_boundary(self):
        document = facts()
        for name, fa, fb, setup, a, b in pairs():
            for consumer in CONSUMERS:
                with self.subTest(pair=name, consumer=consumer):
                    result = run(OPT, program([*setup, a, b, HALT]), selection(), consumer, VERIFY)
                    self.assertEqual(result.returncode, 0, result.stderr)
                    gap = issue_gap(result.stdout, a, b)
                    if consumer == CONSUMERS[0] or gap > 0:  # the scheduler may issue b first
                        self.assertEqual(gap, expected_gap(document, fa, fb))

    def test_vpu_issue_window_fails_closed(self):
        setup = []
        add, relu = FIRST["add"][2], SECOND["relu"][2]
        with tempfile.TemporaryDirectory(prefix="atlas-overlap-facts-") as directory:
            for value in (None, 0, 67):
                path = facts_variant(directory, lambda d: block(d, "vpu.add").update(next_issue_age=value))
                for consumer in CONSUMERS:
                    with self.subTest(next_issue_age=value, consumer=consumer):
                        result = run(OPT, program([add, HALT]), selection(path), consumer)
                        self.assertNotEqual(result.returncode, 0)
                        self.assertIn(f"{RESOLVER}: vpu.add: facts block lacks a next_issue_age", result.stderr)
            # A later next_issue_age moves only the VPU->VPU boundary.
            path = facts_variant(directory, lambda d: block(d, "vpu.add").update(next_issue_age=60))
            vstore = SECOND["vstore"]
            for b, setup, boundary in ((relu, setup, 60), (vstore[2], vstore[1], 66)):
                with self.subTest(second=b[0]):
                    self.assertEqual(self.verify(gapped(setup, add, b, boundary), path).returncode, 0)
                    self.assertNotEqual(self.verify(gapped(setup, add, b, boundary - 1), path).returncode, 0)

    def test_example_tile_kernel(self):
        for consumer in CONSUMERS:
            with self.subTest(consumer=consumer):
                result = run(OPT, program(tile_kernel()), selection(), consumer, VERIFY)
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertEqual(run(EMIT, result.stdout).returncode, 0)
                print(f"\n{consumer}: four-tile load-load-add-store kernel halts at cycle {cycles(result.stdout)}")


if __name__ == "__main__":
    unittest.main()
