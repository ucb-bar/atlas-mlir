"""Selected issue gaps between VLS, VPU and XLU operations (ATLAS_OOT_BIN_DIR, ATLAS_OP_TIMING).

Every gap comes from the facts: a VPU admits the next VPU command at its next_issue_age and the XLU at
its first_free_age, and the XLU keeps both out until its first_free_age. VLOAD and VSTORE hold only their
own path and overlap each other, the VPU and the XLU. An MXU command holds only its own MXU, so it overlaps the
other MXU and every other engine. Operands are disjoint (MREG physical banks, VMEM banks), so only engine
admission decides.
"""
import tempfile
import unittest

from test_delay_insertion import OPT, EMIT, nop, program, run, stream
from test_rtl_gap_coverage import gapped, issue_cycles, issue_gap
from test_rtl_timing import (CONSUMERS, FACTS, HALT, PROVIDER, VERIFY, block, facts, facts_variant,
                             selection, vls, word_base)
from test_rtl_vmul_timing import reduce, unary, vli, vmul
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


def first_free(document, name):
    return block(document, name)["first_free_age"]


def expected_gap(document, first, second):
    if first.startswith("mxu") or second.startswith("mxu"):
        return 1  # MXU0 and MXU1 differ; an MXU meets the other engines only at MREG ports and release rules
    if first.startswith("vpu.") and second.startswith("vpu."):
        return block(document, first)["next_issue_age"]
    if first.startswith("vlsu.") != second.startswith("vlsu.") or {first, second} == {"vlsu.vload", "vlsu.vstore"}:
        return 1  # VLOAD, VSTORE and the VPU/XLU pair overlap; banks and MREGs are disjoint
    return first_free(document, first)


def pairs():
    for x, (fa, sa, a) in FIRST.items():
        for y, (fb, sb, b) in SECOND.items():
            yield f"{x}->{y}", fa, fb, sa + sb, a, b


def cycles(printed):
    """Issue cycle of the terminal."""
    return next(cycle for op, _, cycle in issue_cycles(printed) if op == "trap")


def tile_kernel(tiles=4, pipelined=False):
    """Per tile: load two BF16 pairs, add them, store the sum pair. Tile t uses m(8t)..m(8t+5), reads VMEM
    bank t%3 and writes bank 3+t%3 (a later tile on the same bank uses the next lines). Pipelined, tile t+1's
    loads precede tile t's add in program order."""
    setup, loads, compute = [], [], []
    for t in range(tiles):
        r, src, dst, line = 8 * t, 10 + 2 * t, 11 + 2 * t, 128 * (t // 3)
        setup += [*word_base(src, (t % 3) * BANK_WORDS + line * 8), *word_base(dst, (3 + t % 3) * BANK_WORDS + line * 8)]
        loads.append([vls("vload", r + k, src, 32 * k) for k in range(4)])
        compute.append([vmul(r, r + 2, r + 4, "add"), vls("vstore", r + 4, dst), vls("vstore", r + 5, dst, 32)])
    ops = list(setup)
    if pipelined:
        ops += loads[0]
        for t in range(tiles):
            ops += (loads[t + 1] if t + 1 < tiles else []) + compute[t]
    else:
        for t in range(tiles):
            ops += loads[t] + compute[t]
    return ops + [HALT]


def copy_kernel(tiles=4):
    """Stream `tiles` tiles from VMEM bank 0 to bank 1 through m(4+t)."""
    ops = [*word_base(6, 0), *word_base(8, BANK_WORDS)]
    for t in range(tiles):
        ops += [vls("vload", 4 + t, 6, 32 * t), vls("vstore", 4 + t, 8, 32 * t)]
    return ops + [HALT]


KERNELS = {"four-tile load-load-add-store": tile_kernel,
           "four-tile pipelined load-load-add-store": lambda: tile_kernel(pipelined=True),
           "four-tile copy": copy_kernel}


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
                    self.assertRegex(rejected.stderr, "timing resource conflict: .* busy; issue cycle .*, first permitted cycle")

    def test_both_consumers_issue_at_the_boundary(self):
        document = facts()
        for name, fa, fb, setup, a, b in pairs():
            for consumer in CONSUMERS:
                with self.subTest(pair=name, consumer=consumer):
                    result = run(OPT, program([*setup, a, b, HALT]), selection(), consumer, VERIFY)
                    self.assertEqual(result.returncode, 0, result.stderr)
                    gap, expected = issue_gap(result.stdout, a, b), expected_gap(document, fa, fb)
                    if consumer == CONSUMERS[0]:
                        self.assertEqual(gap, expected)
                    elif gap > 0:  # the scheduler may issue b first, or b's scalar setup between them
                        self.assertTrue(expected <= gap <= expected + len(setup), (gap, expected))

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
                        self.assertIn(f"{PROVIDER}: vpu.add: facts block lacks a next_issue_age", result.stderr)
            # A later next_issue_age moves only the VPU->VPU boundary.
            path = facts_variant(directory, lambda d: block(d, "vpu.add").update(next_issue_age=60))
            for b, boundary in ((relu, 60), (SECOND["xlu"][2], 66)):
                with self.subTest(second=b[0]):
                    self.assertEqual(self.verify(gapped(setup, add, b, boundary), path).returncode, 0)
                    self.assertNotEqual(self.verify(gapped(setup, add, b, boundary - 1), path).returncode, 0)

    def test_same_bank_vload_vstore_is_ordered_by_vmem_bank_windows(self):
        # The RTL asserts on a VLOAD read and VSTORE write to one bank in the same cycle (LSU.scala).
        load, store = block(facts(), "vlsu.vload"), block(facts(), "vlsu.vstore")
        read, write = load["events"]["vmem_read"], store["events"]["vmem_write"]
        setup = [*word_base(6, 0), *word_base(8, 1024)]
        for a, b, boundary in ((vls("vload", 4, 6), vls("vstore", 9, 8), read["last_age"] - write["first_age"] + 1),
                               (vls("vstore", 9, 8), vls("vload", 4, 6), write["last_age"] - read["first_age"] + 1)):
            for gap in (1, boundary - 1, boundary):
                with self.subTest(first=a[0], gap=gap):
                    result = self.verify(gapped(setup, a, b, gap))
                    self.assertEqual(result.returncode == 0, gap >= boundary, result.stderr)
                    if gap < boundary:
                        self.assertIn("VMEM bank 0 busy", result.stderr)

    def test_vstore_source_is_reported_read_until_it_drains(self):
        # The MREG bank tracker reports a VSTORE's source as read through its busy window: a VPU, XLU or MXU pop
        # writing it waits for first_free_age, while a VLOAD (which checks only pending writes) follows the read
        # release (pairs/core_mxu_cross.yaml: vstore_transpose_to_m9, vstore_vli_all_m8, vstore_mxu*_pop_acc_bf16_m8).
        store = block(facts(), "vlsu.vstore")
        drained = store["first_free_age"]
        released = store["events"]["mreg_read"]["last_age"] + store["assumptions"]["scratchpad_read_latency"] + 1
        setup = [*word_base(8, 2 * BANK_WORDS), *word_base(7, 3 * BANK_WORDS)]
        writers = {"transpose": (transpose(30, 9), drained), "vli": (vli("all", 8), drained),
                   "pop": (("mxu_pop", 'format = "bf16", unit = 0 : i32, dst = 8 : i32, slot = 0 : i32, scale_reg = 0 : i32'), drained),
                   "vload": (vls("vload", 9, 7), released)}
        for name, (writer, boundary) in writers.items():
            with self.subTest(writer=name):
                self.assertGreater(boundary, 1)
                self.assertEqual(self.verify(gapped(setup, vls("vstore", 9, 8), writer, boundary)).returncode, 0)
                rejected = self.verify(gapped(setup, vls("vstore", 9, 8), writer, boundary - 1))
                self.assertNotEqual(rejected.returncode, 0)
                self.assertIn("reported read until" if boundary == drained else "reserved for reading", rejected.stderr)

    def test_example_kernels(self):
        for name, kernel in KERNELS.items():
            for consumer in CONSUMERS:
                with self.subTest(kernel=name, consumer=consumer):
                    result = run(OPT, program(kernel()), selection(), consumer, VERIFY)
                    self.assertEqual(result.returncode, 0, result.stderr)
                    self.assertEqual(run(EMIT, result.stdout).returncode, 0)
                    print(f"\n{consumer}: {name} kernel halts at cycle {cycles(result.stdout)}")


if __name__ == "__main__":
    unittest.main()
