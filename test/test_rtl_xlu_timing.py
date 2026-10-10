"""Selected conditional XLU timing from RTL-computed facts.

Set ATLAS_OOT_BIN_DIR and ATLAS_OP_TIMING explicitly.
"""
import tempfile
import unittest

from test_delay_insertion import OPT, EMIT, addi, nop, program, run, without_delays
from test_rtl_timing import (CONSUMERS, FACTS, HALT, MARKER, RESOLVER, block, delay, export, facts,
                             facts_variant, selected, vls)


def transpose(src=3, dst=7):
    return ("xlu_transpose", f"dst = {dst} : i32, src = {src} : i32")


@unittest.skipUnless(FACTS and OPT.is_file() and EMIT.is_file(),
                     "requires explicit RTL timing facts and built tools")
class SelectedXLUTimingTest(unittest.TestCase):
    def selected(self, ops, *passes, **options):
        return selected(ops, *passes, **options)

    def test_both_consumers_and_observed_stream_export(self):
        record = block(facts(), "xlu.transpose")
        for src, dst in ((3, 7), (5, 5), (3, 35), (0, 63), (63, 0)):
            for consumer in CONSUMERS:
                with self.subTest(src=src, dst=dst, consumer=consumer):
                    result = self.selected([transpose(src, dst), HALT], consumer, "--verify-atlas-rtl-timing")
                    self.assertEqual(result.returncode, 0, result.stderr)
                    self.assertIn(RESOLVER, result.stdout)
                    value = export(result.stdout)
                    self.assertFalse(value["scheduling_qualified"])
                    self.assertEqual(value["qualification"], "conditional")
                    instruction = next(x for x in value["instructions"] if "xlu" in x["mnemonic"])
                    footprint = instruction["footprint"]
                    accesses = footprint["accesses"]
                    self.assertEqual(len(accesses), 2)
                    self.assertEqual(sorted(x["age"] for x in accesses),
                                     sorted(record["events"][g]["first_age"] for g in ("read", "write")))
                    self.assertTrue(all(x["count"] == 32 and x["step"] == 1 for x in accesses))
                    self.assertEqual(footprint["done_age"], record["first_free_age"] - 1)
                    self.assertEqual(footprint["write_release"], record["events"]["write"]["last_age"])
                    self.assertEqual(footprint["read_release"], record["events"]["read"]["last_age"] +
                                     record["assumptions"]["scratchpad_read_latency"])
                    self.assertEqual({h["unit"]: h["to"] for h in footprint["holds"]},
                                     {u: record["first_free_age"] - 1 for u in ("XLU", "VLOAD path", "VSTORE path")})
                    self.assertNotEqual(run(EMIT, without_delays(result.stdout)).returncode, 0)

    def test_busy_launch_boundary_applies_to_unrelated_registers(self):
        # DELAY N occupies N+1 cycles; the two launches have gap=N+2.
        boundary = block(facts(), "xlu.transpose")["first_free_age"]
        for gap in (boundary - 1, boundary, boundary + 1):
            result = self.selected([transpose(), delay(gap - 2), transpose(22, 24),
                                    delay(64), nop(), HALT], "--verify-atlas-rtl-timing")
            self.assertEqual(result.returncode == 0, gap >= boundary, result.stderr)

    def test_facts_numbers_drive_the_boundary(self):
        with tempfile.TemporaryDirectory(prefix="atlas-xlu-facts-") as directory:
            slower = facts_variant(directory, lambda d: block(d, "xlu.transpose").update(first_free_age=70))
            for gap, accepted in ((66, False), (69, False), (70, True)):
                with self.subTest(gap=gap):
                    result = self.selected([transpose(), delay(gap - 2), transpose(22, 24), delay(68), nop(), HALT],
                                           "--verify-atlas-rtl-timing", path=slower)
                    self.assertEqual(result.returncode == 0, accepted, result.stderr)
            missing = facts_variant(directory, lambda d: d["op_timing"].remove(block(d, "xlu.transpose")), "missing.json")
            for consumer in CONSUMERS:
                rejected = self.selected([transpose(), HALT], consumer, path=missing)
                self.assertNotEqual(rejected.returncode, 0)
                self.assertIn("xlu.transpose: facts block is missing", rejected.stderr)
            still = self.selected([addi(6, 0, 256), vls("vload"), delay(33), nop(), HALT],
                                  "--verify-atlas-rtl-timing", path=missing)
            self.assertEqual(still.returncode, 0, still.stderr)

    def test_vls_transpose_boundaries_and_marker_are_rechecked(self):
        ops = [addi(6, 0, 0), addi(8, 0, 256), vls("vload", 3),
               transpose(3, 35), vls("vstore", 35, 8), addi(1, 0, 1), MARKER, HALT]
        for consumer in CONSUMERS:
            result = self.selected(ops, consumer, "--verify-atlas-rtl-timing")
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(run(EMIT, result.stdout).returncode, 0)
            self.assertNotEqual(run(OPT, without_delays(result.stdout), "--verify-atlas-rtl-timing").returncode, 0)
            self.assertNotEqual(run(EMIT, without_delays(result.stdout)).returncode, 0)
        for suffix in ([MARKER, delay(64), HALT], [HALT]):
            self.assertNotEqual(self.selected([transpose(), *suffix], "--verify-atlas-rtl-timing").returncode, 0)

    def test_register_ids_are_bounded(self):
        for src, dst in ((-1, 7), (64, 7), (3, -1), (3, 64)):
            self.assertNotEqual(self.selected([transpose(src, dst), HALT], "--insert-atlas-delays").returncode, 0)
        self.assertNotEqual(self.selected([HALT], "--verify-atlas-rtl-timing", digest="0" * 64).returncode, 0)

    def test_pending_dma_requires_wait_before_xlu(self):
        from test_rtl_dma_timing import setup, transfer, wait
        unsafe = [*setup(), transfer(), transpose(), wait(), HALT]
        self.assertNotEqual(self.selected(unsafe, "--insert-atlas-delays", dma="wait").returncode, 0)
        for consumer in CONSUMERS:
            result = self.selected([*setup(), transfer(), wait(), transpose(), HALT],
                                   consumer, "--verify-atlas-rtl-timing", dma="wait")
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn('dma = "wait"', result.stdout)


if __name__ == "__main__":
    unittest.main()
