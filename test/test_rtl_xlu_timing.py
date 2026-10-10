"""Selected XLU timing from RTL-computed facts (ATLAS_OOT_BIN_DIR, ATLAS_OP_TIMING)."""
import tempfile
import unittest

from test_delay_insertion import OPT, EMIT, addi, nop, run, without_delays
from test_rtl_timing import (CONSUMERS, FACTS, HALT, MARKER, VERIFY, block, delay, export, facts,
                             facts_variant, selected, vls)


def transpose(src=3, dst=7):
    return ("xlu_transpose", f"dst = {dst} : i32, src = {src} : i32")


@unittest.skipUnless(FACTS and OPT.is_file() and EMIT.is_file(),
                     "requires explicit RTL timing facts and built tools")
class SelectedXLUTimingTest(unittest.TestCase):
    def check(self, ops, accepted, *passes, **options):
        result = selected(ops, *passes, **options)
        self.assertEqual(result.returncode == 0, accepted, result.stderr)
        return result

    def test_both_consumers_export_facts_numbers(self):
        record = block(facts(), "xlu.transpose")
        busy = record["first_free_age"] - 1
        for src, dst in ((3, 7), (5, 5), (63, 0)):
            for consumer in CONSUMERS:
                with self.subTest(src=src, dst=dst, consumer=consumer):
                    result = self.check([transpose(src, dst), HALT], True, consumer, VERIFY)
                    footprint = next(x for x in export(result.stdout)["instructions"]
                                     if "xlu" in x["mnemonic"])["footprint"]
                    accesses = footprint["accesses"]
                    self.assertEqual(sorted(x["age"] for x in accesses),
                                     sorted(record["events"][g]["first_age"] for g in ("read", "write")))
                    self.assertTrue(all(x["count"] == 32 and x["step"] == 1 for x in accesses))
                    self.assertEqual(footprint["done_age"], busy)
                    self.assertEqual(footprint["write_release"], record["events"]["write"]["last_age"])
                    self.assertEqual(footprint["read_release"], record["events"]["read"]["last_age"] +
                                     record["assumptions"]["scratchpad_read_latency"])
                    self.assertEqual({h["unit"]: h["to"] for h in footprint["holds"]},
                                     {u: busy for u in ("XLU", "VLOAD path", "VSTORE path")})
                    self.assertNotEqual(run(EMIT, without_delays(result.stdout)).returncode, 0)

    def test_facts_drive_the_busy_boundary(self):
        # DELAY N occupies N+1 cycles; the launches are N+2 apart and unrelated.
        boundary = block(facts(), "xlu.transpose")["first_free_age"]
        for gap in (boundary - 1, boundary):
            self.check([transpose(), delay(gap - 2), transpose(22, 24), delay(64), nop(), HALT],
                       gap >= boundary, VERIFY)
        with tempfile.TemporaryDirectory(prefix="atlas-xlu-facts-") as directory:
            slower = facts_variant(directory, lambda d: block(d, "xlu.transpose").update(first_free_age=70))
            for gap in (69, 70):
                with self.subTest(gap=gap):
                    self.check([transpose(), delay(gap - 2), transpose(22, 24), delay(68), nop(), HALT],
                               gap >= 70, VERIFY, path=slower)
            missing = facts_variant(directory, lambda d: d["op_timing"].remove(block(d, "xlu.transpose")))
            for consumer in CONSUMERS:
                rejected = self.check([transpose(), HALT], False, consumer, path=missing)
                self.assertIn("xlu.transpose: facts block is missing", rejected.stderr)
            self.check([addi(6, 0, 256), vls("vload"), delay(33), nop(), HALT], True, VERIFY, path=missing)

    def test_vls_transpose_chain_and_marker_are_rechecked(self):
        ops = [addi(6, 0, 0), addi(8, 0, 256), vls("vload", 3),
               transpose(3, 35), vls("vstore", 35, 8), addi(1, 0, 1), MARKER, HALT]
        for consumer in CONSUMERS:
            result = self.check(ops, True, consumer, VERIFY)
            self.assertEqual(run(EMIT, result.stdout).returncode, 0)
            self.assertNotEqual(run(OPT, without_delays(result.stdout), VERIFY).returncode, 0)
            self.assertNotEqual(run(EMIT, without_delays(result.stdout)).returncode, 0)
        for suffix in ([MARKER, delay(64), HALT], [HALT]):
            self.check([transpose(), *suffix], False, VERIFY)
        for src, dst in ((-1, 7), (64, 7), (3, -1), (3, 64)):
            self.check([transpose(src, dst), HALT], False, CONSUMERS[0])

    def test_pending_dma_requires_wait_before_xlu(self):
        from test_rtl_dma_timing import setup, transfer, wait
        self.check([*setup(), transfer(), transpose(), wait(), HALT], False, CONSUMERS[0], dma="wait")
        for consumer in CONSUMERS:
            self.check([*setup(), transfer(), wait(), transpose(), HALT], True, consumer, VERIFY, dma="wait")


if __name__ == "__main__":
    unittest.main()
