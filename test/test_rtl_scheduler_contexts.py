"""RTL list schedules must remain valid under every concrete block entry."""

import re
import unittest

from test_delay_insertion import EMIT, OPT, addi, nop, run, stream
from test_rtl_timing import (
    FACTS, HALT, VERIFY, VLOAD, branch, export, jal, selected, vls, word_base,
)


@unittest.skipUnless(FACTS and OPT.is_file() and EMIT.is_file(),
                     "requires explicit RTL timing facts and built Atlas tools")
class RTLContextSchedulingTest(unittest.TestCase):
    def schedule(self, ops):
        result = selected(ops, "--schedule-atlas-stream=rtl-best-of=true rtl-dma-estimate=true", VERIFY)
        self.assertEqual(result.returncode, 0, result.stderr)
        emitted = run(EMIT, result.stdout)
        self.assertEqual(emitted.returncode, 0, emitted.stderr)
        return result.stdout

    def test_join_reorders_independent_work_and_preserves_worst_bank_gap(self):
        # The store shares the load's bank only on the else path. Independent
        # scalar work can fill that wait under one common instruction order.
        ops = [addi(6, 0, 256), branch("beq", 7, 0, 6), nop(),
               *word_base(8, 65536 + 256), jal(3), nop(), addi(8, 0, 1280),
               addi(9, 0, 17), VLOAD, vls("vstore", 5, 8), HALT]
        timed = self.schedule(ops)
        printed = stream(timed)
        load = next(i for i, (name, _, _) in enumerate(printed) if name == "vload")
        store = next(i for i, (name, _, _) in enumerate(printed) if name == "vstore")
        scalar = next(i for i, (name, fields, _) in enumerate(printed)
                      if name == "alu_imm" and "dst = 9 : i32" in fields)
        self.assertLess(load, scalar)
        self.assertLess(scalar, store)
        document = export(timed)
        join = next(b for b in document["blocks"] if b.get("entry_contexts"))
        self.assertEqual(sorted(c["entries"][0]["x8"] for c in join["entry_contexts"]),
                         [1280, 65536 + 256])
        instructions = document["instructions"]
        load_cycle = next(i["logical_issue_cycle"] for i in instructions
                          if i["mnemonic"] == "vload")
        store_cycle = next(i["logical_issue_cycle"] for i in instructions
                           if i["mnemonic"] == "vstore")
        self.assertGreaterEqual(store_cycle - load_cycle, 30)

    def test_loop_reordering_keeps_scalar_footprints_and_redirect_slot(self):
        # Every iteration uses the old x6 tile before advancing x6. The
        # independent x9 update moves behind the load, across entry contexts.
        setup = [addi(6, 0, 256), addi(13, 0, 0), addi(14, 0, 3)]
        body = [addi(9, 0, 17), VLOAD, vls("vstore", 4, 6, 32),
                addi(6, 6, 256), addi(13, 13, 1)]
        ops = [*setup, *body, branch("blt", 13, 14, -len(body)), nop(), HALT]
        timed = self.schedule(ops)
        printed = stream(timed)
        load = next(i for i, (name, _, _) in enumerate(printed) if name == "vload")
        scalar = next(i for i, (name, fields, _) in enumerate(printed)
                      if name == "alu_imm" and "dst = 9 : i32" in fields)
        self.assertLess(load, scalar)
        at = next(i for i, (name, _, _) in enumerate(printed) if name == "branch")
        offset = int(re.search(r"offset_bytes = (-?\d+)", printed[at][1]).group(1))
        self.assertEqual(printed[at + offset // 2][0], "vload")
        self.assertEqual(printed[at + 1][0], "alu_imm")
        document = export(timed)
        contexts = document["blocks"][1]["entry_contexts"]
        self.assertEqual([c["entries"][0]["x6"] for c in contexts], [256, 512, 768])
        for context, base in zip(contexts, (256, 512, 768)):
            accesses = context["footprints"][0]["accesses"]
            vmem = next(a for a in accesses if a["resource"] == "vmem")
            self.assertEqual(vmem["first"], base // 8)


if __name__ == "__main__":
    unittest.main()
