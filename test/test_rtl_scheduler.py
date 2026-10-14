import re
import unittest

from rtl_test_support import (DMA_SECOND, EMIT, FACTS, HALT, OPT, PAIRS, VERIFY, VLOAD, addi, branch, checked,
    dma_setup, export, jal, nop, program, selection, stream, transfer, unary, vls, wait, word_base,)

LEGACY = "--schedule-atlas-stream=rtl-dma-estimate=false rtl-best-of=false"
ESTIMATE = "--schedule-atlas-stream=rtl-dma-estimate=true rtl-best-of=false"
BEST = "--schedule-atlas-stream=rtl-dma-estimate=true rtl-best-of=true"
BEST_NO_HINT = "--schedule-atlas-stream=rtl-dma-estimate=false rtl-best-of=true"

def issue_cycle(instruction):
    return instruction.get("logical_issue_cycle", instruction.get("minimum_issue_cycle"))

def estimated_dma_cost(document):
    assert len(document["blocks"]) == 1
    pending, stall, last = {}, 0, 0
    for instruction in document["instructions"]:
        mnemonic = instruction["mnemonic"]
        cycle = issue_cycle(instruction) + stall
        match = re.fullmatch(r"dma\.(load|store|wait)\.ch([0-7])", mnemonic)
        footprint = instruction["footprint"]
        if match:
            kind, channel = match[1], int(match[2])
            if kind == "wait":
                ready, _, extra = pending.pop(channel)
                delay = max(0, ready + extra - cycle)
                stall += delay
                cycle += delay
            else:
                window = next(a for a in footprint["accesses"] if a["resource"] == "vmem" and a["at_completion"])
                size = 32 * window["count"]
                latency = max(1, 2 * ((size + 11) // 4), (size + 63) // 64)
                pending[channel] = [cycle + latency, window["first"] // 8192, 0]
        if mnemonic in ("vload", "vstore"):
            window = next(a for a in footprint["accesses"] if a["resource"] == "vmem")
            bank = window["first"] // 8192
            for entry in pending.values():
                if entry[1] == bank:
                    entry[2] += 32
        last = cycle
    assert not pending
    return last

@unittest.skipUnless(FACTS and OPT.is_file() and EMIT.is_file(),
                     "requires explicit RTL timing facts and built Atlas tools")
class RTLSchedulerTest(unittest.TestCase):
    def schedule(self, ops, policy=BEST, pairs=None, dma=None):
        source = ops if isinstance(ops, str) else program(ops)
        timed = checked(self, source, selection(pairs=pairs, dma=dma), policy, VERIFY)
        return timed, checked(self, timed, tool=EMIT), export(timed)

    def test_policy_flags_preserve_legacy_bytes_when_costs_tie(self):
        cases = [([addi(6, 0, 256), VLOAD, vls("vstore", offset=32), HALT], ())]
        cases += [(ops, (selection(),)) for ops in ([HALT], [addi(4, 0, 7), addi(5, 4, 1), HALT])]
        for ops, options in cases:
            source = program(ops)
            legacy = checked(self, source, *options, LEGACY, VERIFY)
            words = checked(self, legacy, tool=EMIT)
            for policy in (ESTIMATE, BEST_NO_HINT, BEST):
                with self.subTest(ops=ops, selected=bool(options), policy=policy):
                    result = checked(self, source, *options, policy, VERIFY)
                    self.assertEqual(result, legacy)
                    self.assertEqual(checked(self, result, tool=EMIT), words)

    def test_dma_hint_launches_transfer_before_independent_compute(self):
        ops = [*dma_setup(size=1024), *word_base(8, 65536), transfer(),
               vls("vload", 8, 8), unary("exp", 12, 16), wait(), vls("vload", 4), HALT]
        legacy = self.schedule(ops, LEGACY, dma="wait")[2]
        mnemonics = lambda d: [i["mnemonic"] for i in d["instructions"]]
        legacy_order = mnemonics(legacy)
        self.assertLess(legacy_order.index("vexp.bf16"), legacy_order.index("dma.load.ch0"))
        for policy in (ESTIMATE, BEST):
            with self.subTest(policy=policy):
                improved = self.schedule(ops, policy, dma="wait")[2]
                order = mnemonics(improved)
                self.assertLess(order.index("dma.load.ch0"), order.index("vexp.bf16"))
                self.assertLess(order.index("vload"), order.index("dma.wait.ch0"))
                self.assertLess(estimated_dma_cost(improved), estimated_dma_cost(legacy))

    def test_best_of_never_increases_estimated_cost(self):
        fixed = [addi(6, 0, 256), VLOAD, vls("vstore", 5, offset=128), unary("exp", 12, 16), HALT]
        cases = [("fixed", fixed, None, lambda d: issue_cycle(d["instructions"][-1]))]
        cases += [(f"DMA bank {base}", [*dma_setup(size=1024), *word_base(8, base), transfer(),
                   vls("vload", 8, 8), unary("exp", 12, 16), wait(), vls("vload", 4), HALT],
                   "wait", estimated_dma_cost) for base in (1024, 65536)]
        cases += [("queued channels", [*dma_setup(), *DMA_SECOND, transfer(), transfer(channel=1, reg=7, dram=3),
                   unary("exp", 12, 16), wait(), vls("vload", 4), wait(1), vls("vload", 8, 7), HALT],
                   "wait", estimated_dma_cost)]
        for name, ops, dma, cost in cases:
            for pairs in (None, PAIRS) if PAIRS else (None,):
                legacy = self.schedule(ops, LEGACY, pairs=pairs, dma=dma)[2]
                for policy in (BEST_NO_HINT, BEST):
                    with self.subTest(name=name, pairs=bool(pairs), policy=policy):
                        improved = self.schedule(ops, policy, pairs=pairs, dma=dma)[2]
                        self.assertLessEqual(cost(improved), cost(legacy))

    def test_join_reorders_independent_work_and_preserves_worst_bank_gap(self):
        ops = [addi(6, 0, 256), branch("beq", 7, 0, 6), nop(),
               *word_base(8, 65536 + 256), jal(3), nop(), addi(8, 0, 1280),
               addi(9, 0, 17), VLOAD, vls("vstore", 5, 8), HALT]
        legacy = self.schedule(ops, LEGACY)[2]
        for policy in (BEST_NO_HINT, BEST):
            with self.subTest(policy=policy):
                timed, _, document = self.schedule(ops, policy)
                printed = stream(timed)
                load = next(i for i, (name, _, _) in enumerate(printed) if name == "vload")
                store = next(i for i, (name, _, _) in enumerate(printed) if name == "vstore")
                scalar = next(i for i, (name, fields, _) in enumerate(printed)
                              if name == "alu_imm" and "dst = 9 : i32" in fields)
                self.assertLess(load, scalar)
                self.assertLess(scalar, store)
                join = next(b for b in document["blocks"] if b.get("entry_contexts"))
                self.assertEqual(sorted(c["entries"][0]["x8"] for c in join["entry_contexts"]), [1280, 65536 + 256])
                instructions = document["instructions"]
                load_cycle = next(i["logical_issue_cycle"] for i in instructions if i["mnemonic"] == "vload")
                store_cycle = next(i["logical_issue_cycle"] for i in instructions if i["mnemonic"] == "vstore")
                self.assertGreaterEqual(store_cycle - load_cycle, 30)
                self.assertLessEqual(issue_cycle(document["instructions"][-1]), issue_cycle(legacy["instructions"][-1]))

    def test_loop_reordering_keeps_scalar_footprints_and_redirect_slot(self):
        setup = [addi(6, 0, 256), addi(13, 0, 0), addi(14, 0, 3)]
        body = [addi(9, 0, 17), VLOAD, vls("vstore", 4, 6, 32), addi(6, 6, 256), addi(13, 13, 1)]
        ops = [*setup, *body, branch("blt", 13, 14, -len(body)), nop(), HALT]
        timed, _, document = self.schedule(ops)
        printed = stream(timed)
        load = next(i for i, (name, _, _) in enumerate(printed) if name == "vload")
        scalar = next(i for i, (name, fields, _) in enumerate(printed)
                      if name == "alu_imm" and "dst = 9 : i32" in fields)
        self.assertLess(load, scalar)
        at = next(i for i, (name, _, _) in enumerate(printed) if name == "branch")
        offset = int(re.search(r"offset_bytes = (-?\d+)", printed[at][1]).group(1))
        self.assertEqual(printed[at + offset // 2][0], "vload")
        self.assertEqual(printed[at + 1][0], "alu_imm")
        self.assertEqual(document["blocks"][1]["successors"], [1, 2])
        self.assertEqual(document["blocks"][1]["entry_state"], "idle_engines_no_pending_dma")
        contexts = document["blocks"][1]["entry_contexts"]
        self.assertEqual([c["entries"][0]["x6"] for c in contexts], [256, 512, 768])
        for context, base in zip(contexts, (256, 512, 768)):
            accesses = context["footprints"][0]["accesses"]
            vmem = next(a for a in accesses if a["resource"] == "vmem")
            self.assertEqual(vmem["first"], base // 8)

if __name__ == "__main__":
    unittest.main()
