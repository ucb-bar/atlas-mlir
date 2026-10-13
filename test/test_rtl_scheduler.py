"""RTL scheduler policies and reordered streams remain valid under every concrete block entry."""

import re
import unittest

from rtl_test_support import (
    DMA_SECOND, EMIT, FACTS, HALT, KERNELS, OPT, PAIRS, VERIFY, VLOAD, addi, branch, dma_setup,
    export, jal, nop, program, run, selected, selection, stream, transfer, unary, vls, wait,
    word_base,
)


LEGACY = "--schedule-atlas-stream=rtl-dma-estimate=false rtl-best-of=false"
ESTIMATE = "--schedule-atlas-stream=rtl-dma-estimate=true rtl-best-of=false"
BEST = "--schedule-atlas-stream=rtl-dma-estimate=true rtl-best-of=true"
BEST_NO_HINT = "--schedule-atlas-stream=rtl-dma-estimate=false rtl-best-of=true"


def issue_cycle(instruction):
    return instruction.get("logical_issue_cycle", instruction.get("minimum_issue_cycle"))


def estimated_dma_cost(document):
    """Replay the public final stream with model DMA hints and same-bank LSU cost.

    DELAY and guard cycles are already in the exported minimum coordinates.
    A wait moves all later instructions by the estimated stall. VMEM accesses
    are in 32-byte lines and a bank contains 8192 lines.
    """
    assert len(document["blocks"]) == 1
    pending, extra, stall = {}, {}, 0
    last = 0
    for instruction in document["instructions"]:
        mnemonic = instruction["mnemonic"]
        cycle = issue_cycle(instruction) + stall
        match = re.fullmatch(r"dma\.(load|store|wait)\.ch([0-7])", mnemonic)
        footprint = instruction["footprint"]
        if match:
            kind, channel = match[1], int(match[2])
            if kind == "wait":
                ready, _ = pending.pop(channel)
                delay = max(0, ready + extra.pop(channel) - cycle)
                stall += delay
                cycle += delay
            else:
                window = next(a for a in footprint["accesses"]
                              if a["resource"] == "vmem" and a["at_completion"])
                byte_count = 32 * window["count"]
                latency = max(1, 2 * ((byte_count + 11) // 4), (byte_count + 63) // 64)
                pending[channel] = (cycle + latency, window["first"] // 8192)
                extra[channel] = 0
        if mnemonic in ("vload", "vstore"):
            window = next(a for a in footprint["accesses"] if a["resource"] == "vmem")
            bank = window["first"] // 8192
            for channel, (_, dma_bank) in pending.items():
                if dma_bank == bank:
                    extra[channel] += 32
        last = cycle
    assert not pending
    return last


@unittest.skipUnless(FACTS and OPT.is_file() and EMIT.is_file(),
                     "requires explicit RTL timing facts and built Atlas tools")
class RTLSchedulerTest(unittest.TestCase):
    def schedule(self, ops, policy=BEST, pairs=None, dma=None):
        source = ops if isinstance(ops, str) else program(ops)
        result = run(OPT, source, selection(pairs=pairs, dma=dma), policy, VERIFY)
        self.assertEqual(result.returncode, 0, result.stderr)
        emitted = run(EMIT, result.stdout)
        self.assertEqual(emitted.returncode, 0, emitted.stderr)
        return result.stdout, emitted.stdout, export(result.stdout)

    def test_model_provider_ignores_rtl_policy_flags(self):
        source = program([addi(6, 0, 256), vls("vload"), vls("vstore", offset=32), HALT])
        legacy = run(OPT, source, LEGACY, VERIFY)
        self.assertEqual(legacy.returncode, 0, legacy.stderr)
        words = run(EMIT, legacy.stdout)
        self.assertEqual(words.returncode, 0, words.stderr)
        for policy in (ESTIMATE, BEST_NO_HINT, BEST):
            with self.subTest(policy=policy):
                result = run(OPT, source, policy, VERIFY)
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertEqual(result.stdout, legacy.stdout)
                self.assertEqual(run(EMIT, result.stdout).stdout, words.stdout)

    def test_disabled_policy_is_repeatable_and_ties_keep_legacy_bytes(self):
        for ops in ([HALT], [addi(4, 0, 7), addi(5, 4, 1), HALT]):
            legacy = self.schedule(ops, LEGACY)
            self.assertEqual(self.schedule(ops, LEGACY)[:2], legacy[:2])
            for policy in (ESTIMATE, BEST_NO_HINT, BEST):
                with self.subTest(ops=ops, policy=policy):
                    self.assertEqual(self.schedule(ops, policy)[:2], legacy[:2])

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

    def test_best_of_never_increases_fixed_latency_block_cost(self):
        for pairs in (None, PAIRS) if PAIRS else (None,):
            for name, make in KERNELS.items():
                legacy = self.schedule(make(), LEGACY, pairs=pairs)[2]
                limit = issue_cycle(legacy["instructions"][-1])
                for policy in (BEST_NO_HINT, BEST):
                    with self.subTest(name=name, pairs=bool(pairs), policy=policy):
                        improved = self.schedule(make(), policy, pairs=pairs)[2]
                        self.assertLessEqual(issue_cycle(improved["instructions"][-1]), limit)

    def test_best_of_never_increases_documented_dma_estimate(self):
        # Same bank, disjoint windows; another bank; and two queued channels.
        # A consumer after each wait binds the transfer to useful work.
        cases = {
            "same bank": [*dma_setup(size=1024), *word_base(8, 1024), transfer(),
                          vls("vload", 8, 8), unary("exp", 12, 16), wait(), vls("vload", 4), HALT],
            "other bank": [*dma_setup(size=1024), *word_base(8, 65536), transfer(),
                           vls("vload", 8, 8), unary("exp", 12, 16), wait(), vls("vload", 4), HALT],
            "queued channels": [*dma_setup(), *DMA_SECOND, transfer(), transfer(channel=1, reg=7, dram=3),
                                unary("exp", 12, 16), wait(), vls("vload", 4),
                                wait(1), vls("vload", 8, 7), HALT],
        }
        for name, ops in cases.items():
            legacy = self.schedule(ops, LEGACY, dma="wait")[2]
            for policy in (BEST_NO_HINT, BEST):
                with self.subTest(name=name, policy=policy):
                    improved = self.schedule(ops, policy, dma="wait")[2]
                    self.assertLessEqual(estimated_dma_cost(improved), estimated_dma_cost(legacy))

    def test_dma_estimate_preserves_unknown_completion_and_real_waits(self):
        ops = [*dma_setup(size=1024), transfer(), unary("exp", 12, 16), wait(), vls("vload", 4), HALT]
        for policy in (LEGACY, ESTIMATE, BEST_NO_HINT, BEST):
            with self.subTest(policy=policy):
                _, _, document = self.schedule(ops, policy, dma="wait")
                launch = next(i for i in document["instructions"] if i["mnemonic"] == "dma.load.ch0")
                retired = next(i for i in document["instructions"] if i["mnemonic"] == "dma.wait.ch0")
                self.assertIsNone(launch["footprint"]["done_age"])
                self.assertIsNone(launch["footprint"]["dma_cycles"])
                self.assertGreater(retired["issue_epoch"], launch["issue_epoch"])
                self.assertIsNone(document["applicability"]["dma_domain"]["completion_latency"])

    def test_join_order_is_valid_in_every_bank_context(self):
        # Unknown x7 chooses either a same-bank or a different-bank store.
        join = [addi(6, 0, 256), branch("beq", 7, 0, 6), nop(),
                *word_base(8, 65536 + 256), jal(3), nop(), addi(8, 0, 1280),
                vls("vload", 4), vls("vstore", 5, 8), unary("exp", 12, 16), HALT]
        legacy = self.schedule(join, LEGACY)[2]
        for policy in (BEST_NO_HINT, BEST):
            with self.subTest(policy=policy):
                _, _, document = self.schedule(join, policy)
                block = next(b for b in document["blocks"] if "entry_contexts" in b)
                self.assertEqual(len(block["entry_contexts"]), 2)
                self.assertEqual({c["entries"][0]["x8"] for c in block["entry_contexts"]}, {1280, 65792})
                self.assertLessEqual(issue_cycle(document["instructions"][-1]),
                                     issue_cycle(legacy["instructions"][-1]))

    def test_loop_reordering_preserves_induction_and_branch_target(self):
        body = [addi(6, 0, 256), addi(13, 0, 0), addi(14, 0, 3),
                vls("vload", 4), unary("exp", 12, 16), addi(6, 6, 256), addi(13, 13, 1),
                branch("blt", 13, 14, -4), nop(), HALT]
        _, _, document = self.schedule(body)
        loop = document["blocks"][1]
        self.assertEqual(loop["successors"], [1, 2])
        self.assertEqual({c["entries"][0]["x6"] for c in loop["entry_contexts"]}, {256, 512, 768})
        self.assertEqual(loop["entry_state"], "idle_engines_no_pending_dma")

    def test_policy_does_not_admit_unknown_addresses_or_invalid_dma_lifetimes(self):
        cases = [([vls("vload"), HALT], "unknown VLS base"),
                 ([*dma_setup(), transfer(), vls("vload"), wait(), HALT], "in flight"),
                 ([*dma_setup(), transfer(), wait(1), HALT], "channel 1")]
        for ops, diagnostic in cases:
            for policy in (ESTIMATE, BEST_NO_HINT, BEST):
                with self.subTest(diagnostic=diagnostic, policy=policy):
                    result = run(OPT, program(ops), selection(dma="wait"), policy, VERIFY)
                    self.assertNotEqual(result.returncode, 0)
                    self.assertIn(diagnostic, result.stderr)


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
        dma_setup = [addi(6, 0, 256), addi(13, 0, 0), addi(14, 0, 3)]
        body = [addi(9, 0, 17), VLOAD, vls("vstore", 4, 6, 32),
                addi(6, 6, 256), addi(13, 13, 1)]
        ops = [*dma_setup, *body, branch("blt", 13, 14, -len(body)), nop(), HALT]
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
