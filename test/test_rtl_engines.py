"""Engine admission, operand domains and DMA wait contracts."""
import re
import tempfile
import unittest

from rtl_test_support import (BANK_WORDS, CHANNEL_REUSE, CONSUMERS, DMA_SECOND, EIGHT_SLOTS, EMIT, FACTS,
    HALT, IN_FLIGHT, MARKER, OPT, PROVIDER, RTLTestSupport, VERIFY, WAITED,
    addi, binary, block, busy, checked, config, delay, export, facts, facts_variant,
    gapped, issue_gap, mxu, pack, program, reduce, run, stream, transfer,
    transpose, unary, vli, vls, wait, word_base, dma_setup,)


@unittest.skipUnless(FACTS and OPT.is_file() and EMIT.is_file(), "requires explicit RTL timing facts and built tools")
class SelectedEngineTimingTest(RTLTestSupport, unittest.TestCase):
    def test_engine_admission_boundaries(self):
        doc = facts()
        mul = block(doc, "vpu.mul")
        cases = [
            (transpose(), transpose(22, 24), block(doc, "xlu.transpose")["first_free_age"], "timing resource conflict"),
            (transpose(), binary(16, 20, 24), block(doc, "xlu.transpose")["first_free_age"],
             "timing resource conflict"),
            (binary(), binary(6, 8, 10), mul["next_issue_age"], "timing resource conflict"),
            (binary(), transpose(8, 12), mul["first_free_age"], "timing resource conflict"),
            (binary(), vls("vstore", 4), max(mul["next_issue_age"], mul["events"]["write0"]["last_age"]) + 1,
             "m4 reserved for writing"),]
        for unit in (0, 1):
            matmul = f"mxu{unit}.matmul"
            pop = f"mxu{unit}.pop_acc_bf16"
            event = lambda name, group: block(doc, name)["events"][group]["last_age"]
            latency = block(doc, "vlsu.vload")["assumptions"]["scratchpad_read_latency"]
            read_age = event(matmul, "mreg_read0")
            store_age = block(doc, "vlsu.vstore")["events"]["mreg_read"]["first_age"]
            cases += [
                (mxu("pop_acc_bf16", unit, reg=22), vls("vstore", 22),
                 event(pop, "mreg_write0") + 1, "reserved for writing"),
                (vls("vload", 22), mxu("matmul", unit, reg=22),
                 event("vlsu.vload", "mreg_write") + 1, "reserved for writing"),
                (mxu("matmul", unit, reg=22), vls("vload", 22), read_age + latency + 1, "reserved for reading"),
                (mxu("matmul", unit, reg=12), vls("vstore", 44),
                 read_age - store_age + 1, "port busy"),
                (mxu("matmul", unit), mxu("pop_acc_bf16", unit, reg=22),
                 busy(block(doc, matmul)) + 1, "MXU in-flight matmuls"),
            ]
        cases = [([addi(6, 0, 256)], *case) for case in cases]
        store = block(doc, "vlsu.vstore")
        drained = store["first_free_age"]
        released = store["events"]["mreg_read"]["last_age"] + store["assumptions"]["scratchpad_read_latency"] + 1
        setup = [*word_base(8, 2 * BANK_WORDS), *word_base(7, 3 * BANK_WORDS)]
        writers = [(transpose(30, 9), drained), (vli("all", 8), drained),
                   *((mxu("pop_acc_bf16", unit, reg=8), drained) for unit in (0, 1)), (vls("vload", 9, 7), released)]
        cases += [(setup, vls("vstore", 9, 8), writer, bound,
                   "reported read until" if bound == drained else "reserved for reading") for writer, bound in writers]
        for setup, first, second, bound, diagnostic in cases:
            for gap in (bound - 1, bound):
                with self.subTest(first=first, second=second, gap=gap):
                    self.check(gapped(setup, first, second, gap), gap == bound, VERIFY,
                               diagnostic=diagnostic if gap < bound else None)

    def test_admission_uses_selected_facts(self):
        cases = [("mxu1.matmul", lambda b: b["events"]["compute_busy"].update(last_age=40, count=40),
             mxu("matmul", 1), mxu("matmul", 1, reg=22), 41, "MXU in-flight matmuls"),
            ("xlu.transpose", lambda b: b.update(first_free_age=70),
             transpose(), transpose(22, 24), 70, "timing resource conflict"),
            ("vpu.mul", lambda b: b.update(first_free_age=70, next_issue_age=69),
             binary(), binary(6, 8, 10), 69, "timing resource conflict"),
            ("vpu.mul", lambda b: b.update(first_free_age=70, next_issue_age=69),
             binary(), transpose(8, 12), 70, "timing resource conflict"),
            ("vpu.add", lambda b: b.update(next_issue_age=60),
             binary(16, 20, 24, "add"), unary("relu", 0, 6), 60, "timing resource conflict"),]
        with tempfile.TemporaryDirectory() as directory:
            for name, mutate, first, second, bound, diagnostic in cases:
                path = facts_variant(directory, lambda d: mutate(block(d, name)))
                for gap in (bound - 1, bound):
                    with self.subTest(record=name, gap=gap):
                        self.check(gapped([], first, second, gap), gap == bound, VERIFY, path=path,
                                   diagnostic=diagnostic if gap < bound else None)

    def test_engine_facts_fail_closed(self):
        cases = [("mxu1.matmul", lambda b: b["events"]["acc_load"].update(first_age=1, last_age=32, count=32, step=1),
             mxu("matmul", 1), "event group acc_load is not modeled"),
            ("mxu1.matmul", lambda b: b["events"].pop("compute_busy"),
             mxu("matmul", 1), "event group compute_busy is missing"),
            ("mxu1.matmul", lambda b: b["events"]["acc_write"].update(last_age=40), mxu("matmul", 1), "does not sweep"),
            ("vpu.mul", lambda b: b["events"]["write0"].update(count=32), binary(), "does not sweep 1x64"),
            *(("vpu.add", lambda b, value=value: b.update(next_issue_age=value),
               binary(kind="add"), "facts block lacks a next_issue_age") for value in (None, 0, 67)),]
        with tempfile.TemporaryDirectory() as directory:
            for name, mutate, operation, diagnostic in cases:
                path = facts_variant(directory, lambda d: mutate(block(d, name)))
                for consumer in CONSUMERS:
                    with self.subTest(record=name, mutation=diagnostic, consumer=consumer):
                        self.check([operation, HALT], False, consumer, path=path, diagnostic=diagnostic)

    def test_register_domains(self):
        pairs = [((62, 2, 4), True), ((0, 60, 62), True), ((1, 2, 4), False), ((0, 3, 4), False),
                 ((0, 2, 5), False), ((0, 2, 64), False), ((-2, 2, 4), False), ((0, 0, 4), False),
                 ((0, 32, 4), False), ((0, 2, 0), False), ((0, 2, 34), False)]
        for operands, accepted in pairs:
            for consumer in CONSUMERS:
                with self.subTest(operands=operands, consumer=consumer):
                    self.check([binary(*operands), HALT], accepted, consumer)
        for op in (unary("mov", 0, 32), reduce("row_sum", 32, 0), reduce("col_sum", 0, 0),
                   pack("bf16_to_fp8", 0, 1), pack("fp8_to_bf16", 4, 4), unary("square", 2, 34)):
            with self.subTest(operation=op):
                self.check([op, HALT], False, CONSUMERS[0], diagnostic=PROVIDER)
        for src, dst, accepted in ((0, 63, True), (63, 0, True), (-1, 7, False),
                                   (64, 7, False), (3, -1, False), (3, 64, False)):
            with self.subTest(src=src, dst=dst):
                self.check([transpose(src, dst), HALT], accepted, CONSUMERS[0])

    def test_engines_run_beside_pending_dma(self):
        for operation in (binary(), transpose(), *(mxu("matmul", u) for u in (0, 1))):
            for first, second in ((transfer(), operation), (operation, transfer())):
                for consumer in CONSUMERS:
                    with self.subTest(operation=operation, order=first[0], consumer=consumer):
                        result = self.check([*dma_setup(), first, second, wait(), HALT],
                                            True, consumer, VERIFY, dma="wait")
                        checked(self, result.stdout, tool=EMIT)
                        order = [op for op, _, _ in stream(result.stdout)]
                        self.assertLess(order.index(operation[0]), order.index("dma_wait"))
                        if consumer == CONSUMERS[0]:
                            self.assertEqual(issue_gap(result.stdout, first, second), 1)

    def test_unselected_mxu_rules(self):
        for first, second, gap in ((mxu("matmul", 1), mxu("matmul", 1, reg=22), 32),
                                   (mxu("matmul", 0), mxu("pop_acc_bf16", 0, reg=22), 64)):
            with self.subTest(unit=first):
                text = checked(self, program([first, second, HALT]), CONSUMERS[0])
                self.assertIn(f"cycles = {gap - 2} : i32", text)
                self.assertNotIn("atlas.rtl_evidence", text)


@unittest.skipUnless(FACTS and OPT.is_file() and EMIT.is_file(), "requires explicit RTL timing facts and built tools")
class SelectedDMATimingTest(RTLTestSupport, unittest.TestCase):
    def check_consumers(self, ops, accepted, diagnostic=None):
        for consumer in CONSUMERS:
            with self.subTest(consumer=consumer):
                result = self.check(ops, accepted, consumer, VERIFY, dma="wait", diagnostic=diagnostic)
                if accepted:
                    checked(self, result.stdout, tool=EMIT)

    def test_export_completion_is_unknown(self):
        for consumer in CONSUMERS:
            with self.subTest(consumer=consumer):
                checked = self.check(WAITED, True, consumer, VERIFY, dma="wait")
                self.assertIn('dma = "wait"', checked.stdout)
                data = export(checked.stdout)
                self.assertEqual(data["schema"], "atlas.resolved_rtl_timing.v1")
                self.assertEqual(data["provider"], {"id": PROVIDER, "version": 1, "dma_policy": "wait"})
                self.assertIn("dma.load.ch0..7", data["applicability"]["supported_operations"])
                self.assertIsNone(data["applicability"]["dma_domain"]["completion_latency"])
                self.assertEqual(data["applicability"]["dma_domain"]["pending_limit"], 8)
                launch = next(i for i in data["instructions"] if i["mnemonic"] == "dma.load.ch0")
                retired = next(i for i in data["instructions"] if i["mnemonic"] == "dma.wait.ch0")
                self.assertEqual((launch["footprint"]["done_age"], launch["footprint"]["dma_cycles"]), (None, None))
                memory = [a for a in launch["footprint"]["accesses"] if a["resource"] == "vmem"]
                self.assertTrue(memory and all(a["at_completion"] and a["age"] is None and a["step"] is None
                                               for a in memory))
                self.assertGreater(retired["issue_epoch"], launch["issue_epoch"])
                self.assertEqual(retired["epoch_offset"], 0)
                self.assertNotIn("logical_issue_cycle", launch)
                self.assertIn("minimum_issue_cycle", launch)

    def test_dma_queue_and_wait_contracts(self):
        prefix = [*dma_setup(), transfer()]
        base = [*dma_setup(), *DMA_SECOND]
        cases = [([addi(5, 0, 31), config(), HALT], True, None),
            ([*prefix, *word_base(1, 0x90001000), addi(6, 0, 32), addi(2, 0, 32),
              addi(5, 0, 1), config(), wait(), addi(5, 0, 0), config(), HALT], True, None),
            *(([*dma_setup(), transfer(channel=c), wait(c), transfer("store", c), wait(c), HALT], True, None)
              for c in range(8)),
            ([*prefix, wait(1), HALT], False, "has no pending transfer to wait for"),
            ([*prefix, transfer(channel=1), wait(1), HALT], False, "still pending when its block ends"),
            ([*prefix, addi(5, 0, 1), config(), vls("vload"), wait(), HALT], False, IN_FLIGHT),
            ([*prefix, MARKER, wait(), HALT], False, "completion publication requires DMA.WAIT"),
            ([*prefix, HALT], False, "still pending when its block ends"),
            ([wait(), HALT], False, "has no pending transfer to wait for"),
            ([*prefix, wait(), wait(), HALT], False, "has no pending transfer to wait for"),
            ([*base, transfer(), transfer(channel=1, reg=7, dram=3), wait(), wait(1), HALT], True, None),
            ([*base, transfer(), transfer(channel=1, reg=7), wait(), wait(1), HALT], True, None),
            ([*base, transfer(), vls("vload", 4, 9), wait(), HALT], True, None),
            ([*base, transfer(), wait(), HALT], True, None),
            ([*base, transfer(), transfer(reg=7, dram=3), wait(), wait(), HALT], False, CHANNEL_REUSE),
            ([*base, transfer(), transfer(channel=1, dram=3), wait(), wait(1), HALT], False, IN_FLIGHT),
            ([*base, transfer(), transfer("store", 1, 7), wait(), wait(1), HALT], False, IN_FLIGHT),
            ([*base, transfer(), vls("vload"), wait(), HALT], False, IN_FLIGHT)]
        windows = [op for c in range(9) for op in word_base(10 + c, 64 * c)]
        launches = [transfer(channel=c, reg=10 + c) for c in range(8)]
        prefix = [*base, *windows, *launches]
        cases += [([*prefix, *(wait(c) for c in range(8)), HALT], True, None),
            ([*prefix, wait(0), transfer(channel=0, reg=18),
              *(wait(c) for c in range(1, 8)), wait(0), HALT], True, None),
            ([*prefix, wait(1), wait(0), transfer(channel=1, reg=18),
              *(wait(c) for c in range(2, 8)), wait(1), HALT], True, None),
            ([*prefix, wait(1), transfer(channel=1, reg=18),
              *(wait(c) for c in range(8) if c != 1), wait(1), HALT], False, EIGHT_SLOTS),
        ]
        for ops, accepted, diagnostic in cases:
            with self.subTest(ops=ops):
                self.check_consumers(ops, accepted, diagnostic)
        self.check([*dma_setup(), transfer(), delay(100), HALT], False, VERIFY, dma="wait",
                   diagnostic="still pending when its block ends")

    def test_dma_operand_domains(self):
        valid = [(0, 0, 32, 0), (392192, 0xffffffe0, 32, 31), (392192, 0x90000000, 4096, 31)]
        invalid = [(1, 0, 32, 0), (393216, 0, 32, 0), (65528, 0, 64, 0), (0, 1, 32, 0),
                   (0, 0xffffffe0, 64, 0), (0, 0, 0, 0), (0, 0, 31, 0), (0, 0, 33, 0), (0, 0, 4128, 0), (0, 0, 32, 32)]
        for operands in valid + invalid:
            with self.subTest(operands=operands):
                reason = "known 5-bit upper DRAM base" if operands[-1] == 32 else "unaligned, out of range, oversized"
                self.check_consumers([*dma_setup(*operands), transfer(), wait(), HALT], operands in valid,
                                     reason if operands in invalid else None)
        self.check_consumers([addi(6, 0, 0), transfer(), wait(), HALT], False)

    def test_final_boundaries_recheck_the_wait(self):
        selected = self.check(WAITED, True, VERIFY, dma="wait")
        unsafe = re.sub(r'(%\w+) = "atlas.dma_wait"\((%\w+)\)[^\n]*',
                        lambda m: m[1] + ' = "atlas.alu_imm"(' + m[2] + ') <{dst = 0 : i32, immediate = 0 : i32, '
                        'kind = "addi", src = 0 : i32}> : (!atlas.state) -> !atlas.state', selected.stdout)
        self.assertNotEqual(unsafe, selected.stdout)
        for tool, args in ((OPT, [VERIFY]), (EMIT, []), (OPT, ["--convert-atlas-to-llvm"]),
                           (OPT, ["--convert-atlas-to-llvm-calls"])):
            with self.subTest(args=args):
                self.assertNotEqual(run(tool, unsafe, *args).returncode, 0)
        staged = checked(self, selected.stdout, "--convert-atlas-to-llvm-calls")
        checked(self, staged, "--finalize-atlas-llvm-calls")
        lines, index = [], 0
        for line in staged.splitlines():
            if 'atlas.source_op = "atlas.dma_wait"' in line:
                continue
            if "atlas.word_index =" in line:
                line = re.sub(r"atlas.word_index = [0-9]+ : i32", f"atlas.word_index = {index} : i32", line)
                index += 1
            lines.append(line)
        self.assertEqual(len(lines), len(staged.splitlines()) - 1)
        finalized = run(OPT, "\n".join(lines), "--finalize-atlas-llvm-calls")
        self.assertNotEqual(finalized.returncode, 0)
        self.assertIn("DMA", finalized.stderr)

    def test_dma_policy_is_explicit(self):
        self.check(WAITED, False, CONSUMERS[0], diagnostic="dma=wait")
        for policy in ("latency", "none", "true"):
            with self.subTest(policy=policy):
                self.check(WAITED, False, VERIFY, dma=policy, diagnostic="unsupported DMA policy")
        plain = self.check([HALT], True, VERIFY)
        self.assertNotIn("dma =", plain.stdout)
        self.assertEqual(export(plain.stdout)["provider"]["dma_policy"], "none")


if __name__ == "__main__":
    unittest.main()
