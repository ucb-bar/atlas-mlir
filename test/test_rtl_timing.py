"""RTL provider selection, operand domains and final verification."""
import json
from pathlib import Path
import re
import tempfile
import unittest

from rtl_test_support import (
    CONSUMERS, DMA_LOAD, DMA_WAIT, EMIT, FACTS, HALT, MARKER, OPT, PAIRS, PROVIDER,
    RTLTestSupport, VERIFY, VLOAD, VLS_COPY, VLS_LOOP, addi, block, branch, cycles, delay,
    dma_setup, export, facts, facts_variant, jal, lui, nop, pairs_variant, program, run,
    scalar_load, scalar_store, selection, sha256, stream, unary, vls, without_delays,
    word_base,
)


@unittest.skipUnless(FACTS and OPT.is_file() and EMIT.is_file(),
                     "requires explicit RTL timing facts and built Atlas tools")
class SelectedRTLTimingTest(RTLTestSupport, unittest.TestCase):
    def test_both_consumers_and_final_emission(self):
        for consumer in CONSUMERS:
            with self.subTest(consumer=consumer):
                result = self.check(VLS_COPY.read_text(), True, consumer, VERIFY)
                self.assertIn(f'provider = "{PROVIDER}"', result.stdout)
                self.assertIn(f'op_timing_sha256 = "{sha256(FACTS)}"', result.stdout)
                self.assertIn('"atlas.delay"', result.stdout)
                self.assertEqual(run(EMIT, result.stdout).returncode, 0)
                unsafe = run(OPT, without_delays(result.stdout), VERIFY)
                self.assertNotEqual(unsafe.returncode, 0)
                self.assertIn("insufficient issue spacing", unsafe.stderr)

    def test_executable_boundaries_time_selected_streams_without_timing_metadata(self):
        # A handwritten stream carrying a selection but no timing state is timed by the provider at emission and
        # the LLVM handoff; --allow-untimed inspects it without a timing claim.
        unmarked = self.check(VLS_COPY.read_text(), True).stdout
        self.assertNotIn("atlas.timing_state", unmarked)
        for tool, args in ((EMIT, []), (OPT, ["--convert-atlas-to-llvm"]), (OPT, ["--convert-atlas-to-llvm-calls"])):
            with self.subTest(args=args):
                unsafe = run(tool, unmarked, *args)
                self.assertNotEqual(unsafe.returncode, 0)
                self.assertIn("insufficient issue spacing", unsafe.stderr)
        self.assertEqual(run(EMIT, unmarked, "--allow-untimed").returncode, 0)
        timed = self.check(VLS_COPY.read_text(), True, "--insert-atlas-delays").stdout
        stripped = re.sub(r', atlas\.timing_(state|provider) = "[^"]*"', "", timed)
        self.assertNotIn("atlas.timing_state", stripped)
        self.assertEqual(run(EMIT, stripped).stdout, run(EMIT, timed).stdout)
        self.assertEqual(run(OPT, stripped, "--convert-atlas-to-llvm").returncode, 0)

    def test_selection_is_explicit_and_checked(self):
        self.check([HALT], False, VERIFY, digest="0" * 64, diagnostic="mismatch")
        self.check([HALT], False, VERIFY, digest="abc", diagnostic="SHA-256 is required")
        # Without a selection the common verifier keeps the model; the RTL provider needs the facts.
        unselected = run(OPT, program([HALT]), VERIFY)
        self.assertEqual(unselected.returncode, 0, unselected.stderr)
        self.assertIn('atlas.timing_provider = "npu-model-rtl-match-v1"', unselected.stdout)
        explicit = run(OPT, program([HALT]), f"{VERIFY}=provider={PROVIDER}")
        self.assertNotEqual(explicit.returncode, 0)
        self.assertIn("requires selected evidence", explicit.stderr)
        model = run(OPT, program([HALT]), selection(), f"{VERIFY}=provider=npu-model-rtl-match-v1")
        self.assertNotEqual(model.returncode, 0)
        self.assertIn("selected RTL evidence excludes", model.stderr)
        good = self.check([HALT], True, VERIFY).stdout
        self.assertIn(f'atlas.timing_provider = "{PROVIDER}"', good)
        for source, diagnostic in (
                (re.sub(r'hardware_ir_sha256 = "[0-9a-f]{64}"', 'hardware_ir_sha256 = "' + "1" * 64 + '"', good),
                 "different hardware IR"),
                (good.replace(f', provider = "{PROVIDER}"}}', f', provider = "{PROVIDER[:-1]}9"}}'), "unsupported RTL evidence provider"),
                (good.replace(f', provider = "{PROVIDER}"}}', f', dma = "latency", provider = "{PROVIDER}"}}'),
                 "unsupported DMA policy")):
            self.assertNotEqual(source, good)
            checked = run(OPT, source, "--insert-atlas-delays")
            self.assertNotEqual(checked.returncode, 0)
            self.assertIn(diagnostic, checked.stderr)

    def test_missing_unresolved_and_disagreeing_blocks_fail_closed(self):
        load = lambda d: block(d, "vlsu.vload")
        mutations = (
            (lambda d: d["op_timing"].remove(load(d)), "facts block is missing"),
            (lambda d: load(d).update(events=None), "facts block is unresolved"),
            (lambda d: load(d)["events"]["mreg_write"].update(count=31), "does not sweep"),
            (lambda d: load(d)["events"]["mreg_write"].update(step=None, last_age=40), "does not sweep"),
            (lambda d: load(d)["events"].pop("vmem_read"), "event group vmem_read is missing"),
            (lambda d: load(d).update(first_free_age=None), "lacks first_free_age"),
            (lambda d: load(d)["assumptions"].pop("scratchpad_read_latency"), "lacks scratchpad_read_latency"))
        with tempfile.TemporaryDirectory(prefix="atlas-facts-") as directory:
            for mutate, diagnostic in mutations:
                with self.subTest(diagnostic=diagnostic):
                    path = facts_variant(directory, mutate)
                    for consumer in CONSUMERS:
                        result = self.check([addi(6, 0, 256), VLOAD, HALT], False, consumer,
                                            path=path, diagnostic=diagnostic)
                        self.assertIn(f"{PROVIDER}: vlsu.vload: ", result.stderr)
                    self.check([addi(6, 0, 256), VLOAD, delay(33), nop(), HALT], False, VERIFY,
                               path=path, diagnostic=diagnostic)
                    # Other operations still resolve from the same file.
                    self.check([addi(6, 0, 256), vls("vstore"), delay(33), nop(), HALT], True,
                               VERIFY, path=path)
            for mutate in (lambda d: d.update(schema="merlin.op_timing.v0"),
                           lambda d: d.pop("hw_ir"),
                           lambda d: d["op_timing"].append(load(d))):
                self.check([HALT], False, VERIFY, path=facts_variant(directory, mutate),
                           diagnostic="RTL timing facts")

    def test_admission_numbers_come_from_the_facts(self):
        self.assertEqual(block(facts(), "vlsu.vload")["first_free_age"], 35)

        def slower(d):
            for name in ("vlsu.vload", "vlsu.vstore"):
                block(d, name).update(first_free_age=40, next_issue_age=40)
        raw = block(facts(), "vlsu.vload")["events"]["mreg_write"]["last_age"] + 1
        with tempfile.TemporaryDirectory(prefix="atlas-facts-") as directory:
            path = facts_variant(directory, slower)
            # DELAY N occupies N+1 cycles, so the two VLS issue N+2 apart. A path is busy until first_free_age.
            for gap in (35, 39, 40):
                with self.subTest(gap=gap):
                    self.check([addi(6, 0, 256), *word_base(8, 768), VLOAD, delay(gap - 2),
                                vls("vload", 63, 8), delay(38), nop(), HALT], gap >= 40, VERIFY, path=path)
            # VLOAD and VSTORE hold only their own path: the copy is ordered by its RAW on m4.
            for consumer in CONSUMERS:
                with self.subTest(consumer=consumer):
                    result = self.check(VLS_COPY.read_text(), True, consumer, VERIFY, path=path)
                    vector = [i for i in export(result.stdout)["instructions"]
                              if i["mnemonic"] in ("vload", "vstore")]
                    self.assertEqual(vector[1]["logical_issue_cycle"] - vector[0]["logical_issue_cycle"], raw)
                    self.assertEqual(vector[0]["footprint"]["done_age"], 39)
                    self.assertEqual([(h["unit"], h["to"]) for h in vector[0]["footprint"]["holds"]
                                      if h["unit"].endswith("path")], [("VLOAD path", 39)])

    def test_unsupported_domains_reject_in_both_consumers(self):
        for ops in ([VLOAD, HALT], [addi(6, 0, 8), VLOAD, HALT], [("fence", ""), HALT],
                    [("csr", 'kind = "rrw", dst = 0 : i32, source = 1 : i32, address = 3089 : i32'), HALT]):
            for consumer in CONSUMERS:
                with self.subTest(ops=ops, consumer=consumer):
                    self.check(ops, False, consumer, diagnostic=PROVIDER)

    def test_exact_vls_admission_boundary(self):
        # A path is busy until first_free_age. VLOAD and VSTORE overlap; a shared VMEM bank orders them by its
        # read/write windows and a shared MREG by its release ages.
        document = facts()
        load, store = block(document, "vlsu.vload"), block(document, "vlsu.vstore")
        events = {"vload": load["events"]["vmem_read"], "vstore": store["events"]["vmem_write"]}
        latency = store["assumptions"]["scratchpad_read_latency"]
        release = {"vload": load["events"]["mreg_write"]["last_age"],
                   "vstore": store["events"]["mreg_read"]["last_age"] + latency}
        for first in ("vload", "vstore"):
            for second in ("vload", "vstore"):
                for other_mreg, other_words in ((4, 768), (63, 768), (4, 65792), (63, 65792)):
                    if first == second:
                        boundary = block(document, f"vlsu.{first}")["first_free_age"]
                    else:
                        bank = events[first]["last_age"] - events[second]["first_age"] + 1 if other_words == 768 else 1
                        boundary = max(bank, release[first] + 1 if other_mreg == 4 else 1)
                    bank_conflict = first != second and other_mreg == 63 and other_words == 768
                    gaps = sorted({1, boundary - 1, boundary}) if bank_conflict else ((boundary - 1, boundary) if boundary > 1 else (boundary,))
                    for gap in gaps:
                        with self.subTest(first=first, second=second, mreg=other_mreg, words=other_words, gap=gap):
                            prefix = [addi(6, 0, 256), *word_base(8, other_words), vls(first)]
                            between = [] if gap == 1 else [nop()] if gap == 2 else [delay(gap - 2)]
                            result = self.check([*prefix, *between, vls(second, other_mreg, 8), delay(33), nop(), HALT],
                                                gap >= boundary, VERIFY)
                            if gap < boundary:
                                self.assertRegex(result.stderr, "timing resource conflict|insufficient issue spacing")
                                if bank_conflict:
                                    self.assertIn("VMEM bank 0 busy", result.stderr)

    def check_operand_admission(self, setup, instruction, accepted, encoded_immediate=None):
        for consumer in CONSUMERS:
            with self.subTest(consumer=consumer):
                checked = self.check([*setup, instruction, HALT], accepted, consumer, VERIFY,
                                     diagnostic=None if accepted else "VLS effective tile")
                if accepted:
                    emitted = run(EMIT, checked.stdout)
                    self.assertEqual(emitted.returncode, 0, emitted.stderr)
                    if encoded_immediate is not None:
                        words = [int(w, 16) for w in emitted.stdout.split()]
                        vector = [w for w in words if w & 0x7f == 0x07]
                        self.assertEqual([w >> 20 for w in vector], [encoded_immediate])

    def test_each_vmem_bank_first_and_last_legal_tile(self):
        # Six banks of 8192 lines; a 32-line tile starts at line 0 or 8160.
        for bank in range(6):
            for row, kind in ((0, "vload"), (8160, "vstore")):
                with self.subTest(bank=bank, row=row):
                    line = bank * 8192 + row
                    self.check_operand_admission(word_base(6, line * 8), vls(kind, 0 if row == 0 else 63), True)
        for line in (49121, 49152):  # unaligned; past the last bank
            with self.subTest(rejected_line=line):
                self.check_operand_admission(word_base(6, line * 8), VLOAD, False)

    def test_signed_and_raw_immediate_extremes(self):
        # The RTL adds sext(imm12) * 32 to the word base and takes bits 18:3.
        for offset, words, encoded in ((-2048, 65536, 0x800), (2047, 32, 0x7ff), (-1, 32, 0xfff),
                                       (-2048, 458496, 0x800), (-1, 392992, 0xfff)):
            for kind in ("vload", "vstore"):
                with self.subTest(offset=offset, words=words, kind=kind):
                    self.check_operand_admission(word_base(6, words), vls(kind, offset=offset), True,
                                                 encoded_immediate=encoded)
        for offset in (2047, -2048):
            with self.subTest(rejected_offset=offset):
                self.check_operand_admission([addi(6, 0, 0)], vls("vload", offset=offset), False)
        for offset in (-2049, 2048, 4095):
            with self.subTest(unencoded_offset=offset):
                self.check([*word_base(6, 65536), vls("vload", offset=offset), HALT], False,
                           CONSUMERS[0], diagnostic="must be in [-2048, 2047]")

    def test_known_base_wrap_alias_and_hardwired_zero(self):
        # fffff000 + 2047 + 2047 + 258 wraps to 0x100 (line 32); +266 is unaligned.
        prefix = [lui(6, 0xfffff), addi(6, 6, 2047), addi(6, 6, 2047)]
        for increment, accepted in ((258, True), (266, False)):
            with self.subTest(wrap_increment=increment):
                self.check_operand_admission([*prefix, addi(6, 6, increment)], VLOAD, accepted)
        for setup, base, accepted in (([addi(6, 0, 256), addi(6, 6, 8)], 6, False),
                                      ([addi(6, 0, 256), addi(6, 6, 8), addi(6, 6, -8)], 6, True),
                                      ([addi(6, 0, 0), addi(8, 6, 8)], 6, True),
                                      ([addi(6, 0, 0), addi(8, 6, 8)], 8, False),
                                      ([addi(0, 0, 8)], 0, True)):
            with self.subTest(setup=setup, base=base):
                self.check_operand_admission(setup, vls("vload", base=base), accepted)

    def test_scalar_memory_operations_take_their_numbers_from_the_facts(self):
        load = block(facts(), "scalar_lsu.load")
        for consumer in CONSUMERS:
            with self.subTest(consumer=consumer):
                result = self.check([addi(6, 0, 256), scalar_load("lw", 5, 6, 4), addi(7, 5, 1), HALT],
                                    True, consumer, VERIFY)
                self.assertEqual(run(EMIT, result.stdout).returncode, 0)
                unsafe = run(OPT, without_delays(result.stdout), VERIFY)
                self.assertNotEqual(unsafe.returncode, 0)
                self.assertIn("x5", unsafe.stderr)
                footprint = next(i for i in export(result.stdout)["instructions"]
                                 if i["mnemonic"] == "lw")["footprint"]
                accesses = {(a["resource"], a["write"]): a for a in footprint["accesses"]}
                vmem = accesses[("vmem", False)]
                self.assertEqual((vmem["age"], vmem["count"], vmem["first"]),
                                 (load["events"]["vmem_read"]["first_age"], 1, (256 + 4) // 32))
                self.assertEqual(accesses[("xreg", True)]["age"], load["events"]["load_write"]["first_age"])
                holds = {h["unit"]: (h["from"], h["to"]) for h in footprint["holds"]}
                self.assertEqual(holds["scalar load path"], (0, load["first_free_age"] - 1))
                self.assertEqual(holds["scalar write port"], (3, 3))
                self.assertEqual(footprint["done_age"], 3)
        for setup, instruction, accepted in (
                ([addi(6, 0, 256), addi(2, 0, 7)], scalar_store("sw", 2, 6, 8), True),
                ([addi(6, 0, 256)], scalar_load("seld", 3, 6, 0), True),
                ([], scalar_load("lw", 5, 6, 0), False),                      # unknown base
                ([addi(6, 0, 2)], scalar_load("lw", 5, 6, 0), False),          # unaligned
                ([lui(6, 0x180)], scalar_load("lw", 5, 6, 0), False),          # past 1.5 MiB
                ([addi(6, 0, 256)], scalar_load("lb", 5, 6, 0), False),
                ([addi(6, 0, 256)], scalar_store("sh", 2, 6, 0), False)):
            for consumer in CONSUMERS:
                with self.subTest(instruction=instruction, consumer=consumer):
                    self.check([*setup, instruction, HALT], accepted, consumer, VERIFY,
                               diagnostic=None if accepted else PROVIDER)

    def test_terminal_and_marker_admission(self):
        prefix = [addi(6, 0, 256), VLOAD]
        for suffix, diagnostic in (([MARKER, HALT], "atlas.complete waits for prior fixed-latency work"),
                                   ([HALT], "halt precedes completion of in-flight fixed-latency work"),
                                   ([delay(100), HALT], "halt does not wait for a preceding DELAY")):
            self.check([*prefix, *suffix], False, VERIFY, diagnostic=diagnostic)
        self.check([addi(6, 0, 256), scalar_load("lw", 5, 6, 0), HALT], False, VERIFY,
                   diagnostic="halt precedes completion of in-flight fixed-latency work")
        self.check([*prefix, delay(33), nop(), HALT], True, VERIFY)
        for ops in ([addi(1, 0, 1)], [HALT, addi(1, 0, 1)]):
            self.check(ops, False, VERIFY)

    def test_loop_blocks_start_and_end_idle(self):
        for consumer in CONSUMERS:
            with self.subTest(consumer=consumer):
                result = self.check(VLS_LOOP, True, consumer, VERIFY)
                ops = stream(result.stdout)
                at = next(i for i, (op, _, _) in enumerate(ops) if op == "branch")
                offset = int(re.search(r"offset_bytes = (-?\d+)", ops[at][1]).group(1))
                self.assertEqual(ops[at + offset // 2][0], "vload")
                # The block drains before the branch, behind a NOP.
                self.assertEqual([op for op, _, _ in ops[at - 2:at]], ["delay", "alu_imm"])
                self.assertEqual(ops[at - 1][2], "a redirect does not wait for a delay")
                self.assertEqual(run(EMIT, result.stdout).returncode, 0)
                exported = export(result.stdout); blocks = exported["blocks"]
                self.assertEqual([(b["first_word"], b["successors"], b["exit"]) for b in blocks],
                                 [(0, [1], "fall_through"), (3, [1, 2], "branch_with_delay_slot"), (at + 2, [], "halt")])
                self.assertEqual(blocks[1]["successor_issue_offset"], exported["instructions"][at]["logical_issue_cycle"] + 2)
                self.assertEqual([i["block"] for i in exported["instructions"]], [0] * 3 + [1] * (at - 1) + [2])
                unsafe = run(OPT, without_delays(result.stdout), VERIFY)
                self.assertNotEqual(unsafe.returncode, 0)

    def test_block_exit_drain_boundary(self):
        # vstore issues at 35 with done age 34, so successors may start at 70:
        # the branch at 68, after addi (36), DELAY 29 (37..66) and a NOP (67).
        def loop(tail):
            body = [addi(6, 0, 256), addi(13, 0, 0), addi(14, 0, 3), VLOAD, delay(33),
                    vls("vstore", 4, 6, 32), addi(13, 13, 1), *tail]
            return [*body, branch("blt", 13, 14, 3 - len(body)), nop(), HALT]
        self.check(loop([delay(29), nop()]), True, VERIFY)
        self.check(loop([delay(28), nop()]), False, VERIFY,
                   diagnostic="redirect reaches a successor before fixed-latency work drains")
        self.check(loop([delay(30)]), False, VERIFY, diagnostic="redirect does not wait for a preceding DELAY")
        # A fall-through exit drains too; ECALL after the DELAY needs a NOP.
        fall = [addi(6, 0, 256), addi(13, 0, 1), branch("beq", 13, 0, 4), nop(), VLOAD]
        self.check([*fall, delay(33), nop(), HALT], True, VERIFY)
        self.check([*fall, delay(32), nop(), HALT], False, VERIFY,
                   diagnostic="fixed-latency work remains live across a block boundary")
        # The taken branch reaches the ECALL directly.
        to_halt = [addi(6, 0, 256), addi(13, 0, 1), branch("beq", 13, 0, 3), nop(), VLOAD, HALT]
        for consumer in CONSUMERS:
            with self.subTest(consumer=consumer):
                result = self.check(to_halt, True, consumer, VERIFY)
                ops = stream(result.stdout)
                self.assertEqual([op for op, _, _ in ops[-3:]], ["delay", "alu_imm", "trap"])
                self.assertEqual(ops[-2][2], "a halt does not wait for a delay")

    def test_blocks_are_checked_under_each_concrete_entry_context(self):
        # beq skips to the else arm; both arms reach the join's vload. x5 = 1 decides the
        # branch; x7 is unknown at entry, so both arms reach the join with their exact bases.
        def diamond(then_base, else_base, condition=5):
            return [addi(5, 0, 1), branch("beq", condition, 0, 5), nop(), addi(6, 0, then_base), jal(3), nop(),
                    addi(6, 0, else_base), VLOAD, HALT]
        misaligned = "VLS effective tile is unaligned, out of range, or crosses a bank"
        for consumer in CONSUMERS:
            with self.subTest(consumer=consumer):
                self.check(diamond(256, 256), True, consumer, VERIFY)
                self.check(diamond(256, 264), True, consumer, VERIFY)
                self.check(diamond(256, 1280, 7), True, consumer, VERIFY)
                result = self.check(diamond(256, 264, 7), False, consumer, diagnostic=misaligned)
                self.assertIn("note: in the block entry context with x5=0x1 x6=0x108", result.stderr)

        # An induction variable takes one exact value per iteration; the block keeps its order.
        def stride(trips, step=None):
            body = [addi(6, 0, 256), *word_base(15, step or 256), addi(13, 0, 0), addi(14, 0, trips), VLOAD,
                    ("alu_reg", 'kind = "add", dst = 6 : i32, lhs = 6 : i32, rhs = 15 : i32'), addi(13, 13, 1)]
            return [*body, branch("blt", 13, 14, 5 - len(body)), nop(), HALT]
        for consumer in CONSUMERS:
            with self.subTest(consumer=consumer):
                timed = self.check(stride(3), True, consumer, VERIFY).stdout
                self.assertEqual([op for op, _, _ in stream(timed)[5:8]], ["vload", "alu_reg", "alu_imm"])
                exported = export(timed)
                contexts = exported["blocks"][1]["entry_contexts"]
                self.assertEqual([c["entries"][0]["x6"] for c in contexts], [256, 512, 768])
                self.assertEqual([c["footprints"][0]["accesses"][1]["first"] for c in contexts], [32, 64, 96])
                self.assertIsNone(exported["instructions"][5]["footprint"])
                # A bank-sized stride leaves VMEM in the seventh iteration.
                self.check(stride(6, 65536), True, consumer, VERIFY)
                result = self.check(stride(7, 65536), False, consumer, diagnostic=misaligned)
                self.assertIn("x6=0x60100", result.stderr)
        # Contexts differing in a VMEM bank time under the worst one: the vstore shares the vload's
        # bank only on the else path, which needs the load-to-store bank gap.
        join = [addi(6, 0, 256), branch("beq", 7, 0, 6), nop(), *word_base(8, 65536 + 256), jal(3), nop(),
                addi(8, 0, 1280), VLOAD, vls("vstore", 5, 8), HALT]
        for consumer in CONSUMERS:
            with self.subTest(consumer=consumer):
                ops = stream(self.check(join, True, consumer, VERIFY).stdout)
                at = [op for op, _, _ in ops].index("vload")
                self.assertEqual(ops[at + 2][0], "vstore")
                self.assertEqual(cycles(ops[at + 1][1]) + 2, 30)
        # Past the context bound values merge over all paths: an invariant tile still times, a moving one does not.
        unbounded = [addi(6, 0, 256), addi(13, 0, 0), VLOAD, vls("vstore", 4, 6, 32), addi(13, 13, 1),
                     branch("blt", 13, 14, -3), nop(), HALT]
        self.check(unbounded, True, CONSUMERS[0], VERIFY)
        moving = [*unbounded[:4], addi(6, 6, 256), *unbounded[4:]]
        moving[6] = branch("blt", 13, 14, -4)
        result = self.check(moving, False, CONSUMERS[0], diagnostic="unknown VLS base cannot establish bounded address domain")
        self.assertIn("note: concrete register contexts exceed 4096 block entries", result.stderr)

    def test_control_flow_shape_is_checked(self):
        for ops, diagnostic in (
                ([addi(13, 0, 0), branch("beq", 0, 0, 2), VLOAD, HALT], "admits only ADDI or LUI"),
                ([branch("beq", 0, 0, 2), scalar_load("lw", 5, 0, 0), HALT], "admits only ADDI or LUI"),
                ([addi(13, 0, 0), branch("beq", 0, 0, 2), nop()], "targets the stream end"),
                ([jal(3), nop(), addi(1, 0, 1), HALT], "is unreachable"),
                ([addi(13, 0, 0), branch("blt", 13, 0, 3), nop(), HALT, nop()], "requires a terminal"),
                ([("jump", 'kind = "jal", dst = 1 : i32, base = 0 : i32, offset = 4 : i32'), nop(), HALT],
                 "link")):
            for consumer in (*CONSUMERS, VERIFY):
                with self.subTest(ops=ops, consumer=consumer):
                    self.check(ops, False, consumer, diagnostic=diagnostic)

    def test_dma_must_be_waited_within_its_block(self):
        counter = [addi(13, 0, 0), addi(14, 0, 2)]
        across = [*dma_setup(), *counter, DMA_LOAD, addi(13, 13, 1), branch("blt", 13, 14, -2), nop(),
                  DMA_WAIT, HALT]
        inside = [*dma_setup(), *counter, DMA_LOAD, DMA_WAIT, addi(13, 13, 1),
                  branch("blt", 13, 14, -3), nop(), HALT]
        for consumer in CONSUMERS:
            with self.subTest(consumer=consumer):
                self.check(across, False, consumer, dma="wait",
                           diagnostic="still pending when its block ends; selected RTL timing requires atlas.dma_wait on channel 0")
                self.check(inside, True, consumer, VERIFY, dma="wait")
        self.check(across, False, VERIFY, dma="wait",
                   diagnostic="still pending when its block ends")

    def test_program_size_limits(self):
        source = program([nop()] * 32768 + [HALT])
        for consumer in (*CONSUMERS, VERIFY):
            with self.subTest(consumer=consumer):
                self.check(source, False, consumer, diagnostic="32768-word instruction memory")
        # 244 full delays plus the tail place ECALL at cycle 1,000,000 or 1,000,001.
        for excess in (0, 1):
            source = self.check([*[delay(4095)] * 244, delay(574 + excess), nop(), HALT], True).stdout
            for tool, args in ((OPT, [VERIFY]), (EMIT, []), (OPT, ["--convert-atlas-to-llvm"]),
                               (OPT, ["--convert-atlas-to-llvm-calls"])):
                with self.subTest(excess=excess, tool=tool.name, args=args):
                    result = run(tool, source, *args)
                    self.assertEqual(result.returncode == 0, excess == 0, result.stderr)
                    if excess:
                        self.assertIn("exceeds the bounded RTL timing cycle limit", result.stderr)


def first_record(change):
    return lambda document: document["op_pairs"][0].update(change)


@unittest.skipUnless(FACTS and PAIRS and OPT.is_file() and EMIT.is_file(),
                     "requires RTL timing facts, pair facts derived from them and built tools")
class PairFactsSelectionTest(unittest.TestCase):
    SOURCE = program([unary("exp", 8, 16), HALT])

    def test_selection_is_recorded(self):
        result = run(OPT, self.SOURCE, selection(pairs=PAIRS))
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn(f'op_pairs = "{Path(PAIRS).resolve()}"', result.stdout)
        self.assertIn(f'op_pairs_sha256 = "{sha256(PAIRS)}"', result.stdout)
        plain = run(OPT, self.SOURCE, selection())
        self.assertNotIn("op_pairs", plain.stdout)

    def test_selection_reaches_the_resolved_export(self):
        timed = run(OPT, self.SOURCE, selection(pairs=PAIRS), "--insert-atlas-delays", VERIFY)
        self.assertEqual(timed.returncode, 0, timed.stderr)
        exported = run(EMIT, timed.stdout, "--rtl-timing-json")
        self.assertEqual(exported.returncode, 0, exported.stderr)
        document = json.loads(exported.stdout)
        self.assertEqual(document["evidence"]["op_pairs_sha256"], sha256(PAIRS))
        self.assertEqual(document["applicability"]["pair_facts"]["sha256"], sha256(PAIRS))

    def test_invalid_selections_are_rejected(self):
        digest = f"op-timing={Path(FACTS).resolve()} op-timing-sha256={sha256(FACTS)}"
        cases = {
            "hash mismatch": f"--select-atlas-rtl-evidence={digest} op-pairs={PAIRS} op-pairs-sha256={'0' * 64}",
            "selected together": f"--select-atlas-rtl-evidence={digest} op-pairs={PAIRS}",
        }
        with tempfile.TemporaryDirectory() as directory:
            for message, mutate in {
                "different hardware IR": lambda d: d["hw_ir"].update(sha256="0" * 64),
                "different op_timing facts": lambda d: d["op_timing"].update(sha256="0" * 64),
                "not checked against RTL assertions": first_record({"assertions": {"checked": 0, "excluded": 0}}),
                "names no resolved facts block": first_record({"second": "vpu.unknown"}),
                "malformed stable_gap": first_record({"stable_gap": 0}),
                "expected schema merlin.op_pairs.v1": lambda d: d.update(schema="rtl_extract.op_pairs.v1"),
                "malformed capacity entry": lambda d: d["capacity"].append(dict(d["capacity"][0])),
            }.items():
                variant = Path(directory) / str(len(cases))
                variant.mkdir()
                cases[message] = selection(pairs=pairs_variant(variant, mutate))
            for message, option in cases.items():
                with self.subTest(message):
                    result = run(OPT, self.SOURCE, option)
                    self.assertNotEqual(result.returncode, 0)
                    self.assertIn(message, result.stderr)

    def test_capacity_windows_must_match_the_engine(self):
        """A VPU engine is live until next_issue_age and an MXU through its last event; any other window, or none, fails."""
        def window(engine, value):
            def mutate(document):
                entries = [c for c in document["capacity"] if c["engine"] == engine]
                if not entries:
                    entries = [{"engine": engine, "live": 1, "witness": {"source": "-", "gap": 1}}]
                    document["capacity"].append(entries[0])
                if value is None: entries[0].pop("window", None)
                else: entries[0]["window"] = value
            return mutate
        with tempfile.TemporaryDirectory() as directory:
            for n, (engine, value) in enumerate((("vpu", "last_event_age"), ("vpu", None), ("mxu0", "next_issue_age"),
                                                  ("mxu1", "busy"), ("core", "last_event_age"))):
                variant = Path(directory) / str(n)
                variant.mkdir()
                with self.subTest(engine=engine, window=value):
                    result = run(OPT, self.SOURCE, selection(pairs=pairs_variant(variant, window(engine, value))))
                    self.assertNotEqual(result.returncode, 0)
                    self.assertIn("malformed capacity entry", result.stderr)

    def test_recorded_selection_is_rechecked(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "op-pairs.json"
            path.write_bytes(Path(PAIRS).read_bytes())
            selected = run(OPT, self.SOURCE, selection(pairs=path))
            self.assertEqual(selected.returncode, 0, selected.stderr)
            path.write_text(path.read_text() + " ")
            result = run(OPT, selected.stdout, "--insert-atlas-delays")
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("selected pair facts hash mismatch", result.stderr)
            incomplete = selected.stdout.replace("op_pairs_sha256", "op_pairs_digest")
            result = run(OPT, incomplete, "--insert-atlas-delays")
            self.assertIn("incomplete RTL pair facts selection", result.stderr)


if __name__ == "__main__":
    unittest.main()
