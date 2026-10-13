"""RTL provider selection, operand domains and final verification."""
from pathlib import Path
import re
import tempfile
import unittest

from rtl_test_support import (
    CONSUMERS, DMA_LOAD, DMA_WAIT, EMIT, FACTS, HALT, MARKER, OPT, PAIRS, PROVIDER, RTLTestSupport,
    VERIFY, VLOAD, VLS_COPY, VLS_LOOP, addi, block, branch, cycles, delay, dma_setup, export, facts,
    facts_variant, gapped, jal, lui, nop, pairs_variant, program, run, scalar_load, scalar_store,
    selection, sha256, stream, unary, vls, without_delays, word_base)


@unittest.skipUnless(FACTS and OPT.is_file() and EMIT.is_file(), "requires RTL timing facts and Atlas tools")
class SelectedRTLTimingTest(RTLTestSupport, unittest.TestCase):
    def test_consumers_and_executable_boundaries(self):
        unmarked = self.check(VLS_COPY.read_text(), True).stdout
        self.assertNotIn("atlas.timing_state", unmarked)
        for tool, args in ((EMIT, []), (OPT, ["--convert-atlas-to-llvm"]), (OPT, ["--convert-atlas-to-llvm-calls"])):
            with self.subTest(args=args):
                unsafe = run(tool, unmarked, *args)
                self.assertNotEqual(unsafe.returncode, 0)
                self.assertIn("insufficient issue spacing", unsafe.stderr)
        self.assertEqual(run(EMIT, unmarked, "--allow-untimed").returncode, 0)
        for consumer in CONSUMERS:
            with self.subTest(consumer=consumer):
                timed = self.check(VLS_COPY.read_text(), True, consumer, VERIFY).stdout
                stripped = re.sub(r', atlas\.timing_(state|provider) = "[^"]*"', "", timed)
                self.assertNotIn("atlas.timing_state", stripped)
                emitted = run(EMIT, timed)
                self.assertEqual(emitted.returncode, 0, emitted.stderr)
                self.assertEqual(run(EMIT, stripped).stdout, emitted.stdout)
                self.assertEqual(run(OPT, stripped, "--convert-atlas-to-llvm").returncode, 0)
                unsafe = run(OPT, without_delays(timed), VERIFY)
                self.assertNotEqual(unsafe.returncode, 0)
                self.assertIn("insufficient issue spacing", unsafe.stderr)

    def test_selection_is_explicit_and_checked(self):
        self.check([HALT], False, VERIFY, digest="0" * 64, diagnostic="mismatch")
        self.check([HALT], False, VERIFY, digest="abc", diagnostic="SHA-256 is required")
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
        for source, diagnostic in ((good.replace(facts()["hw_ir"]["sha256"], "1" * 64), "different hardware IR"),
                (good.replace(f', provider = "{PROVIDER}"}}', f', provider = "{PROVIDER[:-1]}9"}}'), "unsupported RTL evidence provider"),
                (good.replace(f', provider = "{PROVIDER}"}}', f', dma = "latency", provider = "{PROVIDER}"}}'),
                 "unsupported DMA policy")):
            self.assertNotEqual(source, good)
            checked = run(OPT, source, "--insert-atlas-delays")
            self.assertNotEqual(checked.returncode, 0)
            self.assertIn(diagnostic, checked.stderr)

    def test_missing_unresolved_and_disagreeing_blocks_fail_closed(self):
        load = lambda d: block(d, "vlsu.vload")
        mutations = ((lambda d: d["op_timing"].remove(load(d)), "facts block is missing"),
            (lambda d: load(d).update(events=None), "facts block is unresolved"),
            (lambda d: load(d)["events"]["mreg_write"].update(count=31), "does not sweep"),
            (lambda d: load(d)["events"]["mreg_write"].update(step=None, last_age=40), "does not sweep"),
            (lambda d: load(d)["events"].pop("vmem_read"), "event group vmem_read is missing"),
            (lambda d: load(d).update(first_free_age=None), "lacks first_free_age"),
            (lambda d: load(d)["assumptions"].pop("scratchpad_read_latency"), "lacks scratchpad_read_latency"))
        with tempfile.TemporaryDirectory() as directory:
            for mutate, diagnostic in mutations:
                with self.subTest(diagnostic=diagnostic):
                    path = facts_variant(directory, mutate)
                    for consumer in (*CONSUMERS, VERIFY):
                        ops = [addi(6, 0, 256), VLOAD, *([delay(33), nop()] if consumer == VERIFY else []), HALT]
                        result = self.check(ops, False, consumer, path=path, diagnostic=diagnostic)
                        self.assertIn(f"{PROVIDER}: vlsu.vload: ", result.stderr)
            for mutate in (lambda d: d.update(schema="merlin.op_timing.v0"), lambda d: d.pop("hw_ir"),
                           lambda d: d["op_timing"].append(load(d))):
                self.check([HALT], False, VERIFY, path=facts_variant(directory, mutate), diagnostic="RTL timing facts")

    def test_exact_memory_admission_boundaries(self):
        for first, second, mreg, words, boundary, conflict in (("vload", "vload", 63, 65792, 35, "VLOAD path 0 busy"),
                ("vstore", "vstore", 63, 65792, 35, "VSTORE path 0 busy"),
                *((a, b, r, w, gap, conflict) for a, b, release, bank in (
                    ("vload", "vstore", 35, 30), ("vstore", "vload", 34, 34))
                  for r, w, gap, conflict in ((63, 768, bank, "VMEM bank 0 busy"),
                    (4, 65792, release, "insufficient issue spacing")))):
            for gap in ((boundary - 1, boundary) if boundary > 1 else (1,)):
                with self.subTest(first=first, second=second, mreg=mreg, words=words, gap=gap):
                    self.check(gapped([addi(6, 0, 256), *word_base(8, words)], vls(first),
                                      vls(second, mreg, 8), gap), gap == boundary, VERIFY,
                               diagnostic=None if gap == boundary else conflict)
        with tempfile.TemporaryDirectory() as directory:
            def slower(d):
                for name in ("vlsu.vload", "vlsu.vstore"):
                    block(d, name).update(first_free_age=40, next_issue_age=40)
            path = facts_variant(directory, slower)
            for gap in (39, 40):
                with self.subTest(changed_facts_gap=gap):
                    self.check(gapped([addi(6, 0, 256), *word_base(8, 768)], VLOAD,
                                      vls("vload", 63, 8), gap), gap == 40, VERIFY, path=path,
                               diagnostic=None if gap == 40 else "VLOAD path 0 busy")
        bound = block(facts(), "scalar_lsu.load")["events"]["load_write"]["first_age"] + 1
        for gap in (bound - 1, bound):
            with self.subTest(raw_gap=gap):
                result = self.check(gapped([addi(6, 0, 256)], scalar_load("lw", 5, 6, 4), addi(7, 5, 1), gap),
                                    gap == bound, VERIFY, diagnostic=None if gap == bound else "x5")
                if gap == bound:
                    load = next(i for i in export(result.stdout)["instructions"] if i["mnemonic"] == "lw")
                    self.assertEqual(next(a["first"] for a in load["footprint"]["accesses"] if a["resource"] == "vmem"), 8)

    def test_memory_operand_domains_and_encoding(self):
        rows = [(word_base(6, (bank * 8192 + row) * 8), vls(kind, 0 if row == 0 else 63), True, None)
                for bank in range(6) for row, kind in ((0, "vload"), (8160, "vstore"))]
        rows += [(word_base(6, line * 8), VLOAD, False, None) for line in (49121, 49152)]
        rows += [(word_base(6, words), vls(kind, offset=offset), True, encoded)
                 for offset, words, encoded in ((-2048, 65536, 0x800), (2047, 32, 0x7ff), (-1, 32, 0xfff))
                 for kind in ("vload", "vstore")]
        rows += [([addi(6, 0, 0)], vls("vload", offset=offset), False, None) for offset in (2047, -2048)]
        prefix = [lui(6, 0xfffff), addi(6, 6, 2047), addi(6, 6, 2047)]
        rows += [([*prefix, addi(6, 6, n)], VLOAD, ok, None) for n, ok in ((258, True), (266, False))]
        rows += [([addi(0, 0, 8)], vls("vload", base=0), True, None),
                 ([], VLOAD, False, None), ([addi(6, 0, 8)], VLOAD, False, None)]
        rows += [(setup, instruction, accepted, None) for setup, instruction, accepted in (
                ([addi(6, 0, 256), addi(2, 0, 7)], scalar_store("sw", 2, 6, 8), True),
                ([addi(6, 0, 256)], scalar_load("seld", 3, 6, 0), True), ([], scalar_load("lw", 5, 6, 0), False),
                ([addi(6, 0, 2)], scalar_load("lw", 5, 6, 0), False),
                ([lui(6, 0x180)], scalar_load("lw", 5, 6, 0), False),
                ([addi(6, 0, 256)], scalar_load("lb", 5, 6, 0), False),
                ([addi(6, 0, 256)], scalar_store("sh", 2, 6, 0), False), ([], ("fence", ""), False),
                ([], ("csr", 'kind = "rrw", dst = 0 : i32, source = 1 : i32, address = 3089 : i32'), False))]
        rows += [([], ("alu_reg", f'kind = "{kind}", dst = 6 : i32, lhs = 0 : i32, rhs = 0 : i32'), True, None)
                 for kind in ("xor", "slt", "sltu")]
        rows += [([], ("alu_imm", f'kind = "{kind}", dst = 6 : i32, src = 0 : i32, immediate = 0 : i32'), True, None)
                 for kind in ("xori", "sltiu")]
        rows += [([], scalar_load("seli", 3, 0, 0), True, None)]
        for setup, instruction, accepted, encoded in rows:
            for consumer in CONSUMERS:
                with self.subTest(setup=setup, instruction=instruction, consumer=consumer):
                    result = self.check([*setup, instruction, HALT], accepted, consumer, VERIFY,
                                        diagnostic=None if accepted else (
                                            "VLS effective tile" if instruction[0] in ("vload", "vstore") and setup else PROVIDER))
                    if encoded is not None:
                        emitted = run(EMIT, result.stdout)
                        self.assertEqual(emitted.returncode, 0, emitted.stderr)
                        self.assertEqual([int(w, 16) >> 20 for w in emitted.stdout.split()
                                          if int(w, 16) & 0x7f == 0x07], [encoded])
        for offset in (-2049, 2048, 4095):
            with self.subTest(unencoded_offset=offset):
                self.check([*word_base(6, 65536), vls("vload", offset=offset), HALT], False,
                           CONSUMERS[0], diagnostic="must be in [-2048, 2047]")

    def test_terminal_and_marker_admission(self):
        prefix = [addi(6, 0, 256), VLOAD]
        for suffix, diagnostic in (([delay(31), nop(), MARKER, HALT], "atlas.complete waits for prior fixed-latency work"),
                                   ([delay(31), nop(), HALT], "halt precedes completion of in-flight fixed-latency work"),
                                   ([delay(100), HALT], "halt does not wait for a preceding DELAY")):
            self.check([*prefix, *suffix], False, VERIFY, diagnostic=diagnostic)
        self.check([*prefix, delay(32), nop(), HALT], True, VERIFY)
        self.check([*prefix, delay(32), nop(), MARKER, HALT], True, VERIFY)

    def test_block_exit_drain_boundary(self):
        def loop(tail):
            body = [addi(6, 0, 256), addi(13, 0, 0), addi(14, 0, 3), VLOAD, delay(33),
                    vls("vstore", 4, 6, 32), addi(13, 13, 1), *tail]
            return [*body, branch("blt", 13, 14, 3 - len(body)), nop(), HALT]
        self.check(loop([delay(29), nop()]), True, VERIFY)
        self.check(loop([delay(28), nop()]), False, VERIFY,
                   diagnostic="redirect reaches a successor before fixed-latency work drains")
        self.check(loop([delay(30)]), False, VERIFY, diagnostic="redirect does not wait for a preceding DELAY")
        fall = [addi(6, 0, 256), addi(13, 0, 1), branch("beq", 13, 0, 4), nop(), VLOAD]
        self.check([*fall, delay(33), nop(), HALT], True, VERIFY)
        self.check([*fall, delay(32), nop(), HALT], False, VERIFY,
                   diagnostic="fixed-latency work remains live across a block boundary")
        to_halt = [addi(6, 0, 256), addi(13, 0, 1), branch("beq", 13, 0, 3), nop(), VLOAD, HALT]
        for consumer in CONSUMERS:
            with self.subTest(consumer=consumer):
                ops = stream(self.check(to_halt, True, consumer, VERIFY).stdout)
                self.assertEqual([op for op, _, _ in ops[-3:]], ["delay", "alu_imm", "trap"])
                self.assertEqual(ops[-2][2], "a halt does not wait for a delay")
                timed_loop = stream(self.check(VLS_LOOP, True, consumer, VERIFY).stdout)
                at = next(i for i, (op, _, _) in enumerate(timed_loop) if op == "branch")
                offset = int(re.search(r"offset_bytes = (-?\d+)", timed_loop[at][1]).group(1))
                self.assertEqual(timed_loop[at + offset // 2][0], "vload")

    def test_blocks_are_checked_under_each_concrete_entry_context(self):
        def diamond(then_base, else_base, condition=5):
            return [addi(5, 0, 1), branch("beq", condition, 0, 5), nop(), addi(6, 0, then_base), jal(3), nop(),
                    addi(6, 0, else_base), VLOAD, HALT]
        misaligned = "VLS effective tile is unaligned, out of range, or crosses a bank"
        def stride(trips, step=None):
            body = [addi(6, 0, 256), *word_base(15, step or 256), addi(13, 0, 0), addi(14, 0, trips), VLOAD,
                    ("alu_reg", 'kind = "add", dst = 6 : i32, lhs = 6 : i32, rhs = 15 : i32'), addi(13, 13, 1)]
            return [*body, branch("blt", 13, 14, 5 - len(body)), nop(), HALT]
        join = [addi(6, 0, 256), branch("beq", 7, 0, 6), nop(), *word_base(8, 65536 + 256), jal(3), nop(),
                addi(8, 0, 1280), VLOAD, vls("vstore", 5, 8), HALT]
        for consumer in CONSUMERS:
            with self.subTest(consumer=consumer):
                self.check(diamond(256, 264), True, consumer, VERIFY)
                self.check(diamond(256, 1280, 7), True, consumer, VERIFY)
                result = self.check(diamond(256, 264, 7), False, consumer, diagnostic=misaligned)
                self.assertIn("note: in the block entry context with x5=0x1 x6=0x108", result.stderr)
                timed = self.check(stride(3), True, consumer, VERIFY).stdout
                exported = export(timed)
                contexts = exported["blocks"][1]["entry_contexts"]
                self.assertEqual([c["entries"][0]["x6"] for c in contexts], [256, 512, 768])
                self.assertEqual([c["footprints"][0]["accesses"][1]["first"] for c in contexts], [32, 64, 96])
                self.assertIsNone(exported["instructions"][5]["footprint"])
                self.check(stride(6, 65536), True, consumer, VERIFY)
                result = self.check(stride(7, 65536), False, consumer, diagnostic=misaligned)
                self.assertIn("x6=0x60100", result.stderr)
                ops = stream(self.check(join, True, consumer, VERIFY).stdout)
                at = [op for op, _, _ in ops].index("vload")
                self.assertEqual(ops[at + 2][0], "vstore")
                self.assertEqual(cycles(ops[at + 1][1]) + 2, 30)
        unbounded = [addi(6, 0, 256), addi(13, 0, 0), VLOAD, vls("vstore", 4, 6, 32), addi(13, 13, 1),
                     branch("blt", 13, 14, -3), nop(), HALT]
        self.check(unbounded, True, CONSUMERS[0], VERIFY)
        moving = [*unbounded[:4], addi(6, 6, 256), *unbounded[4:]]
        moving[6] = branch("blt", 13, 14, -4)
        result = self.check(moving, False, CONSUMERS[0], diagnostic="unknown VLS base cannot establish bounded address domain")
        self.assertIn("note: concrete register contexts exceed 4096 block entries", result.stderr)

    def test_control_flow_shape_is_checked(self):
        for ops, diagnostic in (([addi(13, 0, 0), branch("beq", 0, 0, 2), VLOAD, HALT], "admits only ADDI or LUI"),
                ([addi(13, 0, 0), branch("beq", 0, 0, 2), nop()], "targets the stream end"),
                ([jal(3), nop(), addi(1, 0, 1), HALT], "is unreachable"),
                ([addi(13, 0, 0), branch("blt", 13, 0, 3), nop(), HALT, nop()], "requires a terminal"),
                ([("jump", 'kind = "jal", dst = 1 : i32, base = 0 : i32, offset = 4 : i32'), nop(), HALT], "link")):
            for consumer in (*CONSUMERS, VERIFY):
                with self.subTest(ops=ops, consumer=consumer):
                    self.check(ops, False, consumer, diagnostic=diagnostic)

    def test_dma_must_be_waited_within_its_block(self):
        counter = [addi(13, 0, 0), addi(14, 0, 2)]
        across = [*dma_setup(), *counter, DMA_LOAD, addi(13, 13, 1), branch("blt", 13, 14, -2), nop(), DMA_WAIT, HALT]
        inside = [*dma_setup(), *counter, DMA_LOAD, DMA_WAIT, addi(13, 13, 1), branch("blt", 13, 14, -3), nop(), HALT]
        for consumer in CONSUMERS:
            with self.subTest(consumer=consumer):
                self.check(across, False, consumer, dma="wait",
                           diagnostic="still pending when its block ends; selected RTL timing requires atlas.dma_wait on channel 0")
                self.check(inside, True, consumer, VERIFY, dma="wait")
        self.check(across, False, VERIFY, dma="wait", diagnostic="still pending when its block ends")

    def test_program_size_limits(self):
        for words in (32768, 32769):
            for consumer in ((VERIFY,) if words == 32768 else (*CONSUMERS, VERIFY)):
                with self.subTest(words=words, consumer=consumer):
                    self.check(program([nop()] * (words - 1) + [HALT]), words == 32768, consumer,
                               diagnostic=None if words == 32768 else "32768-word instruction memory")
        for excess in (0, 1):
            source = self.check([*[delay(4095)] * 244, delay(574 + excess), nop(), HALT], True).stdout
            for tool, args in ((OPT, [VERIFY]), (EMIT, []), (OPT, ["--convert-atlas-to-llvm"]),
                               (OPT, ["--convert-atlas-to-llvm-calls"])):
                with self.subTest(excess=excess, tool=tool.name, args=args):
                    result = run(tool, source, *args)
                    self.assertEqual(result.returncode == 0, excess == 0, result.stderr)
                    if excess:
                        self.assertIn("exceeds the bounded RTL timing cycle limit", result.stderr)


@unittest.skipUnless(FACTS and PAIRS and OPT.is_file() and EMIT.is_file(), "requires RTL facts, pair facts and tools")
class PairFactsSelectionTest(unittest.TestCase):
    SOURCE = program([unary("exp", 8, 16), HALT])

    def test_selection_is_recorded_exported_and_rechecked(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "op-pairs.json"
            path.write_bytes(Path(PAIRS).read_bytes())
            result = run(OPT, self.SOURCE, selection(pairs=path), CONSUMERS[0], VERIFY)
            self.assertEqual(result.returncode, 0, result.stderr)
            document = export(result.stdout)
            self.assertEqual(document["evidence"]["op_pairs_sha256"], sha256(PAIRS))
            self.assertEqual(document["applicability"]["pair_facts"]["sha256"], sha256(PAIRS))
            path.write_text(path.read_text() + " ")
            for source, diagnostic in ((result.stdout, "selected pair facts hash mismatch"),
                    (result.stdout.replace("op_pairs_sha256", "op_pairs_digest"), "incomplete RTL pair facts selection")):
                with self.subTest(diagnostic=diagnostic):
                    checked = run(OPT, source, CONSUMERS[0])
                    self.assertNotEqual(checked.returncode, 0)
                    self.assertIn(diagnostic, checked.stderr)

    def test_invalid_selections_are_rejected(self):
        def window(engine, value):
            def mutate(d):
                entry = next((c for c in d["capacity"] if c["engine"] == engine), None)
                if entry is None:
                    entry = {"engine": engine, "live": 1, "witness": {"source": "-", "gap": 1}}
                    d["capacity"].append(entry)
                entry.pop("window", None)
                if value is not None:
                    entry["window"] = value
            return mutate
        mutations = [("different hardware IR", lambda d: d["hw_ir"].update(sha256="0" * 64)),
            ("different op_timing facts", lambda d: d["op_timing"].update(sha256="0" * 64)),
            ("not checked against RTL assertions", lambda d: d["op_pairs"][0].update(assertions={"checked": 0, "excluded": 0})),
            ("names no resolved facts block", lambda d: d["op_pairs"][0].update(second="vpu.unknown")),
            ("malformed stable_gap", lambda d: d["op_pairs"][0].update(stable_gap=0)),
            ("expected schema merlin.op_pairs.v1", lambda d: d.update(schema="rtl_extract.op_pairs.v1")),
            ("malformed capacity entry", lambda d: d["capacity"].append(dict(d["capacity"][0]))),
            *(("malformed capacity entry", window(engine, value)) for engine, value in (
                ("vpu", "last_event_age"), ("vpu", None), ("mxu0", "next_issue_age"),
                ("mxu1", "busy"), ("core", "last_event_age")))]
        cases = [("hash mismatch", selection() + f" op-pairs={PAIRS} op-pairs-sha256={'0' * 64}"),
                 ("selected together", selection() + f" op-pairs={PAIRS}")]
        with tempfile.TemporaryDirectory() as directory:
            for n, (diagnostic, mutate) in enumerate(mutations):
                cases.append((diagnostic, selection(pairs=pairs_variant(directory, mutate, f"pairs-{n}.json"))))
            for diagnostic, option in cases:
                with self.subTest(diagnostic=diagnostic, option=option):
                    checked = run(OPT, self.SOURCE, option)
                    self.assertNotEqual(checked.returncode, 0)
                    self.assertIn(diagnostic, checked.stderr)


if __name__ == "__main__":
    unittest.main()
