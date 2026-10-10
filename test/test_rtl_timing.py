"""Selection and final timing checks with RTL-computed timing facts.

Set ATLAS_OP_TIMING to a merlin.op_timing.v1 facts file (tools/rtl_extract) and
ATLAS_OOT_BIN_DIR to the compiler under test.
"""

import hashlib
import json
import os
from pathlib import Path
import re
import tempfile
import unittest

from test_delay_insertion import OPT, EMIT, ROOT, addi, nop, program, run, stream, without_delays


FACTS = os.environ.get("ATLAS_OP_TIMING")
RESOLVER = "atlas.op_timing.serialized.v1"
CONSUMERS = ("--insert-atlas-delays", "--schedule-atlas-stream")
VERIFY = "--verify-atlas-rtl-timing"
HALT = ("trap", 'kind = "ecall"')
MARKER = ("csr", 'kind = "rrw", dst = 0 : i32, source = 1 : i32, address = 3088 : i32')
VLS_COPY = ROOT / "test/examples/ee290_vls_copy.mlir"


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def selection(path=None, digest=None, dma=None):
    path = Path(path or FACTS).resolve()
    options = f"op-timing={path} op-timing-sha256={digest or sha256(path)}"
    return "--select-atlas-rtl-evidence=" + options + (f" dma={dma}" if dma else "")


def selected(ops, *passes, path=None, digest=None, dma=None):
    source = ops if isinstance(ops, str) else program(ops)
    return run(OPT, source, selection(path, digest, dma), *passes)


def facts():
    return json.loads(Path(FACTS).read_text())


def block(document, name):
    return next(b for b in document["op_timing"] if b["name"] == name)


def facts_variant(directory, mutate, name="facts.json"):
    document = facts()
    mutate(document)
    path = Path(directory) / name
    path.write_text(json.dumps(document))
    return path


def delay(n):
    return ("delay", f"cycles = {n} : i32")


def lui(dst, immediate):
    return ("upper", f'kind = "lui", dst = {dst} : i32, immediate = {immediate} : i32')


def vls(kind, mreg=4, base=6, offset=0):
    register = "dst" if kind == "vload" else "src"
    return (kind, f'{register} = {mreg} : i32, base = {base} : i32, '
                  f'offset = {offset} : i32, format = "raw"')


def scalar_load(kind="lw", dst=5, base=6, offset=0):
    return ("scalar_load", f'kind = "{kind}", dst = {dst} : i32, base = {base} : i32, offset = {offset} : i32')


def scalar_store(kind="sw", src=2, base=6, offset=0):
    return ("scalar_store", f'kind = "{kind}", src = {src} : i32, base = {base} : i32, offset = {offset} : i32')


def word_base(dst, words):
    """LUI/ADDI materialization of `words`, compensating ADDI's sign."""
    low = words & 4095
    if low >= 2048:
        low -= 4096
    return [lui(dst, ((words - low) >> 12) & 0xfffff), addi(dst, dst, low)]


def branch(kind, lhs, rhs, offset_words):
    return ("branch", f'kind = "{kind}", lhs = {lhs} : i32, rhs = {rhs} : i32, '
                      f'offset_bytes = {2 * offset_words} : i32')


def jal(offset_words):
    return ("jump", f'kind = "jal", dst = 0 : i32, base = 0 : i32, offset = {2 * offset_words} : i32')


def dma_setup():
    return [*word_base(5, 0), ("dma_config", "channel = 0 : i32, base_reg = 5 : i32"),
            *word_base(6, 0), *word_base(1, 0x90000000), *word_base(2, 128)]


VLOAD = vls("vload")
DMA_LOAD = ("dma", 'direction = "load", channel = 0 : i32, reg = 6 : i32, dram = 1 : i32, size = 2 : i32')
DMA_WAIT = ("dma_wait", "channel = 0 : i32")
# Three iterations over a loop-invariant tile; the loop block is words 3..7.
VLS_LOOP = [addi(6, 0, 256), addi(13, 0, 0), addi(14, 0, 3), VLOAD,
            vls("vstore", 4, 6, 32), addi(13, 13, 1), branch("blt", 13, 14, -3), nop(), HALT]


def export(source):
    result = run(EMIT, source, "--rtl-timing-json")
    if result.returncode:
        raise AssertionError(result.stderr)
    return json.loads(result.stdout)


@unittest.skipUnless(FACTS and OPT.is_file() and EMIT.is_file(),
                     "requires explicit RTL timing facts and built Atlas tools")
class SelectedRTLTimingTest(unittest.TestCase):
    def check(self, ops, accepted, *passes, diagnostic=None, **options):
        result = selected(ops, *passes, **options)
        self.assertEqual(result.returncode == 0, accepted, result.stderr)
        if diagnostic:
            self.assertIn(diagnostic, result.stderr)
        return result

    def test_both_consumers_and_final_emission(self):
        for consumer in CONSUMERS:
            with self.subTest(consumer=consumer):
                result = self.check(VLS_COPY.read_text(), True, consumer, VERIFY)
                self.assertIn(f'resolver = "{RESOLVER}"', result.stdout)
                self.assertIn(f'op_timing_sha256 = "{sha256(FACTS)}"', result.stdout)
                self.assertIn('"atlas.delay"', result.stdout)
                self.assertEqual(run(EMIT, result.stdout).returncode, 0)
                unsafe = run(OPT, without_delays(result.stdout), VERIFY)
                self.assertNotEqual(unsafe.returncode, 0)
                self.assertIn("serialized engine admission", unsafe.stderr)

    def test_selection_is_explicit_and_checked(self):
        self.check([HALT], False, VERIFY, digest="0" * 64, diagnostic="mismatch")
        self.check([HALT], False, VERIFY, digest="abc", diagnostic="SHA-256 is required")
        self.assertNotEqual(run(OPT, program([HALT]), VERIFY).returncode, 0)
        good = self.check([HALT], True, VERIFY).stdout
        for source, diagnostic in (
                (re.sub(r'hardware_ir_sha256 = "[0-9a-f]{64}"', 'hardware_ir_sha256 = "' + "1" * 64 + '"', good),
                 "different hardware IR"),
                (good.replace(RESOLVER, RESOLVER[:-1] + "99"), "unsupported RTL evidence resolver"),
                (good.replace(f'resolver = "{RESOLVER}"', f'dma = "latency", resolver = "{RESOLVER}"'),
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
                        self.assertIn(f"{RESOLVER}: vlsu.vload: ", result.stderr)
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
        with tempfile.TemporaryDirectory(prefix="atlas-facts-") as directory:
            path = facts_variant(directory, slower)
            # DELAY N occupies N+1 cycles, so the two VLS issue N+2 apart.
            for gap in (35, 39, 40):
                with self.subTest(gap=gap):
                    self.check([addi(6, 0, 256), *word_base(8, 768), VLOAD, delay(gap - 2),
                                vls("vstore", 63, 8), delay(38), nop(), HALT], gap >= 40, VERIFY, path=path)
            for consumer in CONSUMERS:
                with self.subTest(consumer=consumer):
                    result = self.check(VLS_COPY.read_text(), True, consumer, VERIFY, path=path)
                    vector = [i for i in export(result.stdout)["instructions"]
                              if i["mnemonic"] in ("vload", "vstore")]
                    self.assertEqual(vector[1]["logical_issue_cycle"] - vector[0]["logical_issue_cycle"], 40)
                    self.assertEqual(vector[0]["footprint"]["done_age"], 39)
                    self.assertEqual([h["to"] for h in vector[0]["footprint"]["holds"]
                                      if h["unit"].endswith("path")], [39, 39])

    def test_unsupported_domains_reject_in_both_consumers(self):
        for ops in ([VLOAD, HALT], [addi(6, 0, 8), VLOAD, HALT], [("fence", ""), HALT],
                    [("csr", 'kind = "rrw", dst = 0 : i32, source = 1 : i32, address = 3089 : i32'), HALT]):
            for consumer in CONSUMERS:
                with self.subTest(ops=ops, consumer=consumer):
                    self.check(ops, False, consumer, diagnostic=RESOLVER)

    def test_exact_serialized_admission_boundary(self):
        # Serialization applies across paths, banks and MREGs, without a RAW edge.
        for first in ("vload", "vstore"):
            for second in ("vload", "vstore"):
                for other_mreg, other_words in ((4, 768), (63, 65792)):
                    for gap in (34, 35):
                        with self.subTest(first=first, second=second, mreg=other_mreg, gap=gap):
                            self.check([addi(6, 0, 256), *word_base(8, other_words), vls(first),
                                        delay(gap - 2), vls(second, other_mreg, 8), delay(33), nop(), HALT],
                                       gap >= 35, VERIFY,
                                       diagnostic=None if gap >= 35 else "serialized engine admission")

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
                               diagnostic=None if accepted else RESOLVER)

    def test_terminal_and_marker_admission(self):
        prefix = [addi(6, 0, 256), VLOAD]
        for suffix, diagnostic in (([MARKER, HALT], "asynchronous writes"),
                                   ([HALT], "asynchronous writes"),
                                   ([delay(100), HALT], "while DELAY is stalled")):
            self.check([*prefix, *suffix], False, VERIFY, diagnostic=diagnostic)
        self.check([addi(6, 0, 256), scalar_load("lw", 5, 6, 0), HALT], False, VERIFY,
                   diagnostic="asynchronous writes")
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
                exported = run(EMIT, result.stdout, "--rtl-timing-json")
                self.assertNotEqual(exported.returncode, 0)
                self.assertIn("supports one straight-line block", exported.stderr)
                self.assertEqual(exported.stdout, "")
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
                   diagnostic="leaves its block before prior work completes")
        self.check(loop([delay(30)]), False, VERIFY, diagnostic="can redirect while DELAY is stalled")
        # A fall-through exit drains too; ECALL after the DELAY needs a NOP.
        fall = [addi(6, 0, 256), addi(13, 0, 1), branch("beq", 13, 0, 4), nop(), VLOAD]
        self.check([*fall, delay(33), nop(), HALT], True, VERIFY)
        self.check([*fall, delay(32), nop(), HALT], False, VERIFY,
                   diagnostic="leaves its block before prior work completes")
        # The taken branch reaches the ECALL directly.
        to_halt = [addi(6, 0, 256), addi(13, 0, 1), branch("beq", 13, 0, 3), nop(), VLOAD, HALT]
        for consumer in CONSUMERS:
            with self.subTest(consumer=consumer):
                result = self.check(to_halt, True, consumer, VERIFY)
                ops = stream(result.stdout)
                self.assertEqual([op for op, _, _ in ops[-3:]], ["delay", "alu_imm", "trap"])
                self.assertEqual(ops[-2][2], "a halt does not wait for a delay")

    def test_join_keeps_only_values_equal_on_every_path(self):
        # beq skips to the else arm; both arms reach the join's vload.
        def diamond(then_base, else_base):
            return [addi(5, 0, 1), branch("beq", 5, 0, 5), nop(), addi(6, 0, then_base), jal(3), nop(),
                    addi(6, 0, else_base), VLOAD, HALT]
        for consumer in CONSUMERS:
            with self.subTest(consumer=consumer):
                self.check(diamond(256, 256), True, consumer, VERIFY)
                self.check(diamond(256, 1280), False, consumer,
                           diagnostic="unknown VLS base cannot establish bounded address domain")
        # An induction variable is unknown at the loop header.
        stride = [addi(6, 0, 256), addi(13, 0, 0), addi(14, 0, 3), VLOAD, addi(6, 6, 256),
                  addi(13, 13, 1), branch("blt", 13, 14, -3), nop(), HALT]
        self.check(stride, False, CONSUMERS[0], diagnostic="unknown VLS base")

    def test_control_flow_shape_is_checked(self):
        for ops, diagnostic in (
                ([addi(13, 0, 0), branch("beq", 0, 0, 2), VLOAD, HALT], "admits only ADDI or LUI"),
                ([branch("beq", 0, 0, 2), scalar_load("lw", 5, 0, 0), HALT], "admits only ADDI or LUI"),
                ([addi(13, 0, 0), branch("beq", 0, 0, 2), nop()], "targets the stream end"),
                ([jal(3), nop(), addi(1, 0, 1), HALT], "is unreachable"),
                ([addi(13, 0, 0), branch("blt", 13, 0, 3), nop(), HALT, nop()], "requires a terminal"),
                ([("jump", 'kind = "jal", dst = 1 : i32, base = 0 : i32, offset = 4 : i32'), nop(), HALT],
                 "link value")):
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
                           diagnostic="requires atlas.dma_wait on channel 0 within the block")
                self.check(inside, True, consumer, VERIFY, dma="wait")
        self.check(across, False, VERIFY, dma="wait",
                   diagnostic="requires a matching DMA wait before its block ends")

    def test_instruction_memory_overflow_is_rejected_before_scheduling(self):
        source = program([nop()] * 32768 + [HALT])
        for consumer in (*CONSUMERS, VERIFY):
            with self.subTest(consumer=consumer):
                self.check(source, False, consumer, diagnostic="32768-word instruction memory")


if __name__ == "__main__":
    unittest.main()
