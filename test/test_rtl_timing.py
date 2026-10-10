"""Compiler selection and final timing checks using RTL-computed timing facts.

Set ATLAS_OP_TIMING to a merlin.op_timing.v1 facts file (tools/rtl_extract) and
ATLAS_OOT_BIN_DIR to the compiler under test. Nothing is selected implicitly.
"""

import hashlib
import json
import os
from pathlib import Path
import re
import tempfile
import unittest

from test_delay_insertion import OPT, EMIT, ROOT, addi, nop, program, run, without_delays


FACTS = os.environ.get("ATLAS_OP_TIMING")
RESOLVER = "atlas.op_timing.serialized.v1"
CONSUMERS = ("--insert-atlas-delays", "--schedule-atlas-stream")
VLOAD = ("vload", 'dst = 4 : i32, base = 6 : i32, offset = 0 : i32, format = "raw"')
VSTORE = ("vstore", 'src = 4 : i32, base = 8 : i32, offset = 0 : i32, format = "raw"')
HALT = ("trap", 'kind = "ecall"')
MARKER = ("csr", 'kind = "rrw", dst = 0 : i32, source = 1 : i32, address = 3088 : i32')


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def selection(path=None, digest=None, dma=None):
    path = Path(path or FACTS).resolve()
    options = {"op-timing": str(path), "op-timing-sha256": digest or sha256(path)}
    if dma is not None:
        options["dma"] = dma
    return "--select-atlas-rtl-evidence=" + " ".join(f"{k}={v}" for k, v in options.items())


def selected(ops, *passes, path=None, digest=None, dma=None):
    source = ops if isinstance(ops, str) else program(ops)
    return run(OPT, source, selection(path, digest, dma), *passes)


def facts():
    return json.loads(Path(FACTS).read_text())


def block(document, name):
    return next(b for b in document["op_timing"] if b["name"] == name)


def facts_variant(directory, mutate, name="facts.json"):
    """A mutated copy of the selected facts, to be selected with its own SHA-256."""
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
    # Ordinary RV32 LUI/ADDI materialization; compensate for ADDI's signed
    # low 12 bits. These are encoded scalar instructions, not timing queries.
    low = words & 4095
    if low >= 2048:
        low -= 4096
    high = ((words - low) >> 12) & 0xfffff
    return [lui(dst, high), addi(dst, dst, low)]


def export(source):
    result = run(EMIT, source, "--rtl-timing-json")
    if result.returncode:
        raise AssertionError(result.stderr)
    return json.loads(result.stdout)


@unittest.skipUnless(FACTS and OPT.is_file() and EMIT.is_file(),
                     "requires explicit RTL timing facts and built Atlas tools")
class SelectedRTLTimingTest(unittest.TestCase):
    def selected(self, source, *passes, **options):
        return selected(source, *passes, **options)

    def test_both_consumers_and_final_emission(self):
        source = (ROOT / "test/examples/ee290_vls_copy.mlir").read_text()
        for consumer in CONSUMERS:
            with self.subTest(consumer=consumer):
                result = self.selected(source, consumer, "--verify-atlas-rtl-timing")
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertIn('atlas.rtl_qualification = "conditional"', result.stdout)
                self.assertIn(f'resolver = "{RESOLVER}"', result.stdout)
                self.assertIn(f'op_timing_sha256 = "{sha256(FACTS)}"', result.stdout)
                self.assertIn('"atlas.delay"', result.stdout)
                self.assertEqual(run(EMIT, result.stdout).returncode, 0)
                # Final checker runs afresh on the actual rewritten stream.
                checked = run(OPT, result.stdout, "--verify-atlas-rtl-timing")
                self.assertEqual(checked.returncode, 0, checked.stderr)
                unsafe = run(OPT, without_delays(result.stdout), "--verify-atlas-rtl-timing")
                self.assertNotEqual(unsafe.returncode, 0)
                self.assertIn("serialized engine admission", unsafe.stderr)

    def test_sha_mismatch_and_unselected_streams_are_rejected(self):
        source = program([HALT])
        result = self.selected(source, "--verify-atlas-rtl-timing", digest="0" * 64)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("mismatch", result.stderr)
        result = self.selected(source, "--verify-atlas-rtl-timing", digest="abc")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("SHA-256 is required", result.stderr)
        self.assertNotEqual(run(OPT, source, "--verify-atlas-rtl-timing").returncode, 0)
        good = self.selected(source, "--verify-atlas-rtl-timing")
        self.assertEqual(good.returncode, 0, good.stderr)
        rehashed = re.sub(r'hardware_ir_sha256 = "[0-9a-f]{64}"',
                          'hardware_ir_sha256 = "' + "1" * 64 + '"', good.stdout)
        self.assertNotEqual(rehashed, good.stdout)
        checked = run(OPT, rehashed, "--verify-atlas-rtl-timing")
        self.assertNotEqual(checked.returncode, 0)
        self.assertIn("different hardware IR", checked.stderr)

    def test_missing_unresolved_and_disagreeing_blocks_fail_closed(self):
        ops = [addi(6, 0, 256), VLOAD, delay(33), nop(), HALT]
        mutations = (
            ("missing", lambda d: d["op_timing"].remove(block(d, "vlsu.vload")), "facts block is missing"),
            ("unresolved", lambda d: block(d, "vlsu.vload").update(events=None), "facts block is unresolved"),
            ("count", lambda d: block(d, "vlsu.vload")["events"]["mreg_write"].update(count=31), "does not sweep"),
            ("uneven", lambda d: block(d, "vlsu.vload")["events"]["mreg_write"].update(step=None, last_age=40), "does not sweep"),
            ("group", lambda d: block(d, "vlsu.vload")["events"].pop("vmem_read"), "event group vmem_read is missing"),
            ("busy", lambda d: block(d, "vlsu.vload").update(first_free_age=None), "lacks first_free_age"),
            ("latency", lambda d: block(d, "vlsu.vload")["assumptions"].pop("scratchpad_read_latency"), "lacks scratchpad_read_latency"))
        with tempfile.TemporaryDirectory(prefix="atlas-facts-") as directory:
            for name, mutate, diagnostic in mutations:
                with self.subTest(mutation=name):
                    path = facts_variant(directory, mutate)
                    for consumer in (*CONSUMERS, "--verify-atlas-rtl-timing"):
                        source = ops if consumer.startswith("--verify") else [addi(6, 0, 256), VLOAD, HALT]
                        result = self.selected(source, consumer, path=path)
                        self.assertNotEqual(result.returncode, 0)
                        self.assertIn(RESOLVER, result.stderr)
                        self.assertIn(diagnostic, result.stderr)
                    # An unrelated operation still resolves from the same file.
                    other = self.selected([addi(6, 0, 256), vls("vstore"), delay(33), nop(), HALT],
                                          "--verify-atlas-rtl-timing", path=path)
                    self.assertEqual(other.returncode, 0, other.stderr)
            for name, mutate in (("schema", lambda d: d.update(schema="merlin.op_timing.v0")),
                                 ("hw_ir", lambda d: d.pop("hw_ir")),
                                 ("duplicate", lambda d: d["op_timing"].append(block(d, "vlsu.vload")))):
                with self.subTest(mutation=name):
                    result = self.selected([HALT], "--verify-atlas-rtl-timing",
                                           path=facts_variant(directory, mutate))
                    self.assertNotEqual(result.returncode, 0)
                    self.assertIn("RTL timing facts", result.stderr)

    def test_admission_numbers_come_from_the_facts(self):
        document = facts()
        self.assertEqual(block(document, "vlsu.vload")["first_free_age"], 35)

        def slower(d):
            for name in ("vlsu.vload", "vlsu.vstore"):
                block(d, name).update(first_free_age=40, next_issue_age=40)
        with tempfile.TemporaryDirectory(prefix="atlas-facts-") as directory:
            path = facts_variant(directory, slower)
            for gap, accepted in ((35, False), (39, False), (40, True)):
                with self.subTest(gap=gap):
                    ops = [addi(6, 0, 256), *word_base(8, 768), vls("vload"), delay(gap - 2),
                           vls("vstore", 63, 8), delay(38), nop(), HALT]
                    checked = self.selected(ops, "--verify-atlas-rtl-timing", path=path)
                    self.assertEqual(checked.returncode == 0, accepted, checked.stderr)
            for consumer in CONSUMERS:
                with self.subTest(consumer=consumer):
                    source = (ROOT / "test/examples/ee290_vls_copy.mlir").read_text()
                    result = self.selected(source, consumer, "--verify-atlas-rtl-timing", path=path)
                    self.assertEqual(result.returncode, 0, result.stderr)
                    vector = [i for i in export(result.stdout)["instructions"]
                              if i["mnemonic"] in ("vload", "vstore")]
                    self.assertEqual(vector[1]["logical_issue_cycle"] - vector[0]["logical_issue_cycle"], 40)
                    self.assertEqual(vector[0]["footprint"]["done_age"], 39)
                    self.assertEqual([h["to"] for h in vector[0]["footprint"]["holds"]
                                      if h["unit"].endswith("path")], [39, 39])

    def test_unsupported_domains_reject_in_both_consumers(self):
        for ops in ([VLOAD, HALT], [addi(6, 0, 8), VLOAD, HALT],
                    [("fence", ""), HALT],
                    [("csr", 'kind = "rrw", dst = 0 : i32, source = 1 : i32, address = 3089 : i32'), HALT]):
            for consumer in CONSUMERS:
                with self.subTest(ops=ops, consumer=consumer):
                    result = self.selected(program(ops), consumer)
                    self.assertNotEqual(result.returncode, 0)
                    self.assertIn(RESOLVER, result.stderr)

    def test_selection_annotations_cannot_overstate_or_silently_fall_back(self):
        result = self.selected(program([HALT]), "--verify-atlas-rtl-timing")
        self.assertEqual(result.returncode, 0, result.stderr)
        for source, diagnostic in (
                (result.stdout.replace('atlas.rtl_qualification = "conditional"',
                                       'atlas.rtl_qualification = "qualified"'),
                 "conditional qualification only"),
                (result.stdout.replace(RESOLVER, RESOLVER[:-1] + "99"),
                 "unsupported RTL evidence resolver"),
                (result.stdout.replace(f'resolver = "{RESOLVER}"', f'dma = "latency", resolver = "{RESOLVER}"'),
                 "unsupported DMA policy"),
                (program([HALT]).replace("module {", 'module attributes {atlas.rtl_qualification = "conditional"} {'),
                 "has no selected evidence")):
            checked = run(OPT, source, "--insert-atlas-delays")
            self.assertNotEqual(checked.returncode, 0)
            self.assertIn(diagnostic, checked.stderr)

    def test_exact_serialized_admission_boundary(self):
        # DELAY occupies N+1 cycles, so the two VLS instructions have gap=N+2.
        # Exercise all path transitions, independently varying VMEM bank and
        # logical MREG reuse; serialization applies even without a RAW edge.
        for first in ("vload", "vstore"):
            for second in ("vload", "vstore"):
                for other_mreg in (4, 63):
                    for other_words in (768, 65792):  # VMEM lines 96 and 8224.
                        for gap in (34, 35, 36):
                            with self.subTest(first=first, second=second, mreg=other_mreg,
                                              bank=other_words >> 16, gap=gap):
                                ops = [addi(6, 0, 256), *word_base(8, other_words),
                                       vls(first), delay(gap - 2),
                                       vls(second, other_mreg, 8), delay(33), nop(), HALT]
                                checked = self.selected(program(ops), "--verify-atlas-rtl-timing")
                                self.assertEqual(checked.returncode == 0, gap >= 35, checked.stderr)
                                if gap == 34:
                                    self.assertIn("serialized engine admission", checked.stderr)

    def check_operand_admission(self, setup, instruction, accepted, encoded_immediate=None):
        for consumer in CONSUMERS:
            with self.subTest(consumer=consumer):
                checked = self.selected(program([*setup, instruction, HALT]), consumer,
                                        "--verify-atlas-rtl-timing")
                self.assertEqual(checked.returncode == 0, accepted, checked.stderr)
                if accepted:
                    emitted = run(EMIT, checked.stdout)
                    self.assertEqual(emitted.returncode, 0, emitted.stderr)
                    if encoded_immediate is not None:
                        words = [int(word, 16) for word in re.findall(r"^([0-9a-fA-F]{8})$", emitted.stdout, re.M)]
                        vector_words = [word for word in words if word & 0x7f == 0x07]
                        self.assertEqual(len(vector_words), 1, emitted.stdout)
                        self.assertEqual(vector_words[0] >> 20, encoded_immediate)
                else:
                    self.assertIn("VLS effective tile", checked.stderr)

    def test_each_vmem_bank_first_and_last_legal_tile(self):
        # Six physical banks each hold 8192 32-byte rows. A 32-row tile may
        # begin at row 0 or 8160; the final tile ends at global row 49151.
        for bank in range(6):
            for row in (0, 8160):
                line = bank * 8192 + row
                for kind in ("vload", "vstore"):
                    with self.subTest(bank=bank, row=row, kind=kind):
                        self.check_operand_admission(word_base(6, line * 8),
                                                     vls(kind, 0 if row == 0 else 63), True)
        # One row past the last legal starting tile is unaligned; the first
        # line beyond the six banks is aligned but outside the VMEM aperture.
        for line in (49121, 49152):
            with self.subTest(rejected_line=line):
                self.check_operand_admission(word_base(6, line * 8), vls("vload"), False)

    def test_signed_and_raw_immediate_extremes_have_legal_effective_tiles(self):
        # RTL adds sign-extended imm12 << 5 to the word base, then selects
        # bits 18:3. These explicit operands yield L=0 or L=8192:
        # 65536 + (-2048)*32 = 0; 32 + 2047*32 = 65536;
        # 32 + (-1)*32 = 0. MLIR uses signed offsets; verify the actual raw
        # emitted fields independently (0x800, 0x7ff and 0xfff).
        # Also place negative encodings at the aperture's last legal tile:
        # signed -2048/-1 yields line49120, whereas incorrectly treating the
        # raw encodings as unsigned would yield out-of-range line65504.
        for offset, words, expected_line, encoded in ((-2048, 65536, 0, 0x800),
                                                     (2047, 32, 8192, 0x7ff), (-1, 32, 0, 0xfff),
                                                     (-2048, 458496, 49120, 0x800),
                                                     (-1, 392992, 49120, 0xfff)):
            for kind in ("vload", "vstore"):
                with self.subTest(offset=offset, words=words, line=expected_line, kind=kind):
                    self.check_operand_admission(word_base(6, words), vls(kind, offset=offset), True,
                                                 encoded_immediate=encoded)
        # Positive maximum without the four-row compensation gives line 8188,
        # not a tile-aligned address; the raw negative maximum at B=0 wraps to
        # an effective masked line 57344, outside all six banks.
        for offset in (2047, -2048):
            with self.subTest(rejected_offset=offset):
                self.check_operand_admission([addi(6, 0, 0)], vls("vload", offset=offset), False)
        for offset in (-2049, 2048, 4095, 4096):
            for kind in ("vload", "vstore"):
                with self.subTest(unencoded_mlir_offset=offset, kind=kind):
                    source = program([*word_base(6, 65536), vls(kind, offset=offset), HALT])
                    for consumer in CONSUMERS:
                        checked = self.selected(source, consumer)
                        self.assertNotEqual(checked.returncode, 0)
                        self.assertIn("must be in [-2048, 2047]", checked.stderr)

    def test_known_base_wrap_alias_and_hardwired_zero(self):
        # fffff000 + 2047 + 2047 = fffffffe; adding 258 wraps to 00000100
        # (VMEM line 32). Adding 266 instead produces unaligned line 33.
        prefix = [lui(6, 0xfffff), addi(6, 6, 2047), addi(6, 6, 2047)]
        for increment, accepted in ((258, True), (266, False)):
            with self.subTest(wrap_increment=increment):
                self.check_operand_admission([*prefix, addi(6, 6, increment)], vls("vload"), accepted)
        # A self-update changes the tracked value; another register's update
        # must not change the source register. Writes to x0 must be ignored.
        cases = [([addi(6, 0, 256), addi(6, 6, 8)], 6, False),
                 ([addi(6, 0, 256), addi(6, 6, 8), addi(6, 6, -8)], 6, True),
                 ([addi(6, 0, 0), addi(8, 6, 8)], 6, True),
                 ([addi(6, 0, 0), addi(8, 6, 8)], 8, False),
                 ([addi(0, 0, 8)], 0, True)]
        for setup, base, accepted in cases:
            with self.subTest(setup=setup, base=base):
                self.check_operand_admission(setup, vls("vload", base=base), accepted)

    def test_scalar_memory_operations_take_their_numbers_from_the_facts(self):
        load = block(facts(), "scalar_lsu.load")
        ops = [addi(6, 0, 256), scalar_load("lw", 5, 6, 4), addi(7, 5, 1), HALT]
        for consumer in CONSUMERS:
            with self.subTest(consumer=consumer):
                result = self.selected(ops, consumer, "--verify-atlas-rtl-timing")
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertEqual(run(EMIT, result.stdout).returncode, 0)
                unsafe = run(OPT, without_delays(result.stdout), "--verify-atlas-rtl-timing")
                self.assertNotEqual(unsafe.returncode, 0)
                self.assertIn("x5", unsafe.stderr)
                entry = next(i for i in export(result.stdout)["instructions"] if i["mnemonic"] == "lw")
                footprint = entry["footprint"]
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
                    checked = self.selected([*setup, instruction, HALT], consumer, "--verify-atlas-rtl-timing")
                    self.assertEqual(checked.returncode == 0, accepted, checked.stderr)
                    if accepted:
                        self.assertEqual(run(EMIT, checked.stdout).returncode, 0)
                    else:
                        self.assertIn(RESOLVER, checked.stderr)
        early = self.selected([addi(6, 0, 256), scalar_load("lw", 5, 6, 0), HALT], "--verify-atlas-rtl-timing")
        self.assertNotEqual(early.returncode, 0)
        self.assertIn("asynchronous writes", early.stderr)

    def test_early_marker_and_terminal_are_rejected(self):
        for suffix, diagnostic in (([MARKER, HALT], "asynchronous writes"),
                                   ([HALT], "asynchronous writes"),
                                   ([delay(100), HALT], "while DELAY is stalled")):
            checked = self.selected(program([addi(6, 0, 256), VLOAD, *suffix]),
                                    "--verify-atlas-rtl-timing")
            self.assertNotEqual(checked.returncode, 0)
            self.assertIn(diagnostic, checked.stderr)
        good = self.selected(program([addi(6, 0, 256), VLOAD, delay(33), nop(), HALT]),
                             "--verify-atlas-rtl-timing")
        self.assertEqual(good.returncode, 0, good.stderr)

    def test_missing_terminal_and_work_after_terminal_are_rejected(self):
        for ops in ([addi(1, 0, 1)], [HALT, addi(1, 0, 1)]):
            result = self.selected(program(ops), "--verify-atlas-rtl-timing")
            self.assertNotEqual(result.returncode, 0)

    def test_instruction_memory_overflow_is_rejected_before_scheduling(self):
        source = program([nop()] * 32768 + [HALT])
        for consumer in (*CONSUMERS, "--verify-atlas-rtl-timing"):
            with self.subTest(consumer=consumer):
                result = self.selected(source, consumer)
                self.assertNotEqual(result.returncode, 0)
                self.assertIn("32768-word instruction memory", result.stderr)


if __name__ == "__main__":
    unittest.main()
