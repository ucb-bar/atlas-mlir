"""Compiler selection and final timing checks using explicitly selected RTL evidence.

Set ATLAS_RTL_EVIDENCE_REPORT to a replay report and ATLAS_OOT_BIN_DIR to the
compiler under test. No workspace-specific evidence is selected implicitly.
"""

import hashlib
import json
import os
from pathlib import Path
import re
import unittest

from test_delay_insertion import OPT, EMIT, ROOT, addi, nop, program, run, without_delays


REPORT = os.environ.get("ATLAS_RTL_EVIDENCE_REPORT")
VLOAD = ("vload", 'dst = 4 : i32, base = 6 : i32, offset = 0 : i32, format = "raw"')
VSTORE = ("vstore", 'src = 4 : i32, base = 8 : i32, offset = 0 : i32, format = "raw"')
HALT = ("trap", 'kind = "ecall"')
MARKER = ("csr", 'kind = "rrw", dst = 0 : i32, source = 1 : i32, address = 3088 : i32')


def delay(n):
    return ("delay", f"cycles = {n} : i32")


def lui(dst, immediate):
    return ("upper", f'kind = "lui", dst = {dst} : i32, immediate = {immediate} : i32')


def vls(kind, mreg=4, base=6, offset=0):
    register = "dst" if kind == "vload" else "src"
    return (kind, f'{register} = {mreg} : i32, base = {base} : i32, '
                  f'offset = {offset} : i32, format = "raw"')


def word_base(dst, words):
    # Ordinary RV32 LUI/ADDI materialization; compensate for ADDI's signed
    # low 12 bits. These are encoded scalar instructions, not timing queries.
    low = words & 4095
    if low >= 2048:
        low -= 4096
    high = ((words - low) >> 12) & 0xfffff
    return [lui(dst, high), addi(dst, dst, low)]


@unittest.skipUnless(REPORT and OPT.is_file() and EMIT.is_file(),
                     "requires an explicit replay report and built Atlas tools")
class SelectedRTLTimingTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        path = Path(REPORT).resolve()
        value = json.loads(path.read_text())
        cls.options = dict(evidence=str(path),
                           **{"evidence-sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                              "manifest-sha256": value["inputs"]["manifest"]["sha256"],
                              "hardware-ir-sha256": value["inputs"]["hardware_ir"]["sha256"],
                              "allow-conditional": "true"})

    def selected(self, source, *passes, **overrides):
        options = {**self.options, **overrides}
        selection = "--select-atlas-rtl-evidence=" + " ".join(f"{k}={v}" for k, v in options.items())
        return run(OPT, source, selection, *passes)

    def test_both_consumers_and_final_emission(self):
        source = (ROOT / "test/examples/ee290_vls_copy.mlir").read_text()
        for consumer in ("--insert-atlas-delays", "--schedule-atlas-stream"):
            with self.subTest(consumer=consumer):
                result = self.selected(source, consumer, "--verify-atlas-rtl-timing")
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertIn('atlas.rtl_qualification = "conditional"', result.stdout)
                self.assertIn('"atlas.delay"', result.stdout)
                self.assertEqual(run(EMIT, result.stdout).returncode, 0)
                # Final checker runs afresh on the actual rewritten stream.
                checked = run(OPT, result.stdout, "--verify-atlas-rtl-timing")
                self.assertEqual(checked.returncode, 0, checked.stderr)
                unsafe = run(OPT, without_delays(result.stdout), "--verify-atlas-rtl-timing")
                self.assertNotEqual(unsafe.returncode, 0)
                self.assertIn("serialized VLS admission", unsafe.stderr)

    def test_mismatches_and_explicit_conditional_selection(self):
        source = program([HALT])
        for field in ("evidence-sha256", "manifest-sha256", "hardware-ir-sha256"):
            with self.subTest(field=field):
                result = self.selected(source, "--verify-atlas-rtl-timing", **{field: "0" * 64})
                self.assertNotEqual(result.returncode, 0)
                self.assertIn("mismatch", result.stderr)
        result = self.selected(source, "--verify-atlas-rtl-timing", **{"allow-conditional": "false"})
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("explicit opt-in", result.stderr)
        self.assertNotEqual(run(OPT, source, "--verify-atlas-rtl-timing").returncode, 0)

    def test_unsupported_domains_reject_in_both_consumers(self):
        for ops in ([VLOAD, HALT], [addi(6, 0, 8), VLOAD, HALT],
                    [("fence", ""), HALT],
                    [("csr", 'kind = "rrw", dst = 0 : i32, source = 1 : i32, address = 3089 : i32'), HALT]):
            for consumer in ("--insert-atlas-delays", "--schedule-atlas-stream"):
                with self.subTest(ops=ops, consumer=consumer):
                    result = self.selected(program(ops), consumer)
                    self.assertNotEqual(result.returncode, 0)
                    self.assertIn("atlas.vls.conservative.v1", result.stderr)

    def test_selection_annotations_cannot_overstate_or_silently_fall_back(self):
        result = self.selected(program([HALT]), "--verify-atlas-rtl-timing")
        self.assertEqual(result.returncode, 0, result.stderr)
        for source, diagnostic in (
                (result.stdout.replace('atlas.rtl_qualification = "conditional"',
                                       'atlas.rtl_qualification = "qualified"'),
                 "conditional qualification only"),
                (result.stdout.replace("atlas.vls.conservative.v1", "atlas.vls.conservative.v99"),
                 "unsupported RTL evidence resolver"),
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
                                    self.assertIn("serialized VLS admission", checked.stderr)

    def check_operand_admission(self, setup, instruction, accepted, encoded_immediate=None):
        for consumer in ("--insert-atlas-delays", "--schedule-atlas-stream"):
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
                    for consumer in ("--insert-atlas-delays", "--schedule-atlas-stream"):
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
        for consumer in ("--insert-atlas-delays", "--schedule-atlas-stream",
                         "--verify-atlas-rtl-timing"):
            with self.subTest(consumer=consumer):
                result = self.selected(source, consumer)
                self.assertNotEqual(result.returncode, 0)
                self.assertIn("32768-word instruction memory", result.stderr)


if __name__ == "__main__":
    unittest.main()
