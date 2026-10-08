"""Compiler selection and final timing checks using explicitly selected RTL evidence.

Set ATLAS_RTL_EVIDENCE_REPORT to a replay report and ATLAS_OOT_BIN_DIR to the
compiler under test. No workspace-specific evidence is selected implicitly.
"""

import hashlib
import json
import os
from pathlib import Path
import unittest

from test_delay_insertion import OPT, EMIT, ROOT, addi, nop, program, run, without_delays


REPORT = os.environ.get("ATLAS_RTL_EVIDENCE_REPORT")
VLOAD = ("vload", 'dst = 4 : i32, base = 6 : i32, offset = 0 : i32, format = "raw"')
VSTORE = ("vstore", 'src = 4 : i32, base = 8 : i32, offset = 0 : i32, format = "raw"')
HALT = ("trap", 'kind = "ecall"')
MARKER = ("csr", 'kind = "rrw", dst = 0 : i32, source = 1 : i32, address = 3088 : i32')


def delay(n):
    return ("delay", f"cycles = {n} : i32")


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
        # Load issues at 2. DELAY at 3 occupies N+1 cycles, so store gap=N+2.
        for gap in (34, 35, 36):
            ops = [addi(6, 0, 256), addi(8, 0, 768), VLOAD, delay(gap - 2),
                   VSTORE, delay(33), nop(), HALT]
            checked = self.selected(program(ops), "--verify-atlas-rtl-timing")
            self.assertEqual(checked.returncode == 0, gap >= 35, checked.stderr)

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
