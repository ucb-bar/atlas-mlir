"""Selected conditional XLU timing and evidence rejection checks.

Set ATLAS_OOT_BIN_DIR, ATLAS_RTL_EVIDENCE_REPORT and
ATLAS_RTL_XLU_EVIDENCE_REPORT explicitly. DMA composition additionally uses
ATLAS_RTL_DMA_EVIDENCE_REPORT; private receipts are never selected by default.
"""
import copy
import hashlib
import json
import os
from pathlib import Path
import tempfile
import unittest

from test_delay_insertion import OPT, EMIT, addi, nop, program, run, without_delays
from test_rtl_timing import HALT, MARKER, delay, vls

REPORT = os.environ.get("ATLAS_RTL_EVIDENCE_REPORT")
XLU_REPORT = os.environ.get("ATLAS_RTL_XLU_EVIDENCE_REPORT")
DMA_REPORT = os.environ.get("ATLAS_RTL_DMA_EVIDENCE_REPORT")
CONSUMERS = ("--insert-atlas-delays", "--schedule-atlas-stream")


def transpose(src=3, dst=7):
    return ("xlu_transpose", f"dst = {dst} : i32, src = {src} : i32")


def identity(path):
    path = Path(path)
    data = path.read_bytes()
    return dict(path=str(path), sha256=hashlib.sha256(data).hexdigest(), bytes=len(data))


@unittest.skipUnless(REPORT and XLU_REPORT and OPT.is_file() and EMIT.is_file(),
                     "requires explicitly selected VLS/XLU receipts and built tools")
class SelectedXLUTimingTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        report, xlu = Path(REPORT).resolve(), Path(XLU_REPORT).resolve()
        value = json.loads(report.read_text())
        cls.options = {"evidence": str(report), "evidence-sha256": identity(report)["sha256"],
                       "manifest-sha256": value["inputs"]["manifest"]["sha256"],
                       "hardware-ir-sha256": value["inputs"]["hardware_ir"]["sha256"],
                       "allow-conditional": "true", "xlu-evidence": str(xlu),
                       "xlu-evidence-sha256": identity(xlu)["sha256"]}
        cls.receipt = json.loads(xlu.read_text())

    def selected(self, ops, *passes, **overrides):
        options = {**self.options, **overrides}
        options = {key: value for key, value in options.items() if value is not None}
        selection = "--select-atlas-rtl-evidence=" + " ".join(f"{k}={v}" for k, v in options.items())
        return run(OPT, ops if isinstance(ops, str) else program(ops), selection, *passes)

    def test_both_consumers_and_observed_stream_export(self):
        for src, dst in ((3, 7), (5, 5), (3, 35), (0, 63), (63, 0)):
            for consumer in CONSUMERS:
                with self.subTest(src=src, dst=dst, consumer=consumer):
                    result = self.selected([transpose(src, dst), HALT], consumer, "--verify-atlas-rtl-timing")
                    self.assertEqual(result.returncode, 0, result.stderr)
                    self.assertIn("atlas.vls_xlu.serialized.v1", result.stdout)
                    exported = run(EMIT, result.stdout, "--rtl-timing-json")
                    self.assertEqual(exported.returncode, 0, exported.stderr)
                    value = json.loads(exported.stdout)
                    self.assertFalse(value["scheduling_qualified"])
                    self.assertEqual(value["qualification"], "conditional")
                    instruction = next(x for x in value["instructions"] if "xlu" in x["mnemonic"])
                    accesses = instruction["footprint"]["accesses"]
                    self.assertEqual(len(accesses), 2)
                    self.assertEqual(sorted(x["age"] for x in accesses), [1, 34])
                    self.assertTrue(all(x["count"] == 32 and x["step"] == 1 for x in accesses))
                    self.assertNotEqual(run(EMIT, without_delays(result.stdout)).returncode, 0)

    def test_busy_launch_boundary_applies_to_unrelated_registers(self):
        # DELAY N occupies N+1 cycles; the two launches have gap=N+2.
        for gap in (65, 66, 67):
            result = self.selected([transpose(), delay(gap - 2), transpose(22, 24),
                                    delay(64), nop(), HALT], "--verify-atlas-rtl-timing")
            self.assertEqual(result.returncode == 0, gap >= 66, result.stderr)

    def test_vls_transpose_boundaries_and_marker_are_rechecked(self):
        ops = [addi(6, 0, 0), addi(8, 0, 256), vls("vload", 3),
               transpose(3, 35), vls("vstore", 35, 8), addi(1, 0, 1), MARKER, HALT]
        for consumer in CONSUMERS:
            result = self.selected(ops, consumer, "--verify-atlas-rtl-timing")
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(run(EMIT, result.stdout).returncode, 0)
            self.assertNotEqual(run(OPT, without_delays(result.stdout), "--verify-atlas-rtl-timing").returncode, 0)
            self.assertNotEqual(run(EMIT, without_delays(result.stdout)).returncode, 0)
        for suffix in ([MARKER, delay(64), HALT], [HALT]):
            self.assertNotEqual(self.selected([transpose(), *suffix], "--verify-atlas-rtl-timing").returncode, 0)

    def test_xlu_receipt_required_and_ids_are_bounded(self):
        self.assertNotEqual(self.selected([transpose(), HALT], "--insert-atlas-delays",
            **{"xlu-evidence": None, "xlu-evidence-sha256": None}).returncode, 0)
        for src, dst in ((-1, 7), (64, 7), (3, -1), (3, 64)):
            self.assertNotEqual(self.selected([transpose(src, dst), HALT], "--insert-atlas-delays").returncode, 0)
        self.assertNotEqual(self.selected([HALT], "--verify-atlas-rtl-timing",
            **{"xlu-evidence-sha256": "0" * 64}).returncode, 0)

    def reject_receipt(self, receipt, directory):
        path = Path(directory) / "receipt.json"
        path.write_text(json.dumps(receipt))
        result = self.selected([HALT], "--verify-atlas-rtl-timing", **{
            "xlu-evidence": str(path), "xlu-evidence-sha256": identity(path)["sha256"]})
        self.assertNotEqual(result.returncode, 0, result.stderr)

    def test_rehashed_receipt_claims_and_recipes_are_rejected(self):
        with tempfile.TemporaryDirectory(prefix="atlas-xlu-claims-") as directory:
            mutations = [lambda r: r.update(state="prepared"),
                         lambda r: r.update(scheduling_qualified=True),
                         lambda r: r.update(integrated_target_execution_qualified=True),
                         lambda r: r["resolved_facts"].update(first_free_age=65),
                         lambda r: r["extraction"].update(first_line=1),
                         lambda r: r["replay_cases"].pop(),
                         lambda r: next(c for c in r["commands"] if c["stage"] == "verilate-xlu")["argv"].remove("--assert")]
            for mutate in mutations:
                changed = copy.deepcopy(self.receipt)
                mutate(changed)
                self.reject_receipt(changed, directory)

    def test_rehashed_observed_logs_do_not_accept_copied_success_flags(self):
        mutations = [("distinct", 1, "mreg_read_reg", 4),
                     ("distinct", 34, "mreg_write_row", 1),
                     ("distinct", 2, "mreg_response", 0),
                     ("distinct", 65, "write_active", 0),
                     ("distinct", 0, "capture", 0),
                     ("busy-unrelated-age65", 65, "capture", 1),
                     ("reuse-age66", 67, "mreg_read_reg", 3),
                     ("two-cycle-response", 35, "mreg_write", 0)]
        with tempfile.TemporaryDirectory(prefix="atlas-xlu-trace-") as directory:
            for name, cycle, key, value in mutations:
                with self.subTest(name=name, cycle=cycle, key=key):
                    changed = copy.deepcopy(self.receipt)
                    command = next(c for c in changed["commands"] if c["stage"] == "replay-" + name)
                    rows = [json.loads(line) for line in Path(command["stdout"]["path"]).read_text().splitlines()]
                    next(row for row in rows if row.get("cycle") == cycle)[key] = value
                    log = Path(directory) / "trace.log"
                    log.write_text("".join(json.dumps(row) + "\n" for row in rows))
                    command["stdout"] = identity(log)
                    self.reject_receipt(changed, directory)

    @unittest.skipUnless(DMA_REPORT, "requires explicit DMA receipt for composition")
    def test_pending_dma_requires_wait_before_xlu(self):
        from test_rtl_dma_timing import setup, transfer, wait
        path = Path(DMA_REPORT).resolve()
        options = {"dma-evidence": str(path), "dma-evidence-sha256": identity(path)["sha256"]}
        unsafe = [*setup(), transfer(), transpose(), wait(), HALT]
        self.assertNotEqual(self.selected(unsafe, "--insert-atlas-delays", **options).returncode, 0)
        for consumer in CONSUMERS:
            result = self.selected([*setup(), transfer(), wait(), transpose(), HALT],
                                   consumer, "--verify-atlas-rtl-timing", **options)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("atlas.vls_dma_xlu.serialized.v1", result.stdout)


if __name__ == "__main__":
    unittest.main()
