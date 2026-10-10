"""Finite selected BF16 multiply timing, consumers, and evidence mutations.

Select ATLAS_OOT_BIN_DIR, ATLAS_RTL_EVIDENCE_REPORT, and
ATLAS_RTL_VMUL_EVIDENCE_REPORT explicitly. Optional XLU/DMA composition uses
ATLAS_RTL_XLU_EVIDENCE_REPORT and ATLAS_RTL_DMA_EVIDENCE_REPORT.
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
VMUL_REPORT = os.environ.get("ATLAS_RTL_VMUL_EVIDENCE_REPORT")
XLU_REPORT = os.environ.get("ATLAS_RTL_XLU_EVIDENCE_REPORT")
DMA_REPORT = os.environ.get("ATLAS_RTL_DMA_EVIDENCE_REPORT")
CONSUMERS = ("--insert-atlas-delays", "--schedule-atlas-stream")


def vmul(lhs=0, rhs=2, dst=4, kind="mul"):
    return ("vpu_binary", f'kind = "{kind}", dst = {dst} : i32, '
                          f'lhs = {lhs} : i32, rhs = {rhs} : i32')


def identity(path):
    path = Path(path)
    data = path.read_bytes()
    return dict(path=str(path), sha256=hashlib.sha256(data).hexdigest(), bytes=len(data))


@unittest.skipUnless(REPORT and VMUL_REPORT and OPT.is_file() and EMIT.is_file(),
                     "requires explicitly selected VLS/VMUL receipts and built tools")
class SelectedVmulTimingTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        report, vmul_report = Path(REPORT).resolve(), Path(VMUL_REPORT).resolve()
        value = json.loads(report.read_text())
        cls.options = {"evidence": str(report), "evidence-sha256": identity(report)["sha256"],
                       "manifest-sha256": value["inputs"]["manifest"]["sha256"],
                       "hardware-ir-sha256": value["inputs"]["hardware_ir"]["sha256"],
                       "allow-conditional": "true", "vmul-evidence": str(vmul_report),
                       "vmul-evidence-sha256": identity(vmul_report)["sha256"]}
        cls.receipt = json.loads(vmul_report.read_text())

    def selected(self, ops, *passes, **overrides):
        options = {k: v for k, v in {**self.options, **overrides}.items() if v is not None}
        selection = "--select-atlas-rtl-evidence=" + " ".join(f"{k}={v}" for k, v in options.items())
        return run(OPT, ops if isinstance(ops, str) else program(ops), selection, *passes)

    def test_both_consumers_and_observed_stream_export(self):
        for lhs, rhs, dst in ((0, 2, 4), (62, 2, 4), (0, 60, 62)):
            for consumer in CONSUMERS:
                with self.subTest(lhs=lhs, rhs=rhs, dst=dst, consumer=consumer):
                    result = self.selected([vmul(lhs, rhs, dst), HALT], consumer, "--verify-atlas-rtl-timing")
                    self.assertEqual(result.returncode, 0, result.stderr)
                    self.assertIn("atlas.vls_vmul.serialized.v1", result.stdout)
                    exported = run(EMIT, result.stdout, "--rtl-timing-json")
                    self.assertEqual(exported.returncode, 0, exported.stderr)
                    value = json.loads(exported.stdout)
                    self.assertFalse(value["scheduling_qualified"])
                    self.assertEqual(value["qualification"], "conditional")
                    instruction = next(i for i in value["instructions"] if i["mnemonic"] == "vmul.bf16")
                    footprint = instruction["footprint"]
                    accesses = footprint["accesses"]
                    self.assertEqual(len(accesses), 3)
                    self.assertEqual(sorted(a["age"] for a in accesses), [0, 0, 2])
                    self.assertTrue(all(a["count"] == 64 and a["step"] == 1 for a in accesses))
                    self.assertEqual(footprint["done_age"], 65)
                    self.assertNotEqual(run(EMIT, without_delays(result.stdout)).returncode, 0)

    def test_pair_domain_and_unreviewed_compute_rejected(self):
        pairs = ((1, 2, 4), (0, 3, 4), (0, 2, 5), (0, 2, 64),
                 (-2, 2, 4), (0, 0, 4), (0, 32, 4), (0, 2, 0), (0, 2, 34))
        for pair in pairs:
            for consumer in CONSUMERS:
                with self.subTest(pair=pair, consumer=consumer):
                    self.assertNotEqual(self.selected([vmul(*pair), HALT], consumer).returncode, 0)
        for kind in ("add", "sub", "max", "min"):
            self.assertNotEqual(self.selected([vmul(kind=kind), HALT], CONSUMERS[0]).returncode, 0)
        for overrides in ({"vmul-evidence": None, "vmul-evidence-sha256": None},
                          {"vmul-evidence-sha256": "0" * 64},
                          {"vmul-evidence": None}, {"vmul-evidence-sha256": None}):
            self.assertNotEqual(self.selected([vmul(), HALT], CONSUMERS[0], **overrides).returncode, 0)

    def test_serialized_reuse_and_store_boundary(self):
        for gap in (65, 66, 67):
            for next_op in (vmul(6, 8, 10), vls("vstore", 4)):
                with self.subTest(gap=gap, next_op=next_op):
                    result = self.selected([addi(6, 0, 0), vmul(), delay(gap - 2),
                                            next_op, delay(64), nop(), HALT], "--verify-atlas-rtl-timing")
                    self.assertEqual(result.returncode == 0, gap >= 66, result.stderr)

    def test_producer_consumer_and_completion_are_rechecked(self):
        ops = [addi(6, 0, 0), addi(8, 0, 1024), vls("vload", 0),
               vls("vload", 1, offset=32), vls("vload", 2, offset=64),
               vls("vload", 3, offset=96), vmul(), vls("vstore", 4, 8),
               vls("vstore", 5, 8, 32), addi(1, 0, 1), MARKER, HALT]
        for consumer in CONSUMERS:
            result = self.selected(ops, consumer, "--verify-atlas-rtl-timing")
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(run(EMIT, result.stdout).returncode, 0)
            unsafe = without_delays(result.stdout)
            self.assertNotEqual(run(OPT, unsafe, "--verify-atlas-rtl-timing").returncode, 0)
            self.assertNotEqual(run(EMIT, unsafe).returncode, 0)
        for suffix in ([MARKER, delay(64), HALT], [HALT]):
            self.assertNotEqual(self.selected([vmul(), *suffix], "--verify-atlas-rtl-timing").returncode, 0)

    def test_llvm_reconstruction_preserves_evidence_and_rechecks(self):
        result = self.selected([vmul(), HALT], CONSUMERS[0], "--verify-atlas-rtl-timing")
        self.assertEqual(result.returncode, 0, result.stderr)
        unsafe = without_delays(result.stdout)
        for entrypoint in ("--convert-atlas-to-llvm", "--convert-atlas-to-llvm-calls"):
            self.assertNotEqual(run(OPT, unsafe, entrypoint).returncode, 0)
        staged = run(OPT, result.stdout, "--convert-atlas-to-llvm-calls")
        self.assertEqual(staged.returncode, 0, staged.stderr)
        self.assertIn("vmul_evidence_sha256", staged.stdout)
        finalized = run(OPT, staged.stdout, "--finalize-atlas-llvm-calls")
        self.assertEqual(finalized.returncode, 0, finalized.stderr)
        self.assertIn("vmul_evidence_sha256", finalized.stdout)

    def reject_receipt(self, changed, directory):
        receipt = Path(directory) / "receipt.json"
        receipt.write_text(json.dumps(changed))
        result = self.selected([HALT], "--verify-atlas-rtl-timing", **{
            "vmul-evidence": str(receipt), "vmul-evidence-sha256": identity(receipt)["sha256"]})
        self.assertNotEqual(result.returncode, 0, result.stderr)
        self.assertIn("RTL evidence", result.stderr)

    def test_rehashed_provenance_claims_are_rejected(self):
        mutations = (lambda r: r.update(state="prepared"),
                     lambda r: r.update(scheduling_qualified=True),
                     lambda r: r["hardware_ir"].update(sha256="0" * 64),
                     lambda r: r["rtl_snapshots"].pop(),
                     lambda r: r["producer_snapshots"].pop(),
                     lambda r: r["cases"].pop())
        with tempfile.TemporaryDirectory(prefix="atlas-vmul-claims-") as directory:
            for index, mutate in enumerate(mutations):
                with self.subTest(index=index):
                    changed = copy.deepcopy(self.receipt)
                    mutate(changed)
                    self.reject_receipt(changed, directory)

    def test_rehashed_event_changes_reject_copied_success_flags(self):
        case = next(c for c in self.receipt["cases"] if c["name"] == "vmul")
        original = json.loads(Path(case["events"]["path"]).read_text())
        mutations = (lambda e: e["commands"][4]["reads"][0].update(age=1),
                     lambda e: e["commands"][4]["responses"][0].update(port=1),
                     lambda e: e["commands"][4]["writes"][63].update(row=30),
                     lambda e: e["commands"][4]["writes"][0].update(data_hex="00" * 32),
                     lambda e: e["commands"][4].update(lhs=2),
                     lambda e: e["commands"][4].update(release_age=65),
                     lambda e: e["instructions"][12].update(word_u32=0),
                     lambda e: e["endpoints"][0]["engine_state"].update(vpu_write_mask=1),
                     lambda e: e.update(full_memory_checked_bytes=2048),
                     lambda e: e.update(terminal_drain_edges=1))
        with tempfile.TemporaryDirectory(prefix="atlas-vmul-events-") as directory:
            events = Path(directory) / "events.json"
            for index, mutate in enumerate(mutations):
                with self.subTest(index=index):
                    changed = copy.deepcopy(original)
                    mutate(changed)
                    events.write_text(json.dumps(changed))
                    report = copy.deepcopy(self.receipt)
                    next(c for c in report["cases"] if c["name"] == "vmul")["events"] = identity(events)
                    self.reject_receipt(report, directory)

    def test_rehashed_build_recipe_does_not_accept_copied_flags(self):
        original = json.loads(Path(self.receipt["compile_phase"]["path"]).read_text())
        with tempfile.TemporaryDirectory(prefix="atlas-vmul-build-") as directory:
            phase = Path(directory) / "phase.json"
            for mutation in ("--assert", "--top-module", "--no-assert", "-CFLAGS"):
                with self.subTest(mutation=mutation):
                    changed = copy.deepcopy(original)
                    argv = changed["command"]["argv"]
                    if mutation == "--no-assert":
                        argv.append("--no-assert")
                    elif mutation == "-CFLAGS":
                        argv[argv.index("-CFLAGS") + 1] = "-std=c++17 -DNDEBUG"
                    else:
                        argv.remove(mutation)
                    phase.write_text(json.dumps(changed))
                    report = copy.deepcopy(self.receipt)
                    report["compile_phase"] = identity(phase)
                    self.reject_receipt(report, directory)

    @unittest.skipUnless(DMA_REPORT and XLU_REPORT, "requires explicit DMA/XLU composition receipts")
    def test_dma_wait_and_xlu_share_serialized_compute_policy(self):
        from test_rtl_dma_timing import setup, transfer, wait
        from test_rtl_xlu_timing import transpose
        options = {}
        for engine, receipt in (("dma", DMA_REPORT), ("xlu", XLU_REPORT)):
            path = Path(receipt).resolve()
            options.update({engine + "-evidence": str(path),
                            engine + "-evidence-sha256": identity(path)["sha256"]})
        unsafe = [*setup(), transfer(), vmul(), wait(), HALT]
        self.assertNotEqual(self.selected(unsafe, CONSUMERS[0], **options).returncode, 0)
        for consumer in CONSUMERS:
            result = self.selected([*setup(), transfer(), wait(), transpose(0, 2),
                                    vmul(2, 4, 6), HALT], consumer, "--verify-atlas-rtl-timing", **options)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("atlas.vls_dma_xlu_vmul.serialized.v1", result.stdout)
            self.assertEqual(run(EMIT, result.stdout).returncode, 0)


if __name__ == "__main__":
    unittest.main()
