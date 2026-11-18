"""Deterministic native work boundaries and terminal allocation exhaustion."""
from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import unittest

from test_virtual_spilling import pressure

ROOT = Path(__file__).resolve().parents[1]
BIN = Path(os.environ.get("ATLAS_OOT_BIN_DIR", ROOT / "build/bin"))


def run(source: str, limit: int, *passes: str):
    return subprocess.run(
        [str(BIN / "atlas-opt"), "--mlir-print-op-on-diagnostic=false",
         f"--atlas-max-work={limit}", "--atlas-report-work", *passes, "-"],
        input=source, text=True, capture_output=True, check=False, timeout=30)


def work_report(result):
    reports = []
    for line in result.stderr.splitlines():
        if line.startswith("{"):
            try:
                value = json.loads(line)
            except json.JSONDecodeError:
                continue
            if value.get("schema") == "atlas-work-budget-v1":
                reports.append(value)
    if len(reports) != 1:
        raise AssertionError(f"expected one work report: {result.stderr}")
    return reports[0]


class NativeWorkBudgetTest(unittest.TestCase):
    def test_process_total_bounds_multiple_affordable_invocations(self):
        function = pressure(count=2).removeprefix("module {").rsplit("}", 1)[0]
        source = "module {" + function + function.replace("@test", "@second") + "}"
        baseline = run(source, 100_000_000, "--schedule-atlas-virtual")
        self.assertEqual(baseline.returncode, 0, baseline.stderr)
        report = work_report(baseline)
        scheduling = report["algorithms"]["virtual-scheduling"]
        self.assertEqual(scheduling["invocations"], 2)
        total = report["total_used"]
        invocation_limit = scheduling["maximum_used"]
        self.assertGreater(total, invocation_limit)
        exact = run(source, invocation_limit, f"--atlas-max-total-work={total}",
                    "--schedule-atlas-virtual")
        self.assertEqual(exact.returncode, 0, exact.stderr)
        self.assertEqual(exact.stdout, baseline.stdout)
        self.assertEqual(work_report(exact)["total_used"], total)
        refused = run(source, invocation_limit, f"--atlas-max-total-work={total - 1}",
                      "--schedule-atlas-virtual")
        self.assertNotEqual(refused.returncode, 0)
        self.assertIn("Atlas process work budget exhausted: virtual-scheduling", refused.stderr)
        self.assertLessEqual(work_report(refused)["total_used"], total - 1)

    def test_immutable_reuse_trial_exhaustion_is_terminal(self):
        from test_immutable_weight_reuse import repeated

        for owner in ("--reuse-atlas-immutable-weights", "--schedule-atlas-virtual",
                      "--lower-atlas-virtual-boundary-to-dma",
                      "--lower-atlas-virtual-to-machine"):
            for flags in ((), ("--atlas-max-total-work=0",)):
                with self.subTest(owner=owner, flags=flags):
                    result = run(repeated(), 1, *flags, owner)
                    self.assertNotEqual(result.returncode, 0)
                    self.assertEqual(result.stdout, "")
                    self.assertIn("register-allocation work budget exhausted during immutable reuse trial",
                                  result.stderr)
                    allocation = work_report(result)["algorithms"]["register-allocation"]
                    self.assertGreater(allocation["exhaustions"], 0)

    def test_scheduler_exact_boundary_preserves_output_and_repeats(self):
        source = pressure(count=2)
        baseline = run(source, 100_000_000, "--schedule-atlas-virtual")
        self.assertEqual(baseline.returncode, 0, baseline.stderr)
        count = work_report(baseline)["algorithms"]["virtual-scheduling"]["maximum_used"]
        self.assertGreater(count, 1)
        exact = run(source, count, "--schedule-atlas-virtual")
        repeated = run(source, count, "--schedule-atlas-virtual")
        for result in (exact, repeated):
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(result.stdout, baseline.stdout)
        self.assertEqual(work_report(exact), work_report(repeated))
        refused = run(source, count - 1, "--schedule-atlas-virtual")
        self.assertNotEqual(refused.returncode, 0)
        self.assertIn("Atlas work budget exhausted: virtual-scheduling", refused.stderr)
        self.assertEqual(work_report(refused)["algorithms"]["virtual-scheduling"]["exhaustions"], 1)

    def test_legacy_spill_cap_has_named_limit_and_usage(self):
        result = run(pressure(), 100_000_000, "--spill-atlas-virtual=max-spills=0")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("spill rewrite work limit reached", result.stderr)
        self.assertIn("budget=spill-atlas-virtual.max-spills, limit=0, used=0, requested=1", result.stderr)

    def test_legacy_mxu_cap_names_the_invoking_pass(self):
        from test_mxu_spilling import mxu_pressure

        source = mxu_pressure(continuation=False)
        for owner in ("spill-atlas-virtual", "spill-atlas-mxu-state"):
            with self.subTest(owner=owner):
                result = run(source, 100_000_000, f"--{owner}=max-spills=0")
                self.assertNotEqual(result.returncode, 0)
                self.assertIn("MXU spill rewrite work limit reached", result.stderr)
                self.assertIn(f"budget={owner}.max-spills, limit=0, used=0, requested=1", result.stderr)
                self.assertNotIn("working slots exhausted", result.stderr)

    def test_allocation_exhaustion_is_terminal_spill_feedback(self):
        # Even this tiny allocation exceeds one work unit. A work refusal
        # must never cause spill retries.
        result = run(pressure(count=1), 1, "--spill-atlas-virtual")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("register-allocation work budget exhausted during spill recovery", result.stderr)
        report = work_report(result)
        allocation = report["algorithms"]["register-allocation"]
        self.assertEqual(allocation["invocations"], 3)
        self.assertGreater(allocation["exhaustions"], 0)
        self.assertNotIn("spill rewrite work limit reached", result.stderr)


if __name__ == "__main__":
    unittest.main()
