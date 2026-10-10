"""Timing passes select a registered provider; the default keeps the npu-model output."""

from __future__ import annotations

import unittest

from test_delay_insertion import addi, program
from test_virtual_lowering import EXAMPLES, lower, run
from verification_support import PROVIDER

DEFAULT = "npu-model-rtl-match-v1"
DISAGREES = "supplied timing policy disagrees with retained provider identity"


class TimingProviderSelectionTest(unittest.TestCase):
    def setUp(self) -> None:
        self.untimed = lower((EXAMPLES / "virtual_dma_mxu.mlir").read_text(), timed=False)

    def checked(self, source: str, *options: str) -> str:
        result = run("atlas-opt", source, *options)
        self.assertEqual(result.returncode, 0, result.stderr)
        return result.stdout

    def rejected(self, source: str, option: str, diagnostic: str) -> None:
        result = run("atlas-opt", source, option)
        self.assertNotEqual(result.returncode, 0, result.stdout)
        self.assertEqual(result.stdout, "")
        self.assertIn(diagnostic, result.stderr)

    def test_naming_the_default_provider_changes_nothing(self) -> None:
        for timing in ("--insert-atlas-delays", "--schedule-atlas-stream"):
            with self.subTest(timing=timing):
                implicit = self.checked(self.untimed, timing)
                explicit = self.checked(self.untimed, f"{timing}=provider={DEFAULT}")
                self.assertEqual(explicit, implicit)
                self.assertIn(PROVIDER, implicit)
                self.assertEqual(self.checked(implicit, f"--verify-atlas-timing=provider={DEFAULT}"),
                                 self.checked(implicit, "--verify-atlas-timing"))

    def test_unregistered_selection_fails_without_fallback(self) -> None:
        legacy = program([addi(1, 0, 0), ("trap", 'kind = "ecall"')])
        for timing, source in (("--insert-atlas-delays", self.untimed), ("--schedule-atlas-stream", self.untimed),
                               ("--verify-atlas-timing", legacy)):
            with self.subTest(timing=timing):
                self.rejected(source, f"{timing}=provider=unknown-policy", "unknown Atlas timing provider: unknown-policy")
                self.rejected(source, f"{timing}=provider=atlas.vls.conservative.v1", "footprint-only coverage cannot borrow model rules")

    def test_retained_identity_rejects_a_different_selection(self) -> None:
        timed = self.checked(self.untimed, "--insert-atlas-delays")
        for timing in ("--insert-atlas-delays", "--schedule-atlas-stream", "--verify-atlas-timing"):
            with self.subTest(timing=timing):
                self.rejected(timed, f"{timing}=provider=another-policy", DISAGREES)


if __name__ == "__main__":
    unittest.main()
