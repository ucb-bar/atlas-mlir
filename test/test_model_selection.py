"""Candidate controls cannot change the builtin timing path."""

import unittest

from test_virtual_ssa import ROOT, run

EXAMPLE = ROOT / "test/examples/virtual_bf16_shared_relu.mlir"


class ModelSelectionKnobsTest(unittest.TestCase):
    def test_candidate_controls_require_npu_model(self):
        source = EXAMPLE.read_text()
        for option, diagnostic in (
            ("--schedule-atlas-virtual=lookahead=1", "lookahead requires the model"),
            ("--lower-atlas-virtual-to-machine=store-windows=1", "store-windows requires the model"),
        ):
            with self.subTest(option=option):
                result = run("atlas-opt", source, option)
                self.assertNotEqual(result.returncode, 0)
                self.assertIn(diagnostic, result.stderr)
        result = run("atlas-opt", "module {}", "--schedule-atlas-stream=wait-placement=source")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("wait-placement=source requires the model", result.stderr)

    def test_invalid_candidate_controls_fail(self):
        source = EXAMPLE.read_text()
        result = run("atlas-opt", source, "--schedule-atlas-virtual=lookahead=3")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("lookahead must be", result.stderr)
        result = run("atlas-opt", "module {}", "--schedule-atlas-stream=wait-placement=unknown")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("wait-placement must be", result.stderr)

    def test_explicit_defaults_preserve_builtin_ir(self):
        source = EXAMPLE.read_text()
        for ordinary, explicit in (
            ("--schedule-atlas-virtual", "--schedule-atlas-virtual=provider=builtin lookahead=0"),
            ("--lower-atlas-virtual-to-machine", "--lower-atlas-virtual-to-machine=provider=builtin"),
        ):
            with self.subTest(ordinary=ordinary):
                before = run("atlas-opt", source, ordinary)
                after = run("atlas-opt", source, explicit)
                self.assertEqual(before.returncode, 0, before.stderr)
                self.assertEqual(after.returncode, 0, after.stderr)
                self.assertEqual(before.stdout, after.stdout)
