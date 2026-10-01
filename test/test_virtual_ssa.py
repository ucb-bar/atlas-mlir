"""Pre-allocation SSA contracts remain distinct from encodable machine ops."""

from __future__ import annotations

import os
from pathlib import Path
import subprocess
import unittest


ROOT = Path(__file__).resolve().parents[1]
BIN = Path(os.environ.get("ATLAS_OOT_BIN_DIR", ROOT / "build/bin"))
EXAMPLE = ROOT / "test/examples/virtual_bf16_ssa.mlir"


def run(tool: str, source: str, *options: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [str(BIN / tool), *options, "-"],
        input=source,
        text=True,
        capture_output=True,
        check=False,
    )


class VirtualSSAStageTest(unittest.TestCase):
    def setUp(self) -> None:
        self.assertTrue((BIN / "atlas-opt").is_file(), "build atlas-opt first")
        self.assertTrue((BIN / "atlas-emit").is_file(), "build atlas-emit first")

    def test_shared_virtual_value_and_two_outputs_round_trip(self) -> None:
        source = EXAMPLE.read_text()
        first = run("atlas-opt", source)
        self.assertEqual(first.returncode, 0, first.stderr)
        second = run("atlas-opt", first.stdout)
        self.assertEqual(second.returncode, 0, second.stderr)
        self.assertEqual(first.stdout, second.stdout)
        checked = run("atlas-opt", first.stdout, "--verify-atlas-virtual-stream")
        self.assertEqual(checked.returncode, 0, checked.stderr)
        self.assertEqual(first.stdout.count('"atlas.virtual_output_bf16"'), 2)
        self.assertEqual(first.stdout.count('"atlas.virtual_input_bf16"'), 1)
        self.assertIn('"atlas.virtual_vpu_binary"', first.stdout)

    def test_wrong_state_and_value_types_fail(self) -> None:
        source = EXAMPLE.read_text()
        for changed, expected in [
            (source.replace("!atlas.virtual_state", "!atlas.state", 1), "virtual_state"),
            (source.replace("!atlas.virtual_bf16", "!atlas.state", 1), "virtual_bf16"),
        ]:
            with self.subTest(expected=expected):
                result = run("atlas-opt", changed)
                self.assertNotEqual(result.returncode, 0)
                self.assertIn(expected, result.stderr)

    def test_invalid_boundary_index_and_instruction_kind_fail(self) -> None:
        source = EXAMPLE.read_text()
        for changed, expected in [
            (source.replace("index = 0 : i32", "index = -1 : i32", 1), "input index"),
            (source.replace("index = 1 : i32", "index = -1 : i32", 1), "output index"),
            (source.replace('kind = "relu"', 'kind = "invented"', 1), "unary VPU kind"),
            (source.replace('kind = "add"', 'kind = "invented"', 1), "binary VPU kind"),
        ]:
            with self.subTest(expected=expected):
                result = run("atlas-opt", changed)
                self.assertNotEqual(result.returncode, 0)
                self.assertIn(expected, result.stderr)

    def test_virtual_stage_cannot_emit_or_lower_as_physical(self) -> None:
        source = EXAMPLE.read_text()
        for tool, options in [
            ("atlas-emit", ()),
            ("atlas-opt", ("--verify-atlas-machine-stream",)),
            ("atlas-opt", ("--convert-atlas-to-llvm",)),
        ]:
            with self.subTest(tool=tool, options=options):
                result = run(tool, source, *options)
                self.assertNotEqual(result.returncode, 0)
                self.assertEqual(result.stdout, "")

    def test_virtual_verifier_rejects_forked_state_and_duplicate_output(self) -> None:
        source = EXAMPLE.read_text()
        forked = source.replace(
            '"atlas.virtual_output_bf16"(%io2, %t2)',
            '"atlas.virtual_output_bf16"(%io1, %t2)',
        )
        duplicate = source.replace("index = 1 : i32", "index = 0 : i32")
        for changed, expected in [
            (forked, "nonlinear virtual state chain"),
            (duplicate, "duplicate virtual output index"),
            (source.replace(
                '  %io0 = "atlas.virtual_start"() : () -> !atlas.virtual_state',
                '  %io0 = "atlas.virtual_start"() : () -> !atlas.virtual_state\n'
                '  %p0 = "atlas.start"() : () -> !atlas.state',
            ),
             "outside the virtual Atlas stage"),
        ]:
            with self.subTest(expected=expected):
                result = run("atlas-opt", changed, "--verify-atlas-virtual-stream")
                self.assertNotEqual(result.returncode, 0)
                self.assertIn(expected, result.stderr)


if __name__ == "__main__":
    unittest.main()
