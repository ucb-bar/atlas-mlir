"""--atlas-compile: the one virtual-to-machine pipeline every tool runs."""

import re
import unittest

from test_virtual_ssa import ROOT, run

# Spilling, boundary DMA, virtual scheduling, lowering, DMA waits, stream
# scheduling, and every verifier, in order.
PASSES = [
    "spill-atlas-mxu-state", "verify-atlas-virtual-stream",
    "lower-atlas-virtual-boundary-to-dma", "schedule-atlas-virtual", "verify-atlas-virtual-stream",
    "spill-atlas-virtual", "verify-atlas-virtual-stream",
    "lower-atlas-virtual-to-machine", "insert-atlas-dma-waits",
    "verify-atlas-machine-stream", "verify-atlas-generated-schedule",
    "schedule-atlas-stream",
    "verify-atlas-machine-stream", "verify-atlas-timing", "verify-atlas-generated-schedule",
]
PROGRAMS = [
    ROOT / "examples/handoff/captured_attention_mlp/00-atlas-virtual-ssa.mlir",
    ROOT / "test/examples/virtual_attention_tile.mlir",
    ROOT / "test/examples/virtual_fp8_mlp_16_tiles.mlir",
    ROOT / "test/examples/virtual_dma_mxu.mlir",
]


class AtlasCompileTest(unittest.TestCase):
    def test_pipeline_runs_every_stage_and_verifier_in_order(self) -> None:
        result = run("atlas-opt", PROGRAMS[0].read_text(), "--atlas-compile", "--dump-pass-pipeline")
        self.assertEqual(result.returncode, 0, result.stderr)
        pipeline = result.stderr[result.stderr.index("builtin.module("):]
        self.assertEqual(re.findall(r"^\s+([a-z0-9-]+)", pipeline, re.M), PASSES)

    def test_compiles_like_the_passes_it_names(self) -> None:
        for program in PROGRAMS:
            with self.subTest(program=program.name):
                source = program.read_text()
                compiled = run("atlas-opt", source, "--atlas-compile")
                self.assertEqual(compiled.returncode, 0, compiled.stderr)
                self.assertIn('atlas.timing_state = "timed"', compiled.stdout)
                spelled = run("atlas-opt", source, *(f"--{name}" for name in PASSES))
                self.assertEqual(spelled.returncode, 0, spelled.stderr)
                self.assertEqual(compiled.stdout, spelled.stdout)
                emitted = run("atlas-emit", compiled.stdout)
                self.assertEqual(emitted.returncode, 0, emitted.stderr)


if __name__ == "__main__":
    unittest.main()
