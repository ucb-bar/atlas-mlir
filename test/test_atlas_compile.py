"""--atlas-compile: the one virtual-to-machine pipeline every tool runs."""

import re
import unittest
from itertools import pairwise

from test_virtual_ssa import ROOT, run

# Spilling, boundary DMA, virtual scheduling, lowering, DMA waits, stream
# scheduling, and every verifier, in order.
PASSES = [
    "spill-atlas-mxu-state", "verify-atlas-virtual-stream",
    "lower-atlas-virtual-boundary-to-dma", "schedule-atlas-virtual", "verify-atlas-virtual-stream",
    "spill-atlas-virtual", "verify-atlas-virtual-stream",
    "lower-atlas-pack-relayout", "spill-atlas-virtual", "verify-atlas-virtual-stream",
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

    def test_lowering_leaves_delay_slots_for_the_stream_scheduler(self) -> None:
        # Direct lowering retains the scalar fallback's PACK back-edge.
        stages = ("--atlas-compile-verify", "--atlas-compile-schedule",
                  "--spill-atlas-virtual", "--verify-atlas-virtual-stream")
        lowered = run("atlas-opt", PROGRAMS[1].read_text(), *stages, "--lower-atlas-virtual-to-machine")
        self.assertEqual(lowered.returncode, 0, lowered.stderr)
        timed = run("atlas-opt", lowered.stdout, "--atlas-compile-time")
        self.assertEqual(timed.returncode, 0, timed.stderr)

        def after_back_edge(text: str) -> str:
            ops = [line for line in text.splitlines() if re.search(r'= "atlas\.(?!start)', line)]
            return next(after for line, after in pairwise(ops) if '"atlas.branch"' in line)

        # Lowered, the loop runs straight into the code after it; timed, its
        # slot holds the destination step, which nothing after it in the row reads.
        self.assertIn("atlas.delay_slots_pending", lowered.stdout)
        self.assertNotIn("dst = 0 : i32", after_back_edge(lowered.stdout))
        self.assertNotIn("atlas.delay_slots_pending", timed.stdout)
        self.assertRegex(after_back_edge(timed.stdout), r'"atlas\.alu_imm".*immediate = 32 : i32')


if __name__ == "__main__":
    unittest.main()
