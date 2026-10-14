"""Pending DMA crosses a PACK loop only with a checked VMEM/CFG proof."""
from __future__ import annotations

import re
import unittest

from test_virtual_lowering import lower, run
from test_delay_insertion import program, nop
from test_untimed_pipeline import DMA_SETUP, WAIT, HALT
from test_virtual_scheduling import cfg_cases
from verification_support import (
    assert_boundaries, last_constant_write, line_index, replace_line, rewrite_line,
)


class DMAPackOverlapTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.source = cfg_cases()["C16 pack beside an explicit DMA"]
        scheduled = run("atlas-opt", cls.source, "--schedule-atlas-virtual")
        if scheduled.returncode:
            raise AssertionError(scheduled.stderr)
        cls.machine = lower(scheduled.stdout, timed=False)

    def test_pending_pack_survives_both_timing_passes_and_handoffs(self) -> None:
        load = line_index(self.machine, '"atlas.dma"', "atlas.virtual_dma_transfer")
        branch = line_index(self.machine, '"atlas.branch"')
        wait = line_index(self.machine, '"atlas.dma_wait"', "atlas.virtual_dma_transfer")
        self.assertLess(load, branch)
        self.assertLess(branch, wait)
        assert_boundaries(self, self.machine)
        for timing in ("--insert-atlas-delays", "--schedule-atlas-stream"):
            timed = run("atlas-opt", self.machine, timing)
            self.assertEqual(timed.returncode, 0, timed.stderr)
            assert_boundaries(self, timed.stdout)

    def test_random_schedules_keep_the_memory_and_cfg_proofs(self) -> None:
        for seed in range(16):
            with self.subTest(seed=seed):
                result = run("atlas-opt", self.source,
                             f"--schedule-atlas-virtual=random-seed={seed}",
                             "--lower-atlas-virtual-to-machine",
                             "--insert-atlas-delays", "--verify-atlas-generated-schedule")
                self.assertEqual(result.returncode, 0, result.stderr)

    def test_source_loop_can_complete_dma_after_its_inner_pack_loop(self) -> None:
        # The DMA completes before the source backedge, so the outer loop's
        # cyclic membership cannot forbid its post-PACK completion block.
        source = cfg_cases()["C14 explicit DMA in a loop body"].replace(
            "    %r =", '    %p8 = "atlas.virtual_pack_fp8"(%x) '
            '{scale_code = 127 : i32} : (!atlas.virtual_bf16) -> !atlas.virtual_fp8\n    %r =')
        self.assertTrue(lower(source, timed=False))
        for seed in [None, *range(16)]:
            with self.subTest(seed=seed):
                schedule = "--schedule-atlas-virtual" + (
                    "" if seed is None else f"=random-seed={seed}")
                result = run("atlas-opt", source, schedule,
                             "--lower-atlas-virtual-to-machine",
                             "--schedule-atlas-stream", "--verify-atlas-generated-schedule")
                self.assertEqual(result.returncode, 0, result.stderr)

    def test_dma_window_overlapping_pack_scratch_is_rejected(self) -> None:
        launch = line_index(self.machine, '"atlas.dma"', "atlas.virtual_dma_transfer")
        constant = last_constant_write(self.machine, launch, 4)
        changed = replace_line(self.machine, constant, "immediate = 32", "immediate = 8")
        assert_boundaries(self, changed, rejects="DMA memory conflict in VMEM")

    def test_scalar_addresses_need_proof_even_with_a_helper_tag(self) -> None:
        load = line_index(self.machine, '"atlas.scalar_load"', 'kind = "lw"')
        unknown = replace_line(self.machine, load, "base = 10", "base = 31")
        assert_boundaries(self, unknown, rejects="cannot prove helper scalar VMEM address")
        # A decidable loop bound cannot turn a pointer into an unbounded walk.
        increment = line_index(self.machine, '"atlas.alu_imm"', "dst = 13", "src = 13")
        stuck = rewrite_line(self.machine, increment,
                             lambda line: re.sub(r"immediate = -?\d+", "immediate = 0", line))
        assert_boundaries(self, stuck, rejects="helper memory proof loop does not exit")
        pointer = line_index(self.machine, '"atlas.upper"', "dst = 10")
        alias = replace_line(self.machine, pointer, "immediate = 32", "immediate = 128")
        assert_boundaries(self, alias, rejects="DMA memory conflict in VMEM")

    def test_pending_loop_cannot_launch_or_wait(self) -> None:
        load = line_index(self.machine, '"atlas.scalar_load"', 'kind = "lw"')
        for operation, fields in (
            ("dma_wait", "channel = 0 : i32"),
            ("dma", 'channel = 2 : i32, direction = "load", dram = 7 : i32, reg = 4 : i32, size = 9 : i32'),
        ):
            with self.subTest(operation=operation):
                def mutate(line: str) -> str:
                    line = line.replace('"atlas.scalar_load"', f'"atlas.{operation}"')
                    return re.sub(r"<\{[^}]+\}>", "<{" + fields + "}>", line, count=1)
                changed = rewrite_line(self.machine, load, mutate)
                assert_boundaries(self, changed, rejects="")
                # Exercise CFG propagation directly on a hand-authored stream,
                # without the earlier generated tag-correspondence diagnostics.
                body = [(operation, fields),
                        ("branch", 'kind = "bne", lhs = 0 : i32, rhs = 0 : i32, offset_bytes = -2 : i32'),
                        nop(), WAIT, HALT]
                result = run("atlas-opt", program(DMA_SETUP + body), "--verify-atlas-timing")
                self.assertNotEqual(result.returncode, 0)
                self.assertIn("CFG incoming edges disagree on pending DMA intervals", result.stderr)

    def test_balanced_dma_commands_inside_helper_still_need_a_loop_proof(self) -> None:
        # These preserve the outer pending set even on the backedge, so CFG
        # identity equality alone is insufficient: the helper proof rejects
        # DMA commands regardless of helper provenance or balanced pairing.
        load = line_index(self.machine, '"atlas.scalar_load"', 'kind = "lw"')
        store = line_index(self.machine, '"atlas.scalar_store"')
        changed = self.machine
        for index, operation, fields in (
            (load, "dma", 'channel = 2 : i32, direction = "load", dram = 7 : i32, reg = 4 : i32, size = 9 : i32'),
            (store, "dma_wait", "channel = 2 : i32"),
        ):
            def mutate(line: str, operation=operation, fields=fields) -> str:
                line = re.sub(r'"atlas.scalar_(?:load|store)"', f'"atlas.{operation}"', line, count=1)
                return re.sub(r"<\{[^}]+\}>", "<{" + fields + "}>", line, count=1)
            changed = rewrite_line(changed, index, mutate)
        assert_boundaries(self, changed, rejects="helper memory proof forbids DMA and nested redirects")

    def test_backedge_cannot_leave_its_pending_interval(self) -> None:
        branch = line_index(self.machine, '"atlas.branch"')
        # Jump to the explicit launch, repeating a command without its WAIT.
        launch = line_index(self.machine, '"atlas.dma"', "atlas.virtual_dma_transfer")
        lines = self.machine.splitlines()
        ops = [i for i, line in enumerate(lines) if ": (!atlas.state)" in line]
        offset = 2 * (ops.index(launch) - ops.index(branch))
        changed = rewrite_line(self.machine, branch,
                               lambda line: re.sub(r"offset_bytes = -?\d+", f"offset_bytes = {offset}", line))
        assert_boundaries(self, changed, rejects="")


if __name__ == "__main__":
    unittest.main()
