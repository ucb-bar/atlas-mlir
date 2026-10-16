"""Untagged scalar loops may carry DMA only with proved independent accesses."""
from __future__ import annotations

import unittest

from test_delay_insertion import addi, nop, program
from test_virtual_lowering import run


def upper(dst, immediate):
    return "upper", f'kind = "lui", dst = {dst} : i32, immediate = {immediate} : i32'


def constant(dst, value):
    high = (value + 2048) >> 12
    return [upper(dst, high), addi(dst, dst, value - (high << 12))]


def loop_program(*, limit=7, source=4096, destination=8192,
                 unknown_limit=False, unknown_address=False, stride=12,
                 store=False, extra_entry=False, dma_inside=False):
    # No PACK operation, helper annotation or source-operation identity exists.
    setup = [*constant(1, source), *constant(2, destination),
             addi(3, 0, 0), addi(4, 23 if unknown_limit else 0, limit),
             upper(6, 32), upper(7, 0x90000), addi(8, 0, 1024),
             ("dma_config", 'channel = 0 : i32, base_reg = 0 : i32'),
             ("dma", f'channel = 0 : i32, direction = "{"store" if store else "load"}", '
                     'reg = 6 : i32, dram = 7 : i32, size = 8 : i32')]
    if unknown_address:
        setup.append(addi(1, 23, 0))
    body = [("scalar_load", 'kind = "lw", dst = 5 : i32, base = 1 : i32, offset = 0 : i32'),
            ("scalar_store", 'kind = "sw", src = 5 : i32, base = 2 : i32, offset = 0 : i32'),
            addi(1, 1, stride), addi(2, 2, 20), addi(3, 3, 1)]
    if dma_inside:
        body.append(("dma_wait", 'channel = 0 : i32'))
    if extra_entry:
        setup += [("branch", 'kind = "beq", lhs = 0 : i32, rhs = 0 : i32, offset_bytes = 6 : i32'), nop()]
    branch = ("branch", f'kind = "blt", lhs = 3 : i32, rhs = 4 : i32, offset_bytes = {-2 * len(body)} : i32')
    return program(setup + body + [branch, nop(),
                                  ("dma_wait", 'channel = 0 : i32'),
                                  ("trap", 'kind = "ecall"')])


class LoopDMAOverlapTest(unittest.TestCase):
    def test_untagged_nonpack_loops_overlap_under_both_timing_passes(self):
        for limit in (1, 7, 32, 256):
            for timing in ("--insert-atlas-delays", "--schedule-atlas-stream"):
                with self.subTest(limit=limit, timing=timing):
                    source = loop_program(limit=limit)
                    self.assertNotIn("pack", source)
                    result = run("atlas-opt", source, timing, "--verify-atlas-timing")
                    self.assertEqual(result.returncode, 0, result.stderr)
                    self.assertGreater(result.stdout.index('"atlas.dma_wait"'),
                                       result.stdout.index('"atlas.branch"'))

    def test_other_scalar_loop_shapes_are_not_tied_to_pack_copy_order(self):
        for stride in (-4, 0, 4, 28):
            with self.subTest(stride=stride):
                result = run("atlas-opt", loop_program(stride=stride),
                             "--insert-atlas-delays", "--verify-atlas-timing")
                self.assertEqual(result.returncode, 0, result.stderr)

    def test_conflicts_in_later_iterations_are_not_missed(self):
        # First read is outside staging; the second enters it. Check read/write
        # direction: a DMA store can overlap loop reads, but not loop writes.
        for options in ({"source": 0x7fffc, "stride": 4},
                        {"destination": 0x80000},
                        {"store": True, "destination": 0x80000}):
            with self.subTest(options=options):
                result = run("atlas-opt", loop_program(**options), "--insert-atlas-delays")
                self.assertNotEqual(result.returncode, 0)
                self.assertIn("conflict", result.stderr)
        result = run("atlas-opt", loop_program(store=True, source=0x80000),
                     "--insert-atlas-delays", "--verify-atlas-timing")
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_unproved_or_unsupported_loops_cannot_carry_pending_dma(self):
        for options in ({"unknown_limit": True}, {"unknown_address": True},
                        {"limit": 257}, {"extra_entry": True}, {"dma_inside": True}):
            with self.subTest(options=options):
                result = run("atlas-opt", loop_program(**options), "--insert-atlas-delays")
                self.assertNotEqual(result.returncode, 0)
                self.assertRegex(result.stderr, r"no decidable exit|conflict|block boundary")

    def test_unsupported_loop_is_legal_once_dma_has_completed(self):
        source = loop_program(unknown_limit=True)
        # Move the completion before entering the loop, preserving the state
        # chain and the loop's relative branch displacement.
        from verification_support import operations, reorder
        ops = operations(source)
        wait = next(op for op in ops if '"atlas.dma_wait"' in op)
        ops.remove(wait)
        launch = next(i for i, op in enumerate(ops) if '"atlas.dma"' in op)
        source = reorder(source, ops[:launch + 1] + [wait] + ops[launch + 1:])
        result = run("atlas-opt", source, "--insert-atlas-delays", "--verify-atlas-timing")
        self.assertEqual(result.returncode, 0, result.stderr)
