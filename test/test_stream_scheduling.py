"""schedule-atlas-stream: list scheduling with the ported npu_model timing model."""

from __future__ import annotations

import collections
import re
import unittest

from test_delay_insertion import (EMIT, OPT, ROOT, addi, cycles, nop, program,
                                  run, stream, without_delays)


def delay_cycles(ops: list[tuple[str, str, str]]) -> int:
    return sum(cycles(fields) + 1 for op, fields, _ in ops if op == "delay")


def instructions(ops: list[tuple[str, str, str]]) -> collections.Counter:
    return collections.Counter(
        (op, re.sub(r"offset_bytes = -?\d+", "", fields))
        for op, fields, reason in ops if op != "delay" and not reason)


def issue_cycles(ops: list[tuple[str, str, str]]) -> list[int]:
    issue, cycle = [], 0
    for op, fields, _ in ops:
        issue.append(cycle)
        cycle += cycles(fields) + 1 if op == "delay" else 1
    return issue


def branch_target(ops: list[tuple[str, str, str]]) -> int:
    index = next(i for i, (op, _, _) in enumerate(ops) if op == "branch")
    offset = int(re.search(r"offset_bytes = (-?\d+)", ops[index][1]).group(1))
    return index + offset // 2


def loop_body(ops: list[tuple[str, str, str]]) -> list[tuple[str, str, str]]:
    branch = next(i for i, (op, _, _) in enumerate(ops) if op == "branch")
    return ops[branch_target(ops):branch + 1]


def work_overlapping_first_dma(ops: list[tuple[str, str, str]]) -> int:
    dma = next(i for i, (op, _, _) in enumerate(ops) if op == "dma")
    wait = next(i for i, (op, _, _) in enumerate(ops) if op == "dma_wait" and i > dma)
    return wait - dma - 1


class StreamSchedulingTest(unittest.TestCase):
    def test_handoff_examples(self) -> None:
        for name in ("mlp", "attention"):
            with self.subTest(example=name):
                source = without_delays(
                    (ROOT / f"test/examples/handoff_{name}_tile.mlir").read_text())
                result = run(OPT, source, "--schedule-atlas-stream")
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertEqual(result.stderr.count("warning:"), 1, result.stderr)
                in_order = stream(run(OPT, source, "--insert-atlas-delays").stdout)
                scheduled = stream(result.stdout)

                self.assertEqual(instructions(scheduled), instructions(in_order))
                self.assertNotEqual(
                    [ops for ops in scheduled if ops[0] != "delay"],
                    [ops for ops in in_order if ops[0] != "delay"])
                self.assertGreater(work_overlapping_first_dma(scheduled), 0)
                self.assertEqual(work_overlapping_first_dma(in_order), 0)
                self.assertLess(delay_cycles(loop_body(scheduled)),
                                delay_cycles(loop_body(in_order)))
                self.assertTrue(all(reason for op, _, reason in scheduled if op == "delay"))

                issue = issue_cycles(scheduled)
                matmul = next(i for i, (op, f, _) in enumerate(scheduled)
                              if op == "mxu_matmul" and "unit = 0" in f)
                pop = next(i for i, (op, f, _) in enumerate(scheduled)
                           if op == "mxu_pop" and "unit = 0" in f)
                self.assertGreaterEqual(issue[pop] - issue[matmul], 64)
                self.assertEqual(scheduled[branch_target(scheduled)][0], "scalar_load")

                checked = run(OPT, result.stdout, "--verify-atlas-machine-stream")
                self.assertEqual(checked.returncode, 0, checked.stderr)
                self.assertEqual(run(EMIT, result.stdout).returncode, 0)
                self.assertNotEqual(
                    run(OPT, result.stdout, "--schedule-atlas-stream").returncode, 0)

    def test_branch_reaches_the_reordered_block_start(self) -> None:
        loop = [
            addi(6, 0, 0), addi(13, 0, 0), addi(14, 0, 3), addi(20, 0, 0),
            addi(20, 20, 1),
            ("vload", 'dst = 0 : i32, base = 6 : i32, offset = 0 : i32, format = "raw"'),
            addi(13, 13, 1),
            ("branch", 'kind = "blt", lhs = 13 : i32, rhs = 14 : i32, offset_bytes = -6 : i32'),
            nop(),
            ("trap", 'kind = "ecall"'),
        ]
        result = run(OPT, program(loop), "--schedule-atlas-stream")
        self.assertEqual(result.returncode, 0, result.stderr)
        ops = stream(result.stdout)
        self.assertEqual([op for op, _, _ in ops[4:]],
                         ["vload", "alu_imm", "alu_imm", "delay", "branch", "alu_imm", "trap"])
        self.assertEqual(branch_target(ops), 4)

    def test_rejects_what_scheduling_cannot_fix(self) -> None:
        dma_then_vload = [
            addi(5, 0, 0), ("dma_config", "channel = 0 : i32, base_reg = 5 : i32"),
            ("dma_wait", "channel = 0 : i32"), addi(6, 0, 0), addi(2, 0, 1024),
            ("upper", 'kind = "lui", dst = 1 : i32, immediate = 589824 : i32'),
            ("dma", 'direction = "load", channel = 0 : i32, reg = 6 : i32, dram = 1 : i32, size = 2 : i32'),
            ("vload", 'dst = 0 : i32, base = 6 : i32, offset = 0 : i32, format = "raw"'),
        ]
        for ops, message in (
            ([addi(1, 0, 1), ("delay", "cycles = 3 : i32"), ("trap", 'kind = "ecall"')],
             "is not allowed in the input"),
            (dma_then_vload, "a delay cannot cover a DMA transfer"),
        ):
            with self.subTest(message=message):
                result = run(OPT, program(ops), "--schedule-atlas-stream")
                self.assertNotEqual(result.returncode, 0)
                self.assertIn(message, result.stderr)


if __name__ == "__main__":
    unittest.main()
