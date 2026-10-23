"""Model-backed delay slots preserve scalar dependencies and PACK copy timing."""

from __future__ import annotations

import hashlib
import os
import pathlib
import tempfile
import unittest

from test_delay_insertion import OPT, addi, program, run, stream
from model_evidence import requires_model_evidence
from test_dma_ring_waits import fixture_evidence
from model_evidence import requires_model_evidence
from test_stream_scheduling import branch_target, issue_cycles
from test_virtual_scheduling import F, S, T, function, inp, outp, start


@requires_model_evidence
class ModelDelaySlotTest(unittest.TestCase):
    def setUp(self) -> None:
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.evidence = fixture_evidence(pathlib.Path(temporary.name))

    def schedule(self, source: str) -> str:
        result = run(OPT, source, self.evidence, "--schedule-atlas-stream=provider=model",
                     "--verify-atlas-machine-stream", "--verify-atlas-timing")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertNotIn("atlas.delay_slots_pending", result.stdout)
        return result.stdout

    def test_dependencies_and_memory_prevent_slot_motion(self) -> None:
        cases = (
            ("branch operand", [], [addi(13, 13, 1)], 14),
            ("RAW to branch input", [], [addi(20, 20, 1), addi(13, 20, 0)], 14),
            ("WAR on live source", [], [addi(21, 20, 0), addi(20, 20, 1)], 20),
            ("scalar load", [addi(6, 0, 0)], [
                ("scalar_load", 'kind = "lw", dst = 20 : i32, base = 6 : i32, offset = 0 : i32'),
                addi(13, 13, 1)], 14),
            ("scalar store", [addi(6, 0, 0)], [
                ("scalar_store", 'kind = "sw", src = 20 : i32, base = 6 : i32, offset = 0 : i32'),
                addi(13, 13, 1)], 14),
        )
        for name, prefix, body, rhs in cases:
            with self.subTest(dependency=name):
                source = program([
                    *prefix, addi(13, 0, 0), addi(14, 0, 3), addi(20, 0, 0), *body,
                    ("branch", 'kind = "blt", lhs = 13 : i32, '
                               f'rhs = {rhs} : i32, offset_bytes = {-2 * len(body)} : i32'),
                    ("trap", 'kind = "ecall"'),
                ]).replace("module {", "module attributes {atlas.delay_slots_pending} {", 1)
                ops = stream(self.schedule(source))
                branch = next(i for i, (op, _, _) in enumerate(ops) if op == "branch")
                self.assertEqual(ops[branch + 1][2], "no instruction can fill the delay slot")
                actual = ops[branch_target(ops):branch]
                self.assertEqual(len(actual), len(body))
                for (operation, fields, _), (expected_operation, expected) in zip(actual, body, strict=True):
                    self.assertEqual(operation, expected_operation)
                    for field in expected.split(", "):
                        self.assertIn(field, fields)

    @requires_model_evidence
    def test_pack_pointer_step_fills_slot_without_serializing_copy(self) -> None:
        facts = pathlib.Path(os.environ["ATLAS_TEST_OP_TIMING"])
        pairs = pathlib.Path(os.environ["ATLAS_TEST_PAIR_TIMING"])
        self.evidence = (f"--select-atlas-model-evidence=op-timing={facts} "
                         f"op-timing-sha256={hashlib.sha256(facts.read_bytes()).hexdigest()} "
                         f"pairs={pairs} pairs-sha256={hashlib.sha256(pairs.read_bytes()).hexdigest()} dma=wait")
        source = function("pack", [
            start(), inp("io0", "s1", "x", 0),
            f'%p = "atlas.virtual_pack_fp8"(%x) {{scale_code = 127 : i32}} : ({T}) -> {F}',
            f'%y = "atlas.virtual_mxu_matmul"(%p, %p) {{unit = 0 : i32}} : ({F}, {F}) -> {T}',
            outp("s1", "s2", "y", 0), f"return %s2 : {S}",
        ])
        lowered = run(OPT, source, self.evidence, "--lower-atlas-virtual-to-machine=provider=model",
                      "--insert-atlas-dma-waits")
        self.assertEqual(lowered.returncode, 0, lowered.stderr)
        scheduled = self.schedule(lowered.stdout)
        lines = scheduled.splitlines()
        branch_line = next(i for i, line in enumerate(lines)
                           if '"atlas.branch"' in line and 'cfg_helper = "pack"' in line)
        self.assertIn('"atlas.alu_imm"', lines[branch_line + 1])
        self.assertIn("immediate = 32 : i32", lines[branch_line + 1])
        ops = stream(scheduled)
        body = ops[branch_target(ops):next(i for i, (op, _, _) in enumerate(ops) if op == "branch") + 2]
        self.assertEqual(sum(op == "scalar_load" for op, _, _ in body), 8)
        self.assertEqual(sum(op == "scalar_store" for op, _, _ in body), 8)
        # The confirmed PACK scalar relations pipeline the sixteen memory ops;
        # dropping their replay contexts expands this 28-cycle body.
        self.assertLessEqual(issue_cycles(body)[-1], 27)
        checked = run(OPT, scheduled, "--verify-atlas-generated-schedule")
        self.assertEqual(checked.returncode, 0, checked.stderr)


if __name__ == "__main__":
    unittest.main()
