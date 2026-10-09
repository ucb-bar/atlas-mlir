"""Untimed lowering retains semantic checks; final timing is checked separately."""

from __future__ import annotations

import re
import unittest

from test_virtual_lowering import ROOT, lower, run, virtual_chain
from test_virtual_dma import copy
from test_tile_contract_verification import records as tile_records
from test_mxu_contract_verification import records as mxu_records
from test_dma_contract_verification import records as dma_records
from test_delay_insertion import addi, nop, program
from test_virtual_mxu_lowering import instructions


UNTIMED = 'atlas.timing_state = "untimed"'
TIMED = 'atlas.timing_state = "timed"'
PROVIDER = 'atlas.timing_provider = "npu-model-rtl-match-v1"'


def source_contracts(text: str) -> tuple:
    return dma_records(text), mxu_records(text), tile_records(text)


class UntimedPipelineTest(unittest.TestCase):
    def checked(self, source: str, *options: str) -> str:
        result = run("atlas-opt", source, *options)
        self.assertEqual(result.returncode, 0, result.stderr)
        return result.stdout

    def sources(self) -> dict[str, str]:
        return {"shared input": (ROOT / "test/examples/virtual_bf16_shared_relu.mlir").read_text(),
                "scalar mailbox and branch": (ROOT / "test/examples/virtual_bf16_dynamic_branch_program.mlir").read_text(),
                "edge-copy loop": (ROOT / "test/examples/virtual_bf16_swap_loop_program.mlir").read_text(),
                "DMA FP8": copy("fp8"), "DMA BF16": copy("bf16"),
                "VPU": (ROOT / "test/examples/virtual_bf16_vpu_program.mlir").read_text(),
                "MXU0": (ROOT / "test/examples/virtual_fp8_matmul_program.mlir").read_text(),
                "MXU1": (ROOT / "test/examples/virtual_fp8_matmul_program.mlir").read_text().replace("unit = 0 : i32", "unit = 1 : i32"),
                "PACK helpers": (ROOT / "test/examples/virtual_fp8_two_layer_mlp.mlir").read_text()}

    def test_lowering_is_untimed_and_preserves_synchronization(self) -> None:
        for name, source in self.sources().items():
            with self.subTest(program=name):
                machine = lower(source, timed=False)
                self.assertIn(UNTIMED, machine)
                self.assertNotIn('"atlas.delay"', machine)
                self.assertNotIn("atlas.delay_reason", machine)
                self.assertIn('"atlas.dma_wait"', machine)
                self.checked(machine, "--verify-atlas-machine-stream", "--verify-atlas-generated-schedule")
                inspect = run("atlas-emit", machine, "--allow-untimed")
                self.assertEqual(inspect.returncode, 0, inspect.stderr)
                self.assertTrue(inspect.stdout)
                for tool, options in (("atlas-emit", ()), ("atlas-opt", ("--convert-atlas-to-llvm",)),
                                      ("atlas-opt", ("--convert-atlas-to-llvm-calls",))):
                    rejected = run(tool, machine, *options)
                    self.assertNotEqual(rejected.returncode, 0, rejected.stdout)
                    self.assertEqual(rejected.stdout, "")
                    self.assertIn("tim", rejected.stderr.lower())

    def test_ordering_then_delay_insertion_preserves_contracts_and_llvm(self) -> None:
        for name, source in self.sources().items():
            for reorder in (False, True):
                with self.subTest(program=name, reorder=reorder):
                    untimed = lower(source, timed=False)
                    expected = source_contracts(untimed)
                    if reorder:
                        untimed = self.checked(untimed, "--schedule-atlas-stream=insert-delays=false")
                        self.assertIn(UNTIMED, untimed)
                        self.assertNotIn('"atlas.delay"', untimed)
                        self.assertEqual(source_contracts(untimed), expected)
                    timed = self.checked(untimed, "--insert-atlas-delays", "--verify-atlas-timing")
                    self.assertIn(TIMED, timed)
                    self.assertIn(PROVIDER, timed)
                    self.assertNotIn("atlas.delay_reason", timed)
                    self.assertEqual(source_contracts(timed), expected)
                    emitted = run("atlas-emit", timed)
                    self.assertEqual(emitted.returncode, 0, emitted.stderr)
                    structured = self.checked(timed, "--convert-atlas-to-llvm-calls")
                    self.assertIn(TIMED, structured)
                    self.assertIn(PROVIDER, structured)
                    self.assertEqual(source_contracts(structured), expected)
                    direct = self.checked(timed, "--convert-atlas-to-llvm")
                    final = self.checked(structured, "--finalize-atlas-llvm-calls")
                    self.assertEqual(final, direct)

    def test_correspondence_remains_required_before_and_after_timing(self) -> None:
        for timed in (False, True):
            with self.subTest(timed=timed):
                machine = lower(virtual_chain(1), timed=timed)
                lines = machine.splitlines()
                index = next(i for i, line in enumerate(lines) if '"atlas.vload"' in line)
                fields = lines[index]
                old = int(re.search(r'dst = (\d+) : i32', fields)[1])
                lines[index] = fields.replace(f"dst = {old} : i32", f"dst = {(old + 2) % 64} : i32", 1)
                corrupted = "\n".join(lines) + "\n"
                self.assertNotEqual(corrupted, machine)
                self.assertEqual(source_contracts(corrupted), source_contracts(machine))
                for tool, options in (("atlas-opt", ("--verify-atlas-generated-schedule",)),
                                      ("atlas-emit", ("--allow-untimed",))):
                    result = run(tool, corrupted, *options)
                    self.assertNotEqual(result.returncode, 0, result.stdout)
                    self.assertEqual(result.stdout, "")
                    self.assertIn("tile contract", result.stderr.lower())

    def test_shortened_timing_is_rejected_at_final_boundaries(self) -> None:
        machine = lower(virtual_chain(1))
        self.assertIn(TIMED, machine)
        changed = re.sub(r'cycles = \d+ : i32', 'cycles = 1 : i32', machine)
        self.assertNotEqual(changed, machine)
        self.assertEqual(source_contracts(changed), source_contracts(machine))
        for tool, options in (("atlas-opt", ("--verify-atlas-timing",)),
                              ("atlas-opt", ("--verify-atlas-generated-schedule",)),
                              ("atlas-emit", ()), ("atlas-emit", ("--allow-untimed",)),
                              ("atlas-opt", ("--convert-atlas-to-llvm",)),
                              ("atlas-opt", ("--convert-atlas-to-llvm-calls",))):
            with self.subTest(tool=tool, options=options):
                result = run(tool, changed, *options)
                self.assertNotEqual(result.returncode, 0, result.stdout)
                self.assertEqual(result.stdout, "")
                self.assertIn("tim", result.stderr.lower())

    def test_timing_selection_is_validated_without_fallback(self) -> None:
        machine = lower(virtual_chain(1))
        for changed in (machine.replace(PROVIDER, 'atlas.timing_provider = "unknown"'),
                        machine.replace(TIMED, 'atlas.timing_state = "unknown"'),
                        machine.replace(PROVIDER + ", ", ""),
                        machine.replace(PROVIDER + ", ", "").replace(TIMED + ", ", "")):
            self.assertNotEqual(changed, machine)
            for tool, options in (("atlas-opt", ("--verify-atlas-timing",)), ("atlas-emit", ())):
                result = run(tool, changed, *options)
                self.assertNotEqual(result.returncode, 0, result.stdout)
                self.assertEqual(result.stdout, "")

    def test_timed_hand_authored_stream_is_rechecked_at_handoff(self) -> None:
        source = program([addi(6, 0, 0), ("vload", 'dst = 0 : i32, base = 6 : i32, offset = 0 : i32, format = "raw"'), ("trap", 'kind = "ecall"')])
        timed = self.checked(source, "--insert-atlas-delays", "--verify-atlas-timing")
        self.assertNotIn("atlas.generated_from_virtual", timed)
        structured = self.checked(timed, "--convert-atlas-to-llvm-calls")
        self.checked(structured, "--finalize-atlas-llvm-calls")
        corrupted = re.sub(r'cycles = \d+ : i32', 'cycles = 1 : i32', timed)
        self.assertNotEqual(corrupted, timed)
        for tool, options in (("atlas-emit", ()), ("atlas-opt", ("--verify-atlas-machine-stream",)),
                              ("atlas-opt", ("--convert-atlas-to-llvm",)), ("atlas-opt", ("--convert-atlas-to-llvm-calls",))):
            with self.subTest(tool=tool, options=options):
                result = run(tool, corrupted, *options)
                self.assertNotEqual(result.returncode, 0, result.stdout)
                self.assertIn("completion", result.stderr)

    def test_completion_publication_waits_for_unrelated_work(self) -> None:
        source = program([addi(6, 0, 0), ("vload", 'dst = 0 : i32, base = 6 : i32, offset = 0 : i32, format = "raw"'),
                          ("csr", 'kind = "rrw", dst = 0 : i32, source = 0 : i32, address = 3088 : i32, atlas.complete = true'),
                          ("trap", 'kind = "ecall"')])
        for timing in ("--insert-atlas-delays", "--schedule-atlas-stream"):
            with self.subTest(timing=timing):
                timed = self.checked(source, timing, "--verify-atlas-timing")
                entries = instructions(timed)
                publication = next(i for i, entry in enumerate(entries) if entry["operation"] == "atlas.csr")
                self.assertTrue(entries[publication]["fields"]["atlas.complete"])
                self.assertEqual(entries[publication - 1]["operation"], "atlas.delay")
                self.checked(timed, "--convert-atlas-to-llvm-calls", "--finalize-atlas-llvm-calls")
                corrupted = re.sub(r'cycles = \d+ : i32', 'cycles = 1 : i32', timed)
                result = run("atlas-opt", corrupted, "--verify-atlas-timing")
                self.assertNotEqual(result.returncode, 0, result.stdout)
                self.assertIn("atlas.complete", result.stderr)

    def test_branch_targets_inserted_timing_prelude(self) -> None:
        source = program([addi(6, 0, 0), ("vload", 'dst = 0 : i32, base = 6 : i32, offset = 0 : i32, format = "raw"'),
                          addi(1, 0, 0), ("branch", 'kind = "beq", lhs = 1 : i32, rhs = 0 : i32, offset_bytes = -2 : i32'),
                          nop(), ("trap", 'kind = "ecall"')])
        timed = self.checked(source, "--insert-atlas-delays", "--verify-atlas-timing")
        entries = instructions(timed)
        branch = next(i for i, entry in enumerate(entries) if entry["operation"] == "atlas.branch")
        target = branch + entries[branch]["fields"]["offset_bytes"] // 2
        self.assertEqual(entries[target]["operation"], "atlas.delay")
        corrupted = re.sub(r'cycles = \d+ : i32', 'cycles = 1 : i32', timed)
        result = run("atlas-opt", corrupted, "--verify-atlas-timing")
        self.assertNotEqual(result.returncode, 0, result.stdout)
        self.assertIn("block boundary", result.stderr)

    def test_completion_semantics_do_not_depend_on_optional_annotation(self) -> None:
        vload = ("vload", 'dst = 0 : i32, base = 6 : i32, offset = 0 : i32, format = "raw"')
        for annotation in ("", ", atlas.complete = false", ", atlas.complete = true"):
            with self.subTest(annotation=annotation):
                publish = ("csr", 'kind = "rrw", dst = 0 : i32, source = 0 : i32, address = 3088 : i32' + annotation)
                # Later padding protects halt but cannot repair early publication.
                bad = program([addi(6, 0, 0), vload, publish, ("delay", "cycles = 256 : i32"), nop(), ("trap", 'kind = "ecall"')])
                result = run("atlas-opt", bad, "--verify-atlas-timing")
                self.assertNotEqual(result.returncode, 0, result.stdout)
                self.assertIn("atlas.complete", result.stderr)
                self.checked(program([addi(6, 0, 0), vload, publish, ("trap", 'kind = "ecall"')]), "--insert-atlas-delays", "--verify-atlas-timing")

    def test_dma_wait_must_precede_completion_publication(self) -> None:
        setup = [addi(6, 0, 0), addi(1, 0, 0), addi(2, 0, 32),
                 ("dma", 'direction = "load", channel = 0 : i32, reg = 6 : i32, dram = 1 : i32, size = 2 : i32')]
        publish = ("csr", 'kind = "rrw", dst = 0 : i32, source = 0 : i32, address = 3088 : i32')
        wait = ("dma_wait", 'channel = 0 : i32')
        halt = ("trap", 'kind = "ecall"')
        for option in ("--verify-atlas-timing", "--insert-atlas-delays", "--schedule-atlas-stream"):
            with self.subTest(option=option):
                result = run("atlas-opt", program(setup + [publish, wait, halt]), option)
                self.assertNotEqual(result.returncode, 0, result.stdout)
                self.assertIn("completion publication requires DMA.WAIT", result.stderr)
        timed = self.checked(program(setup + [wait, publish, halt]), "--insert-atlas-delays", "--verify-atlas-timing")
        self.checked(timed, "--convert-atlas-to-llvm-calls", "--finalize-atlas-llvm-calls")

    def test_dma_completion_cannot_escape_into_an_unchecked_successor(self) -> None:
        setup = [addi(6, 0, 0), addi(1, 0, 0), addi(2, 0, 32),
                 ("dma", 'direction = "load", channel = 0 : i32, reg = 6 : i32, dram = 1 : i32, size = 2 : i32')]
        redirect = [("branch", 'kind = "beq", lhs = 0 : i32, rhs = 0 : i32, offset_bytes = 4 : i32'), nop()]
        wait = ("dma_wait", 'channel = 0 : i32')
        halt = ("trap", 'kind = "ecall"')
        bad = program(setup + redirect + [wait, halt])
        result = run("atlas-opt", bad, "--verify-atlas-timing")
        self.assertNotEqual(result.returncode, 0, result.stdout)
        self.assertIn("DMA completion before a block boundary", result.stderr)
        self.checked(program(setup + [wait] + redirect + [halt]), "--insert-atlas-delays", "--verify-atlas-timing")

    def test_manual_multiple_channels_remain_legal(self) -> None:
        source = program([addi(6, 0, 0), addi(7, 0, 1024), addi(1, 0, 0), addi(2, 0, 32),
                          ("dma", 'direction = "load", channel = 0 : i32, reg = 6 : i32, dram = 1 : i32, size = 2 : i32'),
                          ("dma", 'direction = "load", channel = 1 : i32, reg = 7 : i32, dram = 1 : i32, size = 2 : i32'),
                          ("dma_wait", 'channel = 0 : i32'), ("dma_wait", 'channel = 1 : i32'), ("trap", 'kind = "ecall"')])
        ordered = self.checked(source, "--schedule-atlas-stream=insert-delays=false")
        entries = instructions(ordered, allow_untimed=True)
        dma_events = [entry["operation"] for entry in entries if entry["operation"] in ("atlas.dma", "atlas.dma_wait")]
        self.assertEqual(dma_events[:2], ["atlas.dma", "atlas.dma"])
        self.checked(ordered, "--insert-atlas-delays", "--verify-atlas-timing")


if __name__ == "__main__":
    unittest.main()
