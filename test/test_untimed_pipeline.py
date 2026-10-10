"""Untimed lowering retains semantic checks; final timing is checked separately by the selected provider."""

from __future__ import annotations

import re
import unittest

from test_delay_insertion import addi, nop, program
from test_virtual_dma import copy
from test_virtual_evaluator_core import ROWS, schedules
from test_virtual_lowering import EXAMPLES, lower, run, virtual_chain
from test_virtual_mxu_lowering import instructions
from verification_support import (
    PROVIDER, TIMED, TIMED_FINAL, UNTIMED_STATE as UNTIMED, assert_boundaries, checked, contract_text, handoff_chain,
    lines_of, records, rewrite_line, shorten_all_delays,
)

TAGS = ("atlas.virtual_dma_transfer", "atlas.virtual_mxu_command", "atlas.virtual_tile_command")
CONTRACTS = ("dma", "mxu", "tile", "cfg", "source_memory")
TWO_RESOURCE = {row.name: row.source for row in ROWS if row.name in ("two_dma_reverse_await", "two_chains_same_unit")}
DEFAULT = "npu-model-rtl-match-v1"
VLOAD = ("vload", 'dst = 0 : i32, base = 6 : i32, offset = 0 : i32, format = "raw"')
PUBLISH = ("csr", 'kind = "rrw", dst = 0 : i32, source = 0 : i32, address = 3088 : i32')
HALT = ("trap", 'kind = "ecall"')
WAIT = ("dma_wait", "channel = 0 : i32")
DMA_SETUP = [addi(6, 0, 0), addi(1, 0, 0), addi(2, 0, 32),
             ("dma", 'direction = "load", channel = 0 : i32, reg = 6 : i32, dram = 1 : i32, size = 2 : i32')]


def source_contracts(text: str) -> tuple:
    return tuple(records(text, name) for name in ("dma", "mxu", "tile"))


def contract_facts(text: str) -> tuple:
    return tuple(contract_text(text, name) for name in CONTRACTS)


def sources() -> dict[str, str]:
    mxu = (EXAMPLES / "virtual_fp8_matmul_program.mlir").read_text()
    return {"shared input": (EXAMPLES / "virtual_bf16_shared_relu.mlir").read_text(),
            "scalar mailbox and branch": (EXAMPLES / "virtual_bf16_dynamic_branch_program.mlir").read_text(),
            "edge-copy loop": (EXAMPLES / "virtual_bf16_swap_loop_program.mlir").read_text(),
            "DMA FP8": copy("fp8"), "DMA BF16": copy("bf16"), "VPU": (EXAMPLES / "virtual_bf16_vpu_program.mlir").read_text(),
            "MXU0": mxu, "MXU1": mxu.replace("unit = 0 : i32", "unit = 1 : i32"),
            "PACK helpers": (EXAMPLES / "virtual_fp8_two_layer_mlp.mlir").read_text()}


class UntimedPipelineTest(unittest.TestCase):
    def rejected(self, source: str, *options: str, diagnostic: str, tool: str = "atlas-opt") -> None:
        result = run(tool, source, *options)
        self.assertNotEqual(result.returncode, 0, result.stdout)
        self.assertEqual(result.stdout, "")
        self.assertIn(diagnostic, result.stderr)

    def test_untimed_lowering_preserves_synchronization_contracts_and_llvm(self) -> None:
        for name, source in sources().items():
            untimed = lower(source, timed=False)
            with self.subTest(program=name):
                self.assertIn(UNTIMED, untimed)
                self.assertNotIn('"atlas.delay"', untimed)
                self.assertNotIn("atlas.delay_reason", untimed)
                self.assertIn('"atlas.dma_wait"', untimed)
                self.assertTrue(checked(self, untimed, "--allow-untimed", tool="atlas-emit"))
                for tool, options in TIMED_FINAL[1:]:
                    self.rejected(untimed, *options, tool=tool, diagnostic="untimed Atlas stream requires scheduling or delay insertion")
            expected, tags = source_contracts(untimed), [untimed.count(tag + " =") for tag in TAGS]
            for reorder in (False, True):
                with self.subTest(program=name, reorder=reorder):
                    for stage, text in handoff_chain(self, untimed, reorder=reorder).items():
                        self.assertEqual(source_contracts(text), expected, stage)
                        if stage != "direct":
                            self.assertEqual([text.count(tag + " =") for tag in TAGS], tags, stage)

    def test_correspondence_remains_required_before_and_after_timing(self) -> None:
        for timed in (False, True):
            with self.subTest(timed=timed):
                machine = lower(virtual_chain(1), timed=timed)
                corrupted = rewrite_line(machine, lambda line: '"atlas.vload"' in line,
                                         lambda line: re.sub(r"dst = (\d+) : i32", lambda m: f"dst = {(int(m[1]) + 2) % 64} : i32", line, count=1))
                self.assertEqual(source_contracts(corrupted), source_contracts(machine))
                for tool, options in (("atlas-opt", ("--verify-atlas-generated-schedule",)), ("atlas-emit", ("--allow-untimed",))):
                    self.rejected(corrupted, *options, tool=tool, diagnostic="tile contract")

    def test_shortened_timing_is_rejected_at_final_boundaries(self) -> None:
        machine = lower(virtual_chain(1))
        self.assertIn(TIMED, machine)
        changed = shorten_all_delays(machine)
        self.assertEqual(source_contracts(changed), source_contracts(machine))
        for tool, options in (("atlas-opt", ("--verify-atlas-timing",)), ("atlas-emit", ("--allow-untimed",)), *TIMED_FINAL):
            with self.subTest(tool=tool, options=options):
                self.rejected(changed, *options, tool=tool, diagnostic="timing resource conflict")

    def test_timing_selection_is_validated_without_fallback(self) -> None:
        machine = lower(virtual_chain(1))
        for changed, diagnostic in ((machine.replace(PROVIDER, 'atlas.timing_provider = "unknown"'), "unknown Atlas timing provider: unknown"),
                                    (machine.replace(TIMED, 'atlas.timing_state = "unknown"'), "unknown Atlas timing state"),
                                    (machine.replace(PROVIDER + ", ", ""), "timed Atlas stream requires its timing provider"),
                                    (machine.replace(PROVIDER + ", ", "").replace(TIMED + ", ", ""), "requires an explicit atlas.timing_state")):
            self.assertNotEqual(changed, machine)
            for tool, options in (("atlas-opt", ("--verify-atlas-timing",)), ("atlas-emit", ())):
                self.rejected(changed, *options, tool=tool, diagnostic=diagnostic)

    def test_naming_the_default_provider_changes_nothing(self) -> None:
        untimed = lower((EXAMPLES / "virtual_dma_mxu.mlir").read_text(), timed=False)
        for timing in ("--insert-atlas-delays", "--schedule-atlas-stream"):
            with self.subTest(timing=timing):
                implicit = checked(self, untimed, timing)
                self.assertEqual(checked(self, untimed, f"{timing}=provider={DEFAULT}"), implicit)
                self.assertIn(PROVIDER, implicit)
                self.assertEqual(checked(self, implicit, f"--verify-atlas-timing=provider={DEFAULT}"), checked(self, implicit, "--verify-atlas-timing"))

    def test_unregistered_or_different_provider_fails_without_fallback(self) -> None:
        untimed = lower((EXAMPLES / "virtual_dma_mxu.mlir").read_text(), timed=False)
        timed = checked(self, untimed, "--insert-atlas-delays")
        legacy = program([addi(1, 0, 0), HALT])
        for timing, source in (("--insert-atlas-delays", untimed), ("--schedule-atlas-stream", untimed), ("--verify-atlas-timing", legacy)):
            with self.subTest(timing=timing):
                self.rejected(source, f"{timing}=provider=unknown-policy", diagnostic="unknown Atlas timing provider: unknown-policy")
                self.rejected(source, f"{timing}=provider=atlas.vls.conservative.v1", diagnostic="footprint-only coverage cannot borrow model rules")
                self.rejected(timed, f"{timing}=provider=another-policy", diagnostic="supplied timing policy disagrees with retained provider identity")

    def test_timed_hand_authored_stream_is_rechecked_at_handoff(self) -> None:
        timed = checked(self, program([addi(6, 0, 0), VLOAD, HALT]), "--insert-atlas-delays", "--verify-atlas-timing")
        self.assertNotIn("atlas.generated_from_virtual", timed)
        checked(self, checked(self, timed, "--convert-atlas-to-llvm-calls"), "--finalize-atlas-llvm-calls")
        corrupted = shorten_all_delays(timed)
        for tool, options in (("atlas-emit", ()), ("atlas-opt", ("--verify-atlas-machine-stream",)), *TIMED_FINAL[2:]):
            with self.subTest(tool=tool, options=options):
                self.rejected(corrupted, *options, tool=tool, diagnostic="completion")

    def test_completion_publication_waits_for_unrelated_work(self) -> None:
        source = program([addi(6, 0, 0), VLOAD, (PUBLISH[0], PUBLISH[1] + ", atlas.complete = true"), HALT])
        for timing in ("--insert-atlas-delays", "--schedule-atlas-stream"):
            with self.subTest(timing=timing):
                timed = checked(self, source, timing, "--verify-atlas-timing")
                entries = instructions(timed)
                publication = next(i for i, entry in enumerate(entries) if entry["operation"] == "atlas.csr")
                self.assertTrue(entries[publication]["fields"]["atlas.complete"])
                self.assertEqual(entries[publication - 1]["operation"], "atlas.delay")
                checked(self, timed, "--convert-atlas-to-llvm-calls", "--finalize-atlas-llvm-calls")
                self.rejected(shorten_all_delays(timed), "--verify-atlas-timing", diagnostic="atlas.complete")

    def test_back_edge_skips_the_fall_through_drain(self) -> None:
        source = program([addi(6, 0, 0), VLOAD, addi(1, 0, 0), ("branch", 'kind = "beq", lhs = 1 : i32, rhs = 0 : i32, offset_bytes = -2 : i32'), nop(), HALT])
        timed = checked(self, source, "--insert-atlas-delays", "--verify-atlas-timing")
        entries = instructions(timed)
        branch = next(i for i, entry in enumerate(entries) if entry["operation"] == "atlas.branch")
        self.assertEqual(entries[branch + entries[branch]["fields"]["offset_bytes"] // 2]["operation"], "atlas.alu_imm")
        self.rejected(shorten_all_delays(timed), "--verify-atlas-timing", diagnostic="block boundary")

    def test_completion_semantics_do_not_depend_on_optional_annotation(self) -> None:
        for annotation in ("", ", atlas.complete = false", ", atlas.complete = true"):
            with self.subTest(annotation=annotation):
                publish = (PUBLISH[0], PUBLISH[1] + annotation)
                # Later padding protects halt but cannot repair early publication.
                bad = program([addi(6, 0, 0), VLOAD, publish, ("delay", "cycles = 256 : i32"), nop(), HALT])
                self.rejected(bad, "--verify-atlas-timing", diagnostic="atlas.complete")
                checked(self, program([addi(6, 0, 0), VLOAD, publish, HALT]), "--insert-atlas-delays", "--verify-atlas-timing")

    def test_dma_wait_must_precede_completion_publication(self) -> None:
        for option in ("--verify-atlas-timing", "--insert-atlas-delays", "--schedule-atlas-stream"):
            with self.subTest(option=option):
                self.rejected(program(DMA_SETUP + [PUBLISH, WAIT, HALT]), option, diagnostic="completion publication requires DMA.WAIT")
        timed = checked(self, program(DMA_SETUP + [WAIT, PUBLISH, HALT]), "--insert-atlas-delays", "--verify-atlas-timing")
        checked(self, timed, "--convert-atlas-to-llvm-calls", "--finalize-atlas-llvm-calls")

    def test_dma_completion_cannot_escape_into_an_unchecked_successor(self) -> None:
        redirect = [("branch", 'kind = "beq", lhs = 0 : i32, rhs = 0 : i32, offset_bytes = 4 : i32'), nop()]
        self.rejected(program(DMA_SETUP + redirect + [WAIT, HALT]), "--verify-atlas-timing", diagnostic="DMA completion before a block boundary")
        checked(self, program(DMA_SETUP + [WAIT] + redirect + [HALT]), "--insert-atlas-delays", "--verify-atlas-timing")

    def test_manual_multiple_channels_remain_legal(self) -> None:
        source = program([addi(6, 0, 0), addi(7, 0, 1024), addi(1, 0, 0), addi(2, 0, 32), DMA_SETUP[3],
                          ("dma", 'direction = "load", channel = 1 : i32, reg = 7 : i32, dram = 1 : i32, size = 2 : i32'),
                          WAIT, ("dma_wait", "channel = 1 : i32"), HALT])
        ordered = checked(self, source, "--schedule-atlas-stream=insert-delays=false")
        dma_events = [entry["operation"] for entry in instructions(ordered, allow_untimed=True) if entry["operation"] in ("atlas.dma", "atlas.dma_wait")]
        self.assertEqual(dma_events[:2], ["atlas.dma", "atlas.dma"])
        checked(self, ordered, "--insert-atlas-delays", "--verify-atlas-timing")

    def test_two_resource_schedules_use_both_slots_and_complete_checked_handoffs(self) -> None:
        channels, windows, weights, accumulators = set(), set(), set(), set()
        for name, source in TWO_RESOURCE.items():
            for variant, scheduled in (("original", source), *schedules(source)):
                with self.subTest(case=name, schedule=variant):
                    untimed = lower(scheduled, timed=False)
                    self.assertNotIn('"atlas.delay"', untimed)
                    self.assertNotIn("atlas.delay_reason", untimed)
                    if name == "two_dma_reverse_await":
                        loads = [record for record in records(untimed, "dma") if record["direction"] == "load"]
                        channels.update(record["channel"] for record in loads)
                        windows.update(record["staging_word"] for record in loads)
                    else:
                        mxu = records(untimed, "mxu")
                        self.assertEqual({record["unit"] for record in mxu}, {0})
                        weights.update(record["slot"] for record in mxu if record["kind"] == "weight_fp8")
                        accumulators.update(record["slot"] for record in mxu if record["kind"] == "reset")
                    stages = handoff_chain(self, untimed)
                    for stage in ("ordered", "timed", "structured"):
                        self.assertEqual(contract_facts(stages[stage]), contract_facts(untimed), stage)
        self.assertEqual((channels, windows, weights, accumulators), ({0, 1}, {131072, 131584}, {0, 1}, {0, 1}))

    def test_concurrent_dma_rejects_channel_reuse_wrong_wait_and_unsafe_second_window(self) -> None:
        machine = lower(TWO_RESOURCE["two_dma_reverse_await"], timed=False)
        lines = machine.splitlines()
        first, second = lines_of(machine, "dma", "atlas.virtual_dma_transfer")[:2]
        first_channel, second_channel = (int(re.search(r"channel = (\d+) : i32", lines[i])[1]) for i in (first, second))
        self.assertNotEqual(first_channel, second_channel)
        wait = lines_of(machine, "dma_wait", "atlas.virtual_dma_transfer")[0]
        self.assertIn(f"channel = {second_channel} : i32", lines[wait])
        base = max(i for i in range(second) if '"atlas.alu_imm"' in lines[i] and "dst = 4 : i32" in lines[i])
        retarget = lambda line: re.sub(r"channel = \d+ : i32", f"channel = {first_channel} : i32", line)
        for index, change, diagnostic in ((second, retarget, "channel is pending"), (wait, retarget, "channel and transfer ID"),
                                          (base, lambda line: line.replace("immediate = 512 : i32", "immediate = 0 : i32"), "DMA memory conflict")):
            with self.subTest(diagnostic=diagnostic):
                corrupted = rewrite_line(machine, index, change)
                self.assertEqual(contract_facts(corrupted), contract_facts(machine))
                assert_boundaries(self, corrupted, rejects=diagnostic)


if __name__ == "__main__":
    unittest.main()
