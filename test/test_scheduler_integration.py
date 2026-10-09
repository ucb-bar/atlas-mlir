"""Reference semantics and checked machine handoffs for two-resource schedules.

These checks exercise the compiler and selected timing provider. Selected-core
execution is a separate qualification step.
"""

from __future__ import annotations

import re
import unittest

import test_virtual_evaluator_scheduling as evaluator_scheduling
from test_virtual_evaluator_scheduling import schedules, workloads
from atlas_virtual_evaluator import compare_results, evaluate, parse_program
from test_dma_contract_verification import records as dma_records
from test_mxu_contract_verification import records as mxu_records
from test_untimed_pipeline import PROVIDER, TIMED, UNTIMED, source_contracts
from test_virtual_lowering import emitted, lower, object_words, run


CASES = {"two_dma_reverse_await", "two_chains_same_unit"}
CONTRACT_NAMES = ("dma", "mxu", "tile", "cfg", "buffer")


def contract_attributes(machine: str) -> tuple:
    """Keep complete source facts, including nested CFG/buffer dictionaries."""
    facts = []
    for name in CONTRACT_NAMES:
        marker = f"atlas.virtual_{name}_contract = "
        if marker not in machine:
            continue
        begin = machine.index(marker) + len(marker)
        depth = 0
        quoted = escaped = False
        for end in range(begin, len(machine)):
            char = machine[end]
            if quoted:
                if char == '"' and not escaped:
                    quoted = False
                escaped = char == "\\" and not escaped
                continue
            if char == '"':
                quoted = True
            elif char in "[{":
                depth += 1
            elif char in "]}":
                depth -= 1
                if depth == 0:
                    facts.append((name, machine[begin:end + 1]))
                    break
        else:
            raise AssertionError(f"unterminated {marker}")
    return tuple(facts)


def changed_command(machine: str, operation: str, marker: str, field: str, modulus: int, *, step: int = 1) -> str:
    lines = machine.splitlines()
    index = next(i for i, line in enumerate(lines) if f'"atlas.{operation}"' in line and marker in line)
    pattern = rf"\b{field} = (\d+) : i32"
    match = re.search(pattern, lines[index])
    if match is None:
        raise AssertionError(f"missing {field} on {operation}")
    changed = (int(match[1]) + step) % modulus
    lines[index] = re.sub(pattern, f"{field} = {changed} : i32", lines[index], count=1)
    return "\n".join(lines) + "\n"


class SchedulerIntegrationTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        evaluator_scheduling.VirtualEvaluatorSchedulingTest.setUpClass()

    def checked(self, text: str, *options: str) -> str:
        result = run("atlas-opt", text, *options)
        self.assertEqual(result.returncode, 0, result.stderr)
        return result.stdout

    def cases(self, phase: int):
        return (case for case in workloads(phase) if case[0] in CASES)

    def variants(self, source: str):
        return (("original", source), *schedules(source))

    def test_original_default_and_random_orders_preserve_complete_reference_results(self) -> None:
        for phase in (0, 3):
            for name, source, inputs, literal in self.cases(phase):
                original = evaluate(parse_program(source), inputs)
                compare_results(literal, original)
                for variant, scheduled in self.variants(source):
                    with self.subTest(case=name, phase=phase, schedule=variant):
                        compare_results(original, evaluate(parse_program(scheduled), inputs))

    def test_original_default_and_random_orders_complete_checked_handoffs(self) -> None:
        channels, windows, weights, accumulators = set(), set(), set(), set()
        for name, source, _, _ in self.cases(0):
            for variant, scheduled in self.variants(source):
                with self.subTest(case=name, schedule=variant):
                    untimed = lower(scheduled, timed=False)
                    self.assertIn(UNTIMED, untimed)
                    self.assertNotIn('"atlas.delay"', untimed)
                    self.assertNotIn("atlas.delay_reason", untimed)
                    self.checked(untimed, "--verify-atlas-machine-stream", "--verify-atlas-generated-schedule")
                    facts = contract_attributes(untimed)
                    self.assertTrue(facts)
                    if name == "two_dma_reverse_await":
                        loads = [record for record in dma_records(untimed) if record["direction"] == "load"]
                        channels.update(record["channel"] for record in loads)
                        windows.update(record["staging_word"] for record in loads)
                    else:
                        records = mxu_records(untimed)
                        self.assertEqual({record["unit"] for record in records}, {0})
                        weights.update(record["slot"] for record in records if record["kind"] == "weight_fp8")
                        accumulators.update(record["slot"] for record in records if record["kind"] == "reset")
                    ordered = self.checked(untimed, "--schedule-atlas-stream=insert-delays=false",
                                           "--verify-atlas-machine-stream", "--verify-atlas-generated-schedule")
                    self.assertIn(UNTIMED, ordered)
                    self.assertNotIn('"atlas.delay"', ordered)
                    self.assertEqual(contract_attributes(ordered), facts)
                    timed = self.checked(ordered, "--insert-atlas-delays", "--verify-atlas-timing",
                                         "--verify-atlas-generated-schedule")
                    self.assertIn(TIMED, timed)
                    self.assertIn(PROVIDER, timed)
                    self.assertEqual(contract_attributes(timed), facts)
                    self.assertTrue(emitted(timed))
                    structured = self.checked(timed, "--convert-atlas-to-llvm-calls")
                    self.assertEqual(contract_attributes(structured), facts)
                    direct = self.checked(timed, "--convert-atlas-to-llvm")
                    self.assertEqual(self.checked(structured, "--finalize-atlas-llvm-calls"), direct)
        self.assertEqual(channels, {0, 1})
        self.assertEqual(windows, {131072, 131584})
        self.assertEqual(weights, {0, 1})
        self.assertEqual(accumulators, {0, 1})

    def test_llvm_objects_preserve_final_words(self) -> None:
        for name, source, _, _ in self.cases(0):
            for variant, scheduled in self.variants(source)[:3]:
                with self.subTest(case=name, schedule=variant):
                    ordered = self.checked(lower(scheduled, timed=False), "--schedule-atlas-stream=insert-delays=false")
                    timed = self.checked(ordered, "--insert-atlas-delays", "--verify-atlas-timing")
                    self.assertEqual(object_words(timed), emitted(timed))

    def test_independent_correspondence_rejects_legal_but_wrong_resource_fields(self) -> None:
        for name, source, _, _ in self.cases(0):
            if name == "two_dma_reverse_await":
                mutations = (("dma", "atlas.virtual_dma_transfer", "channel", 8),)
            else:
                mutations = (("mxu_push", 'kind = "weight_fp8"', "slot", 2),
                             ("mxu_matmul", "atlas.virtual_mxu_command", "acc_slot", 2))
            for timed in (False, True):
                machine = lower(schedules(source)[0][1], timed=timed)
                boundaries = [("atlas-opt", ("--verify-atlas-generated-schedule",)),
                              ("atlas-emit", () if timed else ("--allow-untimed",))]
                if timed:
                    boundaries += [("atlas-opt", ("--convert-atlas-to-llvm",)),
                                   ("atlas-opt", ("--convert-atlas-to-llvm-calls",))]
                for operation, marker, field, modulus in mutations:
                    with self.subTest(case=name, field=field, timed=timed):
                        corrupted = changed_command(machine, operation, marker, field, modulus,
                                                    step=2 if field == "channel" else 1)
                        if operation == "dma":
                            first = dma_records(machine)[0]
                            corrupted = changed_command(corrupted, "dma_wait",
                                                        f'atlas.virtual_dma_transfer = {first["id"]} : i32', "channel", 8, step=2)
                            corrupted = changed_command(corrupted, "dma_config",
                                                        f'channel = {first["channel"]} : i32', "channel", 8, step=2)
                        self.assertNotEqual(corrupted, machine)
                        self.assertEqual(contract_attributes(corrupted), contract_attributes(machine))
                        self.assertEqual(source_contracts(corrupted), source_contracts(machine))
                        for tool, options in boundaries:
                            result = run(tool, corrupted, *options)
                            self.assertNotEqual(result.returncode, 0, result.stdout)
                            self.assertEqual(result.stdout, "")
                            if not timed:
                                self.assertIn("contract", result.stderr.lower())

    def test_concurrent_dma_rejects_channel_reuse_wrong_wait_and_unsafe_second_window(self) -> None:
        source = next(source for name, source, _, _ in self.cases(0) if name == "two_dma_reverse_await")
        machine = lower(source, timed=False)
        lines = machine.splitlines()
        launches = [i for i, line in enumerate(lines) if '"atlas.dma"' in line and "atlas.virtual_dma_transfer" in line]
        first, second = launches[:2]
        first_channel = int(re.search(r"channel = (\d+) : i32", lines[first])[1])
        second_channel = int(re.search(r"channel = (\d+) : i32", lines[second])[1])
        self.assertNotEqual(first_channel, second_channel)
        wait = next(i for i, line in enumerate(lines) if '"atlas.dma_wait"' in line and "atlas.virtual_dma_transfer" in line)
        self.assertIn(f"channel = {second_channel} : i32", lines[wait])
        base = max(i for i in range(second) if '"atlas.alu_imm"' in lines[i] and "dst = 4 : i32" in lines[i])
        self.assertIn("immediate = 512 : i32", lines[base])
        mutations = (
            (second, re.sub(r"channel = \d+ : i32", f"channel = {first_channel} : i32", lines[second]), "channel is pending"),
            (wait, re.sub(r"channel = \d+ : i32", f"channel = {first_channel} : i32", lines[wait]), "channel and transfer ID"),
            (base, lines[base].replace("immediate = 512 : i32", "immediate = 0 : i32"), "DMA memory conflict"),
        )
        for index, line, diagnostic in mutations:
            with self.subTest(diagnostic=diagnostic):
                changed = list(lines)
                changed[index] = line
                corrupted = "\n".join(changed) + "\n"
                self.assertNotEqual(corrupted, machine)
                self.assertEqual(contract_attributes(corrupted), contract_attributes(machine))
                for tool, options in (("atlas-opt", ("--verify-atlas-generated-schedule",)),
                                      ("atlas-emit", ("--allow-untimed",))):
                    result = run(tool, corrupted, *options)
                    self.assertNotEqual(result.returncode, 0, result.stdout)
                    self.assertEqual(result.stdout, "")
                    self.assertIn(diagnostic, result.stderr)

    def test_shortened_delays_fail_every_final_handoff(self) -> None:
        for name, source, _, _ in self.cases(0):
            machine = lower(schedules(source)[0][1])
            corrupted = re.sub(r"cycles = \d+ : i32", "cycles = 1 : i32", machine)
            self.assertNotEqual(corrupted, machine)
            self.assertEqual(contract_attributes(corrupted), contract_attributes(machine))
            for tool, options in (("atlas-opt", ("--verify-atlas-timing",)),
                                  ("atlas-opt", ("--verify-atlas-generated-schedule",)),
                                  ("atlas-emit", ()),
                                  ("atlas-opt", ("--convert-atlas-to-llvm",)),
                                  ("atlas-opt", ("--convert-atlas-to-llvm-calls",))):
                with self.subTest(case=name, tool=tool, options=options):
                    result = run(tool, corrupted, *options)
                    self.assertNotEqual(result.returncode, 0, result.stdout)
                    self.assertEqual(result.stdout, "")
                    self.assertRegex(result.stderr.lower(), "tim|issue spacing")

            structured = self.checked(machine, "--convert-atlas-to-llvm-calls")
            lines = structured.splitlines()
            changed = False
            for index, line in enumerate(lines):
                if 'atlas.source_op = "atlas.delay"' not in line:
                    continue
                cycles = re.search(r"cycles = (\d+) : i32", line)
                word = re.search(r"atlas.word = (-?\d+) : i32", line)
                if cycles is None or word is None or int(cycles[1]) <= 1:
                    continue
                # Keep DELAY's encoded word consistent; finalization must check timing.
                encoded = (int(word[1]) & 0xfffff) | (1 << 20)
                line = re.sub(r"cycles = \d+ : i32", "cycles = 1 : i32", line, count=1)
                lines[index] = re.sub(r"atlas.word = -?\d+ : i32", f"atlas.word = {encoded} : i32", line, count=1)
                changed = True
            self.assertTrue(changed, "expected a delay longer than one cycle")
            corrupted = "\n".join(lines) + "\n"
            self.assertEqual(contract_attributes(corrupted), contract_attributes(structured))
            finalized = run("atlas-opt", corrupted, "--finalize-atlas-llvm-calls")
            self.assertNotEqual(finalized.returncode, 0, finalized.stdout)
            self.assertEqual(finalized.stdout, "")
            self.assertRegex(finalized.stderr.lower(), "tim|issue spacing")


if __name__ == "__main__":
    unittest.main()
