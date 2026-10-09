"""Independent MXU execution paths prove physical owners and fresh producers."""

from __future__ import annotations

import unittest

from generated_fixture import BRANCH, JUMP, LABEL, branch_to as branch, jump_to as jump, label
from test_mxu_contract_verification import (
    BOUNDARIES, DELAY, NOP, artifact, chain_fixture, command, record,
)
from test_virtual_lowering import BIN, run


def cfg(facts: list[dict], operations: list[tuple[str, str]]) -> str:
    """Source blocks start at labels; redirects become source edges."""
    return artifact(facts, operations)


def raw_cfg(facts: list[dict], operations: list[tuple[str, str]]) -> str:
    """Physical redirects at Atlas byte displacements, without source edges."""
    labels = {}
    index = 0
    for name, fields in operations:
        if name == LABEL:
            labels[fields] = index
        else:
            index += 1
    emitted = []
    for name, fields in operations:
        if name == LABEL:
            continue
        if name in (BRANCH, JUMP):
            offset = 2 * (labels[fields] - len(emitted))
            name, fields = (("branch", f'kind = "beq", lhs = 1 : i32, rhs = 0 : i32, offset_bytes = {offset} : i32')
                            if name == BRANCH else
                            ("jump", f'kind = "jal", dst = 0 : i32, base = 0 : i32, offset = {offset} : i32'))
        emitted.append((name, fields))
    return artifact(facts, emitted)


def two_sessions() -> tuple[list[dict], list[tuple[str, str]]]:
    # Two reset/pop sessions share a weight. Each dynamic reset needs a fresh
    # execution of its weight producer, even while the other session is pending.
    facts = [record(0, "weight_fp8", 0, 11, 1)]
    operations = [command("mxu_push", 'kind = "weight_fp8", unit = 0 : i32, src = 11 : i32, slot = 1 : i32', 0), DELAY]
    for reset, pop in ((1, 2), (3, 4)):
        facts += [record(reset, "reset", 0, 9, 0, weight_slot=1, weight=0),
                  record(pop, "pop_bf16", 0, 40, 0, previous=reset, scale_reg=0)]
        operations += [command("mxu_matmul", 'unit = 0 : i32, src = 9 : i32, weight_slot = 1 : i32, acc_slot = 0 : i32, accumulate = false', reset), DELAY,
                       command("mxu_pop", 'format = "bf16", unit = 0 : i32, dst = 40 : i32, slot = 0 : i32, scale_reg = 0 : i32', pop), DELAY]
    return facts, operations


class MXUCFGContractVerificationTest(unittest.TestCase):
    def setUp(self) -> None:
        self.assertTrue((BIN / "atlas-opt").is_file(), "build atlas-opt first")
        self.assertTrue((BIN / "atlas-emit").is_file(), "build atlas-emit first")

    def check(self, machine: str, *, accepted: bool) -> None:
        for tool, options in BOUNDARIES:
            with self.subTest(tool=tool, options=options):
                result = run(tool, machine, *options)
                if accepted:
                    self.assertEqual(result.returncode, 0, result.stderr)
                    self.assertTrue(result.stdout)
                else:
                    self.assertNotEqual(result.returncode, 0, result.stdout)
                    self.assertEqual(result.stdout, "")
                    self.assertIn("MXU contract", result.stderr)

    def test_branch_skipping_weight_or_accumulator_producer_fails(self) -> None:
        facts, operations = chain_fixture()
        for producer in (0, 2, 4):
            # Both incoming paths reach the consumer, but one skipped a required
            # producer. All static tags still match and remain in source order.
            changed = (operations[:producer] + [branch("consumer"), NOP]
                       + operations[producer:producer + 2] + [label("consumer")]
                       + operations[producer + 2:])
            with self.subTest(producer=producer):
                self.check(cfg(facts, changed), accepted=False)
        facts = [record(0, "weight_fp8", 0, 11, 1), record(1, "acc_fp8", 0, 9, 0),
                 record(2, "accumulate", 0, 9, 0, weight_slot=1, weight=0, previous=1),
                 record(3, "pop_bf16", 0, 40, 0, previous=2, scale_reg=0)]
        operations = [command("mxu_push", 'kind = "weight_fp8", unit = 0 : i32, src = 11 : i32, slot = 1 : i32', 0), DELAY,
                      branch("consumer"), NOP,
                      command("mxu_push", 'kind = "acc_fp8", unit = 0 : i32, src = 9 : i32, slot = 0 : i32', 1), DELAY,
                      label("consumer"), command("mxu_matmul", 'unit = 0 : i32, src = 9 : i32, weight_slot = 1 : i32, acc_slot = 0 : i32, accumulate = true', 2), DELAY,
                      command("mxu_pop", 'format = "bf16", unit = 0 : i32, dst = 40 : i32, slot = 0 : i32, scale_reg = 0 : i32', 3), DELAY]
        self.check(cfg(facts, operations), accepted=False)

    def test_live_versions_cannot_cross_a_skipped_readout_or_exit(self) -> None:
        facts, operations = chain_fixture()
        changed = operations[:-2] + [branch("exit"), NOP] + operations[-2:] + [label("exit")]
        self.check(cfg(facts, changed), accepted=False)
        first, a = chain_fixture()
        second, b = chain_fixture(first_id=5)
        for entry in second:
            entry["block"] = 1
        changed = a[:-2] + [branch("second"), NOP] + a[-2:] + [label("second")] + b
        self.check(cfg(first + second, changed), accepted=False)

    def test_final_conditional_fallthrough_requires_free_accumulator(self) -> None:
        facts, operations = chain_fixture()
        # The taken edge reaches readout; the untaken edge falls off the stream
        # with an accumulator still live, despite having no represented successor.
        changed = ([jump("producer"), NOP, label("readout")] + operations[-2:]
                   + [("trap", 'kind = "ecall"'), label("producer")]
                   + operations[:-2] + [branch("readout"), NOP])
        machine = raw_cfg(facts, changed)
        # artifact() supplies a final trap; remove it to expose the terminal
        # conditional's implicit stream-end fallthrough rather than a halt block.
        # Retain the earlier halt, which prevents readout from falling through
        # into another execution of the producer.
        original = machine.splitlines()
        trap_lines = [i for i, line in enumerate(original) if '"atlas.trap"' in line]
        del original[trap_lines[-1]]
        self.check("\n".join(original), accepted=False)

    def test_distinct_source_blocks_still_share_physical_slots(self) -> None:
        first, a = chain_fixture()
        second, b = chain_fixture(first_id=5)
        for entry in second:
            entry["block"] = 1
        self.check(cfg(first + second, a + b), accepted=True)
        for reordered in (a[:2] + b[:2] + a[2:] + b[2:],
                          a[:-2] + b[:4] + a[-2:] + b[4:]):
            self.check(cfg(first + second, reordered), accepted=False)
        # A may-live weight at a join remains live even when the other arm
        # consumed all of it. Must-availability alone would erase this obligation.
        changed = a[:2] + [branch("second"), NOP] + a[2:] + [label("second")] + b
        self.check(cfg(first + second, changed), accepted=False)

    def test_loop_cannot_reuse_stale_weight_or_accumulator_versions(self) -> None:
        facts, operations = two_sessions()
        # Loop repeats the first session while the second keeps the physical
        # weight owner id unchanged. The repeated consumer still needs freshness.
        changed = operations[:2] + [label("loop")] + operations[2:6] + [branch("loop"), NOP] + operations[6:]
        self.check(cfg(facts, changed), accepted=False)
        facts, operations = chain_fixture()
        changed = operations[:-2] + [label("pop")] + operations[-2:] + [branch("pop"), NOP]
        self.check(cfg(facts, changed), accepted=False)

    def test_legal_branching_reuse_and_loop_iterations(self) -> None:
        facts, operations = chain_fixture()
        diamond = [branch("right"), NOP, NOP, jump("join"), NOP,
                   label("right"), NOP, label("join")]
        self.check(cfg(facts, operations[:2] + diamond + operations[2:]), accepted=True)
        self.check(cfg(facts, [label("loop")] + operations + [branch("loop"), NOP]), accepted=True)
        facts, operations = two_sessions()
        self.check(cfg(facts, [label("loop")] + operations + [branch("loop"), NOP]), accepted=True)
        first, a = chain_fixture()
        second, b = chain_fixture(first_id=5)
        for entry in second:
            entry["block"] = 1
        # Complete source-local chains in mutually exclusive arms may reuse the
        # same physical slots; either arm leaves them free before the join.
        arms = [branch("right"), NOP] + a + [jump("join"), NOP, label("right")] + b + [label("join")]
        self.check(cfg(first + second, arms), accepted=True)
        self.check(cfg(first + second, [label("loop")] + arms + [branch("loop"), NOP]), accepted=True)

    def test_interleaved_distinct_units_or_slots_are_independent(self) -> None:
        for unit, slot in ((1, 0), (0, 1)):
            first, a = chain_fixture()
            second, b = chain_fixture(unit, slot=slot, first_id=5)
            for entry in second:
                entry["block"] = 1
            if unit == 0:
                second[0]["slot"] = 0
                for entry in second[1:4]:
                    entry["weight_slot"] = 0
                b = [(name, fields.replace("slot = 1 : i32", "slot = 0 : i32", 1) if i == 0 else fields.replace("weight_slot = 1 : i32", "weight_slot = 0 : i32")) for i, (name, fields) in enumerate(b)]
            # Each command keeps its completion delay, which the timed envelope checks.
            interleaved = [op for i in range(0, len(a), 2) for op in a[i:i + 2] + b[i:i + 2]]
            with self.subTest(unit=unit, slot=slot):
                self.check(cfg(first + second, interleaved), accepted=True)


if __name__ == "__main__":
    unittest.main()
