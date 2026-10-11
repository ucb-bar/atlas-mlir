"""Source MXU identities, scales and per-path physical ownership at every machine and LLVM handoff."""

from __future__ import annotations

import re
import unittest

from generated_fixture import artifact, branch_to as branch, jump_to as jump, label, record_text
from test_virtual_lowering import lower, virtual_chain
from test_virtual_mxu_extended import seeded_chain
from test_virtual_mxu_handles import chain
from verification_support import (
    DELAY, NOP, TRAP, UNMARKED, BoundaryChecks, assert_boundaries, checked, contract_text, finalize_rejects, line_index, records,
    replace_contract, rewrite_line, structured_word,
)

CONTRACT = "atlas.virtual_mxu_contract"
TAG = "atlas.virtual_mxu_command"
OUTPUT = ("atlas.output_dram_base = 2415923200", "atlas.output_dram_base = 2415935488")


def record(identity: int, kind: str, unit: int, reg: int, slot: int, *, weight_slot: int = -1,
           weight: int = -1, previous: int = -1, scale_reg: int = -1, scale: int = -1) -> dict:
    return dict(id=identity, block=0, kind=kind, unit=unit, reg=reg, slot=slot,
                weight_slot=weight_slot, weight=weight, previous=previous, scale_reg=scale_reg, scale=scale)


def command(name: str, fields: str, identity: int) -> tuple[str, str]:
    return name, fields + f", {TAG} = {identity} : i32"


def scale(code: int, reg: int = 3, kind: str = "seli") -> tuple[str, str]:
    return "scalar_load", f'kind = "{kind}", dst = {reg} : i32, base = 0 : i32, offset = {code} : i32'


def changed(operations: list, index: int, old: str, new: str) -> list:
    name, text = operations[index]
    assert old in text, (old, text)
    return operations[:index] + [(name, text.replace(old, new))] + operations[index + 1:]


def chain_fixture(unit: int = 0, *, fp8: bool = False, scale_reg: int = 3,
                  slot: int = 0, first_id: int = 0) -> tuple[list[dict], list[tuple[str, str]]]:
    i = first_id
    facts = [record(i, "weight_fp8", unit, 11, 1),
             record(i + 1, "reset", unit, 9, slot, weight_slot=1, weight=i),
             record(i + 2, "accumulate", unit, 9, slot, weight_slot=1, weight=i, previous=i + 1),
             record(i + 3, "accumulate", unit, 9, slot, weight_slot=1, weight=i, previous=i + 2),
             record(i + 4, "pop_fp8" if fp8 else "pop_bf16", unit, 13 if fp8 else 40, slot,
                    previous=i + 3, scale_reg=scale_reg if fp8 else 0, scale=129 if fp8 else -1)]
    operations = [command("mxu_push", f'kind = "weight_fp8", unit = {unit} : i32, src = 11 : i32, slot = 1 : i32', i), DELAY]
    for offset, flag in ((1, "false"), (2, "true"), (3, "true")):
        operations += [command("mxu_matmul", f'unit = {unit} : i32, src = 9 : i32, weight_slot = 1 : i32, acc_slot = {slot} : i32, accumulate = {flag}', i + offset), DELAY]
    if fp8:
        operations.append(scale(129, scale_reg))
    operations += [command("mxu_pop", f'format = "{"fp8" if fp8 else "bf16"}", unit = {unit} : i32, dst = {13 if fp8 else 40} : i32, slot = {slot} : i32, scale_reg = {scale_reg if fp8 else 0} : i32', i + 4), DELAY]
    return facts, operations


def second_block(unit: int = 0, slot: int = 0) -> tuple[list[dict], list[tuple[str, str]]]:
    """A second chain (ids 5..9) owned by source block 1; on unit 0 slot 1 it uses weight slot 0."""
    facts, operations = chain_fixture(unit, slot=slot, first_id=5)
    for entry in facts:
        entry["block"] = 1
    if (unit, slot) == (0, 1):
        facts[0]["slot"] = 0
        for entry in facts[1:4]:
            entry["weight_slot"] = 0
        operations = [(name, fields.replace("slot = 1 : i32", "slot = 0 : i32", 1) if i == 0 else fields.replace("weight_slot = 1 : i32", "weight_slot = 0 : i32"))
                      for i, (name, fields) in enumerate(operations)]
    return facts, operations


def two_sessions() -> tuple[list[dict], list[tuple[str, str]]]:
    # Two reset/pop sessions share a weight. Each dynamic reset needs a fresh
    # execution of its weight producer, even while the other session is pending.
    facts = [record(0, "weight_fp8", 0, 11, 1)]
    operations = [command("mxu_push", 'kind = "weight_fp8", unit = 0 : i32, src = 11 : i32, slot = 1 : i32', 0), DELAY]
    for reset, pop in ((1, 2), (3, 4)):
        facts += [record(reset, "reset", 0, 9, 0, weight_slot=1, weight=0), record(pop, "pop_bf16", 0, 40, 0, previous=reset, scale_reg=0)]
        operations += [command("mxu_matmul", 'unit = 0 : i32, src = 9 : i32, weight_slot = 1 : i32, acc_slot = 0 : i32, accumulate = false', reset), DELAY,
                       command("mxu_pop", 'format = "bf16", unit = 0 : i32, dst = 40 : i32, slot = 0 : i32, scale_reg = 0 : i32', pop), DELAY]
    return facts, operations


class MXUContractVerificationTest(BoundaryChecks, unittest.TestCase):
    REJECTS = "MXU contract"

    def test_legal_command_field_corruptions_fail_all_handoffs(self) -> None:
        for unit in (0, 1):
            facts, operations = chain_fixture(unit)
            self.accepted(artifact(operations, mxu=facts))
            mutations = [(index, f"{field} = {before} : i32", f"{field} = {after} : i32")
                         for index, field, before, after in ((0, "unit", unit, 1 - unit), (0, "src", 11, 12), (0, "slot", 1, 0),
                                                             (2, "unit", unit, 1 - unit), (2, "src", 9, 10), (2, "weight_slot", 1, 0),
                                                             (2, "acc_slot", 0, 1), (8, "dst", 40, 42), (8, "slot", 0, 1))]
            mutations += [(2, "false", "true"), (4, "true", "false")]
            for index, old, new in mutations:
                with self.subTest(unit=unit, command=index, field=old):
                    self.rejected(artifact(changed(operations, index, old, new), mxu=facts))
            self.rejected(artifact(changed(changed(operations, 8, 'format = "bf16"', 'format = "fp8"'), 8, "dst = 40", "dst = 13"), mxu=facts))
        for fmt, other, register in (("bf16", "fp8", 11), ("fp8", "bf16", 40)):
            machine = lower(seeded_chain(0, fmt))
            index = line_index(machine, '"atlas.mxu_push"', f'kind = "acc_{fmt}"')
            self.rejected(rewrite_line(machine, index, lambda line: re.sub(r"src = \d+ : i32", f"src = {register} : i32", line.replace(f'kind = "acc_{fmt}"', f'kind = "acc_{other}"'))))

    def test_source_block_ownership_survives_pack_cfg_expansion(self) -> None:
        for unit in (0, 1):
            virtual = chain(unit).replace(*OUTPUT).replace(
                '    %s1, %a0 = "atlas.virtual_mxu_reset"',
                '    %packed = "atlas.virtual_pack_fp8"(%seed) {scale_code = 127 : i32} : (!atlas.virtual_bf16) -> !atlas.virtual_fp8\n    %s1, %a0 = "atlas.virtual_mxu_reset"')
            with self.subTest(unit=unit):
                machine = lower(virtual.replace("(%s0, %x, %weight)", "(%s0, %packed, %weight)"))
                self.assertIn('"atlas.branch"', machine)
                facts = records(machine, "mxu")
                self.assertEqual([r["block"] for r in facts], [0] * 4)
                self.assertEqual((facts[1]["weight"], facts[2]["previous"], facts[3]["previous"]), (0, 1, 2))
                self.accepted(machine)

    def test_commands_require_exactly_one_valid_tag_each(self) -> None:
        facts, operations = chain_fixture()
        for index in (0, 2, 4, 6, 8):
            with self.subTest(missing=index):
                self.rejected(artifact(operations[:index] + operations[index + 2:], mxu=facts))
            with self.subTest(duplicate=index):
                self.rejected(artifact(operations[:index] + operations[index:index + 2] + operations[index:], mxu=facts))
            for value in (None, "999 : i32", "0 : i32", "-1 : i32", "0 : i64", '"bad"'):
                if index == 0 and value == "0 : i32":
                    continue
                with self.subTest(command=index, tag=value):
                    name, text = operations[index]
                    tagged = (name, re.sub(rf", {TAG} = \d+ : i32", "" if value is None else f", {TAG} = {value}", text))
                    self.rejected(artifact(operations[:index] + [tagged] + operations[index + 1:], mxu=facts))
        self.rejected(artifact([(*NOP[:1], NOP[1] + f", {TAG} = 0 : i32"), *operations], mxu=facts))
        # Identical physical continuations retain their logical order.
        self.assertEqual(re.sub(rf"{TAG} = \d+", "", operations[4][1]), re.sub(rf"{TAG} = \d+", "", operations[6][1]))
        self.rejected(artifact(operations[:4] + operations[6:8] + operations[4:6] + operations[8:], mxu=facts))

    def test_independent_unit_and_slot_chains_can_reorder_or_interleave(self) -> None:
        first, a = chain_fixture()
        for unit, slot in ((1, 0), (0, 1)):
            second, b = second_block(unit, slot)
            # Each command keeps its completion delay, which the timed envelope checks.
            interleaved = [op for i in range(0, len(a), 2) for op in a[i:i + 2] + b[i:i + 2]]
            with self.subTest(unit=unit, slot=slot, interleaved=True):
                self.accepted(artifact(interleaved, mxu=first + second))
            for entry in second:
                entry["block"] = 0
            with self.subTest(unit=unit, slot=slot):
                self.accepted(artifact(b + a, mxu=first + second))

    def test_matching_commands_cannot_overwrite_live_logical_slots(self) -> None:
        first, a = chain_fixture()
        for block in (0, 1):
            second, b = chain_fixture(first_id=5)
            for entry in second:
                entry["block"] = block
            self.accepted(artifact(a + b, mxu=first + second))
            for name, reordered in (("weight", a[:2] + b[:2] + a[2:] + b[2:]), ("accumulator", a[:-2] + b[:4] + a[-2:] + b[4:])):
                with self.subTest(block=block, owner=name):
                    self.rejected(artifact(reordered, mxu=first + second))
        # A may-live weight at a join remains live even when the other arm
        # consumed all of it. Must-availability alone would erase this obligation.
        self.rejected(artifact(a[:2] + [branch("second"), NOP] + a[2:] + [label("second")] + b, mxu=first + second))

    def test_scale_contents_clobber_unknown_restore_and_e0(self) -> None:
        for unit in (0, 1):
            for reg in (0, 3):
                facts, operations = chain_fixture(unit, fp8=True, scale_reg=reg)
                self.accepted(artifact(operations, mxu=facts))
                for clobber in (scale(128, reg), scale(0, reg, "seld")):
                    with self.subTest(unit=unit, reg=reg, clobber=clobber):
                        self.rejected(artifact(operations[:-2] + [clobber] + operations[-2:], mxu=facts))
                        restored = artifact(operations[:-2] + [clobber, scale(129, reg)] + operations[-2:], mxu=facts)
                        assert_boundaries(self, restored, rejects="MXU contract" if 'kind = "seld"' in clobber[1] else None)
                self.accepted(artifact(operations[:-2] + [scale(0, reg + 1, "seld")] + operations[-2:], mxu=facts))
                self.rejected(artifact(operations + [scale(0, reg, "seld")], mxu=facts))
                self.rejected(artifact([op for op in operations if op[0] != "scalar_load"], mxu=facts))
                self.rejected(artifact(changed(operations, -2, f"scale_reg = {reg} : i32", f"scale_reg = {reg + 1} : i32"), mxu=facts))

    def test_scale_cfg_joins_and_loop_fixed_points(self) -> None:
        facts, operations = chain_fixture(fp8=True)
        operations = [op for op in operations if op[0] != "scalar_load"]
        for code in (129, 128):
            diamond = [branch("right"), NOP, scale(129), jump("join"), NOP, label("right"), scale(code), label("join")]
            loop = [scale(129), label("head"), branch("exit"), NOP, scale(code), jump("head"), NOP, label("exit")]
            for shape, prefix in (("diamond", diamond), ("loop", loop)):
                with self.subTest(shape=shape, code=code):
                    assert_boundaries(self, artifact(prefix + operations, mxu=facts), rejects=None if code == 129 else "MXU contract")

    def test_strict_metadata_geometry_and_references(self) -> None:
        facts, operations = chain_fixture()
        machine = artifact(operations, mxu=facts)
        text = contract_text(machine, "mxu")
        replaced = lambda old, new: replace_contract(machine, "mxu", text.replace(old, new, 1))
        mutations = [("nonarray", replace_contract(machine, "mxu", '"bad"'), f"requires an {CONTRACT} array"),
                     ("nondictionary", replace_contract(machine, "mxu", "[0 : i32]"), "MXU contract"),
                     ("empty with commands", replace_contract(machine, "mxu", "[]"), "MXU contract"),
                     ("foreign field", replaced("{", "{foreign = 0 : i32, "), "MXU contract"),
                     ("missing field", replaced("block = 0 : i32, ", ""), "MXU contract"),
                     ("wrong numeric type", replaced("reg = 11 : i32", "reg = 11 : i64"), "MXU contract"),
                     ("wrong kind type", replaced('kind = "reset"', "kind = 0 : i32"), "MXU contract"),
                     ("unsorted", replace_contract(machine, "mxu", record_text(list(reversed(facts)))), "MXU contract")]
        for index, field, value in ((0, "id", 1), (0, "unit", 2), (0, "slot", 2), (0, "kind", "foreign"),
                                    (0, "scale", 0), (1, "weight", 1), (1, "weight_slot", 0),
                                    (2, "previous", 0), (2, "block", 1), (4, "reg", 41)):
            entries = [dict(r) for r in facts]
            entries[index][field] = value
            mutations.append((f"{index}.{field}", replace_contract(machine, "mxu", record_text(entries)), "MXU contract"))
        for name, mutated, diagnostic in mutations:
            with self.subTest(mutation=name):
                self.assertNotEqual(mutated, machine)
                self.rejected(mutated, diagnostic)

    def test_empty_contract_admits_no_mxu_commands(self) -> None:
        machine = lower(virtual_chain(1))
        self.assertEqual(contract_text(machine, "mxu"), "[]")
        self.accepted(machine)
        facts, operations = chain_fixture()
        self.rejected(artifact([(name, re.sub(rf", {TAG} = \d+ : i32", "", fields)) for name, fields in operations], mxu=[]))

    def test_joint_structured_fields_and_encoded_words_cannot_bypass_contract(self) -> None:
        facts, operations = chain_fixture(fp8=True)
        structured = checked(self, artifact(operations, mxu=facts), "--convert-atlas-to-llvm-calls")
        targets = [(('atlas.source_op = "atlas.mxu_matmul"', "accumulate = false"), "src = 9 : i32", "src = 10 : i32", 13, 6, 10),
                   (('atlas.source_op = "atlas.mxu_pop"',), "scale_reg = 3 : i32", "scale_reg = 4 : i32", 13, 6, 4),
                   (('atlas.source_op = "atlas.scalar_load"',), "offset = 129 : i32", "offset = 128 : i32", 20, 12, 128)]
        for needles, before, after, shift, width, field in targets:
            index = line_index(structured, *needles)
            for joint in (False, True):
                with self.subTest(field=before, joint=joint):
                    change = (lambda line: structured_word(line, before, after, shift, width, field)) if joint else (lambda line: line.replace(before, after))
                    mutated = rewrite_line(structured, index, change)
                    self.assertEqual(contract_text(mutated, "mxu"), contract_text(structured, "mxu"))
                    finalize_rejects(self, mutated, "MXU contract" if joint else "")

    def test_structured_finalization_rechecks_metadata_and_command_tags(self) -> None:
        facts, operations = chain_fixture(fp8=True)
        text = checked(self, artifact(operations, mxu=facts), "--convert-atlas-to-llvm-calls")
        header, body = text.split("\n", 1)
        mutations = [("bad MXU array", replace_contract(text, "mxu", '"bad"'), f"requires an {CONTRACT} array"),
                     ("foreign tag", text.replace(f"{TAG} = 0 : i32", f"{TAG} = 999 : i32"), "MXU contract"),
                     ("duplicate tag", text.replace(f"{TAG} = 3 : i32", f"{TAG} = 2 : i32"), "MXU contract"),
                     ("wrong tag type", text.replace(f"{TAG} = 0 : i32", f"{TAG} = 0 : i64"), "MXU contract"),
                     ("missing tag", text.replace(f"{TAG} = 0 : i32, ", ""), "MXU contract"),
                     ("all resource metadata removed", "module attributes {atlas.structured_handoff} {\n" + body, UNMARKED)]
        for name, mutated, diagnostic in mutations:
            with self.subTest(mutation=name):
                self.assertNotEqual(mutated, text)
                finalize_rejects(self, mutated, diagnostic)

    def test_emitted_paths_cannot_skip_producers_readouts_or_freshness(self) -> None:
        facts, operations = chain_fixture()
        first, a = chain_fixture()
        second, b = second_block()
        sessions, ops = two_sessions()
        # Each skip keeps every static tag matching and in source order.
        cases = [(facts, operations[:producer] + [branch("consumer"), NOP] + operations[producer:producer + 2] + [label("consumer")]
                  + operations[producer + 2:]) for producer in (0, 2, 4)]
        cases.append(([record(0, "weight_fp8", 0, 11, 1), record(1, "acc_fp8", 0, 9, 0),
                       record(2, "accumulate", 0, 9, 0, weight_slot=1, weight=0, previous=1), record(3, "pop_bf16", 0, 40, 0, previous=2, scale_reg=0)],
                      [command("mxu_push", 'kind = "weight_fp8", unit = 0 : i32, src = 11 : i32, slot = 1 : i32', 0), DELAY, branch("consumer"), NOP,
                       command("mxu_push", 'kind = "acc_fp8", unit = 0 : i32, src = 9 : i32, slot = 0 : i32', 1), DELAY, label("consumer"),
                       command("mxu_matmul", 'unit = 0 : i32, src = 9 : i32, weight_slot = 1 : i32, acc_slot = 0 : i32, accumulate = true', 2), DELAY,
                       command("mxu_pop", 'format = "bf16", unit = 0 : i32, dst = 40 : i32, slot = 0 : i32, scale_reg = 0 : i32', 3), DELAY]))
        cases += [(facts, operations[:-2] + [branch("exit"), NOP] + operations[-2:] + [label("exit")]),
                  (first + second, a[:-2] + [branch("second"), NOP] + a[-2:] + [label("second")] + b),
                  # The loop repeats the first session while the second keeps the weight owner id unchanged.
                  (sessions, ops[:2] + [label("loop")] + ops[2:6] + [branch("loop"), NOP] + ops[6:]),
                  (facts, operations[:-2] + [label("pop")] + operations[-2:] + [branch("pop"), NOP])]
        for index, (records_, stream) in enumerate(cases):
            with self.subTest(case=index):
                self.rejected(artifact(stream, mxu=records_))

    def test_final_conditional_fallthrough_requires_free_accumulator(self) -> None:
        facts, operations = chain_fixture()
        # The taken edge reaches readout; the untaken edge falls off the stream with a live accumulator. Raw redirects
        # carry no source edges; the halt keeps readout from falling into the producer.
        stream = [("jump", 'kind = "jal", dst = 0 : i32, base = 0 : i32, offset = 10 : i32'), NOP, *operations[-2:], TRAP,
                  *operations[:-2], ("branch", 'kind = "beq", lhs = 1 : i32, rhs = 0 : i32, offset_bytes = -22 : i32'), NOP]
        lines = artifact(stream, mxu=facts).splitlines()
        del lines[[i for i, line in enumerate(lines) if '"atlas.trap"' in line][-1]]
        self.rejected("\n".join(lines))

    def test_legal_branching_reuse_and_loop_iterations(self) -> None:
        facts, operations = chain_fixture()
        diamond = [branch("right"), NOP, NOP, jump("join"), NOP, label("right"), NOP, label("join")]
        self.accepted(artifact(operations[:2] + diamond + operations[2:], mxu=facts))
        self.accepted(artifact([label("loop")] + operations + [branch("loop"), NOP], mxu=facts))
        facts, operations = two_sessions()
        self.accepted(artifact([label("loop")] + operations + [branch("loop"), NOP], mxu=facts))
        first, a = chain_fixture()
        second, b = second_block()
        # Complete source-local chains in mutually exclusive arms may reuse the
        # same physical slots; either arm leaves them free before the join.
        arms = [branch("right"), NOP] + a + [jump("join"), NOP, label("right")] + b + [label("join")]
        self.accepted(artifact(arms, mxu=first + second))
        self.accepted(artifact([label("loop")] + arms + [branch("loop"), NOP], mxu=first + second))


if __name__ == "__main__":
    unittest.main()
