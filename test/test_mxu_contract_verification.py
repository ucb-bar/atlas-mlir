"""Source MXU identities, scales and per-path physical ownership at every machine and LLVM handoff."""

from __future__ import annotations

import re
import unittest

from generated_fixture import BRANCH, JUMP, LABEL, artifact as fixture, branch_to as branch, jump_to as jump, label, record_text
from test_virtual_lowering import lower, virtual_chain
from test_virtual_mxu_extended import seeded_chain
from test_virtual_mxu_handles import chain
from test_virtual_mxu_lowering import source
from verification_support import (
    DELAY, INCOMPLETE, MARKER, NOP, UNMARKED, UNSUPPORTED, assert_boundaries, checked, contract_text, drop_attribute,
    finalize_rejects, line_index, records, replace_contract, rewrite_line, structured_word,
)

CONTRACT = "atlas.virtual_mxu_contract"
TAG = "atlas.virtual_mxu_command"


def record(identity: int, kind: str, unit: int, reg: int, slot: int, *, weight_slot: int = -1,
           weight: int = -1, previous: int = -1, scale_reg: int = -1, scale: int = -1) -> dict:
    return dict(id=identity, block=0, kind=kind, unit=unit, reg=reg, slot=slot,
                weight_slot=weight_slot, weight=weight, previous=previous, scale_reg=scale_reg, scale=scale)


def command(name: str, fields: str, identity: int) -> tuple[str, str]:
    return name, fields + f", {TAG} = {identity} : i32"


def scale(code: int, reg: int = 3, kind: str = "seli") -> tuple[str, str]:
    return "scalar_load", f'kind = "{kind}", dst = {reg} : i32, base = 0 : i32, offset = {code} : i32'


def chain_fixture(unit: int = 0, *, fp8: bool = False, scale_reg: int = 3,
                  slot: int = 0, first_id: int = 0) -> tuple[list[dict], list[tuple[str, str]]]:
    # These fields are hand-authored independently of lowering and the checker.
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


def artifact(facts: list[dict], operations: list[tuple[str, str]]) -> str:
    return fixture(operations, mxu=facts)


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


def raw_cfg(facts: list[dict], operations: list[tuple[str, str]]) -> str:
    """Physical redirects at Atlas byte displacements, without source edges."""
    labels, index = {}, 0
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
            name, fields = (("branch", f'kind = "beq", lhs = 1 : i32, rhs = 0 : i32, offset_bytes = {offset} : i32') if name == BRANCH else
                            ("jump", f'kind = "jal", dst = 0 : i32, base = 0 : i32, offset = {offset} : i32'))
        emitted.append((name, fields))
    return artifact(facts, emitted)


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


class MXUContractVerificationTest(unittest.TestCase):
    def accepted(self, machine: str) -> None:
        assert_boundaries(self, machine)

    def rejected(self, machine: str, diagnostic: str = "MXU contract") -> None:
        assert_boundaries(self, machine, rejects=diagnostic)

    def test_lowering_covers_all_source_families_and_version_dependencies(self) -> None:
        for unit in (0, 1):
            for fmt in ("bf16", "fp8"):
                with self.subTest(unit=unit, fmt=fmt):
                    machine = lower(seeded_chain(unit, fmt, code=173))
                    self.assertIn(MARKER, machine)
                    facts = records(machine, "mxu")
                    self.assertEqual([r["kind"] for r in facts], ["weight_fp8", "acc_" + fmt, "accumulate", "pop_fp8", "acc_fp8", "accumulate", "pop_bf16"])
                    self.assertEqual([r["id"] for r in facts], list(range(7)))
                    self.assertEqual([r["previous"] for r in facts], [-1, -1, 1, 2, -1, 4, 5])
                    self.assertEqual([r["weight"] for r in facts], [-1, -1, 0, -1, -1, 0, -1])
                    self.assertEqual([r["unit"] for r in facts], [unit] * 7)
                    self.assertEqual(facts[3]["scale"], 173)
                    self.assertEqual((facts[6]["scale_reg"], facts[6]["scale"]), (0, -1))
                    self.assertTrue(all(len(r) == 11 for r in facts))
                    self.accepted(machine)
            machine = lower(chain(unit).replace("atlas.output_dram_base = 2415923200", "atlas.output_dram_base = 2415935488"))
            self.assertEqual([r["kind"] for r in records(machine, "mxu")], ["weight_fp8", "reset", "accumulate", "pop_bf16"])
            self.accepted(machine)
            legacy = source(f'    %y = "atlas.virtual_mxu_matmul"(%x, %w) {{unit = {unit} : i32}} : (!atlas.virtual_fp8, !atlas.virtual_fp8) -> !atlas.virtual_bf16', final_state="io2")
            machine = lower(legacy)
            facts = records(machine, "mxu")
            self.assertEqual([r["kind"] for r in facts], ["weight_fp8", "reset", "pop_bf16"])
            self.assertEqual((facts[1]["weight"], facts[2]["previous"]), (0, 1))
            self.accepted(machine)

    def test_legal_command_field_corruptions_fail_all_handoffs(self) -> None:
        def changed(operations, index, old, new):
            result = list(operations)
            name, text = result[index]
            result[index] = name, text.replace(old, new)
            self.assertNotEqual(result, operations)
            return result
        for unit in (0, 1):
            facts, operations = chain_fixture(unit)
            self.accepted(artifact(facts, operations))
            mutations = [(0, "unit", unit, 1 - unit), (0, "src", 11, 12), (0, "slot", 1, 0),
                         (2, "unit", unit, 1 - unit), (2, "src", 9, 10), (2, "weight_slot", 1, 0),
                         (2, "acc_slot", 0, 1), (8, "dst", 40, 42), (8, "slot", 0, 1)]
            for index, field, before, after in mutations:
                with self.subTest(unit=unit, command=index, field=field):
                    self.rejected(artifact(facts, changed(operations, index, f"{field} = {before} : i32", f"{field} = {after} : i32")))
            self.rejected(artifact(facts, changed(operations, 2, "false", "true")))
            self.rejected(artifact(facts, changed(operations, 4, "true", "false")))
            self.rejected(artifact(facts, changed(changed(operations, 8, 'format = "bf16"', 'format = "fp8"'), 8, "dst = 40", "dst = 13")))
        for fmt in ("bf16", "fp8"):
            machine = lower(seeded_chain(0, fmt))
            index = line_index(machine, '"atlas.mxu_push"', f'kind = "acc_{fmt}"')
            other, register = ("fp8", 11) if fmt == "bf16" else ("bf16", 40)
            self.rejected(rewrite_line(machine, index, lambda line: re.sub(r"src = \d+ : i32", f"src = {register} : i32", line.replace(f'kind = "acc_{fmt}"', f'kind = "acc_{other}"'))))

    def test_source_block_ownership_survives_pack_cfg_expansion(self) -> None:
        for unit in (0, 1):
            virtual = chain(unit).replace("atlas.output_dram_base = 2415923200", "atlas.output_dram_base = 2415935488")
            virtual = virtual.replace('    %s1, %a0 = "atlas.virtual_mxu_reset"',
                                      '    %packed = "atlas.virtual_pack_fp8"(%seed) {scale_code = 127 : i32} : (!atlas.virtual_bf16) -> !atlas.virtual_fp8\n    %s1, %a0 = "atlas.virtual_mxu_reset"')
            virtual = virtual.replace("(%s0, %x, %weight)", "(%s0, %packed, %weight)")
            with self.subTest(unit=unit):
                machine = lower(virtual)
                self.assertIn('"atlas.branch"', machine)
                facts = records(machine, "mxu")
                self.assertEqual([r["block"] for r in facts], [0] * 4)
                self.assertEqual((facts[1]["weight"], facts[2]["previous"], facts[3]["previous"]), (0, 1, 2))
                self.accepted(machine)

    def test_commands_require_exactly_one_valid_tag_each(self) -> None:
        facts, operations = chain_fixture()
        for index in (0, 2, 4, 6, 8):
            with self.subTest(missing=index):
                self.rejected(artifact(facts, operations[:index] + operations[index + 2:]))
            with self.subTest(duplicate=index):
                self.rejected(artifact(facts, operations[:index] + operations[index:index + 2] + operations[index:]))
            for value in (None, "999 : i32", "0 : i32", "-1 : i32", "0 : i64", '"bad"'):
                if index == 0 and value == "0 : i32":
                    continue
                with self.subTest(command=index, tag=value):
                    changed = list(operations)
                    name, text = changed[index]
                    changed[index] = name, re.sub(rf", {TAG} = \d+ : i32", "" if value is None else f", {TAG} = {value}", text)
                    self.rejected(artifact(facts, changed))
        self.rejected(artifact(facts, [(*NOP[:1], NOP[1] + f", {TAG} = 0 : i32"), *operations]))
        lines = artifact(facts, operations).splitlines()
        lines[0] = "module {"
        self.rejected("\n".join(lines), UNMARKED)

    def test_identical_physical_continuations_retain_logical_order(self) -> None:
        facts, operations = chain_fixture()
        changed = list(operations)
        changed[4], changed[6] = changed[6], changed[4]
        self.assertEqual(re.sub(rf"{TAG} = \d+", "", operations[4][1]), re.sub(rf"{TAG} = \d+", "", operations[6][1]))
        self.rejected(artifact(facts, changed))

    def test_independent_unit_and_slot_chains_can_reorder_or_interleave(self) -> None:
        first, a = chain_fixture()
        for unit, slot in ((1, 0), (0, 1)):
            second, b = second_block(unit, slot)
            for entry in second:
                entry["block"] = 0
            with self.subTest(unit=unit, slot=slot):
                self.accepted(artifact(first + second, b + a))
            second, b = second_block(unit, slot)
            # Each command keeps its completion delay, which the timed envelope checks.
            interleaved = [op for i in range(0, len(a), 2) for op in a[i:i + 2] + b[i:i + 2]]
            with self.subTest(unit=unit, slot=slot, interleaved=True):
                self.accepted(artifact(first + second, interleaved))

    def test_matching_commands_cannot_overwrite_live_logical_slots(self) -> None:
        first, a = chain_fixture()
        for block in (0, 1):
            second, b = chain_fixture(first_id=5)
            for entry in second:
                entry["block"] = block
            self.accepted(artifact(first + second, a + b))
            for name, reordered in (("weight", a[:2] + b[:2] + a[2:] + b[2:]), ("accumulator", a[:-2] + b[:4] + a[-2:] + b[4:])):
                with self.subTest(block=block, owner=name):
                    self.rejected(artifact(first + second, reordered))
        # A may-live weight at a join remains live even when the other arm
        # consumed all of it. Must-availability alone would erase this obligation.
        self.rejected(artifact(first + second, a[:2] + [branch("second"), NOP] + a[2:] + [label("second")] + b))

    def test_scale_contents_clobber_unknown_restore_and_e0(self) -> None:
        for unit in (0, 1):
            for reg in (0, 3):
                facts, operations = chain_fixture(unit, fp8=True, scale_reg=reg)
                self.accepted(artifact(facts, operations))
                for clobber in (scale(128, reg), scale(0, reg, "seld")):
                    with self.subTest(unit=unit, reg=reg, clobber=clobber):
                        self.rejected(artifact(facts, operations[:-2] + [clobber] + operations[-2:]))
                        restored = artifact(facts, operations[:-2] + [clobber, scale(129, reg)] + operations[-2:])
                        if 'kind = "seld"' in clobber[1]:
                            self.rejected(restored)
                        else:
                            self.accepted(restored)
                self.accepted(artifact(facts, operations[:-2] + [scale(0, reg + 1, "seld")] + operations[-2:]))
                self.rejected(artifact(facts, operations + [scale(0, reg, "seld")]))
                self.rejected(artifact(facts, [op for op in operations if op[0] != "scalar_load"]))
                changed = list(operations)
                changed[-2] = changed[-2][0], changed[-2][1].replace(f"scale_reg = {reg} : i32", f"scale_reg = {reg + 1} : i32")
                self.rejected(artifact(facts, changed))

    def test_scale_cfg_joins_and_loop_fixed_points(self) -> None:
        facts, operations = chain_fixture(fp8=True)
        operations = [op for op in operations if op[0] != "scalar_load"]
        for code in (129, 128):
            diamond = [branch("right"), NOP, scale(129), jump("join"), NOP, label("right"), scale(code), label("join")]
            loop = [scale(129), label("head"), branch("exit"), NOP, scale(code), jump("head"), NOP, label("exit")]
            for shape, prefix in (("diamond", diamond), ("loop", loop)):
                with self.subTest(shape=shape, code=code):
                    if code == 129:
                        self.accepted(artifact(facts, prefix + operations))
                    else:
                        self.rejected(artifact(facts, prefix + operations))

    def test_strict_metadata_geometry_references_and_versions(self) -> None:
        facts, operations = chain_fixture()
        machine = artifact(facts, operations)
        text = contract_text(machine, "mxu")
        replaced = lambda old, new: replace_contract(machine, "mxu", text.replace(old, new, 1))
        mutations = [("missing MXU array", drop_attribute(machine, CONTRACT), INCOMPLETE + " " + CONTRACT),
                     ("missing DMA array", machine.replace("atlas.virtual_dma_contract = [], ", ""), INCOMPLETE + " atlas.virtual_dma_contract"),
                     ("missing version", machine.replace(MARKER + ", ", ""), UNMARKED),
                     ("unit legacy", machine.replace(MARKER, "atlas.generated_from_virtual"), UNSUPPORTED),
                     ("DMA legacy", machine.replace("resource-contract-v4", "dma-contract-v1"), UNSUPPORTED),
                     ("future version", machine.replace("resource-contract-v4", "resource-contract-v999"), UNSUPPORTED),
                     ("bad marker type", machine.replace(MARKER, "atlas.generated_from_virtual = 1 : i32"), UNSUPPORTED),
                     ("nonarray", replace_contract(machine, "mxu", '"bad"'), f"requires an {CONTRACT} array"),
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
            changed = [dict(r) for r in facts]
            changed[index][field] = value
            mutations.append((f"{index}.{field}", replace_contract(machine, "mxu", record_text(changed)), "MXU contract"))
        for name, changed, diagnostic in mutations:
            with self.subTest(mutation=name):
                self.assertNotEqual(changed, machine)
                self.rejected(changed, diagnostic)

    def test_empty_contract_admits_no_mxu_commands(self) -> None:
        machine = lower(virtual_chain(1))
        self.assertIn(MARKER, machine)
        self.assertEqual(contract_text(machine, "mxu"), "[]")
        self.accepted(machine)
        facts, operations = chain_fixture()
        self.rejected(artifact([], [(name, re.sub(rf", {TAG} = \d+ : i32", "", fields)) for name, fields in operations]))

    def test_joint_structured_fields_and_encoded_words_cannot_bypass_contract(self) -> None:
        facts, operations = chain_fixture(fp8=True)
        structured = checked(self, artifact(facts, operations), "--convert-atlas-to-llvm-calls")
        targets = [(('atlas.source_op = "atlas.mxu_matmul"', "accumulate = false"), "src = 9 : i32", "src = 10 : i32", 13, 6, 10),
                   (('atlas.source_op = "atlas.mxu_pop"',), "scale_reg = 3 : i32", "scale_reg = 4 : i32", 13, 6, 4),
                   (('atlas.source_op = "atlas.scalar_load"',), "offset = 129 : i32", "offset = 128 : i32", 20, 12, 128)]
        for needles, before, after, shift, width, field in targets:
            index = line_index(structured, *needles)
            for joint in (False, True):
                with self.subTest(field=before, joint=joint):
                    change = (lambda line: structured_word(line, before, after, shift, width, field)) if joint else (lambda line: line.replace(before, after))
                    changed = rewrite_line(structured, index, change)
                    self.assertEqual(contract_text(changed, "mxu"), contract_text(structured, "mxu"))
                    finalize_rejects(self, changed, "MXU contract" if joint else "")

    def test_structured_finalization_rechecks_metadata_and_command_tags(self) -> None:
        facts, operations = chain_fixture(fp8=True)
        text = checked(self, artifact(facts, operations), "--convert-atlas-to-llvm-calls")
        mutations = [("missing MXU array", drop_attribute(text, CONTRACT), INCOMPLETE + " " + CONTRACT),
                     ("bad MXU array", replace_contract(text, "mxu", '"bad"'), f"requires an {CONTRACT} array"),
                     ("missing DMA array", text.replace("atlas.virtual_dma_contract = [], ", ""), INCOMPLETE + " atlas.virtual_dma_contract"),
                     ("missing version", text.replace(MARKER + ", ", ""), UNMARKED),
                     ("unknown version", text.replace("resource-contract-v4", "resource-contract-v999"), UNSUPPORTED),
                     ("legacy version", text.replace("resource-contract-v4", "dma-contract-v1"), UNSUPPORTED),
                     ("foreign tag", text.replace(f"{TAG} = 0 : i32", f"{TAG} = 999 : i32"), "MXU contract"),
                     ("duplicate tag", text.replace(f"{TAG} = 3 : i32", f"{TAG} = 2 : i32"), "MXU contract"),
                     ("wrong tag type", text.replace(f"{TAG} = 0 : i32", f"{TAG} = 0 : i64"), "MXU contract"),
                     ("missing tag", text.replace(f"{TAG} = 0 : i32, ", ""), "MXU contract")]
        lines = text.splitlines()
        lines[0] = "module attributes {atlas.structured_handoff} {"
        mutations.append(("all resource metadata removed", "\n".join(lines), UNMARKED))
        for name, changed, diagnostic in mutations:
            with self.subTest(mutation=name):
                self.assertNotEqual(changed, text)
                finalize_rejects(self, changed, diagnostic)

    def test_branch_skipping_weight_or_accumulator_producer_fails(self) -> None:
        facts, operations = chain_fixture()
        for producer in (0, 2, 4):
            # Both incoming paths reach the consumer, but one skipped a required
            # producer. All static tags still match and remain in source order.
            changed = (operations[:producer] + [branch("consumer"), NOP] + operations[producer:producer + 2]
                       + [label("consumer")] + operations[producer + 2:])
            with self.subTest(producer=producer):
                self.rejected(artifact(facts, changed))
        facts = [record(0, "weight_fp8", 0, 11, 1), record(1, "acc_fp8", 0, 9, 0),
                 record(2, "accumulate", 0, 9, 0, weight_slot=1, weight=0, previous=1),
                 record(3, "pop_bf16", 0, 40, 0, previous=2, scale_reg=0)]
        operations = [command("mxu_push", 'kind = "weight_fp8", unit = 0 : i32, src = 11 : i32, slot = 1 : i32', 0), DELAY,
                      branch("consumer"), NOP,
                      command("mxu_push", 'kind = "acc_fp8", unit = 0 : i32, src = 9 : i32, slot = 0 : i32', 1), DELAY,
                      label("consumer"), command("mxu_matmul", 'unit = 0 : i32, src = 9 : i32, weight_slot = 1 : i32, acc_slot = 0 : i32, accumulate = true', 2), DELAY,
                      command("mxu_pop", 'format = "bf16", unit = 0 : i32, dst = 40 : i32, slot = 0 : i32, scale_reg = 0 : i32', 3), DELAY]
        self.rejected(artifact(facts, operations))

    def test_live_versions_cannot_cross_a_skipped_readout_or_exit(self) -> None:
        facts, operations = chain_fixture()
        self.rejected(artifact(facts, operations[:-2] + [branch("exit"), NOP] + operations[-2:] + [label("exit")]))
        first, a = chain_fixture()
        second, b = second_block()
        self.rejected(artifact(first + second, a[:-2] + [branch("second"), NOP] + a[-2:] + [label("second")] + b))

    def test_final_conditional_fallthrough_requires_free_accumulator(self) -> None:
        facts, operations = chain_fixture()
        # The taken edge reaches readout; the untaken edge falls off the stream with an
        # accumulator still live. The earlier halt keeps readout from falling into the producer;
        # removing the fixture's final trap exposes the terminal conditional's fallthrough.
        changed = ([jump("producer"), NOP, label("readout")] + operations[-2:] + [("trap", 'kind = "ecall"'), label("producer")]
                   + operations[:-2] + [branch("readout"), NOP])
        lines = raw_cfg(facts, changed).splitlines()
        del lines[[i for i, line in enumerate(lines) if '"atlas.trap"' in line][-1]]
        self.rejected("\n".join(lines))

    def test_loop_cannot_reuse_stale_weight_or_accumulator_versions(self) -> None:
        facts, operations = two_sessions()
        # Loop repeats the first session while the second keeps the physical
        # weight owner id unchanged. The repeated consumer still needs freshness.
        self.rejected(artifact(facts, operations[:2] + [label("loop")] + operations[2:6] + [branch("loop"), NOP] + operations[6:]))
        facts, operations = chain_fixture()
        self.rejected(artifact(facts, operations[:-2] + [label("pop")] + operations[-2:] + [branch("pop"), NOP]))

    def test_legal_branching_reuse_and_loop_iterations(self) -> None:
        facts, operations = chain_fixture()
        diamond = [branch("right"), NOP, NOP, jump("join"), NOP, label("right"), NOP, label("join")]
        self.accepted(artifact(facts, operations[:2] + diamond + operations[2:]))
        self.accepted(artifact(facts, [label("loop")] + operations + [branch("loop"), NOP]))
        facts, operations = two_sessions()
        self.accepted(artifact(facts, [label("loop")] + operations + [branch("loop"), NOP]))
        first, a = chain_fixture()
        second, b = second_block()
        # Complete source-local chains in mutually exclusive arms may reuse the
        # same physical slots; either arm leaves them free before the join.
        arms = [branch("right"), NOP] + a + [jump("join"), NOP, label("right")] + b + [label("join")]
        self.accepted(artifact(first + second, arms))
        self.accepted(artifact(first + second, [label("loop")] + arms + [branch("loop"), NOP]))


if __name__ == "__main__":
    unittest.main()
