"""Source MXU identities and scales survive issued commands and LLVM handoffs."""

from __future__ import annotations

import re
import unittest

from test_virtual_lowering import BIN, lower, run, virtual_chain
from test_virtual_mxu_extended import seeded_chain
from test_virtual_mxu_handles import chain
from test_virtual_mxu_lowering import source


CONTRACT = "atlas.virtual_mxu_contract"
TAG = "atlas.virtual_mxu_command"
VERSION = 'atlas.generated_from_virtual = "resource-contract-v1"'
LOWERED_VERSION = 'atlas.generated_from_virtual = "resource-contract-v3"'
CONTRACT_RE = re.compile(r'atlas\.virtual_mxu_contract = (\[[^\]]*\])')
RECORD_RE = re.compile(r'\{([^{}]*)\}')
FIELD_RE = re.compile(r'(\w+) = (?:(-?\d+) : i32|"([^"]*)")')
BOUNDARIES = (("atlas-opt", ("--verify-atlas-generated-schedule",)),
              ("atlas-emit", ()),
              ("atlas-opt", ("--convert-atlas-to-llvm",)),
              ("atlas-opt", ("--convert-atlas-to-llvm-calls",)))
STATE = "!atlas.state"
NOP = ("alu_imm", 'kind = "addi", dst = 0 : i32, src = 0 : i32, immediate = 0 : i32')
DELAY = ("delay", 'cycles = 256 : i32, atlas.delay_reason = "tensor_completion"')


def contract_text(machine: str) -> str:
    match = CONTRACT_RE.search(machine)
    if match is None:
        raise AssertionError("lowering must retain the source MXU contract")
    return match[1]


def records(machine: str) -> list[dict]:
    return [{name: int(integer) if integer else string
             for name, integer, string in FIELD_RE.findall(record)}
            for record in RECORD_RE.findall(contract_text(machine))]


def replace_contract(machine: str, value: str) -> str:
    return CONTRACT_RE.sub(lambda _: f"{CONTRACT} = {value}", machine, count=1)


def replace_line(machine: str, index: int, line: str) -> str:
    lines = machine.splitlines()
    lines[index] = line
    return "\n".join(lines) + "\n"


def record(identity: int, kind: str, unit: int, reg: int, slot: int, *, weight_slot: int = -1,
           weight: int = -1, previous: int = -1, scale_reg: int = -1, scale: int = -1) -> dict:
    return dict(id=identity, block=0, kind=kind, unit=unit, reg=reg, slot=slot,
                weight_slot=weight_slot, weight=weight, previous=previous, scale_reg=scale_reg, scale=scale)


def record_text(entries: list[dict]) -> str:
    return "[" + ", ".join("{" + ", ".join(
        f'{key} = "{value}"' if isinstance(value, str) else f"{key} = {value} : i32"
        for key, value in entry.items()) + "}" for entry in entries) + "]"


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
    operations = [*operations, ("trap", 'kind = "ecall"')]
    lines = [f'%s0 = "atlas.start"() : () -> {STATE}']
    lines += [f'%s{i + 1} = "atlas.{name}"(%s{i}) {{{fields}}} : ({STATE}) -> {STATE}'
              for i, (name, fields) in enumerate(operations)]
    return f"module attributes {{{VERSION}, atlas.virtual_dma_contract = [], {CONTRACT} = {record_text(facts)}}} {{\n" + "\n".join(lines) + "\n}"


class MXUContractVerificationTest(unittest.TestCase):
    def setUp(self) -> None:
        self.assertTrue((BIN / "atlas-opt").is_file(), "build atlas-opt first")
        self.assertTrue((BIN / "atlas-emit").is_file(), "build atlas-emit first")

    def accepted(self, machine: str) -> None:
        for tool, options in BOUNDARIES:
            with self.subTest(tool=tool, options=options):
                result = run(tool, machine, *options)
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertTrue(result.stdout)

    def rejected(self, machine: str, diagnostic: str = "") -> None:
        for tool, options in BOUNDARIES:
            with self.subTest(tool=tool, options=options):
                result = run(tool, machine, *options)
                self.assertNotEqual(result.returncode, 0, result.stdout)
                self.assertEqual(result.stdout, "")
                self.assertTrue(result.stderr)
                if diagnostic:
                    self.assertIn(diagnostic, result.stderr)

    def test_lowering_covers_all_source_families_and_version_dependencies(self) -> None:
        for unit in (0, 1):
            for fmt in ("bf16", "fp8"):
                with self.subTest(unit=unit, fmt=fmt):
                    machine = lower(seeded_chain(unit, fmt, code=173))
                    self.assertIn(LOWERED_VERSION, machine)
                    facts = records(machine)
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
            self.assertEqual([r["kind"] for r in records(machine)], ["weight_fp8", "reset", "accumulate", "pop_bf16"])
            self.accepted(machine)
            legacy = source(f'    %y = "atlas.virtual_mxu_matmul"(%x, %w) {{unit = {unit} : i32}} : (!atlas.virtual_fp8, !atlas.virtual_fp8) -> !atlas.virtual_bf16', final_state="io2")
            machine = lower(legacy)
            facts = records(machine)
            self.assertEqual([r["kind"] for r in facts], ["weight_fp8", "reset", "pop_bf16"])
            self.assertEqual((facts[1]["weight"], facts[2]["previous"]), (0, 1))
            self.accepted(machine)

    def test_legal_command_field_corruptions_fail_all_handoffs(self) -> None:
        for unit in (0, 1):
            facts, operations = chain_fixture(unit)
            self.accepted(artifact(facts, operations))
            mutations = [(0, "unit", unit, 1 - unit), (0, "src", 11, 12), (0, "slot", 1, 0),
                         (2, "unit", unit, 1 - unit), (2, "src", 9, 10), (2, "weight_slot", 1, 0),
                         (2, "acc_slot", 0, 1), (8, "dst", 40, 42), (8, "slot", 0, 1)]
            for index, field, before, after in mutations:
                with self.subTest(unit=unit, command=index, field=field):
                    changed = list(operations)
                    name, text = changed[index]
                    changed[index] = name, text.replace(f"{field} = {before} : i32", f"{field} = {after} : i32")
                    self.assertNotEqual(changed, operations)
                    self.rejected(artifact(facts, changed), "MXU contract")
            for index in (2, 4):
                changed = list(operations)
                name, text = changed[index]
                changed[index] = name, text.replace("false", "true") if index == 2 else text.replace("true", "false")
                self.rejected(artifact(facts, changed), "MXU contract")
            changed = list(operations)
            name, text = changed[8]
            changed[8] = name, text.replace('format = "bf16"', 'format = "fp8"').replace("dst = 40", "dst = 13")
            self.rejected(artifact(facts, changed), "MXU contract")
        for fmt in ("bf16", "fp8"):
            machine = lower(seeded_chain(0, fmt))
            lines = machine.splitlines()
            index = next(i for i, line in enumerate(lines) if '"atlas.mxu_push"' in line and f'kind = "acc_{fmt}"' in line)
            changed = lines[index].replace(f'kind = "acc_{fmt}"', f'kind = "acc_{"fp8" if fmt == "bf16" else "bf16"}"')
            changed = re.sub(r'src = \d+ : i32', f'src = {11 if fmt == "bf16" else 40} : i32', changed)
            self.rejected(replace_line(machine, index, changed), "MXU contract")

    def test_source_block_ownership_survives_pack_cfg_expansion(self) -> None:
        for unit in (0, 1):
            virtual = chain(unit).replace("atlas.output_dram_base = 2415923200", "atlas.output_dram_base = 2415935488")
            virtual = virtual.replace('    %s1, %a0 = "atlas.virtual_mxu_reset"',
                '    %packed = "atlas.virtual_pack_fp8"(%seed) {scale_code = 127 : i32} : (!atlas.virtual_bf16) -> !atlas.virtual_fp8\n    %s1, %a0 = "atlas.virtual_mxu_reset"')
            virtual = virtual.replace("(%s0, %x, %weight)", "(%s0, %packed, %weight)")
            with self.subTest(unit=unit):
                machine = lower(virtual)
                self.assertIn('"atlas.branch"', machine)
                facts = records(machine)
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
                    text = re.sub(rf', {TAG} = \d+ : i32', "" if value is None else f", {TAG} = {value}", text)
                    changed[index] = name, text
                    self.rejected(artifact(facts, changed))
        self.rejected(artifact(facts, [(*NOP[:1], NOP[1] + f", {TAG} = 0 : i32"), *operations]))
        machine = artifact(facts, operations)
        lines = machine.splitlines()
        lines[0] = "module {"
        self.rejected("\n".join(lines), "virtual-to-machine artifact")
        for marker in ("atlas.generated_from_virtual", 'atlas.generated_from_virtual = "dma-contract-v1"'):
            changed = CONTRACT_RE.sub("", machine).replace(", }", "}").replace(VERSION, marker)
            if marker == "atlas.generated_from_virtual":
                changed = changed.replace(", atlas.virtual_dma_contract = []", "")
            self.rejected(changed)

    def test_identical_physical_continuations_retain_logical_order(self) -> None:
        facts, operations = chain_fixture()
        changed = list(operations)
        changed[4], changed[6] = changed[6], changed[4]
        self.assertEqual(re.sub(rf'{TAG} = \d+', "", operations[4][1]), re.sub(rf'{TAG} = \d+', "", operations[6][1]))
        self.rejected(artifact(facts, changed), "MXU contract")

    def test_independent_unit_and_slot_chains_can_reorder(self) -> None:
        for second_unit, second_slot in ((1, 0), (0, 1)):
            first, a = chain_fixture()
            second, b = chain_fixture(second_unit, slot=second_slot, first_id=5)
            if second_unit == 0:
                # Distinct weight slots are independent too.
                second[0]["slot"] = 0
                for entry in second[1:4]:
                    entry["weight_slot"] = 0
                b = [(name, fields.replace("slot = 1 : i32", "slot = 0 : i32", 1) if index == 0 else fields.replace("weight_slot = 1 : i32", "weight_slot = 0 : i32")) for index, (name, fields) in enumerate(b)]
            with self.subTest(unit=second_unit, slot=second_slot):
                self.accepted(artifact(first + second, b + a))

    def test_matching_commands_cannot_overwrite_live_logical_slots(self) -> None:
        first, a = chain_fixture()
        second, b = chain_fixture(first_id=5)
        self.accepted(artifact(first + second, a + b))
        for name, reordered in (("weight", a[:2] + b[:2] + a[2:] + b[2:]),
                                ("accumulator", a[:-2] + b[:4] + a[-2:] + b[4:])):
            with self.subTest(owner=name):
                self.rejected(artifact(first + second, reordered), "MXU contract")

    def test_scale_contents_clobber_unknown_restore_and_e0(self) -> None:
        for unit in (0, 1):
            for reg in (0, 3):
                facts, operations = chain_fixture(unit, fp8=True, scale_reg=reg)
                self.accepted(artifact(facts, operations))
                for clobber in (scale(128, reg), scale(0, reg, "seld")):
                    with self.subTest(unit=unit, reg=reg, clobber=clobber):
                        self.rejected(artifact(facts, operations[:-2] + [clobber] + operations[-2:]), "MXU contract")
                        restored = artifact(facts, operations[:-2] + [clobber, scale(129, reg)] + operations[-2:])
                        if 'kind = "seld"' in clobber[1]:
                            self.rejected(restored, "MXU contract")
                        else:
                            self.accepted(restored)
                self.accepted(artifact(facts, operations[:-2] + [scale(0, reg + 1, "seld")] + operations[-2:]))
                self.rejected(artifact(facts, operations + [scale(0, reg, "seld")]), "MXU contract")
                self.rejected(artifact(facts, [op for op in operations if op[0] != "scalar_load"]), "MXU contract")
                changed = list(operations)
                changed[-2] = changed[-2][0], changed[-2][1].replace(f"scale_reg = {reg} : i32", f"scale_reg = {reg + 1} : i32")
                self.rejected(artifact(facts, changed), "MXU contract")

    def test_scale_cfg_joins_and_loop_fixed_points(self) -> None:
        facts, operations = chain_fixture(fp8=True)
        operations = [op for op in operations if op[0] != "scalar_load"]
        for code in (129, 128):
            diamond = [("branch", 'kind = "beq", lhs = 1 : i32, rhs = 0 : i32, offset_bytes = 10 : i32'), NOP,
                       scale(129), ("jump", 'kind = "jal", dst = 0 : i32, base = 0 : i32, offset = 6 : i32'), NOP, scale(code)]
            loop = [scale(129), ("branch", 'kind = "beq", lhs = 1 : i32, rhs = 0 : i32, offset_bytes = 10 : i32'), NOP,
                    scale(code), ("jump", 'kind = "jal", dst = 0 : i32, base = 0 : i32, offset = -6 : i32'), NOP]
            for shape, prefix in (("diamond", diamond), ("loop", loop)):
                with self.subTest(shape=shape, code=code):
                    if code == 129:
                        self.accepted(artifact(facts, prefix + operations))
                    else:
                        self.rejected(artifact(facts, prefix + operations), "MXU contract")

    def test_strict_metadata_geometry_references_and_versions(self) -> None:
        facts, operations = chain_fixture()
        machine = artifact(facts, operations)
        text = contract_text(machine)
        mutations = [("missing MXU array", CONTRACT_RE.sub("", machine).replace(", }", "}")),
                     ("missing DMA array", machine.replace("atlas.virtual_dma_contract = [], ", "")),
                     ("missing version", machine.replace(VERSION + ", ", "")),
                     ("unit legacy with MXU", machine.replace(VERSION, "atlas.generated_from_virtual")),
                     ("DMA legacy with MXU", machine.replace("resource-contract-v1", "dma-contract-v1")),
                     ("future version", machine.replace("resource-contract-v1", "resource-contract-v999")),
                     ("bad marker type", machine.replace(VERSION, "atlas.generated_from_virtual = 1 : i32")),
                     ("nonarray", replace_contract(machine, '"bad"')),
                     ("nondictionary", replace_contract(machine, "[0 : i32]")),
                     ("empty with commands", replace_contract(machine, "[]")),
                     ("foreign field", replace_contract(machine, text.replace("{", "{foreign = 0 : i32, ", 1))),
                     ("missing field", replace_contract(machine, text.replace("block = 0 : i32, ", "", 1))),
                     ("wrong numeric type", replace_contract(machine, text.replace("reg = 11 : i32", "reg = 11 : i64", 1))),
                     ("wrong kind type", replace_contract(machine, text.replace('kind = "reset"', "kind = 0 : i32", 1))),
                     ("unsorted", replace_contract(machine, record_text(list(reversed(facts)))))]
        for index, field, value in ((0, "id", 1), (0, "unit", 2), (0, "slot", 2), (0, "kind", "foreign"),
                                    (0, "scale", 0), (1, "weight", 1), (1, "weight_slot", 0),
                                    (2, "previous", 0), (2, "block", 1), (4, "reg", 41)):
            changed = [dict(r) for r in facts]
            changed[index][field] = value
            mutations.append((f"{index}.{field}", replace_contract(machine, record_text(changed))))
        for name, changed in mutations:
            with self.subTest(mutation=name):
                self.assertNotEqual(changed, machine)
                self.rejected(changed)

    def test_empty_new_contract_and_legacy_without_mxu_metadata(self) -> None:
        machine = lower(virtual_chain(1))
        self.assertIn(LOWERED_VERSION, machine)
        self.assertEqual(contract_text(machine), "[]")
        self.accepted(machine)
        facts, operations = chain_fixture()
        untagged = [(name, re.sub(rf', {TAG} = \d+ : i32', "", fields)) for name, fields in operations]
        legacy = artifact([], untagged).replace(f", {CONTRACT} = []", "")
        for marker in ("atlas.generated_from_virtual", 'atlas.generated_from_virtual = "dma-contract-v1"'):
            changed = legacy.replace(VERSION, marker)
            if marker == "atlas.generated_from_virtual":
                changed = changed.replace(", atlas.virtual_dma_contract = []", "")
            self.accepted(changed)
        self.rejected(artifact([], untagged))

    def test_contract_and_tags_survive_llvm_finalization(self) -> None:
        facts, operations = chain_fixture(fp8=True)
        machine = artifact(facts, operations)
        for option in ("--convert-atlas-to-llvm", "--convert-atlas-to-llvm-calls"):
            with self.subTest(option=option):
                result = run("atlas-opt", machine, option)
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertIn(VERSION, result.stdout)
                self.assertEqual(records(result.stdout), facts)
                if option == "--convert-atlas-to-llvm-calls":
                    self.assertEqual(result.stdout.count(TAG + " ="), 5)
                    final = run("atlas-opt", result.stdout, "--finalize-atlas-llvm-calls")
                    direct = run("atlas-opt", machine, "--convert-atlas-to-llvm")
                    self.assertEqual(final.returncode, 0, final.stderr)
                    self.assertEqual(final.stdout, direct.stdout)

    def test_joint_structured_fields_and_encoded_words_cannot_bypass_contract(self) -> None:
        facts, operations = chain_fixture(fp8=True)
        machine = artifact(facts, operations)
        structured = run("atlas-opt", machine, "--convert-atlas-to-llvm-calls")
        self.assertEqual(structured.returncode, 0, structured.stderr)
        lines = structured.stdout.splitlines()
        targets = [(next(i for i, line in enumerate(lines) if 'atlas.source_op = "atlas.mxu_matmul"' in line and "accumulate = false" in line), "src = 9 : i32", "src = 10 : i32", 13, 6, 10),
                   (next(i for i, line in enumerate(lines) if 'atlas.source_op = "atlas.mxu_pop"' in line), "scale_reg = 3 : i32", "scale_reg = 4 : i32", 13, 6, 4),
                   (next(i for i, line in enumerate(lines) if 'atlas.source_op = "atlas.scalar_load"' in line), "offset = 129 : i32", "offset = 128 : i32", 20, 12, 128)]
        for index, before, after, shift, width, value in targets:
            for joint in (False, True):
                with self.subTest(field=before, joint=joint):
                    replacement = lines[index].replace(before, after)
                    if joint:
                        word = int(re.search(r'atlas.word = (-?\d+) : i32', replacement)[1]) & 0xffffffff
                        word = (word & ~(((1 << width) - 1) << shift)) | (value << shift)
                        replacement = re.sub(r'atlas.word = -?\d+ : i32', f"atlas.word = {word} : i32", replacement)
                    changed = replace_line(structured.stdout, index, replacement)
                    self.assertNotEqual(changed, structured.stdout)
                    self.assertEqual(contract_text(changed), contract_text(structured.stdout))
                    final = run("atlas-opt", changed, "--finalize-atlas-llvm-calls")
                    self.assertNotEqual(final.returncode, 0, final.stdout)
                    self.assertEqual(final.stdout, "")
                    if joint:
                        self.assertIn("MXU contract", final.stderr)

    def test_structured_finalization_rechecks_metadata_and_command_tags(self) -> None:
        facts, operations = chain_fixture(fp8=True)
        structured = run("atlas-opt", artifact(facts, operations), "--convert-atlas-to-llvm-calls")
        self.assertEqual(structured.returncode, 0, structured.stderr)
        text = structured.stdout
        mutations = [("missing MXU array", CONTRACT_RE.sub("", text).replace(", ,", ",").replace(", }", "}")),
                     ("bad MXU array", replace_contract(text, '"bad"')),
                     ("missing DMA array", text.replace("atlas.virtual_dma_contract = [], ", "")),
                     ("missing version", text.replace(VERSION + ", ", "")),
                     ("unknown version", text.replace("resource-contract-v1", "resource-contract-v999")),
                     ("legacy with MXU metadata", text.replace("resource-contract-v1", "dma-contract-v1")),
                     ("foreign tag", text.replace(f"{TAG} = 0 : i32", f"{TAG} = 999 : i32")),
                     ("duplicate tag", text.replace(f"{TAG} = 3 : i32", f"{TAG} = 2 : i32")),
                     ("wrong tag type", text.replace(f"{TAG} = 0 : i32", f"{TAG} = 0 : i64")),
                     ("missing tag", text.replace(f"{TAG} = 0 : i32, ", ""))]
        lines = text.splitlines()
        lines[0] = "module attributes {atlas.structured_handoff} {"
        mutations.append(("all resource metadata removed", "\n".join(lines)))
        for name, changed in mutations:
            with self.subTest(mutation=name):
                self.assertNotEqual(changed, text)
                final = run("atlas-opt", changed, "--finalize-atlas-llvm-calls")
                self.assertNotEqual(final.returncode, 0, final.stdout)
                self.assertEqual(final.stdout, "")
                self.assertTrue(final.stderr)


if __name__ == "__main__":
    unittest.main()
