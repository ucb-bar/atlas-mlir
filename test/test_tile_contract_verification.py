"""Source tile endpoints remain accountable at every physical handoff."""

from __future__ import annotations

import re
import unittest

from test_virtual_dma import copy
from test_virtual_dma_lowering import independent_work
from test_virtual_lowering import BIN, ROOT, lower, run, virtual_chain, virtual_pressure
from test_virtual_mxu_handles import program


CONTRACT = "atlas.virtual_tile_contract"
TAG = "atlas.virtual_tile_command"
VERSION = 'atlas.generated_from_virtual = "resource-contract-v2"'
CONTRACT_RE = re.compile(r'atlas\.virtual_tile_contract = (\[[^\]]*\])')
RECORD_RE = re.compile(r'\{([^{}]*)\}')
FIELD_RE = re.compile(r'(\w+) = (?:(-?\d+) : i32|"([^"]*)")')
BOUNDARIES = (("atlas-opt", ("--verify-atlas-generated-schedule",)),
              ("atlas-emit", ()),
              ("atlas-opt", ("--convert-atlas-to-llvm",)),
              ("atlas-opt", ("--convert-atlas-to-llvm-calls",)))
STATE = "!atlas.state"
DELAY = ("delay", 'cycles = 256 : i32, atlas.delay_reason = "tensor_completion"')
NOP = ("alu_imm", 'kind = "addi", dst = 0 : i32, src = 0 : i32, immediate = 0 : i32')


def contract_text(machine: str) -> str:
    match = CONTRACT_RE.search(machine)
    if match is None:
        raise AssertionError("lowering must retain the source tile contract")
    return match[1]


def records(machine: str) -> list[dict]:
    result = []
    for text in RECORD_RE.findall(contract_text(machine)):
        entry = {name: int(integer) if integer else string
                 for name, integer, string in FIELD_RE.findall(text)}
        for address in ("vmem_byte", "dram_byte"):
            entry[address] &= 0xffffffff
        after = re.search(r'after = array<i32(?:: ([^>]*))?>', text)
        if after is None:
            raise AssertionError("tile dependencies must be a dense i32 array")
        entry["after"] = tuple(int(value.strip()) for value in after[1].split(",")) if after[1] else ()
        result.append(entry)
    return result


def record(identity: int, kind: str, *, reg: int = -1, vmem: int = 0,
           dram: int = 0, size: int = 0, channel: int = -1,
           transfer: int = -1, after: tuple[int, ...] = ()) -> dict:
    return dict(id=identity, kind=kind, reg=reg, vmem_byte=vmem,
                dram_byte=dram, bytes=size, channel=channel, transfer=transfer, after=after)


def record_text(entries: list[dict]) -> str:
    def field(key: str, value) -> str:
        if key == "after":
            return f'{key} = array<i32' + (": " + ", ".join(map(str, value)) if value else "") + ">"
        return f'{key} = "{value}"' if isinstance(value, str) else f"{key} = {value} : i32"
    return "[" + ", ".join("{" + ", ".join(field(key, value) for key, value in entry.items()) + "}"
                            for entry in entries) + "]"


def replace_contract(machine: str, text: str) -> str:
    return CONTRACT_RE.sub(lambda _: f"{CONTRACT} = {text}", machine, count=1)


def command(name: str, fields: str, identity: int) -> tuple[str, str]:
    return name, fields + f", {TAG} = {identity} : i32"


def materialize(reg: int, value: int) -> list[tuple[str, str]]:
    low = value & 0xfff
    if low >= 0x800:
        low -= 0x1000
    return [("upper", f'kind = "lui", dst = {reg} : i32, immediate = {((value - low) >> 12) & 0xfffff} : i32'),
            ("alu_imm", f'kind = "addi", dst = {reg} : i32, src = {reg} : i32, immediate = {low} : i32')]


def artifact(facts: list[dict], operations: list[tuple[str, str]]) -> str:
    lines = [f'%s0 = "atlas.start"() : () -> {STATE}']
    lines += [f'%s{i + 1} = "atlas.{name}"(%s{i}) {{{fields}}} : ({STATE}) -> {STATE}'
              for i, (name, fields) in enumerate([*operations, NOP, ("trap", 'kind = "ecall"')])]
    dma = []
    for fact in facts:
        if fact["kind"] not in ("dma_load", "dma_store") or fact["transfer"] < 0:
            continue
        dma.append("{" + f'id = {fact["transfer"]} : i32, direction = "{fact["kind"][4:]}", channel = {fact["channel"]} : i32, '
                   f'staging_word = {fact["vmem_byte"] // 4} : i32, dram_byte = {fact["dram_byte"]} : i32, '
                   f'size_bytes = {fact["bytes"]} : i32, staging_reg = 4 : i32, dram_reg = 7 : i32, size_reg = 2 : i32' + "}")
    return (f'module attributes {{{VERSION}, atlas.timing_state = "timed", atlas.timing_provider = "npu-model-rtl-match-v1", atlas.virtual_dma_contract = [{", ".join(dma)}], atlas.virtual_mxu_contract = [], '
            f"{CONTRACT} = {record_text(facts)}}} {{\n" + "\n".join(lines) + "\n}")


def vector_fixture() -> tuple[list[dict], list[tuple[str, str]]]:
    # Literal endpoints deliberately avoid allocator-selected registers.
    facts = [record(0, "dma_load", vmem=0x80000, dram=0x90000000, size=2048, channel=0, transfer=0),
             record(1, "dma_wait", channel=0, transfer=0, after=(0,)),
             record(2, "vload", reg=40, vmem=0x80000, size=1024, transfer=0, after=(1,)),
             record(3, "vload", reg=41, vmem=0x80400, size=1024, transfer=0, after=(1,)),
             record(4, "vstore", reg=40, vmem=0x80000, size=1024, transfer=1),
             record(5, "vstore", reg=41, vmem=0x80400, size=1024, transfer=1),
             record(6, "dma_store", vmem=0x80000, dram=0x90000000, size=2048, channel=1, transfer=1, after=(4, 5)),
             record(7, "dma_wait", channel=1, transfer=1, after=(6,))]
    operations = [*materialize(4, 131072), *materialize(7, 0x90000000), *materialize(2, 2048),
                  ("dma_config", 'base_reg = 0 : i32, channel = 0 : i32'),
                  ("dma_config", 'base_reg = 0 : i32, channel = 1 : i32'),
                  command("dma", 'direction = "load", channel = 0 : i32, reg = 4 : i32, dram = 7 : i32, size = 2 : i32, atlas.virtual_dma_transfer = 0 : i32', 0),
                  command("dma_wait", 'channel = 0 : i32, atlas.virtual_dma_transfer = 0 : i32', 1)]
    for identity, name, field, reg, offset in ((2, "vload", "dst", 40, 0),
                                             (3, "vload", "dst", 41, 8),
                                             (4, "vstore", "src", 40, 0),
                                             (5, "vstore", "src", 41, 8)):
        operations += [command(name, f'{field} = {reg} : i32, base = 4 : i32, offset = {offset} : i32, format = "raw"', identity), DELAY]
    operations += [command("dma", 'direction = "store", channel = 1 : i32, reg = 4 : i32, dram = 7 : i32, size = 2 : i32, atlas.virtual_dma_transfer = 1 : i32', 6),
                   command("dma_wait", 'channel = 1 : i32, atlas.virtual_dma_transfer = 1 : i32', 7)]
    return facts, operations


def boundary_fixture(*, mailbox: bool = False) -> tuple[list[dict], list[tuple[str, str]]]:
    address = 0x90010000 if mailbox else 0x90000000
    vmem = 0 if mailbox else 0x80000
    facts = [record(0, "dma_load", vmem=vmem, dram=address, size=1024, channel=0),
             record(1, "dma_wait", channel=0, after=(0,))]
    operations = [*materialize(4, vmem // 4), *materialize(7, address), *materialize(2, 1024),
                  ("dma_config", 'base_reg = 0 : i32, channel = 0 : i32'),
                  command("dma", 'direction = "load", channel = 0 : i32, reg = 4 : i32, dram = 7 : i32, size = 2 : i32', 0),
                  command("dma_wait", 'channel = 0 : i32', 1)]
    if mailbox:
        for index, reg in enumerate((10, 11)):
            facts.append(record(2 + index, "mailbox_load", reg=reg, vmem=4 * index, size=4, after=(1,)))
            operations += [command("scalar_load", f'kind = "lw", dst = {reg} : i32, base = 0 : i32, offset = {4 * index} : i32', 2 + index),
                           ("delay", 'cycles = 8 : i32, atlas.delay_reason = "scalar_load_completion"')]
    else:
        facts.append(record(2, "vload", reg=40, vmem=vmem, size=1024, after=(1,)))
        operations += [command("vload", 'dst = 40 : i32, base = 4 : i32, offset = 0 : i32, format = "raw"', 2), DELAY]
    return facts, operations


def store_fixture() -> tuple[list[dict], list[tuple[str, str]]]:
    facts = [record(0, "vstore", reg=40, vmem=0x80000, size=1024),
             record(1, "vstore", reg=41, vmem=0x80400, size=1024),
             record(2, "dma_store", vmem=0x80000, dram=0x90000000, size=2048, channel=1, after=(0, 1)),
             record(3, "dma_wait", channel=1, after=(2,))]
    operations = [*materialize(4, 131072), *materialize(7, 0x90000000), *materialize(2, 2048),
                  ("dma_config", 'base_reg = 0 : i32, channel = 1 : i32'),
                  command("vstore", 'src = 40 : i32, base = 4 : i32, offset = 0 : i32, format = "raw"', 0), DELAY,
                  command("vstore", 'src = 41 : i32, base = 4 : i32, offset = 8 : i32, format = "raw"', 1), DELAY,
                  command("dma", 'direction = "store", channel = 1 : i32, reg = 4 : i32, dram = 7 : i32, size = 2 : i32', 2),
                  command("dma_wait", 'channel = 1 : i32', 3)]
    return facts, operations


class TileContractVerificationTest(unittest.TestCase):
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

    def test_explicit_formats_have_source_derived_halves_and_dependencies(self) -> None:
        for fmt, halves, size in (("fp8", 1, 1024), ("bf16", 2, 2048)):
            machine = lower(copy(fmt))
            facts = records(machine)
            self.assertIn(VERSION, machine)
            self.assertEqual([r["kind"] for r in facts],
                             ["dma_load", "dma_wait"] + ["vload"] * halves + ["vstore"] * halves + ["dma_store", "dma_wait"])
            self.assertEqual([r["id"] for r in facts], list(range(4 + 2 * halves)))
            self.assertEqual((facts[0]["vmem_byte"], facts[0]["dram_byte"], facts[0]["bytes"], facts[0]["transfer"]),
                             (0x80000, 0x80000000, size, 0))
            self.assertEqual(facts[1]["after"], (0,))
            for half in range(halves):
                load, store = facts[2 + half], facts[2 + halves + half]
                self.assertEqual((load["vmem_byte"], load["bytes"], load["after"], load["transfer"]),
                                 (0x80000 + 1024 * half, 1024, (1,), 0))
                self.assertEqual((store["vmem_byte"], store["bytes"], store["transfer"]),
                                 (0x80000 + 1024 * half, 1024, 1))
                self.assertEqual(store["reg"], load["reg"])
                if half:
                    self.assertEqual(load["reg"], facts[2]["reg"] + 1)
            self.assertEqual(facts[-2]["after"], tuple(range(2 + halves, 2 + 2 * halves)))
            self.assertEqual(facts[-1]["after"], (2 + 2 * halves,))
            self.accepted(machine)

    def test_boundary_indices_reserve_two_kib_slots(self) -> None:
        machine = lower(virtual_pressure(2))
        facts = records(machine)
        self.assertEqual([r["dram_byte"] for r in facts if r["kind"] == "dma_load"],
                         [0x90000000, 0x90000400, 0x90000800, 0x90000c00])
        self.assertEqual([r["vmem_byte"] for r in facts if r["kind"] == "vload"], [0, 1024, 2048, 3072])
        self.assertEqual([r["dram_byte"] for r in facts if r["kind"] == "dma_store"],
                         [0x90020000, 0x90020400, 0x90020800, 0x90020c00])
        self.assertTrue(all(r["bytes"] == 1024 for r in facts if r["kind"] != "dma_wait"))
        self.assertTrue(all(r["transfer"] == -1 for r in facts))
        self.accepted(machine)

    def test_each_bf16_half_register_address_and_raw_format_is_checked(self) -> None:
        facts, operations = vector_fixture()
        self.accepted(artifact(facts, operations))
        for index in (10, 12, 14, 16):
            name, fields = operations[index]
            field = "dst" if name == "vload" else "src"
            for label, replacement in (
                    ("register", re.sub(rf'{field} = \d+ : i32', f'{field} = 42 : i32', fields)),
                    ("address", fields.replace("offset = 0", "offset = 8") if "offset = 0" in fields else fields.replace("offset = 8", "offset = 0")),
                    ("unknown pointer", fields.replace("base = 4", "base = 15")),
                    ("format", fields.replace('format = "raw"', 'format = "bf16"'))):
                with self.subTest(index=index, mutation=label):
                    changed = operations.copy()
                    changed[index] = name, replacement
                    self.assertNotEqual(changed, operations)
                    self.rejected(artifact(facts, changed))

    def test_equivalent_base_and_signed_word_offsets_are_accepted(self) -> None:
        facts, operations = vector_fixture()
        for index in (10, 12, 14, 16):
            name, fields = operations[index]
            fields = fields.replace("base = 4", "base = 15")
            fields = fields.replace("offset = 0", "offset = -8") if "offset = 0" in fields else fields.replace("offset = 8", "offset = 0")
            operations[index] = name, fields
        self.accepted(artifact(facts, materialize(15, 131328) + operations))

    def test_rv32_address_wrap_is_proved_before_vmem_domain_checks(self) -> None:
        facts, operations = boundary_fixture()
        facts[0]["vmem_byte"] = 0
        facts[2]["vmem_byte"] = 0
        operations[:2] = materialize(4, 0)
        name, fields = operations[9]
        operations[9] = name, fields.replace("base = 4", "base = 15").replace("offset = 0", "offset = 1")
        wrapped = operations[:9] + materialize(15, 0xffffffe0) + operations[9:]
        self.accepted(artifact(facts, wrapped))
        for base in (0x00100000, 0x000000e0):
            changed = operations[:9] + materialize(15, base) + operations[9:]
            self.rejected(artifact(facts, changed))

    def test_implicit_dma_capture_fields_and_wait_channel_are_checked(self) -> None:
        facts, operations = boundary_fixture()
        self.accepted(artifact(facts, operations))
        for index, before, after in ((1, "immediate = 0", "immediate = 256"),
                                    (3, "immediate = 0", "immediate = 32"),
                                    (5, "immediate = 1024", "immediate = 2048"),
                                    (7, "reg = 4", "reg = 15"),
                                    (7, "dram = 7", "dram = 15"),
                                    (7, "size = 2", "size = 15"),
                                    (8, "channel = 0", "channel = 1")):
            with self.subTest(index=index, field=before):
                changed = operations.copy()
                name, fields = changed[index]
                changed[index] = name, fields.replace(before, after)
                self.assertNotEqual(changed, operations)
                self.rejected(artifact(facts, changed))
        changed = operations.copy()
        for index in (7, 8):
            name, fields = changed[index]
            changed[index] = name, fields.replace("channel = 0", "channel = 1")
        self.rejected(artifact(facts, changed))
        reused = [*operations[:8], ("alu_imm", 'kind = "addi", dst = 7 : i32, src = 0 : i32, immediate = 1 : i32'),
                  ("alu_imm", 'kind = "addi", dst = 2 : i32, src = 0 : i32, immediate = 1 : i32'), *operations[8:]]
        self.accepted(artifact(facts, reused))
        reused_after_wait = [*operations[:9], *reused[8:10], *operations[9:]]
        self.accepted(artifact(facts, reused_after_wait))

    def test_mailbox_loads_have_exact_destination_and_byte_address(self) -> None:
        facts, operations = boundary_fixture(mailbox=True)
        self.accepted(artifact(facts, operations))
        for index in (9, 11):
            name, fields = operations[index]
            for replacement in (re.sub(r'dst = \d+ : i32', 'dst = 12 : i32', fields),
                                fields.replace("offset = 0", "offset = 4") if "offset = 0" in fields else fields.replace("offset = 4", "offset = 0"),
                                fields.replace("base = 0", "base = 15")):
                changed = operations.copy()
                changed[index] = name, replacement
                self.rejected(artifact(facts, changed))
        dynamic = (ROOT / "test/examples/virtual_bf16_dynamic_branch_program.mlir").read_text()
        dynamic = dynamic.replace("(%choose: i1)", "(%choose: i1, %other: i32)")
        machine = lower(dynamic)
        entries = records(machine)
        self.assertEqual([(r["kind"], r["vmem_byte"], r["bytes"], r["after"]) for r in entries[:4]],
                         [("dma_load", 0, 1024, ()), ("dma_wait", 0, 0, (0,)),
                          ("mailbox_load", 0, 4, (1,)), ("mailbox_load", 4, 4, (1,))])
        self.accepted(machine)

    def test_missing_duplicate_foreign_and_untagged_commands_fail(self) -> None:
        facts, operations = vector_fixture()
        for index in (10, 12, 14, 16):
            for value in (None, "999 : i32", "-1 : i32", "0 : i64", '"bad"'):
                with self.subTest(index=index, tag=value):
                    changed = operations.copy()
                    name, fields = changed[index]
                    changed[index] = name, re.sub(rf', {TAG} = \d+ : i32', "" if value is None else f", {TAG} = {value}", fields)
                    self.rejected(artifact(facts, changed))
            self.rejected(artifact(facts, operations[:index] + operations[index + 2:]))
            self.rejected(artifact(facts, operations[:index] + operations[index:index + 2] + operations[index:]))
        self.rejected(artifact(facts, [*operations, ("vload", 'dst = 44 : i32, base = 4 : i32, offset = 0 : i32, format = "raw"'), DELAY]))
        self.rejected(artifact(facts, [*operations, (NOP[0], NOP[1] + f", {TAG} = 0 : i32")]))
        for mailbox in (False, True):
            facts, operations = boundary_fixture(mailbox=mailbox)
            for index, (name, fields) in enumerate(operations):
                if TAG not in fields:
                    continue
                changed = operations.copy()
                changed[index] = name, re.sub(rf', {TAG} = \d+ : i32', "", fields)
                self.rejected(artifact(facts, changed))

    def test_implicit_store_launch_capture_and_completion_are_checked(self) -> None:
        facts, operations = store_fixture()
        self.accepted(artifact(facts, operations))
        for index, before, after in ((3, "immediate = 0", "immediate = 32"),
                                    (5, "immediate = -2048", "immediate = -1024"),
                                    (11, "reg = 4", "reg = 15"),
                                    (11, "dram = 7", "dram = 15"),
                                    (11, "size = 2", "size = 15"),
                                    (12, "channel = 1", "channel = 0")):
            with self.subTest(index=index, field=before):
                changed = operations.copy()
                name, fields = changed[index]
                changed[index] = name, fields.replace(before, after)
                self.assertNotEqual(changed, operations)
                self.rejected(artifact(facts, changed))

    def test_source_dependencies_require_prior_execution(self) -> None:
        facts, operations = boundary_fixture()
        # VLOAD is locally legal before DMA; source correspondence must reject it.
        moved = operations[:7] + operations[9:] + operations[7:9]
        self.rejected(artifact(facts, moved))
        facts, operations = vector_fixture()
        self.accepted(artifact(facts, operations))
        swapped = operations.copy()
        swapped[10:14] = operations[12:14] + operations[10:12]
        self.accepted(artifact(facts, swapped))
        facts[3]["after"] = ()
        self.rejected(artifact(facts, swapped))
        facts, operations = store_fixture()
        self.accepted(artifact(facts, operations))
        self.rejected(artifact(facts, operations[:7] + operations[11:] + operations[7:11]))
        facts, operations = boundary_fixture(mailbox=True)
        self.rejected(artifact(facts, operations[:7] + operations[9:] + operations[7:9]))

    def test_cfg_path_must_execute_recorded_predecessors(self) -> None:
        facts = [record(0, "vstore", reg=13, vmem=0x80000, size=1024),
                 record(1, "vload", reg=13, vmem=0x80400, size=1024, after=(0,))]
        operations = [*materialize(4, 131072),
                      command("vstore", 'src = 13 : i32, base = 4 : i32, offset = 0 : i32, format = "raw"', 0), DELAY,
                      command("vload", 'dst = 13 : i32, base = 4 : i32, offset = 8 : i32, format = "raw"', 1), DELAY]
        machine = artifact(facts, operations)
        lines = machine.splitlines()
        # A legal forward branch skips the store and its completion delay.
        lines[4] = ('%branch = "atlas.branch"(%s2) {kind = "beq", lhs = 0 : i32, rhs = 0 : i32, offset = 16 : i32} : (!atlas.state) -> !atlas.state\n'
                    '%branch_nop = "atlas.alu_imm"(%branch) {kind = "addi", dst = 0 : i32, src = 0 : i32, immediate = 0 : i32} : (!atlas.state) -> !atlas.state\n'
                    + lines[4].replace("(%s2)", "(%branch_nop)"))
        self.rejected("\n".join(lines) + "\n")

    def test_pack_endpoints_preserve_source_result_and_scratch_addresses(self) -> None:
        source = program('    %packed = "atlas.virtual_pack_fp8"(%seed) {scale_code = 127 : i32} : (!atlas.virtual_bf16) -> !atlas.virtual_fp8')
        source = source.replace("atlas.output_dram_base = 2415923200", "atlas.output_dram_base = 2415951872")
        machine = lower(source)
        facts = records(machine)
        store = next(r for r in facts if r["kind"] == "vstore" and r["vmem_byte"] == 0x20000)
        reload = next(r for r in facts if r["kind"] == "vload" and r["vmem_byte"] == 0x20400)
        self.assertEqual((store["reg"], store["bytes"], reload["reg"], reload["bytes"], reload["after"]),
                         (store["reg"], 1024, store["reg"], 1024, (store["id"],)))
        self.accepted(machine)
        for identity in (store["id"], reload["id"]):
            for field in ("base", "src" if identity == store["id"] else "dst"):
                changed = re.sub(rf'([^\n]*{TAG} = {identity} : i32[^\n]*)',
                                 lambda match: re.sub(rf'{field} = \d+ : i32', f'{field} = 15 : i32', match[0]), machine)
                self.assertNotEqual(changed, machine)
                self.rejected(changed)
        endpoints = [record(0, "vstore", reg=13, vmem=0x80000, size=1024),
                     record(1, "vload", reg=13, vmem=0x80400, size=1024, after=(0,))]
        ordered = [*materialize(4, 131072),
                   command("vstore", 'src = 13 : i32, base = 4 : i32, offset = 0 : i32, format = "raw"', 0), DELAY,
                   command("vload", 'dst = 13 : i32, base = 4 : i32, offset = 8 : i32, format = "raw"', 1), DELAY]
        self.accepted(artifact(endpoints, ordered))
        self.rejected(artifact(endpoints, ordered[:2] + ordered[4:] + ordered[2:4]))

    def test_strict_schema_versions_and_dependencies(self) -> None:
        facts, operations = vector_fixture()
        machine = artifact(facts, operations)
        text = contract_text(machine)
        mutations = [CONTRACT_RE.sub("", machine).replace(", }", "}"),
                     replace_contract(machine, '"bad"'), replace_contract(machine, "[0 : i32]"),
                     replace_contract(machine, "[]"),
                     replace_contract(machine, text.replace("bytes = 1024 : i32, ", "", 1)),
                     replace_contract(machine, text.replace("reg = 40 : i32", "reg = 40 : i64", 1)),
                     replace_contract(machine, text.replace("{", "{foreign = 0 : i32, ", 1)),
                     replace_contract(machine, text.replace("id = 1 : i32", "id = 0 : i32", 1)),
                     replace_contract(machine, text.replace("after = array<i32>", "after = array<i32: 0>", 1)),
                     replace_contract(machine, text.replace("after = array<i32>", "after = [0 : i32]", 1)),
                     replace_contract(machine, text.replace('kind = "vload"', 'kind = "foreign"', 1)),
                     replace_contract(machine, text.replace("bytes = 1024 : i32", "bytes = 2048 : i32", 1)),
                     replace_contract(machine, text.replace("vmem_byte = 524288 : i32", "vmem_byte = 524289 : i32", 1)),
                     replace_contract(machine, text.replace("channel = -1 : i32", "channel = 0 : i32", 1)),
                     replace_contract(machine, text.replace("reg = 40 : i32", "reg = 64 : i32", 1))]
        for changed in mutations:
            self.assertNotEqual(changed, machine)
            self.rejected(changed)
        for marker in ('"resource-contract-v1"', '"dma-contract-v1"', "1 : i32"):
            self.rejected(machine.replace('"resource-contract-v2"', marker, 1))

    def test_llvm_handoffs_preserve_contract_and_recheck_corruption(self) -> None:
        facts, operations = vector_fixture()
        machine = artifact(facts, operations)
        for option in ("--convert-atlas-to-llvm", "--convert-atlas-to-llvm-calls"):
            with self.subTest(option=option):
                result = run("atlas-opt", machine, option)
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertEqual(records(result.stdout), facts)
                if option == "--convert-atlas-to-llvm-calls":
                    self.assertEqual(result.stdout.count(TAG + " ="), len(facts))
        structured = run("atlas-opt", machine, "--convert-atlas-to-llvm-calls")
        self.assertEqual(structured.returncode, 0, structured.stderr)
        final = run("atlas-opt", structured.stdout, "--finalize-atlas-llvm-calls")
        direct = run("atlas-opt", machine, "--convert-atlas-to-llvm")
        self.assertEqual(final.returncode, 0, final.stderr)
        self.assertEqual(final.stdout, direct.stdout)
        lines = structured.stdout.splitlines()
        index = next(i for i, line in enumerate(lines) if 'atlas.source_op = "atlas.vload"' in line)
        word = int(re.search(r'atlas.word = (-?\d+) : i32', lines[index])[1])
        lines[index] = lines[index].replace("dst = 40 : i32", "dst = 42 : i32")
        changed_word = (word & 0xffffffff & ~(31 << 7)) | (10 << 7)
        lines[index] = re.sub(r'atlas.word = -?\d+ : i32', f'atlas.word = {changed_word} : i32', lines[index])
        changed = "\n".join(lines) + "\n"
        self.assertEqual(records(changed), facts)
        rejected = run("atlas-opt", changed, "--finalize-atlas-llvm-calls")
        self.assertNotEqual(rejected.returncode, 0, rejected.stdout)
        self.assertEqual(rejected.stdout, "")
        self.assertIn("tile contract", rejected.stderr.lower())

    def test_stream_rewrites_preserve_supported_contract_and_reject_padded_input(self) -> None:
        facts, operations = boundary_fixture()
        machine = artifact(facts, operations)
        # Scheduling currently rejects lowering's explicit tensor completion padding.
        for option in ("--insert-atlas-delays", "--schedule-atlas-stream"):
            with self.subTest(option=option):
                padded = run("atlas-opt", machine, option)
                self.assertNotEqual(padded.returncode, 0, padded.stdout)
                self.assertEqual(padded.stdout, "")
                self.assertIn("delays come from the timing model", padded.stderr)
                dma_only = artifact(facts[:2], operations[:9])
                result = run("atlas-opt", dma_only, option)
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertEqual(records(result.stdout), facts[:2])
                self.assertEqual(result.stdout.count(TAG + " ="), 2)
                self.accepted(result.stdout)
                changed = result.stdout.replace("immediate = 589824", "immediate = 589825", 1)
                self.assertNotEqual(changed, result.stdout)
                self.assertEqual(records(changed), facts[:2])
                self.rejected(changed, "tile contract captured DRAM byte address mismatch")

    def test_pending_dma_and_generated_empty_tensor_scope_remain_valid(self) -> None:
        self.accepted(artifact([], []))
        self.accepted(lower(independent_work()))
        self.accepted(lower(virtual_chain(1)))


if __name__ == "__main__":
    unittest.main()
