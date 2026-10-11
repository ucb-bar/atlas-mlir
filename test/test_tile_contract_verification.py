"""Source tile endpoints remain accountable at every physical handoff."""

from __future__ import annotations

import re
import unittest

from test_virtual_dma import copy
from test_virtual_lowering import ROOT, lower, run, virtual_chain
from test_virtual_mxu_handles import program
from verification_support import (
    NOP, assert_boundaries, checked, constant, contract_text, finalize_rejects, insert_after, last_constant_write,
    BoundaryChecks, line_index, moved, operations, records as contract_records, reorder, replace_contract, rewrite_line,
    shift_constant, structured_word,
)

CONTRACT = "atlas.virtual_tile_contract"
TAG = "atlas.virtual_tile_command"
MISMATCH = "tile contract command kind, format, register, or channel mismatch"
BAD_TAG = "tile contract command tag requires a known nonnegative i32 id"
DYNAMIC = (ROOT / "test/examples/virtual_bf16_dynamic_branch_program.mlir").read_text().replace("(%choose: i1)", "(%choose: i1, %other: i32)")


def records(machine: str) -> list[dict]:
    return contract_records(machine, "tile")


def tag(identity: int) -> str:
    return f"{TAG} = {identity} : i32"


def mutate(machine: str, identity: int, old: str, new: str) -> str:
    """Change a field of the operation issuing tile command `identity`."""
    return rewrite_line(machine, line_index(machine, tag(identity)), lambda line: line.replace(old, new, 1))


def pack_program() -> str:
    source = program('    %packed = "atlas.virtual_pack_fp8"(%seed) {scale_code = 127 : i32} : (!atlas.virtual_bf16) -> !atlas.virtual_fp8')
    return source.replace("atlas.output_dram_base = 2415923200", "atlas.output_dram_base = 2415951872")


class TileContractVerificationTest(BoundaryChecks, unittest.TestCase):
    REJECTS = "tile contract"

    def test_explicit_formats_have_source_derived_halves_and_dependencies(self) -> None:
        for fmt, halves, size in (("fp8", 1, 1024), ("bf16", 2, 2048)):
            machine = lower(copy(fmt))
            facts = records(machine)
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
                self.assertEqual((store["vmem_byte"], store["bytes"], store["transfer"], store["reg"]),
                                 (0x80000 + 1024 * half, 1024, 1, facts[2]["reg"] + half))
            self.assertEqual(facts[-2]["after"], tuple(range(2 + halves, 2 + 2 * halves)))
            self.assertEqual(facts[-1]["after"], (2 + 2 * halves,))
            self.accepted(machine)

    def test_vector_endpoint_fields_and_tags_are_checked(self) -> None:
        machine = lower(copy("bf16"))
        ops = operations(machine)
        for identity, field in ((2, "dst"), (3, "dst"), (4, "src"), (5, "src")):
            index = next(n for n, line in enumerate(ops) if tag(identity) in line)
            reg = re.search(rf"{field} = (\d+) : i32", ops[index])[1]
            mutations = [(f"{field} = {reg} : i32", f"{field} = 42 : i32", MISMATCH), ("offset = 0", "offset = 8", "captured vector VMEM byte address mismatch"),
                         ("base = 4", "base = 15", "cannot prove captured vector VMEM byte address"), ('format = "raw"', 'format = "bf16"', "format must be raw"),
                         (f", {tag(identity)}", "", "requires a command tag on every generated tile transfer")]
            mutations += [(tag(identity), f"{TAG} = {value}", BAD_TAG) for value in ("999 : i32", "-1 : i32", "0 : i64", '"bad"')]
            changes = [(mutate(machine, identity, old, new), diagnostic) for old, new, diagnostic in mutations]
            changes += [(reorder(machine, ops[:index] + ops[index + 1:]), "requires exactly one command for every source record"),
                        (reorder(machine, ops[:index + 1] + ops[index:]), f"has duplicate command id {identity}")]
            for changed, diagnostic in changes:
                with self.subTest(command=identity, diagnostic=diagnostic):
                    self.rejected(changed, diagnostic)
        last = line_index(machine, tag(7))
        self.rejected(insert_after(machine, last, [("vload", 'dst = 44 : i32, base = 4 : i32, offset = 0 : i32, format = "raw"')]),
                      "requires a command tag on every generated tile transfer")
        self.rejected(insert_after(machine, last, [(NOP[0], NOP[1] + f", {tag(0)}")]), "has duplicate command id 0")
        for source in (virtual_chain(1), DYNAMIC):
            boundary = lower(source)
            for identity in range(len(records(boundary))):
                with self.subTest(untagged=identity):
                    self.rejected(mutate(boundary, identity, f", {tag(identity)}", ""))

    def test_equivalent_base_and_signed_word_offsets_are_accepted(self) -> None:
        machine = insert_after(lower(copy("bf16")), 2, constant(15, 0x20100))
        for identity in (2, 3, 4, 5):
            machine = mutate(machine, identity, "base = 4 : i32", "base = 15 : i32")
            if identity in (2, 4):
                machine = mutate(machine, identity, "offset = 0", "offset = -8")
        self.accepted(machine)

    def test_rv32_address_wrap_is_proved_before_vmem_domain_checks(self) -> None:
        for base, diagnostic in ((0xffffffe0, None), (0x00100000, "vector DMA memory access has invalid VMEM span"),
                                 (0x000000e0, "captured vector VMEM byte address mismatch")):
            machine = insert_after(lower(virtual_chain(1)), 2, constant(15, base))
            with self.subTest(base=hex(base)):
                assert_boundaries(self, mutate(mutate(machine, 2, "base = 6", "base = 15"), 2, "offset = 0", "offset = 1"), rejects=diagnostic)

    def test_implicit_dma_capture_fields_and_completion_are_checked(self) -> None:
        machine = lower(virtual_chain(1))
        lines = machine.splitlines()
        load, store = line_index(machine, tag(0)), line_index(machine, tag(7))
        alu = lambda reg: line_index(machine, '"atlas.alu_imm"', f"dst = {reg} : i32")
        load_dram, store_dram = last_constant_write(machine, load, 1), last_constant_write(machine, store, 3)
        mutations = [(alu(6), "immediate = 0", "immediate = 256", "captured DMA VMEM byte address mismatch"),
                     (load_dram, lines[load_dram], shift_constant(lines[load_dram], 32), "captured DRAM byte address mismatch"),
                     (alu(2), "immediate = 1024", "immediate = 1023", "captured byte length mismatch"),
                     (store_dram, lines[store_dram], shift_constant(lines[store_dram], 32), "captured DRAM byte address mismatch"),
                     (store + 1, "channel = 1", "channel = 0", "DMA.WAIT has no pending DMA transfer")]
        mutations += [(index, f"{field} = {reg} : i32", f"{field} = 15 : i32", f"cannot prove captured {name}")
                      for index, regs in ((load, (6, 1, 2)), (store, (8, 3, 2)))
                      for field, reg, name in zip(("reg", "dram", "size"), regs, ("DMA VMEM byte address", "DRAM byte address", "byte length"))]
        for index, old, new, diagnostic in mutations:
            with self.subTest(line=index, field=old):
                self.rejected(rewrite_line(machine, index, lambda line: line.replace(old, new, 1)), diagnostic)
        retarget = lambda line: line.replace("channel = 0", "channel = 1")
        self.rejected(rewrite_line(rewrite_line(machine, load, retarget), load + 1, retarget), MISMATCH)
        reuse = [("alu_imm", f'kind = "addi", dst = {reg} : i32, src = 0 : i32, immediate = 1 : i32') for reg in (3, 8)]
        for index in (store, store + 1):
            with self.subTest(reused_after=index):
                self.accepted(insert_after(machine, index, reuse))

    def test_mailbox_loads_have_exact_destination_and_byte_address(self) -> None:
        machine = lower(DYNAMIC)
        self.assertEqual([(r["kind"], r["vmem_byte"], r["bytes"], r["after"]) for r in records(machine)[:4]],
                         [("dma_load", 0, 1024, ()), ("dma_wait", 0, 0, (0,)), ("mailbox_load", 0, 4, (1,)), ("mailbox_load", 4, 4, (1,))])
        self.accepted(machine)
        for identity, reg, offset in ((2, 10, 0), (3, 11, 4)):
            for old, new, diagnostic in ((f"dst = {reg} : i32", "dst = 12 : i32", MISMATCH),
                                         (f"offset = {offset}", f"offset = {4 - offset}", "captured mailbox VMEM byte address mismatch"),
                                         ("base = 0", "base = 15", "cannot prove captured mailbox VMEM byte address")):
                with self.subTest(command=identity, field=old):
                    self.rejected(mutate(machine, identity, old, new), diagnostic)

    def test_source_and_cfg_dependencies_require_prior_execution(self) -> None:
        chain, vector, mailbox, packed = (lower(source, timed=False) for source in (virtual_chain(1), copy("bf16"), DYNAMIC, pack_program()))
        store = next(r["id"] for r in records(packed) if r["kind"] == "vstore" and r["vmem_byte"] == 0x20000)
        early = lambda machine, chosen, anchor: moved(machine, lambda line: any(tag(i) in line for i in chosen), lambda line: tag(anchor) in line)
        skip = [("branch", 'kind = "beq", lhs = 0 : i32, rhs = 0 : i32, offset_bytes = 8 : i32'), NOP]
        for name, changed in (("VLOAD before its DMA", early(chain, (2,), 0)), ("DMA store before its VSTORE", early(chain, (7, 8), 6)),
                              ("mailbox LW before its DMA", early(mailbox, (2, 3), 0)), ("PACK load before its store", early(packed, (store + 1,), store)),
                              ("branch skips the DMA", insert_after(chain, line_index(chain, tag(0)) - 1, skip))):
            with self.subTest(order=name):
                self.rejected(changed, "predecessor is not established on every emitted path")
        ops = operations(vector)
        first, second = (next(n for n, line in enumerate(ops) if tag(i) in line) for i in (4, 5))
        self.accepted(reorder(vector, ops[:first - 2] + ops[first + 1:second + 1] + ops[first - 2:first + 1] + ops[second + 1:]))

    def test_pack_endpoints_preserve_source_result_and_scratch_addresses(self) -> None:
        machine = lower(pack_program())
        facts = records(machine)
        store = next(r for r in facts if r["kind"] == "vstore" and r["vmem_byte"] == 0x20000)
        reload = next(r for r in facts if r["kind"] == "vload" and r["vmem_byte"] == 0x20400)
        self.assertEqual((store["bytes"], reload["reg"], reload["bytes"], reload["after"]), (1024, store["reg"], 1024, (store["id"],)))
        self.accepted(machine)
        for identity, field in ((store["id"], "src"), (reload["id"], "dst")):
            for name, diagnostic in (("base", "cannot prove captured vector VMEM byte address"), (field, MISMATCH)):
                with self.subTest(command=identity, field=name):
                    changed = rewrite_line(machine, line_index(machine, tag(identity)),
                                           lambda line: re.sub(rf"\b{name} = \d+ : i32", f"{name} = 15 : i32", line))
                    self.rejected(changed, diagnostic)

    def test_strict_schema_and_dependencies(self) -> None:
        machine = lower(copy("bf16"))
        text = contract_text(machine, "tile")
        self.rejected(replace_contract(machine, "tile", '"bad"'), f"requires an {CONTRACT} array")
        mutations = [("nondictionary", "[0 : i32]"), ("empty", "[]")]
        for name, old, new in (("missing field", "bytes = 2048 : i32, ", ""), ("wrong type", "reg = 0 : i32", "reg = 0 : i64"),
                               ("foreign field", "{", "{foreign = 0 : i32, "), ("duplicate id", "id = 1 : i32", "id = 0 : i32"),
                               ("added dependency", "after = array<i32>", "after = array<i32: 0>"),
                               ("dependency type", "after = array<i32>", "after = [0 : i32]"),
                               ("dropped dependency", "after = array<i32: 1>", "after = array<i32>"),
                               ("kind", 'kind = "vload"', 'kind = "foreign"'), ("bytes", "bytes = 1024 : i32", "bytes = 2048 : i32"),
                               ("address", "vmem_byte = 524288 : i32", "vmem_byte = 524289 : i32"),
                               ("channel", "channel = -1 : i32", "channel = 0 : i32"), ("register", "reg = 0 : i32", "reg = 64 : i32")):
            self.assertIn(old, text)
            mutations.append((name, text.replace(old, new, 1)))
        for name, changed in mutations:
            with self.subTest(mutation=name):
                self.rejected(replace_contract(machine, "tile", changed))

    def test_structured_fields_and_words_cannot_jointly_bypass_contract(self) -> None:
        structured = checked(self, lower(copy("bf16")), "--convert-atlas-to-llvm-calls")
        index = line_index(structured, 'atlas.source_op = "atlas.vload"')
        changed = rewrite_line(structured, index, lambda line: structured_word(line, "dst = 0 : i32", "dst = 2 : i32", 7, 5, 2))
        self.assertEqual(records(changed), records(structured))
        finalize_rejects(self, changed, "tile contract")

    def test_stream_rewrites_reject_padded_input(self) -> None:
        timed = lower(virtual_chain(1))
        for option in ("--insert-atlas-delays", "--schedule-atlas-stream"):
            with self.subTest(option=option):
                padded = run("atlas-opt", timed, option)
                self.assertNotEqual(padded.returncode, 0, padded.stdout)
                self.assertIn("delays come from the timing model", padded.stderr)


if __name__ == "__main__":
    unittest.main()
