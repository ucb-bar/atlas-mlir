"""Source-derived DMA facts and captured-operand pending rules at every machine and LLVM handoff.

ScalarCore.scala:489-509 forms a DMA command from S1 register values and diplomatic/memory/DMA.scala:223-229
queues the complete command, so generated schedules may reuse captured registers after launch. This checks
compiler admission, not physical completion or concurrent VMEM range safety.
"""

from __future__ import annotations

from itertools import product
import re
import unittest

from test_virtual_dma import copy
from test_virtual_dma_lowering import STAGING_WORD
from test_virtual_lowering import lower, virtual_chain
from verification_support import (
    DELAY, INCOMPLETE, MARKER as VERSION, NOP, STREAM_REWRITES, TRANSFER, UNMARKED, UNSUPPORTED, assert_boundaries,
    checked, contract_text, dma_wait, drop_attribute, finalize_rejects, insert_after, line_index, lines_of, records,
    remove_line, replace_contract, replace_line, rewrite_line, structured_word,
)

CONTRACT = "atlas.virtual_dma_contract"


def expected_record(identity: int, direction: str, size: int, address: int = 0x80000000) -> dict:
    return dict(id=identity, direction=direction, channel=identity, staging_word=STAGING_WORD, dram_byte=address,
                size_bytes=size, staging_reg=4, dram_reg=7, size_reg=9)


def launch(machine: str, direction: str = "load") -> int:
    """Line of the source transfer's launch in lowered `copy()`: staging x4, DRAM x7, size x9."""
    return line_index(machine, '"atlas.dma"', f'direction = "{direction}"', TRANSFER)


class DMAContractVerificationTest(unittest.TestCase):
    def accepted(self, machine: str) -> None:
        assert_boundaries(self, machine)

    def rejected(self, machine: str, diagnostic: str = "") -> None:
        assert_boundaries(self, machine, rejects=diagnostic)

    def test_lowering_records_source_values_and_complete_schema(self) -> None:
        for fmt, size in (("fp8", 1024), ("bf16", 2048)):
            for address in (-2147483648, 0xfffff800):
                with self.subTest(fmt=fmt, address=address):
                    machine = lower(copy(fmt, address=address))
                    self.assertIn(VERSION, machine)
                    expected = [expected_record(0, "load", size, address & 0xffffffff), expected_record(1, "store", size, address & 0xffffffff)]
                    self.assertEqual(records(machine, "dma"), expected)
                    self.assertEqual(contract_text(machine, "dma").count(" = "), 18)
                    self.accepted(machine)

    def test_source_addition_proofs_wrap_i32_independently(self) -> None:
        source = copy().replace("%addr = arith.constant -2147483648 : i32",
                                "%a = arith.constant -32 : i32\n    %b = arith.constant -2147483616 : i32\n"
                                "    %half = arith.constant 1024 : i32\n    %addr = arith.addi %a, %b : i32")
        machine = lower(source.replace("%size = arith.constant 2048 : i32", "%size = arith.addi %half, %half : i32"))
        self.assertEqual(records(machine, "dma"), [expected_record(0, "load", 2048), expected_record(1, "store", 2048)])
        self.accepted(machine)

    def test_valid_but_wrong_command_values_fail_source_correspondence(self) -> None:
        machine = lower(copy())
        lines = machine.splitlines()
        load = launch(machine)
        address = next(i for i in range(load) if "dst = 7 : i32" in lines[i])
        size = next(i for i in range(load) if "dst = 9 : i32" in lines[i])
        staging = max(i for i in range(load) if '"atlas.alu_imm"' in lines[i] and "dst = 4 : i32" in lines[i])
        upper = next(i for i in range(load) if "dst = 5 : i32" in lines[i])
        mutations = [("DRAM address", address, "immediate = 0", "immediate = 32"), ("byte length", size, "immediate = 0", "immediate = -32"),
                     ("staging word", staging, "immediate = 0", "immediate = 256"), ("DRAM upper word", upper, "immediate = 0", "immediate = 1"),
                     ("direction", load, 'direction = "load"', 'direction = "store"')]
        changes = [(name, replace_line(machine, index, old, new)) for name, index, old, new in mutations]
        for role, field, value in (("staging", "reg", 5), ("DRAM", "dram", 6), ("size", "size", 8)):
            changes.append((role + " register", rewrite_line(machine, load, lambda line: re.sub(rf"\b{field} = \d+ : i32", f"{field} = {value} : i32", line))))
        changes.append(("channel", replace_line(replace_line(machine, load, "channel = 0", "channel = 1"), load + 1, "channel = 0", "channel = 1")))
        for name, changed in changes:
            with self.subTest(mutation=name):
                self.assertEqual(contract_text(changed, "dma"), contract_text(machine, "dma"))
                self.rejected(changed, "DMA contract")

    def test_missing_duplicate_and_foreign_tags_fail(self) -> None:
        machine = lower(copy())
        tagged = [i for i, line in enumerate(machine.splitlines()) if TRANSFER in line]
        for index in tagged:
            with self.subTest(missing=index):
                changed = rewrite_line(machine, index, lambda line: re.sub(rf"{re.escape(TRANSFER)} = \d+ : i32, ", "", line, count=1))
                self.assertNotIn(TRANSFER, changed.splitlines()[index])
                self.assertIn("atlas.virtual_tile_command =", changed.splitlines()[index])
                self.rejected(changed)
        for identity in (0, 999):
            with self.subTest(identity=identity):
                self.rejected(machine.replace(f"{TRANSFER} = 1 : i32", f"{TRANSFER} = {identity} : i32"))
        for value in ("-1 : i32", "0 : i64", '"bad"'):
            with self.subTest(tag=value):
                self.rejected(replace_line(machine, tagged[0], f"{TRANSFER} = 0 : i32", f"{TRANSFER} = {value}"))

    def test_strict_marker_and_record_validation(self) -> None:
        machine = lower(copy())
        contract = contract_text(machine, "dma")
        replaced = lambda old, new: replace_contract(machine, "dma", contract.replace(old, new, 1))
        mutations = [
            ("missing contract", drop_attribute(machine, CONTRACT), f"{INCOMPLETE} {CONTRACT}"),
            ("missing version", machine.replace(VERSION + ", ", ""), UNMARKED),
            ("unknown version", machine.replace(VERSION, 'atlas.generated_from_virtual = "resource-contract-v999"'), UNSUPPORTED),
            ("malformed version", machine.replace(VERSION, "atlas.generated_from_virtual = 1 : i32"), UNSUPPORTED),
            ("unit legacy", machine.replace(VERSION, "atlas.generated_from_virtual"), UNSUPPORTED),
            ("DMA legacy", machine.replace("resource-contract-v5", "dma-contract-v1"), UNSUPPORTED),
            ("nonarray", replace_contract(machine, "dma", '"bad"'), f"requires an {CONTRACT} array"),
            ("nondictionary", replace_contract(machine, "dma", "[0 : i32]"), "DMA contract"),
            ("empty explicit contract", replace_contract(machine, "dma", "[]"), "DMA contract"),
            ("missing field", replaced("size_reg = 9 : i32, ", ""), "DMA contract"),
            ("wrong integer type", replaced("staging_word = 131072 : i32", "staging_word = 131072 : i64"), "DMA contract"),
            ("wrong direction type", replaced('direction = "load"', "direction = 0 : i32"), "DMA contract"),
            ("foreign field", replaced("{", "{foreign = 0 : i32, "), "DMA contract"),
            ("duplicate record", replaced("id = 1 : i32", "id = 0 : i32"), "DMA contract"),
            ("unsorted records", replace_contract(machine, "dma", "[" + ", ".join(reversed(re.findall(r"\{[^{}]*\}", contract))) + "]"), "DMA contract"),
        ]
        for name, changed, diagnostic in mutations:
            with self.subTest(mutation=name):
                self.assertNotEqual(changed, machine)
                self.rejected(changed, diagnostic)

    def test_empty_contract_remains_valid(self) -> None:
        machine = lower(virtual_chain(1))
        self.assertIn(VERSION, machine)
        self.assertEqual(contract_text(machine, "dma"), "[]")
        self.accepted(machine)

    def test_unknown_captured_addresses_are_rejected(self) -> None:
        machine = lower(copy())
        for name, changed in (("DRAM upper word", machine.replace("base_reg = 5 : i32", "base_reg = 15 : i32")),
                              ("DRAM byte address", machine.replace('dst = 10 : i32, immediate = 524288 : i32, kind = "lui"',
                                                                    'dst = 10 : i32, immediate = 524288 : i32, kind = "auipc"'))):
            with self.subTest(capture=name):
                self.assertNotEqual(changed, machine)
                self.assertEqual(contract_text(changed, "dma"), contract_text(machine, "dma"))
                self.rejected(changed, "DMA contract cannot prove captured " + name)

    def test_stream_rewrites_preserve_and_recheck_the_contract(self) -> None:
        untimed = lower(copy(), timed=False)
        for tool, options in STREAM_REWRITES:
            with self.subTest(options=options):
                rewritten = checked(self, untimed, *options)
                self.assertIn(VERSION, rewritten)
                self.assertEqual(contract_text(rewritten, "dma"), contract_text(untimed, "dma"))
                self.accepted(rewritten)
                address = line_index(rewritten, '"atlas.upper"', "dst = 10 : i32")
                self.rejected(replace_line(rewritten, address, "immediate = 524288", "immediate = 524289"), "DMA contract")

    def test_structured_fields_and_words_cannot_jointly_bypass_contract(self) -> None:
        structured = checked(self, lower(copy()), "--convert-atlas-to-llvm-calls")
        index = line_index(structured, 'atlas.source_op = "atlas.alu_imm"', "dst = 7 : i32")
        changed = rewrite_line(structured, index, lambda line: structured_word(line, "immediate = 0", "immediate = 32", 20, 12, 32))
        self.assertEqual(contract_text(changed, "dma"), contract_text(structured, "dma"))
        finalize_rejects(self, changed, "DMA contract")

    def test_all_scalar_write_classes_can_reuse_captured_vmem_dram_and_size_registers(self) -> None:
        machine = lower(copy())
        self.accepted(machine)
        for direction, register, kind in product(("load", "store"), (4, 7, 9), ("alu_reg", "alu_imm", "upper")):
            fields = {"alu_reg": f'kind = "add", dst = {register} : i32, lhs = 0 : i32, rhs = 0 : i32',
                      "alu_imm": f'kind = "addi", dst = {register} : i32, src = 0 : i32, immediate = 1 : i32',
                      "upper": f'kind = "lui", dst = {register} : i32, immediate = 33 : i32'}[kind]
            with self.subTest(direction=direction, register=register, kind=kind):
                self.accepted(insert_after(machine, launch(machine, direction), [(kind, fields)]))

    def test_helper_writes_before_launch_change_the_captured_command(self) -> None:
        machine = lower(copy())
        for register, field, value in ((4, "staging word", 1), (7, "DRAM byte address", 1), (9, "byte length", 32)):
            with self.subTest(register=register):
                clobber = ("alu_imm", f'kind = "addi", dst = {register} : i32, src = 0 : i32, immediate = {value} : i32')
                self.rejected(insert_after(machine, launch(machine) - 1, [clobber]), f"DMA contract captured {field} mismatch")

    def test_pending_configuration_memory_and_control_restrictions_remain(self) -> None:
        machine = lower(copy())
        forbidden = (("configuration", (("dma_config", "channel = 0 : i32, base_reg = 5 : i32"),)),
                     ("VMEM load", (("vload", 'dst = 0 : i32, base = 4 : i32, offset = 0 : i32, format = "raw"'), DELAY)),
                     ("VMEM store", (("vstore", 'src = 0 : i32, base = 4 : i32, offset = 0 : i32, format = "raw"'), DELAY)),
                     ("scalar memory", (("scalar_load", 'kind = "lw", dst = 4 : i32, base = 7 : i32, offset = 0 : i32'),
                                        ("delay", 'cycles = 8 : i32, atlas.delay_reason = "scalar_load_completion"'))),
                     ("branch", (("branch", 'kind = "beq", lhs = 0 : i32, rhs = 0 : i32, offset_bytes = 2 : i32'), NOP)))
        for name, work in forbidden:
            with self.subTest(name=name):
                message = "DMA memory conflict" if name.startswith("VMEM") else "unexpected instruction while DMA is pending"
                self.rejected(insert_after(machine, launch(machine), work), message)

    def test_same_channel_cannot_be_relaunched_before_its_completion(self) -> None:
        machine = lower(copy())
        relaunch = ("dma", f'direction = "load", channel = 0 : i32, reg = 4 : i32, dram = 7 : i32, size = 9 : i32, {TRANSFER} = 2 : i32')
        self.rejected(insert_after(machine, launch(machine), [relaunch]), "another DMA launch while its channel is pending")

    def test_completion_channel_and_identity_must_match_the_pending_transfer(self) -> None:
        machine = lower(copy())
        index = launch(machine) + 1
        self.assertEqual(lines_of(machine, "dma_wait", f"{TRANSFER} = 0 : i32"), [index])
        for old, new, expected in (("channel = 0 : i32", "channel = 1 : i32", "DMA.WAIT has no pending DMA transfer"),
                                   (f"{TRANSFER} = 0 : i32", f"{TRANSFER} = 1 : i32", "DMA.WAIT must match"),
                                   (f", {TRANSFER} = 0 : i32", "", "DMA.WAIT must match")):
            with self.subTest(completion=new or "untagged"):
                self.rejected(replace_line(machine, index, old, new), expected)

    def test_untagged_boundary_transfers_use_the_same_pending_policy(self) -> None:
        # Implicit boundary transfers carry no transfer ID; their completion is the next
        # same-channel DMA.WAIT without an ID, not necessarily the next instruction.
        machine = lower(virtual_chain(1))
        launches, waits = lines_of(machine, "dma"), lines_of(machine, "dma_wait")
        self.assertTrue(launches and len(launches) == len(waits))
        self.assertNotIn(TRANSFER, machine)
        self.assertEqual([w - l for l, w in zip(launches, waits)], [1] * len(launches))
        for index in launches:
            with self.subTest(overlapped=index):
                self.accepted(insert_after(machine, index, [NOP]))
        self.rejected(remove_line(machine, waits[0]), "another DMA launch while its channel is pending")
        self.rejected("\n".join(machine.splitlines()[:waits[-1]] + ["}"]), "generated DMA has no matching DMA.WAIT")
        self.rejected(replace_line(machine, waits[0], "channel = 0 : i32", "channel = 1 : i32"), "DMA.WAIT has no pending DMA transfer")
        self.rejected(insert_after(machine, launches[0], [dma_wait(channel=1, identity=None)]), "DMA.WAIT has no pending DMA transfer")
        self.rejected(replace_line(machine, waits[0], "atlas.virtual_cfg_block", f"{TRANSFER} = 0 : i32, atlas.virtual_cfg_block"), "DMA.WAIT must match")


if __name__ == "__main__":
    unittest.main()
