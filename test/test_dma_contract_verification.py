"""Generated-artifact classification, source-derived DMA facts and captured-operand pending rules at every handoff.

ScalarCore.scala:489-509 forms a DMA command from S1 register values and diplomatic/memory/DMA.scala:223-229
queues the complete command, so generated schedules may reuse captured registers after launch. This checks
compiler admission, not physical completion or concurrent VMEM range safety.
"""

from __future__ import annotations

from itertools import product
import re
import unittest

from test_delay_insertion import program
from test_virtual_dma import copy
from test_virtual_dma_lowering import STAGING_WORD
from test_virtual_lowering import lower, run, virtual_chain
from verification_support import (
    INCOMPLETE, MARKER as VERSION, NOP, STREAM_REWRITES, TRANSFER, UNMARKED, UNSUPPORTED, BoundaryChecks, assert_boundaries,
    checked, constant, contract_text, dma_wait, drop_attribute, finalize_rejects, insert_after, last_constant_write,
    line_index, lines_of, records, remove_line, replace_contract, replace_line, rewrite_line, shift_constant,
    structured_word,
)

CONTRACT = "atlas.virtual_dma_contract"
CONTRACTS = ("dma", "mxu", "tile", "cfg", "source_memory", "buffer")
MARKERS = ('"resource-contract-v1"', '"resource-contract-v2"', '"resource-contract-v3"', '"resource-contract-v4"',
           '"dma-contract-v1"', '"resource-contract-v999"', '"resource-contract-v5 "', "1 : i32", "unit")


def remark(machine: str, marker: str) -> str:
    return machine.replace(VERSION, "atlas.generated_from_virtual" + ("" if marker == "unit" else " = " + marker), 1)


def expected_record(identity: int, direction: str, size: int, address: int = 0x80000000) -> dict:
    return dict(id=identity, direction=direction, channel=identity, staging_word=STAGING_WORD, dram_byte=address,
                size_bytes=size, staging_reg=4, dram_reg=7, size_reg=9)


def launch(machine: str, direction: str = "load") -> int:
    """Line of the source transfer's launch in lowered `copy()`: staging x4, DRAM x7, size x9."""
    return line_index(machine, '"atlas.dma"', f'direction = "{direction}"', TRANSFER)


class DMAContractVerificationTest(BoundaryChecks, unittest.TestCase):
    def test_lowering_records_source_values_and_complete_schema(self) -> None:
        added = copy().replace("%addr = arith.constant -2147483648 : i32",
                               "%a = arith.constant -32 : i32\n    %b = arith.constant -2147483616 : i32\n"
                               "    %half = arith.constant 1024 : i32\n    %addr = arith.addi %a, %b : i32")
        # Source addition proofs wrap i32 independently.
        added = added.replace("%size = arith.constant 2048 : i32", "%size = arith.addi %half, %half : i32")
        cases = [(copy(fmt, address=address), size, address) for fmt, size in (("fp8", 1024), ("bf16", 2048)) for address in (-2147483648, 0xfffff800)]
        for source, size, address in cases + [(added, 2048, 0x80000000)]:
            with self.subTest(size=size, address=address):
                machine = lower(source)
                self.assertIn(VERSION, machine)
                self.assertEqual(records(machine, "dma"), [expected_record(0, "load", size, address & 0xffffffff),
                                                           expected_record(1, "store", size, address & 0xffffffff)])
                self.assertEqual(contract_text(machine, "dma").count(" = "), 18)
                self.accepted(machine)

    def test_valid_but_wrong_command_values_fail_source_correspondence(self) -> None:
        machine = lower(copy())
        lines = machine.splitlines()
        load = launch(machine)
        address = next(i for i in range(load) if "dst = 7 : i32" in lines[i])
        size = next(i for i in range(load) if "dst = 9 : i32" in lines[i])
        staging = last_constant_write(machine, load, 4)
        mutations = [("DRAM address", address, "immediate = 0", "immediate = 32"), ("byte length", size, "immediate = 0", "immediate = -32"),
                     ("staging word", staging, lines[staging], shift_constant(lines[staging], 256)),
                     ("direction", load, 'direction = "load"', 'direction = "store"')]
        changes = [(name, replace_line(machine, index, old, new)) for name, index, old, new in mutations]
        # A second configuration sets a nonzero DRAM upper word before the launch.
        config = line_index(machine, '"atlas.dma_config"')
        changes.append(("DRAM upper word", insert_after(machine, config, [
            *constant(15, 1), ("dma_config", "channel = 0 : i32, base_reg = 15 : i32")])))
        for role, field, value in (("staging", "reg", 5), ("DRAM", "dram", 6), ("size", "size", 8)):
            changes.append((role + " register", rewrite_line(machine, load, lambda line: re.sub(rf"\b{field} = \d+ : i32", f"{field} = {value} : i32", line))))
        changes.append(("channel", replace_line(replace_line(machine, load, "channel = 0", "channel = 1"), load + 1, "channel = 0", "channel = 1")))
        changes = [(name, changed, "DMA contract") for name, changed in changes]
        changes += [("unknown DRAM upper word", machine.replace("base_reg = 0 : i32", "base_reg = 15 : i32"), "DMA contract cannot prove captured DRAM upper word"),
                    ("unknown DRAM byte address", machine.replace('dst = 10 : i32, immediate = 524288 : i32, kind = "lui"', 'dst = 10 : i32, immediate = 524288 : i32, kind = "auipc"'),
                     "DMA contract cannot prove captured DRAM byte address")]
        for name, changed, diagnostic in changes:
            with self.subTest(mutation=name):
                self.assertNotEqual(changed, machine)
                self.assertEqual(contract_text(changed, "dma"), contract_text(machine, "dma"))
                self.rejected(changed, diagnostic)

    def test_generated_artifact_classification_at_every_consumer(self) -> None:
        timed, untimed = lower(copy()), lower(copy(), timed=False)
        structured = checked(self, timed, "--convert-atlas-to-llvm-calls")
        self.accepted(timed)
        assert_boundaries(self, untimed, STREAM_REWRITES)
        cases = [(f"marker {marker}", lambda text, marker=marker: remark(text, marker), UNSUPPORTED) for marker in MARKERS]
        cases.append(("unmarked", lambda text: drop_attribute(text, "atlas.generated_from_virtual"), UNMARKED))
        cases += [(f"missing {name}", lambda text, name=name: drop_attribute(text, f"atlas.virtual_{name}_contract"),
                   f"{INCOMPLETE} atlas.virtual_{name}_contract") for name in CONTRACTS]
        cases.append(("untimed", lambda text: drop_attribute(drop_attribute(text, "atlas.timing_state"), "atlas.timing_provider")
                      if "atlas.timing_provider" in text else drop_attribute(text, "atlas.timing_state"), f"{INCOMPLETE} an explicit atlas.timing_state"))
        for name, mutate, diagnostic in cases:
            with self.subTest(name):
                self.assertNotEqual(mutate(timed), timed)
                self.rejected(mutate(timed), diagnostic)
                assert_boundaries(self, mutate(untimed), STREAM_REWRITES, rejects=diagnostic)
                finalize_rejects(self, mutate(structured), diagnostic)
        # Tags alone, and each contract alone on an untagged stream, are generated metadata too.
        body = timed.split("\n", 1)[1]
        self.rejected("module {\n" + body, UNMARKED)
        untagged = re.sub(r" \{atlas\.[^}]*\} : \(!atlas\.state\)", " : (!atlas.state)", body)
        self.assertNotIn("atlas.virtual_", untagged)
        for name in CONTRACTS:
            with self.subTest(alone=name):
                self.rejected(f"module attributes {{atlas.virtual_{name}_contract = []}} {{\n" + untagged, UNMARKED)

    def test_hand_written_stream_without_metadata_skips_generated_checks(self) -> None:
        source = program([*constant(31, 0), ("jump", 'kind = "jalr", dst = 0 : i32, base = 31 : i32, offset = 0 : i32'), NOP])
        for tool, options in (("atlas-emit", ()), ("atlas-opt", ("--convert-atlas-to-llvm",))):
            with self.subTest(tool=tool):
                self.assertTrue(checked(self, source, *options, tool=tool))
        result = run("atlas-opt", source, "--verify-atlas-generated-schedule")
        self.assertNotEqual(result.returncode, 0, result.stdout)
        self.assertIn("expected an Atlas virtual-to-machine artifact", result.stderr)

    def test_int32_max_transfer_id_is_rejected_without_asserting(self) -> None:
        machine = lower(copy())
        self.rejected(machine.replace(f"{TRANSFER} = 0 : i32", f"{TRANSFER} = 2147483647 : i32"), "DMA contract has no source record for transfer id")

    def test_strict_record_validation(self) -> None:
        machine = lower(copy())
        contract = contract_text(machine, "dma")
        replaced = lambda old, new: replace_contract(machine, "dma", contract.replace(old, new, 1))
        mutations = [
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

    def test_structured_fields_and_words_cannot_jointly_bypass_contract(self) -> None:
        structured = checked(self, lower(copy()), "--convert-atlas-to-llvm-calls")
        index = line_index(structured, 'atlas.source_op = "atlas.alu_imm"', "dst = 7 : i32")
        changed = rewrite_line(structured, index, lambda line: structured_word(line, "immediate = 0", "immediate = 32", 20, 12, 32))
        self.assertEqual(contract_text(changed, "dma"), contract_text(structured, "dma"))
        finalize_rejects(self, changed, "DMA contract")

    def test_captured_registers_are_free_after_launch_but_not_before(self) -> None:
        machine = lower(copy())
        for direction, register, kind in product(("load", "store"), (4, 7, 9), ("alu_reg", "alu_imm", "upper")):
            fields = {"alu_reg": f'kind = "add", dst = {register} : i32, lhs = 0 : i32, rhs = 0 : i32',
                      "alu_imm": f'kind = "addi", dst = {register} : i32, src = 0 : i32, immediate = 1 : i32',
                      "upper": f'kind = "lui", dst = {register} : i32, immediate = 33 : i32'}[kind]
            with self.subTest(direction=direction, register=register, kind=kind):
                self.accepted(insert_after(machine, launch(machine, direction), [(kind, fields)]))
        for register, field, value in ((4, "staging word", 1), (7, "DRAM byte address", 1), (9, "byte length", 32)):
            with self.subTest(before=register):
                clobber = ("alu_imm", f'kind = "addi", dst = {register} : i32, src = 0 : i32, immediate = {value} : i32')
                self.rejected(insert_after(machine, launch(machine) - 1, [clobber]), f"DMA contract captured {field} mismatch")

    def test_pending_configuration_memory_and_control_restrictions_remain(self) -> None:
        machine = lower(copy())
        pending = "unexpected instruction while DMA is pending"
        forbidden = (((("dma_config", "channel = 0 : i32, base_reg = 5 : i32"),), pending),
                     ((("scalar_load", 'kind = "lw", dst = 4 : i32, base = 7 : i32, offset = 0 : i32'),
                       ("delay", 'cycles = 8 : i32, atlas.delay_reason = "scalar_load_completion"')), pending),
                     ((("branch", 'kind = "beq", lhs = 0 : i32, rhs = 0 : i32, offset_bytes = 2 : i32'), NOP), pending),
                     ((("dma", f'direction = "load", channel = 0 : i32, reg = 4 : i32, dram = 7 : i32, size = 9 : i32, {TRANSFER} = 2 : i32'),),
                      "another DMA launch while its channel is pending"))
        for work, message in forbidden:
            with self.subTest(work=work[0][0]):
                self.rejected(insert_after(machine, launch(machine), work), message)

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
        self.assertEqual(contract_text(machine, "dma"), "[]")
        self.accepted(machine)
        launches, waits = lines_of(machine, "dma"), lines_of(machine, "dma_wait")
        self.assertTrue(launches and len(launches) == len(waits))
        self.assertNotIn(TRANSFER, machine)
        self.assertEqual([w - l for l, w in zip(launches, waits)], [1] * len(launches))
        for index in launches:
            with self.subTest(overlapped=index):
                self.accepted(insert_after(machine, index, [NOP]))
        self.rejected(remove_line(machine, waits[0]), "another DMA launch while its channel is pending")
        self.rejected("\n".join(machine.splitlines()[:waits[-1]] + ["}"]), "generated DMA has no matching DMA.WAIT")
        self.rejected(insert_after(machine, launches[0], [dma_wait(channel=1, identity=None)]), "DMA.WAIT has no pending DMA transfer")


if __name__ == "__main__":
    unittest.main()
