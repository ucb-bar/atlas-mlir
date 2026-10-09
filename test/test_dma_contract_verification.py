"""Source-derived explicit DMA facts survive machine and LLVM handoffs."""

from __future__ import annotations

from itertools import product
import re
import unittest

from generated_fixture import CLASSIFICATION, INCOMPLETE, MARKER as VERSION, UNMARKED, UNSUPPORTED, insert_after
from test_virtual_dma import copy
from test_virtual_dma_lowering import MARKER, STAGING_WORD, independent_work
from test_virtual_lowering import BIN, lower, run, virtual_chain


CONTRACT = "atlas.virtual_dma_contract"
CONTRACT_RE = re.compile(r'atlas\.virtual_dma_contract = (\[[^\]]*\])')
RECORD_RE = re.compile(r'\{([^{}]*)\}')
FIELD_RE = re.compile(r'(\w+) = (?:(-?\d+) : i32|"([^"]*)")')
BOUNDARIES = (
    ("atlas-opt", ("--verify-atlas-generated-schedule",)),
    ("atlas-emit", ()),
    ("atlas-opt", ("--convert-atlas-to-llvm",)),
    ("atlas-opt", ("--convert-atlas-to-llvm-calls",)),
)


def contract_text(machine: str) -> str:
    match = CONTRACT_RE.search(machine)
    if match is None:
        raise AssertionError("lowering must retain the source DMA contract")
    return match[1]


def records(machine: str) -> list[dict]:
    return [{name: int(integer) & 0xffffffff if integer else string
             for name, integer, string in FIELD_RE.findall(record)}
            for record in RECORD_RE.findall(contract_text(machine))]


def replace_contract(machine: str, value: str) -> str:
    return CONTRACT_RE.sub(lambda _: f"{CONTRACT} = {value}", machine, count=1)


def replace_line(machine: str, index: int, line: str) -> str:
    lines = machine.splitlines()
    lines[index] = line
    return "\n".join(lines) + "\n"


def expected_record(identity: int, direction: str, size: int, address: int = 0x80000000) -> dict:
    return dict(id=identity, direction=direction, channel=identity,
                staging_word=STAGING_WORD, dram_byte=address, size_bytes=size,
                staging_reg=4, dram_reg=7, size_reg=9)


def line_index(machine: str, *needles: str) -> int:
    return next(i for i, line in enumerate(machine.splitlines()) if all(n in line for n in needles))


def launch(machine: str, direction: str = "load") -> int:
    return line_index(machine, '"atlas.dma"', f'direction = "{direction}"', MARKER)


class DMAContractVerificationTest(unittest.TestCase):
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
                for unintended in CLASSIFICATION:
                    if unintended not in diagnostic:
                        self.assertNotIn(unintended, result.stderr)

    def test_lowering_records_source_values_and_complete_schema(self) -> None:
        for fmt, size in (("fp8", 1024), ("bf16", 2048)):
            for address in (-2147483648, 0xfffff800):
                with self.subTest(fmt=fmt, address=address):
                    machine = lower(copy(fmt, address=address))
                    self.assertIn(VERSION, machine)
                    expected = [expected_record(0, "load", size, address & 0xffffffff),
                                expected_record(1, "store", size, address & 0xffffffff)]
                    self.assertEqual(records(machine), expected)
                    self.assertEqual(len(FIELD_RE.findall(contract_text(machine))), 18)
                    self.accepted(machine)

    def test_source_addition_proofs_wrap_i32_independently(self) -> None:
        source = copy().replace(
            "%addr = arith.constant -2147483648 : i32",
            "%a = arith.constant -32 : i32\n"
            "    %b = arith.constant -2147483616 : i32\n"
            "    %half = arith.constant 1024 : i32\n"
            "    %addr = arith.addi %a, %b : i32",
        ).replace("%size = arith.constant 2048 : i32",
                  "%size = arith.addi %half, %half : i32")
        machine = lower(source)
        self.assertEqual(records(machine), [expected_record(0, "load", 2048),
                                            expected_record(1, "store", 2048)])
        self.accepted(machine)

    def test_valid_but_wrong_command_values_fail_source_correspondence(self) -> None:
        machine = lower(copy())
        lines = machine.splitlines()
        launch = next(i for i, line in enumerate(lines)
                      if '"atlas.dma"' in line and MARKER in line)
        wait = launch + 1
        address = next(i for i in range(launch) if 'dst = 7 : i32' in lines[i])
        size = next(i for i in range(launch) if 'dst = 9 : i32' in lines[i])
        staging = max(i for i in range(launch)
                      if '"atlas.alu_imm"' in lines[i] and 'dst = 4 : i32' in lines[i])
        upper = next(i for i in range(launch) if 'dst = 5 : i32' in lines[i])
        mutations = [
            ("DRAM address", address, lines[address].replace("immediate = 0", "immediate = 32")),
            ("byte length", size, lines[size].replace("immediate = 0", "immediate = -32")),
            ("staging word", staging, lines[staging].replace("immediate = 0", "immediate = 256")),
            ("DRAM upper word", upper, lines[upper].replace("immediate = 0", "immediate = 1")),
            ("direction", launch, lines[launch].replace('direction = "load"', 'direction = "store"')),
        ]
        for role, field, value in (("staging", "reg", 5), ("DRAM", "dram", 6), ("size", "size", 8)):
            mutations.append((role + " register", launch,
                              re.sub(rf'\b{field} = \d+ : i32', f'{field} = {value} : i32', lines[launch])))
        for name, index, replacement in mutations:
            with self.subTest(mutation=name):
                changed = replace_line(machine, index, replacement)
                self.assertNotEqual(changed, machine)
                self.assertEqual(contract_text(changed), contract_text(machine))
                self.rejected(changed, "DMA contract")
        changed = replace_line(machine, launch, lines[launch].replace("channel = 0", "channel = 1"))
        changed = replace_line(changed, wait, lines[wait].replace("channel = 0", "channel = 1"))
        self.rejected(changed, "DMA contract")

    def test_missing_duplicate_and_foreign_tags_fail(self) -> None:
        machine = lower(copy())
        lines = machine.splitlines()
        tagged = [i for i, line in enumerate(lines) if MARKER in line]
        for index in tagged:
            with self.subTest(missing=index):
                replacement, removed = re.subn(
                    rf'{re.escape(MARKER)} = \d+ : i32, ', "", lines[index], count=1)
                self.assertEqual(removed, 1)
                self.assertNotIn(MARKER, replacement)
                self.assertIn("atlas.virtual_tile_command =", replacement)
                changed = replace_line(machine, index, replacement)
                self.assertNotEqual(changed, machine)
                self.rejected(changed)
        for identity in (0, 999):
            with self.subTest(identity=identity):
                changed = "\n".join(line.replace(f"{MARKER} = 1 : i32", f"{MARKER} = {identity} : i32")
                                    if MARKER in line else line for line in lines)
                self.rejected(changed)
        for value in ('-1 : i32', '0 : i64', '"bad"'):
            with self.subTest(tag=value):
                changed = replace_line(machine, tagged[0], lines[tagged[0]].replace(
                    f"{MARKER} = 0 : i32", f"{MARKER} = {value}"))
                self.rejected(changed)

    def test_strict_marker_and_record_validation(self) -> None:
        machine = lower(copy())
        contract = contract_text(machine)
        mutations = [
            ("missing contract", CONTRACT_RE.sub("", machine).replace(", ,", ",").replace(", }", "}"), f"{INCOMPLETE} {CONTRACT}"),
            ("missing version", machine.replace(VERSION + ", ", ""), UNMARKED),
            ("unknown version", machine.replace(VERSION, 'atlas.generated_from_virtual = "resource-contract-v999"'), UNSUPPORTED),
            ("malformed version", machine.replace(VERSION, "atlas.generated_from_virtual = 1 : i32"), UNSUPPORTED),
            ("unit legacy", machine.replace(VERSION, "atlas.generated_from_virtual"), UNSUPPORTED),
            ("DMA legacy", machine.replace("resource-contract-v4", "dma-contract-v1"), UNSUPPORTED),
            ("nonarray", replace_contract(machine, '"bad"'), f"requires an {CONTRACT} array"),
            ("nondictionary", replace_contract(machine, "[0 : i32]"), "DMA contract"),
            ("empty explicit contract", replace_contract(machine, "[]"), "DMA contract"),
            ("missing field", replace_contract(machine, contract.replace("size_reg = 9 : i32, ", "", 1)), "DMA contract"),
            ("wrong integer type", replace_contract(machine, contract.replace("staging_word = 131072 : i32", "staging_word = 131072 : i64", 1)), "DMA contract"),
            ("wrong direction type", replace_contract(machine, contract.replace('direction = "load"', "direction = 0 : i32", 1)), "DMA contract"),
            ("foreign field", replace_contract(machine, contract.replace("{", "{foreign = 0 : i32, ", 1)), "DMA contract"),
            ("duplicate record", replace_contract(machine, contract.replace("id = 1 : i32", "id = 0 : i32", 1)), "DMA contract"),
            ("unsorted records", replace_contract(machine, "[" + ", ".join(reversed(re.findall(r'\{[^{}]*\}', contract))) + "]"), "DMA contract"),
        ]
        for name, changed, diagnostic in mutations:
            with self.subTest(mutation=name):
                self.assertNotEqual(changed, machine)
                self.rejected(changed, diagnostic)

    def test_empty_contract_remains_valid(self) -> None:
        machine = lower(virtual_chain(1))
        self.assertIn(VERSION, machine)
        self.assertEqual(contract_text(machine), "[]")
        self.accepted(machine)

    def test_unknown_captured_addresses_are_rejected(self) -> None:
        machine = lower(copy())
        for name, changed in (
            ("DRAM upper word", machine.replace("base_reg = 5 : i32", "base_reg = 15 : i32")),
            ("DRAM byte address", machine.replace('dst = 10 : i32, immediate = 524288 : i32, kind = "lui"',
                                                    'dst = 10 : i32, immediate = 524288 : i32, kind = "auipc"')),
        ):
            with self.subTest(capture=name):
                self.assertNotEqual(changed, machine)
                self.assertEqual(contract_text(changed), contract_text(machine))
                self.rejected(changed, "DMA contract cannot prove captured " + name)

    def test_contract_survives_llvm_and_stream_rewrites(self) -> None:
        timed, untimed = lower(copy()), lower(copy(), timed=False)
        for option in (None, "--convert-atlas-to-llvm", "--convert-atlas-to-llvm-calls",
                       "--insert-atlas-delays", "--schedule-atlas-stream"):
            with self.subTest(option=option):
                # Stream rewrites take untimed input; LLVM handoffs require timed input.
                machine = untimed if option in ("--insert-atlas-delays", "--schedule-atlas-stream") else timed
                result = run("atlas-opt", machine, *((option,) if option else ()))
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertIn(VERSION, result.stdout)
                self.assertEqual(contract_text(result.stdout), contract_text(machine))
                if option == "--convert-atlas-to-llvm-calls":
                    final = run("atlas-opt", result.stdout, "--finalize-atlas-llvm-calls")
                    direct = run("atlas-opt", machine, "--convert-atlas-to-llvm")
                    self.assertEqual(final.returncode, 0, final.stderr)
                    self.assertEqual(final.stdout, direct.stdout)
                elif option in ("--insert-atlas-delays", "--schedule-atlas-stream"):
                    self.accepted(result.stdout)
                    lines = result.stdout.splitlines()
                    address = next(i for i, line in enumerate(lines)
                                   if '"atlas.upper"' in line and 'dst = 10 : i32' in line)
                    changed = replace_line(result.stdout, address, lines[address].replace(
                        "immediate = 524288", "immediate = 524289"))
                    self.assertNotEqual(changed, result.stdout)
                    self.assertEqual(contract_text(changed), contract_text(machine))
                    self.rejected(changed, "DMA contract")

    def test_structured_fields_and_words_cannot_jointly_bypass_contract(self) -> None:
        machine = lower(copy())
        result = run("atlas-opt", machine, "--convert-atlas-to-llvm-calls")
        self.assertEqual(result.returncode, 0, result.stderr)
        lines = result.stdout.splitlines()
        index = next(i for i, line in enumerate(lines)
                     if 'atlas.source_op = "atlas.alu_imm"' in line and 'dst = 7 : i32' in line)
        word = int(re.search(r'atlas.word = (-?\d+) : i32', lines[index])[1])
        changed_word = ((word & 0xffffffff) & ~(0xfff << 20)) | (32 << 20)
        replacement = lines[index].replace("immediate = 0", "immediate = 32")
        replacement = re.sub(r'atlas.word = -?\d+ : i32', f'atlas.word = {changed_word} : i32', replacement)
        changed = replace_line(result.stdout, index, replacement)
        self.assertNotEqual(changed, result.stdout)
        self.assertEqual(contract_text(changed), contract_text(machine))
        final = run("atlas-opt", changed, "--finalize-atlas-llvm-calls")
        self.assertNotEqual(final.returncode, 0, final.stdout)
        self.assertEqual(final.stdout, "")
        self.assertIn("DMA contract", final.stderr)

    def test_captured_scalar_register_reuse_preserves_source_contract(self) -> None:
        self.accepted(lower(independent_work()))
        machine = lower(copy())
        for direction, register in product(("load", "store"), (4, 7, 9)):
            with self.subTest(direction=direction, register=register):
                write = ("alu_imm", f'kind = "addi", dst = {register} : i32, src = 0 : i32, immediate = 1 : i32')
                self.accepted(insert_after(machine, launch(machine, direction), [write]))


if __name__ == "__main__":
    unittest.main()
