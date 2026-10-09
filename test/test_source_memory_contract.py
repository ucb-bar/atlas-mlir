"""Overlapping source DRAM effects keep their completion order through every handoff."""

from __future__ import annotations

import re
import unittest

from generated_fixture import INCOMPLETE
from test_virtual_lowering import lower, run


CONTRACT = "atlas.virtual_source_memory_contract"
ORDER = "source memory contract: overlapping source predecessor has not completed in this visit"
INCONSISTENT = "source memory contract has malformed or inconsistent source records"
LAUNCHES = ("atlas.virtual_dma_load_fp8", "atlas.virtual_dma_store_fp8")
STATE = "!atlas.virtual_state"
X, Y = 0x90000000, 0x90010000
BOUNDARIES = (("atlas-opt", ("--verify-atlas-generated-schedule",)), ("atlas-emit", ()),
              ("atlas-opt", ("--convert-atlas-to-llvm",)), ("atlas-opt", ("--convert-atlas-to-llvm-calls",)))
UNTIMED = (("atlas-opt", ("--verify-atlas-generated-schedule",)), ("atlas-emit", ("--allow-untimed",)))


def signed(value: int) -> int:
    return value - (1 << 32) if value >= 1 << 31 else value


def transfer(index: int, kind: str, address: str) -> str:
    state, launched, done, handle = f"%s{2 * index + 1}", f"%s{2 * index + 2}", f"%s{2 * index + 3}", f"%t{index}"
    if kind == "store":
        return (f'    {launched}, {handle} = "atlas.virtual_dma_store_fp8"({state}, %tile, {address}, %size) : ({STATE}, !atlas.virtual_fp8, i32, i32) -> ({STATE}, !atlas.virtual_dma_store)\n'
                f'    {done} = "atlas.virtual_dma_wait"({launched}, {handle}) : ({STATE}, !atlas.virtual_dma_store) -> {STATE}')
    return (f'    {launched}, {handle} = "atlas.virtual_dma_load_fp8"({state}, {address}, %size) : ({STATE}, i32, i32) -> ({STATE}, !atlas.virtual_dma_load_fp8)\n'
            f'    {done}, {handle}_value = "atlas.virtual_dma_await_fp8"({launched}, {handle}) : ({STATE}, !atlas.virtual_dma_load_fp8) -> ({STATE}, !atlas.virtual_fp8)')


def program(first: str, second: str, *, second_address: int = X) -> str:
    """Two completed transfers at X; a final store of the input tile to disjoint Y keeps its register distinct."""
    return f'''module {{
  func.func @memory() -> {STATE} attributes {{atlas.input_dram_base = 2432696320 : i64, atlas.output_dram_base = 2449473536 : i64}} {{
    %s0 = "atlas.virtual_start"() : () -> {STATE}
    %s1, %tile = "atlas.virtual_input_fp8"(%s0) {{index = 0 : i32}} : ({STATE}) -> ({STATE}, !atlas.virtual_fp8)
    %x = arith.constant {signed(X)} : i32
    %y = arith.constant {signed(second_address)} : i32
    %z = arith.constant {signed(Y)} : i32
    %size = arith.constant 1024 : i32
{transfer(0, first, "%x")}
{transfer(1, second, "%y")}
{transfer(2, "store", "%z")}
    return %s7 : {STATE}
  }}
}}
'''


LOOP = f'''module {{
  func.func @loop() -> {STATE} attributes {{atlas.input_dram_base = 2432696320 : i64, atlas.output_dram_base = 2449473536 : i64}} {{
    %s0 = "atlas.virtual_start"() : () -> {STATE}
    %s1, %tile = "atlas.virtual_input_fp8"(%s0) {{index = 0 : i32}} : ({STATE}) -> ({STATE}, !atlas.virtual_fp8)
    %zero = arith.constant 0 : i32
    cf.br ^loop(%s1, %zero : {STATE}, i32)
  ^loop(%state : {STATE}, %i : i32):
    %x = arith.constant {signed(X)} : i32
    %size = arith.constant 1024 : i32
    %one = arith.constant 1 : i32
    %limit = arith.constant 2 : i32
    %l1, %first = "atlas.virtual_dma_load_fp8"(%state, %x, %size) : ({STATE}, i32, i32) -> ({STATE}, !atlas.virtual_dma_load_fp8)
    %l2, %a = "atlas.virtual_dma_await_fp8"(%l1, %first) : ({STATE}, !atlas.virtual_dma_load_fp8) -> ({STATE}, !atlas.virtual_fp8)
    %l3, %second = "atlas.virtual_dma_load_fp8"(%l2, %x, %size) : ({STATE}, i32, i32) -> ({STATE}, !atlas.virtual_dma_load_fp8)
    %l4, %b = "atlas.virtual_dma_await_fp8"(%l3, %second) : ({STATE}, !atlas.virtual_dma_load_fp8) -> ({STATE}, !atlas.virtual_fp8)
    %next = arith.addi %i, %one : i32
    %more = arith.cmpi slt, %next, %limit : i32
    cf.cond_br %more, ^loop(%l4, %next : {STATE}, i32), ^exit(%l4 : {STATE})
  ^exit(%final : {STATE}):
    %ex = arith.constant {signed(X)} : i32
    %ez = arith.constant {signed(Y)} : i32
    %esize = arith.constant 1024 : i32
    %e1, %store = "atlas.virtual_dma_store_fp8"(%final, %tile, %ex, %esize) : ({STATE}, !atlas.virtual_fp8, i32, i32) -> ({STATE}, !atlas.virtual_dma_store)
    %e2 = "atlas.virtual_dma_wait"(%e1, %store) : ({STATE}, !atlas.virtual_dma_store) -> {STATE}
    %e3, %load = "atlas.virtual_dma_load_fp8"(%e2, %ex, %esize) : ({STATE}, i32, i32) -> ({STATE}, !atlas.virtual_dma_load_fp8)
    %e4, %value = "atlas.virtual_dma_await_fp8"(%e3, %load) : ({STATE}, !atlas.virtual_dma_load_fp8) -> ({STATE}, !atlas.virtual_fp8)
    %e5, %keep = "atlas.virtual_dma_store_fp8"(%e4, %tile, %ez, %esize) : ({STATE}, !atlas.virtual_fp8, i32, i32) -> ({STATE}, !atlas.virtual_dma_store)
    %e6 = "atlas.virtual_dma_wait"(%e5, %keep) : ({STATE}, !atlas.virtual_dma_store) -> {STATE}
    return %e6 : {STATE}
  }}
}}
'''


def operations(machine: str) -> list[str]:
    return [line for line in machine.splitlines() if ": (!atlas.state)" in line]


def source_operations(machine: str) -> list[tuple[int, int, str]]:
    pattern = r'\{block = (\d+) : i32, id = (\d+) : i32, mxu_commands = array<i32[^>]*>, name = "([^"]+)"'
    return [(int(block), int(identity), name) for block, identity, name in re.findall(pattern, machine)]


def transfer_groups(machine: str) -> list[set[int]]:
    """Each explicit transfer's launch and completion source identities, in source order."""
    return [{identity, identity + 1} for _, identity, name in source_operations(machine) if name in LAUNCHES]


def reorder(machine: str, ordered: list[str]) -> str:
    lines = machine.splitlines()
    first = next(n for n, line in enumerate(lines) if ": (!atlas.state)" in line)
    last = max(n for n, line in enumerate(lines) if ": (!atlas.state)" in line)
    previous = re.search(r"%(\w+) =", next(line for line in lines if '"atlas.start"' in line))[1]
    rebuilt = []
    for index, line in enumerate(ordered):
        line = re.sub(r"^\s*%\w+ =", f"  %r{index} =", line)
        rebuilt.append(re.sub(r'("atlas\.[^"]+"\()%\w+(\))', rf"\g<1>%{previous}\g<2>", line))
        previous = f"r{index}"
    return "\n".join(lines[:first] + rebuilt + lines[last + 1:]) + "\n"


def swap_sources(machine: str, first: set[int], second: set[int]) -> str:
    ops = operations(machine)
    tag = re.compile(r"atlas\.virtual_cfg_source = (\d+) : i32")
    owner = lambda line: int(tag.search(line)[1]) if tag.search(line) else None
    start = next(n for n, line in enumerate(ops) if owner(line) in first)
    middle = next(n for n, line in enumerate(ops) if owner(line) in second)
    end = max(n for n, line in enumerate(ops) if owner(line) in second) + 1
    if start >= middle or any(owner(line) not in (None, *first, *second) for line in ops[start:end]):
        raise AssertionError("source groups must be adjacent and complete")
    return reorder(machine, ops[:start] + ops[middle:end] + ops[start:middle] + ops[end:])


class SourceMemoryContractTest(unittest.TestCase):
    def check(self, machine: str, boundaries, *, rejected: str | None = None) -> None:
        """Accept at every boundary, or reject with the given diagnostic."""
        for tool, flags in boundaries:
            with self.subTest(tool=tool, flags=flags):
                result = run(tool, machine, *flags)
                self.assertEqual(result.returncode == 0, rejected is None, result.stderr)
                if rejected is not None:
                    self.assertIn(rejected, result.stderr)

    def test_completed_overlapping_transfers_keep_source_order(self) -> None:
        for first, second in (("store", "load"), ("load", "store"), ("store", "store")):
            with self.subTest(first=first, second=second):
                untimed = lower(program(first, second), timed=False)
                self.assertIn(CONTRACT, untimed)
                self.check(untimed, UNTIMED)
                earlier, later, _ = transfer_groups(untimed)
                swapped = swap_sources(untimed, earlier, later)
                self.check(swapped, UNTIMED, rejected=ORDER)
                delayed = run("atlas-opt", swapped, "--insert-atlas-delays")
                self.assertNotEqual(delayed.returncode, 0)
                self.assertIn(ORDER, delayed.stderr)
                timed = lower(program(first, second))
                self.check(timed, BOUNDARIES)
                self.check(swap_sources(timed, *transfer_groups(timed)[:2]), BOUNDARIES, rejected=ORDER)

    def test_disjoint_and_read_read_reorders_remain_legal(self) -> None:
        for first, second, address in (("store", "load", X + 1024), ("load", "load", X)):
            with self.subTest(first=first, second=second):
                untimed = lower(program(first, second, second_address=address), timed=False)
                swapped = swap_sources(untimed, *transfer_groups(untimed)[:2])
                self.check(swapped, UNTIMED)
                delayed = run("atlas-opt", swapped, "--insert-atlas-delays")
                self.assertEqual(delayed.returncode, 0, delayed.stderr)
                self.check(delayed.stdout, BOUNDARIES)

    def test_missing_or_weakened_contract_is_rejected(self) -> None:
        timed = lower(program("store", "load"))
        self.check(timed.replace(CONTRACT, "atlas.removed_source_memory_contract", 1), BOUNDARIES,
                   rejected=f"{INCOMPLETE} {CONTRACT}")
        weakened = re.sub(r"predecessors = array<i32: [^>]*>", "predecessors = array<i32>", timed)
        self.assertNotEqual(weakened, timed)
        self.check(weakened, BOUNDARIES, rejected=INCONSISTENT)

    def test_loop_visits_order_overlapping_effects_per_visit(self) -> None:
        untimed = lower(LOOP, timed=False)
        self.check(untimed, UNTIMED)
        self.check(lower(LOOP), BOUNDARIES)
        scheduled = run("atlas-opt", untimed, "--schedule-atlas-stream")
        self.assertEqual(scheduled.returncode, 0, scheduled.stderr)
        self.check(scheduled.stdout, BOUNDARIES)
        first_read, second_read, store, load, _ = transfer_groups(untimed)
        self.check(swap_sources(untimed, first_read, second_read), UNTIMED)
        self.check(swap_sources(untimed, store, load), UNTIMED, rejected=ORDER)


if __name__ == "__main__":
    unittest.main()
