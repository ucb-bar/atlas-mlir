"""Overlapping source DRAM effects keep their completion order through every handoff."""

from __future__ import annotations

import re
import unittest

from test_virtual_lowering import lower, run
from verification_support import assert_boundaries, operations, reorder


CONTRACT = "atlas.virtual_source_memory_contract"
ORDER = "source memory contract: overlapping source predecessor has not completed in this visit"
INCONSISTENT = "source memory contract has malformed or inconsistent source records"
LAUNCHES = ("atlas.virtual_dma_load_fp8", "atlas.virtual_dma_store_fp8")
STATE = "!atlas.virtual_state"
X, Y = 0x90000000, 0x90010000


def signed(value: int) -> int:
    return value - (1 << 32) if value >= 1 << 31 else value


def transfer(index: int, kind: str, address: str) -> str:
    state, done = f"%s{2 * index + 1}", f"%s{2 * index + 3}"
    if kind == "store":
        return f'    {done} = "atlas.virtual_dma_store_fp8"({state}, %tile, {address}, %size) : ({STATE}, !atlas.virtual_fp8, i32, i32) -> {STATE}'
    return f'    {done}, %t{index}_value = "atlas.virtual_dma_load_fp8"({state}, {address}, %size) : ({STATE}, i32, i32) -> ({STATE}, !atlas.virtual_fp8)'


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
    %l2, %a = "atlas.virtual_dma_load_fp8"(%state, %x, %size) : ({STATE}, i32, i32) -> ({STATE}, !atlas.virtual_fp8)
    %l4, %b = "atlas.virtual_dma_load_fp8"(%l2, %x, %size) : ({STATE}, i32, i32) -> ({STATE}, !atlas.virtual_fp8)
    %next = arith.addi %i, %one : i32
    %more = arith.cmpi slt, %next, %limit : i32
    cf.cond_br %more, ^loop(%l4, %next : {STATE}, i32), ^exit(%l4 : {STATE})
  ^exit(%final : {STATE}):
    %ex = arith.constant {signed(X)} : i32
    %ez = arith.constant {signed(Y)} : i32
    %esize = arith.constant 1024 : i32
    %e1 = "atlas.virtual_dma_store_fp8"(%final, %tile, %ex, %esize) : ({STATE}, !atlas.virtual_fp8, i32, i32) -> {STATE}
    %e4, %value = "atlas.virtual_dma_load_fp8"(%e1, %ex, %esize) : ({STATE}, i32, i32) -> ({STATE}, !atlas.virtual_fp8)
    %e5 = "atlas.virtual_dma_store_fp8"(%e4, %tile, %ez, %esize) : ({STATE}, !atlas.virtual_fp8, i32, i32) -> {STATE}
    return %e5 : {STATE}
  }}
}}
'''


def source_operations(machine: str) -> list[tuple[int, int, str]]:
    pattern = r'\{block = (\d+) : i32, id = (\d+) : i32, mxu_commands = array<i32[^>]*>, name = "([^"]+)"'
    return [(int(block), int(identity), name) for block, identity, name in re.findall(pattern, machine)]


def transfer_groups(machine: str) -> list[set[int]]:
    """Each explicit transfer's source identity, in source order: one atomic operation."""
    return [{identity} for _, identity, name in source_operations(machine) if name in LAUNCHES]


def swap_sources(machine: str, first: set[int], second: set[int]) -> str:
    """Move every instruction of the second transfer, in order, before the first transfer starts.

    A load's wait and staging VLOADs may sit later than its launch, before its
    tile's first use, so the second transfer completes entirely before the first.
    """
    ops = operations(machine)
    tag = re.compile(r"atlas\.virtual_cfg_source = (\d+) : i32")
    owner = lambda line: int(tag.search(line)[1]) if tag.search(line) else None
    start = next(n for n, line in enumerate(ops) if owner(line) in first)
    moved = [n for n, line in enumerate(ops) if owner(line) in second]
    if not moved or moved[0] <= start:
        raise AssertionError("the second transfer must follow the first")
    rest = [line for n, line in enumerate(ops[start:], start) if n not in set(moved)]
    return reorder(machine, ops[:start] + [ops[n] for n in moved] + rest)


class SourceMemoryContractTest(unittest.TestCase):
    def test_completed_overlapping_transfers_keep_source_order(self) -> None:
        for first, second in (("store", "load"), ("load", "store"), ("store", "store")):
            for timed in (False, True):
                with self.subTest(first=first, second=second, timed=timed):
                    machine = lower(program(first, second), timed=timed)
                    self.assertIn(CONTRACT, machine)
                    assert_boundaries(self, machine)
                    swapped = swap_sources(machine, *transfer_groups(machine)[:2])
                    assert_boundaries(self, swapped, rejects=ORDER)
                    if not timed:
                        delayed = run("atlas-opt", swapped, "--insert-atlas-delays")
                        self.assertNotEqual(delayed.returncode, 0)
                        self.assertIn(ORDER, delayed.stderr)

    def test_disjoint_and_read_read_reorders_remain_legal(self) -> None:
        for first, second, address in (("store", "load", X + 1024), ("load", "load", X)):
            with self.subTest(first=first, second=second):
                untimed = lower(program(first, second, second_address=address), timed=False)
                swapped = swap_sources(untimed, *transfer_groups(untimed)[:2])
                assert_boundaries(self, swapped)
                delayed = run("atlas-opt", swapped, "--insert-atlas-delays")
                self.assertEqual(delayed.returncode, 0, delayed.stderr)
                assert_boundaries(self, delayed.stdout)

    def test_weakened_contract_is_rejected(self) -> None:
        timed = lower(program("store", "load"))
        weakened = re.sub(r"predecessors = array<i32: [^>]*>", "predecessors = array<i32>", timed)
        self.assertNotEqual(weakened, timed)
        assert_boundaries(self, weakened, rejects=INCONSISTENT)

    def test_loop_visits_order_overlapping_effects_per_visit(self) -> None:
        untimed = lower(LOOP, timed=False)
        assert_boundaries(self, untimed)
        assert_boundaries(self, lower(LOOP))
        scheduled = run("atlas-opt", untimed, "--schedule-atlas-stream")
        self.assertEqual(scheduled.returncode, 0, scheduled.stderr)
        assert_boundaries(self, scheduled.stdout)
        first_read, second_read, store, load, _ = transfer_groups(untimed)
        assert_boundaries(self, swap_sources(untimed, first_read, second_read))
        assert_boundaries(self, swap_sources(untimed, store, load), rejects=ORDER)


if __name__ == "__main__":
    unittest.main()
