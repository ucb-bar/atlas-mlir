"""Staged VMEM words, mailbox words and PACK's relayout survive every physical handoff."""

from __future__ import annotations

import re
import unittest

from test_source_memory_contract import operations, reorder, source_operations
from test_virtual_lowering import lower, run
from verification_support import (
    INCOMPLETE, TIMED_FINAL, UNTIMED, assert_boundaries, contract_text, drop_attribute, handoff_chain, insert_after,
    line_index, records, remove_line, replace_contract, rewrite_line,
)

CONTRACT = "atlas.virtual_buffer_contract"
CAPTURE = "buffer contract: DMA store captures a word its VSTORE did not write"
MAILBOX = "buffer contract: mailbox LW reads a word its mailbox DMA load did not write"
INTERLEAVE = "buffer contract: PACK relayout word differs from the selected row interleave"
STALE = "buffer contract: PACK relayout word is not a copy of its current raw conversion store"
SCALE = "buffer contract: PACK conversion scale register does not hold its source scale code"
INCONSISTENT = "buffer contract has malformed or inconsistent source records"
# Division of labor: CFG correspondence owns register contents, including the i1 mask and PACK's conversion result.
I1_MASK = "CFG contract issued scalar expression differs from source definition"
CONVERSION = "CFG contract memory store read lost its source tensor origin"

S, B, F = "!atlas.virtual_state", "!atlas.virtual_bf16", "!atlas.virtual_fp8"
ABI = ("atlas.input_dram_base = 2415919104 : i64, atlas.output_dram_base = 2415923200 : i64, "
       "atlas.control_dram_base = 2415927296 : i64")


def pack(result: str, tile: str) -> str:
    return f'    %{result} = "atlas.virtual_pack_fp8"(%{tile}) {{scale_code = 127 : i32}} : ({B}) -> {F}'


def matmul(result: str, lhs: str, rhs: str, unit: int = 0) -> str:
    return f'    %{result} = "atlas.virtual_mxu_matmul"(%{lhs}, %{rhs}) {{unit = {unit} : i32}} : ({F}, {F}) -> {B}'


def input_tile(state: str, tile: str, operand: str, index: int) -> str:
    return f'    %{state}, %{tile} = "atlas.virtual_input_bf16"(%{operand}) {{index = {index} : i32}} : ({S}) -> ({S}, {B})'


# Mailbox arguments, a PACK and explicit transfers that reuse one staging window and the mailbox window.
STRAIGHT = f'''module {{
  func.func @straight(%flag: i1, %count: i32) -> {S} attributes {{{ABI}}} {{
    %s0 = "atlas.virtual_start"() : () -> {S}
{input_tile("s1", "x", "s0", 0)}
{pack("p", "x")}
    %addr = arith.constant -1879035904 : i32
    %size = arith.constant 1024 : i32
    %s2, %stored = "atlas.virtual_dma_store_fp8"(%s1, %p, %addr, %size) : ({S}, {F}, i32, i32) -> ({S}, !atlas.virtual_dma_store)
    %s3 = "atlas.virtual_dma_wait"(%s2, %stored) : ({S}, !atlas.virtual_dma_store) -> {S}
    %s4, %loading = "atlas.virtual_dma_load_fp8"(%s3, %addr, %size) : ({S}, i32, i32) -> ({S}, !atlas.virtual_dma_load_fp8)
    %s5, %back = "atlas.virtual_dma_await_fp8"(%s4, %loading) : ({S}, !atlas.virtual_dma_load_fp8) -> ({S}, {F})
{matmul("y", "back", "p")}
    %s6 = "atlas.virtual_output_bf16"(%s5, %y) {{index = 0 : i32}} : ({S}, {B}) -> {S}
    return %s6 : {S}
  }}
}}
'''

# Each branch converts its own input; the join outputs either product.
JOIN = f'''module {{
  func.func @join(%flag: i1, %count: i32) -> {S} attributes {{{ABI}}} {{
    %s0 = "atlas.virtual_start"() : () -> {S}
    cf.cond_br %flag, ^a(%s0 : {S}), ^b(%s0 : {S})
  ^a(%sa: {S}):
{input_tile("sa1", "xa", "sa", 0)}
{pack("pa", "xa")}
{matmul("ya", "pa", "pa")}
    cf.br ^join(%sa1, %ya : {S}, {B})
  ^b(%sb: {S}):
{input_tile("sb1", "xb", "sb", 1)}
{pack("pb", "xb")}
{matmul("yb", "pb", "pb", 1)}
    cf.br ^join(%sb1, %yb : {S}, {B})
  ^join(%sj: {S}, %y: {B}):
    %s1 = "atlas.virtual_output_bf16"(%sj, %y) {{index = 0 : i32}} : ({S}, {B}) -> {S}
    return %s1 : {S}
  }}
}}
'''

# Every visit of the source loop repeats the PACK relayout and reloads one input window.
LOOP = f'''module {{
  func.func @loop(%flag: i1, %count: i32) -> {S} attributes {{{ABI}}} {{
    %s0 = "atlas.virtual_start"() : () -> {S}
    %zero = arith.constant 0 : i32
{input_tile("s1", "x", "s0", 0)}
    cf.br ^loop(%s1, %x, %zero : {S}, {B}, i32)
  ^loop(%st: {S}, %t: {B}, %i: i32):
{pack("p", "t")}
{matmul("y", "p", "p")}
{input_tile("st1", "u", "st", 1)}
    %z = "atlas.virtual_vpu_binary"(%y, %u) {{kind = "add"}} : ({B}, {B}) -> {B}
    %one = arith.constant 1 : i32
    %next = arith.addi %i, %one : i32
    %more = arith.cmpi slt, %next, %count : i32
    cf.cond_br %more, ^loop(%st1, %z, %next : {S}, {B}, i32), ^exit(%st1, %z : {S}, {B})
  ^exit(%se: {S}, %r: {B}):
    %s2 = "atlas.virtual_output_bf16"(%se, %r) {{index = 0 : i32}} : ({S}, {B}) -> {S}
    return %s2 : {S}
  }}
}}
'''


def command(machine: str, kind: str, vmem: int) -> str:
    """The issued line's tag for the tile record of `kind` at VMEM byte `vmem`."""
    identity = next(r["id"] for r in records(machine, "tile") if r["kind"] == kind and r["vmem_byte"] == vmem)
    return f"atlas.virtual_tile_command = {identity} : i32"


def moved(machine: str, chosen, anchor) -> str:
    """Move the issued operations satisfying `chosen`, in order, to just before the first other one satisfying `anchor`."""
    ops = operations(machine)
    group = [line for line in ops if chosen(line)]
    rest = [line for line in ops if not chosen(line)]
    at = next(n for n, line in enumerate(rest) if anchor(line))
    return reorder(machine, rest[:at] + group + rest[at:])


def reads(machine: str) -> list[dict]:
    return [r for r in records(machine, "buffer") if "command" in r]


def stray_store(machine: str) -> str:
    """A scalar SW into the first output half between its VSTORE and its DMA store."""
    staged = line_index(machine, command(machine, "vstore", 0x40000))
    return insert_after(machine, staged, [("upper", 'kind = "lui", dst = 27 : i32, immediate = 64 : i32'),
                                          ("scalar_store", 'kind = "sw", src = 0 : i32, base = 27 : i32, offset = 8 : i32')])


def overwritten_mailbox(machine: str) -> str:
    """The tensor input, which reuses the mailbox window, completes before the mailbox LWs read it."""
    source = next(identity for _, identity, name in source_operations(machine) if name == "atlas.virtual_input_bf16")
    tag = f"atlas.virtual_cfg_source = {source} : i32"
    return moved(machine, lambda line: tag in line, lambda line: "atlas.virtual_scalar_argument" in line)


def row_stride(machine: str) -> str:
    """The relayout pointer advances half a destination row per source row."""
    return rewrite_line(machine, lambda line: '"atlas.alu_imm"' in line and "dst = 12 : i32, immediate = 32 : i32" in line,
                        lambda line: line.replace("immediate = 32 : i32", "immediate = 16 : i32"))


def stale_raw_store(machine: str) -> str:
    """The raw conversion VSTORE issues after the copy loop instead of before it."""
    tag = command(machine, "vstore", 0x20000)
    return moved(machine, lambda line: tag in line, lambda line: command(machine, "vload", 0x20400) in line)


def stale_conversion(machine: str) -> str:
    """The conversion issues after the raw VSTORE that should capture it."""
    ops = operations(machine)
    conversion = next(n for n, line in enumerate(ops) if '"atlas.vpu_pack"' in line)
    store = next(n for n, line in enumerate(ops) if command(machine, "vstore", 0x20000) in line)
    return reorder(machine, ops[:conversion] + ops[conversion + 1:store + 1] + [ops[conversion]] + ops[store + 1:])


def unmasked(machine: str) -> str:
    """The i1 argument keeps its full mailbox word: the ANDI is gone and the LW claims the result."""
    mask = line_index(machine, '"atlas.alu_imm"', 'kind = "andi"', "atlas.virtual_scalar_result")
    value = re.search(r"atlas\.virtual_scalar_result = (\d+) : i32", machine.splitlines()[mask])[1]
    argument = f"atlas.virtual_scalar_argument = {value} : i32"
    machine = remove_line(machine, mask)
    return rewrite_line(machine, line_index(machine, argument),
                        lambda line: line.replace(argument, f"{argument}, atlas.virtual_scalar_result = {value} : i32"))


def wrong_scale(machine: str) -> str:
    return rewrite_line(machine, lambda line: 'kind = "seli"' in line and "atlas.virtual_cfg_helper" in line,
                        lambda line: line.replace("offset = 127 : i32", "offset = 126 : i32"))


class BufferContractVerificationTest(unittest.TestCase):
    def rejected(self, source: str, mutate, diagnostic: str) -> None:
        for timed, boundaries in ((False, UNTIMED), (True, TIMED_FINAL)):
            with self.subTest(timed=timed):
                assert_boundaries(self, mutate(lower(source, timed=timed)), boundaries, rejects=diagnostic)

    def test_lowered_reuse_joins_and_loops_are_accepted_at_every_handoff(self) -> None:
        for name, source in (("straight", STRAIGHT), ("join", JOIN), ("loop", LOOP)):
            with self.subTest(program=name):
                untimed = lower(source, timed=False)
                self.assertIn(CONTRACT, untimed)
                assert_boundaries(self, untimed, UNTIMED)
                assert_boundaries(self, lower(source), TIMED_FINAL)
                stages = handoff_chain(self, untimed)
                self.assertIn(contract_text(untimed, "buffer"), stages["structured"])

    def test_contract_records_every_reader_and_pack(self) -> None:
        machine = lower(STRAIGHT, timed=False)
        readers = reads(machine)
        tiles = records(machine, "tile")
        mailbox = [r for r in readers if tiles[r["command"]]["kind"] == "mailbox_load"]
        self.assertEqual([(r["writer"], r["vmem_byte"], r["bytes"], r["word"]) for r in mailbox], [(0, 0, 4, 0), (0, 4, 4, 1)])
        relayout = [r for r in readers if r["layout"] == "pack"]
        self.assertEqual([(tiles[r["writer"]]["vmem_byte"], r["vmem_byte"]) for r in relayout], [(0x20000, 0x20400)])
        self.assertIn("scale_code = 127 : i32", contract_text(machine, "buffer"))
        self.assertEqual(sum(tiles[r["command"]]["kind"] == "dma_store" for r in readers), 3)

    def test_stray_scalar_store_into_staged_half_is_rejected(self) -> None:
        self.rejected(STRAIGHT, stray_store, CAPTURE)
        elsewhere = lambda machine: stray_store(machine).replace("immediate = 64 : i32", "immediate = 300 : i32", 1)
        assert_boundaries(self, elsewhere(lower(STRAIGHT, timed=False)), UNTIMED)

    def test_mailbox_overwritten_by_tensor_input_is_rejected(self) -> None:
        self.rejected(STRAIGHT, overwritten_mailbox, MAILBOX)

    def test_pack_relayout_requires_row_interleave_of_current_store(self) -> None:
        self.rejected(STRAIGHT, row_stride, INTERLEAVE)
        for name, source in (("straight", STRAIGHT), ("loop", LOOP)):
            with self.subTest(program=name):
                self.rejected(source, stale_raw_store, STALE)
        self.rejected(STRAIGHT, wrong_scale, SCALE)
        self.rejected(STRAIGHT, stale_conversion, CONVERSION)

    def test_dropped_i1_normalization_is_rejected(self) -> None:
        self.rejected(STRAIGHT, unmasked, I1_MASK)

    def test_missing_or_altered_contract_is_rejected(self) -> None:
        machine = lower(STRAIGHT)
        dropped = drop_attribute(machine, CONTRACT)
        self.assertNotIn(CONTRACT, dropped)
        assert_boundaries(self, dropped, TIMED_FINAL, rejects=f"{INCOMPLETE} {CONTRACT}")
        altered = replace_contract(machine, "buffer", contract_text(machine, "buffer").replace('layout = "pack"', 'layout = "copy"'))
        self.assertNotEqual(altered, machine)
        assert_boundaries(self, altered, TIMED_FINAL, rejects=INCONSISTENT)


if __name__ == "__main__":
    unittest.main()
