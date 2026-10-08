"""Compare original virtual CFG execution with its lowered selected-core program."""

from __future__ import annotations

from dataclasses import dataclass
import unittest

from test_virtual_evaluator_core import compare_result, selected_core, tile_bytes
from atlas_virtual_evaluator import MemoryRegion, RuntimeInputs, Scalar, Tile, evaluate, parse_program
from test_virtual_lowering import emitted, lower, object_words


INPUT_BASE, OUTPUT_BASE, CONTROL_BASE = 0x90000000, 0x90004000, 0x90008000
STATE, BF16 = "!atlas.virtual_state", "!atlas.virtual_bf16"
PAIR = f"{STATE}, {BF16}"


@dataclass(frozen=True)
class Case:
    name: str
    source: str
    widths: tuple[int, ...]
    # Each variant gives runtime control bits and independently known output
    # tile indices. Index 2 means ReLU(input 0), evaluated over finite normals.
    variants: tuple[tuple[tuple[int, ...], tuple[int, ...]], ...]


def kernel(arguments: str, body: str) -> str:
    return f'''module {{
  func.func @cfg({arguments}) -> {STATE} attributes {{
    atlas.input_dram_base = {INPUT_BASE} : i64,
    atlas.output_dram_base = {OUTPUT_BASE} : i64,
    atlas.control_dram_base = {CONTROL_BASE} : i64}} {{
    %s0 = "atlas.virtual_start"() : () -> {STATE}
    %s1, %a0 = "atlas.virtual_input_bf16"(%s0) {{index = 0 : i32}} : ({STATE}) -> ({PAIR})
    %s2, %b0 = "atlas.virtual_input_bf16"(%s1) {{index = 1 : i32}} : ({STATE}) -> ({PAIR})
    {body}
  }}
}}
'''


def selection(arguments: str, condition: str) -> str:
    return kernel(arguments, f'''
    {condition}
    cf.cond_br %choose, ^left(%s2, %a0 : {PAIR}), ^right(%s2, %b0 : {PAIR})
  ^left(%ls: {STATE}, %a: {BF16}):
    cf.br ^join(%ls, %a : {PAIR})
  ^right(%rs: {STATE}, %b: {BF16}):
    cf.br ^join(%rs, %b : {PAIR})
  ^join(%js: {STATE}, %selected: {BF16}):
    %out = "atlas.virtual_output_bf16"(%js, %selected) {{index = 0 : i32}} : ({PAIR}) -> {STATE}
    return %out : {STATE}''')


def loop(swap: bool) -> str:
    carried = f"{PAIR}, {BF16}, i32" if swap else f"{PAIR}, i32"
    initial = "%s2, %a0, %b0, %zero" if swap else "%s2, %a0, %zero"
    args = f"%ls: {STATE}, %a: {BF16}, " + (f"%b: {BF16}, " if swap else "") + "%i: i32"
    body_args = f"%bs: {STATE}, %ba: {BF16}, " + (f"%bb: {BF16}, " if swap else "") + "%bi: i32"
    advance = "" if swap else f'%next = "atlas.virtual_vpu_unary"(%ba) {{kind = "relu"}} : ({BF16}) -> {BF16}'
    backedge = "%bs, %bb, %ba, %j" if swap else "%bs, %next, %j"
    exit_types = f"{PAIR}, {BF16}" if swap else PAIR
    exit_values = "%ls, %a, %b" if swap else "%ls, %a"
    exit_args = f"%es: {STATE}, %x: {BF16}" + (f", %y: {BF16}" if swap else "")
    second = f'%out1 = "atlas.virtual_output_bf16"(%out0, %y) {{index = 1 : i32}} : ({PAIR}) -> {STATE}' if swap else ""
    return kernel("%limit: i32", f'''
    %zero = arith.constant 0 : i32
    %one = arith.constant 1 : i32
    cf.br ^loop({initial} : {carried})
  ^loop({args}):
    %more = arith.cmpi ult, %i, %limit : i32
    cf.cond_br %more, ^body({exit_values}, %i : {carried}), ^exit({exit_values} : {exit_types})
  ^body({body_args}):
    {advance}
    %j = arith.addi %bi, %one : i32
    cf.br ^loop({backedge} : {carried})
  ^exit({exit_args}):
    %out0 = "atlas.virtual_output_bf16"(%es, %x) {{index = 0 : i32}} : ({PAIR}) -> {STATE}
    {second}
    return %out{1 if swap else 0} : {STATE}''')


def cases() -> tuple[Case, ...]:
    result = [Case(f"constant_i1_{bit}", selection("", f"%choose = arith.constant {bit} : i1"), (), (((), (0 if bit else 1,)),)) for bit in (0, 1)]
    result.append(Case("runtime_i1", selection("%choose: i1", ""), (1,), (((0,), (1,)), ((1,), (0,)))))
    boundaries = ((0x80000000, 0x7FFFFFFF), (0x7FFFFFFF, 0x80000000), (0xFFFFFFFF, 0), (0x80000000, 0x80000000))
    signed = lambda bits: bits if bits < 0x80000000 else bits - 0x100000000
    predicates = {
        "eq": lambda a, b: a == b, "ne": lambda a, b: a != b,
        "slt": lambda a, b: signed(a) < signed(b), "sle": lambda a, b: signed(a) <= signed(b),
        "sgt": lambda a, b: signed(a) > signed(b), "sge": lambda a, b: signed(a) >= signed(b),
        "ult": lambda a, b: a < b, "ule": lambda a, b: a <= b,
        "ugt": lambda a, b: a > b, "uge": lambda a, b: a >= b,
    }
    for predicate, reference in predicates.items():
        source = selection("%lhs: i32, %rhs: i32", f"%choose = arith.cmpi {predicate}, %lhs, %rhs : i32")
        variants = tuple((bits, (0 if reference(*bits) else 1,)) for bits in boundaries)
        result.append(Case(f"cmpi_{predicate}", source, (32, 32), variants))
    wrapping = selection("%lhs: i32, %rhs: i32", '''%sum = arith.addi %lhs, %rhs : i32
    %zero = arith.constant 0 : i32
    %choose = arith.cmpi eq, %sum, %zero : i32''')
    result.append(Case("wrapping_addi", wrapping, (32, 32), (((0xFFFFFFFF, 1), (0,)), ((0x80000000, 0x80000000), (0,)), ((0xFFFFFFFF, 2), (1,)))))
    for swap in (False, True):
        variants = tuple(((count,), ((1, 0) if count % 2 else (0, 1)) if swap else ((2,) if count else (0,))) for count in (0, 1, 2, 3))
        result.append(Case("tile_swap" if swap else "relu_loop", loop(swap), (32,), variants))
    return tuple(result)


def runtime_inputs(widths: tuple[int, ...], controls: tuple[int, ...], phase: int) -> RuntimeInputs:
    tiles = {index: Tile("bf16", tuple(
        0 if (i + phase) % 19 == 0 else (0x3E80 + index * 0x400 + (i + phase) % 127) | (0x8000 if (i + i // 32 + phase) % 2 else 0)
        for i in range(1024))) for index in (0, 1)}
    mailbox = bytearray(b"\xC3" * 1024)
    for index, bits in enumerate(controls):
        mailbox[index * 4:index * 4 + 4] = bits.to_bytes(4, "little")
    memory = [MemoryRegion(INPUT_BASE + index * 2048, tile_bytes(tile)) for index, tile in tiles.items()]
    memory.append(MemoryRegion(CONTROL_BASE, mailbox))
    memory.extend(MemoryRegion(address, bytes([value]) * 64) for address, value in (
        (INPUT_BASE - 64, 0x51), (INPUT_BASE + 4096, 0x62),
        (OUTPUT_BASE - 64, 0x73), (OUTPUT_BASE + 4096, 0x84),
        (CONTROL_BASE - 64, 0x95), (CONTROL_BASE + 1024, 0xA6),
    ))
    return RuntimeInputs(tiles, tuple(Scalar(width, bits) for width, bits in zip(widths, controls)), tuple(memory))


def assert_reference(test: unittest.TestCase, expected, inputs: RuntimeInputs, indices: tuple[int, ...]) -> None:
    tiles = dict(inputs.tiles)
    tiles[2] = Tile("bf16", tuple(0 if bits & 0x8000 else bits for bits in tiles[0].bits))
    test.assertEqual(dict(expected.outputs), {index: tiles[selection] for index, selection in enumerate(indices)})
    test.assertEqual(expected.memory, inputs.memory)


class VirtualEvaluatorCFGCoreTest(unittest.TestCase):
    def test_all_original_cfg_cases_parse_and_evaluate(self) -> None:
        for case in cases():
            with self.subTest(case=case.name):
                program = parse_program(case.source)
                self.assertEqual(program.control_widths, case.widths)
                for controls, indices in case.variants:
                    for phase in (0, 113):
                        inputs = runtime_inputs(case.widths, controls, phase)
                        expected = evaluate(program, inputs, max_steps=256)
                        assert_reference(self, expected, inputs, indices)

    def test_all_cfg_cases_lower_to_exact_llvm_words(self) -> None:
        for case in cases():
            with self.subTest(case=case.name):
                machine = lower(case.source)
                self.assertEqual(object_words(machine), emitted(machine))

    def test_selected_core_matches_original_cfg_with_fresh_controls_and_tiles(self) -> None:
        with selected_core() as run:
            for case in cases():
                program = parse_program(case.source)
                words = object_words(lower(case.source))
                for controls, indices in case.variants:
                    for phase in (0, 113):
                        with self.subTest(case=case.name, controls=controls, phase=phase):
                            inputs = runtime_inputs(case.widths, controls, phase)
                            expected = evaluate(program, inputs, max_steps=256)
                            assert_reference(self, expected, inputs, indices)
                            preload = [(region.address, region.data) for region in inputs.memory]
                            preload.append((OUTPUT_BASE, b"\xA5" * 4096))
                            result = run(words, preload)
                            self.assertTrue(result.halted, "selected core did not halt")
                            self.assertEqual(result.writes, len(expected.outputs) * 64)
                            compare_result(expected, result.slave.captured, output_base=OUTPUT_BASE)


if __name__ == "__main__":
    unittest.main()
