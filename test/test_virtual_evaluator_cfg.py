"""Scalar and CFG semantics over logical tiles, independent of machine lowering."""

from __future__ import annotations

from pathlib import Path
import sys
import unittest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
from atlas_virtual_evaluator import RuntimeInputs, Scalar, Tile, UnsupportedVirtualMode, VirtualInterfaceError, evaluate, parse_program  # noqa: E402


INPUTS = '''
  %s0 = "atlas.virtual_start"() : () -> !atlas.virtual_state
  %s1, %a = "atlas.virtual_input_bf16"(%s0) {index = 0 : i32} : (!atlas.virtual_state) -> (!atlas.virtual_state, !atlas.virtual_bf16)
  %s2, %b = "atlas.virtual_input_bf16"(%s1) {index = 1 : i32} : (!atlas.virtual_state) -> (!atlas.virtual_state, !atlas.virtual_bf16)
'''
SELECT = '''
  cf.cond_br %choose, ^exit(%s2, %a : !atlas.virtual_state, !atlas.virtual_bf16), ^exit(%s2, %b : !atlas.virtual_state, !atlas.virtual_bf16)
^exit(%state: !atlas.virtual_state, %tile: !atlas.virtual_bf16):
  %done = "atlas.virtual_output_bf16"(%state, %tile) {index = 7 : i32} : (!atlas.virtual_state, !atlas.virtual_bf16) -> !atlas.virtual_state
  func.return %done : !atlas.virtual_state
'''
TILES = {0: Tile("bf16", (0xBF80,) * 1024), 1: Tile("bf16", (0x4000,) * 1024)}


def function(body: str, arguments: str = "") -> str:
    return "module { func.func @test(" + arguments + ") -> !atlas.virtual_state {" + body + "} }"


def example(name: str) -> str:
    return (ROOT / "test/examples" / name).read_text()


class VirtualEvaluatorCFGTest(unittest.TestCase):
    def assert_selection(self, source: str, controls: tuple[Scalar, ...], selected: int) -> None:
        result = evaluate(parse_program(source), RuntimeInputs(TILES, controls))
        self.assertEqual(dict(result.outputs), {7: TILES[selected]})

    def test_all_comparisons_distinguish_signed_and_unsigned_i32_boundaries(self) -> None:
        # Expected answers are explicit to catch sign extension or predicate swaps.
        predicates = ("eq", "ne", "slt", "sle", "sgt", "sge", "ult", "ule", "ugt", "uge")
        cases = (
            (0xFFFFFFFF, 0, (0, 1, 1, 1, 0, 0, 0, 0, 1, 1)),
            (0x80000000, 0x7FFFFFFF, (0, 1, 1, 1, 0, 0, 0, 0, 1, 1)),
            (0x7FFFFFFF, 0x80000000, (0, 1, 0, 0, 1, 1, 1, 1, 0, 0)),
            (0x80000000, 0x80000000, (1, 0, 0, 1, 0, 1, 0, 1, 0, 1)),
            (0, 1, (0, 1, 1, 1, 0, 0, 1, 1, 0, 0)),
        )
        for predicate, position in zip(predicates, range(10)):
            source = function(INPUTS + f"%choose = arith.cmpi {predicate}, %lhs, %rhs : i32\n" + SELECT, "%lhs: i32, %rhs: i32")
            for lhs, rhs, expected in cases:
                with self.subTest(predicate=predicate, lhs=hex(lhs), rhs=hex(rhs)):
                    self.assert_selection(source, (Scalar(32, lhs), Scalar(32, rhs)), 0 if expected[position] else 1)

    def test_addition_wraps_before_comparison_and_negative_constants_keep_bits(self) -> None:
        body = INPUTS + '''
          %sum = arith.addi %lhs, %rhs : i32
          %choose = arith.cmpi eq, %sum, %expected : i32
        ''' + SELECT
        source = function(body, "%lhs: i32, %rhs: i32, %expected: i32")
        for lhs, rhs, expected in ((0xFFFFFFFF, 1, 0), (0x7FFFFFFF, 1, 0x80000000), (0x80000000, 0xFFFFFFFF, 0x7FFFFFFF)):
            with self.subTest(lhs=hex(lhs), rhs=hex(rhs)):
                self.assert_selection(source, tuple(Scalar(32, bits) for bits in (lhs, rhs, expected)), 0)
                self.assert_selection(source, tuple(Scalar(32, bits) for bits in (lhs, rhs, expected ^ 1)), 1)
        source = function(INPUTS + "%negative = arith.constant -1 : i32\n%choose = arith.cmpi eq, %negative, %expected : i32\n" + SELECT, "%expected: i32")
        self.assert_selection(source, (Scalar(32, 0xFFFFFFFF),), 0)
        self.assert_selection(source, (Scalar(32, 0x7FFFFFFF),), 1)

    def test_i1_constants_and_controls_select_both_paths(self) -> None:
        for literal, selected in (("true", 0), ("false", 1)):
            self.assert_selection(function(INPUTS + f"%choose = arith.constant {literal}\n" + SELECT), (), selected)
        source = function(INPUTS + SELECT, "%choose: i1")
        for choice in (0, 1):
            self.assert_selection(source, (Scalar(1, choice),), 0 if choice else 1)

    def test_diamond_passes_state_bf16_i1_and_i32_arguments(self) -> None:
        source = function(INPUTS + '''
          %ten = arith.constant 10 : i32
          %twenty = arith.constant 20 : i32
          cf.cond_br %choose, ^left(%s2, %a : !atlas.virtual_state, !atlas.virtual_bf16), ^right(%s2, %b : !atlas.virtual_state, !atlas.virtual_bf16)
        ^left(%ls: !atlas.virtual_state, %lt: !atlas.virtual_bf16):
          %yes = arith.constant true
          cf.br ^join(%ls, %lt, %yes, %ten : !atlas.virtual_state, !atlas.virtual_bf16, i1, i32)
        ^right(%rs: !atlas.virtual_state, %rt: !atlas.virtual_bf16):
          %no = arith.constant false
          cf.br ^join(%rs, %rt, %no, %twenty : !atlas.virtual_state, !atlas.virtual_bf16, i1, i32)
        ^join(%js: !atlas.virtual_state, %jt: !atlas.virtual_bf16, %flag: i1, %number: i32):
          %correct = arith.cmpi eq, %number, %expected : i32
          cf.cond_br %correct, ^check(%js, %jt, %flag : !atlas.virtual_state, !atlas.virtual_bf16, i1), ^exit(%js, %b : !atlas.virtual_state, !atlas.virtual_bf16)
        ^check(%cs: !atlas.virtual_state, %ct: !atlas.virtual_bf16, %carried: i1):
          cf.cond_br %carried, ^exit(%cs, %ct : !atlas.virtual_state, !atlas.virtual_bf16), ^exit(%cs, %a : !atlas.virtual_state, !atlas.virtual_bf16)
        ^exit(%es: !atlas.virtual_state, %et: !atlas.virtual_bf16):
          %done = "atlas.virtual_output_bf16"(%es, %et) {index = 7 : i32} : (!atlas.virtual_state, !atlas.virtual_bf16) -> !atlas.virtual_state
          func.return %done : !atlas.virtual_state
        ''', "%choose: i1, %expected: i32")
        self.assert_selection(source, (Scalar(1, 1), Scalar(32, 10)), 0)
        self.assert_selection(source, (Scalar(1, 1), Scalar(32, 20)), 1)
        self.assert_selection(source, (Scalar(1, 0), Scalar(32, 20)), 0)
        self.assert_selection(source, (Scalar(1, 0), Scalar(32, 10)), 1)
        renamed = source.replace("%a", "%first").replace("%b", "%second").replace("%js", "%joined_state")
        self.assertEqual(evaluate(parse_program(source), RuntimeInputs(TILES, (Scalar(1, 1), Scalar(32, 10)))), evaluate(parse_program(renamed), RuntimeInputs(TILES, (Scalar(1, 1), Scalar(32, 10)))))

    def test_untaken_paths_have_no_outputs(self) -> None:
        source = function('''
          %s0 = "atlas.virtual_start"() : () -> !atlas.virtual_state
          cf.cond_br %choose, ^take(%s0 : !atlas.virtual_state), ^skip(%s0 : !atlas.virtual_state)
        ^take(%ts: !atlas.virtual_state):
          %next, %tile = "atlas.virtual_input_bf16"(%ts) {index = 0 : i32} : (!atlas.virtual_state) -> (!atlas.virtual_state, !atlas.virtual_bf16)
          %relu = "atlas.virtual_vpu_unary"(%tile) {kind = "relu"} : (!atlas.virtual_bf16) -> !atlas.virtual_bf16
          %out = "atlas.virtual_output_bf16"(%next, %relu) {index = 7 : i32} : (!atlas.virtual_state, !atlas.virtual_bf16) -> !atlas.virtual_state
          func.return %out : !atlas.virtual_state
        ^skip(%ss: !atlas.virtual_state):
          func.return %ss : !atlas.virtual_state
        ''', "%choose: i1")
        special = Tile("bf16", (0x7FC1,) * 1024)
        self.assertEqual(dict(evaluate(parse_program(source), RuntimeInputs({0: special}, (Scalar(1, 0),))).outputs), {})
        self.assertEqual(dict(evaluate(parse_program(source), RuntimeInputs({0: special}, (Scalar(1, 1),))).outputs), {7: special})
        # Declarations still define the input interface even when not executed.
        with self.assertRaises(VirtualInterfaceError):
            evaluate(parse_program(source), RuntimeInputs(controls=(Scalar(1, 0),)))

    def test_unsupported_operation_rejects_even_an_untaken_path(self) -> None:
        source = example("virtual_bf16_dynamic_branch_program.mlir")
        dma = '''%address = arith.constant 0 : i32
          %length = arith.constant 2048 : i32
          %pending_state, %pending = "atlas.virtual_dma_load_bf16"(%right_io, %address, %length) : (!atlas.virtual_state, i32, i32) -> (!atlas.virtual_state, !atlas.virtual_dma_load_bf16)
        '''
        source = source.replace('    %right_result =', dma + '    %right_result =').replace('^join(%right_io,', '^join(%pending_state,')
        for choice in (0, 1):
            with self.subTest(choice=choice):
                with self.assertRaises(UnsupportedVirtualMode):
                    evaluate(parse_program(source), RuntimeInputs({0: TILES[0]}, (Scalar(1, choice),)))

    def test_loop_reexecutes_local_definitions_and_carries_changed_tiles(self) -> None:
        source = example("virtual_bf16_loop_program.mlir")
        program = parse_program(source)
        bits = tuple(0xBF80 if index % 2 else 0x4000 for index in range(1024))
        inputs = RuntimeInputs({0: Tile("bf16", bits)})
        self.assertEqual(evaluate(program, inputs).outputs[0].bits, tuple(0 if value & 0x8000 else value for value in bits))
        # Six entry operations, two full five-operation iterations, the final
        # two-operation condition check, output, and return total twenty steps.
        self.assertEqual(evaluate(program, inputs, max_steps=20), evaluate(program, inputs))
        with self.assertRaises(VirtualInterfaceError):
            evaluate(program, inputs, max_steps=19)

    def test_loop_backedges_bind_swapped_tiles_and_scalars_simultaneously(self) -> None:
        source = example("virtual_bf16_swap_loop_program.mlir")
        source = source.replace("@swap_once()", "@swap_once(%limit: i32)").replace("arith.cmpi slt, %i, %c1", "arith.cmpi slt, %i, %limit")
        program = parse_program(source)
        for limit, selected in ((0, 0), (1, 1), (2, 0), (3, 1)):
            with self.subTest(limit=limit):
                result = evaluate(program, RuntimeInputs(TILES, (Scalar(32, limit),)))
                self.assertEqual(dict(result.outputs), {0: TILES[selected]})
        scalar_swap = function(INPUTS + '''
          %zero = arith.constant 0 : i32
          %one = arith.constant 1 : i32
          cf.br ^loop(%s2, %zero, %zero, %one : !atlas.virtual_state, i32, i32, i32)
        ^loop(%ls: !atlas.virtual_state, %i: i32, %x: i32, %y: i32):
          %more = arith.cmpi slt, %i, %limit : i32
          %next = arith.addi %i, %one : i32
          cf.cond_br %more, ^loop(%ls, %next, %y, %x : !atlas.virtual_state, i32, i32, i32), ^check(%ls, %x : !atlas.virtual_state, i32)
        ^check(%cs: !atlas.virtual_state, %carried: i32):
          %choose = arith.cmpi eq, %carried, %expected : i32
          cf.cond_br %choose, ^exit(%cs, %a : !atlas.virtual_state, !atlas.virtual_bf16), ^exit(%cs, %b : !atlas.virtual_state, !atlas.virtual_bf16)
        ^exit(%es: !atlas.virtual_state, %et: !atlas.virtual_bf16):
          %done = "atlas.virtual_output_bf16"(%es, %et) {index = 7 : i32} : (!atlas.virtual_state, !atlas.virtual_bf16) -> !atlas.virtual_state
          func.return %done : !atlas.virtual_state
        ''', "%limit: i32, %expected: i32")
        for limit, expected in ((1, 1), (2, 0), (3, 1)):
            with self.subTest(scalar_swaps=limit):
                self.assert_selection(scalar_swap, (Scalar(32, limit), Scalar(32, expected)), 0)
                self.assert_selection(scalar_swap, (Scalar(32, limit), Scalar(32, expected ^ 1)), 1)

    def test_repeated_effects_need_fresh_dynamic_state_tokens(self) -> None:
        source = example("virtual_bf16_loop_program.mlir")
        source = source.replace('    %next_tile =', '    %body_next, %unused = "atlas.virtual_input_bf16"(%body_io) {index = 0 : i32} : (!atlas.virtual_state) -> (!atlas.virtual_state, !atlas.virtual_bf16)\n    %next_tile =')
        source = source.replace("cf.br ^loop(%body_io, %next_tile", "cf.br ^loop(%body_next, %next_tile")
        inputs = RuntimeInputs({0: TILES[0]})
        self.assertEqual(evaluate(parse_program(source), inputs).outputs[0], Tile("bf16", (0,) * 1024))
        for stale in ("%body_io", "%io1"):
            with self.subTest(stale=stale):
                changed = source.replace("cf.br ^loop(%body_next, %next_tile", f"cf.br ^loop({stale}, %next_tile")
                with self.assertRaises(VirtualInterfaceError):
                    evaluate(parse_program(changed), inputs)

    def test_stale_branch_tokens_and_foreign_ssa_identities_fail(self) -> None:
        source = function(INPUTS + SELECT, "%choose: i1")
        for changed in (source.replace("^exit(%s2, %a", "^exit(%s1, %a"), source.replace("func.return %done", "func.return %state")):
            with self.assertRaises(VirtualInterfaceError):
                evaluate(parse_program(changed), RuntimeInputs(TILES, (Scalar(1, 1),)))
        program, other = parse_program(source), parse_program(source)
        branch = tuple(program.blocks[0].ops)[-1]
        # The same spelling and type in another parse do not identify a value.
        branch.operands = (other.blocks[0].args[0], *branch.operands[1:])
        with self.assertRaises(VirtualInterfaceError):
            evaluate(program, RuntimeInputs(TILES, (Scalar(1, 1),)))

    def test_dominating_state_alias_cannot_replace_the_current_block_argument(self) -> None:
        source = function(INPUTS + SELECT, "%choose: i1")
        source = source.replace('"atlas.virtual_output_bf16"(%state, %tile)', '"atlas.virtual_output_bf16"(%s2, %tile)')
        # Both values would hold the same dynamic token, but the successor must
        # consume its own current state argument rather than a predecessor alias.
        with self.assertRaisesRegex(VirtualInterfaceError, "current state token"):
            evaluate(parse_program(source), RuntimeInputs(TILES, (Scalar(1, 1),)))

    def test_same_block_forward_scalar_reference_does_not_dominate_its_use(self) -> None:
        source = function('''
          %s = "atlas.virtual_start"() : () -> !atlas.virtual_state
          %sum = arith.addi %later, %later : i32
          %later = arith.constant 1 : i32
          func.return %s : !atlas.virtual_state
        ''')
        with self.assertRaisesRegex(VirtualInterfaceError, "does not dominate"):
            evaluate(parse_program(source), RuntimeInputs())

    def test_step_budget_counts_branches_and_returns_and_bounds_infinite_loops(self) -> None:
        source = function(INPUTS + SELECT, "%choose: i1")
        program, inputs = parse_program(source), RuntimeInputs(TILES, (Scalar(1, 0),))
        self.assertEqual(evaluate(program, inputs, max_steps=6), evaluate(program, inputs))
        for budget in (5, 1, 0, -1, True, 1.5, None):
            with self.subTest(budget=budget):
                with self.assertRaises(VirtualInterfaceError):
                    evaluate(program, inputs, max_steps=budget)
        infinite = function('''
          %s = "atlas.virtual_start"() : () -> !atlas.virtual_state
          cf.br ^loop(%s : !atlas.virtual_state)
        ^loop(%state: !atlas.virtual_state):
          cf.br ^loop(%state : !atlas.virtual_state)
        ''')
        with self.assertRaises(VirtualInterfaceError):
            evaluate(parse_program(infinite), RuntimeInputs(), max_steps=17)

    def test_parser_rejects_handle_crossings_and_branch_type_mismatches(self) -> None:
        source = function(INPUTS + SELECT, "%choose: i1")
        for changed in (source.replace("%tile: !atlas.virtual_bf16", "%tile: i32"), source.replace("%state: !atlas.virtual_state", "%state: !atlas.virtual_dma_store")):
            with self.assertRaises(VirtualInterfaceError):
                parse_program(changed)
        handle = function('''
          %s = "atlas.virtual_start"() : () -> !atlas.virtual_state
          %address = arith.constant 0 : i32
          %size = arith.constant 2048 : i32
          %next, %handle = "atlas.virtual_dma_load_bf16"(%s, %address, %size) : (!atlas.virtual_state, i32, i32) -> (!atlas.virtual_state, !atlas.virtual_dma_load_bf16)
          cf.br ^exit(%next, %handle : !atlas.virtual_state, !atlas.virtual_dma_load_bf16)
        ^exit(%state: !atlas.virtual_state, %pending: !atlas.virtual_dma_load_bf16):
          func.return %state : !atlas.virtual_state
        ''')
        with self.assertRaisesRegex(VirtualInterfaceError, "non-entry block"):
            parse_program(handle)

    def test_malformed_untaken_edges_are_rejected(self) -> None:
        source = function(INPUTS + SELECT, "%choose: i1")
        cases = (
            source.replace("^exit(%s2, %b : !atlas.virtual_state, !atlas.virtual_bf16)", "^exit(%s2 : !atlas.virtual_state)"),
            source.replace("^exit(%s2, %b : !atlas.virtual_state, !atlas.virtual_bf16)", "^exit(%s2, %choose : !atlas.virtual_state, i1)"),
            source.replace("^exit(%s2, %b :", "^exit(%s1, %b :"),
        )
        for changed in cases:
            with self.assertRaises(VirtualInterfaceError):
                evaluate(parse_program(changed), RuntimeInputs(TILES, (Scalar(1, 1),)))

    def test_a_previous_visit_does_not_make_a_nondominating_definition_valid(self) -> None:
        source = function('''
          %s = "atlas.virtual_start"() : () -> !atlas.virtual_state
          %zero = arith.constant 0 : i32
          %one = arith.constant 1 : i32
          cf.br ^head(%s, %zero : !atlas.virtual_state, i32)
        ^head(%hs: !atlas.virtual_state, %i: i32):
          %first = arith.cmpi eq, %i, %zero : i32
          cf.cond_br %first, ^body(%hs, %i : !atlas.virtual_state, i32), ^exit(%hs : !atlas.virtual_state)
        ^body(%bs: !atlas.virtual_state, %j: i32):
          %late = arith.addi %j, %one : i32
          cf.br ^head(%bs, %late : !atlas.virtual_state, i32)
        ^exit(%es: !atlas.virtual_state):
          %invalid = arith.cmpi eq, %late, %one : i32
          func.return %es : !atlas.virtual_state
        ''')
        # The chosen path visits body before exit, but SSA dominance requires
        # this value to cross the edge explicitly as a block argument.
        with self.assertRaises(VirtualInterfaceError):
            evaluate(parse_program(source), RuntimeInputs())


if __name__ == "__main__":
    unittest.main()
