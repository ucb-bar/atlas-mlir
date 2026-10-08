"""MXU0 logical handle ownership through the public virtual evaluator.

The two-handle bounds are target-interface admission, not hardware qualification.
Small exact diagonal tiles distinguish identities without invoking compiler
allocation, scheduling, or the physical model's register/slot state.
"""

from __future__ import annotations

from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
from atlas_virtual_evaluator import RuntimeInputs, Scalar, Tile, UnsupportedVirtualMode, VirtualInterfaceError, evaluate, parse_program  # noqa: E402


S, B, F = "!atlas.virtual_state", "!atlas.virtual_bf16", "!atlas.virtual_fp8"
W, A = "!atlas.virtual_mxu_weight<0>", "!atlas.virtual_mxu_acc<0>"
PREFIX = f'''
  %s0 = "atlas.virtual_start"() : () -> {S}
  %s1, %x = "atlas.virtual_input_fp8"(%s0) {{index = 0 : i32}} : ({S}) -> ({S}, {F})
  %s2, %w = "atlas.virtual_input_fp8"(%s1) {{index = 1 : i32}} : ({S}) -> ({S}, {F})
  %s3, %v = "atlas.virtual_input_fp8"(%s2) {{index = 2 : i32}} : ({S}) -> ({S}, {F})
  %s4, %seed = "atlas.virtual_input_bf16"(%s3) {{index = 3 : i32}} : ({S}) -> ({S}, {B})
'''


def diagonal(format: str, code: int) -> Tile:
    return Tile(format, tuple(code if row == col else 0 for row in range(32) for col in range(32)))


TILES = {0: diagonal("fp8", 0x38), 1: diagonal("fp8", 0x38),
         2: diagonal("fp8", 0x40), 3: diagonal("bf16", 0x4040)}


def weight(before: str, after: str, name: str, source: str = "w") -> str:
    return f'%{after}, %{name} = "atlas.virtual_mxu_load_weight"(%{before}, %{source}) {{unit = 0 : i32}} : ({S}, {F}) -> ({S}, {W})'


def seed(before: str, after: str, name: str, source: str = "seed", format: str = "bf16") -> str:
    return f'%{after}, %{name} = "atlas.virtual_mxu_load_acc_{format}"(%{before}, %{source}) {{unit = 0 : i32}} : ({S}, {B if format == "bf16" else F}) -> ({S}, {A})'


def reset(before: str, after: str, name: str, weights: str = "w0") -> str:
    return f'%{after}, %{name} = "atlas.virtual_mxu_reset"(%{before}, %x, %{weights}) : ({S}, {F}, {W}) -> ({S}, {A})'


def accumulate(before: str, after: str, name: str, previous: str, weights: str = "w0") -> str:
    return f'%{after}, %{name} = "atlas.virtual_mxu_accumulate"(%{before}, %x, %{weights}, %{previous}) : ({S}, {F}, {W}, {A}) -> ({S}, {A})'


def read(before: str, after: str, tile: str, handle: str) -> str:
    return f'%{after}, %{tile} = "atlas.virtual_mxu_readout_bf16"(%{before}, %{handle}) : ({S}, {A}) -> ({S}, {B})'


def output(before: str, after: str, tile: str, index: int = 0) -> str:
    return f'%{after} = "atlas.virtual_output_bf16"(%{before}, %{tile}) {{index = {index} : i32}} : ({S}, {B}) -> {S}'


def legacy(name: str = "product") -> str:
    return f'%{name} = "atlas.virtual_mxu_matmul"(%x, %w) {{unit = 0 : i32}} : ({F}, {F}) -> {B}'


def function(body: str, arguments: str = "") -> str:
    return f'module {{ func.func @handles({arguments}) -> {S} {{\n{PREFIX}\n{body}\n}} }}'


def program(*body: str, final_state: str, flat: bool = False) -> str:
    text = "\n".join((*body, output(final_state, "done", "seed")))
    return f'module {{\n{PREFIX}\n{text}\n}}' if flat else function(text + f'\nfunc.return %done : {S}')


class VirtualEvaluatorMXUHandleTest(unittest.TestCase):
    def assert_outputs(self, source: str, expected: dict[int, Tile], controls: tuple[Scalar, ...] = ()) -> None:
        inputs = RuntimeInputs(TILES, controls)
        result = evaluate(parse_program(source), inputs)
        self.assertEqual(dict(result.outputs), expected)
        self.assertEqual(dict(inputs.tiles), TILES)
        self.assertEqual(result.memory, inputs.memory)

    def reject(self, source: str, controls: tuple[Scalar, ...] = (), error=VirtualInterfaceError) -> None:
        with self.assertRaises(error):
            evaluate(parse_program(source), RuntimeInputs(TILES, controls))

    def test_two_weights_and_chains_read_out_in_reverse_order(self) -> None:
        body = "\n".join((weight("s4", "t0", "w0"), weight("t0", "t1", "w1", "v"),
                          reset("t1", "t2", "a0"), reset("t2", "t3", "a1", "w1"),
                          read("t3", "t4", "y1", "a1"), read("t4", "t5", "y0", "a0"),
                          output("t5", "t6", "y1", 0), output("t6", "t7", "y0", 1),
                          f'func.return %t7 : {S}'))
        self.assert_outputs(function(body), {0: diagonal("bf16", 0x4000), 1: diagonal("bf16", 0x3F80)})

    def test_weight_last_use_retirement_and_new_weight_during_live_chain(self) -> None:
        # w0 remains usable after w1 loads. Its last use frees admission for w2
        # while a1 is still live; each accumulator successor remains distinct.
        body = "\n".join((weight("s4", "t0", "w0"), reset("t0", "t1", "a0"),
                          weight("t1", "t2", "w1", "v"), accumulate("t2", "t3", "a1", "a0"),
                          weight("t3", "t4", "w2", "v"), accumulate("t4", "t5", "a2", "a1", "w1"),
                          accumulate("t5", "t6", "a3", "a2", "w2"), read("t6", "t7", "y", "a3"),
                          output("t7", "done", "y"), f'func.return %done : {S}'))
        self.assert_outputs(function(body), {0: diagonal("bf16", 0x40C0)})

    def test_bf16_and_fp8_seeds_need_no_weights_and_preserve_distinct_chains(self) -> None:
        body = "\n".join((seed("s4", "t0", "a0"), seed("t0", "t1", "a1", "v", "fp8"),
                          read("t1", "t2", "y1", "a1"), read("t2", "t3", "y0", "a0"),
                          output("t3", "t4", "y1", 0), output("t4", "done", "y0", 1),
                          f'func.return %done : {S}'))
        self.assert_outputs(function(body), {0: diagonal("bf16", 0x4000), 1: TILES[3]})

    def test_seed_and_readout_preserve_weights_for_later_reset(self) -> None:
        body = "\n".join((weight("s4", "t0", "w0"), seed("t0", "t1", "a0"),
                          accumulate("t1", "t2", "a1", "a0"), read("t2", "t3", "first", "a1"),
                          reset("t3", "t4", "a2"), read("t4", "t5", "second", "a2"),
                          output("t5", "t6", "first", 0), output("t6", "done", "second", 1),
                          f'func.return %done : {S}'))
        self.assert_outputs(function(body), {0: diagonal("bf16", 0x4080), 1: diagonal("bf16", 0x3F80)})

    def test_dead_weights_allow_new_loads_and_legacy_matmul(self) -> None:
        # Zero-use weights are not resident for admission or legacy overlap.
        body = "\n".join((weight("s4", "t0", "w0"), weight("t0", "t1", "w1"),
                          weight("t1", "t2", "w2"), legacy(), output("t2", "done", "product"),
                          f'func.return %done : {S}'))
        self.assert_outputs(function(body), {0: diagonal("bf16", 0x3F80)})

    def test_legacy_matmul_after_last_weight_use_and_completed_readout(self) -> None:
        body = "\n".join((weight("s4", "t0", "w0"), reset("t0", "t1", "a0"),
                          read("t1", "t2", "first", "a0"), legacy(),
                          output("t2", "t3", "first", 0), output("t3", "done", "product", 1),
                          f'func.return %done : {S}'))
        self.assert_outputs(function(body), {0: diagonal("bf16", 0x3F80), 1: diagonal("bf16", 0x3F80)})

    def test_loop_block_visit_creates_fresh_handle_versions(self) -> None:
        body = f'''
          %zero = arith.constant 0 : i32
          %one = arith.constant 1 : i32
          %limit = arith.constant 2 : i32
          cf.br ^loop(%s4, %zero : {S}, i32)
        ^loop(%ls: {S}, %i: i32):
          {weight("ls", "t0", "w0")}
          {reset("t0", "t1", "a0")}
          {read("t1", "t2", "y", "a0")}
          %j = arith.addi %i, %one : i32
          %again = arith.cmpi ult, %j, %limit : i32
          cf.cond_br %again, ^loop(%t2, %j : {S}, i32), ^exit(%t2, %y : {S}, {B})
        ^exit(%es: {S}, %result: {B}):
          {output("es", "done", "result")}
          func.return %done : {S}
        '''
        self.assert_outputs(function(body), {0: diagonal("bf16", 0x3F80)})

    def test_third_live_weight_and_accumulator_are_rejected(self) -> None:
        weights = (weight("s4", "t0", "w0"), weight("t0", "t1", "w1"), weight("t1", "t2", "w2"),
                   reset("t2", "t3", "a0"), read("t3", "t4", "y0", "a0"),
                   reset("t4", "t5", "a1", "w1"), read("t5", "t6", "y1", "a1"),
                   reset("t6", "t7", "a2", "w2"), read("t7", "t8", "y2", "a2"))
        accumulators = (seed("s4", "t0", "a0"), seed("t0", "t1", "a1"), seed("t1", "t2", "a2"),
                        read("t2", "t3", "y0", "a0"), read("t3", "t4", "y1", "a1"), read("t4", "t5", "y2", "a2"))
        self.reject(program(*weights, final_state="t8"))
        self.reject(program(*accumulators, final_state="t5"))

    def test_stale_and_forked_accumulator_versions_are_rejected(self) -> None:
        prefix = (weight("s4", "t0", "w0"), reset("t0", "t1", "a0"), accumulate("t1", "t2", "a1", "a0"))
        self.reject(program(*prefix, read("t2", "t3", "old", "a0"), read("t3", "t4", "current", "a1"), final_state="t4"))
        self.reject(program(*prefix, accumulate("t2", "t3", "fork", "a0"),
                            read("t3", "t4", "first", "a1"), read("t4", "t5", "second", "fork"), final_state="t5"))

    def test_repeated_readout_and_consumed_accumulation_are_rejected(self) -> None:
        prefix = (weight("s4", "t0", "w0"), reset("t0", "t1", "a0"), read("t1", "t2", "first", "a0"))
        self.reject(program(*prefix, read("t2", "t3", "again", "a0"), final_state="t3"))
        self.reject(program(*prefix, accumulate("t2", "t3", "a1", "a0"), read("t3", "t4", "y", "a1"), final_state="t4"))

    def test_dropped_accumulator_at_return_flat_end_and_branch_is_rejected(self) -> None:
        for flat in (False, True):
            with self.subTest(flat=flat):
                self.reject(program(seed("s4", "t0", "a0"), final_state="t0", flat=flat))
        body = f'''{seed("s4", "t0", "a0")}
          cf.br ^exit(%t0 : {S})
        ^exit(%es: {S}):
          func.return %es : {S}'''
        self.reject(function(body))

    def test_wrong_state_and_mixed_handle_units_are_rejected(self) -> None:
        self.reject(program(weight("s4", "t0", "w0"), reset("s4", "t1", "a0"), read("t1", "t2", "y", "a0"), final_state="t2"))
        source = program(weight("s4", "t0", "w0"), reset("t0", "t1", "a0"), read("t1", "t2", "y", "a0"), final_state="t2")
        self.reject(source.replace(A, "!atlas.virtual_mxu_acc<1>"))

    def test_handles_cannot_cross_blocks_by_capture_or_argument(self) -> None:
        body = f'''{weight("s4", "t0", "w0")}
          cf.br ^next(%t0 : {S})
        ^next(%ns: {S}):
          {reset("ns", "t1", "a0")}
          {read("t1", "t2", "y", "a0")}
          func.return %t2 : {S}'''
        self.reject(function(body))
        passed = body.replace(f'^next(%t0 : {S})', f'^next(%t0, %w0 : {S}, {W})').replace(f'%ns: {S}):', f'%ns: {S}, %nw: {W}):').replace('%ns, %x, %w0', '%ns, %x, %nw')
        self.reject(function(passed))

    def test_legacy_overlap_with_live_weight_or_accumulator_is_rejected(self) -> None:
        self.reject(program(weight("s4", "t0", "w0"), legacy(), reset("t0", "t1", "a0"),
                            read("t1", "t2", "y", "a0"), final_state="t2"))
        self.reject(program(seed("s4", "t0", "a0"), legacy(), read("t0", "t1", "y", "a0"), final_state="t1"))

    def test_untaken_malformed_handle_path_is_rejected_before_execution(self) -> None:
        body = f'''
          cf.cond_br %choose, ^good(%s4 : {S}), ^bad(%s4 : {S})
        ^good(%gs: {S}):
          func.return %gs : {S}
        ^bad(%bs: {S}):
          {seed("bs", "t0", "a0")}
          {read("t0", "t1", "first", "a0")}
          {read("t1", "t2", "again", "a0")}
          func.return %t2 : {S}
        '''
        self.reject(function(body, "%choose: i1"), (Scalar(1, 1),))

    def test_mxu1_seed_and_readout_are_admitted(self) -> None:
        source = program(seed("s4", "t0", "a0"), read("t0", "t1", "y", "a0"), final_state="t1")
        source = source.replace(A, "!atlas.virtual_mxu_acc<1>").replace("unit = 0 : i32", "unit = 1 : i32")
        self.assert_outputs(source, {0: TILES[3]})


if __name__ == "__main__":
    unittest.main()
