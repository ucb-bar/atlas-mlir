"""MXU1 anchor arithmetic and independent per-unit virtual resources.

Expectations are raw literals derived from the selected anchor/FMA sources and
exact rational witnesses, not calls to model numerics or compiler lowering.
The anchor-loss and negative-zero cases are source-derived checks, separate
from historical selected-core observations of the small finite witnesses.
"""

from __future__ import annotations

import unittest

from test_virtual_evaluator_mxu import rtl_arithmetic_cases, sparse as matrix
from test_virtual_evaluator_mxu_handles import A, W, S, TILES, accumulate, diagonal, function, legacy, output, program, read, reset, seed, weight
from atlas_virtual_evaluator import RuntimeInputs, Tile, VirtualInterfaceError, evaluate, parse_program


def on_unit(text: str, unit: int) -> str:
    return text.replace(W, f"!atlas.virtual_mxu_weight<{unit}>").replace(A, f"!atlas.virtual_mxu_acc<{unit}>").replace("unit = 0 : i32", f"unit = {unit} : i32")


def sparse(format: str, codes: tuple[int, ...]) -> Tile:
    return Tile(format, codes + (0,) * (1024 - len(codes)))


def kernel(body: tuple[str, ...], final_state: str, results: tuple[str, ...]) -> str:
    operations = list(body)
    current = final_state
    for index, result in enumerate(results):
        after = f"out{index}"
        operations.append(output(current, after, result, index))
        current = after
    operations.append(f"func.return %{current} : {S}")
    return function("\n".join(operations))


class VirtualEvaluatorMXU1Test(unittest.TestCase):
    def assert_outputs(self, source: str, tiles: dict[int, Tile], expected: dict[int, Tile]) -> None:
        inputs = RuntimeInputs(tiles)
        result = evaluate(parse_program(source), inputs)
        self.assertEqual(dict(result.outputs), expected)
        self.assertEqual(dict(inputs.tiles), tiles)
        self.assertEqual(result.memory, inputs.memory)

    def reject(self, source: str, tiles: dict[int, Tile], error=VirtualInterfaceError) -> None:
        with self.assertRaises(error):
            evaluate(parse_program(source), RuntimeInputs(tiles))

    def test_reset_literals_distinguish_rounding_and_anchor_truncation(self) -> None:
        cases = (
            ("two_half_ulp_terms", (0x38, 0x18, 0x18), (0x38, 0x18, 0x18), 0x3F80, 0x3F81),
            ("above_midpoint", (0x38, 0x18, 0x08), (0x38, 0x18, 0x08), 0x3F80, 0x3F81),
            # +/-448^2 cancel. With anchor 24, the 2^-12 term shifts by
            # 36 and vanishes from the 32-bit IPT integer representation.
            ("anchor_discards_residual", (0x7E, 0xFE, 0x08), (0x7E, 0x7E, 0x08), 0x3980, 0x0000),
            # IPT is unchanged by product permutation; MXU0's K order matters.
            ("residual_before_cancellation", (0x08, 0x7E, 0xFE), (0x08, 0x7E, 0x7E), 0x0000, 0x0000),
        )
        source = kernel((legacy("sa"), on_unit(legacy("ipt"), 1)), "s4", ("sa", "ipt"))
        for name, acts, weights, sa_bits, ipt_bits in cases:
            with self.subTest(case=name):
                tiles = {**TILES, 0: sparse("fp8", acts), 1: sparse("fp8", weights)}
                self.assert_outputs(source, tiles, {0: sparse("bf16", (sa_bits,)), 1: sparse("bf16", (ipt_bits,))})

    def test_instruction_boundary_rounding_has_no_persistent_anchor(self) -> None:
        # First tile is 1+1/256 -> 1. Continuing with 1/256 ties back to 1;
        # retaining exact state across instructions would instead produce 3f81.
        tiles = {**TILES, 0: sparse("fp8", (0x38, 0x18)), 1: sparse("fp8", (0x38, 0x18)),
                 2: sparse("fp8", (0x18,))}
        second = accumulate("t3", "t4", "a1", "a0", "w1").replace("%x,", "%v,")
        second_reset = legacy("second_reset").replace("%x, %w", "%v, %v")
        body = (legacy("first"), weight("s4", "t0", "w0"), reset("t0", "t1", "a0"),
                weight("t1", "t3", "w1", "v"), second, read("t4", "t5", "continued", "a1"), second_reset)
        source = on_unit(kernel(body, "t5", ("first", "continued", "second_reset")), 1)
        self.assert_outputs(source, tiles, {0: sparse("bf16", (0x3F80,)), 1: sparse("bf16", (0x3F80,)), 2: sparse("bf16", (0x3B80,))})

    def test_continuation_rounds_once_per_mxu1_instruction(self) -> None:
        tiles = {**TILES, 0: sparse("fp8", (0x38,)), 1: sparse("fp8", (0x38,)),
                 2: sparse("fp8", (0x18, 0x18))}
        second = accumulate("t2", "t3", "a1", "a0", "w1").replace("%x,", "%v,")
        body = (weight("s4", "t0", "w0"), reset("t0", "t1", "a0"), weight("t1", "t2", "w1", "v"),
                second, read("t3", "t4", "y", "a1"))
        for unit, bits in ((0, 0x3F80), (1, 0x3F81)):
            with self.subTest(unit=unit):
                self.assert_outputs(on_unit(kernel(body, "t4", ("y",)), unit), tiles, {0: sparse("bf16", (bits,))})

    def test_both_units_use_weight_rows_as_output_columns(self) -> None:
        acts, weights, wanted = [0] * 1024, [0] * 1024, [0] * 1024
        acts[2 * 32 + 3], acts[2 * 32 + 7] = 0x38, 0x40
        weights[5 * 32 + 3], weights[20 * 32 + 7] = 0x44, 0x48
        wanted[2 * 32 + 5], wanted[2 * 32 + 20] = 0x4040, 0x4100
        expected = Tile("bf16", tuple(wanted))
        tiles = {**TILES, 0: Tile("fp8", tuple(acts)), 1: Tile("fp8", tuple(weights))}
        source = kernel((legacy("sa"), on_unit(legacy("ipt"), 1)), "s4", ("sa", "ipt"))
        self.assert_outputs(source, tiles, {0: expected, 1: expected})

    def test_two_weights_and_accumulators_are_independent_on_each_unit(self) -> None:
        body = (
            weight("s4", "t0", "sw0"), weight("t0", "t1", "sw1", "v"),
            on_unit(weight("t1", "t2", "iw0"), 1), on_unit(weight("t2", "t3", "iw1", "v"), 1),
            reset("t3", "t4", "sa0", "sw0"), reset("t4", "t5", "sa1", "sw1"),
            on_unit(reset("t5", "t6", "ia0", "iw0"), 1), on_unit(reset("t6", "t7", "ia1", "iw1"), 1),
            on_unit(read("t7", "t8", "iy1", "ia1"), 1), read("t8", "t9", "sy1", "sa1"),
            on_unit(read("t9", "t10", "iy0", "ia0"), 1), read("t10", "t11", "sy0", "sa0"),
        )
        self.assert_outputs(kernel(body, "t11", ("iy1", "sy1", "iy0", "sy0")), TILES,
                            {0: diagonal("bf16", 0x4000), 1: diagonal("bf16", 0x4000),
                             2: diagonal("bf16", 0x3F80), 3: diagonal("bf16", 0x3F80)})

    def test_legacy_other_unit_is_allowed_but_own_unit_overlap_is_rejected(self) -> None:
        body = (on_unit(weight("s4", "t0", "w0"), 1), on_unit(reset("t0", "t1", "a0"), 1),
                legacy("sa"), on_unit(read("t1", "t2", "ipt", "a0"), 1))
        self.assert_outputs(kernel(body, "t2", ("sa", "ipt")), TILES,
                            {0: diagonal("bf16", 0x3F80), 1: diagonal("bf16", 0x3F80)})
        for body, state in (
            ((weight("s4", "t0", "w0"), legacy(), reset("t0", "t1", "a0"), read("t1", "t2", "y", "a0")), "t2"),
            ((seed("s4", "t0", "a0"), legacy(), read("t0", "t1", "y", "a0")), "t1"),
        ):
            with self.subTest(state=state):
                self.reject(on_unit(program(*body, final_state=state), 1), TILES)

    def test_mxu1_stale_forked_and_repeated_readout_versions_are_rejected(self) -> None:
        prefix = (weight("s4", "t0", "w0"), reset("t0", "t1", "a0"), accumulate("t1", "t2", "a1", "a0"))
        cases = (
            (*prefix, read("t2", "t3", "old", "a0"), read("t3", "t4", "current", "a1")),
            (*prefix, accumulate("t2", "t3", "fork", "a0"), read("t3", "t4", "first", "a1"), read("t4", "t5", "second", "fork")),
            (seed("s4", "t0", "a0"), read("t0", "t1", "first", "a0"), read("t1", "t2", "again", "a0")),
        )
        for body, state in zip(cases, ("t4", "t5", "t2")):
            with self.subTest(state=state):
                self.reject(on_unit(program(*body, final_state=state), 1), TILES)

    def test_signed_zero_seed_copy_and_zero_product_continuation_differ(self) -> None:
        negative_zero = Tile("bf16", (0x8000,) * 1024)
        zero_codes = tuple(range(8)) + tuple(range(0x80, 0x88))
        tiles = {**TILES, 0: Tile("fp8", zero_codes * 64), 1: Tile("fp8", (0x38,) * 1024), 3: negative_zero}
        body = (on_unit(seed("s4", "t0", "copy"), 1), on_unit(read("t0", "t1", "raw", "copy"), 1),
                weight("t1", "t2", "sw"), seed("t2", "t3", "sa_acc0"), accumulate("t3", "t4", "sa_acc1", "sa_acc0", "sw"), read("t4", "t5", "sa", "sa_acc1"),
                on_unit(weight("t5", "t6", "iw"), 1), on_unit(seed("t6", "t7", "i0"), 1),
                on_unit(accumulate("t7", "t8", "i1", "i0", "iw"), 1), on_unit(read("t8", "t9", "ipt", "i1"), 1))
        self.assert_outputs(kernel(body, "t9", ("raw", "sa", "ipt")), tiles,
                            {0: negative_zero, 1: negative_zero, 2: Tile("bf16", (0,) * 1024)})

    def test_raw_seed_and_readout_preserve_their_conversion_rules(self) -> None:
        raw_bf16 = (0x8000, 0xFFA1, 0x7F80, 0x0001)
        raw_fp8 = (0x80, 0x81, 0xFF, 0x38, 0xB8, 0x7E, 0xFE, 0x7F)
        decoded = (0x8000, 0x8000, 0x8000, 0x3F80, 0xBF80, 0x43E0, 0xC3E0, 0x0000)
        tiles = {**TILES, 0: Tile("fp8", raw_fp8 * 128), 3: Tile("bf16", raw_bf16 * 256)}
        body = (seed("s4", "t0", "b0"), seed("t0", "t1", "f0", "x", "fp8"),
                read("t1", "t2", "fp8_seed", "f0"), read("t2", "t3", "bf16_seed", "b0"))
        self.assert_outputs(on_unit(kernel(body, "t3", ("fp8_seed", "bf16_seed")), 1), tiles,
                            {0: Tile("bf16", decoded * 128), 1: tiles[3]})

    def test_mxu1_scaled_fp8_readout_uses_the_shared_mxu_converter(self) -> None:
        # Seed the FP8 readout into a fresh accumulator to observe its decoded
        # bits through BF16 boundary I/O, without inventing an FP8 output op.
        source = f'''{seed("s4", "t0", "a0")}
          %scale = "atlas.virtual_scale_constant"() {{code = 127 : i32}} : () -> !atlas.virtual_scale
          %t1, %packed = "atlas.virtual_mxu_readout_fp8"(%t0, %a0, %scale) : ({S}, {A}, !atlas.virtual_scale) -> ({S}, !atlas.virtual_fp8)
          {seed("t1", "t2", "a1", "packed", "fp8")}
          {read("t2", "t3", "y", "a1")}
        '''
        raw = (0x43F0, 0xC3F0, 0x7F80, 0xFF80, 0xFFA1, 0x0001, 0x8000, 0x3F80)
        wanted = (0x0000, 0x8000, 0x43E0, 0xC3E0, 0x0000, 0x0000, 0x0000, 0x3F80)
        tiles = {**TILES, 3: Tile("bf16", raw * 128)}
        self.assert_outputs(on_unit(kernel((source,), "t3", ("y",)), 1), tiles, {0: Tile("bf16", wanted * 128)})
        tiles = {**TILES, 3: diagonal("bf16", 0x3F80)}
        scaled = on_unit(kernel((source.replace("code = 127", "code = 128"),), "t3", ("y",)), 1)
        self.assert_outputs(scaled, tiles, {0: diagonal("bf16", 0x4000)})

    def test_raw_fp8_products_use_multiplier_semantics_on_mxu1(self) -> None:
        codes = tuple(range(8)) + tuple(range(0x80, 0x88)) + (0x7F, 0xFF)
        acts = matrix({(row, 0): code for row, code in enumerate(codes)})
        weights = matrix({(col, 0): code for col, code in enumerate((0x38, 0xB8, 0x7F, 0xFF))})
        wanted = {(16, 0): 0x43F0, (16, 1): 0xC3F0, (16, 2): 0x4861, (16, 3): 0xC861,
                  (17, 0): 0xC3F0, (17, 1): 0x43F0, (17, 2): 0xC861, (17, 3): 0x4861}
        programs = (kernel((legacy("y"),), "s4", ("y",)),
                    kernel((weight("s4", "t0", "w0"), reset("t0", "t1", "a0"), read("t1", "t2", "y", "a0")), "t2", ("y",)))
        for index, program in enumerate(programs):
            source = on_unit(program, 1)
            for x, w, expected in ((acts, weights, wanted), (weights, acts, {(col, row): bits for (row, col), bits in wanted.items()})):
                with self.subTest(explicit=bool(index), swapped=x is weights):
                    self.assert_outputs(source, {**TILES, 0: x, 1: w}, {0: matrix(expected, "bf16")})

    def test_exceptional_fp8_anchor_sum_matches_pinned_rtl_outputs(self) -> None:
        body = (weight("s4", "t0", "w0"), seed("t0", "t1", "a0"),
                accumulate("t1", "t2", "a1", "a0"), read("t2", "t3", "y", "a1"))
        source = on_unit(kernel(body, "t3", ("y",)), 1)
        # The independent RTL lane sums all 32 products with this partial.
        for case in rtl_arithmetic_cases():
            with self.subTest(raw=hex(case["a"][0])):
                tiles = {**TILES, 0: sparse("fp8", tuple(case["a"])), 1: sparse("fp8", tuple(case["w"])),
                         3: sparse("bf16", (case["partial"],))}
                self.assert_outputs(source, tiles, {0: sparse("bf16", (case["ipt"],))})

    def test_mxu1_exceptional_bf16_seeds_are_anchor_integers(self) -> None:
        continued = on_unit(kernel((weight("s4", "t0", "w0"), seed("t0", "t1", "a0"),
                                   accumulate("t1", "t2", "a1", "a0"), read("t2", "t3", "y", "a1")), "t3", ("y",)), 1)
        raw = (0x0001, 0x007F, 0x8001, 0x807F, 0x7F80, 0xFF80, 0x7FC1, 0xFFA1, 0x7FFF, 0xFFFF)
        initial = matrix({(row, 0): code for row, code in enumerate(raw)}, "bf16")
        weights = matrix({(0, 0): 0x38})
        # With zero products, subnormal seeds convert below the normal range
        # and are sanitized to +0. Raw exponent-255 seeds clamp by sign.
        large = (0x7F7F, 0xFF7F, 0x7F7F, 0xFF7F, 0x7F7F, 0xFF7F)
        wanted = matrix({(row + 4, 0): code for row, code in enumerate(large)}, "bf16")
        self.assert_outputs(continued, {**TILES, 0: matrix({}), 1: weights, 3: initial}, {0: wanted})
        for act, product in ((0x38, 0x3F80), (0xB8, 0xBF80)):
            with self.subTest(activation=hex(act)):
                acts = matrix({(row, 0): act for row in range(len(raw))})
                expected = (product,) * 4 + large
                wanted = matrix({(row, 0): code for row, code in enumerate(expected)}, "bf16")
                self.assert_outputs(continued, {**TILES, 0: acts, 1: weights, 3: initial}, {0: wanted})

    def test_all_pinned_rtl_anchor_cases_including_raw_bf16_seeds(self) -> None:
        body = (weight("s4", "t0", "w0"), seed("t0", "t1", "a0"),
                accumulate("t1", "t2", "a1", "a0"), read("t2", "t3", "y", "a1"))
        source = parse_program(on_unit(kernel(body, "t3", ("y",)), 1))
        cases = rtl_arithmetic_cases(all_cases=True)
        for first in range(0, len(cases), 32):
            batch = cases[first:first + 32]
            acts = matrix({(row, k): code for row, case in enumerate(batch) for k, code in enumerate(case["a"])})
            weights = matrix({(row, k): code for row, case in enumerate(batch) for k, code in enumerate(case["w"])})
            initial = matrix({(row, row): case["partial"] for row, case in enumerate(batch)}, "bf16")
            runtime = RuntimeInputs({**TILES, 0: acts, 1: weights, 3: initial})
            result = evaluate(source, runtime).outputs[0]
            for row, case in enumerate(batch):
                with self.subTest(case=first + row, seed=hex(case["partial"])):
                    self.assertEqual(result.bits[row * 32 + row], case["ipt"])
            self.assertEqual(runtime.tiles[3], initial)


if __name__ == "__main__":
    unittest.main()
