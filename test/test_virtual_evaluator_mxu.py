"""MXU0 reference execution with independent raw literals and no Atlas tools."""

from __future__ import annotations

from pathlib import Path
import hashlib
import json
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
from atlas_virtual_evaluator import RuntimeInputs, Scalar, Tile, _mxu_tile, evaluate, parse_program  # noqa: E402

STATE = "!atlas.virtual_state"
FP8 = "!atlas.virtual_fp8"
BF16 = "!atlas.virtual_bf16"
WEIGHT = "!atlas.virtual_mxu_weight<0>"
ACC = "!atlas.virtual_mxu_acc<0>"
SCALE = "!atlas.virtual_scale"

PREFIX = f'''
  %s0 = "atlas.virtual_start"() : () -> {STATE}
  %s1, %x = "atlas.virtual_input_fp8"(%s0) {{index = 0 : i32}} : ({STATE}) -> ({STATE}, {FP8})
  %s2, %w = "atlas.virtual_input_fp8"(%s1) {{index = 1 : i32}} : ({STATE}) -> ({STATE}, {FP8})
  %s3, %seed = "atlas.virtual_input_bf16"(%s2) {{index = 2 : i32}} : ({STATE}) -> ({STATE}, {BF16})
'''


def repeated(values: tuple[int, ...], format="bf16") -> Tile:
    return Tile(format, tuple(values[i % len(values)] for i in range(1024)))


def sparse(values: dict[tuple[int, int], int], format="fp8") -> Tile:
    return Tile(format, tuple(values.get((row, col), 0) for row in range(32) for col in range(32)))


def inputs(x=None, w=None, seed=None, *, controls=()) -> RuntimeInputs:
    return RuntimeInputs({0: x if x is not None else repeated((0,), "fp8"),
                          1: w if w is not None else repeated((0,), "fp8"),
                          2: seed if seed is not None else repeated((0,))}, controls)


class Stream:
    """Small textual fixture builder; expected values never use this builder."""

    def __init__(self):
        self.lines = [PREFIX]
        self.state = "%s3"
        self.count = 0

    def effect(self, name, args, types, result, result_type, attributes=""):
        self.count += 1
        next_state = f"%event{self.count}"
        self.lines.append(f'{next_state}, %{result} = "atlas.virtual_{name}"({self.state}, {args}) '
                          f'{attributes} : ({STATE}, {types}) -> ({STATE}, {result_type})')
        self.state = next_state

    def weight(self):
        self.effect("mxu_load_weight", "%w", FP8, "weight", WEIGHT, "{unit = 0 : i32}")

    def reset(self, result):
        self.effect("mxu_reset", "%x, %weight", f"{FP8}, {WEIGHT}", result, ACC)

    def accumulate(self, previous, result):
        self.effect("mxu_accumulate", f"%x, %weight, %{previous}", f"{FP8}, {WEIGHT}, {ACC}", result, ACC)

    def seed(self, source="seed", format="bf16", result="acc"):
        type = BF16 if format == "bf16" else FP8
        self.effect(f"mxu_load_acc_{format}", f"%{source}", type, result, ACC, "{unit = 0 : i32}")

    def readout(self, accumulator, result):
        self.effect("mxu_readout_bf16", f"%{accumulator}", ACC, result, BF16)

    def fp8_readout(self, code):
        self.lines.append(f'%scale = "atlas.virtual_scale_constant"() {{code = {code} : i32}} : () -> {SCALE}')
        self.effect("mxu_readout_fp8", "%acc, %scale", f"{ACC}, {SCALE}", "packed", FP8)
        self.seed("packed", "fp8", "decoded_acc")
        self.readout("decoded_acc", "decoded")

    def output(self, value, index=0):
        self.count += 1
        next_state = f"%event{self.count}"
        self.lines.append(f'{next_state} = "atlas.virtual_output_bf16"({self.state}, %{value}) '
                          f'{{index = {index} : i32}} : ({STATE}, {BF16}) -> {STATE}')
        self.state = next_state

    def program(self):
        return parse_program("module {" + "\n".join(self.lines) + "}")


def legacy_program():
    stream = Stream()
    stream.lines.append(f'%result = "atlas.virtual_mxu_matmul"(%x, %w) {{unit = 0 : i32}} '
                        f': ({FP8}, {FP8}) -> {BF16}')
    stream.output("result")
    return stream.program()


def reset_program():
    stream = Stream()
    stream.weight()
    stream.reset("acc")
    stream.readout("acc", "result")
    stream.output("result")
    return stream.program()


def rtl_arithmetic_cases(*, all_cases=False):
    import npu_model.configs.numerics as numerics
    path = Path(numerics.__file__).resolve().parents[2] / "tests/rtl/arithmetic.json"
    data = path.read_bytes()
    if hashlib.sha256(data).hexdigest() != "8b2ebb3cb98b4c9e7c7c0f4cb71e86d8a6675c698abec0c3e0525bc693736840":
        raise AssertionError("pinned MXU RTL arithmetic fixture changed")
    cases = json.loads(data)
    return tuple(cases) if all_cases else tuple(cases[index] for index in (1, 127, 129, 255))


class VirtualEvaluatorMxuTest(unittest.TestCase):
    def test_legacy_and_explicit_reset_use_n_by_k_weights_and_fresh_inputs(self):
        x = sparse({(0, 0): 0x38, (0, 1): 0x40, (1, 0): 0x44, (1, 1): 0x48})
        w = sparse({(0, 0): 0x38, (0, 1): 0x44, (1, 0): 0x40, (1, 1): 0x48})
        expected = sparse({(0, 0): 0x40E0, (0, 1): 0x4120, (1, 0): 0x4170, (1, 1): 0x41B0}, "bf16")
        original = inputs(x, w)
        for program in (legacy_program(), reset_program()):
            with self.subTest(explicit=len(program.operations) > 6):
                self.assertEqual(evaluate(program, original).outputs[0], expected)
                self.assertEqual(evaluate(program, inputs(w=w)).outputs[0], repeated((0,)))
                self.assertEqual(evaluate(program, original).outputs[0], expected)
                self.assertEqual(dict(original.tiles), {0: x, 1: w, 2: repeated((0,))})

    def test_each_mac_rounds_bf16_in_increasing_k_order_and_from_the_seed(self):
        # Products 1, 1/256, 1/256: both half-ULP steps tie back to 1.
        forward = sparse({(0, 0): 0x38, (0, 1): 0x18, (0, 2): 0x18})
        reverse = sparse({(0, 0): 0x18, (0, 1): 0x18, (0, 2): 0x38})
        for program in (legacy_program(), reset_program()):
            self.assertEqual(evaluate(program, inputs(forward, forward)).outputs[0], sparse({(0, 0): 0x3F80}, "bf16"))
            self.assertEqual(evaluate(program, inputs(reverse, reverse)).outputs[0], sparse({(0, 0): 0x3F81}, "bf16"))
        small = sparse({(0, 0): 0x18, (0, 1): 0x18})
        seed = sparse({(0, 0): 0x3F80}, "bf16")
        stream = Stream()
        stream.weight()
        stream.seed()
        stream.accumulate("acc", "continued")
        stream.readout("continued", "result")
        stream.output("result")
        self.assertEqual(evaluate(stream.program(), inputs(small, small, seed)).outputs[0], seed)

    def test_reset_and_continuation_store_bf16_and_reuse_resident_weights(self):
        stream = Stream()
        stream.weight()
        stream.reset("first")
        stream.readout("first", "thirty_two")
        stream.output("thirty_two", 0)
        stream.reset("second")
        stream.accumulate("second", "continued")
        stream.readout("continued", "sixty_four")
        stream.output("sixty_four", 1)
        stream.reset("replacement")
        stream.readout("replacement", "reset_again")
        stream.output("reset_again", 2)
        ones = repeated((0x38,), "fp8")
        self.assertEqual(dict(evaluate(stream.program(), inputs(ones, ones)).outputs),
                         {0: repeated((0x4200,)), 1: repeated((0x4280,)), 2: repeated((0x4200,))})

    def test_zero_products_preserve_signed_seed_but_reset_starts_positive_zero(self):
        stream = Stream()
        stream.weight()
        stream.reset("reset")
        stream.readout("reset", "reset_result")
        stream.output("reset_result", 0)
        stream.seed()
        stream.accumulate("acc", "continued")
        stream.readout("continued", "seed_result")
        stream.output("seed_result", 1)
        seed = repeated((0x0000, 0x8000))
        zero = repeated(tuple(range(8)) + tuple(range(0x80, 0x88)), "fp8")
        self.assertEqual(dict(evaluate(stream.program(), inputs(zero, repeated((0xB8,), "fp8"), seed)).outputs),
                         {0: repeated((0,)), 1: seed})

    def test_raw_fp8_products_flush_exponent_zero_but_reserved_codes_multiply_as_480(self):
        codes = tuple(range(8)) + tuple(range(0x80, 0x88)) + (0x7F, 0xFF)
        x = sparse({(row, 0): code for row, code in enumerate(codes)})
        w = sparse({(col, 0): code for col, code in enumerate((0x38, 0xB8, 0x7F, 0xFF))})
        # +/-480 * (+1, -1, +480, -480), including exact +/-230400.
        wanted = {(16, 0): 0x43F0, (16, 1): 0xC3F0, (16, 2): 0x4861, (16, 3): 0xC861,
                  (17, 0): 0xC3F0, (17, 1): 0x43F0, (17, 2): 0xC861, (17, 3): 0x4861}
        for program in (legacy_program(), reset_program()):
            for acts, weights, expected in ((x, w, wanted), (w, x, {(col, row): bits for (row, col), bits in wanted.items()})):
                with self.subTest(explicit=len(program.operations) > 6, swapped=acts is w):
                    runtime = inputs(acts, weights)
                    self.assertEqual(evaluate(program, runtime).outputs[0], sparse(expected, "bf16"))
                    self.assertEqual(dict(runtime.tiles), dict(inputs(acts, weights).tiles))

    def test_exceptional_fp8_single_mac_matches_pinned_rtl_outputs(self):
        stream = Stream()
        stream.weight()
        stream.seed()
        stream.accumulate("acc", "continued")
        stream.readout("continued", "result")
        stream.output("result")
        program = stream.program()
        # The RTL harness's FMA output uses only A[0], W[0], and partial.
        for case in rtl_arithmetic_cases():
            with self.subTest(raw=hex(case["a"][0])):
                runtime = inputs(sparse({(0, 0): case["a"][0]}), sparse({(0, 0): case["w"][0]}),
                                 sparse({(0, 0): case["partial"]}, "bf16"))
                self.assertEqual(evaluate(program, runtime).outputs[0], sparse({(0, 0): case["fma"]}, "bf16"))

    def test_bf16_seed_and_readout_preserve_every_raw_encoding(self):
        stream = Stream()
        stream.seed()
        stream.readout("acc", "result")
        stream.output("result")
        program = stream.program()
        for first in range(0, 65536, 1024):
            with self.subTest(first=hex(first)):
                seed = Tile("bf16", tuple(range(first, first + 1024)))
                runtime = inputs(seed=seed)
                self.assertEqual(evaluate(program, runtime).outputs[0], seed)
                self.assertEqual(runtime.tiles[2], seed)

    def test_fp8_seed_decodes_specials_to_signed_zero_without_a_scale(self):
        raw = (0x00, 0x80, 0x01, 0x81, 0x07, 0x87, 0x7F, 0xFF, 0x08, 0x18, 0x38, 0xB8, 0x7E, 0xFE)
        expected = (0, 0x8000, 0, 0x8000, 0, 0x8000, 0, 0x8000, 0x3C80, 0x3D80, 0x3F80, 0xBF80, 0x43E0, 0xC3E0)
        stream = Stream()
        stream.seed("x", "fp8")
        stream.readout("acc", "result")
        stream.output("result")
        runtime = inputs(x=repeated(raw, "fp8"))
        self.assertEqual(evaluate(stream.program(), runtime).outputs[0], repeated(expected))
        self.assertEqual(runtime.tiles[0], repeated(raw, "fp8"))

    def test_fp8_readout_uses_mxu_scaling_and_reserved_480_encoding(self):
        # Expected values are BF16 after the FP8 result is seeded back. In
        # particular rounded +/-480 produces reserved 7f/ff, decoded to +/-0.
        cases = (
            (126, (0x3F80, 0xBF80), (0x3F00, 0xBF00)),
            (127, (0x3F80, 0xBF80, 0x3F88, 0x3F98, 0x43E8, 0x43E9, 0x43F0, 0xC3E9, 0xC3F0),
                  (0x3F80, 0xBF80, 0x3F80, 0x3FA0, 0x43E0, 0, 0, 0x8000, 0x8000)),
            (128, (0x3F80, 0xBF80), (0x4000, 0xC000)),
            (0, (0x3F80, 0x7F7F, 0xFF7F, 0x7F80, 0xFF80), (0, 0x4000, 0xC000, 0x43E0, 0xC3E0)),
            (254, (0x0080, 0x8080, 0x0001, 0x8001, 0x7FC1, 0xFFC1, 0x8000), (0x4000, 0xC000, 0, 0, 0, 0, 0)),
            (255, (0x0080, 0x8080, 0x3F80), (0x4000, 0xC000, 0x43E0)),
        )
        for code, raw, expected in cases:
            with self.subTest(code=code):
                stream = Stream()
                stream.seed()
                stream.fp8_readout(code)
                stream.output("decoded")
                seed = repeated(raw)
                runtime = inputs(seed=seed)
                self.assertEqual(evaluate(stream.program(), runtime).outputs[0], repeated(expected))
                self.assertEqual(runtime.tiles[2], seed)

    def test_all_pinned_rtl_mxu_converter_cases_cover_every_scale(self):
        cases = rtl_arithmetic_cases(all_cases=True)
        self.assertEqual({case["scale"] for case in cases}, set(range(256)))
        for code in range(256):
            batch = tuple(case for case in cases if case["scale"] == code)
            raw = tuple(case["partial"] for case in batch)
            wanted = tuple(case["quant"] for case in batch)
            with self.subTest(scale=code):
                # Inspect the raw adapter result: reseeding would erase reserved
                # 7f/ff encodings and could hide an incorrect converter result.
                result = _mxu_tile("readout_fp8", (repeated(raw),), code)
                self.assertEqual(result, repeated(wanted, "fp8"))

    def test_exceptional_bf16_seeds_follow_custom_mac_zero_and_nonzero_paths(self):
        stream = Stream()
        stream.weight()
        stream.seed()
        stream.accumulate("acc", "continued")
        stream.readout("continued", "result")
        stream.output("result")
        program = stream.program()
        raw = (0x0001, 0x007F, 0x8001, 0x807F, 0x7F80, 0xFF80, 0x7FC1, 0xFFA1, 0x7FFF, 0xFFFF)
        seed = sparse({(row, 0): code for row, code in enumerate(raw)}, "bf16")
        weights = sparse({(0, 0): 0x38})
        # A zero product bypasses even Inf/NaN unchanged. Any nonzero product
        # replaces an exponent-zero addend; exponent-255 addends clamp finite.
        self.assertEqual(evaluate(program, inputs(w=weights, seed=seed)).outputs[0], seed)
        for act, product in ((0x38, 0x3F80), (0xB8, 0xBF80)):
            with self.subTest(activation=hex(act)):
                expected = (product,) * 4 + (0x7F7F, 0xFF7F, 0x7F7F, 0xFF7F, 0x7F7F, 0xFF7F)
                acts = sparse({(row, 0): act for row in range(len(raw))})
                wanted = sparse({(row, 0): code for row, code in enumerate(expected)}, "bf16")
                runtime = inputs(acts, weights, seed)
                self.assertEqual(evaluate(program, runtime).outputs[0], wanted)
                self.assertEqual(runtime.tiles[2], seed)

    def test_all_pinned_rtl_single_mac_cases_including_raw_bf16_seeds(self):
        stream = Stream()
        stream.weight()
        stream.seed()
        stream.accumulate("acc", "continued")
        stream.readout("continued", "result")
        stream.output("result")
        program = stream.program()
        cases = rtl_arithmetic_cases(all_cases=True)
        # Batch unrelated harness cases on the diagonal; off-diagonal outputs
        # are intentionally unconstrained and do not provide expected values.
        for first in range(0, len(cases), 32):
            batch = cases[first:first + 32]
            acts = sparse({(row, 0): case["a"][0] for row, case in enumerate(batch)})
            weights = sparse({(row, 0): case["w"][0] for row, case in enumerate(batch)})
            seed = sparse({(row, row): case["partial"] for row, case in enumerate(batch)}, "bf16")
            runtime = inputs(acts, weights, seed)
            result = evaluate(program, runtime).outputs[0]
            for row, case in enumerate(batch):
                with self.subTest(case=first + row, seed=hex(case["partial"])):
                    self.assertEqual(result.bits[row * 32 + row], case["fma"])
            self.assertEqual(runtime.tiles[2], seed)

    def test_cfg_executes_only_selected_raw_seed_copy_or_contraction(self):
        source = f'''module {{
          func.func @choose(%take: i1) -> {STATE} {{
            {PREFIX}
            cf.cond_br %take, ^compute(%s3 : {STATE}), ^copy(%s3 : {STATE})
          ^compute(%cs: {STATE}):
            %c1, %weight = "atlas.virtual_mxu_load_weight"(%cs, %w) {{unit = 0 : i32}} : ({STATE}, {FP8}) -> ({STATE}, {WEIGHT})
            %c2, %acc = "atlas.virtual_mxu_load_acc_bf16"(%c1, %seed) {{unit = 0 : i32}} : ({STATE}, {BF16}) -> ({STATE}, {ACC})
            %c3, %continued = "atlas.virtual_mxu_accumulate"(%c2, %x, %weight, %acc) : ({STATE}, {FP8}, {WEIGHT}, {ACC}) -> ({STATE}, {ACC})
            %c4, %result = "atlas.virtual_mxu_readout_bf16"(%c3, %continued) : ({STATE}, {ACC}) -> ({STATE}, {BF16})
            %out = "atlas.virtual_output_bf16"(%c4, %result) {{index = 0 : i32}} : ({STATE}, {BF16}) -> {STATE}
            return %out : {STATE}
          ^copy(%ss: {STATE}):
            %copy_out = "atlas.virtual_output_bf16"(%ss, %seed) {{index = 1 : i32}} : ({STATE}, {BF16}) -> {STATE}
            return %copy_out : {STATE}
          }}
        }}'''
        program = parse_program(source)
        acts = repeated((0x7F,), "fp8")
        seed = repeated((0x7FC1, 0x8001))
        self.assertEqual(dict(evaluate(program, inputs(acts, seed=seed, controls=(Scalar(1, 0),))).outputs), {1: seed})
        # Default weights are zero: the custom MAC's bypass preserves raw seeds.
        self.assertEqual(dict(evaluate(program, inputs(acts, seed=seed, controls=(Scalar(1, 1),))).outputs), {0: seed})


if __name__ == "__main__":
    unittest.main()
