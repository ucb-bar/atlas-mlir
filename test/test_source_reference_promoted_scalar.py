"""A BF16 tensor with a Python scalar and a BF16 result, through the reference's stage proof, bit for bit against torch.

The source binding comes from frontend MLIR through the reference's own provenance parser, as in the reference stage.
torch is the oracle here only for the expected bits; the evaluator under test never imports it.
"""

from unittest import TestCase, main

import numpy as np
import torch

from atlas_mlir_native.reference_eval import EvaluationError, device, evaluate
from atlas_mlir_native.reference_eval.decisions import SelectedDecisions, _source_provenance

_NODE = "g:quantized:root:n1"
_RULES = {"npu_model_revision": device.MODEL_REVISION, "numerics": "rtl", "k_order": "ascending-k32",
          "host_order": {"default": "lexicographic-f32"}}


def _mlir(target, arithmetic, literal, shape=(2, 3)):
    """The frontend lowering of ``x <op> scalar`` with a BF16 result: F32 literal, splat, widen and operate, narrow."""
    dims = "x".join(map(str, shape))
    f32, bf16 = f"tensor<{dims}xf32>", f"tensor<{dims}xbf16>"
    names = ", ".join(f"d{i}" for i in range(len(shape)))
    identity = f"affine_map<({names}) -> ({names})>"
    parallel = ", ".join(["#linalg.iterator_type<parallel>"] * len(shape))
    prov = f'{{prov.origin_node_ids = ["{_NODE}"], prov.aten = "{target}"}}'
    return f'''"builtin.module"() ({{
  "func.func"() <{{sym_name = "forward", function_type = ({bf16}) -> {bf16}}}> ({{
  ^bb0(%0: {bf16}):
    %1 = "arith.constant"() <{{value = {literal!r} : f32}}> {prov} : () -> f32
    %2 = "tensor.splat"(%1) {prov} : (f32) -> {f32}
    %3 = "tensor.empty"() {prov} : () -> {f32}
    %4 = "linalg.generic"(%0, %2, %3) <{{indexing_maps = [{identity}, {identity}, {identity}], iterator_types = [{parallel}], operandSegmentSizes = array<i32: 2, 1>}}> ({{
    ^bb1(%5: bf16, %6: f32, %7: f32):
      %8 = "arith.extf"(%5) : (bf16) -> f32
      %9 = "{arithmetic}"(%8, %6) <{{fastmath = #arith.fastmath<none>}}> : (f32, f32) -> f32
      "linalg.yield"(%9) : (f32) -> ()
    }}) {prov} : ({bf16}, {f32}, {f32}) -> {f32}
    %10 = "tensor.empty"() {prov} : () -> {bf16}
    %11 = "linalg.generic"(%4, %10) <{{indexing_maps = [{identity}, {identity}], iterator_types = [{parallel}], operandSegmentSizes = array<i32: 1, 1>}}> ({{
    ^bb2(%12: f32, %13: bf16):
      %14 = "arith.truncf"(%12) : (f32) -> bf16
      "linalg.yield"(%14) : (bf16) -> ()
    }}) {prov} : ({f32}, {bf16}) -> {bf16}
    "func.return"(%11) : ({bf16}) -> ()
  }}) : () -> ()
}}) : () -> ()
'''


def _rows(widen_input="arg0"):
    # The compiler's three regions: device widen, host F32 arithmetic, device narrow (op ordinals of the module above).
    return [{"id": "widen", "placement": "accelerator", "implementation": "source_exact_float_device",
             "kind": "exact_float_device", "inputs": [widen_input], "buffers": ["op3r0_s0"]},
            {"id": "operate", "placement": "host", "implementation": "source_float_arithmetic",
             "kind": "float_arithmetic", "inputs": ["op1r0", "op3r0_s0"], "buffers": ["op3r0_s1"]},
            {"id": "narrow", "placement": "accelerator", "implementation": "source_exact_float_device",
             "kind": "exact_float_device", "inputs": ["op3r0_s1"], "buffers": ["op5r0_s0"]}]


def _decisions(text, rows):
    decisions = object.__new__(SelectedDecisions)
    decisions._by_node, decisions._bindings = {_NODE: rows}, _source_provenance(text, bindings=True)
    decisions._softmax, decisions.record = {}, {"applied": {}}
    return decisions


def _graph(target, scalar, shape=(2, 3)):
    x = {"id": "x", "op": "placeholder", "target": "x", "input_kind": "USER_INPUT",
         "results": [{"id": "x:v0", "dtype": "bfloat16", "shape": list(shape), "device": "cpu"}]}
    node = {"id": _NODE, "op": "call_function", "target": target, "args": [{"node_id": "x", "value_id": "x:v0"}, scalar],
            "kwargs": {}, "results": [{"id": _NODE + ":v0", "dtype": "bfloat16", "shape": list(shape), "device": "cpu"}]}
    out = {"id": "out", "op": "output", "target": "output", "args": [{"node_id": _NODE, "value_id": _NODE + ":v0"}],
           "kwargs": {}, "results": []}
    return [x, node, out]


def _bits(tensor):
    return tensor.view(torch.int16).numpy().view(np.uint16)


class PromotedScalarTest(TestCase):
    def run_case(self, target, arithmetic, scalar, x, rows=None, literal=None):
        text = _mlir(target, arithmetic, float(np.float32(scalar)) if literal is None else literal, tuple(x.shape))
        decisions = _decisions(text, _rows() if rows is None else rows)
        return evaluate(_graph(target, scalar, tuple(x.shape)), {}, {"x": _bits(x)}, _RULES, decisions=decisions)

    def test_bf16_times_a_python_scalar_matches_torch(self):
        generator = torch.Generator().manual_seed(11)
        for scalar in (960 ** 0.5, 1 / 3, -2.5e-3):
            x = (torch.randn(4096, generator=generator) * 4).to(torch.bfloat16)
            with self.subTest(scalar=scalar):
                # The double rounds once to F32, then the product rounds once to BF16, as torch computes it.
                result = self.run_case("aten.mul.Tensor", "arith.mulf", scalar, x)
                np.testing.assert_array_equal(result.outputs.raw, _bits(x * scalar))
                self.assertTrue(np.all(result.outputs.raw != 0))
        # Rounding the scalar to BF16 first gives other bits, so the test pins the F32 scalar.
        x = (torch.randn(4096, generator=generator) * 4).to(torch.bfloat16)
        self.assertFalse(np.array_equal(_bits(x * (960 ** 0.5)), _bits(x * torch.tensor(960 ** 0.5).to(torch.bfloat16))))

    def test_every_finite_bf16_pattern_matches_torch_bits_or_refuses(self):
        patterns = np.arange(1 << 16, dtype=np.uint32).astype(np.uint16)
        finite = patterns[(patterns & 0x7F80) != 0x7F80]
        self.assertEqual(finite.size, 65280)  # both zeros, subnormals, and the largest finite values included
        values = (finite.astype(np.uint32) << 16).view(np.float32)
        scalars = (0.0, -0.0, 1e-30, 1.5, 3, -(1 << 24), 1 << 24,
                   1 + 2 ** -24, 1 + 3 * 2 ** -24,  # F32 rounding midpoints: ties to even
                   1e-46, -1e-46,  # underflow to zeros of either sign in F32
                   3.4028234663852886e38)  # the largest finite F32
        for scalar in scalars:
            with np.errstate(over="ignore"):
                product = values * np.float32(scalar)
            admitted = finite[np.isfinite(product)]
            x = torch.from_numpy(admitted.view(np.int16).copy()).view(torch.bfloat16)
            with self.subTest(scalar=scalar, admitted=admitted.size):
                result = self.run_case("aten.mul.Tensor", "arith.mulf", scalar, x)
                np.testing.assert_array_equal(result.outputs.raw, _bits(x * scalar))
            if admitted.size < finite.size:
                # A product past the F32 range is not finite, and the narrowing refuses it explicitly.
                overflow = torch.from_numpy(finite[~np.isfinite(product)][:4].view(np.int16).copy()).view(torch.bfloat16)
                with self.subTest(scalar=scalar, overflow=True), self.assertRaisesRegex(EvaluationError, "finite"):
                    self.run_case("aten.mul.Tensor", "arith.mulf", scalar, overflow)
        # BF16 ties: 1.5 times 1 + 2^-7 is exactly halfway between two BF16 values, and rounds to even, as in torch.
        tie = torch.tensor([1 + 2 ** -7, -(1 + 3 * 2 ** -7)], dtype=torch.bfloat16)
        self.assertTrue(np.all((((tie.float() * 1.5).numpy().view(np.uint32)) & 0xFFFF) == 0x8000))
        np.testing.assert_array_equal(self.run_case("aten.mul.Tensor", "arith.mulf", 1.5, tie).outputs.raw, _bits(tie * 1.5))

    def test_non_finite_values_and_unqualified_scalars_refuse(self):
        x = torch.tensor([1.0, -2.0], dtype=torch.bfloat16)
        special = torch.from_numpy(np.array([0x7F80, 0xFF80, 0x7FC0, 0xFFC1], np.uint16).view(np.int16)).view(torch.bfloat16)
        for raw in special:  # infinities and NaNs of both signs
            with self.subTest(raw=int(raw.view(torch.int16))), self.assertRaisesRegex(EvaluationError, "finite"):
                self.run_case("aten.mul.Tensor", "arith.mulf", 3, torch.stack([raw, raw]))
        for scalar, message in ((float("inf"), "finite in F32"), (float("nan"), "finite in F32"),
                                (1e39, "finite in F32"),  # overflows when rounded to F32
                                ((1 << 24) + 1, "beyond 2\\^24"), (-(1 << 24) - 1, "beyond 2\\^24"),
                                (1 << 1000, "beyond 2\\^24"), (True, "Python int or float")):
            with self.subTest(scalar=str(scalar)[:12]), self.assertRaisesRegex(EvaluationError, message):
                self.run_case("aten.mul.Tensor", "arith.mulf", scalar, x, literal=1.0)
        # The scalar and the literal are compared by F32 bits, so a signed zero must match.
        with self.assertRaisesRegex(EvaluationError, "differs from its F32 literal"):
            self.run_case("aten.mul.Tensor", "arith.mulf", 0.0, x, literal=-0.0)

    def test_a_scalar_add_is_not_this_rule(self):
        # torch rounds an added scalar to BF16 first, which the F32-splat lowering does not express.
        x = (torch.randn(4096, generator=torch.Generator().manual_seed(12)) * 4).to(torch.bfloat16)
        widened = (x.float() + float(np.float32(1 / 3))).to(torch.bfloat16)
        self.assertFalse(np.array_equal(_bits(widened), _bits(x + 1 / 3)))
        with self.assertRaisesRegex(EvaluationError, "scalar add is not qualified"):
            self.run_case("aten.add.Tensor", "arith.addf", 1 / 3, x)

    def test_a_scalar_the_lowering_did_not_splat_or_changed_stages_refuse(self):
        x = torch.ones(2, 3, dtype=torch.bfloat16)
        with self.assertRaisesRegex(EvaluationError, "differs from its F32 literal"):
            self.run_case("aten.mul.Tensor", "arith.mulf", 960 ** 0.5, x, literal=30.0)
        two_stage = _rows()[:2]
        with self.assertRaisesRegex(EvaluationError, "exactly three single-result stages"):
            self.run_case("aten.mul.Tensor", "arith.mulf", 960 ** 0.5, x, rows=two_stage)
        with self.assertRaisesRegex(EvaluationError, "edges differ from source operands"):
            self.run_case("aten.mul.Tensor", "arith.mulf", 960 ** 0.5, x, rows=_rows(widen_input="op1r0"))


if __name__ == "__main__":
    main()
