"""Sigmoid source admission, initialized constants and bounded lane agreement."""

import os
import unittest
from dataclasses import replace

from atlas_mlir_native.binding import TARGET_IDENTITY
from atlas_mlir_native.sigmoid import (
    SIGMOID_POLICY,
    SOURCE_ATTRS,
    input_spec,
    prepare_plan,
    prepare_source,
)

from merlin.semantic_compiler.model import (
    KernelRequest,
    SemanticNode,
    TensorType,
    pointwise_identity_maps,
)


def source_request(shape=(2, 3, 35)):
    type_ = TensorType(shape, "f32", "raw-bit-pattern")
    arg = SemanticNode("x", "input", (), type_, effect="input")
    result = SemanticNode(
        "out",
        "source_f32_composite_unqualified",
        (arg.id,),
        type_,
        SOURCE_ATTRS,
        index_maps=pointwise_identity_maps(len(shape), 1),
    )
    return KernelRequest(
        (arg, result),
        (result.id,),
        ("external",),
        ((arg.id, "external"),),
        TARGET_IDENTITY,
        "diagnostic",
    )


def silu_source_request(shape=(2, 3, 35)):
    source = source_request(shape)
    operand, sigmoid = source.nodes
    sigmoid = replace(sigmoid, id="sigmoid")
    product = SemanticNode("out", "f32_mul", (operand.id, sigmoid.id), operand.type,
                           (("numeric_semantics", "ieee754-f32-mul-rne-v1"),),
                           index_maps=pointwise_identity_maps(len(shape), 2))
    return replace(source, nodes=(operand, sigmoid, product))


class SigmoidTest(unittest.TestCase):
    def test_silu_keeps_the_sigmoid_steps_and_one_final_product(self):
        from atlas_mlir_native.pointwise import prepare_plan as loop_plan
        from atlas_mlir_native.pointwise import prepare_source as pointwise_source

        source = silu_source_request((1, 50, 720))
        precision = pointwise_source(source, allow_math=True)
        self.assertEqual(precision.record()["inputs"], ["x"])
        plan = loop_plan(precision.request)
        body = plan.body_request()
        self.assertEqual([n.op for n in body.nodes if n.effect == "pure"],
                         ["sub_bf16_zero", "exp_bf16", "add_bf16_zero", "recip_bf16_bounded", "mul_bf16_zero"])
        self.assertEqual(body.nodes[-1].inputs, ("arg0", "sigmoid"))
        self.assertEqual(body.nodes[-2].inputs, ("denominator",))
        self.assertEqual(plan.input_packed_bytes, (94208,))
        self.assertEqual(plan.packed_bytes, 94208)
        for changed in (
            replace(source, outputs=("out", "sigmoid"), output_storages=("external",) * 2),
            replace(source, nodes=(*source.nodes[:-1], replace(source.nodes[-1], attrs=()))),
        ):
            with self.assertRaises(ValueError):
                pointwise_source(changed, allow_math=True)

    def test_source_policy_domain_and_generated_constants(self):
        import numpy as np
        from atlas_mlir_native.pointwise import prepare_source as pointwise_source

        source = source_request()
        with self.assertRaises(ValueError):
            pointwise_source(source)
        arg, output = source.nodes
        reordered = replace(source, nodes=(arg, replace(output, attrs=tuple(reversed(output.attrs)))))
        self.assertEqual(
            prepare_source(reordered).request.digest(),
            prepare_source(source).request.digest(),
        )
        precision = pointwise_source(source, allow_math=True)
        self.assertEqual(precision.record()["policy"], SIGMOID_POLICY)
        spec = input_spec()
        spec.encode(np.array([0.0, -0.0, -32.0, 32.0, 2.0**-63, -(2.0**-63)], dtype="<f4"))
        for value in (np.inf, -np.inf, np.nan, 33.0, -33.0, 2.0**-64, -(2.0**-64)):
            with self.subTest(value=value), self.assertRaises(ValueError):
                spec.encode(np.array([1.0, value], dtype="<f4"))
        plan = prepare_plan(precision.request)
        self.assertEqual(plan.tile_counts, (1, 2))
        self.assertEqual(plan.input_pad(0), b"\0\0")
        body = plan.body_request()
        self.assertEqual(
            [n.op for n in body.nodes if n.effect == "pure"],
            ["sub_bf16_zero", "exp_bf16", "add_bf16_zero", "recip_bf16_bounded"],
        )
        self.assertEqual(
            [(c.node_id, bytes.fromhex(c.data_hex)) for c in body.constants],
            [("zero", b"\0\0" * 1024), ("one", b"\x80\x3f" * 1024)],
        )
        self.assertEqual(body.input_storages, (("arg0", "external"),))
        self.assertEqual(KernelRequest.from_record(body.record()).record(), body.record())

    def test_changed_source_attributes_shape_effect_or_mode_are_rejected(self):
        source = source_request()
        arg, result = source.nodes
        for bad in (
            replace(result, attrs=()),
            replace(result, index_maps=()),
            replace(
                result,
                attrs=tuple((k, "0x40000000" if k == "literal_bits" else v) for k, v in result.attrs),
            ),
            replace(result, type=replace(result.type, shape=(2, 3, 34))),
        ):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                prepare_source(replace(source, nodes=(arg, bad)))
        with self.assertRaises(ValueError):
            prepare_source(replace(source, lowering_policy="strict-native"))

    def test_application_requires_new_policy_and_old_policies_remain_unchanged(self):
        from atlas_mlir_native.application import AtlasApplicationAdapter
        from atlas_mlir_native.pointwise_math import MATH_POLICY
        from atlas_mlir_native.reduction import REDUCTION_POLICY
        from test_application_c_package import ApplicationLayoutPolicyTest

        from merlin.semantic_compiler.target_binding import NativeCompilationError

        old = {
            **ApplicationLayoutPolicyTest().policy(),
            "schema": "atlas.diagnostic_application_precision.v8",
            "reduction": REDUCTION_POLICY,
            "pointwise_math": MATH_POLICY,
        }
        with self.assertRaises(NativeCompilationError):
            AtlasApplicationAdapter(old, original_source_sha256=old["source_sha256"]).plan(source_request())
        policy = {
            **old,
            "schema": "atlas.diagnostic_application_precision.v9",
            "sigmoid": SIGMOID_POLICY,
        }
        adapter = AtlasApplicationAdapter(policy, original_source_sha256=policy["source_sha256"])
        regions = adapter.plan(source_request()).selected
        self.assertEqual(len(regions), 1)
        self.assertEqual(regions[0].placement, "accelerator")
        self.assertEqual(dict(regions[0].metadata)["kind"], "pointwise_math")
        fused = adapter.plan(silu_source_request()).selected
        self.assertEqual(len(fused), 1)
        self.assertEqual(fused[0].nodes, (0, 1))
        # An independently requested sigmoid must remain a separate result.
        escaped = replace(silu_source_request(), outputs=("out", "sigmoid"),
                          output_storages=("external",) * 2)
        self.assertEqual(len(adapter.plan(escaped).selected), 2)
        for bad in (
            {k: v for k, v in policy.items() if k != "sigmoid"},
            {**policy, "sigmoid": "unknown"},
            {**old, "sigmoid": SIGMOID_POLICY},
        ):
            with self.subTest(bad=bad), self.assertRaises(NativeCompilationError):
                AtlasApplicationAdapter(bad, original_source_sha256=old["source_sha256"])

    @unittest.skipUnless(os.environ.get("ATLAS_RUN_NPU_MODEL") == "1", "select npu_model")
    def test_every_admitted_bf16_encoding_matches_selected_model_primitives(self):
        import torch
        from atlas_mlir_native.reference import evaluate_sigmoid
        from npu_model.configs.numerics import RtlNumerics

        from merlin.semantic_compiler.reference import TensorValue

        codes = [0, 0x8000, *range(0x2000, 0x4201), *range(0xA000, 0xC201)]
        values = torch.tensor(codes, dtype=torch.int32).to(torch.int16).view(torch.bfloat16)
        negative = RtlNumerics.sub(torch.zeros_like(values), values)
        exp = RtlNumerics.unary("exp", negative)
        denominator = RtlNumerics.add(torch.ones_like(values), exp)
        expected = RtlNumerics.unary("rcp", denominator).view(torch.int16).to(torch.int32).bitwise_and(65535).tolist()
        request = prepare_source(source_request((len(codes),))).request
        actual = evaluate_sigmoid(request.nodes[-1], (TensorValue(request.nodes[0].type, tuple(codes)),))
        self.assertEqual(list(actual.elements), expected)
        self.assertEqual(len(codes), 17412)
        self.assertEqual(actual.elements[:2], (0x3F00, 0x3F00))

        # Fusion removes an exact decode/re-encode and one round trip through
        # staging. Check closure of the intervening input-domain guard over
        # every admitted code, then compare the final selected product.
        import numpy as np
        from atlas_mlir_native.binding import BF16_ZERO_BINARY_ATTRS
        from atlas_mlir_native.pointwise import input_spec as multiply_input_spec
        from atlas_mlir_native.reference import evaluate_bf16_binary_zero

        sigmoid_values = (np.array(expected, dtype=np.uint32) << 16).view(np.float32)
        reencoded = multiply_input_spec().encode(sigmoid_values).astype("<u2")
        self.assertEqual(reencoded.tolist(), expected)
        product = SemanticNode("product", "mul_bf16_zero", ("x", "sigmoid"), request.nodes[-1].type,
                               BF16_ZERO_BINARY_ATTRS["mul_bf16_zero"],
                               index_maps=pointwise_identity_maps(1, 2))
        checked = evaluate_bf16_binary_zero(product, (TensorValue(product.type, tuple(codes)), actual))
        direct = RtlNumerics.mul(values, RtlNumerics.unary("rcp", denominator))
        expected_product = direct.view(torch.int16).to(torch.int32).bitwise_and(65535).tolist()
        self.assertEqual(list(checked.elements), expected_product)
        self.assertEqual(checked.elements[:2], (0, 0x8000))
