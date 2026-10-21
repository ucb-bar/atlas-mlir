"""Signed-zero extensions remain separate from normal-only arithmetic."""

import unittest
from dataclasses import replace

from atlas_mlir_native.binding import BF16_ZERO_BINARY_ATTRS, AtlasNativeBinding
from atlas_mlir_native.reference import evaluate_bf16_binary_zero, evaluate_bf16_mul

from merlin.semantic_compiler.model import (
    SemanticNode,
    TensorType,
    pointwise_identity_maps,
)
from merlin.semantic_compiler.reference import TensorValue


class Bf16ZeroArithmeticTest(unittest.TestCase):
    def test_signed_zero_and_normal_identities(self):
        normals = [
            sign | (exponent << 7) | mantissa
            for sign in (0, 0x8000)
            for exponent in (1, 2, 63, 126, 127, 128, 190, 253, 254)
            for mantissa in (0, 1, 63, 64, 126, 127)
        ]
        pairs = [
            (left, right)
            for zero in (0, 0x8000)
            for word in [0, 0x8000, *normals]
            for left, right in ((zero, word), (word, zero))
        ]
        type_ = TensorType((len(pairs),), "bf16", "raw-bit-pattern")
        operands = tuple(TensorValue(type_, tuple(pair[i] for pair in pairs)) for i in range(2))
        for op, attrs in BF16_ZERO_BINARY_ATTRS.items():
            node = SemanticNode("result", op, ("a", "b"), type_, attrs=attrs, index_maps=pointwise_identity_maps(1, 2))
            result = evaluate_bf16_binary_zero(node, operands)
            expected = []
            for left, right in pairs:
                lz, rz = left in (0, 0x8000), right in (0, 0x8000)
                if op.startswith("mul"):
                    value = (left ^ right) & 0x8000
                elif op.startswith("add"):
                    value = left & right & 0x8000 if lz and rz else right if lz else left
                else:
                    value = left & ~right & 0x8000 if lz and rz else right ^ 0x8000 if lz else left
                expected.append(value)
            self.assertEqual(result.elements, tuple(expected))

    def test_normal_only_contract_and_invalid_operands_still_refuse(self):
        type_ = TensorType((1,), "bf16", "raw-bit-pattern")
        node = SemanticNode(
            "result",
            "mul_bf16_zero",
            ("a", "b"),
            type_,
            attrs=BF16_ZERO_BINARY_ATTRS["mul_bf16_zero"],
            index_maps=pointwise_identity_maps(1, 2),
        )
        zero = TensorValue(type_, (0,))
        one = TensorValue(type_, (0x3F80,))
        legacy = replace(
            node,
            op="mul_bf16",
            attrs=(
                ("input_domain", "finite-normal-bf16-product-normal-v1"),
                ("numeric_semantics", "bf16-product-rne-v1"),
            ),
        )
        with self.assertRaisesRegex(ValueError, "finite normal"):
            evaluate_bf16_mul(legacy, (zero, one))
        for code in (1, 0x8001, 0x7F80, 0xFF80, 0x7FC0):
            with self.subTest(code=code), self.assertRaisesRegex(ValueError, "finite normal"):
                evaluate_bf16_binary_zero(node, (zero, TensorValue(type_, (code,))))
        with self.assertRaisesRegex(ValueError, "semantic contract"):
            evaluate_bf16_binary_zero(replace(node, attrs=legacy.attrs), (zero, one))
        with self.assertRaisesRegex(ValueError, "admitted normal range"):
            evaluate_bf16_binary_zero(node, (TensorValue(type_, (0x0080,)), TensorValue(type_, (0x3F00,))))

    def test_exact_cancellation_returns_positive_zero(self):
        type_ = TensorType((2,), "bf16", "raw-bit-pattern")
        left = TensorValue(type_, (0x3F80, 0xBF80))
        for op, right in (("add_bf16_zero", (0xBF80, 0x3F80)), ("sub_bf16_zero", (0x3F80, 0xBF80))):
            node = SemanticNode(
                "result",
                op,
                ("a", "b"),
                type_,
                attrs=BF16_ZERO_BINARY_ATTRS[op],
                index_maps=pointwise_identity_maps(1, 2),
            )
            self.assertEqual(evaluate_bf16_binary_zero(node, (left, TensorValue(type_, right))).elements, (0, 0))

    def test_descriptors_generate_distinct_semantic_variants(self):
        descriptors = {item.name: item for item in AtlasNativeBinding.profile().descriptors}
        for op, attrs in BF16_ZERO_BINARY_ATTRS.items():
            new = descriptors[f"atlas.vpu_{op}_tile"]
            old = descriptors[f"atlas.vpu_{op.removesuffix('_zero')}_tile"]
            self.assertEqual(new.required_attrs, attrs)
            self.assertEqual(new.computation, op)
            self.assertEqual(new.input_storages, old.input_storages)
            self.assertNotEqual(new.required_attrs, old.required_attrs)
