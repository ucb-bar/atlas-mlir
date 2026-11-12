"""Explicit source-bound attention choices preserve private region accounting."""

from dataclasses import replace
import unittest

from atlas_mlir_native.application import AtlasApplicationAdapter
from atlas_mlir_native.application_precision import SCALED_CONTRACTION_POLICY
from atlas_mlir_native.binding import TARGET_IDENTITY
from atlas_mlir_native.pointwise import POINTWISE_POLICY
from atlas_mlir_native.quantized_softmax_value import POLICY
from atlas_mlir_native.rne_add import RNE_ADD_POLICY
from atlas_mlir_native.row_softmax import SOFTMAX_POLICY
from merlin.compile.semantic_regions import extract_semantic_region
from merlin.semantic_compiler.model import KernelRequest, SemanticNode, TensorType, pointwise_identity_maps
from merlin.semantic_compiler.softmax_value_source import recognize_softmax_value
from merlin.semantic_compiler.target_binding import NativeCompilationError
from merlin.tests.ir.test_softmax_value_source import graph


def base():
    return dict(schema="atlas.diagnostic_application_precision.v4", source_sha256="a" * 64,
                target_identity=TARGET_IDENTITY, contraction=dict(policy=SCALED_CONTRACTION_POLICY,
                operand_scale_exponents=[0, 0]), row_softmax=SOFTMAX_POLICY,
                host_support=["transpose", "reshape", "copy"], epilogues=["relu"], pointwise=POINTWISE_POLICY)


def policy(**changes):
    return dict(schema="atlas.diagnostic_application_precision.v23", source_sha256="a" * 64,
                base_policy=base(), rne_add=RNE_ADD_POLICY, softmax_value_schedules=[],
                mxu1_contraction_schedules=[]) | changes


def source():
    original = graph((2, 33), views=True)
    type_ = original.node("scores").type
    mask = SemanticNode("mask", "input", (), type_, effect="input")
    masked = SemanticNode("masked", "f32_add", ("scores", "mask"), type_,
                         (("numeric_semantics", "ieee754-f32-add-rne-v1"),),
                         index_maps=pointwise_identity_maps(2, 2))
    nodes = []
    for node in original.nodes:
        nodes.append(replace(node, inputs=tuple("masked" if key == "scores" else key for key in node.inputs)))
        if node.id == "scores":
            nodes.extend((mask, masked))
    return replace(original, nodes=tuple(nodes), target_identity=TARGET_IDENTITY,
                   lowering_policy="diagnostic", source_identity="a" * 64,
                   input_storages=(*original.input_storages, ("mask", "external")))


class AttentionApplicationPolicyTest(unittest.TestCase):
    def test_private_softmax_and_pv_are_one_admitted_source_region(self):
        graph_ = source()
        found = recognize_softmax_value(graph_, "context", masked=True)
        cut = extract_semantic_region(graph_, found.computations)
        selected = dict(source_output="context", source_request_digest=cut.digest(),
                        provenance_sha256="b" * 64, policy=POLICY, value_scale_exponent=-10)
        plan = AtlasApplicationAdapter(policy(softmax_value_schedules=[selected]),
                                      original_source_sha256="a" * 64).plan(graph_)
        self.assertEqual(len(plan.selected), 1)
        self.assertEqual(dict(plan.selected[0].metadata)["kind"], "softmax_value")
        changed = replace(graph_, outputs=("context", "result"), output_storages=("external", "external"))
        with self.assertRaisesRegex(NativeCompilationError, "exact private source regions"):
            AtlasApplicationAdapter(policy(softmax_value_schedules=[selected]), original_source_sha256="a" * 64).plan(changed)
        changed_policy = policy(softmax_value_schedules=[selected | {"source_request_digest": "0" * 64}])
        with self.assertRaisesRegex(NativeCompilationError, "source digest differs"):
            AtlasApplicationAdapter(changed_policy, original_source_sha256="a" * 64).plan(graph_)

    def test_rne_add_is_separately_selected(self):
        type_ = TensorType((2, 3), "f32", "raw-bit-pattern")
        a, b = (SemanticNode(key, "input", (), type_, effect="input") for key in ("a", "b"))
        add = SemanticNode("sum", "f32_add", ("a", "b"), type_,
                          (("numeric_semantics", "ieee754-f32-add-rne-v1"),),
                          index_maps=pointwise_identity_maps(2, 2))
        graph_ = KernelRequest((a, b, add), ("sum",), ("external",), (("a", "external"), ("b", "external")),
                               TARGET_IDENTITY, "diagnostic", "a" * 64)
        selected = AtlasApplicationAdapter(policy(), original_source_sha256="a" * 64).plan(graph_)
        self.assertEqual(dict(selected.selected[0].metadata)["kind"], "rne_add")
        old = AtlasApplicationAdapter(base(), original_source_sha256="a" * 64).plan(graph_)
        self.assertEqual(dict(old.selected[0].metadata)["kind"], "pointwise")

    def test_unknown_precision_and_duplicate_schedules_refuse(self):
        for selected in (policy(rne_add="unknown"), policy(source_sha256="c" * 64), policy(extra=True),
                         policy(softmax_value_schedules=[{"source_output": "x"}] * 2)):
            with self.subTest(policy=selected), self.assertRaises(NativeCompilationError):
                AtlasApplicationAdapter(selected, original_source_sha256="a" * 64)


if __name__ == "__main__":
    unittest.main()
