"""Explicit GELU admission, finite-domain checks and selected loop bodies."""

import dataclasses
import itertools
import json
import os
import tempfile
import unittest
from pathlib import Path

from atlas_mlir_native.gelu import SOURCE_ATTRS, input_spec, prepare_plan, prepare_source


def source_request(shape=(32, 32)):
    from atlas_mlir_native.binding import TARGET_IDENTITY

    from merlin.semantic_compiler.model import KernelRequest, SemanticNode, TensorType, pointwise_identity_maps

    tensor = TensorType(shape, "f32", "raw-bit-pattern")
    return KernelRequest((
        SemanticNode("x", "input", (), tensor, effect="input"),
        SemanticNode("y", "source_f32_composite_unqualified", ("x",), tensor, SOURCE_ATTRS,
                     index_maps=pointwise_identity_maps(len(shape), 1)),
    ), ("y",), ("external",), (("x", "external"),), TARGET_IDENTITY, "diagnostic")


class GeluPrecisionTest(unittest.TestCase):
    def test_source_attribute_order_does_not_change_admission(self):
        from atlas_mlir_native.gelu import is_source_gelu

        source = source_request()
        expected = prepare_source(source).request.digest()
        for attributes in itertools.permutations(SOURCE_ATTRS):
            with self.subTest(attributes=attributes):
                node = dataclasses.replace(source.nodes[-1], attrs=attributes)
                self.assertTrue(is_source_gelu(node))
                fresh = dataclasses.replace(source, nodes=(source.nodes[0], node))
                self.assertEqual(prepare_source(fresh).request.digest(), expected)

    def test_application_needs_a_separately_selected_gelu_policy(self):
        from atlas_mlir_native.application import AtlasApplicationAdapter
        from atlas_mlir_native.application_precision import SCALED_CONTRACTION_POLICY
        from atlas_mlir_native.binding import TARGET_IDENTITY
        from atlas_mlir_native.gelu import GELU_POLICY
        from atlas_mlir_native.pointwise import POINTWISE_POLICY
        from atlas_mlir_native.pointwise_math import MATH_POLICY
        from atlas_mlir_native.reduction import REDUCTION_POLICY
        from atlas_mlir_native.row_softmax import GUARDED_SOFTMAX_POLICY, MASKED_SOFTMAX_POLICY, SOFTMAX_POLICY
        from atlas_mlir_native.sigmoid import SIGMOID_POLICY

        from merlin.semantic_compiler.target_binding import NativeCompilationError

        policy = {
            "schema": "atlas.diagnostic_application_precision.v16", "source_sha256": "0" * 64,
            "target_identity": TARGET_IDENTITY,
            "contraction": {"policy": SCALED_CONTRACTION_POLICY, "operand_scale_exponents": [4, 4]},
            "host_support": ["transpose", "reshape", "copy"], "epilogues": ["relu"],
            "pointwise": POINTWISE_POLICY, "pointwise_math": MATH_POLICY, "reduction": REDUCTION_POLICY,
            "sigmoid": SIGMOID_POLICY, "row_softmax": SOFTMAX_POLICY, "masked_softmax": MASKED_SOFTMAX_POLICY,
            "guarded_masked_softmax": GUARDED_SOFTMAX_POLICY, "gelu": GELU_POLICY,
            "host_predicates": ["mask"], "host_integer_ops": ["i64_add"],
        }
        adapter = AtlasApplicationAdapter(policy, original_source_sha256=policy["source_sha256"])
        self.assertEqual(len(adapter.plan(source_request()).selected), 1)
        old = {key: value for key, value in policy.items() if key != "gelu"}
        old["schema"] = "atlas.diagnostic_application_precision.v15"
        with self.assertRaisesRegex(NativeCompilationError, "unrecognized source composite"):
            AtlasApplicationAdapter(old, original_source_sha256=old["source_sha256"]).plan(source_request())
        with self.assertRaisesRegex(NativeCompilationError, "GELU policy"):
            AtlasApplicationAdapter({**policy, "gelu": "changed"}, original_source_sha256=policy["source_sha256"])

    def test_coefficients_and_source_shape_are_checked(self):
        import numpy as np

        for shape in ((3, 7), (33, 35), (2, 3, 35)):
            with self.subTest(shape=shape):
                precision = prepare_source(source_request(shape))
                plan = prepare_plan(precision.request)
                body = plan.body_request()
                self.assertEqual(len([n for n in body.nodes if n.effect == "pure"]), 8)
                self.assertEqual(len(body.constants), 4)
                self.assertEqual(precision.record()["coefficient_bf16_bits"],
                                 {"half": 0x3F00, "one": 0x3F80, "cubic": 0x3D37, "root": 0x3F4C})
        raw = np.array([0x3F000000, 0x3F800000, 0x3D372713, 0x3F4C422A], dtype="<u4").view("<f4")
        self.assertEqual(list(input_spec().encode(raw)), [0x3F00, 0x3F80, 0x3D37, 0x3F4C])
        source = source_request()
        changed = dataclasses.replace(source.nodes[-1], attrs=tuple(
            (k, v.replace("3d372713", "3d372714")) for k, v in SOURCE_ATTRS
        ))
        with self.assertRaisesRegex(ValueError, "exact closed"):
            prepare_source(dataclasses.replace(source, nodes=(source.nodes[0], changed)))
        for values in ([2.0**-33], [2.0**33], [float("inf")], [float("nan")]):
            with self.subTest(values=values), self.assertRaises(ValueError):
                input_spec().encode(np.array(values, dtype=np.float32))

    def test_all_admitted_codes_preserve_every_binary_result_domain(self):
        from atlas_mlir_native.reference import evaluate_gelu

        from merlin.semantic_compiler.model import TensorType
        from merlin.semantic_compiler.reference import TensorValue

        codes = (0, 0x8000, *(sign | code for sign in (0, 0x8000) for code in range(0x2F80, 0x4F81)))
        precision = prepare_source(source_request((1, len(codes))))
        macro = precision.request.nodes[-1]
        values = TensorValue(TensorType((1, len(codes)), "bf16", "raw-bit-pattern"), codes)
        result = evaluate_gelu(macro, (values,))
        self.assertEqual(len(result.elements), 16388)
        self.assertEqual(result.elements[:2], (0, 0x8000))
        self.assertTrue(all((code >> 7 & 255) != 255 for code in result.elements))

    @unittest.skipUnless(os.environ.get("MERLIN_NATIVE_ROOT") and os.environ.get("ATLAS_OOT_BIN_DIR"),
                         "select Merlin and Atlas tools")
    def test_selected_gelu_is_one_compact_tiled_program(self):
        from atlas_mlir_native.binding import AtlasNativeBinding
        from atlas_mlir_native.pointwise import plan_abi
        from test_looped_mnk import ROOT, build_snapshot

        from merlin.semantic_compiler.search import SearchLimits

        with tempfile.TemporaryDirectory(dir=os.environ.get("TMPDIR")) as directory:
            root = Path(directory)
            snapshot = build_snapshot(root)
            request = prepare_source(source_request((33, 35))).request
            abi = plan_abi(request)
            folder = root / "compiled"
            manifest = AtlasNativeBinding.compile(
                snapshot, request, fixed_inputs=abi["fixed_inputs"], fixed_outputs=tuple(abi["fixed_outputs"]),
                target_source=ROOT, destination=folder, limits=SearchLimits(),
            )
            self.assertEqual(manifest["engine"], "merlin_native")
            steps = manifest["selected_body_descriptor"]
            self.assertEqual(len(steps), 8)
            program = json.loads((folder / "physical-program.json").read_text())
            self.assertEqual([row["fields"]["kind"] for row in program["instructions"]
                              if row["operation"] == "atlas.vpu_unary"], ["cube", "tanh"])
            self.assertLess((folder / "program.bin").stat().st_size, 8192)
            self.assertIn("^loop", (folder / "atlas-virtual.mlir").read_text())
            from atlas_mlir_native.pointwise_math import selected_steps

            selected = json.loads((folder / "body/native-selection.json").read_text())
            checked = selected_steps(prepare_plan(request), selected)
            self.assertEqual(checked, steps)
            arithmetic = next(v for v in selected["values"] if v["descriptor"] == steps[0])
            selected["values"].append(arithmetic)
            with self.assertRaisesRegex(ValueError, "selected decomposition"):
                selected_steps(prepare_plan(request), selected)
