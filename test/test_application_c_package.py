"""Check a staged application's public CLI, source replay and RV64 linkage."""

import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

from test_looped_mnk import ROOT


class ApplicationLayoutPolicyTest(unittest.TestCase):
    @unittest.skipUnless(
        os.environ.get("MERLIN_NATIVE_ROOT") and os.environ.get("ATLAS_OOT_BIN_DIR"),
        "select Merlin and Atlas build tools",
    )
    def test_folded_accumulator_seed_compiles_from_linalg_with_a_replay_receipt(self):
        from atlas_mlir_native.application import AtlasApplicationAdapter
        from test_looped_mnk import build_snapshot

        from merlin.compile.semantic_application import compile_semantic_application, verify_semantic_application
        from merlin.semantic_compiler.constant_folding import fold_source_constants
        from merlin.semantic_compiler.linalg_bridge import translate_linalg_text
        from merlin.semantic_compiler.search import SearchLimits

        policy = self.policy()
        original = translate_linalg_text(
            (ROOT / "test/examples/constant_matmul_seed.mlir").read_text(),
            entry="forward",
            target_identity=policy["target_identity"],
            lowering_policy="diagnostic",
        ).request
        source, receipt = fold_source_constants(original, eliminate_dead=True)
        self.assertEqual(len(receipt["folds"]), 2)
        self.assertEqual(len(receipt["dead_computations"]), 1)
        policy["source_sha256"] = source.source_identity
        adapter = AtlasApplicationAdapter(policy, original_source_sha256=source.source_identity)
        with tempfile.TemporaryDirectory(dir=os.environ.get("TMPDIR")) as directory:
            root = Path(directory)
            snapshot = build_snapshot(root)
            artifact = root / "application"
            manifest = compile_semantic_application(
                source,
                adapter=adapter,
                snapshot=snapshot,
                target_source=ROOT,
                destination=artifact,
                limits=SearchLimits(),
                name="constant_seed",
                capacity=1 << 20,
                source_folding=receipt,
            )
            self.assertEqual(len(manifest["segments"]), 1)
            self.assertIn("source-constant-folding.json", manifest["stage_artifacts"])
            replay = verify_semantic_application(
                source,
                adapter=adapter,
                snapshot=snapshot,
                target_source=ROOT,
                artifact=artifact,
                source_folding=receipt,
            )
            self.assertEqual(replay["status"], "checked")
            with self.assertRaisesRegex(ValueError, "constant-folding receipt"):
                verify_semantic_application(
                    source, adapter=adapter, snapshot=snapshot, target_source=ROOT, artifact=artifact
                )
            changed = json.loads((artifact / "source-constant-folding.json").read_text())
            changed["folds"][0]["result_sha256"] = "0" * 64
            (artifact / "source-constant-folding.json").write_text(json.dumps(changed))
            with self.assertRaisesRegex(ValueError, "differs from source intake"):
                verify_semantic_application(
                    source,
                    adapter=adapter,
                    snapshot=snapshot,
                    target_source=ROOT,
                    artifact=artifact,
                    source_folding=receipt,
                )

    @unittest.skipUnless(
        os.environ.get("MERLIN_NATIVE_ROOT") and os.environ.get("ATLAS_OOT_BIN_DIR"),
        "select Merlin and Atlas build tools",
    )
    def test_batched_k_chunks_use_flat_members_and_the_existing_batch_wrapper(self):
        from atlas_mlir_native.application import AtlasApplicationAdapter
        from test_looped_mnk import build_snapshot

        from merlin.semantic_compiler.search import SearchLimits
        from merlin.tests.runtime.test_batched_c import source_graph

        source = source_graph((2,))
        shapes = ((2, 1, 4097), (2, 4097, 2), (2, 1, 2), (2, 1, 2))
        policy = self.policy()
        source = replace(
            source,
            target_identity=policy["target_identity"],
            nodes=tuple(
                replace(node, type=replace(node.type, shape=shape))
                for node, shape in zip(source.nodes, shapes, strict=True)
            ),
            constants=(replace(source.constants[0], data_hex="00000000" * 4),),
        )
        adapter = AtlasApplicationAdapter(policy, original_source_sha256=policy["source_sha256"])
        (selected,) = adapter.plan(source).selected
        with tempfile.TemporaryDirectory(dir=os.environ.get("TMPDIR")) as directory:
            root = Path(directory)
            segments = root / "segments"
            segments.mkdir()
            compiled = adapter.compile_region(
                build_snapshot(root), source, selected, "batch", segments, ROOT, SearchLimits()
            )
            self.assertEqual(compiled.programs, ("segments/batch_first", "segments/batch_carry"))
            self.assertEqual(compiled.description["native_export"]["launches_per_call"], 4)
            self.assertEqual(compiled.description["native_export"]["batching"]["batch_count"], 2)
            replayed = adapter.export_region(source, selected, "batch", segments, ROOT)
            self.assertEqual(compiled, replayed)
            for path, payload in compiled.files.items():
                (root / path).parent.mkdir(parents=True, exist_ok=True)
                (root / path).write_bytes(payload)
            c_sources = list(root.rglob("*.c"))
            for index, path in enumerate(c_sources):
                subprocess.run(
                    [
                        "cc",
                        "-std=c11",
                        "-Wall",
                        "-Wextra",
                        "-Werror",
                        "-c",
                        str(path),
                        "-o",
                        str(root / f"source-{index}.o"),
                    ],
                    check=True,
                    capture_output=True,
                )

    def test_runtime_indexed_state_requires_explicit_host_admission(self):
        from atlas_mlir_native.application import AtlasApplicationAdapter
        from atlas_mlir_native.pointwise_math import MATH_POLICY
        from atlas_mlir_native.reduction import REDUCTION_POLICY
        from atlas_mlir_native.row_softmax import MASKED_SOFTMAX_POLICY
        from atlas_mlir_native.sigmoid import SIGMOID_POLICY

        from merlin.semantic_compiler.linalg_bridge import translate_linalg_text
        from merlin.semantic_compiler.target_binding import NativeCompilationError
        from merlin.tests.runtime.test_indexed_update_c import source_text

        old = {
            **self.policy(),
            "schema": "atlas.diagnostic_application_precision.v10",
            "reduction": REDUCTION_POLICY,
            "pointwise_math": MATH_POLICY,
            "sigmoid": SIGMOID_POLICY,
            "masked_softmax": MASKED_SOFTMAX_POLICY,
        }
        source = translate_linalg_text(
            source_text(),
            entry="forward",
            target_identity=old["target_identity"],
            lowering_policy="diagnostic",
        ).request
        policy = {
            **old,
            "schema": "atlas.diagnostic_application_precision.v11",
            "host_support": [*old["host_support"], "indexed_insert_slice"],
        }
        for rejected in (old, {**policy, "schema": old["schema"]}):
            with self.subTest(policy=rejected), self.assertRaises((ValueError, NativeCompilationError)):
                AtlasApplicationAdapter(rejected, original_source_sha256=rejected["source_sha256"]).plan(source)
        adapter = AtlasApplicationAdapter(policy, original_source_sha256=policy["source_sha256"])
        selected = adapter.plan(source).selected
        self.assertEqual(len(selected), 1)
        self.assertEqual(selected[0].placement, "host")
        exported = adapter.export_region(source, selected[0], "cache_update", Path("unused"), ROOT)
        self.assertEqual(exported.programs, ())
        self.assertEqual(exported.description["index_bounds"], [0, 8])
        self.assertFalse(exported.description["failure_writes_output"])

    def test_batched_relu_keeps_separate_native_regions_when_fusion_is_unavailable(self):
        from atlas_mlir_native.application import AtlasApplicationAdapter

        from merlin.semantic_compiler.model import SemanticNode, pointwise_identity_maps
        from merlin.tests.runtime.test_batched_c import source_graph

        policy = self.policy()
        adapter = AtlasApplicationAdapter(policy, original_source_sha256=policy["source_sha256"])
        for batch in ((), (2,), (2, 3)):
            with self.subTest(batch=batch):
                source = source_graph(batch)
                product = source.nodes[-1]
                relu = SemanticNode(
                    "activation",
                    "f32_relu_ordered_select",
                    (product.id,),
                    product.type,
                    (("numeric_semantics", "f32-ordered-lt-positive-zero-select-v1"),),
                    index_maps=pointwise_identity_maps(len(product.type.shape), 1),
                )
                source = replace(
                    source, nodes=(*source.nodes, relu), outputs=(relu.id,), target_identity=policy["target_identity"]
                )
                selected = adapter.plan(source).selected
                self.assertEqual(
                    [dict(r.metadata)["kind"] for r in selected],
                    ["contraction", "pointwise"] if batch else ["contraction"],
                )
                self.assertTrue(all(r.placement == "accelerator" for r in selected))
                self.assertEqual(
                    [dict(r.metadata).get("epilogue") for r in selected], [None, None] if batch else ["relu"]
                )

    def test_masked_softmax_requires_explicit_policy_and_preserves_mask_map(self):
        from atlas_mlir_native.application import AtlasApplicationAdapter
        from atlas_mlir_native.pointwise_math import MATH_POLICY
        from atlas_mlir_native.reduction import REDUCTION_POLICY
        from atlas_mlir_native.row_softmax import MASKED_SOFTMAX_POLICY
        from atlas_mlir_native.sigmoid import SIGMOID_POLICY
        from atlas_mlir_native.softmax_precision import lower_semantic_softmax

        from merlin.semantic_compiler.model import (
            IndexMap,
            SemanticNode,
            TensorType,
            pointwise_identity_maps,
        )
        from merlin.semantic_compiler.target_binding import NativeCompilationError
        from merlin.tests.ir.test_softmax_source import source_graph

        old = {
            **self.policy(),
            "schema": "atlas.diagnostic_application_precision.v9",
            "reduction": REDUCTION_POLICY,
            "pointwise_math": MATH_POLICY,
            "sigmoid": SIGMOID_POLICY,
        }
        policy = {
            **old,
            "schema": "atlas.diagnostic_application_precision.v10",
            "masked_softmax": MASKED_SOFTMAX_POLICY,
        }
        source = source_graph((2, 35))
        scores = source.nodes[0]
        mask = SemanticNode("mask", "input", (), TensorType((35,), "f32", "raw-bit-pattern"), effect="input")
        identity = pointwise_identity_maps(2, 1)[0]
        mapping = IndexMap(2, ((0, 1),), (0,))
        masked = SemanticNode(
            "masked",
            "f32_add",
            (scores.id, mask.id),
            scores.type,
            (("numeric_semantics", "ieee754-f32-add-rne-v1"),),
            index_maps=(identity, mapping, identity),
        )
        nodes = [scores, mask, masked]
        nodes.extend(
            replace(n, inputs=tuple(masked.id if key == scores.id else key for key in n.inputs))
            for n in source.nodes[1:]
        )
        source = replace(
            source,
            nodes=tuple(nodes),
            input_storages=((scores.id, "external"), (mask.id, "external")),
            lowering_policy="diagnostic",
            target_identity=policy["target_identity"],
        )
        legacy = AtlasApplicationAdapter(old, original_source_sha256=old["source_sha256"]).plan(source)
        self.assertEqual([dict(r.metadata)["kind"] for r in legacy.selected], ["pointwise", "softmax"])
        adapter = AtlasApplicationAdapter(policy, original_source_sha256=policy["source_sha256"])
        selected = adapter.plan(source).selected
        self.assertEqual(len(selected), 1)
        self.assertEqual(dict(selected[0].metadata)["kind"], "masked_softmax")
        precision = lower_semantic_softmax(source, allow_masked=True)
        self.assertEqual(precision.source_region.mask_map, mapping)
        conversion = next(r for r in precision.boundary_conversions if r["direction"] == "mask_to_keep")
        self.assertEqual(conversion["source_shape"], [35])
        self.assertEqual(conversion["admitted_f32_codes"], ["0x00000000", "0xff800000"])
        for bad in (
            {**policy, "masked_softmax": "unknown"},
            {k: v for k, v in policy.items() if k != "masked_softmax"},
        ):
            with self.assertRaises(NativeCompilationError):
                AtlasApplicationAdapter(bad, original_source_sha256=policy["source_sha256"])

    def test_embedding_requires_explicit_host_policy_and_preserves_integer_ports(self):
        from atlas_mlir_native.application import AtlasApplicationAdapter
        from atlas_mlir_native.pointwise_math import MATH_POLICY
        from atlas_mlir_native.reduction import REDUCTION_POLICY

        from merlin.semantic_compiler.model import (
            KernelRequest,
            SemanticNode,
            TensorType,
        )
        from merlin.semantic_compiler.target_binding import NativeCompilationError

        policy = {
            **self.policy(),
            "schema": "atlas.diagnostic_application_precision.v8",
            "host_support": ["embedding_lookup"],
            "reduction": REDUCTION_POLICY,
            "pointwise_math": MATH_POLICY,
        }
        indices = SemanticNode(
            "indices",
            "input",
            (),
            TensorType((2,), "i64", "signed-i64"),
            effect="input",
        )
        table = SemanticNode(
            "table",
            "input",
            (),
            TensorType((5, 3), "f32", "raw-bit-pattern"),
            effect="input",
        )
        result = SemanticNode(
            "result",
            "embedding_lookup",
            (indices.id, table.id),
            TensorType((2, 3), "f32", "raw-bit-pattern"),
            (
                ("index_domain", "nonnegative-below-table-rows"),
                ("minimum_index_bits", 4),
            ),
        )
        source = KernelRequest(
            (indices, table, result),
            (result.id,),
            ("external",),
            ((indices.id, "external"), (table.id, "external")),
            policy["target_identity"],
            "diagnostic",
        )
        for schema, support in (("v7", ["embedding_lookup"]), ("v8", ["reshape"])):
            with (
                self.subTest(schema=schema, support=support),
                self.assertRaises(NativeCompilationError),
            ):
                AtlasApplicationAdapter(
                    {
                        **policy,
                        "schema": "atlas.diagnostic_application_precision." + schema,
                        "host_support": support,
                    },
                    original_source_sha256=policy["source_sha256"],
                ).plan(source)
        adapter = AtlasApplicationAdapter(policy, original_source_sha256=policy["source_sha256"])
        selected = adapter.plan(source).selected
        self.assertEqual(len(selected), 1)
        self.assertEqual(selected[0].placement, "host")
        exported = adapter.export_region(source, selected[0], "lookup", Path("unused"), ROOT)
        self.assertEqual(exported.programs, ())
        self.assertEqual(exported.description["index_dtype"], "i64")
        self.assertFalse(exported.description["failure_writes_output"])
        for wrong in (
            replace(result, attrs=()),
            replace(result, type=replace(result.type, shape=(3, 2))),
        ):
            with self.subTest(wrong=wrong), self.assertRaises(ValueError):
                adapter.plan(replace(source, nodes=(indices, table, wrong)))

    def policy(self):
        from atlas_mlir_native.application_precision import SCALED_CONTRACTION_POLICY
        from atlas_mlir_native.binding import TARGET_IDENTITY
        from atlas_mlir_native.pointwise import POINTWISE_POLICY
        from atlas_mlir_native.row_softmax import SOFTMAX_POLICY

        return {
            "schema": "atlas.diagnostic_application_precision.v4",
            "source_sha256": "0" * 64,
            "target_identity": TARGET_IDENTITY,
            "contraction": {
                "policy": SCALED_CONTRACTION_POLICY,
                "operand_scale_exponents": [4, 4],
            },
            "row_softmax": SOFTMAX_POLICY,
            "host_support": ["transpose", "reshape", "copy"],
            "epilogues": ["relu"],
            "pointwise": POINTWISE_POLICY,
        }

    def test_integer_operations_require_explicit_cpu_placement(self):
        from unittest.mock import patch

        from atlas_mlir_native.application import AtlasApplicationAdapter
        from atlas_mlir_native.pointwise_math import MATH_POLICY
        from atlas_mlir_native.reduction import REDUCTION_POLICY
        from atlas_mlir_native.row_softmax import MASKED_SOFTMAX_POLICY
        from atlas_mlir_native.sigmoid import SIGMOID_POLICY

        from merlin.compile.c_integer import INTEGER_OPERATIONS
        from merlin.semantic_compiler.target_binding import NativeCompilationError
        from merlin.tests.runtime.test_integer_c import request_for

        old = {
            **self.policy(),
            "schema": "atlas.diagnostic_application_precision.v12",
            "reduction": REDUCTION_POLICY,
            "pointwise_math": MATH_POLICY,
            "sigmoid": SIGMOID_POLICY,
            "masked_softmax": MASKED_SOFTMAX_POLICY,
            "host_predicates": ["mask"],
        }
        policy = {
            **old,
            "schema": "atlas.diagnostic_application_precision.v13",
            "host_integer_ops": list(INTEGER_OPERATIONS),
            "host_support": ["transpose", "reshape", "copy", "embedding_lookup", "indexed_insert_slice"],
        }
        adapter = AtlasApplicationAdapter(policy, original_source_sha256=policy["source_sha256"])
        for operation in INTEGER_OPERATIONS:
            with self.subTest(operation=operation):
                source = replace(
                    request_for(operation), target_identity=old["target_identity"], lowering_policy="diagnostic"
                )
                region = adapter.plan(source).selected[0]
                self.assertEqual(region.placement, "host")
                compiled = adapter.compile_region(None, source, region, "integer", None, ROOT, None)
                self.assertEqual(compiled.description["arithmetic"], "exact-source-integer")
                self.assertFalse(compiled.programs)
                with self.assertRaisesRegex(NativeCompilationError, "no compiled application route"):
                    AtlasApplicationAdapter(old, original_source_sha256=old["source_sha256"]).plan(source)
        for operations in ([], ["i64_add", "i64_add"], ["source_f32_matmul_unqualified"]):
            with self.assertRaisesRegex(NativeCompilationError, "host integer selection"):
                AtlasApplicationAdapter(
                    {**policy, "host_integer_ops": operations}, original_source_sha256=policy["source_sha256"]
                )
        with patch.dict(sys.modules, {"merlin.compile.c_integer": None}):
            with self.assertRaisesRegex(NativeCompilationError, "integer C emitter"):
                adapter.plan(source)

    def test_float_casts_require_explicit_cpu_placement(self):
        from atlas_mlir_native.application import AtlasApplicationAdapter
        from atlas_mlir_native.gelu import GELU_POLICY
        from atlas_mlir_native.pointwise_math import MATH_POLICY
        from atlas_mlir_native.reduction import REDUCTION_POLICY
        from atlas_mlir_native.row_softmax import GUARDED_SOFTMAX_POLICY, MASKED_SOFTMAX_POLICY
        from atlas_mlir_native.sigmoid import SIGMOID_POLICY

        from merlin.compile.c_float_casts import FLOAT_CAST_OPERATIONS
        from merlin.semantic_compiler.target_binding import NativeCompilationError
        from merlin.tests.runtime.test_float_casts_c import request_for

        policy = {
            **self.policy(), "schema": "atlas.diagnostic_application_precision.v17",
            "reduction": REDUCTION_POLICY, "pointwise_math": MATH_POLICY, "sigmoid": SIGMOID_POLICY,
            "masked_softmax": MASKED_SOFTMAX_POLICY, "guarded_masked_softmax": GUARDED_SOFTMAX_POLICY,
            "gelu": GELU_POLICY, "host_predicates": ["mask"], "host_integer_ops": ["i64_add"],
            "host_float_casts": list(FLOAT_CAST_OPERATIONS),
        }
        adapter = AtlasApplicationAdapter(policy, original_source_sha256=policy["source_sha256"])
        old = {key: value for key, value in policy.items() if key != "host_float_casts"}
        old["schema"] = "atlas.diagnostic_application_precision.v16"
        for operation in FLOAT_CAST_OPERATIONS:
            with self.subTest(operation=operation):
                source = replace(request_for(operation), target_identity=policy["target_identity"],
                                 lowering_policy="diagnostic")
                region, = adapter.plan(source).selected
                self.assertEqual(region.placement, "host")
                compiled = adapter.compile_region(None, source, region, "cast", None, ROOT, None)
                self.assertEqual(compiled.description["arithmetic"], "exact-source-float-conversion")
                self.assertFalse(compiled.programs)
                with self.assertRaisesRegex(NativeCompilationError, "no compiled application route"):
                    AtlasApplicationAdapter(old, original_source_sha256=old["source_sha256"]).plan(source)
        for operations in ([], ["bf16_to_f32", "bf16_to_f32"], ["source_f64_powf_unqualified"]):
            with self.subTest(operations=operations), self.assertRaisesRegex(
                NativeCompilationError, "float cast selection"
            ):
                AtlasApplicationAdapter({**policy, "host_float_casts": operations},
                                        original_source_sha256=policy["source_sha256"])
        identity_policy = {**policy, "host_support": [*policy["host_support"], "identity"]}
        source = replace(request_for("bf16_to_f32"), target_identity=policy["target_identity"],
                         lowering_policy="diagnostic")
        source = replace(source, nodes=(source.nodes[0], replace(
            source.nodes[-1], op="identity", type=source.nodes[0].type, attrs=(), index_maps=(),
        )))
        adapter = AtlasApplicationAdapter(identity_policy, original_source_sha256=policy["source_sha256"])
        region, = adapter.plan(source).selected
        compiled = adapter.compile_region(None, source, region, "identity", None, ROOT, None)
        self.assertEqual(compiled.description["arithmetic"], "byte-preserving")
        self.assertFalse(compiled.programs)

    def test_bucketize_requires_version_14_and_explicit_cpu_placement(self):
        from unittest.mock import patch

        from atlas_mlir_native.application import AtlasApplicationAdapter
        from atlas_mlir_native.pointwise_math import MATH_POLICY
        from atlas_mlir_native.reduction import REDUCTION_POLICY
        from atlas_mlir_native.row_softmax import MASKED_SOFTMAX_POLICY
        from atlas_mlir_native.sigmoid import SIGMOID_POLICY

        from merlin.semantic_compiler.target_binding import NativeCompilationError
        from merlin.tests.runtime.test_bucketize_c import request_for

        policy = {
            **self.policy(),
            "schema": "atlas.diagnostic_application_precision.v14",
            "reduction": REDUCTION_POLICY,
            "pointwise_math": MATH_POLICY,
            "sigmoid": SIGMOID_POLICY,
            "masked_softmax": MASKED_SOFTMAX_POLICY,
            "host_predicates": ["bucketize"],
            "host_integer_ops": ["i64_add"],
        }
        source = replace(request_for(True), target_identity=policy["target_identity"], lowering_policy="diagnostic")
        adapter = AtlasApplicationAdapter(policy, original_source_sha256=policy["source_sha256"])
        region = adapter.plan(source).selected[0]
        self.assertEqual(region.placement, "host")
        compiled = adapter.compile_region(None, source, region, "boundary_count", None, ROOT, None)
        self.assertEqual(compiled.description["arithmetic"], "exact-source-boundary-count")
        self.assertEqual(compiled.inputs, ("values", "limits"))
        self.assertFalse(compiled.programs)
        old = {**policy, "schema": "atlas.diagnostic_application_precision.v13", "host_predicates": ["mask"]}
        with self.assertRaisesRegex(NativeCompilationError, "no compiled application route"):
            AtlasApplicationAdapter(old, original_source_sha256=old["source_sha256"]).plan(source)
        with self.assertRaisesRegex(NativeCompilationError, "host predicate selection"):
            AtlasApplicationAdapter(
                {**old, "host_predicates": ["bucketize"]}, original_source_sha256=old["source_sha256"]
            )
        with patch.dict(sys.modules, {"merlin.compile.c_bucketize": None}):
            with self.assertRaisesRegex(NativeCompilationError, "predicate C emitter"):
                adapter.plan(source)

    def test_source_predicates_require_explicit_host_placement(self):
        from atlas_mlir_native.application import AtlasApplicationAdapter
        from atlas_mlir_native.pointwise_math import MATH_POLICY
        from atlas_mlir_native.reduction import REDUCTION_POLICY
        from atlas_mlir_native.row_softmax import MASKED_SOFTMAX_POLICY
        from atlas_mlir_native.sigmoid import SIGMOID_POLICY

        from merlin.semantic_compiler.target_binding import NativeCompilationError
        from merlin.tests.runtime.test_predicates_c import request_for

        old = {
            **self.policy(),
            "schema": "atlas.diagnostic_application_precision.v11",
            "reduction": REDUCTION_POLICY,
            "pointwise_math": MATH_POLICY,
            "sigmoid": SIGMOID_POLICY,
            "masked_softmax": MASKED_SOFTMAX_POLICY,
        }
        source = replace(request_for("nonzero"), target_identity=old["target_identity"], lowering_policy="diagnostic")
        with self.assertRaisesRegex(NativeCompilationError, "no compiled application route"):
            AtlasApplicationAdapter(old, original_source_sha256=old["source_sha256"]).plan(source)
        selected = {
            **old,
            "schema": "atlas.diagnostic_application_precision.v12",
            "host_predicates": ["f32_nonzero", "mask", "select"],
        }
        adapter = AtlasApplicationAdapter(selected, original_source_sha256=selected["source_sha256"])
        for kind in ("nonzero", "slt", "andi", "extui"):
            source = replace(
                request_for(kind), target_identity=selected["target_identity"], lowering_policy="diagnostic"
            )
            (region,) = adapter.plan(source).selected
            self.assertEqual(region.placement, "host")
            self.assertEqual(dict(region.metadata)["kind"], "predicate")
            compiled = adapter.compile_region(None, source, region, "predicate", None, ROOT, None)
            self.assertFalse(compiled.programs)
            self.assertEqual(compiled.description["arithmetic"], "exact-source-predicate")
            self.assertIn("predicate.c", compiled.files)
        from merlin.semantic_compiler.model import TensorType

        for dtype in ("f32", "f64"):
            floats = request_for("slt")
            floats = replace(
                floats,
                nodes=(
                    *tuple(
                        replace(n, type=TensorType(n.type.shape, dtype, "raw-bit-pattern")) for n in floats.nodes[:-1]
                    ),
                    replace(floats.nodes[-1], attrs=(("operation", "arith.cmpf"), ("predicate", "ogt"))),
                ),
                target_identity=selected["target_identity"],
                lowering_policy="diagnostic",
            )
            (region,) = adapter.plan(floats).selected
            self.assertEqual(region.placement, "host")
            self.assertFalse(adapter.compile_region(None, floats, region, "compare", None, ROOT, None).programs)
        from unittest.mock import patch

        with patch.dict(sys.modules, {"merlin.compile.c_predicates": None}):
            with self.assertRaisesRegex(NativeCompilationError, "tool_unavailable"):
                adapter.plan(source)
        floating_mask = replace(source.nodes[-1], attrs=(("operation", "arith.cmpf"), ("predicate", "ult")))
        with self.assertRaisesRegex(NativeCompilationError, "no compiled implementation"):
            adapter.plan(replace(source, nodes=(*source.nodes[:-1], floating_mask)))
        restricted = {**selected, "host_predicates": ["f32_nonzero"]}
        adapter = AtlasApplicationAdapter(restricted, original_source_sha256=restricted["source_sha256"])
        with self.assertRaisesRegex(NativeCompilationError, "no compiled application route"):
            adapter.plan(source)
        for predicates in ([], ["mask", "mask"], ["source_f32_matmul_unqualified"]):
            bad = {**selected, "host_predicates": predicates}
            with self.assertRaisesRegex(NativeCompilationError, "host predicate selection"):
                AtlasApplicationAdapter(bad, original_source_sha256=bad["source_sha256"])

    def test_indexed_host_layouts_require_explicit_admission(self):
        from atlas_mlir_native.application import AtlasApplicationAdapter

        from merlin.semantic_compiler.model import (
            IndexMap,
            KernelRequest,
            SemanticNode,
            TensorType,
            pointwise_identity_maps,
        )
        from merlin.semantic_compiler.target_binding import NativeCompilationError

        policy = self.policy()
        a = SemanticNode(
            "x",
            "input",
            (),
            TensorType((3, 5), "f32", "raw-bit-pattern"),
            effect="input",
        )
        out = SemanticNode(
            "out",
            "gather",
            ("x",),
            a.type,
            index_maps=(
                IndexMap(2, ((1, 0), (0, -1)), (0, 4)),
                pointwise_identity_maps(2, 1)[0],
            ),
        )
        source = KernelRequest(
            (a, out),
            (out.id,),
            ("external",),
            ((a.id, "external"),),
            policy["target_identity"],
            "diagnostic",
        )
        with self.assertRaises(NativeCompilationError):
            AtlasApplicationAdapter(policy, original_source_sha256=policy["source_sha256"]).plan(source)
        policy.update(schema="atlas.diagnostic_application_precision.v5", host_support=["gather"])
        adapter = AtlasApplicationAdapter(policy, original_source_sha256=policy["source_sha256"])
        selected = adapter.plan(source).selected
        self.assertEqual(len(selected), 1)
        self.assertEqual(selected[0].placement, "host")
        self.assertEqual(dict(selected[0].metadata)["kind"], "layout")
        exported = adapter.export_region(source, selected[0], "layout", Path("unused"), ROOT)
        self.assertEqual(exported.programs, ())
        self.assertEqual(exported.description["arithmetic"], "byte-preserving")
        wrong = replace(out, index_maps=(IndexMap(2, ((1, 0), (0, 1)), (0, 1)), out.index_maps[-1]))
        with self.assertRaises(ValueError):
            adapter.plan(replace(source, nodes=(a, wrong)))

    def test_host_policy_cannot_admit_arithmetic_or_change_legacy_policies(self):
        from atlas_mlir_native.application import AtlasApplicationAdapter

        from merlin.semantic_compiler.target_binding import NativeCompilationError

        policy = self.policy()
        for operations in (
            [],
            ["gather", "gather"],
            ["f32_add"],
            ["matmul"],
            [None],
            "gather",
        ):
            with (
                self.subTest(operations=operations),
                self.assertRaises(NativeCompilationError),
            ):
                AtlasApplicationAdapter(
                    {
                        **policy,
                        "schema": "atlas.diagnostic_application_precision.v5",
                        "host_support": operations,
                    },
                    original_source_sha256=policy["source_sha256"],
                )
        with self.assertRaises(NativeCompilationError):
            AtlasApplicationAdapter(
                {**policy, "host_support": [*policy["host_support"], "gather"]},
                original_source_sha256=policy["source_sha256"],
            )

    def test_reductions_require_explicit_numerical_policy_and_checked_maps(self):
        from atlas_mlir_native.application import AtlasApplicationAdapter
        from atlas_mlir_native.reduction import (
            REDUCTION_POLICY,
            prepare_source,
            request_for,
        )

        from merlin.semantic_compiler.target_binding import NativeCompilationError

        raw = request_for((2, 3, 35), (0, 2))
        x, seed, result = raw.nodes
        source = replace(
            raw,
            nodes=(
                replace(x, type=replace(x.type, dtype="f32"), attrs=()),
                replace(seed, type=replace(seed.type, dtype="f32"), attrs=()),
                replace(
                    result,
                    op="source_f32_reduce_add_unqualified",
                    type=replace(result.type, dtype="f32", numerical_policy="raw-bit-pattern"),
                    attrs=(
                        ("axes", "0,2"),
                        ("reduction_order", "unresolved"),
                        ("numeric_semantics", "source-f32-add-unqualified-v1"),
                        ("scalar_add_order", "input-accumulator"),
                    ),
                ),
            ),
        )
        policy = self.policy()
        with self.assertRaises(NativeCompilationError):
            AtlasApplicationAdapter(policy, original_source_sha256=policy["source_sha256"]).plan(source)
        policy.update(
            schema="atlas.diagnostic_application_precision.v6",
            reduction=REDUCTION_POLICY,
        )
        selected = AtlasApplicationAdapter(policy, original_source_sha256=policy["source_sha256"]).plan(source).selected
        self.assertEqual(len(selected), 1)
        self.assertEqual(selected[0].placement, "accelerator")
        self.assertEqual(dict(selected[0].metadata)["kind"], "reduction")
        precision = prepare_source(source)
        self.assertEqual(precision.request.nodes[-1].type.numerical_policy, REDUCTION_POLICY)
        for mutation in (
            replace(source.nodes[-1], attrs=()),
            replace(source.nodes[-1], index_maps=()),
        ):
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                prepare_source(replace(source, nodes=(*source.nodes[:2], mutation)))
        with self.assertRaises(NativeCompilationError):
            AtlasApplicationAdapter(
                {**policy, "reduction": "unknown"},
                original_source_sha256=policy["source_sha256"],
            )
        from atlas_mlir_native.pointwise_math import MATH_POLICY

        math_policy = {
            **policy,
            "schema": "atlas.diagnostic_application_precision.v7",
            "pointwise_math": MATH_POLICY,
        }
        self.assertTrue(
            AtlasApplicationAdapter(math_policy, original_source_sha256=policy["source_sha256"]).pointwise_math
        )
        for changed in (
            {**math_policy, "pointwise_math": "unknown"},
            {**math_policy, "schema": "atlas.diagnostic_application_precision.v6"},
        ):
            with (
                self.subTest(changed=changed),
                self.assertRaises(NativeCompilationError),
            ):
                AtlasApplicationAdapter(changed, original_source_sha256=policy["source_sha256"])


@unittest.skipUnless(
    all(
        os.environ.get(name)
        for name in (
            "ATLAS_APPLICATION_ARTIFACT",
            "ATLAS_APPLICATION_INPUTS",
            "ATLAS_APPLICATION_SNAPSHOT",
            "ATLAS_LLVM_BIN",
        )
    ),
    "select a compiled application, its staged inputs/snapshot and LLVM",
)
class ApplicationCPackageTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(dir=os.environ.get("TMPDIR"))
        self.root = Path(self.temporary.name)
        self.artifact = Path(os.environ["ATLAS_APPLICATION_ARTIFACT"])
        self.cli = Path(sys.executable).with_name("merlin-targetgen")

    def tearDown(self):
        self.temporary.cleanup()

    def test_scales_are_explicit_and_replayed_for_each_contraction(self):
        from atlas_mlir_native.application import AtlasApplicationAdapter
        from atlas_mlir_native.application_precision import prepare_contraction

        from merlin.compile.semantic_regions import extract_semantic_region
        from merlin.semantic_compiler.model import KernelRequest
        from merlin.semantic_compiler.target_binding import NativeCompilationError

        source = KernelRequest.from_record(json.loads((self.artifact / "source-request.json").read_text()))
        node = next(node for node in source.nodes if node.op == "source_f32_matmul_unqualified")
        region = extract_semantic_region(source, (node.id,))
        old = prepare_contraction(region)
        scaled = prepare_contraction(region, operand_scale_exponents=(4, 3))
        self.assertNotEqual(old.request.digest(), scaled.request.digest())
        self.assertEqual(scaled.output.spec.post_decode_scale_exponent, -7)
        self.assertEqual([item.spec.pre_encode_scale_exponent for item in scaled.inputs], [4, 3])
        self.assertNotEqual(
            scaled.record(),
            prepare_contraction(region, operand_scale_exponents=(3, 4)).record(),
        )
        # A changed unknown attribute must not become eligible because scales
        # were selected, and an unreviewed policy field cannot be ignored.
        changed = replace(
            region,
            nodes=tuple(
                replace(item, attrs=(*item.attrs, ("relaxed", True))) if item.id == node.id else item
                for item in region.nodes
            ),
        )
        with self.assertRaises(ValueError):
            prepare_contraction(changed, operand_scale_exponents=(4, 3))
        policy = json.loads((Path(os.environ["ATLAS_APPLICATION_INPUTS"]) / "precision-policy.json").read_text())
        with self.assertRaises(NativeCompilationError):
            AtlasApplicationAdapter(
                {**policy, "allow_reassociation": True},
                original_source_sha256=policy["source_sha256"],
            )
        for exponents in ((True, 4), (33, 0), (-33, 0), (1,), "4,4"):
            with self.subTest(exponents=exponents), self.assertRaises(ValueError):
                prepare_contraction(region, operand_scale_exponents=exponents)

    def test_relu_fusion_requires_a_closed_unobserved_contraction(self):
        from atlas_mlir_native.application import AtlasApplicationAdapter
        from atlas_mlir_native.application_precision import (
            SCALED_CONTRACTION_POLICY,
            prepare_contraction,
        )
        from atlas_mlir_native.looped_tiling import (
            _virtual,
            prepare_plan,
            split_epilogue,
        )

        from merlin.compile.semantic_regions import extract_semantic_region
        from merlin.semantic_compiler.model import (
            KernelRequest,
            SemanticNode,
            pointwise_identity_maps,
        )
        from merlin.semantic_compiler.target_binding import NativeCompilationError

        graph = KernelRequest.from_record(json.loads((self.artifact / "source-request.json").read_text()))
        matmul = next(node for node in graph.nodes if node.op == "source_f32_matmul_unqualified")
        source = extract_semantic_region(graph, (matmul.id,))
        relu = SemanticNode(
            "activation",
            "f32_relu_ordered_select",
            (matmul.id,),
            matmul.type,
            (("numeric_semantics", "f32-ordered-lt-positive-zero-select-v1"),),
            index_maps=pointwise_identity_maps(2, 1),
        )
        source = replace(source, nodes=(*source.nodes, relu), outputs=(relu.id,))
        precision = prepare_contraction(source, operand_scale_exponents=(4, 4), epilogue="relu")
        self.assertEqual(precision.source.digest(), source.digest())
        self.assertEqual(precision.request.node(precision.request.outputs[0]).op, "relu")
        self.assertEqual(source.nodes[precision.output.source_value.index].id, relu.id)
        contraction, epilogue = split_epilogue(precision.request)
        self.assertEqual(contraction.outputs, ("result",))
        virtual = _virtual(
            prepare_plan(precision.request),
            (0x90000000, 0x90010000),
            0x90020000,
            epilogue,
        )
        self.assertEqual(virtual.count('kind = "relu"'), 1)
        self.assertGreater(virtual.index('kind = "relu"'), virtual.index("^store(%ss:"))
        policy = json.loads((Path(os.environ["ATLAS_APPLICATION_INPUTS"]) / "precision-policy.json").read_text())
        policy.update(
            schema="atlas.diagnostic_application_precision.v3",
            epilogues=["relu"],
            contraction={
                "policy": SCALED_CONTRACTION_POLICY,
                "operand_scale_exponents": [4, 4],
            },
        )
        policy.pop("pointwise", None)
        policy["host_support"] = ["transpose", "reshape", "copy"]
        adapter = AtlasApplicationAdapter(policy, original_source_sha256=policy["source_sha256"])
        selected = adapter.plan(source).selected
        self.assertEqual(len(selected), 1)
        self.assertEqual(selected[0].nodes, (0, 1))
        observed = replace(
            source,
            outputs=(matmul.id, relu.id),
            output_storages=("external", "external"),
        )
        with self.assertRaises(NativeCompilationError):
            adapter.plan(observed)
        shared = SemanticNode(
            "copy",
            "copy",
            (matmul.id,),
            matmul.type,
            index_maps=pointwise_identity_maps(2, 1),
        )
        with self.assertRaises(NativeCompilationError):
            adapter.plan(
                replace(
                    source,
                    nodes=(*source.nodes, shared),
                    outputs=(relu.id, shared.id),
                    output_storages=("external", "external"),
                )
            )
        for bad in (
            replace(relu, attrs=()),
            replace(relu, index_maps=()),
            replace(relu, type=replace(relu.type, numerical_policy="different")),
        ):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                prepare_contraction(
                    replace(source, nodes=(*source.nodes[:-1], bad)),
                    operand_scale_exponents=(4, 4),
                    epilogue="relu",
                )

    def verify(self, artifact, label):
        command = [
            str(self.cli),
            "native-verify",
            "--engine",
            "merlin_native",
            "--support",
            "atlas",
            "--snapshot",
            os.environ["ATLAS_APPLICATION_SNAPSHOT"],
            "--compile-inputs",
            os.environ["ATLAS_APPLICATION_INPUTS"],
            "--application",
            "--mode",
            "diagnostic",
            "--target-source",
            str(ROOT),
            "--artifact",
            str(artifact),
            "--out",
            str(self.root / (label + ".json")),
        ]
        return subprocess.run(command, capture_output=True, text=True, check=False)

    def test_cli_replays_source_regions_and_links_all_C_with_one_runtime(self):
        result = self.verify(self.artifact, "check")
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        manifest = json.loads((self.artifact / "manifest.json").read_text())
        package = self.artifact / "c-package"
        # Compile from only the exported pair, outside the generated source
        # tree, so a missing checkout include cannot be hidden by -I paths.
        isolated = self.root / "isolated-pair"
        isolated.mkdir()
        header = manifest["application_interface"]["header"]
        self.assertEqual(len(manifest["link_sources"]), 1)
        shutil.copyfile(package / header, isolated / Path(header).name)
        for source in manifest["link_sources"]:
            shutil.copyfile(package / source, isolated / Path(source).name)
        llvm = Path(os.environ["ATLAS_LLVM_BIN"])
        linker = shutil.which("ld.lld")
        self.assertIsNotNone(linker)
        objects = []
        for index, source in enumerate(manifest["link_sources"]):
            output = self.root / f"source-{index}.o"
            subprocess.run(
                [
                    str(llvm / "clang"),
                    "--target=riscv64-unknown-elf",
                    "-march=rv64gcv",
                    "-mabi=lp64d",
                    "-ffreestanding",
                    "-std=c11",
                    "-O2",
                    "-Wall",
                    "-Wextra",
                    "-Werror",
                    "-I",
                    str(isolated),
                    "-c",
                    str(isolated / Path(source).name),
                    "-o",
                    str(output),
                ],
                check=True,
                capture_output=True,
            )
            self.assertEqual(int.from_bytes(output.read_bytes()[48:52], "little") & 6, 4)
            objects.append(str(output))
        subprocess.run(
            [linker, "-r", *objects, "-o", str(self.root / "application.o")],
            check=True,
            capture_output=True,
        )
        symbols = subprocess.check_output(
            [str(llvm / "llvm-nm"), "--defined-only", str(self.root / "application.o")],
            text=True,
        )
        self.assertEqual(sum(line.endswith(" atlas_host_init") for line in symbols.splitlines()), 1)
        self.assertTrue(
            any(line.endswith(" " + manifest["application_interface"]["entry"]) for line in symbols.splitlines())
        )

    def test_rehashed_mutations_do_not_bypass_source_replay(self):
        for fault in (
            "C_error_status",
            "native_binary",
            "source_constant",
            "output_order",
            "engine",
            "placement",
        ):
            with self.subTest(fault=fault):
                artifact = self.root / fault
                shutil.copytree(self.artifact, artifact)
                manifest = json.loads((artifact / "manifest.json").read_text())
                if fault == "C_error_status":
                    path = artifact / "c-package" / manifest["application_interface"]["source"]
                    text = path.read_text()
                    self.assertIn("if (status) return status;", text)
                    path.write_text(text.replace("if (status) return status;", "if (status) return 0;"))
                elif fault == "native_binary":
                    path = artifact / manifest["segments"][0]["path"] / "program.bin"
                    data = path.read_bytes()
                    path.write_bytes(bytes([data[0] ^ 1]) + data[1:])
                elif fault == "source_constant":
                    path = artifact / "source-request.json"
                    source = json.loads(path.read_text())
                    data = source["constants"][0]["data_hex"]
                    source["constants"][0]["data_hex"] = ("00" if data[:2] != "00" else "01") + data[2:]
                    path.write_text(json.dumps(source))
                elif fault == "output_order":
                    path = artifact / "program.json"
                    program = json.loads(path.read_text())
                    program["dispatch"]["results"].reverse()
                    path.write_text(json.dumps(program))
                elif fault == "engine":
                    manifest["engine"] = "act_reference"
                else:
                    next(region for region in manifest["regions"] if region["placement"] == "accelerator")[
                        "placement"
                    ] = "host"
                # File hashes alone are insufficient: keep them consistent and
                # require source/candidate replay to catch the changed content.
                manifest["stage_artifacts"] = {
                    name: hashlib.sha256((artifact / name).read_bytes()).hexdigest()
                    for name in manifest["stage_artifacts"]
                }
                (artifact / "manifest.json").write_text(json.dumps(manifest))
                result = self.verify(artifact, fault)
                self.assertNotEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_resource_failure_keeps_diagnostics_and_cannot_reuse_stale_success(self):
        destination = self.root / "small-memory"
        command = [
            str(self.cli),
            "native-compile",
            "--engine",
            "merlin_native",
            "--support",
            "atlas",
            "--snapshot",
            os.environ["ATLAS_APPLICATION_SNAPSHOT"],
            "--compile-inputs",
            os.environ["ATLAS_APPLICATION_INPUTS"],
            "--application",
            "--mode",
            "diagnostic",
            "--target-source",
            str(ROOT),
            "--abi",
            "auto",
            "--application-memory-bytes",
            "1",
            "--out",
            str(destination),
        ]
        result = subprocess.run(command, capture_output=True, text=True, check=False)
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(json.loads(result.stdout)["status"], "resource_limit")
        manifest = (destination / "manifest.json").read_bytes()
        self.assertTrue(json.loads(manifest)["diagnostics_only"])
        self.assertTrue((destination / "source-request.json").is_file())
        self.assertTrue((destination / "segments").is_dir())
        verify = self.verify(destination, "failed-artifact")
        self.assertNotEqual(verify.returncode, 0)
        self.assertEqual(json.loads(verify.stdout)["status"], "resource_limit")
        self.assertNotEqual(subprocess.run(command, capture_output=True, check=False).returncode, 0)
        self.assertEqual((destination / "manifest.json").read_bytes(), manifest)


if __name__ == "__main__":
    unittest.main()
