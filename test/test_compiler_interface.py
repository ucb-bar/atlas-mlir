"""Lightweight declarations and source pin checks; no compiler or board runs."""
from dataclasses import dataclass, replace
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock, patch

from atlas_mlir_native import compiler_interface as interface
from merlin.compiler.backend.protocol import ProposalBudget, ProposalContext
from merlin.compiler.lower.c import host_target


class CompilerInterfaceTests(unittest.TestCase):
    def test_description_names_required_source_roots_and_limits(self):
        description = interface.describe_compiler()
        roots = {op for pattern in description.patterns for op in pattern.root_operations}
        self.assertTrue({"atlas.quantize_act.codes", "atlas.quantize_act.exponent", "atlas.qlinear",
                         "gather", "bf16_add"} <= roots)
        # Host CPU families are the host target's declaration, not Atlas's.
        self.assertNotIn("relu", roots)
        self.assertIn("relu", host_target.describe_compiler().source_domain.operations)
        self.assertNotIn("source_f32_matmul_unqualified", roots)
        self.assertNotIn("i32_add", roots)
        self.assertIn("limited", dict(description.source_domain.constraints)["scope"])
        self.assertIn("unpinned", dict(description.source_domain.constraints)["implementation-pin"])
        self.assertEqual(roots, set(description.source_domain.operations))
        self.assertEqual({pattern.id for pattern in description.patterns},
                         {pattern for family in description.families for pattern in family.patterns})

    def test_description_context_rejects_wrong_provider_and_patterns(self):
        description = interface.describe_compiler()
        context = ProposalContext(*(["a" * 64] * 5), "atlas", interface.implementation_sha256(),
                                  description.patterns, "accelerator", "external", ProposalBudget())
        description.validate_context(context)
        for changed in (replace(context, provider="another"),
                        replace(context, patterns=(replace(context.patterns[0], id="forged"),))):
            with self.assertRaises(ValueError):
                description.validate_context(changed)

    def test_declared_routes_have_valid_recipes(self):
        for kind in ("activation_quantize_device", "already_quantized", "layout_device", "layout",
                     "float_arithmetic", "host_ordered_relu"):
            with self.subTest(kind=kind):
                # float_arithmetic names several host families, so its claim names one.
                # float_arithmetic and already_quantized name several families, so their claims name one.
                family = {"float_arithmetic": (("numerical_family", "host.bias-add"),),
                          "already_quantized": (("numerical_family", "atlas.ordered-quantized-contraction"),)}.get(kind, ())
                recipe = interface.lowering_recipe(SimpleNamespace(metadata=(("kind", kind), *family)))
                interface.validate_recipe(recipe)
                self.assertEqual(dict(recipe.guards)["selected-kind"], kind)
                if kind in {"layout", "float_arithmetic", "host_ordered_relu"}:
                    self.assertEqual(recipe.artifacts, (("c-source", "wrapper"),))

    def test_unknown_and_missing_kind_refused(self):
        for metadata in ((), (("kind", "integer"),), (("kind", "unknown"),)):
            with self.subTest(metadata=metadata), self.assertRaisesRegex(ValueError, "no declared"):
                interface.lowering_recipe(SimpleNamespace(metadata=metadata))

    def test_stale_source_pin_and_forged_family_refused(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            roots = {}
            for name, filename in (("atlas", "application.py"), ("host", "float_arithmetic.py"),
                                   ("runtime", "atlas_host.c")):
                directory = root / name
                directory.mkdir()
                (directory / filename).write_text("original")
                roots[name] = (name, directory, (".py", ".c"))
            with patch.object(interface, "_source_roots", return_value=(roots["atlas"], roots["runtime"])), \
                    patch.object(host_target, "_source_roots", return_value=(roots["host"],)):
                device = interface.lowering_recipe(SimpleNamespace(metadata=(("kind", "layout_device"),)))
                host = interface.lowering_recipe(SimpleNamespace(metadata=(("kind", "float_arithmetic"),
                                                                           ("numerical_family", "host.bias-add"))))
                self.assertEqual(host.pipeline, host_target.PIPELINE)
                # An Atlas recipe pins Atlas sources and the host target's digest; a host recipe pins host sources.
                for name, stale in (("atlas", (device,)), ("runtime", (device,)), ("host", (device, host))):
                    source = next(roots[name][1].iterdir())
                    source.write_text("mutated")
                    for recipe in stale:
                        with self.assertRaisesRegex(ValueError, "sources changed"):
                            interface.validate_recipe(recipe)
                    source.write_text("original")
                for recipe in (device, host):
                    interface.validate_recipe(recipe)
                    forged = replace(recipe, guards=tuple((key, "atlas.fake" if key == "family" else value)
                                                          for key, value in recipe.guards))
                    with self.assertRaisesRegex(ValueError, "guards differ|differs from the declared host families"):
                        interface.validate_recipe(forged)
                # A host family cannot be claimed under the Atlas pipeline.
                with self.assertRaisesRegex(ValueError, "guards differ"):
                    interface.validate_recipe(replace(device, guards=host.guards, artifacts=host.artifacts))

    def test_export_claims_the_declared_source_subset_and_a_generated_cover_refuses_others(self):
        @dataclass(frozen=True)
        class Product:
            description: dict

        product = Product({"existing": "checked"})
        calls = []

        @interface.record_compiler_recipe
        def export(adapter, region, alternative):
            calls.append(region)
            return product

        def source(op):
            return SimpleNamespace(nodes=(SimpleNamespace(op=op, effect="pure", type=SimpleNamespace(dtype="bf16")),))

        alternative = SimpleNamespace(metadata=(("kind", "float_arithmetic"),))
        declared = export(None, source("bf16_add"), alternative)
        self.assertEqual(declared.description["existing"], "checked")
        # Publication serializes to JSON; replay must retain exactly the same record.
        import json
        self.assertEqual(json.loads(json.dumps(declared.description)), declared.description)
        # A host BF16 add is the host target's family, pinned by the host target's sources.
        self.assertEqual(declared.description["compiler_recipe"]["implementation_sha256"],
                         host_target.implementation_sha256())
        self.assertNotIn("compiler_recipe", product.description)
        from merlin.compiler.semantic.target_binding import NativeCompilationError
        generated = SimpleNamespace(_generated_family_claims=())
        with self.assertRaisesRegex(NativeCompilationError, "no implementation family for the float_arithmetic route over f64_mul"):
            export(generated, source("f64_mul"), alternative)
        unknown = SimpleNamespace(metadata=(("kind", "integer"),))
        with self.assertRaisesRegex(NativeCompilationError, "integer route over i32_add"):
            export(generated, source("i32_add"), unknown)
        # Temporary: a legacy hand-written policy still exports an undeclared route of a listed kind without a
        # claim. The list may only shrink; a kind outside it refuses on that path too.
        self.assertIs(export(None, source("f64_mul"), alternative), product)
        unlisted = SimpleNamespace(metadata=(("kind", "unlisted_route"),))
        with self.assertRaisesRegex(NativeCompilationError, "unlisted_route route over i32_add"):
            export(None, source("i32_add"), unlisted)
        self.assertEqual(len(calls), 5)
        self.assertLessEqual(interface.LEGACY_UNCLAIMED_KINDS, {
            "activation_quantize_device", "already_quantized", "contraction", "dynamic_quantize", "exact_float_device",
            "float_arithmetic", "float_cast", "guarded_softmax", "hierarchical_rmsnorm", "host_bf16_linear",
            "host_k1_contraction", "host_ordered_relu", "host_pointwise", "host_softmax", "integer", "layout",
            "layout_device", "masked_softmax", "mxu1_contraction", "pointwise", "pointwise_math", "predicate",
            "reduction", "rmsnorm", "rne_add", "scalar_broadcast", "softmax", "softmax_value", "static_quantize",
            "unary_math"})

        @interface.record_compiler_recipe
        def failed_export(*args):
            raise ValueError("original admission failed")

        with patch.object(interface, "lowering_recipe", side_effect=AssertionError("claim before checks")):
            with self.assertRaisesRegex(ValueError, "original admission failed"):
                failed_export(None, source("bf16_add"), alternative)

    def test_hash_is_stable_and_ignores_symlinks(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source.py"
            source.write_text("source")
            with patch.object(interface, "_source_roots", return_value=(("test", root, (".py",)),)), \
                    patch.object(host_target, "implementation_sha256", return_value="0" * 64):
                original = interface.implementation_sha256()
                self.assertEqual(original, interface.implementation_sha256())
                link = MagicMock(spec=Path)
                link.__str__.return_value = "link.py"
                link.__lt__.side_effect = lambda other: str(link) < str(other)
                link.__gt__.side_effect = lambda other: str(link) > str(other)
                link.is_symlink.return_value = True
                link.is_file.side_effect = AssertionError("must skip symlink before filesystem inspection")
                with patch.object(Path, "iterdir", return_value=[source, link]):
                    self.assertEqual(original, interface.implementation_sha256())
                added = root / "added.py"
                added.write_text("new")
                self.assertNotEqual(original, interface.implementation_sha256())


if __name__ == "__main__":
    unittest.main()
