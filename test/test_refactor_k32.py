"""Regression against the unchanged integration K32 source and selected plan.

These tests check planning and host emission only. Device program equality is a
separate integration gate requiring the declared compute hosts and native tools.
"""
import json
from pathlib import Path
from dataclasses import replace
import tempfile
import unittest

from atlas_mlir_native.application import AtlasApplicationAdapter
from merlin.compiler.semantic.model import KernelRequest
from merlin.compiler.partition.semantic_regions import outline_semantic_regions

FIXTURE = Path(__file__).parent / "fixtures/refactor-k32"


def inputs():
    source = KernelRequest.from_record(json.loads((FIXTURE / "source-request.json").read_text()))
    # The historical fixture predates explicit numerical records. Derive a new
    # diagnostic declaration from its unchanged source attributes; keep the
    # historical file and plan intact. No source arithmetic equivalence claim.
    from atlas_mlir_native.source_translation import NUMERICS_ATTRIBUTE, contraction_numerics_attribute
    source = replace(source, nodes=tuple(
        replace(node, attrs=tuple(sorted({**dict(node.attrs), NUMERICS_ATTRIBUTE:
            contraction_numerics_attribute(node.op, dict(node.attrs)["output_dtype"], bias=len(node.inputs) == 5)}.items())))
        if node.op in {"atlas.qlinear", "atlas.qk_matmul", "atlas.pv_matmul"} else node
        for node in source.nodes))
    policy = json.loads((FIXTURE / "precision-policy.json").read_text())
    return source, AtlasApplicationAdapter(policy, original_source_sha256=policy["source_sha256"])


class RefactorK32Test(unittest.TestCase):
    def test_unchanged_source_and_policy_choose_same_plan(self):
        source, adapter = inputs()
        original = source.record()
        expected = json.loads((FIXTURE / "selected-plan.json").read_text())
        actual = adapter.plan(source).to_dict()
        self.assertEqual(actual, expected)
        self.assertEqual(source.record(), original)

    def test_bias_and_relu_stay_separate_host_regions(self):
        source, adapter = inputs()
        plan = adapter.plan(source)
        outlined = outline_semantic_regions(source, entry="capsule", plan=plan)
        selected = {item.id: item for item in plan.selected}
        with tempfile.TemporaryDirectory() as directory:
            for index, (key, region) in enumerate(outlined.regions):
                alternative = selected[key]
                kind = dict(alternative.metadata)["kind"]
                if kind not in {"layout", "float_arithmetic", "host_ordered_relu"}:
                    continue
                name = f"region_{index:03d}"
                product = adapter.compile_region(None, region, alternative, name, Path(directory), None, None)
                self.assertFalse(product.programs)
                self.assertEqual(product.outputs, region.outputs)
                for filename, data in product.files.items():
                    self.assertEqual(data, (FIXTURE / name / filename).read_bytes())


if __name__ == "__main__":
    unittest.main()
