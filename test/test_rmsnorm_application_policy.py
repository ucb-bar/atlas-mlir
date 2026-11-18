"""Source-coupled normalization owns private arithmetic, preserving residuals."""

from dataclasses import replace
import unittest

from atlas_mlir_native.application import AtlasApplicationAdapter
from atlas_mlir_native.binding import TARGET_IDENTITY
from atlas_mlir_native.hierarchical_rmsnorm import POLICY
from merlin.compile.semantic_regions import extract_semantic_region
from merlin.semantic_compiler.rmsnorm_source import recognize_rmsnorm
from merlin.semantic_compiler.target_binding import NativeCompilationError
from merlin.tests.ir.test_rmsnorm_source import source_graph
from test_attention_application_policy import policy as attention_policy


SOURCE = "a" * 64


def coupled_source():
    """Two genuine norm expressions connected through an ordinary value edge."""
    first = source_graph()
    second = source_graph()
    rename = lambda key: "result" if key == "H" else "second_" + key
    nodes = (*first.nodes, *(replace(node, id=rename(node.id), inputs=tuple(map(rename, node.inputs)))
                            for node in second.nodes if node.id != "H"))
    return replace(first, nodes=nodes, outputs=("second_result", "residual", "second_residual"),
                   output_storages=("external",) * 3,
                   input_storages=(*first.input_storages, ("second_gamma", "external")),
                   constants=(*first.constants, *(replace(c, node_id=rename(c.node_id)) for c in second.constants)),
                   target_identity=TARGET_IDENTITY, source_identity=SOURCE)


def policy(source):
    rows = []
    for result, exponent in (("result", -6), ("second_result", -9)):
        norm = recognize_rmsnorm(source, result)
        cut = extract_semantic_region(source, norm.computations)
        rows.append(dict(source_output=result, source_request_digest=cut.digest(),
                         provenance_sha256="b" * 64, policy=POLICY, output_scale_exponent=exponent))
    return dict(schema="atlas.diagnostic_application_precision.v24", source_sha256=SOURCE,
                base_policy=attention_policy(), rmsnorm_schedules=rows)


class RMSNormApplicationPolicyTest(unittest.TestCase):
    def test_two_norms_keep_value_coupling_and_independent_residual_consumers(self):
        source = coupled_source()
        selected = policy(source)
        adapter = AtlasApplicationAdapter(selected, original_source_sha256=SOURCE)
        plan = adapter.plan(source)
        self.assertEqual([dict(region.metadata)["kind"] for region in plan.selected],
                         ["hierarchical_rmsnorm", "rne_add", "hierarchical_rmsnorm", "rne_add"])
        owned = [i for region in plan.selected for i in region.nodes]
        self.assertEqual(sorted(owned), list(range(len([n for n in source.nodes if n.effect == "pure"]))))
        for row in selected["rmsnorm_schedules"]:
            found = recognize_rmsnorm(source, row["source_output"])
            cut = extract_semantic_region(source, found.computations)
            module, kwargs = adapter._extra_precision("hierarchical_rmsnorm", cut)
            norm = module.prepare_source(cut, **kwargs)
            self.assertEqual(norm.output_scale_exponent, row["output_scale_exponent"])
            self.assertEqual(norm.region.hidden, "H" if row["source_output"] == "result" else "result")
        # The legacy base does not silently absorb or quantize a norm.
        with self.assertRaises(NativeCompilationError):
            AtlasApplicationAdapter(attention_policy(), original_source_sha256=SOURCE).plan(source)

    def test_incomplete_or_stale_region_binding_refuses(self):
        source = coupled_source()
        selected = policy(source)
        for rows in (selected["rmsnorm_schedules"][:1],
                     [row | {"source_request_digest": "0" * 64} for row in selected["rmsnorm_schedules"]],
                     [row | {"source_output": "missing"} for row in selected["rmsnorm_schedules"]]):
            with self.subTest(rows=rows), self.assertRaises(NativeCompilationError):
                AtlasApplicationAdapter(selected | {"rmsnorm_schedules": rows}, original_source_sha256=SOURCE).plan(source)
        changed = replace(source, outputs=(*source.outputs, "inverse"), output_storages=(*source.output_storages, "external"))
        with self.assertRaisesRegex(NativeCompilationError, "exact private source regions"):
            AtlasApplicationAdapter(selected, original_source_sha256=SOURCE).plan(changed)

    def test_unknown_policy_and_invalid_output_boundary_refuse(self):
        selected = policy(coupled_source())
        bad = [selected | {"extra": True}, selected | {"source_sha256": "c" * 64},
               selected | {"rmsnorm_schedules": []}, selected | {"base_policy": None}]
        for change in ({"policy": "unknown"}, {"provenance_sha256": "bad"},
                       {"output_scale_exponent": True}, {"output_scale_exponent": -33}, {"extra": 0}):
            bad.append(selected | {"rmsnorm_schedules": [selected["rmsnorm_schedules"][0] | change]})
        for invalid in bad:
            with self.subTest(policy=invalid), self.assertRaises(NativeCompilationError):
                AtlasApplicationAdapter(invalid, original_source_sha256=SOURCE)


if __name__ == "__main__":
    unittest.main()
