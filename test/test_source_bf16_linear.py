"""Explicit source f32 accumulation admission without rewriting source types."""

from dataclasses import replace
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

from atlas_mlir_native.already_quantized import POLICY as INTAKE_POLICY
from atlas_mlir_native.application import AtlasApplicationAdapter
from atlas_mlir_native.application_precision import CONTRACTION_POLICY
from atlas_mlir_native.binding import TARGET_IDENTITY
from atlas_mlir_native.row_softmax import SOFTMAX_POLICY
from atlas_mlir_native.source_bf16_linear import POLICY, recognize_source_bf16_linear
from merlin.compile.semantic_regions import extract_semantic_region
from merlin.semantic_compiler.model import (
    ConstantBinding, IndexMap, KernelRequest, SemanticNode, TensorType, pointwise_identity_maps,
)


def source_request(m=2, n=5, k=3):
    def type_(shape, dtype):
        return TensorType(shape, dtype, "raw-bit-pattern")

    lhs = SemanticNode("a", "input", (), type_((m, k), "bf16"), effect="input")
    # Exact stored RHS bytes include both zero signs. They are never recoded.
    rhs = SemanticNode("w", "constant", (), type_((k, n), "bf16"), effect="constant")
    seed = SemanticNode("seed", "constant", (), type_((m, n), "f32"), effect="constant")
    widenings = tuple(SemanticNode(key, "bf16_to_f32", (value.id,), type_(value.type.shape, "f32"),
        (("numeric_semantics", "bf16-to-f32-exact-v1"),), index_maps=pointwise_identity_maps(2, 1))
        for key, value in (("aw", lhs), ("ww", rhs)))
    maps = tuple(IndexMap(3, rows, (0, 0)) for rows in (
        ((1, 0, 0), (0, 0, 1)), ((0, 0, 1), (0, 1, 0)),
        ((1, 0, 0), (0, 1, 0)), ((1, 0, 0), (0, 1, 0))))
    matmul = SemanticNode("mm", "source_f32_matmul_unqualified", ("aw", "ww", "seed"), seed.type,
        (("numeric_semantics", "source-f32-mul-add-unqualified-v1"),
         ("reduction_order", "unresolved"), ("scalar_add_order", "accumulator-product")), index_maps=maps)
    output = SemanticNode("y", "f32_to_bf16", ("mm",), type_((m, n), "bf16"),
        (("numeric_semantics", "f32-to-bf16-rne-v1"),), index_maps=pointwise_identity_maps(2, 1))
    rhs_bytes = (b"\x00\x80" + b"\x80\x3f" * (k * n - 1)).hex()
    return KernelRequest((lhs, rhs, seed, *widenings, matmul, output), ("y",), ("external",),
        (("a", "external"),), TARGET_IDENTITY, "diagnostic", constants=(
            ConstantBinding("w", "bf16-le-raw", rhs_bytes),
            ConstantBinding("seed", "f32-le-raw", bytes(4 * m * n).hex())))


def adapter(selected=POLICY):
    policy = dict(schema="atlas.diagnostic_application_precision.v1", source_sha256="0" * 64,
        target_identity=TARGET_IDENTITY, contraction=CONTRACTION_POLICY, row_softmax=SOFTMAX_POLICY,
        host_support=["transpose", "reshape", "copy"], already_quantized=INTAKE_POLICY)
    if selected is not None:
        policy["host_bf16_linear"] = selected
    return AtlasApplicationAdapter(policy, original_source_sha256="0" * 64)


def changed(source, key, **fields):
    return replace(source, nodes=tuple(replace(node, **fields) if node.id == key else node for node in source.nodes))


class SourceBF16LinearTest(unittest.TestCase):
    def test_full_host_application_replays_and_executes_with_default_limits(self):
        compiler = shutil.which("cc")
        if not compiler or not os.environ.get("MERLIN_NATIVE_SNAPSHOT"):
            self.skipTest("requires C compiler and MERLIN_NATIVE_SNAPSHOT")
        from merlin.compile.semantic_application import compile_semantic_application, verify_semantic_application
        from merlin.semantic_compiler.search import SearchLimits
        from merlin.semantic_compiler.snapshot import open_native_snapshot

        source = source_request()
        original = source.record()
        selected_adapter = adapter()
        snapshot = open_native_snapshot(Path(os.environ["MERLIN_NATIVE_SNAPSHOT"]))
        # Keep the saved-capsule example's default one MiB budget, the default
        # search limits and the ordinary semantic arena allocator.
        capacity = 1 << 20
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            artifact = root / "application"
            manifest = compile_semantic_application(source, adapter=selected_adapter, snapshot=snapshot,
                target_source=Path(__file__).parent.parent, destination=artifact, limits=SearchLimits(),
                name="tiny_head", capacity=capacity, allow_host_only=True)
            self.assertEqual(manifest["artifact_kind"], "host_application")
            self.assertEqual(manifest["application_memory_bytes"], capacity)
            self.assertFalse(manifest["segments"])
            self.assertEqual(len(manifest["regions"]), 1)
            compiled = manifest["regions"][0]
            self.assertEqual(compiled["inputs"], ["a", "w", "seed"])
            self.assertEqual(compiled["outputs"], ["y"])
            self.assertEqual(compiled["description"]["source_to_replacement_receipt"]["selected_policy"], POLICY)
            interface = manifest["application_interface"]
            self.assertEqual(interface["inputs"], ["a"])
            self.assertEqual(interface["outputs"], ["y"])
            self.assertLessEqual(interface["workspace_bytes"], capacity)
            self.assertEqual(json.loads((artifact / "source-request.json").read_text()), original)
            replay = verify_semantic_application(source, adapter=selected_adapter, snapshot=snapshot,
                target_source=Path(__file__).parent.parent, artifact=artifact, allow_host_only=True)
            self.assertEqual(replay["status"], "checked")
            self.assertEqual(replay["source_wiring"], "checked")
            package = artifact / "c-package"
            (package / "check.c").write_text(r'''
#include "combined/tiny_head_combined.h"
#include <assert.h>
#include <stdint.h>
#include <string.h>
int main(void) {
  _Alignas(TINY_HEAD_WORKSPACE_ALIGNMENT) unsigned char workspace[TINY_HEAD_WORKSPACE_BYTES];
  tiny_head_driver_state state={0};
  void *context_items[]={0}; tiny_head_driver_contexts contexts={context_items,1};
  uint16_t input[6]={0x3f80,0x4000,0x4040,0x4080,0x40a0,0x40c0};
  uint16_t original[6],output[10];
  const uint16_t expected[10]={0x40a0,0x40c0,0x40c0,0x40c0,0x40c0,
                               0x4130,0x4170,0x4170,0x4170,0x4170};
  const void *inputs[]={input}; void *outputs[]={output};
  memcpy(original,input,sizeof(input));memset(output,0xa5,sizeof(output));
  assert(tiny_head_invoke(&state,workspace,sizeof(workspace),inputs,outputs,&contexts)==0);
  assert(!memcmp(output,expected,sizeof(output)));
  assert(!memcmp(input,original,sizeof(input)));
  input[0]=0x7fc0;memset(output,0xa5,sizeof(output));
  assert(tiny_head_invoke(&state,workspace,sizeof(workspace),inputs,outputs,&contexts)==TINY_HEAD_DRIVER_KERNEL_ERROR);
  for (unsigned i=0;i<sizeof(output);++i) assert(((const unsigned char *)output)[i]==0xa5);
  return 0;
}
''')
            subprocess.run(["nice", "-n", "10", compiler, "-std=c11", "-O2", "-Wall", "-Wextra", "-Werror",
                "-frounding-math", "-ffp-contract=off", *manifest["link_sources"], "check.c", "-lm", "-o", "check"],
                cwd=package, check=True, capture_output=True)
            result = subprocess.run([str(package / "check")], capture_output=True)
            self.assertEqual(result.returncode, 0, result.stderr.decode())
        self.assertEqual(source.record(), original)

    def test_translated_identity_extf_named_matmul_and_rne_truncf(self):
        from merlin.semantic_compiler.linalg_bridge import translate_linalg_text

        text = (Path(__file__).parent / "examples/source_bf16_f32acc.mlir").read_text()
        source = translate_linalg_text(text, entry="forward", target_identity=TARGET_IDENTITY,
                                       lowering_policy="diagnostic").request
        before = source.record()
        found = recognize_source_bf16_linear(source, source.outputs[0])
        self.assertEqual([source.node(key).op for key in found.computations],
                         ["bf16_to_f32", "bf16_to_f32", "source_f32_matmul_unqualified", "f32_to_bf16"])
        self.assertEqual(dict(source.node(found.matmul).attrs)["reduction_order"], "unresolved")
        self.assertEqual(source.node(found.seed).type.dtype, "f32")
        self.assertEqual(adapter().plan(source).selected[0].placement, "host")
        self.assertEqual(source.record(), before)

    def test_selects_complete_host_region_for_independent_shapes_and_k_tails(self):
        for m, n, k in ((1, 1, 1), (3, 7, 2), (2, 3, 31), (1, 5, 32), (4, 2, 33), (2, 7, 65)):
            with self.subTest(shape=(m, n, k)):
                source = source_request(m, n, k)
                before = source.record()
                selected = adapter().plan(source).selected
                self.assertEqual(len(selected), 1)
                self.assertEqual(selected[0].placement, "host")
                self.assertEqual(selected[0].nodes, (0, 1, 2, 3))
                self.assertEqual(tuple(value.buffer for value in selected[0].inputs), ("a", "w", "seed"))
                self.assertEqual(tuple(value.buffer for value in selected[0].outputs), ("y",))
                self.assertEqual(source.record(), before)

    def test_export_preserves_original_typed_source_and_receipts(self):
        source = source_request()
        selected_adapter = adapter()
        alternative = selected_adapter.plan(source).selected[0]
        found = recognize_source_bf16_linear(source, "y")
        region = extract_semantic_region(source, found.computations)
        compiled = selected_adapter.export_region(region, alternative, "head", Path("unused"), None)
        self.assertEqual(compiled.inputs, ("a", "w", "seed"))
        self.assertEqual(compiled.outputs, ("y",))
        receipt = compiled.description["source_to_replacement_receipt"]
        self.assertEqual(receipt["application_source_request_digest"], source.digest())
        self.assertEqual(receipt["source_region_request_digest"], region.digest())
        self.assertEqual(receipt["selected_policy"], POLICY)
        self.assertEqual(receipt["source_reduction_order"], "unresolved")
        self.assertEqual(receipt["selected_reduction_order"], "ascending K, left to right")
        self.assertFalse(receipt["source_types_changed"])
        self.assertFalse(receipt["weight_codes_or_exponents_changed"])
        self.assertEqual([row["type"]["dtype"] for row in receipt["source_computations"]],
                         ["f32", "f32", "f32", "bf16"])
        self.assertEqual(region.node("seed").type.dtype, "f32")
        self.assertEqual(region.constants, source.constants)
        self.assertEqual(source.node("w").type.dtype, "bf16")

    def test_requires_explicit_source_policy_and_preserves_unrelated_f32_refusal(self):
        from merlin.compile.c_bf16_linear import SOURCE_POLICY

        self.assertEqual(POLICY, SOURCE_POLICY)
        source = source_request()
        with self.assertRaisesRegex(Exception, "refuses floating contractions"):
            adapter(None).plan(source)
        historical_policy = "bf16-product-rne-tree32-f32-rne-bf16-rne-tiles-ltr-chop-v1"
        for selected in (historical_policy, "unknown", [], 17):
            with self.subTest(unknown_policy=selected), self.assertRaisesRegex(Exception, "host BF16 linear policy differs"):
                adapter(selected)
        lhs = SemanticNode("u_a", "input", (), TensorType((2, 3), "f32", "raw-bit-pattern"), effect="input")
        rhs = SemanticNode("u_b", "input", (), TensorType((3, 5), "f32", "raw-bit-pattern"), effect="input")
        unrelated = replace(source.node("mm"), id="unrelated", inputs=("u_a", "u_b", "seed"))
        source = replace(source, nodes=(*source.nodes, lhs, rhs, unrelated), outputs=("y", "unrelated"),
                         output_storages=("external", "external"),
                         input_storages=(*source.input_storages, ("u_a", "external"), ("u_b", "external")))
        self.assertEqual(recognize_source_bf16_linear(source, "y").matmul, "mm")
        with self.assertRaisesRegex(Exception, "refuses floating contractions"):
            adapter().plan(source)

    def test_rejects_negative_nonzero_and_input_seed(self):
        source = source_request()
        for bits in ("00000080", "0000803f"):
            seed = replace(source.constants[-1], data_hex=bits + source.constants[-1].data_hex[8:])
            altered = replace(source, constants=(source.constants[0], seed))
            with self.subTest(bits=bits), self.assertRaisesRegex(Exception, "refuses floating contractions"):
                adapter().plan(altered)
        altered = replace(source, nodes=tuple(replace(node, op="input", effect="input")
                          if node.id == "seed" else node for node in source.nodes),
                          constants=(source.constants[0],),
                          input_storages=(*source.input_storages, ("seed", "external")))
        with self.assertRaisesRegex(Exception, "refuses floating contractions"):
            adapter().plan(altered)

    def test_rejects_escaping_widenings_and_f32_result(self):
        source = source_request()
        for key in ("aw", "ww", "mm"):
            escaped = replace(source, outputs=("y", key), output_storages=("external", "external"))
            with self.subTest(output=key), self.assertRaisesRegex(Exception, "refuses floating contractions"):
                adapter().plan(escaped)
            extra = SemanticNode("extra", "copy", (key,), source.node(key).type,
                                 index_maps=pointwise_identity_maps(2, 1))
            escaped = replace(source, nodes=(*source.nodes, extra), outputs=("y", "extra"),
                              output_storages=("external", "external"))
            with self.subTest(consumer=key), self.assertRaisesRegex(Exception, "refuses floating contractions"):
                adapter().plan(escaped)

    def test_rejects_wrong_maps_types_rounding_and_unknown_arithmetic_attributes(self):
        source = source_request()
        bad_map = IndexMap(2, ((0, 1), (1, 0)), (0, 0))
        cases = [changed(source, "aw", index_maps=(bad_map, bad_map)),
                 changed(source, "y", index_maps=(bad_map, bad_map)),
                 changed(source, "y", attrs=(("numeric_semantics", "f32-to-bf16-chop-v1"),)),
                 changed(source, "aw", type=TensorType((2, 3), "f32", "unknown")),
                 changed(source, "mm", attrs=(*source.node("mm").attrs, ("fastmath", "reassoc"))),
                 changed(source, "aw", attrs=(*source.node("aw").attrs, ("extra", True))),
                 changed(source, "seed", attrs=(("extra", True),))]
        maps = list(source.node("mm").index_maps)
        maps[0] = IndexMap(3, ((0, 1, 0), (0, 0, 1)), (0, 0))
        cases.append(changed(source, "mm", index_maps=tuple(maps)))
        for index, altered in enumerate(cases):
            with self.subTest(case=index), self.assertRaisesRegex(Exception, "refuses floating contractions"):
                adapter().plan(altered)


if __name__ == "__main__":
    unittest.main()
