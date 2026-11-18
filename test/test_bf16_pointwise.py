"""BF16 source multiplication keeps RNE and requires a checked input domain."""

import os
from dataclasses import replace

import pytest
import test_application_float_arithmetic as fixtures
from atlas_mlir_native.application import AtlasApplicationAdapter
from atlas_mlir_native.binding import TARGET_IDENTITY
from atlas_mlir_native.pointwise import (
    BF16_POINTWISE_POLICY,
    prepare_plan,
    prepare_source,
)
from merlin.semantic_compiler.model import (
    KernelRequest,
    SemanticNode,
    TensorType,
    pointwise_identity_maps,
)
from merlin.semantic_compiler.target_binding import NativeCompilationError
from merlin.tests.runtime.test_unary_math_c import policy as math_policy


def request(shape=(3, 7), *, shared=False):
    type_ = TensorType(shape, "bf16", "raw-bit-pattern")
    a = SemanticNode("a", "input", (), type_, effect="input")
    b = replace(a, id="b")
    result = SemanticNode("result", "bf16_mul", (a.id, a.id if shared else b.id), type_,
                          attrs=(("numeric_semantics", "ieee754-bf16-mul-rne-v1"),),
                          index_maps=pointwise_identity_maps(len(shape), 2))
    inputs = (a,) if shared else (a, b)
    return KernelRequest((*inputs, result), (result.id,), ("external",),
                         tuple((node.id, "external") for node in inputs), TARGET_IDENTITY, "diagnostic")


@pytest.mark.parametrize("shape,shared", [((3, 7), False), ((33, 35), False), ((2, 3, 35), True)])
def test_checked_bf16_region_needs_no_f32_conversion(shape, shared):
    source = request(shape, shared=shared)
    with pytest.raises(ValueError, match="separately selected"):
        prepare_source(source)
    precision = prepare_source(source, bf16_policy=BF16_POINTWISE_POLICY)
    assert precision.request.nodes[-1].op == "mul_bf16_zero"
    assert dict(precision.request.nodes[-1].attrs)["numeric_semantics"] == "bf16-product-rne-v1"
    assert precision.record()["output_conversion"] == "bf16-bit-preserving"
    assert precision.record()["qualification"].startswith("conditional_source_equivalence")
    plan = prepare_plan(precision.request)
    assert plan.logical_shape == shape
    assert len(plan.body_request().nodes) == (2 if shared else 3)


def test_bf16_application_admission_is_explicit_and_preserves_old_policies():
    old = {**fixtures.policy(), "schema": "atlas.diagnostic_application_precision.v20",
           "host_unary_math": math_policy()}
    with pytest.raises(NativeCompilationError, match="no compiled application route"):
        AtlasApplicationAdapter(old, original_source_sha256=old["source_sha256"]).plan(request())
    selected = {**old, "schema": "atlas.diagnostic_application_precision.v21",
                "bf16_pointwise": BF16_POINTWISE_POLICY}
    adapter = AtlasApplicationAdapter(selected, original_source_sha256=selected["source_sha256"])
    (region,) = adapter.plan(request()).selected
    assert region.placement == "accelerator" and dict(region.metadata)["kind"] == "pointwise"
    for policy in ({**old, "bf16_pointwise": BF16_POINTWISE_POLICY},
                   {**selected, "bf16_pointwise": "chop-or-ignore-rounding"}):
        with pytest.raises(NativeCompilationError, match="policy differs"):
            AtlasApplicationAdapter(policy, original_source_sha256=selected["source_sha256"])


def test_bf16_source_numerics_and_constant_domain_are_not_relaxed():
    from merlin.semantic_compiler.model import ConstantBinding

    source = request()
    output = replace(source.nodes[-1], attrs=(("numeric_semantics", "ieee754-bf16-mul-rtz-v1"),))
    with pytest.raises(ValueError, match="attributes or accounting"):
        prepare_source(replace(source, nodes=(*source.nodes[:-1], output)), bf16_policy=BF16_POINTWISE_POLICY)
    literal = replace(source.nodes[0], op="constant", effect="constant")
    source = replace(source, nodes=(literal, *source.nodes[1:]), input_storages=(("b", "external"),),
                     constants=(ConstantBinding("a", "bf16-le-raw", "817f" * 21),))
    with pytest.raises(ValueError, match="nonfinite"):
        prepare_source(source, bf16_policy=BF16_POINTWISE_POLICY)


@pytest.mark.parametrize("shared", [False, True])
def test_shared_binary_read_does_not_admit_distinct_value_or_result_overlap(shared):
    from atlas_mlir_native.binding import _check_bf16_selection
    from merlin.semantic_compiler.allocate import CandidateGraph, Value

    ids = ("x",) if shared else ("x", "y")
    count = len(ids)
    values = [Value(i, key, "external", 1, (), key, "input") for i, key in enumerate(ids)]
    values += [Value(count + i, "load", "tensor_bf16", 2, (i,), key, "instruction")
               for i, key in enumerate(ids)]
    product, stored = 2 * count, 2 * count + 1
    values += [Value(product, "product", "tensor_bf16", 2, (count, 2 * count - 1), "out", "instruction"),
               Value(stored, "store", "external", 1, (product,), "out", "instruction")]
    graph = CandidateGraph(tuple(values), (stored,))
    names = ([None] * count + ["atlas.load_bf16_tile"] * count
             + ["atlas.vpu_mul_bf16_zero_tile", "atlas.store_bf16_tile"])
    assignment = {**{i: i for i in range(count)}, **{count + i: 2 * i for i in range(count)},
                  product: 2 * count, stored: count}
    def check(bindings):
        return _check_bf16_selection(graph, tuple(range(count, stored + 1)), names, bindings,
                                    ids, tuple(range(count)), (count,))
    assert check(assignment)
    with pytest.raises(NativeCompilationError, match="pairs overlap"):
        check({**assignment, product: assignment[count]})
    if not shared:
        with pytest.raises(NativeCompilationError, match="pairs overlap"):
            check({**assignment, count + 1: assignment[count]})


@pytest.mark.skipif(not os.environ.get("ATLAS_OOT_BIN_DIR"), reason="select native Atlas tools")
def test_shared_bf16_producer_compiles_with_two_ordered_outputs(tmp_path):
    from merlin.compile.semantic_application import (
        compile_semantic_application,
        verify_semantic_application,
    )
    from merlin.semantic_compiler.search import SearchLimits
    from test_looped_mnk import ROOT, build_snapshot

    source = request()
    squared = replace(source.nodes[-1], id="squared", inputs=("result", "result"))
    source = replace(source, nodes=(source.nodes[1], source.nodes[0], source.nodes[-1], squared),
                     outputs=("result", "squared"),
                     output_storages=("external", "external"))
    policy = {**fixtures.policy(), "schema": "atlas.diagnostic_application_precision.v21",
              "host_unary_math": math_policy(), "bf16_pointwise": BF16_POINTWISE_POLICY}
    adapter = AtlasApplicationAdapter(policy, original_source_sha256=policy["source_sha256"])
    snapshot = build_snapshot(tmp_path)
    artifact = tmp_path / "application"
    manifest = compile_semantic_application(
        source, adapter=adapter, snapshot=snapshot, target_source=ROOT, destination=artifact,
        limits=SearchLimits(), name="bf16_chain", capacity=1 << 20,
    )
    assert len(manifest["segments"]) == 2
    result = verify_semantic_application(source, adapter=adapter, snapshot=snapshot,
                                         target_source=ROOT, artifact=artifact)
    assert result["status"] == "checked"


@pytest.mark.skipif(not os.environ.get("ATLAS_OOT_BIN_DIR"), reason="select native Atlas tools")
def test_exact_cast_broadcast_source_compiles_with_one_device_region(tmp_path):
    import shutil
    import subprocess

    from merlin.compile.semantic_application import (
        compile_semantic_application,
        verify_semantic_application,
    )
    from merlin.semantic_compiler.search import SearchLimits
    from merlin.semantic_compiler.source_preparation import prepare_source
    from merlin.tests.ir.test_affine_views import source as broadcast_source
    from test_looped_mnk import ROOT, build_snapshot

    policy = fixtures.policy()
    original = replace(broadcast_source(shape=(50, 720)), target_identity=policy["target_identity"],
                       lowering_policy="diagnostic")
    source, receipt = prepare_source(original)
    adapter = AtlasApplicationAdapter(policy, original_source_sha256=policy["source_sha256"])
    snapshot, artifact = build_snapshot(tmp_path), tmp_path / "application"
    manifest = compile_semantic_application(
        source, adapter=adapter, snapshot=snapshot, target_source=ROOT, destination=artifact,
        limits=SearchLimits(), name="broadcast_product", capacity=1 << 20, source_folding=receipt,
    )
    assert len(manifest["segments"]) == 1
    assert verify_semantic_application(source, adapter=adapter, snapshot=snapshot,
                                       target_source=ROOT, artifact=artifact,
                                       source_folding=receipt)["status"] == "checked"
    package = artifact / "c-package"
    assert len(manifest["link_sources"]) == 1
    subprocess.run([shutil.which("cc"), "-std=c11", "-O2", "-Wall", "-Wextra", "-Werror",
                    "-fno-fast-math", "-ffp-contract=off", "-c", str(package / manifest["link_sources"][0]),
                    "-o", str(tmp_path / "application.o")], check=True, capture_output=True)
