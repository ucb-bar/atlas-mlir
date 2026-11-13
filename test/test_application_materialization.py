"""Real canonical compilation and original-publication rebinding gates."""

from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
import os
import time
import json

import pytest

from atlas_mlir_native.application import AtlasApplicationAdapter
from atlas_mlir_native import exact_float_application as application
from atlas_mlir_native import exact_float_device as device
from atlas_mlir_native.application_materialization import bind_exact_float
from merlin.semantic_compiler.codegen import CodegenOptions, codegen_options
from merlin.semantic_compiler.materialization import canonical_request, materialization_session
from test_exact_float_device import request

ROOT = Path(__file__).resolve().parents[1]
BIN = Path(os.environ.get("ATLAS_OOT_BIN_DIR", ROOT / "build/bin"))


def members(root):
    return {path.relative_to(root).as_posix(): path.read_bytes() for path in root.rglob("*") if path.is_file()}


def renamed(source):
    names = {node.id: "original_" + node.id for node in source.nodes}
    return replace(source, nodes=tuple(replace(node, id=names[node.id], inputs=tuple(names[name] for name in node.inputs))
                                       for node in source.nodes),
                   outputs=tuple(names[name] for name in source.outputs),
                   input_storages=tuple((names[name], storage) for name, storage in source.input_storages),
                   source_identity="original-source-provenance")


@pytest.mark.skipif(not (BIN / "atlas-opt").is_file(), reason="select built Atlas tools")
@pytest.mark.parametrize("op", sorted(device.OPS))
def test_rebind_every_float_kind_matches_all_fresh_publication_bytes(tmp_path, op):
    # Full plus tail bodies exercise nested digest-bound schedules/manifests.
    original = renamed(request(op, device.MAX_LEAF_ELEMENTS[op] + 3))
    canonical = canonical_request(original, source_identity_policy="provenance")
    adapter = AtlasApplicationAdapter.__new__(AtlasApplicationAdapter)
    alternative = SimpleNamespace(metadata=(("kind", "exact_float_device"),))
    with codegen_options(CodegenOptions(io_memory_bytes=67108864)):
        with materialization_session(enabled=False):
            fresh = adapter.compile_region(None, original, alternative, "region", tmp_path / "fresh", ROOT, None)
        with materialization_session(enabled=True):
            materialized = adapter.materialize_region(None, canonical, alternative, "canonical", tmp_path / "private", ROOT, None)
            rebound = adapter.bind_materialization(materialized, original, alternative, "region", tmp_path / "cached", ROOT)
    assert members(tmp_path / "fresh") == members(tmp_path / "cached")
    assert fresh == rebound
    assert not any("canonical" in name for name in members(tmp_path / "cached"))


@pytest.mark.skipif(not (BIN / "atlas-opt").is_file(), reason="select built Atlas tools")
def test_rebinding_refuses_other_source_program_and_damaged_materialization(tmp_path):
    original = request("bf16_neg", 3)
    canonical = canonical_request(original, source_identity_policy="provenance")
    with materialization_session(enabled=True):
        application.compile_source(None, canonical, target_source=ROOT, destination=tmp_path / "canonical", limits=None)
        with pytest.raises(ValueError, match="differs"):
            bind_exact_float(request("bf16_to_f32", 3), canonical, tmp_path / "canonical", tmp_path / "wrong")
        path = tmp_path / "canonical/segments/body_000/program.bin"
        path.write_bytes(path.read_bytes() + b"damage")
        with pytest.raises(ValueError):
            bind_exact_float(original, canonical, tmp_path / "canonical", tmp_path / "damaged")


def test_provider_advertises_capability_without_enabling_reuse():
    assert AtlasApplicationAdapter.supports_materialization_cache is True
    assert not hasattr(AtlasApplicationAdapter, "materialization_cache_supported")


def test_materialization_identity_uses_effective_inputs_and_receipt_mode(tmp_path, monkeypatch):
    class Choice:
        def to_dict(self):
            return dict(id="choice", nodes=[], inputs=[], outputs=[], resident_inputs=[], resident_outputs=[])

    adapter = AtlasApplicationAdapter.__new__(AtlasApplicationAdapter)
    tools = tmp_path / "tools"
    tools.mkdir()
    for name in ("atlas-opt", "atlas-emit"):
        (tools / name).write_bytes(name.encode())
    monkeypatch.setenv("ATLAS_OOT_BIN_DIR", str(tools))
    monkeypatch.setenv("ATLAS_NATIVE_PIPELINE", "mandatory")
    monkeypatch.delenv("ATLAS_CAPTURE_POSTSPILL", raising=False)
    source, choice = request("bf16_neg", 3), Choice()
    original = adapter.materialization_identity(source, choice)
    monkeypatch.setenv("UNRELATED_COMPILER_TEST", "changed")
    working = tmp_path / "working"
    working.mkdir()
    monkeypatch.chdir(working)
    assert adapter.materialization_identity(source, choice) == original
    monkeypatch.setenv("ATLAS_QUANTIZED_OPERAND_REUSE", "row-cache")
    changed = adapter.materialization_identity(source, choice)
    assert changed != original
    assert changed["compilation_identity"]["quantized_operand_reuse"] == "row-cache"
    monkeypatch.setenv("ATLAS_QUANTIZED_OPERAND_REUSE", "ordered")
    assert adapter.materialization_identity(source, choice) == original
    monkeypatch.setenv("ATLAS_EXACT_BINARY_VECTOR_PREFLIGHT", "1")
    changed = adapter.materialization_identity(source, choice)
    assert changed != original
    assert changed["compilation_identity"]["exact_binary_vector_preflight"] is True
    monkeypatch.setenv("ATLAS_EXACT_BINARY_VECTOR_PREFLIGHT", "0")
    assert adapter.materialization_identity(source, choice) == original
    monkeypatch.setenv("ATLAS_EXACT_CAST_VECTOR_PREFLIGHT", "1")
    changed = adapter.materialization_identity(source, choice)
    assert changed != original
    assert changed["compilation_identity"]["exact_cast_vector_preflight"] is True
    monkeypatch.setenv("ATLAS_EXACT_CAST_VECTOR_PREFLIGHT", "0")
    assert adapter.materialization_identity(source, choice) == original
    monkeypatch.setenv("ATLAS_STAGING_STORAGE", "vmem")
    changed = adapter.materialization_identity(source, choice)
    assert changed != original
    assert changed["compilation_identity"]["staging_storage"] == "vmem"
    monkeypatch.setenv("ATLAS_STAGING_STORAGE", "dram")
    assert adapter.materialization_identity(source, choice) == original
    with adapter.publication_options(deterministic_receipts=True):
        changed = adapter.materialization_identity(source, choice)
        assert changed != original
        assert changed["compilation_identity"]["deterministic_receipts"] is True
    assert adapter.materialization_identity(source, choice) == original
    monkeypatch.setenv("ATLAS_NATIVE_PIPELINE", "optimized")
    assert adapter.materialization_identity(source, choice) != original
    monkeypatch.setenv("ATLAS_NATIVE_PIPELINE", "mandatory")
    monkeypatch.setenv("ATLAS_CAPTURE_POSTSPILL", "1")
    assert adapter.materialization_identity(source, choice) != original


def test_unsupported_kinds_decline_materialization():
    for kind in ("contraction", "host_k1_contraction", "vector_quantize", "guarded_softmax"):
        assert not AtlasApplicationAdapter.supports_region_materialization(None, SimpleNamespace(metadata=(("kind", kind),)))


@pytest.mark.skipif(not (BIN / "atlas-opt").is_file(), reason="select built Atlas tools")
def test_generic_application_real_compile_count_wall_time_and_full_bytes(tmp_path):
    from merlin.compile.semantic_application import compile_semantic_application
    from merlin.tests.ir.test_semantic_regions import plan_for

    calls = []

    class Adapter(AtlasApplicationAdapter):
        def plan(self, source):
            plan = plan_for(source, ((0,), (1,), (2,), (3,)))
            return replace(plan, selected=tuple(replace(row, implementation="exact_float_device", placement="accelerator",
                                                       metadata=(("kind", "exact_float_device"),))
                                                  for row in plan.selected))

        def compile_region(self, *args, **kwargs):
            calls.append(1)
            return super().compile_region(*args, **kwargs)

    source = request("bf16_neg", 3)
    pure = source.nodes[-1]
    nodes = list(source.nodes)
    for index in range(3):
        pure = replace(pure, id=f"neg_{index}", inputs=(pure.id,))
        nodes.append(pure)
    source = replace(source, nodes=tuple(nodes), outputs=(pure.id,))
    snapshot_root = tmp_path / "snapshot"
    snapshot_root.mkdir()
    (snapshot_root / "manifest.json").write_text("{}")
    results, metrics = [], []
    for enabled in (False, True):
        calls.clear()
        stats = {}
        destination = tmp_path / str(enabled)
        started = time.monotonic()
        compile_semantic_application(
            source, adapter=Adapter.__new__(Adapter), snapshot=SimpleNamespace(root=snapshot_root),
            target_source=ROOT, destination=destination, limits=None, name="test_application", capacity=65536,
            reuse_region_materializations=enabled, materialization_stats=stats)
        elapsed = time.monotonic() - started
        assert len(calls) == stats["region_compiles"] == (1 if enabled else 4)
        assert stats["region_calls"] == stats["region_replays"] == 4
        assert stats["region_hits"] == (3 if enabled else 0)
        results.append(members(destination))
        metrics.append(dict(enabled=enabled, wall_seconds=elapsed, **stats))
    assert results[0] == results[1]
    print("real_region_materialization_gate=" + json.dumps(metrics, sort_keys=True))
