"""Real canonical compilation and original-publication rebinding gates."""

from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
import os
import time
import json

import pytest

from atlas_mlir_native.options import codegen_options, current_codegen_options
from atlas_mlir_native.application import AtlasApplicationAdapter
from atlas_mlir_native import exact_float_application as application
from atlas_mlir_native import exact_float_device as device
from atlas_mlir_native.application_materialization import bind_exact_float

from merlin.compiler.build.materialization import canonical_request
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
    with codegen_options(replace(current_codegen_options(), io_memory_bytes=67108864)):
        fresh = adapter.compile_region(None, original, alternative, "region", tmp_path / "fresh", ROOT, None)
        materialized = adapter.materialize_region(None, canonical, alternative, "canonical", tmp_path / "private", ROOT, None)
        rebound = adapter.bind_materialization(materialized, original, alternative, "region", tmp_path / "cached", ROOT)
    assert members(tmp_path / "fresh") == members(tmp_path / "cached")
    assert fresh == rebound
    assert not any("canonical" in name for name in members(tmp_path / "cached"))


@pytest.mark.skipif(not (BIN / "atlas-opt").is_file(), reason="select built Atlas tools")
def test_rebinding_refuses_other_source_program_and_damaged_materialization(tmp_path):
    original = request("bf16_neg", 3)
    canonical = canonical_request(original, source_identity_policy="provenance")
    application.compile_source(None, canonical, target_source=ROOT, destination=tmp_path / "canonical", limits=None)
    with pytest.raises(ValueError, match="differs|resource envelope"):
        bind_exact_float(request("bf16_to_f32", 3), canonical, tmp_path / "canonical", tmp_path / "wrong")
    path = tmp_path / "canonical/segments/body_000/program.bin"
    path.write_bytes(path.read_bytes() + b"damage")
    with pytest.raises(ValueError):
        bind_exact_float(original, canonical, tmp_path / "canonical", tmp_path / "damaged")


def test_provider_advertises_capability_without_enabling_reuse():
    assert AtlasApplicationAdapter.supports_materialization_cache is True
    assert not hasattr(AtlasApplicationAdapter, "materialization_cache_supported")


def test_materialization_identity_uses_effective_inputs_and_receipt_mode(tmp_path, monkeypatch, select_codegen_options):
    class Choice:
        def to_dict(self):
            return dict(id="choice", nodes=[], inputs=[], outputs=[], resident_inputs=[], resident_outputs=[])

    adapter = AtlasApplicationAdapter.__new__(AtlasApplicationAdapter)
    tools = tmp_path / "tools"
    tools.mkdir()
    for name in ("atlas-opt", "atlas-emit"):
        (tools / name).write_bytes(name.encode())
    monkeypatch.setenv("ATLAS_OOT_BIN_DIR", str(tools))
    select_codegen_options(native_pipeline="mandatory")
    select_codegen_options(capture_postspill=False)
    source, choice = request("bf16_neg", 3), Choice()
    original = adapter.materialization_identity(source, choice)
    # Invocation-local pending bindings are runtime publication state, not a
    # source policy. They may include non-JSON image objects after any export.
    adapter._resident_loop_packages = {"pending": object()}
    assert adapter.materialization_identity(source, choice) == original
    json.dumps(adapter.materialization_identity(source, choice), allow_nan=False)
    adapter._resident_loop_packages = {}
    assert adapter.materialization_identity(source, choice) == original
    assert set(original["compilation_identity"]) == {
        "schema", "policy", "application_sha256", "binding_sha256", "materialization_sources",
        "tools", "pipeline", "capture_postspill", "deterministic_receipts",
    }
    monkeypatch.setenv("UNRELATED_COMPILER_TEST", "changed")
    working = tmp_path / "working"
    working.mkdir()
    monkeypatch.chdir(working)
    assert adapter.materialization_identity(source, choice) == original
    select_codegen_options(quantized_operand_reuse="row-cache")
    changed = adapter.materialization_identity(source, choice)
    assert changed != original
    assert changed["compilation_identity"]["quantized_operand_reuse"] == "row-cache"
    select_codegen_options(quantized_operand_reuse="ordered")
    assert adapter.materialization_identity(source, choice) == original
    select_codegen_options(exact_binary_vector_preflight=True)
    changed = adapter.materialization_identity(source, choice)
    assert changed != original
    assert changed["compilation_identity"]["exact_binary_vector_preflight"] is True
    select_codegen_options(exact_binary_vector_preflight=False)
    assert adapter.materialization_identity(source, choice) == original
    select_codegen_options(exact_cast_vector_preflight=True)
    changed = adapter.materialization_identity(source, choice)
    assert changed != original
    assert changed["compilation_identity"]["exact_cast_vector_preflight"] is True
    select_codegen_options(exact_cast_vector_preflight=False)
    assert adapter.materialization_identity(source, choice) == original
    select_codegen_options(staging_storage="vmem")
    changed = adapter.materialization_identity(source, choice)
    assert changed != original
    assert changed["compilation_identity"]["staging_storage"] == "vmem"
    select_codegen_options(staging_storage="vmem-inputs")
    inputs_cached = adapter.materialization_identity(source, choice)
    assert inputs_cached != changed and inputs_cached != original
    assert inputs_cached["compilation_identity"]["staging_storage"] == "vmem-inputs"
    select_codegen_options(staging_storage="dram")
    assert adapter.materialization_identity(source, choice) == original
    with adapter.publication_options(deterministic_receipts=True):
        changed = adapter.materialization_identity(source, choice)
        assert changed != original
        assert changed["compilation_identity"]["deterministic_receipts"] is True
    assert adapter.materialization_identity(source, choice) == original
    select_codegen_options(native_pipeline="optimized")
    assert adapter.materialization_identity(source, choice) != original
    select_codegen_options(native_pipeline="mandatory")
    select_codegen_options(capture_postspill=True)
    assert adapter.materialization_identity(source, choice) != original


def test_unsupported_kinds_decline_materialization():
    for kind in ("contraction", "host_k1_contraction", "vector_quantize", "guarded_softmax"):
        assert not AtlasApplicationAdapter.supports_region_materialization(None, SimpleNamespace(metadata=(("kind", kind),)))
