"""Explicit I/O geometry changes allocation without changing target arithmetic."""

import json
import os
import subprocess
import sys

import pytest
from atlas_mlir_native.binding import (
    DRAM_BASE,
    DRAM_SLOTS,
    TILE_BYTES,
    AtlasNativeBinding,
    check_io_memory_manifest,
    dram_slots,
)
from atlas_mlir_native.pointwise import plan_abi, prepare_plan, prepare_source
from merlin.semantic_compiler.codegen import CodegenOptions, codegen_options
from merlin.semantic_compiler.model import (
    KernelRequest,
    SemanticNode,
    TensorType,
    pointwise_identity_maps,
)
from merlin.semantic_compiler.target_binding import NativeCompilationError


def source(shape=(1, 15, 50, 163)):
    type_ = TensorType(shape, "f32", "raw-bit-pattern")
    x = SemanticNode("x", "input", (), type_, effect="input")
    y = SemanticNode("y", "input", (), type_, effect="input")
    out = SemanticNode("result", "f32_mul", (x.id, y.id), type_,
                       (("numeric_semantics", "ieee754-f32-mul-rne-v1"),),
                       index_maps=pointwise_identity_maps(len(shape), 2))
    return KernelRequest((x, y, out), (out.id,), ("external",), ((x.id, "external"), (y.id, "external")),
                         AtlasNativeBinding.profile().target_identity, "diagnostic")


def test_larger_io_window_is_explicit_and_only_changes_the_external_bank():
    baseline = AtlasNativeBinding.profile()
    assert dram_slots() == DRAM_SLOTS
    request = prepare_source(source()).request
    with pytest.raises(NativeCompilationError, match="memory window"):
        plan_abi(request)
    with codegen_options(CodegenOptions(io_memory_bytes=1048576)):
        expanded = AtlasNativeBinding.profile()
        assert dram_slots() == 512 and expanded.digest() != baseline.digest()
        assert expanded.target_identity == baseline.target_identity
        assert expanded.descriptors == baseline.descriptors
        assert expanded.banks[1:] == baseline.banks[1:]
        assert expanded.banks[0].capacity == 512
        abi, plan = plan_abi(request), prepare_plan(request)
        slots = [*abi["fixed_inputs"].values(), *abi["fixed_outputs"]]
        ranges = [(DRAM_BASE + slot * TILE_BYTES, DRAM_BASE + slot * TILE_BYTES + size)
                  for slot, size in zip(slots, (*plan.input_packed_bytes, plan.packed_bytes), strict=True)]
        assert all(DRAM_BASE <= a < b <= DRAM_BASE + 1048576 for a, b in ranges)
        assert any(b > DRAM_BASE + DRAM_SLOTS * TILE_BYTES for a, b in ranges)
        assert all(b <= c or d <= a for i, (a, b) in enumerate(ranges) for c, d in ranges[i+1:])
    assert dram_slots() == DRAM_SLOTS and AtlasNativeBinding.profile().digest() == baseline.digest()


@pytest.mark.parametrize("size", [1, TILE_BYTES - 1, TILE_BYTES + 1, (1 << 32) - DRAM_BASE + TILE_BYTES])
def test_window_alignment_and_current_host_address_abi_are_checked(size):
    with (codegen_options(CodegenOptions(io_memory_bytes=size)),
          pytest.raises(NativeCompilationError, match="32-bit buffer ABI")):
        AtlasNativeBinding.profile()
    assert dram_slots() == DRAM_SLOTS


def test_artifacts_require_their_original_io_configuration_even_for_small_buffers():
    check_io_memory_manifest({})  # Historical artifacts use the original window.
    with codegen_options(CodegenOptions(io_memory_bytes=1048576)):
        check_io_memory_manifest({"io_memory_bytes": 1048576})
        for manifest in ({}, {"io_memory_bytes": 524288}, {"io_memory_bytes": 1048576.0}):
            with pytest.raises(ValueError, match="configuration"):
                check_io_memory_manifest(manifest)
    with pytest.raises(ValueError, match="configuration"):
        check_io_memory_manifest({"io_memory_bytes": 1048576})


@pytest.mark.skipif(not os.environ.get("ATLAS_OOT_BIN_DIR"), reason="select native Atlas tools")
def test_cli_requires_matching_window_for_compilation_and_verification(tmp_path):
    from test_looped_mnk import ROOT, build_snapshot

    request = prepare_source(source((65, 513))).request
    (tmp_path / "request.json").write_text(json.dumps(request.record()))
    with codegen_options(CodegenOptions(io_memory_bytes=1048576)):
        snapshot = build_snapshot(tmp_path)
    common = [sys.executable, "-m", "merlin.targetgen.cli"]
    inputs = ["--engine", "merlin_native", "--support", "atlas", "--snapshot", str(snapshot.root),
              "--request", str(tmp_path / "request.json"), "--mode", "diagnostic"]
    artifact, status = tmp_path / "compiled", tmp_path / "verification.json"
    compile_cmd = [*common, "native-compile", *inputs, "--target-source", str(ROOT), "--abi", "auto",
                   "--out", str(artifact)]
    rejected = subprocess.run(compile_cmd, capture_output=True, text=True)
    assert rejected.returncode == 2 and "snapshot differs" in rejected.stdout
    assert not artifact.exists()
    compiled = subprocess.run([*compile_cmd, "--io-memory-bytes", "1048576"], capture_output=True, text=True)
    assert compiled.returncode == 0, compiled.stdout + compiled.stderr
    assert json.loads((artifact / "manifest.json").read_text())["io_memory_bytes"] == 1048576
    verify_cmd = [*common, "native-verify", *inputs, "--artifact", str(artifact), "--out", str(status)]
    rejected = subprocess.run(verify_cmd, capture_output=True, text=True)
    assert rejected.returncode == 2 and "snapshot differs" in status.read_text()
    verified = subprocess.run([*verify_cmd, "--io-memory-bytes", "1048576"], capture_output=True, text=True)
    assert verified.returncode == 0, verified.stdout + verified.stderr
