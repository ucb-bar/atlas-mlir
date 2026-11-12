"""Public transport conformance; these tests do not qualify Atlas execution."""
from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

pytest.importorskip("merlin.compiler.artifact")

from atlas_mlir_native import backend
from atlas_mlir_native.artifacts import publish_program

from merlin import api
from merlin.compiler.request import CompileOptions, Source
from merlin.compiler.semantic.model import KernelRequest, SemanticNode, TensorType


def invocation(tmp_path, monkeypatch, *, corrupt=False):
    profile = backend.AtlasNativeBinding.profile()
    tensor = TensorType((32, 32), "bf16", "opaque-bf16-bits")
    kernel = KernelRequest(
        nodes=(SemanticNode("x", "input", (), tensor, effect="input"),), outputs=("x",),
        output_storages=("external",), input_storages=(("x", "external"),),
        target_identity=profile.target_identity, lowering_policy="diagnostic",
    )
    monkeypatch.setattr(backend, "open_native_snapshot", lambda _: SimpleNamespace(profile=profile))

    def emitted(snapshot, request, *, destination, **kwargs):
        destination.mkdir()
        result = publish_program(
            destination, words=[0x01020304], virtual="virtual", machine="machine",
            program={"words": [0x01020304]}, plan={"output_addresses": [8192]}, selection={},
            metadata={"engine": "merlin_native", "artifact_kind": "program",
                      "qualification_scope": "diagnostic_only", "request_digest": request.digest(),
                      "target_identity": request.target_identity},
        )
        if corrupt:
            (destination / "program.bin").write_bytes(b"changed")
        return result

    monkeypatch.setattr(backend.AtlasNativeBinding, "compile", emitted)
    options = CompileOptions(parameters={
        "atlas.snapshot": str(tmp_path), "atlas.target-source": str(tmp_path),
        "atlas.fixed-inputs": {"x": 4096}, "atlas.fixed-outputs": [8192],
    })
    return Source(json.dumps(kernel.record()), kind="merlin.kernel-request"), options


def test_public_transport_preserves_binary_kind_and_diagnostic_scope(tmp_path, monkeypatch):
    source, options = invocation(tmp_path, monkeypatch)
    result = api.compile(source, backend=backend.AtlasBackend, options=options, output=tmp_path / "build")
    program = next(item for item in result.artifacts if item.role == "device-program")
    assert program.kind == "device-binary"
    assert program.path.read_bytes() == b"\x04\x03\x02\x01"
    assert result.deployment is None
    assert result.metadata["qualification_scope"] == "diagnostic_only"
    assert result.metadata["execution_observed"] is False
    assert api.inspect(result.directory)["backend"] == "atlas"


def test_public_transport_rechecks_native_publication(tmp_path, monkeypatch):
    source, options = invocation(tmp_path, monkeypatch, corrupt=True)
    with pytest.raises(ValueError, match="binary"):
        api.compile(source, backend=backend.AtlasBackend, options=options, output=tmp_path / "build")
    assert not (tmp_path / "build/compilation.json").exists()


def test_public_transport_rejects_image_options_before_emission(tmp_path, monkeypatch):
    source, options = invocation(tmp_path, monkeypatch)
    with pytest.raises(ValueError, match="host-image"):
        api.compile(source, backend=backend.AtlasBackend, output=tmp_path / "build",
                    options=CompileOptions(platform="zephyr", parameters=options.parameters))
    assert not (tmp_path / "build").exists()


def test_binding_stage_custody_publication_and_original_materialization_hook(tmp_path, monkeypatch):
    """Software transport fixture; native lowering and numerical qualification stay unavailable."""
    import shutil
    from dataclasses import replace
    from pathlib import Path

    from atlas_mlir_native import binding, native

    from merlin.compiler.semantic.allocation_types import (
        AllocationResult,
        CandidateGraph,
        Value,
    )
    from merlin.compiler.semantic.search import SearchLimits, SearchResult
    from merlin.compiler.semantic.target_binding import NativeCompilationError

    root = Path(binding.__file__).resolve().parents[2]
    request = KernelRequest.from_record(json.loads(
        (root / "test/examples/native_bf16_relu_request.json").read_text()
    ))
    graph = CandidateGraph((
        Value(0, "input", "external", 1, (), "x", "input"),
        Value(1, "load", "tensor_bf16", 2, (0,), "x", "instruction"),
        Value(2, "relu", "tensor_bf16", 2, (1,), "y", "instruction"),
        Value(3, "store", "external", 1, (2,), "y", "instruction"),
    ), (3,))
    symbols = {
        "input": {}, "load": {"descriptor": {"name": "atlas.load_bf16_tile"}},
        "relu": {"descriptor": {"name": "atlas.vpu_relu_bf16_tile"}},
        "store": {"descriptor": {"name": "atlas.store_bf16_tile"}},
    }
    selected = SearchResult(
        "selected", request.digest(), SimpleNamespace(digest=lambda: "fixture-candidate"),
        graph, AllocationResult("feasible", (1, 2, 3), {0: 3, 1: 32, 2: 34, 3: 35}),
        1, 1, 0, None, SimpleNamespace(symbols=symbols), check_fingerprint="fixture-transport",
    )
    snapshot_root = tmp_path / "snapshot"
    snapshot_root.mkdir()
    (snapshot_root / "manifest.json").write_text("{}")
    snapshot = SimpleNamespace(root=snapshot_root, profile=binding.AtlasNativeBinding.profile())
    decision_rows = []
    def select(requested, **kwargs):
        assert requested is request
        decision = kwargs["accept_materialization"](selected)
        decision_rows.append(decision)
        return selected if decision.accepted else replace(selected, status="unqualified_target")
    snapshot.select = select
    arguments = dict(fixed_inputs={"x": 3}, fixed_outputs=(35,), target_source=root, limits=SearchLimits())
    materialized = binding._MaterializedCandidate(
        [32, 33, 34, 35], "fixture virtual", "fixture machine", {"instructions": [{"word_u32": 0x73}]},
        [0x73], ("fixture-opt", "fixture-emit"), binding.DRAM_BASE + 3 * binding.TILE_BYTES,
        binding.DRAM_BASE + 35 * binding.TILE_BYTES,
    )
    with monkeypatch.context() as patch:
        patch.setattr(native, "prepare_candidate", lambda *args, **kwargs: materialized)
        manifest = binding.AtlasNativeBinding.compile(snapshot, request, destination=tmp_path / "compiled", **arguments)
    assert (tmp_path / "compiled/program.bin").read_bytes() == b"\x73\0\0\0"
    assert manifest["qualification_scope"] == "diagnostic_only"
    assert manifest["binding_implementation_sources"] == binding.implementation_sources()
    assert manifest["binding_source_sha256"] == manifest["binding_implementation_sources"]["binding.py"]
    assert {"native.py", "planning.py", "profiles.py", "publication.py", "virtual.py"} <= set(
        manifest["binding_implementation_sources"])
    assert not any("reference" in name or "evaluator" in name for name in manifest["binding_implementation_sources"])
    plan = json.loads((tmp_path / "compiled/execution_plan.json").read_text())
    assert plan["output_numerical_policies"] == ["bf16-sign-clamp-v1"]
    assert plan["runtime_domain_guard"] is False
    assert plan["boundary_domain_obligations"][0]["status"] == "caller_premise_unchecked"
    assert all(row.accepted for row in decision_rows)

    # Exercise the actual materializer through the original binding hook; no tool is executed.
    tools = tmp_path / "tools"
    tools.mkdir()
    for name in ("atlas-opt", "atlas-emit"):
        (tools / name).write_bytes(b"transport fixture: must never execute")
    attempts = []
    def refuse_virtual(*args, **kwargs):
        attempts.append(True)
        raise NativeCompilationError("unqualified_target", "fixture physical order declined")
    def select_refused(requested, **kwargs):
        for _ in range(2):
            assert kwargs["accept_materialization"](selected) == (False, "fixture physical order declined")
        return replace(selected, status="unqualified_target", reason="fixture physical order declined")
    snapshot.select = select_refused
    with monkeypatch.context() as patch:
        patch.setenv("ATLAS_OOT_BIN_DIR", str(tools))
        patch.setattr(binding, "_virtual_from_selected", refuse_virtual)
        patch.setattr(binding, "_run", lambda *args, **kwargs: pytest.fail("fixture tool executed"))
        with pytest.raises(NativeCompilationError, match="fixture physical order declined"):
            binding.AtlasNativeBinding.compile(snapshot, request, destination=tmp_path / "declined", **arguments)
    assert len(attempts) == 2 and not (tmp_path / "declined").exists()

    # Only temporary copied compiler sources change; reject drift before publishing a manifest.
    copied = tmp_path / "source"
    copied.mkdir()
    for name in binding.implementation_sources():
        shutil.copyfile(Path(binding.__file__).parent / name, copied / name)
    snapshot.select = select
    def changed_source(*args, **kwargs):
        path = copied / "native.py"
        path.write_bytes(path.read_bytes() + b"\n# changed fixture source\n")
        return materialized
    with monkeypatch.context() as patch:
        patch.setattr(binding, "__file__", str(copied / "binding.py"))
        patch.setattr(native, "prepare_candidate", changed_source)
        with pytest.raises(NativeCompilationError, match="implementation changed"):
            binding.AtlasNativeBinding.compile(snapshot, request, destination=tmp_path / "changed", **arguments)
    assert not (tmp_path / "changed").exists()



def _approve_relu_source(module, entry, options):
    """Original test caller approves meaning, never backend accuracy/hardware."""
    from merlin.compiler.ir.semantics import source_semantic_attributes
    from merlin.compiler.frontend.linalg_bf16_relu import recognize_bf16_sign_clamp_relu

    function = next(op for op in module.body.block.ops if op.sym_name.data == entry)
    relu = tuple(function.body.block.ops)[1]
    recognize_bf16_sign_clamp_relu(relu)
    if options != {"numeric_semantics": "bf16-sign-clamp-v1",
                   "input_domain": "finite-normal-or-zero-bf16"}:
        raise ValueError("original caller did not approve this numerical policy/domain")
    coverage = source_semantic_attributes(module, entry)
    if (len(coverage) != 2 or any(row.location != "operation" or row.key not in
            {"merlin.numeric_semantics", "merlin.input_domain"} for row in coverage)):
        raise ValueError("original caller did not approve extra semantic annotations")
    return coverage


def _system(provider, tmp_path, *, context=None, image=None):
    import yaml

    from merlin.compiler.components import Component
    from merlin.compiler.pipeline import RegionBackend
    from merlin.compiler.system import SystemCompiler
    from merlin.target.system.topology import select_topology

    topology = tmp_path / "topology.yaml"
    document = {
        "schema": "merlin.execution_topology.v1",
        "resources": [{"id": "device0", "kind": "accelerator", "features": [], "shared_group": "device0"}],
        "memory_domains": [{"id": "logical", "capacity_bytes": 1 << 20, "alignment_bytes": 64}],
        "access": [{"resource": "device0", "domain": "logical", "read": True, "write": True,
                    "addressability": "shared", "visibility": "coherent", "completion": "synchronous"}],
    }
    if context is not None:
        document["resources"].append({"id": context.resource, "kind": "cpu", "features": [],
                                      "shared_group": context.resource})
        document["memory_domains"].append({"id": context.memory_domain, "capacity_bytes": 65536,
                                           "alignment_bytes": 64})
        document["access"].append({"resource": context.resource, "domain": context.memory_domain,
            "read": True, "write": True, "addressability": "shared", "visibility": "coherent",
            "completion": "synchronous"})
    topology.write_text(yaml.safe_dump(document))
    def approve_implementation(source, proposal):
        # A declared software-planning test topology is not hardware evidence.
        # Recheck actual source policy/body independently of provider discovery.
        _approve_relu_source(source.module, source.entry,
                             {"numeric_semantics": "bf16-sign-clamp-v1",
                              "input_domain": "finite-normal-or-zero-bf16"})
        assert proposal.alternative.nodes == tuple(range(len(source.operations)))
        assert proposal.alternative.implementation == "atlas.bf16_relu.native.v1"
        assert proposal.lowering.artifacts[0] == ("device-binary", "device-program")
        assert dict(proposal.alternative.metadata)["input_domain"] == "caller_premise_unchecked"
        if context is not None:
            assert json.loads(dict(proposal.alternative.metadata)["c.context"]) == context.record()
            assert dict(proposal.alternative.metadata)["device_storage"] == (
                "original_platform_pre_reserved_outside_available_capacity")
    component = Component(RegionBackend(provider, "kernel"), "device0", "logical", approve_implementation,
                          required_resource_kind="accelerator")
    return SystemCompiler(topology=select_topology(topology), components=(component,), preferred_providers=("atlas",),
                          entry="relu_tile", image=image, validate_source=_approve_relu_source,
                          source_options={"numeric_semantics": "bf16-sign-clamp-v1",
                                          "input_domain": "finite-normal-or-zero-bf16"})


def _configured_transport(tmp_path, monkeypatch, *, plan_override=None):
    """Mock native transport only; real source/proposal/ticket/byte custody."""
    import hashlib
    import shutil
    from pathlib import Path

    from atlas_mlir_native import regions

    from merlin.compiler.semantic.search import SearchLimits
    from merlin.compiler.semantic.snapshot import NativeSnapshot

    root = Path(__file__).resolve().parents[1]
    target = tmp_path / "target"
    (target / "docs").mkdir(parents=True)
    shutil.copyfile(root / "docs/selected-variant-inventory.json", target / "docs/selected-variant-inventory.json")
    snapshot_root = tmp_path / "snapshot"
    (snapshot_root / "bin").mkdir(parents=True)
    profile = backend.AtlasNativeBinding.profile()
    (snapshot_root / "profile.json").write_text(json.dumps(profile.record()))
    bridge = snapshot_root / "bin/merlin-egg-bridge"
    bridge.write_text("mock bridge; never executed")
    manifest = {"scope": "mock native snapshot transport", "profile_sha256": hashlib.sha256(
        (snapshot_root / "profile.json").read_bytes()).hexdigest()}
    (snapshot_root / "manifest.json").write_text(json.dumps(manifest))
    snapshot = NativeSnapshot(snapshot_root, profile, bridge, manifest)
    monkeypatch.setattr(regions, "open_native_snapshot", lambda _: snapshot)
    monkeypatch.setattr(backend, "open_native_snapshot", lambda _: snapshot)
    tools = tmp_path / "tools"
    tools.mkdir()
    for name in ("atlas-opt", "atlas-emit"):
        tool = tools / name
        tool.write_text("mock tool; never executed")
        tool.chmod(0o755)
    calls = []
    def emitted(snapshot, request, *, destination, tool_directory, pipeline, **kwargs):
        assert tool_directory == tools and pipeline == "mandatory"
        calls.append(request.digest())
        destination.mkdir()
        return publish_program(destination, words=[0x01020304], virtual="mock virtual", machine="mock machine",
            program={"words": [0x01020304]}, plan=({"scope": "mock native transport"}
                if plan_override is None else plan_override), selection={},
            metadata={"engine": "merlin_native", "artifact_kind": "program", "qualification_scope": "diagnostic_only",
                      "request_digest": request.digest(), "target_identity": request.target_identity,
                      "native_pipeline": pipeline})
    monkeypatch.setattr(backend.AtlasNativeBinding, "compile", emitted)
    provider = backend.AtlasBackend.configure(snapshot=snapshot, target_source=target, tool_directory=tools,
        pipeline="mandatory", fixed_inputs=(("arg0", 3),), fixed_outputs=(35,), limits=SearchLimits())
    source = (root / "test/examples/native_bf16_relu_linalg.mlir").read_text()
    return provider, source, calls


def test_system_atlas_mocked_transport_binds_original_source_selected_recipe_and_binary(tmp_path, monkeypatch):
    provider, source, calls = _configured_transport(tmp_path, monkeypatch)
    result = api.compile(source, backend=_system(provider, tmp_path), output=tmp_path / "compiled")
    assert len(calls) == 1 and result.metadata["stage"] == "compiled_regions" and result.deployment is None
    program = next(a for a in result.artifacts if a.role == "region.device-program")
    assert program.kind == "device-binary" and program.path.read_bytes() == bytes.fromhex("04030201")
    plan = json.loads((result.directory / "plan.json").read_text())
    assert plan["regions"][0]["lowering_binding"] == "selected_offer"
    child = api.inspect(result.directory / "regions/0")
    assert child["metadata"]["selected_lowering"]["recipe"]["pipeline"] == "atlas.native/mandatory"
    assert child["metadata"]["region_scope"] == "native_device_binary; no_loader_or_callable_abi"
    assert child["metadata"]["execution_observed"] is False


def test_configured_atlas_refuses_unapproved_source_changed_tools_and_changed_configuration(tmp_path, monkeypatch):
    from merlin.compiler.request import CompileRequest

    provider, source, calls = _configured_transport(tmp_path, monkeypatch)
    system = _system(provider, tmp_path)
    with pytest.raises(ValueError, match="no independently admitted source implementations"):
        api.compile(source.replace("32x32", "16x16"), backend=system, output=tmp_path / "small")
    with pytest.raises(ValueError, match="numerical policy and input domain"):
        api.compile(source.replace("bf16-sign-clamp-v1", "different-policy"), backend=system,
                    output=tmp_path / "wrong-policy")
    with pytest.raises(ValueError, match="no additional implicit compile options"):
        provider.validate(CompileRequest(Source(source), tmp_path / "opts",
                          CompileOptions(parameters={"atlas.unknown": True})))
    with pytest.raises(ValueError, match="selected lowering ticket"):
        api.compile(source, backend=provider, output=tmp_path / "no-ticket")
    # Signature obligations cannot be silently dropped while outlining.
    from xdsl.dialects import builtin

    from merlin.compiler.frontend.linalg_mlir import parse_mlir_text
    from merlin.compiler.transforms.certificates import ir_text

    module = parse_mlir_text(source)
    module.body.block.first_op.properties["arg_attrs"] = builtin.ArrayAttr([
        builtin.DictionaryAttr({"merlin.input_domain": builtin.StringAttr("finite-normal-or-zero-bf16")})])
    with pytest.raises(ValueError, match="signature semantic attributes"):
        provider.validate(CompileRequest(Source(ir_text(module)), tmp_path / "signature", CompileOptions()))
    original = provider.fixed_inputs
    provider.fixed_inputs = (("arg0", 4),)
    with pytest.raises(ValueError, match="configuration changed"):
        api.compile(source, backend=system, output=tmp_path / "changed")
    provider.fixed_inputs = original
    (provider.tool_directory / "atlas-emit").write_text("changed selected tool")
    with pytest.raises(ValueError, match="native tool bytes changed"):
        api.compile(source, backend=system, output=tmp_path / "tool-drift")
    assert not calls
    assert not any((tmp_path / name).exists() for name in
                   ("small", "wrong-policy", "opts", "no-ticket", "signature", "changed", "tool-drift"))


def test_system_atlas_real_native_tools_emit_selected_binary_without_llvm_or_loader_claim(tmp_path, monkeypatch):
    import os
    from pathlib import Path

    from merlin.compiler.semantic.search import SearchLimits
    from merlin.compiler.semantic.snapshot import open_native_snapshot

    selected = tuple(os.environ.get(name) for name in (
        "ATLAS_SYSTEM_SNAPSHOT", "ATLAS_SYSTEM_TARGET_SOURCE", "ATLAS_SYSTEM_TOOL_DIRECTORY"))
    if not all(selected):
        pytest.skip("select an admitted native snapshot, target source and built Atlas tool directory")
    snapshot_root, target_source, tools = map(Path, selected)
    provider = backend.AtlasBackend.configure(snapshot=open_native_snapshot(snapshot_root), target_source=target_source,
        tool_directory=tools, pipeline="mandatory", fixed_inputs=(("arg0", 3),), fixed_outputs=(35,),
        limits=SearchLimits())
    # Explicit per-call selection must neither consult nor mutate these legacy defaults.
    monkeypatch.setenv("ATLAS_NATIVE_PIPELINE", "invalid-unselected-default")
    monkeypatch.setenv("ATLAS_OOT_BIN_DIR", str(tmp_path / "unselected-tools"))
    source = (Path(__file__).resolve().parents[1] / "test/examples/native_bf16_relu_linalg.mlir").read_text()
    result = api.compile(source, backend=_system(provider, tmp_path), output=tmp_path / "compiled-native")
    assert result.metadata["stage"] == "compiled_regions" and result.deployment is None
    child = api.inspect(result.directory / "regions/0")
    assert child["metadata"]["qualification_scope"] == "diagnostic_only"
    assert child["metadata"]["execution_observed"] is False
    assert child["metadata"]["native_pipeline"] == "mandatory"
    physical = json.loads((result.directory / "regions/0/physical-program.json").read_text())
    assert sum(row["operation"] == "atlas.vpu_unary" for row in physical["instructions"]) == 1
    binary = next(a for a in result.artifacts if a.role == "region.device-program")
    assert binary.kind == "device-binary" and len(binary.path.read_bytes()) == physical["word_count"] * 4
    assert not any(a.kind == "llvm-ir" or a.kind == "elf" for a in result.artifacts)
    assert os.environ["ATLAS_NATIVE_PIPELINE"] == "invalid-unselected-default"



def _bf16_platform(tmp_path):
    import hashlib
    from pathlib import Path

    from merlin.compiler.artifact import Artifact, Compilation
    from merlin.compiler.backend.stateful import CContext

    root = Path(__file__).resolve().parents[1]
    directory = tmp_path / "platform"
    directory.mkdir()
    for original, name in (("bf16-call-platform.c", "platform.c"), ("bf16-call-platform.h", "platform.h")):
        (directory / name).write_bytes((root / "test" / original).read_bytes())
    contract = {"schema": "atlas.bf16-call-platform.v1",
        "map": {"imem_base": 0x20000, "csr_base": 0x40000, "imem_capacity_bytes": 131072},
        "input": {"address": 0x90001800, "bytes": 2048, "alignment": 64},
        "output": {"address": 0x90011800, "bytes": 2048, "alignment": 64},
        "pre_reserved_outside_available_capacity": True, "control": "single_context"}
    (directory / "contract.json").write_text(json.dumps(contract, sort_keys=True))
    products = tuple(Artifact.from_file(directory / name, kind=kind, role="platform") for name, kind in (
        ("platform.c", "c-source"), ("platform.h", "c-header"), ("contract.json", "json")))
    source_sha = hashlib.sha256((directory / "platform.c").read_bytes()).hexdigest()
    result = Compilation(directory, "test-platform", source_sha,
                         products, metadata={"scope": "fake IO fixture; no device arithmetic/hardware qualification"})
    result.write()
    digest = next(a.sha256 for a in products if a.kind == "json")
    context = CContext("single-device-context", digest, "cpu0", "host", 512, 64, 2048, 64,
                       "call/region.h", "atlas_bf16_call", "atlas_bf16_initialize")
    return result, context


@pytest.mark.parametrize("native", [False, True])
def test_system_atlas_bf16_call_compiles_and_runs_only_fake_transport(tmp_path, monkeypatch, native):
    import os
    import shutil
    from pathlib import Path

    from atlas_mlir_native.call import BF16Call
    from atlas_mlir_native.layout import pack_bf16_pair_halves

    from merlin.compiler.semantic.search import SearchLimits
    from merlin.compiler.semantic.snapshot import open_native_snapshot

    values = tuple(os.environ.get(name) for name in (
        "ATLAS_SYSTEM_SNAPSHOT", "ATLAS_SYSTEM_TARGET_SOURCE", "ATLAS_SYSTEM_TOOL_DIRECTORY"))
    compiler = shutil.which("gcc")
    if not compiler or native and not all(values):
        pytest.skip("select native snapshot/Atlas tools plus the real host C compiler")
    if native:
        snapshot_root, target_source, tools = map(Path, values)
        selected_snapshot = open_native_snapshot(snapshot_root)
    else:
        fake_plan = {"schema": "atlas.native_tile_execution_plan.v8", "shape": [32, 32], "dtype": "bf16",
            "input_bytes": 2048, "output_bytes": 2048, "input_addresses": [0x90001800],
            "output_addresses": [0x90011800], "output_physical_layouts": ["bf16_pair_halves"],
            "boundary_domain_obligations": [{"input": "arg0", "required_domain": "finite-normal-or-zero-bf16",
                                               "status": "caller_premise_unchecked"}]}
        transport, _, _ = _configured_transport(tmp_path, monkeypatch, plan_override=fake_plan)
        selected_snapshot, target_source, tools = transport.snapshot, transport.target_source, transport.tool_directory
    root = Path(__file__).resolve().parents[1]
    from merlin.common.paths import data_path

    runtime_headers = data_path("runtime", "c")
    platform, context = _bf16_platform(tmp_path)
    call_options = dict(context=context, compiler=Path(compiler),
        flags=("-std=c11", "-O2", "-Wall", "-Wextra", "-Werror"), runtime=root / "runtime", headers=runtime_headers,
        platform=platform, contract="contract.json", platform_header="platform.h",
        platform_initialize="test_platform_initialize", link_inputs=("platform.c",))
    call = BF16Call(**call_options)
    provider = backend.AtlasBackend.configure(snapshot=selected_snapshot, target_source=target_source,
        tool_directory=tools, pipeline="mandatory", fixed_inputs=(("arg0", 3),), fixed_outputs=(35,),
        limits=SearchLimits(), call=call)
    source = (root / "test/examples/native_bf16_relu_linalg.mlir").read_text()
    options = CompileOptions(parameters={"merlin.emit": "c-dispatch", "merlin.entry": "model"})
    result = api.compile(source, backend=_system(provider, tmp_path, context=context),
                         output=tmp_path / "compiled-call", options=options)
    assert result.metadata["stage"] == "c_dispatch" and result.deployment is None
    report = json.loads((result.directory / "control/control.json").read_text())
    assert report["context_table"]["minimum_count"] == 1
    assert report["context_initialization"] == "caller_obligation_not_observed"
    region = result.directory / "regions/0"
    child = api.inspect(region)
    assert child["metadata"]["c_call"]["schema"] == "merlin.c_call.v2"
    assert child["metadata"]["c_call"]["inputs"] == [{"shape": [32, 32], "dtype": "bf16"}]
    assert child["metadata"]["runtime_domain_guard"] == "compiled_before_io; execution_unobserved"
    # Independently configure fake readback bytes; no device arithmetic runs.
    raw_input = b"".join(code.to_bytes(2, "little") for code in (0x0000, 0x8000, 0x3f80, 0xbf80) * 256)
    expected = b"".join((0x3f80 + index % 128).to_bytes(2, "little") for index in range(1024))
    data = {"incoming": raw_input, "expected": expected,
            "physical_input": pack_bf16_pair_halves(raw_input), "response": pack_bf16_pair_halves(expected)}
    lines = ["#include <stdint.h>"]
    for name, payload in data.items():
        codes = [int.from_bytes(payload[n:n + 2], "little") for n in range(0, len(payload), 2)]
        lines.append(f"static const uint16_t {name}[1024]={{" + ",".join(hex(code) for code in codes) + "};")
    binary = (region / "program.bin").read_bytes()
    words = [int.from_bytes(binary[n:n + 4], "little") for n in range(0, len(binary), 4)]
    lines.append("static const uint32_t compiled_program[]={" + ",".join(hex(word) for word in words) + "};")
    (tmp_path / "data.h").write_text("\n".join(lines))
    (tmp_path / "main.c").write_text("""#include <string.h>
#include "control/model.h"
#include "regions/0/call/region.h"
#include "regions/0/call/platform/platform.h"
#include "data.h"
int main(void) {
 _Alignas(64) atlas_bf16_call context={0},clone={0};
 _Alignas(64) unsigned char workspace[2048],arena[64];
 _Alignas(64) uint16_t input[1024],output[1024];
 memcpy(input,incoming,sizeof(input)); memset(output,0x5a,sizeof(output));
 test_platform_response((const uint8_t *)response,sizeof(response));
 unsigned effects=test_platform_effects();
 if(atlas_bf16_initialize(&context,workspace,sizeof(workspace)) || test_platform_effects()!=effects) return 1;
 if(!atlas_bf16_initialize(&clone,workspace,sizeof(workspace))) return 2;
 void *items[]={&context}; model_contexts contexts={items,1};
 model_buffer buffers[]={{input,sizeof(input)},{output,sizeof(output)}}; model_state state={0};
 if(model_run(&state,arena,sizeof(arena),buffers,2,&contexts) || memcmp(output,expected,sizeof(output)) ||
    memcmp(input,incoming,sizeof(input)) || memcmp(test_platform_input(),physical_input,sizeof(input)) ||
    !test_platform_loaded(compiled_program,sizeof(compiled_program)/sizeof(*compiled_program))) return 3;
 effects=test_platform_effects(); input[1023]=0x7fc1; state=(model_state){0};
 if(model_run(&state,arena,sizeof(arena),buffers,2,&contexts)!=MODEL_KERNEL_ERROR ||
    test_platform_effects()!=effects || memcmp(output,expected,sizeof(output))) return 4;
 input[1023]=incoming[1023]; state=(model_state){0}; test_platform_mode(1);
 if(model_run(&state,arena,sizeof(arena),buffers,2,&contexts)!=MODEL_KERNEL_ERROR || !context.failed ||
    !context.host.active || !state.active || memcmp(output,expected,sizeof(output))) return 5;
 effects=test_platform_effects();
 if(model_run(&state,arena,sizeof(arena),buffers,2,&contexts)!=MODEL_BUSY ||
    test_platform_effects()!=effects) return 6;
 test_platform_mode(0); if(atlas_region_drain(&context,4)!=ATLAS_HOST_OK) return 7;
 state=(model_state){0}; test_platform_mode(3);
 uint16_t untouched[1024]; memset(untouched,0x5a,sizeof(untouched)); memcpy(output,untouched,sizeof(output));
 if(model_run(&state,arena,sizeof(arena),buffers,2,&contexts)!=MODEL_KERNEL_ERROR || !context.failed ||
    memcmp(output,untouched,sizeof(output))) return 8;
 test_platform_mode(0); if(atlas_region_drain(&context,4)!=ATLAS_HOST_OK) return 9;
 return 0;
}
""")
    executable = tmp_path / "transport"
    import subprocess

    subprocess.run([compiler, "-std=c11", "-O2", "-Wall", "-Wextra", "-Werror", "-I" + str(result.directory),
                    "-I" + str(region / "call/sdk"), "-I" + str(tmp_path), str(tmp_path / "main.c"),
                    *(str(result.directory / name) for name in report["link_inputs"]), "-o", str(executable)],
                   check=True, capture_output=True, text=True)
    subprocess.run([str(executable)], check=True, capture_output=True, text=True)
    # The checked embedding is not an admitted whole-image loading contract.
    assert report["payloads"] == ["regions/0/program.bin"] and not any(a.kind == "elf" for a in result.artifacts)

    if not native:
        from dataclasses import replace

        # A conditional state ceiling must be discharged by the actual host C
        # compiler, rather than accepted because the proposal declares it.
        small_state = replace(context, state_bytes=16)
        bad_call = BF16Call(**{**call_options, "context": small_state})
        bad_provider = backend.AtlasBackend.configure(snapshot=selected_snapshot, target_source=target_source,
            tool_directory=tools, pipeline="mandatory", fixed_inputs=(("arg0", 3),), fixed_outputs=(35,),
            limits=SearchLimits(), call=bad_call)
        with pytest.raises(RuntimeError, match="selected context state ceiling"):
            api.compile(source, backend=_system(bad_provider, tmp_path, context=small_state),
                        output=tmp_path / "bad-state", options=options)
        assert not (tmp_path / "bad-state/compilation.json").exists()
        assert not (tmp_path / "bad-state/control").exists()
        small_workspace = replace(context, workspace_bytes=8)
        with pytest.raises(ValueError, match="workspace ceiling"):
            backend.AtlasBackend.configure(snapshot=selected_snapshot, target_source=target_source,
                tool_directory=tools, pipeline="mandatory", fixed_inputs=(("arg0", 3),), fixed_outputs=(35,),
                limits=SearchLimits(), call=BF16Call(**{**call_options, "context": small_workspace}))
