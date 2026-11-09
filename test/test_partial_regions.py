"""Actual source-driven Atlas regions and scalar complements, without device claims."""
from __future__ import annotations

import json
from pathlib import Path

import pytest
from test_public_backend import _configured_transport

from merlin import api
from merlin.compiler.ir.region import preview_region
from merlin.compiler.ir.semantics import source_semantic_attributes
from merlin.frontends.linalg_bf16_relu import recognize_bf16_sign_clamp_relu
from merlin.frontends.linalg_mlir import parse_mlir_text
from merlin.llvmlower.offload_plan import bind_source
from merlin.transforms.certificates import ir_text
from merlin.transforms.pipeline_state import PipelineContext


def _mixed(source):
    source = source.replace("@relu_tile", "@model")
    source = source.replace("    %empty =", '''    %before_empty = tensor.empty() : tensor<32x32xbf16>
    %before = linalg.transpose ins(%input:tensor<32x32xbf16>) outs(%before_empty:tensor<32x32xbf16>)
      permutation=[1,0]
    %empty =''', 1)
    source = source.replace("ins(%input :", "ins(%before :")
    return source.replace("    func.return %result", '''    %after_empty = tensor.empty() : tensor<32x32xbf16>
    %after = linalg.transpose ins(%result:tensor<32x32xbf16>) outs(%after_empty:tensor<32x32xbf16>)
      permutation=[1,0]
    func.return %after''')


def _source_policy(module, entry, options):
    assert options == {"meaning": "bf16-sign-clamp-v1", "domain": "finite-normal-or-zero-bf16"}
    operations = tuple(module.walk())
    rows = source_semantic_attributes(module, entry)
    for row in rows:
        if row.location != "operation" or row.key not in {"merlin.numeric_semantics", "merlin.input_domain"}:
            raise ValueError("original caller has no inherited/signature/unknown annotation admission")
        recognize_bf16_sign_clamp_relu(operations[row.ordinal])
    return rows


def _bound(source):
    return bind_source(parse_mlir_text(source), entry="model", context=PipelineContext(), graph_mode="ssa",
        validate_source=_source_policy,
        source_options={"meaning": "bf16-sign-clamp-v1", "domain": "finite-normal-or-zero-bf16"})


def _approve_region(source, proposal):
    operations = tuple(source.operations[n] for n in proposal.alternative.nodes)
    actual = preview_region(source.module, source.entry, operations, symbol="independent_source_region")
    relu = tuple(actual.module.body.block.first_op.body.block.ops)[1]
    recognize_bf16_sign_clamp_relu(relu)
    assert proposal.alternative.implementation == "atlas.bf16_relu.native.v1"
    assert dict(proposal.alternative.metadata)["input_domain"] == "caller_premise_unchecked"


def test_partial_offers_bind_actual_sites_and_complete_live_values(tmp_path, monkeypatch):
    provider, source, _ = _configured_transport(tmp_path, monkeypatch)
    with pytest.raises(ValueError, match="unknown effects|source annotations"):
        bind_source(parse_mlir_text(_mixed(source)), entry="model", context=PipelineContext(), graph_mode="ssa")
    bound = _bound(_mixed(source))
    before = ir_text(bound.module)
    proposals = provider.propose(bound, resource="device0", memory_domain="ram")
    assert len(proposals) == 1 and proposals[0].alternative.nodes == (2, 3)
    assert proposals[0].operation_ordinals == tuple(bound.operation_ordinals[n] for n in (2, 3))
    assert proposals[0].alternative.id == "atlas.bf16-relu/region/000002-000004"
    assert len(proposals[0].alternative.inputs) == len(proposals[0].alternative.outputs) == 1
    _approve_region(bound, proposals[0])
    assert ir_text(bound.module) == before


@pytest.mark.parametrize("mutation", ("nonadjacent", "shared_initializer", "nested_attribute", "fastmath"))
def test_partial_family_refuses_invalid_geometry_liveness_and_semantics(tmp_path, monkeypatch, mutation):
    from xdsl.dialects import arith, builtin, func, tensor

    provider, source, _ = _configured_transport(tmp_path, monkeypatch)
    module = parse_mlir_text(_mixed(source))
    function = module.body.block.first_op
    operations = tuple(function.body.block.ops)
    empty, relu = operations[2:4]
    if mutation == "nonadjacent":
        value = arith.ConstantOp(builtin.IntegerAttr(7, builtin.i32))
        function.body.block.insert_op_before(value, relu)
    elif mutation == "shared_initializer":
        function.get_return_op().operands = (*function.get_return_op().operands, empty.results[0])
        function.update_function_type()
    elif mutation == "nested_attribute":
        maximum = next(op for op in relu.walk() if type(op) is arith.MaximumfOp)
        maximum.attributes["required.numerics"] = builtin.UnitAttr()
    else:
        maximum = next(op for op in relu.walk() if type(op) is arith.MaximumfOp)
        maximum.properties["fastmath"] = arith.FastMathFlagsAttr([arith.FastMathFlag("nnan")])
    # This callback only accounts for original annotations to exercise the OOT
    # recognizer's independent refusals; it grants no target/math permission.
    def coverage(current, entry, options):
        return source_semantic_attributes(current, entry)
    bound = bind_source(module, entry="model", context=PipelineContext(), graph_mode="ssa",
                        validate_source=coverage, source_options={})
    assert provider.propose(bound, resource="device0", memory_domain="ram") == ()
    assert type(empty) is tensor.EmptyOp and type(function) is func.FuncOp


def test_identical_relu_sites_have_distinct_source_identity_and_inherited_policies_refuse(tmp_path, monkeypatch):
    from xdsl.dialects import builtin, func

    provider, source, _ = _configured_transport(tmp_path, monkeypatch)
    module = parse_mlir_text(source.replace("@relu_tile", "@model"))
    function = module.body.block.first_op
    empty, relu, returned = tuple(function.body.block.ops)
    mapping = {function.args[0]: relu.results[0]}
    new_empty = empty.clone(value_mapper=mapping)
    new_relu = relu.clone(value_mapper=mapping)
    function.body.block.insert_ops_before([new_empty, new_relu], returned)
    returned.operands = (new_relu.results[0],)
    bound = _bound(ir_text(module))
    proposals = provider.propose(bound, resource="device0", memory_domain="ram")
    assert [p.alternative.nodes for p in proposals] == [(0, 1), (2, 3)]
    assert len({p.numerical_sha256 for p in proposals}) == len({p.operation_ordinals for p in proposals}) == 2
    function.properties["arg_attrs"] = builtin.ArrayAttr([
        builtin.DictionaryAttr({"merlin.input_domain": builtin.StringAttr("finite-normal-or-zero-bf16")})])
    def coverage(current, entry, options):
        return source_semantic_attributes(current, entry)
    inherited = bind_source(module, entry="model", context=PipelineContext(), graph_mode="ssa",
                            validate_source=coverage, source_options={})
    assert provider.propose(inherited, resource="device0", memory_domain="ram") == ()
    assert type(returned) is func.ReturnOp


def test_automatic_three_region_plan_and_actual_native_cpu_complements(tmp_path, monkeypatch):
    import ctypes
    import platform

    import numpy as np
    import yaml

    from merlin.compiler.backend.cabi import CRegionABI
    from merlin.compiler.pipeline import RegionBackend
    from merlin.compiler.system import Component, SystemCompiler, scalar_cpu
    from merlin.llvmlower import codegen, target_data_layout, toolchain
    from merlin.system.topology import select_topology

    if (platform.system() != "Linux" or platform.machine() != "x86_64"
            or toolchain._env("MERLIN_COMPILER_DRIVER", "python") != "stock-cli"
            or not Path(toolchain.clang()).is_file()):
        pytest.skip("select actual stock LLVM host tools for native scalar complements")
    provider, whole, calls = _configured_transport(tmp_path, monkeypatch)
    source = _mixed(whole)
    document = {"schema": "merlin.execution_topology.v1",
        "resources": [{"id": name, "kind": kind, "features": [], "shared_group": name}
                      for name, kind in (("cpu0", "cpu"), ("device0", "accelerator"))],
        "memory_domains": [{"id": "ram", "capacity_bytes": 1 << 20, "alignment_bytes": 64}],
        "access": [{"resource": name, "domain": "ram", "read": True, "write": True, "addressability": "shared",
                    "visibility": "coherent", "completion": "synchronous"} for name in ("cpu0", "device0")]}
    hardware = tmp_path / "hardware.yaml"
    hardware.write_text(yaml.safe_dump(document))
    layout = target_data_layout.observe_index_width(toolchain.clang(), ["--target=x86_64-unknown-linux-gnu"])
    cpu = scalar_cpu(resource="cpu0", memory_domain="ram", entry="cpu_kernel", options=api.CompileOptions(parameters={
        "scalar.data-layout": layout["data_layout"], "scalar.index-bits": 64,
        "scalar.target-flags": ["--target=x86_64-unknown-linux-gnu"]}))
    device = Component(RegionBackend(provider, "device_kernel"), "device0", "ram", _approve_region,
                       required_resource_kind="accelerator")
    backend = SystemCompiler(topology=select_topology(hardware), components=(device, cpu),
        preferred_providers=("atlas", "merlin.scalar"), entry="model", validate_source=_source_policy,
        source_options={"meaning": "bf16-sign-clamp-v1", "domain": "finite-normal-or-zero-bf16"})
    result = api.compile(source, backend=backend, output=tmp_path / "compiled")
    report = json.loads((result.directory / "system.json").read_bytes())
    assert [row["provider"] for row in report["selection"]["execution_order"]] == [
        "merlin.scalar", "atlas", "merlin.scalar"]
    assert len(calls) == 1 and report["source_semantics"]["attributes"]
    cpu_discovery = next(row["report"] for row in report["discovery"] if row["provider"] == "merlin.scalar")
    assert any(row["status"] == "refused" and "annotated" in row["reason"] for row in cpu_discovery["offers"])
    # The device artifact is mocked transport and is never executed here.
    # Native CPU children independently preserve all BF16 bits through real
    # nontrivial layouts, with test reference values at their shared boundary.
    bits = (np.uint16(0x3F80) + (np.arange(1024, dtype=np.uint16) % 127)).reshape(32, 32)
    bits[::2] |= np.uint16(0x8000)
    original = bits.copy()
    current = bits
    for index in (0, 2):
        child = api.Compilation.load(result.directory / "regions" / str(index))
        abi = CRegionABI.from_compilation(child)
        shared = tmp_path / (f"cpu-{index}.so")
        objects = tuple(child.directory / name for name in abi.link_inputs)
        codegen._run([toolchain.clang(), "-shared", *objects, "-lm", "-o", shared],
                     inputs=objects, outputs=(shared,), stage="partial_cpu_native_link")
        library = ctypes.CDLL(str(shared))
        function = getattr(library, abi.symbol)
        function.restype = ctypes.c_int
        function.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_void_p), ctypes.POINTER(ctypes.c_void_p)]
        actual = np.zeros_like(current)
        inputs = (ctypes.c_void_p * 1)(current.ctypes.data)
        outputs = (ctypes.c_void_p * 1)(actual.ctypes.data)
        assert function(None, inputs, outputs) == 0
        np.testing.assert_array_equal(actual, current.T)
        if index == 0:
            current = np.where((actual & np.uint16(0x8000)) != 0, np.uint16(0), actual).astype(np.uint16)
        else:
            current = actual
    expected = np.where((original & np.uint16(0x8000)) != 0, np.uint16(0), original).astype(np.uint16)
    np.testing.assert_array_equal(current, expected)
    np.testing.assert_array_equal(bits, original)
    assert result.metadata["execution_verified"] is False
