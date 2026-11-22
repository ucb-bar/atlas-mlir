"""Generated covers use existing Atlas routes with independent custody/grants."""
import json
from copy import deepcopy
from dataclasses import replace
from hashlib import sha256

import pytest
from atlas_mlir_native.application import AtlasApplicationAdapter
from atlas_mlir_native.binding import TARGET_IDENTITY
from atlas_mlir_native.compiler_interface import (
    describe_compiler,
    describe_numerical_families,
)
from atlas_mlir_native.numerical_contracts import MXU1_PRECISION_AUTHORIZATION
from atlas_mlir_native.source_translation import (
    translate_activation_quantizer,
    translate_prequantized_contraction,
)
from merlin.compiler.frontend.policy_inputs import (
    authenticate_policy_inputs,
    stage_policy_inputs,
)
from merlin.compiler.quantization.policy_generation import generate_capture_policy
from merlin.compiler.semantic.linalg.linalg_bridge import translate_linalg_text
from merlin.compiler.semantic.source_preparation import prepare_source
from merlin.compiler.semantic.target_binding import NativeCompilationError

AUTHORITY = (MXU1_PRECISION_AUTHORIZATION,)
FAMILIES = ("atlas.ordered-quantized-contraction",)


def bundle(tmp_path, k=32, *, quantizer=False, bind_external_parameters=False, producer_policy=None):
    """Small real receipt/schema fixture; no weights or hardware dependency."""
    tmp_path.mkdir(exist_ok=True)
    export, fx, capture = (tmp_path / name for name in ("export", "fx", "capture"))
    for root in (export, fx, capture):
        root.mkdir()

    def write(root, name, value):
        raw = value if isinstance(value, bytes) else json.dumps(value).encode()
        (root / name).write_bytes(raw)
        return {"bytes": len(raw), "sha256": sha256(raw).hexdigest()}

    policy = write(export, "policy.json", producer_policy if producer_policy is not None else
                   {"schema": "atlas_torchao.execution_policy.v1", "format_version": 1})
    manifest = write(export, "manifest.json", {
        "schema": "atlas_torchao.export_manifest.v1", "format_version": 1,
        "files": {"policy.json": policy}, "policy_sha256": policy["sha256"],
    })
    write(fx, "fx_manifest.json", {
        "schema": "atlas_torchao.fx_bundle.v1", "provenance": {"export_manifest_sha256": manifest["sha256"]},
        "files": {"model.pt2": {"sha256": "c" * 64}},
    })
    quantization = {
        "schema": "m2m.quantization_manifest.v2", "adapter_id": "atlas_fp8",
        "contract_sha256": manifest["sha256"], "policy_sha256": policy["sha256"],
        "atlas": {"saved_native_program": {"sha256": "c" * 64}},
    }
    if producer_policy is not None:
        quantization["atlas"]["buffers"] = {}
        quantization["sites"] = [{"site_id": "projection", "execution_route": "atlas.native_fp8"}]
    qdigest = sha256(json.dumps(quantization, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    source = f'''builtin.module attributes {{prov.weights_file = "origin/weights.safetensors", prov.quantization_manifest_sha256 = "{qdigest}"}} {{
      func.func @forward(%a: tensor<2x{k}xi8>, %ae: tensor<i32>, %w: tensor<3x{k}xi8>, %we: tensor<3x1xi32>) -> tensor<2x3xbf16> {{
        %r = "atlas.qlinear"(%a, %ae, %w, %we) {{code_dtype = "uint8", exponent_dtype = "int32", weight_format = "atlas_e4m3", dequantization = "e4m3(code)*2^exponent", rhs_transposed = "true", output_rescale = "2^(lhs_exp+rhs_exp)", bias_order = "after_output_cast", output_dtype = "bfloat16"}} : (tensor<2x{k}xi8>, tensor<i32>, tensor<3x{k}xi8>, tensor<3x1xi32>) -> tensor<2x3xbf16>
        func.return %r : tensor<2x3xbf16>
      }}
    }}'''
    abi = (([2, k], "i8"), ([], "i32"), ([3, k], "i8"), ([3, 1], "i32"))
    selected_families = FAMILIES
    if quantizer:
        source = source.replace(f"%a: tensor<2x{k}xi8>, %ae: tensor<i32>", f"%a: tensor<2x{k}xbf16>")
        source = source.replace('%r = "atlas.qlinear"(%a, %ae, %w, %we)',
            f'''%codes, %ae = "atlas.quantize_act"(%a) {{mode = "static", reduce = "tensor", rule = "ref", static_exp = 0 : i32, rows = 1 : i32, code_dtype = "uint8", exponent_dtype = "int32", weight_format = "atlas_e4m3", dequantization = "e4m3(code)*2^exponent"}} : (tensor<2x{k}xbf16>) -> (tensor<2x{k}xi8>, tensor<i32>)
        %r = "atlas.qlinear"(%codes, %ae, %w, %we)''')
        abi = (([2, k], "bf16"), ([3, k], "i8"), ([3, 1], "i32"))
        selected_families = (*FAMILIES, "atlas.host-static-quantize")
    trace = {"status": "complete", "mlir": {"sha256": sha256(source.encode()).hexdigest()}}
    members = {
        "model.mlir": write(capture, "model.mlir", source.encode()),
        "weights.safetensors": write(capture, "weights.safetensors", b"unused archive bytes"),
        "weights.safetensors.manifest.json": write(capture, "weights.safetensors.manifest.json", {}),
        "frontend-trace.json": write(capture, "frontend-trace.json", trace),
        "quantization-manifest.json": write(capture, "quantization-manifest.json", quantization),
    }
    members["meta.json"] = write(capture, "meta.json", {
        "ok": True, "opaque": 0, "weights": "origin/weights.safetensors", "func_name": "forward",
        "frontend_trace": {"sha256": members["frontend-trace.json"]["sha256"]},
        "input_abi": [{"shape": shape, "dtype": dtype} for shape, dtype in abi],
        "output_abi": [{"shape": [2, 3], "dtype": "bf16"}], "dtype": "bf16",
    })
    write(capture, "capture_receipt.json", {
        "schema": "m2m.capture-receipt.v1", "materialized_abi": {"complete": True, "inputs": len(abi), "lifted_constants": []},
        "lifted_constants": {}, "artifacts": members,
    })
    bound = authenticate_policy_inputs(
        export, fx, capture, export_manifest_schema="atlas_torchao.export_manifest.v1",
        export_policy_schema="atlas_torchao.execution_policy.v1", fx_manifest_schema="atlas_torchao.fx_bundle.v1",
        adapter_id="atlas_fp8", native_program_sha256_path=("atlas", "saved_native_program", "sha256"),
    )
    translators = {"atlas.qlinear": translate_prequantized_contraction, "atlas.quantize_act": translate_activation_quantizer}
    with stage_policy_inputs(bound, capture) as (_, staged):
        source = translate_linalg_text(staged.program, entry=staged.entry, target_identity=TARGET_IDENTITY,
            lowering_policy="diagnostic", source_operation_translators=translators).request
        if bind_external_parameters:
            from merlin.compiler.lower.c.parameters import parameter_source_identity

            source = replace(source, source_identity=parameter_source_identity(source.source_identity, staged))
        source, _ = prepare_source(source)
    generated = generate_capture_policy(bound, capture, describe_compiler(), describe_numerical_families(),
        target_identity=TARGET_IDENTITY, source_operation_translators=translators,
        authorized_precision_changes=AUTHORITY, selected_family_ids=selected_families,
        bind_external_parameters=bind_external_parameters)
    return source, generated, bound, capture


def admit(source, generated, bound, capture, **options):
    return AtlasApplicationAdapter.from_generated_policy(
        source, generated, bound=bound, capture_dir=capture,
        authorized_precision_changes=options.get("authorized_precision_changes", AUTHORITY),
        selected_family_ids=options.get("selected_family_ids", FAMILIES),
        bind_external_parameters=options.get("bind_external_parameters", False))


@pytest.mark.parametrize("k", (32, 64))
def test_generated_cover_selects_existing_route_with_sparse_capabilities(tmp_path, k, monkeypatch):
    source, generated, bound, capture = bundle(tmp_path, k)
    adapter = admit(source, generated, bound, capture)
    plan = adapter.plan(source)
    assert len(plan.selected) == 1
    assert dict(plan.selected[0].metadata)["kind"] == "already_quantized"
    assert adapter.already_quantized
    assert not adapter.pointwise and not adapter.reductions and not adapter.host_softmax
    assert adapter.host_layouts == adapter.host_float_arithmetic == adapter.host_float_casts == ()
    changed = replace(source, source_identity="changed")
    with pytest.raises(NativeCompilationError, match="source identity changed"):
        adapter.plan(changed)
    source, generated, bound, capture = bundle(tmp_path / "static", k, quantizer=True)
    adapter = admit(source, generated, bound, capture,
                    selected_family_ids=(*FAMILIES, "atlas.host-static-quantize"))
    monkeypatch.setattr(adapter, "_is_vector_quantizer", lambda *args: pytest.fail("host quantizer consulted device choice"))
    assert [dict(region.metadata)["kind"] for region in adapter.plan(source).selected] == ["static_quantize", "already_quantized"]
    assert adapter.activation_quantizer_route == "host"
    source, generated, bound, capture = bundle(tmp_path / "external", k, bind_external_parameters=True)
    assert generated["parameter_mode"] == "external"
    with pytest.raises((ValueError, NativeCompilationError)):
        admit(source, generated, bound, capture)
    adapter = admit(source, generated, bound, capture, bind_external_parameters=True)
    assert dict(adapter.plan(source).selected[0].metadata)["kind"] == "already_quantized"


def test_generated_cover_rechecks_source_custody_claims_and_independent_grants(tmp_path):
    source, generated, bound, capture = bundle(tmp_path)
    edits = (
        lambda record: record["selected_families"][0].update(family="atlas.host-layout"),
        lambda record: record["selected_families"][0].update(implementation_policy="different-arithmetic"),
        lambda record: record["authenticated_inputs"].update(weights_sha256="d" * 64),
        lambda record: record.update(preparation_sha256="d" * 64),
        lambda record: record.update(authorized_precision_changes=[]),
        lambda record: record.update(selected_family_ids=["atlas.device-layout"]),
    )
    for edit in edits:
        changed = deepcopy(generated)
        edit(changed)
        with pytest.raises((ValueError, NativeCompilationError)):
            admit(source, changed, bound, capture)
    with pytest.raises((ValueError, NativeCompilationError)):
        admit(source, generated, bound, capture, authorized_precision_changes=())
    with pytest.raises((ValueError, NativeCompilationError)):
        admit(source, generated, bound, capture, selected_family_ids=("atlas.unsupported",))
    with pytest.raises((ValueError, NativeCompilationError)):
        admit(replace(source, source_identity="changed"), generated, bound, capture)
    receipt = capture / "capture_receipt.json"
    receipt.write_bytes(receipt.read_bytes() + b" ")
    with pytest.raises(ValueError, match="receipt changed"):
        admit(source, generated, bound, capture)
