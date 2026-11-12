from __future__ import annotations

from dataclasses import replace
import hashlib
import importlib.metadata
import json
from pathlib import Path
from unittest.mock import patch

import pytest
import torch
import algos
import atlas_fx_ops
import quant_export

from atlas_m2m_adapter import atlas_fp8_adapter, load_program, tensor_record
from m2m.capture.external_quantization import ExternalQuantizationConfig, apply_external_quantization


class RepeatedLinear(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = torch.nn.Linear(4, 4, bias=True, dtype=torch.bfloat16)
        self.head = torch.nn.Linear(4, 4, bias=False, dtype=torch.bfloat16)

    def forward(self, value):
        return self.head(self.proj(self.proj(value)))


@pytest.fixture
def replay(tmp_path):
    torch.manual_seed(7)
    model = RepeatedLinear().eval()
    algo = algos.make("granularity:w=channel,a=token,rule=nc")
    options = {"crosscheck": "off", "mxu": "ipt", "mxu_impl": "torch"}
    report = algos.quantize(model, "ref_conv", ["head"], options, algo=algo)
    quant_export.export(model, tmp_path, mode="ref_conv", report=report, algo=algo,
                        algo_spec="granularity:w=channel,a=token,rule=nc", options=options, skip=["head"])
    inputs = (torch.randn(2, 3, 4, dtype=torch.bfloat16),)
    expected = model(*inputs).detach().clone()
    model.proj.weight = torch.nn.Parameter(torch.empty((4, 4), dtype=torch.bfloat16, device="meta"),
                                           requires_grad=False)
    model.eval()
    model._atlas_export_dir = tmp_path
    config = ExternalQuantizationConfig(
        "atlas_fp8", tmp_path / "manifest.json", tmp_path / "policy.json",
        hashlib.sha256((tmp_path / "manifest.json").read_bytes()).hexdigest(),
        hashlib.sha256((tmp_path / "policy.json").read_bytes()).hexdigest(),
    )
    return model, inputs, expected, config


def invoke(replay):
    model, inputs, _, config = replay
    with patch("load_quantized.apply_export", side_effect=AssertionError("weight conversion")), \
         patch("algos.base.weight_cast", side_effect=AssertionError("weight conversion")), \
         patch("quant_modes.quantize", side_effect=AssertionError("quantization")), \
         patch("pack_fold.fold_export_dir", side_effect=AssertionError("pack fold")), \
         patch("pack_fold.apply_folds", side_effect=AssertionError("pack fold")):
        return apply_external_quantization(model, inputs, config)


def test_entry_point_is_unique_and_owned_by_adapter_distribution():
    entries = [entry for entry in importlib.metadata.entry_points(group="m2m.quantization_adapters")
               if entry.name == "atlas_fp8"]
    assert len(entries) == 1
    assert entries[0].dist.metadata["Name"] == "atlas-m2m-adapter"
    assert entries[0].load() is atlas_fp8_adapter


def test_repeated_calls_keep_bytes_roles_literals_and_node_ids(replay, tmp_path):
    model, inputs, expected, config = replay
    before = {name: tensor_record(value) for name, value in model.named_buffers()}
    program = invoke(replay)
    assert torch.equal(program.module()(*inputs), expected)
    assert "forward" not in model.proj.__dict__
    assert before == {name: tensor_record(value) for name, value in model.named_buffers()}
    manifest = program._m2m_external_quantization_manifest
    assert manifest["schema"] == "m2m.quantization_manifest.v2"
    assert manifest["contract_sha256"] == hashlib.sha256(Path(config.contract_path).read_bytes()).hexdigest()
    assert manifest["policy_sha256"] == hashlib.sha256(Path(config.policy_path).read_bytes()).hexdigest()
    sites = {site["site_id"]: site for site in manifest["sites"]}
    assert sites["proj"]["status"] == "quantized"
    assert sites["proj"]["execution_route"] == "atlas.native_fp8"
    assert sites["head"]["status"] == "skipped"
    calls = manifest["atlas"]["calls"]
    assert len(calls) == 4
    assert len({call["node_id"] for call in calls}) == 4
    linear = [call for call in calls if call["target"] == "atlas.qlinear.default"]
    assert [call["invocation"] for call in linear] == [0, 1]
    for call in linear:
        assert [operand["role"] for operand in call["operands"]] == ["x_codes", "x_exp", "w_codes", "w_exp", "bias"]
        assert call["operands"][0]["node_id"] == call["operands"][1]["node_id"]
        assert [operand["result_index"] for operand in call["operands"][:2]] == [0, 1]
        assert call["operands"][2]["buffer"] == "proj.weight_codes"
        assert call["literals"] == {"out_dtype": "bfloat16", "weight_format": "atlas_e4m3"}
        assert call["results"] == [{"index": 0, "shape": [2, 3, 4], "dtype": "bfloat16"}]
    quantizers = [call for call in calls if call["target"] == "atlas.quantize_act.default"]
    assert all(call["literals"]["rows"] == 32 and call["literals"]["code_dtype"] == "uint8" for call in quantizers)
    assert manifest["atlas"]["buffers"]["proj.weight_codes"]["dtype"] == "uint8"
    assert manifest["atlas"]["buffers"]["proj.weight_scale_exp"]["dtype"] == "int32"
    archive = tmp_path / "replay.pt2"
    torch.export.save(program, archive)
    assert torch.equal(load_program(archive).module()(*inputs), expected)


def test_selected_policy_bytes_must_be_exact(replay):
    model, inputs, _, config = replay
    with pytest.raises(ValueError, match="selected contract/policy bytes"):
        atlas_fp8_adapter(model, inputs, contract_bytes=Path(config.contract_path).read_bytes(),
                          policy_bytes=Path(config.policy_path).read_bytes() + b"\n")


def rewrite_policy(replay, update):
    model, _, _, config = replay
    policy = json.loads(Path(config.policy_path).read_bytes())
    update(policy)
    raw = json.dumps(policy).encode()
    Path(config.policy_path).write_bytes(raw)
    contract = json.loads(Path(config.contract_path).read_bytes())
    contract["files"]["policy.json"].update(sha256=hashlib.sha256(raw).hexdigest(), bytes=len(raw))
    contract["policy_sha256"] = hashlib.sha256(raw).hexdigest()
    Path(config.contract_path).write_text(json.dumps(contract))
    # These tests exercise adapter policy validation, using an explicitly selected
    # new source fixture. Hash-authentication tests retain the original config.
    selected = replace(
        config,
        expected_contract_sha256=hashlib.sha256(Path(config.contract_path).read_bytes()).hexdigest(),
        expected_policy_sha256=hashlib.sha256(raw).hexdigest(),
    )
    return (*replay[:3], selected)


@pytest.mark.parametrize("change,error", [
    (lambda policy: policy.update(pack_fold_summary={}), "folded exports"),
    (lambda policy: policy["sites"]["proj"]["activation"].update(mode="static_per_step"), "mutable activation"),
    (lambda policy: policy["sites"]["proj"].update(pv_pop_shift=1), "folded site"),
    (lambda policy: policy["tensors"].update(count=999), "inventory count"),
])
def test_invalid_policy_features_fail_closed(replay, change, error):
    replay = rewrite_policy(replay, change)
    with pytest.raises(ValueError, match=error):
        invoke(replay)


def test_training_is_refused(replay):
    replay[0].train()
    with pytest.raises(ValueError, match="eval mode"):
        invoke(replay)


def test_conversion_normalizes_before_final_bundle(replay, tmp_path):
    from m2m import convert
    from m2m.capture.bundle import write_bundle
    from atlas_mlir_native.source_normalize import normalize

    model, inputs, _, config = replay
    program = invoke(replay)
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    result = convert(program, inputs, quantization=config, quantization_preapplied=True,
                     capture_trace=True, backend="fx_importer", normalize=normalize,
                     weights_path=str(bundle / "weights.safetensors"))
    assert result.ok, result.diagnostics
    assert "func.call" not in result.mlir_text
    assert "atlas.quantize_act" in result.mlir_text and "atlas.qlinear" in result.mlir_text
    result.module.verify()
    write_bundle(program, inputs, bundle, quant=config, capture_regions=False,
                 capture_trace=True, quantization_preapplied=True, conversion_result=result)
    assert (bundle / "model.mlir").read_text() == result.mlir_text
    manifest = json.loads((bundle / "quantization-manifest.json").read_text())
    assert manifest == program._m2m_external_quantization_manifest
    assert (bundle / "capture_receipt.json").is_file()


def test_duplicate_installed_adapter_fails_closed(replay, tmp_path, monkeypatch):
    duplicate = tmp_path / "duplicate-0.0.1.dist-info"
    duplicate.mkdir()
    (duplicate / "METADATA").write_text("Metadata-Version: 2.1\nName: duplicate\nVersion: 0.0.1\n")
    (duplicate / "entry_points.txt").write_text("[m2m.quantization_adapters]\natlas_fp8 = atlas_m2m_adapter:atlas_fp8_adapter\n")
    monkeypatch.syspath_prepend(str(tmp_path))
    with pytest.raises(ValueError, match="found 2"):
        invoke(replay)


def test_explicit_wrapper_prefix_authenticates_exact_source_names(replay):
    model, inputs, expected, config = replay

    class Boundary(torch.nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model

        def forward(self, value):
            return self.model(value)

    boundary = Boundary(model).eval()
    boundary._atlas_export_dir = model._atlas_export_dir
    boundary._atlas_export_prefix = "model."
    captured = apply_external_quantization(boundary, inputs, config)
    assert torch.equal(captured.module()(*inputs), expected)
    buffers = captured._m2m_external_quantization_manifest["atlas"]["buffers"]
    assert buffers["model.proj.weight_codes"]["source_name"] == "proj.weight_codes"


def test_meta_placeholder_cannot_change_output_dtype(replay):
    replay[0].proj.weight = torch.nn.Parameter(torch.empty((4, 4), dtype=torch.float32, device="meta"),
                                               requires_grad=False)
    with pytest.raises(ValueError, match="meta placeholder dtype or shape"):
        invoke(replay)


def test_tiny_llama_attention_uses_policy_sites_and_uint8(tmp_path):
    import fx_export
    from transformers import LlamaConfig, LlamaForCausalLM

    torch.manual_seed(9)
    config = LlamaConfig(vocab_size=32, hidden_size=16, intermediate_size=32, num_hidden_layers=1,
                         num_attention_heads=2, num_key_value_heads=1, max_position_embeddings=128,
                         tie_word_embeddings=False, use_cache=False)
    config._attn_implementation = "eager"
    model = LlamaForCausalLM(config).to(torch.bfloat16).eval()
    for module in model.modules():
        module._non_persistent_buffers_set.clear()
    algo = algos.make("granularity:w=channel,a=token,rule=nc")
    options = {"crosscheck": "off", "mxu": "ipt", "mxu_impl": "torch"}
    report = algos.quantize(model, "ref_conv", ["lm_head"], options, algo=algo)
    quant_export.export(model, tmp_path, mode="ref_conv", report=report, algo=algo,
                        algo_spec="granularity:w=channel,a=token,rule=nc", options=options, skip=["lm_head"])
    boundary = fx_export.CausalLogits(model).eval()
    inputs = (torch.arange(8, dtype=torch.int64).reshape(1, 8),)
    expected = boundary(*inputs).detach().clone()
    for layer in model.modules():
        if fx_export._is_quant_layer(layer):
            layer.weight = torch.nn.Parameter(torch.empty(layer.weight.shape, dtype=layer.weight.dtype, device="meta"),
                                               requires_grad=False)
    boundary._atlas_export_dir = tmp_path
    boundary._atlas_export_prefix = "model."
    selected = ExternalQuantizationConfig("atlas_fp8", tmp_path / "manifest.json", tmp_path / "policy.json")
    captured = invoke((boundary, inputs, expected, selected))
    assert torch.equal(captured.module()(*inputs), expected)
    extension = captured._m2m_external_quantization_manifest["atlas"]
    assert extension["graph_constants"]
    for name, record in extension["graph_constants"].items():
        assert name not in extension["buffers"]
        assert record["input_kind"] == "CONSTANT_TENSOR"
        assert {key: value for key, value in record.items() if key != "input_kind"} == tensor_record(captured.constants[name])
    calls = extension["calls"]
    by_target = {call["target"] for call in calls}
    assert {"atlas.qk_matmul.default", "atlas.pv_matmul.default"} <= by_target
    for call in calls:
        assert call["site_id"] in json.loads((tmp_path / "policy.json").read_text())["sites"]
        if call["target"] == "atlas.quantize_act.default":
            assert call["results"][0]["dtype"] == "uint8"
            assert call["results"][1]["dtype"] == "int32"
    from m2m import convert
    from atlas_mlir_native.source_normalize import normalize
    converted = convert(captured, inputs, quantization=selected, quantization_preapplied=True,
                        capture_trace=True, backend="fx_importer", normalize=normalize)
    assert converted.ok, converted.diagnostics
    assert "func.call" not in converted.mlir_text
    assert "atlas.qk_matmul" in converted.mlir_text and "atlas.pv_matmul" in converted.mlir_text


def test_layer_policy_identity_must_match_lifted_weights(replay):
    replay[0].proj.layer_name = "head"
    with pytest.raises(ValueError, match="policy identity differs"):
        invoke(replay)


def test_unknown_original_state_is_not_a_graph_literal(replay):
    replay[0].register_buffer("unlisted", torch.tensor(0, dtype=torch.int32))
    with pytest.raises(ValueError, match="state differs.*unlisted"):
        invoke(replay)


def test_static_exponent_tensor_must_match_policy_literal(replay):
    replay = rewrite_policy(replay, lambda policy: policy["sites"]["proj"]["activation"].update(
        mode="static", exp=-3, tensor="proj.weight_scale_exp"))
    with pytest.raises(ValueError, match="static activation tensor differs"):
        invoke(replay)


def test_target_prepare_model_returns_exact_stamped_program(replay):
    from atlas_mlir_native.capture import prepare_model

    model, inputs, expected, selected = replay
    captured, forwarded_inputs = prepare_model(
        model, inputs, selected.contract_path, selected.policy_path,
        expected_contract_sha256=selected.expected_contract_sha256,
        expected_policy_sha256=selected.expected_policy_sha256,
    )
    assert isinstance(captured, torch.export.ExportedProgram)
    assert forwarded_inputs == inputs
    assert all(left is right for left, right in zip(forwarded_inputs, inputs))
    assert torch.equal(captured.module()(*inputs), expected)
    manifest = captured._m2m_external_quantization_manifest
    ids = {node.meta["_m2m_node_id"] for node in captured.graph.nodes}
    assert {call["node_id"] for call in manifest["atlas"]["calls"]} <= ids


def test_custom_quantized_forward_is_not_overwritten(replay):
    original = replay[0].proj.forward
    replay[0].proj.forward = original
    with pytest.raises(ValueError, match="instance forward overrides"):
        invoke(replay)
    assert replay[0].proj.forward is original


def saved_replay(replay, tmp_path):
    program = invoke(replay)
    path = tmp_path / "native.pt2"
    torch.export.save(program, path)
    selected = load_program(path)
    selected._atlas_export_dir = replay[3].contract_path.parent
    selected._atlas_saved_program_path = path
    selected._atlas_saved_program_sha256 = hashlib.sha256(path.read_bytes()).hexdigest()
    return selected


def test_saved_native_entry_preserves_graph_and_source_bytes(replay, tmp_path):
    selected = saved_replay(replay, tmp_path)
    graph = str(selected.graph)
    state = {k: tensor_record(v) for k, v in selected.state_dict.items() if v.device.type != "meta"}
    captured = apply_external_quantization(selected, replay[1], replay[3])
    assert captured is selected
    assert str(captured.graph) == graph
    assert state == {k: tensor_record(v) for k, v in captured.state_dict.items() if v.device.type != "meta"}
    attestation = captured._m2m_external_quantization_manifest["atlas"]["saved_native_program"]
    assert attestation["no_native_calls"] is False
    assert "host_only" not in attestation


def test_saved_host_head_authenticates_bf16_weight(replay, tmp_path):
    original, _, _, config = replay
    class Head(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.head = original.head
        def forward(self, x):
            return self.head(x)
    inputs = (torch.randn(2, 4, dtype=torch.bfloat16),)
    exported = torch.export.export(Head().eval(), inputs)
    path = tmp_path / "host.pt2"
    torch.export.save(exported, path)
    selected = load_program(path)
    selected._atlas_export_dir = config.contract_path.parent
    selected._atlas_saved_program_path = path
    selected._atlas_saved_program_sha256 = hashlib.sha256(path.read_bytes()).hexdigest()
    captured = apply_external_quantization(selected, inputs, config)
    manifest = captured._m2m_external_quantization_manifest
    assert not manifest["atlas"]["calls"]
    assert "host_only" not in manifest["atlas"]["saved_native_program"]
    assert manifest["atlas"]["saved_native_program"]["no_native_calls"] is True
    assert tensor_record(captured.state_dict["head.weight"]) == tensor_record(original.head.weight)


@pytest.mark.parametrize("role", ["contract", "policy"])
@pytest.mark.parametrize("invalid", [None, "", "0" * 63, "G" * 64])
def test_target_prepare_model_requires_source_digests_before_files(tmp_path, role, invalid):
    from atlas_mlir_native.capture import prepare_model

    hashes = {"expected_contract_sha256": "0" * 64, "expected_policy_sha256": "0" * 64}
    hashes[f"expected_{role}_sha256"] = invalid
    # Missing paths and a non-model sentinel prove validation happens before
    # source file access, model mutation, or adapter entry-point selection.
    with patch("m2m.capture.external_quantization._entry_point", side_effect=AssertionError("capture ran")):
        with pytest.raises(ValueError, match="expected source SHA-256"):
            prepare_model(object(), (), tmp_path / "missing-contract", tmp_path / "missing-policy", **hashes)


@pytest.mark.parametrize("case,error", [
    ("codes", "state differs"),
    ("exponent", "state differs"),
    ("ordinary", "state differs"),
    ("source_file", "sha256"),
    ("saved_graph", "saved native graph differs"),
    ("saved_dtype", "saved native graph differs"),
    ("saved_shape", "saved native graph differs"),
    ("contract", "selected contract sha256 differs"),
    ("policy", "selected policy sha256 differs"),
    ("replacement", "selected contract sha256 differs"),
    ("symlink", "export paths must not use symlinks"),
])
def test_mutations_fail_closed(replay, tmp_path, case, error):
    model, inputs, _, config = replay
    if case in {"codes", "exponent", "ordinary"}:
        with torch.no_grad():
            if case == "codes":
                model.proj.weight_codes[0, 0] ^= 1
            elif case == "exponent":
                model.proj.weight_scale_exp[0, 0] += 1
            else:
                model.head.weight[0, 0] += 1
    elif case == "source_file":
        tensor_file = Path(model._atlas_export_dir) / "model.safetensors"
        raw = tensor_file.read_bytes()
        tensor_file.write_bytes(raw[:-1] + bytes([raw[-1] ^ 1]))
    elif case.startswith("saved_"):
        model = saved_replay(replay, tmp_path)
        node = next(n for n in model.graph.nodes if str(n.target) == "atlas.qlinear.default")
        if case == "saved_graph":
            node.args = (*node.args[:-2], "float32", node.args[-1])
        else:
            value = node.meta["val"]
            node.meta["val"] = (torch.empty_like(value, dtype=torch.float32) if case == "saved_dtype"
                                else value.reshape(-1))
    elif case == "symlink":
        alias = tmp_path / "export-alias"
        alias.symlink_to(model._atlas_export_dir, target_is_directory=True)
        model._atlas_export_dir = alias
    elif case == "replacement":
        # Coherent replacement must still fail the originally selected receipt.
        rewrite_policy(replay, lambda policy: policy.update(replacement_marker=True))

    with pytest.raises(ValueError, match=error):
        if case in {"contract", "policy", "replacement"}:
            from atlas_mlir_native.capture import prepare_model

            hashes = {"expected_contract_sha256": config.expected_contract_sha256,
                      "expected_policy_sha256": config.expected_policy_sha256}
            if case != "replacement":
                hashes[f"expected_{case}_sha256"] = "0" * 64
            with patch("m2m.capture.external_quantization._entry_point", side_effect=AssertionError("capture ran")):
                prepare_model(model, inputs, config.contract_path, config.policy_path, **hashes)
        elif case.startswith("saved_"):
            apply_external_quantization(model, inputs, config)
        else:
            invoke(replay)
