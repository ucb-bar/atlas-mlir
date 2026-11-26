"""Saved export intake authenticates source artifacts without graph execution."""
import hashlib
import json
import struct

import pytest

from atlas_mlir_native.export_loader import ExportLoadError, load_export


def dump(path, value):
    path.write_text(json.dumps(value))


def entry(path):
    return {"sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "bytes": path.stat().st_size}


def tensors(path, records):
    header, data = {}, b""
    for name, dtype, shape, raw in records:
        header[name] = {"dtype": dtype, "shape": shape, "data_offsets": [len(data), len(data) + len(raw)]}
        data += raw
    encoded = json.dumps(header).encode()
    path.write_bytes(struct.pack("<Q", len(encoded)) + encoded + data)


@pytest.fixture
def saved(tmp_path):
    export, fx = tmp_path / "export", tmp_path / "fx"
    export.mkdir()
    fx.mkdir()
    tensors(export / "model.safetensors", [
        ("layer.weight_codes", "U8", [1, 2], b"\x38\x40"),
        ("layer.weight_scale_exp", "I32", [1, 1], struct.pack("<i", -4)),
    ])
    tensors(fx / "fixtures.safetensors", [
        ("case_0.input_0", "BF16", [1, 2], b"\x80\x3f\x00\x40"),
        ("case_0.output_0", "BF16", [1, 1], b"\x80\x40"),
    ])
    policy = {"schema": "atlas_torchao.execution_policy.v1", "operand": {"dtype": "f8E4M3", "rounding": "rne"},
              "scale": {"encoding": "e8m0_exponent_int32"}, "sites": {
                  "layer": {"status": "quantized", "weight": {"codes": "layer.weight_codes",
                             "scale_exp": "layer.weight_scale_exp", "codes_shape": [1, 2]}}}}
    dump(export / "policy.json", policy)
    dump(fx / "policy.json", policy)
    dump(export / "manifest.json", {"schema": "atlas_torchao.export_manifest.v1",
         "files": {name: entry(export / name) for name in ("model.safetensors", "policy.json")},
         "policy_sha256": entry(export / "policy.json")["sha256"]})
    (fx / "model.pt2").write_bytes(b"deliberately not a torch archive: never executed")
    dump(fx / "buffer_inventory.json", {
        "wrapper.layer.weight_codes": {"dtype": "torch.uint8", "shape": [1, 2],
            "sha256": hashlib.sha256(b"\x38\x40").hexdigest()},
        "wrapper.layer.weight_scale_exp": {"dtype": "torch.int32", "shape": [1, 1],
            "sha256": hashlib.sha256(struct.pack("<i", -4)).hexdigest()},
    })
    dump(fx / "graph_inspection.json", {})
    dump(fx / "fx_manifest.json", {"schema": "atlas_torchao.fx_bundle.v1",
         "provenance": {"export_manifest_sha256": entry(export / "manifest.json")["sha256"]},
         "native_fp8": {"storage": "uint8", "weight_format": "atlas_e4m3"},
         "files": {name: entry(fx / name) for name in
                   ("model.pt2", "fixtures.safetensors", "policy.json", "buffer_inventory.json", "graph_inspection.json")}})
    dump(tmp_path / "verify_bundle.json", {"ok": True, "failed": [], "checks": {"saved": {"ok": True}}})
    return tmp_path


def rewrite_fx(saved, transform):
    p = saved / "fx/fx_manifest.json"
    value = json.loads(p.read_text())
    transform(value)
    dump(p, value)


def test_bundle_raw_storage_and_inferred_contract(saved):
    bundle = load_export(saved)
    assert bundle.graph_path == saved / "fx/model.pt2"
    assert bundle.metadata["already_quantized"] is True
    assert bundle.metadata["target_contract_sha256"] is None
    assert bundle.metadata["source_contract_sha256"]
    assert bundle.fixtures[0].inputs[0].dtype == "BF16"
    assert bundle.fixtures[0].inputs[0].read_bytes() == b"\x80\x3f\x00\x40"
    assert bundle.fixtures[0].outputs[0].shape == (1, 1)
    bundle.verify_unchanged()
    assert load_export(saved / "export").root == saved
    assert load_export(saved / "fx").root == saved


@pytest.mark.parametrize("name", ["model.pt2", "fixtures.safetensors", "policy.json", "buffer_inventory.json", "graph_inspection.json"])
def test_changed_inventory_member_refuses(saved, name):
    p = saved / "fx" / name
    data = p.read_bytes()
    p.write_bytes(bytes([data[0] ^ 1]) + data[1:])
    with pytest.raises(ExportLoadError, match="hash mismatch"):
        load_export(saved)


def test_incomplete_graph_inventory_refuses(saved):
    rewrite_fx(saved, lambda x: x["files"].pop("model.pt2"))
    with pytest.raises(ExportLoadError, match="omits model.pt2"):
        load_export(saved)


def test_export_cross_binding_refuses(saved):
    rewrite_fx(saved, lambda x: x["provenance"].update(export_manifest_sha256="0" * 64))
    with pytest.raises(ExportLoadError, match="binding mismatch"):
        load_export(saved)


@pytest.mark.parametrize("name", ["../outside", "/tmp/outside", "sub/../../outside", "sub\\outside"])
def test_escaping_inventory_refuses_before_read(saved, name):
    rewrite_fx(saved, lambda x: x["files"].update({name: {"sha256": "0" * 64, "bytes": 0}}))
    with pytest.raises(ExportLoadError, match="inventory path"):
        load_export(saved)


def test_symlink_refuses(saved):
    graph = saved / "fx/model.pt2"
    graph.unlink()
    graph.symlink_to(saved / "verify_bundle.json")
    with pytest.raises(ExportLoadError, match="symlink"):
        load_export(saved)


@pytest.mark.parametrize("change", [
    lambda x: x.update(ok=False),
    lambda x: x.update(failed=["saved"]),
    lambda x: x["checks"]["saved"].update(ok=False),
    lambda x: x.update(checks={}),
])
def test_failed_verification_refuses(saved, change):
    p = saved / "verify_bundle.json"
    value = json.loads(p.read_text())
    change(value)
    dump(p, value)
    with pytest.raises(ExportLoadError, match="verification did not pass"):
        load_export(saved)


def test_conflicting_outer_manifest_refuses(saved):
    dump(saved / "bundle_manifest.json", {"files": {"fx/model.pt2": {"sha256": "0" * 64, "bytes": 45}}})
    with pytest.raises(ExportLoadError, match="mismatch"):
        load_export(saved)


def test_changed_after_intake_refuses(saved):
    bundle = load_export(saved)
    (saved / "fx/model.pt2").write_bytes(b"mutated")
    with pytest.raises(ExportLoadError, match="changed after intake"):
        bundle.verify_unchanged()


def test_malformed_fixture_storage_refuses(saved):
    tensors(saved / "fx/fixtures.safetensors", [("case_0.input_0", "BF16", [9], b"\x00\x00")])
    rewrite_fx(saved, lambda x: x["files"].update({"fixtures.safetensors": entry(saved / "fx/fixtures.safetensors")}))
    with pytest.raises(ExportLoadError, match="shape/length"):
        load_export(saved)


def test_missing_outputs_refuses(saved):
    tensors(saved / "fx/fixtures.safetensors", [("case_0.input_0", "BF16", [1], b"\x00\x00")])
    rewrite_fx(saved, lambda x: x["files"].update({"fixtures.safetensors": entry(saved / "fx/fixtures.safetensors")}))
    with pytest.raises(ExportLoadError, match="output indices"):
        load_export(saved)


def test_duplicate_manifest_keys_refuse(saved):
    p = saved / "fx/fx_manifest.json"
    p.write_text('{"schema": 1, "schema": 2}')
    with pytest.raises(ExportLoadError, match="duplicate JSON"):
        load_export(saved)


def test_intake_does_not_import_torch(saved, monkeypatch):
    import builtins
    original = builtins.__import__
    def guarded(name, *args, **kwargs):
        if name == "torch" or name.startswith("torch."):
            raise AssertionError("framework import during source authentication")
        return original(name, *args, **kwargs)
    monkeypatch.setattr(builtins, "__import__", guarded)
    load_export(saved)


def test_graph_buffer_content_must_match_source(saved):
    p = saved / "fx/buffer_inventory.json"
    value = json.loads(p.read_text())
    value["wrapper.layer.weight_codes"]["sha256"] = "0" * 64
    dump(p, value)
    rewrite_fx(saved, lambda x: x["files"].update({"buffer_inventory.json": entry(p)}))
    with pytest.raises(ExportLoadError, match="graph/source tensor hash mismatch"):
        load_export(saved)
