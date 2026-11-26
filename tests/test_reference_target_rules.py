"""Target selection boundaries and authenticated independent output comparisons."""
import hashlib
import json
import numpy as np
import pytest
from atlas_mlir_native.reference_eval import Tensor
from atlas_mlir_native.reference_eval.target_rules import select_mxu
from atlas_mlir_native.reference_eval.compare import OutputComparison
from atlas_mlir_native.reference_eval import device


def test_already_quantized_route_uses_mxu1_for_every_extent():
    assert [select_mxu(k) for k in (0, 1, 31, 32, 33, 64, 2048)] == [
        "mxu1", "mxu1", "mxu1", "mxu1", "mxu1", "mxu1", "mxu1"]
    for k in (-1, True, 1.5, "32"):
        with pytest.raises(ValueError):
            select_mxu(k)


@pytest.mark.parametrize("k", [1, 32, 33, 64, 2048])
@pytest.mark.parametrize("policy_unit,expected", [(0, "mxu0"), (1, "mxu1"),
                                                   ("mxu0", "mxu0"), ("mxu1", "mxu1")])
def test_tile_route_uses_explicit_policy_independent_of_k(k, policy_unit, expected):
    assert select_mxu(k, route="tile", policy_unit=policy_unit) == expected


def test_route_and_policy_validation():
    with pytest.raises(ValueError, match="explicit policy"):
        select_mxu(64, route="tile")
    with pytest.raises(ValueError, match="unknown compiler route"):
        select_mxu(64, route="unknown")
    for unit in (True, False, -1, 2, 1.0, "MXU1"):
        with pytest.raises(ValueError, match="policy unit"):
            select_mxu(64, route="tile", policy_unit=unit)
    with pytest.raises(ValueError, match="requires MXU1"):
        select_mxu(64, policy_unit=0)
    assert select_mxu(64, route="already-quantized", policy_unit=1) == "mxu1"


def test_device_uses_route_and_explicit_override(monkeypatch):
    monkeypatch.setattr(device, "_authenticate", lambda root: (root, ()))
    monkeypatch.setattr(device, "_qualify", lambda *args: None)
    rules = dict(npu_model_revision=device.MODEL_REVISION[:7], numerics="rtl", k_order="ascending-k32", npu_model_root="unused")
    assert device._rules(rules, contraction=True, k_extent=32) == "mxu1"
    assert device._rules(rules, contraction=True, k_extent=33) == "mxu1"
    assert device._rules(dict(rules, selected_unit="mxu1"), contraction=True, k_extent=33) == "mxu1"
    assert device._rules(dict(rules, contraction_route="tile", policy_unit=0), contraction=True, k_extent=64) == "mxu0"
    with pytest.raises(ValueError, match="explicit policy"):
        device._rules(dict(rules, contraction_route="tile"), contraction=True, k_extent=64)


def comparison(tmp_path):
    np.savez(tmp_path / "expected.npz", output_0=np.array([0x3f80, 0x4000], np.uint16))
    receipt = dict(status="EXPECTED", expected_sha256=hashlib.sha256((tmp_path / "expected.npz").read_bytes()).hexdigest())
    data = json.dumps(receipt).encode()
    (tmp_path / "reference.json").write_bytes(data)
    nodes = [{"args": [[dict(node_id="producer", value_id="v0")]]}]
    return OutputComparison(nodes, tmp_path, hashlib.sha256(data).hexdigest())


def test_compare_match_and_first_raw_bit_mismatch(tmp_path):
    check = comparison(tmp_path)
    check(dict(id="unrelated"), None)
    check(dict(id="producer"), Tensor(np.array([0x3f80, 0x4000], np.uint16), "bf16"))
    assert check.record["compared"] == ["output_0"]
    with pytest.raises(ValueError, match="reference mismatch"):
        check(dict(id="producer"), Tensor(np.array([0x3f80, 0x4001], np.uint16), "bf16"))
    assert check.record["mismatch"]["actual_bits"] == "0x4001"
    assert check.record["mismatch"]["expected_bits"] == "0x4000"
    assert check.record["mismatch"]["index"] == [1]


def test_compare_refuses_archive_drift(tmp_path):
    comparison(tmp_path)
    (tmp_path / "expected.npz").write_bytes(b"changed")
    data = (tmp_path / "reference.json").read_bytes()
    with pytest.raises(ValueError, match="archive differs"):
        OutputComparison([], tmp_path, hashlib.sha256(data).hexdigest())


def cli_capture(root, **trace_fields):
    """Write an authenticated one-cast capture; extra trace fields vary only the trace digest."""
    root.mkdir()
    value = dict(id="x:v0", dtype="int64", shape=[2], device="cpu")
    nodes = [dict(id="x", op="placeholder", target="x", args=[], kwargs={}, results=[value], input_kind="USER_INPUT"),
             dict(id="cast", op="call_function", target="aten.to.dtype_layout", args=[dict(node_id="x", value_id="x:v0")],
                  kwargs={"dtype": {"kind": "dtype", "value": "torch.float32"}},
                  results=[dict(id="cast:v0", dtype="float32", shape=[2], device="cpu")]),
             dict(id="out", op="output", target="output", args=[[dict(node_id="cast", value_id="cast:v0")]],
                  kwargs={}, results=[])]
    (root / "frontend-trace.json").write_text(json.dumps(dict(graphs=dict(quantized=dict(nodes=nodes)), **trace_fields)))
    (root / "weights.safetensors").write_bytes((2).to_bytes(8, "little") + b"{}")
    np.savez(root / "inputs.npz", in0=np.array([1, -3], np.int64))
    artifacts = {name: dict(bytes=(root / name).stat().st_size, sha256=hashlib.sha256((root / name).read_bytes()).hexdigest())
                 for name in ("frontend-trace.json", "weights.safetensors", "inputs.npz")}
    (root / "capture_receipt.json").write_text(json.dumps(dict(artifacts=artifacts)))
    return hashlib.sha256((root / "capture_receipt.json").read_bytes()).hexdigest()


def run_cli(monkeypatch, capsys, *argv):
    from atlas_mlir_native.reference_eval.__main__ import main
    monkeypatch.setattr("sys.argv", ["reference_eval", *map(str, argv)])
    status = main()
    capsys.readouterr()
    return status, json.loads((argv[argv.index("--out") + 1] / "reference.json").read_text())


def test_cli_consumes_its_own_capture_receipt_and_refuses_other_trace(tmp_path, monkeypatch, capsys):
    trusted = cli_capture(tmp_path / "capture")
    status, baseline = run_cli(monkeypatch, capsys, tmp_path / "capture", "--capture", "--receipt-sha256", trusted,
                               "--out", tmp_path / "baseline")
    assert status == 0 and baseline["status"] == "EXPECTED"
    assert baseline["source_trace_sha256"] == baseline["authentication"]["artifacts"]["frontend-trace.json"]
    reference = hashlib.sha256((tmp_path / "baseline" / "reference.json").read_bytes()).hexdigest()
    compare = ("--compare-reference", tmp_path / "baseline", "--compare-receipt-sha256", reference)
    status, replay = run_cli(monkeypatch, capsys, tmp_path / "capture", "--capture", "--receipt-sha256", trusted,
                             *compare, "--out", tmp_path / "replay")
    assert status == 0 and replay["status"] == "EXPECTED"
    assert replay["comparison"]["compared"] == ["output_0"] and replay["comparison"]["mismatch"] is None
    assert replay["expected_sha256"] == baseline["expected_sha256"]
    other = cli_capture(tmp_path / "other", revision="other")
    status, refused = run_cli(monkeypatch, capsys, tmp_path / "other", "--capture", "--receipt-sha256", other,
                              *compare, "--out", tmp_path / "refused")
    assert status == 1 and refused["status"] == "REFUSED"
    assert refused["reason"] == "comparison source trace differs from authenticated capture"
