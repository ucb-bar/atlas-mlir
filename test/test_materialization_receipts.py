"""Deterministic receipts require an explicit publication option."""

import json
from pathlib import Path
import pytest

from atlas_mlir_native import binding
from atlas_mlir_native.region_composition import compile_virtual_with_receipt
from atlas_mlir_native.quantized_loop_sizing import publication_record
from atlas_mlir_native.publication_options import publication_options, valid_receipt_timing


def test_pure_pipeline_receipt_is_byte_identical_across_invocations(tmp_path, monkeypatch):
    tools = tmp_path / "tools"
    tools.mkdir()
    for name in ("atlas-opt", "atlas-emit"):
        (tools / name).write_bytes(name.encode())
    monkeypatch.delenv("ATLAS_CAPTURE_POSTSPILL", raising=False)
    monkeypatch.setattr(binding, "_run", lambda *args, **kwargs: "identical checked machine\n")
    results = []
    for index in range(2):
        output = tmp_path / str(index)
        output.mkdir()
        with publication_options(deterministic_receipts=True):
            compile_virtual_with_receipt("identical source", tools=tools, pipeline="mandatory", destination=output)
        results.append((output / "compilation-pipeline.json").read_bytes())
    assert results[0] == results[1]
    receipt = json.loads(results[0])
    assert "elapsed_s" not in receipt
    assert receipt["receipt_mode"] == "deterministic-v1"
    assert receipt["wall_limit_s"] == 30 and receipt["verifier_limits"] == "compiler defaults"


def test_native_sizing_publication_preserves_facts_and_not_operational_cache(tmp_path):
    results = []
    for name in ("first", "second"):
        directory = tmp_path / name
        command = ["atlas-opt", "--lower-atlas-virtual-to-machine=resource-planning-report="
                   + str(directory / "resource.json") + " program-size-limit-bytes=100", "-"]
        record = dict(identity={"source_request_digest": "original", "compiler_proof_limits": "defaults"},
                      physical_instruction_bound=123, physical_instruction_limit=1000000,
                      resource_command=command,
                      sizing_cache={"directory": str(directory), "files": {"operational.json": name}})
        before = json.dumps(record, sort_keys=True)
        assert publication_record(record) == record
        with publication_options(deterministic_receipts=True):
            results.append(publication_record(record))
        assert json.dumps(record, sort_keys=True) == before
    assert results[0] == results[1]
    assert results[0]["identity"] == {"source_request_digest": "original", "compiler_proof_limits": "defaults"}
    assert results[0]["physical_instruction_bound"] == 123
    assert results[0]["resource_command"][1].endswith("resource-planning-report=resource.json program-size-limit-bytes=100")


def test_default_receipt_retains_bounded_elapsed_time(tmp_path, monkeypatch):
    tools = tmp_path / "tools"
    tools.mkdir()
    for name in ("atlas-opt", "atlas-emit"):
        (tools / name).write_bytes(name.encode())
    monkeypatch.delenv("ATLAS_CAPTURE_POSTSPILL", raising=False)
    monkeypatch.setattr(binding, "_run", lambda *args, **kwargs: "machine")
    times = iter((10.0, 11.25))
    monkeypatch.setattr("time.perf_counter", lambda: next(times))
    compile_virtual_with_receipt("source", tools=tools, pipeline="mandatory", destination=tmp_path)
    receipt = json.loads((tmp_path / "compilation-pipeline.json").read_bytes())
    assert receipt["elapsed_s"] == 1.25
    assert "receipt_mode" not in receipt
    assert valid_receipt_timing(receipt)


@pytest.mark.parametrize("receipt", [
    {}, {"elapsed_s": -1}, {"elapsed_s": 31}, {"elapsed_s": True},
    {"elapsed_s": "1"}, {"elapsed_s": float("nan")},
    {"receipt_mode": None, "elapsed_s": 1},
    {"receipt_mode": "unknown", "elapsed_s": 1},
    {"receipt_mode": "deterministic-v1", "elapsed_s": 1},
])
def test_missing_or_invalid_default_timing_is_rejected(receipt):
    assert not valid_receipt_timing(receipt)


@pytest.mark.parametrize("deterministic", [False, True])
def test_receipt_verification_requires_matching_invocation_mode(deterministic):
    default_receipt = {"elapsed_s": 1.25}
    deterministic_receipt = {"receipt_mode": "deterministic-v1"}
    with publication_options(deterministic_receipts=deterministic):
        assert valid_receipt_timing(default_receipt) is (not deterministic)
        assert valid_receipt_timing(deterministic_receipt) is deterministic
        assert not valid_receipt_timing({})
        assert not valid_receipt_timing(dict(deterministic_receipt, elapsed_s=1.25))


def test_publication_option_scope_restores_default_after_failure():
    record = {"sizing_cache": {"directory": "cache"}}
    with pytest.raises(RuntimeError):
        with publication_options(deterministic_receipts=True):
            assert "sizing_cache" not in publication_record(record)
            with publication_options():
                assert publication_record(record) == record
            raise RuntimeError("stop")
    assert publication_record(record) == record


def test_staged_or_ambiguous_native_selection_envelopes_decline(tmp_path):
    from types import SimpleNamespace
    from atlas_mlir_native.selection_materialization import supports_captured_tile

    source = {"schema": "synthetic source"}
    request = SimpleNamespace(record=lambda: source)
    checked = SimpleNamespace(status="selected", check_fingerprint="checked")
    capture = SimpleNamespace(records=[(request, {}, checked)])
    (tmp_path / "source-request.json").write_text(json.dumps(source))
    manifest = {"schema": "atlas.native_resident_graph_compilation.v1", "native_check_fingerprint": "checked"}
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    assert not supports_captured_tile(tmp_path, capture)
    manifest["schema"] = "atlas.native_tile_compilation.v6"
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    assert supports_captured_tile(tmp_path, capture)
    capture.records *= 2
    assert not supports_captured_tile(tmp_path, capture)
    capture.records.clear()
    assert not supports_captured_tile(tmp_path, capture)
