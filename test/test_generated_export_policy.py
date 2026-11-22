"""Generated export policy reaches the existing staged application compiler."""
import json
from copy import deepcopy
from types import SimpleNamespace

import pytest
from atlas_mlir_native.binding import TARGET_IDENTITY, AtlasNativeBinding
from atlas_mlir_native.generated_export_policy import (
    adapter_for_prepared_source,
    derive_generated_export_policy,
)
from merlin.compiler.frontend.inputs import stage_compile_inputs
from merlin.compiler.semantic.target_binding import NativeCompilationError
from merlin.development.targetgen.execution.native.native_application import _source
from test_export_contract import exported
from test_generated_policy import bundle


def test_generated_export_policy_reaches_native_prepared_source_boundary(tmp_path, monkeypatch):
    _, _, _, capture = bundle(tmp_path / "bundle", producer_policy=exported(), bind_external_parameters=True)
    policy = derive_generated_export_policy(capture.parent / "export", capture)
    assert policy["generated_policy"]["parameter_mode"] == "external"
    assert set(policy["generated_policy"]["compatibility_policy_fields"]) == {"already_quantized"}
    assert "row_softmax" not in policy["generated_policy"]["compatibility_policy_fields"]
    document = tmp_path / "precision.json"
    document.write_text(json.dumps(policy))
    directory = tmp_path / "staged"
    stage_compile_inputs(capture, directory, precision_policy=document)
    args = SimpleNamespace(compile_inputs=directory, mode="diagnostic", linalg_entry=None,
                           application_parameters="external")
    staged, source, adapter, state, receipt, parameters = _source(args, AtlasNativeBinding(), TARGET_IDENTITY)
    assert state is None and parameters is None
    assert source.digest() == policy["generated_policy"]["graph_sha256"] == receipt["after_digest"]
    plan = adapter.plan(source)
    assert len(plan.selected) == 1
    assert dict(plan.selected[0].metadata)["kind"] == "already_quantized"
    with monkeypatch.context() as withheld:
        withheld.setattr(AtlasNativeBinding, "application_adapter_for_prepared_source", None)
        with pytest.raises(NativeCompilationError):
            _source(args, AtlasNativeBinding(), TARGET_IDENTITY)
    edited = deepcopy(policy)
    edited["generated_policy"]["authorized_precision_changes"] = []
    with pytest.raises((ValueError, NativeCompilationError)):
        adapter_for_prepared_source(edited, source=source, staged=staged,
                                   original_source_sha256=staged.original_source_sha256)
    edited = deepcopy(policy)
    edited["checked_export_contract"]["allowed_mxu"] = ["mxu0"]
    with pytest.raises(NativeCompilationError, match="exporter contract changed"):
        adapter_for_prepared_source(edited, source=source, staged=staged,
                                   original_source_sha256=staged.original_source_sha256)


def test_generated_export_policy_rejects_export_authorization_change(tmp_path):
    producer = exported()
    producer["accumulate"]["mxu"] = "sa"
    _, _, _, capture = bundle(tmp_path / "bundle", producer_policy=producer)
    with pytest.raises(ValueError, match="accumulate.mxu"):
        derive_generated_export_policy(capture.parent / "export", capture)


def test_existing_export_intake_defaults_to_graph_generated_policy(tmp_path, monkeypatch):
    from atlas_mlir import driver, export_intake
    from atlas_mlir_native import export_loader

    _, _, _, capture = bundle(tmp_path / "bundle", producer_policy=exported())
    fixture = SimpleNamespace(inputs=(), outputs=())
    authenticated = SimpleNamespace(export_dir=capture.parent / "export", root=capture.parent,
                                    fixtures=(fixture,), metadata={}, inventory={}, verify_unchanged=lambda: None)
    monkeypatch.setattr(export_loader, "load_export", lambda path: authenticated)
    # Capture execution is outside this local orchestration test; the returned
    # receipt is genuinely staged and authenticated by the normal policy path.
    monkeypatch.setattr(driver, "capture", lambda *args, **kwargs: capture)
    def unwanted_archive(*args):
        raise AssertionError("unused unary-math archive must not authorize a source graph")
    monkeypatch.setattr(export_intake, "host_math_contract", unwanted_archive)
    out = tmp_path / "intake"
    out.mkdir()
    arguments = export_intake.prepare(capture.parent, None, out)
    policy = json.loads((out / "precision-policy.json").read_text())
    assert policy["schema"] == "atlas.generated_application_precision.v1"
    assert policy["generated_policy"]["parameter_mode"] == "external"
    assert "--capture" in arguments and "--precision-policy" in arguments
