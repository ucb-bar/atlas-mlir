"""The public build path retains generated policy state at compilation and replay."""
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from atlas_mlir import cli, driver


def test_cli_accepts_and_forwards_application_state(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(cli, "build", lambda *args, **kwargs: (
        calls.append(kwargs) or SimpleNamespace(application="application", report="report.json")))
    state = tmp_path / "state.json"
    assert cli.main(["build", "--capture", str(tmp_path / "capture"),
        "--precision-policy", str(tmp_path / "policy.json"),
        "--application-memory-bytes", "4096", "--application-state", str(state)]) == 0
    assert calls[0]["application_state"] == state


def build_inputs(tmp_path, monkeypatch):
    captured = tmp_path / "capture"
    captured.mkdir()
    (captured / "meta.json").write_text("{}")
    policy = tmp_path / "policy.json"
    policy.write_text(json.dumps({"schema": "atlas.generated_application_precision.v1"}))
    state = tmp_path / "state.json"
    state.write_text("{}")
    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    target = tmp_path / "target"
    (target / "bin").mkdir(parents=True)
    (target / "docs").mkdir()
    for name in ("atlas-opt", "atlas-emit"):
        (target / "bin" / name).write_bytes(b"tool")
    (target / "docs" / "selected-variant-inventory.json").write_text("{}")
    monkeypatch.setattr(driver.toolchain, "data_root", lambda *_: target)
    monkeypatch.setattr(driver.toolchain, "bin_dir", lambda *_: target / "bin")
    monkeypatch.setattr("atlas_mlir.provenance.environment_record", lambda: {})
    monkeypatch.setattr(driver, "_fresh", lambda out, force: out.mkdir())
    return dict(captured=captured, precision_policy=policy, snapshot=snapshot,
                application_state=state, application_memory_bytes=4096)


def test_driver_forwards_same_state_to_compile_and_verify(tmp_path, monkeypatch):
    arguments = build_inputs(tmp_path, monkeypatch)
    commands = []
    monkeypatch.setattr(driver, "_run", lambda command, **kwargs: commands.append(command))
    driver.build(out=tmp_path / "out", **arguments)
    for action in ("native-compile", "native-verify"):
        command, = [command for command in commands if action in command]
        assert command[command.index("--application-state") + 1] == arguments["application_state"]


def test_driver_protects_state_inside_forced_output(tmp_path, monkeypatch):
    arguments = build_inputs(tmp_path, monkeypatch)
    out = tmp_path / "out"
    out.mkdir()
    arguments["application_state"] = out / "state.json"
    arguments["application_state"].write_text("{}")
    with pytest.raises(ValueError, match="build inputs must be outside"):
        driver.build(out=out, force=True, **arguments)
    assert arguments["application_state"].read_text() == "{}"
