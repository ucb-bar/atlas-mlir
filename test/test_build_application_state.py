"""The public build path generates its policy and state from the export and compiles only already-quantized captures."""
import json
from types import SimpleNamespace

import pytest

from atlas_mlir import cli, driver


def test_cli_forwards_the_export_and_has_no_policy_or_state_override(tmp_path, monkeypatch, capsys):
    calls = []
    monkeypatch.setattr(cli, "build", lambda *args, **kwargs: (
        calls.append(kwargs) or SimpleNamespace(application="application", report="report.json")))
    arguments = ["build", "--capture", str(tmp_path / "capture"), "--export", str(tmp_path / "export"),
                 "--application-memory-bytes", "4096"]
    assert cli.main(arguments) == 0
    assert calls[0]["export"] == tmp_path / "export"
    for option in ("--precision-policy", "--application-state", "--already-quantized"):
        with pytest.raises(SystemExit):
            cli.main([*arguments, option, str(tmp_path / "policy.json")])
    capsys.readouterr()


def build_inputs(tmp_path, monkeypatch, quantization="already_materialized", state=True):
    captured = tmp_path / "capture"
    captured.mkdir()
    (captured / "meta.json").write_text(json.dumps({"capture_quantization": quantization}))
    export = tmp_path / "export"
    export.mkdir()
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
    derived = []

    def generated(export_dir, capture_dir, out):
        derived.append((export_dir, capture_dir))
        policy = out / "precision-policy.json"
        policy.write_text("{}")
        if not state:
            return policy, None
        (out / "application-state.json").write_text("{}")
        return policy, out / "application-state.json"

    monkeypatch.setattr(driver, "_generated_policy", generated)
    return dict(captured=captured, export=export, snapshot=snapshot, application_memory_bytes=4096), derived


def test_driver_generates_one_policy_and_forwards_its_state_to_compile_and_verify(tmp_path, monkeypatch):
    arguments, derived = build_inputs(tmp_path, monkeypatch)
    commands = []
    monkeypatch.setattr(driver, "_run", lambda command, **kwargs: commands.append(command))
    driver.build(out=tmp_path / "out", **arguments)
    assert derived == [(arguments["export"], arguments["captured"])]
    stage, = [command for command in commands if "stage-capture" in command]
    assert stage[stage.index("--precision-policy") + 1] == tmp_path / "out" / "precision-policy.json"
    for action in ("native-compile", "native-verify"):
        command, = [command for command in commands if action in command]
        assert command[command.index("--application-state") + 1] == tmp_path / "out" / "application-state.json"
        assert "--already-quantized" in command


def test_driver_without_generated_state_passes_no_state(tmp_path, monkeypatch):
    arguments, _ = build_inputs(tmp_path, monkeypatch, state=False)
    commands = []
    monkeypatch.setattr(driver, "_run", lambda command, **kwargs: commands.append(command))
    driver.build(out=tmp_path / "out", **arguments)
    assert not any("--application-state" in command for command in commands)


def test_driver_refuses_a_capture_that_is_not_already_quantized(tmp_path, monkeypatch):
    arguments, derived = build_inputs(tmp_path, monkeypatch, quantization="applied")
    monkeypatch.setattr(driver, "_run", lambda command, **kwargs: pytest.fail("no step may run"))
    with pytest.raises(ValueError, match="compiles only already-quantized captures"):
        driver.build(out=tmp_path / "out", **arguments)
    assert derived == []
    assert json.loads((tmp_path / "out" / "report.json").read_text())["status"] == "failed"


def test_driver_refuses_inputs_that_cannot_name_a_quantized_export(tmp_path, monkeypatch):
    arguments, _ = build_inputs(tmp_path, monkeypatch)
    loader = tmp_path / "loader.py"
    loader.write_text("")
    for options, message in (
            (dict(arguments, export=None), "--capture needs the --export"),
            (dict(arguments, captured=None, source=loader, export=None), "compiles only already-quantized exports"),
            (dict(arguments, captured=None, source=loader), "--export belongs to --capture")):
        with pytest.raises(ValueError, match=message):
            driver.build(out=tmp_path / "out", **options)


def test_driver_protects_the_export_inside_forced_output(tmp_path, monkeypatch):
    arguments, _ = build_inputs(tmp_path, monkeypatch)
    out = tmp_path / "out"
    out.mkdir()
    arguments["export"] = out / "export"
    arguments["export"].mkdir()
    with pytest.raises(ValueError, match="build inputs must be outside"):
        driver.build(out=out, force=True, **arguments)
    assert arguments["export"].is_dir()
