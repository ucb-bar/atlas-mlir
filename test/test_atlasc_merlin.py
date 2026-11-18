"""Driver boundary tests: refuse destructive reuse, policy drift and timeouts."""
import os
from pathlib import Path
import sys
from unittest.mock import patch

import pytest
from atlas_mlir import cli, driver, toolchain


def test_explicit_target_never_borrows_other_installation(tmp_path):
    with patch.dict(os.environ, {}, clear=True):
        with pytest.raises(toolchain.ToolchainError, match="not found"):
            toolchain.bin_dir(tmp_path)


def test_wheel_share_resolves_sibling_binaries(tmp_path):
    package = tmp_path / "atlas_mlir"
    (package / "bin").mkdir(parents=True)
    (package / "share").mkdir()
    for name in ("atlas-opt", "atlas-emit"):
        (package / "bin" / name).touch()
    with patch.dict(os.environ, {}, clear=True), patch.object(toolchain, "PACKAGE", package):
        assert toolchain.bin_dir(package / "share") == package / "bin"


def test_export_mismatch_refused_before_worker_runs(tmp_path):
    loader = tmp_path / "loader.py"
    loader.touch()
    contract = tmp_path / "manifest.json"
    contract.write_text('{}')
    with patch.object(driver, "_run") as run, pytest.raises(ValueError, match="SHA-256 differs"):
        driver.capture(loader, out=tmp_path / "capture", contract=contract,
                       expected_contract_sha256="0" * 64)
    run.assert_not_called()
    assert not (tmp_path / "capture").exists()


def test_force_cannot_delete_inputs_within_output(tmp_path):
    (tmp_path / ".atlasc").touch()
    policy = tmp_path / "policy.json"
    policy.write_text('{}')
    with pytest.raises(ValueError, match="inputs must be outside"):
        driver.build(captured=tmp_path, precision_policy=policy, out=tmp_path,
                     force=True, application_memory_bytes=1048576)
    assert policy.read_text() == '{}'


def test_timeout_aborts_without_producing_a_success_artifact(tmp_path):
    done = tmp_path / 'done'
    command = [sys.executable, '-c',
               'import time,pathlib; time.sleep(2); pathlib.Path(__import__("sys").argv[1]).touch()', str(done)]
    with pytest.raises(toolchain.ToolFailure, match="timed out; build aborted"):
        driver._run(command, log=tmp_path / 'compile.log', timeout=0.1)
    assert not done.exists()


def test_saved_export_is_not_live_intake(tmp_path):
    pt2 = tmp_path / 'saved.pt2'
    pt2.touch()
    with pytest.raises(ValueError, match="live .py loader"):
        driver.capture(pt2, out=tmp_path / 'capture')


def test_doctor_detects_changed_packaged_runtime(tmp_path):
    import hashlib
    import json
    from atlas_mlir.provenance import build_record
    runtime = tmp_path / 'share/runtime/atlas_host.h'
    runtime.parent.mkdir(parents=True)
    runtime.write_text('original')
    (tmp_path / 'build-record.json').write_text(json.dumps({'files': {
        'share/runtime/atlas_host.h': hashlib.sha256(runtime.read_bytes()).hexdigest()}}))
    with patch.object(toolchain, 'PACKAGE', tmp_path):
        assert build_record()['files']
        runtime.write_text('altered')
        with pytest.raises(toolchain.ToolchainError, match='hash mismatch'):
            build_record()


@pytest.mark.parametrize("failure", ["toolchain", "cache"])
def test_doctor_distinguishes_rust_failures_from_capture_readiness(failure, tmp_path, monkeypatch, capsys):
    from merlin.semantic_compiler import snapshot

    bridge = tmp_path / "installed bridge"
    monkeypatch.setattr(snapshot, "packaged_egg_bridge_path", lambda: bridge)
    monkeypatch.setattr(cli, "environment_record", lambda **_: {})
    monkeypatch.setattr(toolchain, "bin_dir", lambda: tmp_path)

    commands = []

    def run(command, step):
        commands.append(command)
        if command[0] == "cargo" and command[1] == "--version":
            if failure == "toolchain":
                raise toolchain.ToolFailure("rustup could not choose a default toolchain")
            return "cargo 1.90.0"
        if command[0] == "rustc":
            return "rustc 1.90.0\nhost: x86_64-unknown-linux-gnu\n"
        if command[:2] == ["cargo", "fetch"]:
            assert "--offline" in command and "--locked" in command
            assert command[command.index("--manifest-path") + 1] == str(bridge / "Cargo.toml")
            raise toolchain.ToolFailure("no matching package named `egg` found")
        return ""

    monkeypatch.setattr(toolchain, "run", run)
    assert cli.main(["doctor", "--capture"]) == 0
    assert all(command[0] not in {"cargo", "rustc"} for command in commands)
    assert cli.main(["doctor", "--capture", "--build"]) == 1
    error = capsys.readouterr().err
    if failure == "toolchain":
        assert "rustup default stable" in error
        assert "could not choose a default toolchain" in error
    else:
        assert "no matching package named `egg` found" in error
        assert "cargo fetch --locked" in error
        assert f"'{bridge / 'Cargo.toml'}'" in error
