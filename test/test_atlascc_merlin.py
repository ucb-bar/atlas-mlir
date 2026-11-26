"""Driver boundary tests: refuse destructive reuse, policy drift and timeouts."""
import os
from pathlib import Path
import sys
from unittest.mock import patch

import pytest
from atlas_mlir import driver, toolchain


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
    (tmp_path / ".atlascc").touch()
    kept = tmp_path / "export.json"
    kept.write_text('{}')
    with pytest.raises(ValueError, match="inputs must be outside"):
        driver.build(captured=tmp_path, export=tmp_path, out=tmp_path,
                     force=True, application_memory_bytes=1048576)
    assert kept.read_text() == '{}'


def test_timeout_aborts_without_producing_a_success_artifact(tmp_path):
    done = tmp_path / 'done'
    command = [sys.executable, '-c',
               'import time,pathlib; time.sleep(2); pathlib.Path(__import__("sys").argv[1]).touch()', str(done)]
    with pytest.raises(toolchain.ToolFailure, match="made no progress"):
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
