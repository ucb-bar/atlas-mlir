"""Thin orchestration of Merlin's authenticated capture and application compiler."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

from . import toolchain
from .provenance import PINS
from .tile_demo import _fresh

MERLIN_REVISION = PINS["merlin"]


@dataclass(frozen=True)
class BuildResult:
    out: Path
    application: Path
    report: Path
    verification: Path


def _activity(paths):
    """A value that changes whenever a file or directory under paths is created, removed, or modified."""
    count = newest = 0
    pending = [str(p) for p in paths]
    while pending:
        path = pending.pop()
        try:
            stat = os.stat(path, follow_symlinks=False)
        except FileNotFoundError:
            continue
        count, newest = count + 1, max(newest, stat.st_mtime_ns)
        if os.path.isdir(path) and not os.path.islink(path):
            with os.scandir(path) as entries:
                pending.extend(entry.path for entry in entries)
    return count, newest


def _run(command, *, log: Path, timeout: float, env=None, watch=()):
    """Abort the process tree after timeout seconds in which neither its log nor a watched path changed.

    A step that keeps writing its outputs is making progress, however long it runs; never retry a different
    numerical path.
    """
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("w") as stream:
        process = subprocess.Popen(list(map(str, command)), stdout=stream,
                                   stderr=subprocess.STDOUT, env=env, start_new_session=True)
        last, quiet_since = None, time.monotonic()
        try:
            while True:
                try:
                    status = process.wait(timeout=min(30.0, timeout / 4))
                    break
                except subprocess.TimeoutExpired as error:
                    seen, now = _activity([log, *watch]), time.monotonic()
                    if seen != last:
                        last, quiet_since = seen, now
                    elif now - quiet_since >= timeout:
                        raise error
        except (subprocess.TimeoutExpired, KeyboardInterrupt) as error:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait()
            if isinstance(error, KeyboardInterrupt):
                raise
            raise toolchain.ToolFailure(f"{log.stem} made no progress for {timeout:g} s; build aborted. See {log}") from error
    if status:
        tail = "\n".join(log.read_text(errors="replace").splitlines()[-12:])
        raise toolchain.ToolFailure(f"{log.stem} failed ({status}); see {log}\n{tail}")


def _authenticated(path, expected, label):
    if path is None or expected is None:
        raise ValueError(f"{label} requires a file and its independently selected expected SHA-256")
    path = Path(path).resolve(strict=True)
    if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
        raise ValueError(f"{label} SHA-256 differs from the selected export")
    return path


def _require_user_name(env):
    """PyTorch export calls getpass.getuser(), which needs a name in the environment or a passwd entry."""
    if any((os.environ if env is None else env).get(name) for name in ("LOGNAME", "USER", "LNAME", "USERNAME")):
        return
    import pwd
    try:
        pwd.getpwuid(os.getuid())
    except KeyError:
        raise ValueError(f"user id {os.getuid()} has no passwd entry and USER is unset, so PyTorch capture cannot "
                         "name the user; add a passwd entry for this user (for example with useradd) or set USER") from None


def capture(source: Path, *, out: Path, dtype: str = "bf16", contract=None,
            quant_policy=None, expected_contract_sha256=None, expected_policy_sha256=None,
            capture_python=None, timeout=1800, env=None):
    _require_user_name(env)
    source = Path(source).resolve(strict=True)
    if source.suffix != ".py":
        raise ValueError("Merlin intake requires a live .py loader exposing get_model_and_inputs(); saved .pt2 intake is not qualified")
    out = Path(out).resolve()
    if out.exists():
        raise FileExistsError(f"capture output must be fresh: {out}")
    quantized = any(x is not None for x in (contract, quant_policy,
                    expected_contract_sha256, expected_policy_sha256))
    args = []
    if quantized:
        contract = _authenticated(contract, expected_contract_sha256, "contract")
        quant_policy = _authenticated(quant_policy, expected_policy_sha256, "quantization policy")
        args = ["--already-quantized", "--quant-adapter", "atlas_fp8", "--contract", contract,
                "--policy", quant_policy, "--expected-contract-sha256", expected_contract_sha256,
                "--expected-policy-sha256", expected_policy_sha256]
    _run([capture_python or sys.executable, "-m", "atlas_mlir._capture_worker",
          "--loader", source, "--dtype", dtype, "--out", out, "--materialize-bundle", *args],
         log=out.parent / (out.name + ".log"), timeout=timeout, env=env, watch=(out,))
    return out


def build(source: Path | None = None, *, out: Path, precision_policy: Path,
          captured: Path | None = None, snapshot: Path | None = None,
          dtype="bf16", contract=None, quant_policy=None, expected_contract_sha256=None,
          expected_policy_sha256=None, capture_python=None, target_source=None,
          already_quantized=False, application_parameters="external",
          application_name="application", application_memory_bytes=None, application_state=None,
          io_memory_bytes=None, recipe=None, board="sophialake", application_residency=False,
          application_deployment=None, host_clock_hz=None, measured_host_bytes=None,
          force=False, timeout=1800) -> BuildResult:
    """Compile a live loader or a materialized Merlin capture, then replay verification.

    Precision policy must be selected by the caller and bound to exact source IR.
    There is no guessed policy, alternate importer, or failure fallback.
    """
    if (source is None) == (captured is None):
        raise ValueError("give exactly one loader or --capture directory")
    if application_memory_bytes is None or application_memory_bytes <= 0:
        raise ValueError("application_memory_bytes must be an explicit positive arena budget")
    if application_deployment not in (None, "onchip", "board"):
        raise ValueError("application deployment must be onchip, board, or unset")
    if measured_host_bytes is not None and application_deployment != "onchip":
        raise ValueError("a measured host bounds only an on-chip deployment")
    precision_policy = Path(precision_policy).resolve(strict=True)
    application_state = Path(application_state).resolve(strict=True) if application_state is not None else None
    if captured is not None and any(x is not None for x in (contract, quant_policy,
            expected_contract_sha256, expected_policy_sha256)):
        raise ValueError("export contract options belong to live capture, not --capture")
    source = Path(source).resolve(strict=True) if source is not None else None
    captured = Path(captured).resolve(strict=True) if captured is not None else None
    snapshot = Path(snapshot).resolve(strict=True) if snapshot is not None else None
    out = Path(out).resolve()
    # Never let --force delete any of the selected inputs.
    for path in (source, captured, snapshot, precision_policy, application_state, contract, quant_policy):
        if path is not None and Path(path).resolve().is_relative_to(out):
            raise ValueError("build inputs must be outside the output directory")
    target = toolchain.data_root(target_source).resolve()
    bins = toolchain.bin_dir(target)
    from .provenance import environment_record
    provenance = environment_record()
    _fresh(out, force)
    env = os.environ.copy()
    env.update(ATLAS_OOT_BIN_DIR=str(bins), PYTHONNOUSERSITE="1", OMP_NUM_THREADS="1",
               MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")
    command = [sys.executable, "-m", "merlin.development.targetgen.cli"]
    steps = []
    def run(name, arguments):
        _run([*command, *arguments], log=out / "logs" / (name + ".log"), timeout=timeout, env=env, watch=(out,))
        steps.append(name)
    report = {"schema": "atlas_mlir.build_report.v2", "status": "building",
              "route": "merlin-model2mlir-application", "qualification": "diagnostic",
              "environment": provenance, "steps": steps,
              "target": {"tools": {name: hashlib.sha256((bins / name).read_bytes()).hexdigest()
                                      for name in ("atlas-opt", "atlas-emit")},
                         "inventory_sha256": hashlib.sha256((target / "docs/selected-variant-inventory.json").read_bytes()).hexdigest()}}
    report_path = out / "report.json"
    try:
        if captured is None:
            captured = capture(source, out=out / "capture", dtype=dtype, contract=contract,
                quant_policy=quant_policy, expected_contract_sha256=expected_contract_sha256,
                expected_policy_sha256=expected_policy_sha256, capture_python=capture_python,
                timeout=timeout, env=env)
            steps.append("capture")
        capture_record = captured / "atlasc-environment.json"
        if capture_record.is_file():
            report["capture_environment"] = json.loads(capture_record.read_text())
        meta = json.loads((captured / "meta.json").read_text())
        quantized = already_quantized or contract is not None or meta.get("capture_quantization") == "already_materialized"
        if precision_policy.suffix == ".py":
            # A caller-owned policy factory can bind its explicit choices to this
            # capture. Never rewrite a JSON policy's existing source identity.
            spec = importlib.util.spec_from_file_location("_atlasc_precision_policy", precision_policy)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            source_hash = hashlib.sha256((captured / "model.mlir").read_bytes()).hexdigest()
            selected_policy = module.get_precision_policy(source_hash)
            precision_policy = out / "precision-policy.json"
            precision_policy.write_text(json.dumps(selected_policy, sort_keys=True, indent=2) + "\n")
        staged = out / "compile-inputs"
        run("stage-capture", ["stage-capture", "--capture", captured,
                             "--precision-policy", precision_policy, "--out", staged])
        geometry = ["--io-memory-bytes", str(io_memory_bytes)] if io_memory_bytes else []
        if application_deployment == "onchip":
            from atlas_mlir_native.provider_deployment import WINDOW_ADDRESS, WINDOW_BYTES
            if io_memory_bytes not in (None, WINDOW_BYTES):
                raise ValueError("onchip I/O budget differs from provider window")
            geometry = ["--io-memory-base", str(WINDOW_ADDRESS),
                        "--io-memory-bytes", str(WINDOW_BYTES),
                        "--io-memory-capacity-base", str(WINDOW_ADDRESS),
                        "--io-memory-capacity-bytes", str(WINDOW_BYTES)]
        elif application_deployment == "board":
            # The device window ends at the end of the declared DDR; the I/O budget, when given, is its size.
            from atlas_mlir_native.soc import SOC
            ddr = SOC.memory_region("ddr")
            size = ddr.capacity_bytes if io_memory_bytes is None else io_memory_bytes
            if not 0 < size <= ddr.capacity_bytes:
                raise ValueError("board I/O budget exceeds the declared DDR")
            geometry = ["--io-memory-base", str(ddr.end - size), "--io-memory-bytes", str(size),
                        "--io-memory-capacity-base", str(ddr.end - size), "--io-memory-capacity-bytes", str(size)]
        if snapshot is None:
            snapshot = out / "snapshot"
            run("native-build", ["native-build", "--engine", "merlin_native", "--support", "atlas",
                "--source-revision", MERLIN_REVISION, "--cargo-target-dir", out / "cargo",
                *geometry, "--out", snapshot])
        common = ["--engine", "merlin_native", "--support", "atlas", "--snapshot", snapshot,
                  "--target-source", target, "--compile-inputs", staged, "--application",
                  "--application-parameters", application_parameters, "--mode", "diagnostic", *geometry]
        if application_state is not None:
            common += ["--application-state", application_state]
        if quantized:
            common += ["--already-quantized"]
        # Board presets describe DDR geometry. Onchip deployment uses the
        # provider's device window; the packaging layer still owns the board.
        compile_board = None if application_deployment else board
        # A board deployment is Merlin's bounded deployment, placed in the board's DDR instead of the on-chip window.
        for flag, value in (("--recipe", recipe), ("--board", compile_board),
                            ("--application-deployment", application_deployment and "onchip")):
            if value:
                common += [flag, value]
        if application_residency:
            common += ["--application-residency"]
        if host_clock_hz is not None:
            common += ["--host-clock-hz", str(host_clock_hz)]
        if measured_host_bytes is not None:
            # The deploy linked this application's on-chip host once; Merlin stages it when it exceeds the provider host.
            common += ["--measured-host-bytes", str(measured_host_bytes)]
        if application_deployment:
            # Dense reshapes are storage aliases, so no deployment copies them through the device.
            common += ["--application-buffer-views"]
        compile_options = ["--application-name", application_name, "--abi", "auto", "--deterministic-receipts"]
        if application_memory_bytes is not None:
            compile_options += ["--application-memory-bytes", str(application_memory_bytes)]
        application = out / "application"
        run("native-compile", ["native-compile", *common, *compile_options, "--out", application])
        verification = out / "verification.json"
        run("native-verify", ["native-verify", *common, "--artifact", application, "--out", verification])
        report.update(status="verified", application="application", verification="verification.json",
                      precision_policy_sha256=hashlib.sha256(precision_policy.read_bytes()).hexdigest(),
                      artifacts={str(p.relative_to(out)): hashlib.sha256(p.read_bytes()).hexdigest()
                                 for directory in (application, staged)
                                 for p in sorted(directory.rglob("*")) if p.is_file()})
    except (Exception, KeyboardInterrupt) as error:
        report.update(status="failed", error=str(error))
        raise
    finally:
        report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return BuildResult(out, application, report_path, verification)


# Python callers use the same loader/capture contract as the command-line interface.
compile = build
