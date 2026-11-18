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

from . import toolchain
from .tile_demo import _fresh

MERLIN_REVISION = "650cc3400858515bd2722db2603bf35ee6a05f2f"


@dataclass(frozen=True)
class BuildResult:
    out: Path
    application: Path
    report: Path
    verification: Path


def _run(command, *, log: Path, timeout: float, env=None):
    """Abort the process tree on timeout; never retry a different numerical path."""
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("w") as stream:
        process = subprocess.Popen(list(map(str, command)), stdout=stream,
                                   stderr=subprocess.STDOUT, env=env, start_new_session=True)
        try:
            status = process.wait(timeout=timeout)
        except (subprocess.TimeoutExpired, KeyboardInterrupt) as error:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait()
            if isinstance(error, KeyboardInterrupt):
                raise
            raise toolchain.ToolFailure(f"{log.stem} timed out; build aborted. See {log}") from error
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


def capture(source: Path, *, out: Path, dtype: str = "bf16", contract=None,
            quant_policy=None, expected_contract_sha256=None, expected_policy_sha256=None,
            capture_python=None, timeout=1800, env=None):
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
         log=out.parent / (out.name + ".log"), timeout=timeout, env=env)
    return out


def build(source: Path | None = None, *, out: Path, precision_policy: Path,
          captured: Path | None = None, snapshot: Path | None = None,
          dtype="bf16", contract=None, quant_policy=None, expected_contract_sha256=None,
          expected_policy_sha256=None, capture_python=None, target_source=None,
          already_quantized=False, application_parameters="external",
          application_name="application", application_memory_bytes=None,
          io_memory_bytes=None, recipe=None, board=None, application_residency=False,
          application_region_loop=False, force=False, timeout=1800) -> BuildResult:
    """Compile a live loader or a materialized Merlin capture, then replay verification.

    Precision policy must be selected by the caller and bound to exact source IR.
    There is no guessed policy, alternate importer, or failure fallback.
    """
    if (source is None) == (captured is None):
        raise ValueError("give exactly one loader or --capture directory")
    if application_memory_bytes is None or application_memory_bytes <= 0:
        raise ValueError("application_memory_bytes must be an explicit positive arena budget")
    precision_policy = Path(precision_policy).resolve(strict=True)
    if captured is not None and any(x is not None for x in (contract, quant_policy,
            expected_contract_sha256, expected_policy_sha256)):
        raise ValueError("export contract options belong to live capture, not --capture")
    source = Path(source).resolve(strict=True) if source is not None else None
    captured = Path(captured).resolve(strict=True) if captured is not None else None
    snapshot = Path(snapshot).resolve(strict=True) if snapshot is not None else None
    out = Path(out).resolve()
    # Never let --force delete any of the selected inputs.
    for path in (source, captured, snapshot, precision_policy, contract, quant_policy):
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
    command = [sys.executable, "-m", "merlin.targetgen.cli"]
    steps = []
    def run(name, arguments):
        _run([*command, *arguments], log=out / "logs" / (name + ".log"), timeout=timeout, env=env)
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
        if snapshot is None:
            snapshot = out / "snapshot"
            run("native-build", ["native-build", "--engine", "merlin_native", "--support", "atlas",
                "--source-revision", MERLIN_REVISION, "--cargo-target-dir", out / "cargo",
                *geometry, "--out", snapshot])
        common = ["--engine", "merlin_native", "--support", "atlas", "--snapshot", snapshot,
                  "--target-source", target, "--compile-inputs", staged, "--application",
                  "--application-parameters", application_parameters, "--mode", "diagnostic", *geometry]
        if quantized:
            common += ["--already-quantized"]
        for flag, value in (("--recipe", recipe), ("--board", board)):
            if value:
                common += [flag, value]
        if application_residency:
            common += ["--application-residency"]
        compile_options = ["--application-name", application_name, "--abi", "auto", "--deterministic-receipts"]
        if application_memory_bytes is not None:
            compile_options += ["--application-memory-bytes", str(application_memory_bytes)]
        if application_region_loop:
            compile_options += ["--application-region-loop"]
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
