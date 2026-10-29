#!/usr/bin/env python3
"""Observe isolated operation timing from an explicitly selected model."""
from __future__ import annotations

from atlas_model_diagnostics import UnsupportedModelTiming

from collections import defaultdict
from contextlib import closing
from io import StringIO
from dataclasses import asdict, replace
import os
from pathlib import Path
from unittest.mock import Mock, patch

from atlas_model_schema import mreg_event, canonical as encoded, sha256


def manifest(model_root: Path, subdir: str, suffixes=None, sort=False):
    records = []
    root = model_root / subdir
    for directory, children, filenames in os.walk(root):
        children[:] = sorted(name for name in children if
                             not any(blocked in name.lower() for blocked in ("hammer", "vlsi"))
                             and (not suffixes or name != "__pycache__"))
        for name in sorted(filenames):
            if any(blocked in name.lower() for blocked in ("hammer", "vlsi")):
                continue
            path = Path(directory) / name
            if not suffixes or path.suffix in suffixes:
                records.append({"path": str(path.relative_to(model_root)), "sha256": sha256(path.read_bytes())})
    if sort:
        records.sort(key=lambda record: record["path"])
    return sha256(encoded(records)), records


def source_digest(model_root: Path) -> tuple[str, list[dict]]:
    return manifest(model_root, "npu_model", {".py", ".json", ".yaml"})


def numerical_data(model_root: Path):
    return manifest(model_root, "npu_model/configs/data", sort=True)


def configuration():
    from npu_model.configs.hardware.default import DefaultHardwareConfig
    cfg = DefaultHardwareConfig()
    cfg.arch_state_config = replace(cfg.arch_state_config, numerics="rtl", randomize_init=False)
    return cfg, configuration_values(cfg)


def configuration_values(cfg):
    """Canonicalize the actual configuration passed to model execution."""
    values = {}
    for key in dir(cfg):
        if key.startswith("_"):
            continue
        value = getattr(cfg, key)
        if key == "arch_state_config":
            values[key] = asdict(value)
        elif isinstance(value, (str, int, float, bool, dict, type(None))):
            values[key] = value
        elif isinstance(value, type):
            values[key] = value.__module__ + "." + value.__qualname__
    values["ignore_runtime_errors"] = False
    return values


def compress(ages: list[int]) -> dict:
    ages = sorted(ages)
    if not ages:
        raise ValueError("empty group")
    step = ages[1] - ages[0] if len(ages) > 1 else 1
    if step <= 0 or any(b - a != step for a, b in zip(ages, ages[1:])):
        raise ValueError(f"event group is not an even positive sweep: {ages}")
    return {"count": len(ages), "first_age": ages[0], "last_age": ages[-1], "step": step}


def observe_vpu(cfg):
    """Observe one ordinary VPU row sweep through the native port checker."""
    from npu_model.hardware.arch_state import ArchState
    from npu_model.hardware.vpu import VectorExecutionUnit
    from npu_model.logging.logger import Logger
    from npu_model.software.instruction import Uop
    from npu_model.util.converter import stream_to_instrs

    instruction, = stream_to_instrs(StringIO("vadd.bf16 m4, m0, m2"))
    with closing(ArchState(cfg.arch_state_config)) as state:
        unit = VectorExecutionUnit("vpu", Mock(spec=Logger), state, config=cfg)
        uop = Uop(instruction, pc=0)
        grouped = defaultdict(list)
        original = state.conflict_checker.access_mreg

        def access(cycle, bank, row, write, owner):
            original(cycle, bank, row, write, owner)
            grouped[mreg_event(uop.insn.mnemonic, int(bank), write)].append(unit.cycle - 1)

        with patch.object(state.conflict_checker, "access_mreg", access):
            for age in range(4096):
                if age and not unit.has_in_flight:
                    return dict(events={name: compress(ages) for name, ages in sorted(grouped.items())},
                                first_free_age=age)
                unit.tick(uop if age == 0 else None)
                if age == 0 and unit.total_instructions != 1:
                    raise RuntimeError("VPU validation command was not accepted")
    raise RuntimeError("VPU validation command did not drain")


def validate_derived(cfg, candidates, releases, output_dir):
    """Compare vadd MREG events and first-free age; other blocks use source guards."""
    check = dict(name="vadd.bf16", passed=False)
    observed = []
    try:
        block = observe_vpu(cfg)
        observed.append(block)
        candidate = candidates["vadd.bf16"]
        if block != {key: candidate[key] for key in ("events", "first_free_age")}:
            raise RuntimeError("vadd timing differs from native execution")
        check["passed"] = True
    except (ValueError, RuntimeError) as error:
        check["error"] = str(error)
    report = dict(passed=check["passed"],
                  observed_runs=len(observed), checks=[check])
    if not report["passed"]:
        raise UnsupportedModelTiming("VPU native timing differs from facts: " + check["error"])
    return report, observed
