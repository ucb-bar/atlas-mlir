#!/usr/bin/env python3
"""Observe isolated operation timing from an explicitly selected model."""
from __future__ import annotations

from atlas_model_diagnostics import UnsupportedModelTiming

from collections import defaultdict
from contextlib import closing
from io import StringIO
from dataclasses import asdict, replace
import json
import os
from pathlib import Path
from unittest.mock import Mock, patch

from atlas_model_schema import VPU_BLOCKS, MXU_BLOCKS, BINARY, mreg_event, canonical as encoded, sha256


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


def isa_cases(cfg):
    """Select supported operations from IsaSpec and instantiate its named operands."""
    from npu_model.isa_types import Bundled, MatrixReg, ScalarReg, ExponentReg, WeightBuffer, Accumulator, EXU
    names = {mnemonic: ("vpu." + suffix, "vpu") for mnemonic, suffix in VPU_BLOCKS.items()}
    for engine in ("mxu0", "mxu1"):
        names.update({mnemonic + "." + engine: (engine + "." + suffix, engine)
                      for mnemonic, suffix in MXU_BLOCKS.items()})
    names.update({"vload": ("vlsu.vload", "vlsu"), "vstore": ("vlsu.vstore", "vlsu"),
                  "vtrpose.xlu": ("xlu.transpose", "xlu"), "lw": ("scalar_lsu.load", "scalar_lsu"),
                  "seld": ("scalar_lsu.scale_load", "scalar_lsu"), "sw": ("scalar_lsu.store", "scalar_lsu")})
    engines = {EXU.VECTOR: "vpu", EXU.MATRIX_SYSTOLIC: "mxu0", EXU.MATRIX_INNER: "mxu1", EXU.XLU: "xlu"}
    found = {}
    for mnemonic, operation in cfg.isa.operations.items():
        if mnemonic not in names:
            if operation.exu in engines:
                raise UnsupportedModelTiming(f"ISA operation has no timing vocabulary: {mnemonic}")
            continue
        name, engine = names[mnemonic]
        if operation.exu in engines and engines[operation.exu] != engine:
            raise UnsupportedModelTiming(f"ISA execution unit differs for {mnemonic}")
        def operand(param):
            if isinstance(param, Bundled):
                return f"{operand(param.imm)}({operand(param.reg)})"
            typ, role = param.inner, param.repr
            if typ is MatrixReg:
                value = 4 if role == "vd" and mnemonic != "vstore" else 2 if role == "vs2" and mnemonic in BINARY else 0
                return f"m{value}"
            if typ is ScalarReg:
                return "x4" if role == "rd" else "x1" if role == "rs2" else "x0"
            if typ is ExponentReg:
                return "e4" if role == "rd" else "e0"
            if typ is WeightBuffer:
                return "w0"
            if typ is Accumulator:
                return "acc0"
            if issubclass(typ, int) and role == "imm":
                return "0"
            raise UnsupportedModelTiming(f"unsupported ISA operand {mnemonic}: {typ.__name__}/{role}")
        assembly = mnemonic + " " + ", ".join(operand(param) for param in operation.params)
        found[name] = (name, engine, assembly)
    if len(found) != len(names):
        raise UnsupportedModelTiming("selected ISA is missing supported timing operations")
    return [found[name] for name, _ in names.values()]


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
            grouped[mreg_event(uop.insn.mnemonic, "vpu", int(bank), write)].append(unit.cycle - 1)

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
    """Check a representative native VPU execution against source-derived facts."""
    check = dict(name="vpu.add", passed=False)
    observed = []
    try:
        block = observe_vpu(cfg)
        observed.append(block)
        candidate = candidates["vpu.add"]
        if block != {key: candidate[key] for key in ("events", "first_free_age")}:
            raise RuntimeError("source-derived VPU timing differs from native execution")
        check["passed"] = True
    except (ValueError, RuntimeError) as error:
        check["error"] = str(error)
    report = dict(passed=check["passed"],
                  observed_runs=len(observed), checks=[check])
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "validation-report.json").write_text(json.dumps(report, sort_keys=True, indent=2) + "\n")
    if not report["passed"]:
        raise UnsupportedModelTiming(f"derived facts validation failed; see {output_dir / 'validation-report.json'}")
    return report, observed
