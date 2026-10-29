#!/usr/bin/env python3
"""Observe isolated operation timing from an explicitly selected model."""
from __future__ import annotations

from atlas_model_source import UnsupportedModelTiming

from collections import defaultdict
from contextlib import ExitStack, closing
import inspect
from dataclasses import asdict, replace
from unittest.mock import Mock, patch

from atlas_model_source import BINARY, mreg_event, canonical as encoded, sha256


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


def native_uop(cfg, mnemonic):
    """Choose disjoint register operands for an isolated native instruction."""
    from npu_model.isa_types import MatrixReg, ScalarReg, ExponentReg, WeightBuffer, Accumulator
    from npu_model.software.instruction import Uop

    operation = cfg.isa.operations[mnemonic]
    operands = {}
    for role, param in inspect.signature(operation).parameters.items():
        if param.annotation is MatrixReg:
            operands[role] = (4 if role == "vd" and mnemonic != "vstore" else
                              2 if role == "vs2" and mnemonic in BINARY else 0)
        elif param.annotation in (ScalarReg, ExponentReg):
            operands[role] = 4 if role == "rd" else 1 if role == "rs2" else 0
        elif param.annotation in (WeightBuffer, Accumulator, int):
            operands[role] = 0
        else:
            raise ValueError(f"unsupported native probe operand: {mnemonic}/{role}")
    return Uop(operation(**operands), pc=0)


def observe_case(cfg, mnemonic):
    """Observe one isolated instruction through the native execution unit."""
    from npu_model.hardware.arch_state import ArchState
    from npu_model.hardware.exu import StagedExecution
    from npu_model.hardware.lsu import LoadStoreUnit
    from npu_model.hardware.mxu import MatrixExecutionUnitInner, MatrixExecutionUnitSystolic
    from npu_model.hardware.vpu import VectorExecutionUnit
    from npu_model.hardware.xlu import CrossLaneExecutionUnit
    from npu_model.logging.logger import Logger

    uop = native_uop(cfg, mnemonic)
    classes = (VectorExecutionUnit, LoadStoreUnit, CrossLaneExecutionUnit,
               MatrixExecutionUnitSystolic, MatrixExecutionUnitInner)
    unit_class, = [cls for cls in classes if cls.__name__ == str(uop.insn.exu)]
    with closing(ArchState(cfg.arch_state_config)) as state, ExitStack() as patches:
        unit = unit_class(str(uop.insn.exu), Mock(spec=Logger), state, config=cfg)
        state.write_erf(0, 127)
        grouped = defaultdict(list)

        def tap(target, method, group):
            original = getattr(target, method)
            def invoke(*args, **kwargs):
                result = original(*args, **kwargs)
                key = group(*args, **kwargs)
                if key is not None:
                    grouped[key].append(unit.cycle - 1)
                return result
            patches.enter_context(patch.object(target, method, invoke))

        def mreg(cycle, bank, row, write, owner):
            if unit_class is VectorExecutionUnit:
                return mreg_event(mnemonic, int(bank), write)
            if unit_class is CrossLaneExecutionUnit:
                return "write" if write else "read"
            key = "mreg_write" if write else "mreg_read"
            if unit_class is LoadStoreUnit:
                return key
            index = (int(bank) % 2 if write or ".bf16." in mnemonic else
                     0 if mnemonic.startswith("vmatmul") else 1)
            return key + str(index)

        tap(state.conflict_checker, "access_mreg", mreg)
        for method, key in (("read_vmem", "vmem_read"), ("write_vmem", "vmem_write"),
                            ("write_xrf", "load_write"), ("write_erf", "scale_write")):
            tap(state, method, lambda *args, key=key: key)
        if hasattr(unit, "mxu"):
            def sampled(staged, location, index, value):
                if staged.state is state and location[0] == "acc":
                    return "acc_store" if mnemonic.startswith("vmatpop") else "acc_read"
            def retrieved(staged, location):
                if staged.state is state:
                    if location[0] == "wb":
                        return "weight_write"
                    if location[0] == "acc":
                        return "acc_load" if mnemonic.startswith("vmatpush") else "acc_write"
            tap(StagedExecution, "sample", sampled)
            tap(StagedExecution, "result", retrieved)
        for age in range(4096):
            if age and not unit.has_in_flight:
                return dict(events={key: compress(ages) for key, ages in sorted(grouped.items())},
                            first_free_age=age)
            unit.tick(uop if age == 0 else None)
            if age == 0 and unit.total_instructions != 1:
                raise RuntimeError("native validation instruction was not accepted")
    raise RuntimeError("native validation instruction did not drain")


def validate_derived(cfg, candidates):
    """Compare all extracted operation events and first-free ages with native ticks."""
    if not candidates:
        raise UnsupportedModelTiming("native validation has no operations")
    for name, candidate in candidates.items():
        try:
            block = observe_case(cfg, name)
            if block != {key: candidate[key] for key in ("events", "first_free_age")}:
                raise RuntimeError("native timing differs from facts")
        except (ValueError, RuntimeError) as error:
            raise UnsupportedModelTiming(f"{name}: {error}") from error
    return len(candidates)
