#!/usr/bin/env python3
"""Validate fatal runtime faults through bounded, unmodified native Core runs.

These paired probes validate the source contract; observations never derive or
relax timing facts. No model method, exception handler, or unit is patched.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
import re
import sys
from types import SimpleNamespace

from atlas_model_diagnostics import ModelTimingValidationError, model_timing_cli
from atlas_validate_overlap import safe_path

POLICY = "native-core-runtime-fault-pairs-v1"
MAX_CYCLES = 512


@dataclass(frozen=True)
class RuntimeCase:
    name: str
    mechanism: str
    instructions: tuple
    expected_type: str | None = None
    expected_message: str | None = None
    scalar_base: int = 0


def runtime_cases(isa, cfg):
    """Fixed validation coverage; these choices are not derivation inputs."""
    # ECALL detection precedes DELAY stall gating in Core. An ordinary slot
    # must consume the delay before ECALL can prove that engines drained.
    drain = (isa.DELAY(imm=120), isa.ADDI(rd=0, rs1=0, imm=0), isa.ECALL())
    first = isa.VLI_ONE(vd=0, imm=1)
    second = isa.VLI_ONE(vd=2, imm=1)
    return (
        RuntimeCase("issue_busy_positive", "vpu_issue_busy",
                    (first, isa.DELAY(imm=120), second, *drain)),
        RuntimeCase("issue_busy_negative", "vpu_issue_busy", (first, second, *drain),
                    "RuntimeError", r"VPU command vli\.one issued while its issue-busy bit is set"),
        RuntimeCase("bank_positive", "lsu_vmem_bank",
                    (isa.VLOAD(vd=0, rs1=0, imm=0), isa.LW(rd=2, rs1=1, imm=0), *drain),
                    scalar_base=cfg.vmem_bank_bytes),
        RuntimeCase("bank_negative", "lsu_vmem_bank",
                    (isa.VLOAD(vd=0, rs1=0, imm=0), isa.LW(rd=2, rs1=1, imm=0), *drain),
                    "BankConflictError", r"VMEM bank conflict: vload and lw access bank 0"),
        RuntimeCase("mrf_positive", "mrf_reservation",
                    (isa.VLOAD(vd=0, rs1=0, imm=0), isa.VLI_ONE(vd=2, imm=1), *drain)),
        RuntimeCase("mrf_negative", "mrf_reservation",
                    (isa.VLOAD(vd=0, rs1=0, imm=0), isa.VLI_ONE(vd=0, imm=1), *drain),
                    "BankConflictError", r"MRF bank conflict: '[^']+:[0-9]+:vli\.one' accesses register\(s\) \[0\] held by '[^']+:[0-9]+'"),
        RuntimeCase("runtime_positive", "vpu_pair_alignment", (isa.VLI_ALL(vd=0, imm=1), *drain)),
        RuntimeCase("runtime_negative", "vpu_pair_alignment", (isa.VLI_ALL(vd=1, imm=1), *drain),
                    "RuntimeError", r"VPU pair-write destination bank must be even"),
        RuntimeCase("s1_positive", "branch_delay_slot",
                    (isa.JAL(rd=0, imm=4), isa.ADDI(rd=1, rs1=0, imm=1), isa.ECALL())),
        RuntimeCase("s1_negative", "branch_delay_slot",
                    (isa.JAL(rd=0, imm=4), isa.JAL(rd=0, imm=4), isa.ECALL()),
                    "RuntimeError", r"Illegal control-flow instruction 'jal' decoded in a delay-slot position on cycle [0-9]+"),
    )


def _load_model(model):
    root = safe_path(model)
    sys.path.insert(0, str(root))
    import npu_model
    import torch
    if safe_path(Path(npu_model.__file__).parent.parent) != root:
        raise ValueError("imported npu_model differs from --npu-model")
    torch.set_num_threads(1)
    return root


def native_case(case, cfg, *, max_cycles=MAX_CYCLES):
    """Run one complete Core, accepting only its specifically expected fault."""
    from npu_model.hardware.core import Core
    from npu_model.hardware.bank_conflict import BankConflictError
    from npu_model.logging.logger import Logger
    from npu_model.software.program import InstantiableProgram

    logger = SimpleNamespace(**{name: lambda *args, **kwargs: None
                                for name in dir(Logger) if name.startswith("log_")})
    core = Core(cfg, logger)
    result = dict(name=case.name, mechanism=case.mechanism,
                  expectation="reject" if case.expected_type else "positive",
                  words_u32=[insn.to_bytecode() for insn in case.instructions])
    try:
        if core.ignore_runtime_errors is not False:
            raise ModelTimingValidationError("native runtime validation requires ignore_runtime_errors=False")
        core.load_program(InstantiableProgram(list(case.instructions)))
        core.arch_state.write_xrf(1, case.scalar_base)
        while not core.arch_state.halted and core.cycle_count < max_cycles:
            if core.is_finished():
                raise ModelTimingValidationError(f"runtime probe {case.name} exhausted before ECALL")
            try:
                core.tick()
            except (RuntimeError, AssertionError, BankConflictError) as error:
                if core.ignore_runtime_errors is not False:
                    raise ModelTimingValidationError("native runtime validation changed ignore_runtime_errors") from error
                if (type(error).__name__ != case.expected_type or
                        re.fullmatch(case.expected_message or "(?!)", str(error)) is None):
                    raise ModelTimingValidationError(
                        f"runtime probe {case.name} raised unexpected {type(error).__name__}: {error}") from error
                return dict(result, observed="rejected", cycles=core.cycle_count,
                            exception_type=type(error).__name__, exception_message=str(error),
                            ignore_runtime_errors=False)
            if core.ignore_runtime_errors is not False:
                raise ModelTimingValidationError("native runtime validation changed ignore_runtime_errors")
        if case.expected_type:
            raise ModelTimingValidationError(
                f"runtime probe {case.name} did not propagate expected {case.expected_type}: "
                f"{case.expected_message}")
        state = core.arch_state
        if not state.halted or state.halt_reason != "ecall":
            raise ModelTimingValidationError(f"runtime probe {case.name} did not reach ECALL within {max_cycles} cycles")
        if any(unit.has_in_flight for unit in core.exus) or any(state.flags):
            raise ModelTimingValidationError(f"runtime probe {case.name} reached ECALL before engines/DMA drained")
        return dict(result, observed="ecall_drained", cycles=core.cycle_count,
                    ignore_runtime_errors=False)
    finally:
        core.close()


def validate_runtime(model, *, max_cycles=MAX_CYCLES):
    """Return a report only when every positive and negative native control passes."""
    if sys.flags.optimize:
        raise ModelTimingValidationError("native runtime validation requires Python assertions; disable -O/PYTHONOPTIMIZE")
    if type(max_cycles) is not int or not 1 <= max_cycles <= MAX_CYCLES:
        raise ValueError(f"runtime cycle bound must be an integer in 1..{MAX_CYCLES}")
    _load_model(model)
    from npu_model.configs import isa_definition as isa
    from atlas_derive_curve import resolve_curve_path
    from atlas_model_timing import configuration
    cfg, _ = configuration()
    # Use the selected model's supported bytes input, as in native scoring.
    # This binds path resolution before Core construction without changing any
    # model behavior or its curve parameters.
    cfg.dma_memory_params = dict(cfg.dma_memory_params,
                                curves=json.loads(resolve_curve_path(cfg).read_bytes()))
    cases = [native_case(case, cfg, max_cycles=max_cycles) for case in runtime_cases(isa, cfg)]
    return dict(schema="atlas.model_runtime_validation.v1", policy=POLICY,
                authority="validation_only", execution="unmodified_native_core",
                max_cycles_per_case=max_cycles,
                positive_cases=sum(case["expectation"] == "positive" for case in cases),
                negative_cases=sum(case["expectation"] == "reject" for case in cases),
                passed=True, cases=cases)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--npu-model", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    report = validate_runtime(args.npu_model)
    safe_path(args.out).write_text(json.dumps(report, sort_keys=True, indent=2) + "\n")


if __name__ == "__main__":
    model_timing_cli(main)()
