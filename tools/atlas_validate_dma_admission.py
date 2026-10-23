"""Bounded native qualification of frozen asynchronous DMA admission semantics.

Native executions check issue traces and complete architectural state against
serial references. They never select gaps or derive completion/overlap windows.
"""
from __future__ import annotations

from copy import deepcopy
from io import StringIO
import math
from pathlib import Path
import sys
from unittest.mock import Mock

from atlas_derive_dma_admission import derive_dma_admission
from atlas_derive_overlap import safe_path
from atlas_model_diagnostics import ModelTimingValidationError
from atlas_model_schema import canonical, sha256


POLICY = "source-asynchronous-dma-boundaries-v1"
# These bound qualification work and choose mechanisms. They never enter the
# compiler's admission semantics or replace source-derived timing facts.
VALIDATION_POLICY = dict(classification="d", serial_compute_delay_imm=255,
                         native_cycle_limit=50000, response_probe_lead_cycles=8,
                         transfer_boundary="min(max_transfer_beats,max_in_flight+1)",
                         large_transfer="max_transfer_beats")


def _semantics(value):
    return {key: item for key, item in value.items() if key not in {"validation", "producer_sources"}}


def sample_plan(per_op):
    """Freeze representative mechanism boundaries before any model execution."""
    result = []
    def add(name, commands, *, gaps=None, beats=32, second_bank=False, reject=None):
        result.append(dict(id=name, commands=commands, gaps=gaps or [1] * (len(commands) - 1),
                           beats=beats, second_bank=second_bank, rejection=reject))
    def dma(store, channel=0):
        # Two independent VMEM/DRAM spans, even in the same physical bank.
        return (f"dma.store x{4 + 3 * channel}, x{5 + 3 * channel}, x3, ch{channel}" if store else
                f"dma.load x{5 + 3 * channel}, x{4 + 3 * channel}, x3, ch{channel}")
    compute = {"scalar": "addi x17, x0, 7", "vpu": "vli.all m0, 16256", "xlu": "vtrpose.xlu m4, m6",
               "mxu0": "vmatmul.mxu0 acc0, m8, w0", "mxu1": "vmatmul.mxu1 acc0, m8, w0"}
    for unit, command in compute.items():
        for store in (False, True):
            for reverse in (False, True):
                commands = [dma(store), command]
                add(f"compute-{unit}-{int(store)}-{int(reverse)}", commands[::-1] if reverse else commands)
    config = per_op["model"]["configuration"]
    max_beats = config["dma_max_transfer_bytes"] // config["dma_beat_bytes"]
    boundary_beats = min(max_beats, config["dma_max_in_flight"] + 1)
    for first in (False, True):
        for second in (False, True):
            for bank in (False, True):
                add(f"channels-{int(first)}-{int(second)}-{int(bank)}", [dma(first), dma(second, 1)],
                    beats=boundary_beats, second_bank=bank)
    for op in ("vload m12, 0(x15)", "vstore m12, 0(x15)", "lw x16, 0(x15)", "sw x16, 0(x15)"):
        for store in (False, True):
            for reverse in (False, True):
                commands = [dma(store), op]
                add("lsu-" + op.split()[0] + "-" + str(int(store)) + "-" + str(int(reverse)),
                    commands[::-1] if reverse else commands, beats=max_beats)
    # Source-derived latency chooses a *probe context*, not an admitted gap.
    # The LSU stream must overlap a load's returning beat for grant denial.
    curve = min(per_op["dma_timing"]["curves"], key=lambda row: (abs(row["read_percentage"] - 100), row["read_percentage"]))
    probe_gap = max(1, math.ceil(curve["unloaded_latency_cycles"]) - VALIDATION_POLICY["response_probe_lead_cycles"])
    add("load-response-grant-denial", [dma(False), "vload m12, 0(x15)"], gaps=[probe_gap], beats=max_beats)
    # A wait does not reset the curve controller; the final two channels run
    # against preceding mixed history, retaining scalar/VMEM disjointness.
    add("warm-mixed-history", [dma(True), "dma.wait ch0", dma(False), dma(True, 1)], beats=max_beats)
    add("operand-capture", [dma(False), "addi x4, x0, 0", "addi x5, x0, 0", "addi x3, x0, 32"])
    add("same-channel-busy", [dma(False), dma(False)], reject="already set")
    add("zero-transfer", [dma(False)], beats=0, reject="sizes must be multiples")
    add("oversized-transfer", [dma(False)], beats=max_beats + 1, reject="sizes must be multiples")
    return result


def _run(cfg, sample, *, nonzero, serial=False):
    import torch
    from npu_model.hardware.core import Core
    from npu_model.hardware.dma import DmaExecutionUnit
    from npu_model.logging.logger import Logger
    from npu_model.software.program import InstantiableProgram
    from npu_model.util.converter import stream_to_instrs

    commands = sample["commands"]
    assembly, pcs, pending = [], [], set()
    for index, command in enumerate(commands):
        if not serial and index:
            remaining = sample["gaps"][index - 1] - 1
            while remaining:
                occupied = min(4096, remaining)
                assembly.append("delay " + str(occupied - 1))
                remaining -= occupied
        pcs.append(len(assembly))
        assembly.append(command)
        if command.startswith(("dma.load", "dma.store")):
            pending.add(int(command.rsplit("ch", 1)[1]))
        elif command.startswith("dma.wait"):
            pending.discard(int(command.rsplit("ch", 1)[1]))
        if serial:
            if command.startswith(("dma.load", "dma.store")):
                channel = int(command.rsplit("ch", 1)[1])
                assembly.append("dma.wait ch" + str(channel))
                pending.discard(channel)
            elif not command.startswith("dma.wait"):
                assembly.append("delay " + str(VALIDATION_POLICY["serial_compute_delay_imm"]))
    assembly.extend("dma.wait ch" + str(channel) for channel in sorted(pending))
    assembly.append("ecall")
    instructions = stream_to_instrs(StringIO("\n".join(assembly) + "\n"))
    core = Core(config=cfg, logger=Mock(spec=Logger))
    try:
        if core.ignore_runtime_errors:
            raise ModelTimingValidationError("DMA gate requires fatal native runtime errors")
        core.load_program(InstantiableProgram(instructions))
        state = core.arch_state
        bank = cfg.vmem_bank_bytes
        span = 8192
        base = state.cfg.dram_base
        # Deliberately different sources/destinations reveal partial transfers,
        # byte ordering and launch capture mistakes; zero is a separate case.
        data = ((torch.arange(span, dtype=torch.int64) * 13 + 7) % 251).to(torch.uint8) if nonzero else torch.zeros(span, dtype=torch.uint8)
        state.write_vmem(0, 0, data)
        state.write_vmem(bank, 0, data.flip(0))
        state.dram[base:base + span] = data.flip(0)
        state.dram[base + span:base + 2 * span] = data
        for reg in state.mrf:
            reg.fill_(0x30 if nonzero else 0)
        for buffers in state.wb.values():
            for tensor in buffers:
                tensor.fill_(0x30 if nonzero else 0)
        second = bank if sample["second_bank"] else 4096
        for reg, value in {3: sample["beats"] * cfg.dma_beat_bytes, 4: base, 5: 0,
                           7: base + span, 8: second // 4, 15: span, 16: 0x12345678 if nonzero else 0}.items():
            state.write_xrf(reg, value)
        unit = next(unit for unit in core.exus if isinstance(unit, DmaExecutionUnit))
        traces, events, denied = [], [], 0
        error = None
        for _ in range(VALIDATION_POLICY["native_cycle_limit"]):
            try:
                core.tick()
            except (RuntimeError, AssertionError) as exc:
                error = str(exc)
                break
            issued = core.issued_uop
            if issued is not None and issued.pc in pcs:
                traces.append([pcs.index(issued.pc), core.cycle_count])
            event = unit.last_cycle
            for access in (event.vmem_read, event.vmem_write):
                denied += int(access is not None and not access[2])
            if event.a is not None or event.d is not None:
                events.append([core.cycle_count, event.a, event.d])
            if state.halted and not any(exu.has_in_flight for exu in core.exus):
                break
        else:
            raise ModelTimingValidationError("DMA native gate did not drain: " + sample["id"])
        rejection = sample["rejection"]
        if rejection is not None:
            if error is None or rejection not in error:
                raise ModelTimingValidationError("DMA negative failed: " + sample["id"] + ": " + str(error))
            return dict(id=sample["id"], rejected=True, error=error)
        if error is not None:
            raise ModelTimingValidationError("DMA native runtime error: " + sample["id"] + ": " + error)
        if len(traces) != len(commands) or any(state.flags) or unit.backend.outstanding or unit._in_flight:
            raise ModelTimingValidationError("DMA issue/drain mismatch: " + sample["id"])
        relative = [[index, cycle - traces[0][1]] for index, cycle in traces]
        if not serial:
            for index in range(1, len(traces)):
                # A matching wait is the only data-dependent stall in these samples.
                if commands[index].startswith("dma.wait"):
                    continue
                if traces[index][1] - traces[index - 1][1] != sample["gaps"][index - 1]:
                    raise ModelTimingValidationError("DMA exact issue gap mismatch: " + sample["id"])
        tensors = [state.vmem, *state.mrf]
        for unit_name in ("mxu0", "mxu1"):
            tensors.extend(state.wb[unit_name])
            tensors.extend(state.acc[unit_name])
        digest = sha256(b"".join(tensor.contiguous().view(torch.uint8).numpy().tobytes() for tensor in tensors)
                        + state.dram[base:base + 2 * span].numpy().tobytes()
                        + canonical(dict(xrf=state.xrf, erf=[int(x) for x in state.erf], base=state.base)))
        return dict(id=sample["id"], rejected=False, issue_trace=relative, denied_grants=denied,
                    dma_events_sha256=sha256(canonical(events)), state_sha256=digest,
                    retired_age=core.cycle_count - traces[0][1], drained=True)
    finally:
        core.close()


def validate_dma_admission(model, per_op, admission):
    """Return (qualified, deterministic report), refusing any failed gate."""
    model = safe_path(model)
    frozen = deepcopy(admission)
    expected = derive_dma_admission(model, per_op)
    if _semantics(expected) != _semantics(frozen):
        raise ModelTimingValidationError("DMA admission candidate differs from selected source derivation")
    sys.path.insert(0, str(model))
    import npu_model
    import torch
    from atlas_model_timing import configuration
    if safe_path(Path(npu_model.__file__).parent.parent) != model:
        raise ModelTimingValidationError("DMA admission imported a different model")
    torch.set_num_threads(1)
    cfg, config = configuration()
    if config != per_op["model"]["configuration"]:
        raise ModelTimingValidationError("DMA admission native configuration differs from facts")
    plan = sample_plan(per_op)
    results, positives, negatives, numerical = [], 0, 0, 0
    for sample in plan:
        for nonzero in (False, True):
            observed = _run(cfg, sample, nonzero=nonzero)
            if sample["rejection"]:
                negatives += 1
            else:
                serial = _run(cfg, sample, nonzero=nonzero, serial=True)
                if observed["state_sha256"] != serial["state_sha256"]:
                    raise ModelTimingValidationError("DMA serial numerical mismatch: " + sample["id"])
                if sample["id"] == "load-response-grant-denial" and observed["denied_grants"] < 1:
                    raise ModelTimingValidationError("DMA load response denial mechanism was not exercised")
                positives += 1
                numerical += 1
                observed["serial_state_sha256"] = serial["state_sha256"]
            results.append(dict(observed, nonzero=nonzero))
    if admission != frozen or derive_dma_admission(model, per_op) != expected:
        raise ModelTimingValidationError("DMA admission inputs changed during native qualification")
    report = dict(schema="atlas.dma_admission_validation.v1", passed=True, sample_policy=POLICY,
                  classification="bounded_native_qualification", validation_policy=VALIDATION_POLICY,
                  zero_gap_control=dict(gap=0, outcome="unencodable_single_issue", admitted=False),
                  candidate_sha256=sha256(canonical(_semantics(frozen))),
                  sample_descriptors_sha256=sha256(canonical(plan)), samples=plan, inputs_unchanged=True,
                  positive_runs=positives, negative_runs=negatives, numerical_checks=numerical,
                  runs=results, limitations=["Representative native model contexts; not exhaustive operands or RTL qualification.",
                                            "Completion remains dynamic; matching waits and existing alias/ring rules remain required."])
    qualified = deepcopy(frozen)
    qualified["validation"] = dict(passed=True, report_sha256=sha256(canonical(report)),
                                    positive_runs=positives, negative_runs=negatives, numerical_checks=numerical)
    return qualified, report
