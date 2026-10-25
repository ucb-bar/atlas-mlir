"""Check fact-driven DMA predictions, bytes and completion edges against the native model."""
from __future__ import annotations

from atlas_model_diagnostics import ModelTimingValidationError

from collections import deque
from copy import copy
import json
import math
from atlas_validate_common import instructions, native_core, native_state, native_unit, uops


class FactBackend:
    """Small unloaded-curve interpreter, independent of the model controller."""
    def __init__(self, facts):
        self.facts = facts
        self.recent = deque(maxlen=facts["window_beats"])
        self.last_ready = facts["reset_last_ready_cycles"]
        self.pending = []
        self.order = 0
        self.presented = None
        self.selected = []

    def can_accept(self, cycle):
        return len(self.pending) < self.facts["queue_depth"]

    def issue(self, request):
        assert self.can_accept(request.cycle)
        self.recent.append(request.is_store)
        read_pct = 100.0 * (len(self.recent) - sum(self.recent)) / len(self.recent)
        row = min(self.facts["curves"], key=lambda c: (abs(c["read_percentage"] - read_pct), c["read_percentage"]))
        self.selected.append(row["read_percentage"])
        ready = max(request.cycle + max(self.facts["min_latency"], row["unloaded_latency_cycles"]),
                    self.last_ready + row["cycles_per_beat"])
        self.last_ready = ready
        self.pending.append((math.ceil(ready - self.facts["rounding_epsilon_cycles"]), self.order, request))
        self.pending.sort(key=lambda x: x[:2])
        self.order += 1

    def response(self, cycle):
        if self.presented is None and self.pending and self.pending[0][0] <= cycle:
            self.presented = self.pending[0][2]
        return self.presented

    def pop_response(self, cycle):
        assert self.presented is not None
        self.pending.pop(0)
        self.presented = None

    @property
    def outstanding(self):
        return len(self.pending)


def _stream(backend, stores, *, translation=0, stalls=False, gap=0):
    from npu_model.hardware.memory_backend import BeatRequest
    issued, completed, blocked, held = [], [], 0, 0
    max_pending = 0
    index = 0
    for cycle in range(200000):
        response = backend.response(cycle)
        if response is not None:
            if stalls and cycle % 31 < 7:
                held += 1
                assert backend.response(cycle) == response
            else:
                completed.append([cycle, response.source, response.is_store])
                backend.pop_response(cycle)
        if index < len(stores) and cycle >= index * gap:
            if backend.can_accept(cycle):
                backend.issue(BeatRequest(index, translation + index * 32, stores[index], cycle))
                issued.append([cycle, index, stores[index]])
                index += 1
            else:
                blocked += 1
        max_pending = max(max_pending, backend.outstanding)
        if index == len(stores) and not backend.outstanding:
            return dict(issued=issued, completed=completed, blocked_cycles=blocked,
                        held_cycles=held, max_pending=max_pending)
    raise ModelTimingValidationError("backend validation timed out")


def _engine_observation(cfg, *, store, beats, history=()):
    import torch
    size = beats * cfg.dma_beat_bytes
    with native_state(cfg) as state:
        data = (torch.arange(size, dtype=torch.int64) % 251).to(torch.uint8)
        dram_address = state.cfg.dram_base
        state.write_xrf(1, dram_address)
        state.write_xrf(2, 0)
        state.write_xrf(3, size)
        if store:
            state.write_vmem(0, 0, data)
        else:
            state.dram[dram_address:dram_address + size] = data
        state.set_flag(0)
        mnemonic = "dma.store x1, x2, x3, ch0" if store else "dma.load x2, x1, x3, ch0"
        command, = uops(mnemonic)
        unit = native_unit(cfg, state, "dma")
        if history:
            # Drain a prior mixed backend stream, preserving issued history and
            # fractional ready state. The launch follows its last acknowledgement.
            prior = _stream(unit.backend, history)
            unit.cycle = prior["completed"][-1][0]
        launch = unit.cycle + 1
        issued, completed = [], []
        for age in range(200000):
            unit.tick(command if age == 0 else None)
            event = unit.last_cycle
            if event.a is not None:
                issued.append(unit.cycle)
            if event.d is not None:
                completed.append(unit.cycle)
            if age and not unit.has_in_flight:
                result = state.dram[dram_address:dram_address + size] if store else state.read_vmem(0, 0, size)
                if not torch.equal(result, data):
                    raise ModelTimingValidationError("isolated DMA validation data mismatch")
                if unit.backend.outstanding or unit._in_flight or unit._ids_in_flight:
                    raise ModelTimingValidationError("isolated DMA validation did not drain")
                return dict(issued=issued, completed=completed, retired_cycle=unit.cycle, launch_cycle=launch)
        raise ModelTimingValidationError("native isolated DMA validation timed out")


def _mixed_engine_observation(cfg, facts, *, fact_backend=False, deny_grants=False):
    """Three concurrent channels, mixed commands, disjoint source/destination bytes."""
    import torch
    from dataclasses import asdict
    size = facts["beat_bytes"] * (facts["window_beats"] + 1)
    with native_state(cfg) as state:
        base = state.cfg.dram_base
        bank = cfg.vmem_bank_bytes
        read0 = (torch.arange(size, dtype=torch.int64) % 251).to(torch.uint8)
        read1 = ((torch.arange(size, dtype=torch.int64) + 53) % 251).to(torch.uint8)
        store = ((torch.arange(size, dtype=torch.int64) + 97) % 251).to(torch.uint8)
        state.dram[base:base + size] = read0
        state.dram[base + 2 * bank:base + 2 * bank + size] = read1
        state.write_vmem(bank, 0, store)
        for register, value in {1: base, 2: 0, 3: size, 4: base + bank,
                                5: bank // 4, 6: base + 2 * bank, 7: 2 * bank // 4}.items():
            state.write_xrf(register, value)
        commands = uops("dma.load x2, x1, x3, ch0\n"
                        "dma.store x4, x5, x3, ch1\n"
                        "dma.load x7, x6, x3, ch2\n")
        for channel in range(len(commands)):
            state.set_flag(channel)
        unit = native_unit(cfg, state, "dma")
        if fact_backend:
            unit.backend = FactBackend(facts)
        trace = []
        for age in range(200000):
            if deny_grants and age % 19 < 7:
                state.conflict_checker.announce_vmem_ports(unit.cycle + 1, {0: "validation", 1: "validation", 2: "validation"})
            unit.tick(commands[age] if age < len(commands) else None)
            event = asdict(unit.last_cycle)
            if any(event[key] is not None for key in ("a", "d", "vmem_read", "vmem_write")) or unit.complete_count:
                trace.append(dict(event, complete_count=unit.complete_count))
            if age >= len(commands) and not unit.has_in_flight:
                comparisons = (torch.equal(state.read_vmem(0, 0, size), read0),
                               torch.equal(state.read_vmem(2 * bank, 0, size), read1),
                               torch.equal(state.dram[base + bank:base + bank + size], store))
                if not all(comparisons) or unit.backend.outstanding or unit._in_flight:
                    raise ModelTimingValidationError("mixed DMA bytes or drain validation failed")
                return dict(trace=trace, retired_cycle=unit.cycle, bytes_per_command=size,
                            commands=len(commands), bytes_verified=True)
        raise ModelTimingValidationError("mixed native DMA validation timed out")


def _core_wait_observation(cfg, *, store, beats, wait_age=1):
    """Execute real Core S1 and DMA edges, including a precisely placed wait."""
    from npu_model.hardware.dma import DmaExecutionUnit
    from npu_model.software.program import InstantiableProgram
    mnemonic = "dma.store x1, x2, x3, ch0" if store else "dma.load x2, x1, x3, ch0"
    assembly = [mnemonic] + ["addi x0, x0, 0"] * (wait_age - 1) + ["dma.wait ch0"]
    with native_core(cfg) as core:
        state = core.arch_state
        state.write_xrf(1, state.cfg.dram_base)
        state.write_xrf(2, 0)
        state.write_xrf(3, beats * cfg.dma_beat_bytes)
        core.load_program(InstantiableProgram(instructions(assembly)))
        dma = next(unit for unit in core.exus if isinstance(unit, DmaExecutionUnit))
        launch = retired = waited = None
        wait_checks = []
        for _ in range(200000):
            core.tick()
            edge = core.cycle_count
            event = core.last_cycle
            if event["instruction"] == ("dma.store" if store else "dma.load") and event["s1_fire"]:
                launch = edge
            if dma.complete_count:
                retired = edge
            if event["instruction"] == "dma.wait":
                wait_checks.append(dict(cycle=edge, fired=event["s1_fire"], stall=event["stall"]))
                if event["s1_fire"]:
                    waited = edge
            if waited is not None and not dma.has_in_flight:
                return dict(launch_cycle=launch, retired_cycle=retired, wait_issue_cycle=waited,
                            wait_checks=wait_checks, drained=not dma.backend.outstanding)
        raise ModelTimingValidationError("native Core DMA/wait validation timed out")


def validate_curve(cfg, facts, out):
    """Require backend parity, command bounds and Core edges; write native receipts."""
    from atlas_derive_curve import derive_curve, proposal_duration
    from npu_model.hardware.memory_backend import make_memory_backend
    from atlas_model_schema import canonical, sha256
    out.mkdir(parents=True, exist_ok=False)
    checks = []
    variants = [("default", {}), ("channels2", {"channels": 2}),
                ("frequency1", {"frequency_ghz": 1.0}),
                ("history8_queue4", {"window_beats": 8, "queue_depth": 4}),
                ("frequency025_channels2", {"frequency_ghz": 0.25, "channels": 2})]
    streams = [("read", [False] * 129, False, 0), ("write", [True] * 129, False, 0),
               ("mixed_history", [False] * 33 + [True] * 40 + [False] * 39, False, 0),
               ("alternating_held", [False, True] * 65, True, 0),
               ("idle_gap", [False] * 17 + [True] * 17 + [False] * 33, False, 79)]
    try:
        for variant_name, updates in variants:
            selected = copy(cfg)
            selected.dma_memory_params = dict(cfg.dma_memory_params, **updates)
            derived = facts if not updates else derive_curve(selected)
            for name, stores, stalls, gap in streams:
                key = variant_name + "-" + name
                reference = FactBackend(derived)
                expected = _stream(reference, stores, stalls=stalls, gap=gap)
                native = _stream(make_memory_backend(selected), stores, stalls=stalls, gap=gap)
                translated = _stream(make_memory_backend(selected), stores, translation=0x1000000080,
                                     stalls=stalls, gap=gap)
                if native != expected or translated != native:
                    raise ModelTimingValidationError("curve backend fact/native mismatch: " + key)
                receipt = dict(name=key, passed=True, beats=len(stores), address_translation_invariant=True,
                               selected_curves=reference.selected, trace=native)
                (out / (key + ".json")).write_bytes(canonical(receipt) + b"\n")
                checks.append({k: v for k, v in receipt.items() if k not in {"trace", "selected_curves"}} |
                              {k: native[k] for k in ("blocked_cycles", "held_cycles", "max_pending")})
            for store in (False, True):
                for beats in (1, 32, 128):
                    key = variant_name + ("-store-" if store else "-load-") + str(beats)
                    native = _engine_observation(selected, store=store, beats=beats)
                    duration = proposal_duration(derived, store=store, beats=beats)
                    if (native["retired_cycle"] > native["launch_cycle"] + duration or
                            native["issued"][0] != native["launch_cycle"] + derived["startup_cycles"]["store" if store else "load"] or
                            native["retired_cycle"] != native["completed"][-1]):
                        raise ModelTimingValidationError("isolated DMA source bound/edge mismatch: " + key)
                    (out / (key + ".json")).write_bytes(canonical(dict(name=key, passed=True, trace=native)) + b"\n")
                    checks.append(dict(name=key, passed=True, beats=beats, bytes_verified=True,
                                       retired_cycle=native["retired_cycle"]))
            for denied in (False, True):
                key = variant_name + "-mixed-dma" + ("-denied-grants" if denied else "")
                native = _mixed_engine_observation(selected, derived, deny_grants=denied)
                expected = _mixed_engine_observation(selected, derived, fact_backend=True, deny_grants=denied)
                if native != expected:
                    raise ModelTimingValidationError("mixed native DMA fact/backend mismatch: " + key)
                denied_events = sum(not access[2] for event in native["trace"]
                                    for access in (event["vmem_read"], event["vmem_write"]) if access is not None)
                if denied and not denied_events:
                    raise ModelTimingValidationError("mixed DMA denied-grant control did not deny a request")
                (out / (key + ".json")).write_bytes(canonical(dict(name=key, passed=True, **native)) + b"\n")
                checks.append(dict(name=key, passed=True, bytes_verified=True, denied_grant_events=denied_events,
                                   bound_checked=False, commands=native["commands"], retired_cycle=native["retired_cycle"]))
            for store in (False, True):
                key = variant_name + "-retained-mixed-history-" + ("store" if store else "load")
                native = _engine_observation(selected, store=store, beats=32,
                                             history=[False] * 17 + [True] * 13 + [False, True] * 11)
                bound = proposal_duration(derived, store=store, beats=32)
                if native["retired_cycle"] > native["launch_cycle"] + bound:
                    raise ModelTimingValidationError("retained-history DMA bound mismatch: " + key)
                checks.append(dict(name=key, passed=True, bound_cycles=bound,
                                   native_cycles=native["retired_cycle"] - native["launch_cycle"]))
        for store in (False, True):
            for beats in (1, 32):
                early = _core_wait_observation(cfg, store=store, beats=beats)
                retired_age = early["retired_cycle"] - early["launch_cycle"]
                short = _core_wait_observation(cfg, store=store, beats=beats, wait_age=retired_age)
                positive = _core_wait_observation(cfg, store=store, beats=beats, wait_age=retired_age + 1)
                gap = facts["proposal_contract"]["retire_to_wait_issue_cycles"]
                for native in (early, short, positive):
                    if native["wait_issue_cycle"] != native["retired_cycle"] + gap or not native["drained"]:
                        raise ModelTimingValidationError("Core DMA completion/wait visibility mismatch")
                if (short["wait_checks"][0] != dict(cycle=short["retired_cycle"], fired=False, stall="dma.wait") or
                        positive["wait_checks"] != [dict(cycle=positive["wait_issue_cycle"], fired=True, stall=None)]):
                    raise ModelTimingValidationError("Core wait one-short/positive boundary control mismatch")
                key = "core-wait-" + ("store-" if store else "load-") + str(beats)
                (out / (key + ".json")).write_bytes(canonical(dict(early=early, one_short=short, positive=positive)) + b"\n")
                checks.append(dict(name=key, passed=True, one_short_stalled=True, positive_no_stall=True,
                                   retire_to_wait_issue_cycles=gap))
        report = dict(schema="atlas.dma_timing_validation.v1", passed=True, checks=checks,
                      backend_streams=len(variants) * len(streams),
                      isolated_dma_runs=len(variants) * 2 * 3,
                      mixed_dma_runs=len(variants) * 4,
                      retained_history_dma_runs=len(variants) * 2, core_wait_runs=12,
                      fact_sha256=sha256(canonical(facts)),
                      scope="Exact backend parity, source-bound and Core edge validation; no RTL/SoC qualification or fitted constants.")
    except Exception as error:
        report = dict(schema="atlas.dma_timing_validation.v1", passed=False, checks=checks, error=str(error))
        (out / "validation-report.json").write_text(json.dumps(report, sort_keys=True, indent=2) + "\n")
        raise
    (out / "validation-report.json").write_text(json.dumps(report, sort_keys=True, indent=2) + "\n")
    return report
