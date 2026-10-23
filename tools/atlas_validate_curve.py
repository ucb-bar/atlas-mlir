"""Independent fact-driven DMA predictions checked against the native pinned model.

This module never derives facts or fits timings. Its reference consumes only the
exported fact object and compares exact request/response/retirement edges and bytes
with the unchanged backend and DMA engine under deliberately varied streams.
"""
from __future__ import annotations

from collections import deque
from copy import copy
from io import StringIO
import json
import math
from unittest.mock import Mock


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
    raise RuntimeError("backend validation timed out")


def _engine_prediction(facts, stores, beats, launch):
    from npu_model.hardware.memory_backend import BeatRequest
    backend = FactBackend(facts)
    issued, completed = [], []
    index, in_flight = 0, 0
    first = launch + facts["startup_cycles"]["store" if stores else "load"]
    for cycle in range(launch, 200000):
        registered = in_flight
        response = backend.response(cycle)
        if response is not None:
            completed.append(cycle)
            backend.pop_response(cycle)
            in_flight -= 1
        if (index < beats and cycle >= first + index * facts["issue_interval_cycles"] and
                registered < facts["max_in_flight"] and backend.can_accept(cycle)):
            backend.issue(BeatRequest(index, index * facts["beat_bytes"], stores, cycle))
            issued.append(cycle)
            index += 1
            in_flight += 1
        if index == beats and in_flight == 0:
            return dict(issued=issued, completed=completed, retired_cycle=cycle)
    raise RuntimeError("DMA fact prediction timed out")


def _engine_observation(cfg, *, store, beats):
    import torch
    from npu_model.hardware.arch_state import ArchState
    from npu_model.hardware.dma import DmaExecutionUnit
    from npu_model.logging.logger import Logger
    from npu_model.software.instruction import Uop
    from npu_model.util.converter import stream_to_instrs
    size = beats * cfg.dma_beat_bytes
    state = ArchState(cfg.arch_state_config)
    try:
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
        instruction, = stream_to_instrs(StringIO(mnemonic + "\n"))
        unit = DmaExecutionUnit("dma", Mock(spec=Logger), state, config=cfg)
        uop = Uop(instruction, pc=0)
        issued, completed = [], []
        for age in range(200000):
            unit.tick(uop if age == 0 else None)
            event = unit.last_cycle
            if event.a is not None:
                issued.append(unit.cycle)
            if event.d is not None:
                completed.append(unit.cycle)
            if age and not unit.has_in_flight:
                result = state.dram[dram_address:dram_address + size] if store else state.read_vmem(0, 0, size)
                if not torch.equal(result, data):
                    raise RuntimeError("isolated DMA validation data mismatch")
                if unit.backend.outstanding or unit._in_flight or unit._ids_in_flight:
                    raise RuntimeError("isolated DMA validation did not drain")
                return dict(issued=issued, completed=completed, retired_cycle=unit.cycle)
        raise RuntimeError("native isolated DMA validation timed out")
    finally:
        state.close()


def _mixed_engine_observation(cfg, facts, *, fact_backend=False):
    """Three concurrent channels, mixed commands, disjoint source/destination bytes."""
    import torch
    from dataclasses import asdict
    from npu_model.hardware.arch_state import ArchState
    from npu_model.hardware.dma import DmaExecutionUnit
    from npu_model.logging.logger import Logger
    from npu_model.software.instruction import Uop
    from npu_model.util.converter import stream_to_instrs
    state = ArchState(cfg.arch_state_config)
    size = facts["beat_bytes"] * (facts["window_beats"] + 1)
    try:
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
        instructions = stream_to_instrs(StringIO("dma.load x2, x1, x3, ch0\n"
                                                "dma.store x4, x5, x3, ch1\n"
                                                "dma.load x7, x6, x3, ch2\n"))
        commands = [Uop(insn, pc=index * 4) for index, insn in enumerate(instructions)]
        for channel in range(len(commands)):
            state.set_flag(channel)
        unit = DmaExecutionUnit("dma", Mock(spec=Logger), state, config=cfg)
        if fact_backend:
            unit.backend = FactBackend(facts)
        trace = []
        for age in range(200000):
            unit.tick(commands[age] if age < len(commands) else None)
            event = asdict(unit.last_cycle)
            if event["a"] is not None or event["d"] is not None or unit.complete_count:
                trace.append(dict(event, complete_count=unit.complete_count))
            if age >= len(commands) and not unit.has_in_flight:
                comparisons = (torch.equal(state.read_vmem(0, 0, size), read0),
                               torch.equal(state.read_vmem(2 * bank, 0, size), read1),
                               torch.equal(state.dram[base + bank:base + bank + size], store))
                if not all(comparisons) or unit.backend.outstanding or unit._in_flight:
                    raise RuntimeError("mixed DMA bytes or drain validation failed")
                return dict(trace=trace, retired_cycle=unit.cycle, bytes_per_command=size,
                            commands=len(commands), bytes_verified=True)
        raise RuntimeError("mixed native DMA validation timed out")
    finally:
        state.close()


def validate_curve(cfg, facts, out):
    """Require exact independent/native parity; write diagnostic traces and receipt."""
    from atlas_derive_curve import derive_curve
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
                    raise RuntimeError("curve backend fact/native mismatch: " + key)
                receipt = dict(name=key, passed=True, beats=len(stores), address_translation_invariant=True,
                               selected_curves=reference.selected, trace=native)
                (out / (key + ".json")).write_bytes(canonical(receipt) + b"\n")
                checks.append({k: v for k, v in receipt.items() if k not in {"trace", "selected_curves"}} |
                              {k: native[k] for k in ("blocked_cycles", "held_cycles", "max_pending")})
            for store in (False, True):
                for beats in (1, 32, 128):
                    key = variant_name + ("-store-" if store else "-load-") + str(beats)
                    expected = _engine_prediction(derived, store, beats, launch=1)
                    native = _engine_observation(selected, store=store, beats=beats)
                    if native != expected:
                        (out / (key + "-mismatch.json")).write_bytes(canonical(dict(expected=expected, native=native)))
                        raise RuntimeError("isolated DMA fact/native mismatch: " + key)
                    (out / (key + ".json")).write_bytes(canonical(dict(name=key, passed=True, trace=native)) + b"\n")
                    checks.append(dict(name=key, passed=True, beats=beats, bytes_verified=True,
                                       retired_cycle=native["retired_cycle"]))
            key = variant_name + "-mixed-dma"
            native = _mixed_engine_observation(selected, derived)
            expected = _mixed_engine_observation(selected, derived, fact_backend=True)
            if native != expected:
                raise RuntimeError("mixed native DMA fact/backend mismatch: " + key)
            (out / (key + ".json")).write_bytes(canonical(dict(name=key, passed=True, **native)) + b"\n")
            checks.append(dict(name=key, passed=True, bytes_verified=True,
                               commands=native["commands"], retired_cycle=native["retired_cycle"]))
        report = dict(schema="atlas.dma_timing_validation.v1", passed=True, checks=checks,
                      backend_streams=len(variants) * len(streams),
                      isolated_dma_runs=len(variants) * 2 * 3,
                      mixed_dma_runs=len(variants) * 2,
                      fact_sha256=sha256(canonical(facts)),
                      scope="Exact native-model parity; no RTL/SoC qualification or fitted constants.")
    except Exception as error:
        report = dict(schema="atlas.dma_timing_validation.v1", passed=False, checks=checks, error=str(error))
        (out / "validation-report.json").write_text(json.dumps(report, sort_keys=True, indent=2) + "\n")
        raise
    (out / "validation-report.json").write_text(json.dumps(report, sort_keys=True, indent=2) + "\n")
    return report
