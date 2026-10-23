#!/usr/bin/env python3
"""Observe isolated operation timing from an explicitly selected model."""
from __future__ import annotations

from collections import defaultdict
from contextlib import ExitStack
from dataclasses import asdict, replace
from io import StringIO
import json
import os
from pathlib import Path
from unittest.mock import Mock, patch

from atlas_model_observer import observe_mxu
from atlas_model_schema import VPU_BLOCKS, MXU_BLOCKS, BINARY, mreg_event, canonical as encoded, sha256


def manifest(model_root: Path, subdir: str, suffixes=None, sort=False):
    records = []
    kind = "model source" if suffixes else "model numerical data"
    root = model_root / subdir
    if any(word in part.lower() for part in root.parts for word in ("hammer", "vlsi")):
        raise RuntimeError("prohibited manifest path")
    if ".." in root.parts:
        raise RuntimeError("parent traversal is unsupported")
    root = Path(os.path.abspath(root))
    if any(word in part.lower() for part in root.parts for word in ("hammer", "vlsi")):
        raise RuntimeError("prohibited manifest path")
    if any(path.is_symlink() for path in reversed((root, *root.parents))):
        raise RuntimeError(kind + " root contains a symlink")
    if not root.is_dir():
        raise RuntimeError(kind + " root is missing")
    def visit(directory):
        # Inspect names before any entry lookup. os.walk classifies entries before
        # yielding them and can follow a directory symlink while doing so.
        with os.scandir(directory) as stream:
            entries = list(stream)
        if any(word in entry.name.lower() for entry in entries for word in ("hammer", "vlsi")):
            raise RuntimeError(kind + " contains a prohibited entry name")
        directories = []
        for entry in sorted(entries, key=lambda entry: entry.name):
            if suffixes and entry.name == "__pycache__":
                continue
            if entry.is_symlink():
                raise RuntimeError(kind + " contains a symlink")
            path = directory / entry.name
            if entry.is_dir(follow_symlinks=False):
                directories.append(path)
            elif not suffixes or path.suffix in suffixes:
                records.append({"path": str(path.relative_to(model_root)), "sha256": sha256(path.read_bytes())})
        for child in directories:
            visit(child)
    visit(root)
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
                raise RuntimeError(f"ISA operation has no timing vocabulary: {mnemonic}")
            continue
        name, engine = names[mnemonic]
        if operation.exu in engines and engines[operation.exu] != engine:
            raise RuntimeError(f"ISA execution unit differs for {mnemonic}")
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
            raise RuntimeError(f"unsupported ISA operand {mnemonic}: {typ.__name__}/{role}")
        assembly = mnemonic + " " + ", ".join(operand(param) for param in operation.params)
        found[name] = (name, engine, assembly)
    if len(found) != len(names):
        raise RuntimeError("selected ISA is missing supported timing operations")
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
        elif isinstance(value, (str, int, bool, dict, type(None))):
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


def observe_case(block_name: str, engine: str, assembly: str, *, nonzero=False, instrument=True, max_cycles=1024, cfg=None):
    """Execute one instruction, capturing real calls, never latency tables."""
    import torch
    from npu_model.hardware.arch_state import ArchState
    from npu_model.hardware.exu import StagedExecution
    from npu_model.hardware.lsu import LoadStoreUnit
    from npu_model.hardware.mxu import MatrixExecutionUnitInner, MatrixExecutionUnitSystolic
    from npu_model.hardware.vpu import VectorExecutionUnit
    from npu_model.hardware.xlu import CrossLaneExecutionUnit
    from npu_model.logging.logger import Logger
    from npu_model.software.instruction import Uop
    from npu_model.util.converter import stream_to_instrs

    if cfg is None:
        cfg, config_values = configuration()
    else:
        config_values = configuration_values(cfg)
    state = ArchState(cfg.arch_state_config)
    with ExitStack() as cleanup:
        cleanup.callback(state.close)
        finish_tick = lambda: None
        classes = {"vpu": VectorExecutionUnit, "mxu0": MatrixExecutionUnitSystolic,
                   "mxu1": MatrixExecutionUnitInner, "vlsu": LoadStoreUnit,
                   "scalar_lsu": LoadStoreUnit, "xlu": CrossLaneExecutionUnit}
        unit = classes[engine](engine, Mock(spec=Logger), state, config=cfg)
        instruction, = stream_to_instrs(StringIO(assembly))
        uop = Uop(instruction, pc=0)
        mnemonic = instruction.mnemonic
        # Nonzero runs fill every register file and VMEM; zero runs keep reset state.
        if nonzero:
            for bank in range(64):
                state.mrf[bank].view(torch.bfloat16).fill_(0.5 + (bank % 3) * 0.25)
            if mnemonic.startswith("vmat") or mnemonic == "vunpack.fp8.bf16":
                state.write_mrf_fp8(0, torch.ones((32, 32)).to(torch.float8_e4m3fn))
            for mxu in ("mxu0", "mxu1"):
                for slot in (0, 1):
                    state.acc[mxu][slot].fill_(0.5)
                    state.write_wb_fp8(mxu, slot, torch.eye(32).to(torch.float8_e4m3fn))
            state.vmem[:1024] = torch.arange(1024, dtype=torch.int64).remainder(251).to(torch.uint8)
            state.write_xrf(1, 0x12345678)
        state.write_erf(0, 127)
        def snapshot():
            tensors = list(state.mrf) + [t for slots in state.acc.values() for t in slots]
            tensors += [t for slots in state.wb.values() for t in slots] + [state.vmem]
            raw = b"".join(t.contiguous().view(torch.uint8).numpy().tobytes() for t in tensors)
            return sha256(raw + encoded({"xrf": state.xrf, "erf": state.erf}))

        initial_digest = snapshot()
        events, levels, states, controls, reservations = [], [], [], [], []
        def event(group, kind, **fields):
            events.append(dict(group=group, kind=kind, cycle=unit.cycle, age=unit.cycle - 1,
                               uop_id=uop.id, pc=uop.pc, mnemonic=mnemonic, **fields))

        if instrument:
            checker = state.conflict_checker
            original_reserve, original_release = checker.reserve_mreg, checker.release_mreg
            reserved = {}
            def reserve(owner, reads, writes, **kwargs):
                original_reserve(owner, reads, writes, **kwargs)
                reserved[owner] = (set(reads), set(writes))
                reservations.append(dict(kind="reserve", age=unit.cycle - 1, reads=sorted(reads), writes=sorted(writes)))
            def release(owner, *, reads=True, writes=True):
                original_release(owner, reads=reads, writes=writes)
                old_reads, old_writes = reserved.get(owner, (set(), set()))
                released_reads, released_writes = old_reads if reads else set(), old_writes if writes else set()
                reservations.append(dict(kind="release", age=unit.cycle - 1,
                                         reads=sorted(released_reads), writes=sorted(released_writes)))
                reserved[owner] = (old_reads - released_reads, old_writes - released_writes)
            cleanup.enter_context(patch.object(checker, "reserve_mreg", reserve))
            cleanup.enter_context(patch.object(checker, "release_mreg", release))
            original_mreg = state.conflict_checker.access_mreg
            def mreg(cycle, bank, row, write, owner):
                original_mreg(cycle, bank, row, write, owner)
                group = mreg_event(mnemonic, engine, int(bank), write)
                event(group, "mreg_write_edge" if write else "mreg_read_request",
                      bank=int(bank), row=int(row), owner=owner)
            cleanup.enter_context(patch.object(state.conflict_checker, "access_mreg", mreg))
            original_read, original_write = state.read_vmem, state.write_vmem
            def vread(address, imm, size):
                result = original_read(address, imm, size)
                event("vmem_read", "vmem_read_sample", address=int(address + imm), size=int(size))
                return result
            def vwrite(address, imm, data):
                original_write(address, imm, data)
                event("vmem_write", "vmem_write_commit", address=int(address + imm), size=len(data))
            cleanup.enter_context(patch.object(state, "read_vmem", vread))
            cleanup.enter_context(patch.object(state, "write_vmem", vwrite))
            original_x, original_e = state.write_xrf, state.write_erf
            def xwrite(reg, value):
                original_x(reg, value)
                event("load_write", "xrf_write_commit", register=int(reg), value=int(value))
            def ewrite(reg, value):
                original_e(reg, value)
                event("scale_write", "erf_write_commit", register=int(reg), value=int(value))
            cleanup.enter_context(patch.object(state, "write_xrf", xwrite))
            cleanup.enter_context(patch.object(state, "write_erf", ewrite))
            def mxu_event(*, kind, cycle, uop, location, evidence, row=None):
                if cycle != unit.cycle:
                    raise RuntimeError("observer reported a different tick")
                if kind == "acc_read":
                    group = "acc_store" if mnemonic.startswith("vmatpop") else "acc_read"
                elif kind == "acc_write":
                    group = "acc_load" if mnemonic.startswith("vmatpush") else "acc_write"
                else:
                    group = kind
                fields = dict(location=list(location), evidence=evidence)
                if row is not None:
                    fields["row"] = int(row)
                event(group, kind, **fields)
            if engine.startswith("mxu"):
                finish_tick = cleanup.enter_context(observe_mxu(StagedExecution, unit, uop, mxu_event))

        for age in range(max_cycles):
            pre_live = bool(unit.has_in_flight)
            pre = {"age": age, "cycle": age + 1, "has_in_flight": pre_live}
            if engine == "vpu":
                pre["issue_busy_mask"] = unit.issue_busy_mask
            if engine in {"vlsu", "scalar_lsu"}:
                pre.update(vload_busy=unit._vload is not None, vstore_busy=unit._vstore is not None)
            levels.append(pre)
            if age > 0 and not pre_live:
                first_free = age
                break
            unit.tick(uop if age == 0 else None)
            finish_tick()
            states.append(snapshot())
            controls.append(sha256(encoded({key: getattr(state, key) for key in (
                "base", "pc", "npc", "flags", "csrf", "halted", "halt_reason", "redirect_requested", "in_delay_slot")})))
            if age == 0 and unit.total_instructions != 1:
                raise RuntimeError("isolated command was not accepted")
            if instrument and engine.startswith("mxu"):
                event("compute_busy" if mnemonic.startswith("vmatmul") else "data_busy",
                      "observed_pre_edge_or_accepted_live")
        else:
            raise RuntimeError(f"command did not drain within {max_cycles} cycles")
        final_digest = snapshot()
        grouped = defaultdict(list)
        for item in events:
            grouped[item["group"]].append(item["age"])
        block = dict(name=block_name, engine=engine, resolved=True,
                     events={name: compress(ages) for name, ages in sorted(grouped.items())},
                     first_free_age=first_free, next_issue_age=first_free,
                     assumptions={"scratchpad_read_latency": 1}, observation={"accepted": True, "drained": True})
        raw = dict(name=block_name, assembly=assembly, instruction_word=instruction.to_bytecode(),
                   uop_id=uop.id, pc=uop.pc, initialization="nonzero-v1" if nonzero else "zero-v1",
                   initial_state_sha256=initial_digest, final_state_sha256=final_digest,
                   pre_edges=levels, events=events, first_free_age=first_free,
                   configuration=config_values, architectural_states=states, control_states=controls,
                   complete_count=unit.complete_count, total_instructions=unit.total_instructions,
                   busy_cycles=unit.busy_cycles, reservations=reservations)
        return block, raw


def result_signature(raw):
    keys = ("initial_state_sha256", "final_state_sha256", "architectural_states", "control_states", "pre_edges",
            "first_free_age", "complete_count", "total_instructions", "busy_cycles", "instruction_word", "configuration")
    return {key: raw[key] for key in keys}


def observer_manifest():
    return [{"path": name, "sha256": sha256(Path(__file__).with_name(name).read_bytes())}
            for name in ("atlas_model_timing.py", "atlas_model_observer.py")]


def observed_releases(block, raw):
    releases = {"unit": dict(first_age=0, release_age=block["first_free_age"] - 1,
                             available_age=block["first_free_age"])}
    for direction in ("read", "write"):
        banks, ages, starts, live = set(), set(), set(), set()
        for event in raw["reservations"]:
            affected = set(event[direction + "s"])
            if event["kind"] == "reserve":
                if live & affected:
                    raise RuntimeError("MREG reservation repeated before release")
                if affected:
                    starts.add(event["age"])
                live.update(affected)
            elif affected:
                if not affected <= live:
                    raise RuntimeError("MREG release has no matching reservation")
                banks.update(affected)
                ages.add(event["age"])
                live.difference_update(affected)
        if live:
            raise RuntimeError("MREG reservations remain unreleased at drain")
        if banks:
            if len(ages) != 1 or starts != {0}:
                raise RuntimeError("MREG banks have unequal release ages or non-issue reservation starts")
            age, = ages
            releases["mreg_" + direction] = dict(banks=sorted(banks), first_age=0,
                                                  release_age=age, available_age=age + 1)
        key = "vmem_" + direction
        if key in block["events"]:
            event = block["events"][key]
            releases[key] = dict(banks=[0], first_age=event["first_age"], release_age=event["last_age"],
                                 available_age=event["last_age"] + 1)
    return releases


def validate_derived(cfg, candidates, releases, output_dir):
    """Validate source-derived blocks using the unmodified numerical execution paths."""
    checks, observed_blocks, runs = [], [], []
    for name, engine, assembly in isa_cases(cfg):
        accepted = []
        for nonzero in (False, True):
            check = dict(name=name, initialization="nonzero-v1" if nonzero else "zero-v1", passed=False)
            checks.append(check)
            try:
                block, raw = observe_case(name, engine, assembly, nonzero=nonzero, max_cycles=4096, cfg=cfg)
                _, plain = observe_case(name, engine, assembly, nonzero=nonzero, instrument=False, max_cycles=4096, cfg=cfg)
                runs.extend((dict(raw, observer=True), dict(plain, observer=False)))
                if result_signature(raw) != result_signature(plain):
                    raise RuntimeError("observer changed architectural data/control trace or unit progress")
                check["noninterference"] = True
                if encoded(block) != encoded(candidates[name]):
                    raise RuntimeError("source-derived timing block differs from measured export")
                check["byte_identical"] = True
                if observed_releases(block, raw) != releases[name]:
                    raise RuntimeError("source-derived reservation releases differ from model calls")
                check.update(passed=True, releases_equal=True, result_sha256=sha256(encoded(result_signature(raw))))
                accepted.append(block)
            except Exception as error:
                check["error"] = f"{type(error).__name__}: {error}"
        if len(accepted) == 2 and accepted[0] == accepted[1]:
            observed_blocks.append(accepted[0])
    passed = len(observed_blocks) == len(candidates) and all(check["passed"] for check in checks)
    report = dict(schema="atlas.model_derivation_validation.v1", passed=passed, operation_count=len(candidates),
                  observed_runs=sum(run["observer"] for run in runs), checks=checks,
                  unobserved_runs=sum(not run["observer"] for run in runs))
    for filename, value in (("validation-report.json", report), ("raw-observations.json", runs),
                            ("observed-blocks.json", observed_blocks)):
        (output_dir / filename).write_text(json.dumps(value, sort_keys=True, indent=2) + "\n")
    if not passed:
        raise RuntimeError(f"derived facts validation failed; see {output_dir / 'validation-report.json'}")
    return report, runs
