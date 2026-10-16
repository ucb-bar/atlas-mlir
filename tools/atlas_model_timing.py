#!/usr/bin/env python3
"""Experimental isolated timing observations from an explicitly selected model.

No engine timing equations or RTL expectation tables are evaluated here. Ages
come from callbacks during real unit ticks. Register-to-event group mapping is
an explicit compiler contract, not an inferred physical implementation. All
units drain before reuse: these observations establish no overlap/capacity fact.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from contextlib import ExitStack
from dataclasses import asdict, replace
import hashlib
from io import StringIO
import json
import os
from pathlib import Path
import subprocess
import sys
from unittest.mock import Mock, patch

from atlas_model_observer import observe_mxu


MODEL_REVISION = "9e6313bddecb8ba936206eb347f5cd4e5335282c"
SOURCE_SHA256 = "a9a919c3d118af94e2783c8969e05ad19f8d2523ec093922c4f9957641b54426"
REFERENCE_CONFIGURATION_SHA256 = "bdb127152c1ab4b174b2f628687f3b166dad48ea6f00be124ed8351443885e94"
CONFIGURATION_SHA256 = "c41ab668a2a6bf6a9351cb0df4e544703e14818f050f248d83a8e250d260e33a"
NUMERICAL_DATA_SHA256 = "9a279452a0e4115e1096f7cb01a99aa2796de4d33cccf76c378513bb55616ccd"
REFERENCE_SHA256 = "89d0ca34b43d022a08766b03ab5603a391247098aca9ca7e80d6dea88731e363"


VPU_BLOCKS = {
    "vadd.bf16": "add", "vsub.bf16": "sub", "vmul.bf16": "mul",
    "vmaximum.bf16": "pairmax", "vminimum.bf16": "pairmin", "vmov": "mov",
    "vrecip.bf16": "rcp", "vexp.bf16": "exp", "vexp2.bf16": "exp2",
    "vrelu.bf16": "relu", "vsin.bf16": "sin", "vcos.bf16": "cos",
    "vtanh.bf16": "tanh", "vlog2.bf16": "log", "vsqrt.bf16": "sqrt",
    "vsquare.bf16": "square", "vcube.bf16": "cube",
    "vredsum.row.bf16": "rsum", "vredmax.row.bf16": "rmax",
    "vredmin.row.bf16": "rmin", "vredsum.bf16": "csum",
    "vredmax.bf16": "cmax", "vredmin.bf16": "cmin",
    "vpack.bf16.fp8": "fp8pack", "vunpack.fp8.bf16": "fp8unpack",
    "vli.all": "vliAll", "vli.row": "vliRow", "vli.col": "vliCol",
    "vli.one": "vliOne",
}
MXU_BLOCKS = {
    "vmatpush.weight": "push_weight", "vmatpush.acc.fp8": "push_acc_fp8",
    "vmatpush.acc.bf16": "push_acc_bf16", "vmatpop.fp8.acc": "pop_acc_fp8",
    "vmatpop.bf16.acc": "pop_acc_bf16", "vmatmul": "matmul",
    "vmatmul.acc": "matmul_acc",
}
BINARY = {"vadd.bf16", "vsub.bf16", "vmul.bf16", "vmaximum.bf16", "vminimum.bf16"}


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def encoded(value) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()


def source_digest(model_root: Path) -> tuple[str, list[dict]]:
    """Hash source/config bytes, pruning restricted names before traversal."""
    records = []
    for directory, names, files in os.walk(model_root / "npu_model", followlinks=False):
        names[:] = sorted(name for name in names if not any(
            word in name.lower() for word in ("hammer", "vlsi")) and name != "__pycache__")
        if any((Path(directory) / name).is_symlink() for name in names):
            raise RuntimeError("model source contains a directory symlink")
        for name in sorted(files):
            if any(word in name.lower() for word in ("hammer", "vlsi")):
                continue
            path = Path(directory) / name
            if path.is_symlink():
                raise RuntimeError("model source contains a file symlink")
            if path.suffix not in {".py", ".json", ".yaml"}:
                continue
            records.append({"path": str(path.relative_to(model_root)),
                            "sha256": sha256(path.read_bytes())})
    return sha256(encoded(records)), records


def numerical_data(model_root: Path):
    records = []
    for directory, names, files in os.walk(model_root / "npu_model/configs/data", followlinks=False):
        names[:] = sorted(name for name in names if not any(word in name.lower() for word in ("hammer", "vlsi")))
        if any((Path(directory) / name).is_symlink() for name in names):
            raise RuntimeError("model numerical data contains a directory symlink")
        for name in sorted(files):
            if any(word in name.lower() for word in ("hammer", "vlsi")):
                continue
            path = Path(directory) / name
            if path.is_symlink():
                raise RuntimeError("model numerical data contains a file symlink")
            records.append({"path": str(path.relative_to(model_root)), "sha256": sha256(path.read_bytes())})
    records.sort(key=lambda record: record["path"])
    return sha256(encoded(records)), records


def cases():
    """Fixed disjoint operand representatives; no boundary/alias variants."""
    for mnemonic, suffix in VPU_BLOCKS.items():
        if mnemonic.startswith("vli."):
            assembly = f"{mnemonic} m4, 0"
        elif mnemonic in {"vpack.bf16.fp8", "vunpack.fp8.bf16"}:
            assembly = f"{mnemonic} m4, m0, e0"
        elif mnemonic in BINARY:
            assembly = f"{mnemonic} m4, m0, m2"
        else:
            assembly = f"{mnemonic} m4, m0"
        yield "vpu." + suffix, "vpu", assembly
    for mxu in ("mxu0", "mxu1"):
        for mnemonic, suffix in MXU_BLOCKS.items():
            if mnemonic == "vmatpush.weight":
                operands = "w0, m0"
            elif mnemonic.startswith("vmatpush"):
                operands = "acc0, m0"
            elif mnemonic == "vmatpop.fp8.acc":
                operands = "m4, acc0, e0"
            elif mnemonic == "vmatpop.bf16.acc":
                operands = "m4, acc0"
            else:
                operands = "acc0, m0, w0"
            yield f"{mxu}.{suffix}", mxu, f"{mnemonic}.{mxu} {operands}"
    yield "vlsu.vload", "vlsu", "vload m4, 0(x0)"
    yield "vlsu.vstore", "vlsu", "vstore m0, 0(x0)"
    yield "xlu.transpose", "xlu", "vtrpose.xlu m4, m0"
    yield "scalar_lsu.load", "scalar_lsu", "lw x4, 0(x0)"
    yield "scalar_lsu.scale_load", "scalar_lsu", "seld e4, 0(x0)"
    yield "scalar_lsu.store", "scalar_lsu", "sw x1, 0(x0)"


def configuration():
    from npu_model.configs.hardware.default import DefaultHardwareConfig
    cfg = DefaultHardwareConfig()
    cfg.arch_state_config = replace(cfg.arch_state_config, numerics="rtl",
                                    randomize_init=False)
    return cfg, configuration_values(cfg)


def configuration_values(cfg):
    """Canonicalize the actual configuration passed to model execution."""
    # Include all scalar/dict configuration values, including inherited defaults.
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
    # Direct unit ticks propagate failures; export also checks Core's actual default.
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


def observe_case(block_name: str, engine: str, assembly: str, *, nonzero=False,
                 instrument=True, max_cycles=1024):
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

    cfg, config_values = configuration()
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
        # Exact initialization is retained, with both equal-value and nonzero runs.
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
        events, levels, states, controls = [], [], [], []
        def event(group, kind, **fields):
            events.append(dict(group=group, kind=kind, cycle=unit.cycle, age=unit.cycle - 1,
                               uop_id=uop.id, pc=uop.pc, mnemonic=mnemonic, **fields))

        if instrument:
            original_mreg = state.conflict_checker.access_mreg
            def mreg(cycle, bank, row, write, owner):
                original_mreg(cycle, bank, row, write, owner)
                if engine == "vpu":
                    row_reduce = ".row." in mnemonic and mnemonic.startswith("vred")
                    group = ("write" if write else "read") + str(
                        bank - (4 if write else 0) if row_reduce else
                        1 if not write and mnemonic in BINARY and bank in (2, 3) else 0)
                elif engine.startswith("mxu"):
                    if write:
                        group = "mreg_write" + str(bank - 4)
                    else:
                        group = "mreg_read" + str(0 if mnemonic.startswith("vmatmul")
                                                  else bank if ".bf16." in mnemonic else 1)
                elif engine == "xlu":
                    group = "write" if write else "read"
                else:
                    group = "mreg_write" if write else "mreg_read"
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
                     assumptions={"scratchpad_read_latency": 1},
                     observation={"accepted": True, "drained": True})
        raw = dict(name=block_name, assembly=assembly, instruction_word=instruction.to_bytecode(),
                   uop_id=uop.id, pc=uop.pc, initialization="nonzero-v1" if nonzero else "zero-v1",
                   initial_state_sha256=initial_digest, final_state_sha256=final_digest,
                   pre_edges=levels, events=events, first_free_age=first_free,
                   configuration=config_values, architectural_states=states, control_states=controls,
                   complete_count=unit.complete_count, total_instructions=unit.total_instructions,
                   busy_cycles=unit.busy_cycles)
        return block, raw


def model_identity(model_root: Path):
    revision = subprocess.check_output(
        ["git", "-C", str(model_root), "rev-parse", "HEAD"], text=True,
        env=dict(os.environ, GIT_OPTIONAL_LOCKS="0")).strip()
    source_sha, sources = source_digest(model_root)
    data_sha, data = numerical_data(model_root)
    for actual, expected, label in ((revision, MODEL_REVISION, "revision"), (source_sha, SOURCE_SHA256, "source"),
                                    (data_sha, NUMERICAL_DATA_SHA256, "numerical data")):
        if actual != expected:
            raise RuntimeError(f"model {label} mismatch: {actual}")
    return dict(revision=revision, source_sha256=source_sha, numerical_data_sha256=data_sha), sources, data


def reference_facts(path: Path):
    if sha256(path.read_bytes()) != REFERENCE_SHA256:
        raise RuntimeError("prototype reference facts digest mismatch")
    value = json.loads(path.read_text())
    expected = {name for name, _, _ in cases()}
    blocks = value.get("op_timing", [])
    if (value.get("schema") != "atlas.model_timing.v1" or value.get("unresolved") or
            len(blocks) != 49 or {block["name"] for block in blocks} != expected):
        raise RuntimeError("prototype reference must cover exactly the 49 probe operations")
    if value["model"]["configuration_sha256"] != REFERENCE_CONFIGURATION_SHA256:
        raise RuntimeError("prototype reference configuration mismatch")
    return value, {block["name"]: block for block in blocks}


def validate_configuration():
    from npu_model.hardware.core import Core
    from npu_model.logging.logger import Logger
    cfg, values = configuration()
    if sha256(encoded(values)) != CONFIGURATION_SHA256:
        raise RuntimeError("effective model configuration mismatch")
    legacy = {key: value for key, value in values.items() if key != "ignore_runtime_errors"}
    if sha256(encoded(legacy)) != REFERENCE_CONFIGURATION_SHA256:
        raise RuntimeError("prototype configuration profile mismatch")
    core = Core(config=cfg, logger=Mock(spec=Logger))
    try:
        if core.ignore_runtime_errors is not False:
            raise RuntimeError("model Core does not propagate runtime errors by default")
    finally:
        core.arch_state.close()
    return values


def result_signature(raw):
    keys = ("initial_state_sha256", "final_state_sha256", "architectural_states", "control_states", "pre_edges",
            "first_free_age", "complete_count", "total_instructions", "busy_cycles", "instruction_word", "configuration")
    return {key: raw[key] for key in keys}


def observer_manifest():
    return [{"path": name, "sha256": sha256(Path(__file__).with_name(name).read_bytes())}
            for name in ("atlas_model_timing.py", "atlas_model_observer.py")]


def export(model_root: Path, output_dir: Path, reference: Path):
    """Admit no facts unless every operation passes parity and noninterference."""
    with patch.object(sys, "dont_write_bytecode", True):
        return _export(model_root, output_dir, reference)


def _export(model_root: Path, output_dir: Path, reference: Path):
    for path in (model_root, output_dir, reference):
        if any(word in part.lower() for part in path.parts for word in ("hammer", "vlsi")):
            raise RuntimeError("prohibited path")
        if path.is_symlink():
            raise RuntimeError("export paths must not be symlinks")
    if output_dir.exists():
        raise RuntimeError("output directory must be new")
    identity, sources, data_sources = model_identity(model_root)
    reference_value, expected = reference_facts(reference)
    sys.path.insert(0, str(model_root.resolve()))
    import npu_model
    import torch
    if Path(npu_model.__file__).resolve().parent.parent != model_root.resolve():
        raise RuntimeError("selected model import does not match --model-root")
    config = validate_configuration()
    observer_sources = observer_manifest()
    output_dir.mkdir(parents=True)
    blocks, runs, checks, unresolved = [], [], [], []
    for name, engine, assembly in cases():
        observed_blocks = []
        for nonzero in (False, True):
            check = dict(name=name, initialization="nonzero-v1" if nonzero else "zero-v1", passed=False)
            checks.append(check)
            try:
                block, observed = observe_case(name, engine, assembly, nonzero=nonzero)
                _, plain = observe_case(name, engine, assembly, nonzero=nonzero, instrument=False)
                runs.extend((dict(observed, observer=True), dict(plain, observer=False)))
                if block != expected[name]:
                    raise RuntimeError("hook-reference event groups/ages/counts or drain bound differ")
                if result_signature(observed) != result_signature(plain):
                    raise RuntimeError("observer changed architectural data/control trace or unit progress")
                check.update(passed=True, parity=True, noninterference=True,
                             result_sha256=sha256(encoded(result_signature(observed))))
                observed_blocks.append(block)
            except Exception as error:
                check["error"] = f"{type(error).__name__}: {error}"
                unresolved.append(dict(check))
        if len(observed_blocks) == 2 and observed_blocks[0] == observed_blocks[1]:
            blocks.append(observed_blocks[0])
        elif not any(item["name"] == name for item in unresolved):
            unresolved.append(dict(name=name, error="zero/nonzero fact shapes differ"))
    after, after_sources, after_data = model_identity(model_root)
    if (identity, sources, data_sources) != (after, after_sources, after_data):
        unresolved.append(dict(error="model sources changed during observation"))
    if observer_sources != observer_manifest():
        unresolved.append(dict(error="observer sources changed during observation"))
    gate = dict(schema="atlas.model_observer_gate.v1", passed=len(blocks) == 49 and not unresolved,
                operations=49, observed_runs=sum(run["observer"] for run in runs),
                unobserved_runs=sum(not run["observer"] for run in runs), checks=checks, unresolved=unresolved,
                model=identity, observer_sources=observer_sources,
                reference_facts_sha256=REFERENCE_SHA256,
                reference_configuration_sha256=REFERENCE_CONFIGURATION_SHA256,
                effective_configuration_sha256=CONFIGURATION_SHA256,
                wrappers="StagedExecution.sample/result, restored by scoped context managers",
                result_observation="inferred commits published only after a successful unit tick")
    def write(name, value):
        (output_dir / name).write_text(json.dumps(value, sort_keys=True, indent=2) + "\n")
    write("gate-report.json", gate)
    write("raw-observations.json", runs)
    write("source-manifest.json", sources)
    write("numerical-data-manifest.json", data_sources)
    if not gate["passed"]:
        raise RuntimeError(f"model observer gate failed; see {output_dir / 'gate-report.json'}")
    inputs = [dict(name=run["name"], assembly=run["assembly"], instruction_word=run["instruction_word"],
                   initialization=run["initialization"], initial_state_sha256=run["initial_state_sha256"])
              for run in runs if run["observer"]]
    validation = dict(passed=True, operation_count=49, observed_runs=98, unobserved_runs=98,
                      reference_facts_sha256=REFERENCE_SHA256,
                      reference_configuration_sha256=REFERENCE_CONFIGURATION_SHA256,
                      gate_report_sha256=sha256((output_dir / "gate-report.json").read_bytes()))
    payload = dict(schema="atlas.model_timing.v1",
                   model=dict(identity, configuration_sha256=CONFIGURATION_SHA256, configuration=config,
                              configuration_json=encoded(config).decode(), observer_sha256=sha256(encoded(observer_sources)),
                              exporter_sha256=sha256(encoded(observer_sources)), input_sha256=sha256(encoded(inputs))),
                   geometry={"rows": 32, "columns": 32, "mreg_count": 64}, age_convention="issue_edge_zero",
                   observation={"accepted": True, "drained": True}, op_timing=blocks, unresolved=[], validation=validation,
                   limits=["Python-model observations; no new RTL or FPGA execution.",
                           "Fixed disjoint operand probes with two input patterns; no pair/capacity evidence.",
                           "MXU writes are inferred from result RHS lookups and successful pinned-source ticks; row IDs are not observed.",
                           "compute_busy/data_busy are isolated lifetime envelopes; next_issue_age is conservative serialization.",
                           "Scratchpad response latency 1 is an assumption, not a measured response.",
                           "Distributed weight reads remain unobserved; retain conservative weight-slot lifetime.",
                           "DMA timing and backend execution are outside this export."],
                   lineage={"configuration": config, "observer_sources": observer_sources,
                            "inputs": inputs, "raw_observations_sha256": sha256(encoded(runs)),
                            "source_manifest_sha256": sha256(encoded(sources)),
                            "numerical_data_manifest_sha256": sha256(encoded(data_sources)),
                            "reference_model": reference_value["model"],
                            "runtime": {"python": sys.version, "torch": torch.__version__, "threads": torch.get_num_threads()}})
    write("model-timing.json", payload)
    write("receipt.json", dict(validation, model=payload["model"],
                               facts_sha256=sha256((output_dir / "model-timing.json").read_bytes()),
                               observer_sources=observer_sources))
    return payload


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--reference-facts", type=Path, required=True)
    args = parser.parse_args()
    result = export(args.model_root, args.output_dir, args.reference_facts)
    print(json.dumps({"resolved": len(result["op_timing"]), "gate_passed": result["validation"]["passed"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
