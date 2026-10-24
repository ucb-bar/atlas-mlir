"""Derive VectorFSM timing from its issue metadata and pure row planners."""
from __future__ import annotations

from atlas_model_diagnostics import UnsupportedModelTiming

import ast
from collections import defaultdict
import inspect
from pathlib import Path
from types import FunctionType, SimpleNamespace

from atlas_model_schema import (SOURCE_SHAPES, VPU_BLOCKS, ast_digest, mreg_event,
                                reservation, timing_block)
from atlas_model_timing import compress


class UnsupportedVPU(UnsupportedModelTiming):
    """Selected source or geometry has no supported timing interpretation."""


def _normalize_slot_capacity(tree):
    """Read the reviewed ISSUE_SLOTS constant and guard its two consumers.

    Native saturation fixtures cover one through three slots. Only the module
    constant is normalized; range(ISSUE_SLOTS), len(live) == ISSUE_SLOTS and all
    admission/row-planning control flow remain protected by the complete AST.
    """
    from atlas_derive_mxu import _assignment, _method, _one, _name
    cls = _one((n for n in tree.body if isinstance(n, ast.ClassDef)
                and n.name == "VectorExecutionUnit"), "VPU class")
    capacity = _assignment(tree, "ISSUE_SLOTS")
    if (not isinstance(capacity.value, ast.Constant) or type(capacity.value.value) is not int
            or not 1 <= capacity.value.value <= 3):
        raise UnsupportedVPU("unsupported VPU source slot capacity (requires 1..3)")
    slots = list(range(capacity.value.value))
    slot = _assignment(_method(cls, "_accept"), "slot")
    try:
        generator = slot.value.args[0]
        iterator = generator.generators[0].iter
    except (AttributeError, IndexError, TypeError) as error:
        raise UnsupportedVPU("unsupported VPU source slot generator") from error
    if (not isinstance(generator, ast.GeneratorExp) or not isinstance(iterator, ast.Call)
            or not _name(iterator.func, "range") or len(iterator.args) != 1
            or iterator.keywords or not _name(iterator.args[0], "ISSUE_SLOTS")):
        raise UnsupportedVPU("unsupported VPU source slot identifiers")
    count = _one((n for n in ast.walk(_method(cls, "_issue_blocked"))
                  if isinstance(n, ast.Compare) and isinstance(n.left, ast.Call)
                  and isinstance(n.left.func, ast.Name) and n.left.func.id == "len"), "VPU slot count")
    if (len(count.ops) != 1 or not isinstance(count.ops[0], ast.Eq) or len(count.comparators) != 1
            or not _name(count.comparators[0], "ISSUE_SLOTS")):
        raise UnsupportedVPU("VPU issue capacity differs from its source slots")
    capacity.value = ast.Constant(value=2)
    return slots


def _check_source(module, *, latency_table=False):
    path = Path(inspect.getsourcefile(module))
    if any(word in part.lower() for part in path.parts for word in ("hammer", "vlsi")):
        raise UnsupportedVPU("prohibited model source path")
    if ".." in path.parts:
        raise ValueError("parent traversal is unsupported")
    if any(parent.is_symlink() for parent in reversed((path, *path.parents))):
        raise UnsupportedVPU("model source path contains a symlink")
    tree = ast.parse(path.read_text())
    if latency_table:
        tree.body = [node for node in tree.body if not (
            isinstance(node, ast.Assign) and any(isinstance(target, ast.Name) and target.id == "VPU_OP_LATENCIES"
                                               for target in node.targets))]
        if any(isinstance(n, ast.ClassDef) and n.name == "VectorExecutionUnit" for n in tree.body):
            _normalize_slot_capacity(tree)
    digest = ast_digest(tree)
    if digest != SOURCE_SHAPES.get(path.name):
        raise UnsupportedVPU(f"unsupported VPU timing semantics in {path.name}")


def _instruction(cfg, mnemonic, vpu):
    from npu_model.isa import EXU
    try:
        cls = cfg.isa.operations[mnemonic]
    except (AttributeError, KeyError) as error:
        raise UnsupportedVPU(f"missing ISA operation {mnemonic}") from error
    if cls.mnemonic != mnemonic or cls.exu != EXU.VECTOR:
        raise UnsupportedVPU(f"unsupported ISA dispatch for {mnemonic}")
    if mnemonic.startswith("vli."):
        operands = dict(vd=4, imm=0)
    elif mnemonic in {"vpack.bf16.fp8", "vunpack.fp8.bf16"}:
        operands = dict(vd=4, vs2=0, es1=0)
    elif mnemonic in vpu._TWO_INPUT:
        operands = dict(vd=4, vs1=0, vs2=2)
    else:
        operands = dict(vd=4, vs1=0)
    if set(inspect.signature(cls).parameters) != set(operands):
        raise UnsupportedVPU(f"unsupported ISA operands for {mnemonic}")
    return cls(**operands)


def _plans(cfg):
    from npu_model.hardware import bank_conflict, vpu
    # Importing definitions populates the selected IsaSpec; no instruction executes.
    import npu_model.configs.isa_definition  # noqa: F401
    from npu_model.isa import EXU

    _check_source(vpu, latency_table=True)
    _check_source(bank_conflict)
    geometry = cfg.arch_state_config
    if (geometry.mrf_depth, geometry.mrf_width, geometry.num_m_registers) != (32, 32, 64):
        raise UnsupportedVPU("VectorFSM timing requires 64 MREGs of 32 rows of 32 bytes")
    vectors = {name for name, cls in cfg.isa.operations.items() if cls.exu == EXU.VECTOR}
    if vectors != set(VPU_BLOCKS):
        raise UnsupportedVPU("selected ISA has unsupported VPU operation coverage")
    # Run the source's issue planner with an inert StagedExecution constructor.
    # A private globals mapping leaves all imported numerical methods untouched.
    accept = vpu.VectorExecutionUnit._accept
    issue = FunctionType(accept.__code__, dict(accept.__globals__, StagedExecution=lambda *args: None),
                         accept.__name__, accept.__defaults__, accept.__closure__)
    plans = []
    for mnemonic, suffix in VPU_BLOCKS.items():
        unit = object.__new__(vpu.VectorExecutionUnit)
        unit.name, unit.config, unit.lane_id = "vpu", cfg, 0
        unit.arch_state = SimpleNamespace(cfg=geometry, conflict_checker=bank_conflict.BankConflictChecker())
        unit.logger = SimpleNamespace(log_stage_start=lambda *args, **kwargs: None)
        unit.reset()
        unit.cycle = 1
        uop = SimpleNamespace(insn=_instruction(cfg, mnemonic, vpu), id=0, execute_delay=0)
        latency = unit._execution_latency(uop)
        if type(latency) is not int or not 1 <= latency < 4096:
            raise UnsupportedVPU(f"unsupported VPU latency for {mnemonic}: {latency!r}")
        issue(unit, uop)
        if len(unit.operations) != 1:
            raise UnsupportedVPU("VPU issue did not produce exactly one row plan")
        operation, = unit.operations
        if operation.write_last != latency - 1 or operation.issued != 1:
            raise UnsupportedVPU("VPU issue ages disagree with its latency")
        if operation.reads and operation.read_last > operation.write_last:
            raise UnsupportedVPU(f"{mnemonic} completes before its reserved reads release")
        written = [row for age in range(operation.write_last + 1) for row in unit._writes(operation, age)]
        if sorted(written) != [(bank, row) for bank in sorted(operation.writes) for row in range(geometry.mrf_depth)]:
            raise UnsupportedVPU(f"{mnemonic} lifetime truncates or duplicates destination rows")
        plans.append(("vpu." + suffix, mnemonic, unit, operation))
    return plans


def derive_vpu(cfg):
    """Derive blocks and deferred releases from guarded issue/row plans."""
    blocks, releases = {}, {}
    for name, mnemonic, unit, operation in _plans(cfg):
        grouped = defaultdict(list)
        for age in range(operation.write_last + 1):
            requests = unit._reads(operation, age)
            if len(set(requests)) != len(requests):
                raise UnsupportedVPU("duplicate isolated VPU read requests require cache interpretation")
            for write, accesses in ((False, requests), (True, unit._writes(operation, age))):
                for bank, row in accesses:
                    if not 0 <= bank < 64 or not 0 <= row < 32:
                        raise UnsupportedVPU("VPU row plan accesses unsupported geometry")
                    group = mreg_event(mnemonic, "vpu", bank, write)
                    grouped[group].append(age)
        blocks[name] = timing_block(name, "vpu", {group: compress(ages) for group, ages in sorted(grouped.items())},
                                    operation.write_last)
        entry = {}
        for kind, banks, last in (("mreg_read", operation.reads, operation.read_last),
                                  ("mreg_write", operation.writes, operation.write_last)):
            if banks:
                entry[kind] = reservation(last, banks=sorted(banks))
        entry["unit"] = reservation(operation.write_last)
        releases[name] = entry
    return blocks, releases
