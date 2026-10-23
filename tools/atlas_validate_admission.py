"""Bounded native gate for frozen source-derived engine admission predicates."""
from __future__ import annotations

from copy import deepcopy
from io import StringIO
from unittest.mock import Mock

from atlas_derive_admission import derive_admission, semantic_admission
from atlas_derive_overlap import command
from atlas_model_schema import canonical, sha256
from atlas_validate_overlap import initialize_state, state_snapshot


def sample_plan(admission, per_op):
    """Freeze cases from source mechanism boundaries before native execution."""
    result = []
    def add(label, assembly, issues, accepted=True):
        result.append(dict(id=label, assembly=assembly, issues=issues, accepted=accepted))
    blocks = {block["name"]: block for block in per_op["op_timing"]}
    final = admission["vpu"]["final_write_age"]
    for name in sorted(final):
        _, insn = command(name, dict(rs1=0, rs2=2, rd=4))
        xlu = "vtrpose.xlu m22, m20"
        add(name + "-xlu", [insn, xlu], [0, 1])
        add("xlu-" + name, [xlu, insn], [0, 1])
        _, second = command(name, dict(rs1=8, rs2=10, rd=12))
        add(name + "-repeat-final", [insn, second], [0, final[name]])
        add(name + "-repeat-short", [insn, second], [0, final[name]-1], False)
    # Shared logic and double-input issue guards are independent of bank use.
    for index, group in enumerate(admission["vpu"]["logic_groups"]):
        if len(group) < 2:
            continue
        a, b = group[:2]
        _, first = command(a, dict(rs1=0, rs2=2, rd=4))
        _, second = command(b, dict(rs1=8, rs2=10, rd=12))
        add(f"logic-{index}-short", [first, second], [0, final[a]-1], False)
        add(f"logic-{index}-final", [first, second], [0, final[a]])
    for name in admission["vpu"]["double"]:
        _, first = command(name, dict(rs1=0, rs2=2, rd=4))
        _, second = command("vpu.vliOne", dict(rs1=8, rs2=10, rd=12))
        add(name + "-double-short", [first, second], [0, final[name]-1], False)
        add(name + "-double-final", [first, second], [0, final[name]])
    capacity = admission["vpu"]["capacity"]
    if type(capacity) is not int or not 1 <= capacity <= 3:
        raise ValueError("unsupported source VPU slot capacity")
    names = ("vpu.relu", "vpu.exp2", "vpu.sin", "vpu.fp8pack")[:capacity + 1]
    instructions = [command(name, dict(rs1=8*i, rs2=8*i, rd=8*i+4))[1]
                    for i, name in enumerate(names)]
    add("vpu-source-slots-full", instructions, list(range(capacity + 1)), False)
    add("vpu-final-slot-reuse", instructions, [*range(capacity), final["vpu.relu"]])
    xfree = admission["xlu"]["initiation_age"]
    add("xlu-self-busy", ["vtrpose.xlu m4, m0", "vtrpose.xlu m12, m8"], [0, xfree-1], False)
    add("xlu-self-free", ["vtrpose.xlu m4, m0", "vtrpose.xlu m12, m8"], [0, xfree])
    # Independent engines still share direction-aware logical reservations.
    add("vpu-xlu-write-reservation", ["vrelu.bf16 m4, m0", "vtrpose.xlu m12, m4"], [0, 1], False)
    add("xlu-vpu-write-reservation", ["vtrpose.xlu m4, m0", "vrelu.bf16 m12, m4"], [0, 1], False)
    # Different logical registers can still conflict in one physical bank.
    add("xlu-vpu-physical-bank", ["vtrpose.xlu m4, m0", "vrelu.bf16 m12, m32"], [0, 1], False)
    add("vpu-xlu-physical-bank", ["vrelu.bf16 m4, m0", "vtrpose.xlu m12, m32"], [0, 1], False)
    # Column reduction wraps its pair and can share identical native VPU reads.
    # Align RELU's first read with the reduction's next complete source-pair
    # pass. One slot cannot overlap these commands, even with coalesced reads.
    reduction = blocks["vpu.cmax"]["events"]["read0"]
    elementwise = blocks["vpu.relu"]["events"]["read0"]
    shared_read_age = reduction["first_age"] + elementwise["count"] * reduction["step"]
    coalescing_gap = shared_read_age - elementwise["first_age"]
    assembly = ["vredmax.bf16 m4, m0", "vrelu.bf16 m12, m0"]
    add("vpu-read-coalescing", assembly, [0, coalescing_gap], capacity >= 2)
    if capacity == 1:
        add("vpu-read-coalescing-final-slot-reuse", assembly, [0, final["vpu.cmax"]])
    add("vls-paths-independent", ["vload m4, 0(x2)", "vstore m8, 0(x3)"], [0, 1])
    add("vls-paths-bank-conflict", ["vload m4, 0(x2)", "vstore m8, 0(x2)"], [0, 1], False)
    for first, second, directions in (("vload", "vstore", ("read", "write")),
                                       ("vstore", "vload", ("write", "read"))):
        a = blocks["vlsu." + first]["events"]["vmem_" + directions[0]]
        b = blocks["vlsu." + second]["events"]["vmem_" + directions[1]]
        suffix = a["last_age"] - b["first_age"] + 1
        assembly = [f"{first} m4, 0(x2)", f"{second} m8, 0(x2)"]
        add(first + "-" + second + "-same-bank-suffix", assembly, [0, suffix])
        add(first + "-" + second + "-same-bank-short", assembly, [0, suffix-1], False)
    free = admission["vlsu"]["initiation_ages"]["vlsu.vload"]
    add("vload-path-busy", ["vload m4, 0(x2)", "vload m8, 0(x3)"], [0, free-1], False)
    add("vload-path-free", ["vload m4, 0(x2)", "vload m8, 0(x3)"], [0, free])
    gap = admission["scalar_lsu"]["load_response_gap"]
    add("scalar-load-response-busy", ["lw x4, 0(x2)", "lw x5, 0(x6)"], [0, gap-1], False)
    add("scalar-load-response-free", ["lw x4, 0(x2)", "lw x5, 0(x6)"], [0, gap])
    add("scalar-stores-independent", ["sw x4, 0(x2)", "sw x5, 0(x6)"], [0, 1])
    add("scalar-vector-independent", ["vload m4, 0(x2)", "lw x4, 0(x6)"], [0, 1])
    add("scalar-vector-bank-conflict", ["vload m4, 0(x2)", "lw x4, 0(x2)"], [0, 1], False)
    return result


def native_run(sample, nonzero, serial=False):
    """Independent native units, one command/edge, shared native checker/state."""
    from npu_model.hardware.arch_state import ArchState
    from npu_model.hardware.vpu import VectorExecutionUnit
    from npu_model.hardware.xlu import CrossLaneExecutionUnit
    from npu_model.hardware.lsu import LoadStoreUnit
    from npu_model.logging.logger import Logger
    from npu_model.software.instruction import Uop
    from npu_model.util.converter import stream_to_instrs
    from atlas_model_timing import configuration
    cfg, _ = configuration()
    state = ArchState(cfg.arch_state_config)
    logger = Mock(spec=Logger)
    units = [cls(name, logger, state, config=cfg) for cls, name in
             ((VectorExecutionUnit, "Vector0"), (CrossLaneExecutionUnit, "XLU"), (LoadStoreUnit, "LSU"))]
    try:
        initialize_state(state, nonzero, {"2": 0, "3": cfg.vmem_bank_bytes // 4, "6": cfg.vmem_bank_bytes})
        insns = list(stream_to_instrs(StringIO("\n".join(sample["assembly"]))))
        if len(insns) != len(sample["issues"]):
            raise ValueError("native admission sample length differs")
        uops = [Uop(insn, pc=index*4) for index, insn in enumerate(insns)]
        next_index, accepted = 0, []
        for age in range(8192):
            issue = None
            if next_index < len(uops) and (
                    (not serial and age == sample["issues"][next_index]) or
                    (serial and (not accepted or not any(unit.has_in_flight for unit in units)))):
                issue = uops[next_index]
            state.current_uop = issue
            try:
                for unit in units:
                    selected = issue if issue is not None and unit.can_handle(issue) else None
                    before = unit.total_instructions
                    unit.tick(selected)
                    if selected is not None and unit.total_instructions != before + 1:
                        raise ValueError("native unit did not accept exact issue")
            except Exception as error:
                message = str(error)
                resource = (type(error).__name__ == "BankConflictError" or
                            isinstance(error, RuntimeError) and any(text in message for text in
                            ("issue-busy bit", "engine is busy", "path is busy", "prior load response pending")))
                if not resource:
                    raise
                return dict(accepted=False, issued_ages=accepted, fault_age=age,
                            fault_type=type(error).__name__, fault=message)
            if issue is not None:
                accepted.append(age)
                next_index += 1
            if next_index == len(uops) and not any(unit.has_in_flight for unit in units):
                for unit in units:
                    unit.flush_completions()
                return dict(accepted=True, issued_ages=accepted, drained=True,
                            final_state_sha256=sha256(canonical(state_snapshot(state))))
        raise ValueError("native admission sample failed to drain")
    finally:
        state.close()


def validate_admission(model, per_op, admission):
    import torch
    torch.set_num_threads(1)
    # Native execution validates the frozen derived proposal; it cannot modify it.
    if semantic_admission(derive_admission(model, per_op)) != semantic_admission(admission):
        raise ValueError("engine admission differs from frozen source derivation")
    plan = sample_plan(admission, per_op)
    rows = []
    positives = negatives = numerical = 0
    for sample in plan:
        for nonzero in (False, True):
            result = native_run(sample, nonzero)
            if result["accepted"] != sample["accepted"]:
                raise ValueError(f"native engine admission disagreement: {sample['id']}: {result}")
            if result["accepted"]:
                if result["issued_ages"] != sample["issues"] or not result["drained"]:
                    raise ValueError("native engine admission issue/drain mismatch")
                reference = native_run(sample, nonzero, serial=True)
                if not reference["accepted"] or reference["final_state_sha256"] != result["final_state_sha256"]:
                    raise ValueError("native engine admission serial-state mismatch: " + sample["id"])
                positives += 1
                numerical += 1
            else:
                if result["fault_age"] < sample["issues"][1]:
                    raise ValueError("negative sample failed before overlapped command")
                negatives += 1
            rows.append(dict(id=sample["id"], nonzero=nonzero, result=result))
    report = dict(schema="atlas.engine_admission_validation.v1", passed=True,
                  sample_policy="source-engine-boundaries-v1", samples=plan, runs=rows,
                  positive_runs=positives, negative_runs=negatives, numerical_checks=numerical)
    qualified = deepcopy(admission)
    qualified["validation"] = dict(passed=True, report_sha256=sha256(canonical(report)),
                                   positive_runs=positives, negative_runs=negatives,
                                   numerical_checks=numerical)
    return qualified, report
