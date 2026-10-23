"""Validate source-derived weight windows against isolated native MXU edges.

Observation records masks returned by the unchanged native sampler, including
uses of unchanged data. Push/compute boundary probes also check real numerical
results and the native rejection at the immediately preceding issue edge.
"""
from __future__ import annotations

from io import StringIO
from unittest.mock import Mock, patch

from atlas_model_diagnostics import ModelTimingValidationError


def _native(cfg, engine, *, slot=0, push_gap=None, push_first=False, observe=False, mutation=None):
    import torch
    from npu_model.hardware.arch_state import ArchState
    from npu_model.hardware.mxu import MatrixExecutionUnitInner, MatrixExecutionUnitSystolic
    from npu_model.logging.logger import Logger
    from npu_model.software.instruction import Uop
    from npu_model.util.converter import stream_to_instrs

    state = ArchState(cfg.arch_state_config)
    unit_type = MatrixExecutionUnitSystolic if engine == "mxu0" else MatrixExecutionUnitInner
    try:
        state.write_mrf_fp8(0, torch.ones((32, 32)).to(torch.float8_e4m3fn))
        state.write_mrf_fp8(2, torch.full((32, 32), 2.).to(torch.float8_e4m3fn))
        for selected in (0, 1):
            state.write_wb_fp8(engine, selected, torch.eye(32).to(torch.float8_e4m3fn))
        unit = unit_type(engine, Mock(spec=Logger), state, config=cfg)
        compute = f"vmatmul.{engine} acc0, m0, w{slot}\n"
        push = f"vmatpush.weight.{engine} w{slot}, m2\n"
        assembly = compute
        if push_gap is not None:
            assembly = push + compute if push_first else compute + push
        commands = [Uop(insn, pc=i * 4) for i, insn in enumerate(stream_to_instrs(StringIO(assembly)))]
        uses = []
        original = unit._weight_uses
        def weight_uses(age):
            used, first = original(age)
            uses.append((age, used.clone(), first.clone()))
            return used, first
        with patch.object(unit, "_weight_uses", weight_uses if observe else original):
            for age in range(160):
                if mutation is not None and age == mutation["age"]:
                    state.read_wb_u8(engine, slot).view(torch.float8_e4m3fn)[mutation["row"], mutation["column"]] = 2.
                command = commands[0] if age == 0 else commands[1] if push_gap == age else None
                unit.tick(command)
                if age > 0 and not unit.has_in_flight:
                    break
            else:
                raise ModelTimingValidationError("MXU weight boundary probe did not drain")
        if observe:
            # The completing operation leaves _ops on its last use; query the
            # unchanged native predicate once more to verify the closed bound.
            weight_uses(age + 1)
        expected = torch.full((32, 32), 64. if push_first else 1., dtype=torch.bfloat16)
        if mutation is not None and mutation.get("changes_result"):
            # Weight storage rows are output columns; the selected off-diagonal
            # coefficient changes every output row's corresponding column.
            expected[:, mutation["row"]] += 2.
        if not torch.equal(state.acc[engine][0], expected):
            raise ModelTimingValidationError("MXU weight boundary produced an incorrect matmul result")
        if push_gap is not None and not torch.equal(state.read_wb_u8(engine, slot), state.read_mrf_u8(2)):
            raise ModelTimingValidationError("MXU boundary weight push did not commit its bytes")
        return uses
    finally:
        state.close()


def validate_weight_timing(cfg, facts):
    """Require exact masks on all PE use edges and real safe/unsafe overwrites."""
    import torch
    checks = []
    runs = dict(isolated_runs=0, boundary_runs=0, consumer_runs=0, mutable_pe_runs=0)
    def run(kind, engine, **kwargs):
        runs[kind] += 1
        return _native(cfg, engine, **kwargs)
    for engine, value in facts["engines"].items():
        window = value["use_window"]
        grid = torch.arange(32)
        first = window["first_age"] + grid[:, None] * window["row_step"] + grid[None, :] * window["column_step"]
        last = window["last_age"] + grid[:, None] * window["row_step"] + grid[None, :] * window["column_step"]
        for slot in (0, 1):
            uses = run("isolated_runs", engine, slot=slot, observe=True)
            for age, used, native_first in uses:
                if not torch.equal(used, (first <= age) & (age <= last)) or not torch.equal(native_first, first == age):
                    raise ModelTimingValidationError(f"{engine} weight-use mask mismatch at age {age}")
            ages = {age for age, _, _ in uses}
            if not set(range(int(first.min()), int(last.max()) + 2)) <= ages:
                raise ModelTimingValidationError(f"{engine} weight-use observation omits a boundary edge")
            checks.append(dict(engine=engine, slot=slot, kind="isolated_use_masks", passed=True,
                               pe_count=1024, first_age=int(first.min()), last_age=int(last.max()),
                               post_use_age=int(last.max()) + 1))
        # A row writer cannot overwrite until all inner lanes' last reads.
        # Source reads occur before writes, so equal-cycle read/write is safe.
        safe_gap = max(row["last_age"] - i for i, row in enumerate(value["use_rows"]))
        run("boundary_runs", engine, push_gap=safe_gap)
        try:
            run("boundary_runs", engine, push_gap=safe_gap - 1)
        except RuntimeError as error:
            expected = "before previous matmul on same slot drained" if engine == "mxu0" else "weight slot busy"
            if expected not in str(error):
                raise ModelTimingValidationError(f"{engine} early overwrite failed for an unrelated reason") from error
        else:
            raise ModelTimingValidationError(f"{engine} accepted a one-cycle-too-early weight overwrite")
        checks.append(dict(engine=engine, kind="native_overwrite_boundary", passed=True,
                           accepted_gap=safe_gap, rejected_gap=safe_gap - 1, numerical_result_verified=True))
        ready_gap = max(i + 1 - row["first_age"] + 1 for i, row in enumerate(value["use_rows"]))
        run("consumer_runs", engine, push_gap=ready_gap, push_first=True)
        if ready_gap > 1:
            try:
                run("consumer_runs", engine, push_gap=ready_gap - 1, push_first=True)
            except RuntimeError as error:
                if "compute reads buffer with an active push" not in str(error):
                    raise ModelTimingValidationError("early weight consumer failed for an unrelated reason") from error
            else:
                raise ModelTimingValidationError("native MXU accepted an early weight consumer")
        checks.append(dict(engine=engine, kind="native_consumer_boundary", passed=True,
                           accepted_gap=ready_gap, rejected_gap=ready_gap - 1 if ready_gap > 1 else None,
                           preceding_edge_expressible=ready_gap > 1,
                           issue_constraint="one command per unit per edge", numerical_result_verified=True))
        # An asymmetric PE validates the output-column/inner mapping as well
        # as the sampler's first-use change and later-use rejection behavior.
        row, column = 7, 3
        first_use = int(first[row, column])
        last_use = int(last[row, column])
        for age, changes_result, rejected in ((first_use, True, False), (first_use + 1, False, True),
                                               (last_use, False, True), (last_use + 1, False, False)):
            try:
                run("mutable_pe_runs", engine, mutation=dict(age=age, row=row, column=column, changes_result=changes_result))
            except RuntimeError as error:
                if not rejected or "weights changed while a matmul was using them" not in str(error):
                    raise ModelTimingValidationError("native weight mutation failed unexpectedly") from error
            else:
                if rejected:
                    raise ModelTimingValidationError("native MXU missed an active weight mutation")
        checks.append(dict(engine=engine, kind="native_mutable_pe_boundaries", passed=True, row=row, column=column,
                           accepted_first_age=first_use, rejected_first_late_age=first_use + 1,
                           rejected_last_age=last_use, accepted_post_last_age=last_use + 1,
                           numerical_result_verified=True))
    return dict(passed=True, checks=checks, **runs)
