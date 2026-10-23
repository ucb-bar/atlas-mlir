"""Bounded native checks for extracted Core edges; no fitting of graph policy."""
from io import StringIO
from unittest.mock import Mock

from atlas_model_diagnostics import ModelTimingValidationError
from atlas_model_replay import semantic_digest
from atlas_search_policy import POLICY, canonical_hash


def validate_replay(cfg, program):
    from npu_model.hardware.core import Core
    from npu_model.logging.logger import Logger
    from npu_model.software.program import InstantiableProgram
    from npu_model.util.converter import stream_to_instrs
    from atlas_derive_frontend import derive_frontend
    frontend = derive_frontend(cfg)
    validation_policy = POLICY["validation"]["core_edges"]
    rows = []
    for delay in validation_policy["delay_immediates"]:
        core = Core(cfg, Mock(spec=Logger))
        try:
            assembly = f"addi x10, x0, 1\ndelay {delay}\naddi x11, x0, 2\necall\n"
            core.load_program(InstantiableProgram(list(stream_to_instrs(StringIO(assembly)))))
            issues, phases = [], []
            original_s1, original_fetch = core._tick_s1, core.ifu.tick
            def s1():
                phases.append("s1")
                return original_s1()
            def fetch(*args, **kwargs):
                phases.append("fetch")
                return original_fetch(*args, **kwargs)
            core._tick_s1, core.ifu.tick = s1, fetch
            for unit in core.exus:
                original = unit.tick
                def tick(*args, _original=original, **kwargs):
                    phases.append("exu")
                    return _original(*args, **kwargs)
                unit.tick = tick
            for _ in range(validation_policy["ordinary_max_cycles"]):
                phases.clear()
                core.tick()
                if not phases or phases[0] != "s1" or phases[-1] != "fetch" or any(p != "exu" for p in phases[1:-1]):
                    raise ModelTimingValidationError("native Core phase order differs from source contract")
                if core.issued_uop is not None:
                    issues.append((core.issued_uop.insn.mnemonic, core.cycle_count))
                if core.arch_state.halted:
                    break
            expected = frontend["delay"]["base_gap"] + (delay & frontend["delay"]["immediate_mask"])
            if (len(issues) != 3 or issues[1][1] - issues[0][1] != frontend["ordinary_issue_gap"]
                    or issues[2][1] - issues[1][1] != expected):
                raise ModelTimingValidationError("native ordinary/DELAY issue gap differs from extracted edges")
            rows.append(dict(kind="ordinary_delay", immediate=delay, issues=issues,
                             expected_successor_gap=expected, one_short_prediction_mismatch=(expected - 1 != issues[2][1] - issues[1][1])))
        finally:
            core.close()
    core = Core(cfg, Mock(spec=Logger))
    try:
        state = core.arch_state
        assembly = "dma.load x2, x1, x3, ch0\ndma.wait ch0\naddi x10, x0, 1\necall\n"
        core.load_program(InstantiableProgram(list(stream_to_instrs(StringIO(assembly)))))
        state.write_xrf(1, state.cfg.dram_base)
        state.write_xrf(2, 0)
        state.write_xrf(3, cfg.dma_beat_bytes)
        clear, wait = None, None
        for _ in range(validation_policy["dma_wait_max_cycles"]):
            before = bool(state.check_flag(0))
            core.tick()
            after = bool(state.check_flag(0))
            if before and not after:
                clear = core.cycle_count
                if core.issued_uop is not None:
                    raise ModelTimingValidationError("DMA.WAIT observed same-edge flag clear")
            if core.issued_uop is not None and core.issued_uop.insn.mnemonic.startswith("dma.wait"):
                wait = core.cycle_count
            if state.halted:
                break
        if clear is None or wait != clear + program["source"]["clock_step"]:
            raise ModelTimingValidationError("native DMA.WAIT visibility differs from extracted Core edges")
        rows.append(dict(kind="dma_wait", clear_edge=clear, wait_issue_edge=wait,
                         wait_stalled_on_clear_edge=wait != clear))
    finally:
        core.close()
    return dict(schema="atlas.core_edge_validation.v1", passed=True, runs=len(rows), checks=rows,
                program_sha256=semantic_digest(program), validation_policy_sha256=canonical_hash(validation_policy),
                scope="Core phase order, ordinary/DELAY issue gaps, prior-edge DMA.WAIT flags; graph projection policy not validated")
