"""Scoped external observation of the pinned model's isolated MXU commands.

Reads are samples. Writes are result lookups immediately before unconditional
row assignments in the pinned source; publish them only after a successful tick.
This is inferred commit evidence, not a public post-assignment callback.
"""
from contextlib import contextmanager
from unittest.mock import patch


_active = False


@contextmanager
def observe_mxu(staged_type, unit, uop, sink):
    """Yield a successful-tick publisher; restore both methods on every exit."""
    global _active
    if _active:
        raise RuntimeError("MXU observation is process-local and cannot be nested")
    sample, result = staged_type.sample, staged_type.result
    pending = []

    def selected(staged, location):
        return (staged.state is unit.arch_state and staged.unit == unit.mxu and
                staged.uop is uop and location[0] in {"acc", "wb"})

    def sampled(staged, location, index, value):
        answer = sample(staged, location, index, value)
        if selected(staged, location) and location[0] == "acc":
            sink(kind="acc_read", cycle=unit.cycle, uop=uop, location=location,
                 row=int(index), evidence="sample_call")
        return answer

    def retrieved(staged, location):
        answer = result(staged, location)
        if selected(staged, location):
            mnemonic = uop.insn.mnemonic
            if location[0] == "wb" and mnemonic.startswith("vmatpush.weight"):
                kind = "weight_write"
            elif location[0] == "acc" and mnemonic.startswith(("vmatpush", "vmatmul")):
                kind = "acc_write"
            else:
                raise RuntimeError("uncovered MXU result observer site")
            pending.append(dict(kind=kind, cycle=unit.cycle, uop=uop, location=location,
                                evidence="result_rhs_and_successful_tick"))
        return answer

    def finish_tick():
        events = list(pending)
        pending.clear()
        for event in events:
            sink(**event)

    _active = True
    try:
        with patch.object(staged_type, "sample", sampled), patch.object(staged_type, "result", retrieved):
            yield finish_tick
    finally:
        pending.clear()
        _active = False
