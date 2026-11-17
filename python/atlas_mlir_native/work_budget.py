"""Native compiler work limits, independent of host scheduling."""

from merlin.semantic_compiler.target_binding import NativeCompilationError

BUFFER_SOLVER_RLIMIT = 100_000_000
TOOL_HANG_GUARD_SECONDS = 600


class CountedWorkExhausted(NativeCompilationError):
    def __init__(self, reason):
        super().__init__("resource_limit", reason)


def allocate_native_buffers(buffers, banks, **kwargs):
    """Use Merlin's checked solver with a deterministic per-query work limit.

    An inconclusive query aborts the invocation. It cannot select another
    placement, retry with a fresh budget, or justify a different route.
    """
    from merlin.semantic_compiler.buffer_allocation import allocate_buffers

    result = allocate_buffers(
        buffers, banks, timeout_ms=None, rlimit=BUFFER_SOLVER_RLIMIT, **kwargs
    )
    if result.status == "unknown":
        raise CountedWorkExhausted(
            f"native work budget exhausted: buffer_solver_rlimit_per_query={BUFFER_SOLVER_RLIMIT}: "
            + result.reason,
        )
    return result
