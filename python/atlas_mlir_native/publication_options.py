"""Explicit invocation scope for optional deterministic native receipts."""

from contextlib import contextmanager
from contextvars import ContextVar


_deterministic_receipts = ContextVar("atlas_deterministic_receipts", default=False)


def deterministic_receipts_enabled():
    return _deterministic_receipts.get()


@contextmanager
def publication_options(*, deterministic_receipts=False):
    """Keep normal publication bytes unless the caller explicitly opts in."""
    if type(deterministic_receipts) is not bool:
        raise TypeError("deterministic_receipts must be a boolean")
    token = _deterministic_receipts.set(deterministic_receipts)
    try:
        yield
    finally:
        _deterministic_receipts.reset(token)


def valid_receipt_timing(receipt):
    """Require bounded timing by default and a declared deterministic mode."""
    if deterministic_receipts_enabled():
        return receipt.get("receipt_mode") == "deterministic-v1" and "elapsed_s" not in receipt
    if "receipt_mode" in receipt:
        return False
    elapsed = receipt.get("elapsed_s")
    return type(elapsed) in (int, float) and 0 <= elapsed <= 30
