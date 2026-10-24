"""Shared recovery guidance for source derivation and model evidence refusals."""
from __future__ import annotations

from functools import wraps

UPDATE_GUIDANCE = "See docs/model-timing.md#updating-for-npu-model-changes."


def with_update_guidance(error):
    """Keep exactly one recovery line, last even when errors are nested."""
    message = str(error).replace(UPDATE_GUIDANCE, "").rstrip()
    return message + "\n" + UPDATE_GUIDANCE


class ModelTimingRefusal(Exception):
    """A supported model timing or evidence contract could not be established."""

    def __str__(self):
        return with_update_guidance(super().__str__())


class UnsupportedModelTiming(ModelTimingRefusal, ValueError):
    """An unsupported source, ISA or configuration cannot produce timing facts."""


class ModelTimingEvidenceError(ModelTimingRefusal, ValueError):
    """Supplied timing evidence failed its semantic or identity contract."""


class ModelTimingValidationError(ModelTimingRefusal, RuntimeError):
    """Derived timing disagrees with the independent model observations."""


def model_timing_cli(function):
    """Report expected input failures; leave programming errors debuggable."""
    @wraps(function)
    def checked(*args, **kwargs):
        try:
            return function(*args, **kwargs)
        except (OSError, ValueError, ModelTimingRefusal) as error:
            raise SystemExit(str(error)) from None
    return checked
