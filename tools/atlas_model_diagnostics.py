"""Model extraction refusals and CLI recovery guidance."""
from functools import wraps

UPDATE_GUIDANCE = "See docs/model-timing.md#updating-for-npu-model-changes."


class UnsupportedModelTiming(ValueError):
    """The selected model cannot supply supported timing facts."""


def model_timing_cli(function):
    @wraps(function)
    def checked(*args, **kwargs):
        try:
            return function(*args, **kwargs)
        except UnsupportedModelTiming as error:
            raise SystemExit(str(error) + "\n" + UPDATE_GUIDANCE) from None
        except (OSError, ValueError) as error:
            raise SystemExit(str(error)) from None
    return checked
