"""Independent, conservative source-trace evaluation.

Inputs are source graph records and unchanged tensor bytes, never emitted
programs or compiler results. See :mod:`.evaluator` for the admitted subset.
"""

from .evaluator import EvaluationError, EvaluationResult, Tensor, evaluate, inspect_ops

__all__ = ["EvaluationError", "EvaluationResult", "Tensor", "evaluate", "inspect_ops"]
