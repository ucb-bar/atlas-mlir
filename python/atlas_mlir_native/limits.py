"""Default functional execution budget shared by target audit tools."""
DEFAULT_FUNCTIONAL_INSTRUCTION_LIMIT = 1_000_000


def functional_instruction_limit(limits):
    """Use the Atlas default, optionally narrowed by a target API caller.

    Merlin SearchLimits contains search budgets only, so ordinary Merlin calls
    always use DEFAULT_FUNCTIONAL_INSTRUCTION_LIMIT. An Atlas API caller may
    supply a smaller functional_instruction_limit; this is not Merlin vocabulary.
    """
    value = getattr(limits, "functional_instruction_limit", DEFAULT_FUNCTIONAL_INSTRUCTION_LIMIT)
    if type(value) is not int or value <= 0 or value > DEFAULT_FUNCTIONAL_INSTRUCTION_LIMIT:
        raise ValueError("functional instruction limit must be positive and within the default cap")
    return value
