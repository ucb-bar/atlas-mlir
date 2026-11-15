"""Default functional execution budget shared by target audit tools."""
DEFAULT_FUNCTIONAL_INSTRUCTION_LIMIT = 1_000_000


def functional_instruction_limit(limits=None):
    """Resolve a software execution budget, independently of physical IMEM.

    Merlin owns the optional invocation setting; the target supplies the legacy
    default. A target API caller can further constrain an invocation. Every
    accepted image still requires an independently proved finite dynamic bound
    and the unchanged RTL instruction-memory capacity check.
    """
    from merlin.semantic_compiler.codegen import current_codegen_options
    selected = current_codegen_options().execution_instruction_limit
    if selected is None:
        selected = DEFAULT_FUNCTIONAL_INSTRUCTION_LIMIT
    value = getattr(limits, "functional_instruction_limit", selected)
    if type(value) is not int or value <= 0:
        raise ValueError("functional instruction limit must be a positive integer")
    return min(value, selected)
