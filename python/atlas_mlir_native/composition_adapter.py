"""Expose proved adjacent device groups to Merlin's region selector."""
from __future__ import annotations

from merlin.compile.application_plan import select_application_plan

# Recognizers retain ownership of their source/domain proofs. The wrapper owns
# their state and dispatch, so ordinary AtlasApplicationAdapter callers retain
# the existing smaller regions unless they explicitly opt into enlargement.
_ADJACENT_PATTERNS = (
    ("adjacent_pointwise", "_adjacent_groups"),
    ("adjacent_square_reduce", "_square_groups"),
    ("adjacent_reduction_scale", "_reduction_scale_groups"),
    ("adjacent_pointwise_chain", "_pointwise_chain_groups"),
    ("quantized_epilogue", "_quantized_epilogue_groups"),
    ("quantized_epilogue_narrowing", "_quantized_epilogue_narrowing_groups"),
    ("quantized_narrowing", "_quantized_narrowing_groups"),
    ("pointwise_narrowing", "_pointwise_narrowing_groups"),
    ("softmax_narrowing", "_softmax_narrowing_groups"),
    ("softmax_input_widening", "_softmax_input_widening_groups"),
    ("expanded_silu", "_expanded_silu_groups"),
    ("quantized_operand_permutation", "_quantized_operand_permutation_groups"),
)


class AdjacentDeviceAdapter:
    """Opt into proved adjacent device groups while retaining smaller choices."""

    def __init__(self, base, *, patterns=None):
        self.base = base
        if patterns is None:
            self.patterns = _ADJACENT_PATTERNS
        else:
            names = tuple(patterns)
            supported = dict(_ADJACENT_PATTERNS)
            if len(set(names)) != len(names) or any(name not in supported for name in names):
                raise ValueError("unknown or duplicate adjacent device pattern")
            self.patterns = tuple((name, supported[name]) for name in names)
        for _, state in self.patterns:
            setattr(self, state, {})

    def __getattr__(self, name):
        return getattr(self.base, name)

    def application_region_alternatives(self, source):
        from importlib import import_module

        choices = tuple(self.base.plan(source).selected)
        for module, _ in self.patterns:
            choices = import_module("." + module, __package__).alternatives(self, source, choices)
        return choices

    def plan(self, source):
        return select_application_plan(source, self.application_region_alternatives(source))

    def _group(self, region, alternative):
        from importlib import import_module

        kind = dict(alternative.metadata)["kind"]
        for module, state in self.patterns:
            if module == kind:
                return import_module("." + module, __package__), getattr(self, state)[region.digest()]
        return None

    def compile_region(self, snapshot, region, alternative, name, segments, target_source, limits):
        selected = self._group(region, alternative)
        if selected is None:
            return self.base.compile_region(snapshot, region, alternative, name, segments, target_source, limits)
        module, group = selected
        module.compile_group(group, snapshot, target_source=target_source, destination=segments / name, limits=limits)
        return module.export_group(group, segments / name, name, target_source=target_source)

    def export_region(self, region, alternative, name, segments, target_source):
        selected = self._group(region, alternative)
        if selected is None:
            return self.base.export_region(region, alternative, name, segments, target_source)
        module, group = selected
        return module.export_group(group, segments / name, name, target_source=target_source)
