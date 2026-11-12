"""Normal targetgen application factories expose explicit target-owned groups."""
import os
from unittest.mock import patch

import pytest
from atlas_mlir_native.application import AtlasApplicationAdapter
from atlas_mlir_native.binding import AtlasNativeBinding
from atlas_mlir_native.composition_adapter import AdjacentDeviceAdapter
from merlin.semantic_compiler.codegen import CodegenOptions, codegen_options
from test_application_float_arithmetic import policy
from test_expanded_silu import case


def selected_policy():
    from atlas_mlir_native.already_quantized import POLICY as INTAKE
    from atlas_mlir_native.source_exp import POLICY

    return {**policy(), 'already_quantized': INTAKE, 'source_exp': POLICY,
            'host_float_arithmetic': ['f32_neg']}


def factory():
    selected = selected_policy()
    return AtlasNativeBinding.application_adapter(selected, original_source_sha256=selected['source_sha256'])


def test_default_factory_retains_original_regions():
    with patch.dict(os.environ, ATLAS_ADJACENT_PATTERNS=''):
        base = factory()
    assert type(base) is AtlasApplicationAdapter
    with codegen_options(CodegenOptions(io_memory_bytes=67108864)):
        assert len(base.plan(case()).selected) == 7


def test_factory_uses_the_same_proved_groups_and_preserves_base_capabilities():
    with patch.dict(os.environ, ATLAS_ADJACENT_PATTERNS='pointwise_narrowing, expanded_silu'):
        wrapper = factory()
    assert isinstance(wrapper, AdjacentDeviceAdapter)
    assert wrapper.source_operation_translators is wrapper.base.source_operation_translators
    with codegen_options(CodegenOptions(io_memory_bytes=67108864)):
        selected, = wrapper.plan(case()).selected
        assert selected.nodes == tuple(range(7))
        assert dict(selected.metadata)['kind'] == 'expanded_silu'
        assert not wrapper.supports_region_materialization(case(), selected)
        manual = AdjacentDeviceAdapter(wrapper.base, patterns=('pointwise_narrowing', 'expanded_silu'))
        assert manual.plan(case()).selected == wrapper.plan(case()).selected


@pytest.mark.parametrize('selection', ['unknown', 'expanded_silu,expanded_silu', ',expanded_silu', 'expanded_silu,'])
def test_factory_rejects_unknown_duplicate_or_missing_pattern_names(selection):
    with patch.dict(os.environ, ATLAS_ADJACENT_PATTERNS=selection), pytest.raises(ValueError):
        factory()
