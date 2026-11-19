"""Large static quantizers retain bounded counted bodies and exact bytes."""
from pathlib import Path

import pytest

from atlas_mlir_native import activation_quantize_application as application
from test_already_quantized import quantizer_source
from test_quantizer_dispatch_budget import options

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(autouse=True)
def static_quantizer_sizing(select_codegen_options):
    select_codegen_options(static_quantizer_sizing=True)


@pytest.mark.parametrize('pipeline', ['mandatory', 'optimized'])
def test_static_partitioning_is_independent_of_native_pipeline(pipeline):
    source = quantizer_source(count=524320)
    with options(1000000, native_pipeline=pipeline, static_quantizer_sizing=False):
        legacy = application.prepare_plan(source)
        assert [row['code_bytes'] for row in legacy.launches] == [2048] * 256 + [32]
    with options(1000000, native_pipeline=pipeline, static_quantizer_sizing=True):
        enhanced = application.prepare_plan(source)
        assert [row['code_bytes'] for row in enhanced.launches] == [524320]


@pytest.mark.parametrize('count', [1056, 524256, 786432, 12582912])
def test_large_aligned_tensor_uses_selected_budget(count):
    source = quantizer_source(count=count)
    with options():
        plan = application.prepare_plan(source)
        assert len(plan.launches) == 1
        assert plan.launches[0]['code_bytes'] == count
        assert plan.record()['functional_instruction_limit'] == 128000000
    with options(1000000):
        limited = application.prepare_plan(source)
        assert sum(row['code_bytes'] for row in limited.launches) == count
        assert all(50 * (row['code_bytes'] // 32) + 128 <= 1000000
                   for row in limited.launches)
        if count > 640000:
            assert len(limited.launches) > 1


def test_memory_capacity_and_instruction_capacity_keep_aligned_remainders():
    for budget, memory in ((1000000, 8192), (10000, 67108864), (1000, 67108864)):
        with options(budget, memory):
            plan = application.prepare_plan(quantizer_source(count=65568))
            at = 0
            for launch in plan.launches:
                count = launch['code_bytes']
                assert launch['code_offset_bytes'] == at
                assert launch['input_offset_bytes'] == at * 2
                assert launch['exponent_offset_bytes'] == 0
                assert count % 32 == 0
                assert 3 * count + 32 <= memory
                assert 50 * (count // 32) + 128 <= budget
                at += count
            assert at == 65568
    with options(177), pytest.raises(ValueError, match='cannot fit a static quantizer tile'):
        application.prepare_plan(quantizer_source(count=32))
