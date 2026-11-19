"""Selected execution budgets batch whole groups without weakening admission."""
import json
import os
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest
from atlas_mlir_native import activation_quantize_application as application
from atlas_mlir_native import activation_quantize_device as quantizer_device
from atlas_mlir_native import vector_quantize as device
from dataclasses import replace
from atlas_mlir_native.options import codegen_options, current_codegen_options
from merlin.semantic_compiler.target_binding import NativeCompilationError
from test_activation_quantize_device import dynamic_source


@pytest.fixture(autouse=True)
def device_kv_enabled():
    with codegen_options(replace(current_codegen_options(), device_kv_quantizer=True)):
        yield

ROOT = Path(__file__).resolve().parents[1]


def options(budget=128000000, memory=67108864, **kwargs):
    return codegen_options(replace(current_codegen_options(), execution_instruction_limit=budget,
                                         io_memory_bytes=memory, **kwargs))


@pytest.mark.parametrize('reduction', ['tensor', 'head', 'token', 'block', 'head_block'])
@pytest.mark.parametrize('rule', ['ref', 'nc'])
def test_every_dynamic_body_is_proved_once(reduction, rule):
    source = dynamic_source(shape=(2, 5, 32), reduce=reduction, rule=rule, rows=3)
    limits = SimpleNamespace(functional_instruction_limit=1000000)
    with patch.object(quantizer_device, 'prove_plan') as proof:
        schedule = application.prepare_plan(source, limits=limits)
    assert [call.args[0].source for call in proof.call_args_list] == list(schedule.bodies)
    assert all(call.kwargs == {'limits': limits, '_pipeline': schedule.pipeline}
               for call in proof.call_args_list)
    assert all(call.args[0].group_elements <= 4096 for call in proof.call_args_list)


def test_resource_refused_packs_retry_and_preserve_complete_groups():
    attempted = []

    def prove(plan, **kwargs):
        packed = plan.count // plan.group_elements
        attempted.append(packed)
        if packed > 2:
            raise NativeCompilationError('resource_limit', 'selected instruction bound')

    source = dynamic_source(shape=(7, 64, 64), reduce='head')
    with patch.object(quantizer_device, 'prove_plan', side_effect=prove):
        schedule = application.prepare_plan(source)
    assert attempted == [7, 6, 5, 4, 3, 2, 1]
    assert [launch['code_bytes'] for launch in schedule.launches] == [8192, 8192, 8192, 4096]
    assert [launch['code_offset_bytes'] for launch in schedule.launches] == [0, 8192, 16384, 24576]
    assert [launch['input_offset_bytes'] for launch in schedule.launches] == [0, 16384, 32768, 49152]
    assert [launch['exponent_offset_bytes'] for launch in schedule.launches] == [0, 8, 16, 24]
    assert [launch['exponent_bytes'] for launch in schedule.launches] == [8, 8, 8, 4]


@pytest.mark.parametrize('status', ['compile_error', 'unqualified_target', 'search_timeout', 'tool_unavailable'])
def test_non_resource_planning_errors_propagate_without_repacking(status):
    from atlas_mlir_native.application import AtlasApplicationAdapter

    source = dynamic_source(shape=(8, 64, 64), reduce='head')
    error = NativeCompilationError(status, 'proof failed')
    with patch.object(quantizer_device, 'prove_plan', side_effect=error) as proof:
        with pytest.raises(NativeCompilationError) as caught:
            AtlasApplicationAdapter._is_vector_quantizer(source, source.outputs)
    assert caught.value is error
    assert proof.call_count == 1


def test_only_singleton_resource_refusal_selects_host():
    from atlas_mlir_native.application import AtlasApplicationAdapter

    source = dynamic_source(shape=(2, 1, 32), reduce='head')
    with patch.object(quantizer_device, 'prove_plan', side_effect=NativeCompilationError('resource_limit', 'bound')) as proof:
        assert not AtlasApplicationAdapter._is_vector_quantizer(source, source.outputs)
    assert [call.args[0].count for call in proof.call_args_list] == [64, 32]


@pytest.mark.parametrize('emitted,status', [('not JSON', 'compile_error'), ('[]', 'compile_error'),
                                          ('{"selected_rtl_revision":"different"}', 'unqualified_target')])
def test_emitter_failures_do_not_select_host(emitted, status):
    from atlas_mlir_native.application import AtlasApplicationAdapter

    source = dynamic_source(shape=(2, 1, 32), reduce='head')
    with patch('atlas_mlir_native.binding._run', side_effect=['machine', '', emitted]):
        with pytest.raises(NativeCompilationError) as caught:
            AtlasApplicationAdapter._is_vector_quantizer(source, source.outputs)
    assert caught.value.status == status


@pytest.mark.parametrize('error', [ValueError('invalid bound'), KeyError('missing'), IndexError('missing')])
def test_bound_analyzer_failures_do_not_select_host(error):
    from atlas_mlir_native.application import AtlasApplicationAdapter
    from atlas_mlir_native.binding import RTL_REVISION

    source = dynamic_source(shape=(2, 1, 32), reduce='head')
    with patch('atlas_mlir_native.binding._run', side_effect=['machine', '', json.dumps({'selected_rtl_revision': RTL_REVISION})]), \
         patch('atlas_mlir_native.restoration_budget.instruction_bound', side_effect=error):
        with pytest.raises(NativeCompilationError) as caught:
            AtlasApplicationAdapter._is_vector_quantizer(source, source.outputs)
    assert caught.value.status == 'compile_error'
    assert caught.value.__cause__ is error


@pytest.mark.skipif(not os.environ.get('ATLAS_OOT_BIN_DIR'), reason='select Atlas tools')
@pytest.mark.parametrize('pipeline,rows,packed', [
    ('mandatory',64,8), ('mandatory',128,4), ('mandatory',256,2),
    ('optimized',64,8), ('optimized',128,4), ('optimized',256,2)])
def test_default_budget_reduces_head_packs_before_host(pipeline, rows, packed):
    from atlas_mlir_native.application import AtlasApplicationAdapter

    source = dynamic_source(shape=(32, rows, 64), reduce='head')
    with options(1000000), patch.dict(os.environ, ATLAS_NATIVE_PIPELINE=pipeline):
        schedule = application.prepare_plan(source)
        assert AtlasApplicationAdapter._is_vector_quantizer(source, source.outputs)
        assert schedule.bodies[0].nodes[0].type.shape == (packed, rows, 64)
        assert len(schedule.launches) == (32 + packed - 1) // packed
        assert sum(launch['code_bytes'] for launch in schedule.launches) == 32 * rows * 64
        assert sum(launch['exponent_bytes'] for launch in schedule.launches) == 32 * 4
        for launch in schedule.launches:
            assert launch['code_offset_bytes'] % (rows * 64) == 0
            assert launch['code_bytes'] % (rows * 64) == 0
            plan = device.prepare_plan(schedule.bodies[launch['body_index']])
            assert quantizer_device.prove_plan(plan)[3]['admitted']


@pytest.mark.skipif(not os.environ.get('ATLAS_OOT_BIN_DIR'), reason='select Atlas tools')
@pytest.mark.parametrize('heads,length', [(32, 4096), (13, 1024), (10, 96), (9, 32768)])
def test_selected_budget_batches_whole_heads(heads, length):
    source = dynamic_source(shape=(heads, 1, length), reduce='head')
    with options():
        plan = application.prepare_plan(source)
        assert len(plan.launches) == 1
        assert plan.launches[0]['code_bytes'] == heads * length
        assert plan.launches[0]['exponent_bytes'] == heads * 4
        assert plan.record()['functional_instruction_limit'] == 128000000
        limited = application.prepare_plan(source, limits=SimpleNamespace(functional_instruction_limit=1000000))
        assert limited.record()['functional_instruction_limit'] == 1000000
        assert len(limited.launches) > 1
    with options(1000000):
        assert application.prepare_plan(source).record() == limited.record()


@patch.object(quantizer_device, 'prove_plan')
def test_exponent_slot_io_and_indivisible_group_limits_remain_active(_proof):
    with options():
        plan = application.prepare_plan(dynamic_source(shape=(257, 1, 32), reduce='head'))
        assert [x['exponent_bytes'] for x in plan.launches] == [1024, 4]
        for length in (1056, 2016, 4064):
            assert len(application.prepare_plan(dynamic_source(shape=(10, 1, length), reduce='head')).launches) == 10
        with pytest.raises(ValueError, match='complete 32-element tiles'):
            application.prepare_plan(dynamic_source(shape=(2, 1, 4129), reduce='head'))
    with options(memory=8192):
        assert len(application.prepare_plan(dynamic_source(shape=(10, 1, 1024), reduce='head')).launches) == 5
        with pytest.raises(ValueError, match='I/O window'):
            application.prepare_plan(dynamic_source(shape=(1, 1, 4096), reduce='head'))
    with options(1000000), pytest.raises(ValueError, match='indivisible dynamic reduction group'):
        application.prepare_plan(dynamic_source(shape=(1, 1, 33792), reduce='head'))


@pytest.mark.skipif(not os.environ.get('ATLAS_OOT_BIN_DIR'), reason='select Atlas tools')
def test_replayed_budget_and_resource_rejections(tmp_path):
    source = dynamic_source(shape=(12, 1, 1024), reduce='head')
    with options():
        application.compile_source(None, source, target_source=ROOT, destination=tmp_path / 'compiled',
                                   limits=SimpleNamespace(functional_instruction_limit=1000000))
        exported = application.prepare_c_package(source, tmp_path / 'compiled', 'quant', target_source=ROOT)
        assert exported.description['launches_per_invocation'] == 2
        schedule = exported.description['quantizer_schedule']
        assert schedule['functional_instruction_limit'] == 1000000
        application.compile_source(None, source, target_source=ROOT, destination=tmp_path / 'large', limits=None)
        assert application.prepare_c_package(source, tmp_path / 'large', 'large', target_source=ROOT).description['launches_per_invocation'] == 1
    with options(1000000), pytest.raises(ValueError, match='differs from checked source'):
        application.prepare_c_package(source, tmp_path / 'large', 'quant', target_source=ROOT)
    with options(1), pytest.raises(NativeCompilationError, match='instruction bound'):
        device.compile_source(None, source, target_source=ROOT, destination=tmp_path / 'too_many', limits=None)
    with options(program_size_limit_bytes=4), pytest.raises(NativeCompilationError, match='policy limit'):
        device.compile_source(None, source, target_source=ROOT, destination=tmp_path / 'image', limits=None)
