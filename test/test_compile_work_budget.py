"""Tool guards abort; counted refusals are classified separately from bugs."""
import json
import subprocess
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import pytest
from atlas_mlir_native.binding import _run
from atlas_mlir_native import work_budget
from merlin.compiler.semantic.target_binding import NativeCompilationError


def test_counted_exhaustion_keeps_resource_limit_status():
    error = work_budget.CountedWorkExhausted('counted work exhausted')
    assert isinstance(error, NativeCompilationError)
    assert error.status == 'resource_limit'
    assert error.reason == 'counted work exhausted'


def test_hang_detector_is_internal_error():
    with patch('subprocess.run', side_effect=subprocess.TimeoutExpired(['atlas-opt'], 600)) as run:
        with pytest.raises(NativeCompilationError) as caught:
            _run(['atlas-opt', '-'], input_text='module {}')
    assert caught.value.status == 'compile_error'
    assert 'internal error' in caught.value.reason
    assert run.call_args.kwargs['timeout'] == work_budget.TOOL_HANG_GUARD_SECONDS
    assert run.call_args.args[0] == ['atlas-opt', '-']


@pytest.mark.parametrize('stderr,status', [
    ('Atlas work budget exhausted: register-allocation', 'resource_limit'),
    ('Atlas process work budget exhausted: virtual-scheduling', 'resource_limit'),
    ('spill rewrite work limit reached (budget=spill-atlas-virtual.max-spills)', 'resource_limit'),
    ('invalid operation', 'compile_error'),
])
def test_refusal_classification(stderr, status):
    with patch('subprocess.run', return_value=subprocess.CompletedProcess([], 1, '', stderr)):
        with pytest.raises(NativeCompilationError) as caught:
            _run(['atlas-opt', '-'], input_text='module {}')
    assert caught.value.status == status
    assert isinstance(caught.value, work_budget.CountedWorkExhausted) == (status == 'resource_limit')
    assert stderr in caught.value.reason


def test_reduction_buffer_solver_exhaustion_is_named(monkeypatch):
    from atlas_mlir_native.reduction import plan_abi, request_for
    monkeypatch.setattr(work_budget, 'BUFFER_SOLVER_RLIMIT', 1)
    with pytest.raises(work_budget.CountedWorkExhausted) as caught:
        plan_abi(request_for((32, 32), (1,)))
    assert caught.value.status == 'resource_limit'
    assert 'buffer_solver_rlimit_per_query=1' in caught.value.reason


def test_counted_exhaustion_cannot_become_quantized_capacity_fallback(tmp_path, monkeypatch):
    from atlas_mlir_native import quantized_loop_sizing as sizing
    from merlin.compiler.semantic.codegen import codegen_options, current_codegen_options
    from test_quantized_panel_geometry import plan_for

    tools = tmp_path / 'tools'
    tools.mkdir()
    for name in ('atlas-opt', 'atlas-emit'):
        (tools / name).write_bytes(b'tool identity')
    monkeypatch.setenv('ATLAS_OOT_BIN_DIR', str(tools))
    monkeypatch.setattr(sizing, '_CACHE', {})
    exhausted = work_budget.CountedWorkExhausted('Atlas work budget exhausted: x')

    def reject(command, **kwargs):
        options = next(arg for arg in command if arg.startswith('--lower-atlas-virtual-to-machine='))
        report = Path(options.split('resource-planning-report=', 1)[1].split(' ', 1)[0])
        report.write_text(json.dumps(dict(schema='atlas.virtual_resource_plan.v1',
            planning_succeeded=False, machine_artifact_qualified=False,
            untimed_words=dict(words=32769, limit=32768))))
        raise exhausted

    with codegen_options(replace(current_codegen_options(), io_memory_bytes=67108864)):
        with patch.object(sizing, '_run', side_effect=reject) as run:
            with pytest.raises(work_budget.CountedWorkExhausted) as caught:
                sizing.size_counted(plan_for(225, 65, 5632),
                    target_source=Path(__file__).resolve().parents[1], operand_reuse='panel-cache')
    assert caught.value is exhausted
    assert run.call_count == 1


def test_retained_sizing_failure_keeps_counted_exhaustion(tmp_path, monkeypatch):
    from atlas_mlir_native import quantized_loop_sizing as sizing
    from merlin.compiler.semantic.codegen import codegen_options, current_codegen_options
    from test_quantized_panel_geometry import plan_for

    tools = tmp_path / 'tools'
    tools.mkdir()
    for name in ('atlas-opt', 'atlas-emit'):
        (tools / name).write_bytes(b'tool identity')
    monkeypatch.setenv('ATLAS_OOT_BIN_DIR', str(tools))
    monkeypatch.setattr(sizing, '_CACHE', {})

    def prepare(command, **kwargs):
        options = next(arg for arg in command if arg.startswith('--lower-atlas-virtual-to-machine='))
        report = Path(options.split('resource-planning-report=', 1)[1].split(' ', 1)[0])
        report.write_text('{}')
        return 'module {}'

    with codegen_options(replace(current_codegen_options(), io_memory_bytes=67108864)):
        with patch.object(sizing, '_run', side_effect=prepare), \
             patch.object(sizing, 'untimed_bound', return_value=dict(
                 instruction_bound=1, exceeds_physical_instruction_limit=False)), \
             patch('atlas_mlir_native.region_composition.compile_virtual_with_receipt',
                   side_effect=work_budget.CountedWorkExhausted('Atlas work budget exhausted: x')) as checked:
            plan = plan_for(225, 65, 5632)
            for _ in range(2):
                with pytest.raises(work_budget.CountedWorkExhausted):
                    sizing.size_counted(plan, target_source=Path(__file__).resolve().parents[1],
                                        operand_reuse='ordered')
    assert checked.call_count == 1
