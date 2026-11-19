"""Batched panels preserve independent matrix products and scale ownership."""
from dataclasses import replace
import json
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from atlas_mlir_native.options import codegen_options, current_codegen_options
from atlas_mlir_native.already_quantized import prepare_source
from atlas_mlir_native.quantized_loop import execution_layout, virtual_source
from atlas_mlir_native.quantized_loop_sizing import size_counted

from merlin.semantic_compiler.target_binding import NativeCompilationError
from test_exponent_page_cache import fixture, OPTIONS


def test_geometry_replay_and_only_proved_capacity_retry():
    plan = prepare_source(fixture(128, 128, 64, (2, 3), 'batch', 'batch', False))
    with codegen_options(replace(current_codegen_options(), io_memory_bytes=OPTIONS.io_memory_bytes, execution_instruction_limit=OPTIONS.execution_instruction_limit)):
        old = execution_layout(plan, operand_reuse='panel-cache', exponent_page_cache=True)
        layout = execution_layout(plan, operand_reuse='batch-panel-cache', exponent_page_cache=True)
        assert old['operand_reuse'] == 'row-cache'
        assert layout['operand_reuse'] == 'panel-cache'
        assert layout['operand_reuse_panel_m_tiles'] == 4
        expected = virtual_source(plan, layout)
        with codegen_options(replace(current_codegen_options(), quantized_operand_reuse='ordered', exponent_page_cache=False)):
            assert virtual_source(plan, layout) == expected
        for key in ('operand_reuse_panel_m_tiles', 'operand_reuse_panel_k_tiles'):
            with pytest.raises(ValueError, match='schedule differs'):
                virtual_source(plan, {**layout, key: 1})
    refused = SimpleNamespace(record=lambda: dict(route='k32_tiles', reason='encoded_instruction_memory'))
    accepted = SimpleNamespace(record=lambda: dict(route='counted', reason='physical_bound_admitted'))
    with patch('atlas_mlir_native.quantized_loop_sizing._size_counted',
               side_effect=[refused, refused, accepted]) as size:
        assert size_counted(plan, operand_reuse='batch-panel-cache') is accepted
        assert [call.kwargs['operand_reuse'] for call in size.call_args_list] == [
            'batch-panel-cache', 'panel-cache', 'paired-cache']
    with patch('atlas_mlir_native.quantized_loop_sizing._size_counted',
               side_effect=NativeCompilationError('compile_error', 'native proof failure')) as size:
        with pytest.raises(NativeCompilationError, match='native proof failure'):
            size_counted(plan, operand_reuse='batch-panel-cache')
        assert size.call_count == 1


@pytest.mark.parametrize('m,n,k,left,right', [
    (225, 35, 3073, 'batch', 'batch'),
    (64, 35, 2049, 'batch', 'batch'), (65, 35, 65, 'batch', 'batch'),
])
def test_ineligible_batch_geometry_retains_original_program(m, n, k, left, right):
    with codegen_options(replace(current_codegen_options(), io_memory_bytes=OPTIONS.io_memory_bytes, execution_instruction_limit=OPTIONS.execution_instruction_limit)):
        plan = prepare_source(fixture(m, n, k, (2,), left, right, False))
        old = execution_layout(plan, operand_reuse='panel-cache', exponent_page_cache=True)
        new = execution_layout(plan, operand_reuse='batch-panel-cache', exponent_page_cache=True)
        assert new['operand_reuse'] != 'panel-cache'
        assert virtual_source(plan, old) == virtual_source(plan, new)


def test_native_binding_replays_batched_panel_selection(tmp_path):
    from unittest import TestCase
    from merlin.semantic_compiler.materialization import canonical_request
    from test_quantized_operand_reuse import export_recorded_policy

    source = fixture(64, 35, 33, (2,), 'batch', 'batch', False)
    canonical = canonical_request(source, source_identity_policy='provenance')
    with codegen_options(replace(current_codegen_options(), io_memory_bytes=OPTIONS.io_memory_bytes, execution_instruction_limit=OPTIONS.execution_instruction_limit)):
        export_recorded_policy(TestCase(), source, canonical, tmp_path, 'batch',
                               quantized_operand_reuse='batch-panel-cache', exponent_page_cache=True)
        layout = json.loads((tmp_path / 'bound/execution_plan.json').read_text())
        assert layout['operand_reuse_requested'] == 'batch-panel-cache'
