"""Shape-selected panels preserve ordered results, bounds and replay identity."""
from atlas_mlir_native.options import codegen_options, current_codegen_options
from dataclasses import replace
import json
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from atlas_mlir_native.already_quantized import prepare_source
from atlas_mlir_native.quantized_loop import execution_layout, virtual_source
from atlas_mlir_native.quantized_loop_sizing import size_counted
from atlas_mlir_native.quantized_panel_geometry import panel_geometry

from merlin.semantic_compiler.target_binding import NativeCompilationError
from test_already_quantized import source_request


def plan_for(m, n, k, *, batch=()):
    return prepare_source(source_request(m=m, n=n, k=k, batch=batch))


def test_geometry_accounts_for_partial_traffic_and_keeps_frame_reserve():
    assert panel_geometry(plan_for(1024, 3072, 768)) == (4, 38)
    assert panel_geometry(plan_for(1024, 768, 3072)) == (8, 21)
    for m in (33, 97, 225, 1024):
        for k in (257, 768, 1025, 2048, 3073, 5632):
            geometry = panel_geometry(plan_for(m, 35, k))
            if geometry is not None:
                rows, columns = geometry
                assert (rows + 1) * columns * 1024 <= 192 * 1024
                assert ((m + 31) // 32) % rows == 0
    for m, n, k, batch in ((32, 65, 1024, ()), (65, 65, 1024, ()),
                            (128, 32, 1024, ()), (128, 65, 64, ()),
                            (128, 65, 6145, ()), (63, 65, 1024, (2,))):
        assert panel_geometry(plan_for(m, n, k, batch=batch)) is None


def test_recorded_geometry_is_independent_of_environment_and_not_mutable():
    with codegen_options(replace(current_codegen_options(), io_memory_bytes=67108864)):
        plan = plan_for(97, 35, 768)
        layout = execution_layout(plan, operand_reuse='panel-cache')
        expected = virtual_source(plan, layout)
        with codegen_options(replace(current_codegen_options(), quantized_operand_reuse='ordered')):
            assert virtual_source(plan, layout) == expected
        for key in ('operand_reuse_panel_m_tiles', 'operand_reuse_panel_k_tiles', 'operand_reuse_cached_tiles'):
            with pytest.raises(ValueError, match='schedule differs'):
                virtual_source(plan, {**layout, key: 1})


def test_retry_requires_proved_capacity_and_preserves_actual_policy():
    plan = plan_for(225, 65, 5632)
    accepted = SimpleNamespace(record=lambda: {'route': 'counted', 'reason': 'physical_bound_admitted',
                                               'operand_reuse_requested': 'paired-cache'})
    for reason in ('encoded_instruction_memory', 'untimed_instruction_bound', 'physical_instruction_bound'):
        refused = SimpleNamespace(record=lambda: {'route': 'k32_tiles', 'reason': reason})
        with patch('atlas_mlir_native.quantized_loop_sizing._size_counted', side_effect=[refused, accepted]) as size:
            assert size_counted(plan, operand_reuse='panel-cache') is accepted
            assert [call.kwargs['operand_reuse'] for call in size.call_args_list] == ['panel-cache', 'paired-cache']
    refused = SimpleNamespace(record=lambda: {'route': 'k32_tiles', 'reason': 'selected_dram_window'})
    with patch('atlas_mlir_native.quantized_loop_sizing._size_counted', return_value=refused) as size:
        assert size_counted(plan, operand_reuse='panel-cache') is refused
        assert size.call_count == 1
    with patch('atlas_mlir_native.quantized_loop_sizing._size_counted',
               side_effect=NativeCompilationError('compile_error', 'test proof failure')) as size:
        with pytest.raises(NativeCompilationError, match='test proof failure'):
            size_counted(plan, operand_reuse='panel-cache')
        assert size.call_count == 1


def test_selected_panels_survive_native_rebinding_and_c_export(tmp_path):
    from unittest import TestCase
    from merlin.semantic_compiler.materialization import canonical_request
    from test_quantized_operand_reuse import export_recorded_policy

    source = source_request(m=97, n=35, k=257, batch=())
    canonical = canonical_request(source, source_identity_policy='provenance')
    with codegen_options(replace(current_codegen_options(), io_memory_bytes=67108864, execution_instruction_limit=128000000)):
        export_recorded_policy(TestCase(), source, canonical, tmp_path, 'panels', quantized_operand_reuse='panel-cache')
        execution = json.loads((tmp_path/'bound/execution_plan.json').read_text())
        assert execution['operand_reuse'] == 'panel-cache'
        assert execution['operand_reuse_panel_m_tiles'] == 4


def test_real_oversized_panel_uses_checked_paired_candidate():
    from atlas_mlir_native.quantized_loop_sizing import _size_counted
    from test_exponent_restoration import ROOT

    with codegen_options(replace(current_codegen_options(), io_memory_bytes=67108864, execution_instruction_limit=128000000, program_size_limit_bytes=100000)):
        plan = plan_for(225, 65, 5632)
        refused = _size_counted(plan, target_source=ROOT, operand_reuse='panel-cache')
        assert refused.record()['route'] == 'k32_tiles'
        assert refused.record()['reason'] == 'encoded_instruction_memory'
        selected = size_counted(plan, target_source=ROOT, operand_reuse='panel-cache')
        assert selected.record()['route'] == 'counted'
        assert selected.record()['operand_reuse_requested'] == 'paired-cache'
