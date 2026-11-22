"""Share weight pushes while preserving ordered BF16 partials and publication."""
from dataclasses import replace
import json
import os
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from atlas_mlir_native.already_quantized import prepare_source
from atlas_mlir_native.binding import DRAM_BASE
from atlas_mlir_native.options import codegen_options, current_codegen_options
from atlas_mlir_native.quantized_loop import execution_layout, virtual_source
from atlas_mlir_native.quantized_loop_sizing import size_counted
from atlas_mlir_native.restoration_budget import instruction_bound
from merlin.semantic_compiler.target_binding import NativeCompilationError
from test_already_quantized import source_request
from test_exponent_page_cache import fixture
from test_exponent_restoration import compile_fixture
from test_quantized_operand_reuse import memory_fixture


def options():
    return codegen_options(replace(current_codegen_options(), io_memory_bytes=67108864,
                                   execution_instruction_limit=128000000,
                                   exponent_page_cache=True))


def test_interleaving_is_explicit_authenticated_and_shares_weight_pushes():
    plan = prepare_source(source_request(m=128, n=64, k=256, batch=()))
    with options():
        before = execution_layout(plan, operand_reuse='batch-panel-cache')
        after = execution_layout(plan, operand_reuse='interleaved-panel-cache')
        assert 'operand_reuse_panel_order' not in before
        assert after['operand_reuse_panel_order'] == 'k-major'
        old, new = virtual_source(plan, before), virtual_source(plan, after)
        assert new.count('"atlas.virtual_mxu_load_weight"') * 4 == old.count('"atlas.virtual_mxu_load_weight"')
        assert '%panel_w_' not in new
        # Saved source replay does not depend on the currently selected recipe.
        with codegen_options(replace(current_codegen_options(), quantized_operand_reuse='ordered')):
            assert virtual_source(plan, after) == new
        for forged in ({**after, 'operand_reuse_panel_order': 'row-major'},
                       {key: value for key, value in after.items() if key != 'operand_reuse_panel_order'},
                       {**before, 'operand_reuse_panel_order': 'k-major'}):
            with pytest.raises(ValueError, match='schedule differs'):
                virtual_source(plan, forged)


@pytest.mark.parametrize('m,n,k', [(32, 64, 256), (65, 35, 65), (256, 64, 5120)])
def test_unselected_geometries_keep_the_original_source(m, n, k):
    plan = prepare_source(source_request(m=m, n=n, k=k, batch=()))
    with options():
        before = execution_layout(plan, operand_reuse='batch-panel-cache')
        after = execution_layout(plan, operand_reuse='interleaved-panel-cache')
        assert 'operand_reuse_panel_order' not in after
        assert virtual_source(plan, before) == virtual_source(plan, after)


def test_only_proved_capacity_refusals_retry_the_original_panel_order():
    plan = prepare_source(source_request(m=64, n=64, k=256, batch=()))
    accepted = SimpleNamespace(record=lambda: dict(route='counted', reason='physical_bound_admitted'))
    for reason in ('encoded_instruction_memory', 'untimed_instruction_bound', 'physical_instruction_bound'):
        refused = SimpleNamespace(record=lambda: dict(route='k32_tiles', reason=reason))
        with patch('atlas_mlir_native.quantized_loop_sizing._size_counted',
                   side_effect=[refused, accepted]) as size:
            assert size_counted(plan, operand_reuse='interleaved-panel-cache') is accepted
            assert [call.kwargs['operand_reuse'] for call in size.call_args_list] == [
                'interleaved-panel-cache', 'batch-panel-cache']
    with patch('atlas_mlir_native.quantized_loop_sizing._size_counted',
               side_effect=NativeCompilationError('compile_error', 'failed proof')) as size:
        with pytest.raises(NativeCompilationError, match='failed proof'):
            size_counted(plan, operand_reuse='interleaved-panel-cache')
        assert size.call_count == 1


@pytest.mark.skipif(not os.environ.get('ATLAS_OOT_BIN_DIR'), reason='select Atlas tools')
def test_native_rebinding_and_c_export_preserve_the_selected_order(tmp_path):
    from unittest import TestCase
    from merlin.semantic_compiler.materialization import canonical_request
    from test_quantized_operand_reuse import export_recorded_policy

    source = source_request(m=33, n=35, k=65, batch=())
    canonical = canonical_request(source, source_identity_policy='provenance')
    with options():
        export_recorded_policy(TestCase(), source, canonical, tmp_path, 'interleaved',
                               quantized_operand_reuse='interleaved-panel-cache', exponent_page_cache=True)
        execution = json.loads((tmp_path / 'bound/execution_plan.json').read_text())
        assert execution['operand_reuse_requested'] == 'interleaved-panel-cache'
        assert execution['operand_reuse_panel_order'] == 'k-major'


@pytest.mark.skipif(not (os.environ.get('ATLAS_RUST_NPU_MODEL') and os.environ.get('ATLAS_OOT_BIN_DIR')),
                    reason='select Rust simulator and Atlas tools')
@pytest.mark.parametrize('pipeline', ['mandatory', 'optimized'])
@pytest.mark.parametrize('m,n,k,batch', [
    (33, 35, 65, ()), (97, 65, 257, ()), (64, 96, 2049, ()), (64, 35, 65, (2,)),
])
def test_rust_outputs_guards_and_late_rejection_match_original_panels(tmp_path, pipeline, m, n, k, batch):
    from rust_model import execute

    request = (fixture(m, n, k, batch, 'batch', 'batch', False) if batch else
               source_request(m=m, n=n, k=k, batch=()))
    plan = prepare_source(request)
    with options():
        layouts = [execution_layout(plan, operand_reuse=policy)
                   for policy in ('batch-panel-cache', 'interleaved-panel-cache')]
        assert layouts[1]['operand_reuse_panel_order'] == 'k-major'
        programs = []
        for layout in layouts:
            source = virtual_source(plan, layout)
            machine, program = compile_fixture(source, pipeline=pipeline)
            assert instruction_bound(source, machine, program)['admitted']
            assert len(program['instructions']) * 4 <= 131072
            programs.append(program)
        layout = layouts[0]
        initial = memory_fixture(plan, layout)
        for refused in (False, True):
            if refused:
                geometry = layout['exponent_layouts'][1]
                # Reject a later scale page, after valid output tiles have run.
                at = layout['exponent_addresses'][1] - DRAM_BASE + geometry['bytes'] - 2048
                initial[at:at+4] = (1000).to_bytes(4, 'little')
            outputs, reports = [], []
            for index, program in enumerate(programs):
                actual, stats = execute(program, initial, DRAM_BASE, tmp_path / f'{refused}-{index}')
                first = layout['output_addresses'][0] - DRAM_BASE
                status = layout['status_address'] - DRAM_BASE
                end = status + layout['status_bytes']
                assert actual[:first] == initial[:first]
                assert actual[end:] == initial[end:]
                assert bool(int.from_bytes(actual[status:status+4], 'little')) == refused
                outputs.append(actual)
                reports.append(stats)
            assert outputs[0] == outputs[1]
            # The optional measurement build exposes actual issued beat counts.
            for counter in ('dram_read_bytes', 'dram_write_bytes'):
                if counter in reports[0]:
                    assert reports[0][counter] == reports[1][counter]
