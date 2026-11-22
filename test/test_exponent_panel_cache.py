"""Row pages have bounded panel lifetimes and unchanged refusal behavior."""
from dataclasses import replace
import os

import pytest

from atlas_mlir_native.already_quantized import prepare_source
from atlas_mlir_native.binding import DRAM_BASE
from atlas_mlir_native.options import codegen_options, current_codegen_options
from atlas_mlir_native.quantized_loop import execution_layout, virtual_source
from atlas_mlir_native.restoration_budget import instruction_bound
from test_exponent_page_cache import fixture
from test_exponent_restoration import compile_fixture
from test_quantized_operand_reuse import memory_fixture


def options():
    return codegen_options(replace(current_codegen_options(), io_memory_bytes=67108864,
                                   execution_instruction_limit=128000000, exponent_page_cache=True))


def test_panel_roles_are_recorded_checked_and_legacy_absence_replays():
    plan = prepare_source(fixture(128, 128, 257, (), 'row', 'row', True))
    with options():
        layout = execution_layout(plan, operand_reuse='interleaved-panel-cache')
        assert layout['exponent_panel_cache_roles'] == ['lhs', 'rhs']
        selected = virtual_source(plan, layout)
        legacy = {key: value for key, value in layout.items() if key != 'exponent_panel_cache_roles'}
        assert virtual_source(plan, legacy) != selected
        with codegen_options(replace(current_codegen_options(), exponent_page_cache=False)):
            assert virtual_source(plan, layout) == selected
        for roles in ([], ['lhs'], ['rhs', 'lhs'], ['lhs', 'rhs', 'unknown']):
            with pytest.raises(ValueError, match='panel cache roles'):
                virtual_source(plan, {**layout, 'exponent_panel_cache_roles': roles})
        for policy in ('ordered', 'batch-panel-cache'):
            ordinary = execution_layout(plan, operand_reuse=policy)
            assert 'exponent_panel_cache_roles' not in ordinary
            with pytest.raises(ValueError, match='panel cache roles'):
                virtual_source(plan, {**ordinary, 'exponent_panel_cache_roles': ['lhs', 'rhs']})


@pytest.mark.skipif(not (os.environ.get('ATLAS_RUST_NPU_MODEL') and os.environ.get('ATLAS_OOT_BIN_DIR')),
                    reason='select Rust simulator and Atlas tools')
@pytest.mark.parametrize('pipeline', ['mandatory', 'optimized'])
@pytest.mark.parametrize('m,n,k,batch,left,right,shared', [
    (97, 65, 257, (), 'row', 'row', True),
    (64, 96, 2049, (), 'row', 'scalar', True),
    (64, 35, 65, (2,), 'row', 'row', False),
    (64, 35, 65, (2,), 'row', 'row', True),
])
def test_panel_pages_match_outputs_rejections_and_reduce_reads(
        tmp_path, pipeline, m, n, k, batch, left, right, shared):
    from rust_model import execute

    with options():
        plan = prepare_source(fixture(m, n, k, batch, left, right, shared))
        layout = execution_layout(plan, operand_reuse='interleaved-panel-cache')
        assert layout['exponent_panel_cache_roles']
        legacy = {key: value for key, value in layout.items() if key != 'exponent_panel_cache_roles'}
        programs = []
        for selected in (legacy, layout):
            source = virtual_source(plan, selected)
            machine, program = compile_fixture(source, pipeline=pipeline)
            assert instruction_bound(source, machine, program)['admitted']
            assert len(program['instructions'])*4 <= 131072
            programs.append(program)
        initial = memory_fixture(plan, layout)
        for rejected_role in (None, 0, 1):
            memory = bytearray(initial)
            if rejected_role is not None:
                at = (layout['exponent_addresses'][rejected_role]-DRAM_BASE
                      + layout['exponent_layouts'][rejected_role]['bytes']-2048)
                memory[at:at+4] = (1000).to_bytes(4, 'little')
            results = [execute(program, memory, DRAM_BASE, tmp_path / f'{rejected_role}-{index}')
                       for index, program in enumerate(programs)]
            assert results[0][0] == results[1][0]
            actual, stats = results[1]
            output = layout['output_addresses'][0]-DRAM_BASE
            status = layout['status_address']-DRAM_BASE
            end = status+layout['status_bytes']
            assert bool(int.from_bytes(actual[status:status+4], 'little')) == (rejected_role is not None)
            assert actual[:output] == memory[:output] and actual[end:] == memory[end:]
            if 'dram_read_bytes' in stats:
                assert stats['dram_read_bytes'] < results[0][1]['dram_read_bytes']
                assert stats['dram_write_bytes'] == results[0][1]['dram_write_bytes']
