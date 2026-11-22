"""Row-cache fallback retains LHS scales across columns, but refreshes each row/batch."""
from dataclasses import replace
import json
from math import prod
import os

import pytest

from atlas_mlir_native.already_quantized import prepare_source
from atlas_mlir_native.binding import DRAM_BASE
from atlas_mlir_native.options import codegen_options, current_codegen_options
from atlas_mlir_native.quantized_loop import execution_layout, virtual_source
from atlas_mlir_native.restoration_budget import instruction_bound
from test_exponent_page_cache import fixture
from test_exponent_panel_cache import options
from test_exponent_restoration import compile_fixture
from test_quantized_operand_reuse import memory_fixture


def test_row_roles_are_explicit_and_replay_independently_of_current_options():
    with options():
        plan = prepare_source(fixture(3,65,257,(),'row','row',True))
        layout = execution_layout(plan, operand_reuse='interleaved-panel-cache')
        assert layout['exponent_row_cache_roles'] == ['lhs']
        selected = virtual_source(plan, layout)
        legacy = {key:value for key,value in layout.items() if key != 'exponent_row_cache_roles'}
        assert virtual_source(plan, legacy) != selected
        with codegen_options(replace(current_codegen_options(), exponent_page_cache=False)):
            assert virtual_source(plan, layout) == selected
        for roles in ([], ['rhs'], ['lhs','rhs'], True):
            with pytest.raises(ValueError, match='row cache roles'):
                virtual_source(plan, {**layout, 'exponent_row_cache_roles':roles})
        ordinary = execution_layout(plan, operand_reuse='batch-panel-cache')
        assert 'exponent_row_cache_roles' not in ordinary
        with pytest.raises(ValueError, match='row cache roles'):
            virtual_source(plan, {**ordinary, 'exponent_row_cache_roles':['lhs']})


@pytest.mark.skipif(not os.environ.get('ATLAS_OOT_BIN_DIR'), reason='select Atlas tools')
def test_row_cache_requires_dominating_initialization(monkeypatch):
    from atlas_mlir_native.exponent_restoration import Restoration

    monkeypatch.setattr(Restoration, 'row_entry', lambda self,state,addresses: ([],state))
    with options():
        plan = prepare_source(fixture(3,65,257,(),'row','row',True))
        layout = execution_layout(plan, operand_reuse='interleaved-panel-cache')
        with pytest.raises(AssertionError, match='initializ'):
            compile_fixture(virtual_source(plan, layout), pipeline='optimized')


@pytest.mark.skipif(not os.environ.get('ATLAS_OOT_BIN_DIR'), reason='select Atlas tools')
def test_row_cache_survives_native_rebinding_and_c_export(tmp_path):
    from unittest import TestCase
    from merlin.compiler.build.materialization import canonical_request
    from test_quantized_operand_reuse import export_recorded_policy

    source = fixture(3,35,65,(),'row','row',True)
    canonical = canonical_request(source, source_identity_policy='provenance')
    with options():
        export_recorded_policy(TestCase(), source, canonical, tmp_path, 'rows',
                               quantized_operand_reuse='interleaved-panel-cache', exponent_page_cache=True)
    execution = json.loads((tmp_path / 'bound/execution_plan.json').read_text())
    assert execution['exponent_row_cache_roles'] == ['lhs']


@pytest.mark.skipif(not (os.environ.get('ATLAS_RUST_NPU_MODEL') and os.environ.get('ATLAS_OOT_BIN_DIR')),
                    reason='select Rust simulator and Atlas tools')
@pytest.mark.parametrize('pipeline,m,n,k,batch,right,shared', [
    (pipeline,*shape) for pipeline in ('mandatory','optimized') for shape in (
        (3,65,257,(),'row',True),
        (65,35,65,(),'row',True),
        (3,35,65,(2,),'row',False),
        (3,35,65,(2,),'batch',False),
    )
] + [('mandatory',3,35,6145,(),'scalar',True)])
def test_rows_batches_tails_and_late_refusals_keep_all_bytes(
        tmp_path, pipeline, m, n, k, batch, right, shared):
    from rust_model import execute

    with options():
        plan = prepare_source(fixture(m,n,k,batch,'row',right,shared))
        layout = execution_layout(plan, operand_reuse='interleaved-panel-cache')
        assert layout['exponent_row_cache_roles'] == ['lhs']
        legacy = {key:value for key,value in layout.items() if key != 'exponent_row_cache_roles'}
        programs = []
        for selected in (legacy, layout):
            virtual = virtual_source(plan, selected)
            machine, program = compile_fixture(virtual, pipeline=pipeline)
            assert instruction_bound(virtual, machine, program)['admitted']
            programs.append(program)
        initial = memory_fixture(plan, layout)
        for rejected_role in (None,0,1):
            memory = bytearray(initial)
            if rejected_role is not None:
                address = (layout['exponent_addresses'][rejected_role] - DRAM_BASE
                           + layout['exponent_layouts'][rejected_role]['bytes'] - 2048)
                memory[address:address+4] = (1000).to_bytes(4,'little')
            results = [execute(program, memory, DRAM_BASE, tmp_path / f'{rejected_role}-{i}')
                       for i,program in enumerate(programs)]
            assert results[0][0] == results[1][0]
            actual, stats = results[1]
            output, status = layout['output_addresses'][0]-DRAM_BASE, layout['status_address']-DRAM_BASE
            end = status+layout['status_bytes']
            assert bool(int.from_bytes(actual[status:status+4],'little')) == (rejected_role is not None)
            assert actual[:output] == memory[:output] and actual[end:] == memory[end:]
            saved = prod(batch)*((m+31)//32)*(((n+31)//32)-1)*128
            difference = results[0][1]['dram_read_bytes'] - stats['dram_read_bytes']
            # A legacy scalar-refusal fallback can reload the same page again.
            assert difference == saved if rejected_role is None else difference >= saved
            assert results[0][1]['dram_write_bytes'] == stats['dram_write_bytes']
