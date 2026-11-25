"""Constant mask setup is bounded, reused, and absent for scalar RHS factors."""
import os
from unittest.mock import patch

import pytest

from atlas_mlir_native import exponent_restoration
from atlas_mlir_native.already_quantized import prepare_source
from atlas_mlir_native.binding import DRAM_BASE
from atlas_mlir_native.quantized_loop import execution_layout, virtual_source
from atlas_mlir_native.restoration_budget import instruction_bound
from test_exponent_page_cache import fixture
from quantized_efficiency_support import options
from test_exponent_restoration import compile_fixture
from test_quantized_prefix_reuse import memory_fixture


@pytest.mark.skipif(not (os.environ.get('ATLAS_RUST_NPU_MODEL') and os.environ.get('ATLAS_OOT_BIN_DIR')),
                    reason='select Rust simulator and Atlas tools')
@pytest.mark.parametrize('pipeline', ['mandatory', 'optimized'])
@pytest.mark.parametrize('shape,policy', [
    ((1,1,32,(),'scalar','scalar',True), 'ordered'),
    ((3,35,33,(),'scalar','row',True), 'ordered'),
    ((64,65,65,(),'row','row',True), 'batch-panel-cache'),
    ((3,35,65,(2,),'row','batch',False), 'ordered'),
])
def test_mask_setup_preserves_scale_domains_tails_and_all_memory(tmp_path, pipeline, shape, policy):
    from rust_model import execute

    original_entry = exponent_restoration._vector_entry
    def legacy_entry(**kwargs):
        return original_entry(**{**kwargs, 'row_rhs':True})

    with options():
        plan = prepare_source(fixture(*shape))
        layout = execution_layout(plan, operand_reuse=policy)
        with patch.object(exponent_restoration, '_MASK_CLEAR_WORDS_PER_ITERATION', 1), \
             patch.object(exponent_restoration, '_vector_entry', legacy_entry):
            baseline = virtual_source(plan, layout)
        selected = virtual_source(plan, layout)
        assert ('^restore_mask_header' in selected) == (shape[5] == 'row')
        programs = []
        for virtual in (baseline, selected):
            machine, program = compile_fixture(virtual, pipeline=pipeline)
            assert instruction_bound(virtual, machine, program)['admitted']
            programs.append(program)
        initial = memory_fixture(plan, layout)
        for rejected_role in (None,0,1):
            memory = bytearray(initial)
            if rejected_role is not None:
                address = (layout['exponent_addresses'][rejected_role]-DRAM_BASE
                           + layout['exponent_layouts'][rejected_role]['bytes']-2048)
                memory[address:address+4] = (1000).to_bytes(4,'little')
            results = [execute(program,memory,DRAM_BASE,tmp_path / f'{rejected_role}-{i}')
                       for i,program in enumerate(programs)]
            assert results[0][0] == results[1][0]
            actual, after = results[1]
            before = results[0][1]
            status = layout['status_address']-DRAM_BASE
            assert actual[status:status+4] == int(rejected_role is not None).to_bytes(4,'little')
            assert actual[:layout['output_addresses'][0]-DRAM_BASE] == memory[:layout['output_addresses'][0]-DRAM_BASE]
            assert actual[status+32:] == memory[status+32:]
            assert after['cycles'] < before['cycles']
            assert after['total_instructions'] < before['total_instructions']
            assert after['dram_read_bytes'] == before['dram_read_bytes']
            assert after['dram_write_bytes'] == before['dram_write_bytes']
