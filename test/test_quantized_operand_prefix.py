"""Single M/N tiles load valid rows and synthesize their private zero padding."""
import os
from unittest.mock import patch

import pytest

from atlas_mlir_native import quantized_loop
from atlas_mlir_native.already_quantized import prepare_source
from atlas_mlir_native.binding import DRAM_BASE
from atlas_mlir_native.restoration_budget import instruction_bound
from test_exponent_page_cache import fixture
from quantized_efficiency_support import options
from test_exponent_restoration import compile_fixture
from test_quantized_prefix_reuse import memory_fixture


@pytest.mark.skipif(not (os.environ.get('ATLAS_RUST_NPU_MODEL') and os.environ.get('ATLAS_OOT_BIN_DIR')),
                    reason='select Rust simulator and Atlas tools')
@pytest.mark.parametrize('pipeline', ['mandatory', 'optimized'])
@pytest.mark.parametrize('shape,policy', [
    ((1,65,257,(),'row','row',True), 'ordered'),
    ((1,65,2048,(),'row','row',True), 'batch-panel-cache'),
    ((3,5,65,(),'row','row',True), 'ordered'),
    ((65,3,33,(),'row','row',True), 'batch-panel-cache'),
    ((3,35,65,(2,),'row','batch',False), 'batch-panel-cache'),
    ((3,35,65,(2,),'scalar','batch',False), 'ordered'),
    ((35,3,65,(2,),'batch','row',False), 'ordered'),
    ((3,5,97,(2,),'row','row',True), 'ordered'),
])
def test_prefix_loads_preserve_outputs_rejections_batches_and_guards(tmp_path,pipeline,shape,policy):
    from rust_model import execute

    original = quantized_loop._virtual
    def full_tiles(*args,**kwargs):
        return original(*args,**{**kwargs,'input_prefix_bytes':None})

    with options():
        plan = prepare_source(fixture(*shape))
        layout = quantized_loop.execution_layout(plan,operand_reuse=policy)
        with patch.object(quantized_loop,'_virtual',full_tiles):
            baseline = quantized_loop.virtual_source(plan,layout)
        selected = quantized_loop.virtual_source(plan,layout)
        programs = []
        for virtual in (baseline,selected):
            machine,program = compile_fixture(virtual,pipeline=pipeline)
            assert instruction_bound(virtual,machine,program)['admitted']
            programs.append(program)
        initial = memory_fixture(plan,layout)
        output = layout['output_addresses'][0]-DRAM_BASE
        status = layout['status_address']-DRAM_BASE
        for rejected_role in (None,0,1):
            memory = bytearray(initial)
            if rejected_role is not None:
                address = (layout['exponent_addresses'][rejected_role]-DRAM_BASE
                           + layout['exponent_layouts'][rejected_role]['bytes']-2048)
                memory[address:address+4] = (1000).to_bytes(4,'little')
            results = [execute(program,memory,DRAM_BASE,tmp_path/f'{rejected_role}-{i}')
                       for i,program in enumerate(programs)]
            assert results[0][0] == results[1][0]
            actual,after = results[1]
            assert actual[status:status+4] == int(rejected_role is not None).to_bytes(4,'little')
            assert actual[:output] == memory[:output] and actual[status+32:] == memory[status+32:]
            assert after['dram_read_bytes'] < results[0][1]['dram_read_bytes']
            assert after['dram_write_bytes'] == results[0][1]['dram_write_bytes']
        # The partial load, not the stored padding, supplies zero to inactive
        # rows. Poison those bytes and keep every logical source value fixed.
        memory = bytearray(initial)
        for rows,address,size in zip((plan.m,plan.n),layout['input_addresses'],layout['input_buffer_bytes'],strict=True):
            if rows < 32:
                for first in range(address-DRAM_BASE,address-DRAM_BASE+size,1024):
                    memory[first+rows*32:first+1024] = bytes([0x38])*(1024-rows*32)
        actual,_ = execute(programs[1],memory,DRAM_BASE,tmp_path/'poisoned')
        expected,_ = execute(programs[1],initial,DRAM_BASE,tmp_path/'zero-padding')
        assert actual[output:status+32] == expected[output:status+32]
        assert actual[:output] == memory[:output] and actual[status+32:] == memory[status+32:]
