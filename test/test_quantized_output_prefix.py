"""Output prefixes preserve logical lanes, refusal and packed tile strides."""
import os

import pytest

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
@pytest.mark.parametrize('pipeline',['mandatory','optimized'])
@pytest.mark.parametrize('shape,policy',[
    ((3,65,65,(),'row','row',True),'batch-panel-cache'),
    ((3,5,65,(2,),'row','row',True),'ordered'),
    ((35,3,65,(2,),'row','batch',False),'ordered'),
    ((1,17,33,(),'scalar','row',True),'ordered'),
    ((32,16,33,(),'row','row',True),'ordered'),
    ((32,32,33,(),'row','row',True),'ordered'),
])
def test_output_prefix_matches_full_store_and_preserves_padding(tmp_path,pipeline,shape,policy):
    from rust_model import execute

    with options():
        plan = prepare_source(fixture(*shape))
        layout = execution_layout(plan,operand_reuse=policy)
        legacy = execution_layout(plan,operand_reuse=policy,output_prefix=False)
        programs = []
        for selected in (legacy,layout):
            virtual = virtual_source(plan,selected)
            machine,program = compile_fixture(virtual,pipeline=pipeline)
            assert instruction_bound(virtual,machine,program)['admitted']
            programs.append(program)
        initial = memory_fixture(plan,layout)
        first = layout['output_addresses'][0]-DRAM_BASE
        status = layout['status_address']-DRAM_BASE
        size = layout.get('output_tile_store_bytes',2048)
        for rejected_role in (None,0,1):
            memory = bytearray(initial)
            if rejected_role is not None:
                at = (layout['exponent_addresses'][rejected_role]-DRAM_BASE
                      +layout['exponent_layouts'][rejected_role]['bytes']-2048)
                memory[at:at+4] = (1000).to_bytes(4,'little')
            results = [execute(program,memory,DRAM_BASE,tmp_path/f'{rejected_role}-{index}')
                       for index,program in enumerate(programs)]
            expected = bytearray(results[0][0])
            for at in range(first,status,2048):
                expected[at+size:at+2048] = memory[at+size:at+2048]
            actual,after = results[1]
            assert actual == expected
            assert actual[:first] == memory[:first] and actual[status+32:] == memory[status+32:]
            assert actual[status:status+4] == int(rejected_role is not None).to_bytes(4,'little')
            assert after['dram_read_bytes'] == results[0][1]['dram_read_bytes']
            assert results[0][1]['dram_write_bytes']-after['dram_write_bytes'] == (
                (2048-size)*layout['output_bytes']//2048)
