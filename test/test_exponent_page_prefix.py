"""Exponent pages need only their first 32 signed words, including refusal paths."""
from unittest.mock import patch
import os

import pytest

from atlas_mlir_native import exponent_restoration, quantized_panel_reuse
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
@pytest.mark.parametrize('pipeline', ['mandatory','optimized'])
@pytest.mark.parametrize('shape,policy', [
    ((3,35,33,(),'scalar','row',True), 'ordered'),
    ((65,35,65,(),'row','row',True), 'batch-panel-cache'),
    ((64,65,257,(),'row','row',True), 'batch-panel-cache'),
    ((3,35,65,(2,),'batch','batch',False), 'ordered'),
])
def test_partial_page_loads_match_full_pages_and_preserve_padding(tmp_path, pipeline, shape, policy):
    from rust_model import execute

    with options():
        plan = prepare_source(fixture(*shape))
        layout = execution_layout(plan, operand_reuse=policy)
        programs = []
        for size in (2048,128):
            with patch.object(exponent_restoration, 'EXPONENT_PAGE_PAYLOAD_BYTES', size), \
                 patch.object(quantized_panel_reuse, 'EXPONENT_PAGE_PAYLOAD_BYTES', size):
                virtual = virtual_source(plan, layout)
            machine, program = compile_fixture(virtual, pipeline=pipeline)
            assert instruction_bound(virtual, machine, program)['admitted']
            programs.append(program)
        initial = memory_fixture(plan, layout)
        # These unused source words must not become scale values or be overwritten.
        for address,geometry in zip(layout['exponent_addresses'],layout['exponent_layouts'],strict=True):
            for page in range(address-DRAM_BASE,address-DRAM_BASE+geometry['bytes'],2048):
                initial[page+128:page+2048] = bytes([0xa5])*1920
        for rejected_role in (None,0,1):
            memory = bytearray(initial)
            if rejected_role is not None:
                address = (layout['exponent_addresses'][rejected_role]-DRAM_BASE
                           + layout['exponent_layouts'][rejected_role]['bytes']-2048)
                memory[address:address+4] = (1000).to_bytes(4,'little')
            results = [execute(program,memory,DRAM_BASE,tmp_path / f'{rejected_role}-{i}')
                       for i,program in enumerate(programs)]
            assert results[0][0] == results[1][0]
            actual, stats = results[1]
            output, status = layout['output_addresses'][0]-DRAM_BASE,layout['status_address']-DRAM_BASE
            end = status+layout['status_bytes']
            assert bool(int.from_bytes(actual[status:status+4],'little')) == (rejected_role is not None)
            assert actual[:output] == memory[:output] and actual[end:] == memory[end:]
            difference = results[0][1]['dram_read_bytes'] - stats['dram_read_bytes']
            assert difference > 0 and difference % 1920 == 0
            assert results[0][1]['dram_write_bytes'] == stats['dram_write_bytes']
