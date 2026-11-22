"""Completion needs one beat; reserved padding is neither read nor overwritten."""
import os
from unittest.mock import patch

import pytest

from atlas_mlir_native import exponent_restoration
from atlas_mlir_native.already_quantized import prepare_source
from atlas_mlir_native.binding import DRAM_BASE
from atlas_mlir_native.quantized_loop import execution_layout, virtual_source
from atlas_mlir_native.restoration_budget import instruction_bound
from test_exponent_page_cache import fixture
from test_exponent_panel_cache import options
from test_exponent_restoration import compile_fixture
from test_quantized_operand_reuse import memory_fixture


@pytest.mark.skipif(not (os.environ.get('ATLAS_RUST_NPU_MODEL') and os.environ.get('ATLAS_OOT_BIN_DIR')),
                    reason='select Rust simulator and Atlas tools')
@pytest.mark.parametrize('pipeline', ['mandatory', 'optimized'])
@pytest.mark.parametrize('shape,policy', [
    ((3,35,33,(),'scalar','row',True), 'ordered'),
    ((64,65,65,(),'row','row',True), 'interleaved-panel-cache'),
    ((3,35,65,(2,),'batch','batch',False), 'ordered'),
])
def test_status_prefix_preserves_outputs_refusal_reset_and_padding(tmp_path, pipeline, shape, policy):
    from rust_model import execute

    with options():
        plan = prepare_source(fixture(*shape))
        layout = execution_layout(plan, operand_reuse=policy)
        programs = []
        for size in (2048,32):
            with patch.object(exponent_restoration, 'STATUS_WRITE_BYTES', size):
                virtual = virtual_source(plan, layout)
            machine, program = compile_fixture(virtual, pipeline=pipeline)
            assert instruction_bound(virtual, machine, program)['admitted']
            programs.append(program)
        initial = memory_fixture(plan, layout)
        status = layout['status_address']-DRAM_BASE
        output = layout['output_addresses'][0]-DRAM_BASE
        # Reuse a frame after failure: stale status must not leak into success.
        initial[status:status+4] = (1).to_bytes(4,'little')
        for rejected_role in (None,0,1):
            memory = bytearray(initial)
            if rejected_role is not None:
                address = (layout['exponent_addresses'][rejected_role]-DRAM_BASE
                           + layout['exponent_layouts'][rejected_role]['bytes']-2048)
                memory[address:address+4] = (1000).to_bytes(4,'little')
            results = [execute(program,memory,DRAM_BASE,tmp_path / f'{rejected_role}-{i}')
                       for i,program in enumerate(programs)]
            baseline, before = results[0]
            actual, after = results[1]
            assert actual[:status+32] == baseline[:status+32]
            assert actual[status:status+4] == int(rejected_role is not None).to_bytes(4,'little')
            assert actual[status+4:status+32] == bytes(28)
            assert actual[:output] == memory[:output]
            assert actual[status+32:] == memory[status+32:]
            assert before['dram_write_bytes']-after['dram_write_bytes'] == 2016
            assert before['dram_read_bytes'] == after['dram_read_bytes']
