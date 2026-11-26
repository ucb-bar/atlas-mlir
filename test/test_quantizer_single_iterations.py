"""Single-iteration specialization preserves static and grouped exponents."""
from dataclasses import replace
import os

import pytest

from atlas_mlir_native.activation_quantize_device import prepare_plan, virtual_program
from atlas_mlir_native.binding import DRAM_BASE
from atlas_mlir_native.options import codegen_options, current_codegen_options
from atlas_mlir_native.restoration_budget import instruction_bound
from test_activation_quantize_device import dynamic_source, host_reference
from test_already_quantized import quantizer_source
from test_exponent_restoration import compile_fixture


@pytest.mark.skipif(not (os.environ.get('ATLAS_RUST_NPU_MODEL') and os.environ.get('ATLAS_OOT_BIN_DIR')),
                    reason='select Rust simulator and Atlas tools')
@pytest.mark.parametrize('pipeline', ['mandatory', 'optimized'])
@pytest.mark.parametrize('source', [
    pytest.param(quantizer_source(count=32), id='static-prefix'),
    pytest.param(quantizer_source(count=1024), id='static-tile'),
    pytest.param(quantizer_source(count=2048), id='static-multiple-tiles'),
    pytest.param(dynamic_source(shape=(6,32), reduce='block', rows=2), id='block-two-exponents'),
    pytest.param(dynamic_source(shape=(2,4,32), reduce='head_block', rows=2), id='head-block'),
    pytest.param(dynamic_source(shape=(2,3,32), reduce='head'), id='head-irregular'),
    pytest.param(dynamic_source(shape=(1,4096), reduce='tensor'), id='tensor-multiple-tiles'),
])
def test_source_outputs_and_exponent_broadcasts(tmp_path, pipeline, source):
    from rust_model import execute

    with codegen_options(replace(current_codegen_options(), io_memory_bytes=1048576)):
        plan = prepare_plan(source)
        virtual = virtual_program(plan)
        machine, program = compile_fixture(virtual, pipeline=pipeline)
        assert instruction_bound(virtual, machine, program)['admitted']
    special = (0,0x8000,1,0x807f,0x7f80,0xff80,0x7fc1,0xffa3)
    raw = b''.join((special[i%37] if i%37 < 8 else
        (0x3b00 + i%1024) | ((i%3 == 0) << 15)).to_bytes(2,'little') for i in range(plan.count))
    expected = host_reference(source, raw)
    memory = bytearray([0xa5])*(plan.workspace_bytes+64)
    memory[:len(raw)] = raw
    actual, _ = execute(program, memory, DRAM_BASE, tmp_path / 'run')
    assert actual[:len(raw)] == raw
    for address, value in zip((plan.code_address, plan.exponent_address), expected, strict=True):
        start = address-DRAM_BASE
        assert actual[start:start+len(value)] == value
    assert actual[plan.workspace_bytes:] == bytes([0xa5])*64
