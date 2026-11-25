"""Single-tile dynamic groups retain exact input bytes across scale selection."""
from dataclasses import replace
import os

import pytest

from atlas_mlir_native.activation_quantize_device import prepare_plan, virtual_program
from atlas_mlir_native.binding import DRAM_BASE
from atlas_mlir_native.options import codegen_options, current_codegen_options
from atlas_mlir_native.restoration_budget import instruction_bound
from test_activation_quantize_device import dynamic_source, host_reference
from test_exponent_restoration import compile_fixture


@pytest.mark.skipif(not (os.environ.get('ATLAS_RUST_NPU_MODEL') and os.environ.get('ATLAS_OOT_BIN_DIR')),
                    reason='select Rust simulator and Atlas tools')
@pytest.mark.parametrize('pipeline', ['mandatory', 'optimized'])
@pytest.mark.parametrize('width', [32,64,96,128,256,512,1024,2048])
def test_exact_group_codes_exponents_guards_and_read_traffic(tmp_path, pipeline, width):
    from rust_model import execute

    source = dynamic_source(shape=(3,width), reduce='token', rule='nc')
    with codegen_options(replace(current_codegen_options(), io_memory_bytes=1048576)):
        plan = prepare_plan(source)
        virtual = virtual_program(plan)
        machine, program = compile_fixture(virtual, pipeline=pipeline)
        assert instruction_bound(virtual, machine, program)['admitted']
    special = (0,0x8000,1,0x807f,0x7f80,0xff80,0x7fc1,0xffa3)
    raw = bytearray()
    for index in range(plan.count):
        row, lane = divmod(index, width)
        value = (special[lane] if row == 0 and lane < len(special) else
                 (0x3b00 + row*128 + lane%128) | ((lane%3 == 0) << 15))
        raw.extend(value.to_bytes(2,'little'))
    expected_codes, expected_exponents = host_reference(source, bytes(raw))
    memory = bytearray([0xa5])*(plan.workspace_bytes+64)
    memory[:len(raw)] = raw
    actual, stats = execute(program, memory, DRAM_BASE, tmp_path / 'run')
    code_at, exponent_at = plan.code_address-DRAM_BASE, plan.exponent_address-DRAM_BASE
    assert actual[:len(raw)] == raw
    assert actual[code_at:code_at+len(expected_codes)] == expected_codes
    assert actual[exponent_at:exponent_at+len(expected_exponents)] == expected_exponents
    assert actual[plan.workspace_bytes:] == bytes([0xa5])*64
    single_tile = width <= 1024 and width & (width-1) == 0
    assert stats['dram_read_bytes'] == len(raw)*(1 if single_tile else 2)
