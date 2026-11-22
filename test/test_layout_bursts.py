"""Contiguous affine spans use bounded aligned bursts of arbitrary beat count."""
import os

import pytest

from atlas_mlir_native.layout_device import prepare_source, virtual_program
from atlas_mlir_native.restoration_budget import instruction_bound
from merlin.semantic_compiler.model import IndexMap
from test_exponent_restoration import compile_fixture
from test_layout_device import reference, request


def test_dense_spans_are_not_split_at_unnecessary_power_of_two_boundaries():
    for size in (96, 192, 320, 1536, 3072):
        plan = prepare_source(request('copy', ((size,),), (size,), dtype='i8'))
        assert len(plan.segments) == 1
        assert plan.segments[0]['counts'] == []
        assert plan.segments[0]['block_bytes'] == size
    plan = prepare_source(request('copy', ((8480,),), (8480,), dtype='i8'))
    assert plan.segments[0]['counts'] == [5]
    assert plan.segments[0]['block_bytes'] == 1696


@pytest.mark.skipif(not (os.environ.get('ATLAS_RUST_NPU_MODEL') and os.environ.get('ATLAS_OOT_BIN_DIR')),
                    reason='select Rust simulator and Atlas tools')
@pytest.mark.parametrize('pipeline', ['mandatory', 'optimized'])
@pytest.mark.parametrize('source', [
    request('copy', ((192,),), (192,), dtype='i8'),
    request('copy', ((320,),), (320,), dtype='i8'),
    request('copy', ((1536,),), (1536,), dtype='i8'),
    request('copy', ((8480,),), (8480,), dtype='i8'),
    request('transpose', ((2,3,96),), (3,2,96), dtype='i8',
            mapping=IndexMap(3, ((0,1,0),(1,0,0),(0,0,1)), (0,0,0))),
])
def test_bursts_execute_every_byte_with_affine_rows_and_outer_guards(tmp_path, pipeline, source):
    from rust_model import execute

    plan = prepare_source(source)
    virtual = virtual_program(plan)
    machine, program = compile_fixture(virtual, pipeline=pipeline)
    assert instruction_bound(virtual, machine, program)['admitted']
    base = plan.io_memory_base-64
    memory = bytearray([0xa5])*(plan.workspace_bytes+128)
    raw = bytes((i*37+i//64*11) % 256 for i in range(plan.input_bytes[0]))
    memory[64:64+len(raw)] = raw
    actual, stats = execute(program, memory, base, tmp_path / 'model')
    output = plan.output_address-base
    assert actual[output:output+plan.output_bytes] == reference(source, [raw])
    assert actual[:output] == memory[:output]
    assert actual[output+plan.physical_bytes:] == memory[output+plan.physical_bytes:]
    if 'dram_read_bytes' in stats:
        assert stats['dram_read_bytes'] == stats['dram_write_bytes'] == len(raw)
