"""Unit dimensions cannot create movement or constrain DMA alignment."""
import itertools
import os

import pytest

from atlas_mlir_native.layout_device import prepare_source, virtual_program
from atlas_mlir_native.restoration_budget import instruction_bound
from merlin.compiler.semantic.model import IndexMap
from test_exponent_restoration import compile_fixture
from test_layout_device import request, reference


def singleton_transpose(shape, permutation, dtype):
    return request('transpose', (shape,), tuple(shape[i] for i in permutation), dtype=dtype,
                   mapping=IndexMap(len(shape), tuple(tuple(int(i == axis) for axis in permutation)
                                                    for i in range(len(shape))), (0,)*len(shape)))


@pytest.mark.parametrize('dtype', ['i8', 'bf16', 'f32', 'i64'])
def test_complete_singleton_permutations_have_the_same_address_stream(dtype):
    for shape in ((1, 32, 32), (32, 1, 32), (32, 32, 1), (1, 1, 32)):
        for permutation in itertools.permutations(range(3)):
            source = singleton_transpose(shape, permutation, dtype)
            plan = prepare_source(source)
            # Only moving unit dimensions preserves the dense order.
            dense_order = [axis for axis in permutation if shape[axis] != 1] == [
                axis for axis in range(3) if shape[axis] != 1]
            if dense_order:
                assert all(segment['mode'] == 'dma' for segment in plan.segments)
                raw = bytes((i*37) % 256 for i in range(plan.input_bytes[0]))
                assert reference(source, [raw]) == raw


@pytest.mark.skipif(not (os.environ.get('ATLAS_RUST_NPU_MODEL') and os.environ.get('ATLAS_OOT_BIN_DIR')),
                    reason='select Rust simulator and Atlas tools')
@pytest.mark.parametrize('pipeline', ['mandatory', 'optimized'])
@pytest.mark.parametrize('shape,permutation,dtype', [
    ((32,32,1), (2,0,1), 'bf16'), ((1,3,32), (1,0,2), 'i8'),
    ((32,1,32), (2,1,0), 'i8'), ((1,3,7), (1,2,0), 'f32'),
])
def test_public_layouts_execute_exactly_with_tails_and_real_permutations(tmp_path, pipeline, shape, permutation, dtype):
    from rust_model import execute

    source = singleton_transpose(shape, permutation, dtype)
    plan = prepare_source(source)
    virtual = virtual_program(plan)
    machine, program = compile_fixture(virtual, pipeline=pipeline)
    assert instruction_bound(virtual, machine, program)['admitted']
    raw = bytes((i*37 + i//64*11) % 256 for i in range(plan.input_bytes[0]))
    base = plan.io_memory_base-64
    memory = bytearray([0xa5])*(plan.workspace_bytes+128)
    memory[64:64+len(raw)] = raw
    actual, stats = execute(program, memory, base, tmp_path / 'model')
    output = plan.output_address-base
    assert actual[output:output+plan.output_bytes] == reference(source, [raw])
    assert actual[:output] == memory[:output]
    assert actual[output+plan.physical_bytes:] == memory[output+plan.physical_bytes:]
    if 'dram_read_bytes' in stats and all(segment['mode'] == 'dma' for segment in plan.segments):
        assert stats['dram_read_bytes'] == stats['dram_write_bytes'] == plan.physical_bytes
