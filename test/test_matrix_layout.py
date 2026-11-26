"""Physical matrix handoffs keep every byte on device in one program."""

import json
import os
from pathlib import Path

import numpy as np
import pytest

from atlas_mlir_native.matrix_layout import (
    DENSE_BF16, MATRIX_BF16, bf16_matrix_transition,
)
from test_program_sharing import execute, tool
from merlin.compiler.semantic.codegen import CodegenOptions, codegen_options
from merlin.compiler.semantic.target_binding import NativeCompilationError

NATIVE = bool(os.environ.get('ATLAS_OOT_BIN_DIR'))
RUST = bool(os.environ.get('ATLAS_RUST_NPU_MODEL'))
BASE = 0x90000000


@pytest.fixture(autouse=True)
def selected_memory_window():
    with codegen_options(CodegenOptions(io_memory_bytes=67108864)):
        yield


def matrix_bytes(dense, rows, columns):
    # Independent coordinate permutation of raw 16-bit codes, with no float math.
    codes = np.frombuffer(dense, dtype='<u2').reshape(rows // 32, 32, columns // 32, 2, 16)
    return codes.transpose(0, 2, 3, 1, 4).copy().tobytes()


def compile_virtual(source, pipeline):
    from atlas_mlir_native.restoration_budget import instruction_bound

    native = tool('atlas-opt', source, pipeline)
    assert native.returncode == 0, native.stderr
    checked = tool('atlas-opt', native.stdout, '--verify-atlas-generated-schedule',
                   '--verify-atlas-machine-stream', '--verify-atlas-timing')
    assert checked.returncode == 0, checked.stderr
    emitted = tool('atlas-emit', native.stdout, '--program-json')
    assert emitted.returncode == 0, emitted.stderr
    program = json.loads(emitted.stdout)
    traps = [row for row in program['instructions'] if row['operation'] == 'atlas.trap']
    assert len(traps) == 1 and traps[0]['fields']['kind'] == 'ecall'
    assert instruction_bound(source, native.stdout, program, max_instructions=128000000)['admitted']
    return program


@pytest.mark.parametrize('change', [
    dict(rows=True), dict(rows=0), dict(rows=33), dict(columns=48),
    dict(rows=1 << 30), dict(source_layout='f32'),
    dict(destination_layout=MATRIX_BF16), dict(destination_address=BASE + 32),
    dict(destination_address=BASE), dict(source_address=BASE + 1),
    dict(source_address=True), dict(destination_address=0x100000000),
    dict(io_memory_base=0x40000),
])
def test_incomplete_layouts_aliases_and_invalid_windows_are_refused(change):
    options = dict(rows=32, columns=32, source_address=BASE,
                   destination_address=BASE + 8192,
                   source_layout=MATRIX_BF16, destination_layout=DENSE_BF16)
    options.update(change)
    with pytest.raises((ValueError, NativeCompilationError)):
        bf16_matrix_transition(**options)


@pytest.mark.skipif(not NATIVE, reason='select Atlas tools')
def test_member_input_extent_is_not_an_invocation_wide_policy():
    from test_program_sharing import copy_stages

    original = copy_stages(2)
    marker = 'atlas.input_dram_base'
    source = original.replace(marker, 'atlas.input_dram_bytes = 2048 : i64, ' + marker, 1)
    joined = tool('atlas-opt', source, '--compose-atlas-program=share-bodies=true')
    assert joined.returncode == 0, joined.stderr
    assert 'atlas.input_dram_bytes' not in joined.stdout
    for extent in ('0 : i64', '-1 : i64', '4294967296 : i64', '2048 : i32', '"unknown"'):
        invalid = original.replace(marker, f'atlas.input_dram_bytes = {extent}, ' + marker, 1)
        refused = tool('atlas-opt', invalid, '--compose-atlas-program=share-bodies=true')
        assert refused.returncode != 0
        assert 'bounded optional input DRAM extent' in refused.stderr


@pytest.mark.skipif(not NATIVE or not RUST, reason='select Atlas tools and Rust')
@pytest.mark.parametrize('shape', [(32, 32), (64, 96), (96, 64), (64, 2048), (1024, 768)])
@pytest.mark.parametrize('reverse', [False, True])
def test_raw_matrix_layouts_preserve_every_code_and_guard(tmp_path, shape, reverse):
    rows, columns = shape
    size = rows * columns * 2
    source_offset, output_offset = 4096, size + 8192
    source_layout, output_layout = (DENSE_BF16, MATRIX_BF16) if reverse else (MATRIX_BF16, DENSE_BF16)
    source = bf16_matrix_transition(rows=rows, columns=columns,
        source_address=BASE + source_offset, destination_address=BASE + output_offset,
        source_layout=source_layout, destination_layout=output_layout)
    program = compile_virtual(source, '--atlas-compile=enhanced-virtual-scheduling=true')
    # Runs at model-layer sizes exercise every BF16 encoding, including NaNs.
    dense = ((np.arange(rows * columns, dtype=np.uint32) * 73 + 31) & 65535).astype('<u2').tobytes()
    physical = matrix_bytes(dense, rows, columns)
    raw, result = (dense, physical) if reverse else (physical, dense)
    capacity = (output_offset + size + 8191) // 4096 * 4096
    initial = bytearray(b'\xa5' * capacity)
    initial[source_offset:source_offset + size] = raw
    expected = bytearray(initial)
    expected[output_offset:output_offset + size] = result
    assert execute(tmp_path, program, initial, 0) == expected


@pytest.mark.skipif(not NATIVE or not RUST, reason='select Atlas tools and Rust')
@pytest.mark.parametrize('shape', [(32, 64, 32), (64, 96, 65)])
@pytest.mark.parametrize('pipeline', ['--atlas-compile-unscheduled',
                                     '--atlas-compile=enhanced-virtual-scheduling=true'])
def test_matrix_to_dense_arithmetic_runs_in_one_dispatch(tmp_path, shape, pipeline):
    from atlas_mlir_native.exact_float_device import prepare_plan as float_plan, virtual_program
    from atlas_mlir_native.looped_tiling import _virtual, prepare_plan
    from atlas_mlir_native.resident_graph import compose_fragments
    from merlin.compiler.transforms.tiling import materialize_looped_mnk_inputs
    from test_exact_float_device import request as float_request
    from test_looped_mnk import source_request

    rows, columns, reduction = shape
    plan = prepare_plan(source_request(rows, columns, reduction))
    size = rows * columns * 2
    a = np.zeros((rows, reduction), dtype=np.uint8)
    w = np.zeros((columns, reduction), dtype=np.uint8)
    for row in range(rows):
        a[row, row % 32::32] = 0x38
    for column in range(columns):
        w[column, column % 32::32] = 0xb8 if column % 2 else 0x38
    packed = materialize_looped_mnk_inputs(plan, {'a': a.tobytes(), 'w': w.tobytes()})
    addresses = (BASE, BASE + len(packed[0]))
    matrix_address = (addresses[1] + len(packed[1]) + 2047) // 2048 * 2048
    dense_address = matrix_address + size
    arithmetic = float_plan(float_request('bf16_neg', rows * columns), io_memory_base=dense_address)
    stages = [
        _virtual(plan, addresses, matrix_address),
        bf16_matrix_transition(rows=rows, columns=columns, source_address=matrix_address,
            destination_address=dense_address, source_layout=MATRIX_BF16,
            destination_layout=DENSE_BF16),
        virtual_program(arithmetic),
    ]
    joined = compose_fragments(stages, tools=Path(os.environ['ATLAS_OOT_BIN_DIR']),
                               boundaries_lowered=True, share_bodies=True)
    program = compile_virtual(joined, pipeline)
    if shape == (32, 64, 32) and pipeline == '--atlas-compile-unscheduled':
        reordered = compose_fragments([stages[1], stages[0], stages[2]],
            tools=Path(os.environ['ATLAS_OOT_BIN_DIR']), boundaries_lowered=True)
        refused = tool('atlas-opt', reordered, pipeline)
        assert refused.returncode != 0, 'relayout must not read an unproduced matrix'
        assert 'initialized' in refused.stderr
    # An independent integer contraction supplies the expected exact small integers.
    av = (a != 0).astype(np.int32)
    wv = np.where(w == 0xb8, -1, w != 0).astype(np.int32)
    codes = ((av @ wv.T).astype('<f4').view('<u4') >> 16).astype('<u2')
    dense = codes.tobytes()
    result = (codes ^ 0x8000).tobytes()
    capacity = (arithmetic.status_address - BASE + 8191) // 4096 * 4096
    for trial in range(2):
        initial = bytearray(bytes([0x5a if trial else 0xa5]) * capacity)
        for address, payload in zip(addresses, packed, strict=True):
            initial[address - BASE:address - BASE + len(payload)] = payload
        expected = bytearray(initial)
        for address, payload in ((matrix_address, matrix_bytes(dense, rows, columns)),
                                 (dense_address, dense), (arithmetic.output_address, result),
                                 (arithmetic.status_address, bytes(32))):
            expected[address - BASE:address - BASE + len(payload)] = payload
        assert execute(tmp_path, program, initial, trial) == expected
