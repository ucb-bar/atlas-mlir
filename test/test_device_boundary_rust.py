"""Boundary scans preserve all logical bits, padding, guards and sticky status."""
import json
import os
import subprocess

import numpy as np
import pytest
from atlas_mlir_native.binding import DRAM_BASE
from atlas_mlir_native.device_boundary import virtual_transition
from atlas_mlir_native.resident_graph import compose_fragments
from test_device_boundary import RANGES, packed, reference
from test_exponent_restoration import BIN, compile_fixture

pytestmark = pytest.mark.skipif(
    not (os.environ.get('ATLAS_RUST_NPU_MODEL') and os.environ.get('ATLAS_OOT_BIN_DIR')),
    reason='select Rust simulator and Atlas tools')
SOURCE, DESTINATION, STATUS = DRAM_BASE + 2048, DRAM_BASE + 0x10000, DRAM_BASE + 0x18000


def compile_program(virtual, root):
    _, physical = compile_fixture(virtual)
    program = root / 'program.bin'
    program.write_bytes(b''.join(row['word_u32'].to_bytes(4, 'little') for row in physical['instructions']))
    return program


def check_memory(program, initial, expected, root):
    root.mkdir()
    memory = root / 'memory.bin'
    memory.write_bytes(initial)
    config = root / 'config.json'
    config.write_text(json.dumps({'compute': 'cpu', 'dram_base': DRAM_BASE, 'dram_size': len(initial)}))
    state = root / 'state'
    subprocess.run([os.environ['ATLAS_RUST_NPU_MODEL'], str(program), '--config', str(config),
        '--load', f'{DRAM_BASE}:{memory}', '--dump-state', str(state)], check=True, capture_output=True)
    actual = b''.join((state / f'dram_{address:016x}.bin').read_bytes()
                     for address in range(DRAM_BASE, DRAM_BASE + len(initial), 4096))
    assert actual == expected


@pytest.mark.parametrize('kind,rows,columns', [
    ('identity', 1, 64), ('identity', 3, 7), ('identity', 17, 17), ('identity', 35, 37),
    ('column_to_row', 3, 1), ('column_to_row', 17, 1), ('column_to_row', 35, 1),
    ('broadcast_column', 3, 1), ('broadcast_column', 33, 1), ('broadcast_column', 64, 1),
])
@pytest.mark.parametrize('canonical,ranges,pad', [
    (False, RANGES, 0), (True, ((0x3f80, 0x4000),), 0x3f80), (False, (), 0),
])
def test_layouts_tails_and_raw_encodings(tmp_path, kind, rows, columns, canonical, ranges, pad):
    virtual = virtual_transition(source_address=SOURCE, destination_address=DESTINATION,
        status_address=STATUS, rows=rows, columns=columns, kind=kind,
        admitted_code_ranges=ranges, pad=pad, canonicalize_negative_zero=canonical)
    program = compile_program(virtual, tmp_path)
    for invalid in (False, True):
        codes = [0x3f80, 0x8000, 0x4000, 0xbf80, 0] if ranges else [0x7fc1, 0xffff, 0x7f80, 1, 0x8000]
        values = np.resize(np.array(codes, dtype='<u2'), (rows, columns))
        if pad:
            values.fill(0x3f80)
        if invalid:
            values[-1, -1] = 0x7f80
        payload = packed(values, 0x7fc1)  # Nonlogical producer lanes must not reject.
        wanted, rejected = reference(values, kind, ranges, pad, canonical)
        initial = bytearray([0xa5]) * 0x20000
        initial[SOURCE - DRAM_BASE:SOURCE - DRAM_BASE + len(payload)] = payload
        expected = bytearray(initial)
        expected[DESTINATION - DRAM_BASE:DESTINATION - DRAM_BASE + len(wanted)] = wanted
        expected[STATUS - DRAM_BASE:STATUS - DRAM_BASE + 2048] = int(rejected).to_bytes(4, 'little') + bytes(2044)
        check_memory(program, initial, expected, tmp_path / str(invalid))


def test_later_admitted_boundary_keeps_earlier_rejection(tmp_path):
    last = DRAM_BASE + 0x12000
    fragments = [virtual_transition(source_address=SOURCE, destination_address=DESTINATION,
        status_address=STATUS, rows=3, columns=1, kind='broadcast_column', admitted_code_ranges=RANGES,
        canonicalize_negative_zero=True),
        virtual_transition(source_address=DESTINATION, destination_address=last,
        status_address=STATUS, rows=3, columns=32, admitted_code_ranges=RANGES, initialize_status=False)]
    program = compile_program(compose_fragments(fragments, tools=BIN, boundaries_lowered=True), tmp_path)
    values = np.array([[0x7f80], [0x8000], [0x3f80]], dtype='<u2')
    wanted, rejected = reference(values, 'broadcast_column', RANGES, 0, True)
    assert rejected
    initial = bytearray([0xa5]) * 0x20000
    initial[SOURCE - DRAM_BASE:SOURCE - DRAM_BASE + 2048] = packed(values, 0x7fc1)
    expected = bytearray(initial)
    for address in (DESTINATION, last):
        expected[address - DRAM_BASE:address - DRAM_BASE + len(wanted)] = wanted
    expected[STATUS - DRAM_BASE:STATUS - DRAM_BASE + 2048] = bytes([1]) + bytes(2047)
    check_memory(program, initial, expected, tmp_path / 'joined')
