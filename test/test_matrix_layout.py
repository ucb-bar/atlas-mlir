"""Physical matrix handoffs keep every byte on device in one program."""

import os

import pytest

from atlas_mlir_native.matrix_layout import (
    DENSE_BF16, MATRIX_BF16, bf16_matrix_transition,
)
from test_program_sharing import tool
from merlin.semantic_compiler.target_binding import NativeCompilationError

NATIVE = bool(os.environ.get('ATLAS_OOT_BIN_DIR'))
BASE = 0x90000000


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
