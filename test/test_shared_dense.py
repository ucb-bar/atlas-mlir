"""Source, storage and input-domain refusals for the shared dense adapter."""

from dataclasses import replace
from pathlib import Path

import pytest

pytest.importorskip("merlin.compile.scheduling.dense_tile_ir")

from atlas_mlir_native.shared_dense import pack_inputs, plan, virtual_program
from tools.check_shared_dense import source_for

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("shape", [(32, 32, 32), (32, 64, 32), (64, 32, 64)])
def test_actual_checked_calls_drive_virtual_expansion(shape):
    selected, schedule = plan(source_for(ROOT, *shape), entry="mxu0_tile")
    program = virtual_program(selected, schedule)
    outputs = shape[0]//32 * (shape[2]//32)
    assert program.count('"atlas.virtual_output_bf16"') == outputs
    if shape[1] == 32:
        assert program.count('"atlas.virtual_mxu_matmul"') == outputs
        assert "virtual_mxu_accumulate" not in program
    else:
        assert program.count('"atlas.virtual_mxu_reset"') == outputs
        assert program.count('"atlas.virtual_mxu_accumulate"') == outputs * (shape[1]//32-1)
    with pytest.raises(ValueError, match="selection differs"):
        virtual_program(replace(selected, k=selected.k+32), schedule)


@pytest.mark.parametrize("shape", [(31, 32, 32), (32, 33, 32), (32, 32, 31), (320, 320, 320)])
def test_unqualified_tails_and_large_expansions_refuse(shape):
    with pytest.raises(ValueError):
        plan(source_for(ROOT, *shape), entry="mxu0_tile")


@pytest.mark.parametrize("code", [0x80, 0x01, 0x87, 0x7f, 0xff])
def test_runtime_guard_rejects_codes_outside_source_domain(code):
    selected, _ = plan(source_for(ROOT, 32, 32, 32), entry="mxu0_tile")
    with pytest.raises(ValueError, match="domain"):
        pack_inputs(selected, bytes([code])*1024, bytes(1024))


def test_wrong_input_extent_refuses_and_transposed_maps_preserve_weights():
    selected, _ = plan(source_for(ROOT, 32, 32, 64), entry="mxu0_tile")
    with pytest.raises(ValueError, match="extent"):
        pack_inputs(selected, bytes(1023), bytes(2048))
    a = bytes(1024)
    w = bytes(0x38 if r == c % 32 else 0 for c in range(64) for r in range(32))
    slots = pack_inputs(selected, a, w)
    assert slots == ((0, a), (2048, w[:1024]), (4096, w[1024:]))
    kn = bytes(w[c*32+r] for r in range(32) for c in range(64))
    assert pack_inputs(replace(selected, rhs_transposed=False), a, kn) == slots
