"""Counted lowering consumes checked shared calls, including K beyond old limits."""

import os
from dataclasses import replace
from pathlib import Path

import pytest
from atlas_mlir_native.shared_counted_dense import counted_dense_program
from atlas_mlir_native.shared_dense import (
    INPUT_BASE,
    POLICY,
    compile_virtual,
    schedule_dense,
)
from merlin.transforms.ordered_float_dense import OrderedFloatDense


@pytest.mark.parametrize("k", [32, 64, 96, 2048, 5632])
@pytest.mark.parametrize("pipeline", ["source-order", "optimized"])
def test_counted_program_has_bounded_size(k, pipeline, tmp_path):
    bins = os.environ.get("ATLAS_OOT_BIN_DIR")
    if not bins:
        pytest.skip("select built Atlas tools")
    selected = OrderedFloatDense("0" * 64, "tile", 32, k, 32, False, POLICY)
    scheduled = schedule_dense(selected, max_tile_products=256)
    virtual = counted_dense_program(
        selected, scheduled, output_base=INPUT_BASE + (2 * (k // 32) + 1) * 2048
    )
    manifest = compile_virtual(
        "diagnostic tile contract",
        virtual,
        destination=tmp_path / "compile",
        atlas_bin_dir=Path(bins),
        pipeline=pipeline,
    )
    assert 0 < manifest["words"] < 128
    assert ("arith.cmpi ult" in virtual) == (k > 32)


def test_counted_does_not_reinterpret_mutated_selection_or_overlap():
    selected = OrderedFloatDense("0" * 64, "tile", 32, 96, 32, False, POLICY)
    scheduled = schedule_dense(selected)
    with pytest.raises(ValueError, match="selection differs"):
        counted_dense_program(
            replace(selected, k=64), scheduled, output_base=INPUT_BASE + 16384
        )
    with pytest.raises(ValueError, match="overlap"):
        counted_dense_program(selected, scheduled, output_base=INPUT_BASE + 2048)


@pytest.mark.parametrize("limit", [0, -1, True, 2.5])
def test_checking_budget_is_explicit_positive_integer(limit):
    selected = OrderedFloatDense("0" * 64, "tile", 32, 32, 32, False, POLICY)
    with pytest.raises(ValueError):
        schedule_dense(selected, max_tile_products=limit)


def test_boundary_encoding_and_host_replay_refuse_wrong_operands():
    torch = pytest.importorskip("torch")
    pytest.importorskip("npu_model")
    import numpy as np

    from test.run_shared_layer import ReplayContractions, encoded

    values = np.array(
        [
            0.0,
            -0.0,
            2**-7,
            -(2**-7),
            2**-6,
            1e20,
            -1e20,
            -0.7187499403953552,
            -0.7187501788139343,
        ],
        dtype=np.float32,
    )
    assert encoded(values) == bytes([0, 0, 0, 0, 8, 126, 254, 179, 180])
    for invalid in (np.nan, np.inf, -np.inf):
        with pytest.raises(ValueError, match="finite"):
            encoded(np.array([invalid], dtype=np.float32))
    a = np.array([[1.0, 2.0]], dtype=np.float32)
    b = np.array([[2.0], [1.0]], dtype=np.float32)
    c = np.array([[4.0]], dtype=np.float32)
    replay = ReplayContractions([(a, b, c)])
    with replay:
        got = torch.mm(torch.from_numpy(a), torch.from_numpy(b))
    assert replay.index == 1
    np.testing.assert_array_equal(got.numpy(), c)
    with pytest.raises(AssertionError), ReplayContractions([(a, b, c)]):
        torch.mm(torch.from_numpy(a + 0.1), torch.from_numpy(b))
