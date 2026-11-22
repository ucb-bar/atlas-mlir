"""Nonzero literal witnesses for compiler-placed Aten device arithmetic."""
import os
from unittest import mock

import numpy as np
import pytest

from atlas_mlir_native.reference_eval import device
from atlas_mlir_native.reference_eval import placed_device as placed
from atlas_mlir_native.reference_eval.evaluator import Tensor


def rules(**updates):
    root = os.environ.get("ATLAS_NPU_MODEL_ROOT")
    assert root, "ATLAS_NPU_MODEL_ROOT must name the pinned model; qualification never skips"
    return {"npu_model_revision": device.MODEL_REVISION, "numerics": "rtl", "npu_model_root": root, **updates}


def f32(raw):
    return Tensor(np.asarray(raw, np.uint32), "f32")


# Compiler declarations, as atlas-mlir row_softmax.py names them.
ABSOLUTE32 = "bf16-signed-zero-or-normal-exp8-to132-abs-at-most32-v1"
ROWSPAN80 = "bf16-row-relative-span80-pad32-v1"
POLICIES = {"source_softmax": "bf16-tiled-row-softmax-ltr-fp32-rne-chop-rcp-lut-v1",
            "source_masked_softmax": "bf16-tiled-masked-row-softmax-zero-neginf-v1"}


def softmax_rules(route, domain=ABSOLUTE32, **declaration):
    """A selected softmax region as decisions.SelectedDecisions passes its producer's declaration."""
    masked = route == "source_masked_softmax"
    policy = POLICIES[route]
    softmax = {"region": "region_7", "policy": policy, "score_domain": domain,
               "exclusion_bits": 0xff7f0000 if masked else None, **declaration}
    return rules(selected_routes=[route], softmax=softmax)


def test_n387_square_uses_bf16_multiply_and_widened_output():
    # The actual first RMSNorm-square discriminator from the source diagnostic.
    value = f32([0x3bf90000])
    result = placed.call("aten.pow.Tensor_Scalar", [value, 2], {}, rules())
    np.testing.assert_array_equal(result.raw, [0x38720000])
    assert result.dtype == "f32"
    assert not result.raw.flags.writeable
    # Binary32 source multiplication would instead produce 0x38723100.
    assert result.raw[0] != 0x38723100


def test_operands_narrow_rne_and_rtl_add_truncates():
    result = placed.call("aten.mul.Scalar", [f32([0x3f818001]), 1], {}, rules())
    np.testing.assert_array_equal(result.raw, [0x3f820000])
    # 1 + 3/256 is halfway between BF16 codes 0x3f81 and 0x3f82.
    # RTL add chops to 0x3f81; an ordinary BF16 addition rounds to 0x3f82.
    result = placed.call("aten.add.Tensor", [f32([[0x3f800000]]), f32([0x3c400000])], {}, rules())
    np.testing.assert_array_equal(result.raw, [[0x3f810000]])
    result = placed.call("aten.sub.Scalar", [f32([0x40000000]), 1], {}, rules())
    np.testing.assert_array_equal(result.raw, [0x3f800000])


def test_reduction_uses_rne_32_tree_then_chopped_seeded_tile_carry():
    words = np.zeros((1, 33), np.uint32)
    words[0, :2] = [0x3f800000, 0x3c000000]  # first tile -> 0x3f81
    words[0, 32] = 0x3b800000                 # next tile = half BF16 ulp
    result = placed.call("aten.sum.dim_IntList", [f32(words), [-1], True], {}, rules())
    np.testing.assert_array_equal(result.raw, [[0x3f810000]])
    # Summing all 33 values and RNE only once would produce 0x3f82.
    tree = np.full((1, 32), 0x3b800000, np.uint32)
    tree[0, 0] = 0x3f800000
    result = placed.call("aten.sum.dim_IntList", [f32(tree), [-1], False], {}, rules())
    np.testing.assert_array_equal(result.raw, [0x3f900000])


def test_mean_is_sum_then_device_reciprocal_multiply():
    value = f32(np.full((2, 32), 0x3f800000, np.uint32))
    result = placed.call("aten.mean.dim", [value, [-1], True], {}, rules())
    np.testing.assert_array_equal(result.raw, [[0x3f800000], [0x3f800000]])
    bf = Tensor(np.full((1, 32), 0x3f80, np.uint16), "bf16")
    result = placed.call("aten.mean.dim", [bf, [1], False], {}, rules())
    np.testing.assert_array_equal(result.raw, [0x3f80])
    assert result.dtype == "bf16"


def test_lut_sequences_and_bitwise_negation():
    for target, raw, expected in (("rsqrt", 0x40800000, 0x3f000000),
                                   ("div", 0x40800000, 0x40000000),
                                   ("exp", 0, 0x3f800000)):
        args = [f32([raw]), 2] if target == "div" else [f32([raw])]
        result = placed.call(f"aten.{target}.{'Scalar' if target == 'div' else 'default'}", args, {}, rules())
        np.testing.assert_array_equal(result.raw, [expected])
    result = placed.call("aten.neg.default", [f32([0x3f800001, 0x80000000])], {}, rules())
    np.testing.assert_array_equal(result.raw, [0xbf800001, 0])


def test_nonnegative_reduction_requires_explicit_policy():
    value = f32([[0x00800000, 0x00800000]])
    with pytest.raises(ValueError, match="code domain"):
        placed.call("aten.sum.dim_IntList", [value, [-1]], {}, rules())
    result = placed.call("aten.sum.dim_IntList", [value, [-1]], {},
                         rules(reduction_policy=placed.NONNEGATIVE_REDUCTION_POLICY))
    np.testing.assert_array_equal(result.raw, [0x01000000])


def test_fail_closed_rules_arguments_domains_and_table_identity():
    value = f32([0x3f800000])
    with pytest.raises(ValueError, match="pinned npu_model"):
        placed.call("aten.mul.Tensor", [value, value], {}, {})
    with pytest.raises(ValueError, match="square exponent"):
        placed.call("aten.pow.Tensor_Scalar", [value, 3], {}, rules())
    with pytest.raises(ValueError, match="scalar modifier"):
        placed.call("aten.add.Tensor", [value, value], {"alpha": 2}, rules())
    with pytest.raises(ValueError, match="code domain"):
        placed.call("aten.div.Scalar", [value, -1], {}, rules())
    with pytest.raises(ValueError, match="NaN payload"):
        placed.call("aten.mul.Scalar", [f32([0x7fc00001]), 1], {}, rules())
    with pytest.raises(ValueError, match="final contiguous axes"):
        placed.call("aten.sum.dim_IntList", [f32([[0x3f800000]]), [0]], {}, rules())
    with mock.patch.object(placed, "_pinned_table", return_value=b"modified"):
        with pytest.raises(ValueError, match="table differs"):
            placed.call("aten.rsqrt.default", [value], {}, rules())


def test_no_framework_arithmetic_entry_points_or_host_fallback():
    import torch
    from atlas_mlir_native.reference_eval import host
    # The model API owns its instruction implementation. The adapter must not
    # invoke ordinary torch function kernels or fall back to host arithmetic.
    with mock.patch.object(host, "call", side_effect=AssertionError("host fallback")), \
         mock.patch.multiple(torch, add=mock.DEFAULT, mul=mock.DEFAULT,
                             sum=mock.DEFAULT, exp=mock.DEFAULT, rsqrt=mock.DEFAULT) as kernels:
        result = placed.call("aten.pow.Tensor_Scalar", [f32([0x3bf90000]), 2], {}, rules())
        np.testing.assert_array_equal(result.raw, [0x38720000])
        for kernel in kernels.values():
            kernel.assert_not_called()


@pytest.mark.parametrize("op,left,right,expected", [
    ("add", 0x3d96, 0x3c07, 0x3da7),
    ("add", 0x3b51, 0xbe25, 0xbe22),
    ("add", 0x3f80, 0x3c40, 0x3f82),
    ("sub", 0x3f80, 0x3f80, 0),
    ("add", 0x8000, 0x8000, 0x8000),
    ("add", 0x0001, 0x0001, 0x0002),
])
def test_exact_route_rne_witnesses(op, left, right, expected):
    args = [Tensor(np.array([code], np.uint16), "bf16") for code in (left, right)]
    result = placed.call(f"aten.{op}.Tensor", args, {},
                         rules(selected_routes=["source_exact_float_device"]))
    np.testing.assert_array_equal(result.raw, [expected])


def test_exact_route_never_falls_back_to_raw_vpu():
    value = f32([0x3f800000])
    exact = rules(selected_routes=["source_exact_float_device"])
    result = placed.call("aten.add.Tensor", [value, value], {}, exact)
    np.testing.assert_array_equal(result.raw, [0x40000000])
    with pytest.raises(ValueError, match="unsupported source_exact"):
        placed.call("aten.mul.Tensor", [value, value], {}, exact)
    with pytest.raises(ValueError, match="conflicting exact"):
        placed.call("aten.neg.default", [value], {},
                    rules(selected_routes=["source_exact_float_device", "source_pointwise"]))


def test_selected_softmax_masks_padding_and_multiple_tiles():
    words = np.full((2, 33), 0xff7f0000, np.uint32)
    words[0, [0, 32]] = 0
    words[1, 0] = 0
    value = f32(words)
    result = placed.call("aten.softmax.int", [value, -1], {},
                         softmax_rules("source_masked_softmax"))
    expected = np.zeros_like(words)
    expected[0, [0, 32]] = 0x3f000000
    expected[1, 0] = 0x3f800000
    np.testing.assert_array_equal(result.raw, expected)
    with pytest.raises(ValueError, match="retained key"):
        placed.call("aten.softmax.int", [f32([[0xff7f0000]]), -1], {},
                    softmax_rules("source_masked_softmax"))


def test_plain_softmax_keeps_every_key_by_its_policy():
    plain = softmax_rules("source_softmax")
    result = placed.call("aten.softmax.int", [f32(np.zeros((2, 64), np.uint32)), -1], {}, plain)
    np.testing.assert_array_equal(result.raw, np.full((2, 64), 0x3c800000, np.uint32))
    # A partial last tile is multiplied by its column mask only; padding never reaches the sum.
    tail = placed.call("aten.softmax.int", [f32(np.zeros((1, 33), np.uint32)), -1], {}, plain)
    assert np.all(tail.raw == tail.raw[0, 0]) and tail.raw[0, 0] not in (0, 0x3c800000)
    # The plain route has no mask: the finite sentinel is an ordinary score outside its domain.
    with pytest.raises(ValueError, match="leaves the declared bf16-signed-zero-or-normal-exp8-to132-abs-at-most32-v1 domain"):
        placed.call("aten.softmax.int", [f32([[0, 0xff7f0000]]), -1], {}, plain)
    with pytest.raises(ValueError, match="declares no semantics"):
        placed.call("aten.mul.Tensor", [f32([1]), f32([1])], {}, plain)
    with pytest.raises(ValueError, match="one selected device row-softmax route"):
        placed.call("aten.softmax.int", [f32([[0]]), -1], {}, rules(selected_routes=["source_pointwise"]))


def test_masked_softmax_evaluates_its_source_mask_add():
    masked = softmax_rules("source_masked_softmax")
    scores, mask = f32([[0x3f800000, 0x40000000, 0, 0x3f000000]]), f32([[0, 0xff7f0000, 0, 0xff7f0000]])
    combined = placed.call("aten.add.Tensor", [scores, mask], {}, masked)
    np.testing.assert_array_equal(combined.raw, [[0x3f800000, 0xff7f0000, 0, 0xff7f0000]])
    result = placed.call("aten.softmax.int", [combined, -1], {}, masked)
    assert result.raw[0, 1] == result.raw[0, 3] == 0
    assert result.raw[0, 0] > result.raw[0, 2] > 0
    # Other mask forms refuse instead of being approximated.
    for other in (0xff800000, 0x3f800000, 0x80000000):
        with pytest.raises(ValueError, match="mask operand codes"):
            placed.call("aten.add.Tensor", [scores, f32([[0, other, 0, 0]])], {}, masked)
    with pytest.raises(ValueError, match="differs from the finite sentinel"):
        placed.call("aten.add.Tensor", [f32([[0x7f000000, 0x3f800000]]), f32([[0xff7f0000, 0]])], {}, masked)
    with pytest.raises(ValueError, match="differs from the finite sentinel"):
        placed.call("aten.add.Tensor", [f32([[0xff7f0000, 0x3f800000]]), f32([[0, 0]])], {}, masked)


@pytest.mark.parametrize("dtype, sentinel, bits, width", [("f32", 0xff7f0000, np.uint32, 32), ("bf16", 0xff7f, np.uint16, 16)])
def test_masked_softmax_mask_role_comes_from_source_operand_order(dtype, sentinel, bits, width):
    masked = softmax_rules("source_masked_softmax")
    one = 0x3f800000 >> (32 - width)
    zeros = Tensor(np.zeros((2, 32), bits), dtype)
    partial = np.zeros((1, 32), bits)
    partial[:, 16:] = sentinel
    # Zero scores with a partial mask broadcast over rows: the mask is the second operand.
    combined = placed.call("aten.add.Tensor", [zeros, Tensor(partial, dtype)], {}, masked)
    np.testing.assert_array_equal(combined.raw, np.broadcast_to(partial, (2, 32)))
    result = placed.call("aten.softmax.int", [combined, -1], {}, masked)
    assert np.all(result.raw[:, 16:] == 0) and np.all(result.raw[:, :16] == result.raw[0, 0]) and result.raw[0, 0] != 0
    # Zero scores with an all-zero mask keep every key and give a nonzero uniform row.
    combined = placed.call("aten.add.Tensor", [zeros, Tensor(np.zeros((1, 32), bits), dtype)], {}, masked)
    result = placed.call("aten.softmax.int", [combined, -1], {}, masked)
    assert np.all(result.raw == result.raw[0, 0]) and result.raw[0, 0] != 0
    # Reversed operands put the mask where the route reads scores and refuse.
    scores = Tensor(np.full((2, 32), one, bits), dtype)
    with pytest.raises(ValueError, match="mask operand codes"):
        placed.call("aten.add.Tensor", [Tensor(partial, dtype), scores], {}, masked)
    with pytest.raises(ValueError, match="scores operand must have the sum's shape"):
        placed.call("aten.add.Tensor", [Tensor(partial, dtype), zeros], {}, masked)


def test_exact_f32_add_retains_source_precision_and_mask_sentinel():
    result = placed.call("aten.add.Tensor", [f32([0x3f800001, 0x3f800000]),
                          Tensor(np.array([0, 0xff7f], np.uint16), "bf16")], {},
                         rules(selected_routes=["source_exact_float_device"]))
    np.testing.assert_array_equal(result.raw, [0x3f800001, 0xff7f0000])


def test_composite_silu_exact_cast_and_device_math_routes():
    result = placed.call("aten.silu.default", [f32([0, 0x80000000])], {},
                         rules(selected_routes=["source_exact_float_device", "source_pointwise_math"]))
    np.testing.assert_array_equal(result.raw, [0, 0x80000000])


def test_rowspan80_admits_declared_scores_beyond_bounded32():
    route = softmax_rules("source_masked_softmax", ROWSPAN80)
    # A retained -32.5 score is legal relative to the zero row maximum.
    result = placed.call("aten.softmax.int", [f32([[0, 0xc2020000]]), -1], {}, route)
    assert result.raw[0, 0] == 0x3f800000
    assert result.raw[0, 1] != 0
    with pytest.raises(ValueError, match="maximum/span"):
        placed.call("aten.softmax.int", [f32([[0x42440000]]), -1], {}, route)


def test_softmax_policy_domain_and_mask_come_from_the_selected_declaration():
    scores = [f32([[0, 0x3f800000]]), -1]
    # The declared row-relative domain admits a -32.5 score that the absolute domain refuses.
    assert placed.call("aten.softmax.int", [f32([[0, 0xc2020000]]), -1], {},
                       softmax_rules("source_softmax", ROWSPAN80)).raw[0, 1] != 0
    with pytest.raises(ValueError, match="leaves the declared"):
        placed.call("aten.softmax.int", [f32([[0, 0xc2020000]]), -1], {}, softmax_rules("source_softmax"))
    # Caller target rules no longer choose the policy, and every declared field is checked.
    stale = {**softmax_rules("source_masked_softmax"), "softmax_policy": "bf16-minfinite-zero-rowspan80-nonempty-v1"}
    for refused, reason in [
            (stale, "may not declare softmax_policy"),
            (rules(selected_routes=["source_softmax"]), "requires its declared row-softmax policy, not None"),
            (softmax_rules("source_masked_softmax", policy=POLICIES["source_softmax"]),
             "requires its declared row-softmax policy"),
            (softmax_rules("source_softmax", "bf16-row-relative-span96-pad32-v1"), "unsupported declared softmax score domain"),
            (softmax_rules("source_masked_softmax", exclusion_bits=None), "declared finite exclusion bits 0xff7f0000, not None")]:
        with pytest.raises(ValueError, match=reason):
            placed.call("aten.softmax.int", scores, {}, refused)
    with pytest.raises(ValueError, match="declared finite exclusion bits"):
        placed.call("aten.add.Tensor", [f32([[0, 0]]), f32([[0, 0xff7f0000]])], {},
                    softmax_rules("source_masked_softmax", exclusion_bits=0xff800000))
