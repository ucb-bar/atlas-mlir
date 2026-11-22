"""Generic source data motion and arithmetic domains, without framework kernels."""
import numpy as np
import pytest

from atlas_mlir_native.reference_eval import host
from atlas_mlir_native.reference_eval.evaluator import Tensor


def f32(values):
    return Tensor(np.asarray(values, np.float32), "f32")


def test_reverse_subtract_and_reciprocal_source_rounding():
    x = f32([1, 2, 4, -0.0])
    np.testing.assert_array_equal(host.call("aten.rsub.Scalar", [x, 3], {"alpha": 2}).raw.view(np.float32), [1, -1, -5, 3])
    np.testing.assert_array_equal(host.call("aten.reciprocal.default", [x], {}).raw, [0x3f800000, 0x3f000000, 0x3e800000, 0xff800000])
    with pytest.raises(ValueError, match="integer reciprocal"):
        host.call("aten.reciprocal.default", [Tensor(np.array([2], np.int64), "i64")], {})


def test_im2col_raw_padding_channel_kernel_and_block_order():
    x = f32([[[[1, 2], [3, 4]], [[5, 6], [7, 8]]]])
    out = host.call("aten.im2col.default", [x, [2, 2], [1, 1], [0, 0], [1, 1]], {})
    np.testing.assert_array_equal(out.raw.view(np.float32), [[[1], [2], [3], [4], [5], [6], [7], [8]]])
    padded = host.call("aten.im2col.default", [f32([[[1, 2], [3, 4]]]), [1, 1], [1, 1], [1, 0], [2, 1]], {})
    np.testing.assert_array_equal(padded.raw.view(np.float32), [[0, 0, 3, 4]])
    with pytest.raises(ValueError, match="pairs"):
        host.call("aten.im2col.default", [x, [2, 2], [0, 1], [0, 0], [1, 1]], {})


def test_bucketize_boundary_equality_and_flags():
    x, bounds = f32([-1, 0, 1, 2, 3]), f32([0, 2])
    np.testing.assert_array_equal(host.call("aten.bucketize.Tensor", [x, bounds], {}).raw, [0, 0, 1, 1, 2])
    out = host.call("aten.bucketize.Tensor", [x, bounds], {"right": True, "out_int32": True})
    assert out.dtype == "i32"
    np.testing.assert_array_equal(out.raw, [0, 1, 1, 2, 2])
    with pytest.raises(ValueError, match="strictly increasing"):
        host.call("aten.bucketize.Tensor", [x, f32([2, 1])], {})


def test_shape_handlers_preserve_raw_storage_and_split_tail():
    x = Tensor(np.arange(24, dtype=np.uint16).reshape(2, 3, 4), "bf16")
    flat = host.call("aten.flatten.using_ints", [x, 1, -1], {})
    assert flat.raw.shape == (2, 12)
    np.testing.assert_array_equal(flat.raw, x.raw.reshape(2, 12))
    chunks = host.call("aten.split.Tensor", [flat, 5, 1], {})
    assert [c.raw.shape for c in chunks] == [(2, 5), (2, 5), (2, 2)]
    np.testing.assert_array_equal(np.concatenate([c.raw for c in chunks], axis=1), flat.raw)
    a, b = host.call("aten.broadcast_tensors.default", [[f32([[1], [2]]), f32([3, 4, 5])]], {})
    assert a.raw.shape == b.raw.shape == (2, 3)
    expanded = host.call("aten.expand_as.default", [f32([3, 4, 5]), a], {})
    np.testing.assert_array_equal(expanded.raw, b.raw)
    assert host.call("aten.flatten.using_ints", [f32(3)], {}).raw.shape == (1,)


def test_min_returns_first_equal_index_and_preserves_negative_zero():
    x = f32([[3, -0.0, 0.0], [2, 1, 1]])
    val, idx = host.call("aten.min.dim", [x, -1, True], {})
    np.testing.assert_array_equal(val.raw, [[0x80000000], [0x3f800000]])
    np.testing.assert_array_equal(idx.raw, [[1], [1]])
    with pytest.raises(ValueError, match="NaN"):
        host.call("aten.min.dim", [f32([float("nan")]), 0], {})


def test_cumsum_explicit_order_and_integer_promotion():
    x = f32([[16777216, 1, -16777216], [1, 2, 3]])
    with pytest.raises(ValueError, match="host_order"):
        host.call("aten.cumsum.default", [x, 1], {})
    out = host.call("aten.cumsum.default", [x, 1], {}, {"host_order": "lexicographic-f32"})
    np.testing.assert_array_equal(out.raw.view(np.float32), [[16777216, 16777216, 0], [1, 3, 6]])
    out = host.call("aten.cumsum.default", [Tensor(np.array([1, 2, 3], np.int32), "i32"), 0], {})
    assert out.dtype == "i64"
    np.testing.assert_array_equal(out.raw, [1, 3, 6])


def test_source_copy_constructor_mask_and_cpu_transfer():
    x = Tensor(np.array([0x8000, 0x7fc1], np.uint16), "bf16")
    for target in ("aten.lift_fresh_copy.default", "aten.detach_.default"):
        np.testing.assert_array_equal(host.call(target, [x], {}).raw, x.raw)
    out = host.call("aten.new_ones.default", [x, [2, 1]], {})
    assert out.dtype == "bf16"
    np.testing.assert_array_equal(out.raw, [[0x3f80], [0x3f80]])
    mask = Tensor(np.array([True, False]), "bool")
    out = host.call("aten.where.ScalarOther", [mask, f32([3, 4]), -2], {})
    np.testing.assert_array_equal(out.raw.view(np.float32), [3, -2])
    out = host.call("aten.__and__.Tensor", [mask, Tensor(np.array([True, True]), "bool")], {})
    np.testing.assert_array_equal(out.raw, [True, False])
    np.testing.assert_array_equal(host.call("aten.to.device", [x, "cpu", "bf16"], {}).raw, x.raw)
    with pytest.raises(ValueError, match="CPU"):
        host.call("aten.to.device", [x, "cuda", "bf16"], {})


def test_max_pool_keeps_first_equal_skips_padding_and_refuses_nan():
    def bf16(codes, shape):
        return Tensor(np.asarray(codes, np.uint16).reshape(shape), "bf16")

    zeros = bf16([0x8000, 0x0000, 0x0000, 0x8000], (1, 1, 2, 2))
    assert host.call("aten.max_pool2d.default", [zeros, [2, 2]], {}).raw.tolist() == [[[[0x8000]]]]
    with pytest.raises(ValueError, match="NaN"):
        host.call("aten.max_pool2d.default", [bf16([0x7fc1, 0x4000, 0xffc2, 0x3f80], (1, 1, 2, 2)), [2, 2]], {})
    x = bf16([0xc000, 0xbf80, 0xc040, 0xbf00], (1, 1, 2, 2))  # all negative: -inf padding never wins
    out = host.call("aten.max_pool2d.default", [x, [3, 3], [1, 1], [1, 1]], {})
    assert out.raw.tolist() == [[[[0xbf00, 0xbf00], [0xbf00, 0xbf00]]]]
    dilated = host.call("aten.max_pool2d.default", [f32(np.arange(16).reshape(1, 1, 4, 4)), [2, 2], [1, 1], [0, 0], [2, 2]], {})
    np.testing.assert_array_equal(dilated.raw.view(np.float32), [[[[10, 11], [14, 15]]]])
    with pytest.raises(ValueError, match="ceil_mode"):
        host.call("aten.max_pool2d.default", [x, [2, 2], [2, 2], [0, 0], [1, 1], True], {})


def test_batch_norm_inference_uses_fused_binary32_steps():
    def bf16(codes, shape=None):
        array = np.asarray(codes, np.uint16)
        return Tensor(array.reshape(shape) if shape else array, "bf16")

    x = bf16([0x4040, 0x3f80, 0x0000, 0x8000], (1, 2, 1, 2))
    weight, bias, mean, var = bf16([0x4000, 0x3f80]), bf16([0x3f00, 0x8000]), bf16([0x3f80, 0x0000]), bf16([0x4080, 0x3f80])
    out = host.call("aten.batch_norm.default", [x, weight, bias, mean, var, False, 0.1, 0.0, False], {})
    # Channel 0: alpha = 2 / 2 = 1, beta = 0.5 - 1 = -0.5. Channel 1: alpha = 1, beta = -0 - 0 * 1 = -0.
    assert out.raw.tolist() == [[[[0x4020, 0x3f00]], [[0x0000, 0x8000]]]]
    with pytest.raises(ValueError, match="non-finite"):
        host.call("aten.batch_norm.default", [bf16([0x7f80, 0, 0, 0], (1, 2, 1, 2)), weight, bias, mean, var, False, 0.1, 1e-5, False], {})
    with pytest.raises(ValueError, match="inference"):
        host.call("aten.batch_norm.default", [x, weight, bias, mean, var, True, 0.1, 1e-5, False], {})


def _bf16(codes):
    return Tensor(np.asarray(codes, np.uint16), "bf16")


def test_bf16_sigmoid_uses_binary32_opmath_and_host_expf(tmp_path):
    import subprocess
    import sys
    from pathlib import Path

    from merlin.runtime import host_math_supplier as host_math

    root = Path(host_math.__file__).resolve().parent
    subprocess.run([sys.executable, str(root / "build.py"), "--target", "x86", "--out", str(tmp_path)],
                   check=True, capture_output=True)
    rules = {"host_math": {"binding_path": str(root / "__init__.py"), "build_dir": str(tmp_path / "x86")}}
    x = _bf16([0x0000, 0x8000, 0x3f80, 0xc0a0, 0x7f80, 0xff80, 0x0001, 0x42c8])
    out = host.call("aten.sigmoid.default", [x], {}, rules)
    assert out.dtype == "bf16"
    assert out.raw.tolist() == [0x3f00, 0x3f00, 0x3f3b, 0x3bdb, 0x3f80, 0x0000, 0x3f00, 0x3f80]
    with pytest.raises(ValueError, match="NaN"):
        host.call("aten.sigmoid.default", [_bf16([0x3f80, 0x7fc0])], {}, rules)
    with pytest.raises(ValueError, match="BF16 only"):
        host.call("aten.sigmoid.default", [f32([0])], {}, rules)


def test_inference_dropout_is_identity_and_training_dropout_refuses():
    x = _bf16([0x3f80, 0x8000, 0x7fc1])
    assert host.call("aten.dropout.default", [x, 0.5, False], {}).raw.tolist() == [0x3f80, 0x8000, 0x7fc1]
    with pytest.raises(ValueError, match="training"):
        host.call("aten.dropout.default", [x, 0.5, True], {})
