"""Independent host arithmetic: source order and fail-closed math admission."""
from pathlib import Path
import json
import os

import numpy as np
import pytest

from atlas_mlir_native.reference_eval import host
from atlas_mlir_native.reference_eval.evaluator import Tensor


def f32(values):
    return Tensor(np.array(values, np.float32), "f32")


def values(tensor):
    return tensor.raw.view(np.float32)


def test_constructors_and_broadcast_source_arithmetic():
    index = host.call("aten.arange.start_step", [2, 8, 2], {})
    assert index.dtype == "i64"
    np.testing.assert_array_equal(index.raw, [2, 4, 6])
    ones = host.call("aten.ones.default", [[2, 1]], {"dtype": "torch.float32"})
    result = host.call("aten.add.Tensor", [ones, f32([[2, 3, 4]])], {"alpha": 2})
    np.testing.assert_array_equal(values(result), [[5, 7, 9], [5, 7, 9]])
    assert host.call("aten.full.default", [[1], 1.00390625], {"dtype": "bf16"}).raw[0] == 0x3f80
    with pytest.raises(ValueError, match="floating arange"):
        host.call("aten.arange.default", [2.5], {})
    with pytest.raises(ValueError, match="CPU"):
        host.call("aten.zeros.default", [[2]], {"device": "cuda"})


def test_bf16_rounding_negation_and_comparison():
    x = Tensor(np.array([0x3f80, 0x8000, 0xbf80], np.uint16), "bf16")
    y = host.call("aten.add.Scalar", [x, 0.00390625], {})
    np.testing.assert_array_equal(y.raw, [0x3f80, 0x3b80, 0xbf7f])
    np.testing.assert_array_equal(host.call("aten.neg.default", [x], {}).raw,
                                  [0xbf80, 0, 0x3f80])
    np.testing.assert_array_equal(host.call("aten.gt.Scalar", [x, 0], {}).raw,
                                  [True, False, False])
    np.testing.assert_array_equal(host.call("aten.relu.default", [x], {}).raw,
                                  [0x3f80, 0x8000, 0])
    left = Tensor(np.array([0x3f80, 0x8000, 1], np.uint16), "bf16")
    right = Tensor(np.array([0x3b80, 0x8000, 1], np.uint16), "bf16")
    np.testing.assert_array_equal(host.call("aten.add.Tensor", [left, right], {}).raw,
                                  [0x3f80, 0x8000, 2])


def test_ordered_sum_rounds_every_add_and_preserves_dimension():
    x = f32([[16777216, 1, -16777216], [1, 2, 3]])
    with pytest.raises(ValueError, match="host_order"):
        host.call("aten.sum.dim_IntList", [x, [-1], True], {})
    result = host.call("aten.sum.dim_IntList", [x, [-1], True], {},
                       {"host_order": "lexicographic-f32"})
    np.testing.assert_array_equal(values(result), [[0], [6]])
    mean = host.call("aten.mean.dim", [x, [1], False], {},
                     {"host_order": {"aten.mean.dim": "lexicographic-f32"}})
    np.testing.assert_array_equal(values(mean), [0, 2])


def test_host_matmul_refuses_device_and_uses_unfused_k_order():
    left = f32([[16777216, 1, -16777216]])
    right = f32([[1], [1], [1]])
    with pytest.raises(ValueError, match="explicitly declared host"):
        host.call("aten.matmul.default", [left, right], {})
    rules = {"execution": "host", "host_order": "lexicographic-f32"}
    result = host.call("aten.matmul.default", [left, right], {}, rules)
    np.testing.assert_array_equal(values(result), [[0]])
    linear = host.call("aten.linear.default", [f32([[2, 3]]), f32([[4, 5]]), f32([1])], {}, rules)
    np.testing.assert_array_equal(values(linear), [[24]])
    batched = host.call("aten.matmul.default", [f32([1, 2]), f32([[[1, 3], [2, 4]]])], {}, rules)
    np.testing.assert_array_equal(values(batched), [[5, 11]])
    rotary = host.call("aten.matmul.default", [f32([[[0.5], [-0.0]]]), f32([[[2, 4]]])], {})
    np.testing.assert_array_equal(rotary.raw, [[[0x3f800000, 0x40000000], [0x80000000, 0x80000000]]])
    bf_input = Tensor(np.array([[0x3f80, 0x3f80]], np.uint16), "bf16")
    bf_weight = Tensor(np.array([[0x3f80, 0x3b80]], np.uint16), "bf16")
    bf_bias = Tensor(np.array([0x3b80], np.uint16), "bf16")
    bf_linear = host.call("aten.linear.default", [bf_input, bf_weight, bf_bias], {}, rules)
    # Round matrix output to BF16 before source bias add; fusing the bias into
    # the F32 accumulator would instead produce the next BF16 code, 0x3f81.
    np.testing.assert_array_equal(bf_linear.raw, [[0x3f80]])


def test_mask_and_shape_ops_do_not_interpret_raw_float_words_as_integers():
    mask = Tensor(np.array([[True, False]], np.bool_), "bool")
    result = host.call("aten.masked_fill.Scalar", [f32([[1, 2]]), mask, -3], {})
    np.testing.assert_array_equal(values(result), [[-3, 2]])
    tri = host.call("aten.triu.default", [f32([[1, 2], [3, 4]]), 1], {})
    np.testing.assert_array_equal(values(tri), [[0, 2], [0, 0]])


def test_math_has_no_numpy_or_libm_fallback():
    with pytest.raises(ValueError, match=r"expf requires .*binding_path/build_dir"):
        host.call("aten.exp.default", [f32([0])], {})
    with pytest.raises(ValueError, match=r"erff requires .*binding_path/build_dir"):
        host.call("aten.gelu.default", [f32([0])], {})
    with pytest.raises(ValueError, match="source square exponent"):
        host.call("aten.pow.Tensor_Scalar", [f32([2]), 0.5], {})


def test_math_binding_rejects_symlinks_before_import(tmp_path):
    alias = tmp_path / "binding.py"
    alias.symlink_to(tmp_path / "missing.py")
    with pytest.raises(ValueError, match="symlink"):
        host.call("aten.sin.default", [f32([0])], {},
                  {"host_math": {"binding_path": str(alias), "build_dir": str(tmp_path)}})


def host_math_test_rules():
    rules_path = os.environ.get("ATLAS_REFERENCE_TEST_RULES")
    assert rules_path, (
        "ATLAS_REFERENCE_TEST_RULES must name the rules JSON with the "
        "merlin.runtime.host_math_supplier binding and build directory; qualification never skips"
    )
    return json.loads(Path(rules_path).read_text())


def test_host_math_rules_require_explicit_configuration(monkeypatch):
    monkeypatch.delenv("ATLAS_REFERENCE_TEST_RULES", raising=False)
    with pytest.raises(AssertionError, match="ATLAS_REFERENCE_TEST_RULES must name the rules JSON"):
        host_math_test_rules()


def test_merlin_host_math_shared_policy_real_bit_abi():
    rules = host_math_test_rules()
    assert Path(rules["host_math"]["binding_path"]).is_file(), (
        "merlin.runtime.host_math_supplier binding is required for host-math qualification"
    )
    result = host.call("aten.sin.default", [f32([0, -0.0, 1])], {}, rules)
    np.testing.assert_array_equal(result.raw, [0, 0x80000000, 0x3f576aa4])
    exp = host.call("aten.exp.default", [f32([0])], {}, rules)
    np.testing.assert_array_equal(exp.raw, [0x3f800000])
