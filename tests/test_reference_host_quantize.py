"""Authored host quantization shares raw encoders without device dispatch."""
import numpy as np
import pytest

from atlas_mlir_native.reference_eval import device
from atlas_mlir_native.reference_eval.evaluator import Tensor, _call


def source(dtype):
    values = np.array([[1, -2], [0.5, 4]], np.float32)
    return Tensor(values, "f32") if dtype == "f32" else Tensor(
        (values.view(np.uint32) >> 16).astype(np.uint16), "bf16")


def rules(mode):
    return {"execution": "host", "authenticated_placement": True,
            "selected_routes": ["source_" + mode + "_quantize"]}


def invoke(dtype="f32", mode="dynamic", reduce="tensor", **changes):
    options = dict(x=source(dtype), mode=mode, reduce=reduce, rule="ref",
                   static_exp=0 if mode == "static" else None, code_dtype="uint8")
    options.update(changes)
    return _call("atlas.quantize_act.default", [], options, rules(mode))


@pytest.mark.parametrize("dtype", ["f32", "bf16"])
@pytest.mark.parametrize("mode", ["static", "dynamic"])
def test_nonzero_host_quantizer_known_codes(dtype, mode, monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("host quantization called device dispatch or model admission")
    monkeypatch.setattr(device, "call", forbidden)
    monkeypatch.setattr(device, "_rules", forbidden)
    codes, exponent = invoke(dtype, mode)
    assert codes.dtype == "u8" and exponent.dtype == "i32"
    np.testing.assert_array_equal(codes.raw, [[56, 192], [48, 72]] if mode == "static" else [[104, 240], [96, 120]])
    assert exponent.raw.shape == ()
    assert exponent.raw == (0 if mode == "static" else -6)


@pytest.mark.parametrize("dtype", ["f32", "bf16"])
@pytest.mark.parametrize("reduce", ["token", "block", "head_block"])
def test_dynamic_row_groups(dtype, reduce):
    codes, exponent = invoke(dtype, reduce=reduce, rows=1)
    np.testing.assert_array_equal(exponent.raw, [[-7], [-6]])
    np.testing.assert_array_equal(codes.raw, [[112, 248], [96, 120]])


def test_head_reduction_and_nc_scale():
    _, exponent = invoke(reduce="head")
    np.testing.assert_array_equal(exponent.raw, [[-6]])
    x = Tensor(np.array([[3.75]], np.float32), "f32")
    codes, exponent = invoke(x=x, rule="nc")
    assert exponent.raw == -6
    np.testing.assert_array_equal(codes.raw, [[119]])


def test_f32_rounds_to_bf16_before_fp8():
    # This BF16 halfway value rounds down to 1.0625, then FP8 ties to 1.0.
    # Direct FP8 rounding of the F32 source would instead produce 1.125.
    x = Tensor(np.array([[1.06640625]], np.float32), "f32")
    codes, _ = invoke(mode="static", x=x)
    np.testing.assert_array_equal(codes.raw, [[56]])


def test_static_requires_fixed_supported_exponent():
    for exponent in (None, 128, -(1 << 31) - 1):
        with pytest.raises(ValueError):
            invoke(mode="static", static_exp=exponent)


@pytest.mark.parametrize("change", [
    {"authenticated_placement": False}, {"selected_routes": []},
    {"selected_routes": ["source_static_quantize"]},
    {"selected_routes": ["source_dynamic_quantize", "source_pointwise"]},
])
def test_rejects_missing_or_conflicting_route(change):
    selected = {**rules("dynamic"), **change}
    with pytest.raises(ValueError, match="authenticated selected route"):
        _call("atlas.quantize_act.default", [source("f32"), "dynamic", "tensor", "ref", None, "uint8"], {}, selected)


@pytest.mark.parametrize("change", [
    {"code_dtype": "int8"}, {"tag": 1}, {"rows": True}, {"rows": 0},
    {"rows": 1 << 31}, {"static_exp": True}, {"static_exp": 1},
    {"rule": "other"}, {"reduce": "other"}, {"x": Tensor(np.array([1], np.int32), "i32")},
    {"x": Tensor(np.array([], np.float32), "f32")},
    {"x": Tensor(np.array(1, np.float32), "f32")},
])
def test_rejects_invalid_quantizer_contract(change):
    with pytest.raises(ValueError):
        invoke(**change)


def test_host_other_atlas_op_still_refused(monkeypatch):
    monkeypatch.setattr(device, "call", lambda *a: pytest.fail("device fallback"))
    with pytest.raises(ValueError, match="no host source adapter"):
        _call("atlas.qlinear.default", [], {}, rules("dynamic"))
