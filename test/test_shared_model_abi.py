"""Mixed host tensor arguments must be checked before native invocation."""

import numpy as np
import pytest

pytest.importorskip("npu_model")
from test.run_shared_layer import model_arrays


def test_integer_ids_boolean_masks_and_scalar_buffers(tmp_path):
    np.savez(
        tmp_path / "inputs.npz",
        arg0=np.array([2, 1], dtype=np.int64),
        arg1=np.array([True, False]),
        arg2=np.array(3, dtype=np.int64),
    )
    manifest = {
        "signature": {
            "inputs": [[2], [2], []],
            "inputs_dtypes": ["i64", "i1", "i64"],
            "outputs": [[2]],
            "outputs_dtypes": ["f32"],
        }
    }
    inputs, outputs = model_arrays(manifest, tmp_path)
    assert [v.dtype for v in inputs] == [np.int64, np.bool_, np.int64]
    assert inputs[-1].shape == ()
    assert outputs[0].dtype == np.float32
    assert np.isnan(outputs[0]).all()
    manifest["signature"]["inputs_dtypes"][0] = "f32"
    with pytest.raises(ValueError, match="ABI"):
        model_arrays(manifest, tmp_path)
    manifest["signature"]["inputs_dtypes"][0] = "i64"
    manifest["signature"]["inputs"][0] = [1, 2]
    with pytest.raises(ValueError, match="ABI"):
        model_arrays(manifest, tmp_path)
