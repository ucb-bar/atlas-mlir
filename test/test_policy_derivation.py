"""Capture bindings name only export-site tensors, never lifted graph constants."""

from hashlib import sha256
import json

import pytest

from atlas_mlir_native.policy_derivation import PolicyDerivationError, _capture_bindings


@pytest.mark.parametrize("constants,refused", [({"lifted": {"shape": []}}, False), ({}, True)])
def test_lifted_graph_constants_bind_no_export_site_tensor(tmp_path, constants, refused):
    raw = json.dumps({"0": {"kind": "constant", "weight": "lifted"},
                      "1": {"kind": "param", "weight": "state"}}).encode()
    (tmp_path / "weights.safetensors.manifest.json").write_bytes(raw)
    receipt = {"artifacts": {"weights.safetensors.manifest.json": {
        "bytes": len(raw), "sha256": sha256(raw).hexdigest()}}}
    quantization = {"sites": [], "atlas": {"graph_constants": constants,
                                           "buffers": {"state": {"source_name": "site.weight"}}}}
    if refused:
        with pytest.raises(PolicyDerivationError) as error:
            _capture_bindings(tmp_path, receipt, quantization)
        assert error.value.field == "capture.provenance"
    else:
        assert _capture_bindings(tmp_path, receipt, quantization) == (set(), {"site.weight"})
