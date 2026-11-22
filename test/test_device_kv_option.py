"""Device KV planning is explicit and preserves the existing default route."""
from dataclasses import replace
from unittest.mock import patch

import pytest
from atlas_mlir_native import activation_quantize_application as quantizer
from atlas_mlir_native import activation_quantize_device
from atlas_mlir_native.layout_device import prepare_source, virtual_program
from atlas_mlir_native.options import codegen_options, current_codegen_options
from test_activation_quantize_device import dynamic_source
from test_layout_device import cases


def test_dynamic_default_uses_existing_group_geometry_without_proof():
    source = dynamic_source(shape=(1, 65, 64), reduce="head")
    with codegen_options(replace(current_codegen_options(), device_kv_quantizer=False)), \
            patch.object(activation_quantize_device, "prove_plan") as proof:
        with pytest.raises(ValueError, match="1024-element tiles"):
            quantizer.prepare_plan(source)
        ordinary = quantizer.prepare_plan(dynamic_source(shape=(2, 64, 64), reduce="head"))
        assert "dynamic_group_admission" not in ordinary.record()
        proof.assert_not_called()
    with codegen_options(replace(current_codegen_options(), device_kv_quantizer=True)), \
            patch.object(activation_quantize_device, "prove_plan") as proof:
        selected = quantizer.prepare_plan(source)
        assert selected.launches[0]["code_bytes"] == 65 * 64
        assert "dynamic_group_admission" in selected.record()
        assert proof.call_count == 1


def test_ragged_transpose_default_keeps_scalar_program():
    source = cases()["ragged_xlu"]
    with codegen_options(replace(current_codegen_options(), device_kv_quantizer=False)):
        default = prepare_source(source)
        assert [segment["mode"] for segment in default.segments] == ["scalar"]
        assert 'kind = "raw"' not in virtual_program(default)
    with codegen_options(replace(current_codegen_options(), device_kv_quantizer=True)):
        selected = prepare_source(source)
        assert [segment["mode"] for segment in selected.segments] == ["ragged_xlu"]
        assert 'kind = "raw"' in virtual_program(selected)
