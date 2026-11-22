"""Exporter decoding retains producer refusals without enabling a policy menu."""

from copy import deepcopy
from dataclasses import FrozenInstanceError

import pytest
from atlas_mlir_native.binding import MXU1_CAP448_POLICY
from atlas_mlir_native.export_contract import decode_export_contract
from atlas_mlir_native.numerical_contracts import MXU1_PRECISION_AUTHORIZATION
from atlas_mlir_native.policy_derivation import PolicyDerivationError


def exported():
    return {
        "schema": "atlas_torchao.execution_policy.v1",
        "format_version": 1,
        "operand": {
            "dtype": "f8E4M3",
            "exponent_bits": 4,
            "mantissa_bits": 3,
            "bias": 7,
            "exponent_range": [1, 15],
            "exponent_15_finite": True,
            "finite_max": 448.0,
            "min_normal": 0.015625,
            "reserved_codes": [127, 255],
            "subnormal_flush": True,
            "signed_zero": False,
            "rounding": "rne",
            "saturate": True,
        },
        "accumulate": {
            "unit": "MXU",
            "mxu": "ipt",
            "readout": "bf16",
            "exported_mode_accumulation": "framework",
        },
        "scale": {"encoding": "e8m0_exponent_int32"},
        "sites": {
            "projection": {
                "kind": "linear",
                "status": "quantized",
                "weight": {"codes": "codes", "scale_exp": "exponents"},
            }
        },
    }


def test_decoder_records_reviewed_mapping_and_defaults_without_legacy_capabilities():
    policy = exported()
    checked, grants = decode_export_contract(policy)
    assert grants == (MXU1_PRECISION_AUTHORIZATION,)
    record = checked.record()
    assert record["export"] == policy
    assert record["schema_defaults"]["accumulate.k_order"] == {
        "order": "ascending",
        "tile": 32,
    }
    assert record["schema_defaults"]["accumulate.units"] == {
        "mxu1": {"mxu_model": "ipt"}
    }
    assert record["schema_defaults"]["runtime.chip_options.chip_vadd"] == "rn"
    assert (
        not {"host_support", "pointwise", "row_softmax", "bf16_pointwise"}
        & record.keys()
    )
    policy["sites"]["projection"]["status"] = "changed"
    checked.sites["projection"]["status"] = "also changed"
    assert checked.sites["projection"]["status"] == "quantized"
    with pytest.raises(FrozenInstanceError):
        checked.target_identity = "changed"

    # Both existing exporter spellings retain exactly the reviewed MXU1 choice.
    exact = exported()
    exact["accumulate"].update(
        mxu=MXU1_CAP448_POLICY,
        exported_mode_accumulation="hardware",
        allowed_mxu=["mxu1"],
        units={"mxu1": {"mxu_model": "ipt"}},
        k_order={"order": "ascending", "tile": 32},
        allowed_mxu_semantics="descriptive only",
    )
    checked, exact_grants = decode_export_contract(exact)
    assert exact_grants == grants
    assert checked.allowed_mxu == ("mxu1",)
    assert "accumulate.k_order" not in checked.defaulted_fields


@pytest.mark.parametrize(
    ("section", "field", "value", "diagnostic"),
    (
        ("operand", "rounding", "chop", "operand.rounding"),
        ("operand", "reserved_codes", [127], "operand.reserved_codes"),
        ("accumulate", "unit", "VPU", "accumulate.unit"),
        ("accumulate", "readout", "f32", "accumulate.readout"),
        ("accumulate", "mxu", "sa", "accumulate.mxu"),
        ("accumulate", "allowed_mxu", ["mxu0"], "accumulate.allowed_mxu"),
        ("accumulate", "units", {"mxu1": {"mxu_model": "sa"}}, "accumulate.units"),
        (
            "accumulate",
            "k_order",
            {"order": "descending", "tile": 32},
            "accumulate.k_order",
        ),
        (
            "accumulate",
            "k_order",
            {"order": "ascending", "tile": 64},
            "accumulate.k_order",
        ),
        ("scale", "encoding", "float_scale", "scale.encoding"),
    ),
)
def test_decoder_preserves_numeric_field_refusals(
    section, field, value, diagnostic
):
    policy = exported()
    policy[section][field] = value
    with pytest.raises(PolicyDerivationError) as decoded:
        decode_export_contract(policy)
    assert decoded.value.field == diagnostic


@pytest.mark.parametrize(
    ("change", "diagnostic"),
    (
        ({"status": "skipped"}, "sites.projection.weight.dtype"),
        ({"kind": "Conv2d", "status": "host", "weight": {"dtype": "float32"}}, "sites.projection.weight.dtype"),
        ({"kind": "attention", "status": "host", "weight": {"dtype": "bfloat16"}}, "sites.projection.status"),
        ({"status": "pending"}, "sites.projection.status"),
        ({"residual_passes": 0}, "sites.projection.residual_passes"),
        ({"allowed_mxu": ["mxu0"]}, "sites.projection.allowed_mxu"),
        ({"kind": "unreviewed"}, "sites.projection.kind"),
        (
            {"kind": "conv2d", "conv": {"lowering": "direct convolution"}},
            "sites.projection.conv.lowering",
        ),
    ),
)
def test_capture_scope_ignores_unused_sites_but_preserves_used_site_refusals(
    change, diagnostic
):
    policy = exported()
    policy["sites"]["unused"] = {**deepcopy(policy["sites"]["projection"]), **change}
    checked, grants = decode_export_contract(policy, captured=({"projection"}, set()))
    assert set(checked.sites) == {"projection"} and grants == (
        MXU1_PRECISION_AUTHORIZATION,
    )
    policy["sites"]["projection"].update(change)
    with pytest.raises(PolicyDerivationError) as error:
        decode_export_contract(policy, captured=({"projection"}, set()))
    assert error.value.field == diagnostic


def test_capture_bindings_conv_prefix_and_storage_only_sites_keep_their_original_scope():
    policy = exported()
    policy["sites"]["projection"].update(
        kind="conv2d",
        conv={
            "lowering": "im2col; per-tensor cast of the unfolded input (explicit producer prefix)"
        },
    )
    checked, grants = decode_export_contract(policy, captured=(set(), {"codes"}))
    assert set(checked.sites) == {"projection"} and grants == (
        MXU1_PRECISION_AUTHORIZATION,
    )
    empty, grants = decode_export_contract(policy, captured=(set(), set()))
    assert empty.sites == {} and grants == ()
    with pytest.raises(PolicyDerivationError, match="native call names a site absent"):
        decode_export_contract(policy, captured=({"absent"}, set()))
    with pytest.raises(PolicyDerivationError, match="sites.projection.conv.lowering"):
        decode_export_contract(
            {
                **policy,
                "sites": {
                    "projection": {
                        **policy["sites"]["projection"],
                        "conv": {
                            "lowering": "im2col; per-tensor cast of the unfolded input (a(b))"
                        },
                    }
                },
            }
        )

    for site in (
        {"kind": "embedding", "status": "quantized"},
        {"kind": "linear", "status": "skipped", "weight": {"dtype": "bfloat16"}},
        {"kind": "Conv2d", "status": "host", "weight": {"tensor": "patch.weight", "dtype": "bfloat16"}},
    ):
        checked, grants = decode_export_contract(
            {**exported(), "sites": {"storage": site}}
        )
        assert checked.sites == {"storage": site} and grants == ()


def test_decoder_requires_reviewed_schema_and_retains_chip_rounding_refusal():
    policy = exported()
    with pytest.raises(PolicyDerivationError, match="policy.schema"):
        decode_export_contract({**policy, "schema": "unreviewed"})
    with pytest.raises(PolicyDerivationError, match="policy.format_version"):
        decode_export_contract({**policy, "format_version": 2})
    policy["runtime"] = {"chip_options": {"chip_vadd": "chop"}}
    with pytest.raises(PolicyDerivationError, match="runtime.chip_options.chip_vadd"):
        decode_export_contract(policy)
