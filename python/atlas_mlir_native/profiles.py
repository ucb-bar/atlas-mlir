"""Construct the existing selected Atlas resource and finite numerical profiles.

Compiler-side admission/replay retains its diagnostic scope. Independent reference
and evaluator authority remain separate; no hardware execution is established.
"""

from __future__ import annotations

from dataclasses import replace

from merlin.compiler.semantic.allocate import (
    StorageBank,
)
from merlin.compiler.semantic.model import (
    pointwise_identity_maps,
)
from merlin.compiler.quantization.contraction.transform import (
    MatmulTileProfile,
)
from merlin.compiler.quantization.contraction.values import FloatBoundarySpec
from merlin.compiler.semantic.rules import (
    AddressConstraint,
    AxisBound,
    InstructionDescriptor,
    IntegerAttributeBound,
)
from merlin.compiler.semantic.snapshot import NativeTargetProfile

from .admission import (
    _mxu0_accumulate_maps,
    _mxu0_maps,
    _row_reduce_maps,
    _transpose_maps,
)
from .pack_lane import ROUNDED_PACK_DOMAIN, ROUNDED_PACK_POLICY
from .policy import (
    BF16_ADD_ATTRS,
    BF16_DRAM_PAIR_LAYOUT,
    BF16_EXP_ATTRS,
    BF16_MUL_ATTRS,
    BF16_RECIP_ATTRS,
    BF16_RELU_ATTRS,
    BF16_ROW_MAX_ATTRS,
    BF16_ROW_SUM_ATTRS,
    BF16_SUB_ATTRS,
    BF16_TENSOR_PAIR_LAYOUT,
    DRAM_SLOTS,
    FINITE_ANCHOR_POLICY,
    FP8_DRAM_TILE_LAYOUT,
    FP8_TENSOR_TILE_LAYOUT,
    MXU0_ACC_TILE_LAYOUT,
    MXU0_INPUT_DOMAIN,
    MXU0_POLICY,
    MXU1_FINITE_INPUT_DOMAIN,
    MXU1_FINITE_POLICY,
    MXU1_INPUT_DOMAIN,
    MXU1_POLICY,
    PACK_ATTRS,
    PACK_POLICY,
    RTL_REVISION,
    SCALED_PACK_DOMAIN,
    SCALED_PACK_POLICY,
    SCALED_PACK_RANGE,
    TARGET_IDENTITY,
)


def diagnostic_mxu0_f32_profile() -> MatmulTileProfile:
    """Describe the source-bound f32-to-MXU0 tile as an explicit approximation."""
    operand_conversion = FloatBoundarySpec(
        "f32-to-e4m3-finite-normal-rne-flush-subnormal", "fp8_e4m3", "encode",
        max_normal_exponent_code=14, flush_below_normal=True,
        admitted_code_ranges=((0, 0), (8, 119), (136, 247)),
    )
    return MatmulTileProfile(
        TARGET_IDENTITY, (32, 32, 32), _mxu0_maps(),
        ("fp8_e4m3", "fp8_e4m3"), ("raw-bit-pattern", "raw-bit-pattern"),
        (MXU0_INPUT_DOMAIN, MXU0_INPUT_DOMAIN),
        ("f32-to-e4m3-finite-normal-rne-flush-subnormal",) * 2,
        "bf16", MXU0_POLICY, "bf16-to-f32-exact", "external", (1, 1),
        (
            ("schema", "atlas.oot.quantized_linalg_policy.v3"),
            ("selected_rtl_revision", RTL_REVISION),
            ("source_dtype", "f32"),
            ("operand_encoding", "e4m3_finite_normal_rne_flush_subnormal"),
            ("bias_encoding", "bf16_rne"),
            ("vpu_add", "selected_f32_sum_bf16_chop"),
            ("precision_transform", "explicit_diagnostic_candidate"),
        ),
        (("mxu_unit", 0), ("activation_pack_scale_code", 127)),
        ("softmax_realization",),
        (operand_conversion, operand_conversion, FloatBoundarySpec("bf16-to-f32-exact", "bf16", "decode")),
    )


def diagnostic_mxu1_f32_profile() -> MatmulTileProfile:
    """Keep MXU1's anchor-tree arithmetic distinct from ordered MXU0."""
    base = diagnostic_mxu0_f32_profile()
    operand_conversion = FloatBoundarySpec(
        "f32-to-e4m3-finite-normal-rne-flush-subnormal", "fp8_e4m3", "encode",
        max_normal_exponent_code=14, flush_below_normal=True,
        admitted_code_ranges=((0, 0), (56, 56), (64, 64), (184, 184), (192, 192)),
    )
    return MatmulTileProfile(
        TARGET_IDENTITY, base.loop_shape, base.maps,
        base.operand_dtypes, base.operand_policies,
        (MXU1_INPUT_DOMAIN, MXU1_INPUT_DOMAIN),
        base.operand_conversion_policies,
        base.result_dtype, MXU1_POLICY, base.result_conversion_policy,
        base.storage, base.input_extents, base.required_policy_fields,
        (("mxu_unit", 1), ("activation_pack_scale_code", 127)),
        base.optional_policy_fields,
        (operand_conversion, operand_conversion, base.conversion_specs[2]),
    )


def diagnostic_mxu0_scaled_f32_profile(scale_code: int) -> MatmulTileProfile:
    """Quantize one source-bound tile with inverse power-of-two operand scales.

    The emitted MXU consumes already encoded FP8 operands. This boundary
    policy records the selected pack-equivalent activation scale and the
    compensating constant-weight scale; it does not insert a VPU pack.
    """
    if type(scale_code) is not int or scale_code not in range(SCALED_PACK_RANGE[0], SCALED_PACK_RANGE[1] + 1):
        raise ValueError("source-bound MXU0 scale code is outside the selected pack range")
    base = diagnostic_mxu0_f32_profile()
    activation_shift = 127 - scale_code
    policies = (
        f"f32-to-e4m3-activation-e8m0-{scale_code}-v1",
        f"f32-to-e4m3-weight-compensation-e8m0-{scale_code}-v1",
    )
    conversions = tuple(
        FloatBoundarySpec(
            policy, "fp8_e4m3", "encode", max_normal_exponent_code=14,
            flush_below_normal=True, admitted_code_ranges=base.conversion_specs[0].admitted_code_ranges,
            pre_encode_scale_exponent=shift,
        )
        for policy, shift in zip(policies, (activation_shift, -activation_shift), strict=True)
    )
    return MatmulTileProfile(
        base.target_identity, base.loop_shape, base.maps,
        base.operand_dtypes, base.operand_policies, base.operand_domains,
        policies, base.result_dtype, base.result_policy,
        base.result_conversion_policy, base.storage, base.input_extents,
        base.required_policy_fields,
        (("mxu_unit", 0), ("activation_pack_scale_code", scale_code)),
        base.optional_policy_fields,
        (*conversions, base.conversion_specs[2]),
    )


def _finite_anchor_policy_profile(profile: MatmulTileProfile) -> MatmulTileProfile:
    """Bind one diagnostic tile to a separately identified precision policy."""
    fields = tuple(
        (key, "atlas.oot.quantized_linalg_policy.v4" if key == "schema" else value)
        for key, value in profile.required_policy_fields
    )
    return replace(profile, required_policy_fields=(
        *fields, ("mxu1_reset_arithmetic", FINITE_ANCHOR_POLICY),
    ))


def _canonicalized_zero_boundary_profile(profile: MatmulTileProfile, schema: str) -> MatmulTileProfile:
    """Keep an explicit policy identity for the lossy FP8 zero-sign choice."""
    if profile.conversion_specs is None:
        raise ValueError("positive-zero boundary needs executable operand conversions")
    inputs = tuple(replace(spec, policy=f"{spec.policy}-positive-zero",
                           canonicalize_negative_zero=True)
                   for spec in profile.conversion_specs[:2])
    required = tuple(
        (key, schema if key == "schema" else value)
        for key, value in profile.required_policy_fields
    )
    return replace(
        profile,
        operand_conversion_policies=tuple(spec.policy for spec in inputs),
        conversion_specs=(*inputs, profile.conversion_specs[2]),
        required_policy_fields=(*required, ("zero_sign_policy", "canonicalize_encoded_negative_zero")),
    )


def _positive_zero_boundary_profile(profile: MatmulTileProfile) -> MatmulTileProfile:
    if dict(profile.selected_decision).get("mxu_unit") != 0:
        raise ValueError("positive-zero boundary currently needs MXU0")
    return _canonicalized_zero_boundary_profile(profile, "atlas.oot.quantized_linalg_policy.v5")


def _mixed_positive_zero_boundary_profile(profile: MatmulTileProfile) -> MatmulTileProfile:
    if (
        dict(profile.selected_decision).get("mxu_unit") not in (0, 1)
        or dict(profile.required_policy_fields).get("mxu1_reset_arithmetic") != FINITE_ANCHOR_POLICY
    ):
        raise ValueError("mixed positive-zero boundary needs the finite-anchor MXU profile")
    return _canonicalized_zero_boundary_profile(profile, "atlas.oot.quantized_linalg_policy.v6")


def diagnostic_mxu1_finite_f32_profile(scale_code: int) -> MatmulTileProfile:
    """Describe source-derived finite-normal MXU1 reset in diagnostic mode.

    The policy is separate from the five-code MXU1 profile. It permits
    finite-normal E4M3 encodings and positive zero after the selected f32
    conversion, with the same inverse power-of-two weight compensation used
    by the explicit MXU0 scale policy.
    """
    if type(scale_code) is not int or not SCALED_PACK_RANGE[0] <= scale_code <= SCALED_PACK_RANGE[1]:
        raise ValueError("finite MXU1 scale code is outside the selected range")
    base = (diagnostic_mxu0_f32_profile() if scale_code == 127
            else diagnostic_mxu0_scaled_f32_profile(scale_code))
    return _finite_anchor_policy_profile(replace(
        base,
        operand_domains=(MXU1_FINITE_INPUT_DOMAIN, MXU1_FINITE_INPUT_DOMAIN),
        result_policy=MXU1_FINITE_POLICY,
        selected_decision=(("mxu_unit", 1), ("activation_pack_scale_code", scale_code)),
    ))


def profile() -> NativeTargetProfile:
    def signature(dtype: str) -> dict:
        return {
            "output_dtype": dtype, "numerical_policy": "raw-bit-pattern", "ranks": (2,),
            "input_dtypes": (dtype,), "input_numerical_policies": ("raw-bit-pattern",),
            "input_ranks": (2,),
            "output_axis_bounds": (AxisBound(0, 32, 32), AxisBound(1, 32, 32)),
            "input_axis_bounds": ((AxisBound(0, 32, 32), AxisBound(1, 32, 32)),),
        }

    descriptors = (
        InstructionDescriptor("atlas.load_bf16_tile", "identity", ("external",),
                              "tensor_bf16", extent=2, value_preserving_copy=True,
                              **signature("bf16")),
        InstructionDescriptor("atlas.load_bf16_tile_mixed", "identity", ("external",),
                              "tensor_bf16", extent=2, value_preserving_copy=True,
                              source_required_attrs=(("input_domain", PACK_ATTRS[0][1]),),
                              validity=(AddressConstraint("at_least", "out", value=32),
                                        AddressConstraint("at_most", "out", value=32)),
                              **signature("bf16")),
        InstructionDescriptor("atlas.vpu_relu_bf16_tile", "relu", ("tensor_bf16",),
                              "tensor_bf16", extent=2, required_attrs=BF16_RELU_ATTRS,
                              index_maps=pointwise_identity_maps(2, 1),
                              input_read_offsets=(32,), completion_offset=65,
                              **signature("bf16")),
        InstructionDescriptor("atlas.vpu_exp_bf16_tile", "exp_bf16", ("tensor_bf16",),
                              "tensor_bf16", extent=2, required_attrs=BF16_EXP_ATTRS,
                              index_maps=pointwise_identity_maps(2, 1),
                              input_read_offsets=(32,), completion_offset=65,
                              input_policy_alternatives=((MXU0_POLICY,),),
                              **signature("bf16")),
        InstructionDescriptor("atlas.vpu_recip_bf16_tile", "recip_bf16", ("tensor_bf16",),
                              "tensor_bf16", extent=2, required_attrs=BF16_RECIP_ATTRS,
                              index_maps=pointwise_identity_maps(2, 1),
                              input_read_offsets=(32,), completion_offset=65,
                              input_policy_alternatives=((MXU0_POLICY,),),
                              **signature("bf16")),
        InstructionDescriptor(
            "atlas.vpu_row_max_bf16_tile", "row_max_bf16", ("tensor_bf16",),
            "tensor_bf16", extent=2, required_attrs=BF16_ROW_MAX_ATTRS,
            index_maps=_row_reduce_maps(), input_read_offsets=(32,), completion_offset=65,
            input_policy_alternatives=((MXU0_POLICY,),), **signature("bf16"),
        ),
        InstructionDescriptor(
            "atlas.vpu_row_sum_bf16_tile", "row_sum_bf16", ("tensor_bf16",),
            "tensor_bf16", extent=2, required_attrs=BF16_ROW_SUM_ATTRS,
            index_maps=_row_reduce_maps(), input_read_offsets=(32,), completion_offset=65,
            input_policy_alternatives=((MXU0_POLICY,),), **signature("bf16"),
        ),
        InstructionDescriptor(
            "atlas.vpu_add_bf16_tile", "add_bf16", ("tensor_bf16", "tensor_bf16"),
            "tensor_bf16", "bf16", "raw-bit-pattern", (2,), extent=2,
            input_dtypes=("bf16", "bf16"),
            input_numerical_policies=("raw-bit-pattern", "raw-bit-pattern"),
            input_policy_alternatives=((MXU0_POLICY,), (MXU0_POLICY,)),
            input_ranks=(2, 2), required_attrs=BF16_ADD_ATTRS,
            index_maps=pointwise_identity_maps(2, 2),
            input_read_offsets=(32, 32), completion_offset=65,
            output_axis_bounds=(AxisBound(0, 32, 32), AxisBound(1, 32, 32)),
            input_axis_bounds=(
                (AxisBound(0, 32, 32), AxisBound(1, 32, 32)),
                (AxisBound(0, 32, 32), AxisBound(1, 32, 32)),
            ),
        ),
        InstructionDescriptor(
            "atlas.vpu_sub_bf16_tile", "sub_bf16", ("tensor_bf16", "tensor_bf16"),
            "tensor_bf16", "bf16", "raw-bit-pattern", (2,), extent=2,
            input_dtypes=("bf16", "bf16"),
            input_numerical_policies=("raw-bit-pattern", "raw-bit-pattern"),
            input_policy_alternatives=((MXU0_POLICY,), (MXU0_POLICY,)),
            input_ranks=(2, 2), required_attrs=BF16_SUB_ATTRS,
            index_maps=pointwise_identity_maps(2, 2),
            input_read_offsets=(32, 32), completion_offset=65,
            output_axis_bounds=(AxisBound(0, 32, 32), AxisBound(1, 32, 32)),
            input_axis_bounds=(
                (AxisBound(0, 32, 32), AxisBound(1, 32, 32)),
                (AxisBound(0, 32, 32), AxisBound(1, 32, 32)),
            ),
        ),
        InstructionDescriptor(
            "atlas.vpu_mul_bf16_tile", "mul_bf16", ("tensor_bf16", "tensor_bf16"),
            "tensor_bf16", "bf16", "raw-bit-pattern", (2,), extent=2,
            input_dtypes=("bf16", "bf16"),
            input_numerical_policies=("raw-bit-pattern", "raw-bit-pattern"),
            input_policy_alternatives=((MXU0_POLICY,), (MXU0_POLICY,)),
            input_ranks=(2, 2), required_attrs=BF16_MUL_ATTRS,
            index_maps=pointwise_identity_maps(2, 2),
            input_read_offsets=(32, 32), completion_offset=65,
            output_axis_bounds=(AxisBound(0, 32, 32), AxisBound(1, 32, 32)),
            input_axis_bounds=(
                (AxisBound(0, 32, 32), AxisBound(1, 32, 32)),
                (AxisBound(0, 32, 32), AxisBound(1, 32, 32)),
            ),
        ),
        InstructionDescriptor(
            "atlas.vpu_relu_bf16_from_mxu0_tile", "relu", ("tensor_bf16",),
            "tensor_bf16", "bf16", "raw-bit-pattern", (2,), extent=2,
            input_dtypes=("bf16",), input_numerical_policies=(MXU0_POLICY,),
            input_ranks=(2,), required_attrs=BF16_RELU_ATTRS,
            index_maps=pointwise_identity_maps(2, 1),
            input_read_offsets=(32,), completion_offset=65,
            output_axis_bounds=(AxisBound(0, 32, 32), AxisBound(1, 32, 32)),
            input_axis_bounds=((AxisBound(0, 32, 32), AxisBound(1, 32, 32)),),
        ),
        InstructionDescriptor("atlas.store_bf16_tile", "identity", ("tensor_bf16",),
                              "external", value_preserving_copy=True,
                              **signature("bf16")),
        InstructionDescriptor("atlas.load_fp8_tile", "identity", ("external",),
                              "tensor_fp8", value_preserving_copy=True,
                              input_layouts=(FP8_DRAM_TILE_LAYOUT,),
                              output_layout=FP8_TENSOR_TILE_LAYOUT,
                              **signature("fp8_e4m3")),
        InstructionDescriptor("atlas.xlu_transpose_fp8", "transpose", ("tensor_fp8",),
                              "tensor_fp8", index_maps=_transpose_maps(),
                              input_layouts=(FP8_TENSOR_TILE_LAYOUT,),
                              output_layout=FP8_TENSOR_TILE_LAYOUT,
                              **signature("fp8_e4m3")),
        InstructionDescriptor(
            "atlas.xlu_transpose_unit_fp8_tile", "transpose", ("tensor_fp8",),
            "tensor_fp8", "fp8_e4m3", PACK_POLICY, (2,),
            input_dtypes=("fp8_e4m3",), input_numerical_policies=(PACK_POLICY,),
            input_ranks=(2,), index_maps=_transpose_maps(),
            input_read_offsets=(32,), completion_offset=65,
            output_axis_bounds=(AxisBound(0, 32, 32), AxisBound(1, 32, 32)),
            input_axis_bounds=((AxisBound(0, 32, 32), AxisBound(1, 32, 32)),),
        ),
        InstructionDescriptor(
            "atlas.xlu_transpose_scaled_fp8_tile", "transpose", ("tensor_fp8",),
            "tensor_fp8", "fp8_e4m3", SCALED_PACK_POLICY, (2,),
            input_dtypes=("fp8_e4m3",), input_numerical_policies=(SCALED_PACK_POLICY,),
            input_ranks=(2,), index_maps=_transpose_maps(),
            input_read_offsets=(32,), completion_offset=65,
            output_axis_bounds=(AxisBound(0, 32, 32), AxisBound(1, 32, 32)),
            input_axis_bounds=((AxisBound(0, 32, 32), AxisBound(1, 32, 32)),),
        ),
        InstructionDescriptor(
            "atlas.xlu_transpose_rounded_fp8_tile", "transpose", ("tensor_fp8",),
            "tensor_fp8", "fp8_e4m3", ROUNDED_PACK_POLICY, (2,),
            input_dtypes=("fp8_e4m3",), input_numerical_policies=(ROUNDED_PACK_POLICY,),
            input_ranks=(2,), index_maps=_transpose_maps(),
            input_read_offsets=(32,), completion_offset=65,
            output_axis_bounds=(AxisBound(0, 32, 32), AxisBound(1, 32, 32)),
            input_axis_bounds=((AxisBound(0, 32, 32), AxisBound(1, 32, 32)),),
        ),
        InstructionDescriptor("atlas.store_fp8_tile", "identity", ("tensor_fp8",),
                              "external", value_preserving_copy=True,
                              input_layouts=(FP8_TENSOR_TILE_LAYOUT,),
                              output_layout=FP8_DRAM_TILE_LAYOUT,
                              **signature("fp8_e4m3")),
        InstructionDescriptor(
            "atlas.vpu_pack_unit_fp8_tile", "pack_fp8", ("tensor_bf16",),
            "tensor_fp8", "fp8_e4m3", PACK_POLICY, (2,),
            input_dtypes=("bf16",), input_numerical_policies=("raw-bit-pattern",),
            input_policy_alternatives=((MXU0_POLICY,),),
            input_ranks=(2,), required_attrs=PACK_ATTRS,
            index_maps=pointwise_identity_maps(2, 1),
            input_read_offsets=(34,), completion_offset=65,
            validity=(AddressConstraint("at_least", "in0", value=32),
                      AddressConstraint("at_most", "in0", value=60)),
            output_axis_bounds=(AxisBound(0, 32, 32), AxisBound(1, 32, 32)),
            input_axis_bounds=((AxisBound(0, 32, 32), AxisBound(1, 32, 32)),),
        ),
        InstructionDescriptor(
            "atlas.store_fp8_unit_tile", "identity", ("tensor_fp8",),
            "external", "fp8_e4m3", PACK_POLICY, (2,),
            input_dtypes=("fp8_e4m3",), input_numerical_policies=(PACK_POLICY,),
            input_ranks=(2,), value_preserving_copy=True,
            output_axis_bounds=(AxisBound(0, 32, 32), AxisBound(1, 32, 32)),
            input_axis_bounds=((AxisBound(0, 32, 32), AxisBound(1, 32, 32)),),
        ),
        InstructionDescriptor(
            "atlas.vpu_pack_scaled_fp8_tile", "pack_fp8", ("tensor_bf16",),
            "tensor_fp8", "fp8_e4m3", SCALED_PACK_POLICY, (2,),
            input_dtypes=("bf16",), input_numerical_policies=("raw-bit-pattern",),
            input_policy_alternatives=((MXU0_POLICY,),),
            input_ranks=(2,), required_attrs=(("input_domain", SCALED_PACK_DOMAIN),),
            parameter_attrs=(IntegerAttributeBound("scale_code", *SCALED_PACK_RANGE),),
            index_maps=pointwise_identity_maps(2, 1),
            input_read_offsets=(34,), completion_offset=65,
            validity=(AddressConstraint("at_least", "in0", value=32),
                      AddressConstraint("at_most", "in0", value=60)),
            output_axis_bounds=(AxisBound(0, 32, 32), AxisBound(1, 32, 32)),
            input_axis_bounds=((AxisBound(0, 32, 32), AxisBound(1, 32, 32)),),
        ),
        InstructionDescriptor(
            "atlas.vpu_pack_rounded_fp8_tile", "pack_fp8", ("tensor_bf16",),
            "tensor_fp8", "fp8_e4m3", ROUNDED_PACK_POLICY, (2,),
            input_dtypes=("bf16",), input_numerical_policies=("raw-bit-pattern",),
            input_policy_alternatives=((MXU0_POLICY,),),
            input_ranks=(2,), required_attrs=(("input_domain", ROUNDED_PACK_DOMAIN),),
            parameter_attrs=(IntegerAttributeBound("scale_code", *SCALED_PACK_RANGE),),
            index_maps=pointwise_identity_maps(2, 1),
            input_read_offsets=(34,), completion_offset=65,
            validity=(AddressConstraint("at_least", "in0", value=32),
                      AddressConstraint("at_most", "in0", value=60)),
            output_axis_bounds=(AxisBound(0, 32, 32), AxisBound(1, 32, 32)),
            input_axis_bounds=((AxisBound(0, 32, 32), AxisBound(1, 32, 32)),),
        ),
        InstructionDescriptor(
            "atlas.store_fp8_scaled_tile", "identity", ("tensor_fp8",),
            "external", "fp8_e4m3", SCALED_PACK_POLICY, (2,),
            input_dtypes=("fp8_e4m3",), input_numerical_policies=(SCALED_PACK_POLICY,),
            input_ranks=(2,), value_preserving_copy=True,
            output_axis_bounds=(AxisBound(0, 32, 32), AxisBound(1, 32, 32)),
            input_axis_bounds=((AxisBound(0, 32, 32), AxisBound(1, 32, 32)),),
        ),
        InstructionDescriptor(
            "atlas.store_fp8_rounded_tile", "identity", ("tensor_fp8",),
            "external", "fp8_e4m3", ROUNDED_PACK_POLICY, (2,),
            input_dtypes=("fp8_e4m3",), input_numerical_policies=(ROUNDED_PACK_POLICY,),
            input_ranks=(2,), value_preserving_copy=True,
            output_axis_bounds=(AxisBound(0, 32, 32), AxisBound(1, 32, 32)),
            input_axis_bounds=((AxisBound(0, 32, 32), AxisBound(1, 32, 32)),),
        ),
        InstructionDescriptor(
            "atlas.mxu0_reset_tile", "matmul", ("tensor_fp8", "tensor_fp8"),
            "tensor_bf16", "bf16", MXU0_POLICY, (2,), extent=2,
            input_layouts=(FP8_TENSOR_TILE_LAYOUT, FP8_TENSOR_TILE_LAYOUT),
            output_layout=BF16_TENSOR_PAIR_LAYOUT,
            input_dtypes=("fp8_e4m3", "fp8_e4m3"),
            input_numerical_policies=("raw-bit-pattern", "raw-bit-pattern"),
            input_policy_alternatives=((PACK_POLICY, SCALED_PACK_POLICY, ROUNDED_PACK_POLICY),
                                       (PACK_POLICY, SCALED_PACK_POLICY, ROUNDED_PACK_POLICY)),
            input_ranks=(2, 2),
            output_axis_bounds=(AxisBound(0, 32, 32), AxisBound(1, 32, 32)),
            input_axis_bounds=(
                (AxisBound(0, 32, 32), AxisBound(1, 32, 32)),
                (AxisBound(0, 32, 32), AxisBound(1, 32, 32)),
            ),
            shape_contract="bounded", index_maps=_mxu0_maps(),
            # The OOT allocator may choose any even BF16 pair in the
            # shared register file after considering later VPU uses.
            validity=(AddressConstraint("at_least", "out", value=32),
                      AddressConstraint("at_most", "out", value=60)),
        ),
        InstructionDescriptor(
            "atlas.mxu0_reset_acc_tile", "matmul", ("tensor_fp8", "tensor_fp8"),
            "tensor_mxu0_acc", "bf16", MXU0_POLICY, (2,),
            input_layouts=(FP8_TENSOR_TILE_LAYOUT, FP8_TENSOR_TILE_LAYOUT),
            output_layout=MXU0_ACC_TILE_LAYOUT,
            input_dtypes=("fp8_e4m3", "fp8_e4m3"),
            input_numerical_policies=("raw-bit-pattern", "raw-bit-pattern"),
            input_ranks=(2, 2),
            output_axis_bounds=(AxisBound(0, 32, 32), AxisBound(1, 32, 32)),
            input_axis_bounds=(
                (AxisBound(0, 32, 32), AxisBound(1, 32, 32)),
                (AxisBound(0, 32, 32), AxisBound(1, 32, 32)),
            ),
            shape_contract="bounded", index_maps=_mxu0_maps(),
            input_read_offsets=(32, 32), completion_offset=96,
        ),
        InstructionDescriptor(
            "atlas.mxu0_accumulate_tile", "matmul_accumulate",
            ("tensor_fp8", "tensor_fp8", "tensor_mxu0_acc"),
            "tensor_mxu0_acc", "bf16", MXU0_POLICY, (2,),
            input_layouts=(FP8_TENSOR_TILE_LAYOUT, FP8_TENSOR_TILE_LAYOUT,
                           MXU0_ACC_TILE_LAYOUT),
            output_layout=MXU0_ACC_TILE_LAYOUT,
            input_dtypes=("fp8_e4m3", "fp8_e4m3", "bf16"),
            input_numerical_policies=("raw-bit-pattern", "raw-bit-pattern", MXU0_POLICY),
            input_ranks=(2, 2, 2),
            output_axis_bounds=(AxisBound(0, 32, 32), AxisBound(1, 32, 32)),
            input_axis_bounds=(
                (AxisBound(0, 32, 32), AxisBound(1, 32, 32)),
                (AxisBound(0, 32, 32), AxisBound(1, 32, 32)),
                (AxisBound(0, 32, 32), AxisBound(1, 32, 32)),
            ),
            shape_contract="bounded", index_maps=_mxu0_accumulate_maps(),
            validity=(AddressConstraint("eq_offset", "out", "in2", 0),),
            input_read_offsets=(32, 32, 94), completion_offset=96,
            in_place_inputs=(2,),
        ),
        InstructionDescriptor(
            "atlas.store_bf16_mxu0_acc_tile", "identity", ("tensor_mxu0_acc",),
            "external", "bf16", MXU0_POLICY, (2,),
            input_layouts=(MXU0_ACC_TILE_LAYOUT,),
            output_layout=BF16_DRAM_PAIR_LAYOUT,
            input_dtypes=("bf16",), input_numerical_policies=(MXU0_POLICY,),
            input_ranks=(2,), value_preserving_copy=True,
            output_axis_bounds=(AxisBound(0, 32, 32), AxisBound(1, 32, 32)),
            input_axis_bounds=((AxisBound(0, 32, 32), AxisBound(1, 32, 32)),),
            input_read_offsets=(32,), completion_offset=64,
        ),
        InstructionDescriptor(
            "atlas.store_bf16_mxu0_tile", "identity", ("tensor_bf16",),
            "external", "bf16", MXU0_POLICY, (2,),
            input_layouts=(BF16_TENSOR_PAIR_LAYOUT,),
            output_layout=BF16_DRAM_PAIR_LAYOUT,
            input_dtypes=("bf16",), input_numerical_policies=(MXU0_POLICY,),
            input_ranks=(2,),
            output_axis_bounds=(AxisBound(0, 32, 32), AxisBound(1, 32, 32)),
            input_axis_bounds=((AxisBound(0, 32, 32), AxisBound(1, 32, 32)),),
            value_preserving_copy=True,
        ),
        InstructionDescriptor(
            "atlas.mxu1_reset_tile", "matmul", ("tensor_fp8", "tensor_fp8"),
            "tensor_bf16", "bf16", MXU1_POLICY, (2,), extent=2,
            input_dtypes=("fp8_e4m3", "fp8_e4m3"),
            input_numerical_policies=("raw-bit-pattern", "raw-bit-pattern"),
            input_ranks=(2, 2),
            output_axis_bounds=(AxisBound(0, 32, 32), AxisBound(1, 32, 32)),
            input_axis_bounds=(
                (AxisBound(0, 32, 32), AxisBound(1, 32, 32)),
                (AxisBound(0, 32, 32), AxisBound(1, 32, 32)),
            ),
            shape_contract="bounded", index_maps=_mxu0_maps(),
            validity=(AddressConstraint("at_least", "out", value=32),
                      AddressConstraint("at_most", "out", value=60)),
        ),
        InstructionDescriptor(
            "atlas.store_bf16_mxu1_tile", "identity", ("tensor_bf16",),
            "external", "bf16", MXU1_POLICY, (2,),
            input_dtypes=("bf16",), input_numerical_policies=(MXU1_POLICY,),
            input_ranks=(2,),
            output_axis_bounds=(AxisBound(0, 32, 32), AxisBound(1, 32, 32)),
            input_axis_bounds=((AxisBound(0, 32, 32), AxisBound(1, 32, 32)),),
            value_preserving_copy=True,
        ),
        InstructionDescriptor(
            "atlas.mxu1_finite_reset_tile", "matmul", ("tensor_fp8", "tensor_fp8"),
            "tensor_bf16", "bf16", MXU1_FINITE_POLICY, (2,), extent=2,
            input_dtypes=("fp8_e4m3", "fp8_e4m3"),
            input_numerical_policies=("raw-bit-pattern", "raw-bit-pattern"),
            input_ranks=(2, 2),
            output_axis_bounds=(AxisBound(0, 32, 32), AxisBound(1, 32, 32)),
            input_axis_bounds=(
                (AxisBound(0, 32, 32), AxisBound(1, 32, 32)),
                (AxisBound(0, 32, 32), AxisBound(1, 32, 32)),
            ),
            shape_contract="bounded", index_maps=_mxu0_maps(),
            validity=(AddressConstraint("at_least", "out", value=32),
                      AddressConstraint("at_most", "out", value=60)),
        ),
        InstructionDescriptor(
            "atlas.store_bf16_mxu1_finite_tile", "identity", ("tensor_bf16",),
            "external", "bf16", MXU1_FINITE_POLICY, (2,),
            input_dtypes=("bf16",), input_numerical_policies=(MXU1_FINITE_POLICY,),
            input_ranks=(2,),
            output_axis_bounds=(AxisBound(0, 32, 32), AxisBound(1, 32, 32)),
            input_axis_bounds=((AxisBound(0, 32, 32), AxisBound(1, 32, 32)),),
            value_preserving_copy=True,
        ),
    )
    banks = (
        StorageBank("external", "selected_dram_window", DRAM_SLOTS, "2048_byte_tile"),
        StorageBank("tensor_bf16", "tensor_register_file", 62,
                    "1024_byte_register", alignment=2),
        StorageBank("tensor_fp8", "tensor_register_file", 32,
                    "1024_byte_register"),
        StorageBank("tensor_mxu0_acc", "mxu0_accumulator_slots", 1,
                    "one_accumulator_slot"),
    )
    return NativeTargetProfile(TARGET_IDENTITY, descriptors, banks)
