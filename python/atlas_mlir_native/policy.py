"""Selected Atlas diagnostic numerical, layout and native pipeline constants.

Compiler-side admission/replay retains its diagnostic scope. Independent reference
and evaluator authority remain separate; no hardware execution is established.
"""

from __future__ import annotations

from .pack_lane import ROUNDED_PACK_POLICY

RTL_REVISION = "0079c0541111197741a231c002e3843fa6f545b2"

TARGET_IDENTITY = f"atlas-rtl-{RTL_REVISION}:movement-mxu01-vpu-pack-tile-v21"

FP8_DRAM_TILE_LAYOUT = "atlas.dram.fp8-e4m3-row-major-32x32-in-2k-slot.v1"

FP8_TENSOR_TILE_LAYOUT = "atlas.tensor.fp8-e4m3-row-major-32x32.v1"

BF16_DRAM_PAIR_LAYOUT = "atlas.dram.bf16-pair-halves-32x32-in-2k-slot.v1"

BF16_TENSOR_PAIR_LAYOUT = "atlas.tensor.bf16-pair-halves-32x32.v1"

MXU0_ACC_TILE_LAYOUT = "atlas.mxu0.accumulator-slot-32x32-bf16.v1"

MXU0_POLICY = "mxu0-e4m3-ordered-bf16-finite-v1"

MXU0_INPUT_DOMAIN = "finite-normal-e4m3-or-positive-zero"

MXU1_POLICY = "mxu1-e4m3-anchor-bf16-bounded-v1"

MXU1_INPUT_DOMAIN = "e4m3-positive-zero-or-signed-one-two-v1"

MXU1_FINITE_POLICY = "mxu1-e4m3-anchor-bf16-finite-reset-v1"

MXU1_FINITE_INPUT_DOMAIN = "finite-normal-e4m3-or-positive-zero"

FINITE_ANCHOR_POLICY = "selected-source-anchor-finite-reset-v1"

PACK_POLICY = "fp8-e4m3-unit-e8m0-seven-codes-v1"

PACK_ATTRS = (("input_domain", "bf16-seven-powers-and-zero-v1"), ("scale_code", 127))

SCALED_PACK_POLICY = "fp8-e4m3-scaled-e8m0-exact-normal-122-128-v1"

SCALED_PACK_DOMAIN = "bf16-positive-zero-or-exact-normal-under-scale-122-128-v1"

SCALED_PACK_RANGE = (122, 128)

PACK_RULE_BY_POLICY = {
    PACK_POLICY: "atlas.vpu_pack_unit_fp8_tile",
    SCALED_PACK_POLICY: "atlas.vpu_pack_scaled_fp8_tile",
    ROUNDED_PACK_POLICY: "atlas.vpu_pack_rounded_fp8_tile",
}

TRANSPOSE_RULE_BY_POLICY = {
    PACK_POLICY: "atlas.xlu_transpose_unit_fp8_tile",
    SCALED_PACK_POLICY: "atlas.xlu_transpose_scaled_fp8_tile",
    ROUNDED_PACK_POLICY: "atlas.xlu_transpose_rounded_fp8_tile",
}

STORE_RULE_BY_POLICY = {
    PACK_POLICY: "atlas.store_fp8_unit_tile",
    SCALED_PACK_POLICY: "atlas.store_fp8_scaled_tile",
    ROUNDED_PACK_POLICY: "atlas.store_fp8_rounded_tile",
}

PACK_RULE_NAMES = frozenset(PACK_RULE_BY_POLICY.values())

TRANSPOSE_RULE_NAMES = frozenset(TRANSPOSE_RULE_BY_POLICY.values())

STORE_RULE_NAMES = frozenset(STORE_RULE_BY_POLICY.values())

PACK_RULE_BY_TRANSPOSE_RULE = {
    transpose: PACK_RULE_BY_POLICY[policy]
    for policy, transpose in TRANSPOSE_RULE_BY_POLICY.items()
}

BF16_RELU_ATTRS = (
    ("input_domain", "finite-normal-or-zero-bf16"),
    ("numeric_semantics", "bf16-sign-clamp-v1"),
)

BF16_ADD_ATTRS = (
    ("input_domain", "finite-normal-exact-fp32-sum-v1"),
    ("numeric_semantics", "selected-f32-rne-bf16-chop-v1"),
)

BF16_SUB_ATTRS = (
    ("input_domain", "finite-normal-exact-fp32-difference-or-zero-v1"),
    ("numeric_semantics", "selected-f32-rne-bf16-chop-v1"),
)

BF16_MUL_ATTRS = (
    ("input_domain", "finite-normal-bf16-product-normal-v1"),
    ("numeric_semantics", "bf16-product-rne-v1"),
)

BF16_ROW_MAX_ATTRS = (
    ("input_domain", "bf16-raw-all-codes-v1"),
    ("numeric_semantics", "selected-rtl-bf16-total-order-row-max-v1"),
)

BF16_ROW_SUM_ATTRS = (
    ("input_domain", "bf16-normal-or-positive-zero-fp32-tree-normal-result-v1"),
    ("numeric_semantics", "selected-rtl-bf16-row-sum-fp32-rne-tree-bf16-rne-v1"),
)

BF16_EXP_ATTRS = (
    ("input_domain", "bf16-normal-minus80-plus80-or-signed-zero-v1"),
    ("numeric_semantics", "selected-rtl-bf16-exp-q9p12-lut-rne-v1"),
)

BF16_RECIP_ATTRS = (
    ("input_domain", "bf16-positive-normal-1-to-32-v1"),
    ("numeric_semantics", "selected-rtl-bf16-rcp-lut-trunc-v1"),
)

BF16_UNARY_ATTRS = {"exp_bf16": BF16_EXP_ATTRS, "recip_bf16": BF16_RECIP_ATTRS}

BF16_UNARY_MODES = {
    "atlas.vpu_exp_bf16_tile": "exp",
    "atlas.vpu_recip_bf16_tile": "recip",
}

BF16_ROW_REDUCTION_ATTRS = {
    "row_max_bf16": BF16_ROW_MAX_ATTRS,
    "row_sum_bf16": BF16_ROW_SUM_ATTRS,
}

BF16_ROW_REDUCTION_MODES = {
    "atlas.vpu_row_max_bf16_tile": "row_max",
    "atlas.vpu_row_sum_bf16_tile": "row_sum",
}

BF16_BINARY_ATTRS = {
    "add_bf16": BF16_ADD_ATTRS,
    "sub_bf16": BF16_SUB_ATTRS,
    "mul_bf16": BF16_MUL_ATTRS,
}

BF16_BINARY_MODES = {
    "atlas.vpu_add_bf16_tile": "add",
    "atlas.vpu_sub_bf16_tile": "sub",
    "atlas.vpu_mul_bf16_tile": "mul",
}

TILE_BYTES = 2048

DRAM_BASE = 0x90000000

DRAM_SLOTS = 128

NATIVE_PIPELINES = {
    "mandatory": ("--atlas-compile-unscheduled",),
    "optimized": ("--atlas-compile",),
}

NATIVE_CHECKS = ("--verify-atlas-generated-schedule", "--verify-atlas-machine-stream")
