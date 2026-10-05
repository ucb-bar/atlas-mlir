"""Selected Atlas matrix-unit tile description for native rule and placement study.

This profile composes the selected movement contract. Its static timing is
diagnostic; selected graphs can be executed on the standalone selected core.
"""

from __future__ import annotations

import importlib.resources
import json
import subprocess
from dataclasses import dataclass, replace
from pathlib import Path

from merlin.semantic_compiler.allocate import StorageBank
from merlin.semantic_compiler.model import IndexMap
from merlin.semantic_compiler.rules import (
    AddressConstraint,
    AxisBound,
    AxisEquality,
    InstructionDescriptor,
)
from merlin.semantic_compiler.snapshot import NativeTargetProfile

from .movement import MovementContract, _canonical, _sha


@dataclass(frozen=True)
class Mxu0Contract:
    record: dict[str, object]
    movement: MovementContract

    @classmethod
    def load(cls) -> Mxu0Contract:
        path = importlib.resources.files("atlas_native_support").joinpath(
            "mxu0_contract.json"
        )
        record = json.loads(path.read_text())
        movement = MovementContract.load()
        expected = {
            "schema",
            "scope",
            "movement_contract_identity",
            "rtl_revision",
            "software_spec_sha256",
            "arithmetic_submodule_revision",
            "source_sha256",
            "geometry",
            "numerical_policy",
            "weight_orientation",
            "accumulation",
            "mxu1",
            "xlu",
            "vpu_relu",
            "vpu_add",
            "vpu_mul",
            "vpu_sub",
            "vpu_square",
            "vpu_recip",
            "vpu_log2",
            "vpu_sqrt",
            "vpu_exp2",
            "vpu_min",
            "vpu_max",
            "vpu_col_min",
            "vpu_col_max",
            "vpu_col_sum",
            "vpu_row_min",
            "vpu_row_max",
            "vpu_row_sum",
            "vfp8_conversion",
            "bf16_result_layout",
        }
        if (
            set(record) != expected
            or record["schema"] != "atlas.native_tensor_contract.v21"
            or (record["scope"] != "diagnostic_execution_unqualified_timing")
        ):
            raise ValueError("unrecognized MXU0 selection contract")
        if record["movement_contract_identity"] != movement.identity or (
            record["rtl_revision"] != movement.record["rtl_revision"]
        ):
            raise ValueError("MXU0 selection differs from its movement base")
        geometry = record["geometry"]
        if (
            not isinstance(geometry, dict)
            or set(geometry)
            != {
                "array_rows",
                "array_cols",
                "accumulator_rows",
                "matrix_register_bytes",
                "matrix_register_count",
                "weight_slots",
                "accumulator_slots",
            }
            or any(type(value) is not int or value <= 0 for value in geometry.values())
        ):
            raise ValueError("MXU0 geometry is incomplete")
        rows, cols = geometry["array_rows"], geometry["array_cols"]
        register_bytes = geometry["matrix_register_bytes"]
        layout = record["bf16_result_layout"]
        if (
            not isinstance(layout, dict)
            or set(layout)
            != {
                "kind",
                "register_count",
                "element_bytes",
            }
            or layout["kind"] != "column_halves_row_major"
            or (
                type(layout["register_count"]) is not int
                or layout["register_count"] != 2
                or type(layout["element_bytes"]) is not int
                or layout["element_bytes"] != 2
            )
        ):
            raise ValueError("MXU0 BF16 pair layout is unqualified")
        if (
            rows != geometry["accumulator_rows"]
            or register_bytes != rows * cols
            or (
                register_bytes % movement.block_bytes
                or movement.record["vmem_capacity_bytes"] % register_bytes
                or movement.record["dram_window_bytes"] % register_bytes
                or geometry["matrix_register_count"] % layout["register_count"]
                or cols % layout["register_count"]
                or rows * cols * layout["element_bytes"]
                != layout["register_count"] * register_bytes
            )
        ):
            raise ValueError("MXU0 geometry does not fit the selected physical stores")
        numerical = record["numerical_policy"]
        if (
            not isinstance(numerical, dict)
            or set(numerical)
            != {
                "operand_dtype",
                "operand_policy",
                "result_dtype",
                "result_policy",
            }
            or any(
                not isinstance(value, str) or not value for value in numerical.values()
            )
        ):
            raise ValueError("MXU0 numerical identity is incomplete")
        if record["weight_orientation"] != "output_by_reduction":
            raise ValueError("MXU0 weight orientation is unqualified")
        if record["accumulation"] != {
            "semantic_operation": "contraction_accumulate",
            "initial_value": "previous_ordered_bf16_accumulator",
            "physical_update": "same_mxu0_accumulator_slot",
            "instruction": "VMATMUL.ACC.MXU0",
        }:
            raise ValueError("MXU0 continuation contract is unqualified")
        if record["mxu1"] != {
            "operand_dtype": "fp8_e4m3",
            "operand_policy": "e4m3_anchor_exact_power_of_two_subset",
            "admitted_encodings": ["00", "08", "18", "38", "98", "b8"],
            "result_dtype": "bf16",
            "result_policy": "mxu1_anchor_exact_sum_bf16_rne_k32",
            "reduction_arithmetic": "anchor_aligned_integer_sum_then_bf16_rne",
        }:
            raise ValueError("MXU1 bounded arithmetic contract is unqualified")
        if record["xlu"] != {
            "operation": "transpose",
            "operand_dtype": "fp8_e4m3",
            "numerical_policy": "e4m3_exponent_1_14_or_zero",
            "shape": [rows, cols],
            "physical_layout": "row_major_fp8",
            "in_place_admitted": False,
        }:
            raise ValueError("XLU transpose contract is unqualified")
        if record["vpu_relu"] != {
            "operation": "relu",
            "operand_dtype": "bf16",
            "operand_policy": numerical["result_policy"],
            "result_policy": numerical["result_policy"],
            "shape": [rows, cols],
            "physical_layout": layout["kind"],
            "register_count": layout["register_count"],
            "semantic_domain": "finite_bf16_from_admitted_mxu0",
            "negative_result": "positive_zero",
            "nonnegative_result": "preserve_bits",
            "instruction": "VRELU",
            "in_place_admitted": False,
        }:
            raise ValueError("VPU ReLU contract is unqualified")
        if record["vpu_add"] != {
            "operation": "add",
            "operand_dtype": "bf16",
            "operand_policy": numerical["result_policy"],
            "result_policy": "vpu_add_widen_f32_rne_then_high16",
            "shape": [rows, cols],
            "physical_layout": layout["kind"],
            "register_count": layout["register_count"],
            "semantic_domain": "finite_bf16_from_admitted_mxu0",
            "intermediate": "binary32_round_nearest_even",
            "readout": "upper_16_bits_of_binary32",
            "instruction": "VADD.BF16",
            "in_place_admitted": False,
        }:
            raise ValueError("VPU BF16 add contract is unqualified")
        if record["vpu_mul"] != {
            "operation": "mul",
            "operand_dtype": "bf16",
            "operand_policy": numerical["result_policy"],
            "result_policy": "vpu_mul_exact_product_bf16_rne",
            "shape": [rows, cols],
            "physical_layout": layout["kind"],
            "register_count": layout["register_count"],
            "semantic_domain": "normal_bf16_operands_and_normal_result_from_admitted_mxu0",
            "arithmetic": "exact_product_then_bf16_round_nearest_even",
            "instruction": "VMUL.BF16",
            "in_place_admitted": False,
        }:
            raise ValueError("VPU BF16 multiply contract is unqualified")
        if record["vpu_sub"] != {
            "operation": "sub",
            "operand_dtype": "bf16",
            "operand_policy": numerical["result_policy"],
            "result_policy": "vpu_sub_widen_f32_rne_then_high16",
            "shape": [rows, cols],
            "physical_layout": layout["kind"],
            "register_count": layout["register_count"],
            "semantic_domain": "normal_bf16_operands_and_normal_nonzero_result_from_admitted_mxu0",
            "intermediate": "binary32_round_nearest_even",
            "readout": "upper_16_bits_of_binary32",
            "instruction": "VSUB.BF16",
            "in_place_admitted": False,
        }:
            raise ValueError("VPU BF16 subtract contract is unqualified")
        if record["vpu_square"] != {
            "operation": "square",
            "operand_dtype": "bf16",
            "operand_policy": numerical["result_policy"],
            "result_policy": "vpu_square_normal_mantissa_truncate",
            "shape": [rows, cols],
            "physical_layout": layout["kind"],
            "register_count": layout["register_count"],
            "semantic_domain": "normal_bf16_operand_and_normal_result_from_admitted_mxu0",
            "arithmetic": "exact_square_then_bf16_mantissa_truncate",
            "instruction": "VSQUARE.BF16",
            "in_place_admitted": False,
        }:
            raise ValueError("VPU BF16 square contract is unqualified")
        if record["vpu_recip"] != {
            "operation": "reciprocal",
            "operand_dtype": "bf16",
            "operand_policies": ["bf16_raw_bits", numerical["result_policy"]],
            "result_policy": "vpu_recip_lut_m1n16_selected_rtl",
            "shape": [rows, cols],
            "physical_layout": layout["kind"],
            "register_count": layout["register_count"],
            "semantic_domain": "all_bf16_raw_encodings_isolated_pair_program",
            "lut_entries": 128,
            "fixed_integer_bits": 1,
            "fixed_fraction_bits": 16,
            "output_fraction_bits": 7,
            "exponent_policy": "wrap_to_8_bits_before_specials",
            "instruction": "VRECIP.BF16",
            "in_place_admitted": False,
        }:
            raise ValueError("VPU BF16 reciprocal contract is unqualified")
        if record["vpu_log2"] != {
            "operation": "log2",
            "operand_dtype": "bf16",
            "operand_policies": ["bf16_raw_bits", numerical["result_policy"]],
            "result_policy": "vpu_log2_lut_q9n16_selected_rtl",
            "shape": [rows, cols],
            "physical_layout": layout["kind"],
            "register_count": layout["register_count"],
            "semantic_domain": "all_bf16_raw_encodings_isolated_pair_program",
            "lut_entries": 128,
            "fixed_integer_bits": 9,
            "fixed_fraction_bits": 16,
            "output_fraction_bits": 7,
            "input_sign_policy": "ignored",
            "negative_result_policy": "input_exponent_below_bias",
            "instruction": "VLOG2",
            "in_place_admitted": False,
            "arithmetic_source_sha256": {
                "LogLUT.scala": "13ee6e5367003d163996660086a25027f2f3b4e0d93c69a890136725b7b35e37",
                "LUTParams.scala": "684aac051fe104a04784545bdc2df29b4dfc06734c1309c1aee5ba4242dbbcb5",
            },
        }:
            raise ValueError("VPU BF16 log2 contract is unqualified")
        if record["vpu_sqrt"] != {
            "operation": "sqrt",
            "operand_dtype": "bf16",
            "operand_policies": ["bf16_raw_bits", numerical["result_policy"]],
            "result_policy": "vpu_sqrt_lut_q1n16_selected_rtl",
            "shape": [rows, cols],
            "physical_layout": layout["kind"],
            "register_count": layout["register_count"],
            "semantic_domain": "all_bf16_raw_encodings_isolated_pair_program",
            "lut_entries": 128,
            "fixed_integer_bits": 1,
            "fixed_fraction_bits": 16,
            "output_fraction_bits": 7,
            "input_sign_policy": "ignored",
            "special_exponent_policy": "zero_or_subnormal_to_positive_zero_infinity_to_positive_infinity_nan_to_positive_zero",
            "instruction": "VSQRT",
            "in_place_admitted": False,
            "arithmetic_source_sha256": {
                "SqrtLUT.scala": "490cf82e396c2a400330d8a0e625423f25ee49dfe51aac66e09785081f01b54b",
                "LUTParams.scala": "684aac051fe104a04784545bdc2df29b4dfc06734c1309c1aee5ba4242dbbcb5",
            },
        }:
            raise ValueError("VPU BF16 sqrt contract is unqualified")
        if record["vpu_exp2"] != {
            "operation": "exp2",
            "operand_dtype": "bf16",
            "operand_policy": "bf16_exp2_integer_special_subset_v1",
            "result_policy": "vpu_exp2_integer_special_selected_rtl_v1",
            "shape": [rows, cols],
            "physical_layout": layout["kind"],
            "register_count": layout["register_count"],
            "semantic_domain": "17_directed_bf16_integer_zero_infinity_codes",
            "admitted_input_codes_hex": [
                "0000", "8000", "7f80", "ff80", "3f80", "bf80", "4000", "c000",
                "4080", "c080", "4100", "c100", "42b0", "42b2", "42c8", "c2b2", "c2c8",
            ],
            "selected_positive_overflow_integer": 89,
            "instruction": "VEXP2",
            "in_place_admitted": False,
            "arithmetic_source_sha256": {
                "src/main/scala/sp26-fp-units/vpuLUTs/ExLUT.scala": "a4c61f023e39a86fc5c894a7a0bd03a666e09daa9886309b6ed470140d6d52ed",
                "src/main/scala/sp26-fp-units/Qmn.scala": "d717095d4504896fce8216ac359ec3db2653b89e53c26b4d060f0180fcfe9185",
                "src/main/scala/sp26-fp-units/common.scala": "21182c1398c92fc2fc1c99d4f9d841ce061844b0e91b1d95b045d3adbcef3e0f",
            },
        }:
            raise ValueError("VPU BF16 exp2 bounded contract is unqualified")
        for kind in ("min", "max"):
            if record[f"vpu_{kind}"] != {
                "operation": kind,
                "operand_dtype": "bf16",
                "operand_policy": numerical["result_policy"],
                "result_policy": f"vpu_{kind}_bf16_sign_folded_raw_order",
                "shape": [rows, cols],
                "physical_layout": layout["kind"],
                "register_count": layout["register_count"],
                "semantic_domain": "finite_bf16_from_admitted_mxu0",
                "comparison": "sign_folded_unsigned_16_bit_code",
                "equal_tie": "second_operand",
                "instruction": f"V{kind.upper()}.BF16",
                "in_place_admitted": False,
            }:
                raise ValueError(f"VPU BF16 {kind} contract is unqualified")
        for kind in ("min", "max"):
            if record[f"vpu_col_{kind}"] != {
                "operation": f"physical_col_{kind}",
                "operand_dtype": "bf16",
                "operand_policy": numerical["result_policy"],
                "result_policy": f"vpu_col_{kind}_bf16_selected_rtl_64x16",
                "input_shape": [rows, cols],
                "output_shape": [2 * rows, cols // 2],
                "physical_layout": "two_consecutive_32x16_register_rows",
                "register_count": layout["register_count"],
                "semantic_domain": "finite_bf16_from_admitted_mxu0",
                "comparison": "sign_folded_unsigned_16_bit_code",
                "reduction_axis": "physical_rows_64",
                "broadcast_rows": 2 * rows,
                "instruction": f"VRED{kind.upper()}.BF16",
                "diagnostic_delay_cycles": 256,
                "in_place_admitted": False,
            }:
                raise ValueError(f"VPU BF16 RTL column {kind} contract is unqualified")
        if record["vpu_col_sum"] != {
            "operation": "physical_col_sum",
            "operand_dtype": "bf16",
            "operand_policy": numerical["result_policy"],
            "result_policy": "vpu_col_sum_bf16_fp32_rne_sequential_then_high16_selected_rtl_64x16",
            "input_shape": [rows, cols],
            "output_shape": [2 * rows, cols // 2],
            "physical_layout": "two_consecutive_32x16_register_rows",
            "register_count": layout["register_count"],
            "semantic_domain": "finite_normal_or_positive_zero_bf16_with_normal_or_zero_prefix_partials",
            "reduction_axis": "physical_rows_64",
            "initial_accumulator": "positive_zero",
            "intermediate_rounding": "binary32_rne_each_add",
            "readout": "upper_16_bits_of_binary32",
            "broadcast_rows": 2 * rows,
            "instruction": "VREDSUM.BF16",
            "diagnostic_delay_cycles": 256,
            "in_place_admitted": False,
        }:
            raise ValueError("VPU BF16 RTL column sum contract is unqualified")
        for kind in ("min", "max"):
            if record[f"vpu_row_{kind}"] != {
                "operation": f"physical_row_{kind}",
                "operand_dtype": "bf16",
                "operand_policy": numerical["result_policy"],
                "result_policy": f"vpu_row_{kind}_bf16_selected_rtl_pair_broadcast",
                "shape": [rows, cols],
                "physical_layout": layout["kind"],
                "register_count": layout["register_count"],
                "semantic_domain": "finite_bf16_from_admitted_mxu0",
                "comparison": "sign_folded_unsigned_16_bit_code",
                "reduction_axis": "logical_columns_32_across_two_register_halves",
                "broadcast_columns": cols,
                "instruction": f"VRED{kind.upper()}.ROW.BF16",
                "diagnostic_delay_cycles": 128,
                "in_place_admitted": False,
            }:
                raise ValueError(f"VPU BF16 RTL row {kind} contract is unqualified")
        if record["vpu_row_sum"] != {
            "operation": "physical_row_sum",
            "operand_dtype": "bf16",
            "operand_policy": numerical["result_policy"],
            "result_policy": "vpu_row_sum_bf16_fp32_rne_tree_then_bf16_rne",
            "shape": [rows, cols],
            "physical_layout": layout["kind"],
            "register_count": layout["register_count"],
            "semantic_domain": "finite_normal_or_positive_zero_bf16_with_normal_or_zero_tree_partials",
            "reduction_axis": "logical_columns_32_across_two_register_halves",
            "tree": "five_levels_adjacent_pairs",
            "intermediate_rounding": "binary32_rne_each_add",
            "final_rounding": "bf16_rne",
            "broadcast_columns": cols,
            "instruction": "VREDSUM.ROW.BF16",
            "diagnostic_delay_cycles": 128,
            "in_place_admitted": False,
        }:
            raise ValueError("VPU BF16 RTL row sum contract is unqualified")
        conversion = record["vfp8_conversion"]
        if conversion != {
            "scope": "diagnostic_exact_anchor_values_no_exceptions",
            "admitted_scale_codes": [126, 127, 128],
            "scale_register": 0,
            "scale_exponent_rule": "e8m0_code_minus_127",
            "input_bf16_words_hex": ["0000", "3d80", "3f80", "bd80", "bf80"],
            "input_policy": "bf16_anchor_exact_no_exceptions_column_halves_row_major",
            "pack_row_order": "consecutive_physical_bf16_rows_then_32_fp8_bytes",
            "unpack_row_order": "inverse_consecutive_physical_rows_to_bf16_pair",
            "pack_result_policies": {
                "126": "fp8_pack_e8m0_126_consecutive_rows_anchor_no_exceptions",
                "127": "fp8_pack_e8m0_127_consecutive_rows_anchor_no_exceptions",
                "128": "fp8_pack_e8m0_128_consecutive_rows_anchor_no_exceptions",
            },
            "unpack_result_policies": {
                "126": "bf16_unpack_e8m0_126_inverse_rows_anchor_no_exceptions",
                "127": "bf16_unpack_e8m0_127_inverse_rows_anchor_no_exceptions",
                "128": "bf16_unpack_e8m0_128_inverse_rows_anchor_no_exceptions",
            },
            "downstream_mxu1_pack_code": 127,
        }:
            raise ValueError("bounded E8M0 pack/unpack contract is unqualified")
        sources = record["source_sha256"]
        if (
            not isinstance(sources, dict)
            or not sources
            or any(
                not isinstance(name, str)
                or not isinstance(digest, str)
                or len(digest) != 64
            for name, digest in sources.items()
            )
        ):
            raise ValueError("MXU0 source identities are incomplete")
        if not {
            "src/main/scala/atlas/mxu/ipt/InnerProductTrees.scala",
            "src/main/scala/atlas/mxu/ipt/AnchorAccumulationTree.scala",
            "src/main/scala/atlas/common/InnerProductTreeParams.scala",
            "src/main/scala/atlas/mxu/FPUtils.scala",
            "src/main/scala/atlas/xlu/XLU.scala",
            "src/main/scala/atlas/vector/VectorEngine.scala",
            "src/main/scala/atlas/vector/VectorFSM.scala",
            "src/main/scala/atlas/vector/VectorEngineTop.scala",
            "src/main/scala/atlas/scalar/ScalarCore.scala",
            "src/main/scala/atlas/scalar/IDecode.scala",
            "src/main/scala/atlas/vector/laneBoxes/AddSubSumVec.scala",
            "src/main/scala/atlas/vector/laneBoxes/MulRec.scala",
            "src/main/scala/atlas/vector/laneBoxes/SquareCubeVec.scala",
            "src/main/scala/atlas/vector/laneBoxes/VectorParam.scala",
            "src/main/scala/atlas/vector/laneBoxes/FP8Pack.scala",
            "src/main/scala/atlas/vector/laneBoxes/FP8Unpack.scala",
            "src/main/scala/atlas/scalar/ScalingFactorRegFile.scala",
            "src/main/scala/atlas/scalar/ScalarISA.scala",
            "src/main/scala/atlas/vector/laneBoxes/PairwiseMin.scala",
            "src/main/scala/atlas/vector/laneBoxes/PairwiseMax.scala",
            "src/main/scala/atlas/scalar/Instructions.scala",
            "src/main/scala/atlas/vector/laneBoxes/RowMin.scala",
            "src/main/scala/atlas/vector/laneBoxes/RowMax.scala",
            "src/main/scala/atlas/vector/laneBoxes/SumRedu.scala",
            "src/main/scala/atlas/vector/laneBoxes/ExpLane.scala",
        } <= set(sources):
            raise ValueError("Atlas tensor source pins are incomplete")
        if (
            not isinstance(record["software_spec_sha256"], str)
            or len(record["software_spec_sha256"]) != 64
            or (
            not isinstance(record["arithmetic_submodule_revision"], str)
            or len(record["arithmetic_submodule_revision"]) != 40
            )
        ):
            raise ValueError("MXU0 numerical source pins are incomplete")
        return cls(record, movement)

    @property
    def identity(self) -> str:
        return "atlas-tensor-selection-" + _sha(_canonical(self.record))

    def verify_source(self, rtl_root: Path, software_spec: Path | None = None) -> None:
        self.movement.verify_source(rtl_root)
        if (
            software_spec is not None
            and _sha(software_spec.read_bytes()) != self.record["software_spec_sha256"]
        ):
            raise ValueError("selected Atlas software semantics changed")
        for relative, digest in self.record["source_sha256"].items():
            if _sha((rtl_root / relative).read_bytes()) != digest:
                raise ValueError(f"selected MXU0 source changed: {relative}")
        gitlink = subprocess.run(
            ["git", "ls-tree", "HEAD", "dependencies/sp26-fp-units"],
            cwd=rtl_root,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.split()
        if (
            len(gitlink) != 4
            or gitlink[:2] != ["160000", "commit"]
            or (gitlink[2] != self.record["arithmetic_submodule_revision"])
        ):
            raise ValueError("selected MXU0 arithmetic submodule pin changed")


def _tile_bounds(contract: Mxu0Contract) -> tuple[AxisBound, AxisBound]:
    geometry = contract.record["geometry"]
    return (
        AxisBound(0, geometry["array_rows"], geometry["array_rows"]),
        AxisBound(1, geometry["array_cols"], geometry["array_cols"]),
    )


def _physical_col_bounds(contract: Mxu0Contract) -> tuple[AxisBound, AxisBound]:
    geometry = contract.record["geometry"]
    return (
        AxisBound(0, 2 * geometry["array_rows"], 2 * geometry["array_rows"]),
        AxisBound(1, geometry["array_cols"] // 2, geometry["array_cols"] // 2),
    )


def _copy(
    contract: Mxu0Contract,
    name: str,
    input_storage: str,
    output_storage: str,
    dtype: str,
    policy: str,
    extent: int,
) -> InstructionDescriptor:
    return InstructionDescriptor(
        name,
        "identity",
        (input_storage,),
        output_storage,
        dtype,
        policy,
        (2,),
        extent=extent,
        input_dtypes=(dtype,),
        input_numerical_policies=(policy,),
        input_ranks=(2,),
        output_axis_bounds=_tile_bounds(contract),
        value_preserving_copy=True,
    )


def _physical_col_copy(
    contract: Mxu0Contract,
    name: str,
    input_storage: str,
    output_storage: str,
    dtype: str,
    policy: str,
    extent: int,
) -> InstructionDescriptor:
    bounds = _physical_col_bounds(contract)
    return replace(
        _copy(contract, name, input_storage, output_storage, dtype, policy, extent),
        output_axis_bounds=bounds,
        input_axis_bounds=(bounds,),
    )


def admits_fp8_tile(contract: Mxu0Contract, policy: str, payload: bytes) -> bool:
    """Check a compiler or invocation operand against its declared policy."""
    if policy == contract.record["numerical_policy"]["operand_policy"]:
        return all(value == 0 or 1 <= ((value >> 3) & 15) <= 14 for value in payload)
    mxu1 = contract.record["mxu1"]
    if policy == mxu1["operand_policy"]:
        admitted = {int(value, 16) for value in mxu1["admitted_encodings"]}
        return all(value in admitted for value in payload)
    conversion = contract.record["vfp8_conversion"]
    if policy in conversion["pack_result_policies"].values():
        return False  # generated packed data is never an external input here
    return False


def admits_bf16_anchor_pair(
    contract: Mxu0Contract, policy: str, payload: bytes
) -> bool:
    """Validate the full physical pair before any selected-core preload."""
    conversion = contract.record["vfp8_conversion"]
    tile_bytes = contract.record["geometry"]["matrix_register_bytes"]
    allowed = {int(word, 16) for word in conversion["input_bf16_words_hex"]}
    return (
        policy == conversion["input_policy"]
        and len(payload) == 2 * tile_bytes
        and all(
            int.from_bytes(payload[index : index + 2], "little") in allowed
            for index in range(0, len(payload), 2)
        )
    )


def admits_bf16_raw_pair(
    contract: Mxu0Contract, policy: str, payload: bytes
) -> bool:
    """Admit every BF16 bit pattern for selected raw-input unary VPU modes."""
    return (
        policy == contract.record["vpu_recip"]["operand_policies"][0]
        and type(payload) is bytes
        and len(payload) == 2 * contract.record["geometry"]["matrix_register_bytes"]
    )


def admits_bf16_exp2_pair(
    contract: Mxu0Contract, policy: str, payload: bytes
) -> bool:
    """Check every runtime lane against the independently tested exp2 subset."""
    mode = contract.record["vpu_exp2"]
    tile_bytes = contract.record["geometry"]["matrix_register_bytes"]
    admitted = {int(code, 16) for code in mode["admitted_input_codes_hex"]}
    return (
        policy == mode["operand_policy"]
        and type(payload) is bytes
        and len(payload) == 2 * tile_bytes
        and all(
            int.from_bytes(payload[index : index + 2], "little") in admitted
            for index in range(0, len(payload), 2)
        )
    )


def mxu0_profile(contract: Mxu0Contract) -> NativeTargetProfile:
    """Describe selected MXU0, bounded MXU1, and XLU in one physical profile."""
    geometry = contract.record["geometry"]
    numerical = contract.record["numerical_policy"]
    rows = geometry["array_rows"]
    tile_bytes = geometry["matrix_register_bytes"]
    operand_dtype = numerical["operand_dtype"]
    operand_policy = numerical["operand_policy"]
    result_dtype = numerical["result_dtype"]
    result_policy = numerical["result_policy"]
    result_registers = contract.record["bf16_result_layout"]["register_count"]
    loop = 3
    maps = (
        IndexMap(loop, ((1, 0, 0), (0, 0, 1)), (0, 0)),  # activation[m, k]
        IndexMap(loop, ((0, 1, 0), (0, 0, 1)), (0, 0)),  # weight[n, k]
        IndexMap(loop, ((1, 0, 0), (0, 1, 0)), (0, 0)),  # output[m, n]
    )
    accumulate_maps = (*maps[:2], maps[2], maps[2])
    mxu1 = contract.record["mxu1"]
    transpose_maps = (
        IndexMap(2, ((0, 1), (1, 0)), (0, 0)),
        IndexMap(2, ((1, 0), (0, 1)), (0, 0)),
    )
    descriptors = (
        _copy(
            contract,
            "dma_load_fp8",
            "external_fp8",
            "vmem_fp8",
            operand_dtype,
            operand_policy,
            1,
        ),
        _copy(
            contract,
            "vload_fp8",
            "vmem_fp8",
            "vrf_fp8",
            operand_dtype,
            operand_policy,
            1,
        ),
        _copy(
            contract,
            "mxu0_push_weight",
            "vrf_fp8",
            "mxu0_weight",
            operand_dtype,
            operand_policy,
            1,
        ),
        InstructionDescriptor(
            "mxu0_matmul_reset",
            "contraction",
            ("vrf_fp8", "mxu0_weight"),
            "mxu0_accum",
            result_dtype,
            result_policy,
            (2,),
            required_attrs=(("reduction_order", "k_ascending"),),
            input_dtypes=(operand_dtype, operand_dtype),
            input_numerical_policies=(operand_policy, operand_policy),
            input_ranks=(2, 2),
            output_axis_bounds=_tile_bounds(contract),
            input_axis_bounds=(
                (AxisBound(1, geometry["array_rows"], geometry["array_rows"]),),
                (AxisBound(1, geometry["array_rows"], geometry["array_rows"]),),
            ),
            index_maps=maps,
            shape_contract="relations",
            shape_equalities=(
                AxisEquality("in0", 0, "out", 0),
                              AxisEquality("in1", 0, "out", 1),
                AxisEquality("in0", 1, "in1", 1),
            ),
        ),
        InstructionDescriptor(
            "mxu0_matmul_accumulate",
            contract.record["accumulation"]["semantic_operation"],
            ("vrf_fp8", "mxu0_weight", "mxu0_accum"),
            "mxu0_accum",
            result_dtype,
            result_policy,
            (2,),
            required_attrs=(("reduction_order", "k_ascending"),),
            input_dtypes=(operand_dtype, operand_dtype, result_dtype),
            input_numerical_policies=(operand_policy, operand_policy, result_policy),
            input_ranks=(2, 2, 2),
            output_axis_bounds=_tile_bounds(contract),
            input_axis_bounds=(
                               (AxisBound(1, geometry["array_rows"], geometry["array_rows"]),),
                (AxisBound(1, geometry["array_rows"], geometry["array_rows"]),),
                _tile_bounds(contract),
            ),
            index_maps=accumulate_maps,
            shape_contract="relations",
            shape_equalities=(
                AxisEquality("in0", 0, "out", 0),
                              AxisEquality("in1", 0, "out", 1),
                              AxisEquality("in0", 1, "in1", 1),
                              AxisEquality("in2", 0, "out", 0),
                AxisEquality("in2", 1, "out", 1),
            ),
            validity=(AddressConstraint("eq_offset", "out", "in2", 0),),
            input_read_offsets=(0, 0, 0),
            completion_offset=1,
            in_place_inputs=(2,),
        ),
        _copy(
            contract,
            "mxu0_pop_bf16",
            "mxu0_accum",
            "vrf_bf16",
            result_dtype,
            result_policy,
            result_registers,
        ),
        _copy(
            contract,
            "vstore_bf16_pair",
            "vrf_bf16",
            "vmem_bf16",
            result_dtype,
            result_policy,
            result_registers,
        ),
        _copy(
            contract,
            "dma_store_bf16_pair",
            "vmem_bf16",
            "external_bf16",
            result_dtype,
            result_policy,
            result_registers,
        ),
        _copy(
            contract,
            "mxu1_dma_load_fp8",
            "external_fp8",
            "vmem_fp8",
            mxu1["operand_dtype"],
            mxu1["operand_policy"],
            1,
        ),
        _copy(
            contract,
            "mxu1_vload_fp8",
            "vmem_fp8",
            "vrf_fp8",
            mxu1["operand_dtype"],
            mxu1["operand_policy"],
            1,
        ),
        _copy(
            contract,
            "mxu1_push_weight",
            "vrf_fp8",
            "mxu1_weight",
            mxu1["operand_dtype"],
            mxu1["operand_policy"],
            1,
        ),
        InstructionDescriptor(
            "mxu1_matmul_reset",
            "contraction",
            ("vrf_fp8", "mxu1_weight"),
            "mxu1_accum",
            mxu1["result_dtype"],
            mxu1["result_policy"],
            (2,),
            required_attrs=(("reduction_order", "anchor_aligned_k32"),),
            input_dtypes=(mxu1["operand_dtype"], mxu1["operand_dtype"]),
            input_numerical_policies=(mxu1["operand_policy"], mxu1["operand_policy"]),
            input_ranks=(2, 2),
            output_axis_bounds=_tile_bounds(contract),
            input_axis_bounds=(
                (AxisBound(1, rows, rows),),
                (AxisBound(1, rows, rows),),
            ),
            index_maps=maps,
            shape_contract="relations",
            shape_equalities=(
                AxisEquality("in0", 0, "out", 0),
                              AxisEquality("in1", 0, "out", 1),
                AxisEquality("in0", 1, "in1", 1),
            ),
        ),
        _copy(
            contract,
            "mxu1_pop_bf16",
            "mxu1_accum",
            "vrf_bf16",
            mxu1["result_dtype"],
            mxu1["result_policy"],
            result_registers,
        ),
        _copy(
            contract,
            "mxu1_vstore_bf16_pair",
            "vrf_bf16",
            "vmem_bf16",
            mxu1["result_dtype"],
            mxu1["result_policy"],
            result_registers,
        ),
        _copy(
            contract,
            "mxu1_dma_store_bf16_pair",
            "vmem_bf16",
            "external_bf16",
            mxu1["result_dtype"],
            mxu1["result_policy"],
            result_registers,
        ),
        InstructionDescriptor(
            "xlu_transpose_fp8",
            "transpose",
            ("vrf_fp8",),
            "vrf_fp8",
            operand_dtype,
            operand_policy,
            (2,),
            input_dtypes=(operand_dtype,),
            input_numerical_policies=(operand_policy,),
            input_ranks=(2,),
            output_axis_bounds=_tile_bounds(contract),
            input_axis_bounds=(_tile_bounds(contract),),
            index_maps=transpose_maps,
        ),
        InstructionDescriptor(
            "vpu_relu_bf16",
            contract.record["vpu_relu"]["operation"],
            ("vrf_bf16",),
            "vrf_bf16",
            result_dtype,
            contract.record["vpu_relu"]["result_policy"],
            (2,),
            extent=result_registers,
            input_dtypes=(result_dtype,),
            input_numerical_policies=(result_policy,),
            input_ranks=(2,),
            output_axis_bounds=_tile_bounds(contract),
            input_axis_bounds=(_tile_bounds(contract),),
            input_read_offsets=(2 * rows + 3,),
            completion_offset=2 * rows + 8,
        ),
        InstructionDescriptor(
            "vpu_add_bf16",
            contract.record["vpu_add"]["operation"],
            ("vrf_bf16", "vrf_bf16"),
            "vrf_bf16",
            result_dtype,
            contract.record["vpu_add"]["result_policy"],
            (2,),
            extent=result_registers,
            input_dtypes=(result_dtype, result_dtype),
            input_numerical_policies=(result_policy, result_policy),
            input_ranks=(2, 2),
            output_axis_bounds=_tile_bounds(contract),
            input_axis_bounds=(_tile_bounds(contract), _tile_bounds(contract)),
            input_read_offsets=(2 * rows + 3, 2 * rows + 3),
            completion_offset=2 * rows + 8,
        ),
        _copy(
            contract,
            "vstore_vpu_add_bf16_pair",
            "vrf_bf16",
            "vmem_bf16",
            result_dtype,
            contract.record["vpu_add"]["result_policy"],
            result_registers,
        ),
        _copy(
            contract,
            "dma_store_vpu_add_bf16_pair",
            "vmem_bf16",
            "external_bf16",
            result_dtype,
            contract.record["vpu_add"]["result_policy"],
            result_registers,
        ),
        InstructionDescriptor(
            "vpu_mul_bf16",
            contract.record["vpu_mul"]["operation"],
            ("vrf_bf16", "vrf_bf16"),
            "vrf_bf16",
            result_dtype,
            contract.record["vpu_mul"]["result_policy"],
            (2,),
            extent=result_registers,
            input_dtypes=(result_dtype, result_dtype),
            input_numerical_policies=(result_policy, result_policy),
            input_ranks=(2, 2),
            output_axis_bounds=_tile_bounds(contract),
            input_axis_bounds=(_tile_bounds(contract), _tile_bounds(contract)),
            input_read_offsets=(2 * rows + 3, 2 * rows + 3),
            completion_offset=2 * rows + 8,
        ),
        _copy(
            contract,
            "vstore_vpu_mul_bf16_pair",
            "vrf_bf16",
            "vmem_bf16",
            result_dtype,
            contract.record["vpu_mul"]["result_policy"],
            result_registers,
        ),
        _copy(
            contract,
            "dma_store_vpu_mul_bf16_pair",
            "vmem_bf16",
            "external_bf16",
            result_dtype,
            contract.record["vpu_mul"]["result_policy"],
            result_registers,
        ),
        InstructionDescriptor(
            "vpu_sub_bf16",
            contract.record["vpu_sub"]["operation"],
            ("vrf_bf16", "vrf_bf16"),
            "vrf_bf16",
            result_dtype,
            contract.record["vpu_sub"]["result_policy"],
            (2,),
            extent=result_registers,
            input_dtypes=(result_dtype, result_dtype),
            input_numerical_policies=(result_policy, result_policy),
            input_ranks=(2, 2),
            output_axis_bounds=_tile_bounds(contract),
            input_axis_bounds=(_tile_bounds(contract), _tile_bounds(contract)),
            input_read_offsets=(2 * rows + 3, 2 * rows + 3),
            completion_offset=2 * rows + 8,
        ),
        _copy(
            contract,
            "vstore_vpu_sub_bf16_pair",
            "vrf_bf16",
            "vmem_bf16",
            result_dtype,
            contract.record["vpu_sub"]["result_policy"],
            result_registers,
        ),
        _copy(
            contract,
            "dma_store_vpu_sub_bf16_pair",
            "vmem_bf16",
            "external_bf16",
            result_dtype,
            contract.record["vpu_sub"]["result_policy"],
            result_registers,
        ),
        InstructionDescriptor(
            "vpu_square_bf16",
            contract.record["vpu_square"]["operation"],
            ("vrf_bf16",),
            "vrf_bf16",
            result_dtype,
            contract.record["vpu_square"]["result_policy"],
            (2,),
            extent=result_registers,
            input_dtypes=(result_dtype,),
            input_numerical_policies=(result_policy,),
            input_ranks=(2,),
            output_axis_bounds=_tile_bounds(contract),
            input_axis_bounds=(_tile_bounds(contract),),
            input_read_offsets=(2 * rows + 3,),
            completion_offset=2 * rows + 8,
        ),
        _copy(
            contract,
            "vstore_vpu_square_bf16_pair",
            "vrf_bf16",
            "vmem_bf16",
            result_dtype,
            contract.record["vpu_square"]["result_policy"],
            result_registers,
        ),
        _copy(
            contract,
            "dma_store_vpu_square_bf16_pair",
            "vmem_bf16",
            "external_bf16",
            result_dtype,
            contract.record["vpu_square"]["result_policy"],
            result_registers,
        ),
        _copy(
            contract,
            "vstore_fp8",
            "vrf_fp8",
            "vmem_fp8",
            operand_dtype,
            operand_policy,
            1,
        ),
        _copy(
            contract,
            "dma_store_fp8",
            "vmem_fp8",
            "external_fp8",
            operand_dtype,
            operand_policy,
            1,
        ),
    )
    recip = contract.record["vpu_recip"]
    raw_policy = recip["operand_policies"][0]
    descriptors += (
        _copy(
            contract, "dma_load_bf16_raw_pair", "external_bf16", "vmem_bf16",
            result_dtype, raw_policy, result_registers,
        ),
        _copy(
            contract, "vload_bf16_raw_pair", "vmem_bf16", "vrf_bf16",
            result_dtype, raw_policy, result_registers,
        ),
        *(
            InstructionDescriptor(
                f"vpu_recip_bf16_{origin}",
                recip["operation"],
                ("vrf_bf16",),
                "vrf_bf16",
                result_dtype,
                recip["result_policy"],
                (2,),
                extent=result_registers,
                input_dtypes=(result_dtype,),
                input_numerical_policies=(policy,),
                input_ranks=(2,),
                output_axis_bounds=_tile_bounds(contract),
                input_axis_bounds=(_tile_bounds(contract),),
                input_read_offsets=(2 * rows + 3,),
                completion_offset=2 * rows + 8,
            )
            for origin, policy in zip(("raw", "mxu0"), recip["operand_policies"], strict=True)
        ),
        _copy(
            contract, "vstore_vpu_recip_bf16_pair", "vrf_bf16", "vmem_bf16",
            result_dtype, recip["result_policy"], result_registers,
        ),
        _copy(
            contract, "dma_store_vpu_recip_bf16_pair", "vmem_bf16", "external_bf16",
            result_dtype, recip["result_policy"], result_registers,
        ),
    )
    log2 = contract.record["vpu_log2"]
    descriptors += (
        *(
            InstructionDescriptor(
                f"vpu_log2_bf16_{origin}",
                log2["operation"],
                ("vrf_bf16",),
                "vrf_bf16",
                result_dtype,
                log2["result_policy"],
                (2,),
                extent=result_registers,
                input_dtypes=(result_dtype,),
                input_numerical_policies=(policy,),
                input_ranks=(2,),
                output_axis_bounds=_tile_bounds(contract),
                input_axis_bounds=(_tile_bounds(contract),),
                input_read_offsets=(2 * rows + 3,),
                completion_offset=2 * rows + 8,
            )
            for origin, policy in zip(("raw", "mxu0"), log2["operand_policies"], strict=True)
        ),
        _copy(
            contract, "vstore_vpu_log2_bf16_pair", "vrf_bf16", "vmem_bf16",
            result_dtype, log2["result_policy"], result_registers,
        ),
        _copy(
            contract, "dma_store_vpu_log2_bf16_pair", "vmem_bf16", "external_bf16",
            result_dtype, log2["result_policy"], result_registers,
        ),
    )
    sqrt = contract.record["vpu_sqrt"]
    descriptors += (
        *(
            InstructionDescriptor(
                f"vpu_sqrt_bf16_{origin}",
                sqrt["operation"],
                ("vrf_bf16",),
                "vrf_bf16",
                result_dtype,
                sqrt["result_policy"],
                (2,),
                extent=result_registers,
                input_dtypes=(result_dtype,),
                input_numerical_policies=(policy,),
                input_ranks=(2,),
                output_axis_bounds=_tile_bounds(contract),
                input_axis_bounds=(_tile_bounds(contract),),
                input_read_offsets=(2 * rows + 3,),
                completion_offset=2 * rows + 8,
            )
            for origin, policy in zip(("raw", "mxu0"), sqrt["operand_policies"], strict=True)
        ),
        _copy(
            contract, "vstore_vpu_sqrt_bf16_pair", "vrf_bf16", "vmem_bf16",
            result_dtype, sqrt["result_policy"], result_registers,
        ),
        _copy(
            contract, "dma_store_vpu_sqrt_bf16_pair", "vmem_bf16", "external_bf16",
            result_dtype, sqrt["result_policy"], result_registers,
        ),
    )
    exp2 = contract.record["vpu_exp2"]
    descriptors += (
        _copy(
            contract, "dma_load_bf16_exp2_pair", "external_bf16", "vmem_bf16",
            result_dtype, exp2["operand_policy"], result_registers,
        ),
        _copy(
            contract, "vload_bf16_exp2_pair", "vmem_bf16", "vrf_bf16",
            result_dtype, exp2["operand_policy"], result_registers,
        ),
        InstructionDescriptor(
            "vpu_exp2_bf16_bounded",
            exp2["operation"],
            ("vrf_bf16",),
            "vrf_bf16",
            result_dtype,
            exp2["result_policy"],
            (2,),
            extent=result_registers,
            input_dtypes=(result_dtype,),
            input_numerical_policies=(exp2["operand_policy"],),
            input_ranks=(2,),
            output_axis_bounds=_tile_bounds(contract),
            input_axis_bounds=(_tile_bounds(contract),),
            input_read_offsets=(2 * rows + 3,),
            completion_offset=2 * rows + 8,
        ),
        _copy(
            contract, "vstore_vpu_exp2_bf16_pair", "vrf_bf16", "vmem_bf16",
            result_dtype, exp2["result_policy"], result_registers,
        ),
        _copy(
            contract, "dma_store_vpu_exp2_bf16_pair", "vmem_bf16", "external_bf16",
            result_dtype, exp2["result_policy"], result_registers,
        ),
    )
    for kind in ("min", "max"):
        numerical = contract.record[f"vpu_{kind}"]
        descriptors += (
            InstructionDescriptor(
                f"vpu_{kind}_bf16",
                numerical["operation"],
                ("vrf_bf16", "vrf_bf16"),
                "vrf_bf16",
                result_dtype,
                numerical["result_policy"],
                (2,),
                extent=result_registers,
                input_dtypes=(result_dtype, result_dtype),
                input_numerical_policies=(result_policy, result_policy),
                input_ranks=(2, 2),
                output_axis_bounds=_tile_bounds(contract),
                input_axis_bounds=(_tile_bounds(contract), _tile_bounds(contract)),
                input_read_offsets=(2 * rows + 3, 2 * rows + 3),
                completion_offset=2 * rows + 8,
            ),
            _copy(
                contract,
                f"vstore_vpu_{kind}_bf16_pair",
                "vrf_bf16",
                "vmem_bf16",
                result_dtype,
                numerical["result_policy"],
                result_registers,
            ),
            _copy(
                contract,
                f"dma_store_vpu_{kind}_bf16_pair",
                "vmem_bf16",
                "external_bf16",
                result_dtype,
                numerical["result_policy"],
                result_registers,
            ),
        )
    for kind in ("min", "max", "sum"):
        numerical = contract.record[f"vpu_col_{kind}"]
        descriptors += (
            InstructionDescriptor(
                f"vpu_col_{kind}_bf16",
                numerical["operation"],
                ("vrf_bf16",),
                "vrf_bf16",
                result_dtype,
                numerical["result_policy"],
                (2,),
                extent=result_registers,
                input_dtypes=(result_dtype,),
                input_numerical_policies=(result_policy,),
                input_ranks=(2,),
                output_axis_bounds=_physical_col_bounds(contract),
                input_axis_bounds=(_tile_bounds(contract),),
                input_read_offsets=(numerical["diagnostic_delay_cycles"] - 1,),
                completion_offset=numerical["diagnostic_delay_cycles"],
                shape_contract="bounded",
            ),
            _physical_col_copy(
                contract,
                f"vstore_vpu_col_{kind}_bf16_pair",
                "vrf_bf16",
                "vmem_bf16",
                result_dtype,
                numerical["result_policy"],
                result_registers,
            ),
            _physical_col_copy(
                contract,
                f"dma_store_vpu_col_{kind}_bf16_pair",
                "vmem_bf16",
                "external_bf16",
                result_dtype,
                numerical["result_policy"],
                result_registers,
            ),
        )
    for kind in ("min", "max"):
        numerical = contract.record[f"vpu_row_{kind}"]
        descriptors += (
            InstructionDescriptor(
                f"vpu_row_{kind}_bf16",
                numerical["operation"],
                ("vrf_bf16",),
                "vrf_bf16",
                result_dtype,
                numerical["result_policy"],
                (2,),
                extent=result_registers,
                input_dtypes=(result_dtype,),
                input_numerical_policies=(result_policy,),
                input_ranks=(2,),
                output_axis_bounds=_tile_bounds(contract),
                input_axis_bounds=(_tile_bounds(contract),),
                input_read_offsets=(numerical["diagnostic_delay_cycles"] - 1,),
                completion_offset=numerical["diagnostic_delay_cycles"],
            ),
            _copy(
                contract, f"vstore_vpu_row_{kind}_bf16_pair", "vrf_bf16", "vmem_bf16",
                result_dtype, numerical["result_policy"], result_registers,
            ),
            _copy(
                contract, f"dma_store_vpu_row_{kind}_bf16_pair", "vmem_bf16", "external_bf16",
                result_dtype, numerical["result_policy"], result_registers,
            ),
        )
    row_sum = contract.record["vpu_row_sum"]
    descriptors += (
        InstructionDescriptor(
            "vpu_row_sum_bf16",
            row_sum["operation"],
            ("vrf_bf16",),
            "vrf_bf16",
            result_dtype,
            row_sum["result_policy"],
            (2,),
            extent=result_registers,
            input_dtypes=(result_dtype,),
            input_numerical_policies=(result_policy,),
            input_ranks=(2,),
            output_axis_bounds=_tile_bounds(contract),
            input_axis_bounds=(_tile_bounds(contract),),
            input_read_offsets=(row_sum["diagnostic_delay_cycles"] - 1,),
            completion_offset=row_sum["diagnostic_delay_cycles"],
        ),
        _copy(
            contract, "vstore_vpu_row_sum_bf16_pair", "vrf_bf16", "vmem_bf16",
            result_dtype, row_sum["result_policy"], result_registers,
        ),
        _copy(
            contract, "dma_store_vpu_row_sum_bf16_pair", "vmem_bf16", "external_bf16",
            result_dtype, row_sum["result_policy"], result_registers,
        ),
    )
    conversion = contract.record["vfp8_conversion"]
    bf16_input_policy = conversion["input_policy"]
    descriptors += (
        _copy(
            contract,
            "dma_load_bf16_anchor_pair",
            "external_bf16",
            "vmem_bf16",
            "bf16",
            bf16_input_policy,
            result_registers,
        ),
        _copy(
            contract,
            "vload_bf16_anchor_pair",
            "vmem_bf16",
            "vrf_bf16",
            "bf16",
            bf16_input_policy,
            result_registers,
        ),
    )
    for scale_code in conversion["admitted_scale_codes"]:
        packed_policy = conversion["pack_result_policies"][str(scale_code)]
        unpacked_policy = conversion["unpack_result_policies"][str(scale_code)]
        descriptors += (
            InstructionDescriptor(
                f"vpu_pack_e8m0_{scale_code}",
                "vfp8_pack",
                ("vrf_bf16",),
                "vrf_fp8",
                operand_dtype,
                packed_policy,
                (2,),
                required_attrs=(("scale_e8m0", scale_code),),
                input_dtypes=("bf16",),
                input_numerical_policies=(bf16_input_policy,),
                input_ranks=(2,),
                output_axis_bounds=_tile_bounds(contract),
                input_axis_bounds=(_tile_bounds(contract),),
                input_read_offsets=(2 * rows + 3,),
                completion_offset=2 * rows + 8,
            ),
            InstructionDescriptor(
                f"vpu_unpack_e8m0_{scale_code}",
                "vfp8_unpack",
                ("vrf_fp8",),
                "vrf_bf16",
                result_dtype,
                unpacked_policy,
                (2,),
                required_attrs=(("scale_e8m0", scale_code),),
                extent=result_registers,
                input_dtypes=(operand_dtype,),
                input_numerical_policies=(packed_policy,),
                input_ranks=(2,),
                output_axis_bounds=_tile_bounds(contract),
                input_axis_bounds=(_tile_bounds(contract),),
                input_read_offsets=(rows + 3,),
                completion_offset=2 * rows + 8,
            ),
            _copy(
                contract,
                f"vstore_pack_e8m0_{scale_code}",
                "vrf_fp8",
                "vmem_fp8",
                operand_dtype,
                packed_policy,
                1,
            ),
            _copy(
                contract,
                f"dma_store_pack_e8m0_{scale_code}",
                "vmem_fp8",
                "external_fp8",
                operand_dtype,
                packed_policy,
                1,
            ),
            _copy(
                contract,
                f"vstore_unpack_e8m0_{scale_code}_pair",
                "vrf_bf16",
                "vmem_bf16",
                result_dtype,
                unpacked_policy,
                result_registers,
            ),
            _copy(
                contract,
                f"dma_store_unpack_e8m0_{scale_code}_pair",
                "vmem_bf16",
                "external_bf16",
                result_dtype,
                unpacked_policy,
                result_registers,
            ),
        )
    mxu1_reset = next(item for item in descriptors if item.name == "mxu1_matmul_reset")
    descriptors += (
        replace(
            mxu1_reset,
            name="mxu1_matmul_packed_e8m0_127",
            input_numerical_policies=(
                conversion["pack_result_policies"][
                    str(conversion["downstream_mxu1_pack_code"])
                ],
                mxu1["operand_policy"],
            ),
        ),
    )
    banks = (
        StorageBank(
            "external_fp8",
            "dram",
            contract.movement.record["dram_window_bytes"] // tile_bytes,
            "mreg_bytes",
        ),
        StorageBank(
            "external_bf16",
            "dram",
            contract.movement.record["dram_window_bytes"] // tile_bytes,
            "mreg_bytes",
            alignment=result_registers,
        ),
        StorageBank(
            "vmem_fp8",
            "vmem",
            contract.movement.record["vmem_capacity_bytes"] // tile_bytes,
            "mreg_bytes",
        ),
        StorageBank(
            "vmem_bf16",
            "vmem",
            contract.movement.record["vmem_capacity_bytes"] // tile_bytes,
            "mreg_bytes",
            alignment=result_registers,
        ),
        StorageBank(
            "vrf_fp8", "matrix_registers", geometry["matrix_register_count"], "mreg"
        ),
        StorageBank(
            "vrf_bf16",
            "matrix_registers",
            geometry["matrix_register_count"],
            "mreg",
            alignment=result_registers,
        ),
        StorageBank(
            "mxu0_weight", "mxu0_weight_slots", geometry["weight_slots"], "slot"
        ),
        StorageBank(
            "mxu0_accum",
            "mxu0_accumulator_slots",
            geometry["accumulator_slots"],
            "slot",
        ),
        StorageBank(
            "mxu1_weight", "mxu1_weight_slots", geometry["weight_slots"], "slot"
        ),
        StorageBank(
            "mxu1_accum",
            "mxu1_accumulator_slots",
            geometry["accumulator_slots"],
            "slot",
        ),
    )
    return NativeTargetProfile(contract.identity, descriptors, banks)
