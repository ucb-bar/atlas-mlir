"""Reviewed MLIR machine-IR view of the selected Atlas native descriptor catalog.

This is a consumer of the existing selected contracts and profiles, not an ISA
description.  A change to the source catalog changes the generated signatures
and the coverage manifest.  Address legality, shape relations, numerical
preconditions, and temporal scheduling remain in Merlin's native compiler.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.resources
import json
import re
from pathlib import Path

from merlin.common.schemas import validate
from merlin.semantic_compiler.rules import InstructionDescriptor
from merlin.targetgen.contract.toolchain import LLVM_COMMIT, LLVM_VERSION
from merlin.targetgen.generate import mlir_scaffold

from .movement import MovementContract, movement_profile
from .mxu0 import Mxu0Contract, mxu0_profile

# Authored interface selection: these are the profile operations whose physical
# input/output roles have been reviewed for the current diagnostic machine IR.
# Keeping this list explicit avoids turning every new descriptor into an MLIR
# operation without review.  No instruction semantics or encoding lives here.
_REVIEWED = (
    "dma_load_wait",
    "dma_store_wait",
    "dma_load_fp8",
    "vload_fp8",
    "mxu0_push_weight",
    "mxu0_matmul_reset",
    "mxu0_matmul_accumulate",
    "mxu0_pop_bf16",
    "vstore_bf16_pair",
    "dma_store_bf16_pair",
    "vstore_fp8",
    "dma_store_fp8",
    "mxu1_dma_load_fp8",
    "mxu1_vload_fp8",
    "mxu1_push_weight",
    "mxu1_matmul_reset",
    "mxu1_pop_bf16",
    "mxu1_vstore_bf16_pair",
    "mxu1_dma_store_bf16_pair",
    "xlu_transpose_fp8",
    "vpu_relu_bf16",
    "vpu_add_bf16",
    "vstore_vpu_add_bf16_pair",
    "dma_store_vpu_add_bf16_pair",
    "vpu_mul_bf16",
    "vstore_vpu_mul_bf16_pair",
    "dma_store_vpu_mul_bf16_pair",
    "vpu_sub_bf16",
    "vstore_vpu_sub_bf16_pair",
    "dma_store_vpu_sub_bf16_pair",
    "vpu_square_bf16",
    "vstore_vpu_square_bf16_pair",
    "dma_store_vpu_square_bf16_pair",
    "dma_load_bf16_raw_pair",
    "vload_bf16_raw_pair",
    "vpu_recip_bf16_raw",
    "vpu_recip_bf16_mxu0",
    "vstore_vpu_recip_bf16_pair",
    "dma_store_vpu_recip_bf16_pair",
    "vpu_log2_bf16_raw",
    "vpu_log2_bf16_mxu0",
    "vstore_vpu_log2_bf16_pair",
    "dma_store_vpu_log2_bf16_pair",
    "vpu_sqrt_bf16_raw",
    "vpu_sqrt_bf16_mxu0",
    "vstore_vpu_sqrt_bf16_pair",
    "dma_store_vpu_sqrt_bf16_pair",
    "dma_load_bf16_exp2_pair",
    "vload_bf16_exp2_pair",
    "vpu_exp2_bf16_bounded",
    "vstore_vpu_exp2_bf16_pair",
    "dma_store_vpu_exp2_bf16_pair",
    "vpu_min_bf16",
    "vstore_vpu_min_bf16_pair",
    "dma_store_vpu_min_bf16_pair",
    "vpu_max_bf16",
    "vstore_vpu_max_bf16_pair",
    "dma_store_vpu_max_bf16_pair",
    "vpu_col_min_bf16",
    "vstore_vpu_col_min_bf16_pair",
    "dma_store_vpu_col_min_bf16_pair",
    "vpu_col_max_bf16",
    "vstore_vpu_col_max_bf16_pair",
    "dma_store_vpu_col_max_bf16_pair",
    "vpu_col_sum_bf16",
    "vstore_vpu_col_sum_bf16_pair",
    "dma_store_vpu_col_sum_bf16_pair",
    "vpu_row_min_bf16",
    "vstore_vpu_row_min_bf16_pair",
    "dma_store_vpu_row_min_bf16_pair",
    "vpu_row_max_bf16",
    "vstore_vpu_row_max_bf16_pair",
    "dma_store_vpu_row_max_bf16_pair",
    "vpu_row_sum_bf16",
    "vstore_vpu_row_sum_bf16_pair",
    "dma_store_vpu_row_sum_bf16_pair",
    "dma_load_bf16_anchor_pair",
    "vload_bf16_anchor_pair",
    "vpu_pack_e8m0_126",
    "vpu_unpack_e8m0_126",
    "vstore_pack_e8m0_126",
    "dma_store_pack_e8m0_126",
    "vstore_unpack_e8m0_126_pair",
    "dma_store_unpack_e8m0_126_pair",
    "vpu_pack_e8m0_127",
    "vpu_unpack_e8m0_127",
    "vstore_pack_e8m0_127",
    "dma_store_pack_e8m0_127",
    "vstore_unpack_e8m0_127_pair",
    "dma_store_unpack_e8m0_127_pair",
    "vpu_pack_e8m0_128",
    "vpu_unpack_e8m0_128",
    "vstore_pack_e8m0_128",
    "dma_store_pack_e8m0_128",
    "vstore_unpack_e8m0_128_pair",
    "dma_store_unpack_e8m0_128_pair",
    "mxu1_matmul_packed_e8m0_127",
)


def _identifier(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")


def _port_type(storage: str, dtype: str, policy: str) -> str:
    return _identifier(f"{storage}_{dtype}_{policy}")


def _descriptor_signature(descriptor: InstructionDescriptor) -> tuple[dict, set[str]]:
    input_types = tuple(
        _port_type(storage, dtype, policy)
        for storage, dtype, policy in zip(
            descriptor.input_storages,
            descriptor.input_dtypes,
            descriptor.input_numerical_policies,
            strict=True,
        )
    )
    output_type = _port_type(
        descriptor.output_storage, descriptor.output_dtype, descriptor.numerical_policy
    )
    signature = {
        "operands": [
            {"name": f"source{index}", "type": f"!atlas.{kind}"}
            for index, kind in enumerate(input_types)
        ],
        "results": [{"name": "result", "type": f"!atlas.{output_type}"}],
        "attributes": [
            {"name": f"source{index}Address", "type": "i64"}
            for index in range(len(input_types))
        ]
        + [{"name": "resultAddress", "type": "i64"}],
        # The typed MLIR generator owns effects as part of each operation's
        # checked signature. A selected machine operation reads and writes
        # physical state even when its semantic value is pure.
        "effects": ["read", "write"],
    }
    return signature, {*input_types, output_type}


def selected_dialect_plan() -> tuple[dict, dict]:
    """Return the authored plan and exact descriptor-coverage manifest."""
    movement = MovementContract.load()
    tensor = Mxu0Contract.load()
    requirements_path = importlib.resources.files("atlas_native_support").joinpath(
        "requirements.json"
    )
    requirements = json.loads(requirements_path.read_text())
    if requirements.get("act_required") or not requirements.get("merlin_revision"):
        raise ValueError("selected native compiler dependency pin is missing")
    descriptors = {
        descriptor.name: descriptor
        for profile in (movement_profile(movement), mxu0_profile(tensor))
        for descriptor in profile.descriptors
    }
    if len(descriptors) != sum(
        len(profile.descriptors)
        for profile in (movement_profile(movement), mxu0_profile(tensor))
    ):
        raise ValueError("selected Atlas descriptor names are not unique")
    missing = set(_REVIEWED) - set(descriptors)
    if missing:
        raise ValueError(
            f"reviewed Atlas dialect operation disappeared: {sorted(missing)}"
        )

    ops: list[dict] = []
    type_names: set[str] = set()
    type_identities: dict[str, tuple[str, str, str]] = {}
    for name in _REVIEWED:
        descriptor = descriptors[name]
        ports = [
            *zip(
                descriptor.input_storages,
                descriptor.input_dtypes,
                descriptor.input_numerical_policies,
                strict=True,
            ),
            (
                descriptor.output_storage,
                descriptor.output_dtype,
                descriptor.numerical_policy,
            ),
        ]
        for storage, dtype, policy in ports:
            token = _port_type(storage, dtype, policy)
            identity = (storage, dtype, policy)
            if token in type_identities and type_identities[token] != identity:
                raise ValueError(f"Atlas MLIR type token collision: {token}")
            type_identities[token] = identity
        signature, used_types = _descriptor_signature(descriptor)
        type_names.update(used_types)
        ops.append(
            {
                "name": name,
                "summary": f"Selected Atlas {descriptor.computation} in physical {descriptor.output_storage}",
                "signature": signature,
            }
        )
    plan = {
        "target": "atlas",
        "dialect_name": "atlas",
        "ops": ops,
        "types": [
            {
                "name": name,
                "parameters": [],
                "summary": f"Selected Atlas {name} physical value",
            }
            for name in sorted(type_names)
        ],
        "lowering": [],
        "tests": [{"lit": "selected_machine_ir_roundtrip"}],
    }
    problems = validate(plan, "dialect_plan")
    if problems:
        raise ValueError(f"selected Atlas dialect_plan is invalid: {problems}")
    manifest = {
        "schema": "atlas.generated_dialect_view.v1",
        "movement_contract_identity": movement.identity,
        "tensor_contract_identity": tensor.identity,
        "rtl_revision": tensor.record["rtl_revision"],
        "merlin_revision": requirements["merlin_revision"],
        "llvm_commit": LLVM_COMMIT,
        "llvm_version": LLVM_VERSION,
        "covered_descriptors": list(_REVIEWED),
        "omitted_descriptors": {
            name: "physical MLIR interface not reviewed in this consumer view"
            for name in sorted(set(descriptors) - set(_REVIEWED))
        },
        "omitted_families": [
            "scalar/control/CSR",
            "unselected DMA modes",
            "VPU modes beyond admitted BF16 subset",
            "XLU modes beyond FP8 transpose",
            "scale/pack modes beyond admitted E8M0 anchor subset",
        ],
        "unrepresented_obligations": [
            "shape/index-map verifier",
            "numerical-domain verifier",
            "address range/alignment verifier",
            "temporal dependency verifier",
            "instruction encoding",
            "LLVM lowering",
        ],
        "source_profile": "diagnostic_execution_unqualified_timing",
    }
    return plan, manifest


def generate_dialect(
    destination: Path, *, target_source: Path, software_spec: Path
) -> dict:
    """Write a fresh generated MLIR/C++ package under an invocation-owned root."""
    if destination.exists():
        raise FileExistsError(f"fresh dialect output required: {destination}")
    Mxu0Contract.load().verify_source(target_source, software_spec)
    plan, manifest = selected_dialect_plan()
    artifacts = mlir_scaffold.generate(plan)
    destination.mkdir(parents=True)
    for artifact in artifacts:
        artifact.write(destination)
    (destination / "dialect_plan.json").write_text(
        json.dumps(plan, sort_keys=True, indent=2) + "\n", encoding="utf-8"
    )
    manifest["dialect_plan_sha256"] = hashlib.sha256(
        json.dumps(plan, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    (destination / "coverage.json").write_text(
        json.dumps(manifest, sort_keys=True, indent=2) + "\n", encoding="utf-8"
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate the selected Atlas typed MLIR view"
    )
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--target-source", required=True, type=Path)
    parser.add_argument("--software-spec", required=True, type=Path)
    args = parser.parse_args()
    manifest = generate_dialect(
        args.output, target_source=args.target_source, software_spec=args.software_spec
    )
    print(
        json.dumps(
            {"status": "PASS", "output": str(args.output), **manifest}, sort_keys=True
        )
    )


if __name__ == "__main__":
    main()
