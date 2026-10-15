#!/usr/bin/env python3
"""Prepare independent Q/K-to-score inputs for the selected AtlasCore test."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
from pathlib import Path

TARGET_IDENTITY = (
    "atlas-rtl-0079c0541111197741a231c002e3843fa6f545b2:"
    "movement-mxu01-vpu-pack-tile-v21"
)
REQUEST_DIGEST = "d204cf30682aef0c47a067df61b38d22ba4e37a469ea59e55cca94de087fb108"


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--compiled", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--case", required=True, choices=("one", "two"))
    parser.add_argument("--name", required=True)
    args = parser.parse_args()
    if args.out.exists():
        parser.error(f"output directory must be fresh: {args.out}")
    if re.fullmatch(r"[a-z][a-z0-9_]*", args.name) is None:
        parser.error("fixture name must use lowercase letters, digits, and underscores")

    manifest_path = args.compiled / "manifest.json"
    plan_path = args.compiled / "execution_plan.json"
    physical_path = args.compiled / "physical-program.json"
    program_path = args.compiled / "program.bin"
    manifest = json.loads(manifest_path.read_text())
    plan = json.loads(plan_path.read_text())
    physical = json.loads(physical_path.read_text())
    program = program_path.read_bytes()
    inputs = plan.get("input_addresses")
    outputs = plan.get("output_addresses")
    operations = [row.get("operation") for row in physical.get("instructions", [])]
    if (not isinstance(inputs, list) or len(inputs) != 3
            or not isinstance(outputs, list) or len(outputs) != 1
            or not all(type(address) is int and address > 0 and address % 32 == 0
                       for address in (*inputs, *outputs))):
        parser.error("native tile has invalid input or output addresses")
    if (manifest.get("engine") != "merlin_native"
            or manifest.get("artifact_kind") != "program"
            or manifest.get("compile_mode") != "diagnostic"
            or manifest.get("qualification_scope") != "diagnostic_only"
            or manifest.get("request_digest") != REQUEST_DIGEST
            or manifest.get("target_identity") != TARGET_IDENTITY
            or manifest.get("binary_sha256") != digest(program_path)
            or manifest.get("execution_plan_sha256") != digest(plan_path)
            or manifest.get("physical_program_sha256") != digest(physical_path)
            or plan.get("schema") != "atlas.native_tile_execution_plan.v8"
            or plan.get("operation") != "mxu0_bf16_dag"
            or plan.get("shape") != [32, 32]
            or plan.get("input_payload_bytes") != [1024] * 3
            or plan.get("output_bytes") != 2048
            or plan.get("output_physical_layouts") != ["bf16_pair_halves"]
            or plan.get("output_numerical_policies") != ["mxu0-e4m3-ordered-bf16-finite-v1"]
            or plan.get("weight_addresses") != [inputs[1], inputs[2], None]
            or len(plan.get("weight_source_nodes", [])) != 3
            or plan["weight_source_nodes"][:2] != ["arg1", "arg2"]
            or len(plan.get("interlayer_pack_scale_codes", [])) != 2
            or any(row.get("scale_code") != 127
                   for row in plan["interlayer_pack_scale_codes"])
            or physical.get("word_count") != len(program) // 4
            or operations.count("atlas.mxu_matmul") != 3
            or operations.count("atlas.vpu_pack") != 2
            or not program or len(program) % 4 or len(program) > 128 * 1024):
        parser.error("native compilation does not match the Q/K RTL fixture")
    ranges = [(address, address + 1024) for address in inputs]
    ranges.append((outputs[0], outputs[0] + 2048))
    if any(left[0] < right[1] and right[0] < left[1]
           for index, left in enumerate(ranges) for right in ranges[index + 1:]):
        parser.error("input and output windows overlap")

    k_weight = 0x10 if args.case == "one" else 0x18
    score = 0x4200 if args.case == "one" else 0x4280
    # Each uniform Q dot product is 32*(1*1/32)=1. K is 1 or 2, so
    # every score cell is 32 or 64 after the two exact unit-scale packs.
    payloads = (bytes([0x38]) * 1024, bytes([0x10]) * 1024,
                bytes([k_weight]) * 1024)
    args.out.mkdir(parents=True)
    shutil.copyfile(program_path, args.out / "program.bin")
    for index, payload in enumerate(payloads):
        (args.out / f"input-{index}.bin").write_bytes(payload)
    (args.out / "expected.bin").write_bytes(score.to_bytes(2, "little") * 1024)
    properties = [
        "schema=atlas.rtl_tile_fixture.v1",
        f"name={args.name}",
        "input_count=3",
        "output_count=1",
        f"program_sha256={digest(program_path)}",
        f"compiler_manifest_sha256={digest(manifest_path)}",
        f"execution_plan_sha256={digest(plan_path)}",
        f"output.address={outputs[0]:#x}",
        f"reference=uniform_qk_pack_{args.case}_v1",
    ]
    properties.extend(f"input.{index}.address={address:#x}"
                      for index, address in enumerate(inputs))
    (args.out / "manifest.properties").write_text("\n".join(properties) + "\n")


if __name__ == "__main__":
    main()
