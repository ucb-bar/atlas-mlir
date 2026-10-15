#!/usr/bin/env python3
"""Prepare exact finite-normal VSUB witnesses for a native Atlas tile."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import struct
from pathlib import Path

TARGET_IDENTITY = (
    "atlas-rtl-0079c0541111197741a231c002e3843fa6f545b2:"
    "movement-mxu01-vpu-pack-tile-v21"
)
REQUEST_DIGEST = "7498d5c12d4a587970550b3aa72ef563a77fcc61ad3ef65009ec234bad883626"


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--compiled", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--case", required=True, choices=("zero", "chop"))
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
    if (not isinstance(inputs, list) or len(inputs) != 2
            or not isinstance(outputs, list) or len(outputs) != 1
            or not all(type(address) is int and address > 0 and address % 32 == 0
                       for address in (*inputs, *outputs))):
        parser.error("native tile has invalid input or output addresses")
    arithmetic = [row for row in physical.get("instructions", [])
                  if row.get("operation") == "atlas.vpu_binary"]
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
            or plan.get("operation") != "bf16_sub"
            or plan.get("shape") != [32, 32]
            or plan.get("output_bytes") != 2048
            or plan.get("output_physical_layouts") != ["bf16_pair_halves"]
            or plan.get("input_domain") != "finite-normal-exact-fp32-difference-or-zero-v1"
            or len(arithmetic) != 1 or arithmetic[0].get("fields", {}).get("kind") != "sub"
            or physical.get("word_count") != len(program) // 4
            or not program or len(program) % 4 or len(program) > 128 * 1024):
        parser.error("native compilation does not match the BF16 subtraction fixture")
    ranges = [(address, address + 2048) for address in (*inputs, *outputs)]
    if any(left[0] < right[1] and right[0] < left[1]
           for index, left in enumerate(ranges) for right in ranges[index + 1:]):
        parser.error("input and output windows overlap")

    if args.case == "zero":
        # x - x = +0 under selected RNE for these finite-normal operands.
        pairs = ((0x3F80, 0x3F80, 0), (0xBF80, 0xBF80, 0))
    else:
        # 1 - (-3/512) chops to 1, while nearest-even BF16 gives 0x3F81.
        pairs = ((0x3F80, 0xBBC0, 0x3F80),
                 (0xBF80, 0x3BC0, 0xBF80),
                 (0x3F00, 0x3F80, 0xBF00),
                 (0x4000, 0xBF00, 0x4020))
    args.out.mkdir(parents=True)
    shutil.copyfile(program_path, args.out / "program.bin")
    for operand in range(2):
        (args.out / f"input-{operand}.bin").write_bytes(b"".join(
            struct.pack("<H", pairs[index % len(pairs)][operand]) for index in range(1024)
        ))
    (args.out / "expected.bin").write_bytes(b"".join(
        struct.pack("<H", pairs[index % len(pairs)][2]) for index in range(1024)
    ))
    properties = [
        "schema=atlas.rtl_tile_fixture.v1",
        f"name={args.name}",
        "input_count=2",
        "output_count=1",
        f"program_sha256={digest(program_path)}",
        f"compiler_manifest_sha256={digest(manifest_path)}",
        f"execution_plan_sha256={digest(plan_path)}",
        f"input.0.address={inputs[0]:#x}",
        f"input.1.address={inputs[1]:#x}",
        f"output.address={outputs[0]:#x}",
        f"reference=exact_finite_normal_sub_{args.case}_v1",
    ]
    (args.out / "manifest.properties").write_text("\n".join(properties) + "\n")


if __name__ == "__main__":
    main()
