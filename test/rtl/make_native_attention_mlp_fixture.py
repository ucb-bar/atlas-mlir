#!/usr/bin/env python3
"""Prepare exact two-output attention and MLP panels for a native Atlas image."""

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
REQUEST_DIGEST = "6b1a223ec9eff1204ae49567e88833f318fbf9e5df24a9f8b9f44974553b7f5b"


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _panel(case: str) -> tuple[tuple[bytes, ...], bytes]:
    if case not in {"uniform", "value_columns"}:
        raise ValueError("unsupported attention and MLP panel")
    token = bytes([0x38]) * 1024
    projection_weight = bytes([0x10]) * 1024
    value_weight = (projection_weight if case == "uniform" else
                    bytes(0x10 if feature % 2 == 0 else 0x08
                          for feature in range(32) for _ in range(32)))
    identity_weight = bytes(0x38 if row == column else 0
                            for row in range(32) for column in range(32))
    # Q and K are 1, so every probability is exactly 1/32. The value
    # projection is 1 or 1/2 by feature. Both MLP weights are identity
    # matrices, and the positive ReLU leaves each feature unchanged.
    logical = tuple(0x3F80 if case == "uniform" or feature % 2 == 0 else 0x3F00
                    for _row in range(32) for feature in range(32))
    physical = tuple(logical[row * 32 + column + half * 16]
                     for half in range(2) for row in range(32) for column in range(16))
    expected_tile = struct.pack("<1024H", *physical)
    return (token, projection_weight, projection_weight, value_weight,
            identity_weight, identity_weight), expected_tile * 2


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--compiled", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--case", required=True, choices=("uniform", "value_columns"))
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
    if (manifest.get("engine") != "merlin_native"
            or manifest.get("artifact_kind") != "program"
            or manifest.get("compile_mode") != "diagnostic"
            or manifest.get("qualification_scope") != "diagnostic_only"
            or manifest.get("request_digest") != REQUEST_DIGEST
            or manifest.get("target_identity") != TARGET_IDENTITY
            or manifest.get("binary_sha256") != _digest(program_path)
            or manifest.get("execution_plan_sha256") != _digest(plan_path)
            or manifest.get("physical_program_sha256") != _digest(physical_path)
            or physical.get("selected_rtl_revision")
               != "0079c0541111197741a231c002e3843fa6f545b2"
            or plan.get("operation") != "mxu0_bf16_dag"
            or plan.get("shape") != [32, 32]
            or plan.get("input_payload_bytes") != [1024] * 6
            or plan.get("output_bytes") != 2048
            or plan.get("output_physical_layouts") != ["bf16_pair_halves"] * 2
            or plan.get("output_numerical_policies") != [
                "bf16-sign-clamp-v1", "mxu0-e4m3-ordered-bf16-finite-v1"
            ]
            or plan.get("transpose_source_nodes") != [
                {"result": "value_transposed", "source": "value_fp8"}
            ]
            or plan.get("weight_source_nodes") != [
                "arg1", "arg2", "op10r0", "value_weight", "value_transposed",
                "mlp_weight1", "mlp_weight2",
            ]
            or len(plan.get("interlayer_pack_scale_codes", [])) != 6
            or any(row.get("scale_code") != 127
                   for row in plan["interlayer_pack_scale_codes"])
            or operations.count("atlas.mxu_matmul") != 7
            or operations.count("atlas.vpu_pack") != 6
            or operations.count("atlas.xlu_transpose") != 1
            or physical.get("word_count") != len(program) // 4
            or not program or len(program) % 4 or len(program) > 128 * 1024
            or not isinstance(inputs, list) or len(inputs) != 6
            or not isinstance(outputs, list) or len(outputs) != 2
            or outputs[1] != outputs[0] + 2048
            or not all(type(address) is int and address > 0 and address % 32 == 0
                       for address in (*inputs, *outputs))):
        parser.error("native compilation does not match the attention and MLP fixture")
    windows = [(address, address + 1024) for address in inputs]
    windows.append((outputs[0], outputs[0] + 4096))
    if any(left[0] < right[1] and right[0] < left[1]
           for index, left in enumerate(windows) for right in windows[index + 1:]):
        parser.error("input and output windows overlap")

    payloads, expected = _panel(args.case)
    args.out.mkdir(parents=True)
    shutil.copyfile(program_path, args.out / "program.bin")
    for index, payload in enumerate(payloads):
        (args.out / f"input-{index}.bin").write_bytes(payload)
    (args.out / "expected.bin").write_bytes(expected)
    properties = [
        "schema=atlas.rtl_tile_fixture.v1",
        f"name={args.name}",
        "input_count=6",
        "output_count=2",
        f"program_sha256={_digest(program_path)}",
        f"compiler_manifest_sha256={_digest(manifest_path)}",
        f"execution_plan_sha256={_digest(plan_path)}",
        f"output.address={outputs[0]:#x}",
        f"reference=directed_native_attention_mlp_{args.case}_v1",
    ]
    properties.extend(f"input.{index}.address={address:#x}"
                      for index, address in enumerate(inputs))
    (args.out / "manifest.properties").write_text("\n".join(properties) + "\n")


if __name__ == "__main__":
    main()
