#!/usr/bin/env python3
"""Build finite BF16 VEXP panels for a checked Merlin-native Atlas image."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import struct
from pathlib import Path

from atlas_mlir_native.exp_lane import vexp_bf16_bits

TARGET_IDENTITY = (
    "atlas-rtl-0079c0541111197741a231c002e3843fa6f545b2:"
    "movement-mxu01-vpu-pack-tile-v21"
)
REQUEST_DIGEST = "ca912168d95976904ecf00990cb667cb795cc3662a962de53b2e904890566fa2"


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _panel(phase: int) -> tuple[bytes, bytes]:
    # Ten significands at fifteen exponents, both signs, plus signed zero.
    # The upper exponent contributes only values within the admitted bound.
    fractions = (0, 1, 3, 7, 16, 31, 32, 63, 95, 127)
    codes = (0, 0x8000) + tuple(
        sign | (exponent << 7) | fraction
        for sign in (0, 0x8000)
        for exponent in range(119, 134)
        for fraction in fractions
        if exponent < 133 or fraction <= 32
    )
    assert len(codes) == len(set(codes)) == 296
    logical = tuple(codes[(position * 131 + phase * 47) % len(codes)]
                    for position in range(1024))
    physical = tuple(logical[row * 32 + column + half * 16]
                     for half in range(2) for row in range(32) for column in range(16))
    return (struct.pack("<1024H", *physical),
            struct.pack("<1024H", *(vexp_bf16_bits(code) for code in physical)))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--compiled", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--phase", required=True, type=int, choices=(0, 1))
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
    exponentials = [row for row in physical.get("instructions", [])
                    if row.get("operation") == "atlas.vpu_unary"]
    if (manifest.get("engine") != "merlin_native"
            or manifest.get("artifact_kind") != "program"
            or manifest.get("compile_mode") != "diagnostic"
            or manifest.get("qualification_scope") != "diagnostic_only"
            or manifest.get("request_digest") != REQUEST_DIGEST
            or manifest.get("target_identity") != TARGET_IDENTITY
            or manifest.get("binary_sha256") != _digest(program_path)
            or manifest.get("execution_plan_sha256") != _digest(plan_path)
            or manifest.get("physical_program_sha256") != _digest(physical_path)
            or plan.get("schema") != "atlas.native_tile_execution_plan.v8"
            or plan.get("operation") != "bf16_exp"
            or plan.get("shape") != [32, 32]
            or plan.get("output_bytes") != 2048
            or plan.get("output_physical_layouts") != ["bf16_pair_halves"]
            or plan.get("input_domain") != "bf16-normal-minus80-plus80-or-signed-zero-v1"
            or not isinstance(inputs, list) or len(inputs) != 1
            or not isinstance(outputs, list) or len(outputs) != 1
            or not all(type(address) is int and address > 0 and address % 32 == 0
                       for address in (*inputs, *outputs))
            or inputs[0] < outputs[0] + 2048 and outputs[0] < inputs[0] + 2048
            or len(exponentials) != 1
            or exponentials[0].get("fields", {}).get("kind") != "exp"
            or physical.get("word_count") != len(program) // 4
            or not program or len(program) % 4 or len(program) > 128 * 1024):
        parser.error("native compilation does not match the BF16 VEXP fixture")

    source, expected = _panel(args.phase)
    args.out.mkdir(parents=True)
    shutil.copyfile(program_path, args.out / "program.bin")
    (args.out / "input-0.bin").write_bytes(source)
    (args.out / "expected.bin").write_bytes(expected)
    properties = [
        "schema=atlas.rtl_tile_fixture.v1",
        f"name={args.name}",
        "input_count=1",
        "output_count=1",
        f"program_sha256={_digest(program_path)}",
        f"compiler_manifest_sha256={_digest(manifest_path)}",
        f"execution_plan_sha256={_digest(plan_path)}",
        f"input.0.address={inputs[0]:#x}",
        f"output.address={outputs[0]:#x}",
        f"reference=selected_exp_lane_q9p12_lut_port_phase_{args.phase}",
    ]
    (args.out / "manifest.properties").write_text("\n".join(properties) + "\n")


if __name__ == "__main__":
    main()
