#!/usr/bin/env python3
"""Prepare directed Q/K softmax panels for a checked Merlin-native image."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import struct
from pathlib import Path

from atlas_mlir_native.exp_lane import vexp_bf16_bits
from atlas_mlir_native.recip_lane import vrecip_bf16_bits

TARGET_IDENTITY = (
    "atlas-rtl-0079c0541111197741a231c002e3843fa6f545b2:"
    "movement-mxu01-vpu-pack-tile-v21"
)
REQUEST_DIGEST = "54d13ac24af645cc142ba7137fd5c8d97c4b144b85e5ff5d55c1f4bd850ce44b"


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _bf16_value(code: int) -> float:
    return struct.unpack("<f", struct.pack("<I", code << 16))[0]


def _f32(value: float) -> float:
    return struct.unpack("<f", struct.pack("<f", value))[0]


def _bf16_rne(value: float) -> int:
    raw = struct.unpack("<I", struct.pack("<f", value))[0]
    rounded = (raw + 0x7FFF + ((raw >> 16) & 1)) >> 16
    if not 0 < ((rounded >> 7) & 0xFF) < 255:
        raise ValueError("directed softmax result left finite-normal BF16")
    return rounded


def _tree_sum(codes: tuple[int, ...]) -> int:
    values = [_bf16_value(code) for code in codes]
    while len(values) > 1:
        values = [_f32(values[index] + values[index + 1])
                  for index in range(0, len(values), 2)]
    return _bf16_rne(values[0])


def _panel(case: str) -> tuple[tuple[bytes, bytes, bytes], bytes]:
    if case == "uniform":
        # 32 × (1 × 1/32) gives Q=K=1 in every cell. Every score is 32.
        inputs = (bytes([0x38]) * 1024, bytes([0x10]) * 1024,
                  bytes([0x10]) * 1024)
        scores = tuple(0x4200 for _ in range(1024))
    elif case == "patterned":
        # Q[m,k] is 1/2 or 1. K[n,k] is the source-row factor times a
        # feature factor of 2 at k%4==0 and 1 elsewhere. Its 32 features
        # sum to 40; every score is exactly 40*q_m*source_factor_n.
        tokens = bytes(0x30 if row % 2 == 0 else 0x38
                       for row in range(32) for _column in range(32))
        wk = bytes(0x18 if row % 4 == 0 else 0x10
                   for row in range(32) for _column in range(32))
        inputs = (tokens, bytes([0x10]) * 1024, wk)
        scores = tuple(_bf16_rne(40 * (0.5 if row % 2 == 0 else 1)
                                * (0.5 if column % 2 == 0 else 1))
                       for row in range(32) for column in range(32))
    else:
        raise ValueError("unsupported directed softmax case")

    logical = []
    for row in range(32):
        score_row = scores[row * 32:(row + 1) * 32]
        maximum = max(score_row)
        centered = tuple(0 if code == maximum else
                         _bf16_rne(_f32(_bf16_value(code) - _bf16_value(maximum)))
                         for code in score_row)
        exponential = tuple(vexp_bf16_bits(code) for code in centered)
        total = _tree_sum(exponential)
        if not 0x3F80 <= total <= 0x4200:
            raise ValueError("directed row sum left reciprocal input domain")
        inverse = vrecip_bf16_bits(total)
        logical.extend(_bf16_rne(_f32(_bf16_value(code) * _bf16_value(inverse)))
                       for code in exponential)
    physical = tuple(logical[row * 32 + column + half * 16]
                     for half in range(2) for row in range(32) for column in range(16))
    return inputs, struct.pack("<1024H", *physical)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--compiled", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--case", required=True, choices=("uniform", "patterned"))
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
    kinds = [(row.get("operation"), row.get("fields", {}).get("kind"))
             for row in physical.get("instructions", [])
             if row.get("operation") in {"atlas.vpu_binary", "atlas.vpu_unary",
                                          "atlas.vpu_reduce"}]
    expected_kinds = [
        ("atlas.vpu_reduce", "row_max"), ("atlas.vpu_binary", "sub"),
        ("atlas.vpu_unary", "exp"), ("atlas.vpu_reduce", "row_sum"),
        ("atlas.vpu_unary", "recip"), ("atlas.vpu_binary", "mul"),
    ]
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
            or plan.get("operation") != "mxu0_bf16_dag"
            or plan.get("shape") != [32, 32]
            or plan.get("input_payload_bytes") != [1024] * 3
            or plan.get("output_bytes") != 2048
            or plan.get("output_physical_layouts") != ["bf16_pair_halves"]
            or plan.get("output_numerical_policies") != ["bf16-product-rne-v1"]
            or not isinstance(inputs, list) or len(inputs) != 3
            or not isinstance(outputs, list) or len(outputs) != 1
            or not all(type(address) is int and address > 0 and address % 32 == 0
                       for address in (*inputs, *outputs))
            or plan.get("weight_addresses") != [inputs[1], inputs[2], None]
            or len(plan.get("interlayer_pack_scale_codes", [])) != 2
            or any(row.get("scale_code") != 127
                   for row in plan["interlayer_pack_scale_codes"])
            or operations.count("atlas.mxu_matmul") != 3
            or operations.count("atlas.vpu_pack") != 2
            or kinds != expected_kinds
            or physical.get("word_count") != len(program) // 4
            or not program or len(program) % 4 or len(program) > 128 * 1024):
        parser.error("native compilation does not match the Q/K softmax fixture")
    windows = [(address, address + 1024) for address in inputs]
    windows.append((outputs[0], outputs[0] + 2048))
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
        "input_count=3",
        "output_count=1",
        f"program_sha256={_digest(program_path)}",
        f"compiler_manifest_sha256={_digest(manifest_path)}",
        f"execution_plan_sha256={_digest(plan_path)}",
        f"output.address={outputs[0]:#x}",
        f"reference=directed_qk_softmax_{args.case}_v1",
    ]
    properties.extend(f"input.{index}.address={address:#x}"
                      for index, address in enumerate(inputs))
    (args.out / "manifest.properties").write_text("\n".join(properties) + "\n")


if __name__ == "__main__":
    main()
