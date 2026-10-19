"""Build a selected-core MXU1 fixture from a verified sparse constant tile.

The expected bytes use exact single-product arithmetic. This deliberately
does not model MXU1 accumulation or approve the source precision policy.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import struct
from pathlib import Path

SELECTED_RTL = "0079c0541111197741a231c002e3843fa6f545b2"
FINITE_POLICY = "mxu1-e4m3-anchor-bf16-finite-reset-v1"


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def bf16_power_of_two(code: int) -> bytes:
    exponent = (code >> 3) & 15
    if code & 0x87 or not 1 <= exponent <= 14:
        raise ValueError("sparse weight must use positive normal power-of-two E4M3 codes")
    bits = struct.unpack("<I", struct.pack("<f", 2.0 ** (exponent - 7)))[0]
    assert bits & 0xffff == 0
    return (bits >> 16).to_bytes(2, "little")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--compiled", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--name", required=True)
    parser.add_argument("--pattern", choices=("all_ones", "one_hot"), required=True)
    args = parser.parse_args()
    if args.out.exists():
        parser.error(f"output directory must be fresh: {args.out}")
    if re.fullmatch(r"[a-z][a-z0-9_]*", args.name) is None:
        parser.error("fixture name must use lowercase letters, digits, and underscores")

    root = args.compiled
    manifest_bytes = (root / "manifest.json").read_bytes()
    plan_bytes = (root / "execution_plan.json").read_bytes()
    program = (root / "program.bin").read_bytes()
    weight = (root / "constants/rhs.bin").read_bytes()
    manifest = json.loads(manifest_bytes)
    plan = json.loads(plan_bytes)
    constant_rows = manifest.get("constant_inputs")
    if (
        not str(manifest.get("target_identity", "")).startswith(f"atlas-rtl-{SELECTED_RTL}:")
        or manifest.get("engine") != "merlin_native"
        or manifest.get("binary_sha256") != digest(program)
        or manifest.get("execution_plan_sha256") != digest(plan_bytes)
        or not isinstance(constant_rows, list) or len(constant_rows) != 1
        or constant_rows[0].get("role") != "rhs"
        or constant_rows[0].get("path") != "constants/rhs.bin"
        or constant_rows[0].get("status") != "packed"
        or constant_rows[0].get("sha256") != digest(weight)
        or constant_rows[0].get("bytes") != 1024
        or plan.get("schema") != "atlas.native_tile_execution_plan.v8"
        or plan.get("operation") != "mxu1_reset"
        or plan.get("numerical_policy") != FINITE_POLICY
        or plan.get("shape") != [32, 32]
        or plan.get("input_bytes") != 1024
        or plan.get("output_bytes") != 2048
        or plan.get("output_physical_layouts") != ["bf16_pair_halves"]
        or len(weight) != 1024
        or not program or len(program) % 4 or len(program) > 128 * 1024
    ):
        parser.error(
            "compiled tile, plan, or constant identity is outside the selected MXU1 fixture"
        )
    inputs = plan.get("input_addresses")
    output = plan.get("output_address")
    if (
        not isinstance(inputs, list) or len(inputs) != 2
        or any(type(address) is not int or address <= 0 or address % 32 for address in inputs)
        or inputs[0] == inputs[1]
        or type(output) is not int or output <= 0 or output % 32
        or constant_rows[0].get("address") != inputs[1]
        or any(address < output + 2048 and output < address + 1024 for address in inputs)
    ):
        parser.error("compiled tile has invalid or overlapping physical input/output addresses")

    weight_positions = []
    for row in range(32):
        active = [(column, code) for column, code in enumerate(weight[row * 32:(row + 1) * 32])
                  if code]
        if len(active) != 1:
            parser.error("each physical weight row must have exactly one nonzero code")
        try:
            bf16_power_of_two(active[0][1])
        except ValueError as error:
            parser.error(str(error))
        weight_positions.append(active[0])

    activation = bytes(
        0x38 if args.pattern == "all_ones" or row == column else 0
        for row in range(32) for column in range(32)
    )
    expected = bytearray(2048)
    for row in range(32):
        for column, (weight_k, weight_code) in enumerate(weight_positions):
            if args.pattern == "all_ones" or row == weight_k:
                physical = (0 if column < 16 else 1024) + row * 32 + (column % 16) * 2
                expected[physical:physical + 2] = bf16_power_of_two(weight_code)

    args.out.mkdir(parents=True)
    (args.out / "program.bin").write_bytes(program)
    (args.out / "input-0.bin").write_bytes(activation)
    (args.out / "input-1.bin").write_bytes(weight)
    (args.out / "expected.bin").write_bytes(expected)
    properties = (
        "schema=atlas.rtl_tile_fixture.v1",
        f"name={args.name}",
        "input_count=2",
        f"program_sha256={digest(program)}",
        f"input.0.address={inputs[0]:#x}",
        f"input.1.address={inputs[1]:#x}",
        f"output.address={output:#x}",
        f"constant_sha256={digest(weight)}",
        f"compiler_manifest_sha256={digest(manifest_bytes)}",
        f"reference=sparse_power_two_single_product_{args.pattern}_v1",
    )
    (args.out / "manifest.properties").write_text("\n".join(properties) + "\n")


if __name__ == "__main__":
    main()
