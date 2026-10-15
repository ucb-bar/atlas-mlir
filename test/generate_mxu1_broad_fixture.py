"""Create a deterministic finite-normal MXU1 lane fixture outside the source tree."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import struct
from pathlib import Path

from mxu1_anchor_source_model import mxu1_reset_lane
from test_mxu_numerics import fp8, round_bf16

SEED = 0xA71A66
PALETTE = (0,) + tuple(
    sign | (exponent << 3) | mantissa
    for sign in (0, 0x80)
    for exponent in range(1, 15)
    for mantissa in range(8)
)
SELECTED_RTL = "0079c0541111197741a231c002e3843fa6f545b2"
FINITE_POLICY = "mxu1-e4m3-anchor-bf16-finite-reset-v1"


def digest(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def build_fixture(directory: Path) -> dict:
    """Write two FP8 tiles and source-mirror/exact BF16 result tiles."""
    rng = random.Random(SEED)

    def operand() -> bytearray:
        values = list(PALETTE) * 4
        values.extend(rng.choice(PALETTE) for _ in range(1024 - len(values)))
        rng.shuffle(values)
        return bytearray(values)

    activation = operand()
    weight = operand()
    # This cell distinguishes anchor alignment from an exact 32-product sum.
    activation[:32] = bytes([0x77, 0x77] + [0x08] * 30)
    weight[:32] = bytes([0x77, 0xF7] + [0x08] * 30)
    for values in (activation, weight):
        for index, code in enumerate(sorted(set(PALETTE) - set(values)), start=32):
            values[index] = code
        if set(values) != set(PALETTE):
            raise AssertionError("fixture does not cover all finite-normal codes")

    source = bytearray()
    exact = bytearray()
    truncated_products = 0
    mismatches = []
    for row in range(32):
        acts = activation[32 * row:32 * (row + 1)]
        for column in range(32):
            weights = weight[32 * column:32 * (column + 1)]
            source_bits, truncated = mxu1_reset_lane(weights, acts)
            exact_bits = round_bf16(sum((fp8(w) * fp8(a)
                                         for w, a in zip(weights, acts, strict=True)), start=0))
            source.extend(struct.pack("<H", source_bits))
            exact.extend(struct.pack("<H", exact_bits))
            truncated_products += truncated
            if source_bits != exact_bits:
                mismatches.append((row, column, source_bits, exact_bits, truncated))

    directory.mkdir(parents=True, exist_ok=True)
    payloads = {
        "activation.bin": activation,
        "weight.bin": weight,
        "source-bf16.bin": source,
        "exact-bf16.bin": exact,
    }
    for name, payload in payloads.items():
        (directory / name).write_bytes(payload)
    report = {
        "schema": "atlas.selected_source_mxu1_broad_fixture.v1",
        "seed": SEED,
        "scope": "one reset tile; all 224 signed finite-normal E4M3 codes plus positive zero per operand",
        "codes_per_operand": len(PALETTE),
        "cells": 1024,
        "truncated_products": truncated_products,
        "source_vs_exact_mismatches": len(mismatches),
        "first_mismatches": [
            {"row": row, "column": column, "source_bf16": f"0x{observed:04x}",
             "exact_bf16": f"0x{expected:04x}", "truncated_products": truncated}
            for row, column, observed, expected, truncated in mismatches[:12]
        ],
        "cancellation_cell": {
            "source_bf16": f"0x{struct.unpack_from('<H', source, 0)[0]:04x}",
            "exact_bf16": f"0x{struct.unpack_from('<H', exact, 0)[0]:04x}",
        },
        "files": {name: hashlib.sha256(payload).hexdigest()
                  for name, payload in payloads.items()},
    }
    (directory / "fixture.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return report


def build_core_fixture(source: Path, compiled: Path, destination: Path, name: str) -> None:
    """Bind the independent lane panel to a compiled selected-core image."""
    if destination.exists():
        raise ValueError(f"core fixture destination must be fresh: {destination}")
    report_bytes = (source / "fixture.json").read_bytes()
    report = json.loads(report_bytes)
    if (
        report.get("schema") != "atlas.selected_source_mxu1_broad_fixture.v1"
        or report.get("cells") != 1024
        or report.get("codes_per_operand") != len(PALETTE)
        or report.get("source_vs_exact_mismatches", 0) < 1
    ):
        raise ValueError("source fixture does not carry the required broad MXU1 witness")
    payloads = {}
    for filename, length in (("activation.bin", 1024), ("weight.bin", 1024),
                             ("source-bf16.bin", 2048)):
        payload = (source / filename).read_bytes()
        if len(payload) != length or report.get("files", {}).get(filename) != digest(payload):
            raise ValueError(f"source fixture file is stale or malformed: {filename}")
        payloads[filename] = payload
    if (payloads["source-bf16.bin"][:2] != b"\x00\x00"
            or report.get("cancellation_cell") != {
                "source_bf16": "0x0000", "exact_bf16": "0x3bf0"
            }):
        raise ValueError("source fixture lost its anchor cancellation discriminator")

    program = (compiled / "program.bin").read_bytes()
    manifest = json.loads((compiled / "manifest.json").read_text())
    plan_bytes = (compiled / "execution_plan.json").read_bytes()
    plan = json.loads(plan_bytes)
    addresses = plan.get("input_addresses")
    output = plan.get("output_address")
    if (
        manifest.get("engine") != "merlin_native"
        or not str(manifest.get("target_identity", "")).startswith(
            f"atlas-rtl-{SELECTED_RTL}:")
        or manifest.get("binary_sha256") != digest(program)
        or manifest.get("execution_plan_sha256") != digest(plan_bytes)
        or not program or len(program) % 4 or len(program) > 128 * 1024
        or plan.get("schema") != "atlas.native_tile_execution_plan.v8"
        or plan.get("operation") != "mxu1_reset"
        or plan.get("numerical_policy") != FINITE_POLICY
        or plan.get("shape") != [32, 32]
        or plan.get("input_bytes") != 1024 or plan.get("output_bytes") != 2048
        or plan.get("output_physical_layouts") != ["bf16_pair_halves"]
        or not isinstance(addresses, list) or len(addresses) != 2
        or any(type(address) is not int or address <= 0 or address % 32
               for address in addresses)
        or type(output) is not int or output <= 0 or output % 32
        or addresses[0] == addresses[1]
        or any(address < output + 2048 and output < address + 1024
               for address in addresses)
    ):
        raise ValueError("compiled image does not match the selected MXU1 reset interface")

    logical = payloads["source-bf16.bin"]
    expected = bytearray(2048)
    for row in range(32):
        for column in range(32):
            logical_offset = 2 * (row * 32 + column)
            physical_offset = ((0 if column < 16 else 1024)
                               + row * 32 + 2 * (column % 16))
            expected[physical_offset:physical_offset + 2] = logical[
                logical_offset:logical_offset + 2]

    destination.mkdir(parents=True)
    (destination / "program.bin").write_bytes(program)
    (destination / "input-0.bin").write_bytes(payloads["activation.bin"])
    (destination / "input-1.bin").write_bytes(payloads["weight.bin"])
    (destination / "expected.bin").write_bytes(expected)
    properties = (
        "schema=atlas.rtl_tile_fixture.v1",
        f"name={name}",
        "input_count=2",
        f"program_sha256={digest(program)}",
        f"input.0.address={addresses[0]:#x}",
        f"input.1.address={addresses[1]:#x}",
        f"output.address={output:#x}",
        f"source_fixture_sha256={digest(report_bytes)}",
        "reference=selected_source_anchor_mirror_v1",
    )
    (destination / "manifest.properties").write_text("\n".join(properties) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--compiled", type=Path)
    parser.add_argument("--core-out", type=Path)
    parser.add_argument("--core-name", default="mxu1_broad_finite")
    args = parser.parse_args()
    if (args.compiled is None) != (args.core_out is None):
        parser.error("--compiled and --core-out must be supplied together")
    if re.fullmatch(r"[a-z][a-z0-9_]*", args.core_name) is None:
        parser.error("--core-name must use lowercase letters, digits, and underscores")
    if args.core_out is not None and args.core_out.exists():
        parser.error(f"core fixture destination must be fresh: {args.core_out}")
    report = build_fixture(args.out)
    if args.compiled is not None:
        build_core_fixture(args.out, args.compiled, args.core_out, args.core_name)
    print(json.dumps({key: report[key] for key in (
        "codes_per_operand", "cells", "truncated_products",
        "source_vs_exact_mismatches", "cancellation_cell",
    )}, sort_keys=True))


if __name__ == "__main__":
    main()
