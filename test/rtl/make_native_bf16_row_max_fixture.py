#!/usr/bin/env python3
"""Prepare selected raw-BF16 row-maximum witnesses for a native Atlas tile."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "test"))
from test_vpu_row_max_reference import (
    _panel,  # Test-side oracle lives outside the compiler.
)

TARGET_IDENTITY = (
    "atlas-rtl-0079c0541111197741a231c002e3843fa6f545b2:"
    "movement-mxu01-vpu-pack-tile-v21"
)
REQUEST_DIGEST = "6a0ff68ec6f3ffd8ccb50fb1e9908e1c2dbcfa16e5680226899c1b093b722ba6"


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--compiled", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--phase", required=True, type=int, choices=(0, 2))
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
    reductions = [row for row in physical.get("instructions", [])
                  if row.get("operation") == "atlas.vpu_reduce"]
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
            or plan.get("operation") != "bf16_row_max"
            or plan.get("shape") != [32, 32]
            or plan.get("output_bytes") != 2048
            or plan.get("output_physical_layouts") != ["bf16_pair_halves"]
            or plan.get("input_domain") != "bf16-raw-all-codes-v1"
            or not isinstance(inputs, list) or len(inputs) != 1
            or not isinstance(outputs, list) or len(outputs) != 1
            or not all(type(address) is int and address > 0 and address % 32 == 0
                       for address in (*inputs, *outputs))
            or inputs[0] < outputs[0] + 2048 and outputs[0] < inputs[0] + 2048
            or len(reductions) != 1 or reductions[0].get("fields", {}).get("kind") != "row_max"
            or physical.get("word_count") != len(program) // 4
            or not program or len(program) % 4 or len(program) > 128 * 1024):
        parser.error("native compilation does not match the BF16 row-maximum fixture")

    source, expected, _ = _panel(args.phase)
    if len(source) != 2048 or len(expected) != 2048:
        parser.error("independent row-maximum panel has the wrong physical size")
    args.out.mkdir(parents=True)
    shutil.copyfile(program_path, args.out / "program.bin")
    (args.out / "input-0.bin").write_bytes(source)
    (args.out / "expected.bin").write_bytes(expected)
    properties = [
        "schema=atlas.rtl_tile_fixture.v1",
        f"name={args.name}",
        "input_count=1",
        "output_count=1",
        f"program_sha256={digest(program_path)}",
        f"compiler_manifest_sha256={digest(manifest_path)}",
        f"execution_plan_sha256={digest(plan_path)}",
        f"input.0.address={inputs[0]:#x}",
        f"output.address={outputs[0]:#x}",
        f"reference=selected_raw_bf16_row_order_phase_{args.phase}",
    ]
    (args.out / "manifest.properties").write_text("\n".join(properties) + "\n")


if __name__ == "__main__":
    main()
