#!/usr/bin/env python3
"""Prepare a selected-core fixture for the captured 32x32 attention/MLP block.

Uniform inputs make every attention row uniform. The fixed sparse 0.5 weights
then give exactly representable final outputs of 1/16 or 1/8. The expected
bytes are constructed here, separately from the Atlas instruction emitter.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
EXAMPLES = ROOT / "test/examples"
SELECTED_RTL = "0079c0541111197741a231c002e3843fa6f545b2"
PANELS = ("half", "one", "striped")


def panel_payloads(panel: str) -> tuple[bytes, bytes]:
    if panel == "half":
        return bytes([0x30]) * 1024, (0x3D80).to_bytes(2, "little") * 1024
    if panel == "one":
        return bytes([0x38]) * 1024, (0x3E00).to_bytes(2, "little") * 1024
    if panel != "striped":
        raise ValueError(f"unknown fixture panel: {panel}")

    # Each row is identical, so every key score in its attention row is
    # identical and softmax gives 1/32. The value and final MLP weights are
    # sparse permutations, each with a factor 1/2. Thus output column j is
    # input column (j + 2 + 3) mod 32, divided by eight.
    runtime_input = bytes(0x30 if column % 2 == 0 else 0x38
                          for _row in range(32) for column in range(32))
    expected = bytearray(2048)
    for row in range(32):
        for column in range(32):
            source_code = runtime_input[row * 32 + (column + 5) % 32]
            bits = 0x3D80 if source_code == 0x30 else 0x3E00
            physical = (column // 16) * 1024 + row * 32 + (column % 16) * 2
            expected[physical:physical + 2] = bits.to_bytes(2, "little")
    return runtime_input, bytes(expected)


def two_output_panel_payloads(panel: str) -> tuple[bytes, bytes]:
    """Expected ReLU intermediate followed by the final MLP tile."""
    runtime, final = panel_payloads(panel)
    intermediate = bytearray(2048)
    for row in range(32):
        for column in range(32):
            # The value projection shifts the source by two columns and
            # halves it; the first MLP weight halves it again. Every row's
            # attention probabilities are uniform for these panels.
            source_code = runtime[row * 32 + (column + 2) % 32]
            bits = {0x30: 0x3E00, 0x38: 0x3E80}[source_code]
            physical = (column // 16) * 1024 + row * 32 + (column % 16) * 2
            intermediate[physical:physical + 2] = bits.to_bytes(2, "little")
    return runtime, bytes(intermediate) + final


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--compiled", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--name", required=True)
    parser.add_argument("--panel", choices=sorted(PANELS), required=True)
    parser.add_argument("--two-output", action="store_true",
                        help="check the ReLU intermediate and final MLP output")
    args = parser.parse_args()
    if args.out.exists():
        parser.error(f"output directory must be fresh: {args.out}")
    if re.fullmatch(r"[a-z][a-z0-9_]*", args.name) is None:
        parser.error("fixture name must use lowercase letters, digits, and underscores")

    compiled = args.compiled
    manifest_path = compiled / "import-manifest.json"
    program_path = compiled / "program.bin"
    manifest = json.loads(manifest_path.read_text())
    program = manifest.get("program", {})
    outputs = manifest.get("outputs", [])
    output_count = 2 if args.two_output else 1
    source_file = ("captured_attention_mlp32_two_outputs_linalg.mlir"
                   if args.two_output else "captured_attention_mlp32_linalg.mlir")
    slots = manifest.get("input_slots", [])
    if (manifest.get("schema") != "atlas.oot.captured_linalg_import.v1"
            or manifest.get("selected_rtl_revision") != SELECTED_RTL
            or manifest.get("source_sha256") != digest(
                EXAMPLES / source_file)
            or manifest.get("argument_manifest_sha256") != digest(
                EXAMPLES / "captured_attention_mlp32_arguments.json")
            or manifest.get("policy_sha256") != digest(
                EXAMPLES / "atlas_attention_mlp_numeric_policy.json")
            or manifest.get("weights_sha256") != digest(
                EXAMPLES / "captured_attention_mlp32_weights.safetensors")
            or manifest.get("source_operations_accounted") != 61
            or manifest.get("mxu_units") != [0, 1, 0, 1, 0, 1, 0]
            or program.get("program_sha256") != digest(program_path)
            or program.get("word_count") != program_path.stat().st_size // 4
            or len(outputs) != output_count
            or any(row.get("index") != index or
                   row.get("dram_address") != 0x90010000 + 2048 * index or
                   row.get("payload_bytes") != 2048 or
                   row.get("physical_layout") != "bf16_pair_halves"
                   for index, row in enumerate(outputs))
            or len(slots) != 7
            or [slot.get("slot") for slot in slots] != list(range(7))
            or [slot.get("payload_bytes") for slot in slots]
                != [1024, 1024, 1024, 1024, 2048, 1024, 1024]
            or [slot.get("source_kind") for slot in slots]
                != ["input", "param", "param", "param", "generated", "param", "param"]
            or [slot.get("physical_role") for slot in slots]
                != ["activation_fp8", "weight_fp8_nk", "weight_fp8_nk",
                    "weight_fp8_nk", "constant_bf16", "weight_fp8_nk", "weight_fp8_nk"]):
        parser.error("compiled source, target, precision, or layout differs from fixture")

    constants = {row.get("slot"): row for row in manifest.get("constants", [])}
    if set(constants) != set(range(1, 7)):
        parser.error("compiled constants must cover the six fixed slots")
    payloads = []
    for slot in slots[1:]:
        row = constants[slot["slot"]]
        relative = row.get("payload_file")
        if not isinstance(relative, str) or Path(relative).is_absolute():
            parser.error("constant payload path must be relative")
        path = compiled / relative
        if (not path.resolve().is_relative_to(compiled.resolve())
                or row.get("payload_bytes") != slot["payload_bytes"]
                or row.get("dram_address") != slot["dram_address"]
                or row.get("payload_sha256") != digest(path)):
            parser.error(f"constant slot {slot['slot']} does not match its manifest")
        payloads.append(path.read_bytes())

    runtime_input, expected = (two_output_panel_payloads(args.panel)
                               if args.two_output else panel_payloads(args.panel))
    args.out.mkdir(parents=True)
    shutil.copyfile(program_path, args.out / "program.bin")
    (args.out / "input-0.bin").write_bytes(runtime_input)
    for index, payload in enumerate(payloads, start=1):
        (args.out / f"input-{index}.bin").write_bytes(payload)
    (args.out / "expected.bin").write_bytes(expected)
    properties = [
        "schema=atlas.rtl_tile_fixture.v1",
        f"name={args.name}",
        "input_count=7",
        f"output_count={output_count}",
        f"program_sha256={digest(program_path)}",
        *(f"input.{index}.address={slot['dram_address']:#x}"
          for index, slot in enumerate(slots)),
        f"output.address={outputs[0]['dram_address']:#x}",
        f"compiler_manifest_sha256={digest(manifest_path)}",
        f"source_sha256={manifest['source_sha256']}",
        f"policy_sha256={manifest['policy_sha256']}",
        f"reference=uniform_attention_sparse_mlp_{args.panel}_v1",
    ]
    (args.out / "manifest.properties").write_text("\n".join(properties) + "\n")


if __name__ == "__main__":
    main()
