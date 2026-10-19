"""Turn a checked host launch's FP8 windows into an independent MXU0 RTL fixture."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

from make_ordered_tile_fixture import _bf16, _fp8, _round_bf16


def _logical_position(k: int, head: int) -> tuple[int, int]:
    if k < head:
        return 0, 32 - head + k
    return 1 + (k - head) // 32, (k - head) % 32


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture", required=True, type=Path)
    parser.add_argument("--program", required=True, type=Path)
    parser.add_argument("--execution-plan", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--name", required=True)
    parser.add_argument("--trial", required=True, type=int)
    parser.add_argument("--tile-index", required=True, type=int)
    parser.add_argument("--m", required=True, type=int)
    parser.add_argument("--k", required=True, type=int)
    parser.add_argument("--n", required=True, type=int)
    args = parser.parse_args()
    if args.out.exists() or args.out.is_symlink():
        parser.error(f"fresh output directory required: {args.out}")
    if (re.fullmatch(r"[a-z][a-z0-9_]*", args.name) is None
            or min(args.m, args.k, args.n) <= 0 or args.k > 256
            or args.trial < 0 or args.tile_index < 0):
        parser.error("invalid fixture name, shape, trial, or tile index")
    n_tiles = (args.n + 31) // 32
    m_start, n_start = divmod(args.tile_index, n_tiles)
    m_start *= 32
    n_start *= 32
    if m_start >= args.m:
        parser.error("tile index exceeds the source output grid")
    head = args.k % 32 or 32
    windows = 2 * ((args.k + 31) // 32)
    plan = json.loads(args.execution_plan.read_text())
    addresses = plan.get("input_addresses")
    output = plan.get("output_address")
    operation = "mxu0_reset" if windows == 2 else f"mxu0_k{windows // 2}"
    if (plan.get("schema") != "atlas.native_tile_execution_plan.v8"
            or plan.get("operation") != operation
            or plan.get("numerical_policy") != "mxu0-e4m3-ordered-bf16-finite-v1"
            or plan.get("shape") != [32, 32]
            or plan.get("input_bytes") != 1024
            or plan.get("output_bytes") != 2048
            or plan.get("output_physical_layouts") != ["bf16_pair_halves"]
            or not isinstance(addresses, list) or len(addresses) != windows
            or any(type(address) is not int or address <= 0 or address % 32
                   for address in addresses)
            or type(output) is not int or output <= 0 or output % 32):
        parser.error("execution plan is not a supported MXU0 K-window tile")
    program = args.program.read_bytes()
    if not program or len(program) % 4:
        parser.error("program is empty or not word-aligned")

    inputs = []
    for index in range(windows):
        path = args.capture / (
            f"trial-{args.trial}-tile-{args.tile_index}-input-{index}.bin"
        )
        raw = path.read_bytes()
        if len(raw) != 1024:
            parser.error(f"captured input must be exactly 1024 bytes: {path}")
        inputs.append(raw)

    expected = bytearray(2048)
    valid_m = min(32, args.m - m_start)
    valid_n = min(32, args.n - n_start)
    for row in range(valid_m):
        for column in range(valid_n):
            acc = 0
            for k in range(args.k):
                segment, offset = _logical_position(k, head)
                lhs = inputs[2 * segment][row * 32 + offset]
                rhs = inputs[2 * segment + 1][column * 32 + offset]
                acc = _round_bf16(_bf16(acc) + _fp8(lhs) * _fp8(rhs))
            physical = (1024 if column >= 16 else 0) + row * 32 + (column % 16) * 2
            expected[physical:physical + 2] = acc.to_bytes(2, "little")

    args.out.mkdir(parents=True)
    (args.out / "program.bin").write_bytes(program)
    for index, raw in enumerate(inputs):
        (args.out / f"input-{index}.bin").write_bytes(raw)
    (args.out / "expected.bin").write_bytes(expected)
    lines = [
        "schema=atlas.rtl_tile_fixture.v1",
        f"name={args.name}",
        f"input_count={windows}",
        f"program_sha256={hashlib.sha256(program).hexdigest()}",
        f"output.address={output:#x}",
        f"source_shape={args.m},{args.k},{args.n}",
        f"tile_start={m_start},{n_start}",
        "reference=host_capture_exact_e4m3_ordered_bf16_v1",
    ]
    lines.extend(f"input.{index}.address={address:#x}"
                 for index, address in enumerate(addresses))
    (args.out / "manifest.properties").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
