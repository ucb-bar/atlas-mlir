"""Prepare finite MXU0 tile inputs and expected outputs independently of a compiler."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import struct
from fractions import Fraction
from pathlib import Path


def _power2(exponent: int) -> Fraction:
    return Fraction(1 << exponent) if exponent >= 0 else Fraction(1, 1 << -exponent)


def _fp8(code: int) -> Fraction:
    exponent = (code >> 3) & 15
    if exponent == 0:
        if code & 7:
            raise ValueError("FP8 subnormals are outside the finite tile reference")
        return Fraction(0)
    if exponent == 15 and (code & 7) == 7:
        raise ValueError("NaN is outside the finite tile reference")
    value = Fraction(8 + (code & 7), 8) * _power2(exponent - 7)
    return -value if code & 0x80 else value


def _bf16(code: int) -> Fraction:
    if code == 0:
        return Fraction(0)
    exponent = (code >> 7) & 255
    if not 1 <= exponent <= 254:
        raise ValueError("BF16 result left the finite-normal tile reference")
    value = Fraction(128 + (code & 127), 128) * _power2(exponent - 127)
    return -value if code & 0x8000 else value


def _round_bf16(value: Fraction) -> int:
    if value == 0:
        return 0
    sign = 0x8000 if value < 0 else 0
    magnitude = abs(value)
    exponent = magnitude.numerator.bit_length() - magnitude.denominator.bit_length()
    if magnitude < _power2(exponent):
        exponent -= 1
    if not -126 <= exponent <= 127:
        raise ValueError("reference result left the finite-normal BF16 domain")
    scaled = magnitude / _power2(exponent - 7)
    lower = scaled.numerator // scaled.denominator
    remainder = scaled - lower
    significand = lower + (remainder > Fraction(1, 2)
                           or (remainder == Fraction(1, 2) and lower & 1))
    if significand == 256:
        significand = 128
        exponent += 1
    if exponent > 127:
        raise ValueError("reference BF16 result overflowed")
    return sign | ((exponent + 127) << 7) | (significand - 128)


def _code(*, weight: bool, row: int, k: int, panel: str) -> int:
    if panel == "ones":
        return 0x38
    palette = (0x30, 0x38, 0x40)  # exact E4M3 values 0.5, 1, 2
    return palette[((2 * row + k) if weight else (row + 2 * k)) % len(palette)]


def _expected_cell(row: int, column: int, k_extent: int, panel: str) -> int:
    accumulator = 0
    for k in range(k_extent):
        accumulator = _round_bf16(
            _bf16(accumulator)
            + _fp8(_code(weight=False, row=row, k=k, panel=panel))
            * _fp8(_code(weight=True, row=column, k=k, panel=panel))
        )
    return accumulator


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--program", required=True, type=Path)
    parser.add_argument("--execution-plan", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--name", required=True)
    parser.add_argument("--m", required=True, type=int)
    parser.add_argument("--k", required=True, type=int)
    parser.add_argument("--n", required=True, type=int)
    parser.add_argument("--m-start", type=int, default=0)
    parser.add_argument("--n-start", type=int, default=0)
    parser.add_argument("--panel", choices=("ones", "patterned"), default="ones")
    args = parser.parse_args()
    if args.out.exists():
        parser.error(f"output directory must be fresh: {args.out}")
    if re.fullmatch(r"[a-z][a-z0-9_]*", args.name) is None:
        parser.error("fixture name must use lowercase letters, digits, and underscores")
    if (args.m <= 0 or args.n <= 0 or not 1 <= args.k <= 256
            or args.m_start < 0 or args.n_start < 0
            or args.m_start % 32 or args.n_start % 32
            or args.m_start >= args.m or args.n_start >= args.n):
        parser.error("invalid positive shape or 32-aligned output tile start")

    tiles = (args.k + 31) // 32
    plan = json.loads(args.execution_plan.read_text())
    addresses = plan.get("input_addresses")
    output = plan.get("output_address")
    operation = "mxu0_reset" if tiles == 1 else f"mxu0_k{tiles}"
    if (plan.get("schema") != "atlas.native_tile_execution_plan.v8"
            or plan.get("operation") != operation
            or plan.get("numerical_policy") != "mxu0-e4m3-ordered-bf16-finite-v1"
            or plan.get("shape") != [32, 32]
            or plan.get("input_bytes") != 1024
            or plan.get("output_bytes") != 2048
            or plan.get("output_physical_layouts") != ["bf16_pair_halves"]
            or not isinstance(addresses, list) or len(addresses) != 2 * tiles
            or not all(isinstance(address, int) and address % 32 == 0
                       for address in addresses)
            or not isinstance(output, int) or output % 32):
        parser.error("execution plan does not match the requested MXU0 K windows")
    program = args.program.read_bytes()
    if not program or len(program) % 4 or len(program) > 128 * 1024:
        parser.error("program is empty, unaligned, or exceeds selected IMEM")

    head = args.k % 32 or 32
    valid_m = min(32, args.m - args.m_start)
    valid_n = min(32, args.n - args.n_start)
    inputs = []
    for window in range(2 * tiles):
        k_tile, source = divmod(window, 2)
        valid_rows = valid_m if source == 0 else valid_n
        leading_zeros = 32 - head if k_tile == 0 else 0
        source_start = 0 if k_tile == 0 else head + (k_tile - 1) * 32
        tile = bytearray(1024)
        for row in range(valid_rows):
            for column in range(leading_zeros, 32):
                logical_k = source_start + column - leading_zeros
                tile[row * 32 + column] = _code(
                    weight=source == 1,
                    row=(args.n_start if source else args.m_start) + row,
                    k=logical_k, panel=args.panel,
                )
        inputs.append(bytes(tile))

    # Expected cells use logical source rows and K order, independently of the
    # packed physical inputs above. The finite arithmetic reference is exact.
    if args.panel == "ones":
        encoded = struct.unpack("<I", struct.pack("<f", float(args.k)))[0]
        if encoded & 0xffff:
            parser.error("selected K is not exactly representable in BF16")
        ones_bf16 = encoded >> 16
    expected = bytearray(2048)
    for row in range(valid_m):
        for column in range(valid_n):
            byte = (0 if column < 16 else 1024) + row * 32 + (column % 16) * 2
            value = (ones_bf16 if args.panel == "ones" else _expected_cell(
                args.m_start + row, args.n_start + column, args.k, args.panel,
            ))
            expected[byte:byte + 2] = value.to_bytes(2, "little")

    args.out.mkdir(parents=True)
    (args.out / "program.bin").write_bytes(program)
    for index, tile in enumerate(inputs):
        (args.out / f"input-{index}.bin").write_bytes(tile)
    (args.out / "expected.bin").write_bytes(expected)
    manifest = [
        "schema=atlas.rtl_tile_fixture.v1",
        f"name={args.name}",
        f"input_count={len(inputs)}",
        f"program_sha256={hashlib.sha256(program).hexdigest()}",
        f"output.address={output:#x}",
        f"source_shape={args.m},{args.k},{args.n}",
        f"tile_start={args.m_start},{args.n_start}",
        f"reference=exact_e4m3_{args.panel}_ordered_bf16_v1",
    ]
    manifest.extend(f"input.{index}.address={address:#x}"
                    for index, address in enumerate(addresses))
    (args.out / "manifest.properties").write_text("\n".join(manifest) + "\n")


if __name__ == "__main__":
    main()
