"""Independent finite arithmetic and physical tail placement for RTL fixtures."""

from __future__ import annotations

import json
import struct
import sys
from fractions import Fraction
from pathlib import Path

import pytest
from atlas_mlir_native.layout import unpack_bf16_pair_halves, unpack_native_bf16_output

sys.path.insert(0, str(Path(__file__).resolve().parent / "rtl"))
from make_ordered_tile_fixture import _round_bf16, main


def test_rounding_ties_use_even_bf16_significand() -> None:
    assert _round_bf16(Fraction(257, 256)) == 0x3F80
    assert _round_bf16(Fraction(259, 256)) == 0x3F82
    assert _round_bf16(Fraction(-257, 256)) == 0xBF80


def test_patterned_tail_uses_global_rows_and_physical_halves(
    tmp_path: Path, monkeypatch,
) -> None:
    program = tmp_path / "program.bin"
    program.write_bytes((0x00000073).to_bytes(4, "little"))
    execution_plan = tmp_path / "plan.json"
    execution_plan.write_text(json.dumps({
        "schema": "atlas.native_tile_execution_plan.v8",
        "operation": "mxu0_reset",
        "numerical_policy": "mxu0-e4m3-ordered-bf16-finite-v1",
        "shape": [32, 32],
        "input_bytes": 1024,
        "output_bytes": 2048,
        "dtype": "bf16",
        "output_physical_layouts": ["bf16_pair_halves"],
        "input_addresses": [0x90000000, 0x90000400],
        "output_address": 0x90001000,
    }))
    output = tmp_path / "fixture"
    monkeypatch.setattr(sys, "argv", [
        "make_ordered_tile_fixture.py", "--program", str(program),
        "--execution-plan", str(execution_plan), "--out", str(output),
        "--name", "patterned_tail", "--m", "33", "--k", "1", "--n", "17",
        "--m-start", "32", "--panel", "patterned",
    ])
    main()

    activation = (output / "input-0.bin").read_bytes()
    weight = (output / "input-1.bin").read_bytes()
    expected = (output / "expected.bin").read_bytes()
    assert activation[:31] == bytes(31)
    assert activation[31] == 0x40  # global source row 32 has value 2 at K=0
    assert activation[32:] == bytes(992)
    assert weight[31] == 0x30  # global output column 0 has value 0.5
    assert weight[32 + 31] == 0x40  # column 1 has value 2
    assert struct.unpack_from("<H", expected, 0)[0] == 0x3F80  # 2 * 0.5
    assert struct.unpack_from("<H", expected, 2)[0] == 0x4080  # 2 * 2
    assert struct.unpack_from("<H", expected, 1024)[0] == 0x4080  # column 16
    assert struct.unpack_from("<H", expected, 1026)[0] == 0  # inactive column
    assert struct.unpack_from("<H", expected, 32)[0] == 0  # inactive row
    logical = unpack_bf16_pair_halves(expected)
    assert struct.unpack_from("<H", logical, 0)[0] == 0x3F80
    assert struct.unpack_from("<H", logical, 2)[0] == 0x4080
    assert struct.unpack_from("<H", logical, 32)[0] == 0x4080
    assert struct.unpack_from("<H", logical, 34)[0] == 0
    assert "reference=exact_e4m3_patterned_ordered_bf16_v1" in (
        output / "manifest.properties"
    ).read_text()


def test_pair_halves_unpack_preserves_last_row_and_refuses_short_tile() -> None:
    physical = bytearray(2048)
    physical[31 * 32:31 * 32 + 2] = b"\x34\x12"
    physical[1024 + 31 * 32 + 15 * 2:1024 + 31 * 32 + 16 * 2] = b"\x78\x56"
    logical = unpack_bf16_pair_halves(bytes(physical))
    assert logical[(31 * 32) * 2:(31 * 32) * 2 + 2] == b"\x34\x12"
    assert logical[-2:] == b"\x78\x56"
    with pytest.raises(ValueError, match="2048"):
        unpack_bf16_pair_halves(bytes(2046))


def test_native_output_decoding_requires_declared_physical_layout() -> None:
    plan = {
        "schema": "atlas.native_tile_execution_plan.v8",
        "shape": [32, 32], "dtype": "bf16", "output_bytes": 2048,
        "output_addresses": [0x90001000, 0x90001800],
        "output_physical_layouts": ["bf16_pair_halves", "bf16_pair_halves"],
    }
    physical = bytes(2048)
    assert unpack_native_bf16_output(plan, physical, 1) == physical
    for invalid in ({**plan, "output_physical_layouts": ["row_major", "bf16_pair_halves"]},
                    {**plan, "output_physical_layouts": ["bf16_pair_halves"]},
                    {key: value for key, value in plan.items() if key != "output_physical_layouts"}):
        with pytest.raises(ValueError, match="declared BF16"):
            unpack_native_bf16_output(invalid, physical)
    with pytest.raises(ValueError, match="declared BF16"):
        unpack_native_bf16_output(plan, physical, 2)
