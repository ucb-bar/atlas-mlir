"""Preflight the mixed FP8/BF16 selected-core fixture format."""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent / "rtl"))
from run_compiled_program_atlas_core import fixture_identity


def make_fixture(root: Path) -> None:
    root.mkdir()
    program = (0x00000073).to_bytes(4, "little")
    (root / "program.bin").write_bytes(program)
    (root / "expected.bin").write_bytes(bytes(2048))
    lengths = [1024, 1024, 1024, 1024, 2048, 1024, 1024]
    for index, length in enumerate(lengths):
        (root / f"input-{index}.bin").write_bytes(bytes([index]) * length)
    properties = [
        "schema=atlas.rtl_tile_fixture.v1",
        "name=attention_mlp_test",
        "input_count=7",
        f"program_sha256={hashlib.sha256(program).hexdigest()}",
        *(f"input.{index}.address={0x90000000 + index * 2048:#x}"
          for index in range(7)),
        "output.address=0x90010000",
    ]
    (root / "manifest.properties").write_text("\n".join(properties) + "\n")


def test_mixed_width_fixture_counts_all_dma_beats(tmp_path: Path) -> None:
    root = tmp_path / "fixture"
    make_fixture(root)
    name, count, read_beats, cells, _ = fixture_identity(root)
    assert (name, count, read_beats, cells) == ("attention_mlp_test", 7, 256, 1024)


def test_two_output_fixture_counts_both_output_tiles(tmp_path: Path) -> None:
    root = tmp_path / "fixture"
    make_fixture(root)
    (root / "expected.bin").write_bytes(bytes(4096))
    with (root / "manifest.properties").open("a") as manifest:
        manifest.write("output_count=2\n")
    name, count, read_beats, cells, _ = fixture_identity(root)
    assert (name, count, read_beats, cells) == ("attention_mlp_test", 7, 256, 2048)


def test_two_output_fixture_rejects_short_expected_tile(tmp_path: Path) -> None:
    root = tmp_path / "fixture"
    make_fixture(root)
    with (root / "manifest.properties").open("a") as manifest:
        manifest.write("output_count=2\n")
    with pytest.raises(ValueError, match="byte count"):
        fixture_identity(root)


def test_mixed_width_fixture_rejects_overlap(tmp_path: Path) -> None:
    root = tmp_path / "fixture"
    make_fixture(root)
    properties = root.joinpath("manifest.properties").read_text()
    root.joinpath("manifest.properties").write_text(
        properties.replace("input.5.address=0x90002800", "input.5.address=0x90002000")
    )
    with pytest.raises(ValueError, match="overlap"):
        fixture_identity(root)


def test_mixed_width_fixture_rejects_unexpected_input_size(tmp_path: Path) -> None:
    root = tmp_path / "fixture"
    make_fixture(root)
    (root / "input-4.bin").write_bytes(bytes(1056))
    with pytest.raises(ValueError, match="byte count"):
        fixture_identity(root)


def test_resident_fixture_requires_a_watchdog_and_reports_bounded_dma(tmp_path: Path) -> None:
    root = tmp_path / "resident"
    make_fixture(root)
    manifest = root / "manifest.properties"
    manifest.write_text(manifest.read_text().replace("atlas.rtl_tile_fixture.v1", "atlas.rtl_tile_fixture.v3") + "output_count=17\n")
    (root / "expected.bin").write_bytes(bytes(17*2048))
    with pytest.raises(ValueError, match="watchdog"):
        fixture_identity(root)
    with manifest.open("a") as stream:
        stream.write("max_cycles=200000\n")
    _, _, beats, words, _ = fixture_identity(root)
    assert beats is None  # Intermediate reads are checked by address bounds.
    assert words == 17*1024
