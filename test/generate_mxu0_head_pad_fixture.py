"""Generate bounded MXU0 head-pad cases from the independent ordered oracle."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import struct
from pathlib import Path

from test_mxu_numerics import dot

SEED = 0xA71A00
PALETTE = (0x00, 0x20, 0x28, 0x30, 0x38, 0x40,
           0xA0, 0xA8, 0xB0, 0xB8, 0xC0)
CASE_COUNT = 32
CONTINUATION_CASE_COUNT = 16


def generate(destination: Path) -> dict[str, object]:
    rng = random.Random(SEED)
    records = bytearray()
    for _ in range(CASE_COUNT):
        activation = bytes(rng.choice(PALETTE) for _ in range(31))
        weight = bytes(rng.choice(PALETTE) for _ in range(31))
        expected = dot(weight, activation)
        records.extend(b"\x00" + activation + b"\x00" + weight)
        records.extend(struct.pack("<H", expected))
    assert len(records) == CASE_COUNT * 66
    continuation = bytearray()
    for _ in range(CONTINUATION_CASE_COUNT):
        activation = bytes(rng.choice(PALETTE) for _ in range(65))
        weight = bytes(rng.choice(PALETTE) for _ in range(65))
        a_tiles = (bytes(31) + activation[:1], activation[1:33], activation[33:65])
        w_tiles = (bytes(31) + weight[:1], weight[1:33], weight[33:65])
        for a_tile, w_tile in zip(a_tiles, w_tiles, strict=True):
            continuation.extend(a_tile + w_tile)
        continuation.extend(struct.pack("<H", dot(weight, activation)))
    assert len(continuation) == CONTINUATION_CASE_COUNT * 194
    destination.mkdir(parents=True, exist_ok=True)
    payload = destination / "head-pad-cases.bin"
    continuation_payload = destination / "head-pad-continuation-cases.bin"
    if payload.exists() or continuation_payload.exists():
        raise FileExistsError(f"fixture already exists in {destination}")
    payload.write_bytes(records)
    continuation_payload.write_bytes(continuation)
    manifest = {
        "schema": "atlas.mxu0_head_pad_fixture.v1",
        "selected_rtl_revision": "0079c0541111197741a231c002e3843fa6f545b2",
        "reference": "independent ordered FP8 product and per-step BF16 rounding",
        "numerical_domain": "positive zero and selected finite-normal E4M3 codes",
        "seed": SEED,
        "case_count": CASE_COUNT,
        "record_bytes": 66,
        "payload_sha256": hashlib.sha256(records).hexdigest(),
        "continuation_case_count": CONTINUATION_CASE_COUNT,
        "continuation_record_bytes": 194,
        "continuation_payload_sha256": hashlib.sha256(continuation).hexdigest(),
    }
    (destination / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
    )
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(generate(args.out), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
