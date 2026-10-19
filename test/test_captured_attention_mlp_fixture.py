"""Check the directed RTL panel against the captured block's f32 weights."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from safetensors.numpy import load_file

sys.path.insert(0, str(Path(__file__).resolve().parent / "rtl"))
from make_captured_attention_mlp_fixture import (
    panel_payloads,
    two_output_panel_payloads,
)


def physical_bf16(values: np.ndarray) -> bytes:
    physical = bytearray(2048)
    for row in range(32):
        for column in range(32):
            f32_bits = np.asarray(values[row, column], dtype=np.float32).view(np.uint32).item()
            assert f32_bits & 0xFFFF == 0
            address = (column // 16) * 1024 + row * 32 + (column % 16) * 2
            physical[address:address + 2] = (f32_bits >> 16).to_bytes(2, "little")
    return bytes(physical)


def test_two_outputs_match_source_weights() -> None:
    weights = load_file(Path(__file__).parent / "examples/captured_attention_mlp32_weights.safetensors")
    for panel in ("half", "one", "striped"):
        runtime, expected = two_output_panel_payloads(panel)
        codes = np.frombuffer(runtime, dtype=np.uint8).reshape(32, 32)
        assert set(codes.flat) <= {0x30, 0x38}
        tokens = np.where(codes == 0x30, np.float32(0.5), np.float32(1.0))
        query = tokens @ weights["query.weight"].T
        key = tokens @ weights["key.weight"].T
        value = tokens @ weights["value.weight"].T
        scores = (query @ key.T) / np.sqrt(np.float32(32.0))
        exponentials = np.exp(scores - scores.max(axis=-1, keepdims=True))
        probabilities = exponentials / exponentials.sum(axis=-1, keepdims=True)
        hidden = np.maximum((probabilities @ value) @ weights["fc1.weight"].T, 0)
        final = hidden @ weights["fc2.weight"].T
        assert expected == physical_bf16(hidden) + physical_bf16(final)


def test_striped_panel_matches_source_weights() -> None:
    runtime, expected = panel_payloads("striped")
    codes = np.frombuffer(runtime, dtype=np.uint8).reshape(32, 32)
    assert set(codes.flat) == {0x30, 0x38}
    tokens = np.where(codes == 0x30, np.float32(0.5), np.float32(1.0))
    weights = load_file(Path(__file__).parent / "examples/captured_attention_mlp32_weights.safetensors")

    query = tokens @ weights["query.weight"].T
    key = tokens @ weights["key.weight"].T
    value = tokens @ weights["value.weight"].T
    scores = (query @ key.T) / np.sqrt(np.float32(32.0))
    exponentials = np.exp(scores - scores.max(axis=-1, keepdims=True))
    probabilities = exponentials / exponentials.sum(axis=-1, keepdims=True)
    result = np.maximum(
        (probabilities @ value) @ weights["fc1.weight"].T, 0
    ) @ weights["fc2.weight"].T

    assert physical_bf16(result) == expected
