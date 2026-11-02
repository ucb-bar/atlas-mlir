"""Precision policies: a named preset, or a policy JSON file passed through unchanged."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from .toolchain import rtl_revision

PRESETS = {
    "fp8": "E4M3 operands, BF16 intermediates, every contraction on MXU0, pack scale 127",
}


def resolve(policy: str | Path, linalg: str) -> dict[str, Any]:
    path = Path(policy)
    if path.suffix == ".json" or path.is_file():
        return json.loads(path.read_text())
    if str(policy) not in PRESETS:
        raise ValueError(f"unknown policy {policy!r}; use a JSON file or one of: {', '.join(PRESETS)}")
    contractions = len(re.findall(r"=\s*linalg\.matmul\b", linalg))
    if not contractions:
        raise ValueError("the fp8 preset needs at least one linalg.matmul; pass a policy file")
    result: dict[str, Any] = {
        "schema": "atlas.oot.quantized_linalg_policy.v1",
        "selected_rtl_revision": rtl_revision(),
        "source_dtype": "f32",
        "operand_encoding": "e4m3_finite_normal_rne_flush_subnormal",
        "bias_encoding": "bf16_rne",
        "vpu_add": "selected_f32_sum_bf16_chop",
        "interlayer_pack_scale_code": 127,
        # MXU0 admits any finite normal E4M3 operand; MXU1 is narrower.
        "mxu_units": [0] * contractions,
        "precision_transform": "explicit_diagnostic_candidate",
    }
    if "math.exp" in linalg:
        result["softmax_realization"] = "bf16_vpu_exp_recip_mul"
    return result
