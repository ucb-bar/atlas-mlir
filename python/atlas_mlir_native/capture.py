"""Target-owned external adapter intake for immutable exported FP8 state."""
from __future__ import annotations

from hashlib import sha256
from importlib import import_module
from pathlib import Path


def load_export(path, *, ops_module="atlas_fx_ops", **kwargs):
    """Register the source schemas before deserializing an exported program."""
    import torch

    import_module(ops_module)
    return torch.export.load(path, **kwargs)


def prepare_model(model, inputs, contract, policy, *, export_prefix="", expected_contract_sha256, expected_policy_sha256):
    """Apply the selected external adapter to a model reconstructed from raw state.

    The caller supplies architecture and raw state. This boundary never constructs
    quantized weights or changes their dtype. The returned ExportedProgram keeps
    the exact node identities used by the adapter's per-call receipt.
    """
    from m2m.capture.external_quantization import (
        ExternalQuantizationConfig, apply_external_quantization,
    )

    for expected in (expected_contract_sha256, expected_policy_sha256):
        if not isinstance(expected, str) or len(expected) != 64 or any(c not in "0123456789abcdef" for c in expected):
            raise ValueError("expected source SHA-256 must be a lowercase 64-character digest")
    contract, policy = Path(contract).resolve(strict=True), Path(policy).resolve(strict=True)
    config = ExternalQuantizationConfig(
        "atlas_fp8", contract, policy,
        expected_contract_sha256, expected_policy_sha256,
    )
    model._atlas_export_dir = contract.parent
    model._atlas_export_prefix = export_prefix
    captured = apply_external_quantization(model, tuple(inputs), config)
    return captured, tuple(inputs)


def normalize_capture(module, manifest):
    """Worker hook: normalize before capture hashes and bundle publication."""
    from .source_normalize import normalize

    return normalize(module, manifest)


def compiler_input_view(staged, manifest):
    """Consume validated manifest provenance for a compiler-owned input view.

    The archived staged program is immutable. The two authenticated module provenance attributes are consumed into this
    receipt. The archived program retains both attributes.
    """
    from dataclasses import replace
    import json

    from xdsl.dialects.builtin import StringAttr
    from merlin.frontends.linalg_mlir import parse_mlir_text
    from merlin.xdsl_dialects._common import text

    raw = staged.program.encode()
    if sha256(raw).hexdigest() != staged.program_sha256:
        raise ValueError("compiler view source differs from validated staged program")
    expected = sha256(json.dumps(manifest, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()
    module = parse_mlir_text(staged.program)
    suffix = "\n" if staged.program.endswith("\n") else ""
    if text(module, generic=True) + suffix != staged.program:
        raise ValueError("compiler view requires exact staged parser/serializer roundtrip")
    key = "prov.quantization_manifest_sha256"
    if module.attributes.get(key) != StringAttr(expected):
        raise ValueError("compiler view manifest digest differs from authenticated selection")
    if any(key in op.attributes for op in module.walk() if op is not module):
        raise ValueError("compiler view cannot consume nested manifest attributes")
    scheme_key = "prov.quantization"
    scheme = module.attributes.get(scheme_key)
    if scheme != StringAttr("external:atlas_fp8"):
        raise ValueError("compiler view quantization scheme differs from selected adapter")
    if any(scheme_key in op.attributes for op in module.walk() if op is not module):
        raise ValueError("compiler view cannot consume nested quantization attributes")
    del module.attributes[key]
    del module.attributes[scheme_key]
    compiler_text = text(module, generic=True) + suffix
    compiler_hash = sha256(compiler_text.encode()).hexdigest()
    receipt = {
        "schema": "atlas.compiler_input_view.v1",
        "consumed_attributes": {key: expected, scheme_key: scheme.data},
        "quantization_manifest_sha256": expected,
        "staged_program_sha256": staged.program_sha256,
        "staged_program_bytes": len(raw),
        "compiler_program_sha256": compiler_hash,
        "compiler_program_bytes": len(compiler_text.encode()),
        "byte_delta": len(raw) - len(compiler_text.encode()),
    }
    return replace(staged, program=compiler_text, program_sha256=compiler_hash), receipt
