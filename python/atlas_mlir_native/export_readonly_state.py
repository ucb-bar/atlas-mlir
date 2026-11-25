"""Explicit read-only captured state from authenticated exporter declarations.

Registered buffers keep their state role. They are never frozen into constants.
A saved-program signature supplies independent mutation information; when its
payload is unavailable, only explicit offline weight roles authorize read-only.
"""
from __future__ import annotations

import json
from hashlib import sha256
from pathlib import Path

from merlin.compiler.frontend.inputs import load_lifted_argument_bytes
from merlin.compiler.frontend.policy_inputs import stage_policy_inputs
from merlin.compiler.frontend.signature import _WEIGHT_DTYPES
from merlin.compiler.lower.c.parameters import StateDeclaration
from merlin.compiler.semantic.target_binding import NativeCompilationError


def _refuse(reason):
    raise NativeCompilationError("unsupported_semantics", "capture.readonly_state: " + reason)


def derive_readonly_state(bound, checked_export, export_dir, capture_dir):
    """Account for every buffer with checked bytes and explicit producer authority."""
    with stage_policy_inputs(bound, capture_dir) as (directory, staged):
        buffers = [(i, row) for i, row in enumerate(staged.argument_bindings) if row["kind"] == "buffer"]
        if not buffers:
            return None, {"schema": "atlas.readonly-state-evidence.v1", "ports": []}
        quantization = json.loads(bound.quantization_manifest_json)
        records = quantization.get("atlas", {}).get("buffers", {})
        if type(records) is not dict:
            _refuse("missing captured buffer declarations")
        from .soc import CAPTURED_STATE_READ_BYTES

        payloads = load_lifted_argument_bytes(directory, staged, kinds=("buffer",), max_bytes=CAPTURED_STATE_READ_BYTES)
        static_weights = set()
        for site in checked_export.sites.values():
            weight = site.get("weight")
            if (site.get("status") == "quantized" and type(weight) is dict
                    and weight.get("mode") == "static (offline)"):
                static_weights.update(weight.get(key) for key in ("codes", "scale_exp")
                                      if type(weight.get(key)) is str)
        program = Path(export_dir).absolute().parent / "fx" / "model.pt2"
        signature = None
        if program.exists() or program.is_symlink():
            from merlin.compiler.frontend.saved_program_signature import (
                read_saved_program_signature,
            )

            signature = read_saved_program_signature(program, expected_sha256=bound.native_program_sha256)
        serialized_buffers = {row.buffer_target: row for row in signature.buffers} if signature else {}
        ports, evidence = [], []
        for index, binding in buffers:
            name = binding["weight"]
            record = records.get(name)
            if type(record) is not dict or type(record.get("source_name")) is not str:
                _refuse("buffer lacks an authenticated exporter binding: " + name)
            payload = payloads[index]
            storage = _WEIGHT_DTYPES.get(record.get("dtype"))
            if (storage is None or record.get("shape") != binding["shape"]
                    or record.get("bytes") != len(payload) or record.get("sha256") != sha256(payload).hexdigest()
                    or not (storage[0] == binding["dtype"] or storage[1] == "U8" and binding["dtype"] == "i8")):
                _refuse("buffer bytes, shape or storage dtype differ: " + name)
            target = record["source_name"]
            declared = serialized_buffers.get(target)
            if signature is not None:
                if declared is None or declared.mutated:
                    _refuse("saved program does not declare a read-only buffer: " + target)
                authority = "authenticated_saved_program_signature"
            elif target in static_weights:
                authority = "authenticated_explicit_offline_weight_role"
            else:
                _refuse("no independent read-only declaration for buffer: " + target)
            ports.append({"argument_index": index, "name": name, "mode": "read_only",
                          "output_index": None, "initialization": "captured"})
            evidence.append({"argument_index": index, "source_name": target,
                             "payload_sha256": sha256(payload).hexdigest(), "authority": authority})
        declaration = {"schema": "merlin.application_state.v1", "source_program_sha256": staged.program_sha256,
                       "ports": ports}
        StateDeclaration.from_record(declaration, staged)
        return declaration, {"schema": "atlas.readonly-state-evidence.v1", "ports": evidence,
                             "saved_program_sha256": signature.archive_sha256 if signature else None,
                             "signature_member": signature.signature_member if signature else None}
