"""Publish Atlas command programs without a host LLVM representation.

Selection and numerical policy remain in the binding. This module serializes
its admitted outputs and records the identity of the bytes actually written;
it does not establish execution or semantic qualification.
"""

from __future__ import annotations

import hashlib
import json
import struct
from pathlib import Path
from typing import Mapping, Sequence


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def publish_program(
    destination: Path,
    *,
    words: Sequence[int],
    virtual: str,
    machine: str,
    program: Mapping[str, object],
    plan: Mapping[str, object],
    selection: Mapping[str, object],
    metadata: Mapping[str, object],
    constant_rows: Sequence[Mapping[str, object]] = (),
    qualification_machine: str | None = None,
    qualification_program: Mapping[str, object] | None = None,
) -> dict[str, object]:
    """Write the selected program into an existing staging directory.

    The binding supplies target-specific policy and previously written constant
    rows. A manifest is written last. Merlin's independent publication checker
    remains the consumer responsible for admitting the resulting file set.
    """
    binary = b"".join(struct.pack("<I", word) for word in words)
    plan_bytes = (json.dumps(plan, sort_keys=True) + "\n").encode()
    stages = {
        "atlas-virtual.mlir": virtual.encode(),
        "atlas-machine.mlir": machine.encode(),
        "physical-program.json": (json.dumps(program, indent=2) + "\n").encode(),
        "native-selection.json": (
            json.dumps(selection, sort_keys=True, indent=2) + "\n"
        ).encode(),
    }
    if qualification_machine is not None:
        stages["mandatory-qualification-machine.mlir"] = qualification_machine.encode()
        stages["mandatory-qualification-program.json"] = (
            json.dumps(qualification_program, sort_keys=True, indent=2) + "\n"
        ).encode()
    identities: dict[str, object] = {
        "binary_sha256": _sha256(binary),
        "execution_plan_sha256": _sha256(plan_bytes),
        "virtual_sha256": _sha256(stages["atlas-virtual.mlir"]),
        "machine_sha256": _sha256(stages["atlas-machine.mlir"]),
        "physical_program_sha256": _sha256(stages["physical-program.json"]),
        "stage_artifacts": {
            **{name: _sha256(payload) for name, payload in stages.items()},
            **{str(row["path"]): row["sha256"] for row in constant_rows},
        },
        "artifact_publisher_source_sha256": _sha256(Path(__file__).read_bytes()),
    }
    overlap = identities.keys() & metadata.keys()
    if overlap:
        raise ValueError(f"publication metadata overrides computed identities: {sorted(overlap)}")
    manifest = dict(metadata) | identities
    if constant_rows:
        manifest["constant_inputs"] = list(constant_rows)
    (destination / "program.bin").write_bytes(binary)
    (destination / "execution_plan.json").write_bytes(plan_bytes)
    for name, payload in stages.items():
        (destination / name).write_bytes(payload)
    (destination / "manifest.json").write_bytes(
        (json.dumps(manifest, sort_keys=True) + "\n").encode()
    )
    return manifest
