"""Deterministic multi-launch materialization for independent MXU0 output tiles.

The outer partition only chooses region boundaries. Every segment still goes
through Merlin-native rule generation, extraction, allocation and checking.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

from merlin.semantic_compiler.model import KernelRequest
from merlin.semantic_compiler.search import SearchLimits
from merlin.semantic_compiler.snapshot import NativeSnapshot

from .mxu0 import Mxu0Contract
from .mxu0_emit import compile_mxu0, verify_mxu0_artifact
from .mxu0_launch import prepare_mxu0_launch
from .mxu0_spatial_tiling import (
    MAX_COMBINED_SPATIAL_NODES,
    SpatialTilePlan,
    lower_spatial_contraction,
)


def _encoded(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode() + b"\n"


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@dataclass(frozen=True)
class ProgramSetLaunch:
    program: Path
    program_sha256: str
    preloads: tuple[tuple[int, bytes], ...]
    output_id: str
    output_address: int
    output_length: int


def compile_mxu0_program_set(
    snapshot: NativeSnapshot,
    tiling: SpatialTilePlan,
    *,
    fixed_inputs: dict[str, int],
    fixed_outputs: tuple[int, ...],
    rtl_root: Path,
    destination: Path,
    limits: SearchLimits,
    contract: Mxu0Contract,
) -> dict[str, object]:
    """Compile one ordered-K chain per output tile with disjoint external ABI."""
    if destination.exists():
        raise FileExistsError("native program set needs a fresh destination")
    if tiling.source.target_identity != contract.identity:
        raise ValueError("program set target identity differs from selected Atlas")
    partitions = tiling.partition_outputs(fixed_inputs, fixed_outputs)
    if len(partitions) < 2:
        raise ValueError("program set needs multiple independent output tiles")
    destination.mkdir(parents=True)
    (destination / "segments").mkdir()
    (destination / "source_request.json").write_bytes(_encoded(tiling.source.record()))
    (destination / "tiling.json").write_bytes(_encoded(tiling.record()))
    rows: list[dict[str, object]] = []
    for index, (request, inputs, outputs) in enumerate(partitions):
        segment_path = destination / "segments" / f"{index:03d}"
        segment = compile_mxu0(
            snapshot, request, fixed_inputs=inputs, fixed_outputs=outputs,
            rtl_root=rtl_root, destination=segment_path, limits=limits,
            contract=contract,
        )
        output_id, m_start, n_start = tiling.output_tiles[index]
        rows.append({
            "index": index,
            "output_id": output_id,
            "m_start": m_start,
            "n_start": n_start,
            "request_digest": request.digest(),
            "fixed_inputs": inputs,
            "fixed_outputs": list(outputs),
            "binary_sha256": segment["binary_sha256"],
            "manifest_sha256": _sha(segment_path / "manifest.json"),
            "instruction_words": segment["instruction_words"],
        })
    manifest: dict[str, object] = {
        "schema": "atlas.native_tensor_program_set.v1",
        "status": "diagnostic_program_set",
        "engine": "merlin_native",
        "target_identity": contract.identity,
        "source_request_digest": tiling.source.digest(),
        "lowered_request_digest": tiling.lowered.digest(),
        "tiling_sha256": _sha(destination / "tiling.json"),
        "partition_policy": "one_complete_ordered_k_chain_per_output_tile",
        "max_combined_spatial_nodes": MAX_COMBINED_SPATIAL_NODES,
        "fixed_source_inputs": fixed_inputs,
        "fixed_source_outputs": list(fixed_outputs),
        "segments": rows,
        "instruction_words_total": sum(int(row["instruction_words"]) for row in rows),
        "runtime_scope": "sequential_standalone_core_launches_not_yet_qualified",
    }
    (destination / "manifest.json").write_bytes(_encoded(manifest))
    return manifest


def verify_mxu0_program_set(
    snapshot: NativeSnapshot,
    destination: Path,
    *,
    rtl_root: Path,
    expected_request_digest: str,
    software_spec: Path | None = None,
) -> dict[str, object]:
    """Rebuild the partition and independently replay each native segment."""
    manifest = json.loads((destination / "manifest.json").read_text())
    source = KernelRequest.from_record(json.loads((destination / "source_request.json").read_text()))
    contract = Mxu0Contract.load()
    tiling = lower_spatial_contraction(source, contract)
    if (
        manifest.get("schema") != "atlas.native_tensor_program_set.v1"
        or manifest.get("status") != "diagnostic_program_set"
        or manifest.get("engine") != "merlin_native"
        or manifest.get("target_identity") != contract.identity
        or manifest.get("source_request_digest") != expected_request_digest
        or source.digest() != expected_request_digest
        or manifest.get("lowered_request_digest") != tiling.lowered.digest()
        or (destination / "tiling.json").read_bytes() != _encoded(tiling.record())
        or manifest.get("tiling_sha256") != _sha(destination / "tiling.json")
        or manifest.get("partition_policy") != "one_complete_ordered_k_chain_per_output_tile"
        or manifest.get("max_combined_spatial_nodes") != MAX_COMBINED_SPATIAL_NODES
    ):
        raise ValueError("native program set source, target, or partition changed")
    if {item.name for item in destination.iterdir()} != {
        "manifest.json", "source_request.json", "tiling.json", "segments",
    }:
        raise ValueError("native program set file set changed")
    partitions = tiling.partition_outputs(
        manifest["fixed_source_inputs"], tuple(manifest["fixed_source_outputs"]),
    )
    rows = manifest.get("segments")
    segment_root = destination / "segments"
    if not isinstance(rows, list) or len(rows) != len(partitions) or {
        item.name for item in segment_root.iterdir()
    } != {f"{index:03d}" for index in range(len(partitions))}:
        raise ValueError("native program set segment count or paths changed")
    instruction_words_total = 0
    for index, (request, inputs, outputs) in enumerate(partitions):
        row = rows[index]
        output_id, m_start, n_start = tiling.output_tiles[index]
        segment_path = segment_root / f"{index:03d}"
        segment_manifest = json.loads((segment_path / "manifest.json").read_text())
        expected = {
            "index": index,
            "output_id": output_id,
            "m_start": m_start,
            "n_start": n_start,
            "request_digest": request.digest(),
            "fixed_inputs": inputs,
            "fixed_outputs": list(outputs),
            "binary_sha256": segment_manifest["binary_sha256"],
            "manifest_sha256": _sha(segment_path / "manifest.json"),
            "instruction_words": segment_manifest["instruction_words"],
        }
        if row != expected or segment_manifest.get("request_digest") != request.digest():
            raise ValueError("native program set segment identity or ABI changed")
        verify_mxu0_artifact(
            snapshot, segment_path, rtl_root=rtl_root,
            expected_request_digest=request.digest(), software_spec=software_spec,
        )
        instruction_words_total += int(segment_manifest["instruction_words"])
    if manifest.get("instruction_words_total") != instruction_words_total:
        raise ValueError("native program set instruction accounting changed")
    return manifest


def prepare_mxu0_program_set_launch(
    destination: Path,
    runtime_inputs: dict[str, bytes],
    *,
    expected_request_digest: str,
) -> tuple[ProgramSetLaunch, ...]:
    """Check and panelize one invocation without running a reference model."""
    manifest = json.loads((destination / "manifest.json").read_text())
    source = KernelRequest.from_record(json.loads((destination / "source_request.json").read_text()))
    contract = Mxu0Contract.load()
    tiling = lower_spatial_contraction(source, contract)
    if (
        manifest.get("schema") != "atlas.native_tensor_program_set.v1"
        or manifest.get("status") != "diagnostic_program_set"
        or manifest.get("engine") != "merlin_native"
        or manifest.get("target_identity") != contract.identity
        or source.digest() != expected_request_digest
        or manifest.get("source_request_digest") != expected_request_digest
        or manifest.get("lowered_request_digest") != tiling.lowered.digest()
        or (destination / "tiling.json").read_bytes() != _encoded(tiling.record())
        or manifest.get("tiling_sha256") != _sha(destination / "tiling.json")
        or manifest.get("partition_policy") != "one_complete_ordered_k_chain_per_output_tile"
        or manifest.get("max_combined_spatial_nodes") != MAX_COMBINED_SPATIAL_NODES
    ):
        raise ValueError("program set invocation identity or partition changed")
    if {item.name for item in destination.iterdir()} != {
        "manifest.json", "source_request.json", "tiling.json", "segments",
    }:
        raise ValueError("program set invocation file set changed")
    partitions = tiling.partition_outputs(
        manifest["fixed_source_inputs"], tuple(manifest["fixed_source_outputs"]),
    )
    rows = manifest.get("segments")
    segment_root = destination / "segments"
    if not isinstance(rows, list) or len(rows) != len(partitions) or {
        item.name for item in segment_root.iterdir()
    } != {f"{index:03d}" for index in range(len(partitions))}:
        raise ValueError("program set invocation has missing or extra segments")
    panels = tiling.panelize_inputs(runtime_inputs)
    launches: list[ProgramSetLaunch] = []
    instruction_words_total = 0
    for index, (request, inputs, outputs) in enumerate(partitions):
        output_id, m_start, n_start = tiling.output_tiles[index]
        segment = segment_root / f"{index:03d}"
        segment_manifest = json.loads((segment / "manifest.json").read_text())
        expected = {
            "index": index,
            "output_id": output_id,
            "m_start": m_start,
            "n_start": n_start,
            "request_digest": request.digest(),
            "fixed_inputs": inputs,
            "fixed_outputs": list(outputs),
            "binary_sha256": segment_manifest["binary_sha256"],
            "manifest_sha256": _sha(segment / "manifest.json"),
            "instruction_words": segment_manifest["instruction_words"],
        }
        if rows[index] != expected or segment_manifest.get("request_digest") != request.digest():
            raise ValueError("program set invocation segment identity or ABI changed")
        payloads = {node_id: panels[node_id] for node_id in inputs}
        preloads = prepare_mxu0_launch(segment, payloads)
        plan = json.loads((segment / "execution_plan.json").read_text())
        if len(plan["outputs"]) != 1 or plan["outputs"][0]["source"] != output_id:
            raise ValueError("program set invocation output identity changed")
        output = plan["outputs"][0]
        instruction_words_total += int(segment_manifest["instruction_words"])
        launches.append(ProgramSetLaunch(
            segment / "program.bin", segment_manifest["binary_sha256"],
            preloads, output_id,
            output["byte_address"], output["byte_length"],
        ))
    if manifest.get("instruction_words_total") != instruction_words_total:
        raise ValueError("program set invocation instruction accounting changed")
    return tuple(launches)
