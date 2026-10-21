"""Compose checked reset and seeded loops for an ordered K-streaming plan."""

from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path

from merlin.semantic_compiler.buffer_allocation import (
    Buffer,
    allocate_buffers,
    check_buffer_allocation,
)
from merlin.semantic_compiler.model import SemanticNode
from merlin.semantic_compiler.target_binding import (
    NativeCompilationError,
    verify_native_publication,
)


def chunk_bodies(request, plan):
    """Derive a small set of bodies; the source K range is never reassociated."""
    from .binding import MXU0_SEED_ATTRS, _mxu0_accumulate_maps
    from .looped_tiling import split_epilogue

    source, epilogue = split_epilogue(request)
    if plan.full.source != source:
        raise ValueError("chunk geometry differs from the source contraction")
    if plan.k_block_tiles is None or plan.reduction_launches < 2 or plan.accumulator_in_place:
        raise ValueError("chunk compilation requires separate carry storage and multiple K chunks")
    first = plan.first_body.source
    a, w, result = plan.body.source.nodes
    seed_id = result.id + "__seed"
    while seed_id in {node.id for node in request.nodes}:
        seed_id += "_"
    seed = SemanticNode(seed_id, "input", (), result.type, MXU0_SEED_ATTRS, "input")
    accumulation = replace(
        result, op="matmul_accumulate", inputs=(a.id, w.id, seed.id), index_maps=_mxu0_accumulate_maps()
    )
    carry = replace(
        plan.body.source,
        nodes=(a, w, seed, accumulation),
        input_storages=(*plan.body.source.input_storages, (seed.id, "external")),
    )
    bodies = [("first", first)]
    if epilogue is None or plan.reduction_launches > 2:
        bodies.append(("carry", carry))
    if epilogue is not None:
        final = replace(epilogue, type=replace(epilogue.type, shape=result.type.shape))
        bodies.append(("final", replace(carry, nodes=(*carry.nodes, final), outputs=(final.id,))))
    return tuple(bodies)


def _frame_buffers(request, plan, fixed_inputs, fixed_outputs):
    from .binding import TILE_BYTES, AtlasNativeBinding

    if set(fixed_inputs) != {node.id for node in request.nodes[:2]} or len(fixed_outputs) != 1:
        raise NativeCompilationError("compile_error", "chunk ABI needs both source operands and one output")
    bank = next(bank for bank in AtlasNativeBinding.profile().banks if bank.name == "external")
    ab, wb, ob = plan.body.buffer_bytes
    fixed = (fixed_inputs[request.nodes[0].id], fixed_inputs[request.nodes[1].id], None, fixed_outputs[0])
    buffers = tuple(
        Buffer(name, bank.name, (size + TILE_BYTES - 1) // TILE_BYTES, 0, 1, fixed_start=address)
        for name, size, address in zip(("a", "w", "seed", "out"), (ab, wb, ob, ob), fixed, strict=True)
    )
    return buffers, bank


def place_frames(request, plan, fixed_inputs, fixed_outputs):
    buffers, bank = _frame_buffers(request, plan, fixed_inputs, fixed_outputs)
    result = allocate_buffers(buffers, (bank,))
    if result.status != "sat":
        raise NativeCompilationError(
            "infeasible_candidate" if result.status == "unsat" else "search_timeout",
            "chunk buffer placement: " + result.reason,
        )
    frames = dict(result.addresses)
    check_frames(request, plan, fixed_inputs, fixed_outputs, frames)
    return frames


def check_frames(request, plan, fixed_inputs, fixed_outputs, frames):
    buffers, bank = _frame_buffers(request, plan, fixed_inputs, fixed_outputs)
    check_buffer_allocation(buffers, (bank,), frames)


def compile_members(snapshot, request, plan, *, fixed_inputs, fixed_outputs, root, prefix, target_source, limits):
    """Write flat native leaves, shared by raw-kernel and application packaging."""
    from .looped_tiling import compile_looped

    if prefix and (not prefix.isascii() or not prefix.isidentifier()):
        raise ValueError("chunk member prefix must be an ASCII identifier")
    frames = place_frames(request, plan, fixed_inputs, fixed_outputs)
    members = []
    for role, body in chunk_bodies(request, plan):
        relative = prefix + role
        inputs = {body.nodes[0].id: frames["a"], body.nodes[1].id: frames["w"]}
        if role != "first":
            inputs[body.nodes[2].id] = frames["seed"]
        manifest = compile_looped(
            snapshot,
            body,
            fixed_inputs=inputs,
            fixed_outputs=(frames["out"],),
            target_source=target_source,
            destination=root / relative,
            limits=limits,
        )
        verify_native_publication(
            root / relative,
            manifest,
            engine="merlin_native",
            request_digest=body.digest(),
            target_identity=request.target_identity,
        )
        members.append((role, relative, body, manifest))
    return frames, tuple(members)


def compile_chunked(snapshot, request, plan, *, fixed_inputs, fixed_outputs, target_source, destination, limits):
    """Publish a full-source request with a checked, flat set of native bodies."""
    destination.mkdir(parents=True, exist_ok=False)
    segments = destination / "segments"
    segments.mkdir()
    frames, members = compile_members(
        snapshot,
        request,
        plan,
        fixed_inputs=fixed_inputs,
        fixed_outputs=fixed_outputs,
        root=segments,
        prefix="",
        target_source=target_source,
        limits=limits,
    )
    files = {
        "source-request.json": request.record(),
        "chunked-mnk-tiling.json": plan.record(),
        "chunk-bindings.json": {"fixed_inputs": fixed_inputs, "fixed_outputs": list(fixed_outputs), "frames": frames},
    }
    for name, data in files.items():
        (destination / name).write_text(json.dumps(data, sort_keys=True, indent=2) + "\n")
    rows = []
    for role, relative, body, detail in members:
        rows.append(
            {
                "index": len(rows),
                "role": role,
                "path": "segments/" + relative,
                "request_digest": body.digest(),
                "binary_sha256": detail["binary_sha256"],
                "manifest_sha256": hashlib.sha256((segments / relative / "manifest.json").read_bytes()).hexdigest(),
            }
        )
    manifest = {
        "schema": "atlas.native_chunked_mnk_compilation.v1",
        "artifact_kind": "program_set",
        "engine": "merlin_native",
        "compile_mode": "diagnostic",
        "qualification_scope": "diagnostic_only",
        "request_digest": request.digest(),
        "target_identity": request.target_identity,
        "execution_requires": "ordered_chunked_host_dispatch",
        "segments": rows,
        "native_snapshot_sha256": hashlib.sha256((snapshot.root / "manifest.json").read_bytes()).hexdigest(),
        "chunk_lowering_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "stage_artifacts": {
            path.relative_to(destination).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted(destination.rglob("*"))
            if path.is_file()
        },
    }
    (destination / "manifest.json").write_text(json.dumps(manifest, sort_keys=True, indent=2) + "\n")
    return manifest
