"""Bind checked affine pointwise loops to the existing Atlas C launch interface."""

from __future__ import annotations

import hashlib
import json
import os
from math import prod
from pathlib import Path

from merlin.compile.c_conversions import render_bf16_codes_c, render_f32_array_c
from merlin.compile.c_pointwise import render_pointwise_c
from merlin.semantic_compiler.target_binding import verify_native_publication

from .binding import (
    DRAM_BASE,
    NATIVE_CHECKS,
    NATIVE_PIPELINES,
    RTL_REVISION,
    TILE_BYTES,
    _run,
    check_io_memory_manifest,
    dram_slots,
)
from .pointwise import (
    prepare_plan,
    prepare_source,
    selected_body_descriptor,
    virtual_program,
)
from .tile_io_c import render_tile_io_package


def prepare_c_package(
    source,
    compiled,
    name,
    *,
    target_source,
    runtime_linkage="separate",
    allow_math=False,
    bf16_policy=None,
):
    precision = prepare_source(source, allow_math=allow_math, bf16_policy=bf16_policy)
    request = precision.request
    arity = len(request.nodes) - 1
    plan = prepare_plan(request)
    manifest = json.loads((compiled / "manifest.json").read_text())
    check_io_memory_manifest(manifest)
    verify_native_publication(
        compiled,
        manifest,
        engine="merlin_native",
        request_digest=request.digest(),
        target_identity=request.target_identity,
    )
    body_manifest = json.loads((compiled / "body/manifest.json").read_text())
    inventory_sha = hashlib.sha256((target_source / "docs/selected-variant-inventory.json").read_bytes()).hexdigest()
    if body_manifest.get("target_inventory_sha256") != inventory_sha:
        raise ValueError("pointwise runtime differs from the compiled target inventory")
    verify_native_publication(
        compiled / "body",
        body_manifest,
        engine="merlin_native",
        request_digest=plan.body_request().digest(),
        target_identity=request.target_identity,
    )
    if manifest.get("schema") != "atlas.native_pointwise_compilation.v1" or manifest.get(
        "body_check_fingerprint"
    ) != body_manifest.get("native_check_fingerprint"):
        raise ValueError("pointwise body differs from its native publication")
    descriptor = selected_body_descriptor(plan, compiled / "body")
    if manifest.get("selected_body_descriptor") != descriptor:
        raise ValueError("pointwise loop differs from its native selected tile")
    execution = json.loads((compiled / "execution_plan.json").read_text())
    addresses, outputs = (
        execution.get("input_addresses"),
        execution.get("output_addresses"),
    )
    if not isinstance(addresses, list) or len(addresses) != arity or not isinstance(outputs, list) or len(outputs) != 1:
        raise ValueError("pointwise image needs every input range and one output range")
    output = outputs[0]
    ranges = []
    sizes = (*plan.input_packed_bytes, plan.packed_bytes)
    for address, size in zip([*addresses, output], sizes, strict=True):
        if (
            type(address) is not int
            or address < DRAM_BASE
            or (address - DRAM_BASE) % TILE_BYTES
            or address + size > DRAM_BASE + dram_slots() * TILE_BYTES
        ):
            raise ValueError("pointwise byte address is unaligned or outside the selected window")
        ranges.append((address, address + size))
    if any(a < d and c < b for i, (a, b) in enumerate(ranges) for c, d in ranges[i + 1 :]):
        raise ValueError("pointwise input/output ranges overlap")
    expected = {
        "schema": "atlas.pointwise_execution_plan.v1",
        "shape": list(plan.shape),
        "input_addresses": addresses,
        "input_buffer_bytes": list(plan.input_packed_bytes),
        "output_addresses": [output],
        "output_bytes": plan.packed_bytes,
        "output_physical_layouts": ["mn_tiles_bf16_pair_halves"],
        "tiling": plan.record(),
    }
    if execution != expected or (compiled / "atlas-virtual.mlir").read_text() != virtual_program(
        plan, addresses, output, descriptor
    ):
        raise ValueError("pointwise program or ABI differs from source replay")
    binary = (compiled / "program.bin").read_bytes()
    physical = json.loads((compiled / "physical-program.json").read_text())
    if physical.get("selected_rtl_revision") != RTL_REVISION or binary != b"".join(
        row["word_u32"].to_bytes(4, "little") for row in physical["instructions"]
    ):
        raise ValueError("pointwise binary differs from its selected physical words")
    # Hashes detect changed files. Replay also binds a consistently rehashed
    # machine program to this source operation, placement and pass pipeline.
    pipeline = manifest.get("lowering_pipeline")
    if pipeline not in NATIVE_PIPELINES:
        raise ValueError("pointwise compilation has an unknown lowering pipeline")
    tools = Path(os.environ.get("ATLAS_OOT_BIN_DIR", target_source / "build/bin"))
    machine = _run(
        [str(tools / "atlas-opt"), *NATIVE_PIPELINES[pipeline], "-"],
        input_text=virtual_program(plan, addresses, output, descriptor),
    )
    _run(
        [str(tools / "atlas-opt"), *NATIVE_CHECKS, "--verify-atlas-timing", "-"],
        input_text=machine,
    )
    replayed = json.loads(_run([str(tools / "atlas-emit"), "--program-json", "-"], input_text=machine))
    if binary != b"".join(row["word_u32"].to_bytes(4, "little") for row in replayed["instructions"]):
        raise ValueError("pointwise encoded program differs from source lowering")
    elements = [prod(node.type.shape) for node in request.nodes]
    source_dtype = source.nodes[-1].type.dtype
    encode = render_bf16_codes_c if source_dtype == "bf16" else render_f32_array_c
    source_inputs = tuple(node.id for node in source.nodes if node.effect in {"input", "constant"})
    helpers = (
        render_pointwise_c(plan, name)
        + "\n"
        + "\n".join(
            encode(spec, count, name + f"_encode_{i}")
            for i, (count, spec) in enumerate(zip(elements[:-1], precision.operand_specs, strict=True))
        )
    )
    io_record, files = render_tile_io_package(
        request,
        binary,
        execution,
        name,
        target_source=target_source,
        runtime_linkage=runtime_linkage,
        packed_sizes=sizes,
        tile_bytes=plan.tile_bytes,
        helpers=helpers,
        source_dtype=source_dtype,
        source_input_indices=tuple(source_inputs.index(key) for key in precision.record()["inputs"]),
    )
    return {
        **io_record,
        "precision_plan": precision.record(),
        "input_node_ids": list(source_inputs),
        "output_node_ids": list(source.outputs),
        "native_manifest_sha256": hashlib.sha256((compiled / "manifest.json").read_bytes()).hexdigest(),
    }, files
