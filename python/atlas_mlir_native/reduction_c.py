"""Export checked seeded reductions through the shared Atlas C launch interface."""

from __future__ import annotations

import hashlib
import json
import os
from math import prod
from pathlib import Path

from merlin.compile.c_conversions import render_f32_array_c
from merlin.compile.c_reduction import render_reduction_c
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
from .reduction import (
    body_request,
    input_spec,
    prepare_plan,
    prepare_source,
    selected_body_descriptors,
    virtual_program,
)
from .tile_io_c import render_tile_io_package


def prepare_c_package(source, compiled, name, *, target_source, runtime_linkage="separate"):
    precision = prepare_source(source)
    request = precision.request
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
    body = json.loads((compiled / "body/manifest.json").read_text())
    verify_native_publication(
        compiled / "body",
        body,
        engine="merlin_native",
        request_digest=body_request(plan).digest(),
        target_identity=request.target_identity,
    )
    inventory = hashlib.sha256((target_source / "docs/selected-variant-inventory.json").read_bytes()).hexdigest()
    if (
        manifest.get("schema") != "atlas.native_reduction_compilation.v1"
        or manifest.get("body_check_fingerprint") != body.get("native_check_fingerprint")
        or manifest.get("target_inventory_sha256") != inventory
        or body.get("target_inventory_sha256") != inventory
        or manifest.get("selected_body_descriptors") != selected_body_descriptors(plan, compiled / "body")
    ):
        raise ValueError("reduction body, inventory or native selection identity differs")
    execution = json.loads((compiled / "execution_plan.json").read_text())
    addresses, outputs = (
        execution.get("input_addresses"),
        execution.get("output_addresses"),
    )
    if not isinstance(addresses, list) or len(addresses) != 2 or not isinstance(outputs, list) or len(outputs) != 1:
        raise ValueError("reduction needs data, seed and output addresses")
    ranges = []
    for address, size in zip([*addresses, *outputs], plan.packed_sizes, strict=True):
        if (
            type(address) is not int
            or address < DRAM_BASE
            or (address - DRAM_BASE) % TILE_BYTES
            or address + size > DRAM_BASE + dram_slots() * TILE_BYTES
        ):
            raise ValueError("reduction address is unaligned or outside the selected window")
        ranges.append((address, address + size))
    if any(a < d and c < b for i, (a, b) in enumerate(ranges) for c, d in ranges[i + 1 :]):
        raise ValueError("reduction physical buffers overlap")
    expected = {
        "schema": "atlas.reduction_execution_plan.v1",
        "input_addresses": addresses,
        "input_buffer_bytes": list(plan.packed_sizes[:2]),
        "output_addresses": outputs,
        "output_bytes": plan.packed_sizes[2],
        "output_physical_layouts": ["row_column_zero_bf16_pair_tiles"],
        "tiling": plan.record(),
    }
    virtual = virtual_program(plan, addresses, outputs[0])
    if execution != expected or (compiled / "atlas-virtual.mlir").read_text() != virtual:
        raise ValueError("reduction program or ABI differs from source replay")
    pipeline = manifest.get("lowering_pipeline")
    if pipeline not in NATIVE_PIPELINES:
        raise ValueError("reduction has an unknown lowering pipeline")
    tools = Path(os.environ.get("ATLAS_OOT_BIN_DIR", target_source / "build/bin"))
    machine = _run([str(tools / "atlas-opt"), *NATIVE_PIPELINES[pipeline], "-"], input_text=virtual)
    _run(
        [str(tools / "atlas-opt"), *NATIVE_CHECKS, "--verify-atlas-timing", "-"],
        input_text=machine,
    )
    replay = json.loads(_run([str(tools / "atlas-emit"), "--program-json", "-"], input_text=machine))
    binary = (compiled / "program.bin").read_bytes()
    if replay.get("selected_rtl_revision") != RTL_REVISION or binary != b"".join(
        row["word_u32"].to_bytes(4, "little") for row in replay["instructions"]
    ):
        raise ValueError("reduction binary differs from checked source lowering")
    helpers = (
        render_reduction_c(plan, name)
        + "\n"
        + "\n".join(
            render_f32_array_c(input_spec(), prod(node.type.shape), name + f"_encode_{i}")
            for i, node in enumerate(request.nodes[:2])
        )
    )
    record, files = render_tile_io_package(
        request,
        binary,
        execution,
        name,
        target_source=target_source,
        runtime_linkage=runtime_linkage,
        packed_sizes=plan.packed_sizes,
        tile_bytes=plan.data_plan().tile_bytes,
        helpers=helpers,
    )
    return {
        **record,
        "precision_plan": precision.record(),
        "input_node_ids": list(source.nodes[-1].inputs),
        "output_node_ids": list(source.outputs),
        "native_manifest_sha256": hashlib.sha256((compiled / "manifest.json").read_bytes()).hexdigest(),
    }, files
