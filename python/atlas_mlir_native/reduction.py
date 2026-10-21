"""Native tile selection inside compact seeded tensor-reduction loops."""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass, replace
from pathlib import Path

from merlin.semantic_compiler.buffer_allocation import Buffer, allocate_buffers
from merlin.semantic_compiler.model import (
    IndexMap,
    KernelRequest,
    SemanticNode,
    TensorType,
    pointwise_identity_maps,
)
from merlin.semantic_compiler.reduction_tiling import ReductionTiling
from merlin.semantic_compiler.target_binding import (
    NativeCompilationError,
    verify_native_publication,
)

from .binding import (
    DRAM_BASE,
    NATIVE_CHECKS,
    NATIVE_PIPELINES,
    RTL_REVISION,
    TARGET_IDENTITY,
    TILE_BYTES,
    AtlasNativeBinding,
    _native_pipeline,
    _run,
    dram_slots,
)
from .pointwise import input_spec as pointwise_input_spec

REDUCTION_POLICY = "bf16-adjacent-fp32-tree-rne-seeded-tile-sum-ltr-chop-v1"


def input_spec():
    """Bounded RNE encoding, explicitly canonicalizing source negative zero."""
    return replace(pointwise_input_spec(), policy=REDUCTION_POLICY, canonicalize_negative_zero=True)


@dataclass(frozen=True)
class ReductionPrecision:
    source: KernelRequest
    request: KernelRequest

    def record(self):
        return {
            "schema": "atlas.reduction_precision.v1",
            "source_request_digest": self.source.digest(),
            "request_digest": self.request.digest(),
            "input_conversion": input_spec().record(),
            "arithmetic": REDUCTION_POLICY,
            "output_conversion": "bf16-to-f32-exact",
            "qualification": "diagnostic_precision_change; original_model_quality_unmeasured",
        }


def prepare_source(source):
    """Select new arithmetic explicitly; this is not an f32 equality rewrite."""
    if (
        source.target_identity != TARGET_IDENTITY
        or source.lowering_policy != "diagnostic"
        or source.boundary_layouts
        or source.output_layouts
        or source.physical_extents
    ):
        raise ValueError("reduction precision needs a plain diagnostic source graph")
    pure = [node for node in source.nodes if node.effect == "pure"]
    if len(pure) != 1 or len(pure[0].inputs) != 2:
        raise ValueError("reduction needs one computation with explicit data and seed")
    result = pure[0]
    attrs = dict(result.attrs)
    dimension = "axis" if "axis" in attrs else "axes"
    value = attrs.get(dimension)
    axes = (
        (value,) if dimension == "axis" else tuple(int(a) for a in value.split(",")) if isinstance(value, str) else ()
    )
    expected = {dimension: value}
    if result.op == "source_f32_reduce_add_unqualified":
        expected.update(
            reduction_order="unresolved",
            numeric_semantics="source-f32-add-unqualified-v1",
            scalar_add_order=attrs.get("scalar_add_order"),
        )
        if expected["scalar_add_order"] not in {
            "input-accumulator",
            "accumulator-input",
        }:
            raise ValueError("source reduction scalar operand order is unknown")
    elif result.op == "f32_reduce_add":
        expected.update(reduction_order="axis-ascending", numeric_semantics="ieee754-f32-add-rne-v1")
    else:
        raise ValueError("source operation is not an admitted addition reduction")
    if (
        attrs != expected
        or source.outputs != (result.id,)
        or {n.id for n in source.nodes} != {*result.inputs, result.id}
        or any(n.type.dtype != "f32" or n.type.numerical_policy != "raw-bit-pattern" for n in source.nodes)
    ):
        raise ValueError("source reduction attributes, types or accounting differ")
    data, seed = (source.node(key) for key in result.inputs)
    if any(n.effect not in {"input", "constant"} or n.index_maps for n in (data, seed)):
        raise ValueError("source reduction operands must be explicit boundary values")
    request = request_for(data.type.shape, axes)
    if (
        seed.type.shape != request.nodes[1].type.shape
        or result.type.shape != request.nodes[2].type.shape
        or result.index_maps != request.nodes[2].index_maps
    ):
        raise ValueError("source reduction shapes or indexing maps differ")
    request = replace(request, source_identity=source.digest())
    prepare_plan(request)
    import numpy as np

    constants = {item.node_id: item for item in source.constants}
    for node in (data, seed):
        if node.effect == "constant":
            binding = constants[node.id]
            if binding.encoding != "f32-le-raw":
                raise ValueError("reduction parameter needs exact f32 bytes")
            input_spec().encode(np.frombuffer(bytes.fromhex(binding.data_hex), dtype="<f4"))
    return ReductionPrecision(source, request)


def request_for(shape, axes):
    """Construct a public BF16 request; source-float precision selection is separate."""
    rank = len(shape)
    axes = tuple(axes)
    if (
        not axes
        or any(type(axis) is not int or not 0 <= axis < rank for axis in axes)
        or tuple(sorted(set(axes))) != axes
    ):
        raise ValueError("reduction axes must be distinct ascending source dimensions")
    kept = tuple(i for i in range(rank) if i not in axes)
    if not kept:
        raise ValueError("reduction request needs a ranked output")
    projection = IndexMap(
        rank,
        tuple(tuple(int(i == axis) for i in range(rank)) for axis in kept),
        (0,) * len(kept),
    )
    catalog = {d.computation: d for d in AtlasNativeBinding.profile().descriptors}
    data_type = TensorType(tuple(shape), "bf16", "raw-bit-pattern")
    seed_type = TensorType(tuple(shape[i] for i in kept), "bf16", "raw-bit-pattern")
    return KernelRequest(
        (
            SemanticNode(
                "data",
                "input",
                (),
                data_type,
                (catalog["row_sum_bf16"].required_attrs[0],),
                "input",
            ),
            SemanticNode(
                "seed",
                "input",
                (),
                seed_type,
                (catalog["add_bf16_rne"].required_attrs[0],),
                "input",
            ),
            SemanticNode(
                "result",
                "reduce_sum",
                ("data", "seed"),
                replace(seed_type, numerical_policy=REDUCTION_POLICY),
                (("axes", ",".join(map(str, axes))),),
                index_maps=(
                    pointwise_identity_maps(rank, 1)[0],
                    projection,
                    projection,
                ),
            ),
        ),
        ("result",),
        ("external",),
        (("data", "external"), ("seed", "external")),
        TARGET_IDENTITY,
        "diagnostic",
    )


def prepare_plan(request):
    if (
        request.target_identity != TARGET_IDENTITY
        or request.lowering_policy != "diagnostic"
        or len(request.nodes) != 3
        or request.nodes[-1].op != "reduce_sum"
    ):
        raise NativeCompilationError("unsupported_semantics", "reduction needs its selected BF16 policy")
    data, seed, out = request.nodes
    axes_text = dict(out.attrs).get("axes")
    try:
        axes = tuple(int(v) for v in axes_text.split(",")) if isinstance(axes_text, str) else ()
    except ValueError as error:
        raise NativeCompilationError("unsupported_semantics", "reduction axes are not canonical integers") from error
    row, add = descriptors()
    if (
        out.attrs != (("axes", ",".join(map(str, axes))),)
        or out.type.dtype != "bf16"
        or out.type.numerical_policy != REDUCTION_POLICY
        or any(node.type.numerical_policy != "raw-bit-pattern" for node in (data, seed))
        or data.attrs != (row.required_attrs[0],)
        or seed.attrs != (add.required_attrs[0],)
        or request.input_storages != ((data.id, "external"), (seed.id, "external"))
        or request.output_storages != ("external",)
    ):
        raise NativeCompilationError("unsupported_semantics", "reduction policies, attributes or storages differ")
    tile = tuple(axis.minimum for axis in row.output_axis_bounds)
    plan = ReductionTiling(request, axes, tile, 2, b"\0\0")
    if plan.matrix_shape[1] > 65536:
        raise NativeCompilationError("resource_limit", "reduction exceeds the selected finite column bound")
    return plan


def descriptors():
    catalog = AtlasNativeBinding.profile().descriptors
    return (
        next(d for d in catalog if d.computation == "row_sum_bf16"),
        next(d for d in catalog if d.computation == "add_bf16_rne"),
    )


def body_request(plan):
    row, add = descriptors()
    tile = TensorType(plan.tile_shape, "bf16", "raw-bit-pattern")
    return KernelRequest(
        (
            SemanticNode("data", "input", (), tile, (row.required_attrs[0],), "input"),
            SemanticNode("seed", "input", (), tile, (add.required_attrs[0],), "input"),
            SemanticNode(
                "partial",
                row.computation,
                ("data",),
                tile,
                row.required_attrs,
                index_maps=row.index_maps,
            ),
            SemanticNode(
                "result",
                add.computation,
                ("seed", "partial"),
                tile,
                add.required_attrs,
                index_maps=add.index_maps,
            ),
        ),
        ("result",),
        ("external",),
        (("data", "external"), ("seed", "external")),
        plan.source.target_identity,
        "diagnostic",
        plan.source.digest(),
    )


def plan_abi(request):
    plan = prepare_plan(request)
    bank = next(b for b in AtlasNativeBinding.profile().banks if b.name == "external")
    names = [n.id for n in request.nodes]
    buffers = tuple(
        Buffer(name, bank.name, (size + TILE_BYTES - 1) // TILE_BYTES, 0, 1)
        for name, size in zip(names, plan.packed_sizes, strict=True)
    )
    solved = allocate_buffers(buffers, (bank,))
    if solved.status != "sat":
        raise NativeCompilationError("resource_limit", "packed reduction buffers exceed the selected window")
    addresses = dict(solved.addresses)
    return {
        "fixed_inputs": {name: addresses[name] for name in names[:2]},
        "fixed_outputs": [addresses[names[2]]],
    }


def selected_body_descriptors(plan, compiled):
    selected = json.loads((compiled / "native-selection.json").read_text())
    required = [d.name for d in descriptors()]
    computational = [
        value["descriptor"]
        for value in selected["values"]
        if value["descriptor"] is not None
        and value["descriptor"] not in {"atlas.load_bf16_tile", "atlas.store_bf16_tile"}
    ]
    if (
        selected.get("engine") != "merlin_native"
        or selected.get("request_digest") != body_request(plan).digest()
        or selected.get("target_identity") != plan.source.target_identity
        or sorted(computational) != sorted(required)
    ):
        raise NativeCompilationError(
            "unsupported_semantics",
            "native reduction body differs from its tree and seeded combination",
        )
    return required


def virtual_program(plan, addresses, output):
    state, tensor = "!atlas.virtual_state", "!atlas.virtual_bf16"
    rows = plan.seed_plan().tile_counts[0]
    columns = plan.data_plan().tile_counts[1]
    tile = plan.data_plan().tile_bytes
    signed = lambda value: value if value < 2**31 else value - 2**32
    return f"""module {{
  func.func @reduce_sum() -> {state} attributes {{atlas.input_dram_base = {addresses[0]} : i64, atlas.output_dram_base = {output} : i64, atlas.output_dram_bytes = {plan.packed_sizes[2]} : i64}} {{
    %start = "atlas.virtual_start"() : () -> {state}
    %zero = arith.constant 0 : i32
    %base = arith.constant {signed(addresses[0])} : i32
    %seed_base = arith.constant {signed(addresses[1])} : i32
    %out_base = arith.constant {signed(output)} : i32
    cf.br ^rows(%start,%zero,%base,%seed_base,%out_base : {state},i32,i32,i32,i32)
  ^rows(%rs: {state},%row: i32,%data: i32,%seed: i32,%out: i32):
    %row_count = arith.constant {rows} : i32
    %more_rows = arith.cmpi ult, %row, %row_count : i32
    cf.cond_br %more_rows, ^initialize(%rs : {state}), ^done(%rs : {state})
  ^initialize(%is: {state}):
    %bytes = arith.constant {tile} : i32
    %initialized, %initial = "atlas.virtual_dma_load_bf16"(%is,%seed,%bytes) : ({state},i32,i32) -> ({state},{tensor})
    cf.br ^columns(%initialized,%zero,%data,%initial : {state},i32,i32,{tensor})
  ^columns(%cs: {state},%col: i32,%pointer: i32,%acc: {tensor}):
    %column_count = arith.constant {columns} : i32
    %more_columns = arith.cmpi ult, %col, %column_count : i32
    cf.cond_br %more_columns, ^step(%cs : {state}), ^store(%cs,%acc : {state},{tensor})
  ^step(%ss: {state}):
    %loaded, %tile = "atlas.virtual_dma_load_bf16"(%ss,%pointer,%bytes) : ({state},i32,i32) -> ({state},{tensor})
    %partial = "atlas.virtual_vpu_reduce"(%tile) {{kind = "row_sum"}} : ({tensor}) -> {tensor}
    %next_acc = "atlas.virtual_vpu_binary"(%acc,%partial) {{kind = "add"}} : ({tensor},{tensor}) -> {tensor}
    %one = arith.constant 1 : i32
    %next_col = arith.addi %col, %one : i32
    %next_pointer = arith.addi %pointer, %bytes : i32
    cf.br ^columns(%loaded,%next_col,%next_pointer,%next_acc : {state},i32,i32,{tensor})
  ^store(%os: {state},%result: {tensor}):
    %stored = "atlas.virtual_dma_store_bf16"(%os,%result,%out,%bytes) : ({state},{tensor},i32,i32) -> {state}
    %one_row = arith.constant 1 : i32
    %next_row = arith.addi %row, %one_row : i32
    %row_bytes = arith.constant {columns * tile} : i32
    %next_data = arith.addi %data, %row_bytes : i32
    %next_seed = arith.addi %seed, %bytes : i32
    %next_out = arith.addi %out, %bytes : i32
    cf.br ^rows(%stored,%next_row,%next_data,%next_seed,%next_out : {state},i32,i32,i32,i32)
  ^done(%end: {state}):
    return %end : {state}
  }}
}}
"""


def compile_reduction(
    snapshot,
    request,
    *,
    fixed_inputs,
    fixed_outputs,
    target_source,
    destination,
    limits,
):
    plan = prepare_plan(request)
    if set(fixed_inputs) != {request.nodes[0].id, request.nodes[1].id} or len(fixed_outputs) != 1:
        raise NativeCompilationError("compile_error", "reduction ABI must bind data, seed and output")
    slots = tuple(fixed_inputs[n.id] for n in request.nodes[:2]) + tuple(fixed_outputs)
    if any(
        type(slot) is not int or slot < 0 or slot * TILE_BYTES + size > dram_slots() * TILE_BYTES
        for slot, size in zip(slots, plan.packed_sizes, strict=True)
    ):
        raise NativeCompilationError("resource_limit", "reduction bindings exceed the selected byte window")
    addresses = tuple(DRAM_BASE + slot * TILE_BYTES for slot in slots)
    ranges = [(address, address + size) for address, size in zip(addresses, plan.packed_sizes, strict=True)]
    if any(a < d and c < b for i, (a, b) in enumerate(ranges) for c, d in ranges[i + 1 :]):
        raise NativeCompilationError("infeasible_candidate", "reduction physical buffers overlap")
    destination.mkdir(parents=True, exist_ok=False)
    body = body_request(plan)
    abi = AtlasNativeBinding.plan_boundary_abi(body)
    checked = AtlasNativeBinding.compile(
        snapshot,
        body,
        fixed_inputs=abi["fixed_inputs"],
        fixed_outputs=tuple(abi["fixed_outputs"]),
        target_source=target_source,
        destination=destination / "body",
        limits=limits,
        allow_staging=False,
    )
    verify_native_publication(
        destination / "body",
        checked,
        engine="merlin_native",
        request_digest=body.digest(),
        target_identity=request.target_identity,
    )
    required = selected_body_descriptors(plan, destination / "body")
    tools = Path(os.environ.get("ATLAS_OOT_BIN_DIR", target_source / "build/bin"))
    virtual = virtual_program(plan, addresses[:2], addresses[2])
    pipeline = _native_pipeline()
    mandatory = _run(
        [str(tools / "atlas-opt"), *NATIVE_PIPELINES["mandatory"], "-"],
        input_text=virtual,
    )
    _run(
        [str(tools / "atlas-opt"), *NATIVE_CHECKS, "--verify-atlas-timing", "-"],
        input_text=mandatory,
    )
    machine = (
        mandatory
        if pipeline == "mandatory"
        else _run(
            [str(tools / "atlas-opt"), *NATIVE_PIPELINES[pipeline], "-"],
            input_text=virtual,
        )
    )
    _run(
        [str(tools / "atlas-opt"), *NATIVE_CHECKS, "--verify-atlas-timing", "-"],
        input_text=machine,
    )
    program_text = _run([str(tools / "atlas-emit"), "--program-json", "-"], input_text=machine)
    program = json.loads(program_text)
    if program.get("selected_rtl_revision") != RTL_REVISION:
        raise NativeCompilationError("unqualified_target", "reduction emitter uses a different RTL")
    binary = b"".join(row["word_u32"].to_bytes(4, "little") for row in program["instructions"])
    execution = {
        "schema": "atlas.reduction_execution_plan.v1",
        "input_addresses": list(addresses[:2]),
        "input_buffer_bytes": list(plan.packed_sizes[:2]),
        "output_addresses": [addresses[2]],
        "output_bytes": plan.packed_sizes[2],
        "output_physical_layouts": ["row_column_zero_bf16_pair_tiles"],
        "tiling": plan.record(),
    }
    for name, value in {
        "source-request.json": request.record(),
        "execution_plan.json": execution,
        "reduction-tiling.json": plan.record(),
    }.items():
        (destination / name).write_text(json.dumps(value, sort_keys=True, indent=2) + "\n")
    for name, text in {
        "atlas-virtual.mlir": virtual,
        "atlas-machine.mlir": machine,
        "physical-program.json": program_text,
    }.items():
        (destination / name).write_text(text)
    (destination / "program.bin").write_bytes(binary)
    if pipeline != "mandatory":
        (destination / "mandatory-qualification-machine.mlir").write_text(mandatory)
        (destination / "mandatory-qualification-program.json").write_text(
            _run([str(tools / "atlas-emit"), "--program-json", "-"], input_text=mandatory)
        )
    sha = lambda raw: hashlib.sha256(raw).hexdigest()
    manifest = {
        "schema": "atlas.native_reduction_compilation.v1",
        "artifact_kind": "program",
        "engine": "merlin_native",
        "compile_mode": "diagnostic",
        "qualification_scope": "diagnostic_only",
        "request_digest": request.digest(),
        "target_identity": request.target_identity,
        "binary_sha256": sha(binary),
        "execution_plan_sha256": sha((destination / "execution_plan.json").read_bytes()),
        "body_check_fingerprint": checked["native_check_fingerprint"],
        "selection_scope": "representative_row_tree_and_seeded_combination",
        "selected_body_descriptors": required,
        "lowering_pipeline": pipeline,
        "semantic_qualification_pipeline": "mandatory",
        "native_snapshot_sha256": sha((snapshot.root / "manifest.json").read_bytes()),
        "io_memory_bytes": dram_slots() * TILE_BYTES,
        "reduction_lowering_source_sha256": sha(Path(__file__).read_bytes()),
        "target_inventory_sha256": sha((target_source / "docs/selected-variant-inventory.json").read_bytes()),
        "stage_artifacts": {
            p.relative_to(destination).as_posix(): sha(p.read_bytes())
            for p in sorted(destination.rglob("*"))
            if p.is_file() and p != destination / "program.bin"
        },
    }
    (destination / "manifest.json").write_text(json.dumps(manifest, sort_keys=True, indent=2) + "\n")
    return manifest
