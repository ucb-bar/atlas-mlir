"""Compose selected tile DAGs and compact matrix loops into one Atlas image.

Matrix compositions use private tile-aligned DRAM allocations. Larger BF16
tile DAGs reuse staging storage after completed region executions. Neither
route changes arithmetic, infers precision, or inserts CPU computation.
"""

from __future__ import annotations

import hashlib
import io
import json
import os
from dataclasses import replace
from math import prod
from pathlib import Path
from time import monotonic

from merlin.compile.semantic_regions import extract_semantic_region
from merlin.semantic_compiler.model import TensorType
from merlin.semantic_compiler.target_binding import (
    NativeCompilationError,
    verify_native_publication,
)

from .binding import (
    BF16_BINARY_ATTRS,
    BF16_RELU_ATTRS,
    BF16_ROW_REDUCTION_ATTRS,
    BF16_UNARY_ATTRS,
    DRAM_BASE,
    MXU0_INPUT_DOMAIN,
    MXU0_POLICY,
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
from .pack_tiling import ROUNDED_PACK_DOMAIN, ROUNDED_PACK_POLICY
from .looped_tiling import retry_without_operand_reuse


def _bf16_body_work(request):
    # Each primitive needs an instruction, each boundary a load, and each
    # distinct output a store. Staging must budget for movement as well as ALU work.
    return len(request.nodes) + len(set(request.outputs))


def is_staged_bf16(request):
    from merlin.semantic_compiler.search import SearchLimits

    from .binding import _bf16_tile_request

    profile = AtlasNativeBinding.profile()
    bank = next(bank for bank in profile.banks if bank.name == "tensor_bf16")
    width = next(d.extent for d in profile.descriptors if d.name == "atlas.load_bf16_tile")
    return _bf16_tile_request(request) and (
        sum(n.effect == "pure" for n in request.nodes) > bank.capacity // width
        or _bf16_body_work(request) > SearchLimits().candidate_nodes
    )


def is_resident_graph(request, *, allow_staging=True):
    return (allow_staging and is_staged_bf16(request)) or (
        len([n for n in request.nodes if n.op == "matmul"]) > 1
        and any(n.type.shape != (32, 32) for n in request.nodes)
    )


def bf16_boundary_request(region):
    """Retain exact primitive computations and declare their boundary premises."""
    attributes = {**BF16_BINARY_ATTRS, **BF16_UNARY_ATTRS, **BF16_ROW_REDUCTION_ATTRS, "relu": BF16_RELU_ATTRS}
    nodes = []
    for node in region.nodes:
        if node.effect == "pure":
            nodes.append(node)
            continue
        domains = {attributes[user.op][0] for user in region.nodes
                   if user.effect == "pure" and node.id in user.inputs and user.op in attributes}
        if len(domains) > 1:
            raise ValueError("staged input has conflicting primitive-domain declarations")
        nodes.append(replace(node, op="input", effect="input", attrs=tuple(domains)))
    return replace(region, nodes=tuple(nodes), constants=(),
                   input_storages=tuple((n.id, "external") for n in nodes if n.effect == "input"),
                   source_identity=region.digest())


def staged_memory_plan(request):
    from merlin.compile.semantic_regions import (
        partition_semantic_dag,
        semantic_staging_buffers,
    )
    from merlin.semantic_compiler.buffer_allocation import (
        Buffer,
        allocate_buffers,
        check_buffer_allocation,
        first_fit_buffer_assignment,
    )
    from merlin.semantic_compiler.search import SearchLimits

    def admitted(region):
        try:
            body = bf16_boundary_request(region)
        except ValueError:
            return False
        return _bf16_body_work(body) <= SearchLimits().candidate_nodes

    try:
        cuts = partition_semantic_dag(request, max_nodes=8, accept=admitted)
    except ValueError as error:
        raise NativeCompilationError("unsupported_semantics", "no qualified BF16 staging boundaries") from error
    sources = [n.id for n in request.nodes if n.effect in {"input", "constant"}]
    readonly = len(sources)
    fixed = {key: index for index, key in enumerate((*sources, *request.outputs))}
    bank = next(b for b in AtlasNativeBinding.profile().banks if b.name == "external")
    original = semantic_staging_buffers(request, cuts, storage=bank.name,
                                        extents=dict.fromkeys((n.id for n in request.nodes), 1), fixed=fixed)
    buffers = (Buffer("$readonly", bank.name, readonly, 0, len(cuts) + 1, fixed_start=0, persistent=True),) + tuple(
        replace(b, view_of="$readonly", view_offset=fixed[b.name]) if b.name in sources else b for b in original
    )
    proposal = first_fit_buffer_assignment(buffers, (bank,))
    pinned = tuple(replace(b, fixed_start=proposal[b.name]) for b in buffers) if proposal else buffers
    allocated = allocate_buffers(pinned, (bank,), timeout_ms=5000)
    if allocated.status != "sat":
        raise NativeCompilationError("resource_limit" if allocated.status == "unsat" else "search_timeout",
                                     "staged DRAM allocation: " + allocated.reason)
    slots = dict(allocated.addresses)
    check_buffer_allocation(buffers, (bank,), slots)
    check_buffer_allocation(original, (bank,), {b.name: slots[b.name] for b in original})
    slots.pop("$readonly")
    sizes = dict.fromkeys(slots, TILE_BYTES)
    return cuts, slots, sizes, [vars(b) for b in buffers]


def regions(request):
    """Cut exact semantic edges; representation changes are checked below."""
    if (
        request.target_identity != TARGET_IDENTITY
        or request.lowering_policy != "diagnostic"
        or any(n.effect not in {"pure", "input", "constant"} for n in request.nodes)
    ):
        raise NativeCompilationError(
            "unsupported_semantics",
            "resident graph requires pure computations and the selected diagnostic target",
        )
    if request.physical_extents or request.boundary_layouts or request.output_layouts:
        raise NativeCompilationError(
            "unsupported_semantics",
            "resident graph requires plain logical matrices",
        )
    if any(
        len(n.type.shape) != 2 or any(d % 32 for d in n.type.shape)
        for n in request.nodes
    ):
        raise NativeCompilationError(
            "unsupported_semantics",
            "resident edges require tile-aligned matrices; internal tail repacking is not implemented",
        )
    pure = [n for n in request.nodes if n.effect == "pure"]
    users = {n.id: [] for n in request.nodes}
    for n in pure:
        for operand in n.inputs:
            users[operand].append(n.id)
    prefix = []
    for n in pure:
        if n.type.shape != (32, 32) or any(
            request.node(k).type.shape != (32, 32) for k in n.inputs
        ):
            break
        prefix.append(n.id)
    # A tile DAG currently exports completed BF16 tensors. Keep frontier packs
    # as explicit device conversions instead of lying about an FP8 output ABI.
    while prefix:
        cut = extract_semantic_region(request, tuple(prefix))
        packed_roots = {k for k in cut.outputs if request.node(k).type.dtype != "bf16"}
        if not packed_roots:
            break
        prefix = [k for k in prefix if k not in packed_roots]
    result = [tuple(prefix)] if prefix else []
    done = set(prefix)
    for n in pure:
        if n.id in done:
            continue
        keys = (n.id,)
        if n.op == "matmul" and n.id not in request.outputs and len(users[n.id]) == 1:
            child = request.node(users[n.id][0])
            if child.op == "relu" and child.inputs == (n.id,):
                keys += (child.id,)
        result.append(keys)
        done.update(keys)
    if done != {n.id for n in pure}:
        raise NativeCompilationError(
            "compile_error", "resident partition omitted a computation"
        )
    return tuple(extract_semantic_region(request, keys) for keys in result)


def boundary_request(region):
    """Expose completed bytes at a region boundary without re-quantizing them.

    A packed FP8 edge must already carry the selected RTL pack policy. Its
    admitted MXU input range remains a diagnostic execution obligation; pack
    saturation by itself does not prove that range for every possible input.
    """
    pure = [n for n in region.nodes if n.effect == "pure"]
    if tuple(n.op for n in pure) in {("matmul",), ("matmul", "relu")}:
        dot = pure[0]
        inputs = tuple(region.node(k) for k in dot.inputs)
        if (
            dot.type.numerical_policy != MXU0_POLICY
            or len(dot.inputs) != 2
            or any(
                n.effect != "input"
                or n.type.dtype != "fp8_e4m3"
                or n.type.numerical_policy
                not in {"raw-bit-pattern", ROUNDED_PACK_POLICY}
                for n in inputs
            )
        ):
            raise NativeCompilationError(
                "unsupported_semantics",
                "resident contraction needs admitted FP8 physical edges",
            )
        nodes = tuple(
            replace(
                n,
                type=replace(n.type, numerical_policy="raw-bit-pattern"),
                attrs=(("input_domain", MXU0_INPUT_DOMAIN),),
            )
            for n in inputs
        )
        return replace(
            region,
            nodes=(*nodes, *pure),
            input_storages=tuple((n.id, "external") for n in nodes),
        )
    if pure[0].op == "pack_fp8":
        pack = pure[0]
        source = region.node(pack.inputs[0])
        if (
            len(pure) != 1
            or source.effect != "input"
            or source.type.dtype != "bf16"
            or source.type.numerical_policy not in {"raw-bit-pattern", MXU0_POLICY}
        ):
            raise NativeCompilationError(
                "unsupported_semantics", "resident pack needs a completed BF16 input"
            )
        source = replace(
            source,
            type=TensorType(source.type.shape, "bf16", "raw-bit-pattern"),
            attrs=(("input_domain", ROUNDED_PACK_DOMAIN),),
        )
        return replace(
            region, nodes=(source, pack), input_storages=((source.id, "external"),)
        )
    if len(pure) == 1 and pure[0].op == "relu":
        source = region.node(pure[0].inputs[0])
        if (
            source.effect != "input"
            or source.type.dtype != "bf16"
            or source.type.numerical_policy not in {"raw-bit-pattern", MXU0_POLICY}
        ):
            raise NativeCompilationError(
                "unsupported_semantics", "resident ReLU needs a completed BF16 input"
            )
        source = replace(
            source,
            type=replace(source.type, numerical_policy="raw-bit-pattern"),
            attrs=(BF16_RELU_ATTRS[0],),
        )
        return replace(
            region, nodes=(source, pure[0]), input_storages=((source.id, "external"),)
        )
    if len(pure) == 1 and pure[0].op in BF16_BINARY_ATTRS:
        operation = pure[0]
        attrs = BF16_BINARY_ATTRS[operation.op]
        inputs = tuple(region.node(k) for k in operation.inputs)
        if any(
            n.effect not in {"input", "constant"}
            or n.type.dtype != "bf16"
            or n.type.numerical_policy not in {"raw-bit-pattern", MXU0_POLICY}
            for n in inputs
        ):
            raise NativeCompilationError(
                "unsupported_semantics",
                "resident pointwise operation needs completed BF16 inputs",
            )
        for constant in region.constants:
            raw = bytes.fromhex(constant.data_hex)
            if constant.encoding != "bf16-le-raw" or any(
                (code & 0x7F80) == 0x7F80 or ((code & 0x7F80) == 0 and code & 0x7F)
                for i in range(0, len(raw), 2)
                for code in (int.from_bytes(raw[i : i + 2], "little"),)
            ):
                raise NativeCompilationError(
                    "unsupported_semantics",
                    "resident pointwise constant exceeds finite-normal-or-zero BF16",
                )
        nodes = tuple(
            replace(
                n,
                op="input",
                effect="input",
                inputs=(),
                index_maps=(),
                type=replace(n.type, numerical_policy="raw-bit-pattern"),
                attrs=(attrs[0],),
            )
            for n in inputs
        )
        return replace(
            region,
            nodes=(*nodes, operation),
            constants=(),
            input_storages=tuple((n.id, "external") for n in nodes),
        )
    return region  # A complete, single-tile DAG retains its original policies.


def memory_plan(request):
    """Allocate one immutable frame and one private frame, using the shared allocator."""
    from merlin.semantic_compiler.buffer_allocation import Buffer, allocate_buffers

    if is_staged_bf16(request):
        return staged_memory_plan(request)[:3]
    cuts = regions(request)
    # Preserve each tile DAG's indexed ABI. Place its immutable operands first
    # in first-use order; large matrix operands follow in their own spans.
    source_ids = tuple(
        dict.fromkeys(
            n.id
            for cut in cuts
            for n in cut.nodes
            if request.node(n.id).effect in {"input", "constant"}
        )
    )
    sources = [request.node(k) for k in source_ids]
    if set(source_ids) != {
        n.id for n in request.nodes if n.effect in {"input", "constant"}
    }:
        raise NativeCompilationError(
            "unsupported_semantics", "resident graph contains unused boundary values"
        )
    written = [request.node(k) for region in cuts for k in region.outputs]
    names = [n.id for n in (*sources, *written)]
    if len(set(names)) != len(names):
        raise NativeCompilationError(
            "compile_error", "resident buffer has more than one producer"
        )
    sizes = {
        n.id: prod(n.type.shape) * (2 if n.type.dtype == "bf16" else 1)
        for n in (*sources, *written)
    }
    if any(n.type.dtype not in {"bf16", "fp8_e4m3"} for n in (*sources, *written)):
        raise NativeCompilationError(
            "unsupported_semantics", "resident buffer encoding is unsupported"
        )
    extents = {k: (size + TILE_BYTES - 1) // TILE_BYTES for k, size in sizes.items()}
    readonly = sum(extents[n.id] for n in sources)
    mutable = sum(extents[n.id] for n in written)
    bank = next(b for b in AtlasNativeBinding.profile().banks if b.name == "external")
    buffers = [
        Buffer("$readonly", bank.name, readonly, 0, 1, fixed_start=0),
        Buffer("$mutable", bank.name, mutable, 0, 1, fixed_start=readonly),
    ]
    for parent, nodes in (("$readonly", sources), ("$mutable", written)):
        offset = 0
        for n in nodes:
            buffers.append(
                Buffer(
                    n.id,
                    bank.name,
                    extents[n.id],
                    0,
                    1,
                    view_of=parent,
                    view_offset=offset,
                )
            )
            offset += extents[n.id]
    allocated = allocate_buffers(tuple(buffers), (bank,))
    if allocated.status != "sat":
        raise NativeCompilationError(
            "resource_limit" if allocated.status == "unsat" else "search_timeout",
            "resident DRAM allocation: " + allocated.reason,
        )
    slots = {k: v for k, v in allocated.addresses if k in names}
    return cuts, slots, sizes


def plan_abi(request):
    _, slots, _ = memory_plan(request)
    return {
        "fixed_inputs": {
            n.id: slots[n.id]
            for n in request.nodes
            if n.effect in {"input", "constant"}
        },
        "fixed_outputs": [slots[k] for k in request.outputs],
    }


def compose_fragments(fragments, *, tools, boundaries_lowered=False):
    """Link parsed functions; never rewrite SSA names or addresses as text."""
    from xdsl.dialects.builtin import ArrayAttr, ModuleOp, StringAttr
    from xdsl.dialects.func import FuncOp
    from xdsl.printer import Printer

    from merlin.frontends.linalg_mlir import parse_mlir_text

    functions, names = [], []
    for i, text in enumerate(fragments):
        explicit = text if boundaries_lowered else _run(
            [str(tools / "atlas-opt"), "--lower-atlas-virtual-boundary-to-dma", "-"],
            input_text=text,
        )
        module = parse_mlir_text(explicit)
        ops = list(module.body.block.ops)
        if len(ops) != 1 or not isinstance(ops[0], FuncOp) or module.attributes:
            raise NativeCompilationError(
                "compile_error",
                "resident fragment must contain one function and no module attributes",
            )
        function = ops[0]
        name = f"region_{i}"
        function.properties["sym_name"] = StringAttr(name)
        if "atlas.output_dram_bytes" not in function.attributes:
            # Indexed boundary lowering has already materialized every output;
            # an explicit byte range is supplied by the caller below.
            raise NativeCompilationError(
                "compile_error", "resident fragment lacks output allocation bounds"
            )
        module.body.block.detach_op(function)
        functions.append(function)
        names.append(StringAttr(name))
    linked = ModuleOp(functions, attributes={"atlas.program_entries": ArrayAttr(names)})
    stream = io.StringIO()
    Printer(stream=stream).print_op(linked)
    return _run(
        [str(tools / "atlas-opt"), "--compose-atlas-program", "-"],
        input_text=stream.getvalue(),
    )


@retry_without_operand_reuse
def compile_resident(
    snapshot,
    request,
    *,
    fixed_inputs,
    fixed_outputs,
    target_source,
    destination,
    limits,
):
    from merlin.semantic_compiler.codegen import (
        choose_reduction_unroll,
        codegen_options,
        current_codegen_options,
    )

    staged_bf16 = is_staged_bf16(request)
    deadline = monotonic() + limits.wall_timeout_s if staged_bf16 else None
    cuts, slots, sizes = memory_plan(request)
    abi = plan_abi(request)
    if fixed_inputs != abi["fixed_inputs"] or tuple(fixed_outputs) != tuple(
        abi["fixed_outputs"]
    ):
        raise NativeCompilationError(
            "unsupported_semantics",
            "resident graph requires its checked automatic frame layout",
        )
    destination.mkdir(parents=True, exist_ok=False)
    tools = Path(os.environ.get("ATLAS_OOT_BIN_DIR", target_source / "build/bin"))
    from xdsl.dialects.builtin import IntegerAttr
    from xdsl.printer import Printer

    from merlin.frontends.linalg_mlir import parse_mlir_text

    fragments, records = [], []
    key = AtlasNativeBinding.tuning_key(snapshot, request, fixed_inputs, fixed_outputs, limits)
    unroll, codegen = choose_reduction_unroll(key)
    codegen["operand_reuse_enabled"] = key["operand_reuse_enabled"]
    for i, region in enumerate(cuts):
        if deadline is not None and monotonic() >= deadline:
            raise NativeCompilationError("search_timeout", "staged compilation exceeded its total time budget")
        body = bf16_boundary_request(region) if staged_bf16 else (
            boundary_request(region)
            if len([n for n in region.nodes if n.effect == "pure"]) <= 2
            else region
        )
        folder = destination / "regions" / str(i)
        if body.nodes[-1].op == "pack_fp8":
            from .pack_tiling import compile_pack

            compile_region = compile_pack
        else:
            compile_region = AtlasNativeBinding.compile
        with codegen_options(
            replace(current_codegen_options(), reduction_unroll=str(unroll),
                    program_size_limit_bytes=codegen["program_size_limit_bytes"])
        ):
            extra = {"allow_staging": False} if compile_region == AtlasNativeBinding.compile else {}
            manifest = compile_region(
                snapshot,
                body,
                fixed_inputs={
                    n.id: slots[n.id]
                    for n in body.nodes
                    if n.effect in {"input", "constant"}
                },
                fixed_outputs=tuple(slots[k] for k in body.outputs),
                target_source=target_source,
                destination=folder,
                limits=replace(limits, wall_timeout_s=max(1, int(deadline - monotonic()))) if deadline else limits,
                **extra,
            )
        verify_native_publication(
            folder,
            manifest,
            engine="merlin_native",
            request_digest=body.digest(),
            target_identity=request.target_identity,
        )
        if manifest.get("artifact_kind") != "program" or manifest.get(
            "execution_requires"
        ):
            raise NativeCompilationError(
                "resource_limit",
                "resident region requires host streaming or segmentation",
            )
        explicit = _run(
            [str(tools / "atlas-opt"), "--lower-atlas-virtual-boundary-to-dma", "-"],
            input_text=(folder / "atlas-virtual.mlir").read_text(),
        )
        module = parse_mlir_text(explicit)
        function = next(iter(module.body.block.ops))
        start = min(slots[k] for k in body.outputs)
        end = max(
            slots[k] + (sizes[k] + TILE_BYTES - 1) // TILE_BYTES for k in body.outputs
        )
        function.attributes["atlas.output_dram_base"] = IntegerAttr(
            DRAM_BASE + start * TILE_BYTES, 64
        )
        function.attributes["atlas.output_dram_bytes"] = IntegerAttr(
            (end - start) * TILE_BYTES, 64
        )
        if staged_bf16:
            readonly = len(fixed_inputs)
            function.attributes["atlas.output_dram_base"] = IntegerAttr(DRAM_BASE + readonly * TILE_BYTES, 64)
            function.attributes["atlas.output_dram_bytes"] = IntegerAttr(
                (max(slots.values()) + 1 - readonly) * TILE_BYTES, 64
            )
        stream = io.StringIO()
        Printer(stream=stream).print_op(module)
        fragments.append(stream.getvalue())
        body_execution = json.loads((folder / "execution_plan.json").read_text())
        records.append(
            {
                "nodes": [n.id for n in region.nodes if n.effect == "pure"],
                "outputs": list(region.outputs),
                "request_digest": body.digest(),
                "manifest_sha256": hashlib.sha256(
                    (folder / "manifest.json").read_bytes()
                ).hexdigest(),
            }
        )
        if staged_bf16:
            records[-1]["source_slice_digest"] = region.digest()
            records[-1]["native_check_fingerprint"] = manifest["native_check_fingerprint"]
        if "reduction_extent" in body_execution:
            count = body_execution["tiling"]["loop_counts"][2]
            records[-1].update(
                reduction_tiles=count,
                effective_K_unroll=(
                    max(1, count - 1) if body_execution.get("operand_reuse") else min(unroll, max(1, count - 1))
                ),
                operand_reuse=body_execution.get("operand_reuse"),
            )
    # Staged fragments have already checked their local boundaries. Their
    # workspace reads are initialized by preceding fragments, so rechecking
    # them as isolated entries would discard that context. The joined CFG is
    # checked by the normal lowering and memory-initialization passes below.
    virtual = compose_fragments(fragments, tools=tools, boundaries_lowered=staged_bf16)
    (destination / "atlas-virtual.mlir").write_text(virtual)
    pipeline = _native_pipeline()
    machine = _run(
        [str(tools / "atlas-opt"), *NATIVE_PIPELINES[pipeline], "-"], input_text=virtual
    )
    _run(
        [str(tools / "atlas-opt"), *NATIVE_CHECKS, "--verify-atlas-timing", "-"],
        input_text=machine,
    )
    program = json.loads(
        _run([str(tools / "atlas-emit"), "--program-json", "-"], input_text=machine)
    )
    if program.get("selected_rtl_revision") != RTL_REVISION:
        raise NativeCompilationError(
            "unqualified_target", "resident emitter uses a different RTL"
        )
    binary = b"".join(
        row["word_u32"].to_bytes(4, "little") for row in program["instructions"]
    )
    execution = {
        "schema": "atlas.resident_graph_execution_plan.v1",
        "launches_per_invocation": 1,
        "inputs": [
            {
                "node": n.id,
                "address": DRAM_BASE + slots[n.id] * TILE_BYTES,
                "bytes": sizes[n.id],
                "type": n.type.record(),
            }
            for n in request.nodes
            if n.effect in {"input", "constant"}
        ],
        "outputs": [
            {
                "node": k,
                "address": DRAM_BASE + slots[k] * TILE_BYTES,
                "bytes": sizes[k],
                "type": request.node(k).type.record(),
            }
            for k in request.outputs
        ],
        "buffers": [
            {"node": k, "address": DRAM_BASE + slots[k] * TILE_BYTES, "bytes": sizes[k]}
            for k in slots
        ],
        "regions": records,
        "lowering_pipeline": pipeline,
        "codegen": codegen,
        "limitations": [
            "tile_aligned_edges",
            "no_dram_reuse",
            "packed_mxu_input_domains_require_execution_check",
        ],
    }
    if staged_bf16:
        execution.update(
            schema="atlas.native_tile_execution_plan.v8", shape=[32, 32], dtype="bf16",
            input_addresses=[DRAM_BASE + fixed_inputs[n.id] * TILE_BYTES
                             for n in request.nodes if n.effect in {"input", "constant"}],
            output_addresses=[DRAM_BASE + slot * TILE_BYTES for slot in fixed_outputs],
            input_bytes=TILE_BYTES, output_bytes=TILE_BYTES,
            output_physical_layouts=["bf16_pair_halves"] * len(request.outputs),
            operation="bf16_staged", numerical_policy="per-operation-declared",
            staging_workspace={"address": DRAM_BASE + len(fixed_inputs) * TILE_BYTES,
                               "bytes": (max(slots.values()) + 1 - len(fixed_inputs)) * TILE_BYTES},
            limitations=["diagnostic_primitive_domains_require_execution_check"],
        )
    for name, data in {
        "source-request.json": request.record(),
        "execution_plan.json": execution,
        "physical-program.json": program,
    }.items():
        (destination / name).write_text(
            json.dumps(data, sort_keys=True, indent=2) + "\n"
        )
    (destination / "atlas-virtual.mlir").write_text(virtual)
    (destination / "atlas-machine.mlir").write_text(machine)
    def sha(data):
        return hashlib.sha256(data).hexdigest()
    constants = []
    if request.constants:
        from .layout import pack_bf16_matrix

        (destination / "constants").mkdir()
        for i, binding in enumerate(request.constants):
            if binding.encoding != "bf16-le-raw":
                raise NativeCompilationError(
                    "unsupported_semantics", "resident constant encoding is unsupported"
                )
            payload = pack_bf16_matrix(
                bytes.fromhex(binding.data_hex),
                request.node(binding.node_id).type.shape,
            )
            relative = f"constants/constant-{i:03}.bin"
            (destination / relative).write_bytes(payload)
            constants.append(
                {
                    "node_id": binding.node_id,
                    "address": DRAM_BASE + slots[binding.node_id] * TILE_BYTES,
                    "bytes": len(payload),
                    "encoding": binding.encoding,
                    "physical_layout": "bf16_pair_halves" if staged_bf16 else "mn_tiles_bf16_pair_halves",
                    "path": relative,
                    "sha256": sha(payload),
                }
            )
    if staged_bf16:
        execution["constant_inputs"] = constants
        (destination / "execution_plan.json").write_text(json.dumps(execution, sort_keys=True, indent=2) + "\n")
        selection = {
            "schema": "merlin.native_staged_selection.v1", "engine": "merlin_native",
            "request_digest": request.digest(), "target_identity": request.target_identity,
            "check_scope": "exact_source_partition_and_checked_region_materializations",
            "regions": records,
        }
        fingerprint = sha(json.dumps(selection, sort_keys=True).encode())
        selection["check_fingerprint"] = fingerprint
        (destination / "native-selection.json").write_text(json.dumps(selection, sort_keys=True, indent=2) + "\n")
    manifest = {
        "schema": "atlas.native_resident_graph_compilation.v1",
        "artifact_kind": "program",
        "engine": "merlin_native",
        "compile_mode": "diagnostic",
        "qualification_scope": "diagnostic_only",
        "request_digest": request.digest(),
        "target_identity": request.target_identity,
        "binary_sha256": sha(binary),
        "execution_plan_sha256": sha(
            (destination / "execution_plan.json").read_bytes()
        ),
        "constant_inputs": constants,
        "native_snapshot_sha256": sha((snapshot.root / "manifest.json").read_bytes()),
        "io_memory_bytes": dram_slots() * TILE_BYTES,
        "target_inventory_sha256": sha(
            (target_source / "docs/selected-variant-inventory.json").read_bytes()
        ),
        "provider_sources_sha256": {
            p.name: sha(p.read_bytes())
            for p in sorted(Path(__file__).parent.glob("*.py"))
        },
        "atlas_opt_sha256": sha((tools / "atlas-opt").read_bytes()),
        "atlas_emit_sha256": sha((tools / "atlas-emit").read_bytes()),
        "stage_artifacts": {
            p.relative_to(destination).as_posix(): sha(p.read_bytes())
            for p in sorted(destination.rglob("*"))
            if p.is_file()
        },
    }
    if staged_bf16:
        manifest["native_check_fingerprint"] = fingerprint
        if deadline is not None and monotonic() >= deadline:
            raise NativeCompilationError("search_timeout", "staged publication exceeded its total time budget")
    manifest["stage_artifacts"]["program.bin"] = sha(binary)
    (destination / "program.bin").write_bytes(binary)
    (destination / "manifest.json").write_text(
        json.dumps(manifest, sort_keys=True, indent=2) + "\n"
    )
    return manifest
