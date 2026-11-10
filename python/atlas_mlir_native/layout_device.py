"""Exact affine layouts expressed as typed virtual DMA, slot and XLU operations.

Only raw bytes cross the host boundary. The common compiler allocates and times
these operations and verifies its six ordinary source contracts.
"""
from __future__ import annotations
from dataclasses import dataclass
from hashlib import sha256
from math import prod, gcd
from pathlib import Path
import json
import os

from merlin.compile.layout_c import _element_bytes, _flat_map
from merlin.semantic_compiler.model import IndexMap, pointwise_identity_maps
from merlin.semantic_compiler.target_binding import NativeCompilationError, verify_native_publication
from .binding import DRAM_BASE, TARGET_IDENTITY, TILE_BYTES, RTL_REVISION, NATIVE_CHECKS, _native_pipeline, _scheduled_virtual_pipeline, _run, dram_slots


def _aligned(n):
    return (n + 31) // 32 * 32


def _strides(shape, width=1):
    return tuple(prod(shape[i + 1:]) * width for i in range(len(shape)))


def _flatten(mapping, shape, width):
    strides = _strides(shape, width)
    return (sum(a * b for a, b in zip(mapping.offsets, strides, strict=True)),
            tuple(sum(row[j] * stride for row, stride in zip(mapping.coefficients, strides, strict=True))
                  for j in range(mapping.loop_rank)))


@dataclass(frozen=True)
class LayoutPlan:
    source: object
    operands: tuple
    output: object
    width: int
    segments: tuple
    io_memory_base: int = DRAM_BASE

    @property
    def input_bytes(self):
        return tuple(prod(node.type.shape) * self.width for node in self.operands)

    @property
    def input_addresses(self):
        sizes = self.input_bytes
        return tuple(self.io_memory_base + sum(_aligned(n) for n in sizes[:i]) for i in range(len(sizes)))

    @property
    def output_bytes(self):
        return prod(self.output.type.shape) * self.width

    @property
    def output_address(self):
        return self.io_memory_base + sum(_aligned(n) for n in self.input_bytes)

    @property
    def physical_bytes(self):
        return _aligned(self.output_bytes)

    @property
    def workspace_bytes(self):
        return self.output_address - self.io_memory_base + self.physical_bytes

    def record(self):
        return dict(schema="atlas.layout_device.v1", source_request_digest=self.source.digest(),
                    input_addresses=list(self.input_addresses), input_payload_bytes=[_aligned(n) for n in self.input_bytes],
                    input_logical_bytes=list(self.input_bytes), output_addresses=[self.output_address],
                    output_bytes=self.physical_bytes, logical_output_bytes=self.output_bytes,
                    workspace_bytes=self.workspace_bytes, segments=list(self.segments),
                    launches_per_invocation=1, runtime_arithmetic="none; raw-byte transport only",
                    weight_conversion="none", padding="private input tail zero-filled; output tail private")


def _segment(index, shape, source_strides, output_strides, source_offset, output_offset, width):
    # Collapse the common dense suffix into aligned DMA runs. The remainder is
    # a small counted affine iteration space, independent of tensor names.
    shape, ss, ds = list(shape), list(source_strides), list(output_strides)
    block = width
    while shape and ss[-1] == ds[-1] == block:
        block *= shape.pop()
        ss.pop()
        ds.pop()
    if block % 32 == 0 and source_offset % 32 == output_offset % 32 == 0 and all(s % 32 == 0 for s in (*ss, *ds)):
        chunk = gcd(block, 4096)
        if chunk < block:
            shape.append(block // chunk)
            ss.append(chunk)
            ds.append(chunk)
        mode, block = "dma", chunk
    else:
        # Scalar fallback traverses bytes in dense output order, with one
        # zero-initialized 32-byte output window and aligned source windows.
        # It is needed for small ragged inner transposes such as 64 by 5.
        mode = "scalar"
        shape, ss, ds = list(shape), list(ss), list(ds)
        if block != 1:
            shape.append(block)
            ss.append(1)
            ds.append(1)
        block = 1
    return dict(input=index, mode=mode, counts=shape, source_strides=ss, output_strides=ds,
                source_offset=source_offset, output_offset=output_offset, block_bytes=block,
                source_row_bytes=0, output_row_bytes=0)


def prepare_source(source, *, io_memory_base=DRAM_BASE):
    """Plan byte transport in an explicitly selected device I/O window."""
    if (type(io_memory_base) is not int or io_memory_base < 0 or io_memory_base % 32
            or io_memory_base + dram_slots() * TILE_BYTES > 1 << 32):
        raise ValueError("device I/O window must be 32-byte aligned within the 32-bit buffer ABI")
    if (source.target_identity != TARGET_IDENTITY or source.lowering_policy != "diagnostic"
            or source.physical_extents or source.boundary_layouts or source.output_layouts
            or source.output_storages != ("external",)):
        raise ValueError("device layout needs selected target and dense external output")
    computations = [node for node in source.nodes if node.effect == "pure"]
    if len(computations) != 1:
        raise ValueError("device layout needs one pure source operation")
    output = computations[0]
    if source.outputs != (output.id,) or output.op not in {"reshape", "copy", "gather", "transpose", "concat"}:
        raise ValueError("unsupported device layout operation")
    operands = tuple(source.node(key) for key in output.inputs)
    if (not operands or len({node.id for node in operands}) != len(operands)
            or {node.id for node in source.nodes} != {output.id, *(node.id for node in operands)}
            or any(node.effect not in {"input", "constant"} or node.inputs or node.index_maps or node.attrs
                   or node.op != node.effect
                   or (node.type.dtype, node.type.numerical_policy) != (output.type.dtype, output.type.numerical_policy)
                   for node in operands)):
        raise ValueError("device layout needs complete dense operands with unchanged element types")
    if dict(source.input_storages) != {node.id: "external" for node in operands if node.effect == "input"}:
        raise ValueError("device layout input storage differs")
    width = _element_bytes(output.type.dtype)
    if width not in (1, 2, 4, 8) or any(not 0 < prod(n.type.shape) * width < 1 << 28 for n in (*operands, output)):
        raise ValueError("device layout extent or element width exceeds selected address domain")
    constants = {b.node_id: b for b in source.constants}
    if set(constants) != {node.id for node in operands if node.effect == "constant"}:
        raise ValueError("device layout constant bindings differ")
    for node in operands:
        if node.effect == "constant":
            constants[node.id].validate_type(node.type)
    shape = output.type.shape
    identity = pointwise_identity_maps(len(shape), 1)[0]
    segments = []
    if output.op in {"reshape", "copy"}:
        if (len(operands) != 1 or output.attrs or output.index_maps
                or prod(operands[0].type.shape) != prod(shape)):
            raise ValueError("device reshape/copy must preserve dense element order and count")
        segments.append(_segment(0, (prod(shape),), (width,), (width,), 0, 0, width))
    elif output.op in {"gather", "transpose"}:
        if len(operands) != 1 or output.attrs or len(output.index_maps) != 2 or output.index_maps[-1] != identity:
            raise ValueError("device gather needs one affine source map and identity output")
        mapping = output.index_maps[0]
        _flat_map(mapping, shape, operands[0].type.shape)
        if output.op == "transpose" and (len(mapping.coefficients) != len(shape)
                or mapping.offsets != (0,) * len(shape)
                or any(set(row) - {0, 1} or sum(row) != 1 for row in mapping.coefficients)
                or len({row.index(1) for row in mapping.coefficients}) != len(shape)
                or tuple(shape[row.index(1)] for row in mapping.coefficients) != operands[0].type.shape):
            raise ValueError("device transpose needs a complete dimension permutation")
        offset, strides = _flatten(mapping, operands[0].type.shape, width)
        if (output.op == "transpose" and width == 1 and len(shape) >= 2
                and shape[-1] % 32 == shape[-2] % 32 == 0
                and mapping.coefficients == tuple(tuple(int(j == (i if i < len(shape)-2 else 2*len(shape)-3-i))
                                                     for j in range(len(shape))) for i in range(len(shape)))):
            counts = (*shape[:-2], shape[-2] // 32, shape[-1] // 32)
            ss = (*strides[:-2], strides[-2] * 32, strides[-1] * 32)
            ds = (*_strides(shape)[:-2], shape[-1] * 32, 32)
            segments.append(dict(input=0, mode="xlu", counts=list(counts), source_strides=list(ss),
                                 output_strides=list(ds), source_offset=0, output_offset=0, block_bytes=1024,
                                 source_row_bytes=operands[0].type.shape[-1], output_row_bytes=shape[-1]))
        else:
            segments.append(_segment(0, shape, strides, _strides(shape, width), offset, 0, width))
    else:
        axis = dict(output.attrs).get("axis")
        if (type(axis) is not int or not 0 <= axis < len(shape) or output.attrs != (("axis", axis),)
                or output.index_maps or len(operands) < 2
                or any(len(node.type.shape) != len(shape) or any(n != shape[i] for i, n in enumerate(node.type.shape) if i != axis) for node in operands)
                or sum(node.type.shape[axis] for node in operands) != shape[axis]):
            raise ValueError("device concat axis or dimensions differ")
        start = 0
        for index, node in enumerate(operands):
            segments.append(_segment(index, node.type.shape, _strides(node.type.shape, width), _strides(shape, width), 0,
                                     start * _strides(shape, width)[axis], width))
            start += node.type.shape[axis]
    if any(s["mode"] == "scalar" for s in segments):
        if len(segments) != 1 or segments[0]["output_offset"] or tuple(segments[0]["output_strides"]) != _strides(tuple(segments[0]["counts"])):
            raise ValueError("ragged segmented scalar scatter is outside selected layout contract")
        if prod(segments[0]["counts"]) > 16384:
            raise ValueError("ragged scalar layout exceeds default proof work budget; aligned DMA/XLU needed")
    plan = LayoutPlan(source, operands, output, width, tuple(segments), io_memory_base)
    if plan.workspace_bytes > dram_slots() * TILE_BYTES or plan.io_memory_base + plan.workspace_bytes > 1 << 32:
        raise ValueError("device layout buffers exceed selected I/O window")
    if not (plan.io_memory_base >= 0x80000000 or
            0x08000000 <= plan.io_memory_base and
            plan.io_memory_base + plan.workspace_bytes <= 0x08010000):
        raise ValueError("device layout buffers leave selected DMA memory windows")
    return plan


def virtual_program(plan, *, size_limit=None):
    """Build byte transport from typed virtual DMA, slots, XLU and counted CFG.

    Allocation, instruction timing, DMA waits and all six source contracts are
    produced by the common compiler. No physical location appears in this IR.
    """
    import itertools

    state, tile, spill = "!atlas.virtual_state", "!atlas.virtual_fp8", "!atlas.virtual_spill"
    signed = lambda n: n if n < 2**31 else n - 2**32
    budget = f"atlas.program_size_limit_bytes = {size_limit} : i64, " if size_limit is not None else ""
    lines = ["module {", f" func.func @layout() -> {state} attributes {{{budget}atlas.input_dram_base = {plan.io_memory_base} : i64, atlas.output_dram_base = {plan.output_address} : i64, atlas.output_dram_bytes = {plan.physical_bytes} : i64}} {{",
             f'  %s0 = "atlas.virtual_start"() : () -> {state}']
    current = "%s0"
    serial = 0

    def fresh(prefix):
        nonlocal serial
        serial += 1
        return f"%{prefix}{serial}"

    def constant(n):
        v = fresh("c")
        lines.append(f"  {v} = arith.constant {signed(n)} : i32")
        return v

    def add(base, n):
        if not n:
            return base
        rhs, value = constant(n), fresh("p")
        lines.append(f"  {value} = arith.addi {base}, {rhs} : i32")
        return value

    def slot():
        value = fresh("slot")
        lines.append(f'  {value} = "atlas.virtual_slot"() {{kind = "fp8", words = 256 : i32, slot_word = -1 : i32, purpose = "scratch"}} : () -> {spill}')
        return value

    # Slots are declared in the entry block and receive addresses from spilling.
    source_slot, output_slot, transpose_slot = slot(), slot(), slot()

    def load(address, size=32):
        nonlocal current
        n, token, value = constant(size), fresh("s"), fresh("tile")
        lines.append(f'  {token}, {value} = "atlas.virtual_dma_load_fp8"({current}, {address}, {n}) : ({state}, i32, i32) -> ({state}, {tile})')
        current = token
        return value

    def store(value, address, size=32):
        nonlocal current
        n, token = constant(size), fresh("s")
        lines.append(f'  {token} = "atlas.virtual_dma_store_fp8"({current}, {value}, {address}, {n}) : ({state}, {tile}, i32, i32) -> {state}')
        current = token

    def save(value, memory):
        nonlocal current
        token = fresh("s")
        lines.append(f'  {token} = "atlas.virtual_save"({current}, {memory}, {value}) : ({state}, {spill}, {tile}) -> {state}')
        current = token

    def restore(memory):
        nonlocal current
        token, value = fresh("s"), fresh("tile")
        lines.append(f'  {token}, {value} = "atlas.virtual_restore"({current}, {memory}) : ({state}, {spill}) -> ({state}, {tile})')
        current = token
        return value

    def scalar_copy(source_memory, source_offset, target_memory, target_offset, width=1):
        nonlocal current
        token, value = fresh("s"), fresh("byte")
        kind = {1: "lbu", 2: "lhu", 4: "lw"}[width]
        lines.append(f'  {token}, {value} = "atlas.virtual_slot_load"({current}, {source_memory}) {{kind = "{kind}", byte_offset = {source_offset} : i32}} : ({state}, {spill}) -> ({state}, i32)')
        current = token
        token = fresh("s")
        kind = {1: "sb", 2: "sh", 4: "sw"}[width]
        lines.append(f'  {token} = "atlas.virtual_slot_store"({current}, {target_memory}, {value}) {{kind = "{kind}", byte_offset = {target_offset} : i32}} : ({state}, {spill}, i32) -> {state}')
        current = token

    def repeat(count, source, destination, source_stride, destination_stride, body):
        nonlocal current
        if count == 1:
            body(source, destination)
            return
        label = fresh("loop")[1:]
        zero = constant(0)
        lines.append(f"  cf.br ^{label}({current}, {zero}, {source}, {destination} : {state}, i32, i32, i32)")
        token, index, sp, dp = fresh("s"), fresh("i"), fresh("p"), fresh("p")
        lines.append(f" ^{label}({token}: {state}, {index}: i32, {sp}: i32, {dp}: i32):")
        limit, more = constant(count), fresh("more")
        lines.append(f"  {more} = arith.cmpi ult, {index}, {limit} : i32")
        lines.append(f"  cf.cond_br {more}, ^{label}_body({token} : {state}), ^{label}_done({token} : {state})")
        before = fresh("s")
        lines.append(f" ^{label}_body({before}: {state}):")
        current = before
        # A constant pointer is retained directly when its stride is zero.
        body(sp if source_stride else source, dp if destination_stride else destination)
        next_index = add(index, 1)
        next_source = add(sp, source_stride)
        next_destination = add(dp, destination_stride)
        lines.append(f"  cf.br ^{label}({current}, {next_index}, {next_source}, {next_destination} : {state}, i32, i32, i32)")
        current = fresh("s")
        lines.append(f" ^{label}_done({current}: {state}):")

    for segment in plan.segments:
        source_base = plan.input_addresses[segment["input"]] + segment["source_offset"]
        destination_base = plan.output_address + segment["output_offset"]
        if segment["mode"] == "scalar":
            # A dense output suffix repeats in a counted loop whenever both
            # pointers advance by DMA beats. Ragged rows may be grouped to
            # reach that alignment without interpreting any stored bits.
            counts = list(segment["counts"])
            ss, ds = segment["source_strides"], segment["output_strides"]
            repeat_count, source_step, output_step = 1, 0, 0
            for dimension, count in enumerate(counts):
                if count == 1:
                    continue
                if ss[dimension] >= 0:
                    group = 32 // gcd(ss[dimension], 32)
                    group = group * (32 // gcd(ds[dimension], 32)) // gcd(group, 32 // gcd(ds[dimension], 32))
                    if count % group == 0:
                        counts[dimension] = group
                        repeat_count = count // group
                        source_step, output_step = ss[dimension] * group, ds[dimension] * group
                break
            mapping = {}
            for coordinate in itertools.product(*(range(n) for n in counts)):
                source = source_base % 32 + sum(i * stride for i, stride in zip(coordinate, ss))
                destination = sum(i * stride for i, stride in zip(coordinate, ds))
                mapping[destination] = source
            # Negative affine maps remain compile-time addresses; the counted
            # address proof intentionally accepts only forward nonwrapping runs.
            source_origin = source_base // 32 * 32
            minimum = min(mapping.values()) // 32 * 32
            source_origin += minimum
            mapping = {destination: source - minimum for destination, source in mapping.items()}

            def scalar_transport(sp, dp):
                for output_start in range(0, len(mapping), 32):
                    members = [(i, mapping[i]) for i in range(output_start, min(output_start + 32, len(mapping)))]
                    windows = sorted({source // 32 * 32 for _, source in members})
                    if all(source == windows[0] + i - output_start for i, source in members):
                        store(load(add(sp, windows[0])), add(dp, output_start))
                        continue
                    for window_index, window in enumerate(windows):
                        value = load(add(sp, window))
                        save(value, source_slot)
                        if window_index == 0:
                            save(value, output_slot)
                        for i, source in members:
                            if source // 32 * 32 == window:
                                scalar_copy(source_slot, source - window, output_slot, i - output_start)
                    store(restore(output_slot), add(dp, output_start))

            repeat(repeat_count, constant(source_origin), constant(destination_base),
                   source_step, output_step, scalar_transport)
            continue

        def transport(sp, dp):
            if segment["mode"] == "dma":
                for start in range(0, segment["block_bytes"], 1024):
                    size = min(1024, segment["block_bytes"] - start)
                    store(load(add(sp, start), size), add(dp, start), size)
                return
            # Assemble the strided 32x32 raw tile, transpose with the existing
            # XLU op, and scatter its rows through 32-byte DMA windows.
            if segment["source_row_bytes"] == 32:
                value = load(sp, 1024)
                save(value, output_slot)
            else:
                rows_per_window = max(1, (1024 - 32) // segment["source_row_bytes"] + 1)
                for first_row in range(0, 32, rows_per_window):
                    rows = min(rows_per_window, 32 - first_row)
                    size = (rows - 1) * segment["source_row_bytes"] + 32
                    loaded = load(add(sp, first_row * segment["source_row_bytes"]), size)
                    save(loaded, source_slot)
                    if first_row == 0:
                        save(loaded, output_slot)
                        save(loaded, transpose_slot)
                    for row in range(first_row, first_row + rows):
                        for word in range(8):
                            scalar_copy(source_slot, (row - first_row) * segment["source_row_bytes"] + word * 4,
                                        transpose_slot, row * 32 + word * 4, 4)
                value = restore(transpose_slot)
            transposed = fresh("tile")
            lines.append(f'  {transposed} = "atlas.virtual_xlu_transpose"({value}) : ({tile}) -> {tile}')
            if segment["output_row_bytes"] == 32:
                store(transposed, dp, 1024)
            else:
                save(transposed, transpose_slot)
                for row in range(32):
                    for word in range(8):
                        scalar_copy(transpose_slot, row * 32 + word * 4, output_slot, word * 4, 4)
                    store(restore(output_slot), add(dp, row * segment["output_row_bytes"]))

        counts = segment["counts"]
        ss, ds = segment["source_strides"], segment["output_strides"]
        def affine_runs(dimension, sp, dp):
            if dimension == len(counts):
                transport(sp, dp)
                return
            if ss[dimension] >= 0 and ds[dimension] >= 0:
                repeat(counts[dimension], sp, dp, ss[dimension], ds[dimension],
                       lambda source, destination: affine_runs(dimension + 1, source, destination))
            else:
                for index in range(counts[dimension]):
                    affine_runs(dimension + 1, add(sp, index * ss[dimension]), add(dp, index * ds[dimension]))
        affine_runs(0, constant(source_base), constant(destination_base))
    lines.extend((f"  return {current} : {state}", " }", "}", ""))
    return "\n".join(lines)

def compile_source(snapshot, source, *, target_source, destination, limits, io_memory_base=DRAM_BASE):
    plan = prepare_source(source, io_memory_base=io_memory_base)
    from . import layout_prefix_application
    schedule = layout_prefix_application.prepare_plan(plan)
    if schedule is not None:
        if io_memory_base != DRAM_BASE:
            raise ValueError("custom I/O base is unsupported for layout prefix schedules")
        return layout_prefix_application.compile_source(snapshot, source, schedule,
            target_source=target_source, destination=destination, limits=limits)
    from merlin.semantic_compiler.codegen import current_codegen_options
    virtual = virtual_program(plan, size_limit=current_codegen_options().program_size_limit_bytes)
    pipeline = _native_pipeline()
    tools = Path(os.environ.get("ATLAS_OOT_BIN_DIR", Path(target_source) / "build/bin"))
    machine = _run([str(tools / "atlas-opt"), *_scheduled_virtual_pipeline(pipeline), "-"], input_text=virtual)
    # The mandatory pipeline ends in the private finalizer's full
    # verifyAtlasArtifact(llvmBlock=true), including all six contracts, strict
    # timing/CFG DMA, machine/LLVM checks, and the original timing metadata.
    # Its immutable result needs no duplicate postcheck. Optimized retains it.
    if pipeline != "mandatory":
        _run([str(tools / "atlas-opt"), *NATIVE_CHECKS, "--verify-atlas-timing", "-"], input_text=machine)
    program = json.loads(_run([str(tools / "atlas-emit"), "--program-json", "-"], input_text=machine))
    if program.get("selected_rtl_revision") != RTL_REVISION:
        raise NativeCompilationError("unqualified_target", "layout emitter RTL differs")
    binary = b"".join(row["word_u32"].to_bytes(4, "little") for row in program["instructions"])
    from .restoration_budget import instruction_bound
    try:
        bound = instruction_bound(virtual, machine, program)
    except (ValueError, KeyError, IndexError) as error:
        raise NativeCompilationError("resource_limit", "layout instruction bound is unproved: " + str(error)) from error
    if not bound["admitted"]:
        raise NativeCompilationError("resource_limit", "layout worst-case instruction bound exceeds 1000000: " + str(bound["instruction_bound"]))
    budget = current_codegen_options().program_size_limit_bytes
    if budget is not None and len(binary) > budget:
        raise ValueError("device layout exceeds selected program budget")
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=False)
    for member, record in {"source-request.json": source.record(), "execution_plan.json": plan.record(), "physical-program.json": program,
                           "layout-instruction-bound.json": bound}.items():
        (destination / member).write_text(json.dumps(record, sort_keys=True, indent=2) + "\n")
    for member, payload in {"program.bin": binary, "atlas-virtual.mlir": virtual.encode(), "atlas-machine.mlir": machine.encode()}.items():
        (destination / member).write_bytes(payload)
    manifest = dict(schema="atlas.native_layout_device.v1", artifact_kind="program", engine="merlin_native",
                    request_digest=source.digest(), target_identity=source.target_identity,
                    binary_sha256=sha256(binary).hexdigest(), selected_rtl_revision=RTL_REVISION, lowering_pipeline=pipeline,
                    execution_plan_sha256=sha256((destination / "execution_plan.json").read_bytes()).hexdigest(),
                    stage_artifacts={p.name: sha256(p.read_bytes()).hexdigest() for p in sorted(destination.iterdir())})
    (destination / "manifest.json").write_text(json.dumps(manifest, sort_keys=True, indent=2) + "\n")
    return manifest


def prepare_c_package(source, compiled, name, *, target_source, runtime_linkage="separate", io_memory_base=DRAM_BASE):
    from merlin.compile.c_sources import c_source_pair
    from .c_runtime import program_array, runtime_headers, runtime_members
    plan = prepare_source(source, io_memory_base=io_memory_base)
    compiled, target_source = Path(compiled), Path(target_source)
    if (compiled / "layout-prefix-manifest.json").exists():
        if io_memory_base != DRAM_BASE:
            raise ValueError("custom I/O base is unsupported for layout prefix schedules")
        from .layout_prefix_application import prepare_c_package as prepare_prefix
        return prepare_prefix(source, compiled, name, target_source=target_source, runtime_linkage=runtime_linkage)
    manifest = json.loads((compiled / "manifest.json").read_text())
    verify_native_publication(compiled, manifest, engine="merlin_native", request_digest=source.digest(), target_identity=source.target_identity)
    from merlin.semantic_compiler.codegen import current_codegen_options
    virtual = virtual_program(plan, size_limit=current_codegen_options().program_size_limit_bytes)
    if (json.loads((compiled / "source-request.json").read_text()) != source.record()
            or json.loads((compiled / "execution_plan.json").read_text()) != plan.record()
            or (compiled / "atlas-virtual.mlir").read_text() != virtual):
        raise ValueError("device layout source or execution plan differs from publication")
    tools = Path(os.environ.get("ATLAS_OOT_BIN_DIR", target_source / "build/bin"))
    machine = _run([str(tools / "atlas-opt"), *_scheduled_virtual_pipeline(manifest["lowering_pipeline"]), "-"], input_text=virtual)
    if machine != (compiled / "atlas-machine.mlir").read_text():
        raise ValueError("device layout machine source differs")
    program = json.loads(_run([str(tools / "atlas-emit"), "--program-json", "-"], input_text=machine))
    binary = b"".join(row["word_u32"].to_bytes(4, "little") for row in program["instructions"])
    if binary != (compiled / "program.bin").read_bytes():
        raise ValueError("device layout binary differs")
    from .restoration_budget import instruction_bound
    bound = instruction_bound(virtual, machine, program)
    if not bound["admitted"] or json.loads((compiled / "layout-instruction-bound.json").read_text()) != bound:
        raise ValueError("device layout instruction bound differs or exceeds default")
    inputs = [key for key, _ in source.input_storages]
    bindings = {b.node_id: b for b in source.constants}
    upper = name.upper()
    buffers = [f"{{{address}ULL,{_aligned(size)}u,ATLAS_HOST_DEVICE_READ}}" for address, size in zip(plan.input_addresses, plan.input_bytes)]
    buffers.append(f"{{{plan.output_address}ULL,{plan.physical_bytes}u,ATLAS_HOST_DEVICE_WRITE}}")
    constants, writes, checks = [], [], []
    for index, node in enumerate(plan.operands):
        size, address = plan.input_bytes[index], plan.input_addresses[index]
        if node.effect == "constant":
            raw = bytes.fromhex(bindings[node.id].data_hex)
            constants.append(f"static const uint8_t {name}_constant_{index}[{len(raw)}]={{{','.join(map(str, raw))}}};")
            pointer = f"{name}_constant_{index}"
        else:
            port = inputs.index(node.id)
            pointer = f"inputs[{port}]"
            checks.extend((f"!{name}_valid({pointer},{size}u)",
                           f"{name}_overlap({pointer},{size}u,outputs[0],{plan.output_bytes}u)"))
        writes.append(f"if (region->io.write(region->io.context,{address}ULL,{pointer},{size}u)) return {upper}_MEMORY_ERROR;")
        if size % 32:
            writes.append(f"if (region->io.write(region->io.context,{address + size}ULL,{name}_zero,{_aligned(size)-size}u)) return {upper}_MEMORY_ERROR;")
    for pointer in ["outputs[0]", *(f"inputs[{inputs.index(node.id)}]" for node in plan.operands if node.effect == "input")]:
        size = plan.output_bytes if pointer == "outputs[0]" else plan.input_bytes[next(i for i, n in enumerate(plan.operands) if n.effect == "input" and f"inputs[{inputs.index(n.id)}]" == pointer)]
        checks.extend(f"{name}_overlap({pointer},{size}u,{context},sizeof(*{context}))" for context in ("ctx", "region", "region->host"))
    declarations = f'''
#define {upper}_WORKSPACE_BYTES 1u
#define {upper}_SOURCE_WORKSPACE_BYTES 1u
typedef enum {{ {upper}_OK=0,{upper}_INVALID,{upper}_MEMORY_ERROR,{upper}_DEVICE_ERROR }} {name}_status;
typedef struct {{ atlas_host *host; atlas_tile_memory_io io; size_t max_polls; atlas_host_buffer buffers[{len(buffers)}]; }} {name}_context;
typedef struct {{ {name}_context *region; }} {name}_source_context;
{name}_status {name}_bind({name}_context *,atlas_host *,atlas_tile_memory_io,void *,size_t,size_t);
{name}_status {name}_source_bind({name}_source_context *,{name}_context *,void *,size_t);
int {name}_source_dispatch(void *,const void *const *,void *const *);
'''
    implementation = f'''
#include <string.h>
{program_array(name + '_words', binary)}
{os.linesep.join(constants)}
''' + (f"static const uint8_t {name}_zero[32]={{0}};\n" if any(size % 32 for size in plan.input_bytes) else "") + f'''
static int {name}_valid(const void *p,size_t n) {{ return p && (uintptr_t)p<=UINTPTR_MAX-n; }}
static int {name}_overlap(const void *a,size_t an,const void *b,size_t bn) {{
  return (uintptr_t)a<(uintptr_t)b+bn && (uintptr_t)b<(uintptr_t)a+an;
}}
{name}_status {name}_bind({name}_context *ctx,atlas_host *host,atlas_tile_memory_io io,void *scratch,size_t bytes,size_t polls) {{
  (void)scratch;(void)bytes;
  if (!ctx || !host || !host->mmio.read32 || !host->mmio.write32 || host->active || !io.write || !io.read || !polls ||
      {name}_overlap(ctx,sizeof(*ctx),host,sizeof(*host))) return {upper}_INVALID;
  *ctx=({name}_context){{host,io,polls,{{{','.join(buffers)}}}}};return {upper}_OK;
}}
{name}_status {name}_source_bind({name}_source_context *ctx,{name}_context *region,void *scratch,size_t bytes) {{
  (void)scratch;(void)bytes;
  if (!ctx || !region || !region->host || region->host->active ||
      {name}_overlap(ctx,sizeof(*ctx),region,sizeof(*region)) ||
      {name}_overlap(ctx,sizeof(*ctx),region->host,sizeof(*region->host))) return {upper}_INVALID;
  ctx->region=region;return {upper}_OK;
}}
int {name}_source_dispatch(void *opaque,const void *const *inputs,void *const *outputs) {{
  {'(void)inputs;' if not inputs else ''}
  {name}_source_context *ctx=opaque;
  if (!ctx || !ctx->region || {'!inputs || ' if inputs else ''}!outputs || !{name}_valid(outputs[0],{plan.output_bytes}u)) return {upper}_INVALID;
  {name}_context *region=ctx->region;
  if (!region->host || region->host->active || !region->io.write || !region->io.read || !region->max_polls ||
      {' || '.join(checks)}) return {upper}_INVALID;
  {os.linesep.join(writes)}
  atlas_host_result status=atlas_host_ensure_loaded(region->host,{name}_words,{len(binary)//4}u);
  if (status==ATLAS_HOST_OK) status=atlas_host_start(region->host,region->buffers,{len(buffers)}u);
  if (status==ATLAS_HOST_OK) status=atlas_host_wait(region->host,region->max_polls);
  if (status!=ATLAS_HOST_OK) return {upper}_DEVICE_ERROR;
  if (region->io.read(region->io.context,{plan.output_address}ULL,outputs[0],{plan.output_bytes}u)) return {upper}_MEMORY_ERROR;
  return {upper}_OK;
}}
'''
    public = runtime_headers(target_source / "runtime")
    members, files, runtime = runtime_members(public, (target_source / "runtime/atlas_host.c").read_text(), runtime_linkage)
    rendered = c_source_pair(name, public_headers=public, private_headers={}, sources={**members, "layout.c": implementation}, declarations=declarations)
    files.update({key: value.encode() for key, value in rendered.items()})
    files["program.bin"] = binary
    record = dict(kind="layout_device", workspace_bytes=1, source_workspace_bytes=1,
                  context_type=name + "_context", source_context_type=name + "_source_context",
                  binding_function=name + "_bind", source_binding_function=name + "_source_bind",
                  source_dispatch_function=name + "_source_dispatch", dispatch_function=name + "_source_dispatch",
                  input_node_ids=inputs, output_node_ids=list(source.outputs), launches_per_invocation=1,
                  layout_device_plan=plan.record(), native_manifest_sha256=sha256((compiled / "manifest.json").read_bytes()).hexdigest(), **runtime)
    return record, files
