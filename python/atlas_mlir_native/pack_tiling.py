"""Keep BF16-to-FP8 conversion on Atlas between tiled regions.

The buffer ABI is row-tile then column-tile order. BF16 tiles contain the
two physical register halves; FP8 tiles are compact row-major bytes. Every
physical input lane, including padding, must be initialized by its producer.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass, replace
from math import prod
from pathlib import Path

from merlin.semantic_compiler.model import (
    KernelRequest,
    TensorType,
    pointwise_identity_maps,
)
from merlin.semantic_compiler.target_binding import NativeCompilationError

from .binding import (
    PACK_RULE_BY_POLICY,
    ROUNDED_PACK_DOMAIN,
    ROUNDED_PACK_POLICY,
    TARGET_IDENTITY,
    AtlasNativeBinding,
)


@dataclass(frozen=True)
class PackTiling:
    request: KernelRequest
    tile_shape: tuple[int, int]
    scale_code: int

    @property
    def counts(self):
        return tuple(
            (n + t - 1) // t
            for n, t in zip(
                self.request.nodes[0].type.shape, self.tile_shape, strict=True
            )
        )

    @property
    def tiles(self):
        return prod(self.counts)

    @property
    def buffer_bytes(self):
        cells = self.tiles * prod(self.tile_shape)
        return cells * 2, cells

    def body_request(self):
        return replace(
            self.request,
            nodes=tuple(
                replace(n, type=replace(n.type, shape=self.tile_shape))
                for n in self.request.nodes
            ),
        )

    def record(self):
        return {
            "schema": "atlas.pack_tiling.v1",
            "source_request_digest": self.request.digest(),
            "shape": list(self.request.nodes[0].type.shape),
            "tile_shape": list(self.tile_shape),
            "tile_counts": list(self.counts),
            "input_bytes": self.buffer_bytes[0],
            "output_bytes": self.buffer_bytes[1],
            "input_layout": "mn_tiles_bf16_pair_halves",
            "output_layout": "mn_tiles_fp8_row_major",
            "padding": "all_physical_input_lanes_initialized_by_producer",
            "numerical_policy": ROUNDED_PACK_POLICY,
            "scale_code": self.scale_code,
            "body_request_digest": self.body_request().digest(),
        }


def prepare_plan(request):
    if (
        len(request.nodes) != 2
        or request.target_identity != TARGET_IDENTITY
        or request.lowering_policy != "diagnostic"
        or request.constants
        or request.physical_extents
        or request.boundary_layouts
        or request.output_layouts
    ):
        raise NativeCompilationError(
            "unsupported_semantics", "pack loop needs one plain diagnostic conversion"
        )
    source, result = request.nodes
    code = dict(result.attrs).get("scale_code")
    if (
        source.op != "input"
        or source.effect != "input"
        or source.inputs
        or source.index_maps
        or len(source.type.shape) != 2
        or source.type != TensorType(source.type.shape, "bf16", "raw-bit-pattern")
        or source.attrs != (("input_domain", ROUNDED_PACK_DOMAIN),)
        or result.op != "pack_fp8"
        or result.effect != "pure"
        or result.inputs != (source.id,)
        or result.type != TensorType(source.type.shape, "fp8_e4m3", ROUNDED_PACK_POLICY)
        or result.index_maps != pointwise_identity_maps(2, 1)
        or result.attrs != (("input_domain", ROUNDED_PACK_DOMAIN), ("scale_code", code))
        or type(code) is not int
        or request.input_storages != ((source.id, "external"),)
        or request.outputs != (result.id,)
        or request.output_storages != ("external",)
    ):
        raise NativeCompilationError(
            "unsupported_semantics",
            "pack loop types, policy, maps or boundaries differ",
        )
    descriptor = next(
        d
        for d in AtlasNativeBinding.profile().descriptors
        if d.name == PACK_RULE_BY_POLICY[ROUNDED_PACK_POLICY]
    )
    bound = next(b for b in descriptor.parameter_attrs if b.name == "scale_code")
    if not bound.minimum <= code <= bound.maximum:
        raise NativeCompilationError(
            "unsupported_semantics", "pack scale exceeds the selected descriptor"
        )
    if any(axis.minimum != axis.maximum for axis in descriptor.output_axis_bounds):
        raise NativeCompilationError(
            "unqualified_target", "pack loop needs a fixed instruction tile"
        )
    shape = tuple(axis.minimum for axis in descriptor.output_axis_bounds)
    return PackTiling(request, shape, code)


def select_body(snapshot, plan, *, limits):
    """Select and check the conversion; a virtual opcode alone is not selection."""
    request = plan.body_request()
    selected = snapshot.select(request, limits=limits)
    if (
        selected.status != "selected"
        or not selected.graph
        or not selected.rules
        or not selected.allocation
    ):
        raise NativeCompilationError(
            selected.status, "native pack-body selection: " + selected.reason
        )
    instructions = [
        selected.rules.symbols[value.symbol].get("descriptor", {}).get("name")
        for value in selected.graph.values
        if value.kind == "instruction"
    ]
    expected = [
        "atlas.load_bf16_tile",
        PACK_RULE_BY_POLICY[ROUNDED_PACK_POLICY],
        "atlas.store_fp8_rounded_tile",
    ]
    if instructions != expected:
        raise NativeCompilationError(
            "unqualified_target", "pack loop selected an unadmitted realization"
        )
    return {
        "engine": selected.engine,
        "request_digest": request.digest(),
        "source_request_digest": plan.request.digest(),
        "check_fingerprint": selected.check_fingerprint,
        "candidate_digest": selected.candidate.digest(),
        "instructions": instructions,
    }


def virtual_program(plan, input_address, output_address, *, size_limit=None):
    """One loop over compact tiles, before the existing allocator and scheduler."""
    from .binding import DRAM_BASE, TILE_BYTES, dram_slots

    sizes = plan.buffer_bytes
    if size_limit is not None and (type(size_limit) is not int or size_limit <= 0):
        raise NativeCompilationError(
            "compile_error", "pack program size limit must be positive"
        )
    budget = (
        f"atlas.program_size_limit_bytes = {size_limit} : i64, "
        if size_limit is not None
        else ""
    )
    ranges = []
    for address, size in zip((input_address, output_address), sizes, strict=True):
        if (
            type(address) is not int
            or address < DRAM_BASE
            or address % TILE_BYTES
            or address + size > DRAM_BASE + dram_slots() * TILE_BYTES
        ):
            raise NativeCompilationError(
                "resource_limit", "pack buffer exceeds the selected diagnostic window"
            )
        ranges.append((address, address + size))
    if ranges[0][0] < ranges[1][1] and ranges[1][0] < ranges[0][1]:
        raise NativeCompilationError(
            "infeasible_candidate", "pack input and output buffers overlap"
        )
    # The output owns complete allocation slots. The final partial slot's
    # private padding is permitted but never written by this loop.
    capacity = (sizes[1] + TILE_BYTES - 1) // TILE_BYTES * TILE_BYTES
    signed = lambda value: value if value < 1 << 31 else value - (1 << 32)
    cells = prod(plan.tile_shape)
    state, bf16, fp8 = (
        "!atlas.virtual_state",
        "!atlas.virtual_bf16",
        "!atlas.virtual_fp8",
    )
    return f"""module {{
  func.func @pack_tiles() -> {state} attributes {{{budget}atlas.input_dram_base = {input_address} : i64, atlas.output_dram_base = {output_address} : i64, atlas.output_dram_bytes = {capacity} : i64}} {{
    %s = "atlas.virtual_start"() : () -> {state}
    %zero = arith.constant 0 : i32
    %abase = arith.constant {signed(input_address)} : i32
    %obase = arith.constant {signed(output_address)} : i32
    cf.br ^loop(%s,%zero,%abase,%obase : {state},i32,i32,i32)
  ^loop(%token: {state},%i: i32,%ap: i32,%op: i32):
    %count = arith.constant {plan.tiles} : i32
    %more = arith.cmpi ult, %i, %count : i32
    cf.cond_br %more, ^body(%token : {state}), ^done(%token : {state})
  ^body(%t: {state}):
    %input_bytes = arith.constant {cells * 2} : i32
    %output_bytes = arith.constant {cells} : i32
    %s1, %a = "atlas.virtual_dma_load_bf16"(%t,%ap,%input_bytes) : ({state},i32,i32) -> ({state},{bf16})
    %r = "atlas.virtual_pack_fp8"(%a) {{scale_code = {plan.scale_code} : i32}} : ({bf16}) -> {fp8}
    %s2 = "atlas.virtual_dma_store_fp8"(%s1,%r,%op,%output_bytes) : ({state},{fp8},i32,i32) -> {state}
    %one = arith.constant 1 : i32
    %next = arith.addi %i, %one : i32
    %an = arith.addi %ap, %input_bytes : i32
    %on = arith.addi %op, %output_bytes : i32
    cf.br ^loop(%s2,%next,%an,%on : {state},i32,i32,i32)
  ^done(%end: {state}):
    return %end : {state}
  }}
}}
"""


def compile_pack(
    snapshot,
    request,
    *,
    fixed_inputs,
    fixed_outputs,
    target_source,
    destination,
    limits,
):
    """Compile one complete packed conversion; no CPU computation is inserted."""
    from .binding import (
        DRAM_BASE,
        NATIVE_CHECKS,
        NATIVE_PIPELINES,
        RTL_REVISION,
        TILE_BYTES,
        _native_pipeline,
        _run,
    )

    plan = prepare_plan(request)
    if set(fixed_inputs) != {request.nodes[0].id} or len(fixed_outputs) != 1:
        raise NativeCompilationError(
            "compile_error", "pack loop needs one input and one output binding"
        )
    slots = (fixed_inputs[request.nodes[0].id], fixed_outputs[0])
    if any(type(slot) is not int or slot < 0 for slot in slots):
        raise NativeCompilationError(
            "compile_error", "pack bindings must be nonnegative allocation slots"
        )
    addresses = tuple(DRAM_BASE + slot * TILE_BYTES for slot in slots)
    from merlin.semantic_compiler.codegen import current_codegen_options

    virtual = virtual_program(
        plan, *addresses, size_limit=current_codegen_options().program_size_limit_bytes
    )
    evidence = select_body(snapshot, plan, limits=limits)
    destination.mkdir(parents=True, exist_ok=False)
    tools = Path(os.environ.get("ATLAS_OOT_BIN_DIR", target_source / "build/bin"))
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
            "unqualified_target", "pack emitter uses a different RTL"
        )
    binary = b"".join(
        row["word_u32"].to_bytes(4, "little") for row in program["instructions"]
    )
    execution = {
        "schema": "atlas.pack_execution_plan.v1",
        "input_addresses": [addresses[0]],
        "input_buffer_bytes": [plan.buffer_bytes[0]],
        "output_addresses": [addresses[1]],
        "output_bytes": plan.buffer_bytes[1],
        "output_physical_layouts": ["mn_tiles_fp8_row_major"],
        "tiling": plan.record(),
    }
    for name, data in {
        "source-request.json": request.record(),
        "execution_plan.json": execution,
        "pack-tiling.json": plan.record(),
        "native-selection.json": evidence,
        "physical-program.json": program,
    }.items():
        (destination / name).write_text(
            json.dumps(data, sort_keys=True, indent=2) + "\n"
        )
    (destination / "atlas-virtual.mlir").write_text(virtual)
    (destination / "atlas-machine.mlir").write_text(machine)
    (destination / "program.bin").write_bytes(binary)
    sha = lambda data: hashlib.sha256(data).hexdigest()
    manifest = {
        "schema": "atlas.native_pack_compilation.v1",
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
        "body_check_fingerprint": evidence["check_fingerprint"],
        "lowering_pipeline": pipeline,
        "native_snapshot_sha256": sha((snapshot.root / "manifest.json").read_bytes()),
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
            p.name: sha(p.read_bytes())
            for p in sorted(destination.iterdir())
            if p.is_file()
        },
    }
    (destination / "manifest.json").write_text(
        json.dumps(manifest, sort_keys=True, indent=2) + "\n"
    )
    return manifest
