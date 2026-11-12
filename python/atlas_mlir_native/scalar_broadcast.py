"""Exact immutable scalar broadcasts using counted raw-byte DMA tile copies.

The target compiler rematerializes one immutable tile. The device repeats that
tile; invocation-time C only transports bytes and launches one checked image.
No floating-point interpretation, conversion, or arithmetic is performed.
"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
from math import prod
import os
from pathlib import Path

from merlin.semantic_compiler.model import IndexMap, KernelRequest, pointwise_identity_maps
from merlin.semantic_compiler.target_binding import NativeCompilationError, verify_native_publication

from .binding import DRAM_BASE, NATIVE_CHECKS, RTL_REVISION, TARGET_IDENTITY, TILE_BYTES, _native_pipeline, _scheduled_virtual_pipeline, _run, dram_slots

BYTE_TILE = 1024


@dataclass(frozen=True)
class BroadcastPlan:
    source: KernelRequest
    scalar: bytes
    output_bytes: int

    @property
    def tiles(self):
        return (self.output_bytes + BYTE_TILE - 1) // BYTE_TILE

    @property
    def physical_bytes(self):
        return self.tiles * BYTE_TILE

    @property
    def tile(self):
        return self.scalar * (BYTE_TILE // len(self.scalar))

    def record(self):
        return dict(schema="atlas.scalar_broadcast.v1", source_request_digest=self.source.digest(),
                    scalar_node=self.source.nodes[0].id, scalar_hex=self.scalar.hex(),
                    output_node=self.source.outputs[0], output_bytes=self.physical_bytes,
                    logical_output_bytes=self.output_bytes, output_addresses=[DRAM_BASE + TILE_BYTES],
                    input_addresses=[DRAM_BASE], input_payload_bytes=[BYTE_TILE],
                    physical_output_bytes=self.physical_bytes, tile_bytes=BYTE_TILE,
                    tiles=self.tiles, tile_sha256=sha256(self.tile).hexdigest(),
                    constant_materialization="one immutable raw-byte tile at target compilation",
                    runtime_arithmetic="none", padding="private final tile repeats scalar bytes")


def prepare_source(source):
    """Admit the complete immutable scalar gather, including its exact maps."""
    if (source.target_identity != TARGET_IDENTITY or source.lowering_policy != "diagnostic"
            or len(source.nodes) != 2 or len(source.constants) != 1 or source.physical_extents
            or source.boundary_layouts or source.output_layouts):
        raise ValueError("scalar broadcast needs one immutable scalar and one dense gather")
    scalar, output = source.nodes
    rank = len(output.type.shape)
    if (scalar.op != "constant" or scalar.effect != "constant" or scalar.inputs or scalar.attrs
            or scalar.index_maps or scalar.type.shape != () or scalar.type.numerical_policy != "raw-bit-pattern"
            or output.op != "gather" or output.effect != "pure" or output.attrs
            or output.inputs != (scalar.id,) or not rank
            or (output.type.dtype, output.type.numerical_policy) != (scalar.type.dtype, scalar.type.numerical_policy)
            or output.index_maps != (IndexMap(rank, (), ()), pointwise_identity_maps(rank, 1)[0])
            or source.outputs != (output.id,) or source.output_storages != ("external",)
            or source.input_storages):
        raise ValueError("scalar broadcast source, maps, type, or boundary differs")
    binding = source.constants[0]
    if binding.node_id != scalar.id:
        raise ValueError("scalar broadcast constant binding differs")
    binding.validate_type(scalar.type)
    raw = bytes.fromhex(binding.data_hex)
    if len(raw) not in (1, 2, 4, 8):
        raise ValueError("scalar broadcast needs whole-byte scalar storage up to eight bytes")
    return BroadcastPlan(source, raw, prod(output.type.shape) * len(raw))


def virtual_program(plan, *, size_limit=None):
    """Existing typed byte transport retains full CFG, DMA, and buffer proofs."""
    output = DRAM_BASE + TILE_BYTES
    if output + plan.physical_bytes > DRAM_BASE + dram_slots() * TILE_BYTES:
        raise NativeCompilationError("resource_limit", "scalar broadcast exceeds selected I/O window")
    signed = lambda n: n if n < 2**31 else n - 2**32
    budget = f"atlas.program_size_limit_bytes = {size_limit} : i64, " if size_limit is not None else ""
    state, tile = "!atlas.virtual_state", "!atlas.virtual_fp8"
    return f'''module {{
  func.func @scalar_broadcast() -> {state} attributes {{{budget}atlas.input_dram_base = {DRAM_BASE} : i64, atlas.output_dram_base = {output} : i64, atlas.output_dram_bytes = {plan.physical_bytes} : i64}} {{
    %s = "atlas.virtual_start"() : () -> {state}
    %base = arith.constant {signed(DRAM_BASE)} : i32
    %bytes = arith.constant {BYTE_TILE} : i32
    %loaded, %tile = "atlas.virtual_dma_load_fp8"(%s,%base,%bytes) : ({state},i32,i32) -> ({state},{tile})
    %zero = arith.constant 0 : i32
    %obase = arith.constant {signed(output)} : i32
    cf.br ^loop(%loaded,%zero,%obase : {state},i32,i32)
  ^loop(%token: {state},%i: i32,%op: i32):
    %count = arith.constant {plan.tiles} : i32
    %more = arith.cmpi ult, %i, %count : i32
    cf.cond_br %more, ^body(%token : {state}), ^done(%token : {state})
  ^body(%t: {state}):
    %store_bytes = arith.constant {BYTE_TILE} : i32
    %stored = "atlas.virtual_dma_store_fp8"(%t,%tile,%op,%store_bytes) : ({state},{tile},i32,i32) -> {state}
    %one = arith.constant 1 : i32
    %next = arith.addi %i, %one : i32
    %onext = arith.addi %op, %store_bytes : i32
    cf.br ^loop(%stored,%next,%onext : {state},i32,i32)
  ^done(%end: {state}):
    return %end : {state}
  }}
}}
'''


def compile_source(snapshot, source, *, target_source, destination, limits):
    from merlin.semantic_compiler.codegen import current_codegen_options

    plan = prepare_source(source)
    virtual = virtual_program(plan, size_limit=current_codegen_options().program_size_limit_bytes)
    tools = Path(os.environ.get("ATLAS_OOT_BIN_DIR", Path(target_source) / "build/bin"))
    pipeline = _native_pipeline()
    machine = _run([str(tools / "atlas-opt"), *_scheduled_virtual_pipeline(pipeline), "-"], input_text=virtual)
    _run([str(tools / "atlas-opt"), *NATIVE_CHECKS, "--verify-atlas-timing", "-"], input_text=machine)
    program = json.loads(_run([str(tools / "atlas-emit"), "--program-json", "-"], input_text=machine))
    if program.get("selected_rtl_revision") != RTL_REVISION:
        raise NativeCompilationError("unqualified_target", "scalar broadcast emitter RTL differs")
    binary = b"".join(row["word_u32"].to_bytes(4, "little") for row in program["instructions"])
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=False)
    files = {"source-request.json": source.record(), "execution_plan.json": plan.record(),
             "physical-program.json": program}
    for member, record in files.items():
        (destination / member).write_text(json.dumps(record, sort_keys=True, indent=2) + "\n")
    for member, payload in {"constant-tile.bin": plan.tile, "program.bin": binary,
                            "atlas-virtual.mlir": virtual.encode(), "atlas-machine.mlir": machine.encode()}.items():
        (destination / member).write_bytes(payload)
    manifest = dict(schema="atlas.native_scalar_broadcast.v1", artifact_kind="program", engine="merlin_native",
                    request_digest=source.digest(), target_identity=source.target_identity,
                    binary_sha256=sha256(binary).hexdigest(),
                    execution_plan_sha256=sha256((destination / "execution_plan.json").read_bytes()).hexdigest(),
                    lowering_pipeline=pipeline, selected_rtl_revision=RTL_REVISION,
                    stage_artifacts={p.name: sha256(p.read_bytes()).hexdigest() for p in sorted(destination.iterdir())})
    (destination / "manifest.json").write_text(json.dumps(manifest, sort_keys=True, indent=2) + "\n")
    return manifest


def prepare_c_package(source, compiled, name, *, target_source, runtime_linkage="separate"):
    from merlin.compile.c_sources import c_source_pair
    from .c_runtime import program_array, runtime_headers, runtime_members

    plan = prepare_source(source)
    compiled, target_source = Path(compiled), Path(target_source)
    manifest = json.loads((compiled / "manifest.json").read_text())
    verify_native_publication(compiled, manifest, engine="merlin_native", request_digest=source.digest(), target_identity=source.target_identity)
    if (json.loads((compiled / "source-request.json").read_text()) != source.record()
            or json.loads((compiled / "execution_plan.json").read_text()) != plan.record()
            or (compiled / "constant-tile.bin").read_bytes() != plan.tile):
        raise ValueError("scalar broadcast source or constant tile differs from publication")
    # Replay source lowering and final encoding, not only mutable manifest hashes.
    tools = Path(os.environ.get("ATLAS_OOT_BIN_DIR", target_source / "build/bin"))
    virtual = (compiled / "atlas-virtual.mlir").read_text()
    from merlin.semantic_compiler.codegen import current_codegen_options
    if virtual != virtual_program(plan, size_limit=current_codegen_options().program_size_limit_bytes):
        raise ValueError("scalar broadcast virtual source differs")
    machine = _run([str(tools / "atlas-opt"), *_scheduled_virtual_pipeline(manifest["lowering_pipeline"]), "-"], input_text=virtual)
    if machine != (compiled / "atlas-machine.mlir").read_text():
        raise ValueError("scalar broadcast machine source differs")
    program = json.loads(_run([str(tools / "atlas-emit"), "--program-json", "-"], input_text=machine))
    binary = b"".join(row["word_u32"].to_bytes(4, "little") for row in program["instructions"])
    if binary != (compiled / "program.bin").read_bytes():
        raise ValueError("scalar broadcast program differs")
    upper = name.upper()
    declarations = f'''
#define {upper}_WORKSPACE_BYTES 1u
#define {upper}_SOURCE_WORKSPACE_BYTES 1u
typedef enum {{ {upper}_OK=0,{upper}_INVALID,{upper}_MEMORY_ERROR,{upper}_DEVICE_ERROR }} {name}_status;
typedef struct {{ atlas_host *host; atlas_tile_memory_io io; size_t max_polls; atlas_host_buffer buffers[2]; }} {name}_context;
typedef struct {{ {name}_context *region; }} {name}_source_context;
{name}_status {name}_bind({name}_context *,atlas_host *,atlas_tile_memory_io,void *,size_t,size_t);
{name}_status {name}_source_bind({name}_source_context *,{name}_context *,void *,size_t);
int {name}_source_dispatch(void *,const void *const *,void *const *);
'''
    implementation = f'''
#include <string.h>
{program_array(name + '_words', binary)}
static const uint8_t {name}_tile[{BYTE_TILE}]={{{','.join(map(str, plan.tile))}}};
static int {name}_valid(const void *p,size_t n) {{ return p && (uintptr_t)p<=UINTPTR_MAX-n; }}
static int {name}_overlap(const void *a,size_t an,const void *b,size_t bn) {{
  return (uintptr_t)a<(uintptr_t)b+bn && (uintptr_t)b<(uintptr_t)a+an;
}}
{name}_status {name}_bind({name}_context *ctx,atlas_host *host,atlas_tile_memory_io io,void *scratch,size_t bytes,size_t polls) {{
  (void)scratch;(void)bytes;
  if (!ctx || !host || !host->mmio.read32 || !host->mmio.write32 || host->active || !io.write || !io.read || !polls ||
      {name}_overlap(ctx,sizeof(*ctx),host,sizeof(*host))) return {upper}_INVALID;
  *ctx=({name}_context){{host,io,polls,{{{{{DRAM_BASE}ULL,{BYTE_TILE}u,ATLAS_HOST_DEVICE_READ}},
    {{{DRAM_BASE + TILE_BYTES}ULL,{plan.physical_bytes}u,ATLAS_HOST_DEVICE_WRITE}}}}}};
  return {upper}_OK;
}}
{name}_status {name}_source_bind({name}_source_context *ctx,{name}_context *region,void *scratch,size_t bytes) {{
  (void)scratch;(void)bytes;
  if (!ctx || !region || !region->host || region->host->active ||
      {name}_overlap(ctx,sizeof(*ctx),region,sizeof(*region)) ||
      {name}_overlap(ctx,sizeof(*ctx),region->host,sizeof(*region->host))) return {upper}_INVALID;
  ctx->region=region;return {upper}_OK;
}}
int {name}_source_dispatch(void *opaque,const void *const *inputs,void *const *outputs) {{
  (void)inputs;
  {name}_source_context *ctx=opaque;
  if (!ctx || !ctx->region || !outputs || !{name}_valid(outputs[0],{plan.output_bytes}u)) return {upper}_INVALID;
  {name}_context *region=ctx->region;
  if (!region->host || region->host->active || !region->io.write || !region->io.read || !region->max_polls ||
      {name}_overlap(outputs[0],{plan.output_bytes}u,ctx,sizeof(*ctx)) ||
      {name}_overlap(outputs[0],{plan.output_bytes}u,region,sizeof(*region)) ||
      {name}_overlap(outputs[0],{plan.output_bytes}u,region->host,sizeof(*region->host))) return {upper}_INVALID;
  if (region->io.write(region->io.context,{DRAM_BASE}ULL,{name}_tile,{BYTE_TILE}u)) return {upper}_MEMORY_ERROR;
  atlas_host_result status=atlas_host_ensure_loaded(region->host,{name}_words,{len(binary)//4}u);
  if (status==ATLAS_HOST_OK) status=atlas_host_start(region->host,region->buffers,2u);
  if (status==ATLAS_HOST_OK) status=atlas_host_wait(region->host,region->max_polls);
  if (status!=ATLAS_HOST_OK) return {upper}_DEVICE_ERROR;
  if (region->io.read(region->io.context,{DRAM_BASE + TILE_BYTES}ULL,outputs[0],{plan.output_bytes}u)) return {upper}_MEMORY_ERROR;
  return {upper}_OK;
}}
'''
    public = runtime_headers(target_source / "runtime")
    members, files, runtime = runtime_members(public, (target_source / "runtime/atlas_host.c").read_text(), runtime_linkage)
    rendered = c_source_pair(name, public_headers=public, private_headers={}, sources={**members, "broadcast.c": implementation}, declarations=declarations)
    files.update({key: value.encode() for key, value in rendered.items()})
    files["program.bin"] = binary
    record = dict(kind="scalar_broadcast", workspace_bytes=1, source_workspace_bytes=1,
                  context_type=name + "_context", source_context_type=name + "_source_context",
                  binding_function=name + "_bind", source_binding_function=name + "_source_bind",
                  source_dispatch_function=name + "_source_dispatch", dispatch_function=name + "_source_dispatch",
                  input_node_ids=[], output_node_ids=list(source.outputs), launches_per_invocation=1,
                  scalar_broadcast_plan=plan.record(), native_manifest_sha256=sha256((compiled / "manifest.json").read_bytes()).hexdigest(), **runtime)
    return record, files
