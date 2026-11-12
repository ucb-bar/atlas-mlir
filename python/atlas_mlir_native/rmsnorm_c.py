"""Export authenticated RMSNorm row loops through the shared host C runtime."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from merlin.compile.c_conversions import render_f32_array_c
from merlin.compile.c_sources import c_source_pair
from merlin.semantic_compiler.target_binding import verify_native_publication

from .binding import AtlasNativeBinding, DRAM_BASE, TILE_BYTES
from .c_runtime import program_array, runtime_headers, runtime_members
from .row_loop import verify_row_loop


def prepare_c_package(precision, compiled, name, *, target_source, runtime_linkage="separate"):
    """Replay the selected row loop before publishing its source-float ABI."""
    compiled, target_source = Path(compiled), Path(target_source)
    if precision.request.source_identity != precision.source.digest():
        raise ValueError("RMSNorm precision request is not bound to its source")
    plan, manifest = verify_row_loop(precision.request, compiled, target_source=target_source)
    before = (compiled / "manifest.json").read_bytes()
    binary = (compiled / "program.bin").read_bytes()
    record, files = _render_package(precision, plan, binary, name, target_source=target_source,
                                    runtime_linkage=runtime_linkage)
    verify_native_publication(compiled, manifest, engine="merlin_native",
                              request_digest=precision.request.digest(),
                              target_identity=precision.request.target_identity)
    if before != (compiled / "manifest.json").read_bytes() or binary != (compiled / "program.bin").read_bytes():
        raise ValueError("RMSNorm publication changed during C export")
    return {**record, "native_manifest_sha256": hashlib.sha256(before).hexdigest()}, files


def _render_package(precision, plan, binary, name, *, target_source, runtime_linkage):
    request = precision.request
    if (plan.tiling.source != request or request.nodes[-1].op != "rmsnorm"
            or len(precision.input_ids) != 2 or len(set(precision.input_ids)) != 2
            or len(precision.operand_specs) != 2):
        raise ValueError("RMSNorm C export needs its checked two-input precision plan")
    m, n = plan.tiling.shape
    if n % 32 or n < 32 or m < 1:
        raise ValueError("RMSNorm C export requires complete column tiles")
    counts = (m * n, n)
    if (len(precision.source.outputs) != 1
            or precision.source.node(precision.source.outputs[0]).type.shape != (m, n)
            or precision.source.node(precision.source.outputs[0]).type.dtype != "f32"):
        raise ValueError("RMSNorm source result shape or type differs")
    for node_id, shape in zip(precision.input_ids, ((m, n), (n,)), strict=True):
        node = precision.source.node(node_id)
        if (node.type.shape != shape or node.type.dtype != "f32"
                or node.type.numerical_policy != "raw-bit-pattern" or node.effect not in {"input", "constant"}):
            raise ValueError("RMSNorm source boundary shape, type or role differs")
    body = plan.tiling.regions[0].request
    abi = AtlasNativeBinding.plan_boundary_abi(body)
    columns = n // 32
    expected_inputs = tuple(f"data_{i}" for i in range(columns)) + tuple(f"gamma_{i}" for i in range(columns))
    if (tuple(key for key, _ in body.input_storages) != expected_inputs
            or len(body.outputs) != columns or plan.output_bytes != columns * TILE_BYTES):
        raise ValueError("RMSNorm selected body has an unsupported boundary layout")
    constants = {row.node_id: row for row in body.constants}
    if set(abi["fixed_inputs"]) != {*expected_inputs, *constants}:
        raise ValueError("RMSNorm selected ABI does not account for every input and constant")
    if list(abi["fixed_inputs"].values()) != list(range(len(abi["fixed_inputs"]))):
        raise ValueError("RMSNorm readonly ABI is not dense")
    rows = []
    for row in plan.tiling.regions:
        if row.row_start % 32 or row.valid_rows != min(32, m - row.row_start):
            raise ValueError("RMSNorm row frame or tail differs")
        rows.append(row.row_start)
    if rows != list(range(0, m, 32)):
        raise ValueError("RMSNorm C export has missing or duplicate row groups")
    readonly_slots = len(abi["fixed_inputs"])
    if plan.readonly_bytes != readonly_slots * TILE_BYTES or plan.writable_bytes % TILE_BYTES:
        raise ValueError("RMSNorm frame byte sizes differ from the selected ABI")
    constant_arrays, constant_stages = [], []
    for index, (key, binding) in enumerate(constants.items()):
        raw = bytes.fromhex(binding.data_hex)
        if binding.encoding != "bf16-le-raw" or len(raw) != TILE_BYTES:
            raise ValueError("RMSNorm literal requires one raw BF16 tile")
        paired = bytearray(TILE_BYTES)
        for r in range(32):
            for c in range(32):
                logical = 2 * (r * 32 + c)
                physical = (c // 16) * 1024 + (r * 16 + c % 16) * 2
                paired[physical:physical + 2] = raw[logical:logical + 2]
        constant_arrays.append(f"static const uint8_t {name}_constant_{index}[2048]={{" +
                               ",".join(str(x) for x in paired) + "};")
        address = DRAM_BASE + abi["fixed_inputs"][key] * TILE_BYTES
        constant_stages.append(f"    if (region->io.write(region->io.context,{address}ULL+group*{plan.readonly_bytes}u,"
                               f"{name}_constant_{index},2048u)) return {name.upper()}_MEMORY_ERROR;")
    input_addresses = [DRAM_BASE + abi["fixed_inputs"][key] * TILE_BYTES for key in expected_inputs]
    output_addresses = [DRAM_BASE + slot * TILE_BYTES for slot in abi["fixed_outputs"]]
    # Every address is replayed through the checked row plan, including private
    # scratch. The C loop applies precisely these two selected frame strides.
    for group in range(plan.groups):
        for address in (*input_addresses, *output_addresses,
                        *(DRAM_BASE + abi["fixed_inputs"][key] * TILE_BYTES for key in constants)):
            plan.address(address, group)
    raw_bytes, source_bytes = TILE_BYTES + 4 * counts[0], 2 * sum(counts)
    upper = name.upper()
    declarations = f"""
#define {upper}_WORKSPACE_BYTES {raw_bytes}u
#define {upper}_SOURCE_WORKSPACE_BYTES {source_bytes}u
#define {upper}_SOURCE_INPUT_COUNT 2u
#define {upper}_SOURCE_OUTPUT_COUNT 1u
#define {upper}_SOURCE_INPUT_0_BYTES {counts[0] * 4}u
#define {upper}_SOURCE_INPUT_1_BYTES {counts[1] * 4}u
#define {upper}_SOURCE_OUTPUT_0_BYTES {counts[0] * 4}u
typedef enum {{ {upper}_OK=0,{upper}_INVALID,{upper}_INPUT_DOMAIN,{upper}_MEMORY_ERROR,{upper}_DEVICE_ERROR }} {name}_status;
typedef struct {{ atlas_host *host; atlas_tile_memory_io io; uint8_t *workspace; size_t max_polls;
                  atlas_host_buffer buffers[2]; }} {name}_context;
typedef struct {{ {name}_context *region; uint8_t *workspace; }} {name}_source_context;
/* Keep contexts and private workspaces alive until completion or recovery. */
{name}_status {name}_bind({name}_context *,atlas_host *,atlas_tile_memory_io,void *,size_t,size_t);
{name}_status {name}_source_bind({name}_source_context *,{name}_context *,void *,size_t);
{name}_status {name}_source_run({name}_source_context *,const float *const *,float *,atlas_host_result *);
int {name}_source_dispatch(void *,const void *const *,void *const *);
"""
    checks = []
    protected = [("ctx", "sizeof(*ctx)"), ("region", "sizeof(*region)"),
                 ("region->host", "sizeof(*region->host)"), ("ctx->workspace", f"{source_bytes}u"),
                 ("region->workspace", f"{raw_bytes}u")]
    for pointer, size in [("out", counts[0] * 4), ("inputs[0]", counts[0] * 4), ("inputs[1]", counts[1] * 4)]:
        checks.append(f"!{name}_valid({pointer},{size}u)")
        checks.extend(f"{name}_overlap({pointer},{size}u,{p},{s})" for p, s in protected)
    checks.extend((f"{name}_overlap(inputs[0],{counts[0] * 4}u,inputs[1],{counts[1] * 4}u)",
                   f"{name}_overlap(inputs[0],{counts[0] * 4}u,out,{counts[0] * 4}u)",
                   f"{name}_overlap(inputs[1],{counts[1] * 4}u,out,{counts[0] * 4}u)"))
    inputs_c = ",".join(f"{a}ULL" for a in input_addresses)
    outputs_c = ",".join(f"{a}ULL" for a in output_addresses)
    helpers = "\n".join(render_f32_array_c(spec, count, name + f"_encode_{i}")
                         for i, (spec, count) in enumerate(zip(precision.operand_specs, counts, strict=True)))
    implementation = f"""
#include <string.h>
{program_array(name + '_words', binary)}
{chr(10).join(constant_arrays)}
static const uintptr_t {name}_inputs[]={{{inputs_c}}};
static const uintptr_t {name}_outputs[]={{{outputs_c}}};
static int {name}_valid(const void *p,size_t bytes) {{ return p && (uintptr_t)p<=UINTPTR_MAX-bytes; }}
static int {name}_overlap(const void *a,size_t an,const void *b,size_t bn) {{
  return (uintptr_t)a<(uintptr_t)b+bn && (uintptr_t)b<(uintptr_t)a+an;
}}
static size_t {name}_pair(size_t r,size_t c) {{ return (c/16u)*1024u+(r*16u+c%16u)*2u; }}
{name}_status {name}_bind({name}_context *ctx,atlas_host *host,atlas_tile_memory_io io,void *scratch,size_t bytes,size_t polls) {{
  if (!ctx || !host || !host->mmio.read32 || !host->mmio.write32 || host->active || !io.write || !io.read || !polls ||
      bytes<{raw_bytes}u || !{name}_valid(scratch,{raw_bytes}u) ||
      {name}_overlap(scratch,{raw_bytes}u,ctx,sizeof(*ctx)) || {name}_overlap(scratch,{raw_bytes}u,host,sizeof(*host)) ||
      {name}_overlap(ctx,sizeof(*ctx),host,sizeof(*host))) return {upper}_INVALID;
  *ctx=({name}_context){{host,io,scratch,polls,{{
      {{{DRAM_BASE}ULL,{plan.groups * plan.readonly_bytes}u,ATLAS_HOST_DEVICE_READ}},
      {{{plan.writable_base}ULL,{plan.groups * plan.writable_bytes}u,ATLAS_HOST_DEVICE_READ_WRITE}}}}}};
  return {upper}_OK;
}}
{name}_status {name}_source_bind({name}_source_context *ctx,{name}_context *region,void *scratch,size_t bytes) {{
  if (!ctx || !region || !region->host || !region->workspace || region->host->active || bytes<{source_bytes}u ||
      !{name}_valid(scratch,{source_bytes}u) || {name}_overlap(scratch,{source_bytes}u,ctx,sizeof(*ctx)) ||
      {name}_overlap(scratch,{source_bytes}u,region,sizeof(*region)) ||
      {name}_overlap(scratch,{source_bytes}u,region->host,sizeof(*region->host)) ||
      {name}_overlap(scratch,{source_bytes}u,region->workspace,{raw_bytes}u) ||
      {name}_overlap(ctx,sizeof(*ctx),region,sizeof(*region)) ||
      {name}_overlap(ctx,sizeof(*ctx),region->host,sizeof(*region->host)) ||
      {name}_overlap(ctx,sizeof(*ctx),region->workspace,{raw_bytes}u)) return {upper}_INVALID;
  *ctx=({name}_source_context){{region,scratch}};
  return {upper}_OK;
}}
{name}_status {name}_source_run({name}_source_context *ctx,const float *const *inputs,float *out,atlas_host_result *device) {{
  if (device) *device=ATLAS_HOST_INVALID;
  if (!ctx || !ctx->region || !ctx->region->host || !ctx->workspace || !inputs ||
      !ctx->region->workspace || !ctx->region->io.write || !ctx->region->io.read || !ctx->region->max_polls) return {upper}_INVALID;
  {name}_context *region=ctx->region;
  if (region->host->active) {{ if(device) *device=ATLAS_HOST_BUSY; return {upper}_DEVICE_ERROR; }}
  if ({' || '.join(checks)}) return {upper}_INVALID;
  uint8_t *data=ctx->workspace,*gamma=data+{counts[0] * 2}u;
  if ({name}_encode_0(inputs[0],data) || {name}_encode_1(inputs[1],gamma)) return {upper}_INPUT_DOMAIN;
  uint8_t *physical=region->workspace,*result=physical+2048u;
  for (size_t group=0;group<{plan.groups}u;++group) {{
    size_t row_start=group*32u;
    for (size_t column=0;column<{columns}u;++column) {{
      __builtin_memset(physical,0,2048u);
      for (size_t r=0;r<32u && row_start+r<{m}u;++r)
        for (size_t c=0;c<32u;++c)
          __builtin_memcpy(physical+{name}_pair(r,c),data+((row_start+r)*{n}u+column*32u+c)*2u,2u);
      if (region->io.write(region->io.context,{name}_inputs[column]+group*{plan.readonly_bytes}u,physical,2048u)) return {upper}_MEMORY_ERROR;
      for (size_t r=0;r<32u;++r) for (size_t c=0;c<32u;++c)
        __builtin_memcpy(physical+{name}_pair(r,c),gamma+(column*32u+c)*2u,2u);
      if (region->io.write(region->io.context,{name}_inputs[{columns}u+column]+group*{plan.readonly_bytes}u,physical,2048u)) return {upper}_MEMORY_ERROR;
    }}
{chr(10).join(constant_stages)}
    __builtin_memset(physical,0,2048u);
    for (size_t at=0;at<{plan.writable_bytes}u;at+=2048u)
      if (region->io.write(region->io.context,{plan.writable_base}ULL+group*{plan.writable_bytes}u+at,physical,2048u)) return {upper}_MEMORY_ERROR;
  }}
  atlas_host_result status=atlas_host_ensure_loaded(region->host,{name}_words,{len(binary) // 4}u);
  if (status==ATLAS_HOST_OK) status=atlas_host_start(region->host,region->buffers,2u);
  if (status==ATLAS_HOST_OK) status=atlas_host_wait(region->host,region->max_polls);
  if (device) *device=status;
  if (status!=ATLAS_HOST_OK) return {upper}_DEVICE_ERROR;
  for (size_t group=0;group<{plan.groups}u;++group) {{
    for (size_t column=0;column<{columns}u;++column) {{
      uintptr_t address={name}_outputs[column]-{plan.readonly_bytes}u+{plan.groups * plan.readonly_bytes}u+group*{plan.writable_bytes}u;
      if (region->io.read(region->io.context,address,physical,2048u)) return {upper}_MEMORY_ERROR;
      for (size_t r=0;r<32u && group*32u+r<{m}u;++r) for (size_t c=0;c<32u;++c) {{
        size_t at={name}_pair(r,c);
        uint32_t code=(uint32_t)physical[at]|((uint32_t)physical[at+1u]<<8);
        if ((code&0x7f80u)==0x7f80u) return {upper}_INPUT_DOMAIN;
        uint32_t bits=code<<16;
        __builtin_memcpy(result+((group*32u+r)*{n}u+column*32u+c)*4u,&bits,4u);
      }}
    }}
  }}
  __builtin_memcpy(out,result,{counts[0] * 4}u);
  return {upper}_OK;
}}
int {name}_source_dispatch(void *ctx,const void *const *inputs,void *const *outputs) {{
  if (!inputs || !outputs) return {upper}_INVALID;
  const float *typed[]={{inputs[0],inputs[1]}};
  return {name}_source_run(ctx,typed,outputs[0],0);
}}
"""
    public = runtime_headers(target_source / "runtime")
    members, files, runtime = runtime_members(public, (target_source / "runtime/atlas_host.c").read_text(), runtime_linkage)
    rendered = c_source_pair(name, public_headers=public, private_headers={},
                             sources={**members, "rmsnorm_io.c": helpers + implementation}, declarations=declarations)
    files.update({key: value.encode() for key, value in rendered.items()})
    files["program.bin"] = binary
    return {"workspace_bytes": raw_bytes, "source_workspace_bytes": source_bytes,
            "context_type": name + "_context", "source_context_type": name + "_source_context",
            "binding_function": name + "_bind", "source_binding_function": name + "_source_bind",
            "source_dispatch_function": name + "_source_dispatch", "precision_plan": precision.record(),
            "input_node_ids": list(precision.input_ids), "output_node_ids": list(precision.source.outputs),
            "row_loop_plan": plan.record(), "launches_per_invocation": 1,
            "output_conversion": "bf16-to-f32-exact", **runtime}, files
