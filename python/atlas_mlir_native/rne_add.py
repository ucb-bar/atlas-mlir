"""Select BF16 RNE addition through independently packed two-value row sums.

The host converts and rearranges operands. Existing semantic reduction
selection and Atlas lowering perform every addition on the device.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from math import prod

from merlin.compile.c_conversions import render_f32_array_c
from merlin.compile.c_sources import c_source_pair
from merlin.semantic_compiler.model import KernelRequest, pointwise_identity_maps

from .binding import TARGET_IDENTITY, AtlasNativeBinding
from .pointwise import input_spec as pointwise_input_spec
from .reduction import prepare_source as prepare_reduction_source, request_for
from .reduction_c import prepare_c_package as prepare_reduction_c_package

RNE_ADD_POLICY = "f32-bf16-rne-pair-row-sum-normal-or-zero-v1"
BATCH_ROWS = 32


def input_spec():
    return replace(pointwise_input_spec(), policy=RNE_ADD_POLICY)


@dataclass(frozen=True)
class RNEAddPrecision:
    source: KernelRequest
    reduction_source: KernelRequest
    request: KernelRequest
    input_ids: tuple[str, ...]
    count: int

    def record(self):
        return {
            "schema": "atlas.rne_add_precision.v1",
            "policy": RNE_ADD_POLICY,
            "source_request_digest": self.source.digest(),
            "reduction_source_request_digest": self.reduction_source.digest(),
            "request_digest": self.request.digest(),
            "input_conversion": input_spec().record(),
            "arithmetic": "bf16-rne-add-of-bf16-rne-operands",
            "output_conversion": "bf16-to-f32-exact",
            "batch_rows": BATCH_ROWS,
            "element_count": self.count,
            "launch_count": (self.count + BATCH_ROWS - 1) // BATCH_ROWS,
            "padding": "first-two-columns-then-positive-zero",
            "refused_pairs": "both-encoded-operands-negative-zero",
            "qualification": "explicit-diagnostic-precision-change; device-addition",
        }


def prepare_source(source, *, policy):
    if policy != RNE_ADD_POLICY:
        raise ValueError("RNE addition requires its explicitly selected policy")
    pure = [node for node in source.nodes if node.effect == "pure"]
    if (source.target_identity != TARGET_IDENTITY or source.lowering_policy != "diagnostic"
            or source.physical_extents or source.boundary_layouts or source.output_layouts
            or source.constants or len(pure) != 1):
        raise ValueError("RNE addition needs one plain diagnostic source computation")
    result = pure[0]
    keys = tuple(dict.fromkeys(result.inputs))
    if (result.op != "f32_add" or len(result.inputs) != 2 or not result.type.shape
            or source.outputs != (result.id,)
            or dict(result.attrs) != {"numeric_semantics": "ieee754-f32-add-rne-v1"}
            or result.index_maps != pointwise_identity_maps(len(result.type.shape), 2)
            or result.type.dtype != "f32" or result.type.numerical_policy != "raw-bit-pattern"
            or {node.id for node in source.nodes} != {*keys, result.id}
            or any(source.node(key).op != "input" or source.node(key).effect != "input"
                   or source.node(key).inputs or source.node(key).attrs or source.node(key).index_maps
                   or source.node(key).type != result.type for key in keys)):
        raise ValueError("RNE addition source types, indexing, or accounting differ")
    count = prod(result.type.shape)
    if count <= 0 or count * 12 + 1024 > 2**32 - 1:
        raise ValueError("RNE addition exceeds its bounded host buffer ABI")
    raw = request_for((BATCH_ROWS, 2), (1,))
    data, seed, output = raw.nodes
    floating = lambda node: replace(node.type, dtype="f32", numerical_policy="raw-bit-pattern")
    reduction_source = replace(raw, nodes=(
        replace(data, type=floating(data), attrs=()),
        replace(seed, type=floating(seed), attrs=()),
        replace(output, type=floating(output), op="f32_reduce_add", attrs=(
            ("axes", "1"), ("reduction_order", "axis-ascending"),
            ("numeric_semantics", "ieee754-f32-add-rne-v1"))),
    ), source_identity=source.digest())
    reduction = prepare_reduction_source(reduction_source)
    return RNEAddPrecision(source, reduction_source, reduction.request, keys, count)


def compile_source(snapshot, source, *, policy, target_source, destination, limits):
    precision = prepare_source(source, policy=policy)
    abi = AtlasNativeBinding.plan_boundary_abi(precision.request)
    return AtlasNativeBinding.compile(snapshot, precision.request, fixed_inputs=abi["fixed_inputs"],
        fixed_outputs=tuple(abi["fixed_outputs"]), target_source=target_source,
        destination=destination, limits=limits)


def prepare_c_package(source, compiled, name, *, policy, target_source, runtime_linkage="separate"):
    precision = prepare_source(source, policy=policy)
    child = name + "_rows"
    inner, files = prepare_reduction_c_package(precision.reduction_source, compiled, child,
        target_source=target_source, runtime_linkage=runtime_linkage)
    count, upper, child_upper = precision.count, name.upper(), child.upper()
    child_bytes = inner["source_workspace_bytes"]
    data_at, seed_at, block_at = child_bytes, child_bytes + 256, child_bytes + 384
    codes_at = child_bytes + 512
    result_at = (codes_at + 2 * count * len(precision.input_ids) + 3) & ~3
    converted = result_at + 4 * count
    declarations = f"""
#define {upper}_WORKSPACE_BYTES {inner['workspace_bytes']}u
#define {upper}_SOURCE_WORKSPACE_BYTES {converted}u
#define {upper}_SOURCE_INPUT_COUNT {len(precision.input_ids)}u
#define {upper}_SOURCE_OUTPUT_COUNT 1u
typedef {child}_context {name}_context;
typedef {child}_status {name}_status;
enum {{ {upper}_OK=0,{upper}_INVALID=1,{upper}_INPUT_DOMAIN=2,{upper}_MEMORY_ERROR=3,{upper}_DEVICE_ERROR=4 }};
typedef struct {{ {name}_context *region; {child}_source_context rows; uint8_t *workspace; }} {name}_source_context;
{name}_status {name}_bind({name}_context *,atlas_host *,atlas_tile_memory_io,void *,size_t,size_t);
{name}_status {name}_source_bind({name}_source_context *,{name}_context *,void *,size_t);
{name}_status {name}_source_run({name}_source_context *,const float *const *,float *,atlas_host_result *);
int {name}_source_dispatch(void *,const void *const *,void *const *);
"""
    encoders = "\n".join(render_f32_array_c(input_spec(), count, f"{name}_encode_{i}")
                          for i in range(len(precision.input_ids)))
    checks = []
    for i in range(len(precision.input_ids)):
        checks.extend((f"!{name}_valid(inputs[{i}],{4*count}u)",
            f"{name}_overlap(inputs[{i}],{4*count}u,out,{4*count}u)",
            f"{name}_overlap(inputs[{i}],{4*count}u,ctx,sizeof(*ctx))",
            f"{name}_overlap(inputs[{i}],{4*count}u,region,sizeof(*region))",
            f"{name}_overlap(inputs[{i}],{4*count}u,region->host,sizeof(*region->host))",
            f"{name}_overlap(inputs[{i}],{4*count}u,ctx->workspace,{converted}u)",
            f"{name}_overlap(inputs[{i}],{4*count}u,region->workspace,{inner['workspace_bytes']}u)"))
    encoding = " || ".join(f"{name}_encode_{i}(inputs[{i}],ctx->workspace+{codes_at+2*count*i}u)"
                           for i in range(len(precision.input_ids)))
    lhs, rhs = (precision.input_ids.index(key) for key in source.node(source.outputs[0]).inputs)
    typed = ",".join(f"(const float *)inputs[{i}]" for i in range(len(precision.input_ids)))
    implementation = f"""
static int {name}_valid(const void *p,size_t n) {{ return p && n && (uintptr_t)p<=UINTPTR_MAX-n; }}
static int {name}_overlap(const void *a,size_t n,const void *b,size_t m) {{
  return (uintptr_t)a<(uintptr_t)b+m && (uintptr_t)b<(uintptr_t)a+n;
}}
{name}_status {name}_bind({name}_context *ctx,atlas_host *host,atlas_tile_memory_io io,void *scratch,size_t bytes,size_t polls) {{
  return {child}_bind(ctx,host,io,scratch,bytes,polls);
}}
{name}_status {name}_source_bind({name}_source_context *ctx,{name}_context *region,void *scratch,size_t bytes) {{
  if (!ctx || !region || !region->host || !region->workspace || bytes<{converted}u ||
      !{name}_valid(scratch,{converted}u) || ((uintptr_t)scratch&3u) ||
      {name}_overlap(scratch,{converted}u,ctx,sizeof(*ctx)) ||
      {name}_overlap(scratch,{converted}u,region,sizeof(*region)) ||
      {name}_overlap(scratch,{converted}u,region->host,sizeof(*region->host)) ||
      {name}_overlap(scratch,{converted}u,region->workspace,{inner['workspace_bytes']}u) ||
      {name}_overlap(ctx,sizeof(*ctx),region,sizeof(*region)) ||
      {name}_overlap(ctx,sizeof(*ctx),region->host,sizeof(*region->host)) ||
      {name}_overlap(ctx,sizeof(*ctx),region->workspace,{inner['workspace_bytes']}u)) return ({name}_status){upper}_INVALID;
  {child}_status status={child}_source_bind(&ctx->rows,region,scratch,{child_bytes}u);
  if (status!={child_upper}_OK) return status;
  ctx->region=region;ctx->workspace=scratch;return ({name}_status){upper}_OK;
}}
{name}_status {name}_source_run({name}_source_context *ctx,const float *const *inputs,float *out,atlas_host_result *device) {{
  if (device) *device=ATLAS_HOST_INVALID;
  if (!ctx || !ctx->region || !ctx->region->host || !ctx->workspace || !inputs ||
      ctx->rows.region!=ctx->region || ctx->rows.workspace!=ctx->workspace ||
      !ctx->region->workspace) return ({name}_status){upper}_INVALID;
  {name}_context *region=ctx->region;
  if (region->host->active) {{ if(device) *device=ATLAS_HOST_BUSY;return ({name}_status){upper}_DEVICE_ERROR; }}
  if (!{name}_valid(out,{4*count}u) ||
      {name}_overlap(out,{4*count}u,ctx,sizeof(*ctx)) ||
      {name}_overlap(out,{4*count}u,region,sizeof(*region)) ||
      {name}_overlap(out,{4*count}u,region->host,sizeof(*region->host)) ||
      {name}_overlap(out,{4*count}u,ctx->workspace,{converted}u) ||
      {name}_overlap(out,{4*count}u,region->workspace,{inner['workspace_bytes']}u) ||
      {' || '.join(checks)}) return ({name}_status){upper}_INVALID;
  if ({encoding}) return ({name}_status){upper}_INPUT_DOMAIN;
  const uint8_t *a=ctx->workspace+{codes_at+2*count*lhs}u,*b=ctx->workspace+{codes_at+2*count*rhs}u;
  for (size_t i=0;i<{count}u;++i)
    if (a[2*i]==0 && a[2*i+1]==128u && b[2*i]==0 && b[2*i+1]==128u) return ({name}_status){upper}_INPUT_DOMAIN;
  float *data=(float *)(ctx->workspace+{data_at}u),*seed=(float *)(ctx->workspace+{seed_at}u);
  float *block=(float *)(ctx->workspace+{block_at}u);
  __builtin_memset(seed,0,128u);
  for (size_t first=0;first<{count}u;first+=32u) {{
    __builtin_memset(data,0,256u);
    size_t lanes={count}u-first<32u?{count}u-first:32u;
    for (size_t row=0;row<lanes;++row) {{
      size_t i=first+row;
      uint32_t left=((uint32_t)a[2*i]|((uint32_t)a[2*i+1]<<8))<<16;
      uint32_t right=((uint32_t)b[2*i]|((uint32_t)b[2*i+1]<<8))<<16;
      __builtin_memcpy(data+2*row,&left,4u);__builtin_memcpy(data+2*row+1,&right,4u);
    }}
    const float *operands[]={{data,seed}};
    {child}_status status={child}_source_run(&ctx->rows,operands,block,device);
    if (status!={child_upper}_OK) return status;
    __builtin_memcpy(ctx->workspace+{result_at}u+4*first,block,4*lanes);
  }}
  __builtin_memcpy(out,ctx->workspace+{result_at}u,{4*count}u);return ({name}_status){upper}_OK;
}}
int {name}_source_dispatch(void *ctx,const void *const *inputs,void *const *outputs) {{
  if (!inputs || !outputs) return {upper}_INVALID;
  const float *typed[]={{{typed}}};return {name}_source_run(ctx,typed,outputs[0],0);
}}
"""
    rendered = c_source_pair(name, public_headers={child + ".h": files.pop(child + ".h").decode()},
        private_headers={}, sources={child + ".c": files.pop(child + ".c").decode(),
                                     "rne_add_source.c": encoders + implementation}, declarations=declarations)
    files.update({key: value.encode() for key, value in rendered.items()})
    return {**inner, "schema": "atlas.rne_add_c_package.v1", "source_workspace_bytes": converted,
        "precision_plan": precision.record(), "input_node_ids": list(precision.input_ids),
        "output_node_ids": list(source.outputs), "launches_per_call": (count+31)//32,
        "context_type": name + "_context", "source_context_type": name + "_source_context",
        "binding_function": name + "_bind", "source_binding_function": name + "_source_bind",
        "dispatch_function": name + "_source_dispatch"}, files
