"""Render the shared Atlas BF16 tile I/O and synchronous C launch interface."""

from __future__ import annotations

from math import prod

from merlin.compile.c_sources import c_source_pair

from .c_runtime import program_array, runtime_members


def render_tile_io_package(
    request,
    binary,
    execution,
    name,
    *,
    target_source,
    runtime_linkage,
    packed_sizes,
    tile_bytes,
    helpers,
    source_dtype="f32",
    source_input_indices=None,
):
    addresses = execution["input_addresses"]
    output = execution["output_addresses"][0]
    htext = (target_source / "runtime/atlas_host.h").read_text()
    ctext = (target_source / "runtime/atlas_host.c").read_text()
    members, files, runtime = runtime_members(htext, ctext, runtime_linkage)
    sizes = tuple(packed_sizes)
    t = tile_bytes
    arity = len(sizes) - 1
    indices = tuple(range(arity)) if source_input_indices is None else tuple(source_input_indices)
    if (source_dtype not in {"f32", "bf16"} or len(indices) != arity
            or any(type(index) is not int or index < 0 for index in indices)
            or set(indices) != set(range(len(set(indices))))):
        raise ValueError("tile I/O source dtype or input mapping differs")
    for i, index in enumerate(indices):
        if any(request.nodes[j].type != request.nodes[i].type for j in range(i) if indices[j] == index):
            raise ValueError("shared tile I/O source has inconsistent shapes or types")
    source_width = 4 if source_dtype == "f32" else 2
    source_type = "float" if source_dtype == "f32" else "uint8_t"
    if t != 2048 or len(sizes) != len(request.nodes) or any(size <= 0 or size % t for size in sizes):
        raise ValueError("tile I/O C pair layout differs from the selected target")
    raw_workspace = sum(sizes) + t
    elements = [prod(node.type.shape) for node in request.nodes]
    source_workspace = 2 * sum(elements)
    upper = name.upper()
    input_elements = [elements[indices.index(index)] for index in range(len(set(indices)))]
    source_sizes = "\n".join(
        f"#define {upper}_SOURCE_INPUT_{index}_BYTES {count * source_width}u"
        for index, count in enumerate(input_elements)
    )
    buffers = ",".join(
        f"{{{addr}u,{size}u,ATLAS_HOST_DEVICE_{mode}}}"
        for addr, size, mode in zip([*addresses, output], sizes, (*("READ",) * arity, "WRITE"), strict=True)
    )
    declarations = f"""
#define {upper}_WORKSPACE_BYTES {raw_workspace}u
#define {upper}_SOURCE_WORKSPACE_BYTES {source_workspace}u
#define {upper}_SOURCE_INPUT_COUNT {len(input_elements)}u
#define {upper}_SOURCE_OUTPUT_COUNT 1u
{source_sizes}
#define {upper}_SOURCE_OUTPUT_0_BYTES {elements[-1] * source_width}u
typedef enum {{ {upper}_OK=0,{upper}_INVALID,{upper}_INPUT_DOMAIN,{upper}_MEMORY_ERROR,{upper}_DEVICE_ERROR }} {name}_status;
typedef struct {{ atlas_host *host; atlas_tile_memory_io io; uint8_t *workspace; size_t max_polls;
                  atlas_host_buffer buffers[{arity + 1}]; }} {name}_context;
typedef struct {{ {name}_context *region; uint8_t *workspace; }} {name}_source_context;
/* Preserve contexts and both workspaces through completion or recovery. */
{name}_status {name}_bind({name}_context *,atlas_host *,atlas_tile_memory_io,void *,size_t,size_t);
{name}_status {name}_source_bind({name}_source_context *,{name}_context *,void *,size_t);
/* Dense source {source_dtype} operands/results in their recorded logical shapes and order. */
/* BF16 source values use little-endian code bytes, with no float reinterpretation. */
{name}_status {name}_source_run({name}_source_context *,const {source_type} *const *,{source_type} *,atlas_host_result *);
int {name}_source_dispatch(void *,const void *const *,void *const *);
"""
    checks = []
    for i, size in enumerate(elements[:-1]):
        index = indices[i]
        checks += [
            f"!{name}_valid(inputs[{index}],{size * source_width}u)",
            f"{name}_overlap(inputs[{index}],{size * source_width}u,out,{elements[-1] * source_width}u)",
            f"{name}_overlap(inputs[{index}],{size * source_width}u,ctx->workspace,{source_workspace}u)",
            f"{name}_overlap(inputs[{index}],{size * source_width}u,region->workspace,{raw_workspace}u)",
            f"{name}_overlap(inputs[{index}],{size * source_width}u,ctx,sizeof(*ctx))",
            f"{name}_overlap(inputs[{index}],{size * source_width}u,region,sizeof(*region))",
            f"{name}_overlap(inputs[{index}],{size * source_width}u,region->host,sizeof(*region->host))",
        ]
    converted = "; ".join(f"uint8_t *operand_{i}=ctx->workspace+{sum(elements[:i]) * 2}u" for i in range(arity))
    encoding = " || ".join(f"{name}_encode_{i}(inputs[{indices[i]}],operand_{i})" for i in range(arity))
    packing = " ".join(f"{name}_pack_{i}(region->workspace+{sum(sizes[:i])}u,operand_{i});" for i in range(arity))
    writing = "\n".join(
        f"  for (size_t at=0;at<{sizes[i]}u;at+={t}u) {{\n"
        f"    {name}_pair(physical,region->workspace+{sum(sizes[:i])}u+at,1);\n"
        f"    if (region->io.write(region->io.context,{address}u+at,physical,{t}u)) return {upper}_MEMORY_ERROR;\n  }}"
        for i, address in enumerate(addresses)
    )
    typed_inputs = ",".join(f"inputs[{i}]" for i in range(len(set(indices))))
    decode = (f"for (size_t i=0;i<{elements[-1]}u;++i) {{ "
              "uint32_t bits=((uint32_t)result[2*i]|((uint32_t)result[2*i+1]<<8))<<16; "
              "__builtin_memcpy(out+i,&bits,4u); }" if source_dtype == "f32"
              else f"__builtin_memcpy(out,result,{elements[-1] * 2}u);")
    implementation = f"""
{program_array(name + "_words", binary)}
static int {name}_valid(const void *p,size_t n) {{ return p && n && (uintptr_t)p<=UINTPTR_MAX-n; }}
static int {name}_overlap(const void *a,size_t n,const void *b,size_t m) {{
  return (uintptr_t)a<(uintptr_t)b+m && (uintptr_t)b<(uintptr_t)a+n;
}}
static void {name}_pair(uint8_t *out,const uint8_t *in,int pack) {{
  for (size_t r=0;r<32u;++r) for (size_t c=0;c<32u;++c) {{
    size_t logical=(r*32u+c)*2u,physical=(c/16u)*1024u+(r*16u+c%16u)*2u;
    __builtin_memcpy(out+(pack?physical:logical),in+(pack?logical:physical),2u);
  }}
}}
{name}_status {name}_bind({name}_context *ctx,atlas_host *host,atlas_tile_memory_io io,void *scratch,size_t bytes,size_t polls) {{
  if (!ctx || !host || !host->mmio.read32 || !host->mmio.write32 || host->active || !io.write || !io.read || !polls || bytes<{raw_workspace}u ||
      !{name}_valid(scratch,{raw_workspace}u) || {name}_overlap(scratch,{raw_workspace}u,ctx,sizeof(*ctx)) ||
      {name}_overlap(scratch,{raw_workspace}u,host,sizeof(*host)) || {name}_overlap(ctx,sizeof(*ctx),host,sizeof(*host))) return {upper}_INVALID;
  *ctx=({name}_context){{host,io,scratch,polls,{{{buffers}}}}}; return {upper}_OK;
}}
{name}_status {name}_source_bind({name}_source_context *ctx,{name}_context *region,void *scratch,size_t bytes) {{
  if (!ctx || !region || !region->host || region->host->active || !region->workspace || bytes<{source_workspace}u ||
      !{name}_valid(scratch,{source_workspace}u) || {name}_overlap(scratch,{source_workspace}u,ctx,sizeof(*ctx)) ||
      {name}_overlap(scratch,{source_workspace}u,region,sizeof(*region)) ||
      {name}_overlap(scratch,{source_workspace}u,region->host,sizeof(*region->host)) ||
      {name}_overlap(scratch,{source_workspace}u,region->workspace,{raw_workspace}u) ||
      {name}_overlap(ctx,sizeof(*ctx),region,sizeof(*region)) || {name}_overlap(ctx,sizeof(*ctx),region->host,sizeof(*region->host)) ||
      {name}_overlap(ctx,sizeof(*ctx),region->workspace,{raw_workspace}u)) return {upper}_INVALID;
  *ctx=({name}_source_context){{region,scratch}}; return {upper}_OK;
}}
{name}_status {name}_source_run({name}_source_context *ctx,const {source_type} *const *inputs,{source_type} *out,atlas_host_result *device) {{
  if (device) *device=ATLAS_HOST_INVALID;
  if (!ctx || !ctx->region || !ctx->region->host || !ctx->workspace || !inputs ||
      !ctx->region->workspace || !ctx->region->io.write || !ctx->region->io.read || !ctx->region->max_polls) return {upper}_INVALID;
  {name}_context *region=ctx->region;
  if (region->host->active) {{ if(device) *device=ATLAS_HOST_BUSY; return {upper}_DEVICE_ERROR; }}
  if (!{name}_valid(out,{elements[-1] * source_width}u) || {name}_overlap(out,{elements[-1] * source_width}u,ctx,sizeof(*ctx)) ||
      {name}_overlap(out,{elements[-1] * source_width}u,region,sizeof(*region)) ||
      {name}_overlap(out,{elements[-1] * source_width}u,region->host,sizeof(*region->host)) ||
      {name}_overlap(out,{elements[-1] * source_width}u,ctx->workspace,{source_workspace}u) ||
      {name}_overlap(out,{elements[-1] * source_width}u,region->workspace,{raw_workspace}u) ||
      {" || ".join(checks)}) return {upper}_INVALID;
  {converted}; uint8_t *result=ctx->workspace+{sum(elements[:-1]) * 2}u;
  if ({encoding}) return {upper}_INPUT_DOMAIN;
  {packing}
  uint8_t *physical=region->workspace+{sum(sizes)}u;
  atlas_host_result status=atlas_host_ensure_loaded(region->host,{name}_words,{len(binary) // 4}u);
  if (device) *device=status; if (status!=ATLAS_HOST_OK) return {upper}_DEVICE_ERROR;
{writing}
  status=atlas_host_start(region->host,region->buffers,{arity + 1}u);
  if (status==ATLAS_HOST_OK) status=atlas_host_wait(region->host,region->max_polls);
  if (device) *device=status; if (status!=ATLAS_HOST_OK) return {upper}_DEVICE_ERROR;
  for (size_t at=0;at<{sizes[-1]}u;at+={t}u) {{
    if (region->io.read(region->io.context,{output}u+at,physical,{t}u)) return {upper}_MEMORY_ERROR;
    {name}_pair(region->workspace+{sum(sizes[:-1])}u+at,physical,0);
  }}
  {name}_assemble(result,region->workspace+{sum(sizes[:-1])}u);
  {decode}
  return {upper}_OK;
}}
int {name}_source_dispatch(void *context,const void *const *inputs,void *const *outputs) {{
  if (!inputs || !outputs) return {upper}_INVALID;
  const {source_type} *typed[]={{{typed_inputs}}};
  return {name}_source_run(context,typed,outputs[0],0);
}}
"""
    rendered = c_source_pair(
        name,
        public_headers={"atlas_host.h": htext},
        private_headers={},
        sources={**members, "tile_io.c": helpers + implementation},
        declarations=declarations,
    )
    files.update({key: value.encode() for key, value in rendered.items()})
    files["program.bin"] = binary
    return {
        "workspace_bytes": raw_workspace,
        "source_workspace_bytes": source_workspace,
        **({"source_dtype": source_dtype, "source_input_indices": list(indices)}
           if source_dtype != "f32" or indices != tuple(range(arity)) else {}),
        **runtime,
    }, files
