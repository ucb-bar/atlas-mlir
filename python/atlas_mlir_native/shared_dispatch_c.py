"""Shared host dispatch routines, emitted once per combined application source.

Each region keeps its public bind, source-bind, and dispatch symbols as thin
wrappers that pass a per-region constant plan or shape to these routines, so
host code grows by a small table per region instead of a full dispatch body.
Identical text shares one guarded copy; two different copies of one routine in
a translation unit fail to compile instead of silently using either.
"""
from __future__ import annotations

from hashlib import sha256


def guarded(name: str, text: str) -> str:
    """Emit text once per translation unit, keyed by its exact bytes."""
    guard = f"ATLAS_SHARED_{name.upper()}_{sha256(text.encode()).hexdigest()[:16].upper()}"
    return f"#ifndef {guard}\n#define {guard}\n{text}#endif\n"


# Header types: the device region context shared by checked image dispatch routines.
REGION_TYPES = guarded("region_types", "typedef struct { atlas_host *host; atlas_tile_memory_io io; uint8_t *workspace; "
                       "size_t max_polls; } atlas_region_context;\n")

CHECKS = guarded("checks", f"""#include <stddef.h>
#include <stdint.h>
#include <string.h>
typedef struct {{ const void *address; size_t bytes; }} atlas_span;
static __attribute__((unused)) int atlas_span_valid(const void *p,size_t n) {{ return p && n && n<=UINTPTR_MAX-(uintptr_t)p; }}
static __attribute__((unused)) int atlas_overlap(const void *a,size_t an,const void *b,size_t bn) {{ return (uintptr_t)a<(uintptr_t)b+bn && (uintptr_t)b<(uintptr_t)a+an; }}
/* Shared routines are unused in some standalone regions. Nonzero unless every span is a valid nonempty range, spans before fixed are
 * pairwise disjoint, every later span avoids them, and each span from
 * exclusive on also avoids every earlier span. */
static __attribute__((unused)) int atlas_spans_conflict(const atlas_span *span,size_t count,size_t fixed,size_t exclusive) {{
  for (size_t i=0;i<count;++i) {{
    size_t n=span[i].bytes,limit=i<fixed || i>=exclusive?i:fixed;
    if (!atlas_span_valid(span[i].address,n)) return 1;
    for (size_t j=0;j<limit;++j) if (atlas_overlap(span[i].address,n,span[j].address,span[j].bytes)) return 1;
  }}
  return 0;
}}
/* Region binding: context, host, and workspace are valid and disjoint, the
 * host is idle with MMIO hooks, and memory I/O and a poll bound are present. */
static __attribute__((unused)) int atlas_bind_conflict(const void *context,size_t context_bytes,const atlas_host *host,const void *workspace,
                               size_t workspace_bytes,size_t bytes,const atlas_tile_memory_io *io,size_t polls) {{
  const atlas_span span[3]={{{{context,context_bytes}},{{host,sizeof(*host)}},{{workspace,workspace_bytes}}}};
  return atlas_spans_conflict(span,3u,3u,3u) || bytes<workspace_bytes || host->active || !host->mmio.read32 ||
         !host->mmio.write32 || !io->read || !io->write || !polls;
}}
/* Source binding: both contexts are valid and disjoint, and the source context
 * also avoids the region's host and workspace, which must be valid and idle.
 * The host and workspace fields are read at their offsets only after the region is validated. */
static __attribute__((unused)) int atlas_source_bind_conflict(const void *context,size_t context_bytes,const void *region,size_t region_bytes,
                                      size_t host_field,size_t workspace_field,size_t workspace_bytes) {{
  const atlas_span first[2]={{{{context,context_bytes}},{{region,region_bytes}}}};
  if (atlas_spans_conflict(first,2u,2u,2u)) return 1;
  atlas_host *host; uint8_t *workspace;
  __builtin_memcpy(&host,(const uint8_t *)region+host_field,sizeof(host));
  __builtin_memcpy(&workspace,(const uint8_t *)region+workspace_field,sizeof(workspace));
  const atlas_span span[3]={{{{context,context_bytes}},{{host,sizeof(*host)}},{{workspace,workspace_bytes}}}};
  return atlas_spans_conflict(span,3u,1u,3u) || host->active;
}}
/* Region dispatch ABI. The source context, region context, host, workspace,
 * and the input and output pointer tables are pairwise disjoint. Table entries
 * are read only after that check; each input then avoids every one of them,
 * and each output also avoids every earlier input and output. */
typedef struct {{ uint32_t context_bytes,region_bytes,workspace_bytes,inputs,outputs; const uint32_t *value_bytes; }} atlas_dispatch_shape;
static __attribute__((unused)) atlas_span atlas_value_span(const atlas_dispatch_shape *shape,const void *const *inputs,void *const *outputs,size_t i) {{
  return (atlas_span){{i<shape->inputs?inputs[i]:outputs[i-shape->inputs],shape->value_bytes[i]}};
}}
static __attribute__((unused)) int atlas_dispatch_conflict(const atlas_dispatch_shape *shape,const void *context,const void *region,
                                   const atlas_host *host,const void *workspace,const void *const *inputs,void *const *outputs) {{
  atlas_span fixed[6];
  size_t count=0;
  fixed[count++]=(atlas_span){{context,shape->context_bytes}};
  fixed[count++]=(atlas_span){{region,shape->region_bytes}};
  fixed[count++]=(atlas_span){{host,sizeof(*host)}};
  fixed[count++]=(atlas_span){{workspace,shape->workspace_bytes}};
  if (shape->inputs) fixed[count++]=(atlas_span){{inputs,shape->inputs*sizeof(*inputs)}};
  fixed[count++]=(atlas_span){{outputs,shape->outputs*sizeof(*outputs)}};
  if (atlas_spans_conflict(fixed,count,count,count)) return 1;
  /* Values are read from the tables on demand, so any port count is checked without a span buffer. */
  for (size_t i=0;i<shape->inputs+shape->outputs;++i) {{
    atlas_span value=atlas_value_span(shape,inputs,outputs,i);
    if (!atlas_span_valid(value.address,value.bytes)) return 1;
    for (size_t j=0;j<count;++j) if (atlas_overlap(value.address,value.bytes,fixed[j].address,fixed[j].bytes)) return 1;
    for (size_t j=0;i>=shape->inputs && j<i;++j) {{
      atlas_span other=atlas_value_span(shape,inputs,outputs,j);
      if (atlas_overlap(value.address,value.bytes,other.address,other.bytes)) return 1;
    }}
  }}
  return 0;
}}
""")


def shape(name: str, *, context: str, region: str, workspace_bytes: int, inputs: list[int], outputs: list[int]) -> str:
    """One region's constant dispatch shape for atlas_dispatch_conflict."""
    if not outputs:
        raise ValueError("region dispatch has no outputs")
    values = ",".join(f"{size}u" for size in [*inputs, *outputs])
    return (f"static const uint32_t {name}_value_bytes[]={{{values}}};\n"
            f"static const atlas_dispatch_shape {name}_shape={{sizeof({context}),sizeof({region}),"
            f"{workspace_bytes}u,{len(inputs)}u,{len(outputs)}u,{name}_value_bytes}};\n")


# Header types for single-image copy regions: the region context and its source context.
COPY_TYPES = REGION_TYPES + guarded("copy_types", "typedef struct { atlas_region_context *region; } atlas_copy_source_context;\n")

# One device image over constant and runtime operands: write each operand (zero padded to 32 bytes),
# run the image once, and read one output. Return codes: 1 invalid, 2 memory error, 3 device error.
COPY_DISPATCH = guarded("copy_dispatch", """typedef struct { const uint8_t *constant; uintptr_t address; uint32_t port,bytes; } atlas_copy_write;
typedef struct {
  const uint32_t *words; const atlas_copy_write *writes; const atlas_host_buffer *buffers;
  uintptr_t output_address; uint32_t word_count,write_count,buffer_count,inputs,output_bytes;
} atlas_copy_plan;
static const uint8_t atlas_copy_zero[32]={0};
static __attribute__((unused)) int atlas_copy_valid(const void *p,size_t n) { return p && (uintptr_t)p<=UINTPTR_MAX-n; }
static __attribute__((unused)) int atlas_copy_bind(atlas_region_context *ctx,atlas_host *host,const atlas_tile_memory_io *io,size_t polls) {
  if (!ctx || !host || !host->mmio.read32 || !host->mmio.write32 || host->active || !io->write || !io->read || !polls ||
      atlas_overlap(ctx,sizeof(*ctx),host,sizeof(*host))) return 1;
  *ctx=(atlas_region_context){host,*io,0,polls}; return 0;
}
static __attribute__((unused)) int atlas_copy_source_bind(atlas_copy_source_context *ctx,atlas_region_context *region) {
  if (!ctx || !region || !region->host || region->host->active ||
      atlas_overlap(ctx,sizeof(*ctx),region,sizeof(*region)) ||
      atlas_overlap(ctx,sizeof(*ctx),region->host,sizeof(*region->host))) return 1;
  ctx->region=region; return 0;
}
static __attribute__((unused)) int atlas_copy_dispatch(const atlas_copy_plan *p,void *opaque,const void *const *inputs,void *const *outputs) {
  atlas_copy_source_context *ctx=opaque;
  if (!ctx || !ctx->region || (p->inputs && !inputs) || !outputs || !atlas_copy_valid(outputs[0],p->output_bytes)) return 1;
  atlas_region_context *region=ctx->region;
  if (!region->host || region->host->active || !region->io.write || !region->io.read || !region->max_polls) return 1;
  /* Each runtime operand is valid and avoids the output; the output and every runtime operand avoid both contexts and the host. */
  for (size_t i=0;i<=p->write_count;++i) {
    const void *pointer=outputs[0]; size_t size=p->output_bytes;
    if (i<p->write_count) {
      if (p->writes[i].constant) continue;
      pointer=inputs[p->writes[i].port]; size=p->writes[i].bytes;
      if (!atlas_copy_valid(pointer,size) || atlas_overlap(pointer,size,outputs[0],p->output_bytes)) return 1;
    }
    if (atlas_overlap(pointer,size,ctx,sizeof(*ctx)) || atlas_overlap(pointer,size,region,sizeof(*region)) ||
        atlas_overlap(pointer,size,region->host,sizeof(*region->host))) return 1;
  }
  for (size_t i=0;i<p->write_count;++i) {
    const atlas_copy_write *w=&p->writes[i];
    if (region->io.write(region->io.context,w->address,w->constant?w->constant:(const uint8_t *)inputs[w->port],w->bytes)) return 2;
    if (w->bytes%32u && region->io.write(region->io.context,w->address+w->bytes,atlas_copy_zero,32u-w->bytes%32u)) return 2;
  }
  atlas_host_result status=atlas_host_ensure_loaded(region->host,p->words,p->word_count);
  if (status==ATLAS_HOST_OK) status=atlas_host_start(region->host,p->buffers,p->buffer_count);
  if (status==ATLAS_HOST_OK) status=atlas_host_wait(region->host,region->max_polls);
  if (status!=ATLAS_HOST_OK) return 3;
  if (region->io.read(region->io.context,p->output_address,outputs[0],p->output_bytes)) return 2;
  return 0;
}
""")


def copy_region(name: str, *, words: str, word_count: int, writes: list[tuple[str | None, int, int, int]],
                buffers: list[tuple[int, int, str]], output_address: int, output_bytes: int, inputs: int) -> tuple[str, str]:
    """Declarations and C for one copy region; writes are (constant symbol or None, address, input port, bytes)."""
    upper = name.upper()
    declarations = COPY_TYPES + f'''
#define {upper}_WORKSPACE_BYTES 1u
#define {upper}_SOURCE_WORKSPACE_BYTES 1u
typedef enum {{ {upper}_OK=0,{upper}_INVALID,{upper}_MEMORY_ERROR,{upper}_DEVICE_ERROR }} {name}_status;
typedef atlas_region_context {name}_context;
typedef atlas_copy_source_context {name}_source_context;
{name}_status {name}_bind({name}_context *,atlas_host *,atlas_tile_memory_io,void *,size_t,size_t);
{name}_status {name}_source_bind({name}_source_context *,{name}_context *,void *,size_t);
int {name}_source_dispatch(void *,const void *const *,void *const *);
'''
    rows = ",".join(f"{{{symbol or 0},{address}ULL,{port}u,{size}u}}" for symbol, address, port, size in writes) or "{0,0,0,0}"
    table = ",".join(f"{{{address}ULL,{size}u,{access}}}" for address, size, access in buffers)
    implementation = CHECKS + COPY_DISPATCH + f'''
static const atlas_copy_write {name}_writes[]={{{rows}}};
static const atlas_host_buffer {name}_buffers[]={{{table}}};
static const atlas_copy_plan {name}_plan={{{words},{name}_writes,{name}_buffers,{output_address}ULL,{word_count}u,{len(writes)}u,
  {len(buffers)}u,{inputs}u,{output_bytes}u}};
{name}_status {name}_bind({name}_context *ctx,atlas_host *host,atlas_tile_memory_io io,void *scratch,size_t bytes,size_t polls) {{
  (void)scratch;(void)bytes; return ({name}_status)atlas_copy_bind(ctx,host,&io,polls);
}}
{name}_status {name}_source_bind({name}_source_context *ctx,{name}_context *region,void *scratch,size_t bytes) {{
  (void)scratch;(void)bytes; return ({name}_status)atlas_copy_source_bind(ctx,region);
}}
int {name}_source_dispatch(void *opaque,const void *const *inputs,void *const *outputs) {{
  return atlas_copy_dispatch(&{name}_plan,opaque,inputs,outputs);
}}
'''
    return declarations, implementation
