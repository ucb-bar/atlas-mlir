"""Shared C slicing dispatcher with checked pointers and atomic publication."""
from .shared_dispatch_c import CHECKS, guarded, shape

# Stage one launch of a constant launch table: point the body's input and output tables at their slices
# and set its first index. Each region calls its bodies directly by index, so no body is reached through
# a function pointer, and pointer tables are sized for the region's widest launch.
SLICE_RUN = guarded("slice_run", """typedef struct {
  uint32_t body,source,first_field,first,inputs,outputs,refs; /* refs: (input, offset) pairs, then workspace offsets */
} atlas_slice_launch;
typedef struct { const atlas_slice_launch *launches; const uint32_t *refs,*copies; uint32_t launch_count,copy_count,max_inputs,max_outputs; } atlas_slice_plan;
static __attribute__((unused)) void *atlas_slice_stage(const atlas_slice_plan *plan,size_t l,uint8_t *region,uint8_t *workspace,
                                                       const void *const *inputs,const void **body_inputs,void **body_outputs) {
  const atlas_slice_launch *x=&plan->launches[l];
  const uint32_t *r=plan->refs+x->refs;
  if (x->inputs>plan->max_inputs || !x->outputs || x->outputs>plan->max_outputs) return 0;
  for (size_t i=0;i<x->inputs;++i) body_inputs[i]=(const uint8_t *)inputs[r[2u*i]]+r[2u*i+1u];
  for (size_t i=0;i<x->outputs;++i) body_outputs[i]=workspace+r[2u*x->inputs+i];
  if (x->first_field!=UINT32_MAX) { size_t first=x->first; __builtin_memcpy(region+x->first_field,&first,sizeof(first)); }
  return region+x->source;
}
static __attribute__((unused)) int atlas_slice_publish(const atlas_slice_plan *plan,const uint8_t *workspace,void *const *outputs) {
  for (size_t i=0;i<plan->copy_count;++i) __builtin_memcpy(outputs[i],workspace+plan->copies[2u*i],plan->copies[2u*i+1u]);
  return 0;
}
""")


def render_slice_dispatch(name, *, workspace_bytes, fields, bind_calls, input_bytes,
                          output_slices, launches):
    """Return declarations and C source; output_slices are (workspace offset,size).

    Each launch is dict(body=index, inputs=[(input index, byte offset)],
    outputs=[workspace byte offset], first=None or the body's input_first).
    """
    upper=name.upper();n=len(input_bytes)
    declarations=f'''
#define {upper}_WORKSPACE_BYTES {workspace_bytes}u
#define {upper}_SOURCE_WORKSPACE_BYTES 1u
typedef enum {{ {upper}_OK=0,{upper}_INVALID=1,{upper}_DOMAIN=2,{upper}_MEMORY_ERROR=3,{upper}_DEVICE_ERROR=4 }} {name}_status;
typedef struct {{ atlas_host *host; uint8_t *workspace;
{fields}
}} {name}_context;
typedef struct {{ {name}_context *region; }} {name}_source_context;
{name}_status {name}_bind({name}_context *,atlas_host *,atlas_tile_memory_io,void *,size_t,size_t);
{name}_status {name}_source_bind({name}_source_context *,{name}_context *,void *,size_t);
int {name}_source_dispatch(void *,const void *const *,void *const *);
'''
    rows,refs=[],[]
    for launch in launches:
        if not launch['outputs']:
            raise ValueError('slice launch has no outputs')
        first=launch.get('first')
        field='UINT32_MAX' if first is None else f'(uint32_t)offsetof({name}_context,source_{launch["body"]}.input_first)'
        rows.append(f'{{{launch["body"]}u,(uint32_t)offsetof({name}_context,source_{launch["body"]}),{field},'
                    f'{first or 0}u,{len(launch["inputs"])}u,{len(launch["outputs"])}u,{len(refs)}u}}')
        refs+=[value for pair in launch['inputs'] for value in pair]+list(launch['outputs'])
    cases='\n'.join(f'    case {body}u: status={name}_body_{body:03d}_source_dispatch(source,body_inputs,body_outputs); break;'
                     for body in sorted({launch['body'] for launch in launches}))
    max_inputs=max((len(launch['inputs']) for launch in launches),default=0)
    max_outputs=max((len(launch['outputs']) for launch in launches),default=0)
    copies=','.join(f'{offset}u,{size}u' for offset,size in output_slices)
    implementation=f'''
#include <stddef.h>
#include <string.h>
{CHECKS}{SLICE_RUN}{shape(name,context=name+'_source_context',region=name+'_context',workspace_bytes=workspace_bytes,
               inputs=list(input_bytes),outputs=[size for _,size in output_slices])}
static const atlas_slice_launch {name}_launches[]={{{','.join(rows)}}};
static const uint32_t {name}_refs[]={{{','.join(f'{value}u' for value in refs) or '0u'}}},{name}_copies[]={{{copies}}};
static const atlas_slice_plan {name}_plan={{{name}_launches,{name}_refs,{name}_copies,{len(rows)}u,{len(output_slices)}u,{max_inputs}u,{max_outputs}u}};
{name}_status {name}_bind({name}_context *ctx,atlas_host *host,atlas_tile_memory_io io,void *workspace,size_t bytes,size_t polls) {{
  if (atlas_bind_conflict(ctx,sizeof(*ctx),host,workspace,{workspace_bytes}u,bytes,&io,polls)) return {upper}_INVALID;
  ctx->host=host;ctx->workspace=workspace;
{chr(10).join(bind_calls)}
  return {upper}_OK;
}}
{name}_status {name}_source_bind({name}_source_context *ctx,{name}_context *region,void *workspace,size_t bytes) {{
  (void)workspace;(void)bytes;
  if (atlas_source_bind_conflict(ctx,sizeof(*ctx),region,sizeof(*region),offsetof({name}_context,host),offsetof({name}_context,workspace),{workspace_bytes}u))
    return {upper}_INVALID;
  ctx->region=region;return {upper}_OK;
}}
int {name}_source_dispatch(void *opaque,const void *const *inputs,void *const *outputs) {{
  {'(void)inputs;' if not n else ''}
  {name}_source_context *ctx=opaque;
  if (!atlas_span_valid(ctx,sizeof(*ctx)) || !atlas_span_valid(ctx->region,sizeof(*ctx->region))) return {upper}_INVALID;
  {name}_context *region=ctx->region;
  if (atlas_dispatch_conflict(&{name}_shape,ctx,region,region->host,region->workspace,inputs,outputs)) return {upper}_INVALID;
  if (region->host->active) return {upper}_DEVICE_ERROR;
  const void *body_inputs[{max(1,max_inputs)}]={{0}}; void *body_outputs[{max(1,max_outputs)}];
  for (size_t l=0;l<{len(rows)}u;++l) {{
    void *source=atlas_slice_stage(&{name}_plan,l,(uint8_t *)region,region->workspace,inputs,body_inputs,body_outputs);
    int status={upper}_INVALID;
    if (!source) return {upper}_INVALID;
    switch ({name}_launches[l].body) {{
{cases}
    default: return {upper}_INVALID;
    }}
    if (status) return status;
  }}
  return atlas_slice_publish(&{name}_plan,region->workspace,outputs);
}}
'''
    return declarations,implementation
