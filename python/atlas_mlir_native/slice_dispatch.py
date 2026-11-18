"""Shared C slicing dispatcher with checked pointers and atomic publication."""
def render_slice_dispatch(name, *, workspace_bytes, fields, bind_calls, input_bytes,
                          output_slices, launch_calls):
    """Return declarations and C source; output_slices are (workspace offset,size)."""
    upper=name.upper();n=len(input_bytes);m=len(output_slices)
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
    states=[('ctx','sizeof(*ctx)'),('region','sizeof(*region)'),('region->host','sizeof(*region->host)'),
            ('region->workspace',f'{workspace_bytes}u'),('inputs',f'{n}u*sizeof(*inputs)'),('outputs',f'{m}u*sizeof(*outputs)')]
    states=[(ptr,size) for ptr,size in states if n or ptr!='inputs']
    checks=[f'  if (!{name}_range({ptr},{size})) return {upper}_INVALID;' for ptr,size in states]
    for i,(ptr,size) in enumerate(states):
        for other,osize in states[:i]:checks.append(f'  if ({name}_overlap({ptr},{size},{other},{osize})) return {upper}_INVALID;')
    values=[(f'inputs[{i}]',size) for i,size in enumerate(input_bytes)]+[(f'outputs[{i}]',size) for i,(_,size) in enumerate(output_slices)]
    for i,(ptr,size) in enumerate(values):
        checks.append(f'  if (!{name}_range({ptr},{size}u)) return {upper}_INVALID;')
        for other,osize in states:checks.append(f'  if ({name}_overlap({ptr},{size}u,{other},{osize})) return {upper}_INVALID;')
        if ptr.startswith('outputs'):
            for other,osize in values[:i]:checks.append(f'  if ({name}_overlap({ptr},{size}u,{other},{osize}u)) return {upper}_INVALID;')
    copies='\n'.join(f'  __builtin_memcpy(outputs[{i}],region->workspace+{offset}u,{size}u);' for i,(offset,size) in enumerate(output_slices))
    implementation=f'''
#include <string.h>
static int {name}_range(const void *p,size_t n) {{ return p && n && n<=UINTPTR_MAX-(uintptr_t)p; }}
static int {name}_overlap(const void *a,size_t an,const void *b,size_t bn) {{ return (uintptr_t)a<(uintptr_t)b+bn && (uintptr_t)b<(uintptr_t)a+an; }}
{name}_status {name}_bind({name}_context *ctx,atlas_host *host,atlas_tile_memory_io io,void *workspace,size_t bytes,size_t polls) {{
  if (!{name}_range(ctx,sizeof(*ctx)) || !{name}_range(host,sizeof(*host)) || !{name}_range(workspace,{workspace_bytes}u) || bytes<{workspace_bytes}u ||
      {name}_overlap(ctx,sizeof(*ctx),host,sizeof(*host)) || {name}_overlap(ctx,sizeof(*ctx),workspace,{workspace_bytes}u) || {name}_overlap(host,sizeof(*host),workspace,{workspace_bytes}u) ||
      host->active || !host->mmio.read32 || !host->mmio.write32 || !io.read || !io.write || !polls) return {upper}_INVALID;
  ctx->host=host;ctx->workspace=workspace;
{chr(10).join(bind_calls)}
  return {upper}_OK;
}}
{name}_status {name}_source_bind({name}_source_context *ctx,{name}_context *region,void *workspace,size_t bytes) {{
  (void)workspace;(void)bytes;
  if (!{name}_range(ctx,sizeof(*ctx)) || !{name}_range(region,sizeof(*region)) || !{name}_range(region->host,sizeof(*region->host)) || !{name}_range(region->workspace,{workspace_bytes}u) ||
      {name}_overlap(ctx,sizeof(*ctx),region,sizeof(*region)) || {name}_overlap(ctx,sizeof(*ctx),region->host,sizeof(*region->host)) || {name}_overlap(ctx,sizeof(*ctx),region->workspace,{workspace_bytes}u) || region->host->active) return {upper}_INVALID;
  ctx->region=region;return {upper}_OK;
}}
int {name}_source_dispatch(void *opaque,const void *const *inputs,void *const *outputs) {{
  {'(void)inputs;' if not n else ''}
  {name}_source_context *ctx=opaque;
  if (!{name}_range(ctx,sizeof(*ctx)) || !{name}_range(ctx->region,sizeof(*ctx->region))) return {upper}_INVALID;
  {name}_context *region=ctx->region;
{chr(10).join(checks)}
  if (region->host->active) return {upper}_DEVICE_ERROR;
{chr(10).join(launch_calls)}
{copies}
  return {upper}_OK;
}}
'''
    return declarations,implementation
