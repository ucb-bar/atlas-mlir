"""Generate application context setup from each compiled region's public ABI."""

from merlin.compile.semantic_application import ApplicationCBinding


def application_binding(name, regions, interface):
    external = interface.get("model_parameters")
    raw = max(1, *(row["description"].get("workspace_bytes", 0) for row in regions))
    converted = max(1, *(row["description"].get("source_workspace_bytes", 0) for row in regions))
    declarations, initialization, pointers = [], [], []
    for i, row in enumerate(regions):
        if row["placement"] == "host":
            pointers.append("0")
            continue
        data = row["description"]
        declarations += [f"  {data['context_type']} region_{i};", f"  {data['source_context_type']} source_{i};"]
        initialization += [
            f"  if ({data['binding_function']}(&ctx->region_{i},host,io,ctx->device_scratch,sizeof(ctx->device_scratch),max_polls)) return -1;",
            f"  if ({data['source_binding_function']}(&ctx->source_{i},&ctx->region_{i},ctx->conversion_scratch,sizeof(ctx->conversion_scratch))) return -1;",
        ]
        pointers.append(f"&ctx->source_{i}")
    fields = "\n".join(declarations)
    public = f"""
/* Zero-initialize and keep this context alive through completion or recovery.
 * It contains all private region scratch; prefer static storage on bare metal.
 * Share one initialized atlas_host across applications using the same device. */
typedef struct {{
  atlas_host *host;
  {name}_driver_state state;
{f"  {name}_parameters parameters;" if external else ""}
  {name}_driver_contexts dispatch;
  void *items[{len(regions)}];
{fields}
  _Alignas(64) unsigned char device_scratch[{raw}];
  _Alignas(64) unsigned char conversion_scratch[{converted}];
  _Alignas(64) unsigned char workspace[{max(1, interface["workspace_bytes"])}];
}} {name}_context;
int {name}_bind({name}_context *,atlas_host *,atlas_tile_memory_io,size_t max_polls);
{f"int {name}_bind_parameters({name}_context *,const void *,size_t);" if external else ""}
int {name}_run({name}_context *,const void *const *inputs,void *const *outputs);
/* After the platform has completed/reset the device, discard a failed call. */
int {name}_recover({name}_context *);
"""
    mixed = interface.get("pointer_abi") == "opaque_native_dense_buffers"
    inputs = ",".join(f"(const float *)inputs[{i}]" for i in range(len(interface["inputs"]))) or "0"
    outputs = ",".join(f"(float *)outputs[{i}]" for i in range(len(interface["outputs"])))
    typed = "" if mixed else (f"  const float *typed_inputs[]={{{inputs}}};\n  float *typed_outputs[]={{{outputs}}};\n")
    parameter_bind = f"""
int {name}_bind_parameters({name}_context *ctx,const void *data,size_t bytes) {{
  if (!ctx || !ctx->host || !data || bytes!={external['payload_bytes']}u ||
      (uintptr_t)data>UINTPTR_MAX-bytes || (uintptr_t)ctx>UINTPTR_MAX-sizeof(*ctx)) return -1;
  if (ctx->state.active || ctx->host->active) return -2;
  if ((uintptr_t)data<(uintptr_t)ctx+sizeof(*ctx) && (uintptr_t)ctx<(uintptr_t)data+bytes) return -1;
  if ((uintptr_t)data<(uintptr_t)ctx->host+sizeof(*ctx->host) &&
      (uintptr_t)ctx->host<(uintptr_t)data+bytes) return -1;
  return {name}_parameters_bind(&ctx->parameters,data,bytes);
}}
""" if external else ""
    implementation = f"""
int {name}_bind({name}_context *ctx,atlas_host *host,atlas_tile_memory_io io,size_t max_polls) {{
  if (!ctx || !host || !max_polls) return -1;
  if (ctx->state.active || host->active) return -2;
{chr(10).join(initialization)}
  ctx->host=host;
  ctx->state=({name}_driver_state){{0}};
{chr(10).join(f"  ctx->items[{i}]={value};" for i, value in enumerate(pointers))}
  ctx->dispatch=({name}_driver_contexts){{ctx->items,{len(regions)}}};
  return 0;
}}
int {name}_run({name}_context *ctx,const void *const *inputs,void *const *outputs) {{
  if (!ctx || !ctx->host || !outputs{" || !inputs" if interface["inputs"] else ""}) return -1;
{typed}  return {name}_invoke(&ctx->state,ctx->workspace,sizeof(ctx->workspace),
      {"&ctx->parameters," if external else ""}
      {"inputs,outputs" if mixed else "typed_inputs,typed_outputs"},&ctx->dispatch);
}}
{parameter_bind}
int {name}_recover({name}_context *ctx) {{
  if (!ctx || !ctx->host || ctx->host->active) return -1;
  ctx->state=({name}_driver_state){{0}};
  return 0;
}}
"""
    return ApplicationCBinding(
        public,
        implementation,
        {
            "context_type": name + "_context",
            "bind": name + "_bind",
            "invoke": name + "_run",
            "recover": name + "_recover",
            "private_region_scratch_included": True,
            "raw_scratch_bytes": raw,
            "conversion_scratch_bytes": converted,
            "completion": "synchronous",
            "host_float_abi": "lp64d",
            "device_context_ownership": "caller",
            **({"bind_parameters": name + "_bind_parameters"} if external else {}),
        },
    )
