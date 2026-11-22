"""Generate application context setup from each compiled region's public ABI."""

from merlin.compiler.build.regions import ApplicationCBinding


def application_binding(name, regions, interface):
    external = interface.get("model_parameters")
    raw = max(1, *(row["description"].get("workspace_bytes", 0) for row in regions))
    converted = max(1, *(row["description"].get("source_workspace_bytes", 0) for row in regions))
    sizes = dict(device_scratch=raw, conversion_scratch=converted,
                 workspace=max(1, interface["workspace_bytes"]))
    deployment = interface.get("deployment")
    fixed = {}
    if deployment is not None:
        provider = deployment["provider"]
        alignment = provider["allocation_alignment"]
        context_bytes = provider["host_context_reservation_bytes"]
        if (type(alignment) is not int or alignment <= 0 or alignment & (alignment - 1)
                or type(context_bytes) is not int or context_bytes <= 0):
            raise ValueError("invalid deployment provider context budget or alignment")
        for key, size in sizes.items():
            matches = [row for row in deployment["allocations"] if row["name"] == key]
            if (len(matches) != 1 or type(matches[0].get("address")) is not int
                    or matches[0]["address"] <= 0 or matches[0]["address"] % alignment
                    or type(matches[0].get("bytes")) is not int or matches[0]["bytes"] < size
                    or matches[0]["address"] + matches[0]["bytes"] > 1 << 32):
                raise ValueError("deployment allocation differs for " + key)
            fixed[key] = matches[0]["address"]
    def capacity(key):
        return f"{sizes[key]}u" if deployment is not None else f"sizeof(ctx->{key})"

    declarations, initialization, pointers = [], [], []
    for i, row in enumerate(regions):
        if row["placement"] == "host":
            pointers.append("0")
            continue
        data = row["description"]
        declarations += [f"  {data['context_type']} region_{i};", f"  {data['source_context_type']} source_{i};"]
        initialization += [
            f"  if ({data['binding_function']}(&ctx->region_{i},host,io,ctx->device_scratch,{capacity('device_scratch')},max_polls)) return -1;",
            f"  if ({data['source_binding_function']}(&ctx->source_{i},&ctx->region_{i},ctx->conversion_scratch,{capacity('conversion_scratch')})) return -1;",
        ]
        pointers.append(f"&ctx->source_{i}")
    fields = "\n".join(declarations)
    scratch_fields = ("\n".join(f"  unsigned char *{key};" for key in sizes) if deployment is not None else
                      "\n".join(f"  _Alignas(64) unsigned char {key}[{size}];" for key, size in sizes.items()))
    context_assert = (f'_Static_assert(sizeof({name}_context)<={context_bytes}u,"deployment host context exceeds reservation");\n'
                      if deployment is not None else "")
    scratch_note = ("It points to private region scratch in the published deployment window."
                    if deployment is not None else
                    "It contains all private region scratch; prefer static storage on bare metal.")
    public = f"""
/* Zero-initialize and keep this context alive through completion or recovery.
 * {scratch_note}
 * Share one initialized atlas_host across applications using the same device. */
typedef struct {{
  atlas_host *host;
  {name}_driver_state state;
{f"  {name}_parameters parameters;" if external else ""}
  {name}_driver_contexts dispatch;
  void *items[{len(regions)}];
{fields}
{scratch_fields}
}} {name}_context;
{context_assert}int {name}_bind({name}_context *,atlas_host *,atlas_tile_memory_io,size_t max_polls);
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
{"" if initialization else "  (void)io;" + chr(10)}{''.join(f"  ctx->{key}=(unsigned char *)(uintptr_t){address}u;" + chr(10) for key, address in fixed.items())}{chr(10).join(initialization)}
  ctx->host=host;
  ctx->state=({name}_driver_state){{0}};
{chr(10).join(f"  ctx->items[{i}]={value};" for i, value in enumerate(pointers))}
  ctx->dispatch=({name}_driver_contexts){{ctx->items,{len(regions)}}};
  return 0;
}}
int {name}_run({name}_context *ctx,const void *const *inputs,void *const *outputs) {{
  if (!ctx || !ctx->host || !outputs{" || !inputs" if interface["inputs"] else ""}) return -1;
{typed}  return {name}_invoke(&ctx->state,ctx->workspace,{capacity('workspace')},
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
    if deployment is not None and all(row["placement"] == "host" for row in regions):
        # No device region contributes the host ABI that the fixed consumer binding names.
        # The headers come in dependency order, so their project includes are already satisfied.
        from pathlib import Path

        from .c_runtime import runtime_headers

        headers = runtime_headers(Path(__file__).resolve().parents[2] / "runtime").values()
        public = "".join(line for text in headers for line in text.splitlines(keepends=True)
                         if not line.lstrip().startswith('#include "')) + public
    return ApplicationCBinding(
        public,
        implementation,
        {
            "context_type": name + "_context",
            "bind": name + "_bind",
            "invoke": name + "_run",
            "recover": name + "_recover",
            "private_region_scratch_included": deployment is None,
            "raw_scratch_bytes": raw,
            "conversion_scratch_bytes": converted,
            "completion": "synchronous",
            "host_float_abi": "lp64d",
            "device_context_ownership": "caller",
            **({"host_context_reservation_bytes": context_bytes,
                "private_region_scratch_storage": "published_deployment_window"}
               if deployment is not None else {}),
            **({"bind_parameters": name + "_bind_parameters"} if external else {}),
        },
    )
