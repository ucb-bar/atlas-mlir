"""Bind ordered host chunk iteration to checked native matrix bodies."""

from __future__ import annotations

import hashlib
import json

from merlin.compile.c_sources import c_source_pair
from merlin.compile.c_tiling import (
    render_streamed_k_driver_c,
    streamed_k_driver_workspace_bytes,
)
from merlin.semantic_compiler.model import KernelRequest
from merlin.semantic_compiler.target_binding import verify_native_publication

from .binding import DRAM_BASE, TILE_BYTES
from .c_runtime import runtime_members
from .chunked_tiling import check_frames, chunk_bodies


def prepare_c_package(request_path, compiled, name, *, target_source, runtime_linkage="embedded", **options):
    """Check the full request, decomposition, physical frames and every leaf."""
    from .looped_tiling import prepare_execution_plan

    if options.get("compile_inputs") is not None:
        raise ValueError("source-bound chunked constants require application compilation")
    source = KernelRequest.from_record(json.loads(request_path.read_text()))
    _, _, plan = prepare_execution_plan(source)
    if plan is None or plan.k_block_tiles is None:
        raise ValueError("chunked C export needs an ordered K partition")
    manifest_bytes = (compiled / "manifest.json").read_bytes()
    manifest = json.loads(manifest_bytes)
    verify_native_publication(
        compiled,
        manifest,
        engine="merlin_native",
        request_digest=source.digest(),
        target_identity=source.target_identity,
    )
    if (
        manifest.get("schema") != "atlas.native_chunked_mnk_compilation.v1"
        or manifest.get("execution_requires") != "ordered_chunked_host_dispatch"
    ):
        raise ValueError("chunked publication has the wrong execution requirement")
    if json.loads((compiled / "chunked-mnk-tiling.json").read_text()) != plan.record():
        raise ValueError("chunk decomposition differs from the source")
    if KernelRequest.from_record(json.loads((compiled / "source-request.json").read_text())) != source:
        raise ValueError("chunk source differs from its publication")
    bindings = json.loads((compiled / "chunk-bindings.json").read_text())
    if set(bindings) != {"fixed_inputs", "fixed_outputs", "frames"}:
        raise ValueError("chunk binding fields differ from the selected ABI")
    check_frames(source, plan, bindings["fixed_inputs"], tuple(bindings["fixed_outputs"]), bindings["frames"])
    bodies = chunk_bodies(source, plan)
    if len(manifest["segments"]) != len(bodies):
        raise ValueError("chunk publication has missing or extra bodies")
    roots = {}
    for row, (role, body) in zip(manifest["segments"], bodies, strict=True):
        if row.get("role") != role:
            raise ValueError("chunk body order differs from the source")
        member = compiled / row["path"]
        if KernelRequest.from_record(json.loads((member / "source-request.json").read_text())) != body:
            raise ValueError("chunk body differs from the source decomposition")
        execution = json.loads((member / "execution_plan.json").read_text())
        roles = ("a", "w") if role == "first" else ("a", "w", "seed")
        if execution["input_addresses"] != [DRAM_BASE + bindings["frames"][key] * TILE_BYTES for key in roles] or (
            execution["output_addresses"] != [DRAM_BASE + bindings["frames"]["out"] * TILE_BYTES]
        ):
            raise ValueError("chunk physical bindings differ from the checked frame")
        roots[role] = member
    record, files = prepare_members(
        source, plan, roots, name, target_source=target_source, runtime_linkage=runtime_linkage, **options
    )
    verify_native_publication(
        compiled,
        manifest,
        engine="merlin_native",
        request_digest=source.digest(),
        target_identity=source.target_identity,
    )
    if (compiled / "manifest.json").read_bytes() != manifest_bytes:
        raise ValueError("chunk publication changed during C export")
    return {**record, "native_manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest()}, files


def prepare_members(
    source,
    plan,
    roots,
    name,
    *,
    target_source,
    runtime_linkage="separate",
    compile_inputs=None,
    semantic_source=None,
    semantic_operand_scale_exponents=None,
    semantic_epilogue=None,
):
    """Compose the same native leaves in a kernel or a flat application set."""
    from .looped_c import prepare_c_package as prepare_leaf

    bodies = chunk_bodies(source, plan)
    if set(roots) != {role for role, _ in bodies}:
        raise ValueError("chunk C export has missing or extra native bodies")
    headers, sources, records, children, files = {}, {}, {}, {}, {}
    for role, body in bodies:
        leaf = roots[role]
        if KernelRequest.from_record(json.loads((leaf / "source-request.json").read_text())) != body:
            raise ValueError("chunk leaf differs from the full source")
        child = name + "_" + role
        record, payloads = prepare_leaf(
            leaf / "source-request.json", leaf, child, target_source=target_source, runtime_linkage="separate"
        )
        records[role], children[role] = record, child
        headers[child + ".h"] = payloads[child + ".h"].decode()
        sources[child + ".c"] = payloads[child + ".c"].decode()
        files["programs/" + role + ".bin"] = (leaf / "program.bin").read_bytes()
    runtime = target_source / "runtime"
    header, host_source = (runtime / "atlas_host.h").read_text(), (runtime / "atlas_host.c").read_text()
    host_members, runtime_files, runtime_record = runtime_members(header, host_source, runtime_linkage)
    headers["atlas_host.h"] = header
    sources.update(host_members)
    driver_bytes = streamed_k_driver_workspace_bytes(plan)
    child_bytes = max(record["workspace_bytes"] for record in records.values())
    workspace = driver_bytes + child_bytes
    if workspace > (1 << 64) - 1:
        raise ValueError("chunk C workspace exceeds host size_t")
    m, n, k = plan.full.extents
    upper = name.upper()
    from .contraction_source_c import prepare_source_wrappers

    wrappers = prepare_source_wrappers(
        source,
        manifest={},
        member=None,
        execution=None,
        name=name,
        workspace=workspace,
        m=m,
        n=n,
        compile_inputs=compile_inputs,
        semantic_source=semantic_source,
        semantic_operand_scale_exponents=semantic_operand_scale_exponents,
        semantic_epilogue=semantic_epilogue,
    )
    if wrappers.packing:
        sources["source_conversion.c"] = wrappers.packing
    fields = "\n".join(f"  {child}_context {role};" for role, child in children.items())
    declarations = f"""
#define {upper}_A_BYTES {m * k}ULL
#define {upper}_W_BYTES {n * k}ULL
#define {upper}_OUTPUT_BYTES {m * n * 2}ULL
#define {upper}_WORKSPACE_BYTES {workspace}ULL
#define {upper}_PROGRAM_COUNT {len(bodies)}u
typedef enum {{ {upper}_OK=0,{upper}_INVALID,{upper}_INPUT_DOMAIN,
               {upper}_MEMORY_ERROR,{upper}_DEVICE_ERROR }} {name}_status;
typedef struct {{
  atlas_host *host;
  uint8_t *workspace;
  atlas_host_result device;
  unsigned ready;
{fields}
}} {name}_context;
{name}_status {name}_bind({name}_context *,atlas_host *,atlas_tile_memory_io,void *,size_t,size_t);
{name}_status {name}_run({name}_context *,const uint8_t *,const uint8_t *,uint8_t *,atlas_host_result *);
int {name}_dispatch(void *,const void *const *,void *const *);
const uint32_t *{name}_program(unsigned,size_t *);
"""
    declarations += wrappers.declarations
    bind = "\n".join(
        f"  if ({child}_bind(&ctx->{role},host,io,(uint8_t *)workspace+{driver_bytes}ULL,{child_bytes}ULL,polls)) return {upper}_INVALID;"
        for role, child in children.items()
    )
    callbacks = []
    for phase, role in ((0, "first"), (1, "carry"), (2, "final" if "final" in children else "carry")):
        if role not in children:
            continue
        extra = "" if role == "first" else ",seed"
        callbacks.append(
            f"    case {phase}: return (int){children[role]}_run(&ctx->{role},a,w{extra},out,&ctx->device);"
        )
    program_cases = "\n".join(
        f"    case {i}: return {children[role]}_program(count);" for i, (role, _) in enumerate(bodies)
    )
    implementation = (
        render_streamed_k_driver_c(plan, name)
        + f"""
static int {name}_valid_range(const void *p,size_t bytes) {{ return p && bytes<=UINTPTR_MAX-(uintptr_t)p; }}
static int {name}_overlaps(const void *a,size_t an,const void *b,size_t bn) {{
  return (uintptr_t)a<(uintptr_t)b+bn && (uintptr_t)b<(uintptr_t)a+an;
}}
{name}_status {name}_bind({name}_context *ctx,atlas_host *host,atlas_tile_memory_io io,
                         void *workspace,size_t bytes,size_t polls) {{
  if (!ctx || !host || host->active || bytes<{workspace}ULL || !polls ||
      !{name}_valid_range(workspace,{workspace}ULL) ||
      {name}_overlaps(workspace,{workspace}ULL,ctx,sizeof(*ctx)) ||
      {name}_overlaps(workspace,{workspace}ULL,host,sizeof(*host))) return {upper}_INVALID;
  *ctx=({name}_context){{.host=host,.workspace=workspace,.device=ATLAS_HOST_INVALID}};
{bind}
  ctx->ready=1;
  return {upper}_OK;
}}
static int {name}_invoke_chunk(void *context,unsigned phase,const uint8_t *a,
                               const uint8_t *w,const uint8_t *seed,uint8_t *out) {{
  {name}_context *ctx=context;
  (void)seed;
  switch (phase) {{
{chr(10).join(callbacks)}
    default: return {upper}_INVALID;
  }}
}}
{name}_status {name}_run({name}_context *ctx,const uint8_t *a,const uint8_t *w,
                        uint8_t *out,atlas_host_result *device) {{
  if (device) *device=ATLAS_HOST_INVALID;
  if (!ctx || !ctx->ready || !ctx->host || !ctx->workspace) return {upper}_INVALID;
  if (ctx->host->active) {{ if (device) *device=ATLAS_HOST_BUSY; return {upper}_DEVICE_ERROR; }}
  const void *values[]={{a,w,out}}; const size_t sizes[]={{{m * k}ULL,{n * k}ULL,{m * n * 2}ULL}};
  for (unsigned i=0;i<3;++i)
    if (!{name}_valid_range(values[i],sizes[i]) ||
        {name}_overlaps(values[i],sizes[i],ctx,sizeof(*ctx)) ||
        {name}_overlaps(values[i],sizes[i],ctx->host,sizeof(*ctx->host)) ||
        {name}_overlaps(values[i],sizes[i],ctx->workspace,{workspace}ULL)) return {upper}_INVALID;
  for (unsigned input=0;input<2;++input) {{
    const uint8_t *codes=values[input];
    for (size_t i=0;i<sizes[input];++i) {{
      unsigned exponent=(codes[i]>>3)&15u;
      if (codes[i] && (!exponent || exponent==15u)) return {upper}_INPUT_DOMAIN;
    }}
  }}
  ctx->device=ATLAS_HOST_INVALID;
  int status={name}_run_chunks(ctx,{name}_invoke_chunk,ctx->workspace,{driver_bytes}ULL,a,w,out);
  if (device) *device=ctx->device;
  return status<0 ? {upper}_INVALID : ({name}_status)status;
}}
int {name}_dispatch(void *ctx,const void *const *in,void *const *out) {{
  if (!in || !out) return {upper}_INVALID;
  return (int){name}_run(ctx,in[0],in[1],out[0],0);
}}
const uint32_t *{name}_program(unsigned index,size_t *count) {{
  switch (index) {{
{program_cases}
    default: if (count) *count=0; return 0;
  }}
}}
"""
    )
    implementation += wrappers.implementation
    pair = c_source_pair(
        name,
        public_headers=headers,
        private_headers=wrappers.private_headers,
        sources=sources,
        declarations=declarations,
        implementation=implementation,
    )
    files.update({path: data.encode() for path, data in pair.items()})
    files.update(runtime_files)
    files["README.md"] = f"""# {name}

Compile {name}.c and include {name}.h. The C file embeds every native body;
programs/ contains the same device words separately. Initialize atlas_host and
bind {upper}_WORKSPACE_BYTES host bytes through {name}_bind. Pass dense encoded
FP8 A[M,K] and W[N,K] to {name}_run; output is dense BF16 little-endian [M,N].
The call checks both complete input domains before launching. It retains BF16
carry bits without conversion and applies any epilogue only after the last K
chunk. Outputs remain unchanged on errors. Preserve all contexts/workspaces
through a timed-out launch, then finish/reset it before reuse. Calls serialize
use of a host and reload the selected body before each launch.

The platform supplies memory/MMIO/cache services, startup and final linking.
No host wrapper or final ELF is included. This diagnostic export does not
qualify original-model accuracy or execution on an integrated SoC/FPGA.
""".encode()
    return {
        "schema": "atlas.chunked_c_package.v1",
        "status": "generated_c_package",
        "engine": "merlin_native",
        "package_name": name,
        "request_digest": source.digest(),
        "target_identity": source.target_identity,
        "workspace_bytes": workspace,
        "driver_workspace_bytes": driver_bytes,
        "child_workspace_bytes": child_bytes,
        "streaming": plan.record(),
        "launches_per_call": plan.launch_counts[0] * plan.launch_counts[1] * plan.reduction_launches,
        "host_inputs": [
            {"id": node.id, "dtype": "fp8_e4m3", "shape": list(node.type.shape), "bytes": size}
            for node, size in zip(source.nodes[:2], (m * k, n * k), strict=True)
        ],
        "host_output": {"id": source.outputs[0], "dtype": "bf16-le-raw", "shape": [m, n], "bytes": m * n * 2},
        "entry_function": name + "_run",
        "dispatch_function": name + "_dispatch",
        "dispatch_completion": "synchronous",
        "startup_included": False,
        "elf_included": False,
        "precision_conversion": wrappers.region is not None,
        "executed": False,
        **({"source_region": wrappers.region} if wrappers.region is not None else {}),
        "programs": [
            {"role": role, "binary_sha256": hashlib.sha256(files["programs/" + role + ".bin"]).hexdigest()}
            for role, _ in bodies
        ],
        **runtime_record,
    }, files
