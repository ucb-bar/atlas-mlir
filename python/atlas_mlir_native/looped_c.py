"""Export the checked packed-loop ABI as an application-owned C/H pair."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from merlin.compile.c_sources import c_source_pair
from merlin.compile.c_tiling import (
    render_looped_mnk_c,
    render_looped_seed_c,
    render_streamed_mnk_c,
)
from merlin.semantic_compiler.model import KernelRequest
from merlin.semantic_compiler.target_binding import verify_native_publication

from .binding import (
    DRAM_BASE,
    RTL_REVISION,
    TILE_BYTES,
    check_io_memory_manifest,
    dram_slots,
)
from .looped_tiling import (
    _operand_cache,
    _virtual,
    prepare_execution_plan,
    split_epilogue,
    split_seeded_request,
)


def prepare_c_package(
    request_path: Path,
    compiled: Path,
    name: str,
    *,
    target_source: Path | None = None,
    compile_inputs: Path | None = None,
    runtime_linkage: str = "embedded",
    semantic_source: KernelRequest | None = None,
    semantic_operand_scale_exponents: tuple[int, int] | None = None,
    semantic_epilogue: str | None = None,
):
    """Validate source and publication before rendering raw FP8/BF16 host calls."""
    if compiled.is_symlink() or not compiled.is_dir():
        raise ValueError("native compilation must be a real directory")

    def member(filename):
        path = compiled / filename
        if path.is_symlink() or not path.is_file():
            raise ValueError("missing or linked native artifact: " + filename)
        return path.read_bytes()

    def sha(data):
        return hashlib.sha256(data).hexdigest()

    if json.loads(member("manifest.json")).get("schema") == "atlas.native_chunked_mnk_compilation.v1":
        from .chunked_c import prepare_c_package as prepare_chunked

        return prepare_chunked(
            request_path,
            compiled,
            name,
            target_source=target_source or Path(__file__).resolve().parents[2],
            runtime_linkage=runtime_linkage,
            compile_inputs=compile_inputs,
            semantic_source=semantic_source,
            semantic_operand_scale_exponents=semantic_operand_scale_exponents,
            semantic_epilogue=semantic_epilogue,
        )
    source = KernelRequest.from_record(json.loads(request_path.read_text()))
    tiling, device_tiling, streaming = prepare_execution_plan(source)
    geometry, seed = split_seeded_request(source)
    _, epilogue = split_epilogue(geometry)
    manifest_bytes = member("manifest.json")
    manifest = json.loads(manifest_bytes)
    check_io_memory_manifest(manifest)
    verify_native_publication(
        compiled,
        manifest,
        engine="merlin_native",
        request_digest=source.digest(),
        target_identity=source.target_identity,
    )
    if manifest.get("schema") != "atlas.native_looped_mnk_compilation.v1":
        raise ValueError("C export requires a checked whole-contraction compilation")
    if manifest.get("execution_requires") != ("streamed_host_dispatch" if streaming else None):
        raise ValueError("loop publication has the wrong host dispatch requirement")
    if KernelRequest.from_record(json.loads(member("source-request.json"))).digest() != source.digest():
        raise ValueError("published loop source differs from the requested source")
    if json.loads(member("looped-mnk-tiling.json")) != tiling.record():
        raise ValueError("published loop plan differs from the source")
    execution_bytes = member("execution_plan.json")
    execution = json.loads(execution_bytes)
    addresses = execution.get("input_addresses")
    output = execution.get("output_addresses")
    if (
        not isinstance(addresses, list)
        or len(addresses) != (3 if seed else 2)
        or not isinstance(output, list)
        or len(output) != 1
    ):
        raise ValueError("looped C export has the wrong input or output ranges")
    sizes = (
        *device_tiling.buffer_bytes[:2],
        *((device_tiling.buffer_bytes[2],) if seed else ()),
        device_tiling.buffer_bytes[2],
    )
    ranges = []
    for address, size in zip(addresses + output, sizes, strict=True):
        if (
            type(address) is not int
            or address < DRAM_BASE
            or (address - DRAM_BASE) % TILE_BYTES
            or address + size > DRAM_BASE + dram_slots() * TILE_BYTES
        ):
            raise ValueError("looped C export range exceeds the selected diagnostic window")
        ranges.append((address, address + size))
    if any(a < d and c < b for i, (a, b) in enumerate(ranges) for c, d in ranges[i + 1 :]):
        raise ValueError("looped C input and output ranges overlap")
    expected = {
        "schema": "atlas.looped_mnk_execution_plan.v1",
        "scope": "diagnostic_selected_standalone_core",
        "shape": list(device_tiling.extents[:2]),
        "reduction_extent": tiling.extents[2],
        "input_addresses": addresses,
        "input_buffer_bytes": list(sizes[:-1]),
        "output_addresses": output,
        "output_bytes": sizes[-1],
        "output_physical_layouts": ["mn_tiles_bf16_pair_halves"],
        "numerical_policy": tiling.qualified_policy,
        "tiling": device_tiling.record(),
        **(
            {
                "accumulator_seed": {
                    "input_index": 2,
                    "conversion": "none",
                    "physical_layout": "mn_tiles_bf16_pair_halves",
                }
            }
            if seed
            else {}
        ),
        **({"streaming": streaming.record()} if streaming else {}),
        **({"epilogue": epilogue.record()} if epilogue is not None else {}),
    }
    codegen = execution.get("codegen")
    if codegen is not None:
        if codegen.get("reduction_unroll") not in (1, 2, 4, 8):
            raise ValueError("unrecognized reduction unroll factor")
        expected["codegen"] = codegen
    reuse_enabled = bool(execution.get("operand_reuse"))
    if "operand_reuse" in execution:
        expected["operand_reuse"] = _operand_cache(device_tiling, reuse_enabled)
    virtual = _virtual(
        device_tiling,
        tuple(addresses[:2]),
        output[0],
        epilogue,
        seed_address=addresses[2] if seed else None,
        k_unroll=codegen["reduction_unroll"] if codegen else 1,
        operand_reuse=reuse_enabled,
    )
    if codegen and codegen["program_size_limit_bytes"] is not None:
        virtual = virtual.replace(
            "attributes {",
            f"attributes {{atlas.program_size_limit_bytes = {codegen['program_size_limit_bytes']} : i64, ",
            1,
        )
    if execution != expected or member("atlas-virtual.mlir").decode() != virtual:
        raise ValueError("loop execution ABI differs from its checked source lowering")
    binary = member("program.bin")
    program = json.loads(member("physical-program.json"))
    if (
        program.get("selected_rtl_revision") != RTL_REVISION
        or not binary
        or len(binary) % 4
        or binary != b"".join(row["word_u32"].to_bytes(4, "little") for row in program["instructions"])
    ):
        raise ValueError("loop binary differs from the selected instruction words")
    runtime = (target_source or Path(__file__).resolve().parents[2]) / "runtime"
    # Runtime ownership stays with the explicitly selected target checkout.
    if not (runtime / "atlas_host.h").is_file():
        raise ValueError("selected Atlas runtime sources are unavailable")
    if sha((runtime.parent / "docs/selected-variant-inventory.json").read_bytes()) != manifest.get(
        "target_inventory_sha256"
    ):
        raise ValueError("selected runtime checkout differs from the compiled target inventory")
    public = {"atlas_host.h": (runtime / "atlas_host.h").read_text()}
    host_source = (runtime / "atlas_host.c").read_text()
    from .c_runtime import program_array, runtime_members

    host_members, runtime_files, runtime_record = runtime_members(public["atlas_host.h"], host_source, runtime_linkage)
    m, n, k = tiling.extents
    ab, wb, ob = device_tiling.buffer_bytes
    workspace = ab + wb + ob + 2048 + (ob if seed else 0) + (m * n * 2 if streaming else 0)
    upper = name.upper()
    from .contraction_source_c import prepare_source_wrappers

    wrappers = prepare_source_wrappers(
        source,
        manifest=manifest,
        member=member,
        execution=execution,
        name=name,
        workspace=workspace,
        m=m,
        n=n,
        compile_inputs=compile_inputs,
        semantic_source=semantic_source,
        semantic_operand_scale_exponents=semantic_operand_scale_exponents,
        semantic_epilogue=semantic_epilogue,
        seed=seed,
    )
    region, region_declarations, region_source = wrappers.region, wrappers.declarations, wrappers.implementation
    private_headers, conversion_source = wrappers.private_headers, wrappers.packing
    declarations = f"""
#define {upper}_M {m}u
#define {upper}_N {n}u
#define {upper}_K {k}u
#define {upper}_A_BYTES {m * k}u
#define {upper}_W_BYTES {n * k}u
#define {upper}_OUTPUT_BYTES {m * n * 2}u
#define {upper}_WORKSPACE_BYTES {workspace}u
{f"#define {upper}_SEED_BYTES {m * n * 2}u" if seed else ""}

typedef enum {{ {upper}_OK = 0, {upper}_INVALID, {upper}_INPUT_DOMAIN,
               {upper}_MEMORY_ERROR, {upper}_DEVICE_ERROR }} {name}_status;
typedef struct {{
  atlas_host *host;
  atlas_tile_memory_io io;
  uint8_t *workspace;
  size_t max_polls;
  atlas_host_buffer buffers[{4 if seed else 3}];
}} {name}_context;

/* Keep context, host, callback state and workspace alive through completion.
 * No heap allocation. Workspace must be disjoint from application buffers. */
{name}_status {name}_bind({name}_context *context, atlas_host *host,
                         atlas_tile_memory_io io, void *workspace,
                         size_t workspace_bytes, size_t max_polls);
/* Row-major raw FP8 A[M,K], W[N,K]; row-major BF16 little-endian output[M,N].
 * Inputs must be finite normal E4M3 or positive zero. Outputs stay unchanged
 * on failure. A timeout leaves the launch active; finish/reset it before reuse. */
{name}_status {name}_run({name}_context *context, const uint8_t *a,
                        const uint8_t *w{", const uint8_t *seed_bf16_le" if seed else ""}, uint8_t *output_bf16_le,
                        atlas_host_result *device_result);
const uint32_t *{name}_program(size_t *word_count);
int {name}_dispatch(void *context, const void *const *inputs, void *const *outputs);
"""
    declarations += region_declarations
    helpers = (
        render_streamed_mnk_c(streaming, name) if streaming else render_looped_mnk_c(tiling, name)
    ) + conversion_source
    seed_argument = ", const uint8_t *seed" if seed else ""
    seed_pack = seed_checks = seed_write = seed_buffer = ""
    if seed:
        helpers += render_looped_seed_c(tiling, name, b"\0\0")
        seed_checks = f"""if (!{name}_valid_range(seed,{m * n * 2}u) ||
      {name}_overlaps(seed,{m * n * 2}u,out,{m * n * 2}u) ||
      {name}_overlaps(seed,{m * n * 2}u,ctx,sizeof(*ctx)) ||
      {name}_overlaps(seed,{m * n * 2}u,ctx->host,sizeof(*ctx->host)) ||
      {name}_overlaps(seed,{m * n * 2}u,ctx->workspace,{workspace}u)) return {upper}_INVALID;
  for (size_t i=0;i<{m * n}u;++i) {{
    unsigned bits=(unsigned)seed[2*i]|((unsigned)seed[2*i+1]<<8);
    unsigned exponent=bits&0x7f80u;
    if (bits && (!exponent || exponent==0x7f80u)) return {upper}_INPUT_DOMAIN;
  }}"""
        seed_pack = f"""uint8_t *packed_seed=temporary+2048u;
  {name}_pack_seed(packed_seed,seed);
  for (size_t tile=0;tile<{ob // 2048}u;++tile) {{
    uint8_t *logical=packed_seed+tile*2048u;
    for (size_t row=0;row<32u;++row)
      for (size_t col=0;col<32u;++col)
        __builtin_memcpy(temporary+(col/16u)*1024u+(row*16u+col%16u)*2u,
                         logical+(row*32u+col)*2u,2u);
    __builtin_memcpy(logical,temporary,2048u);
  }}"""
        seed_write = (
            f"if (ctx->io.write(ctx->io.context,{addresses[2]}u,packed_seed,{ob}u)) return {upper}_MEMORY_ERROR;"
        )
        seed_buffer = f"{{{addresses[2]}u,{ob}u,ATLAS_HOST_DEVICE_READ}},"
    pre_pack = f"{name}_pack_rows(packed_a,a,{m}u,32u);\n  {name}_pack_rows(packed_w,w,{n}u,32u);"
    stream_start = stream_pack = stream_end = ""
    assemble = f"{name}_assemble_output(out,packed_out);"
    if streaming:
        bm, bn, _ = device_tiling.extents
        pre_pack = ""
        stream_start = f"""uint8_t *staged=temporary+2048u;
  for (size_t first_m=0;first_m<{m}u;first_m+={bm}u)
    for (size_t first_n=0;first_n<{n}u;first_n+={bn}u) {{"""
        stream_pack = f"""{name}_pack_block(packed_a,a,{m}u,first_m,{bm}u,32u);
      {name}_pack_block(packed_w,w,{n}u,first_n,{bn}u,32u);"""
        assemble = f"{name}_scatter_block(staged,packed_out,first_m,first_n);"
        stream_end = f"}}\n  __builtin_memcpy(out,staged,{m * n * 2}u);"
    implementation = f"""
{program_array(name + "_words", binary)}
const uint32_t *{name}_program(size_t *count) {{
  if (count) *count = {len(binary) // 4}u;
  return {name}_words;
}}
static int {name}_valid_range(const void *pointer, size_t bytes) {{
  return pointer && bytes && bytes <= UINTPTR_MAX-(uintptr_t)pointer;
}}
static int {name}_overlaps(const void *a, size_t an, const void *b, size_t bn) {{
  return (uintptr_t)a < (uintptr_t)b+bn && (uintptr_t)b < (uintptr_t)a+an;
}}
{name}_status {name}_bind({name}_context *ctx, atlas_host *host,
                         atlas_tile_memory_io io, void *workspace,
                         size_t bytes, size_t polls) {{
  if (!ctx || !host || !host->mmio.read32 || !host->mmio.write32 || host->active ||
      !io.write || !io.read || !polls || bytes < {workspace}u ||
      !{name}_valid_range(workspace,{workspace}u) ||
      {name}_overlaps(workspace,{workspace}u,ctx,sizeof(*ctx)) ||
      {name}_overlaps(workspace,{workspace}u,host,sizeof(*host)) ||
      {name}_overlaps(ctx,sizeof(*ctx),host,sizeof(*host))) return {upper}_INVALID;
  *ctx = ({name}_context){{host,io,(uint8_t *)workspace,polls,{{
    {{{addresses[0]}u,{ab}u,ATLAS_HOST_DEVICE_READ}},
    {{{addresses[1]}u,{wb}u,ATLAS_HOST_DEVICE_READ}},
    {seed_buffer}
    {{{output[0]}u,{ob}u,ATLAS_HOST_DEVICE_WRITE}}
  }}}};
  return {upper}_OK;
}}
static int {name}_valid_fp8(const uint8_t *values, size_t bytes) {{
  for (size_t i = 0; i < bytes; ++i) {{
    unsigned exponent = (values[i] >> 3) & 15u;
    if (values[i] != 0 && (exponent == 0 || exponent == 15)) return 0;
  }}
  return 1;
}}
{name}_status {name}_run({name}_context *ctx, const uint8_t *a,
                        const uint8_t *w{seed_argument}, uint8_t *out, atlas_host_result *device) {{
  if (device) *device = ATLAS_HOST_INVALID;
  if (!ctx || !ctx->host || !ctx->io.write || !ctx->io.read || !ctx->workspace || !ctx->max_polls)
    return {upper}_INVALID;
  if (ctx->host->active) {{ if (device) *device = ATLAS_HOST_BUSY; return {upper}_DEVICE_ERROR; }}
  if (!{name}_valid_range(a,{m * k}u) || !{name}_valid_range(w,{n * k}u) ||
      !{name}_valid_range(out,{m * n * 2}u) ||
      {name}_overlaps(out,{m * n * 2}u,ctx,sizeof(*ctx)) ||
      {name}_overlaps(out,{m * n * 2}u,ctx->host,sizeof(*ctx->host)) ||
      {name}_overlaps(out,{m * n * 2}u,a,{m * k}u) || {name}_overlaps(out,{m * n * 2}u,w,{n * k}u) ||
      {name}_overlaps(ctx->workspace,{workspace}u,a,{m * k}u) ||
      {name}_overlaps(ctx->workspace,{workspace}u,w,{n * k}u) ||
      {name}_overlaps(ctx->workspace,{workspace}u,out,{m * n * 2}u)) return {upper}_INVALID;
  if (!{name}_valid_fp8(a,{m * k}u) || !{name}_valid_fp8(w,{n * k}u)) return {upper}_INPUT_DOMAIN;
  {seed_checks}
  uint8_t *packed_a = ctx->workspace, *packed_w = packed_a+{ab}u;
  uint8_t *packed_out = packed_w+{wb}u, *temporary = packed_out+{ob}u;
  {pre_pack}
  {seed_pack}
  atlas_host_result result = atlas_host_ensure_loaded(ctx->host,{name}_words,{len(binary) // 4}u);
  if (device) *device = result;
  if (result != ATLAS_HOST_OK) return {upper}_DEVICE_ERROR;
  {stream_start}
  {stream_pack}
  if (ctx->io.write(ctx->io.context,{addresses[0]}u,packed_a,{ab}u) ||
      ctx->io.write(ctx->io.context,{addresses[1]}u,packed_w,{wb}u)) return {upper}_MEMORY_ERROR;
  {seed_write}
  result = atlas_host_start(ctx->host,ctx->buffers,{4 if seed else 3}u);
  if (result == ATLAS_HOST_OK) result = atlas_host_wait(ctx->host,ctx->max_polls);
  if (device) *device = result;
  if (result != ATLAS_HOST_OK) return {upper}_DEVICE_ERROR;
  if (ctx->io.read(ctx->io.context,{output[0]}u,packed_out,{ob}u)) return {upper}_MEMORY_ERROR;
  for (size_t tile = 0; tile < {ob // 2048}u; ++tile) {{
    uint8_t *physical = packed_out+tile*2048u;
    for (size_t row = 0; row < 32u; ++row)
      for (size_t column = 0; column < 32u; ++column)
        __builtin_memcpy(temporary+(row*32u+column)*2u,
               physical+(column/16u)*1024u+(row*16u+column%16u)*2u,2u);
    __builtin_memcpy(physical,temporary,2048u);
  }}
  {assemble}
  {stream_end}
  return {upper}_OK;
}}
int {name}_dispatch(void *context, const void *const *inputs, void *const *outputs) {{
  if (!inputs || !outputs) return {upper}_INVALID;
  return (int){name}_run(({name}_context *)context,inputs[0],inputs[1]{",inputs[2]" if seed else ""},outputs[0],0);
}}
"""
    implementation += region_source
    files = {
        filename: text.encode()
        for filename, text in c_source_pair(
            name,
            public_headers=public,
            private_headers=private_headers,
            sources={**host_members, "looped_packing.c": helpers},
            declarations=declarations,
            implementation=implementation,
        ).items()
    }
    files.update(runtime_files)
    files[name + ".bin"] = binary
    files[name + ".words.txt"] = "".join(
        f"{int.from_bytes(binary[i : i + 4], 'little'):08x}\n" for i in range(0, len(binary), 4)
    ).encode()
    files["README.md"] = f"""# {name}

Compile `{name}.c` and include `{name}.h`. The C file embeds the Atlas program;
`{name}.bin` contains the identical little-endian device words. No host ELF,
startup, linker script, main function or allocator is included.

Initialize `atlas_host` using the platform MMIO/cache services. Allocate
`{upper}_WORKSPACE_BYTES` host bytes, then call `{name}_bind`. Pass dense raw
FP8 A[M,K] and W[N,K] buffers to `{name}_run`; output is raw BF16 little-endian
M-by-N bytes. This API does not quantize floats. It validates the declared
finite-normal-or-positive-zero FP8 input domain before device writes.

The source packs operands, loads the image when it is not already resident, launches
{streaming.launch_counts[0] * streaming.launch_counts[1] if streaming else 1} time(s), waits,
reads all output tiles and removes physical BF16 pair/tail padding. The
application supplies DRAM read/write callbacks, coherent memory or cache
maintenance, startup and the final link. Reserve the fixed device ranges in
`manifest.json`; host linker relocation does not change Atlas DMA addresses.

Calls are synchronous. Check both the returned status and device status.
Outputs remain unchanged on errors. Keep the context, workspace, host and
callback state alive through an outstanding timed-out launch; finish or reset
that launch before reuse. Serialize use of a host. `{name}_dispatch` exposes
Merlin's pointer-array dispatch ABI with the same encoded types.

Use C11 with Clang/GCC and the same ABI as your application (RV64GCV bring-up
uses lp64d). Link the ordinary memory functions required by your toolchain.
Standalone AtlasCore and software-model evidence do not qualify the selected
complete SoC, cache policy or FPGA bitstream.
""".encode()
    if seed:
        files["README.md"] += (
            b"\nThe third input is a completed BF16 accumulator, row major: finite normal values or positive zero. "
            b"Its bits are copied without conversion; the input remains unchanged.\n"
        )
    if region is not None:
        files["README.md"] += f"""
`{name}_region_run` embeds the declared source parameters and accepts encoded
runtime region operands with BF16 byte output. For source float arrays, first
bind a `{name}_source_context` with separate `{upper}_SOURCE_WORKSPACE_BYTES`
scratch using `{name}_source_bind`. Call `{name}_source_run` or
`{name}_source_dispatch`; these apply the recorded conversions and axis order,
execute the same region and decode BF16 results into float output. Their output
stays unchanged on failure. Both workspaces and contexts must survive a timeout.
This diagnostic precision transformation is not a quality pass or a complete
model execution plan.
""".encode()
    verify_native_publication(
        compiled,
        manifest,
        engine="merlin_native",
        request_digest=source.digest(),
        target_identity=source.target_identity,
    )
    if member("manifest.json") != manifest_bytes or member("execution_plan.json") != execution_bytes:
        raise ValueError("native compilation changed during C export")
    record = {
        "schema": "atlas.looped_c_package.v1",
        "status": "generated_c_package",
        "package_name": name,
        "engine": "merlin_native",
        "request_digest": source.digest(),
        "target_identity": source.target_identity,
        "native_manifest_sha256": sha(manifest_bytes),
        "device_layout": execution,
        "workspace_bytes": workspace,
        "launches_per_call": streaming.launch_counts[0] * streaming.launch_counts[1] if streaming else 1,
        **({"streaming": streaming.record()} if streaming else {}),
        "host_inputs": [
            {"id": node.id, "shape": list(node.type.shape), "dtype": "fp8_e4m3", "bytes": size}
            for node, size in zip(source.nodes[:2], (m * k, n * k), strict=True)
        ]
        + ([{"id": seed.id, "shape": [m, n], "dtype": "bf16-le-raw", "bytes": m * n * 2}] if seed else []),
        "host_output": {"id": source.outputs[0], "shape": [m, n], "dtype": "bf16-le-raw", "bytes": m * n * 2},
        "source_members": {
            "atlas_host.h": sha(public["atlas_host.h"].encode()),
            "atlas_host.c": sha(host_source.encode()),
            "looped_packing.c": sha(helpers.encode()),
            **{filename: sha(text.encode()) for filename, text in private_headers.items()},
        },
        "entry_function": name + "_run",
        "dispatch_function": name + "_dispatch",
        "dispatch_completion": "synchronous",
        "startup_included": False,
        "elf_included": False,
        "precision_conversion": region is not None,
        "executed": False,
        **runtime_record,
    }
    if runtime_files:
        files["README.md"] += (
            b"\nLink `runtime/atlas_host.c` once for the application, alongside every region C file.\n"
        )
    if region is not None:
        record["source_region"] = region
    return record, files
