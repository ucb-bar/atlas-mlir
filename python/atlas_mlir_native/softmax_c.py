"""Export checked row-softmax images through the existing Atlas host runtime."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from merlin.compile.c_sources import c_source_pair
from merlin.compile.c_tiling import render_row_softmax_c
from merlin.semantic_compiler.model import KernelRequest
from merlin.semantic_compiler.target_binding import verify_native_publication

from .binding import (
    DRAM_BASE,
    RTL_REVISION,
    TILE_BYTES,
    AtlasNativeBinding,
    check_io_memory_manifest,
    dram_slots,
)
from .layout import pack_bf16_pair_halves
from .row_softmax import prepare_row_softmax


def prepare_c_package(
    source_path: Path, compiled: Path, name: str, *, target_source: Path, runtime_linkage: str = "embedded"
):
    """Bind compiled-0, compiled-1, ... to their regenerated source row groups.

    Inputs and outputs are dense raw BF16, not sampled tensors or interpreter
    callbacks. Each image was selected and checked by the normal native path.
    This exporter validates publication identities and the physical boundary;
    it does not confer numerical or full-system qualification.
    """
    source = KernelRequest.from_record(json.loads(source_path.read_text()))
    return _prepare_c_package(source, compiled, name, target_source=target_source, runtime_linkage=runtime_linkage)


def _prepare_c_package(
    source: KernelRequest,
    compiled: Path,
    name: str,
    *,
    target_source: Path,
    precision=None,
    runtime_linkage: str = "embedded",
    semantic_source: KernelRequest | None = None,
    row_compilations: tuple[Path, ...] | None = None,
    row_loop_compilation: Path | None = None,
    allow_masked=False,
    allow_guarded=False,
):
    staged_precision = precision
    if semantic_source is not None:
        if precision is not None:
            raise ValueError("select staged or semantic softmax source binding, not both")
        from .softmax_precision import lower_semantic_softmax

        precision = lower_semantic_softmax(semantic_source, allow_masked=allow_masked, allow_guarded=allow_guarded)
        if precision.request.digest() != source.digest():
            raise ValueError("semantic softmax differs from its checked native source")
    if compiled.is_symlink() or not compiled.is_dir():
        raise ValueError("softmax compilation must be a real directory")

    def member(root, filename):
        relative = Path(filename)
        path = root / relative
        if (
            relative.is_absolute()
            or relative.as_posix() != filename
            or ".." in relative.parts
            or path.is_symlink()
            or not path.is_file()
            or not path.resolve().is_relative_to(root.resolve())
        ):
            raise ValueError("missing or linked softmax artifact: " + filename)
        return path.read_bytes()

    def digest(payload):
        return hashlib.sha256(payload).hexdigest()

    plan = prepare_row_softmax(source)
    if row_compilations is None and row_loop_compilation is None and (compiled / "manifest.json").is_file():
        if json.loads(member(compiled, "manifest.json")).get("schema") == "atlas.row_loop_compilation.v1":
            row_loop_compilation = compiled
    row_loop = None
    if row_loop_compilation is not None:
        from .row_loop import verify_row_loop

        if row_compilations is not None or not row_loop_compilation.resolve().is_relative_to(compiled.resolve()):
            raise ValueError("select one contained row-loop compilation or separate row programs")
        row_loop, loop_manifest = verify_row_loop(source, row_loop_compilation, target_source=target_source)
    if row_compilations is not None and (
        len(row_compilations) != len(plan.regions)
        or len(set(row_compilations)) != len(row_compilations)
        or any(not root.resolve().is_relative_to(compiled.resolve()) for root in row_compilations)
    ):
        raise ValueError("softmax needs exactly one distinct compilation per row group")
    runtime = target_source / "runtime"
    inventory = member(target_source, "docs/selected-variant-inventory.json")
    htext, ctext = (
        member(runtime, "atlas_host.h").decode(),
        member(runtime, "atlas_host.c").decode(),
    )
    from .c_runtime import program_array, runtime_members

    host_members, runtime_files, runtime_record = runtime_members(htext, ctext, runtime_linkage)
    m, n = plan.shape
    columns, groups = plan.column_tiles, len(plan.regions)
    input_tiles = columns * (2 if plan.masked else 1)
    mask_parameter = ", const uint8_t *keep" if plan.masked else ""
    output_bytes = m * n * 2
    workspace = output_bytes + 2 * TILE_BYTES
    embedded, initializers, images, files = [], [], [], {}
    publications = []
    slots = max(
        sum(node.effect in {"input", "constant"} for node in row.request.nodes) + columns for row in plan.regions
    )
    arena_bytes = 0
    selected_rows = plan.regions if row_loop is None else plan.regions[:1]
    for index, region in enumerate(selected_rows):
        root = (
            row_loop_compilation / "body"
            if row_loop_compilation is not None
            else compiled / f"compiled-{index}"
            if row_compilations is None
            else row_compilations[index]
        )
        if root.is_symlink() or not root.is_dir():
            raise ValueError("missing or linked softmax row compilation")
        manifest_bytes = member(root, "manifest.json")
        manifest = json.loads(manifest_bytes)
        check_io_memory_manifest(manifest)
        verify_native_publication(
            root,
            manifest,
            engine="merlin_native",
            request_digest=region.request.digest(),
            target_identity=source.target_identity,
        )
        if staged_precision is not None and row_loop is None:
            expected_row = {
                **plan.record(),
                "selected_row_index": index,
                "selected_request_digest": region.request.digest(),
            }
            if (
                json.loads(member(root, "precision-plan.json")) != precision.record()
                or manifest.get("softmax_source_request_digest") != source.digest()
                or json.loads(member(root, "row-softmax-tiling.json")) != expected_row
            ):
                raise ValueError("softmax source precision or row binding differs from the parsed program")
        if manifest.get("target_inventory_sha256") != digest(inventory):
            raise ValueError("softmax runtime checkout differs from the selected inventory")
        selection = json.loads(member(root, "native-selection.json"))
        if (
            selection.get("request_digest") != region.request.digest()
            or selection.get("engine") != "merlin_native"
            or not manifest.get("native_check_fingerprint")
            or selection.get("check_fingerprint") != manifest["native_check_fingerprint"]
        ):
            raise ValueError("softmax selected graph differs from its source")
        execution = json.loads(member(root, "execution_plan.json"))
        sources = [node.id for node in region.request.nodes if node.effect in {"input", "constant"}]
        abi = AtlasNativeBinding.plan_boundary_abi(region.request)
        # The native allocator uses tile slots; the host ABI uses byte addresses.
        bindings = {key: DRAM_BASE + slot * TILE_BYTES for key, slot in abi["fixed_inputs"].items()}
        addresses = [bindings[key] for key in sources]
        outputs = [DRAM_BASE + slot * TILE_BYTES for slot in abi["fixed_outputs"]]
        if (
            execution.get("schema") != "atlas.native_tile_execution_plan.v8"
            or execution.get("dtype") != "bf16"
            or execution.get("shape") != [32, 32]
            or execution.get("input_addresses") != addresses
            or execution.get("output_addresses") != outputs
            or execution.get("output_physical_layouts") != ["bf16_pair_halves"] * columns
            or execution.get("input_bytes") != TILE_BYTES
            or execution.get("output_bytes") != TILE_BYTES
        ):
            raise ValueError("softmax physical boundary differs from the checked source ABI")
        if any(
            address < DRAM_BASE or address + TILE_BYTES > DRAM_BASE + dram_slots() * TILE_BYTES
            for address in addresses + outputs
        ):
            raise ValueError("softmax boundary exceeds the selected diagnostic window")
        from .resident_graph import is_staged_bf16, staged_memory_plan

        staging = execution.get("staging_workspace")
        if is_staged_bf16(region.request):
            _, staged_slots, _, _ = staged_memory_plan(region.request)
            expected_staging = {
                "address": DRAM_BASE + len(sources) * TILE_BYTES,
                "bytes": (max(staged_slots.values()) + 1 - len(sources)) * TILE_BYTES,
            }
            if staging != expected_staging:
                raise ValueError("softmax staging workspace differs from its checked allocation")
        elif staging is not None:
            raise ValueError("unexpected softmax staging workspace")
        arena_bytes = max(arena_bytes, max(addresses + outputs) + TILE_BYTES - DRAM_BASE)
        if staging is not None:
            arena_bytes = max(arena_bytes, staging["address"] + staging["bytes"] - DRAM_BASE)
        if row_loop is not None:
            addresses = [row_loop.address(address, 0) for address in addresses]
            outputs = [row_loop.address(address, 0) for address in outputs]
            arena_bytes = row_loop.end - DRAM_BASE
        constants = manifest.get("constant_inputs", [])
        expected_constants = []
        for ordinal, constant in enumerate(region.request.constants):
            path = f"constants/constant-{ordinal:03d}.bin"
            payload = pack_bf16_pair_halves(bytes.fromhex(constant.data_hex))
            expected_constants.append(
                {
                    "address": bindings[constant.node_id],
                    "bytes": TILE_BYTES,
                    "encoding": constant.encoding,
                    "node_id": constant.node_id,
                    "path": path,
                    "physical_layout": "bf16_pair_halves",
                    "sha256": digest(payload),
                }
            )
            if member(root, path) != payload:
                raise ValueError("softmax tail mask differs from its source")
            symbol = f"{name}_constant_{index}_{ordinal}"
            embedded.append(
                f"static const uint8_t {symbol}[{TILE_BYTES}] = {{" + ",".join(str(value) for value in payload) + "};"
            )
        if constants != expected_constants:
            raise ValueError("softmax declared constants differ from its source")
        program_root = row_loop_compilation if row_loop_compilation is not None else root
        binary = member(program_root, "program.bin")
        program = json.loads(member(program_root, "physical-program.json"))
        if (
            program.get("selected_rtl_revision") != RTL_REVISION
            or not binary
            or len(binary) % 4
            or binary != b"".join(row["word_u32"].to_bytes(4, "little") for row in program["instructions"])
        ):
            raise ValueError("softmax binary differs from its selected instructions")
        symbol = f"{name}_words_{index}"
        embedded.append(program_array(symbol, binary))
        device_buffers = [(address, TILE_BYTES, "READ") for address in addresses]
        if row_loop is not None:
            device_buffers = [
                (DRAM_BASE, row_loop.groups * row_loop.readonly_bytes, "READ"),
                (row_loop.writable_base, row_loop.groups * row_loop.writable_bytes, "READ_WRITE"),
            ]
        elif staging is not None:
            device_buffers.append((staging["address"], staging["bytes"], "READ_WRITE"))
        else:
            device_buffers.extend((address, TILE_BYTES, "WRITE") for address in outputs)
        buffers = ",".join(
            f"{{{address}u,{size}u,ATLAS_HOST_DEVICE_{access}}}" for address, size, access in device_buffers
        )
        masks = ",".join(f"{name}_constant_{index}_{j}" for j in range(len(constants))) or "0"
        initializers.append(
            "{"
            + f"{symbol},{len(binary) // 4}u,{region.row_start}u,"
            + "{"
            + ",".join(f"{value}u" for value in addresses[:input_tiles])
            + "},"
            + "{"
            + ",".join(f"{value}u" for value in outputs)
            + "},"
            + f"{len(constants)}u,"
            + "{"
            + masks
            + "},"
            + "{"
            + (",".join(f"{value}u" for value in addresses[input_tiles:]) or "0")
            + "},"
            + f"{len(device_buffers)}u,"
            + "{"
            + buffers
            + "}}"
        )
        files[f"programs/row-{index}.bin"] = binary
        files[f"programs/row-{index}.words.txt"] = "".join(
            f"{int.from_bytes(binary[i : i + 4], 'little'):08x}\n" for i in range(0, len(binary), 4)
        ).encode()
        images.append(
            {
                "row_start": region.row_start,
                "valid_rows": region.valid_rows,
                "request_digest": region.request.digest(),
                "native_manifest_sha256": digest(member(program_root, "manifest.json")),
                "binary_sha256": digest(binary),
                "word_count": len(binary) // 4,
                "device_layout": execution,
            }
        )
        publications.append((root, manifest, manifest_bytes, region.request.digest()))
    program_count = len(selected_rows)
    row_start = "image->start_row" if row_loop is None else "group*32u"
    input_offset = "0u" if row_loop is None else f"group*{row_loop.readonly_bytes}u"
    output_offset = "0u" if row_loop is None else f"group*{row_loop.writable_bytes}u"
    mask_check, mask_write, keep_helper = "", "", ""
    if plan.masked:
        mask_check = f"""
  if (!{name}_valid(keep,{output_bytes}u) ||
      {name}_overlap(keep,{output_bytes}u,in,{output_bytes}u) ||
      {name}_overlap(keep,{output_bytes}u,out,{output_bytes}u) ||
      {name}_overlap(keep,{output_bytes}u,ctx,sizeof(*ctx)) ||
      {name}_overlap(keep,{output_bytes}u,ctx->host,sizeof(*ctx->host)) ||
      {name}_overlap(keep,{output_bytes}u,ctx->workspace,{workspace}u)) return {name.upper()}_INVALID;
  for (size_t row=0;row<{m}u;++row) {{
    unsigned any=0;
    for (size_t col=0;col<{n}u;++col) {{
      size_t at=(row*{n}u+col)*2u;
      unsigned code=(unsigned)keep[at]|((unsigned)keep[at+1u]<<8);
      if (code!=0u && code!=0x3f80u) return {name.upper()}_INPUT_DOMAIN;
      if (!code && ((unsigned)in[at]|((unsigned)in[at+1u]<<8))!=0xc200u) return {name.upper()}_INPUT_DOMAIN;
      any|=code!=0u;
    }}
    if (!any && {int(not plan.empty_rows_zero)}u) return {name.upper()}_INPUT_DOMAIN;
  }}
"""
        keep_helper = f"""
static void {name}_pack_keep_tile(uint8_t *out,const uint8_t *in,size_t start,size_t column) {{
  for (size_t row=0;row<32u;++row) for (size_t col=0;col<32u;++col) {{
    size_t c=column*32u+col; uint16_t code=0;
    if (c<{n}u) {{
      if (start+row>={m}u) code=0x3f80u;
      else {{ size_t at=((start+row)*{n}u+c)*2u; code=(uint16_t)(in[at]|((unsigned)in[at+1u]<<8)); }}
    }}
    out[(row*32u+col)*2u]=(uint8_t)code;
    out[(row*32u+col)*2u+1u]=(uint8_t)(code>>8);
  }}
}}
"""
        mask_write = f"""
      {name}_pack_keep_tile(logical,keep,{row_start},col);
      for (size_t row=0;row<32u;++row) for (size_t c=0;c<32u;++c)
        __builtin_memcpy(physical+(c/16u)*1024u+(row*16u+c%16u)*2u,logical+(row*32u+c)*2u,2u);
      if (ctx->io.write(ctx->io.context,image->inputs[{columns}u+col]+{input_offset},physical,{TILE_BYTES}u))
        return {name.upper()}_MEMORY_ERROR;
"""
    upper = name.upper()
    load_code = f"""
    atlas_host_result result=atlas_host_ensure_loaded(ctx->host,image->words,image->count);
    if (device) *device=result;
    if (result!=ATLAS_HOST_OK) return {upper}_DEVICE_ERROR;
"""
    launch_code = f"""
    atlas_host_result result=atlas_host_start(ctx->host,image->buffers,image->buffer_count);
    if (result==ATLAS_HOST_OK) result=atlas_host_wait(ctx->host,ctx->max_polls);
    if (device) *device=result;
    if (result!=ATLAS_HOST_OK) return {upper}_DEVICE_ERROR;
"""
    read_code = f"""
    for (size_t col=0; col < {columns}u; ++col) {{
      if (ctx->io.read(ctx->io.context,image->outputs[col]+{output_offset},physical,{TILE_BYTES}u))
        return {upper}_MEMORY_ERROR;
      for (size_t row=0; row<32u; ++row) for (size_t c=0; c<32u; ++c)
        __builtin_memcpy(logical+(row*32u+c)*2u,
                         physical+(c/16u)*1024u+(row*16u+c%16u)*2u,2u);
      {name}_scatter_row_tile(ctx->workspace,logical,{row_start},col);
    }}
"""
    before_rows, after_rows = "", ""
    if row_loop is not None:
        before_rows = f"  {{ const {name}_image *image=&{name}_images[0];{load_code}  }}\n"
        after_rows = f"""
  {{ const {name}_image *image=&{name}_images[0];{launch_code}  }}
  for (size_t group=0; group < {groups}u; ++group) {{
    const {name}_image *image=&{name}_images[0];{read_code}
  }}
"""
    declarations = f"""
#define {upper}_ROWS {m}u
#define {upper}_COLUMNS {n}u
#define {upper}_BYTES {output_bytes}u
#define {upper}_WORKSPACE_BYTES {workspace}u
#define {upper}_DEVICE_ARENA_BASE {DRAM_BASE}u
#define {upper}_DEVICE_ARENA_BYTES {arena_bytes}u
#define {upper}_PROGRAM_COUNT {program_count}u
typedef enum {{ {upper}_OK=0, {upper}_INVALID, {upper}_INPUT_DOMAIN,
                {upper}_MEMORY_ERROR, {upper}_DEVICE_ERROR }} {name}_status;
typedef struct {{
  atlas_host *host;
  atlas_tile_memory_io io;
  uint8_t *workspace;
  size_t max_polls;
}} {name}_context;
/* Keep host, context, workspace and callback state alive through a timeout.
 * The platform serializes calls sharing one Atlas host. No heap allocation. */
{name}_status {name}_bind({name}_context *, atlas_host *, atlas_tile_memory_io,
                         void *workspace, size_t bytes, size_t max_polls);
/* Dense row-major BF16 little-endian input/output. All rows share the selected
 * per-row normalization across columns. Output is unchanged on any error. */
{name}_status {name}_run({name}_context *, const uint8_t *scores_bf16_le{mask_parameter},
                        uint8_t *probabilities_bf16_le, atlas_host_result *);
int {name}_dispatch(void *, const void *const *inputs, void *const *outputs);
const uint32_t *{name}_program(size_t index, size_t *word_count);
"""
    implementation = (
        "\n".join(embedded)
        + f"""
_Static_assert(SIZE_MAX >= {workspace}ULL, "softmax workspace exceeds host size_t");
typedef struct {{
  const uint32_t *words;
  size_t count, start_row;
  uintptr_t inputs[{input_tiles}], outputs[{columns}];
  size_t constant_count;
  const uint8_t *constants[1];
  uintptr_t constant_addresses[1];
  size_t buffer_count;
  atlas_host_buffer buffers[{slots}];
}} {name}_image;
static const {name}_image {name}_images[{program_count}] = {{
  {", ".join(initializers)}
}};
const uint32_t *{name}_program(size_t index, size_t *count) {{
  if (count) *count = index < {program_count}u ? {name}_images[index].count : 0;
  return index < {program_count}u ? {name}_images[index].words : 0;
}}
static int {name}_valid(const void *p, size_t bytes) {{
  return p && bytes && bytes <= UINTPTR_MAX-(uintptr_t)p;
}}
static int {name}_overlap(const void *a, size_t an, const void *b, size_t bn) {{
  return (uintptr_t)a < (uintptr_t)b+bn && (uintptr_t)b < (uintptr_t)a+an;
}}
{name}_status {name}_bind({name}_context *ctx, atlas_host *host, atlas_tile_memory_io io,
                         void *scratch, size_t bytes, size_t polls) {{
  if (!ctx || !host || !host->mmio.read32 || !host->mmio.write32 || host->active ||
      !io.write || !io.read || !polls || bytes < {workspace}u ||
      !{name}_valid(scratch,{workspace}u) ||
      {name}_overlap(scratch,{workspace}u,ctx,sizeof(*ctx)) ||
      {name}_overlap(scratch,{workspace}u,host,sizeof(*host)) ||
      {name}_overlap(ctx,sizeof(*ctx),host,sizeof(*host))) return {upper}_INVALID;
  *ctx = ({name}_context){{host,io,scratch,polls}};
  return {upper}_OK;
}}
{name}_status {name}_run({name}_context *ctx, const uint8_t *in{mask_parameter}, uint8_t *out,
                        atlas_host_result *device) {{
  if (device) *device = ATLAS_HOST_INVALID;
  if (!ctx || !ctx->host || !ctx->workspace || !ctx->io.write || !ctx->io.read || !ctx->max_polls)
    return {upper}_INVALID;
  if (ctx->host->active) {{ if (device) *device=ATLAS_HOST_BUSY; return {upper}_DEVICE_ERROR; }}
  if (!{name}_valid(in,{output_bytes}u) || !{name}_valid(out,{output_bytes}u) ||
      {name}_overlap(in,{output_bytes}u,out,{output_bytes}u) ||
      {name}_overlap(in,{output_bytes}u,ctx,sizeof(*ctx)) ||
      {name}_overlap(out,{output_bytes}u,ctx,sizeof(*ctx)) ||
      {name}_overlap(in,{output_bytes}u,ctx->host,sizeof(*ctx->host)) ||
      {name}_overlap(out,{output_bytes}u,ctx->host,sizeof(*ctx->host)) ||
      {name}_overlap(in,{output_bytes}u,ctx->workspace,{workspace}u) ||
      {name}_overlap(out,{output_bytes}u,ctx->workspace,{workspace}u)) return {upper}_INVALID;
  for (size_t at=0; at < {output_bytes}u; at+=2) {{
    unsigned code=(unsigned)in[at] | ((unsigned)in[at+1]<<8), mag=code&32767u;
    unsigned exponent=(code>>7)&255u;
    if (mag>0x4200u || (mag && (exponent<8u || exponent>132u))) return {upper}_INPUT_DOMAIN;
  }}
{mask_check}  uint8_t *logical=ctx->workspace+{output_bytes}u, *physical=logical+{TILE_BYTES}u;
{before_rows}
  for (size_t group=0; group < {groups}u; ++group) {{
    const {name}_image *image=&{name}_images[{"group" if row_loop is None else "0"}];
{load_code if row_loop is None else ""}
    for (size_t col=0; col < {columns}u; ++col) {{
      {name}_pack_row_tile(logical,in,{row_start},col);
      for (size_t row=0; row<32u; ++row) for (size_t c=0; c<32u; ++c)
        __builtin_memcpy(physical+(c/16u)*1024u+(row*16u+c%16u)*2u,
                         logical+(row*32u+c)*2u,2u);
      if (ctx->io.write(ctx->io.context,image->inputs[col]+{input_offset},physical,{TILE_BYTES}u))
        return {upper}_MEMORY_ERROR;
{mask_write}    }}
    for (size_t c=0; c<image->constant_count; ++c)
      if (ctx->io.write(ctx->io.context,image->constant_addresses[c]+{input_offset},image->constants[c],{TILE_BYTES}u))
        return {upper}_MEMORY_ERROR;
    {"{" + launch_code + "}" + read_code if row_loop is None else ""}
  }}
{after_rows}
  __builtin_memcpy(out,ctx->workspace,{output_bytes}u);
  return {upper}_OK;
}}
int {name}_dispatch(void *ctx, const void *const *inputs, void *const *outputs) {{
  if (!inputs || !outputs) return {upper}_INVALID;
  return (int){name}_run(ctx,inputs[0]{",inputs[1]" if plan.masked else ""},outputs[0],0);
}}
"""
    )
    source_interface = None
    extra_readme = ""
    if precision is not None:
        from .softmax_c_source import render_source_interface

        more_declarations, more_source, source_interface, extra_readme = render_source_interface(
            precision, name, workspace
        )
        declarations += more_declarations
        implementation += more_source
    files.update(
        {
            filename: payload.encode()
            for filename, payload in c_source_pair(
                name,
                public_headers={"atlas_host.h": htext},
                private_headers={},
                sources={
                    **host_members,
                    "row_packing.c": render_row_softmax_c(plan, name) + keep_helper,
                },
                declarations=declarations,
                implementation=implementation,
            ).items()
        }
    )
    files.update(runtime_files)
    launch_description = (
        "All row groups run in one device loop and one launch, after initializing their inputs and constants."
        if row_loop is not None
        else "Each row group initializes inputs and constants, launches, waits and reads back."
    )
    files["README.md"] = f"""# {name}

Compile `{name}.c` as C11 and include `{name}.h`. The C source embeds all
{program_count} Atlas images and tail masks. `programs/` exposes the identical binaries
and hex words for inspection. No Python, heap allocator, host main or ELF is used.

Initialize an `atlas_host` with platform MMIO/cache services, reserve the device
ranges in the manifest (`{upper}_DEVICE_ARENA_BASE`, `{upper}_DEVICE_ARENA_BYTES`), and call `{name}_bind` with
`{upper}_WORKSPACE_BYTES` scratch bytes. Call `{name}_run` with dense {m}-by-{n}
BF16 little-endian buffers, or use `{name}_dispatch` for Merlin's pointer-array
ABI. The application supplies device-memory read/write callbacks, startup,
linker script and final host build; use the application's float ABI (lp64d for
the current Shuttle bring-up).

Input scores must be signed zero or normal BF16 with magnitude at most 32 and
exponent code at least 8. The call checks this domain before writing device
memory. This selected policy uses Atlas exponential/reciprocal approximations
and tile-wise rounding; it does not promise original-model softmax accuracy.

Calls are synchronous. {launch_description}
Matching instructions remain loaded in IMEM. Normalization includes every valid
column; it is not a separate softmax per tile. Output stays unchanged on errors.
Keep host, context, scratch and callback state alive after a timeout, and finish
or reset the outstanding launch before reuse. Serialize calls sharing a host.
Standalone-core evidence does not qualify an integrated SoC or FPGA bitstream.
""".encode()
    for root, manifest, original, request_digest in publications:
        verify_native_publication(
            root,
            manifest,
            engine="merlin_native",
            request_digest=request_digest,
            target_identity=source.target_identity,
        )
        if member(root, "manifest.json") != original:
            raise ValueError("softmax compilation changed during C export")
    if row_loop_compilation is not None:
        _, final_loop_manifest = verify_row_loop(source, row_loop_compilation, target_source=target_source)
        if final_loop_manifest != loop_manifest:
            raise ValueError("row-loop compilation changed during C export")
    return {
        "schema": "atlas.row_softmax_c_package.v1",
        "status": "generated_c_package",
        "package_name": name,
        "engine": "merlin_native",
        "request_digest": source.digest(),
        "target_identity": source.target_identity,
        "tiling": plan.record(),
        "images": images,
        **({"row_loop": row_loop.record(), "launches_per_invocation": 1} if row_loop is not None else {}),
        "workspace_bytes": workspace,
        "device_arena_base": DRAM_BASE,
        "device_arena_bytes": arena_bytes,
        "entry_function": name + "_run",
        "dispatch_function": name + "_dispatch",
        "dispatch_completion": "synchronous",
        "source_members": {
            "atlas_host.h": digest(htext.encode()),
            "atlas_host.c": digest(ctext.encode()),
        },
        "startup_included": False,
        "elf_included": False,
        "executed": False,
        **runtime_record,
        **({"source_interface": source_interface} if source_interface is not None else {}),
    }, {
        **files,
        "README.md": files["README.md"]
        + extra_readme.encode()
        + (
            b"\nLink `runtime/atlas_host.c` once for the application, alongside every region C file.\n"
            if runtime_files
            else b""
        ),
    }
