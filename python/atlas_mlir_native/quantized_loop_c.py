"""Checked C packaging for one on-device already-quantized contraction."""
from __future__ import annotations

from hashlib import sha256
import json
from math import prod
from pathlib import Path

from merlin.semantic_compiler.model import KernelRequest
from merlin.semantic_compiler.target_binding import verify_native_publication

from .already_quantized import prepare_source
from .binding import TARGET_IDENTITY
from .quantized_loop import execution_layout
from .quantized_loop_sizing import selected_route


def prepare_c_package(source, compiled, name, *, target_source):
    """Export one checked contraction image and preserve dense source inputs."""
    if (Path(compiled)/'row-block-manifest.json').is_file():
        from .quantized_row_blocks_c import prepare_c_package as prepare_rows
        return prepare_rows(source, compiled, name, target_source=target_source)
    from merlin.compile.c_sources import c_source_pair
    from .c_runtime import program_array, runtime_headers, runtime_members
    plan = prepare_source(source)
    if selected_route(plan,compiled) == 'k32_tiles':
        if plan.operand_permutation is not None:
            raise ValueError('overlarge contraction must retain its operand permutation')
        if plan.narrowing_input_id is not None:
            raise ValueError('overlarge contraction must retain its separate narrowing cast')
        if getattr(plan, 'epilogue_precision', None) is not None:
            raise ValueError('overlarge contraction must retain its separate epilogue')
        from .already_quantized import prepare_c_package as prepare_tiles
        record, files = prepare_tiles(source, compiled, name, target_source=target_source)
        record = {**record, 'execution_route': 'bounded_k32_tile_dispatch',
                  'launches_per_invocation': prod(plan.batch_shape) * ((plan.m + 31)//32) *
                      ((plan.n + 31)//32) * ((plan.k + 31)//32),
                  'restoration_placement': 'host; original exact integer exponent restoration',
                  'counted_loop_declined': json.loads((Path(compiled)/'quantized-route.json').read_text())['reason']}
        return record, files
    compiled,target_source = Path(compiled),Path(target_source)
    if json.loads((compiled/"already-quantized-plan.json").read_text()) != plan.record():
        raise ValueError("already-quantized source plan differs from its publication")
    manifest = json.loads((compiled/"manifest.json").read_text())
    verify_native_publication(compiled,manifest,engine="merlin_native",request_digest=source.digest(),target_identity=TARGET_IDENTITY)
    if KernelRequest.from_record(json.loads((compiled/"source-request.json").read_text())).digest() != source.digest():
        raise ValueError("already-quantized native source differs from its publication")
    execution = json.loads((compiled/"execution_plan.json").read_text())
    from .exponent_page_cache import recorded_enabled
    if execution != execution_layout(plan, operand_reuse=execution.get("operand_reuse_requested", "ordered"),
                                     exponent_page_cache=recorded_enabled(execution)):
        raise ValueError("already-quantized loop execution ABI differs from its publication")
    if sha256((target_source/"docs/selected-variant-inventory.json").read_bytes()).hexdigest() != manifest["target_inventory_sha256"]:
        raise ValueError("already-quantized runtime differs from its checked target inventory")
    binary = (compiled/"program.bin").read_bytes()
    physical = json.loads((compiled/"physical-program.json").read_text())
    if binary != b"".join(row["word_u32"].to_bytes(4,"little") for row in physical["instructions"]):
        raise ValueError("already-quantized binary differs from checked instruction words")
    m,n,k,batches = plan.m,plan.n,plan.k,prod(plan.batch_shape)
    inputs = [source.node(key) for key in plan.input_ids]
    sizes = [prod(node.type.shape)*{"i8":1,"ui8":1,"u8":1,"fp8_e4m3":1,"i32":4,"si32":4,"bf16":2,"f32":4}[node.type.dtype] for node in inputs]
    a,b = (source.node(key) for key in plan.operand_ids)
    ia,ib = (plan.input_ids.index(key) for key in plan.physical_operand_ids)
    lhs_index = f'(batch*{m}u+mt*32u+row)*{k}u+kt*32u+c'
    rhs_index = f'(batch*{n}u+nt*32u+row)*{k}u+kt*32u+c'
    if plan.operand_permutation is not None:
        logical, _, _ = plan.operand_permutation
        if logical == plan.operand_ids[0]:
            lhs_index = f'(batch*{k}u+kt*32u+c)*{m}u+mt*32u+row'
        else:
            rhs_index = f'(batch*{k}u+kt*32u+c)*{n}u+nt*32u+row'
    iea,ieb = (plan.input_ids.index(key) for key in plan.exponent_ids)
    output_bytes = batches*m*n*(2 if plan.output_dtype=="bf16" else 4)
    mc,nc,kc = ((value+31)//32 for value in (m,n,k))
    rhs_batches = batches if b.type.shape[:-2] else 1
    lhs_bytes,rhs_bytes = execution["input_buffer_bytes"][:2]
    device_output_bytes = execution["output_bytes"]
    exponent_bytes = execution['exponent_buffer_bytes']
    exponent_offsets = [lhs_bytes+rhs_bytes, lhs_bytes+rhs_bytes+exponent_bytes[0]]
    output_offset = exponent_offsets[1]+exponent_bytes[1]
    status_offset = output_offset+device_output_bytes
    extra_offset = status_offset+execution['status_bytes']
    extra_bytes = 0
    extra_encoder, extra_packing = '', ''
    if plan.epilogue_precision is not None:
        from merlin.compile.c_conversions import render_f32_array_c, render_bf16_codes_c
        if (len(execution['input_addresses']) != 3 or len(execution['input_buffer_bytes']) != 3
                or execution['input_addresses'][2] != execution['status_address']+execution['status_bytes']
                or execution['input_buffer_bytes'][2] != device_output_bytes):
            raise ValueError('quantized epilogue requires one complete readonly BF16 buffer after output and status')
        extra_bytes = execution['input_buffer_bytes'][2]
        extra_index = plan.input_ids.index(plan.epilogue_other_id)
        spec = plan.epilogue_precision.operand_specs[1-plan.epilogue_contract_operand]
        encode = render_bf16_codes_c if plan.epilogue_other_dtype == 'bf16' else render_f32_array_c
        extra_encoder = encode(spec,batches*m*n,name+'_encode_extra')
        extra_packing = f"""
  /* Preserve the original extra-operand conversion and code-domain guard. */
  if ({name}_encode_extra(inputs[{extra_index}],ctx->workspace)) return 2;
  __builtin_memset(region->workspace+{extra_offset}u,0,{extra_bytes}u);
  for (size_t batch=0;batch<{batches}u;++batch) for (size_t mt=0;mt<{mc}u;++mt)
    for (size_t nt=0;nt<{nc}u;++nt)
      for (size_t row=0;row<32u && mt*32u+row<{m}u;++row)
        for (size_t c=0;c<32u && nt*32u+c<{n}u;++c) {{
          size_t tile=((batch*{mc}u+mt)*{nc}u+nt)*2048u;
          size_t physical=tile+(c/16u)*1024u+(row*16u+c%16u)*2u;
          size_t logical=((batch*{m}u+mt*32u+row)*{n}u+nt*32u+c)*2u;
          __builtin_memcpy(region->workspace+{extra_offset}u+physical,ctx->workspace+logical,2u);
        }}
"""
    raw,converted,count = extra_offset+extra_bytes,output_bytes,len(inputs)
    runtime = target_source/"runtime"
    public = runtime_headers(runtime)
    host_members,runtime_files,runtime_record = runtime_members(public,(runtime/"atlas_host.c").read_text(),"separate")
    upper=name.upper()
    declarations=f"""
#define {upper}_WORKSPACE_BYTES {raw}u
#define {upper}_SOURCE_WORKSPACE_BYTES {converted}u
#define {upper}_SOURCE_INPUT_COUNT {count}u
typedef struct {{ atlas_host *host; atlas_tile_memory_io io; uint8_t *workspace; size_t max_polls; }} {name}_context;
typedef struct {{ {name}_context *region; uint8_t *workspace; }} {name}_source_context;
int {name}_bind({name}_context *,atlas_host *,atlas_tile_memory_io,void *,size_t,size_t);
int {name}_source_bind({name}_source_context *,{name}_context *,void *,size_t);
int {name}_source_dispatch(void *,const void *const *,void *const *);
"""
    arrays=program_array(name+"_words",binary)
    addresses=execution["input_addresses"]
    out=execution["output_addresses"][0]
    write_inputs = [(addresses[0], 0, lhs_bytes), (addresses[1], lhs_bytes, rhs_bytes),
                    *zip(execution['exponent_addresses'], exponent_offsets, exponent_bytes, strict=True)]
    if extra_bytes:
        write_inputs.append((addresses[2],extra_offset,extra_bytes))
    writes = '\n'.join(f'  if (region->io.write(region->io.context,{address}u,region->workspace+{offset}u,{size}u)) return 3;'
                       for address, offset, size in write_inputs)
    buffers = ',\n      '.join(f'{{{address}u,{size}u,ATLAS_HOST_DEVICE_READ}}' for address, _, size in write_inputs)
    buffers += f',\n      {{{out}u,{device_output_bytes}u,ATLAS_HOST_DEVICE_WRITE}},\n      {{{execution["status_address"]}u,{execution["status_bytes"]}u,ATLAS_HOST_DEVICE_WRITE}}'
    launch=f"""
  status=atlas_host_ensure_loaded(region->host,{name}_words,{len(binary)//4}u);
  if (status!=ATLAS_HOST_OK) {{ if (device) *device=status; return 4; }}
{writes}
  atlas_host_buffer buffers[]={{{buffers}}};
  status=atlas_host_start(region->host,buffers,{len(write_inputs)+2}u);
  if (status==ATLAS_HOST_OK) status=atlas_host_wait(region->host,region->max_polls);
  if (device) *device=status;
  if (status!=ATLAS_HOST_OK) return 4;
  if (region->io.read(region->io.context,{execution['status_address']}u,region->workspace+{status_offset}u,4u)) return 3;
  if (region->workspace[{status_offset}u] || region->workspace[{status_offset+1}u] ||
      region->workspace[{status_offset+2}u] || region->workspace[{status_offset+3}u]) return 2;
  if (region->io.read(region->io.context,{out}u,region->workspace+{output_offset}u,{device_output_bytes}u)) return 3;
"""
    pack_exponents = []
    for index, offset, geometry in zip((iea, ieb), exponent_offsets, execution['exponent_layouts'], strict=True):
        if geometry['mode'] == 'row':
            pack_exponents.append(f"""
  for (size_t batch=0;batch<{geometry['batches']}u;++batch)
    for (size_t row=0;row<{geometry['rows']}u;++row)
      __builtin_memcpy(region->workspace+{offset}u+(batch*{geometry['tiles']}u+row/32u)*2048u+(row%32u)*4u,
                       (const uint8_t *)inputs[{index}]+(batch*{geometry['rows']}u+row)*4u,4u);
""")
        else:
            pack_exponents.append(f"""
  for (size_t page=0;page<{geometry['pages']}u;++page)
    __builtin_memcpy(region->workspace+{offset}u+page*2048u,(const uint8_t *)inputs[{index}]+page*4u,4u);
""")
    exponent_packing = ''.join(pack_exponents)
    storing = ("uint16_t value=(uint16_t)word; __builtin_memcpy(ctx->workspace+index*2u,&value,2u);" if plan.output_dtype=="bf16"
        else "uint32_t value=word<<16; __builtin_memcpy(ctx->workspace+index*4u,&value,4u);")
    implementation=extra_encoder+arrays+f"""
static int {name}_range(const void *p,size_t n) {{ return p && n && (uintptr_t)p<=UINTPTR_MAX-n; }}
static int {name}_overlap(const void *a,size_t an,const void *b,size_t bn) {{ return (uintptr_t)a<(uintptr_t)b+bn && (uintptr_t)b<(uintptr_t)a+an; }}
int {name}_bind({name}_context *ctx,atlas_host *host,atlas_tile_memory_io io,void *scratch,size_t bytes,size_t polls) {{
  if (!ctx || !host || host->active || !host->mmio.read32 || !host->mmio.write32 || !io.read || !io.write || !polls ||
      bytes<{raw}u || !{name}_range(scratch,{raw}u) || {name}_overlap(scratch,{raw}u,ctx,sizeof(*ctx)) ||
      {name}_overlap(scratch,{raw}u,host,sizeof(*host))) return 1;
  if ({name}_overlap(ctx,sizeof(*ctx),host,sizeof(*host))) return 1;
  *ctx=({name}_context){{host,io,scratch,polls}}; return 0;
}}
int {name}_source_bind({name}_source_context *ctx,{name}_context *region,void *scratch,size_t bytes) {{
  if (!ctx || !region || !region->host || region->host->active || !region->workspace || bytes<{converted}u ||
      !{name}_range(scratch,{converted}u) || {name}_overlap(scratch,{converted}u,ctx,sizeof(*ctx)) ||
      {name}_overlap(scratch,{converted}u,region,sizeof(*region)) ||
      {name}_overlap(scratch,{converted}u,region->host,sizeof(*region->host)) ||
      {name}_overlap(scratch,{converted}u,region->workspace,{raw}u) ||
      {name}_overlap(ctx,sizeof(*ctx),region,sizeof(*region)) ||
      {name}_overlap(ctx,sizeof(*ctx),region->host,sizeof(*region->host)) ||
      {name}_overlap(ctx,sizeof(*ctx),region->workspace,{raw}u)) return 1;
  *ctx=({name}_source_context){{region,scratch}}; return 0;
}}
int {name}_source_dispatch(void *context,const void *const *inputs,void *const *outputs) {{
  {name}_source_context *ctx=context;
  if (!ctx || !ctx->region || !ctx->region->host || !ctx->region->workspace || !ctx->workspace || !inputs || !outputs || !outputs[0]) return 1;
  {name}_context *region=ctx->region;
  atlas_host_result *device=0,status=ATLAS_HOST_INVALID;
  if (region->host->active) return 4;
  const size_t sizes[{count}]={{{','.join(str(size)+'u' for size in sizes)}}};
  void *out=outputs[0];
  const void *reserved[]={{ctx,region,region->host,ctx->workspace,region->workspace,inputs,outputs}};
  const size_t reserved_sizes[]={{sizeof(*ctx),sizeof(*region),sizeof(*region->host),{converted}u,{raw}u,{count}u*sizeof(*inputs),sizeof(*outputs)}};
  if (!{name}_range(out,{output_bytes}u)) return 1;
  for (size_t r=0;r<7u;++r) {{
    if (!{name}_range(reserved[r],reserved_sizes[r]) || {name}_overlap(out,{output_bytes}u,reserved[r],reserved_sizes[r])) return 1;
    for (size_t i=0;i<{count}u;++i) if (!{name}_range(inputs[i],sizes[i]) ||
        (r<5u && {name}_overlap(inputs[i],sizes[i],reserved[r],reserved_sizes[r])) ||
        {name}_overlap(inputs[i],sizes[i],out,{output_bytes}u)) return 1;
  }}
  for (size_t table=5u;table<7u;++table) for (size_t r=0;r<5u;++r)
    if ({name}_overlap(reserved[table],reserved_sizes[table],reserved[r],reserved_sizes[r])) return 1;
  const uint8_t *a=inputs[{ia}],*b=inputs[{ib}];
  for (size_t which=0;which<2u;++which) {{
    const uint8_t *v=which?b:a; size_t size=which?{prod(b.type.shape)}u:{prod(a.type.shape)}u;
    for (size_t i=0;i<size;++i) if (!(v[i]==0u || (v[i]>=8u && v[i]<=126u) || (v[i]>=136u && v[i]<=254u))) return 2;
  }}
  /* Pack complete logical operands once, preserving every admitted FP8 code. */
  __builtin_memset(region->workspace,0,{output_offset}u);
  for (size_t batch=0;batch<{batches}u;++batch) for (size_t mt=0;mt<{mc}u;++mt)
    for (size_t kt=0;kt<{kc}u;++kt) for (size_t row=0;row<32u && mt*32u+row<{m}u;++row)
      for (size_t c=0;c<32u && kt*32u+c<{k}u;++c)
        region->workspace[((batch*{mc}u+mt)*{kc}u+kt)*1024u+row*32u+c]=a[{lhs_index}];
  for (size_t batch=0;batch<{rhs_batches}u;++batch) for (size_t nt=0;nt<{nc}u;++nt)
    for (size_t kt=0;kt<{kc}u;++kt) for (size_t row=0;row<32u && nt*32u+row<{n}u;++row)
      for (size_t c=0;c<32u && kt*32u+c<{k}u;++c)
        region->workspace[{lhs_bytes}u+((batch*{nc}u+nt)*{kc}u+kt)*1024u+row*32u+c]=b[{rhs_index}];
{exponent_packing}
{extra_packing}
{launch}
  for (size_t batch=0;batch<{batches}u;++batch) for (size_t first_m=0;first_m<{m}u;first_m+=32u)
    for (size_t first_n=0;first_n<{n}u;first_n+=32u) {{
      size_t tile=((batch*{mc}u+first_m/32u)*{nc}u+first_n/32u)*2048u;
      for (size_t row=0;row<32u && first_m+row<{m}u;++row) for (size_t c=0;c<32u && first_n+c<{n}u;++c) {{
        size_t at=tile+(c/16u)*1024u+(row*16u+c%16u)*2u;
        uint32_t word=(uint32_t)region->workspace[{output_offset}u+at]|((uint32_t)region->workspace[{output_offset+1}u+at]<<8);
        size_t index=(batch*{m}u+first_m+row)*{n}u+first_n+c;
        {storing}
      }}
    }}
  __builtin_memcpy(out,ctx->workspace,{output_bytes}u); return 0;
}}
"""
    files={key:value.encode() for key,value in c_source_pair(name,public_headers=public,private_headers={},
        sources={**host_members,"quantized_dispatch.c":implementation},declarations=declarations).items()}
    files.update(runtime_files)
    record=dict(kind="already_quantized",schema="atlas.already_quantized_c_package.v1",package_name=name,
        execution_route='checked_counted_image',launches_per_invocation=1,restoration_placement='device',
        context_type=name+"_context",source_context_type=name+"_source_context",binding_function=name+"_bind",
        source_binding_function=name+"_source_bind",workspace_bytes=raw,source_workspace_bytes=converted,
        readout_scale_domain="device admitted signed exponent sums in [-133,127]; finite normal BF16 or positive zero",
        rejection_status=dict(address=execution["status_address"],bytes=4,success=0,rejection=1),
        precision_plan=plan.record(),input_node_ids=list(plan.input_ids),output_node_ids=[plan.output_id],
        source_interface=dict(workspace_bytes=converted,failure_output="unchanged",completion="synchronous"),**runtime_record)
    if plan.epilogue_precision is not None:
        record.update(kind='quantized_epilogue',schema='atlas.quantized_epilogue_c_package.v1',
            epilogue=dict(precision=plan.epilogue_precision.record(),contract_operand=plan.epilogue_contract_operand,
                          other_id=plan.epilogue_other_id,other_dtype=plan.epilogue_other_dtype,
                          extra_address=execution['input_addresses'][2],extra_bytes=extra_bytes,
                          extra_layout='contraction batch/M-tile/N-tile BF16 pair halves; positive-zero padding',
                          failure_output='unchanged; shared device status checked before output transfer'))
    if plan.narrowing_input_id is not None:
        record.update(kind='quantized_narrowing', schema='atlas.quantized_narrowing_c_package.v1',
                      narrowing=plan.record()['narrowing'])
    return record,files
