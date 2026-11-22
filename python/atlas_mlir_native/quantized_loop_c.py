"""Checked C packaging for one on-device already-quantized contraction."""
from __future__ import annotations

from hashlib import sha256
import json
from math import prod
from pathlib import Path

from merlin.compiler.semantic.model import KernelRequest
from merlin.compiler.semantic.target_binding import verify_native_publication

from .already_quantized import prepare_source
from .binding import TARGET_IDENTITY
from .exponent_restoration import EXPONENT_PAGE_PAYLOAD_BYTES
from .quantized_loop import execution_layout
from .quantized_loop_sizing import selected_route
from .shared_dispatch_c import guarded


MATMUL_TYPES = guarded("matmul_types",
    "typedef struct { atlas_region_context *region; uint8_t *workspace; } atlas_matmul_source_context;\n")
# One checked counted contraction launch. Per-region constants live in the plan;
# the checks, packing, launch, and readout follow the region ABI in this order.
# A streamed plan packs, writes, and reads back one tile at a time through one tile of host scratch.
MATMUL_DISPATCH = guarded("matmul_dispatch", """typedef struct { uint32_t mode,input,offset,batches,rows,tiles,pages; } atlas_exponent_pack;
typedef struct { uint32_t address,offset,bytes,resident; } atlas_transfer;
typedef struct {
  const uint32_t *words; uint32_t word_count,raw,converted,output_bytes,count,sizes[8],resident;
  uint32_t ia,ib,lhs_columns,rhs_columns,f32_output,a_elements,b_elements;
  uint32_t batches,rhs_batches,m,n,k,mc,nc,kc,rhs_offset,pack_bytes;
  atlas_exponent_pack exponents[2]; /* mode 0: resident, 1: rows, 2: pages */
  int (*extra)(const void *,uint8_t *); uint32_t extra_input,extra_offset,extra_bytes;
  uint32_t read_count; atlas_transfer reads[5]; /* device-read buffers in launch order; resident ones are not written */
  uint32_t status_address,status_bytes,status_offset,output_address,output_offset,device_output_bytes,streamed;
} atlas_matmul_plan;
static __attribute__((unused)) int atlas_matmul_bind(atlas_region_context *ctx,atlas_host *host,const atlas_tile_memory_io *io,void *scratch,
                             size_t bytes,size_t polls,const atlas_matmul_plan *plan) {
  size_t raw=plan->raw;
  if (!ctx || !host || host->active || !host->mmio.read32 || !host->mmio.write32 || !io->read || !io->write || !polls ||
      bytes<raw || !atlas_span_valid(scratch,raw) || atlas_overlap(scratch,raw,ctx,sizeof(*ctx)) ||
      atlas_overlap(scratch,raw,host,sizeof(*host))) return 1;
  if (atlas_overlap(ctx,sizeof(*ctx),host,sizeof(*host))) return 1;
  *ctx=(atlas_region_context){host,*io,scratch,polls}; return 0;
}
static __attribute__((unused)) int atlas_matmul_source_bind(atlas_matmul_source_context *ctx,atlas_region_context *region,void *scratch,size_t bytes,
                                    const atlas_matmul_plan *plan) {
  size_t converted=plan->converted,raw=plan->raw;
  if (!ctx || !region || !region->host || region->host->active || !region->workspace || bytes<converted ||
      !atlas_span_valid(scratch,converted) || atlas_overlap(scratch,converted,ctx,sizeof(*ctx)) ||
      atlas_overlap(scratch,converted,region,sizeof(*region)) ||
      atlas_overlap(scratch,converted,region->host,sizeof(*region->host)) ||
      atlas_overlap(scratch,converted,region->workspace,raw) ||
      atlas_overlap(ctx,sizeof(*ctx),region,sizeof(*region)) ||
      atlas_overlap(ctx,sizeof(*ctx),region->host,sizeof(*region->host)) ||
      atlas_overlap(ctx,sizeof(*ctx),region->workspace,raw)) return 1;
  *ctx=(atlas_matmul_source_context){region,scratch}; return 0;
}
static __attribute__((unused)) int atlas_matmul_dispatch(const atlas_matmul_plan *p,void *context,const void *const *inputs,void *const *outputs) {
  atlas_matmul_source_context *ctx=context;
  if (!ctx || !ctx->region || !ctx->region->host || !ctx->region->workspace || !ctx->workspace || !inputs || !outputs || !outputs[0]) return 1;
  atlas_region_context *region=ctx->region;
  if (region->host->active) return 4;
  void *out=outputs[0];
  const void *reserved[]={ctx,region,region->host,ctx->workspace,region->workspace,inputs,outputs};
  const size_t reserved_sizes[]={sizeof(*ctx),sizeof(*region),sizeof(*region->host),p->converted,p->raw,p->count*sizeof(*inputs),sizeof(*outputs)};
  if (!atlas_span_valid(out,p->output_bytes)) return 1;
  for (size_t r=0;r<7u;++r) {
    if (!atlas_span_valid(reserved[r],reserved_sizes[r]) || atlas_overlap(out,p->output_bytes,reserved[r],reserved_sizes[r])) return 1;
    for (size_t i=0;i<p->count;++i) if (!(p->resident>>i&1u) && (!atlas_span_valid(inputs[i],p->sizes[i]) ||
        (r<5u && atlas_overlap(inputs[i],p->sizes[i],reserved[r],reserved_sizes[r])) ||
        atlas_overlap(inputs[i],p->sizes[i],out,p->output_bytes))) return 1;
  }
  for (size_t table=5u;table<7u;++table) for (size_t r=0;r<5u;++r)
    if (atlas_overlap(reserved[table],reserved_sizes[table],reserved[r],reserved_sizes[r])) return 1;
  const uint8_t *a=inputs[p->ia],*b=inputs[p->ib];
  for (size_t which=0;which<2u;++which) {
    if (p->resident>>(which?p->ib:p->ia)&1u) continue;
    const uint8_t *v=which?b:a; size_t size=which?p->b_elements:p->a_elements;
    for (size_t i=0;i<size;++i) if (!(v[i]==0u || (v[i]>=8u && v[i]<=126u) || (v[i]>=136u && v[i]<=254u))) return 2;
  }
  /* Pack complete logical operands once, or one tile at a time when streamed, preserving every admitted FP8 code. */
  uint8_t *w=region->workspace;
  size_t m=p->m,n=p->n,k=p->k,mc=p->mc,nc=p->nc,kc=p->kc;
  const atlas_tile_memory_io *io=&region->io;
  /* Preserve the original extra-operand conversion and code-domain guard, before the image loads. */
  if (p->extra && p->extra(inputs[p->extra_input],ctx->workspace)) return 2;
  atlas_host_result status=atlas_host_ensure_loaded(region->host,p->words,p->word_count);
  if (status!=ATLAS_HOST_OK) return 4;
  if (!p->streamed) for (size_t i=0;i<2u;++i)
    if (!p->reads[i].resident) __builtin_memset(w+p->reads[i].offset,0,p->reads[i].bytes);
  if (!(p->resident>>p->ia&1u))
    for (size_t batch=0;batch<p->batches;++batch) for (size_t mt=0;mt<mc;++mt) for (size_t kt=0;kt<kc;++kt) {
      size_t tile=((batch*mc+mt)*kc+kt)*1024u; uint8_t *t=p->streamed?w:w+tile;
      if (p->streamed) __builtin_memset(w,0,1024u);
      for (size_t row=0;row<32u && mt*32u+row<m;++row) for (size_t c=0;c<32u && kt*32u+c<k;++c)
        t[row*32u+c]=a[p->lhs_columns?(batch*k+kt*32u+c)*m+mt*32u+row:(batch*m+mt*32u+row)*k+kt*32u+c];
      if (p->streamed && io->write(io->context,p->reads[0].address+tile,w,1024u)) return 3;
    }
  if (!(p->resident>>p->ib&1u))
    for (size_t batch=0;batch<p->rhs_batches;++batch) for (size_t nt=0;nt<nc;++nt) for (size_t kt=0;kt<kc;++kt) {
      size_t tile=((batch*nc+nt)*kc+kt)*1024u; uint8_t *t=p->streamed?w:w+p->rhs_offset+tile;
      if (p->streamed) __builtin_memset(w,0,1024u);
      for (size_t row=0;row<32u && nt*32u+row<n;++row) for (size_t c=0;c<32u && kt*32u+c<k;++c)
        t[row*32u+c]=b[p->rhs_columns?(batch*k+kt*32u+c)*n+nt*32u+row:(batch*n+nt*32u+row)*k+kt*32u+c];
      if (p->streamed && io->write(io->context,p->reads[1].address+tile,w,1024u)) return 3;
    }
  for (size_t e=0;e<2u;++e) {
    const atlas_exponent_pack *x=&p->exponents[e];
    const uint8_t *source=inputs[x->input];
    size_t pages=x->mode==1u?x->batches*x->tiles:x->mode==2u?x->pages:0u;
    for (size_t page=0;page<pages;++page) {
      uint8_t *t=p->streamed?w:w+x->offset+page*2048u;
      __builtin_memset(t,0,@EXPONENT_PAGE_PAYLOAD_BYTES@u);
      if (x->mode==2u) __builtin_memcpy(t,source+page*4u,4u);
      else for (size_t row=page%x->tiles*32u;row<page%x->tiles*32u+32u && row<x->rows;++row)
        __builtin_memcpy(t+(row%32u)*4u,source+(page/x->tiles*x->rows+row)*4u,4u);
      if (p->streamed && io->write(io->context,p->reads[2u+e].address+page*2048u,w,@EXPONENT_PAGE_PAYLOAD_BYTES@u)) return 3;
    }
  }
  if (p->extra) {
    if (!p->streamed) __builtin_memset(w+p->extra_offset,0,p->extra_bytes);
    for (size_t batch=0;batch<p->batches;++batch) for (size_t mt=0;mt<mc;++mt) for (size_t nt=0;nt<nc;++nt) {
      size_t tile=((batch*mc+mt)*nc+nt)*2048u; uint8_t *t=p->streamed?w:w+p->extra_offset+tile;
      if (p->streamed) __builtin_memset(w,0,2048u);
      for (size_t row=0;row<32u && mt*32u+row<m;++row) for (size_t c=0;c<32u && nt*32u+c<n;++c)
        __builtin_memcpy(t+(c/16u)*1024u+(row*16u+c%16u)*2u,ctx->workspace+((batch*m+mt*32u+row)*n+nt*32u+c)*2u,2u);
      if (p->streamed && io->write(io->context,p->reads[4].address+tile,w,2048u)) return 3;
    }
  }
  atlas_host_buffer buffers[7];
  for (size_t i=0;i<p->read_count;++i) {
    const atlas_transfer *t=&p->reads[i];
    if (!p->streamed && !t->resident) {
      if (i==2u || i==3u) {
        for (size_t page=0;page<t->bytes;page+=2048u)
          if (io->write(io->context,t->address+page,w+t->offset+page,@EXPONENT_PAGE_PAYLOAD_BYTES@u)) return 3;
      } else if (io->write(io->context,t->address,w+t->offset,t->bytes)) return 3;
    }
    buffers[i]=(atlas_host_buffer){t->address,t->bytes,ATLAS_HOST_DEVICE_READ};
  }
  buffers[p->read_count]=(atlas_host_buffer){p->output_address,p->device_output_bytes,ATLAS_HOST_DEVICE_WRITE};
  buffers[p->read_count+1u]=(atlas_host_buffer){p->status_address,p->status_bytes,ATLAS_HOST_DEVICE_WRITE};
  status=atlas_host_start(region->host,buffers,p->read_count+2u);
  if (status==ATLAS_HOST_OK) status=atlas_host_wait(region->host,region->max_polls);
  if (status!=ATLAS_HOST_OK) return 4;
  if (io->read(io->context,p->status_address,w+p->status_offset,4u)) return 3;
  if (w[p->status_offset] || w[p->status_offset+1u] || w[p->status_offset+2u] || w[p->status_offset+3u]) return 2;
  if (!p->streamed && io->read(io->context,p->output_address,w+p->output_offset,p->device_output_bytes)) return 3;
  for (size_t batch=0;batch<p->batches;++batch) for (size_t first_m=0;first_m<m;first_m+=32u)
    for (size_t first_n=0;first_n<n;first_n+=32u) {
      size_t tile=((batch*mc+first_m/32u)*nc+first_n/32u)*2048u;
      if (p->streamed && io->read(io->context,p->output_address+tile,w,2048u)) return 3;
      for (size_t row=0;row<32u && first_m+row<m;++row) for (size_t c=0;c<32u && first_n+c<n;++c) {
        size_t at=(p->streamed?0u:tile)+(c/16u)*1024u+(row*16u+c%16u)*2u;
        uint32_t word=(uint32_t)w[p->output_offset+at]|((uint32_t)w[p->output_offset+1u+at]<<8);
        size_t index=(batch*m+first_m+row)*n+first_n+c;
        if (p->f32_output) { uint32_t value=word<<16; __builtin_memcpy(ctx->workspace+index*4u,&value,4u); }
        else { uint16_t value=(uint16_t)word; __builtin_memcpy(ctx->workspace+index*2u,&value,2u); }
      }
    }
  __builtin_memcpy(out,ctx->workspace,p->output_bytes); return 0;
}
""".replace('@EXPONENT_PAGE_PAYLOAD_BYTES@',str(EXPONENT_PAGE_PAYLOAD_BYTES)))


def _resident_dispatch_helpers():
    # Keep every ordinary package byte-identical. The separate resident ABI
    # retains the checked packing/readout routine and changes only image loading.
    text = '\n'.join(MATMUL_DISPATCH.splitlines()[2:-1])+'\n'
    text = text.replace('atlas_matmul_', 'atlas_resident_matmul_')
    text = text.replace('atlas_exponent_pack', 'atlas_resident_exponent_pack').replace('atlas_transfer', 'atlas_resident_transfer')
    text = text.replace('{region,scratch};', '{region,scratch,0};')
    # Each region binds and loads its own image by a direct call before dispatch, and passes the status, so the
    # generated call graph has no function pointer that a static stack bound would have to follow.
    text = text.replace('void *const *outputs) {\n',
        'void *const *outputs,atlas_host_result loaded) {\n')
    text = text.replace('atlas_host_ensure_loaded(region->host,p->words,p->word_count)', 'loaded')
    return guarded('resident_matmul_dispatch', text)


RESIDENT_MATMUL_TYPES = guarded('resident_matmul_types',
    'typedef struct { atlas_region_context *region; uint8_t *workspace; uint32_t binding; } atlas_resident_matmul_source_context;\n')


def _resident_runtime_table(state):
    name = state['name']
    rows = [state['image'].binding_addresses(bound) for bound in state['bindings']]
    count = len(rows[0])
    if not count or any(len(row) != count for row in rows):
        raise ValueError('resident runtime address table differs')
    values = ',\n'.join('  {'+','.join(f'{address}u' for address in row)+'}' for row in rows)
    return f"""
/* atlas resident binding table begin */
static const uint32_t {name}_resident_addresses[{len(rows)}][{count}]={{
{values}
}};
static const uint32_t {name}_resident_count={len(rows)}u;
static atlas_host_result {name}_resident_load(atlas_host *host,uint8_t *workspace,uint32_t index) {{
  if (index>={name}_resident_count) return ATLAS_HOST_INVALID;
  return {name}_resident_image_bind(host,(uint32_t *)(void *)(workspace+{state['image_offset']}u),
    {name.upper()}_RESIDENT_IMAGE_WORD_COUNT,{name}_resident_addresses[index],{count}u);
}}
/* atlas resident binding table end */
"""


def _resident_dispatch(state):
    name = state['name']
    ports = [row['node_id'] for row in state['bindings'][0].execution_plan['resident_bindings']]
    overrides = '\n  '.join(f'plan.reads[{i}].address={name}_resident_addresses[ctx->binding][{ports.index(key)}];'
                            for i, key in enumerate(state['read_ids']) if key in state['resident_ids'])
    return f"""atlas_resident_matmul_source_context *ctx=context;
  if (!ctx || ctx->binding>={name}_resident_count) return 1;
  atlas_resident_matmul_plan plan={name}_plan;
  {overrides}
  if (!ctx->region || !ctx->region->host || !ctx->workspace) return 1;
  return atlas_resident_matmul_dispatch(&plan,context,inputs,outputs,
    {name}_resident_load(ctx->region->host,ctx->workspace,ctx->binding));"""


def _resident_runtime_receipt(state):
    rows = [dict(binding=bound.receipt, execution_plan=bound.execution_plan,
                 addresses=list(state['image'].binding_addresses(bound))) for bound in state['bindings']]
    record = dict(schema='atlas.resident_loop_c_runtime.v1', image=state['source_receipt'],
                  trip_bindings=rows, mutable_image_storage='source_context_workspace_tail',
                  load_convention='forced_load_each_dispatch')
    record['table_sha256'] = sha256(json.dumps(rows, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    return record


def prepare_resident_c_package(source, compiled, name, *, target_source):
    """Keep private pending bindings until Merlin finalizes the shared body."""
    from .resident_loop_image import ResidentLoopImage

    image = ResidentLoopImage(source, compiled, target_source=target_source)
    state = dict(image=image, bindings=[image.bind(source)], sources=[source],
                 representative_source=source, compiled_root=Path(compiled), target_source=Path(target_source))
    record, files = prepare_c_package(source, compiled, name, target_source=target_source,
                                     _resident_runtime=state)
    state.update(record=record, files=files)
    return record, files, state


def finalize_resident_c_package(state, *, target_source=None):
    """Publish one immutable template and the authenticated per-trip address table."""
    from .resident_loop_publication import verify_resident_loop_runtime

    name = state['name']
    receipt = _resident_runtime_receipt(state)
    verify_resident_loop_runtime(state['representative_source'], state['compiled_root'], state['sources'],
                                 receipt, name=name, target_source=(state['target_source'] if target_source is None
                                                                  else Path(target_source)))
    files = dict(state['files'])
    body = files[name+'.c'].decode()
    start = body.index('/* atlas resident binding table begin */')
    end = body.index('/* atlas resident binding table end */', start)+len('/* atlas resident binding table end */')
    files[name+'.c'] = (body[:start]+_resident_runtime_table(state).strip()+body[end:]).encode()
    return files, receipt


def prepare_c_package(source, compiled, name, *, target_source, _resident_runtime=None):
    """Export one checked contraction image and preserve dense source inputs."""
    if (Path(compiled)/'row-block-manifest.json').is_file():
        from .quantized_row_blocks_c import prepare_c_package as prepare_rows
        return prepare_rows(source, compiled, name, target_source=target_source)
    from merlin.compiler.lower.c.dispatch import c_source_pair
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
                                     exponent_page_cache=recorded_enabled(execution),
                                     output_prefix='output_tile_store_bytes' in execution):
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
    resident_ids = {item['node_id'] for item in execution.get('resident_bindings', [])}
    resident_indices = {plan.input_ids.index(key) for key in resident_ids}
    output_bytes = batches*m*n*(2 if plan.output_dtype=="bf16" else 4)
    mc,nc,kc = ((value+31)//32 for value in (m,n,k))
    rhs_batches = batches if b.type.shape[:-2] else 1
    lhs_bytes,rhs_bytes = execution["input_buffer_bytes"][:2]
    device_output_bytes = execution["output_bytes"]
    exponent_bytes = execution['exponent_buffer_bytes']
    operand_offsets = [0, lhs_bytes]
    exponent_offsets = [lhs_bytes+rhs_bytes, lhs_bytes+rhs_bytes+exponent_bytes[0]]
    output_offset = exponent_offsets[1]+exponent_bytes[1]
    status_offset = output_offset+device_output_bytes
    if resident_ids:
        from .binding import io_memory_base
        base = io_memory_base()
        # Resident host offsets are zero and are never written.
        operand_offsets = [0 if node_id in resident_ids else address-base for node_id, address in
                           zip(plan.physical_operand_ids, execution['input_addresses'][:2], strict=True)]
        exponent_offsets = [0 if node_id in resident_ids else address-base for node_id, address in
                            zip(plan.exponent_ids, execution['exponent_addresses'], strict=True)]
        output_offset = execution['output_addresses'][0]-base
        status_offset = execution['status_address']-base
    extra_offset = status_offset+execution['status_bytes']
    extra_bytes = extra_index = 0
    extra_encoder = ''
    if plan.epilogue_precision is not None:
        from merlin.compiler.lower.c.conversions import render_f32_array_c, render_bf16_codes_c
        if (len(execution['input_addresses']) != 3 or len(execution['input_buffer_bytes']) != 3
                or execution['input_addresses'][2] != execution['status_address']+execution['status_bytes']
                or execution['input_buffer_bytes'][2] != device_output_bytes):
            raise ValueError('quantized epilogue requires one complete readonly BF16 buffer after output and status')
        extra_bytes = execution['input_buffer_bytes'][2]
        extra_index = plan.input_ids.index(plan.epilogue_other_id)
        spec = plan.epilogue_precision.operand_specs[1-plan.epilogue_contract_operand]
        encode = render_bf16_codes_c if plan.epilogue_other_dtype == 'bf16' else render_f32_array_c
        extra_encoder = encode(spec,batches*m*n,name+'_encode_extra')
        # A uniform pointer type for the shared routine; the encoder keeps its own signature.
        extra_encoder += f"static int {name}_extra(const void *source,uint8_t *output) {{ return {name}_encode_extra(source,output); }}\n"
    raw,converted,count = extra_offset+extra_bytes,output_bytes,len(inputs)
    image_offset = (converted+3)//4*4
    if _resident_runtime is not None:
        converted = image_offset+len(binary)
    # The host image is a copy of the device frame. On chip the host scratch shares the I/O window with the frame, the
    # dense source inputs, the output, its converted copy, and the instruction image, so when they cannot all fit there,
    # each tile is packed and written, or read back and restored, in turn through one tile of host scratch.
    from .binding import validate_io_window
    from .options import current_codegen_options
    frame = [*zip(execution['input_addresses'], execution['input_buffer_bytes'], strict=True),
             *zip(execution['exponent_addresses'], exponent_bytes, strict=True),
             (execution['output_addresses'][0], device_output_bytes), (execution['status_address'], execution['status_bytes'])]
    shared = max(a+s for a, s in frame)-min(a for a, _ in frame)+sum(sizes)+output_bytes+converted+len(binary)
    streamed = current_codegen_options().application_deployment == 'onchip' and shared+raw > validate_io_window()[1]
    if streamed:
        raw, status_offset, output_offset = 2048, 0, 0
    runtime = target_source/"runtime"
    public = runtime_headers(runtime)
    host_members,runtime_files,runtime_record = runtime_members(public,(runtime/"atlas_host.c").read_text(),"separate")
    upper=name.upper()
    read_spans = [(execution["input_addresses"][0], operand_offsets[0], lhs_bytes),
                  (execution["input_addresses"][1], operand_offsets[1], rhs_bytes),
                  *zip(execution['exponent_addresses'], exponent_offsets, exponent_bytes, strict=True)]
    read_ids = [*plan.physical_operand_ids, *plan.exponent_ids]
    if extra_bytes:
        read_spans.append((execution["input_addresses"][2], extra_offset, extra_bytes))
        read_ids.append(plan.epilogue_other_id)
    if count > 8 or len(read_spans) > 5:
        raise ValueError("quantized contraction exceeds the shared dispatch plan")
    out = execution["output_addresses"][0]
    exponent_packs = []
    for index, offset, geometry in zip((iea, ieb), exponent_offsets, execution['exponent_layouts'], strict=True):
        if index in resident_indices:
            exponent_packs.append("{0u,0u,0u,0u,0u,0u,0u}")
        elif geometry['mode'] == 'row':
            exponent_packs.append(f"{{1u,{index}u,{offset}u,{geometry['batches']}u,{geometry['rows']}u,{geometry['tiles']}u,0u}}")
        else:
            exponent_packs.append(f"{{2u,{index}u,{offset}u,0u,0u,0u,{geometry['pages']}u}}")
    lhs_columns = rhs_columns = 0
    if plan.operand_permutation is not None:
        lhs_columns = int(plan.operand_permutation[0] == plan.operand_ids[0])
        rhs_columns = 1 - lhs_columns
    resident_mask = sum(1 << index for index in resident_indices)
    reads = ",".join(f"{{{address}u,{offset}u,{size}u,{int(node_id in resident_ids)}u}}"
                     for (address, offset, size), node_id in zip(read_spans, read_ids, strict=True))
    from .shared_dispatch_c import CHECKS, REGION_TYPES
    helper = 'atlas_matmul' if _resident_runtime is None else 'atlas_resident_matmul'
    declarations = REGION_TYPES + (MATMUL_TYPES if _resident_runtime is None else RESIDENT_MATMUL_TYPES) + f"""
#define {upper}_WORKSPACE_BYTES {raw}u
#define {upper}_SOURCE_WORKSPACE_BYTES {converted}u
#define {upper}_SOURCE_INPUT_COUNT {count}u
typedef atlas_region_context {name}_context;
typedef {helper}_source_context {name}_source_context;
int {name}_bind({name}_context *,atlas_host *,atlas_tile_memory_io,void *,size_t,size_t);
int {name}_source_bind({name}_source_context *,{name}_context *,void *,size_t);
int {name}_source_dispatch(void *,const void *const *,void *const *);
"""
    runtime_source = ''
    image_words = name+'_words'
    if _resident_runtime is not None:
        image_words = '0'
        image_c = _resident_runtime['image'].emit_c(name+'_resident_image')
        # Headers are flattened below with the existing checked runtime headers.
        public[name+'_resident_image.h'] = image_c.header
        runtime_source = image_c.source
        declarations += f"int {name}_source_bind_index({name}_source_context *,{name}_context *,void *,size_t,uint32_t);\n"
        _resident_runtime.update(name=name, image_offset=image_offset,
                                 read_ids=read_ids, resident_ids=resident_ids,
                                 source_receipt=image_c.receipt)
        runtime_source += _resident_runtime_table(_resident_runtime)
    implementation = extra_encoder + (program_array(name + "_words", binary) if _resident_runtime is None else '') + CHECKS + (MATMUL_DISPATCH if _resident_runtime is None else _resident_dispatch_helpers()) + f"""
static const {helper}_plan {name}_plan={{{image_words},{len(binary)//4}u,{raw}u,{converted}u,{output_bytes}u,{count}u,
  {{{','.join(f'{size}u' for size in sizes)}}},{resident_mask}u,{ia}u,{ib}u,{lhs_columns}u,{rhs_columns}u,{int(plan.output_dtype != 'bf16')}u,
  {prod(a.type.shape)}u,{prod(b.type.shape)}u,{batches}u,{rhs_batches}u,{m}u,{n}u,{k}u,{mc}u,{nc}u,{kc}u,{operand_offsets[1]}u,{output_offset}u,
  {{{','.join(exponent_packs)}}},{name + '_extra' if extra_bytes else '0'},{extra_index}u,{extra_offset}u,{extra_bytes}u,
  {len(read_spans)}u,{{{reads}}},
  {execution['status_address']}u,{execution['status_bytes']}u,{status_offset}u,{out}u,{output_offset}u,{device_output_bytes}u,{int(streamed)}u}};
int {name}_bind({name}_context *ctx,atlas_host *host,atlas_tile_memory_io io,void *scratch,size_t bytes,size_t polls) {{
  return {helper}_bind(ctx,host,&io,scratch,bytes,polls,&{name}_plan);
}}
int {name}_source_bind({name}_source_context *ctx,{name}_context *region,void *scratch,size_t bytes) {{
  return {helper}_source_bind(ctx,region,scratch,bytes,&{name}_plan);
}}
int {name}_source_dispatch(void *context,const void *const *inputs,void *const *outputs) {{
  {_resident_dispatch(_resident_runtime) if _resident_runtime is not None else f'return atlas_matmul_dispatch(&{name}_plan,context,inputs,outputs);'}
}}
"""
    if _resident_runtime is not None:
        implementation += f"""
int {name}_source_bind_index({name}_source_context *ctx,{name}_context *region,void *scratch,size_t bytes,uint32_t index) {{
  if (index>={name}_resident_count || {name}_source_bind(ctx,region,scratch,bytes)) return 1;
  ctx->binding=index; return 0;
}}
"""
    files={key:value.encode() for key,value in c_source_pair(name,public_headers=public,private_headers={},
        sources={**host_members,**({"resident_runtime.c":runtime_source} if runtime_source else {}),
                 "quantized_dispatch.c":implementation},declarations=declarations).items()}
    files.update(runtime_files)
    record=dict(kind="already_quantized",schema="atlas.already_quantized_c_package.v1",package_name=name,
        execution_route='checked_counted_image',launches_per_invocation=1,restoration_placement='device',
        exponent_page_upload_bytes=EXPONENT_PAGE_PAYLOAD_BYTES,
        context_type=name+"_context",source_context_type=name+"_source_context",binding_function=name+"_bind",
        source_binding_function=name+"_source_bind",workspace_bytes=raw,source_workspace_bytes=converted,
        readout_scale_domain="device admitted signed exponent sums in [-133,127]; finite normal BF16 or positive zero",
        rejection_status=dict(address=execution["status_address"],bytes=4,success=0,rejection=1),
        precision_plan=plan.record(),input_node_ids=list(plan.input_ids),output_node_ids=[plan.output_id],
        source_interface=dict(workspace_bytes=converted,failure_output="unchanged",completion="synchronous"),**runtime_record)
    if resident_ids:
        record['resident_bindings'] = execution['resident_bindings']
    if _resident_runtime is not None:
        record.update(source_binding_index_function=name+'_source_bind_index', source_binding_index=0,
                      resident_loop_runtime=_resident_runtime_receipt(_resident_runtime))
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
