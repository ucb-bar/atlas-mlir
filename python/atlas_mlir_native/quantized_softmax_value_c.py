"""Publish one checked exp-pack/PV body through a bounded source C interface."""
from __future__ import annotations

import hashlib
import json
from math import prod
from pathlib import Path

from merlin.common.paths import runtime_dir
from merlin.compile.c_conversions import render_f32_array_c
from merlin.compile.c_sources import c_source_pair
from merlin.semantic_compiler.precision_values import FloatBoundarySpec
from merlin.semantic_compiler.softmax_source import recognize_masked_row_softmax
from merlin.semantic_compiler.target_binding import verify_native_publication

from .binding import DRAM_BASE, TILE_BYTES, check_io_memory_manifest, dram_slots
from .c_runtime import program_array, runtime_headers, runtime_members
from .layout import pack_bf16_pair_halves
from .quantized_softmax_value import prepare_source, score_spec
from .resident_graph import mixed_memory_plan


def value_spec(exponent):
    return FloatBoundarySpec('f32-cap448-quantized-softmax-value-input-v1', 'fp8_e4m3', 'encode',
        max_normal_exponent_code=15, flush_below_normal=True,
        pre_encode_scale_exponent=-exponent, canonicalize_negative_zero=True,
        underflow_rounding='round-normal-significand-then-flush')


def _member(root, name):
    relative = Path(name)
    path = root / relative
    if (relative.is_absolute() or '..' in relative.parts or relative.as_posix() != name
            or path.is_symlink() or not path.is_file() or not path.resolve().is_relative_to(root.resolve())):
        raise ValueError('missing or unsafe quantized softmax value artifact: '+name)
    return path.read_bytes()


def _checked_execution(plan, compiled, target_source):
    if compiled.is_symlink() or not compiled.is_dir():
        raise ValueError('quantized softmax value needs a real body compilation')
    manifest = json.loads(_member(compiled, 'manifest.json'))
    check_io_memory_manifest(manifest)
    verify_native_publication(compiled, manifest, engine='merlin_native',
        request_digest=plan.body.digest(), target_identity=plan.body.target_identity)
    source_bytes = _member(compiled, 'semantic-source.json')
    plan_bytes = _member(compiled, 'quantized-softmax-value-plan.json')
    source_hash, plan_hash = hashlib.sha256(source_bytes).hexdigest(), hashlib.sha256(plan_bytes).hexdigest()
    stages = manifest.get('stage_artifacts', {})
    if (json.loads(source_bytes) != plan.source.record() or json.loads(plan_bytes) != plan.record()
            or stages.get('semantic-source.json') != source_hash
            or stages.get('quantized-softmax-value-plan.json') != plan_hash
            or manifest.get('source_softmax_value_request_digest') != plan.source.digest()
            or manifest.get('source_softmax_value_plan_sha256') != plan_hash):
        raise ValueError('quantized softmax value source, plan, or provenance binding differs')
    inventory = _member(target_source, 'docs/selected-variant-inventory.json')
    if manifest.get('target_inventory_sha256') != hashlib.sha256(inventory).hexdigest():
        raise ValueError('quantized softmax value target inventory differs')
    captured = json.loads(_member(compiled, 'source-request.json'))
    if captured != plan.body.record():
        raise ValueError('quantized softmax value body source differs')
    execution = json.loads(_member(compiled, 'execution_plan.json'))
    _, slots, sizes, liveness = mixed_memory_plan(plan.body)
    def row(key):
        return dict(node=key, address=DRAM_BASE+slots[key]*TILE_BYTES, bytes=sizes[key])
    inputs = [dict(**row(n.id), type=n.type.record()) for n in plan.body.nodes if n.effect in {'input', 'constant'}]
    outputs = [dict(**row(key), type=plan.body.node(key).type.record()) for key in plan.body.outputs]
    buffers = [row(key) for key in slots]
    if (execution.get('schema') != 'atlas.native_tile_execution_plan.v8'
            or execution.get('inputs') != inputs or execution.get('outputs') != outputs
            or execution.get('buffers') != buffers or execution.get('source_liveness_buffers') != liveness
            or execution.get('backend_verification_tier') != 'verified'
            or execution.get('launches_per_invocation') != 1):
        raise ValueError('quantized softmax value execution buffers or verified tier differ')
    if max(slots.values()) >= dram_slots():
        raise ValueError('quantized softmax value body exceeds diagnostic memory')
    constants = []
    for index, binding in enumerate(plan.body.constants):
        data = bytes.fromhex(binding.data_hex)
        packed = pack_bf16_pair_halves(data) if binding.encoding == 'bf16-le-raw' else data
        descriptor = dict(node_id=binding.node_id, address=row(binding.node_id)['address'], bytes=len(packed),
            encoding=binding.encoding, physical_layout='bf16_pair_halves' if binding.encoding == 'bf16-le-raw' else 'fp8_row_major',
            path=f'constants/constant-{index:03d}.bin', sha256=hashlib.sha256(packed).hexdigest())
        if _member(compiled, descriptor['path']) != packed:
            raise ValueError('quantized softmax value constant payload differs')
        constants.append(descriptor)
    if manifest.get('constant_inputs') != constants or execution.get('constant_inputs') != constants:
        raise ValueError('quantized softmax value constant binding differs')
    binary = _member(compiled, 'program.bin')
    if hashlib.sha256(binary).hexdigest() != manifest.get('binary_sha256'):
        raise ValueError('quantized softmax value binary differs')
    return execution, binary, slots


def prepare_c_package(source, compiled, name, *, value_scale_exponent, target_source, runtime_linkage='separate', provenance_sha256=None):
    plan = prepare_source(source, value_scale_exponent=value_scale_exponent, provenance_sha256=provenance_sha256)
    compiled, target_source = Path(compiled), Path(target_source)
    execution, binary, slots = _checked_execution(plan, compiled, target_source)
    return _render_package(plan, execution, binary, slots, name, target_source, runtime_linkage)


def _render_package(plan, execution, binary, slots, name, target_source, runtime_linkage):
    """Render transport only; admission is performed by prepare_c_package."""
    region, body = plan.region, plan.body
    masked = recognize_masked_row_softmax(plan.source, region.probabilities)
    batch, rows, keys, channels = prod(region.batch_shape), region.query_rows, region.keys, region.channels
    score_count, value_count, output_count = batch*rows*keys, batch*keys*channels, batch*rows*channels
    score_bytes, result_bytes = 2*score_count, 2*output_count
    result_at, workspace = score_bytes+value_count, score_bytes+value_count+result_bytes
    mask_count = prod(masked.mask_shape)
    if max(score_count, value_count, output_count, mask_count) > 1 << 28 or workspace > 2**32-1:
        raise ValueError('quantized softmax value source buffers exceed bounded host ABI')
    upper = name.upper()
    addresses = {entry['node']: entry['address'] for entry in execution['inputs']+execution['outputs']}
    readonly = sum(n.effect in {'input', 'constant'} for n in body.nodes)
    end = max(slots.values())+1
    ranges = [(DRAM_BASE, readonly*TILE_BYTES, 'READ'),
              (DRAM_BASE+readonly*TILE_BYTES, (end-readonly)*TILE_BYTES, 'READ_WRITE')]
    if any(size <= 0 for _, size, _ in ranges):
        raise ValueError('quantized softmax value body needs readonly and mutable slot ranges')
    buffers = ','.join(f'{{{address}u,{size}u,ATLAS_HOST_DEVICE_{mode}}}' for address,size,mode in ranges)
    public = runtime_headers(target_source/'runtime')
    members, files, runtime = runtime_members(public, _member(target_source, 'runtime/atlas_host.c').decode(), runtime_linkage)
    declarations = f'''
#define {upper}_WORKSPACE_BYTES 2048u
#define {upper}_SOURCE_WORKSPACE_BYTES {workspace}u
#define {upper}_SOURCE_INPUT_COUNT 3u
#define {upper}_SOURCE_OUTPUT_COUNT 1u
#define {upper}_SOURCE_INPUT_0_BYTES {score_count*4}u
#define {upper}_SOURCE_INPUT_1_BYTES {value_count*4}u
#define {upper}_SOURCE_INPUT_2_BYTES {mask_count*4}u
#define {upper}_SOURCE_OUTPUT_0_BYTES {output_count*4}u
/* Preserve both contexts and workspaces through completion or device recovery. */
typedef enum {{ {upper}_OK=0,{upper}_INVALID,{upper}_INPUT_DOMAIN,{upper}_MEMORY_ERROR,{upper}_DEVICE_ERROR }} {name}_status;
typedef struct {{ atlas_host *host; atlas_tile_memory_io io; uint8_t *workspace; size_t max_polls; atlas_host_buffer buffers[2]; }} {name}_context;
typedef struct {{ {name}_context *region; uint8_t *workspace; }} {name}_source_context;
{name}_status {name}_bind({name}_context *,atlas_host *,atlas_tile_memory_io,void *,size_t,size_t);
{name}_status {name}_source_bind({name}_source_context *,{name}_context *,void *,size_t);
/* Dense f32 scores, values, and additive mask; output is published after all groups succeed. */
{name}_status {name}_source_run({name}_source_context *,const float *const *,float *,atlas_host_result *);
int {name}_source_dispatch(void *,const void *const *,void *const *);
'''
    strides, stride = [], 1
    for extent in reversed(masked.mask_shape):
        strides.insert(0, stride)
        stride *= extent
    coefficients = [sum(row[d]*s for row,s in zip(masked.mask_map.coefficients,strides,strict=True)) for d in range(len(masked.shape))]
    coefficients = [0 if extent == 1 else coefficient for coefficient,extent in zip(coefficients,masked.shape,strict=True)]
    offset = sum(o*s for o,s in zip(masked.mask_map.offsets,strides,strict=True))
    steps = '\n'.join(f'  size_t d{d}=index%{masked.shape[d]}u; index/={masked.shape[d]}u;' for d in reversed(range(len(masked.shape))))
    expression = str(offset)+''.join(f'+({c}LL)*(int64_t)d{d}' for d,c in enumerate(coefficients))
    constant_arrays, writes = [], []
    for i, binding in enumerate(body.constants):
        raw = bytes.fromhex(binding.data_hex)
        data = pack_bf16_pair_halves(raw) if binding.encoding == 'bf16-le-raw' else raw
        symbol = f'{name}_constant_{i}'
        constant_arrays.append(f'static const uint8_t {symbol}[]={{'+','.join(str(b) for b in data)+'};')
        writes.append(f'    if (region->io.write(region->io.context,{addresses[binding.node_id]}u,{symbol},{len(data)}u)) return {upper}_MEMORY_ERROR;')
    key_tiles, channel_tiles = (keys+31)//32, (channels+31)//32
    for kt in range(key_tiles):
        for keep in (False, True):
            key = f'keep_{kt}' if keep else f'scores_{kt}'
            writes.append(f'''    for (size_t r=0;r<32u;++r) for (size_t k=0;k<32u;++k) {{
      size_t row=first+r,key={kt*32}u+k; uint16_t code={'0u' if keep else '0xc220u'};
      if (row<{rows}u && key<{keys}u) {{
        size_t at=(batch*{rows}u+row)*{keys}u+key; uint32_t mask;
        __builtin_memcpy(&mask,(const uint8_t *)inputs[2]+4u*{name}_mask_index(at),4u);
        if (!mask) code={'0x3f80u' if keep else '(uint16_t)((uint16_t)scores[2u*at]|((uint16_t)scores[2u*at+1u]<<8))'};
      }}
      size_t at=(k/16u)*1024u+(r*16u+k%16u)*2u;
      tile[at]=(uint8_t)code;tile[at+1u]=(uint8_t)(code>>8);
    }}
    if (region->io.write(region->io.context,{addresses[key]}u,tile,2048u)) return {upper}_MEMORY_ERROR;''')
    for nt in range(channel_tiles):
        for kt in range(key_tiles):
            writes.append(f'''    for (size_t d=0;d<32u;++d) for (size_t k=0;k<32u;++k) {{
      size_t channel={nt*32}u+d,key={kt*32}u+k;
      tile[d*32u+k]=(channel<{channels}u && key<{keys}u)?values[(batch*{keys}u+key)*{channels}u+channel]:0u;
    }}
    if (region->io.write(region->io.context,{addresses[f'value_{nt}_{kt}']}u,tile,1024u)) return {upper}_MEMORY_ERROR;''')
    reads=[]
    for nt, key in enumerate(body.outputs):
        reads.append(f'''    if (region->io.read(region->io.context,{addresses[key]}u,tile,2048u)) return {upper}_MEMORY_ERROR;
    for (size_t r=0;r<32u && first+r<{rows}u;++r) for (size_t d=0;d<32u && {nt*32}u+d<{channels}u;++d) {{
      size_t src=(d/16u)*1024u+(r*16u+d%16u)*2u;
      size_t dst=((batch*{rows}u+first+r)*{channels}u+{nt*32}u+d)*2u;
      result[dst]=tile[src];result[dst+1u]=tile[src+1u];
    }}''')
    # Every pointer span is checked before either conversion writes or MMIO.
    reserved = [('ctx','sizeof(*ctx)'),('region','sizeof(*region)'),('region->host','sizeof(*region->host)'),('ctx->workspace',f'{workspace}u'),('region->workspace','2048u')]
    checks = []
    spans = [(f'inputs[{i}]',f'{count*4}u') for i,count in enumerate((score_count,value_count,mask_count))]
    for pointer,size in [*spans,('out',f'{output_count*4}u'),('inputs','3u*sizeof(*inputs)')]:
        checks.append(f'!{name}_valid({pointer},{size})')
        checks.extend(f'{name}_overlap({pointer},{size},{other},{extent})' for other,extent in reserved)
    checks.extend(f'{name}_overlap({p},{n},out,{output_count*4}u)' for p,n in [*spans,('inputs','3u*sizeof(*inputs)')])
    device_checks = ' || '.join(f'{name}_overlap(device,sizeof(*device),{p},{n})' for p,n in [*reserved,*spans,('out',f'{output_count*4}u'),('inputs','3u*sizeof(*inputs)')])
    implementation = f'''
{program_array(name+'_words', binary)}
{chr(10).join(constant_arrays)}
static int {name}_valid(const void *p,size_t n) {{ return p && n && (uintptr_t)p<=UINTPTR_MAX-n; }}
static int {name}_overlap(const void *a,size_t n,const void *b,size_t m) {{ return (uintptr_t)a<(uintptr_t)b+m && (uintptr_t)b<(uintptr_t)a+n; }}
static size_t {name}_mask_index(size_t index) {{
{steps}
  return (size_t)({expression});
}}
{name}_status {name}_bind({name}_context *ctx,atlas_host *host,atlas_tile_memory_io io,void *scratch,size_t bytes,size_t polls) {{
  if (!ctx || !host || !host->mmio.read32 || !host->mmio.write32 || host->active || !io.write || !io.read || !polls || bytes<2048u ||
      !{name}_valid(scratch,2048u) || {name}_overlap(scratch,2048u,ctx,sizeof(*ctx)) || {name}_overlap(scratch,2048u,host,sizeof(*host)) ||
      {name}_overlap(ctx,sizeof(*ctx),host,sizeof(*host))) return {upper}_INVALID;
  *ctx=({name}_context){{host,io,scratch,polls,{{{buffers}}}}}; return {upper}_OK;
}}
{name}_status {name}_source_bind({name}_source_context *ctx,{name}_context *region,void *scratch,size_t bytes) {{
  if (!ctx || !region || !region->host || region->host->active || !region->workspace || bytes<{workspace}u ||
      !{name}_valid(scratch,{workspace}u) || {name}_overlap(scratch,{workspace}u,ctx,sizeof(*ctx)) ||
      {name}_overlap(scratch,{workspace}u,region,sizeof(*region)) || {name}_overlap(scratch,{workspace}u,region->host,sizeof(*region->host)) ||
      {name}_overlap(scratch,{workspace}u,region->workspace,2048u) || {name}_overlap(ctx,sizeof(*ctx),region,sizeof(*region)) ||
      {name}_overlap(ctx,sizeof(*ctx),region->host,sizeof(*region->host)) || {name}_overlap(ctx,sizeof(*ctx),region->workspace,2048u)) return {upper}_INVALID;
  *ctx=({name}_source_context){{region,scratch}}; return {upper}_OK;
}}
{name}_status {name}_source_run({name}_source_context *ctx,const float *const *inputs,float *out,atlas_host_result *device) {{
  if (!ctx || !ctx->region || !ctx->region->host || !ctx->workspace || !inputs || !ctx->region->workspace ||
      !ctx->region->io.read || !ctx->region->io.write || !ctx->region->max_polls || !{name}_valid(inputs,3u*sizeof(*inputs))) return {upper}_INVALID;
  {name}_context *region=ctx->region;
  if ({' || '.join(checks)}) return {upper}_INVALID;
  if (device && (!{name}_valid(device,sizeof(*device)) || {device_checks})) return {upper}_INVALID;
  if (device) *device=ATLAS_HOST_INVALID;
  if (region->host->active) {{ if (device) *device=ATLAS_HOST_BUSY; return {upper}_DEVICE_ERROR; }}
  for (size_t i=0;i<{mask_count}u;++i) {{
    uint32_t bits;__builtin_memcpy(&bits,(const uint8_t *)inputs[2]+4u*i,4u);
    if (bits!=0u && bits!=0xff800000u && bits!=0xff7fffffu) return {upper}_INPUT_DOMAIN;
  }}
  uint8_t *scores=ctx->workspace,*values=ctx->workspace+{score_bytes}u,*result=ctx->workspace+{result_at}u,*tile=region->workspace;
  if ({name}_encode_scores(inputs[0],scores) || {name}_encode_values(inputs[1],values)) return {upper}_INPUT_DOMAIN;
  atlas_host_result status=atlas_host_ensure_loaded(region->host,{name}_words,{len(binary)//4}u);
  if (device) *device=status;
  if (status!=ATLAS_HOST_OK) return {upper}_DEVICE_ERROR;
  for (size_t batch=0;batch<{batch}u;++batch) for (size_t first=0;first<{rows}u;first+=32u) {{
{chr(10).join(writes)}
    status=atlas_host_start(region->host,region->buffers,2u);
    if (status==ATLAS_HOST_OK) status=atlas_host_wait(region->host,region->max_polls);
    if (device) *device=status;
    if (status!=ATLAS_HOST_OK) return {upper}_DEVICE_ERROR;
{chr(10).join(reads)}
  }}
  for (size_t i=0;i<{output_count}u;++i) {{
    uint32_t bits=((uint32_t)result[2u*i]|((uint32_t)result[2u*i+1u]<<8))<<16;
    __builtin_memcpy((uint8_t *)out+4u*i,&bits,4u);
  }}
  return {upper}_OK;
}}
int {name}_source_dispatch(void *context,const void *const *inputs,void *const *outputs) {{
  if (!inputs || !outputs) return {upper}_INVALID;
  const float *typed[]={{inputs[0],inputs[1],inputs[2]}};
  return {name}_source_run(context,typed,outputs[0],0);
}}
'''
    converters = render_f32_array_c(score_spec(), score_count, name+'_encode_scores')+'\n'+render_f32_array_c(value_spec(plan.value_scale_exponent), value_count, name+'_encode_values')
    rendered = c_source_pair(name,public_headers=public,private_headers={'finite_float_boundary.h':_member(runtime_dir()/'c','finite_float_boundary.h').decode()},sources={**members,'quantized_softmax_value.c':converters+implementation},declarations=declarations)
    files.update({key:value.encode() for key,value in rendered.items()})
    files['program.bin']=binary
    files['source-plan.json']=(json.dumps(plan.record(),indent=2)+'\n').encode()
    return {**runtime,'schema':'atlas.quantized_softmax_value_c_package.v1','workspace_bytes':2048,
        'source_workspace_bytes':workspace,'input_node_ids':[region.scores,region.values,region.mask],
        'output_node_ids':[region.result],'context_type':name+'_context','source_context_type':name+'_source_context',
        'binding_function':name+'_bind','source_binding_function':name+'_source_bind','dispatch_function':name+'_source_dispatch',
        'entry_function':name+'_source_run','launches_per_call':plan.groups,'source_plan':plan.record(),
        'precision_plan':plan.record(),'device_buffers':[dict(address=a,bytes=s,access=m) for a,s,m in ranges]},files
