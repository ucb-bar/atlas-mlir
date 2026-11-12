"""Checked four-image RMSNorm C transport with atomic FP8 source publication.

The host converts declared inputs, compares their words, and moves completed
tiles. All normalization arithmetic and the FP8 boundary execute natively.
"""
from hashlib import sha256
import json
from pathlib import Path

from merlin.common.paths import runtime_dir
from merlin.compile.c_conversions import render_f32_array_c
from merlin.compile.c_sources import c_source_pair
from merlin.compile.launch_schedule import make_launch_schedule
from merlin.semantic_compiler.target_binding import verify_native_publication

from .binding import AtlasNativeBinding, DRAM_BASE, TILE_BYTES, check_io_memory_manifest, dram_slots
from .c_runtime import program_array, runtime_headers, runtime_members
from .hierarchical_rmsnorm import prepare_source, operand_spec, runtime_admission
from .layout import pack_bf16_pair_halves
from .resident_graph import mixed_memory_plan

STAGES = ('square_reduce', 'reduce', 'inverse', 'publish')


def _member(root, name):
    relative = Path(name)
    path = root / relative
    if (relative.is_absolute() or '..' in relative.parts or relative.as_posix() != name
            or any(p.is_symlink() for p in (path, *path.parents))
            or not path.is_file() or path.stat().st_nlink != 1):
        raise ValueError('missing or unsafe hierarchical RMSNorm artifact: '+name)
    return path.read_bytes()


def _stage_layout(body, kind):
    """Derive addresses and complete device spans from the semantic request."""
    abi = AtlasNativeBinding.plan_boundary_abi(body)
    sources = [n for n in body.nodes if n.effect in {'input', 'constant'}]
    addresses = {k:DRAM_BASE+v*TILE_BYTES for k,v in abi['fixed_inputs'].items()}
    output = DRAM_BASE+abi['fixed_outputs'][0]*TILE_BYTES
    if kind == 'publish':
        _, slots, sizes, liveness = mixed_memory_plan(body)
        def row(key):
            return dict(node=key, address=DRAM_BASE+slots[key]*TILE_BYTES, bytes=sizes[key])
        typed_inputs = [dict(**row(n.id), type=n.type.record()) for n in sources]
        typed_outputs = [dict(**row(k), type=body.node(k).type.record()) for k in body.outputs]
        if addresses != {n.id:row(n.id)['address'] for n in sources} or output != row(body.outputs[0])['address']:
            raise ValueError('hierarchical RMSNorm ABI and mixed memory allocation differ')
        readonly, end = len(sources), max(slots.values())+1
        ranges = [(DRAM_BASE, readonly*TILE_BYTES, 'READ'),
                  (DRAM_BASE+readonly*TILE_BYTES, (end-readonly)*TILE_BYTES, 'READ_WRITE')]
        expected = dict(inputs=typed_inputs, outputs=typed_outputs,
            buffers=[row(k) for k in slots], source_liveness_buffers=liveness,
            staging_workspace=dict(address=DRAM_BASE+readonly*TILE_BYTES, bytes=(end-readonly)*TILE_BYTES),
            operation='mixed_tile_staged', backend_verification_tier='verified', launches_per_invocation=1,
            input_payload_bytes=[2048]*len(sources), output_payload_bytes=[1024])
        write_spans = [row(n.id) for n in body.nodes if n.effect=='pure']
    else:
        ranges = [(addresses[n.id], 2048, 'READ') for n in sources]+[(output, 2048, 'WRITE')]
        expected = dict(operation='bf16_row_sum' if kind == 'reduce' else 'bf16_mixed',
                        input_bytes=2048, output_address=output)
        write_spans = [dict(node=body.outputs[0],address=output,bytes=2048)]
    if any(a < DRAM_BASE or s <= 0 or a+s > DRAM_BASE+dram_slots()*TILE_BYTES for a,s,_ in ranges):
        raise ValueError('hierarchical RMSNorm stage exceeds diagnostic memory')
    return dict(addresses=addresses, output=output, output_bytes=1024 if kind=='publish' else 2048,
                ranges=ranges, write_spans=write_spans, expected=expected)


def _checked_stages(plan, compiled, target_source):
    parent_bytes = _member(compiled, 'hierarchical-rmsnorm-plan.json')
    parent = json.loads(parent_bytes)
    source_bytes = _member(compiled, 'semantic-source.json')
    if json.loads(source_bytes) != plan.source.record() or set(parent) != set(plan.record())|{'compiled_stages'}:
        raise ValueError('hierarchical RMSNorm source or plan binding differs')
    if {k:v for k,v in parent.items() if k!='compiled_stages'} != plan.record() or set(parent['compiled_stages']) != set(STAGES):
        raise ValueError('hierarchical RMSNorm policy or provenance binding differs')
    inventory_hash = sha256(_member(target_source, 'docs/selected-variant-inventory.json')).hexdigest()
    result = {}
    for kind in STAGES:
        root, body = compiled/kind, plan.stages[kind]
        manifest_bytes = _member(root, 'manifest.json')
        execution_bytes = _member(root, 'execution_plan.json')
        binary = _member(root, 'program.bin')
        manifest, execution = json.loads(manifest_bytes), json.loads(execution_bytes)
        expected_record = dict(request_digest=body.digest(), program_sha256=sha256(binary).hexdigest(),
            manifest_path=kind+'/manifest.json', manifest_sha256=sha256(manifest_bytes).hexdigest(),
            execution_plan_sha256=sha256(execution_bytes).hexdigest())
        if parent['compiled_stages'][kind] != expected_record:
            raise ValueError('hierarchical RMSNorm compiled stage hashes differ')
        if (manifest.get('binary_sha256') != expected_record['program_sha256']
                or manifest.get('execution_plan_sha256') != expected_record['execution_plan_sha256']
                or manifest.get('target_inventory_sha256') != inventory_hash
                or json.loads(_member(root, 'source-request.json')) != body.record()):
            raise ValueError('hierarchical RMSNorm stage source or target differs')
        # Check all publication paths before the generic verifier reads them.
        for member, digest in manifest.get('stage_artifacts', {}).items():
            if sha256(_member(root, member)).hexdigest() != digest:
                raise ValueError('hierarchical RMSNorm stage artifact differs')
        check_io_memory_manifest(manifest)
        verify_native_publication(root, manifest, engine='merlin_native',
            request_digest=body.digest(), target_identity=body.target_identity)
        physical = json.loads(_member(root, 'physical-program.json'))
        instructions = physical.get('instructions', [])
        if (not binary or len(binary)%4 or physical.get('schema') != 'atlas.physical_program.v1'
                or physical.get('word_endianness') != 'little' or physical.get('word_count') != len(binary)//4
                or physical.get('timing_state') != 'timed' or len(instructions) != len(binary)//4
                or any(row.get('word_index') != i or type(row.get('word_u32')) is not int
                       or row['word_u32'] != int.from_bytes(binary[4*i:4*i+4], 'little')
                       for i,row in enumerate(instructions))):
            raise ValueError('hierarchical RMSNorm physical words or timing publication differ')
        layout = _stage_layout(body, kind)
        sources = [n for n in body.nodes if n.effect in {'input','constant'}]
        if (execution.get('schema') != 'atlas.native_tile_execution_plan.v8'
                or execution.get('input_addresses') != [layout['addresses'][n.id] for n in sources]
                or execution.get('output_addresses') != [layout['output']]
                or execution.get('output_bytes') != layout['output_bytes']
                or execution.get('output_physical_layouts') != ['fp8_row_major' if kind=='publish' else 'bf16_pair_halves']
                or any(execution.get(k) != v for k,v in layout['expected'].items())):
            raise ValueError('hierarchical RMSNorm execution buffers or verified tier differ')
        constants = []
        for index, binding in enumerate(body.constants):
            if binding.encoding != 'bf16-le-raw':
                raise ValueError('hierarchical RMSNorm constants require BF16')
            data = pack_bf16_pair_halves(bytes.fromhex(binding.data_hex))
            descriptor = dict(node_id=binding.node_id, address=layout['addresses'][binding.node_id], bytes=len(data),
                encoding=binding.encoding, physical_layout='bf16_pair_halves', path=f'constants/constant-{index:03d}.bin',
                sha256=sha256(data).hexdigest())
            if _member(root, descriptor['path']) != data:
                raise ValueError('hierarchical RMSNorm constant payload differs')
            constants.append(descriptor)
        if manifest.get('constant_inputs', []) != constants or execution.get('constant_inputs', []) != constants:
            raise ValueError('hierarchical RMSNorm constants differ')
        result[kind] = dict(**layout, binary=binary, publication=expected_record)
    # The rendered code uses captured bytes, and admission refuses concurrent replacements.
    if _member(compiled, 'hierarchical-rmsnorm-plan.json') != parent_bytes or _member(compiled, 'semantic-source.json') != source_bytes:
        raise ValueError('hierarchical RMSNorm compilation changed during admission')
    return result


def prepare_c_package(source, compiled, name, *, policy, target_source, runtime_linkage='separate'):
    plan = prepare_source(source, policy)
    compiled, target_source = Path(compiled), Path(target_source)
    stages = _checked_stages(plan, compiled, target_source)
    record, files = _render_package(plan, stages, name, target_source, runtime_linkage)
    if _checked_stages(plan, compiled, target_source) != stages:
        raise ValueError('hierarchical RMSNorm compilation changed during rendering')
    record['compilation_binding'] = dict(
        semantic_source_sha256=sha256(_member(compiled, 'semantic-source.json')).hexdigest(),
        compiled_plan_sha256=sha256(_member(compiled, 'hierarchical-rmsnorm-plan.json')).hexdigest(),
        stages={kind:stages[kind]['publication'] for kind in STAGES})
    return record, files


def _render_package(plan, stages, name, target_source, runtime_linkage):
    rows, width = plan.region.matrix_shape
    count, tiles, groups = rows*width, width//32, width//1024
    # Input conversion, atomic FP8 result, partial matrices and four logical tiles.
    result_at = 2*count+2*width
    partial_at = result_at+count
    workspace = partial_at+groups*2048+8192
    if count > 1<<28 or workspace > 2**32-1:
        raise ValueError('hierarchical RMSNorm buffers exceed bounded host ABI')
    upper = name.upper()
    public = runtime_headers(target_source/'runtime')
    members, files, runtime = runtime_members(public, _member(target_source, 'runtime/atlas_host.c').decode(), runtime_linkage)
    declarations = f'''
#define {upper}_WORKSPACE_BYTES 2048u
#define {upper}_SOURCE_WORKSPACE_BYTES {workspace}u
#define {upper}_SOURCE_INPUT_COUNT 2u
#define {upper}_SOURCE_OUTPUT_COUNT 1u
#define {upper}_SOURCE_INPUT_0_BYTES {count*4}u
#define {upper}_SOURCE_INPUT_1_BYTES {width*4}u
#define {upper}_SOURCE_OUTPUT_0_BYTES {count*4}u
typedef enum {{ {upper}_OK=0,{upper}_INVALID,{upper}_INPUT_DOMAIN,{upper}_MEMORY_ERROR,{upper}_DEVICE_ERROR }} {name}_status;
typedef struct {{ atlas_host *host; atlas_tile_memory_io io; uint8_t *workspace; size_t max_polls; }} {name}_context;
typedef struct {{ {name}_context *region; uint8_t *workspace; }} {name}_source_context;
{name}_status {name}_bind({name}_context *,atlas_host *,atlas_tile_memory_io,void *,size_t,size_t);
{name}_status {name}_source_bind({name}_source_context *,{name}_context *,void *,size_t);
/* H and gamma are immutable dense f32. Exact unpack of completed FP8 is atomic. */
{name}_status {name}_source_run({name}_source_context *,const float *const *,float *,atlas_host_result *);
int {name}_source_dispatch(void *,const void *const *,void *const *);
'''
    fragments = []
    for kind in STAGES:
        stage, body = stages[kind], plan.stages[kind]
        symbol = name+'_'+kind
        fragments.append(program_array(symbol+'_words', stage['binary']))
        writes = []
        for i,binding in enumerate(body.constants):
            data = pack_bf16_pair_halves(bytes.fromhex(binding.data_hex))
            fragments.append(f'static const uint8_t {symbol}_constant_{i}[]={{'+','.join(map(str,data))+'};')
            writes.append(f'  if (region->io.write(region->io.context,{stage["addresses"][binding.node_id]}u,{symbol}_constant_{i},{len(data)}u)) return {upper}_MEMORY_ERROR;')
        inputs = [n for n in body.nodes if n.effect=='input']
        for i,node in enumerate(inputs):
            writes.append(f'''  {name}_to_native(in{i},region->workspace);
  if (region->io.write(region->io.context,{stage['addresses'][node.id]}u,region->workspace,2048u)) return {upper}_MEMORY_ERROR;''')
        ranges = ','.join(f'{{{a}u,{s}u,ATLAS_HOST_DEVICE_{m}}}' for a,s,m in stage['ranges'])
        fragments.append(f'''static {name}_status {symbol}({name}_context *region,{','.join(f'const uint8_t *in{i}' for i in range(len(inputs)))},uint8_t *out,atlas_host_result *device) {{
  static const atlas_host_buffer buffers[]={{{ranges}}};
  atlas_host_result status=atlas_host_ensure_loaded(region->host,{symbol}_words,{len(stage['binary'])//4}u);
  if (device) *device=status;
  if (status!=ATLAS_HOST_OK) return {upper}_DEVICE_ERROR;
{chr(10).join(writes)}
  status=atlas_host_start(region->host,buffers,{len(stage['ranges'])}u);
  if (status==ATLAS_HOST_OK) status=atlas_host_wait(region->host,region->max_polls);
  if (device) *device=status;
  if (status!=ATLAS_HOST_OK) return {upper}_DEVICE_ERROR;
  if (region->io.read(region->io.context,{stage['output']}u,region->workspace,{stage['output_bytes']}u)) return {upper}_MEMORY_ERROR;
  {f'__builtin_memcpy(out,region->workspace,1024u);' if kind=='publish' else f'{name}_from_native(region->workspace,out);'}
  return {upper}_OK;
}}
''')
    reserved = [('ctx','sizeof(*ctx)'),('region','sizeof(*region)'),('region->host','sizeof(*region->host)'),('ctx->workspace',f'{workspace}u'),('region->workspace','2048u')]
    spans = [('inputs[0]',f'{count*4}u'),('inputs[1]',f'{width*4}u'),('out',f'{count*4}u'),('inputs','2u*sizeof(*inputs)')]
    checks = []
    for p,n in spans:
        checks.append(f'!{name}_valid({p},{n})')
        checks.extend(f'{name}_overlap({p},{n},{q},{m})' for q,m in reserved)
    checks.extend(f'{name}_overlap({p},{n},out,{count*4}u)' for p,n in [*spans[:2],spans[3]])
    device_checks = ' || '.join(f'{name}_overlap(device,sizeof(*device),{p},{n})' for p,n in reserved+spans)
    threshold = runtime_admission(width)['nonzero_row_abs_max_min_bf16']
    implementation = f'''
static int {name}_valid(const void *p,size_t n) {{ return p && n && (uintptr_t)p<=UINTPTR_MAX-n; }}
static int {name}_overlap(const void *a,size_t n,const void *b,size_t m) {{ return (uintptr_t)a<(uintptr_t)b+m && (uintptr_t)b<(uintptr_t)a+n; }}
static void {name}_to_native(const uint8_t *src,uint8_t *dst) {{
  for(size_t r=0;r<32u;++r) for(size_t c=0;c<32u;++c) {{
    size_t a=(r*32u+c)*2u,b=(c/16u)*1024u+(r*16u+c%16u)*2u;
    dst[b]=src[a];dst[b+1u]=src[a+1u];
  }}
}}
static void {name}_from_native(const uint8_t *src,uint8_t *dst) {{
  for(size_t r=0;r<32u;++r) for(size_t c=0;c<32u;++c) {{
    size_t a=(r*32u+c)*2u,b=(c/16u)*1024u+(r*16u+c%16u)*2u;
    dst[a]=src[b];dst[a+1u]=src[b+1u];
  }}
}}
{chr(10).join(fragments)}
{name}_status {name}_bind({name}_context *ctx,atlas_host *host,atlas_tile_memory_io io,void *scratch,size_t bytes,size_t polls) {{
  if (!ctx || !host || !host->mmio.read32 || !host->mmio.write32 || host->active || !io.write || !io.read || !polls || bytes<2048u ||
      !{name}_valid(scratch,2048u) || {name}_overlap(scratch,2048u,ctx,sizeof(*ctx)) || {name}_overlap(scratch,2048u,host,sizeof(*host)) ||
      {name}_overlap(ctx,sizeof(*ctx),host,sizeof(*host))) return {upper}_INVALID;
  *ctx=({name}_context){{host,io,scratch,polls}};return {upper}_OK;
}}
{name}_status {name}_source_bind({name}_source_context *ctx,{name}_context *region,void *scratch,size_t bytes) {{
  if (!ctx || !region || !region->host || region->host->active || !region->workspace || bytes<{workspace}u ||
      !{name}_valid(scratch,{workspace}u) || {name}_overlap(scratch,{workspace}u,ctx,sizeof(*ctx)) ||
      {name}_overlap(scratch,{workspace}u,region,sizeof(*region)) || {name}_overlap(scratch,{workspace}u,region->host,sizeof(*region->host)) ||
      {name}_overlap(scratch,{workspace}u,region->workspace,2048u) || {name}_overlap(ctx,sizeof(*ctx),region,sizeof(*region)) ||
      {name}_overlap(ctx,sizeof(*ctx),region->host,sizeof(*region->host)) || {name}_overlap(ctx,sizeof(*ctx),region->workspace,2048u)) return {upper}_INVALID;
  *ctx=({name}_source_context){{region,scratch}};return {upper}_OK;
}}
{name}_status {name}_source_run({name}_source_context *ctx,const float *const *inputs,float *out,atlas_host_result *device) {{
  if (!ctx || !ctx->region || !ctx->region->host || !ctx->workspace || !inputs || !ctx->region->workspace ||
      !ctx->region->io.read || !ctx->region->io.write || !ctx->region->max_polls || !{name}_valid(inputs,2u*sizeof(*inputs))) return {upper}_INVALID;
  {name}_context *region=ctx->region;
  if ({' || '.join(checks)}) return {upper}_INVALID;
  if (device && (!{name}_valid(device,sizeof(*device)) || {device_checks})) return {upper}_INVALID;
  if (device) *device=ATLAS_HOST_INVALID;
  if (region->host->active) {{if(device)*device=ATLAS_HOST_BUSY;return {upper}_DEVICE_ERROR;}}
  uint8_t *hidden=ctx->workspace,*gamma=hidden+{2*count}u,*result=ctx->workspace+{result_at}u;
  uint8_t *partials=ctx->workspace+{partial_at}u,*totals=partials+{groups*2048}u,*inverse=totals+2048u,*htile=inverse+2048u,*gtile=htile+2048u;
  if ({name}_encode_hidden(inputs[0],hidden) || {name}_encode_gamma(inputs[1],gamma)) return {upper}_INPUT_DOMAIN;
  /* This sufficient domain proves every intermediate obligation before MMIO.
     Mean is zero or [2^-33,4]; epsilon addition is exact FP32 (<=24 bits).
     Nonzero products have bounds: normalized [2^-26,2^10], weighted
     [2^-50,2^18], prescaled [2^-82,2^50]; none under/overflow BF16. */
  for(size_t r=0;r<{rows}u;++r) {{
    uint16_t maximum=0u;
    for(size_t c=0;c<{width}u;++c) {{
      size_t a=(r*{width}u+c)*2u;uint16_t word=(uint16_t)(((uint16_t)hidden[a]|((uint16_t)hidden[a+1u]<<8))&0x7fffu);
      if(word>0x4000u) return {upper}_INPUT_DOMAIN;
      if(word>maximum) maximum=word;
    }}
    if(maximum && maximum<{threshold}u) return {upper}_INPUT_DOMAIN;
  }}
  for(size_t first=0;first<{rows}u;first+=32u) {{
    __builtin_memset(partials,0,{groups*2048}u);__builtin_memset(totals,0,2048u);
    {name}_status status;
    for(size_t ct=0;ct<{tiles}u;++ct) {{
      __builtin_memset(htile,0,2048u);
      for(size_t r=0;r<32u && first+r<{rows}u;++r)
        __builtin_memcpy(htile+r*64u,hidden+((first+r)*{width}u+ct*32u)*2u,64u);
      status={name}_square_reduce(region,htile,htile,device);if(status) return status;
      for(size_t r=0;r<32u;++r) __builtin_memcpy(partials+(ct/32u)*2048u+(r*32u+ct%32u)*2u,htile+r*64u,2u);
    }}
    for(size_t g=0;g<{groups}u;++g) {{
      status={name}_reduce(region,partials+g*2048u,htile,device);if(status) return status;
      for(size_t r=0;r<32u;++r) __builtin_memcpy(totals+(r*32u+g)*2u,htile+r*64u,2u);
    }}
    status={name}_reduce(region,totals,htile,device);if(status) return status;
    status={name}_inverse(region,htile,inverse,device);if(status) return status;
    for(size_t ct=0;ct<{tiles}u;++ct) {{
      __builtin_memset(htile,0,2048u);
      for(size_t r=0;r<32u;++r) {{
        if(first+r<{rows}u) __builtin_memcpy(htile+r*64u,hidden+((first+r)*{width}u+ct*32u)*2u,64u);
        __builtin_memcpy(gtile+r*64u,gamma+ct*64u,64u);
      }}
      status={name}_publish(region,htile,inverse,gtile,htile,device);if(status) return status;
      for(size_t r=0;r<32u && first+r<{rows}u;++r) __builtin_memcpy(result+(first+r)*{width}u+ct*32u,htile+r*32u,32u);
    }}
  }}
  /* Validate the entire native FP8 result before touching the public output. */
  for(size_t i=0;i<{count}u;++i) {{unsigned code=result[i],e=(code>>3)&15u,m=code&7u;
    if(code && (!e || (e==15u && m==7u))) return {upper}_INPUT_DOMAIN;
  }}
  for(size_t i=0;i<{count}u;++i) {{unsigned code=result[i];uint32_t bits=0u;
    if(code) {{unsigned e=(unsigned)((int)((code>>3)&15u)+120+({plan.output_scale_exponent}));
      bits=((uint32_t)(code&128u)<<24)|(e<<23)|((uint32_t)(code&7u)<<20);}}
    __builtin_memcpy((uint8_t *)out+i*4u,&bits,4u);
  }}
  return {upper}_OK;
}}
int {name}_source_dispatch(void *context,const void *const *inputs,void *const *outputs) {{
  if(!inputs || !outputs) return {upper}_INVALID;
  const float *typed[]={{inputs[0],inputs[1]}};return {name}_source_run(context,typed,outputs[0],0);
}}
'''
    converters = render_f32_array_c(operand_spec(),count,name+'_encode_hidden')+'\n'+render_f32_array_c(operand_spec(),width,name+'_encode_gamma')
    rendered = c_source_pair(name, public_headers=public,
        private_headers={'finite_float_boundary.h':_member(runtime_dir()/'c','finite_float_boundary.h').decode()},
        sources={**members,'hierarchical_rmsnorm.c':converters+implementation}, declarations=declarations)
    files.update({k:v.encode() for k,v in rendered.items()})
    for kind in STAGES:
        files['programs/'+kind+'.bin'] = stages[kind]['binary']
    files['source-plan.json'] = (json.dumps(plan.record(),sort_keys=True,indent=2)+'\n').encode()
    schedule = make_launch_schedule([sha256(stages[k]['binary']).hexdigest() for k in STAGES],
        [dict(repeat=plan.row_groups,runs=[dict(image=i,count=n) for i,n in enumerate((tiles,groups+1,1,tiles))])])
    return {**runtime, 'schema':'atlas.hierarchical_rmsnorm_c_package.v1', 'workspace_bytes':2048,
        'source_workspace_bytes':workspace, 'input_node_ids':[plan.region.hidden,plan.region.gamma],
        'output_node_ids':[plan.region.result], 'context_type':name+'_context', 'source_context_type':name+'_source_context',
        'binding_function':name+'_bind', 'source_binding_function':name+'_source_bind',
        'dispatch_function':name+'_source_dispatch','entry_function':name+'_source_run',
        'source_plan':plan.record(),'precision_plan':plan.record(),'native_program_paths':list(STAGES),
        'device_launch_schedule':schedule,
        'host_operations':['declared BF16 input conversion','comparison-only whole-array admission',
                           'completed tile byte permutation and column transport','exact FP8-to-f32 unpack'],
        'qualification':'diagnostic source-bound C transport; combined decoder qualification is separate',
        'device_buffers_by_image':[[dict(address=a,bytes=s,access=m) for a,s,m in stages[k]['ranges']] for k in STAGES]}, files
