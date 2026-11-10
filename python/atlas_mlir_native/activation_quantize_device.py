"""Exact activation quantization through ordinary virtual source contracts."""
from __future__ import annotations

from dataclasses import dataclass
from math import prod
from .binding import DRAM_BASE


def _aligned(n, alignment=32):
    return (n + alignment - 1) // alignment * alignment


@dataclass(frozen=True)
class QuantizePlan:
    source: object
    x: object
    codes: object
    exponents: object
    attrs: dict
    count: int
    width: int
    group_elements: int
    head_elements: int
    row_elements: int
    engine: str = "vector"

    @property
    def input_bytes(self):
        # Prefix transfers use complete aligned 32-element windows.
        return _aligned(self.count, 32) * self.width

    @property
    def code_address(self):
        return DRAM_BASE + self.input_bytes

    @property
    def exponent_address(self):
        return self.code_address + _aligned(self.count)

    @property
    def exponent_bytes(self):
        return _aligned(prod(self.exponents.type.shape) * 4)

    @property
    def workspace_bytes(self):
        return self.exponent_address + self.exponent_bytes - DRAM_BASE

    def record(self):
        result = dict(schema='atlas.activation_quantize_vector_plan.v1',
                    source_request_digest=self.source.digest(), count=self.count,
                    input_dtype=self.x.type.dtype, mode=self.attrs['mode'],
                    group_elements=self.group_elements, head_elements=self.head_elements,
                    row_elements=self.row_elements, input_address=DRAM_BASE,
                    input_bytes=self.input_bytes, code_address=self.code_address,
                    code_bytes=_aligned(self.count), exponent_address=self.exponent_address,
                    exponent_bytes=self.exponent_bytes, workspace_bytes=self.workspace_bytes,
                    launch_count=1, weight_conversion='none',
                    qualification='ordinary virtual pipeline and six source contracts',
                    input_addresses=[DRAM_BASE], input_payload_bytes=[self.input_bytes],
                    output_addresses=[self.code_address], output_bytes=self.workspace_bytes-self.input_bytes)
        if self.engine == 'vector':
            vector_path=self.width==2 and not any(n%32 for n in (self.count,self.group_elements,self.head_elements))
            result.update(schema='atlas.activation_quantize_vector_plan.v1',
                          engine='vector' if vector_path else 'unsupported_device_body',
                          qualification='ordinary virtual scheduling, allocation, DMA, MXU, tile, CFG, source-memory and buffer contracts',
                          exceptional_correction='read-only scalar finite-only reduction on exceptional tiles',
                          exponent_selection='scalar once per authored group')
        return result

    def descriptor(self):
        from .binding import TILE_BYTES, dram_slots
        result = dict(shape=list(self.x.type.shape), input_dtype=self.x.type.dtype,
                    mode=self.attrs['mode'], reduce=self.attrs['reduce'], rule=self.attrs['rule'],
                    rows=self.attrs['rows'], static_exp=self.attrs.get('static_exp', 0),
                    source_request_digest=self.source.digest(), io_memory_bytes=dram_slots()*TILE_BYTES)
        if self.engine != "scalar":
            result["engine"] = self.engine
        return result


def prepare_plan(source):
    from .binding import TARGET_IDENTITY, TILE_BYTES, dram_slots
    from .static_quantize import prepare_source as static
    from .dynamic_quantize import prepare_source as dynamic
    node = next((node for node in source.nodes if node.op == 'atlas.quantize_act.codes'), None)
    if node is None:
        raise ValueError('activation quantizer requires its code projection')
    if dict(node.attrs).get('mode') == 'static':
        x, codes, exponents, _ = static(source)
        attrs = dict(codes.attrs)
    else:
        x, codes, exponents, attrs = dynamic(source)
    if (source.target_identity != TARGET_IDENTITY or source.lowering_policy != 'diagnostic' or
        source.constants or x.effect != 'input' or x.op != 'input' or
        source.input_storages != ((x.id, 'external'),) or source.output_storages != ('external', 'external')):
        raise ValueError('activation quantizer requires the selected target and dense external source boundaries')
    count = prod(x.type.shape)
    if not count or count >= 1 << 28:
        raise ValueError('activation quantizer extent is outside bounded RV32 addressing')
    k = x.type.shape[-1] if x.type.shape else 1
    m = x.type.shape[-2] if len(x.type.shape) > 1 else 1
    reduce = attrs['reduce']
    group = count if attrs['mode'] == 'static' or reduce == 'tensor' else k * m if reduce == 'head' else k if reduce == 'token' else k * attrs['rows']
    head = k * m if reduce == 'head_block' else count
    plan = QuantizePlan(source, x, codes, exponents, attrs, count,
                        2 if x.type.dtype == 'bf16' else 4, min(group, head, count), head, k)
    if plan.workspace_bytes > dram_slots()*TILE_BYTES or DRAM_BASE+plan.workspace_bytes > 1 << 32:
        raise ValueError('activation quantizer buffers exceed the selected I/O window')
    return plan


def virtual_program(plan):
    from .virtual_quantizer import program
    return program(plan)


def compile_source(snapshot, region, *, target_source, destination, limits, _plan=None):
    import hashlib
    import json
    import os
    from pathlib import Path
    from merlin.semantic_compiler.codegen import current_codegen_options
    from merlin.semantic_compiler.target_binding import verify_native_publication
    from .binding import NATIVE_CHECKS, RTL_REVISION, _run, _native_pipeline, _scheduled_virtual_pipeline
    plan=_plan or prepare_plan(region)
    destination=Path(destination)
    destination.mkdir(parents=True, exist_ok=False)
    tools=Path(os.environ.get('ATLAS_OOT_BIN_DIR', Path(target_source)/'build/bin'))
    virtual=virtual_program(plan)
    pipeline = _native_pipeline()
    machine=_run([str(tools/'atlas-opt'),*_scheduled_virtual_pipeline(pipeline),'-'], input_text=virtual)
    _run([str(tools/'atlas-opt'),*NATIVE_CHECKS,'--verify-atlas-timing','-'], input_text=machine)
    program=json.loads(_run([str(tools/'atlas-emit'),'--program-json','-'], input_text=machine))
    if program.get('selected_rtl_revision') != RTL_REVISION:
        raise ValueError('activation quantizer emitter uses a different RTL revision')
    from .restoration_budget import instruction_bound
    from merlin.semantic_compiler.target_binding import NativeCompilationError
    try:
        from .limits import functional_instruction_limit
        bound=instruction_bound(virtual,machine,program,max_instructions=functional_instruction_limit(limits))
    except (ValueError,KeyError,IndexError) as error:
        raise NativeCompilationError('resource_limit','activation quantizer instruction bound is unproved: '+str(error)) from error
    if not bound['admitted']:
        raise NativeCompilationError('resource_limit','activation quantizer worst-case instruction bound exceeds '+str(bound['max_instructions'])+': '+str(bound['instruction_bound']))
    binary=b''.join(row['word_u32'].to_bytes(4,'little') for row in program['instructions'])
    budget=current_codegen_options().program_size_limit_bytes
    if budget is not None and len(binary)>budget:
        raise ValueError('activation quantizer exceeds the selected program budget')
    def encoded(record):
        return (json.dumps(record,sort_keys=True,indent=2)+'\n').encode()
    files={'source-request.json':encoded(region.record()),'quantizer-plan.json':encoded(plan.record()),
           'atlas-virtual.mlir':virtual.encode(),'atlas-machine.mlir':machine.encode(),
           'physical-program.json':encoded(program),'execution_plan.json':encoded(plan.record()),'program.bin':binary,
           'quantizer-instruction-bound.json':encoded(bound)}
    for key,data in files.items():
        (destination/key).write_bytes(data)
    artifacts={key:hashlib.sha256(data).hexdigest() for key,data in files.items()}
    manifest=dict(schema='atlas.activation_quantize_compilation.v1',engine='merlin_native',artifact_kind='program',
                  request_digest=region.digest(),target_identity=region.target_identity,compile_mode='diagnostic',
                  qualification_scope='diagnostic_only',binary_sha256=artifacts['program.bin'],
                  execution_plan_sha256=artifacts['execution_plan.json'],stage_artifacts=artifacts,
                  program_bytes=len(binary),lowering_pipeline=pipeline,
                  launches_per_invocation=1,io_memory_bytes=plan.descriptor()['io_memory_bytes'])
    (destination/'manifest.json').write_bytes(encoded(manifest))
    verify_native_publication(destination,manifest,engine='merlin_native',request_digest=region.digest(),target_identity=region.target_identity)
    return manifest


def prepare_c_package(region, compiled, name, *, target_source, runtime_linkage='separate', _plan=None):
    import hashlib
    import json
    import os
    from pathlib import Path
    from merlin.compile.c_sources import c_source_pair
    from merlin.compile.semantic_application import CompiledCRegion
    from merlin.semantic_compiler.target_binding import verify_native_publication
    from .binding import NATIVE_CHECKS, _run, _scheduled_virtual_pipeline, NATIVE_PIPELINES
    from .c_runtime import RUNTIME_SHARED_FILES, program_array, runtime_headers, runtime_members
    if runtime_linkage != 'separate':
        raise ValueError('activation quantizer exports require shared runtime linkage')
    plan=_plan or prepare_plan(region)
    compiled,target_source=Path(compiled),Path(target_source)
    manifest=json.loads((compiled/'manifest.json').read_text())
    verify_native_publication(compiled,manifest,engine='merlin_native',request_digest=region.digest(),target_identity=region.target_identity)
    if (json.loads((compiled/'source-request.json').read_text()) != region.record() or
        json.loads((compiled/'quantizer-plan.json').read_text()) != plan.record() or
        json.loads((compiled/'execution_plan.json').read_text()) != plan.record() or
        (compiled/'atlas-virtual.mlir').read_text() != virtual_program(plan)):
        raise ValueError('activation quantizer publication differs from the exact source descriptor')
    tools=Path(os.environ.get('ATLAS_OOT_BIN_DIR',target_source/'build/bin'))
    pipeline = manifest.get('lowering_pipeline')
    if pipeline not in NATIVE_PIPELINES:
        raise ValueError('unknown activation quantizer lowering pipeline')
    regenerated=_run([str(tools/'atlas-opt'),*_scheduled_virtual_pipeline(pipeline),'-'],input_text=virtual_program(plan))
    if regenerated != (compiled/'atlas-machine.mlir').read_text():
        raise ValueError('activation quantizer machine differs from replay of the exact source descriptor')
    _run([str(tools/'atlas-opt'),*NATIVE_CHECKS,'--verify-atlas-timing','-'],input_text=regenerated)
    emitted=json.loads(_run([str(tools/'atlas-emit'),'--program-json','-'],input_text=regenerated))
    binary=b''.join(row['word_u32'].to_bytes(4,'little') for row in emitted['instructions'])
    if binary != (compiled/'program.bin').read_bytes():
        raise ValueError('activation quantizer binary differs from replay of its exact typed source')
    from .restoration_budget import instruction_bound
    from .limits import functional_instruction_limit
    saved_bound=json.loads((compiled/'quantizer-instruction-bound.json').read_text())
    from types import SimpleNamespace
    cap=functional_instruction_limit(SimpleNamespace(functional_instruction_limit=saved_bound['max_instructions']))
    bound=instruction_bound(virtual_program(plan),regenerated,emitted,max_instructions=cap)
    if not bound['admitted'] or json.loads((compiled/'quantizer-instruction-bound.json').read_text())!=bound:
        raise ValueError('activation quantizer instruction bound differs or exceeds default')
    upper=name.upper()
    input_bytes=plan.count*plan.width
    exponent_bytes=prod(plan.exponents.type.shape)*4
    sizes={plan.codes.id:plan.count,plan.exponents.id:exponent_bytes}
    offsets={plan.codes.id:plan.code_address-DRAM_BASE,plan.exponents.id:plan.exponent_address-DRAM_BASE}
    # These checks mirror the existing region wrappers' full-span ABI rules.
    # In particular, table entries are read only after the complete tables and
    # mutable runtime state have been checked for aliasing.
    states=[('ctx','sizeof(*ctx)'),('region','sizeof(*region)'),
            ('region->host','sizeof(*region->host)'),('region->workspace',f'{plan.workspace_bytes}u')]
    tables=[('inputs','sizeof(*inputs)'),('outputs','2u*sizeof(*outputs)')]
    checks=[]
    for pointer,size in states+tables:
        checks.append(f'  if (!{name}_range({pointer},{size})) return {upper}_INVALID;')
    for index,(pointer,size) in enumerate(states+tables):
        for other,other_size in (states+tables)[:index]:
            checks.append(f'  if ({name}_overlap({pointer},{size},{other},{other_size})) return {upper}_INVALID;')
    values=[('inputs[0]',input_bytes)]+[(f'outputs[{index}]',sizes[key]) for index,key in enumerate(region.outputs)]
    for index,(pointer,size) in enumerate(values):
        checks.append(f'  if (!{name}_range({pointer},{size}u)) return {upper}_INVALID;')
        for other,other_size in states+tables:
            checks.append(f'  if ({name}_overlap({pointer},{size}u,{other},{other_size})) return {upper}_INVALID;')
        for other,other_size in values[:index]:
            checks.append(f'  if ({name}_overlap({pointer},{size}u,{other},{other_size}u)) return {upper}_INVALID;')
    copies='\n'.join(f'  __builtin_memcpy(outputs[{index}],region->workspace+{offsets[key]}u,{sizes[key]}u);' for index,key in enumerate(region.outputs))
    declarations=f'''
#define {upper}_WORKSPACE_BYTES {plan.workspace_bytes}u
#define {upper}_SOURCE_WORKSPACE_BYTES 1u
typedef enum {{ {upper}_OK=0, {upper}_INVALID=1, {upper}_MEMORY_ERROR=3, {upper}_DEVICE_ERROR=4 }} {name}_status;
typedef struct {{ atlas_host *host; atlas_tile_memory_io io; uint8_t *workspace; size_t max_polls; atlas_host_buffer buffers[3]; }} {name}_context;
typedef struct {{ {name}_context *region; }} {name}_source_context;
{name}_status {name}_bind({name}_context *,atlas_host *,atlas_tile_memory_io,void *,size_t,size_t);
{name}_status {name}_source_bind({name}_source_context *,{name}_context *,void *,size_t);
int {name}_source_dispatch(void *,const void *const *,void *const *);
'''
    implementation=program_array(name+'_words',binary)+f'''
static int {name}_range(const void *p,size_t n) {{ return p && n && n<=UINTPTR_MAX-(uintptr_t)p; }}
static int {name}_overlap(const void *a,size_t an,const void *b,size_t bn) {{ return (uintptr_t)a<(uintptr_t)b+bn && (uintptr_t)b<(uintptr_t)a+an; }}
{name}_status {name}_bind({name}_context *ctx,atlas_host *host,atlas_tile_memory_io io,void *workspace,size_t bytes,size_t polls) {{
  if (!{name}_range(ctx,sizeof(*ctx)) || !{name}_range(host,sizeof(*host)) ||
      !{name}_range(workspace,{plan.workspace_bytes}u) || bytes<{plan.workspace_bytes}u ||
      {name}_overlap(ctx,sizeof(*ctx),host,sizeof(*host)) ||
      {name}_overlap(ctx,sizeof(*ctx),workspace,{plan.workspace_bytes}u) ||
      {name}_overlap(host,sizeof(*host),workspace,{plan.workspace_bytes}u) ||
      host->active || !host->mmio.read32 || !host->mmio.write32 || !io.read || !io.write || !polls) return {upper}_INVALID;
  *ctx=({name}_context){{host,io,workspace,polls,{{
    {{2415919104u,{plan.input_bytes}u,ATLAS_HOST_DEVICE_READ}},
    {{{plan.code_address}u,{_aligned(plan.count)}u,ATLAS_HOST_DEVICE_WRITE}},
    {{{plan.exponent_address}u,{plan.exponent_bytes}u,ATLAS_HOST_DEVICE_WRITE}}
  }}}}; return {upper}_OK;
}}
{name}_status {name}_source_bind({name}_source_context *ctx,{name}_context *region,void *workspace,size_t bytes) {{
  (void)workspace;(void)bytes;
  if (!{name}_range(ctx,sizeof(*ctx)) || !{name}_range(region,sizeof(*region)) ||
      !{name}_range(region->host,sizeof(*region->host)) || !{name}_range(region->workspace,{plan.workspace_bytes}u) ||
      {name}_overlap(ctx,sizeof(*ctx),region,sizeof(*region)) ||
      {name}_overlap(ctx,sizeof(*ctx),region->host,sizeof(*region->host)) ||
      {name}_overlap(ctx,sizeof(*ctx),region->workspace,{plan.workspace_bytes}u) || region->host->active) return {upper}_INVALID;
  ctx->region=region;return {upper}_OK;
}}
int {name}_source_dispatch(void *opaque,const void *const *inputs,void *const *outputs) {{
  {name}_source_context *ctx=opaque;
  if (!{name}_range(ctx,sizeof(*ctx)) || !{name}_range(ctx->region,sizeof(*ctx->region))) return {upper}_INVALID;
  {name}_context *region=ctx->region;
{chr(10).join(checks)}
  if (region->host->active) return {upper}_DEVICE_ERROR;
  __builtin_memcpy(region->workspace,inputs[0],{input_bytes}u);
  __builtin_memset(region->workspace+{input_bytes}u,0,{plan.input_bytes-input_bytes}u);
  atlas_host_result result=atlas_host_ensure_loaded(region->host,{name}_words,{len(binary)//4}u);
  if (result!=ATLAS_HOST_OK) return {upper}_DEVICE_ERROR;
  if (region->io.write(region->io.context,2415919104u,region->workspace,{plan.input_bytes}u)) return {upper}_MEMORY_ERROR;
  result=atlas_host_start(region->host,region->buffers,3u);
  if (result==ATLAS_HOST_OK) result=atlas_host_wait(region->host,region->max_polls);
  if (result!=ATLAS_HOST_OK) return {upper}_DEVICE_ERROR;
  if (region->io.read(region->io.context,{plan.code_address}u,region->workspace+{plan.code_address-DRAM_BASE}u,{_aligned(plan.count)}u) ||
      region->io.read(region->io.context,{plan.exponent_address}u,region->workspace+{plan.exponent_address-DRAM_BASE}u,{plan.exponent_bytes}u)) return {upper}_MEMORY_ERROR;
{copies}
  return {upper}_OK;
}}
'''
    public=runtime_headers(target_source/'runtime')
    _,runtime_files,_=runtime_members(public,(target_source/'runtime/atlas_host.c').read_text(),runtime_linkage)
    files={key:value.encode() for key,value in c_source_pair(name,public_headers=public,private_headers={},
            sources={'activation_quantize.c':implementation},declarations=declarations).items()}
    files.update(runtime_files)
    description=dict(kind='activation_quantize_device',context_type=name+'_context',source_context_type=name+'_source_context',
                     binding_function=name+'_bind',source_binding_function=name+'_source_bind',dispatch_function=name+'_source_dispatch',
                     workspace_bytes=plan.workspace_bytes,source_workspace_bytes=1,launches_per_call=1,
                     input_node_ids=[plan.x.id],output_node_ids=list(region.outputs),weight_conversion='none',
                     native_manifest_sha256=hashlib.sha256((compiled/'manifest.json').read_bytes()).hexdigest())
    return CompiledCRegion(name+'_source_dispatch',name+'.h',(plan.x.id,),tuple(region.outputs),files,
                           RUNTIME_SHARED_FILES,('segments/'+name,),description)
