"""Exact activation quantization through ordinary virtual source contracts."""
from __future__ import annotations

from .binding import io_memory_base as selected_io_base

from dataclasses import dataclass
from math import prod
from .binding import DRAM_BASE
from .shared_dispatch_c import REGION_TYPES, guarded


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
    io_memory_base: int = DRAM_BASE

    @property
    def input_bytes(self):
        # Prefix transfers use complete aligned 32-element windows.
        return _aligned(self.count, 32) * self.width

    @property
    def code_address(self):
        return self.io_memory_base + self.input_bytes

    @property
    def exponent_address(self):
        return self.code_address + _aligned(self.count)

    @property
    def exponent_bytes(self):
        return _aligned(prod(self.exponents.type.shape) * 4)

    @property
    def workspace_bytes(self):
        return self.exponent_address + self.exponent_bytes - self.io_memory_base

    def record(self):
        result = dict(schema='atlas.activation_quantize_vector_plan.v1',
                    source_request_digest=self.source.digest(), count=self.count,
                    input_dtype=self.x.type.dtype, mode=self.attrs['mode'],
                    group_elements=self.group_elements, head_elements=self.head_elements,
                    row_elements=self.row_elements, input_address=self.io_memory_base,
                    input_bytes=self.input_bytes, code_address=self.code_address,
                    code_bytes=_aligned(self.count), exponent_address=self.exponent_address,
                    exponent_bytes=self.exponent_bytes, workspace_bytes=self.workspace_bytes,
                    launch_count=1, weight_conversion='none',
                    qualification='ordinary virtual pipeline and six source contracts',
                    input_addresses=[self.io_memory_base], input_payload_bytes=[self.input_bytes],
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


def prepare_plan(source, *, io_memory_base=None):
    io_memory_base = selected_io_base() if io_memory_base is None else io_memory_base
    if type(io_memory_base) is not int or io_memory_base % 32:
        raise ValueError("I/O memory base requires an aligned integer address")
    from .binding import validate_explicit_io_base
    validate_explicit_io_base(io_memory_base)
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
                        2 if x.type.dtype == 'bf16' else 4, min(group, head, count), head, k, io_memory_base=io_memory_base)
    from .memory_windows import dma_span_in_window
    if (type(io_memory_base) is not int or io_memory_base % 32 or
        plan.workspace_bytes > dram_slots()*TILE_BYTES or
        not dma_span_in_window(io_memory_base, plan.workspace_bytes)):
        raise ValueError('activation quantizer buffers exceed the selected I/O window')
    return plan


def virtual_program(plan):
    from merlin.compiler.semantic.target_binding import NativeCompilationError
    from .virtual_quantizer import program
    try:
        return program(plan)
    except ValueError as error:
        # A shape this body cannot cover is a named refusal, so a caller choosing among
        # slice widths can try another width.
        raise NativeCompilationError('unsupported_semantics', str(error)) from error


def prove_plan(plan, *, target_source=None, limits=None, _pipeline=None):
    """Lower and prove a body without publishing files or creating directories."""
    import json
    import os
    from pathlib import Path
    from merlin.compiler.semantic.target_binding import NativeCompilationError
    from .binding import NATIVE_CHECKS, RTL_REVISION, _run, _native_pipeline, _scheduled_virtual_pipeline
    if target_source is None:
        target_source = Path(__file__).resolve().parents[2]
    tools=Path(os.environ.get('ATLAS_OOT_BIN_DIR', Path(target_source)/'build/bin'))
    virtual=virtual_program(plan)
    pipeline = _native_pipeline() if _pipeline is None else _pipeline
    if pipeline not in ("mandatory", "optimized"):
        raise ValueError("unknown quantizer proof pipeline")
    machine=_run([str(tools/'atlas-opt'),*_scheduled_virtual_pipeline(pipeline),'-'], input_text=virtual)
    _run([str(tools/'atlas-opt'),*NATIVE_CHECKS,'--verify-atlas-timing','-'], input_text=machine)
    emitted=_run([str(tools/'atlas-emit'),'--program-json','-'], input_text=machine)
    try:
        program=json.loads(emitted)
    except json.JSONDecodeError as error:
        raise NativeCompilationError('compile_error','activation quantizer emitter returned invalid program JSON: '+str(error)) from error
    if not isinstance(program, dict):
        raise NativeCompilationError('compile_error','activation quantizer emitter returned a non-object program')
    if program.get('selected_rtl_revision') != RTL_REVISION:
        raise NativeCompilationError('unqualified_target','activation quantizer emitter uses a different RTL revision')
    from .restoration_budget import instruction_bound
    try:
        from .limits import functional_instruction_limit
        bound=instruction_bound(virtual,machine,program,max_instructions=functional_instruction_limit(limits))
    except (ValueError,KeyError,IndexError) as error:
        raise NativeCompilationError('compile_error','activation quantizer instruction bound is unproved: '+str(error)) from error
    if not bound['admitted']:
        raise NativeCompilationError('resource_limit','activation quantizer worst-case instruction bound exceeds '+str(bound['max_instructions'])+': '+str(bound['instruction_bound']))
    return virtual, machine, program, bound, pipeline


def compile_source(snapshot, region, *, target_source, destination, limits, io_memory_base=None, _plan=None):
    io_memory_base = selected_io_base() if io_memory_base is None else io_memory_base
    import hashlib
    import json
    from pathlib import Path
    from merlin.compiler.semantic.codegen import current_codegen_options
    from merlin.compiler.semantic.target_binding import verify_native_publication
    plan=_plan or prepare_plan(region, io_memory_base=io_memory_base)
    virtual,machine,program,bound,pipeline=prove_plan(plan,target_source=target_source,limits=limits)
    destination=Path(destination)
    destination.mkdir(parents=True, exist_ok=False)
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


QUANTIZE_TYPES = REGION_TYPES + guarded("quantize_types",
    "typedef struct { atlas_region_context *region; size_t input_first; } atlas_quantize_source_context;\n")
# One checked launch: pack the input, load the image, start, wait, and read codes and exponents.
QUANTIZE_DISPATCH = guarded("quantize_dispatch", """typedef struct {
  const uint32_t *words; uint32_t word_count,copy_bytes,input_bytes,count;
  int (*pack)(void *,const void *,size_t,size_t);
  atlas_dispatch_shape shape; uint32_t output_offset[2],output_bytes[2];
  uint32_t input_address,code_address,code_bytes,exponent_address,exponent_bytes;
} atlas_quantize_plan;
static __attribute__((unused)) int atlas_quantize_bind(atlas_region_context *ctx,atlas_host *host,const atlas_tile_memory_io *io,void *workspace,
                               size_t bytes,size_t polls,const atlas_quantize_plan *plan) {
  if (atlas_bind_conflict(ctx,sizeof(*ctx),host,workspace,plan->shape.workspace_bytes,bytes,io,polls)) return 1;
  *ctx=(atlas_region_context){host,*io,workspace,polls}; return 0;
}
static __attribute__((unused)) int atlas_quantize_source_bind(atlas_quantize_source_context *ctx,atlas_region_context *region,const atlas_quantize_plan *plan) {
  if (atlas_source_bind_conflict(ctx,sizeof(*ctx),region,sizeof(*region),offsetof(atlas_region_context,host),offsetof(atlas_region_context,workspace),
                                 plan->shape.workspace_bytes)) return 1;
  ctx->region=region;ctx->input_first=0;return 0;
}
static __attribute__((unused)) int atlas_quantize_dispatch(const atlas_quantize_plan *plan,void *opaque,const void *const *inputs,void *const *outputs) {
  atlas_quantize_source_context *ctx=opaque;
  if (!atlas_span_valid(ctx,sizeof(*ctx)) || !atlas_span_valid(ctx->region,sizeof(*ctx->region))) return 1;
  atlas_region_context *region=ctx->region;
  if (atlas_dispatch_conflict(&plan->shape,ctx,region,region->host,region->workspace,inputs,outputs)) return 1;
  if (region->host->active) return 4;
  if (plan->pack) { if (plan->pack(region->workspace,inputs[0],ctx->input_first,plan->count)) return 1; }
  else __builtin_memcpy(region->workspace,inputs[0],plan->copy_bytes);
  __builtin_memset(region->workspace+plan->copy_bytes,0,plan->input_bytes-plan->copy_bytes);
  if (atlas_host_ensure_loaded(region->host,plan->words,plan->word_count)!=ATLAS_HOST_OK) return 4;
  if (region->io.write(region->io.context,plan->input_address,region->workspace,plan->input_bytes)) return 3;
  const atlas_host_buffer buffers[3]={{plan->input_address,plan->input_bytes,ATLAS_HOST_DEVICE_READ},
      {plan->code_address,plan->code_bytes,ATLAS_HOST_DEVICE_WRITE},{plan->exponent_address,plan->exponent_bytes,ATLAS_HOST_DEVICE_WRITE}};
  atlas_host_result result=atlas_host_start(region->host,buffers,3u);
  if (result==ATLAS_HOST_OK) result=atlas_host_wait(region->host,region->max_polls);
  if (result!=ATLAS_HOST_OK) return 4;
  for (size_t i=1;i<3u;++i)
    if (region->io.read(region->io.context,buffers[i].address,
                        region->workspace+(buffers[i].address-plan->input_address),buffers[i].size_bytes)) return 3;
  for (size_t i=0;i<2u;++i) __builtin_memcpy(outputs[i],region->workspace+plan->output_offset[i],plan->output_bytes[i]);
  return 0;
}
""")


def prepare_c_package(region, compiled, name, *, target_source, runtime_linkage='separate', io_memory_base=None, _plan=None, input_permutation=None):
    io_memory_base = selected_io_base() if io_memory_base is None else io_memory_base
    import hashlib
    import json
    import os
    from pathlib import Path
    from merlin.compiler.lower.c.dispatch import c_source_pair
    from merlin.compiler.build.regions import CompiledCRegion
    from merlin.compiler.semantic.target_binding import verify_native_publication
    from .binding import NATIVE_CHECKS, _run, _scheduled_virtual_pipeline, NATIVE_PIPELINES
    from .c_runtime import RUNTIME_SHARED_FILES, program_array, runtime_headers, runtime_members
    if runtime_linkage != 'separate':
        raise ValueError('activation quantizer exports require shared runtime linkage')
    plan=_plan or prepare_plan(region, io_memory_base=io_memory_base)
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
    source_bytes=input_bytes
    packing_helper=''
    if input_permutation is not None:
        from merlin.compiler.lower.c.layout import render_permutation_slice_copy
        physical,logical=input_permutation
        if ((logical.type.dtype,logical.type.numerical_policy) !=
                (plan.x.type.dtype,plan.x.type.numerical_policy)
                or plan.count > prod(logical.type.shape)):
            raise ValueError('quantizer input slice differs from the logical permutation')
        packing_helper=render_permutation_slice_copy(physical,logical,name+'_pack_input')
        source_bytes=prod(physical.type.shape)*plan.width
    exponent_bytes=prod(plan.exponents.type.shape)*4
    sizes={plan.codes.id:plan.count,plan.exponents.id:exponent_bytes}
    offsets={plan.codes.id:plan.code_address-plan.io_memory_base,plan.exponents.id:plan.exponent_address-plan.io_memory_base}
    from .shared_dispatch_c import CHECKS
    pack=name+'_pack_input' if input_permutation is not None else '0'
    outputs=[(offsets[key],sizes[key]) for key in region.outputs]
    if len(outputs) != 2:
        raise ValueError('activation quantizer publishes codes and exponents')
    value_bytes=','.join(f'{size}u' for size in [source_bytes,*(size for _,size in outputs)])
    declarations=f'''{QUANTIZE_TYPES}
#define {upper}_WORKSPACE_BYTES {plan.workspace_bytes}u
#define {upper}_SOURCE_WORKSPACE_BYTES 1u
typedef enum {{ {upper}_OK=0, {upper}_INVALID=1, {upper}_MEMORY_ERROR=3, {upper}_DEVICE_ERROR=4 }} {name}_status;
typedef atlas_region_context {name}_context;
typedef atlas_quantize_source_context {name}_source_context;
{name}_status {name}_bind({name}_context *,atlas_host *,atlas_tile_memory_io,void *,size_t,size_t);
{name}_status {name}_source_bind({name}_source_context *,{name}_context *,void *,size_t);
int {name}_source_dispatch(void *,const void *const *,void *const *);
'''
    implementation=program_array(name+'_words',binary)+f'''
{packing_helper}
{CHECKS}{QUANTIZE_DISPATCH}
static const uint32_t {name}_value_bytes[]={{{value_bytes}}};
static const atlas_quantize_plan {name}_plan={{{name}_words,{len(binary)//4}u,{input_bytes}u,{plan.input_bytes}u,{plan.count}u,{pack},
  {{sizeof(atlas_quantize_source_context),sizeof(atlas_region_context),{plan.workspace_bytes}u,1u,2u,{name}_value_bytes}},
  {{{outputs[0][0]}u,{outputs[1][0]}u}},{{{outputs[0][1]}u,{outputs[1][1]}u}},
  {plan.io_memory_base}u,{plan.code_address}u,{_aligned(plan.count)}u,{plan.exponent_address}u,{plan.exponent_bytes}u}};
{name}_status {name}_bind({name}_context *ctx,atlas_host *host,atlas_tile_memory_io io,void *workspace,size_t bytes,size_t polls) {{
  return ({name}_status)atlas_quantize_bind(ctx,host,&io,workspace,bytes,polls,&{name}_plan);
}}
{name}_status {name}_source_bind({name}_source_context *ctx,{name}_context *region,void *workspace,size_t bytes) {{
  (void)workspace;(void)bytes; return ({name}_status)atlas_quantize_source_bind(ctx,region,&{name}_plan);
}}
int {name}_source_dispatch(void *opaque,const void *const *inputs,void *const *outputs) {{
  return atlas_quantize_dispatch(&{name}_plan,opaque,inputs,outputs);
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
