"""Exact source float operations through the common Atlas virtual pipeline.

Every checked operation scans its whole source domain before storing results.
NaNs and invalid infinity cancellation produce a device status with no output
writes. Sign flips accept all encodings and preserve NaN payloads.
"""
from dataclasses import dataclass
from math import prod
from .binding import DRAM_BASE
from .activation_quantize_device import _aligned

OPS = {"bf16_to_f32": (2,4,"bf16-to-f32-exact-v1"),
       "f32_to_bf16": (4,2,"f32-to-bf16-rne-v1"),
       "bf16_neg": (2,2,"bf16-sign-bit-flip-v1"),
       "f32_neg": (4,4,"f32-sign-bit-flip-v1"),
       "bf16_add": (2,2,"ieee754-bf16-add-rne-v1"),
       "bf16_sub": (2,2,"ieee754-bf16-sub-rne-v1")}

# Tile-aligned policies leave at least 3% of the unchanged instruction budget
# unused in the checked conversion and sign-flip bodies. Binary operations
# retain their existing tile size because both numeric arms are charged.
MAX_LEAF_ELEMENTS = {"bf16_to_f32": 20480, "f32_to_bf16": 18432,
                     "bf16_neg": 43008, "f32_neg": 43008,
                     "bf16_add": 1024, "bf16_sub": 1024}


def chunk_elements(plan):
    """Respect both the operation policy and the selected I/O window."""
    available = plan.descriptor()["io_memory_bytes"] - 32
    per_element = len(plan.operands) * plan.width + plan.output_width
    capacity = available // per_element
    capacity -= capacity % plan.tile_lanes
    if capacity <= 0:
        raise ValueError("exact float cannot fit a tile in selected I/O window")
    return min(MAX_LEAF_ELEMENTS[plan.op], capacity)

@dataclass(frozen=True)
class FloatPlan:
    source: object
    node: object
    operands: tuple
    count: int
    width: int
    output_width: int
    @property
    def op(self): return self.node.op
    @property
    def tile_lanes(self): return min(1024,2048//max(self.width,self.output_width),_aligned(self.count,32))
    @property
    def input_bytes(self): return _aligned(self.count,self.tile_lanes)*self.width
    @property
    def output_address(self): return DRAM_BASE+len(self.operands)*self.input_bytes
    @property
    def output_bytes(self): return _aligned(self.count,self.tile_lanes)*self.output_width
    @property
    def status_address(self): return self.output_address+self.output_bytes
    @property
    def workspace_bytes(self): return self.status_address+32-DRAM_BASE
    def descriptor(self):
        from .binding import TILE_BYTES,dram_slots
        return dict(operation=self.op,count=self.count,input_count=len(self.operands),right_input=[n.id for n in self.operands].index(self.node.inputs[-1]),source_request_digest=self.source.digest(),io_memory_bytes=dram_slots()*TILE_BYTES)
    def record(self):
        return dict(schema="atlas.exact_float.v1",source_request_digest=self.source.digest(),operation=self.op,
                    input_node_ids=[n.id for n in self.operands],output_node_ids=[self.node.id],
                    input_addresses=[DRAM_BASE+i*self.input_bytes for i in range(len(self.operands))],
                    input_payload_bytes=[self.input_bytes]*len(self.operands),output_addresses=[self.output_address],
                    output_bytes=self.output_bytes+32,result_buffer_bytes=self.output_bytes,status_address=self.status_address,workspace_bytes=self.workspace_bytes,
                    logical_output_bytes=self.count*self.output_width,runtime_arithmetic="none",launches_per_invocation=1,
                    boundary_transport_layout="contiguous-full-tile-DMA-with-private-zero-padding",
                    arithmetic_routes=["qualified-residual-vector", "pair-row-sum-vector", "exceptional-scalar"] if self.op in ("bf16_add","bf16_sub") else ["scalar-bit-arithmetic"],
                    verification="common-virtual-scheduling-allocation-and-all-generated-contracts",
                    vector_admission={"qualified":"signed-zero or abs-code2f80..4f80",
                                      "pair":"normal-or-signed-zero inputs except both-negative-zero; private result must be normal or provably exact zero"} if self.op in ("bf16_add","bf16_sub") else None)

def prepare_plan(source):
    from .binding import TARGET_IDENTITY,TILE_BYTES,dram_slots
    from merlin.semantic_compiler.model import TensorType,pointwise_identity_maps
    pure=[n for n in source.nodes if n.effect=="pure"]
    if len(pure)!=1 or pure[0].op not in OPS: raise ValueError("exact float needs one supported operation")
    n=pure[0]; width,ow,policy=OPS[n.op]; keys=tuple(dict.fromkeys(n.inputs))
    binary=n.op in ("bf16_add","bf16_sub")
    operands=tuple(source.node(k) for k in keys)
    if (source.target_identity!=TARGET_IDENTITY or source.lowering_policy!="diagnostic" or
        source.constants or source.physical_extents or source.boundary_layouts or source.output_layouts or
        not n.type.shape or not 0<prod(n.type.shape)<1<<27 or len(n.inputs)!=(2 if binary else 1) or
        source.outputs!=(n.id,) or source.output_storages!=("external",) or
        source.input_storages!=tuple((k,"external") for k in keys) or
        {v.id for v in source.nodes}!={*keys,n.id} or
        n.type.numerical_policy!="raw-bit-pattern" or dict(n.attrs)!={"numeric_semantics":policy} or
        n.index_maps!=pointwise_identity_maps(len(n.type.shape),len(n.inputs)) or
        n.type.dtype!=("f32" if n.op=="bf16_to_f32" else "bf16" if n.op=="f32_to_bf16" else n.op.split("_")[0]) or
        any(v.op!="input" or v.effect!="input" or v.inputs or v.attrs or v.index_maps or
            v.type!=TensorType(n.type.shape,"bf16" if width==2 else "f32","raw-bit-pattern") for v in operands)):
        raise ValueError("exact float source types, maps, domain, or boundaries differ")
    plan=FloatPlan(source,n,operands,prod(n.type.shape),width,ow)
    return plan

def virtual_program(p):
    from .float_virtual import virtual_program
    return virtual_program(p)


def _compute(b,p,left,right=None):
    c=b.constant; op=b.binary
    if p.op=='bf16_to_f32': return op('shl',left,c(16))
    if p.op=='f32_to_bf16':
        odd=op('and',op('shr',left,c(16)),c(1))
        return op('shr',op('add',left,op('add',odd,c(32767))),c(16))
    if p.op.endswith('_neg'): return op('xor',left,c(1<<(8*p.width-1)))
    return _add(b,left,right)

def _valid(b,p,left,right=None):
    c=b.constant; op=b.binary; cmp=b.compare
    mask=c(32767 if p.width==2 else 0x7fffffff)
    infinity=c(32640 if p.width==2 else 0x7f800000)
    a=op('and',left,mask); bad=cmp('ult',infinity,a)
    if p.op in ('bf16_add','bf16_sub'):
        d=op('and',right,mask)
        bad=op('or',bad,cmp('ult',infinity,d))
        both_inf=op('and',cmp('eq',a,infinity),cmp('eq',d,infinity))
        opposite=op('shr',op('xor',left,right),c(15))
        bad=op('or',bad,op('and',both_inf,opposite))
    return bad

def _add(b,left,right):
    """Align exact significands with 16 guard bits and one sticky bit, then RNE."""
    c=b.constant; op=b.binary; cmp=b.compare; choose=b.select
    def imm(kind,value,n): return op(kind,value,c(n))
    swap=cmp('ult',imm('and',left,32767),imm('and',right,32767))
    a=choose(right,left,swap); d=choose(left,right,swap)
    sign=imm('shl',imm('shr',a,15),15)
    ae=imm('and',imm('shr',a,7),255); de=imm('and',imm('shr',d,7),255)
    exponent=op('add',ae,cmp('eq',ae,c(0)))
    small_exp=op('add',de,cmp('eq',de,c(0)))
    gap=op('sub',exponent,small_exp)
    large=imm('shl',op('or',imm('and',a,127),imm('shl',cmp('ult',c(0),ae),7)),16)
    small=imm('shl',op('or',imm('and',d,127),imm('shl',cmp('ult',c(0),de),7)),16)
    gap=choose(gap,c(24),cmp('ult',gap,c(24)))
    shifted=op('shr',small,gap)
    sticky=cmp('ult',c(0),op('xor',op('shl',shifted,gap),small))
    small=op('or',shifted,sticky)
    opposite=imm('shr',op('xor',a,d),15)
    significand=choose(op('sub',large,small),op('add',large,small),opposite)
    carry=imm('shr',significand,24)
    rounded_shift=op('or',imm('shr',significand,1),imm('and',significand,1))
    significand=choose(rounded_shift,significand,carry)
    exponent=op('add',exponent,carry)
    for _ in range(8):
        normalize=op('and',cmp('ult',significand,c(128<<16)),cmp('ule',c(2),exponent))
        significand=choose(imm('shl',significand,1),significand,normalize)
        exponent=op('sub',exponent,normalize)
    rounded=imm('shr',significand,16); remainder=imm('and',significand,65535)
    up=op('or',cmp('ult',c(32768),remainder),op('and',cmp('eq',remainder,c(32768)),imm('and',rounded,1)))
    rounded=op('add',rounded,up); carry=imm('shr',rounded,8)
    exponent=op('add',exponent,carry); rounded=choose(imm('shr',rounded,1),rounded,carry)
    encoded=op('add',imm('shl',imm('sub',exponent,1),7),rounded)
    rounded=choose(rounded,encoded,cmp('ult',rounded,c(128)))
    rounded=choose(rounded,c(32640),cmp('ult',exponent,c(255)))
    rounded=op('or',rounded,sign)
    zero_sign=imm('and',op('and',a,d),32768)
    rounded=choose(zero_sign,rounded,cmp('eq',significand,c(0)))
    rounded=choose(d,rounded,cmp('eq',imm('and',d,32767),c(32640)))
    return choose(a,rounded,cmp('eq',imm('and',a,32767),c(32640)))


def machine_program(p):
    """Compile the SSA program through the standard target pipeline."""
    import os
    from pathlib import Path
    from .binding import _scheduled_virtual_pipeline, _run
    tools=Path(os.environ.get("ATLAS_OOT_BIN_DIR",Path(__file__).resolve().parents[2]/"build/bin"))
    return _run([str(tools/"atlas-opt"),*_scheduled_virtual_pipeline(),"-"],input_text=virtual_program(p))


def compile_source(snapshot,region,*,target_source,destination,limits):
    import hashlib,json,os
    from pathlib import Path
    from merlin.semantic_compiler.codegen import current_codegen_options
    from merlin.semantic_compiler.target_binding import verify_native_publication
    from .binding import NATIVE_CHECKS,RTL_REVISION,_run
    p=prepare_plan(region)
    maximum=MAX_LEAF_ELEMENTS[p.op]
    if p.count>maximum: raise ValueError(f"exact float leaf requires at most {maximum} elements")
    if p.workspace_bytes>p.descriptor()["io_memory_bytes"]: raise ValueError("exact float exceeds selected I/O window")
    tools=Path(os.environ.get("ATLAS_OOT_BIN_DIR",Path(target_source)/"build/bin"))
    virtual=virtual_program(p)
    machine=machine_program(p)
    _run([str(tools/"atlas-opt"),*NATIVE_CHECKS,"--verify-atlas-timing","-"],input_text=machine)
    program=json.loads(_run([str(tools/"atlas-emit"),"--program-json","-"],input_text=machine))
    if program.get("selected_rtl_revision")!=RTL_REVISION: raise ValueError("exact float emitter RTL differs")
    from .restoration_budget import instruction_bound
    from merlin.semantic_compiler.target_binding import NativeCompilationError
    try:
        bound=instruction_bound(virtual,machine,program)
    except (ValueError,KeyError,IndexError) as error:
        raise NativeCompilationError("resource_limit","exact float instruction bound is unproved: "+str(error)) from error
    if not bound["admitted"]:
        raise NativeCompilationError("resource_limit","exact float worst-case instruction bound exceeds 1000000: "+str(bound["instruction_bound"]))
    binary=b"".join(r["word_u32"].to_bytes(4,"little") for r in program["instructions"])
    budget=current_codegen_options().program_size_limit_bytes
    if budget is not None and len(binary)>budget: raise ValueError("exact float exceeds selected program budget")
    encoded=lambda r:(json.dumps(r,sort_keys=True,indent=2)+"\n").encode()
    files={"source-request.json":encoded(region.record()),"float-plan.json":encoded(p.record()),
           "execution_plan.json":encoded(p.record()),"atlas-virtual.mlir":virtual.encode(),
           "atlas-machine.mlir":machine.encode(),"physical-program.json":encoded(program),"program.bin":binary,
           "float-instruction-bound.json":encoded(bound)}
    destination=Path(destination);destination.mkdir(parents=True,exist_ok=False)
    for filename,data in files.items():(destination/filename).write_bytes(data)
    artifacts={k:hashlib.sha256(v).hexdigest() for k,v in files.items()}
    manifest=dict(schema="atlas.exact_float_compilation.v1",engine="merlin_native",artifact_kind="program",
                  request_digest=region.digest(),target_identity=region.target_identity,compile_mode="diagnostic",
                  qualification_scope="diagnostic_only",binary_sha256=artifacts["program.bin"],
                  execution_plan_sha256=artifacts["execution_plan.json"],stage_artifacts=artifacts,
                  program_bytes=len(binary),launches_per_invocation=1,io_memory_bytes=p.descriptor()["io_memory_bytes"])
    (destination/"manifest.json").write_bytes(encoded(manifest))
    verify_native_publication(destination,manifest,engine="merlin_native",request_digest=region.digest(),target_identity=region.target_identity)
    return manifest


def prepare_c_package(region,compiled,name,*,target_source,runtime_linkage="separate"):
    import hashlib,json,os
    from pathlib import Path
    from merlin.compile.c_sources import c_source_pair
    from merlin.compile.semantic_application import CompiledCRegion
    from merlin.semantic_compiler.target_binding import verify_native_publication
    from .binding import NATIVE_CHECKS,_run
    from .c_runtime import RUNTIME_SHARED_FILES,program_array,runtime_headers,runtime_members
    if runtime_linkage!="separate":raise ValueError("exact float requires shared runtime linkage")
    p=prepare_plan(region);compiled,target_source=Path(compiled),Path(target_source)
    manifest=json.loads((compiled/"manifest.json").read_text())
    verify_native_publication(compiled,manifest,engine="merlin_native",request_digest=region.digest(),target_identity=region.target_identity)
    if (json.loads((compiled/"source-request.json").read_text())!=region.record() or
        json.loads((compiled/"float-plan.json").read_text())!=p.record() or
        json.loads((compiled/"execution_plan.json").read_text())!=p.record() or
        (compiled/"atlas-virtual.mlir").read_text()!=virtual_program(p)):
        raise ValueError("exact float publication differs from complete source descriptor")
    tools=Path(os.environ.get("ATLAS_OOT_BIN_DIR",target_source/"build/bin"))
    machine=machine_program(p)
    if machine!=(compiled/"atlas-machine.mlir").read_text():raise ValueError("exact float machine differs from source replay")
    _run([str(tools/"atlas-opt"),*NATIVE_CHECKS,"--verify-atlas-timing","-"],input_text=machine)
    program=json.loads(_run([str(tools/"atlas-emit"),"--program-json","-"],input_text=machine))
    binary=b"".join(r["word_u32"].to_bytes(4,"little") for r in program["instructions"])
    if binary!=(compiled/"program.bin").read_bytes():raise ValueError("exact float binary differs from source replay")
    from .restoration_budget import instruction_bound
    bound=instruction_bound(virtual_program(p),machine,program)
    if not bound["admitted"] or json.loads((compiled/"float-instruction-bound.json").read_text())!=bound:
        raise ValueError("exact float instruction bound differs or exceeds default")
    upper=name.upper();n=len(p.operands);logical=p.count*p.width;obytes=p.count*p.output_width
    states=[("ctx","sizeof(*ctx)"),("region","sizeof(*region)"),("region->host","sizeof(*region->host)"),
            ("region->workspace",f"{p.workspace_bytes}u"),("inputs",f"{n}u*sizeof(*inputs)"),("outputs","sizeof(*outputs)")]
    checks=[f"  if (!{name}_range({ptr},{size})) return {upper}_INVALID;" for ptr,size in states]
    for i,(ptr,size) in enumerate(states):
        for other,osize in states[:i]:checks.append(f"  if ({name}_overlap({ptr},{size},{other},{osize})) return {upper}_INVALID;")
    values=[(f"inputs[{i}]",logical) for i in range(n)]+[("outputs[0]",obytes)]
    for i,(ptr,size) in enumerate(values):
        checks.append(f"  if (!{name}_range({ptr},{size}u)) return {upper}_INVALID;")
        for other,osize in states:checks.append(f"  if ({name}_overlap({ptr},{size}u,{other},{osize})) return {upper}_INVALID;")
        # Source operands may alias each other, as permitted by the source ABI.
        if ptr.startswith("outputs"):
            for other,osize in values[:i]:checks.append(f"  if ({name}_overlap({ptr},{size}u,{other},{osize}u)) return {upper}_INVALID;")
    buffers=[f"{{{DRAM_BASE+i*p.input_bytes}u,{p.input_bytes}u,ATLAS_HOST_DEVICE_READ}}" for i in range(n)]
    buffers += [f"{{{p.output_address}u,{p.output_bytes+32}u,ATLAS_HOST_DEVICE_WRITE}}"]
    declarations=f'''
#define {upper}_WORKSPACE_BYTES {p.workspace_bytes}u
#define {upper}_SOURCE_WORKSPACE_BYTES 1u
typedef enum {{ {upper}_OK=0,{upper}_INVALID=1,{upper}_DOMAIN=2,{upper}_MEMORY_ERROR=3,{upper}_DEVICE_ERROR=4 }} {name}_status;
typedef struct {{ atlas_host *host; atlas_tile_memory_io io; uint8_t *workspace; size_t max_polls; atlas_host_buffer buffers[{n+1}]; }} {name}_context;
typedef struct {{ {name}_context *region; }} {name}_source_context;
{name}_status {name}_bind({name}_context *,atlas_host *,atlas_tile_memory_io,void *,size_t,size_t);
{name}_status {name}_source_bind({name}_source_context *,{name}_context *,void *,size_t);
int {name}_source_dispatch(void *,const void *const *,void *const *);
'''
    copies=[]
    for i in range(n):
        at=i*p.input_bytes
        copies.append(f'''  __builtin_memcpy(region->workspace+{at}u,inputs[{i}],{logical}u);
  __builtin_memset(region->workspace+{at+logical}u,0,{p.input_bytes-logical}u);
  if (region->io.write(region->io.context,{DRAM_BASE+at}u,region->workspace+{at}u,{p.input_bytes}u)) return {upper}_MEMORY_ERROR;''')
    implementation=program_array(name+"_words",binary)+f'''
static int {name}_range(const void *p,size_t n) {{ return p && n && n<=UINTPTR_MAX-(uintptr_t)p; }}
static int {name}_overlap(const void *a,size_t an,const void *b,size_t bn) {{ return (uintptr_t)a<(uintptr_t)b+bn && (uintptr_t)b<(uintptr_t)a+an; }}
{name}_status {name}_bind({name}_context *ctx,atlas_host *host,atlas_tile_memory_io io,void *workspace,size_t bytes,size_t polls) {{
  if (!{name}_range(ctx,sizeof(*ctx)) || !{name}_range(host,sizeof(*host)) || !{name}_range(workspace,{p.workspace_bytes}u) || bytes<{p.workspace_bytes}u ||
      {name}_overlap(ctx,sizeof(*ctx),host,sizeof(*host)) || {name}_overlap(ctx,sizeof(*ctx),workspace,{p.workspace_bytes}u) ||
      {name}_overlap(host,sizeof(*host),workspace,{p.workspace_bytes}u) || host->active || !host->mmio.read32 || !host->mmio.write32 || !io.read || !io.write || !polls) return {upper}_INVALID;
  *ctx=({name}_context){{host,io,workspace,polls,{{{','.join(buffers)}}}}};return {upper}_OK;
}}
{name}_status {name}_source_bind({name}_source_context *ctx,{name}_context *region,void *workspace,size_t bytes) {{
  (void)workspace;(void)bytes;
  if (!{name}_range(ctx,sizeof(*ctx)) || !{name}_range(region,sizeof(*region)) || !{name}_range(region->host,sizeof(*region->host)) ||
      !{name}_range(region->workspace,{p.workspace_bytes}u) || {name}_overlap(ctx,sizeof(*ctx),region,sizeof(*region)) ||
      {name}_overlap(ctx,sizeof(*ctx),region->host,sizeof(*region->host)) || {name}_overlap(ctx,sizeof(*ctx),region->workspace,{p.workspace_bytes}u) || region->host->active) return {upper}_INVALID;
  ctx->region=region;return {upper}_OK;
}}
int {name}_source_dispatch(void *opaque,const void *const *inputs,void *const *outputs) {{
  {name}_source_context *ctx=opaque;
  if (!{name}_range(ctx,sizeof(*ctx)) || !{name}_range(ctx->region,sizeof(*ctx->region))) return {upper}_INVALID;
  {name}_context *region=ctx->region;
{chr(10).join(checks)}
  if (region->host->active) return {upper}_DEVICE_ERROR;
{chr(10).join(copies)}
  atlas_host_result status=atlas_host_ensure_loaded(region->host,{name}_words,{len(binary)//4}u);
  if (status==ATLAS_HOST_OK) status=atlas_host_start(region->host,region->buffers,{n+1}u);
  if (status==ATLAS_HOST_OK) status=atlas_host_wait(region->host,region->max_polls);
  if (status!=ATLAS_HOST_OK) return {upper}_DEVICE_ERROR;
  if (region->io.read(region->io.context,{p.status_address}u,region->workspace+{p.status_address-DRAM_BASE}u,32u)) return {upper}_MEMORY_ERROR;
  if (region->workspace[{p.status_address-DRAM_BASE}u] || region->workspace[{p.status_address-DRAM_BASE+1}u] || region->workspace[{p.status_address-DRAM_BASE+2}u] || region->workspace[{p.status_address-DRAM_BASE+3}u]) return {upper}_DOMAIN;
  if (region->io.read(region->io.context,{p.output_address}u,region->workspace+{p.output_address-DRAM_BASE}u,{p.output_bytes}u)) return {upper}_MEMORY_ERROR;
  __builtin_memcpy(outputs[0],region->workspace+{p.output_address-DRAM_BASE}u,{obytes}u);
  return {upper}_OK;
}}
'''
    public=runtime_headers(target_source/"runtime")
    _,shared,_=runtime_members(public,(target_source/"runtime/atlas_host.c").read_text(),runtime_linkage)
    files={k:v.encode() for k,v in c_source_pair(name,public_headers=public,private_headers={},sources={"exact_float.c":implementation},declarations=declarations).items()};files.update(shared)
    description=dict(kind="exact_float_leaf",context_type=name+"_context",source_context_type=name+"_source_context",binding_function=name+"_bind",
                     source_binding_function=name+"_source_bind",dispatch_function=name+"_source_dispatch",workspace_bytes=p.workspace_bytes,
                     source_workspace_bytes=1,launches_per_call=1,input_node_ids=[v.id for v in p.operands],output_node_ids=list(region.outputs),
                     weight_conversion="none",native_manifest_sha256=hashlib.sha256((compiled/"manifest.json").read_bytes()).hexdigest())
    return CompiledCRegion(name+"_source_dispatch",name+".h",tuple(v.id for v in p.operands),tuple(region.outputs),files,RUNTIME_SHARED_FILES,("segments/"+name,),description)
