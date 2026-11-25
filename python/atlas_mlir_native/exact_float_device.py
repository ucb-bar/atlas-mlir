"""Exact source float operations through the common Atlas virtual pipeline.

By default, checked operations scan the whole source domain before storing
results. NaNs and invalid infinity cancellation then return a device status
without output writes. Explicitly private binary results allow tile-local
validation; callers must discard the entire result on failure. Sign flips
accept all encodings and preserve NaN payloads.
"""
from .shared_dispatch_c import guarded
from .binding import io_memory_base as selected_io_base

from dataclasses import dataclass
from math import prod
from .binding import DRAM_BASE
from .memory_windows import dma_span_in_window
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

# Conservative scheduling envelope for private binary tiles, including the
# scalar fallback. It only selects a candidate size: compilation and replay
# independently prove the complete physical instruction bound and IMEM size.
PRIVATE_BINARY_TILE_ELEMENTS = 1024
PRIVATE_BINARY_TILE_INSTRUCTIONS = 640 * PRIVATE_BINARY_TILE_ELEMENTS
PRIVATE_BINARY_FIXED_INSTRUCTIONS = 4096



FLOAT_TYPES = guarded("float_types", "typedef struct { atlas_region_context *region; size_t input_first; } atlas_float_source_context;\n")
# One checked exact float launch. Inputs are staged in the region workspace (packed inputs through
# the region's slice copy, all checked before any device I/O), then the image runs once and the
# status word and output are read back. Return codes follow the region status enum.
FLOAT_DISPATCH = guarded("float_dispatch", """typedef struct { uint32_t offset,copy,payload; } atlas_float_input;
typedef struct {
  const uint32_t *words; const atlas_host_buffer *buffers; const atlas_float_input *inputs; const atlas_dispatch_shape *shape;
  int (*pack)(void *,const void *,size_t,size_t); uintptr_t base;
  uint32_t word_count,count,packed,period,pack_count,status_offset,output_offset,output_bytes,result_bytes;
} atlas_float_plan;
static __attribute__((unused)) int atlas_float_dispatch(const atlas_float_plan *p,void *opaque,const void *const *inputs,void *const *outputs) {
  atlas_float_source_context *ctx=opaque;
  if (!atlas_span_valid(ctx,sizeof(*ctx)) || !atlas_span_valid(ctx->region,sizeof(*ctx->region))) return 1;
  atlas_region_context *region=ctx->region;
  if (atlas_dispatch_conflict(p->shape,ctx,region,region->host,region->workspace,inputs,outputs)) return 1;
  if (region->host->active) return 4;
  uint8_t *w=region->workspace;
  for (size_t pass=0;pass<2u;++pass) for (size_t i=0;i<p->count;++i) {
    const atlas_float_input *x=&p->inputs[i];
    if (!pass) {
      if (i==p->packed) {
        size_t first=p->period?ctx->input_first%p->period:ctx->input_first;
        if (p->pack(w+x->offset,inputs[i],first,p->pack_count)) return 1;
      } else __builtin_memcpy(w+x->offset,inputs[i],x->copy);
      __builtin_memset(w+x->offset+x->copy,0,x->payload-x->copy);
    }
    /* Without packing each input is written as soon as it is staged; with packing, after every slice is validated. */
    if ((p->packed==UINT32_MAX)==!pass && region->io.write(region->io.context,p->base+x->offset,w+x->offset,x->payload)) return 3;
  }
  atlas_host_result status=atlas_host_ensure_loaded(region->host,p->words,p->word_count);
  if (status==ATLAS_HOST_OK) status=atlas_host_start(region->host,p->buffers,p->count+1u);
  if (status==ATLAS_HOST_OK) status=atlas_host_wait(region->host,region->max_polls);
  if (status!=ATLAS_HOST_OK) return 4;
  uint8_t *flag=w+p->status_offset;
  if (region->io.read(region->io.context,p->base+p->status_offset,flag,32u)) return 3;
  if (flag[0] || flag[1] || flag[2] || flag[3]) return 2;
  if (region->io.read(region->io.context,p->base+p->output_offset,w+p->output_offset,p->output_bytes)) return 3;
  __builtin_memcpy(outputs[0],w+p->output_offset,p->result_bytes);
  return 0;
}
""")

def max_leaf_elements(plan):
    """Scale the conservative slice policy; prove the actual image separately."""
    from .limits import DEFAULT_FUNCTIONAL_INSTRUCTION_LIMIT, functional_instruction_limit
    budget = functional_instruction_limit()
    if plan.private_output:
        tiles = max(1, (budget - PRIVATE_BINARY_FIXED_INSTRUCTIONS)
                    // PRIVATE_BINARY_TILE_INSTRUCTIONS)
        return tiles * PRIVATE_BINARY_TILE_ELEMENTS
    factor = max(1, budget // DEFAULT_FUNCTIONAL_INSTRUCTION_LIMIT)
    maximum = MAX_LEAF_ELEMENTS[plan.op] * factor
    if plan.op in ('bf16_add', 'bf16_sub'):
        # One route word per tile in the private 2-KiB admission table.
        maximum = min(maximum, 512 * plan.tile_lanes)
    return maximum


def chunk_elements(plan):
    """Respect both the operation policy and the selected I/O window."""
    available = plan.descriptor()["io_memory_bytes"] - 32
    per_element = len(plan.operands) * plan.width + plan.output_width
    capacity = available // per_element
    capacity -= capacity % plan.tile_lanes
    if capacity <= 0:
        raise ValueError("exact float cannot fit a tile in selected I/O window")
    return min(max_leaf_elements(plan), capacity)

@dataclass(frozen=True)
class FloatPlan:
    source: object
    node: object
    operands: tuple
    count: int
    width: int
    output_width: int
    io_memory_base: int = DRAM_BASE
    private_output: bool = False
    repeated_input: object = None
    @property
    def op(self): return self.node.op
    @property
    def tile_lanes(self): return min(1024,2048//max(self.width,self.output_width),_aligned(self.count,32))
    @property
    def input_bytes(self): return _aligned(self.count,self.tile_lanes)*self.width
    @property
    def cached_tiles(self):
        from .float_operand_cache import cached_tiles
        return cached_tiles(self)
    @property
    def input_payload_bytes(self):
        return tuple(self.cached_tiles*self.tile_lanes*self.width
                     if self.cached_tiles and i == self.repeated_input.port else self.input_bytes
                     for i in range(len(self.operands)))
    @property
    def input_addresses(self):
        sizes = self.input_payload_bytes
        return tuple(self.io_memory_base+sum(sizes[:i]) for i in range(len(sizes)))
    @property
    def output_address(self): return self.io_memory_base+sum(self.input_payload_bytes)
    @property
    def output_bytes(self): return _aligned(self.count,self.tile_lanes)*self.output_width
    @property
    def status_address(self): return self.output_address+self.output_bytes
    @property
    def workspace_bytes(self): return self.status_address+32-self.io_memory_base
    def descriptor(self):
        from .binding import TILE_BYTES,dram_slots
        return dict(operation=self.op,count=self.count,input_count=len(self.operands),right_input=[n.id for n in self.operands].index(self.node.inputs[-1]),source_request_digest=self.source.digest(),io_memory_bytes=dram_slots()*TILE_BYTES)
    def record(self):
        return dict(schema="atlas.exact_float.v1",
                    **({"private_output": True, "publication_contract": "device result is private until zero status; discard all result bytes on failure"} if self.private_output else {}),
                    **({"input_cache": dict(self.repeated_input.record(), tiles=self.cached_tiles,
                        initialization="each invocation", transport="complete repeated dense tile cycle")}
                       if self.cached_tiles else {}),
                    source_request_digest=self.source.digest(),operation=self.op,
                    input_node_ids=[n.id for n in self.operands],output_node_ids=[self.node.id],
                    input_addresses=list(self.input_addresses),
                    input_payload_bytes=list(self.input_payload_bytes),output_addresses=[self.output_address],
                    output_bytes=self.output_bytes+32,result_buffer_bytes=self.output_bytes,status_address=self.status_address,workspace_bytes=self.workspace_bytes,
                    logical_output_bytes=self.count*self.output_width,runtime_arithmetic="none",launches_per_invocation=1,
                    boundary_transport_layout=("compact-repeated-input-cycle-and-dense-private-output" if self.cached_tiles
                                               else "contiguous-full-tile-DMA-with-private-zero-padding"),
                    arithmetic_routes=["qualified-residual-vector", "pair-row-sum-vector", "exceptional-scalar"] if self.op in ("bf16_add","bf16_sub") else ["scalar-bit-arithmetic"],
                    verification="common-virtual-scheduling-allocation-and-all-generated-contracts",
                    vector_admission={"qualified":"signed-zero or abs-code2f80..4f80",
                                      "pair":"normal-or-signed-zero inputs except both-negative-zero; private result must be normal or provably exact zero"} if self.op in ("bf16_add","bf16_sub") else None)

def prepare_source_plan(source, *, io_memory_base=None):
    """Validate logical source semantics before bounded physical slicing."""
    io_memory_base = selected_io_base() if io_memory_base is None else io_memory_base
    from .binding import TARGET_IDENTITY
    from merlin.compiler.semantic.model import TensorType,pointwise_identity_maps
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
    plan=FloatPlan(source,n,operands,prod(n.type.shape),width,ow,io_memory_base)
    if type(io_memory_base) is not int or io_memory_base < 0 or io_memory_base % 32:
        raise ValueError("exact float buffers leave selected DMA memory windows")
    from .binding import validate_explicit_io_base
    validate_explicit_io_base(io_memory_base)
    return plan


def prepare_plan(source, *, io_memory_base=None, private_output=False, input_layout=None, cache_input=False):
    """Admit a physical leaf only when its complete frame fits DMA memory."""
    from .binding import dram_slots, TILE_BYTES

    plan = prepare_source_plan(source, io_memory_base=io_memory_base)
    if type(private_output) is not bool or private_output and plan.op not in ('bf16_add', 'bf16_sub'):
        raise ValueError('private float publication currently requires an exact binary operation')
    if private_output:
        from dataclasses import replace
        plan = replace(plan, private_output=True)
    if type(cache_input) is not bool:
        raise ValueError('input cache selection must be Boolean')
    if cache_input:
        from .float_operand_cache import with_repeated_input
        plan = with_repeated_input(plan, input_layout)
    if (plan.workspace_bytes > dram_slots() * TILE_BYTES
            or not dma_span_in_window(plan.io_memory_base, plan.workspace_bytes)):
        raise ValueError("exact float buffers leave selected DMA memory windows")
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


def compile_source(snapshot,region,*,target_source,destination,limits,io_memory_base=None,private_output=False,input_layout=None,cache_input=False):
    io_memory_base = selected_io_base() if io_memory_base is None else io_memory_base
    import hashlib,json,os
    from pathlib import Path
    from atlas_mlir_native.options import current_codegen_options
    from merlin.compiler.semantic.target_binding import verify_native_publication
    from .binding import NATIVE_CHECKS,RTL_REVISION,_run
    p=prepare_plan(region,io_memory_base=io_memory_base,private_output=private_output,
                   input_layout=input_layout,cache_input=cache_input)
    maximum=max_leaf_elements(p)
    if p.count>maximum: raise ValueError(f"exact float leaf requires at most {maximum} elements")
    if p.workspace_bytes>p.descriptor()["io_memory_bytes"]: raise ValueError("exact float exceeds selected I/O window")
    tools=Path(os.environ.get("ATLAS_OOT_BIN_DIR",Path(target_source)/"build/bin"))
    virtual=virtual_program(p)
    machine=machine_program(p)
    _run([str(tools/"atlas-opt"),*NATIVE_CHECKS,"--verify-atlas-timing","-"],input_text=machine)
    program=json.loads(_run([str(tools/"atlas-emit"),"--program-json","-"],input_text=machine))
    if program.get("selected_rtl_revision")!=RTL_REVISION: raise ValueError("exact float emitter RTL differs")
    from .restoration_budget import instruction_bound
    from .limits import functional_instruction_limit
    from merlin.compiler.semantic.target_binding import NativeCompilationError
    try:
        bound=instruction_bound(virtual,machine,program,max_instructions=functional_instruction_limit(limits))
    except (ValueError,KeyError,IndexError) as error:
        raise NativeCompilationError("resource_limit","exact float instruction bound is unproved: "+str(error)) from error
    if not bound["admitted"]:
        raise NativeCompilationError("resource_limit",f"exact float worst-case instruction bound exceeds {bound['max_instructions']}: "+str(bound["instruction_bound"]))
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


def prepare_c_package(region,compiled,name,*,target_source,runtime_linkage="separate",io_memory_base=None,input_layout=None,private_output=False,cache_input=False):
    io_memory_base = selected_io_base() if io_memory_base is None else io_memory_base
    import hashlib,json,os
    from pathlib import Path
    from merlin.compiler.lower.c.dispatch import c_source_pair
    from merlin.compiler.build.regions import CompiledCRegion
    from merlin.compiler.semantic.target_binding import verify_native_publication
    from .binding import NATIVE_CHECKS,_run
    from .c_runtime import RUNTIME_SHARED_FILES,program_array,runtime_headers,runtime_members
    if runtime_linkage!="separate":raise ValueError("exact float requires shared runtime linkage")
    p=prepare_plan(region,io_memory_base=io_memory_base,private_output=private_output,
                   input_layout=input_layout,cache_input=cache_input);compiled,target_source=Path(compiled),Path(target_source)
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
    saved_bound = json.loads((compiled / "float-instruction-bound.json").read_text())
    from .limits import functional_instruction_limit
    from types import SimpleNamespace
    cap=functional_instruction_limit(SimpleNamespace(functional_instruction_limit=saved_bound['max_instructions']))
    bound=instruction_bound(virtual_program(p),machine,program,schema=saved_bound["schema"],max_instructions=cap)
    if not bound["admitted"] or json.loads((compiled/"float-instruction-bound.json").read_text())!=bound:
        raise ValueError("exact float instruction bound differs or exceeds selected budget")
    upper=name.upper();n=len(p.operands);logical=p.count*p.width;obytes=p.count*p.output_width
    input_sizes = [logical] * n
    mapped_index, packing_helper = None, ""
    if input_layout is not None:
        from merlin.compiler.lower.c.layout import render_indexed_slice_copy
        physical, view = input_layout
        mapped_index = [operand.id for operand in p.operands].index(view.id)
        if (p.operands[mapped_index].type.dtype != view.type.dtype
                or p.operands[mapped_index].type.numerical_policy != view.type.numerical_policy
                or p.count > prod(view.type.shape)):
            raise ValueError("exact float input packing differs from its logical operand")
        packing_helper = render_indexed_slice_copy(physical, view, name + "_pack_input")
        input_sizes[mapped_index] = prod(physical.type.shape) * p.width
    from .shared_dispatch_c import CHECKS, REGION_TYPES, shape
    buffers=",".join(f"{{{address}u,{size}u,ATLAS_HOST_DEVICE_READ}}"
                     for address,size in zip(p.input_addresses,p.input_payload_bytes,strict=True))
    buffers+=f",{{{p.output_address}u,{p.output_bytes+32}u,ATLAS_HOST_DEVICE_WRITE}}"
    declarations=REGION_TYPES+FLOAT_TYPES+f'''
#define {upper}_WORKSPACE_BYTES {p.workspace_bytes}u
#define {upper}_SOURCE_WORKSPACE_BYTES 1u
typedef enum {{ {upper}_OK=0,{upper}_INVALID=1,{upper}_DOMAIN=2,{upper}_MEMORY_ERROR=3,{upper}_DEVICE_ERROR=4 }} {name}_status;
typedef atlas_region_context {name}_context;
typedef atlas_float_source_context {name}_source_context;
{name}_status {name}_bind({name}_context *,atlas_host *,atlas_tile_memory_io,void *,size_t,size_t);
{name}_status {name}_source_bind({name}_source_context *,{name}_context *,void *,size_t);
int {name}_source_dispatch(void *,const void *const *,void *const *);
'''
    rows=[]
    for i in range(n):
        at=p.input_addresses[i]-p.io_memory_base
        payload=p.input_payload_bytes[i]
        copied=payload if i == mapped_index and p.cached_tiles and i == p.repeated_input.port else logical
        rows.append(f"{{{at}u,{copied}u,{payload}u}}")
    packed,period,pack_count="UINT32_MAX",0,0
    if mapped_index is not None:
        packed,pack_count=f"{mapped_index}u",p.count
        if p.cached_tiles and mapped_index == p.repeated_input.port:
            period,pack_count=p.repeated_input.period,p.input_payload_bytes[mapped_index]//p.width
    pack=name+"_pack_input" if mapped_index is not None else "0"
    implementation=packing_helper+program_array(name+"_words",binary)+CHECKS+FLOAT_DISPATCH+shape(
        name,context=name+"_source_context",region=name+"_context",workspace_bytes=p.workspace_bytes,
        inputs=input_sizes,outputs=[obytes])+f'''
static const atlas_host_buffer {name}_buffers[]={{{buffers}}};
static const atlas_float_input {name}_inputs[]={{{','.join(rows)}}};
static const atlas_float_plan {name}_plan={{{name}_words,{name}_buffers,{name}_inputs,&{name}_shape,{pack},{p.io_memory_base}u,
  {len(binary)//4}u,{n}u,{packed},{period}u,{pack_count}u,{p.status_address-p.io_memory_base}u,
  {p.output_address-p.io_memory_base}u,{p.output_bytes}u,{obytes}u}};
{name}_status {name}_bind({name}_context *ctx,atlas_host *host,atlas_tile_memory_io io,void *workspace,size_t bytes,size_t polls) {{
  if (atlas_bind_conflict(ctx,sizeof(*ctx),host,workspace,{p.workspace_bytes}u,bytes,&io,polls)) return {upper}_INVALID;
  *ctx=({name}_context){{host,io,workspace,polls}};return {upper}_OK;
}}
{name}_status {name}_source_bind({name}_source_context *ctx,{name}_context *region,void *workspace,size_t bytes) {{
  (void)workspace;(void)bytes;
  if (atlas_source_bind_conflict(ctx,sizeof(*ctx),region,sizeof(*region),offsetof({name}_context,host),offsetof({name}_context,workspace),{p.workspace_bytes}u))
    return {upper}_INVALID;
  ctx->region=region;ctx->input_first=0;return {upper}_OK;
}}
int {name}_source_dispatch(void *opaque,const void *const *inputs,void *const *outputs) {{
  return atlas_float_dispatch(&{name}_plan,opaque,inputs,outputs);
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
