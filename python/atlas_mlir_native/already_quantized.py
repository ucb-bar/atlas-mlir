"""Byte-preserving native intake for explicitly quantized contractions.

Exponent tensors remain signed data. The only numerical conversion after the
MXU contraction restores their sum at readout. No weight encoder is involved.
"""
from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
from math import prod
from pathlib import Path

import numpy as np
from merlin.semantic_compiler.model import KernelRequest, SemanticNode, TensorType
from merlin.semantic_compiler.target_binding import NativeCompilationError, verify_native_publication

from .binding import (AtlasNativeBinding, DRAM_BASE, MXU1_CAP448_INPUT_DOMAIN,
    MXU1_CAP448_POLICY, MXU1_CAP448_SEED_DOMAIN, TARGET_IDENTITY, TILE_BYTES,
    _mxu0_maps, _mxu0_accumulate_maps)

POLICY = "atlas-already-quantized-e4m3-signed-exponents-mxu1-model47-v1"
OPS = ("atlas.qlinear", "atlas.qk_matmul", "atlas.pv_matmul")


def _digest(value):
    return sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _fail(message):
    raise NativeCompilationError("unsupported_semantics", message)


def valid_codes(values):
    values = np.asarray(values)
    if values.dtype != np.uint8 or np.any(~((values == 0) | ((values >= 8) & (values <= 126)) | ((values >= 136) & (values <= 254)))):
        raise ValueError("already-quantized operand requires unchanged finite normal cap448 codes or positive zero")
    return values


def pack_operand(codes, *, row=0, k=0):
    """Copy a logical [rows,K] code window, padding inactive lanes with +0."""
    values = valid_codes(codes)
    if values.ndim != 2 or row < 0 or k < 0:
        raise ValueError("FP8 packing requires a nonnegative rank-two window")
    out = np.zeros((32, 32), dtype=np.uint8)
    view = values[row:row+32, k:k+32]
    out[:view.shape[0], :view.shape[1]] = view
    return out.tobytes()


def restore_bf16(codes, lhs_exponents, rhs_exponents, *, output_dtype="bf16"):
    """Restore signed exponents without decoding or changing operand codes."""
    words = np.asarray(codes)
    lhs, rhs = np.asarray(lhs_exponents), np.asarray(rhs_exponents)
    if words.dtype != np.uint16 or any(value.dtype.kind != "i" for value in (lhs, rhs)):
        raise ValueError("readout requires raw BF16 codes and signed integer exponent data")
    if any(np.any((value < -127) | (value > 127)) for value in (lhs, rhs)):
        raise ValueError("signed exponents leave the admitted operand domain")
    shifts = lhs.astype(np.int64) + rhs.astype(np.int64)
    shifts = np.broadcast_to(shifts, words.shape)
    if np.any((shifts < -133) | (shifts > 127)):
        raise ValueError("readout factor leaves the finite nonzero BF16 power-of-two domain [-133,127]")
    normal = (words & 0x7fff) != 0
    exponent = ((words >> 7) & 255).astype(np.int64)
    if np.any(normal & ((exponent == 0) | (exponent == 255) | (exponent + shifts <= 0) | (exponent + shifts >= 255))):
        raise ValueError("scaled readout leaves the finite normal BF16 domain")
    result = np.where(normal, (words & 0x807f) | ((exponent + shifts) << 7), words).astype(np.uint16)
    if output_dtype == "bf16":
        return result
    if output_dtype != "f32":
        raise ValueError("already-quantized readout supports BF16 or f32")
    return (result.astype(np.uint32) << 16).view(np.float32)


def _restore_bf16_c(name):
    """Restore exact nonzero BF16 powers of two with normal results or zero."""
    return f"""
static int {name}_restore(uint32_t word,int32_t lhs,int32_t rhs,uint32_t *out) {{
  int64_t scale=(int64_t)lhs+(int64_t)rhs;
  if (scale< -133 || scale>127) return 2;
  if (word&0x7fffu) {{
    int64_t exponent=(int64_t)((word>>7)&255u)+scale;
    if (!(word&0x7f80u) || (word&0x7f80u)==0x7f80u || exponent<=0 || exponent>=255) return 2;
    word=(word&0x807fu)|((uint32_t)exponent<<7);
  }}
  *out=word; return 0;
}}
"""


@dataclass(frozen=True)
class QuantizedPlan:
    source: KernelRequest
    region_id: str
    operand_ids: tuple[str, str]
    exponent_ids: tuple[str, str]
    input_ids: tuple[str, ...]
    batch_shape: tuple[int, ...]
    m: int
    n: int
    k: int
    output_dtype: str
    reset: KernelRequest
    continuation: KernelRequest
    epilogue_precision: object | None = None
    epilogue_contract_operand: int | None = None
    epilogue_other_id: str | None = None
    epilogue_other_dtype: str | None = None
    narrowing_input_id: str | None = None
    operand_permutation: tuple[str, str, tuple[int, ...]] | None = None

    @property
    def physical_operand_ids(self):
        if self.operand_permutation is None:
            return self.operand_ids
        logical, physical, _ = self.operand_permutation
        return tuple(physical if key == logical else key for key in self.operand_ids)

    @property
    def output_id(self):
        return self.region_id

    def record(self):
        record = dict(schema="atlas.already_quantized_contraction.v1", policy=POLICY,
            source_request_digest=self.source.digest(), region_id=self.region_id,
            operand_ids=list(self.operand_ids), exponent_ids=list(self.exponent_ids),
            input_ids=list(self.input_ids), batch_shape=list(self.batch_shape), shape=[self.m,self.n,self.k],
            output_dtype=self.output_dtype, weight_conversion="none; unchanged logical FP8 codes",
            exponent_interpretation="signed i32; restore 2^(lhs_exp+rhs_exp) after MXU readout",
            k_order="K32 MXU1 reset followed by explicit ordered BF16 seeded continuation",
            padding="positive zero, logical operands unchanged", reset_request_digest=self.reset.digest(),
            continuation_request_digest=self.continuation.digest())
        if self.epilogue_precision is not None:
            record.update(schema="atlas.already_quantized_epilogue.v1", epilogue=dict(
                precision=self.epilogue_precision.record(), contract_operand=self.epilogue_contract_operand,
                other_id=self.epilogue_other_id, other_dtype=self.epilogue_other_dtype,
                arithmetic_order="unchanged ordered K chain, signed-exponent restoration, then original BF16 RNE multiply",
                input_conversion="original other-operand encoder and guard; checked device admission of restored contraction",
                physical_layout="contraction batch/M-tile/N-tile BF16 pair halves; positive-zero padding"))
        if self.narrowing_input_id is not None:
            record.update(schema="atlas.already_quantized_narrowing.v1", narrowing=dict(
                input_id=self.narrowing_input_id, source_dtype="f32", output_dtype="bf16",
                admission=("bounded non-NaN BF16 epilogue product; exact FP32 widening"
                           if self.epilogue_precision is not None else
                           "restored finite normal BF16 or signed zero; exact FP32 widening"),
                arithmetic=("unchanged ordered K, exponent restoration and original BF16 RNE multiply; "
                            "omit widening then narrowing" if self.epilogue_precision is not None else
                            "unchanged ordered K and exponent restoration; omit widening then narrowing")))
        if self.operand_permutation is not None:
            logical, physical, axes = self.operand_permutation
            record['operand_permutation'] = dict(logical_operand=logical, physical_operand=physical,
                axes=list(axes), placement='existing raw-byte input tile packing; no intermediate copy')
        return record


def _body(source, continued):
    fp8 = TensorType((32,32), "fp8_e4m3", "raw-bit-pattern")
    inputs = tuple(SemanticNode(role,"input",(),fp8,(("input_domain",MXU1_CAP448_INPUT_DOMAIN),),"input") for role in ("a","w"))
    if continued:
        inputs += (SemanticNode("seed","input",(),TensorType((32,32),"bf16","raw-bit-pattern"),
            (("input_domain",MXU1_CAP448_SEED_DOMAIN),),"input"),)
    result = SemanticNode("result","matmul_accumulate" if continued else "matmul",tuple(node.id for node in inputs),
        TensorType((32,32),"bf16",MXU1_CAP448_POLICY),index_maps=_mxu0_accumulate_maps() if continued else _mxu0_maps())
    return KernelRequest((*inputs,result),("result",),("external",),tuple((node.id,"external") for node in inputs),
        TARGET_IDENTITY,"diagnostic",_digest([source.digest(), POLICY, continued]))


def prepare_source(source):
    pure = [node for node in source.nodes if node.effect == "pure"]
    if (len(pure) == 3 and pure[0].op in OPS and pure[1].op == "f32_mul"
            and pure[2].op == "f32_to_bf16"):
        from .quantized_epilogue_narrowing import prepare_source as prepare_product_narrowing
        return prepare_product_narrowing(source)
    if any(node.op in {'gather', 'transpose'} for node in pure):
        from .quantized_operand_permutation import prepare_source as prepare_permutation
        return prepare_permutation(source)
    if (len(pure) == 2 and pure[0].op in OPS
            and pure[1].op in ("f32_mul", "bf16_mul")):
        from .quantized_epilogue import prepare_source as prepare_epilogue
        return prepare_epilogue(source)
    if len(pure) == 2 and pure[0].op in OPS and pure[1].op == "f32_to_bf16":
        from .quantized_narrowing import prepare_source as prepare_narrowing
        return prepare_narrowing(source)
    from .source_translation import lower_already_quantized_contraction
    intake = lower_already_quantized_contraction(source)
    result = source.node(intake.output_id)
    if source.target_identity != TARGET_IDENTITY or source.lowering_policy != "diagnostic" or result.op not in OPS:
        _fail("already-quantized contraction requires the selected diagnostic Atlas target")
    if dict(result.attrs).get("weight_format") != "atlas_e4m3":
        _fail("already-quantized contraction requires Atlas E4M3 weight format")
    if dict(result.attrs).get("pop_shift", 0) != 0:
        _fail("folded PV pop_shift is unsupported; preserve the unfolded graph")
    if len(result.inputs) != 4:
        _fail("already-quantized contraction bias requires a separate checked BF16 addition")
    if result.type.dtype not in ("bf16","f32"):
        _fail("already-quantized result requires BF16 or f32")
    a,b = (source.node(key) for key in intake.operand_ids)
    # The first implementation accepts identical batch prefixes, and a shared
    # rank-two qlinear weight. Other broadcasts require an explicit layout op.
    batch = a.type.shape[:-2]
    if b.type.shape[:-2] not in (batch, ()) or result.type.shape != (*batch,intake.m,intake.n):
        _fail("already-quantized operand batches require identical prefixes or one shared weight")
    for key, node in zip(intake.exponent_ids,(a,b),strict=True):
        shape = source.node(key).type.shape
        if shape not in ((), (*node.type.shape[:-1],1), (*node.type.shape[:-2],1,1)):
            _fail("already-quantized signed exponents require scalar, per-row, or per-head shape")
    for binding in source.constants:
        node = source.node(binding.node_id)
        if binding.node_id in intake.operand_ids:
            valid_codes(np.frombuffer(bytes.fromhex(binding.data_hex),dtype=np.uint8))
    return QuantizedPlan(source,intake.output_id,intake.operand_ids,intake.exponent_ids,
        tuple(node.id for node in source.nodes if node.effect in ("input","constant")),tuple(batch),
        intake.m,intake.n,intake.k,result.type.dtype,_body(source,False),_body(source,True))


def _native_bodies(plan):
    """Publish exactly the bodies that the ordered K32 dispatch can call."""
    bodies = (("reset", plan.reset),)
    return bodies + (("continue", plan.continuation),) if plan.k > 32 else bodies


def compile_source(snapshot, source, *, target_source, destination, limits):
    plan = prepare_source(source)
    destination = Path(destination)
    for role,body in _native_bodies(plan):
        abi = AtlasNativeBinding.plan_boundary_abi(body)
        AtlasNativeBinding.compile(snapshot,body,fixed_inputs=abi["fixed_inputs"],fixed_outputs=tuple(abi["fixed_outputs"]),
            target_source=target_source,destination=destination/role,limits=limits,allow_staging=False)
    (destination/"already-quantized-plan.json").write_text(json.dumps(plan.record(),sort_keys=True,indent=2)+"\n")
    return plan


def prepare_c_package(source, compiled, name, *, target_source):
    """Export the used checked K32 images with unchanged code/exponent inputs."""
    from merlin.compile.c_sources import c_source_pair
    from .c_runtime import program_array, runtime_headers, runtime_members
    plan = prepare_source(source)
    compiled,target_source = Path(compiled),Path(target_source)
    if json.loads((compiled/"already-quantized-plan.json").read_text()) != plan.record():
        raise ValueError("already-quantized source plan differs from its publication")
    images = []
    for role,body in _native_bodies(plan):
        root = compiled/role
        manifest = json.loads((root/"manifest.json").read_text())
        verify_native_publication(root,manifest,engine="merlin_native",request_digest=body.digest(),target_identity=TARGET_IDENTITY)
        if KernelRequest.from_record(json.loads((root/"source-request.json").read_text())).digest() != body.digest():
            raise ValueError("already-quantized native body differs from its publication")
        abi = AtlasNativeBinding.plan_boundary_abi(body)
        execution = json.loads((root/"execution_plan.json").read_text())
        addresses = [DRAM_BASE+abi["fixed_inputs"][node.id]*TILE_BYTES for node in body.nodes if node.effect == "input"]
        out = DRAM_BASE+abi["fixed_outputs"][0]*TILE_BYTES
        if execution["input_addresses"] != addresses or execution["output_addresses"] != [out] or execution.get("operation") != "mxu1_cap448_seeded_ordered":
            raise ValueError("already-quantized native operand ABI differs")
        if sha256((target_source/"docs/selected-variant-inventory.json").read_bytes()).hexdigest() != manifest["target_inventory_sha256"]:
            raise ValueError("already-quantized runtime differs from its checked target inventory")
        binary = (root/"program.bin").read_bytes()
        physical = json.loads((root/"physical-program.json").read_text())
        if binary != b"".join(row["word_u32"].to_bytes(4,"little") for row in physical["instructions"]):
            raise ValueError("already-quantized binary differs from checked instruction words")
        images.append((role,binary,addresses,out))
    m,n,k,batches = plan.m,plan.n,plan.k,prod(plan.batch_shape)
    inputs = [source.node(key) for key in plan.input_ids]
    sizes = [prod(node.type.shape)*{"i8":1,"ui8":1,"u8":1,"fp8_e4m3":1,"i32":4,"si32":4,"bf16":2,"f32":4}[node.type.dtype] for node in inputs]
    a,b = (source.node(key) for key in plan.operand_ids)
    ia,ib = (plan.input_ids.index(key) for key in plan.operand_ids)
    iea,ieb = (plan.input_ids.index(key) for key in plan.exponent_ids)
    def exp_index(key,operand,row):
        shape = source.node(key).type.shape
        if not shape: return "0u"
        batch = "batch" if operand.type.shape[:-2] else "0u"
        if shape[-2:] == (1,1): return batch
        return f"({batch}*{operand.type.shape[-2]}u+{row})"
    ae = exp_index(plan.exponent_ids[0],a,"first_m+row")
    be = exp_index(plan.exponent_ids[1],b,"first_n+c")
    lhs_scan = exp_index(plan.exponent_ids[0],a,"row")
    rhs_scan = exp_index(plan.exponent_ids[1],b,"column")
    wb = "batch" if b.type.shape[:-2] else "0u"
    output_bytes = batches*m*n*(2 if plan.output_dtype=="bf16" else 4)
    raw,converted,count = 8192,output_bytes,len(inputs)
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
    arrays="\n".join(program_array(name+"_"+role+"_words",binary) for role,binary,_,_ in images)
    cases=[]
    for role,binary,addresses,out in images:
        writes="\n".join(f"      if (region->io.write(region->io.context,{address}u,region->workspace+{offset}u,{size}u)) return 3;"
            for address,offset,size in zip(addresses,(0,1024,2048),(1024,1024,2048),strict=False))
        buffers=",".join(f"{{{address}u,{2048 if i==2 else 1024}u,ATLAS_HOST_DEVICE_READ}}" for i,address in enumerate(addresses))
        buffers+=f",{{{out}u,2048u,ATLAS_HOST_DEVICE_WRITE}}"
        cases.append(f"""{{
      status=atlas_host_ensure_loaded(region->host,{name}_{role}_words,{len(binary)//4}u);
      if (status!=ATLAS_HOST_OK) {{ if (device) *device=status; return 4; }}
{writes}
      atlas_host_buffer buffers[]={{{buffers}}};
      status=atlas_host_start(region->host,buffers,{len(addresses)+1}u);
      if (status==ATLAS_HOST_OK) status=atlas_host_wait(region->host,region->max_polls);
      if (device) *device=status;
      if (status!=ATLAS_HOST_OK) return 4;
      if (region->io.read(region->io.context,{out}u,region->workspace+4096u,2048u)) return 3;
    }}""")
    storing = ("uint16_t value=(uint16_t)word; __builtin_memcpy(ctx->workspace+index*2u,&value,2u);" if plan.output_dtype=="bf16"
        else "uint32_t value=word<<16; __builtin_memcpy(ctx->workspace+index*4u,&value,4u);")
    dispatch = cases[0] if len(cases) == 1 else f"if (!first_k) {cases[0]} else {cases[1]}"
    implementation=arrays+_restore_bf16_c(name)+f"""
static int {name}_range(const void *p,size_t n) {{ return p && n && (uintptr_t)p<=UINTPTR_MAX-n; }}
static int {name}_overlap(const void *a,size_t an,const void *b,size_t bn) {{ return (uintptr_t)a<(uintptr_t)b+bn && (uintptr_t)b<(uintptr_t)a+an; }}
static int32_t {name}_exponent(const void *p,size_t i) {{ int32_t v; __builtin_memcpy(&v,(const uint8_t *)p+4u*i,4u); return v; }}
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
  const size_t exponent_inputs[2]={{{iea}u,{ieb}u}};
  for (size_t which=0;which<2u;++which) for (size_t i=0;i<sizes[exponent_inputs[which]]/4u;++i) {{
    int32_t exponent={name}_exponent(inputs[exponent_inputs[which]],i);
    if (exponent< -127 || exponent>127) return 2;
  }}
  for (size_t which=0;which<2u;++which) {{
    const uint8_t *v=which?b:a; size_t size=which?{prod(b.type.shape)}u:{prod(a.type.shape)}u;
    for (size_t i=0;i<size;++i) if (!(v[i]==0u || (v[i]>=8u && v[i]<=126u) || (v[i]>=136u && v[i]<=254u))) return 2;
  }}
  /* Check every paired scale before program loading or device I/O. */
  for (size_t batch=0;batch<{batches}u;++batch) {{
    int32_t lhs_min=127,lhs_max=-127,rhs_min=127,rhs_max=-127;
    for (size_t row=0;row<{m}u;++row) {{
      int32_t value={name}_exponent(inputs[{iea}],{lhs_scan});
      if (value<lhs_min) lhs_min=value;
      if (value>lhs_max) lhs_max=value;
    }}
    for (size_t column=0;column<{n}u;++column) {{
      int32_t value={name}_exponent(inputs[{ieb}],{rhs_scan});
      if (value<rhs_min) rhs_min=value;
      if (value>rhs_max) rhs_max=value;
    }}
    if ((int64_t)lhs_min+(int64_t)rhs_min< -133 ||
        (int64_t)lhs_max+(int64_t)rhs_max>127) return 2;
  }}
  for (size_t batch=0;batch<{batches}u;++batch) for (size_t first_m=0;first_m<{m}u;first_m+=32u)
    for (size_t first_n=0;first_n<{n}u;first_n+=32u) {{
      for (size_t first_k=0;first_k<{k}u;first_k+=32u) {{
        __builtin_memset(region->workspace,0,2048u);
        for (size_t row=0;row<32u;++row) for (size_t c=0;c<32u;++c) if (first_k+c<{k}u) {{
          if (first_m+row<{m}u) region->workspace[row*32u+c]=a[(batch*{m}u+first_m+row)*{k}u+first_k+c];
          if (first_n+row<{n}u) region->workspace[1024u+row*32u+c]=b[({wb}*{n}u+first_n+row)*{k}u+first_k+c];
        }}
        {dispatch}
        for (size_t i=0;i<2048u;i+=2u) {{
          uint32_t word=region->workspace[4096u+i]|((uint32_t)region->workspace[4097u+i]<<8);
          if (word && (!(word&0x7f80u) || (word&0x7f80u)==0x7f80u)) return 2;
        }}
        __builtin_memcpy(region->workspace+2048u,region->workspace+4096u,2048u);
      }}
      for (size_t row=0;row<32u && first_m+row<{m}u;++row) for (size_t c=0;c<32u && first_n+c<{n}u;++c) {{
        size_t at=(c/16u)*1024u+(row*16u+c%16u)*2u;
        uint32_t word=(uint32_t)region->workspace[4096u+at]|((uint32_t)region->workspace[4097u+at]<<8);
        if ({name}_restore(word,{name}_exponent(inputs[{iea}],{ae}),{name}_exponent(inputs[{ieb}],{be}),&word)) return 2;
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
    from merlin.compile.launch_schedule import make_launch_schedule
    runs = [dict(image=0, count=1)]
    k_tiles = (k + 31) // 32
    if k_tiles > 1:
        runs.append(dict(image=1, count=k_tiles-1))
    schedule = make_launch_schedule([sha256(binary).hexdigest() for _,binary,_,_ in images],
        [dict(repeat=batches*((m+31)//32)*((n+31)//32), runs=runs)])
    record=dict(kind="already_quantized",schema="atlas.already_quantized_c_package.v1",package_name=name,
        context_type=name+"_context",source_context_type=name+"_source_context",binding_function=name+"_bind",
        source_binding_function=name+"_source_bind",workspace_bytes=raw,source_workspace_bytes=converted,
        readout_scale_domain="paired signed exponent sums in [-133,127]; finite nonzero BF16 powers of two",
        precision_plan=plan.record(),input_node_ids=list(plan.input_ids),output_node_ids=[plan.output_id],
        source_interface=dict(workspace_bytes=converted,failure_output="unchanged",completion="synchronous"),
        native_program_paths=[role for role,_,_,_ in images], device_launch_schedule=schedule, **runtime_record)
    return record,files
